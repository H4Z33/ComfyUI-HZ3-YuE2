"""Whisper transcriptions and an LLM section-styling pass for YuE2 covers.

HZ3_YuE2_Whisper          -- transcribe an input AUDIO into raw lyrics using the
                             HuggingFace Transformers Whisper pipe (no extra
                             dependency beyond transformers/torch, already present).
HZ3_YuE2_MixMashGenius    -- an Ollama agent that turns raw lyrics + an ABC score
                             + free-text instructions into a per-section styled
                             [Section]-tag output: a style prompt, sectioned
                             lyrics and the untouched ABC (passthrough).
"""

from __future__ import annotations

import bisect
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import torch

import folder_paths

from .score_analysis import inspect_score


# ---------------------------------------------------------------------------
# Shared Ollama client (json in / json out)
# ---------------------------------------------------------------------------
def _ollama_chat(system, payload, model, endpoint, temperature, timeout):
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }
    request = urllib.request.Request(endpoint.rstrip("/") + "/api/chat",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama is unavailable at {endpoint}: {exc.reason}") from exc
    content = raw.get("message", {}).get("content", "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(content)


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------
def _model_dirs():
    directories = []
    try:
        directories.extend(Path(p) for p in folder_paths.get_folder_paths("whisper"))
    except KeyError:
        pass
    directories.append(Path(folder_paths.models_dir) / "whisper")
    directories.append(Path(folder_paths.models_dir))
    return list(dict.fromkeys(directories))


def _resolve_model(model_name):
    """Return a local checkpoint dir if one exists, else the HF id unchanged."""
    candidate = Path(model_name)
    if candidate.exists():
        return str(candidate)
    for directory in _model_dirs():
        local = directory / model_name
        # accepts a bare folder (openai/whisper-small) or a HF snapshot dir
        config = local / "config.json"
        if local.exists() and config.exists():
            return str(local)
        snapshot = local / "snapshots" if local.exists() else None
        if local.exists() and any((local / sub).is_dir() for sub in ("snapshots",)) and not config.exists():
            return str(local)
    return model_name


def _load_pipe(model_name, device):
    from transformers import pipeline
    local = _resolve_model(model_name)
    device_id = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
    dtype = torch.float16 if device_id != -1 else torch.float32
    try:
        # chunk_length_s/stride_length_s: generic seq2seq chunked transcription.
        # Emits the experimental warning but transcribes long audio far better
        # than Whisper's own native chunking for this use case.
        return pipeline(
            "automatic-speech-recognition",
            model=local,
            feature_extractor=local,
            tokenizer=local,
            device=device_id,
            torch_dtype=dtype,
            chunk_length_s=30,
            stride_length_s=5,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Whisper model {model_name!r}. Put a checkpoint under "
            f"models/whisper (e.g. models/whisper/openai/whisper-small) or use an "
            f"internet-reachable HuggingFace id. Original error: {exc}"
        ) from exc


def _to_mono_16k(audio):
    waveform = audio["waveform"].detach().float().cpu()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 2:
        waveform = waveform.mean(0)
    import torchaudio
    if int(audio["sample_rate"]) != 16000:
        waveform = torchaudio.functional.resample(
            waveform, int(audio["sample_rate"]), 16000
        )
    return waveform.numpy().astype("float32")


def _seg_times(chunk):
    """Whisper chunk timestamps can be `(start, end)`, a bare float, or contain
    None endpoints (e.g. no predicted ending). Normalize to (start, end|None)."""
    ts = chunk.get("timestamp")
    if isinstance(ts, (tuple, list)) and len(ts) == 2:
        return ts[0], ts[1]
    if isinstance(ts, (int, float)):
        return ts, None
    return None, None


def _rounded(value):
    return round(value, 2) if value is not None else None


BACKENDS = ["whisper small", "whisper medium", "fast"]

# module-level pipeline cache so a new node instance / re-run does not reload
# the model every execution
_PIPE_CACHE = {}


def _get_pipe(backend, model, device):
    key = (backend, model, device)
    if key not in _PIPE_CACHE:
        if backend in ("whisper small", "whisper medium"):
            _PIPE_CACHE[key] = _load_pipe(model, device)
        elif backend == "fast":
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise RuntimeError("'fast' (faster-whisper) is not installed. Run: pip install faster-whisper")
            _PIPE_CACHE[key] = WhisperModel(model, device=device, compute_type=_compute_type(device))
        else:
            raise ValueError(f"Unknown backend: {backend!r}")
    return _PIPE_CACHE[key]


def _backend_model(backend):
    return {
        "whisper small": "openai/whisper-small",
        "whisper medium": "openai/whisper-medium",
        "fast": "small",
    }[backend]


def _compute_type(device):
    return "int8" if device == "cpu" else "float16"


def _transcribe(backend, mono, language, task, device):
    """Run one backend and return (full_text, [(start, end, text), ...])."""
    lang = None
    if language and language.strip().lower() not in ("auto", ""):
        lang = language.strip().lower()
    model = _backend_model(backend)

    if backend in ("whisper small", "whisper medium"):
        # Transformers Whisper pipeline
        pipe = _get_pipe(backend, model, device)
        gen_kwargs = {"task": task}
        if lang:
            gen_kwargs["language"] = lang
        result = pipe(mono, generate_kwargs=gen_kwargs, return_timestamps="word")
        chunks = []
        for chunk in result.get("chunks") or []:
            start, end = _seg_times(chunk)
            chunks.append((start, end, (chunk.get("text") or "").strip()))
        return (result.get("text") or "").strip(), chunks

    if backend == "fast":
        _pipe = _get_pipe(backend, model, device)
        segments, _info = _pipe.transcribe(mono, language=lang, task=task, beam_size=5)
        chunks = []
        pieces = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                chunks.append((segment.start, segment.end, text))
                pieces.append(text)
        return "\n".join(pieces), chunks

    raise ValueError(f"Unknown backend: {backend!r}")


def _build_lines(chunks, fallback_text):
    """Split atomic transcript units into (start, end, text) lines, breaking on
    sentence punctuation and on pauses (works for sung audio)."""
    _SENT = tuple("!.?…;:")
    lines = []
    current = []
    cstart = None
    cend = None
    for start, end, text in chunks:
        text = (text or "").strip()
        if not text:
            continue
        if current and start is not None and cend is not None and (start - cend) > 0.5:
            lines.append((cstart, cend, " ".join(current)))
            current = []
            cstart = cend = None
        if not current:
            cstart = start
        current.append(text)
        cend = end
        if text.rstrip().endswith(_SENT):
            lines.append((cstart, cend, " ".join(current)))
            current = []
            cstart = cend = None
    if current:
        lines.append((cstart, cend, " ".join(current)))
    if not lines and fallback_text:
        lines.append((None, None, fallback_text))
    return lines


_FILLER_RE = re.compile(
    r"^(?:\s*(?:m[úu]sic\w*|music\w*|[♪♫]|la+|\b(?:la|ah|uh|ooh|oh|yeah|mm|um|na)\b)\s*)*$",
    re.IGNORECASE,
)


def _vocal_onsets(abc):
    """Return (sorted vocal-note onset times in seconds, bpm) mapped from a
    SheetSage ABC. On failure returns ([], None)."""
    try:
        from .score_analysis import inspect_score
        info = inspect_score(abc)
        bpm = info["bpm"]
        spt = 60.0 / (bpm * 256.0)
    except Exception:
        return [], None
    onsets = sorted(note["start"] * spt for note in info["roll"]["tracks"].get("Vocal", []))
    return onsets, bpm


def _phrase_times(onsets, bpm, gap_beats=1.5):
    """Group note onsets into sung phrases; return the start time of each
    phrase (an onset separated from the previous by > gap_beats beats)."""
    if not onsets:
        return []
    gap_s = gap_beats * 60.0 / bpm
    starts = [onsets[0]]
    prev = onsets[0]
    for onset in onsets[1:]:
        if onset - prev > gap_s:
            starts.append(onset)
        prev = onset
    return starts


def _split_at_phrases(atoms, phrase_times):
    """Split word/segment atoms into one (start,end,text) line per sung phrase
    boundary, using the SheetSage ABC phrase starts as the segmenter."""
    if not phrase_times:
        return []
    lines = []
    cur = []
    cstart = None
    cend = None
    for start, end, text in atoms:
        text = (text or "").strip()
        if not text:
            continue
        idx = bisect.bisect_right(phrase_times, start if start is not None else -1) - 1
        idx = max(0, idx)
        bound = phrase_times[idx]
        if cur and cstart is not None and bound != cstart:
            lines.append((cstart, cend, " ".join(cur)))
            cur = []
        if not cur:
            cstart = bound
        cur.append(text)
        cend = end
    if cur:
        lines.append((cstart, cend, " ".join(cur)))
    return lines


def _silence_spans(mono, min_gap=0.45):
    """Silent spans (start_s, end_s) of the 16k mono audio, based on RMS energy.
    These locate the sung regions vs the quiet/instrumental parts."""
    import numpy as np
    sr = 16000
    win = int(sr * 0.03)
    hop = int(sr * 0.01)
    n = len(mono)
    if n < win:
        return []
    rms = np.array([float(np.sqrt(np.mean(mono[s:s + win] ** 2))) for s in range(0, n - win + 1, hop)], dtype=np.float64)
    thr = max(float(np.median(rms)) * 0.4, 10 ** (-60 / 20))
    silent = rms < thr
    times = np.arange(len(rms)) * hop / float(sr)
    spans = []
    start = None
    for i in range(len(silent)):
        if silent[i] and start is None:
            start = i
        elif (not silent[i]) and start is not None:
            if times[i] - times[start] >= min_gap:
                spans.append((float(times[start]), float(times[i])))
            start = None
    if start is not None and times[-1] - times[start] >= min_gap:
        spans.append((float(times[start]), float(times[-1])))
    return spans


def _drop_silent(atoms, spans, margin=0.06):
    """Keep only transcript atoms that overlap actual vocal (non-silent) audio;
    drop atoms fully inside a silence gap."""
    if not spans:
        return atoms
    kept = []
    for start, end, text in atoms:
        if not (text or "").strip():
            continue
        if start is None or end is None:
            kept.append((start, end, text))
            continue
        inside = any((start - margin) >= a and (end + margin) <= b for (a, b) in spans)
        if not inside:
            kept.append((start, end, text))
    return kept


def _audio_cuts(mono, min_gap=0.45):
    """Detect silence gaps in the (16k mono) audio and return the midpoint
    second of each gap >= min_gap. These are the phrase/paragraph boundaries
    of an already-vocal-separated stem."""
    import numpy as np
    sr = 16000
    win = int(sr * 0.03)
    hop = int(sr * 0.01)
    n = len(mono)
    if n < win:
        return []
    rms = []
    for st in range(0, n - win + 1, hop):
        rms.append(float(np.sqrt(np.mean(mono[st:st + win] ** 2))))
    rms = np.array(rms, dtype=np.float64)
    thr = max(float(np.median(rms)) * 0.4, 10 ** (-60 / 20))
    silent = rms < thr
    if not silent.any():
        return []
    times = np.arange(len(rms)) * hop / float(sr)  # seconds
    cuts = []
    start = None
    for i in range(len(silent)):
        if silent[i] and start is None:
            start = i
        elif (not silent[i]) and start is not None:
            if times[i] - times[start] >= min_gap:
                cuts.append((times[start] + times[i]) / 2.0)
            start = None
    if start is not None and times[-1] - times[start] >= min_gap:
        cuts.append((times[start] + times[-1]) / 2.0)
    return cuts


def _segment(atoms, fallback_text, onsets, bpm):
    """Best-effort segmentation: prefer ABC phrase boundaries; else split by
    punctuation/pauses; drop instrumental-filler hallucination lines."""
    if onsets and bpm:
        phrase_times = _phrase_times(onsets, bpm)
        lines = _split_at_phrases(atoms, phrase_times)
        lines = [ln for ln in lines
                 if not (_snap_to_onset(ln[0], onsets, window=3.0) is None
                         and _FILLER_RE.fullmatch(ln[2] or ""))]
        if not lines:
            lines = _build_lines(atoms, fallback_text)
        return lines
    return _build_lines(atoms, fallback_text)


def _snap_to_onset(value, onsets, window=2.0):
    if value is None:
        return value
    best = None
    for o in onsets:
        if o < value - window:
            continue
        if o > value + window:
            break
        if best is None or abs(o - value) < abs(best - value):
            best = o
    return best


def _align_with_abc(lines, onsets):
    """Snap each line start to the nearest ABC vocal onset; drop instrumental
    filler hallucinations that have no vocal onset nearby."""
    if not onsets:
        return lines
    aligned = []
    for start, end, text in lines:
        snapped = _snap_to_onset(start, onsets)
        if snapped is None and _FILLER_RE.fullmatch(text or ""):
            continue
        aligned.append((snapped if snapped is not None else start, end, text))
    return aligned


class HZ3_YuE2_Transcribe:
    CATEGORY = "HZ3 YuE2"
    FUNCTION = "transcribe"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("lyrics", "segments", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Transcribe the input audio to lyrics. Backends: transformers Whisper (small/medium) or faster-whisper (fast). SheetSage keeps producing the ABC separately."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "backend": (BACKENDS, {"default": "whisper medium"}),
                "language": ("STRING", {"default": "auto", "tooltip": "auto detects the spoken language. Or force an ISO-639-1 code (es, en, ...)."}),
                "task": (["transcribe", "translate"], {"default": "transcribe"}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            },
        }

    def transcribe(self, audio, backend, language, task, device):
        text, chunks = _transcribe(backend, mono, language, task, device)

        # Read the audio's silences: drop transcript content that falls inside
        # non-vocal gaps so only actually-sung content is extracted.
        spans = _silence_spans(mono)
        chunks = _drop_silent(chunks, spans)
        cuts = [(a + b) / 2.0 for a, b in spans]  # phrase boundaries = silence midpoints

        if backend == "fast":
            # faster-whisper already returns per-phrase segments: keep them.
            lines = [(s, e, t) for s, e, t in chunks if (t or "").strip()]
        else:
            lines = _split_at_phrases(chunks, cuts) if cuts else []
            if not lines:
                lines = _build_lines(chunks, text)
        if not lines:
            lines = [(None, None, text)]

        lyrics = "\n".join(t for _, _, t in lines)
        segments = [{"start": _rounded(s), "end": _rounded(e), "text": t} for s, e, t in lines]
        report = (f"Transcribe · {backend} · {task} · {language or 'auto'} · "
                  f"{len(lines)} lines · silence_gaps: {len(spans)}")
        visible = "LYRICS (RAW)\n" + lyrics
        return {"ui": {"text": [visible]},
                "result": (lyrics, json.dumps(segments, ensure_ascii=False), report)}


# ---------------------------------------------------------------------------
# MixMash Genius (Ollama): sectioned lyrics + per-section style
# ---------------------------------------------------------------------------
GENIUS_SYSTEM = """You are a music editor that turns raw transcribed lyrics and an ABC score into a YuE2 cover plan.

You receive JSON with:
- lyrics: a raw transcript (may contain non-sung phrases, repeated lines, a chorus, ad-libs).
- instructions: free-text direction from the user. Follow it exactly, especially if it says which source drives a section or a specific build.
- score: measured BPM, meter, key and total seconds derived from the ABC.
- sections: the actual section list of the score with bars, seconds and vocal_notes. A section with vocal_notes == 0 is instrumental-only (no sung melody).

TASKS:
1. Rebuild lyrics into the score's section order with a [Section] label per block:
   - [Intro], [Verse 1], [Verse 2], [Chorus], ..., ending with [Outro].
   - Put an empty [Instrumental] block where a section has no sung melody, and keep unsung or off-melody lines (repeated chorus phrases, ad-libs) exactly as you found them.
   - Do not invent, translate or rewrite the Spanish words; only restructure and place them.
2. Produce ONE section-tagged style string, sections in the same order. For each [Section] write a short musical instruction (delivery, drums, bass, harmony, timbre, production) that applies the user's instructions to that section. Where reliable, use the measured BPM/meter/key. Keep the section naming the instructions/score uses.
3. abc is handled by the caller; never touch or return it.

Return JSON only with exactly these keys:
- "lyrics": the sectioned lyrics as a single string.
- "style": the single section-tagged style string.
No commentary, no markdown, no confidence values."""


def _score_evidence(abc):
    if not (abc or "").strip():
        return {"bpm": None, "meter": None, "key": None, "seconds": None,
                "sections": [], "note": "No ABC supplied."}
    try:
        info = inspect_score(abc)
    except Exception as exc:  # malformed ABC should not hard-fail the whole node
        return {"bpm": None, "meter": None, "key": None, "seconds": None,
                "sections": [], "parsing_error": str(exc)}
    roll = info["roll"]
    bar_ticks = int(roll["bar_ticks"])
    seconds_per_tick = 60.0 / info["bpm"] / 256.0
    vocal_tracks = roll.get("tracks", {}).get("Vocal", [])
    sections = []
    for sec in roll.get("sections", []):
        start_ticks = sec["start"] * bar_ticks
        end_ticks = (sec["start"] + sec["bars"]) * bar_ticks
        vocal_notes = sum(start_ticks <= note["start"] < end_ticks for note in vocal_tracks)
        sections.append({
            "name": sec["name"],
            "bars": sec["bars"],
            "seconds": round((end_ticks - start_ticks) * seconds_per_tick, 2),
            "vocal_notes": vocal_notes,
            "instrumental_only": vocal_notes == 0,
        })
    return {"bpm": info["bpm"], "meter": info["meter"], "key": info["key"],
            "seconds": round(info["seconds"], 2), "sections": sections,
            "note": None}


class HZ3_YuE2_MixMashGenius:
    CATEGORY = "HZ3 YuE2"
    FUNCTION = "compose"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("style", "lyrics", "abc", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Ollama agent: convert raw lyrics + ABC + user instructions into sectioned lyrics and a per-section style for YuE2. The ABC passes through unchanged."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "abc": ("STRING", {"forceInput": True, "multiline": True, "tooltip": "Connect the SheetSage2 ABC."}),
                "instructions": ("STRING", {"forceInput": True, "multiline": True, "tooltip": "Connect a text box with the direction to apply per section."}),
                "model": ("STRING", {"default": "deepseek-v4.1-flash:cloud"}),
                "endpoint": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "temperature": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05}),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 900}),
            }
        }

    def compose(self, lyrics, abc, instructions, model, endpoint, temperature, timeout):
        evidence = _score_evidence(abc)
        payload = {
            "language_locale": "es-MX",
            "lyrics": lyrics,
            "instructions": instructions,
            "score": {k: evidence[k] for k in ("bpm", "meter", "key", "seconds", "sections")},
            "output_contract": "{\"lyrics\": string, \"style\": string}",
        }
        result = _ollama_chat(GENIUS_SYSTEM, payload, model, endpoint, temperature, timeout)
        style = str(result.get("style", "")).replace("\r\n", "\n").strip()
        new_lyrics = str(result.get("lyrics", "")).replace("\r\n", "\n").strip()
        if not style or not new_lyrics:
            raise RuntimeError("MixMash Genius returned an incomplete answer (need style and lyrics).")
        report = json.dumps(result, ensure_ascii=False, indent=2)
        visible = ("SECTIONED LYRICS:\n" + new_lyrics +
                   "\n\nSECTIONED STYLE:\n" + style)
        if evidence.get("note"):
            visible += "\n\n" + evidence["note"]
        return {"ui": {"text": [visible]},
                "result": (style, new_lyrics, abc, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_Transcribe": HZ3_YuE2_Transcribe,
    "HZ3_YuE2_Whisper": HZ3_YuE2_Transcribe,  # backwards-compatible alias
    "HZ3_YuE2_MixMashGenius": HZ3_YuE2_MixMashGenius,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_Transcribe": "HZ3 YuE2 · Transcribe",
    "HZ3_YuE2_Whisper": "HZ3 YuE2 · Transcribe",
    "HZ3_YuE2_MixMashGenius": "HZ3 YuE2 · MixMash Genius (Ollama)",
}
