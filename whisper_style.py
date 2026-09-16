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
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

import folder_paths
import comfy.model_management as _mm

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


def _check_interrupt():
    """Honor ComfyUI's Cancel/Stop (throws if the UI interrupt flag is set)."""
    _mm.throw_exception_if_processing_interrupted()


def _interruptible(callable_fn):
    """Run a blocking call in a daemon thread and poll ComfyUI's interrupt flag,
    so Cancel/Stop returns immediately instead of hanging on a C-library call."""
    box = {}

    def _run():
        try:
            box["value"] = callable_fn()
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    while thread.is_alive():
        _check_interrupt()
        time.sleep(0.05)
    if "error" in box:
        raise box["error"]
    return box["value"]


BACKENDS = ["whisper small", "whisper medium", "whisper large", "fast"]

# module-level pipeline cache so a new node instance / re-run does not reload
# the model every execution
_PIPE_CACHE = {}


def _get_pipe(backend, model, device):
    key = (backend, model, device)
    if key not in _PIPE_CACHE:
        if backend in ("whisper small", "whisper medium", "whisper large"):
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
        "whisper large": "openai/whisper-large-v3",
        "fast": "small",
    }[backend]


def _compute_type(device):
    return "int8" if device == "cpu" else "float16"


def _transcribe(backend, mono, language, task, device, beam_size=5):
    """Run one backend and return (full_text, [(start, end, text), ...])."""
    lang = None
    if language and language.strip().lower() not in ("auto", ""):
        lang = language.strip().lower()
    model = _backend_model(backend)
    beams = max(1, int(beam_size))

    if backend in ("whisper small", "whisper medium", "whisper large"):
        # Transformers Whisper pipeline
        pipe = _get_pipe(backend, model, device)
        gen_kwargs = {"task": task, "num_beams": beams}
        if lang:
            gen_kwargs["language"] = lang
        result = _interruptible(lambda: pipe(mono, generate_kwargs=gen_kwargs, return_timestamps="word"))
        chunks = []
        for chunk in result.get("chunks") or []:
            start, end = _seg_times(chunk)
            chunks.append((start, end, (chunk.get("text") or "").strip()))
        return (result.get("text") or "").strip(), chunks

    if backend == "fast":
        _pipe = _get_pipe(backend, model, device)
        # VAD is OFF by default; enabling it skips the non-speech intro (so it
        # can't hallucinate a fake first segment) and splits on real speech.
        segments, _info = _pipe.transcribe(
            mono, language=lang, task=task, beam_size=beams,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        chunks = []
        pieces = []
        for segment in segments:
            _check_interrupt()
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


def _first_voice(mono_len_s, spans):
    """Start time of the first actually-vocal region (after an instrumental
    lead-in), derived from the silence spans."""
    if not spans:
        return 0.0
    if spans[0][0] <= 0.25:  # audio starts silent (intro) -> vocals begin after it
        return spans[0][1]
    return 0.0


def _drop_preshape(atoms, first_voice, margin=0.1):
    """Drop transcript atoms entirely before the first real vocal onset (i.e.
    hallucinated content from the instrumental lead-in)."""
    if first_voice <= 0.15:
        return atoms
    kept = []
    for start, end, text in atoms:
        if start is None or end is None:
            kept.append((start, end, text))
            continue
        if end <= first_voice + margin:
            continue
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
                "backend": (BACKENDS, {"default": "fast"}),
                "language": ("STRING", {"default": "auto", "tooltip": "auto detects the spoken language. Or force an ISO-639-1 code (es, en, ...)."}),
                "task": (["transcribe", "translate"], {"default": "transcribe"}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
                "beam_size": ("INT", {"default": 5, "min": 1, "max": 20, "tooltip": "Beam search width. Higher = more accurate but slower."}),
            },
            "optional": {
                "force_rerun": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off (default): the node caches normally and only re-transcribes when an input changes. On: force re-transcription on every run.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # force_rerun On -> always re-transcribe (nan != nan). Off (default) ->
        # normal caching: only re-run when an input actually changes.
        if kwargs.get("force_rerun"):
            return float("nan")
        return None

    def transcribe(self, audio, backend, language, task, device, force_rerun=False, beam_size=5):
        mono = _to_mono_16k(audio)
        text, chunks = _transcribe(backend, mono, language, task, device, beam_size)

        # Read the audio's silences: drop transcript content that falls inside
        # non-vocal gaps so only actually-sung content is extracted.
        spans = _silence_spans(mono)
        chunks = _drop_silent(chunks, spans)
        fv = _first_voice(len(mono) / 16000.0, spans)
        chunks = _drop_preshape(chunks, fv)
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
GENIUS_SYSTEM = """You are a music editor that turns an audio section structure + real lyrics + a user direction into a per-section YuE2 cover plan.

You receive JSON with:
- structure: the AUTHORITATIVE ordered list of sections, each {name, start, end} (seconds). Use exactly this order and count; never add, drop, or reorder sections.
- lyrics: the real lyrics, each line with its whisper timestamps {start, end, text}.
- instructions: the user's free-text direction. Apply it across the song, building in intensity from start to end (soft/acoustic early, fuller/louder at the choruses and the end, etc.).
- score: measured BPM, meter, key and total seconds (may be empty if no score).

TASKS:
1. Place every lyric line into the structure section whose [start, end) span contains its timestamp. Sections with no lines (intro, interlude, outro, instrumental) get empty lyrics.
2. For EVERY section (including intro/interlude/outro), write ONE concise style paragraph that explicitly states:
   - tempo/BPM (use score.bpm when present, otherwise describe the beat feel),
   - music genre / style and how the instruments are used,
   - the VOICE (singer gender, age, tone and delivery),
   - instrumentation, harmony, production and dynamics,
   - how intensity builds toward the choruses and the final section.
3. Return one entry per structure section, in the same order.

Return JSON only with the single key:
- "sections": [ { "name": <structure name>, "lyrics": <lines for this section or "">, "style": <the paragraph> } ]
No commentary, no markdown, no extra keys."""


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


# Seconds of instrumentation shared across an interface when the overlap SWITCH is ON.
_OVERLAP_SECONDS = 4.0


def _parse_structure(data):
    """Parse the structure JSON (from SheetSage2 Audio to ABC + Sections) into a
    sorted list of {name, start, end}. Accepts a bare list or a dict with 'sections'."""
    if isinstance(data, str):
        payload = json.loads(data) if data.strip() else []
    else:
        payload = data
    if isinstance(payload, dict):
        payload = payload.get("sections") or payload.get("structure") or payload.get("clips") or []
    if not isinstance(payload, list):
        raise ValueError("`structure` must be a JSON list of {name, start, end} objects.")
    sections = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("start") is None or item.get("end") is None:
            continue
        sections.append({
            "name": str(item.get("name") or item.get("kind") or "section"),
            "start": round(float(item["start"]), 3),
            "end": round(float(item["end"]), 3),
        })
    sections.sort(key=lambda s: s["start"])
    sections = [s for s in sections if s["end"] - s["start"] > 1e-6]
    if not sections:
        raise ValueError("No valid sections were parsed from `structure`.")
    return sections


class HZ3_YuE2_MixMashGenius:
    CATEGORY = "HZ3 YuE2"
    FUNCTION = "compose"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("style", "lyrics", "durations", "segments_abc", "added_seconds", "report")
    OUTPUT_IS_LIST = (True, True, True, True, True, False)
    OUTPUT_NODE = True
    DESCRIPTION = ("MixMash Genius: per-section style + lyrics from abc + structure + "
                   "instructions. The `overlap` SWITCH applies the two-sided instrumental "
                   "overlap across sections (grows segments_abc, corrects durations, and "
                   "reports the per-section traslape for the Conditioning Overlap node). "
                   "OFF: no overlap. style[i]/lyrics[i]/durations[i]/segments_abc[i]/added_seconds[i] are index-aligned.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc": ("STRING", {"forceInput": True, "multiline": True, "tooltip": "The whole-song ABC from 'HZ3 YuE2 · SheetSage2 Audio to ABC + Sections'. Split into per-section fragments internally."}),
                "overlap": ("BOOLEAN", {"default": True,
                                        "tooltip": "SWITCH: ON applies the two-sided instrumental overlap across sections (grows segments_abc and corrects durations). OFF: no overlap, segments_abc and durations left unchanged."}),
                "structure": ("STRING", {"forceInput": True, "multiline": True, "tooltip": "Connect `structure` from 'HZ3 YuE2 · SheetSage2 Audio to ABC + Sections'."}),
                "instructions": ("STRING", {"forceInput": True, "multiline": True, "tooltip": "Direction to apply, e.g. 'start soft/acoustic and build to a rock finale with an older, raspy male voice'."}),
                "lyrics": ("STRING", {"multiline": True, "default": "", "tooltip": "Real lyrics (the split's per-section text or the raw transcript)."}),
                "model": ("STRING", {"default": "deepseek-v4.1-flash:cloud"}),
                "endpoint": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "temperature": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05}),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 900}),
            },
        }

    def compose(self, abc, overlap, structure, instructions, lyrics, model, endpoint,
                temperature, timeout):
        from .overlap_sections import apply_abc_overlap
        from .sheetsage2_sections import split_abc_by_sections

        def first(value, default=""):
            if value is None:
                return default
            if isinstance(value, (list, tuple)):
                return value[0] if value else default
            return value

        abc = first(abc, "")
        structure = first(structure, "")
        instructions = first(instructions, "")
        lyrics = first(lyrics, "")
        model = str(first(model, "deepseek-v4.1-flash:cloud"))
        endpoint = str(first(endpoint, "http://127.0.0.1:11434"))
        overlap_on = bool(first(overlap, False))
        try:
            temperature = float(first(temperature, 0.35))
        except (TypeError, ValueError):
            temperature = 0.35
        try:
            timeout = int(first(timeout, 180))
        except (TypeError, ValueError):
            timeout = 180

        evidence = _score_evidence(abc)
        sections = _parse_structure(structure)
        segments = [fragment for _, fragment in split_abc_by_sections(abc)] if (abc or "").strip() else []
        payload = {
            "language_locale": "es-MX",
            "structure": sections,
            "segments_abc": segments,
            "lyrics": lyrics,
            "instructions": instructions,
            "score": {k: evidence[k] for k in ("bpm", "meter", "key", "seconds")},
            "output_contract": '{"sections": [{"name": string, "lyrics": string, "style": string}]}',
        }
        result = _interruptible(lambda: _ollama_chat(GENIUS_SYSTEM, payload, model, endpoint,
                                                    temperature, timeout))
        result_sections = result.get("sections")
        if not isinstance(result_sections, list) or not result_sections:
            raise RuntimeError("MixMash Genius did not return a 'sections' list.")

        styles = []
        lyric_blocks = []
        orig_durations = []
        for index, sec in enumerate(sections):
            item = result_sections[index] if index < len(result_sections) and isinstance(result_sections[index], dict) else {}
            style = str(item.get("style", "") or "").strip().replace("\n", " ")
            block_lyrics = str(item.get("lyrics", "") or "").strip()
            styles.append(style if style else f"[{sec['name']}] (style not provided)")
            lyric_blocks.append(f"[{sec['name']}]" + (f"\n{block_lyrics}" if block_lyrics else ""))
            orig_durations.append(round(float(sec["end"]) - float(sec["start"]), 3))

        # Overlap SWITCH: ON applies the two-sided instrumental overlap; OFF leaves unchanged.
        if overlap_on and segments:
            overlapped, added = apply_abc_overlap(segments, _OVERLAP_SECONDS)
        else:
            overlapped = list(segments)
            added = [0.0] * len(segments)
        durations = [
            round(orig_durations[i] + (added[i] if i < len(added) else 0.0), 3)
            for i in range(len(orig_durations))
        ]

        report = json.dumps({
            "ollama": result,
            "conditioning_overlap": {
                "overlap_enabled": bool(overlap_on),
                "overlap_amount": _OVERLAP_SECONDS,
                "overlap_seconds_per_section": added,
                "section_durations": durations,
                "sections": [sec["name"] for sec in sections],
            },
        }, ensure_ascii=False, indent=2)
        visible = (
            "PER-SECTION STYLE (style[0..N-1]):\n" +
            "\n\n".join(f"{sections[i]['name']}:\n{styles[i]}" for i in range(len(sections))) +
            "\n\nSECTIONED LYRICS (lyrics[0..N-1]):\n" + "\n\n".join(lyric_blocks) +
            "\n\nDURATIONS (corrected):\n" + ", ".join(str(d) for d in durations) +
            "\n\nOVERLAP PER SECTION (s) [for Conditioning Overlap]:\n" +
            ", ".join(str(a) for a in added))
        return {"ui": {"text": [visible]},
                "result": (styles, lyric_blocks, durations, overlapped, added, report)}


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
