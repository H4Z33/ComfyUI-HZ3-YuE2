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


BACKENDS = ["whisper small", "whisper medium", "fast", "x"]


def _backend_model(backend):
    return {
        "whisper small": "openai/whisper-small",
        "whisper medium": "openai/whisper-medium",
        "fast": "small",
        "x": "small",
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
        pipe = _load_pipe(model, device)
        gen_kwargs = {"task": task}
        if lang:
            gen_kwargs["language"] = lang
        result = pipe(mono, generate_kwargs=gen_kwargs, return_timestamps=True)
        chunks = []
        for chunk in result.get("chunks") or []:
            start, end = _seg_times(chunk)
            chunks.append((start, end, (chunk.get("text") or "").strip()))
        return (result.get("text") or "").strip(), chunks

    if backend == "fast":
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError("'fast' (faster-whisper) is not installed. Run: pip install faster-whisper")
        _pipe = WhisperModel(model, device=device, compute_type=_compute_type(device))
        segments, _info = _pipe.transcribe(mono, language=lang, task=task, beam_size=5)
        chunks = []
        pieces = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                chunks.append((segment.start, segment.end, text))
                pieces.append(text)
        return "\n".join(pieces), chunks

    if backend == "x":
        try:
            import whisperx
        except ImportError:
            raise RuntimeError("'x' (WhisperX) is not installed. Run: pip install whisperx")
        _pipe = whisperx.load_model(model, device, compute_type=_compute_type(device))
        result = _pipe.transcribe(mono, batch_size=16, language=lang, task=task)
        chunks = []
        pieces = []
        for segment in result.get("segments") or []:
            text = (segment.get("text") or "").strip()
            if text:
                chunks.append((segment.get("start"), segment.get("end"), text))
                pieces.append(text)
        return "\n".join(pieces), chunks

    raise ValueError(f"Unknown backend: {backend!r}")


def _build_lines(chunks, fallback_text):
    lines = []
    current = []
    prev_end = None
    for start, end, text in chunks:
        if not text:
            continue
        if start is not None and prev_end is not None and start - prev_end > 0.35 and current:
            lines.append(" ".join(current))
            current = []
        current.append(text)
        prev_end = end
    if current:
        lines.append(" ".join(current))
    if not lines and fallback_text:
        lines.append(fallback_text)
    return lines


class HZ3_YuE2_Transcribe:
    CATEGORY = "HZ3 YuE2"
    FUNCTION = "transcribe"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("lyrics", "segments", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Transcribe the input audio to lyrics. Backends: transformers Whisper (small/medium), faster-whisper (fast) or WhisperX (x). SheetSage keeps producing the ABC separately."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "backend": (BACKENDS, {"default": "whisper medium"}),
                "language": ("STRING", {"default": "auto", "tooltip": "auto detects the spoken language. Or force an ISO-639-1 code (es, en, ...)."}),
                "task": (["transcribe", "translate"], {"default": "transcribe"}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            }
        }

    def transcribe(self, audio, backend, language, task, device):
        mono = _to_mono_16k(audio)
        text, chunks = _transcribe(backend, mono, language, task, device)
        lines = _build_lines(chunks, text)
        lyrics = "\n".join(lines)
        segments = [{"start": _rounded(s), "end": _rounded(e), "text": t} for s, e, t in chunks]
        report = (f"Transcribe · {backend} · {task} · {language or 'auto'} · "
                  f"{len(chunks)} segments · {len(lines)} lines")
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
