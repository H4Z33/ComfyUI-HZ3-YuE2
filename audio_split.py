"""Split one audio into musical-section clips from whisper timings + optional ABC.

Each clip keeps its TRUE real length (no padding) and is emitted as a LIST of
AUDIO (one per output socket) so short intro/instrumental clips stay short.

Boundaries always come from the REAL whisper timings. Vocal lines are grouped into
runs separated by gaps >= min_gap; the lead-in becomes its own INTRO clip, interior
gaps and the trailing tail become INSTRUMENTAL clips. When `abc` is supplied and
`subdivide` is on, each vocal run is further split into the ABC's vocal sections
(verse/chorus/...) by their bar shares, labelling each piece.

Note: the ABC never provides exact real times (sections are reconstructed as
bars*bpm and inspect_score drops intro/instrumental-only sections), so cuts are
from whisper; the ABC only names/subdivides the vocal runs.
"""

from __future__ import annotations

import bisect
import json

import torch


def _as_seconds(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_segments(data):
    """Return a sorted list of {start, end, text} from a JSON segment payload."""
    if isinstance(data, str):
        payload = json.loads(data) if data.strip() else []
    else:
        payload = data
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = payload.get("segments") or payload.get("results") or []
    else:
        raise ValueError("segments must be a JSON list of {start, end, text} objects.")
    if not isinstance(raw, list):
        raise ValueError("segments must be a JSON list of {start, end, text} objects.")

    segments = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = _as_seconds(item.get("start"), None)
        end = _as_seconds(item.get("end"), None)
        if start is None or end is None:
            raise ValueError("Each segment needs numeric start and end fields.")
        text = str(item.get("text", "") or "").strip()
        if end <= start:
            raise ValueError(f"Segment has end <= start ({start}..{end}); fix the boundaries.")
        segments.append({"start": start, "end": end, "text": text})
    segments.sort(key=lambda segment: segment["start"])
    if not segments:
        raise ValueError("No valid segments were parsed from the segment payload.")
    return segments


def _abc_vocal_sections(abc):
    """Return [(name, bars)] vocal section list from the ABC, or None."""
    if not (abc or "").strip():
        return None
    try:
        from .score_analysis import inspect_score
        info = inspect_score(abc)
        sections = info["roll"]["sections"]
        if not sections or sum(int(section["bars"]) for section in sections) <= 0:
            return None
        return [(str(section["name"] or "section"), int(section["bars"]))
                for section in sections]
    except Exception:
        return None


def _vocal_items(runs, abc_sections, subdivide):
    """Build [(start, end, name, text)] vocal clips from the runs.

    With subdivision, a run is cut at the ABC section bands (by bar share) and
    whole whisper lines are assigned to the section containing their midpoint.
    """
    items = []
    for run_index, run in enumerate(runs):
        run_start = run[0]["start"]
        run_end = run[-1]["end"]
        run_span = run_end - run_start
        text = " ".join(segment["text"] for segment in run)

        if subdivide and abc_sections and run_span > 0:
            total_bars = sum(bars for _, bars in abc_sections)
            boundaries = []
            cumulative = 0.0
            for name, bars in abc_sections:
                cumulative += bars / total_bars if total_bars else 0.0
                boundaries.append(cumulative)
            bands = [[] for _ in abc_sections]
            for segment in run:
                mid = (segment["start"] + segment["end"]) / 2.0
                frac = (mid - run_start) / run_span
                idx = bisect.bisect_right(boundaries, frac)
                bands[min(idx, len(bands) - 1)].append(segment)
            for band_index, band in enumerate(bands):
                if not band:
                    continue
                name = abc_sections[band_index][0]
                band_text = " ".join(segment["text"] for segment in band)
                items.append((band[0]["start"], band[-1]["end"], name, band_text))
        else:
            if abc_sections and run_index < len(abc_sections):
                name = abc_sections[run_index][0]
            else:
                name = f"vocal {run_index + 1}"
            items.append((run_start, run_end, name, text))
    return items


def _build_slots(segments, duration_seconds, abc, separate_intro, min_gap, subdivide):
    """Return (slots, abc_sections). slots: (start, end, kind, name, text) in order."""
    abc_sections = _abc_vocal_sections(abc)

    # Vocal runs: a gap >= min_gap closes the run (an instrumental section).
    runs = [[segments[0]]]
    for segment in segments[1:]:
        if segment["start"] - runs[-1][-1]["end"] >= min_gap:
            runs.append([])
        runs[-1].append(segment)

    vocal_items = _vocal_items(runs, abc_sections, subdivide)
    vocal_items.sort(key=lambda item: item[0])

    if not separate_intro and vocal_items:
        vocal_items[0] = (0.0, vocal_items[0][1], vocal_items[0][2], vocal_items[0][3])

    slots = []
    cursor = 0.0
    first = True
    for start, end, name, text in vocal_items:
        if first and start > 1e-6:
            slots.append((0.0, min(start, duration_seconds), "intro", "intro", ""))
        elif start - cursor >= min_gap:
            slots.append((cursor, min(start, duration_seconds), "instrumental", "instrumental", ""))
        if start < duration_seconds:
            slots.append((min(start, duration_seconds), min(end, duration_seconds),
                          "vocal", name, text))
        first = False
        cursor = min(max(start, end, cursor), duration_seconds)
    if duration_seconds - cursor >= min_gap:
        slots.append((cursor, duration_seconds, "instrumental", "instrumental", ""))
    elif cursor < duration_seconds and slots:
        slots[-1] = (slots[-1][0], duration_seconds, slots[-1][2], slots[-1][3])

    slots = [slot for slot in slots if slot[1] - slot[0] > 1e-6]
    return slots, abc_sections


class HZ3_YuE2_SplitAudioSegments:
    CATEGORY = "HZ3 YuE2/Audio"
    FUNCTION = "split"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("segments_audio", "layout", "report")
    OUTPUT_IS_LIST = (True, False, False)
    DESCRIPTION = (
        "Slice one audio into musical-section clips from REAL whisper timings, emitted "
        "as a LIST of AUDIO with each clip at its TRUE length (no padding). Lead-in "
        "becomes an intro clip, vocal lines group into runs, gaps >= min_gap become "
        "instrumental clips. Optional abc labels and subdivides the vocal runs by "
        "section (verse/chorus/...). Use Get Batch Item to pull one clip by index."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "segments": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "JSON list of {start, end, text} from faster-whisper (or wrap under 'segments').",
                }),
                "separate_intro": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True: the lead-in before the first vocal line becomes its own INTRO clip. False: it merges into the first clip.",
                }),
                "min_gap": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 120.0,
                    "step": 0.5,
                    "tooltip": "Whisper gap (s) between lines that closes a vocal run and emits an instrumental clip.",
                }),
                "subdivide": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When abc is provided, subdivide each vocal run into the ABC's section bands (verse/chorus/...).",
                }),
            },
            "optional": {
                "abc": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Optional ABC (e.g. from SheetSage2). Used to label and subdivide the vocal runs; the parsed section list is echoed in layout.",
                }),
            },
        }

    def split(self, audio, segments, separate_intro=True, min_gap=4.0, subdivide=True, abc=""):
        waveform = audio["waveform"]          # [batch, channels, samples]
        sample_rate = int(audio["sample_rate"])
        if waveform.shape[0] != 1:
            raise ValueError("Expected a single audio clip (batch of 1) to split.")
        channels = waveform.shape[1]
        total = waveform.shape[-1]
        duration_seconds = total / float(sample_rate)

        parsed = _parse_segments(segments)
        slots, abc_sections = _build_slots(parsed, duration_seconds, abc,
                                           separate_intro, min_gap, subdivide)

        clips = []          # list of AUDIO dicts, each at its true length
        layout = []
        for index, (start, end, kind, name, text) in enumerate(slots):
            i0 = max(0, round(start * sample_rate))
            i1 = min(total, round(end * sample_rate))
            if i1 <= i0:
                continue
            clip = waveform[:, :, i0:i1]      # [1, channels, n_samples], true length
            clips.append({"waveform": clip, "sample_rate": sample_rate})
            layout.append({
                "index": len(clips) - 1,
                "kind": kind,
                "name": name,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round((end - start), 3),
                "text": text,
            })
        if not clips:
            raise ValueError("None of the segments produced an audible audio clip.")

        payload = {"clips": layout}
        if abc_sections:
            payload["abc_sections"] = [{"name": name, "bars": bars} for name, bars in abc_sections]
        layout_json = json.dumps(payload, ensure_ascii=False, indent=2)
        report = (
            f"Split audio ({duration_seconds:.2f} s) into {len(clips)} true-length clip(s): "
            + "\n".join(
                f"  [{j['index']}] {j['kind']}:{j['name']} {j['start']:.2f}-{j['end']:.2f}s "
                f"({j['duration']:.2f}s)"
                for j in layout
            )
        )
        return {"ui": {"text": [layout_json]}, "result": (clips, layout_json, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_SplitAudioSegments": HZ3_YuE2_SplitAudioSegments,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_SplitAudioSegments": "HZ3 YuE2 · Split Audio by Segments",
}
