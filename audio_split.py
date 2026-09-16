"""Split one audio clip into musical-section clips from whisper segments + ABC.

Boundaries always come from the REAL whisper timings (reliable). Sections are then
grouped into vocal "runs" separated by instrumental gaps (>= min_gap seconds), the
lead-in before the first vocal line becomes its own INTRO clip, and any trailing
non-vocal tail becomes an OUTRO/INSTRUMENTAL clip. The output is packed into a
single batched AUDIO (one clip per batch index).

The optional `abc` adds SECTION NAMES: whenever the ABC has exactly as many vocal
sections as there are vocal runs, each run is labelled *verse/chorus/...* from the
ABC; otherwise runs are labelled "vocal N". The parsed ABC section list (names and
bar shares) is also echoed in the layout so you can see the structure.

Note: the ABC text alone cannot give exact real-time section boundaries (sections
are reconstructed as bars * bpm and inspect_score drops intro/instrumental-only
sections). Those cuts therefore come from whisper/energy, not from the ABC.
"""

from __future__ import annotations

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
    """Return [(name, bars)] vocal section list from the ABC, or None.

    Uses inspect_score's roll sections (vocal-only, tiled over the vocal span) so
    the share of each section is known even though real times are not.
    """
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


def _build_slots(segments, duration_seconds, abc, separate_intro, min_gap):
    """Return (slots, abc_sections).

    slots: list of (start, end, kind, name, text) in time order, contiguous.
    kinds: intro / vocal / instrumental (also used for trailing tail).
    """
    abc_sections = _abc_vocal_sections(abc)

    # Vocal runs: consecutive lines separated by < min_gap belong to the same run;
    # a gap >= min_gap closes the run (an instrumental section separates them).
    runs = [[segments[0]]]
    for segment in segments[1:]:
        if segment["start"] - runs[-1][-1]["end"] >= min_gap:
            runs.append([])
        runs[-1].append(segment)

    vocal_items = []  # (start, end, name, text)
    for index, run in enumerate(runs):
        start = run[0]["start"]
        end = run[-1]["end"]
        text = " ".join(segment["text"] for segment in run)
        name = "vocal"
        if abc_sections and index < len(abc_sections):
            name = abc_sections[index][0]
        else:
            name = f"vocal {index + 1}"
        vocal_items.append((start, end, name, text))
    vocal_items.sort(key=lambda item: item[0])

    if not separate_intro and vocal_items:
        # Merge the intro into the first vocal clip (its start becomes 0).
        vocal_items[0] = (0.0, vocal_items[0][1], vocal_items[0][2], vocal_items[0][3])

    slots = []
    cursor = 0.0
    first = True
    for start, end, name, text in vocal_items:
        if first and start > 1e-6:
            # Lead-in before the first vocal line is always its own INTRO clip
            # (independent of min_gap), unless separate_intro merged it away.
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
    elif cursor < duration_seconds:
        # Small tail: fold into the last slot so nothing is dropped.
        if slots:
            slots[-1] = (slots[-1][0], duration_seconds, slots[-1][2], slots[-1][3])
        else:
            slots.append((cursor, duration_seconds, "vocal", "vocal", ""))

    # Keep only positive-length slots.
    slots = [slot for slot in slots if slot[1] - slot[0] > 1e-6]
    return slots, abc_sections


class HZ3_YuE2_SplitAudioSegments:
    CATEGORY = "HZ3 YuE2/Audio"
    FUNCTION = "split"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("segments_audio", "layout", "report")
    DESCRIPTION = (
        "Slice one audio into musical-section clips using REAL whisper timings. The "
        "lead-in becomes an intro clip, vocal lines are grouped into runs, and gaps "
        ">= min_gap become instrumental clips. Optional abc labels the runs by section "
        "name. Outputs one batched AUDIO (one clip per batch index) for a single "
        "SheetSage2 audio-to-ABC pass."
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
                    "tooltip": "Whisper gap (s) in a vocal run used to close it and emit an instrumental clip.",
                }),
            },
            "optional": {
                "abc": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Optional ABC (e.g. from SheetSage2). Used to name the vocal runs by section; the parsed section list is echoed in layout.",
                }),
            },
        }

    def split(self, audio, segments, separate_intro=True, min_gap=4.0, abc=""):
        waveform = audio["waveform"]          # [batch, channels, samples]
        sample_rate = int(audio["sample_rate"])
        if waveform.shape[0] != 1:
            raise ValueError("Expected a single audio clip (batch of 1) to split.")
        channels = waveform.shape[1]
        total = waveform.shape[-1]
        duration_seconds = total / float(sample_rate)

        parsed = _parse_segments(segments)
        slots, abc_sections = _build_slots(parsed, duration_seconds, abc,
                                           separate_intro, min_gap)

        clips = []
        layout = []
        for index, (start, end, kind, name, text) in enumerate(slots):
            i0 = max(0, round(start * sample_rate))
            i1 = min(total, round(end * sample_rate))
            if i1 <= i0:
                continue
            clip = waveform[:, :, i0:i1]      # [1, channels, n_samples]
            clips.append(clip)
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

        # ABC structure echo (informative; boundaries come from whisper).
        if abc_sections:
            layout_meta = {
                "abc_sections": [{"name": name, "bars": bars} for name, bars in abc_sections],
            }
        else:
            layout_meta = {}

        max_samples = max(clip.shape[-1] for clip in clips)
        batch = len(clips)
        output = torch.zeros(
            (batch, channels, max_samples),
            dtype=waveform.dtype,
            device=waveform.device,
        )
        for index, clip in enumerate(clips):
            output[index:index + 1, :, : clip.shape[-1]].copy_(clip)

        payload = {"clips": layout, **layout_meta}
        layout_json = json.dumps(payload, ensure_ascii=False, indent=2)
        report = (
            f"Split audio ({duration_seconds:.2f} s) into {batch} clip(s): "
            + "\n".join(
                f"  [{j['index']}] {j['kind']}:{j['name']} {j['start']:.2f}-{j['end']:.2f}s "
                f"({j['duration']:.2f}s)"
                for j in layout
            )
        )
        result_audio = {"waveform": output, "sample_rate": sample_rate}
        return {"ui": {"text": [layout_json]}, "result": (result_audio, layout_json, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_SplitAudioSegments": HZ3_YuE2_SplitAudioSegments,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_SplitAudioSegments": "HZ3 YuE2 · Split Audio by Segments",
}
