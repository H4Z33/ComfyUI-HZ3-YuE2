"""Split one audio into SECTION clips using the SheetSage2 structure, mapping lyrics.

Takes the SECTION STRUCTURE (real timestamps) from the companion
`HZ3 YuE2 · SheetSage2 Audio to ABC + Sections` node, cuts the audio at those
boundaries (intro/interlude/outro included), and assigns every faster-whisper line
to the section that contains its midpoint. Emits one AUDIO per section at its true
length (list output) plus a layout with each section's aligned lyrics.

This node only maps lyrics onto the structure; the audio cuts come from the real
section timestamps, so whisper is NOT used to place boundaries any more.
"""

from __future__ import annotations

import json

import torch


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
        start = item.get("start")
        end = item.get("end")
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            raise ValueError("Each segment needs numeric start and end fields.")
        text = str(item.get("text", "") or "").strip()
        if end <= start:
            raise ValueError(f"Segment has end <= start ({start}..{end}); fix the boundaries.")
        segments.append({"start": start, "end": end, "text": text})
    segments.sort(key=lambda segment: segment["start"])
    if not segments:
        raise ValueError("No valid segments were parsed from the segment payload.")
    return segments


def _parse_structure(data):
    """Return a sorted list of {name, start, end} from the SheetSage2 structure JSON."""
    if isinstance(data, str):
        payload = json.loads(data) if data.strip() else []
    else:
        payload = data
    if isinstance(payload, dict):
        payload = payload.get("sections") or payload.get("structure") or payload.get("clips") or []
    if not isinstance(payload, list):
        raise ValueError("structure must be a JSON list of {name, start, end} objects.")
    sections = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("start") is None or item.get("end") is None:
            continue
        sections.append({
            "name": str(item.get("name") or item.get("kind") or "section"),
            "start": float(item["start"]),
            "end": float(item["end"]),
        })
    sections.sort(key=lambda section: section["start"])
    sections = [section for section in sections if section["end"] - section["start"] > 1e-6]
    if not sections:
        raise ValueError("No valid sections were parsed from the structure payload.")
    return sections


def _coverage(sections, duration):
    """Ensure sections tile [0, duration]: prepend intro, append outro."""
    sections = [dict(section) for section in sections]
    for section in sections:
        section["start"] = max(0.0, min(duration, section["start"]))
        section["end"] = max(0.0, min(duration, section["end"]))
        section.setdefault("lines", [])
        section.setdefault("text", "")
    sections = [section for section in sections if section["end"] - section["start"] > 1e-6]
    if sections[0]["start"] > 1e-3:
        sections.insert(0, {"name": "intro", "start": 0.0,
                            "end": sections[0]["start"], "lines": [], "text": ""})
    if sections[-1]["end"] < duration - 1e-3:
        sections.append({"name": "outro", "start": sections[-1]["end"],
                         "end": duration, "lines": [], "text": ""})
    return sections


def _assign_lines(sections, segments):
    """Assign each whisper line to the section whose span contains its midpoint."""
    for section in sections:
        section["lines"] = []
        section["text"] = ""
    for segment in segments:
        mid = (segment["start"] + segment["end"]) / 2.0
        target = None
        for section in sections:
            if section["start"] - 1e-6 <= mid < section["end"] + 1e-6:
                target = section
                break
        if target is None:  # line outside any section -> nearest by start
            target = min(sections, key=lambda section: abs(segment["start"] - section["start"]))
        target["lines"].append(segment)
        target["text"] = " ".join(line["text"] for line in target["lines"])


class HZ3_YuE2_SplitAudioSegments:
    CATEGORY = "HZ3 YuE2/Audio"
    FUNCTION = "split"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("segments_audio", "layout", "report")
    OUTPUT_IS_LIST = (True, False, False)
    DESCRIPTION = (
        "Cut one audio into SECTION clips using the structure (real timestamps) from "
        "'SheetSage2 Audio to ABC + Sections', and assign each faster-whisper line to "
        "its section. Emits one AUDIO per section (list) with the aligned lyrics in the "
        "layout."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "segments": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "JSON list of {start, end, text} from faster-whisper.",
                }),
                "structure": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "JSON structure from 'SheetSage2 Audio to ABC + Sections' (list of {name, start, end} with real times).",
                }),
            },
        }

    def split(self, audio, segments, structure):
        waveform = audio["waveform"]          # [batch, channels, samples]
        sample_rate = int(audio["sample_rate"])
        if waveform.shape[0] != 1:
            raise ValueError("Expected a single audio clip (batch of 1) to split.")
        channels = waveform.shape[1]
        total = waveform.shape[-1]
        duration_seconds = total / float(sample_rate)

        parsed = _parse_segments(segments)
        sections = _coverage(_parse_structure(structure), duration_seconds)
        _assign_lines(sections, parsed)

        clips = []          # list of AUDIO dicts, each at its true length
        layout = []
        for index, section in enumerate(sections):
            i0 = max(0, round(section["start"] * sample_rate))
            i1 = min(total, round(section["end"] * sample_rate))
            if i1 <= i0:
                continue
            clip = waveform[:, :, i0:i1]
            clips.append({"waveform": clip, "sample_rate": sample_rate})
            layout.append({
                "index": len(clips) - 1,
                "name": section["name"],
                "start": round(section["start"], 3),
                "end": round(section["end"], 3),
                "duration": round(section["end"] - section["start"], 3),
                "text": section["text"],
                "vocal_notes": len([1 for line in section["lines"] if line["text"]]),
            })
        if not clips:
            raise ValueError("None of the sections produced an audible audio clip.")

        layout_json = json.dumps({"sections": layout}, ensure_ascii=False, indent=2)
        report = (
            f"Split audio ({duration_seconds:.2f} s) into {len(clips)} section(s): "
            + "\n".join(
                f"  [{j['index']}] {j['name']} {j['start']:.2f}-{j['end']:.2f}s "
                f"({j['duration']:.2f}s) | {j['text'][:60]}"
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
