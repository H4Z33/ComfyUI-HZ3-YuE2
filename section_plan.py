"""Section-aware planning and multi-take conditioning assembly for YuE2 covers.

SectionPlan parses the native two-voice ABC into an ordered section timeline
(bars, seconds, semantic frames) with optional per-section style overrides, so a
cover can be laid out and generated section by section.

AssembleSections is the complement: it concatenates one conditioning take per
section into a single contiguous YuE2 conditioning, rebasing chunk KV offsets
and frame timestamps so the result feeds directly into a KSampler via the usual
yue2_frames / EmptyYuE2LatentAudio plumbing.
"""

from __future__ import annotations

import copy
import json
import re

import torch

from comfy.text_encoders.yue2 import FRAMES_PER_SECOND

from .conditioning_edit import _entry
from .score_analysis import inspect_score


# Localized section-name normalization used to match [Section] labels against
# style_overrides lines and lyric headers. Mirrors procedural_prosody's aliases.
_ALIASES = {
    "coro": "chorus", "estribillo": "chorus", "verso": "verse", "estrofa": "verse",
    "intro": "intro", "interludio": "interlude", "puente": "bridge", "salida": "outro",
}


def _normalize_section(value):
    value = re.sub(r"\s+\d+\s*$", "", (value or "").strip().lower())
    # Keep trailing numerals intact for repeated tags like [Verse 2].
    return _ALIASES.get(value, value)


def _section_timeline(score_abc):
    info = inspect_score(score_abc)
    roll = info["roll"]
    sections = roll.get("sections", [])
    if not sections:
        raise ValueError("The ABC has no named sections (% comment). Add sections to build a plan.")
    seconds_per_tick = 60.0 / info["bpm"] / 256.0
    bar_ticks = int(roll["bar_ticks"])
    timeline = []
    cumulative_frames = 0
    for index, section in enumerate(roll["sections"]):
        start_ticks = section["start"] * bar_ticks
        duration_ticks = section["bars"] * bar_ticks
        duration_seconds = float(duration_ticks) * seconds_per_tick
        start_seconds = float(start_ticks) * seconds_per_tick
        end_seconds = start_seconds + duration_seconds
        start_frame = round(start_seconds * FRAMES_PER_SECOND)
        end_frame = round(end_seconds * FRAMES_PER_SECOND)
        timeline.append({
            "instance": index + 1,
            "name": _normalize_section(section["name"]),
            "label": section["name"],
            "bars": section["bars"],
            "start_seconds": round(start_seconds, 3),
            "duration_seconds": round(duration_seconds, 3),
            "end_seconds": round(end_seconds, 3),
            "start_frame": start_frame,
            "duration_frames": end_frame - start_frame,
            "end_frame": end_frame,
        })
        cumulative_frames += end_frame - start_frame
    return info, timeline, cumulative_frames


def _parse_overrides(text):
    """Return a {normalized_name: override} map from 'Section: text' lines."""
    overrides = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(":")
        name = _normalize_section(name)
        if not name or not value.strip():
            raise ValueError(f"Invalid style override line: {raw!r}")
        overrides[name] = value.strip()
    return overrides


def _render_plan(timeline, overrides, info, total_frames):
    lines = ["# YuE2 section plan",
             f"BPM {info['bpm']} · {info['meter']} · key of {info['key']} · "
             f"{total_frames / FRAMES_PER_SECOND:.2f} s · {FRAMES_PER_SECOND} frames/s"]
    for section in timeline:
        frame_info = f"{section['start_frame']}:{section['end_frame']}"
        name = section["label"]
        if section["name"] in overrides:
            hint = overrides[section["name"]]
            lines.append(
                f"{name} ({section['bars']} bars) {section['start_seconds']:.2f}-"
                f"{section['end_seconds']:.2f}s [{frame_info}] -> {hint}"
            )
        else:
            lines.append(
                f"{name} ({section['bars']} bars) {section['start_seconds']:.2f}-"
                f"{section['end_seconds']:.2f}s [{frame_info}]"
            )
    return "\n".join(lines)


def _lyric_per_section(info):
    roll = info["roll"]
    tracks = roll.get("tracks", {}).get("Vocal", [])
    bar_ticks = int(roll["bar_ticks"])
    result = {}
    for section in roll.get("sections", []):
        start_tick = section["start"] * bar_ticks
        end_tick = (section["start"] + section["bars"]) * bar_ticks
        result[_normalize_section(section["name"])] = sum(
            start_tick <= note["start"] < end_tick for note in tracks
        )
    return result


class HZ3_YuE2_SectionPlan:
    CATEGORY = "HZ3 YuE2/Score"
    FUNCTION = "plan"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("plan", "sections", "seconds", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Parse the ABC into a timed, ordered section plan (bars, seconds, frames) with optional per-section style overrides."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "score_abc": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                "style_overrides": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "One per section, 'Section: text'. Overrides are merged into plan/sections and are usable to guide per-section style.",
                }),
                "lyrics": ("STRING", {"forceInput": True}),
            },
        }

    def plan(self, score_abc, style_overrides="", lyrics=""):
        info, timeline, total_frames = _section_timeline(score_abc)
        overrides = _parse_overrides(style_overrides)
        note_per_section = _lyric_per_section(info) if lyrics != "" else {}

        for index, section in enumerate(timeline):
            if section["name"] in overrides:
                section["style_override"] = overrides[section["name"]]
            if lyrics != "":
                section["vocal_notes"] = note_per_section.get(section["name"], 0)

        sections_json = json.dumps(timeline, ensure_ascii=False, indent=2)
        plan_text = _render_plan(timeline, overrides, info, total_frames)
        if lyrics != "":
            plan_text += "\n# vocal notes per section"
            for section in timeline:
                plan_text += f"\n{section['label']}: {section.get('vocal_notes', 0)} notes"

        seconds = total_frames / FRAMES_PER_SECOND
        report = f"Planned {len(timeline)} sections · {seconds:.2f} s"
        return {"ui": {"text": [plan_text]},
                "result": (plan_text, sections_json, seconds, report)}


def assemble_conditionings(takes, start_frames):
    if len(takes) != len(start_frames):
        raise ValueError("Each take needs a corresponding start frame.")
    if not takes:
        raise ValueError("Provide at least one take to assemble.")
    pieces = []
    chunks = []
    kv_cursor = 0
    base_device = None
    base_dtype = None
    base_metadata = None
    for index, (take, start) in enumerate(zip(takes, start_frames), 1):
        context, metadata, take_chunks, take_frames = _entry(take, f"take {index}")
        if base_device is None:
            base_device, base_dtype = context.device, context.dtype
            base_metadata = metadata
        if take_frames < 1:
            raise ValueError(f"take {index} is empty.")
        for chunk in take_chunks:
            chunk_start, chunk_end, kv_start, kv_end = chunk
            frame_count = chunk_end - chunk_start
            prefix_count = (kv_end - kv_start) - frame_count - 1
            if prefix_count < 1:
                raise ValueError(f"take {index} has an inconsistent chunk: {chunk}")
            piece = torch.cat((
                context[:, kv_start:kv_start + prefix_count],
                context[:, kv_start + prefix_count:kv_end - 1],
                context[:, kv_end - 1:kv_end],
            ), dim=1)
            out_start = start + chunk_start
            out_end = out_start + frame_count
            pieces.append(piece.to(device=base_device, dtype=base_dtype))
            chunks.append((out_start, out_end, kv_cursor, kv_cursor + piece.shape[1]))
            kv_cursor += piece.shape[1]
    if not pieces:
        raise ValueError("No section conditioning frames were produced.")

    prev_end = 0
    for chunk_start, chunk_end, _kv_start, _kv_end in chunks:
        if chunk_start != prev_end:
            raise ValueError(
                f"Assembled conditionings are not contiguous (gap/overlap at frame "
                f"{chunk_start}..{chunk_end}). Sections must tile exactly."
            )
        prev_end = chunk_end

    assembled = torch.cat(pieces, dim=1)
    total_frames = chunks[-1][1]
    metadata = copy.copy(base_metadata)
    metadata["yue2_chunks"] = tuple(chunks)
    metadata["yue2_frames"] = total_frames
    metadata["yue2_truncated"] = False
    metadata["hz3_section_assembly"] = True
    return [[assembled, metadata]], total_frames


class HZ3_YuE2_AssembleSections:
    CATEGORY = "HZ3 YuE2/Conditioning"
    FUNCTION = "assemble"
    RETURN_TYPES = ("CONDITIONING", "FLOAT", "STRING")
    RETURN_NAMES = ("conditioning", "seconds", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Concatenate one conditioning take per section into a single contiguous YuE2 conditioning, tiled in order."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "take_1": ("CONDITIONING",),
            },
            "optional": {
                **{f"take_{index}": ("CONDITIONING",) for index in range(2, 9)},
                "sections": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect HZ3 YuE2 · SectionPlan.sections to label each take by section name.",
                }),
            },
        }

    def assemble(self, take_1, **kwargs):
        takes = [take_1]
        names = []
        for index in range(2, 9):
            value = kwargs.get(f"take_{index}")
            if value is not None:
                takes.append(value)
                names.append(f"take {index}")
        sections = kwargs.get("sections")
        section_labels = []
        if isinstance(sections, str):
            try:
                parsed = json.loads(sections)
                if isinstance(parsed, list):
                    section_labels = [
                        str(item.get("label", item.get("name", f"take {i}")))
                        for i, item in enumerate(parsed, 1)
                    ]
            except json.JSONDecodeError:
                section_labels = []

        start_frames = []
        cursor = 0
        for take in takes:
            _context, _metadata, _chunks, take_frames = _entry(take, "take")
            start_frames.append(cursor)
            cursor += take_frames

        conditioning, total_frames = assemble_conditionings(takes, start_frames)
        seconds = total_frames / FRAMES_PER_SECOND

        labels = section_labels[:len(takes)]
        while len(labels) < len(takes):
            labels.append(f"take {len(labels) + 1}")
        per_take = []
        for index, take in enumerate(takes, 1):
            _ctx, _meta, _ch, frames = _entry(take, f"take {index}")
            per_take.append(f"{labels[index - 1]}: {start_frames[index - 1] / FRAMES_PER_SECOND:.2f}-"
                            f"{start_frames[index - 1] / FRAMES_PER_SECOND + frames / FRAMES_PER_SECOND:.2f}s "
                            f"({frames} frames)")
        report = f"Assembled {len(takes)} sections · {seconds:.2f} s\n" + "\n".join(per_take)
        conditioning[0][1]["hz3_section_layout"] = per_take
        return {"ui": {"text": [report]}, "result": (conditioning, seconds, report)}


class HZ3_YuE2_ConcatConditionings:
    CATEGORY = "HZ3 YuE2/Conditioning"
    FUNCTION = "assemble"
    RETURN_TYPES = ("CONDITIONING", "FLOAT", "STRING")
    RETURN_NAMES = ("conditioning", "seconds", "report")
    INPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Concatenate a LIST of per-section YuE2 conditionings (e.g. one from each per-section "
        "YuE2 Generate Music) into one contiguous conditioning, in connection order. Connect "
        "every per-section conditioning to the single `conditionings` list input."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditionings": ("CONDITIONING", {
                "tooltip": "Connect each per-section conditioning (from YuE2 Generate Music) to this list input; order = connection order.",
            }),
        }}

    def assemble(self, conditionings):
        takes = [take for take in (conditionings or []) if take is not None]
        if not takes:
            raise ValueError("Connect at least one per-section conditioning to concatenate.")
        start_frames = []
        cursor = 0
        for index, take in enumerate(takes, 1):
            _ctx, _meta, _chunks, frames = _entry(take, f"conditioning {index}")
            start_frames.append(cursor)
            cursor += frames
        conditioning, total_frames = assemble_conditionings(takes, start_frames)
        seconds = total_frames / FRAMES_PER_SECOND
        report = (
            f"Concatenated {len(takes)} per-section conditioning(s) in order · "
            f"{seconds:.2f} s · {total_frames} frames"
        )
        return {"ui": {"text": [report]}, "result": (conditioning, seconds, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_SectionPlan": HZ3_YuE2_SectionPlan,
    "HZ3_YuE2_AssembleSections": HZ3_YuE2_AssembleSections,
    "HZ3_YuE2_ConcatConditionings": HZ3_YuE2_ConcatConditionings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_SectionPlan": "HZ3 YuE2 · Section Plan",
    "HZ3_YuE2_AssembleSections": "HZ3 YuE2 · Assemble Sections",
    "HZ3_YuE2_ConcatConditionings": "HZ3 YuE2 · Concat Conditionings (list)",
}
