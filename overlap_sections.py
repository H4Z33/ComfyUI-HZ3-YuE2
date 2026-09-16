"""Overlap instrumental notes across adjacent per-section ABC fragments.

For every internal boundary, the LAST instrumental (Ins) notes of the previous
section (covering `overlap` seconds -> mapped to bars) are duplicated so they also
appear at the START of the next section's Ins voice. This ties consecutive sections
together and avoids an abrupt cut at the seam. The first and last sections have no
predecessor/ successor respectively and are not extended from that side.

Each fragment keeps its own header; the pickup is spliced inside the target
section's own Ins voice (after its first `V: Ins` selection line) so the section
naming/bar alignment is preserved as much as possible.
"""

from __future__ import annotations

import json
import re


def _parse_section(abc):
    """Split one section fragment into {header, body_lines, bpm, num, den, spb,
    ins_bars} where ins_bars is the flat list of Ins bars in document order."""
    lines = (abc or "").replace("\r\n", "\n").split("\n")
    header_end = None
    bpm = None
    num = None
    den = None
    for index, line in enumerate(lines):
        if line.startswith("Q:"):
            match = re.search(r"=\s*(\d+)", line)
            if match:
                bpm = int(match.group(1))
        elif line.startswith("M:"):
            match = re.search(r"M:\s*(\d+)\s*/\s*(\d+)", line)
            if match:
                num, den = int(match.group(1)), int(match.group(2))
        elif line.startswith("K:"):
            header_end = index
            break
    if header_end is None:
        header_end = max(0, len(lines) - 1)
    header = lines[:header_end + 1]
    body = lines[header_end + 1:]
    spb = (num * 60.0 / bpm) if (bpm and num) else None

    ins_bars = []
    index = 0
    while index < len(body):
        if body[index].strip().startswith("V: Ins"):
            index += 1
            while index < len(body) and not (body[index].strip().startswith("V:")
                                             or body[index].strip().startswith("% ")):
                toks = [t for t in body[index].split("|") if t.strip()]
                ins_bars.extend(toks)
                index += 1
        else:
            index += 1
    return {"header": header, "body": body, "bpm": bpm, "num": num, "den": den,
            "spb": spb, "ins_bars": ins_bars}


def _render_bar_line(bars):
    return ("|".join(bars)) + ("|" if bars else "")


def _prepend_ins_pickup(abc, pickup_bars):
    """Insert the pickup bars into the target section's Ins voice (after its first
    'V: Ins' line). If the section has no Ins voice, add an Ins block first."""
    lines = (abc or "").replace("\r\n", "\n").split("\n")
    if not pickup_bars:
        return abc
    pickup_line = _render_bar_line(pickup_bars)

    for index, line in enumerate(lines):
        if line.strip() == "V: Ins":
            lines.insert(index + 1, pickup_line)
            return "\n".join(lines) + "\n"
    # no Ins voice: add an Ins block right after the header end / before first V: Vocal
    for index, line in enumerate(lines):
        if line.strip().startswith("V: Vocal") or line.strip().startswith("% "):
            lines[index:index] = ["V: Ins", pickup_line]
            return "\n".join(lines) + "\n"
    return abc


def apply_abc_overlap(segments_abc, overlap_seconds=4.0):
    """Return (overlapped_segments, added_seconds) applying the instrumental overlap.

    `overlap_seconds` is mapped to whole bars using each source section's own BPM/beat.
    `added_seconds[i]` is 0 for the first section and >0 for later sections that gain
    a pickup from their predecessor's tail.
    """
    parsed = [_parse_section(abc) for abc in segments_abc]
    overlapped = list(segments_abc)
    added_seconds = [0.0] * len(segments_abc)

    for index in range(len(parsed) - 1):
        source = parsed[index]
        target = parsed[index + 1]
        spb = source["spb"]
        bars = source["ins_bars"]
        if not bars or spb is None:
            continue
        count = 1
        if overlap_seconds > 0 and spb > 0:
            count = max(1, round(overlap_seconds / spb))
        count = min(count, len(bars))
        pickup = bars[-count:]
        # Splice into the target's Ins voice.
        overlapped[index + 1] = _prepend_ins_pickup(segments_abc[index + 1], pickup)
        added_seconds[index + 1] = round(count * spb, 3)

    return overlapped, added_seconds


class HZ3_YuE2_OverlapABCSections:
    CATEGORY = "HZ3 YuE2/Score"
    FUNCTION = "overlap"
    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("segments_abc", "added_seconds", "report")
    OUTPUT_IS_LIST = (True, True, False)
    INPUT_IS_LIST = (True, False, False)
    DESCRIPTION = (
        "Duplicate the instrumental (Ins) tail of each section into the START of the next "
        "one, so consecutive sections overlap. Returns the overlapped segments_abc (list) "
        "and added_seconds (list, 0 for the first section). First/last sections have no "
        "neighbour from that side."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "segments_abc": ("STRING", {
                "tooltip": "Connect each per-section ABC fragment (from 'SheetSage2 Audio to ABC + Sections'.segments_abc) here."}),
            "overlap": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 120.0, "step": 0.5,
                                  "tooltip": "Overlap in seconds -> mapped to whole instrumental bars of the source section."}),
        }}

    def overlap(self, segments_abc, overlap=4.0):
        segments = list(segments_abc or [])
        if not segments:
            raise ValueError("Connect at least one per-section ABC.")
        overlapped, added = apply_abc_overlap(segments, float(overlap))
        report = (f"Applied instrumental overlap (target ~{overlap:.1f}s):\n"
                  + "\n".join(f"  [{i}] +{added[i]:.2f}s" for i in range(len(added))))
        return {"ui": {"text": [report]}, "result": (overlapped, added, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_OverlapABCSections": HZ3_YuE2_OverlapABCSections,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_OverlapABCSections": "HZ3 YuE2 · Overlap ABC Sections",
}
