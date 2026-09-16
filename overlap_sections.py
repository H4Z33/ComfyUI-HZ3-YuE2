"""Overlap section ABCs across adjacent fragments: chords + instrumental, both sides.

At every interface the shared material is duplicated into BOTH neighbours:
- the END of the previous section gains the FIRST chords+instrumental of the next,
- the START of the next section (just before its '% [tag]') gains the LAST
  chords+instrumental of the previous.

Only the musical bars are copied (V: Vocal with inline chords, V: Ins); section tags
are never duplicated. `added_seconds[i]` records how much time each section grew
(leading + trailing pickups) so a downstream Conditioning Overlap node can apply the
same overlap in frames.
"""

from __future__ import annotations

import json
import re


def _parse_section(abc):
    """Return {lines, bpm, num, den, spb, name, vocal, ins, header_end}
    where vocal/ins are ordered flat bars (vocal keeps inline chord symbols)."""
    lines = (abc or "").replace("\r\n", "\n").split("\n")
    header_end = None
    bpm = num = den = None
    for index, line in enumerate(lines):
        if line.startswith("Q:"):
            m = re.search(r"=\s*(\d+)", line)
            if m:
                bpm = int(m.group(1))
        elif line.startswith("M:"):
            m = re.search(r"M:\s*(\d+)\s*/\s*(\d+)", line)
            if m:
                num, den = int(m.group(1)), int(m.group(2))
        elif line.startswith("K:"):
            header_end = index
            break
    if header_end is None:
        header_end = max(0, len(lines) - 1)
    spb = (num * 60.0 / bpm) if (bpm and num) else None

    vocal, ins = [], []
    name = ""
    current = None
    for line in lines[header_end + 1:]:
        stripped = line.strip()
        if stripped.startswith("% "):
            name = stripped[2:].strip()
            current = None
        elif stripped == "V: Vocal":
            current = vocal
        elif stripped == "V: Ins":
            current = ins
        elif stripped.startswith("V:"):
            current = None
        elif current is not None:
            for bar in line.split("|"):
                if bar.strip():
                    current.append(bar.strip())
    return {"lines": lines, "header_end": header_end, "bpm": bpm, "num": num,
            "den": den, "spb": spb, "name": name, "vocal": vocal, "ins": ins}


def _render(bars):
    return ("|".join(bars)) + ("|" if bars else "")


def _voice_block(vocal_bars, ins_bars):
    return ["V: Vocal", _render(vocal_bars), "V: Ins", _render(ins_bars)]


def _insert_prepend(lines, header_end, block):
    """Insert a voice block right before the section tag (after the header)."""
    out = list(lines)
    insert_at = header_end + 1
    for index in range(header_end + 1, len(out)):
        if out[index].strip().startswith("% ") or out[index].strip().startswith("V:"):
            insert_at = index
            break
    out[insert_at:insert_at] = block
    return out


def apply_abc_overlap(segments_abc, overlap_seconds=4.0):
    """Return (overlapped_segments, added_seconds) applying the two-sided overlap."""
    parsed = [_parse_section(abc) for abc in segments_abc]
    overlapped = list(segments_abc)
    added = [0.0] * len(segments_abc)
    n = len(parsed)

    for index in range(n - 1):
        prev = parsed[index]
        nxt = parsed[index + 1]
        if prev["spb"] is None or nxt["spb"] is None:
            continue

        def section_bars(sec, overlap):
            n = max(len(sec["vocal"]), len(sec["ins"]))
            if overlap <= 0 or sec["spb"] is None or sec["spb"] <= 0 or n <= 0:
                return 0
            return max(1, min(round(overlap / sec["spb"]), n))

        k_prev = section_bars(prev, overlap_seconds)
        k_next = section_bars(nxt, overlap_seconds)
        if k_prev <= 0 and k_next <= 0:
            continue

        # Previous gains the NEXT's first bars appended at its end.
        if k_next > 0:
            nv = nxt["vocal"][:k_next]
            ni = nxt["ins"][:k_next]
            prev_lines = _insert_append(parsed[index]["lines"], _voice_block(nv, ni))
            overlapped[index] = "\n".join(prev_lines) + "\n"
            parsed[index] = _parse_section(overlapped[index])
            added[index] += round(k_next * nxt["spb"], 3)

        # Next gains the PREVIOUS's last bars prepended before its tag.
        if k_prev > 0:
            pv = prev["vocal"][-k_prev:]
            pi = prev["ins"][-k_prev:]
            nxt_lines = _insert_prepend(parsed[index + 1]["lines"], parsed[index + 1]["header_end"], _voice_block(pv, pi))
            overlapped[index + 1] = "\n".join(nxt_lines) + "\n"
            parsed[index + 1] = _parse_section(overlapped[index + 1])
            added[index + 1] += round(k_prev * prev["spb"], 3)

    added = [round(v, 3) for v in added]
    return overlapped, added


def _insert_append(lines, block):
    return list(lines) + block


class HZ3_YuE2_OverlapABCSections:
    CATEGORY = "HZ3 YuE2/Score"
    FUNCTION = "overlap"
    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("segments_abc", "added_seconds", "report")
    OUTPUT_IS_LIST = (True, True, False)
    INPUT_IS_LIST = (True, False, False)
    DESCRIPTION = (
        "Overlap section ABCs at every interface: each section gains the first "
        "chords+instrumental (Vocal+Ins) of the NEXT at its end, and the last "
        "chords+instrumental of the PREVIOUS before its tag. Section tags are never "
        "duplicated. added_seconds[i] = time each section grew (for the Conditioning "
        "Overlap node)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "segments_abc": ("STRING", {
                "tooltip": "Connect each per-section ABC fragment (from 'SheetSage2 Audio to ABC + Sections'.segments_abc) here."}),
            "overlap": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 120.0, "step": 0.5,
                                  "tooltip": "Overlap (s) duplicated from/into each neighbour, mapped to whole bars per section."}),
        }}

    def overlap(self, segments_abc, overlap=4.0):
        segments = list(segments_abc or [])
        if not segments:
            raise ValueError("Connect at least one per-section ABC.")
        overlapped, added = apply_abc_overlap(segments, float(overlap))
        report = (f"Two-sided overlap (target ~{overlap:.1f}s), added per section (s):\n"
                  + "\n".join(f"  [{i}] +{added[i]:.2f}s" for i in range(len(added))))
        return {"ui": {"text": [report]}, "result": (overlapped, added, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_OverlapABCSections": HZ3_YuE2_OverlapABCSections,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_OverlapABCSections": "HZ3 YuE2 · Overlap ABC Sections",
}
