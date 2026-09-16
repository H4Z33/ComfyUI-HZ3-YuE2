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
    lines = (abc or "").replace("\r\n", "\n").splitlines()
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


def _chords_only(bars):
    """Keep only the CHORD symbols of each vocal bar (over rests); discard the melody
    notes, so the duplicated seam carries harmony but not a second vocal melody."""
    out = []
    for bar in bars:
        chords = re.findall(r'"([^"]*)"', bar)
        out.append("".join(f'"{c}"z' for c in chords) if chords else "z")
    return out


def _voice_block(vocal_bars, ins_bars):
    return ["V: Vocal", _render(_chords_only(vocal_bars)), "V: Ins", _render(ins_bars)]


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


def _section_bars(sec, overlap_seconds):
    """Whole bars of this section covering ~ the overlap window."""
    n = max(len(sec["vocal"]), len(sec["ins"]))
    spb = sec["spb"]
    if overlap_seconds <= 0 or spb is None or spb <= 0 or n <= 0:
        return 0
    return max(1, min(round(overlap_seconds / spb), n))


def apply_abc_overlap(segments_abc, overlap_seconds=4.0):
    """Return (overlapped_segments, added_seconds).

    At every interface ONLY ONE material is used: the LAST bars of the left section
    (chords-only Vocal + Ins). It is appended to the left section's own END and the
    SAME block is prepended to the right section's START before its tag. So both sides
    of a seam are identical (same content, same length), avoiding long endings: the
    end-append equals the start-prepend. section i's own tail is used for the boundary
    i-(i+1); first section has no prepend, last has no append from its own boundary.
    """
    parsed = [_parse_section(abc) for abc in segments_abc]
    overlapped = list(segments_abc)
    added = [0.0] * len(segments_abc)
    n = len(parsed)

    for index in range(n - 1):
        left = parsed[index]
        k = _section_bars(left, overlap_seconds)
        if k <= 0:
            continue
        block = _voice_block(left["vocal"][-k:], left["ins"][-k:])
        added_sec = round(k * left["spb"], 3)

        # Left section's OWN end gains its own last bars (boundary material).
        left_lines = _insert_append(left["lines"], block)
        overlapped[index] = "\n".join(left_lines) + "\n"
        parsed[index] = _parse_section(overlapped[index])
        added[index] += added_sec

        # The RIGHT section's start gains the SAME block (before its tag).
        right = parsed[index + 1]
        right_lines = _insert_prepend(right["lines"], right["header_end"], block)
        overlapped[index + 1] = "\n".join(right_lines) + "\n"
        parsed[index + 1] = _parse_section(overlapped[index + 1])
        added[index + 1] += added_sec

    added = [round(v, 3) for v in added]
    return overlapped, added


def _insert_append(lines, block):
    return list(lines) + block
