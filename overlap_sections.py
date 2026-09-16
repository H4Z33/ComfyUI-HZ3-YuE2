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
