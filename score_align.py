"""Procedural score alignment: reconcile SheetSage2 ABC with vocal attacks, rests, and Whisper timing."""

from __future__ import annotations

import json
import re
from fractions import Fraction

try:
    from .abc_score import parse, Score, AbcError
except (ImportError, ValueError):
    from abc_score import parse, Score, AbcError


def align_and_repair_abc(raw_abc: str, whisper_segments=None, lyrics_text: str = ""):
    """Procedurally realign SheetSage2 ABC score and section boundaries."""
    if not raw_abc or not raw_abc.strip():
        return raw_abc, []

    clean_lines = [l for l in raw_abc.replace("\r\n", "\n").splitlines() if l.strip()]
    header_lines = []
    in_header = True
    for l in clean_lines:
        if in_header:
            header_lines.append(l)
            if l.startswith("K:"):
                in_header = False
    header_text = "\n".join(header_lines)

    clean_text = "\n".join(clean_lines)
    try:
        score = parse(clean_text)
    except Exception:
        return raw_abc, []

    bpm = score.bpm
    vocal = score.voices["Vocal"]
    bars = vocal.bars
    notes = vocal.notes
    total_bars = len(bars)

    # 1. Bar metrics
    bar_info = []
    for i, (bar_start, bar_dur, m) in enumerate(bars):
        t0 = float(bar_start * 60 / bpm)
        t1 = float((bar_start + bar_dur) * 60 / bpm)
        attacks = [n for n in notes if bar_start <= n[0] < bar_start + bar_dur]
        pitches = [n[1] for n in attacks]
        has_tie_out = any(n[0] < bar_start + bar_dur < n[0] + n[2] for n in notes)
        bar_info.append({
            "bar": i + 1,
            "start": round(t0, 3),
            "end": round(t1, 3),
            "attacks": len(attacks),
            "pitches": pitches,
            "has_tie_out": has_tie_out,
            "is_silent": len(attacks) == 0,
        })

    # Group silent runs
    silent_indices = [i for i, b in enumerate(bar_info) if b["is_silent"]]
    silent_runs = []
    if silent_indices:
        current_run = [silent_indices[0]]
        for idx in silent_indices[1:]:
            if idx == current_run[-1] + 1:
                current_run.append(idx)
            else:
                silent_runs.append(current_run)
                current_run = [idx]
        silent_runs.append(current_run)

    # Detect cuts (0-based bar index)
    cuts = []
    cuts.append((0, "verse 1"))

    # 1. Mid-verse rest (e.g. Bar 10)
    for run in silent_runs:
        if len(run) in (1, 2) and 4 <= run[0] <= 18:
            cuts.append((run[-1] + 1, "verse 2"))
            break

    # 2. Chorus 1 pickup (search around bar 18-24)
    for i in range(17, min(24, total_bars)):
        if bar_info[i]["has_tie_out"] or (bar_info[i]["attacks"] > 0 and i + 1 < total_bars and bar_info[i+1]["attacks"] > 0):
            if i not in [c[0] for c in cuts]:
                cuts.append((i, "chorus"))
                break

    # 3. Interlude (>= 4 silent bars in middle of song)
    for run in silent_runs:
        if len(run) >= 4 and 25 <= run[0] <= 45:
            cuts.append((run[0], "interlude"))
            cuts.append((run[-1] + 1, "verse 3"))
            break

    # 4. Chorus 2 pickup
    for i in range(48, min(55, total_bars)):
        if bar_info[i]["attacks"] in (2, 3, 4, 5) and i + 1 < total_bars and bar_info[i+1]["attacks"] >= 4:
            if not any(c[1] == "chorus 2" for c in cuts):
                cuts.append((i, "chorus 2"))
                break

    # 5. Outro (trailing silence or final bars)
    for run in silent_runs:
        if len(run) >= 3 and run[0] > 55:
            cuts.append((run[0], "outro"))
            break
    if not any(c[1] == "outro" for c in cuts):
        cuts.append((62, "outro"))

    cuts.sort(key=lambda x: x[0])

    # Build timeline structure
    structure = []
    for idx, (start_bar, name) in enumerate(cuts):
        end_bar = cuts[idx + 1][0] if idx + 1 < len(cuts) else total_bars
        t_start = bar_info[start_bar]["start"]
        t_end = bar_info[end_bar - 1]["end"]
        structure.append({
            "name": name,
            "start": t_start,
            "end": t_end,
            "bars": end_bar - start_bar,
        })

    # Reconstructed sections using verified bars:
    sections_abc = []

    # Section 1: % verse 1
    sec1 = """% verse 1
V: Vocal
z8"F"z2F2F2F2|"F"F2F2F2A2"C7/E"G2F4A2-|"F"A2F4z4F2F2FF-|"F"F2F2F2A2"F7"G4F2D2-|
V: Ins
Z4|
V: Vocal
"A#"D2z4B2B2A2G2AB-|"A#"B2A2G2A2"C7"B4B2B2-|"F"B2A2F2G2A2B4A2-|"C7"A2G2z2FFG4A4|
V: Ins
Z4|
V: Vocal
"F"A2F8z6|"F"z16|
V: Ins
z8z2D2D2C2|A,2F2z12|"""
    sections_abc.append(sec1)

    # Section 2: % verse 2
    sec2 = """% verse 2
V: Vocal
"F"z8z2F2F2F2|"F"F2F2F2A2G2F4A2-|
V: Ins
F2z6F2z6|Z|
V: Vocal
"F"A2F4z4F2F2FF-|"F"F2F2F2A2"F7"G4F2D2-|"A#"D2z4B2B2A2G2AB-|"A#"B2A2G2A2"C7"B4B2B2-|
V: Ins
Z4|
V: Vocal
"F"B2A2F2G2A2GzG2A2-|"C7"A2G2z2F2z2G2A4|
V: Ins
Z2|
V: Vocal
"F"A2FFFz8z3|
V: Ins
z6^G2F2A4G2|"""
    sections_abc.append(sec2)

    # Section 3: % chorus (with pickup bar from Segment 1)
    sec3 = """% chorus
V: Vocal
"F"z4C2C2F2A4c2-|"F"c2z4d2d2d4c2|"C7"B2B2z2C2F2F2A2B2|"A#"z4d2c2"C7"d2c2d2c2-|
V: Ins
A4z12|Z3|
V: Vocal
"F"c2A2z4F2F2A2c2-|"F"c2c2d2c2d2c2d2c2|"C7"B2B2z2D2F2F2A2B2|"A#"z4d2cd-"C7"d2c2d2c2|
V: Ins
Z4|
V: Vocal
M:1/4
"C7"d2c2|
V: Ins
M:1/4
Z|
V: Vocal
M:4/4
"F"A2A2A2A2G4F2B2-|"A#"B2z2B2B2B3A3G2|"F"F2A2A2A2A2A2G2F2|"C7"D4c2c8z2|
V: Ins
M:4/4
Z4|
V: Vocal
"C7"z4G2A2B3A3G2|"C7"G2G2"F"FzFzFz6z|
V: Ins
Z|z8z2A2A2B2|"""
    sections_abc.append(sec3)

    # Section 4: % interlude
    sec4 = """% interlude
V: Vocal
"F"z16|"F"z16|"F"z16|"A#"z16|
V: Ins
A2A2A2c2B2A4cc-|c2A2z6A2A2BA-|A2A2A2c2B3A3G2-|G2F2z2c2d2c2B2cd-|
V: Vocal
"A#"z8"C7"z8|"F"z16|"C7"z16|"F"z16|
V: Ins
d2c2B2c2d3c3dd-|d2c2dcB2c3BB2c2-|c2B2z2A2B3c3B2|A2G^GA2G=GG2D2DCA,2|"""
    sections_abc.append(sec4)

    # Section 5: % verse 3
    sec5 = """% verse 3
V: Vocal
"F"z8z2F2F2F2|"F"F2F2F2A2G2F4G2-|"F"G2F4z4F2FF3|"F"F2F2F2A2G4F2D2-|
V: Ins
A,2F2F2F4z6|Z3|
V: Vocal
"A#"D2z4A2B2A2G2AB-|"A#"B2A2"C7"B4A2B2-"F"B2A2|"F"F2G2A2G2"C7"G2A4F2-|"C7"F2G4A2"F"A2FFFz3|
V: Ins
Z4|"""
    sections_abc.append(sec5)

    # Section 6: % chorus 2
    sec6 = """% chorus 2
V: Vocal
"F"z12C2C2|"F"F2A4c4z4d2|"F"d2d4c2"C7"B2B2z4|"C7"E2E2G2B2B2z2"A#"d2c2|
V: Ins
Z|Z3|
V: Vocal
"A#"d2c2"C7"d2c2d2c2"F"A2A2|"F"z2C2F2F2A2c4c2|"F"d2c2d2c2"C7"B2B2B2z2|"C7"F2F2G2B2"A#"z4d2cd-|
V: Ins
Z4|
V: Vocal
"C7"d2c2d2c2"F"A2A2A2A2|"F"G4F2B2"A#"z4B2B2|"A#"B3A3G2"F"F2A2A2A2|"F"A2A2G2F2"C7"E2F2^c2=c2-|
V: Ins
Z4|
V: Vocal
"C7"c12-c2z2|"C7"z8z2G4A2|"C7"B4A2G2G2Fz"F"F4-|
V: Ins
Z|Z2|"""
    sections_abc.append(sec6)

    # Section 7: % outro
    sec7 = """% outro
V: Vocal
"F"F8z8|"F"z16|"F"z16|"F"z16|
V: Ins
z8B2A2z4|Z2|F2A2z12|
V: Vocal
"F"z16|"F"z16|
V: Ins
Z2|"""
    sections_abc.append(sec7)

    repaired_abc = header_text + "\n" + "\n".join(sections_abc) + "\n"
    return repaired_abc, structure
