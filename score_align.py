"""Procedural score alignment: reconcile SheetSage2 ABC with vocal attacks, rests, and timing."""

from __future__ import annotations

import json
import logging
import re
from fractions import Fraction

try:
    from .abc_score import parse, Score, AbcError, CHORD, PITCH_NAME
except (ImportError, ValueError):
    from abc_score import parse, Score, AbcError, CHORD, PITCH_NAME

logger = logging.getLogger("HZ3.ScoreAlign")


def sanitize_chord(chord: str) -> str:
    """Sanitize SheetSage2 chord symbols to standard qualities supported by abc_score."""
    if not chord:
        return ""
    if CHORD.fullmatch(chord):
        return chord
    m = re.match(r"^([A-G](?:bb|##|b|#)?)(.*?)(?:/([A-G](?:bb|##|b|#)?).*?)?$", chord)
    if not m:
        return ""
    root, qual, bass = m.group(1), m.group(2), m.group(3)
    clean_qual = ""
    if "dim7" in qual:
        clean_qual = "dim7"
    elif "dim" in qual:
        clean_qual = "dim"
    elif "aug" in qual:
        clean_qual = "aug"
    elif "m7b5" in qual:
        clean_qual = "m7b5"
    elif any(k in qual for k in ("maj7", "Gj7", "M7", "maj")):
        clean_qual = "maj7"
    elif any(k in qual for k in ("m7", "min7")):
        clean_qual = "m7"
    elif "7sus4" in qual:
        clean_qual = "7sus4"
    elif "sus4" in qual:
        clean_qual = "sus4"
    elif "sus2" in qual:
        clean_qual = "sus2"
    elif "m6" in qual:
        clean_qual = "m6"
    elif "6" in qual:
        clean_qual = "6"
    elif "7" in qual:
        clean_qual = "7"
    elif any(k in qual for k in ("m", "min")):
        clean_qual = "m"

    cand = root + clean_qual
    if bass and re.match(r"^[A-G](?:bb|##|b|#)?$", bass):
        cand += "/" + bass

    if CHORD.fullmatch(cand):
        return cand
    if CHORD.fullmatch(root):
        return root
    return ""


def sanitize_abc(text: str) -> str:
    """Sanitize chord annotations in ABC body after the K: header line."""
    lines = text.splitlines()
    k_idx = -1
    for i, l in enumerate(lines):
        if l.startswith("K:"):
            k_idx = i
            break
    if k_idx == -1:
        return text

    x_line = "X:1"
    t_line = "T:"
    m_line = "M:4/4"
    l_line = "L:1/8"
    q_line = "Q:1/4=120"
    k_line = "K:C"
    for l in lines[:k_idx + 1]:
        if l.startswith("X:"):
            x_line = l
        elif l.startswith("M:"):
            m_line = l
        elif l.startswith("L:"):
            l_line = l
        elif l.startswith("Q:"):
            q_line = l
        elif l.startswith("K:"):
            k_line = l

    header = "\n".join([
        x_line,
        t_line,
        m_line,
        l_line,
        q_line,
        'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
        'V: Ins clef=treble name="Ins Melody" snm="Inst."',
        k_line,
    ])
    body = "\n".join(lines[k_idx + 1:])

    def repl(match):
        cleaned = sanitize_chord(match.group(1))
        return f'"{cleaned}"' if cleaned else ""

    clean_body = re.sub(r'"([^"\n]*)"', repl, body)
    clean_body = re.sub(r"^V:\s*Vocal\b", "V: Vocal", clean_body, flags=re.MULTILINE)
    clean_body = re.sub(r"^V:\s*Ins\b", "V: Ins", clean_body, flags=re.MULTILINE)

    # In native YuE2 ABC, chord symbols belong in Vocal, not Ins
    body_lines = clean_body.splitlines()
    in_ins = False
    fixed_body_lines = []
    for bl in body_lines:
        if bl.startswith("V: Ins"):
            in_ins = True
        elif bl.startswith("V: Vocal"):
            in_ins = False
        if in_ins and not bl.startswith("V:"):
            fixed_body_lines.append(re.sub(r'"[^"\n]*"', '', bl))
        else:
            fixed_body_lines.append(bl)
    clean_body = "\n".join(fixed_body_lines)
    # Collapse polyphonic note clusters [CEG]8 or [G,B,D]8 to single monophonic note G,8
    clean_body = re.sub(r"\[([A-Ga-gz][,']*)[A-Ga-gz,']*\]([0-9]*)", r"\1\2", clean_body)
    return header + "\n" + clean_body


def normalize_section_label(label: str) -> str:
    """Clean SheetSage2 OCR/encoding quirks into clean lowercase musical labels."""
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", "", label).strip().lower()
    if "intro" in cleaned:
        return "intro"
    if "outro" in cleaned:
        return "outro"
    if "interlu" in cleaned or ("int" in cleaned and "lu" in cleaned):
        return "interlude"
    if "bridge" in cleaned:
        return "bridge"
    if "horus" in cleaned or "chor" in cleaned:
        return "chorus"
    if "vers" in cleaned or ("v" in cleaned and "rs" in cleaned):
        return "verse"
    if "solo" in cleaned:
        return "solo"
    return cleaned or "section"


def align_and_repair_abc(raw_abc: str, whisper_segments=None, lyrics_text: str = ""):
    """Procedurally realign SheetSage2 ABC score, fix chord quirks, detect pickups and outros.

    Returns:
        (repaired_abc: str, structure: list[dict])
    """
    if not raw_abc or not str(raw_abc).strip():
        return raw_abc, []

    try:
        clean_text = sanitize_abc(str(raw_abc).replace("\r\n", "\n").strip())
        raw_lines = [l.strip() for l in clean_text.splitlines() if l.strip()]

        k_idx = -1
        for i, l in enumerate(raw_lines):
            if l.startswith("K:"):
                k_idx = i
                break
        if k_idx == -1:
            return clean_text, []

        counts = {}
        for l in raw_lines:
            if l.startswith("% "):
                norm = normalize_section_label(l[2:])
                counts[norm] = counts.get(norm, 0) + 1

        new_lines = []
        running = {}
        for l in raw_lines:
            if l.startswith("% "):
                norm = normalize_section_label(l[2:])
                if counts[norm] > 1 and norm in ("verse", "chorus"):
                    running[norm] = running.get(norm, 0) + 1
                    disp = f"{norm} {running[norm]}"
                else:
                    disp = norm
                new_lines.append(f"% {disp}")
            else:
                new_lines.append(l)

        # Detect trailing outro: if the last section has trailing silent measures in Vocal
        has_outro = any(l.startswith("% outro") for l in new_lines)
        if not has_outro and len(new_lines) >= 6:
            for i in range(len(new_lines) - 1, max(k_idx, len(new_lines) - 6), -1):
                if new_lines[i].startswith("V: Vocal") and i + 1 < len(new_lines):
                    next_line = new_lines[i + 1]
                    notes_only = re.sub(r'"[^"]*"', "", next_line).strip()
                    if (re.search(r"z\d+\|.*z\d+\|", notes_only) or re.fullmatch(r"[zZ\d\|\s]+", notes_only)) and not re.search(r"[A-Ga-g]", notes_only):
                        new_lines.insert(i, "% outro")
                        break

        # Voice Healing Pass: Ensure every section/group has BOTH V: Vocal and V: Ins
        # with identical measure counts, matching native YuE2 two-voice invariants.
        healed_lines = []
        body_start_idx = k_idx + 1
        healed_lines.extend(new_lines[:body_start_idx])

        b_idx = body_start_idx
        while b_idx < len(new_lines):
            line = new_lines[b_idx]
            if line.startswith("% "):
                healed_lines.append(line)
                b_idx += 1
                continue

            if line.startswith("V: Vocal"):
                v_headers = [line]
                b_idx += 1
                while b_idx < len(new_lines) and new_lines[b_idx].startswith(("M:", "K:")):
                    v_headers.append(new_lines[b_idx])
                    b_idx += 1
                v_music = new_lines[b_idx] if b_idx < len(new_lines) else "Z|"
                b_idx += 1

                # Calculate bar count from Vocal
                v_bars = [b.strip() for b in v_music[:-1].split("|") if b.strip()] if v_music.endswith("|") else [v_music]
                bar_count = len(v_bars) if v_bars else 4

                # Look ahead for V: Ins
                if b_idx < len(new_lines) and new_lines[b_idx].startswith("V: Ins"):
                    i_headers = [new_lines[b_idx]]
                    b_idx += 1
                    while b_idx < len(new_lines) and new_lines[b_idx].startswith(("M:", "K:")):
                        i_headers.append(new_lines[b_idx])
                        b_idx += 1
                    i_music = new_lines[b_idx] if b_idx < len(new_lines) else ("Z|" * bar_count)
                    b_idx += 1
                else:
                    # Missing V: Ins -> Synthesize silent measures matching Vocal
                    i_headers = ["V: Ins"]
                    i_music = "Z|" * bar_count

                healed_lines.extend(v_headers)
                healed_lines.append(v_music)
                healed_lines.extend(i_headers)
                healed_lines.append(i_music)
            elif line.startswith("V: Ins"):
                # Group started with V: Ins without V: Vocal
                i_headers = [line]
                b_idx += 1
                while b_idx < len(new_lines) and new_lines[b_idx].startswith(("M:", "K:")):
                    i_headers.append(new_lines[b_idx])
                    b_idx += 1
                i_music = new_lines[b_idx] if b_idx < len(new_lines) else "Z|"
                b_idx += 1
                i_bars = [b.strip() for b in i_music[:-1].split("|") if b.strip()] if i_music.endswith("|") else [i_music]
                bar_count = len(i_bars) if i_bars else 4

                # Insert synthesized Vocal rests before Ins
                healed_lines.append("V: Vocal")
                healed_lines.append("Z|" * bar_count)
                healed_lines.extend(i_headers)
                healed_lines.append(i_music)
            else:
                healed_lines.append(line)
                b_idx += 1

        new_lines = healed_lines

        result_abc = "\n".join(new_lines) + "\n"

        # Try strict parse to extract high precision timeline and bar metrics
        try:
            score = parse(result_abc)
            bpm = score.bpm
            bars = score.voices["Vocal"].bars
            total_bars = len(bars)
        except Exception:
            score = None
            bpm = 120
            for hl in raw_lines[:k_idx + 1]:
                m = re.match(r"^Q:1/4=(\d+)", hl)
                if m:
                    bpm = int(m.group(1))
                    break
            bars = []
            total_bars = 0

        # Walk through lines to calculate bars and timeline for each section
        sec_info = []
        cur_bar_cursor = 0
        target_vocal = False

        for l in new_lines[k_idx + 1:]:
            if l.startswith("% "):
                sec_name = l[2:].strip()
                sec_info.append({"name": sec_name, "start_bar": cur_bar_cursor, "bars": 0})
            elif l.startswith("V: Vocal"):
                target_vocal = True
            elif l.startswith("V: Ins"):
                target_vocal = False
            elif "|" in l and target_vocal:
                chunks = [c.strip() for c in l[:-1].split("|") if c.strip()]
                for c in chunks:
                    rest = re.fullmatch(r"Z([2-4])?", c)
                    num = int(rest.group(1) or "1") if rest else 1
                    cur_bar_cursor += num
                    if sec_info:
                        sec_info[-1]["bars"] += num

        # Build structure timeline
        structure = []
        cur_t = 0.0
        seconds_per_bar_approx = float(4.0 * 60.0 / bpm)

        for s in sec_info:
            b_cnt = s["bars"]
            start_b = s["start_bar"]
            end_b = start_b + b_cnt
            if bars and end_b <= len(bars) and start_b < len(bars):
                t_start = float(bars[start_b][0] * 60 / bpm)
                t_end = float((bars[end_b - 1][0] + bars[end_b - 1][1]) * 60 / bpm)
            else:
                t_start = cur_t
                t_end = cur_t + (b_cnt * seconds_per_bar_approx)
            structure.append({
                "name": s["name"],
                "start": round(t_start, 2),
                "end": round(t_end, 2),
                "bars": b_cnt,
            })
            cur_t = t_end

        return result_abc, structure

    except Exception as exc:
        logger.warning(f"align_and_repair_abc fallback: {exc}")
        return raw_abc, []
