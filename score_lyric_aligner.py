"""HZ3 YuE2 · Lyrics alignment library (audio forced alignment, Whisper, ABC score).

Library module used by HZ3_YuE2_KaraokeVisualizer. It registers no ComfyUI node: the
former "Score & Lyrics Aligner" node was merged into the Karaoke & Audio Visualizer.

Alignment sources, best first:

1. ``forced``  - CTC forced alignment of the clean lyrics onto the audio (ideally the
                 separated vocal stem) with torchaudio's multilingual MMS_FA model.
                 Word-level timestamps anchored to what was actually sung.
2. ``whisper`` - clean lyric lines matched monotonically (DP) to Whisper / LRC segments.
3. ``score``   - nominal ABC timing (sections, vocal phrases, note onsets).

The clean lyrics text is authoritative: it is never rewritten, only timed.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import comfy.model_management as _mm
except ImportError:
    _mm = None

try:
    from .score_align import align_and_repair_abc, normalize_section_label
    from .score_analysis import inspect_score, lyric_syllables
except (ImportError, ValueError):
    from score_align import align_and_repair_abc, normalize_section_label
    from score_analysis import inspect_score, lyric_syllables

logger = logging.getLogger("HZ3.LyricAlign")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def normalize_for_comparison(text: str) -> str:
    """Normalize text for phonetic/fuzzy similarity comparison."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", without_accents)
    return " ".join(cleaned.split())


def text_similarity(s1: str, s2: str) -> float:
    """Normalized text similarity combining character sequence ratio and word Jaccard."""
    n1 = normalize_for_comparison(s1)
    n2 = normalize_for_comparison(s2)
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0
    seq_ratio = difflib.SequenceMatcher(None, n1, n2).ratio()
    words1 = set(n1.split())
    words2 = set(n2.split())
    jaccard = len(words1 & words2) / float(len(words1 | words2)) if words1 and words2 else 0.0
    return 0.6 * seq_ratio + 0.4 * jaccard


def romanize_word(word: str) -> str:
    """Map a lyric word onto the MMS_FA alphabet (a-z and apostrophe).

    Spanish/English only need accent stripping (á->a, ñ->n, ü->u). Digits, punctuation
    and non-Latin characters are dropped; a word that becomes empty is timed by
    interpolation between its neighbours instead of being force-aligned.
    """
    text = unicodedata.normalize("NFKD", str(word))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    for src, dst in (("ß", "ss"), ("æ", "ae"), ("œ", "oe"), ("ø", "o"), ("ł", "l"), ("đ", "d"),
                     ("þ", "th"), ("’", "'"), ("‘", "'"), ("`", "'")):
        text = text.replace(src, dst)
    return re.sub(r"[^a-z']", "", text).strip("'")


def word_syllable_count(w: str) -> int:
    """Count syllables in a word using lyric_syllables analysis."""
    clean = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]", "", w)
    if not clean:
        return 1
    return max(1, len(lyric_syllables(clean)))


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------
def parse_whisper_input(raw_input: str) -> list[dict]:
    """Parse Whisper input from a JSON array, a JSON object with 'segments', or LRC text."""
    if not raw_input or not str(raw_input).strip():
        return []
    raw = str(raw_input).strip()

    if raw.startswith("[") or raw.startswith("{"):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                data = data.get("segments", data.get("chunks", []))
            if isinstance(data, list):
                segments = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    s = float(item.get("start", 0.0) or 0.0)
                    e = float(item.get("end", s + 2.0) or (s + 2.0))
                    t = str(item.get("text", item.get("word", ""))).strip()
                    if t:
                        segments.append({"start": s, "end": max(s + 0.1, e), "text": t, "words": item.get("words", [])})
                if segments:
                    return segments
        except Exception:
            pass

    lrc_pattern = re.compile(r"^\[(\d{1,2}):(\d{2}(?:\.\d+)?)\](.*)$")
    lrc_entries = []
    for line in raw.splitlines():
        m = lrc_pattern.match(line.strip())
        if m:
            t_sec = int(m.group(1)) * 60.0 + float(m.group(2))
            content = m.group(3).strip()
            if content and not re.match(r"^[a-zA-Z]{2,4}:", content):
                lrc_entries.append((t_sec, content))
    if lrc_entries:
        lrc_entries.sort(key=lambda x: x[0])
        segments = []
        for idx, (t_start, content) in enumerate(lrc_entries):
            t_end = lrc_entries[idx + 1][0] if idx + 1 < len(lrc_entries) else t_start + 3.5
            segments.append({"start": t_start, "end": max(t_start + 0.2, t_end), "text": content, "words": []})
        return segments

    return []


SECTION_HEADER_RE = re.compile(r"^\s*\[([a-zA-Z0-9_\s\-]+)\]\s*$")
ANNOTATION_RE = re.compile(r"^\s*[\(\[\{][^\)\]\}]*[\)\]\}]\s*$")  # "(instrumental)", "(solo)", "[x2]"


def parse_clean_lyrics(lyrics_text: str) -> list[dict]:
    """Parse clean lyrics into lines preserving [Section] tags and their order.

    ``section_index`` increments at every header occurrence, so repeated [Chorus]
    headers stay separate blocks (required for section matching and star tokens).
    Lines that are only a parenthesized annotation ("(instrumental)") are not lyrics:
    they are neither aligned nor displayed.
    """
    lines = []
    curr_sec = ""
    section_index = 0
    for raw in (lyrics_text or "").splitlines():
        line_str = raw.strip()
        if not line_str:
            continue
        m = SECTION_HEADER_RE.match(line_str)
        if m:
            curr_sec = m.group(1).strip()
            section_index += 1
            continue
        if ANNOTATION_RE.match(line_str):
            continue
        lines.append({
            "text": line_str,
            "section": curr_sec,
            "norm_section": normalize_section_label(curr_sec) if curr_sec else "",
            "section_index": section_index,
        })
    return lines


# ---------------------------------------------------------------------------
# ABC score timeline
# ---------------------------------------------------------------------------
def extract_score_timeline(abc_text: str, lyrics_text: str = "", intro_silence: float = 0.0) -> dict[str, Any]:
    """Repair and parse the ABC into seconds-based notes, chords and sections.

    Never raises: when the strict parser rejects the score, tempo/meter/key come from
    the header and sections from the procedural repair pass (``parsed`` is False).
    """
    repaired_abc, structure = align_and_repair_abc(abc_text or "", lyrics_text=lyrics_text)
    info = None
    if repaired_abc and str(repaired_abc).strip():
        try:
            info = inspect_score(repaired_abc, lyrics_text)
        except Exception as exc:
            logger.warning(f"inspect_score strict parse fallback: {exc}")

    timeline: dict[str, Any] = {
        "repaired_abc": repaired_abc,
        "parsed": info is not None,
        "bpm": 120.0,
        "meter": "4/4",
        "key": "C",
        "score_seconds": 0.0,
        "vocal_notes": [],
        "ins_notes": [],
        "chords": [],
        "sections": [],
    }

    if info:
        bpm = float(info.get("bpm", 120)) or 120.0
        seconds_per_tick = 60.0 / (bpm * 256.0)
        roll = info.get("roll", {})
        timeline.update({
            "bpm": bpm,
            "meter": info.get("meter", "4/4"),
            "key": info.get("key", "C"),
            "score_seconds": float(info.get("seconds", 0.0)) + intro_silence,
        })
        for track, key, default_pitch in (("Vocal", "vocal_notes", 60), ("Ins", "ins_notes", 48)):
            for n in roll.get("tracks", {}).get(track, []):
                s = n["start"] * seconds_per_tick + intro_silence
                dur = n["duration"] * seconds_per_tick
                timeline[key].append({"start": s, "end": s + dur, "duration": dur, "pitch": int(n.get("pitch", default_pitch))})
        for c in roll.get("chords", []):
            timeline["chords"].append({"start": c["start"] * seconds_per_tick + intro_silence, "symbol": c.get("symbol", "")})
        timeline["chords"].sort(key=lambda x: x["start"])
        bar_ticks = roll.get("bar_ticks", 1024)
        for s in roll.get("sections", []):
            t_start = s["start"] * bar_ticks * seconds_per_tick + intro_silence
            t_end = t_start + s["bars"] * bar_ticks * seconds_per_tick
            name = str(s.get("name", "section")).strip()
            timeline["sections"].append({"name": name.lower(), "norm": normalize_section_label(name), "start": t_start, "end": t_end})
    else:
        text = str(repaired_abc or abc_text or "")
        bpm_m = re.search(r"^Q:1/4=(\d+)", text, re.MULTILINE)
        meter_m = re.search(r"^M:(\S+)", text, re.MULTILINE)
        key_m = re.search(r"^K:(\S+)", text, re.MULTILINE)
        bpm = float(bpm_m.group(1)) if bpm_m else 120.0
        timeline.update({
            "bpm": bpm,
            "meter": meter_m.group(1) if meter_m else "4/4",
            "key": key_m.group(1) if key_m else "C",
        })
        if structure:
            for s in structure:
                name = str(s.get("name", "section")).strip()
                timeline["sections"].append({
                    "name": name.lower(),
                    "norm": normalize_section_label(name),
                    "start": float(s["start"]) + intro_silence,
                    "end": float(s["end"]) + intro_silence,
                })
        else:
            approx_sec_dur = 16.0 * (60.0 / bpm)
            cur_t = intro_silence
            for s_name in re.findall(r"^%\s*([a-zA-Z0-9_\s\-]+)", text, re.MULTILINE):
                timeline["sections"].append({
                    "name": s_name.lower().strip(),
                    "norm": normalize_section_label(s_name),
                    "start": cur_t,
                    "end": cur_t + approx_sec_dur,
                })
                cur_t += approx_sec_dur
        timeline["score_seconds"] = timeline["sections"][-1]["end"] if timeline["sections"] else intro_silence + 60.0

    return timeline


# ---------------------------------------------------------------------------
# Procedural helpers (score phrases, word distribution)
# ---------------------------------------------------------------------------
def cluster_notes_into_phrases(notes: list[dict], min_pause: float = 0.25) -> list[list[dict]]:
    """Group musical notes into phrases separated by rests >= min_pause seconds."""
    phrases: list[list[dict]] = []
    cur_phrase: list[dict] = []
    for n in notes:
        if cur_phrase and n["start"] - cur_phrase[-1]["end"] >= min_pause:
            phrases.append(cur_phrase)
            cur_phrase = []
        cur_phrase.append(n)
    if cur_phrase:
        phrases.append(cur_phrase)
    return phrases


def adjust_phrases_to_target_count(phrases: list[list[dict]], target_count: int) -> list[list[dict]]:
    """Merge or split phrases until the phrase count matches the target line count."""
    if not phrases or target_count <= 0:
        return phrases
    phrases = list(phrases)

    while len(phrases) > target_count and len(phrases) > 1:
        min_gap = float("inf")
        min_idx = 0
        for p_i in range(len(phrases) - 1):
            g = phrases[p_i + 1][0]["start"] - phrases[p_i][-1]["end"]
            if g < min_gap:
                min_gap = g
                min_idx = p_i
        phrases[min_idx] = phrases[min_idx] + phrases[min_idx + 1]
        phrases.pop(min_idx + 1)

    while len(phrases) < target_count:
        best_p_idx = -1
        best_dur = -1.0
        for p_i, p in enumerate(phrases):
            dur = p[-1]["end"] - p[0]["start"]
            if dur > best_dur:
                best_dur = dur
                best_p_idx = p_i
        if best_p_idx == -1 or best_dur <= 0.5:
            break
        target_phrase = phrases[best_p_idx]
        if len(target_phrase) >= 2:
            max_gap = -1.0
            split_idx = len(target_phrase) // 2
            for n_i in range(len(target_phrase) - 1):
                gap = target_phrase[n_i + 1]["start"] - target_phrase[n_i]["end"]
                if gap > max_gap:
                    max_gap = gap
                    split_idx = n_i + 1
            phrases[best_p_idx] = target_phrase[:split_idx]
            phrases.insert(best_p_idx + 1, target_phrase[split_idx:])
        else:
            note = target_phrase[0]
            mid_t = (note["start"] + note["end"]) / 2.0
            phrases[best_p_idx] = [{"start": note["start"], "end": mid_t, "pitch": note.get("pitch", 60)}]
            phrases.insert(best_p_idx + 1, [{"start": mid_t, "end": note["end"], "pitch": note.get("pitch", 60)}])

    return phrases


def map_words_to_timeline(
    clean_line_text: str,
    line_start: float,
    line_end: float,
    notes_in_span: list[dict],
    syllable_mode: str = "ABC Notes",
) -> list[dict]:
    """Distribute the words of a line across [line_start, line_end] (no audio evidence)."""
    words = clean_line_text.split()
    if not words:
        return []

    line_dur = max(0.1, line_end - line_start)
    syl_counts = [word_syllable_count(w) for w in words]
    tot_syl = max(1, sum(syl_counts))
    timed_words = []

    if syllable_mode == "ABC Notes" and len(notes_in_span) >= 2:
        num_notes = len(notes_in_span)
        cur_note_idx = 0
        for w, sc in zip(words, syl_counts):
            note_slice_len = max(1, round(num_notes * (sc / float(tot_syl))))
            end_note_idx = min(num_notes - 1, cur_note_idx + note_slice_len - 1)
            w_start = notes_in_span[cur_note_idx]["start"]
            w_end = notes_in_span[end_note_idx]["end"]
            if w_end <= w_start:
                w_end = w_start + 0.25
            w_start = max(line_start, min(line_end - 0.05, w_start))
            w_end = max(w_start + 0.05, min(line_end, w_end))
            timed_words.append({"text": w, "start": round(w_start, 3), "end": round(w_end, 3)})
            cur_note_idx = min(num_notes - 1, end_note_idx + 1)
    else:
        cur_t = line_start
        for idx, (w, sc) in enumerate(zip(words, syl_counts)):
            weight = (sc / float(tot_syl)) if syllable_mode != "Equal Word Split" else (1.0 / len(words))
            w_dur = max(0.05, line_dur * weight)
            w_end = cur_t + w_dur if idx < len(words) - 1 else line_end
            timed_words.append({"text": w, "start": round(cur_t, 3), "end": round(w_end, 3)})
            cur_t = w_end

    return timed_words


def _make_line(l_item: dict, start: float, end: float, words: list[dict], **extra) -> dict:
    line = {
        "text": l_item["text"],
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "section": l_item.get("section", ""),
        "section_index": int(l_item.get("section_index", 0)),
        "words": words,
    }
    line.update(extra)
    return line


# ---------------------------------------------------------------------------
# Source 3: nominal ABC score timing
# ---------------------------------------------------------------------------
def _section_has_vocals(section: dict, melody: list[dict]) -> bool:
    return any(section["start"] - 0.05 <= n["start"] <= section["end"] + 0.05 for n in melody)


def _lines_from_phrases(lines: list[dict], phrases: list[list[dict]], span_start: float, span_end: float,
                        span_notes: list[dict], syllable_weighting: str) -> list[dict]:
    phrases = adjust_phrases_to_target_count(phrases, len(lines))
    out = []
    for idx, l_item in enumerate(lines):
        if phrases:
            p = phrases[min(idx, len(phrases) - 1)]
            l_start, l_end, p_notes = p[0]["start"], p[-1]["end"], p
        else:
            line_dur = max(1.0, (span_end - span_start) / max(1, len(lines)))
            l_start = span_start + idx * line_dur
            l_end = span_start + (idx + 1) * line_dur
            p_notes = [n for n in span_notes if l_start <= n["start"] <= l_end]
        if l_end <= l_start:
            l_end = l_start + 2.0
        words = map_words_to_timeline(l_item["text"], l_start, l_end, p_notes, syllable_mode=syllable_weighting)
        out.append(_make_line(l_item, l_start, l_end, words))
    return out


_ALLOC_W_LABEL = 0.5      # lyric block label differs from the ABC section label
_ALLOC_W_CAPACITY = 1.0   # |log(phrases / lines)| of a section that hosts lyrics
_ALLOC_W_SKIP = 0.8       # a section with sung phrases that receives no lyrics
_ALLOC_MAX_COMBOS = 300_000


def allocate_blocks_to_sections(blocks: list[dict], sections: list[dict]) -> list[int]:
    """Monotone assignment lyric block -> ABC section index minimizing label/capacity mismatch.

    ``blocks`` carry ``norm`` and ``lines``; ``sections`` carry ``norm`` and ``phrases``.
    Sections may host several consecutive blocks (a long ABC verse holding two lyric verses)
    and sung sections left without lyrics are penalized. Exhaustive when small, greedy otherwise.
    """
    B, S = len(blocks), len(sections)
    if B == 0 or S == 0:
        return []

    def total_cost(assign: list[int]) -> float:
        cost = 0.0
        lines_per_section = [0] * S
        for b, s in enumerate(assign):
            if blocks[b]["norm"] and sections[s]["norm"] and blocks[b]["norm"] != sections[s]["norm"]:
                cost += _ALLOC_W_LABEL
            lines_per_section[s] += len(blocks[b]["lines"])
        for s in range(S):
            phrases = len(sections[s]["phrases"])
            if lines_per_section[s] > 0:
                cost += _ALLOC_W_CAPACITY * abs(math.log((phrases + 1.0) / (lines_per_section[s] + 1.0)))
            elif phrases > 0:
                cost += _ALLOC_W_SKIP * min(1.0, phrases / 4.0)
        return cost

    combos = math.comb(B + S - 1, B)
    if combos <= _ALLOC_MAX_COMBOS:
        best: tuple[float, list[int]] | None = None
        assign: list[int] = []

        def rec(b: int, start: int) -> None:
            nonlocal best
            if b == B:
                c = total_cost(assign)
                if best is None or c < best[0]:
                    best = (c, list(assign))
                return
            for s in range(start, S):
                assign.append(s)
                rec(b + 1, s)
                assign.pop()

        rec(0, 0)
        return best[1] if best else list(range(min(B, S))) + [S - 1] * max(0, B - S)

    # Greedy: same label ahead of the cursor, else the next section that sings.
    out: list[int] = []
    cursor = 0
    for block in blocks:
        target = None
        for idx in range(cursor, S):
            if block["norm"] and block["norm"] == sections[idx]["norm"]:
                target = idx
                break
        if target is None:
            for idx in range(cursor, S):
                if sections[idx]["phrases"]:
                    target = idx
                    break
        if target is None:
            target = min(cursor, S - 1)
        out.append(target)
        cursor = target
    return out


def align_lines_by_score(clean_lines: list[dict], timeline: dict, syllable_weighting: str = "ABC Notes") -> list[dict]:
    """Nominal timing: lyric section blocks -> ABC sections -> vocal phrases -> words."""
    if not clean_lines:
        return []
    sections = timeline.get("sections", [])
    melody = timeline.get("vocal_notes") or timeline.get("ins_notes") or []
    total = float(timeline.get("score_seconds", 0.0)) or 60.0

    # Consecutive lines under the same header occurrence form one block.
    blocks: list[dict] = []
    for l in clean_lines:
        if blocks and blocks[-1]["index"] == l["section_index"]:
            blocks[-1]["lines"].append(l)
        else:
            blocks.append({"index": l["section_index"], "norm": l["norm_section"], "lines": [l]})
    has_tags = any(b["norm"] for b in blocks)

    if not has_tags or not sections:
        phrases = cluster_notes_into_phrases(melody, min_pause=0.25)
        return _lines_from_phrases(clean_lines, phrases, 0.0, total, melody, syllable_weighting)

    sec_data = []
    for s in sections:
        notes = [n for n in melody if s["start"] - 0.05 <= n["start"] <= s["end"] + 0.05]
        sec_data.append({**s, "notes": notes, "phrases": cluster_notes_into_phrases(notes, min_pause=0.25)})
    assignment = allocate_blocks_to_sections(blocks, sec_data)

    timed: list[dict] = []
    for s_idx, sec in enumerate(sec_data):
        hosted = [blocks[b] for b, a in enumerate(assignment) if a == s_idx]
        if not hosted:
            continue
        phrases = list(sec["phrases"])
        total_lines = sum(len(b["lines"]) for b in hosted)
        if len(hosted) == 1:
            timed.extend(_lines_from_phrases(hosted[0]["lines"], phrases, sec["start"], sec["end"], sec["notes"], syllable_weighting))
            continue
        # Several lyric blocks share one ABC section: split its phrases (or time) by line count.
        phrases = adjust_phrases_to_target_count(phrases, total_lines) if phrases else []
        p_cursor = 0
        t_cursor = sec["start"]
        for block in hosted:
            n_lines = len(block["lines"])
            if len(phrases) >= total_lines:
                share = phrases[p_cursor:p_cursor + n_lines]
                p_cursor += n_lines
                b_start, b_end = share[0][0]["start"], share[-1][-1]["end"]
            else:
                share = []
                b_start = t_cursor
                b_end = t_cursor + (sec["end"] - sec["start"]) * n_lines / float(total_lines)
                t_cursor = b_end
            b_notes = [n for n in sec["notes"] if b_start - 0.05 <= n["start"] <= b_end + 0.05]
            timed.extend(_lines_from_phrases(block["lines"], share, b_start, b_end, b_notes, syllable_weighting))
    return timed


# ---------------------------------------------------------------------------
# Source 2: Whisper / LRC segments
# ---------------------------------------------------------------------------
def align_clean_lines_to_whisper(clean_lines: list[dict], whisper_segments: list[dict]) -> list[tuple[dict, float, float]]:
    """Monotonically align clean lyric lines to Whisper segment timestamps (dynamic programming)."""
    N = len(clean_lines)
    M = len(whisper_segments)
    if N == 0:
        return []
    if M == 0:
        return [(l, 0.0, 3.0) for l in clean_lines]

    sim_matrix = [[text_similarity(clean_lines[i]["text"], whisper_segments[j]["text"]) for j in range(M)] for i in range(N)]

    INF = float("inf")
    dp = [[INF] * (M + 1) for _ in range(N + 1)]
    parent: list[list[tuple | None]] = [[None] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0.0
    for j in range(1, M + 1):
        dp[0][j] = dp[0][j - 1] + 0.35
        parent[0][j] = (0, j - 1, "skip_whisper")

    for i in range(1, N + 1):
        dp[i][0] = dp[i - 1][0] + 0.8
        parent[i][0] = (i - 1, 0, "skip_clean")
        for j in range(1, M + 1):
            best_cost = dp[i - 1][j - 1] + (1.0 - sim_matrix[i - 1][j - 1])
            best_op = (i - 1, j - 1, "1to1")
            if j >= 2:
                merged_whisper = whisper_segments[j - 2]["text"] + " " + whisper_segments[j - 1]["text"]
                c = dp[i - 1][j - 2] + (1.0 - text_similarity(clean_lines[i - 1]["text"], merged_whisper))
                if c < best_cost:
                    best_cost, best_op = c, (i - 1, j - 2, "1to2")
            if i >= 2:
                merged_clean = clean_lines[i - 2]["text"] + " " + clean_lines[i - 1]["text"]
                c = dp[i - 2][j - 1] + (1.0 - text_similarity(merged_clean, whisper_segments[j - 1]["text"]))
                if c < best_cost:
                    best_cost, best_op = c, (i - 2, j - 1, "2to1")
            c = dp[i][j - 1] + 0.35
            if c < best_cost:
                best_cost, best_op = c, (i, j - 1, "skip_whisper")
            c = dp[i - 1][j] + 0.80
            if c < best_cost:
                best_cost, best_op = c, (i - 1, j, "skip_clean")
            dp[i][j] = best_cost
            parent[i][j] = best_op

    cur_i, cur_j = N, M
    matched_ranges: dict[int, list[int]] = {idx: [] for idx in range(N)}
    while cur_i > 0 or cur_j > 0:
        p = parent[cur_i][cur_j]
        if not p:
            break
        prev_i, prev_j, op = p
        if op == "1to1":
            matched_ranges[cur_i - 1].append(cur_j - 1)
        elif op == "1to2":
            matched_ranges[cur_i - 1].extend([cur_j - 2, cur_j - 1])
        elif op == "2to1":
            matched_ranges[cur_i - 2].append(cur_j - 1)
            matched_ranges[cur_i - 1].append(cur_j - 1)
        cur_i, cur_j = prev_i, prev_j

    results = []
    last_known_end = 0.0
    for idx, l in enumerate(clean_lines):
        w_indices = sorted(set(matched_ranges[idx]))
        if w_indices:
            start_t = whisper_segments[w_indices[0]]["start"]
            end_t = whisper_segments[w_indices[-1]]["end"]
            shared_lines = [k for k, v in matched_ranges.items() if set(v) == set(w_indices)]
            if len(shared_lines) > 1:
                pos = shared_lines.index(idx)
                part_dur = max(0.2, end_t - start_t) / len(shared_lines)
                line_s = start_t + pos * part_dur
                results.append((l, line_s, line_s + part_dur))
                last_known_end = line_s + part_dur
            else:
                results.append((l, start_t, end_t))
                last_known_end = end_t
        else:
            est_s = last_known_end + 0.3
            results.append((l, est_s, est_s + 3.0))
            last_known_end = est_s + 3.0
    return results


def align_lines_by_whisper(clean_lines: list[dict], whisper_segments: list[dict], timeline: dict,
                           syllable_weighting: str = "ABC Notes") -> list[dict]:
    """Line bounds from Whisper segments; words from ABC note onsets inside each span."""
    melody = timeline.get("vocal_notes") or timeline.get("ins_notes") or []
    timed = []
    for l_item, l_start, l_end in align_clean_lines_to_whisper(clean_lines, whisper_segments):
        span_notes = [n for n in melody if l_start - 0.15 <= n["start"] <= l_end + 0.15]
        words = map_words_to_timeline(l_item["text"], l_start, l_end, span_notes, syllable_mode=syllable_weighting)
        timed.append(_make_line(l_item, l_start, l_end, words))
    return timed


# ---------------------------------------------------------------------------
# Source 1: CTC forced alignment on audio (torchaudio MMS_FA)
# ---------------------------------------------------------------------------
FA_SAMPLE_RATE = 16000
FA_FRAME_SECONDS = 0.02          # 320-sample stride of the MMS_FA wav2vec2 encoder
FA_WINDOW_SECONDS = 30.0         # emissions are computed per window ...
FA_CONTEXT_SECONDS = 3.0         # ... with this much extra audio on both sides for context
FA_STAR_LOGPROB = -2.0           # per-frame cost of the wildcard token (unscripted vocals)
FA_MODEL_FILENAME = "mms_fa_ctc_alignment_mling_uroman.pt"
FA_MIN_MEAN_CONFIDENCE = 0.10    # below this the audio most likely has no sung lyrics
FA_LOW_LINE_CONFIDENCE = 0.40    # reported as weak in the alignment report
FA_HIGH_LINE_CONFIDENCE = 0.60   # trusted anchors for the audio-vs-score offset
FA_DEAD_LINE_CONFIDENCE = 0.20   # words forced onto frames that do not support them
FA_TRUNCATION_RATIO = 0.85       # audio shorter than this fraction of the score: expect unsung lyrics
FA_MAX_TRIM_PASSES = 3

_FA_MODEL_CACHE: dict[str, Any] = {}


def _model_roots() -> list[Path]:
    """Candidate ``models`` roots: extra search-path roots (Comfy Desktop's shared models
    folder) before the default ``models_dir``; a bare checkout falls back to ./models."""
    roots: list[Path] = []
    if folder_paths is None:
        return [Path(__file__).resolve().parent / "models"]
    default_root = Path(folder_paths.models_dir).resolve()
    try:
        for p in folder_paths.get_folder_paths("checkpoints"):
            root = Path(p).resolve().parent
            if root != default_root and root not in roots:
                roots.append(root)
    except Exception:
        pass
    roots.append(default_root)
    return roots


def fa_model_dir() -> Path:
    """Directory holding the MMS_FA weights: wherever they already are, else the first root."""
    roots = _model_roots()
    for root in roots:
        if (root / "mms_fa" / FA_MODEL_FILENAME).exists():
            return root / "mms_fa"
    for root in roots:
        if root.exists():
            return root / "mms_fa"
    return roots[-1] / "mms_fa"


def resolve_alignment_device(device: str = "auto") -> str:
    device = (device or "auto").lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def load_forced_aligner() -> tuple[torch.nn.Module, dict[str, int]]:
    """Load (and cache on CPU) the MMS_FA model and its character dictionary."""
    if "model" not in _FA_MODEL_CACHE:
        from torchaudio.pipelines import MMS_FA as bundle
        model_dir = fa_model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        if not (model_dir / FA_MODEL_FILENAME).exists():
            logger.info(f"Downloading MMS_FA forced-alignment model (~1.2 GB) to {model_dir}")
        model = bundle.get_model(
            with_star=False,
            dl_kwargs={"model_dir": str(model_dir), "file_name": FA_MODEL_FILENAME, "map_location": "cpu"},
        )
        model.eval()
        _FA_MODEL_CACHE["model"] = model
        _FA_MODEL_CACHE["dictionary"] = bundle.get_dict(star=None)
    return _FA_MODEL_CACHE["model"], _FA_MODEL_CACHE["dictionary"]


def audio_to_mono16k(audio: dict | None) -> np.ndarray | None:
    """ComfyUI AUDIO dict -> float32 mono waveform at 16 kHz (None when absent)."""
    if not isinstance(audio, dict) or "waveform" not in audio:
        return None
    waveform = audio["waveform"].detach().float().cpu()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 2:
        waveform = waveform.mean(0)
    sr = int(audio.get("sample_rate", FA_SAMPLE_RATE))
    if sr != FA_SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, FA_SAMPLE_RATE)
    return waveform.numpy().astype(np.float32)


def _check_interrupt():
    if _mm is not None:
        _mm.throw_exception_if_processing_interrupted()


def compute_emissions(model: torch.nn.Module, mono16k: np.ndarray, device: str) -> torch.Tensor:
    """Frame log-probabilities (T, C) for the whole waveform, computed in context windows."""
    stride = int(round(FA_SAMPLE_RATE * FA_FRAME_SECONDS))
    win = int(FA_WINDOW_SECONDS * FA_SAMPLE_RATE)
    ctx = int(FA_CONTEXT_SECONDS * FA_SAMPLE_RATE)
    f_ctx, f_win = ctx // stride, win // stride

    mono = torch.from_numpy(np.ascontiguousarray(mono16k, dtype=np.float32))
    n = int(mono.numel())
    total_frames = max(1, (n + stride - 1) // stride)
    padded = torch.nn.functional.pad(mono, (ctx, ctx + win))

    pieces = []
    frames_done = 0
    for start in range(0, n, win):
        _check_interrupt()
        seg = padded[start:start + win + 2 * ctx].unsqueeze(0).to(device)
        with torch.inference_mode():
            emission, _ = model(seg)
        emission = emission[0].float()
        need = min(f_win, total_frames - frames_done)
        pieces.append(emission[f_ctx:f_ctx + need])
        frames_done += need
        del seg, emission
    out = torch.cat(pieces, dim=0)
    if out.shape[0] < total_frames:  # never expected; keep the frame clock honest
        out = torch.nn.functional.pad(out, (0, 0, 0, total_frames - out.shape[0]), value=math.log(1e-6))
    return out


FA_SILENCE_DB_BELOW_REF = 45.0   # frames this far under the loud level are silence: blank only
FA_SILENCE_PENALTY = -20.0


def gate_silence(emission: torch.Tensor, mono16k: np.ndarray) -> tuple[torch.Tensor, float]:
    """Force near-silent frames to the blank token.

    wav2vec2 normalizes its features per frame, so digital silence (fade-outs, the gap
    before the first note) turns into confident garbage characters; without this gate the
    last line of a song can land in the silent tail with a perfect score.
    Returns the gated emission and the fraction of silent frames.
    """
    stride = int(round(FA_SAMPLE_RATE * FA_FRAME_SECONDS))
    rms = _rms_envelope(np.asarray(mono16k, dtype=np.float32), hop=stride, win=2 * stride)
    frames = emission.shape[0]
    if len(rms) < frames:
        rms = np.pad(rms, (0, frames - len(rms)), mode="edge" if len(rms) else "constant")
    rms = rms[:frames]
    db = 20.0 * np.log10(rms + 1e-9)
    ref = float(np.percentile(db, 95))
    silent = torch.from_numpy(db < ref - FA_SILENCE_DB_BELOW_REF).to(emission.device)
    if not bool(silent.any()):
        return emission, 0.0
    gated = emission.clone()
    gated[silent, 1:] = FA_SILENCE_PENALTY
    gated[silent, 0] = 0.0
    return gated, float(silent.float().mean())


def build_alignment_units(clean_lines: list[dict], star_between_sections: bool = True) -> list[dict]:
    """Words to align plus wildcard units at the edges and between lyric sections."""
    units: list[dict] = [{"kind": "star"}]
    prev_section_index = None
    for li, line in enumerate(clean_lines):
        sec_idx = line.get("section_index", 0)
        if star_between_sections and prev_section_index is not None and sec_idx != prev_section_index:
            if units[-1]["kind"] != "star":
                units.append({"kind": "star"})
        prev_section_index = sec_idx
        for wi, word in enumerate(line["text"].split()):
            rom = romanize_word(word)
            units.append({"kind": "word" if rom else "skip", "line": li, "word": wi, "text": word, "rom": rom})
    if units[-1]["kind"] != "star":
        units.append({"kind": "star"})
    return units


def run_forced_alignment(emission: torch.Tensor, units: list[dict], dictionary: dict[str, int]) -> None:
    """Viterbi-align the unit token sequence to the emissions; annotates word units in place."""
    import torchaudio.functional as F

    star_id = emission.shape[1]
    star_col = torch.full((emission.shape[0], 1), FA_STAR_LOGPROB, dtype=emission.dtype, device=emission.device)
    emission = torch.cat([emission, star_col], dim=1)

    token_groups: list[list[int]] = []
    group_units: list[dict] = []
    for u in units:
        if u["kind"] == "word":
            ids = [dictionary[c] for c in u["rom"] if c in dictionary]
            if not ids:
                u["kind"] = "skip"
                continue
        elif u["kind"] == "star":
            ids = [star_id]
        else:
            continue
        token_groups.append(ids)
        group_units.append(u)

    flat = [t for g in token_groups for t in g]
    if not flat:
        return
    repeats = sum(1 for a, b in zip(flat, flat[1:]) if a == b)
    if emission.shape[0] < len(flat) + repeats:
        raise ValueError(
            f"Audio too short for the lyrics: {emission.shape[0]} frames for {len(flat) + repeats} required tokens."
        )

    targets = torch.tensor([flat], dtype=torch.int32, device=emission.device)
    aligned, scores = F.forced_align(emission.unsqueeze(0), targets, blank=0)
    spans = F.merge_tokens(aligned[0], scores[0].exp(), blank=0)

    pos = 0
    for u, g in zip(group_units, token_groups):
        s = spans[pos:pos + len(g)]
        pos += len(g)
        u["char_spans"] = [(int(sp.start), int(sp.end), float(sp.score)) for sp in s]
        u["start_frame"] = int(s[0].start)
        u["end_frame"] = int(s[-1].end)
        u["score"] = float(sum(sp.score for sp in s) / len(s))
        u["min_frames"] = len(g) + sum(1 for a, b in zip(g, g[1:]) if a == b)
    _repair_stray_fragments(units)


FA_STRAY_CHAR_GAP = int(2.0 / FA_FRAME_SECONDS)   # a lone character this far from the rest of its word
FA_STRAY_WORD_GAP = int(2.5 / FA_FRAME_SECONDS)   # tiny words this far from the rest of their line


def _repair_stray_fragments(units: list[dict]) -> None:
    """Undo two CTC artifacts: a lone character or a tiny word matched far away from its line.

    Blank frames are free during instrumental passages, so the Viterbi path may drop the
    'p' of "pude" onto a transient ten seconds early, or start a line with "Y" at some
    stray vowel long before the phrase. The fragments are snapped next to the main cluster
    (order is preserved), and the word/line spans shrink back to what was really sung.
    """
    # 1. Inside a word: a single character separated from the rest by a long silence.
    for u in units:
        spans = u.get("char_spans")
        if not spans or len(spans) < 2:
            continue
        gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
        if max(gaps) < FA_STRAY_CHAR_GAP:
            continue
        i = max(range(len(gaps)), key=lambda k: gaps[k])
        head, tail = spans[:i + 1], spans[i + 1:]
        if len(head) == 1 and len(tail) >= 1:
            spans = [(tail[0][0] - 1, tail[0][0], head[0][2])] + tail
        elif len(tail) == 1 and len(head) >= 1:
            spans = head + [(head[-1][1], head[-1][1] + 1, tail[0][2])]
        else:
            continue
        u["char_spans"] = spans
        u["start_frame"], u["end_frame"] = spans[0][0], spans[-1][1]
        u["score"] = float(sum(sp[2] for sp in spans) / len(spans))

    # 2. Inside a line: tiny words (<= 3 letters in total) stranded far from the phrase.
    by_line: dict[int, list[dict]] = {}
    for u in units:
        if u["kind"] == "word" and "start_frame" in u:
            by_line.setdefault(u["line"], []).append(u)
    for words in by_line.values():
        if len(words) < 2:
            continue
        gaps = [words[i + 1]["start_frame"] - words[i]["end_frame"] for i in range(len(words) - 1)]
        i = max(range(len(gaps)), key=lambda k: gaps[k])
        if gaps[i] < FA_STRAY_WORD_GAP:
            continue
        head, tail = words[:i + 1], words[i + 1:]
        if sum(len(w["rom"]) for w in head) <= 3:
            cursor = tail[0]["start_frame"]
            for w in reversed(head):
                span = max(1, len(w["rom"]))
                w["start_frame"], w["end_frame"] = cursor - span - 1, cursor - 1
                cursor = w["start_frame"]
        elif sum(len(w["rom"]) for w in tail) <= 3:
            cursor = head[-1]["end_frame"]
            for w in tail:
                span = max(1, len(w["rom"]))
                w["start_frame"], w["end_frame"] = cursor + 1, cursor + 1 + span
                cursor = w["end_frame"]


def _rms_envelope(mono16k: np.ndarray, hop: int = 160, win: int = 400) -> np.ndarray:
    n = len(mono16k)
    if n < win:
        return np.zeros(1, dtype=np.float32)
    frames = 1 + (n - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(frames)[:, None]
    return np.sqrt(np.mean(np.square(mono16k[idx]), axis=1)).astype(np.float32)


def _extend_line_ends(lines: list[dict], mono16k: np.ndarray | None, is_stem: bool, audio_seconds: float) -> None:
    """CTC marks the onset of the last character; extend line ends over the sung release."""
    hop_s = 0.01
    rms = _rms_envelope(mono16k) if (mono16k is not None and is_stem) else None
    for idx, line in enumerate(lines):
        if not line["words"] or line.get("beyond_audio"):
            continue
        last = line["words"][-1]
        next_start = lines[idx + 1]["start"] if idx + 1 < len(lines) else audio_seconds
        limit = min(max(last["end"], next_start - 0.05), last["end"] + (1.5 if is_stem else 0.35))
        new_end = last["end"]
        if rms is not None:
            a = int(last["start"] / hop_s)
            b = max(a + 1, int(last["end"] / hop_s))
            ref = float(rms[a:b].max()) if b <= len(rms) and b > a else 0.0
            thr = 0.3 * ref
            t = last["end"]
            while t + hop_s <= limit and int(t / hop_s) < len(rms) and rms[int(t / hop_s)] > thr and ref > 0:
                t += hop_s
            new_end = t
        else:
            new_end = limit
        last["end"] = round(max(last["end"], new_end), 3)
        line["end"] = round(max(line["end"], last["end"]), 3)


def _lines_from_units(clean_lines: list[dict], units: list[dict], audio_seconds: float) -> list[dict]:
    """Turn aligned word units into timed lines, interpolating unalignable words/lines."""
    per_line: dict[int, list[dict]] = {i: [] for i in range(len(clean_lines))}
    for u in units:
        if u["kind"] in ("word", "skip"):
            per_line[u["line"]].append(u)

    lines: list[dict] = []
    for li, l_item in enumerate(clean_lines):
        words_units = per_line[li]
        aligned = [u for u in words_units if u["kind"] == "word" and "start_frame" in u]
        if aligned:
            l_start = aligned[0]["start_frame"] * FA_FRAME_SECONDS
            l_end = max(aligned[-1]["end_frame"] * FA_FRAME_SECONDS, l_start + 0.05)
            conf = float(sum(u["score"] for u in aligned) / len(aligned))
        else:
            l_start = l_end = None
            conf = 0.0
        lines.append(_make_line(l_item, l_start or 0.0, l_end or 0.0, [], confidence=round(conf, 3)))
        lines[-1]["_units"] = words_units
        lines[-1]["_placed"] = l_start is not None

    # Lines with no alignable word: fit them into the gap between their neighbours.
    for li, line in enumerate(lines):
        if line["_placed"]:
            continue
        prev_end = next((lines[k]["end"] for k in range(li - 1, -1, -1) if lines[k]["_placed"]), 0.0)
        next_start = next((lines[k]["start"] for k in range(li + 1, len(lines)) if lines[k]["_placed"]), audio_seconds)
        gap = max(0.3, next_start - prev_end)
        dur = min(2.0, gap * 0.8)
        line["start"] = round(prev_end + (gap - dur) / 2.0, 3)
        line["end"] = round(line["start"] + dur, 3)

    def spread(target: list[dict], skip_units: list[dict], t0: float, t1: float) -> None:
        step = max(0.02, (t1 - t0) / max(1, len(skip_units)))
        for k, su in enumerate(skip_units):
            target.append({"text": su["text"], "start": round(t0 + k * step, 3), "end": round(t0 + (k + 1) * step, 3)})

    for line in lines:
        units_here = line.pop("_units")
        line.pop("_placed")
        timed_words: list[dict] = []
        aligned_idx = [k for k, u in enumerate(units_here) if u["kind"] == "word" and "start_frame" in u]
        if not aligned_idx:
            spread(timed_words, units_here, line["start"], line["end"])
            line["words"] = timed_words
            continue

        first = aligned_idx[0]
        lead = units_here[:first]
        if lead:  # unalignable words before the first aligned one: short slots ahead of it
            first_start = units_here[first]["start_frame"] * FA_FRAME_SECONDS
            lead_start = max(0.0, first_start - 0.25 * len(lead))
            line["start"] = round(min(line["start"], lead_start), 3)
            spread(timed_words, lead, lead_start, first_start)

        cursor = line["start"]
        pending: list[dict] = []
        for u in units_here[first:]:
            if u["kind"] == "word" and "start_frame" in u:
                w_start = u["start_frame"] * FA_FRAME_SECONDS
                w_end = max(u["end_frame"] * FA_FRAME_SECONDS, w_start + 0.04)
                if pending:
                    spread(timed_words, pending, cursor, max(cursor + 0.05, w_start))
                    pending = []
                timed_words.append({"text": u["text"], "start": round(w_start, 3), "end": round(w_end, 3), "score": round(u["score"], 3)})
                cursor = w_end
            else:
                pending.append(u)
        if pending:  # unalignable trailing words: short slots after the last aligned one
            tail_end = cursor + 0.25 * len(pending)
            line["end"] = round(max(line["end"], tail_end), 3)
            spread(timed_words, pending, cursor, tail_end)
        line["words"] = timed_words
    return lines


def _squeezed_word(u: dict) -> bool | None:
    """True/False for long words (>= 4 letters); None for short words (no evidence either way)."""
    if len(u.get("rom", "")) < 4 or "start_frame" not in u:
        return None
    return (u["end_frame"] - u["start_frame"]) <= u["min_frames"] + 1


def _line_evidence(clean_lines: list[dict], units: list[dict]) -> list[dict]:
    """Per line: mean word confidence, fraction of squeezed long words, aligned word count."""
    scores: dict[int, list[float]] = {i: [] for i in range(len(clean_lines))}
    squeezed: dict[int, list[bool]] = {i: [] for i in range(len(clean_lines))}
    for u in units:
        if u["kind"] == "word" and "start_frame" in u:
            scores[u["line"]].append(u["score"])
            flag = _squeezed_word(u)
            if flag is not None:
                squeezed[u["line"]].append(flag)
    evidence = []
    for li in range(len(clean_lines)):
        evidence.append({
            "conf": float(np.mean(scores[li])) if scores[li] else None,
            "squeezed": (sum(squeezed[li]) / len(squeezed[li])) if squeezed[li] else None,
            "n_words": len(scores[li]),
        })
    return evidence


def _alignment_health(clean_lines: list[dict], units: list[dict]) -> dict[str, Any]:
    """Dead lines (forced onto unsupported frames) and whether the tail of the transcript is dead."""
    evidence = _line_evidence(clean_lines, units)
    evaluated = [i for i, e in enumerate(evidence) if e["conf"] is not None]
    dead = [
        i for i in evaluated
        if evidence[i]["conf"] < FA_DEAD_LINE_CONFIDENCE
        or (evidence[i]["squeezed"] is not None and evidence[i]["squeezed"] >= 0.6)
    ]
    # The tail is dead when the last line is dead, or merely weak right after a dead line
    # (a take cut mid-phrase leaves a half-sung line followed by scraps).
    tail_dead = bool(evaluated) and (
        evaluated[-1] in dead
        or (len(evaluated) >= 2 and evaluated[-2] in dead and evidence[evaluated[-1]]["conf"] < FA_LOW_LINE_CONFIDENCE)
    )
    healthy = bool(evaluated) and not tail_dead and len(dead) <= max(1, int(0.1 * len(evaluated)))
    return {"evidence": evidence, "dead": dead, "tail_dead": tail_dead, "healthy": healthy, "evaluated": evaluated}


def _trailing_dead_cut(health: dict[str, Any]) -> int | None:
    """First line of the trailing run of dead lines, or None when the tail is alive."""
    if not health["tail_dead"]:
        return None
    dead = set(health["dead"])
    cut = None
    for li in reversed(health["evaluated"]):
        if li in dead:
            cut = li
        else:
            break
    return cut


def forced_align_lines(
    clean_lines: list[dict],
    mono16k: np.ndarray,
    device: str = "auto",
    is_stem: bool = False,
    star_between_sections: bool = True,
    expected_seconds: float = 0.0,
) -> tuple[list[dict], dict[str, Any]]:
    """Forced-align clean lyric lines onto the audio. Returns (timed_lines, stats).

    CTC forced alignment must place every token inside the audio. When the generation is
    shorter than the lyrics (``expected_seconds`` from the ABC clearly exceeds the audio),
    the surplus text would be crammed into instrumental gaps and, with repeated verses or
    choruses, pull the wrong copy of a line onto the sung audio. So the number of leading
    lines that fit is searched (binary search on a health check) and the rest is flagged
    ``beyond_audio``. With a complete take, only a dead trailing run is trimmed.
    """
    if not clean_lines:
        return [], {"error": "no lyrics"}
    if mono16k is None or len(mono16k) < FA_SAMPLE_RATE // 2:
        raise ValueError("Alignment audio is empty or shorter than half a second.")

    device = resolve_alignment_device(device)
    model, dictionary = load_forced_aligner()
    audio_seconds = len(mono16k) / float(FA_SAMPLE_RATE)
    stats: dict[str, Any] = {"device": device, "audio_seconds": round(audio_seconds, 2), "trimmed_lines": 0, "passes": 0}

    if device.startswith("cuda") and _mm is not None:
        try:
            _mm.free_memory(2 * 1024 ** 3, torch.device(device))
        except Exception:
            pass

    try:
        model.to(device)
        try:
            emission = compute_emissions(model, mono16k, device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("MMS_FA ran out of GPU memory; retrying forced alignment on CPU.")
            model.to("cpu")
            torch.cuda.empty_cache()
            device = "cpu"
            stats["device"] = "cpu (cuda OOM)"
            emission = compute_emissions(model, mono16k, device)
    finally:
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    emission, silent_fraction = gate_silence(emission, mono16k)
    stats["silent_fraction"] = round(silent_fraction, 3)

    cache: dict[int, tuple[list[dict], dict[str, Any]]] = {}

    def attempt(k: int) -> tuple[list[dict], dict[str, Any]]:
        """Align the first k lines; cached. Infeasible transcripts count as fully dead."""
        if k not in cache:
            stats["passes"] += 1
            subset = clean_lines[:k]
            units_k = build_alignment_units(subset, star_between_sections=star_between_sections)
            try:
                run_forced_alignment(emission, units_k, dictionary)
                health = _alignment_health(subset, units_k)
            except ValueError as exc:  # audio shorter than the minimum CTC length of the text
                logger.info("forced alignment infeasible for %d lines: %s", k, exc)
                health = {"evidence": [], "dead": list(range(k)), "tail_dead": True, "healthy": False, "evaluated": []}
            cache[k] = (units_k, health)
        return cache[k]

    n = len(clean_lines)
    keep = n
    units, health = attempt(n)
    truncated_take = expected_seconds > 0 and audio_seconds < FA_TRUNCATION_RATIO * expected_seconds
    if not health["healthy"] and truncated_take:
        # Largest prefix whose alignment is healthy (monotone enough for a binary search).
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if attempt(mid)[1]["healthy"] else (lo, mid - 1)
        keep = lo if lo > 0 else n
        units, health = attempt(keep)
        stats["truncation_search"] = True
    for _ in range(FA_MAX_TRIM_PASSES):
        # Complete take (or after the search): drop a dead trailing run the singer never reached.
        cut = _trailing_dead_cut(health)
        if cut is None or cut == 0:
            break
        keep = cut
        units, health = attempt(keep)
    del emission

    active = clean_lines[:keep]
    dropped = clean_lines[keep:]
    lines = _lines_from_units(active, units, audio_seconds)
    _extend_line_ends(lines, mono16k, is_stem, audio_seconds)

    # Lyrics the audio never reached (generation cut short): timed past the end, flagged.
    cursor = max(audio_seconds, lines[-1]["end"] if lines else 0.0) + 1.0
    for l_item in dropped:
        words = map_words_to_timeline(l_item["text"], cursor, cursor + 2.0, [], syllable_mode="Phonetic Syllables")
        lines.append(_make_line(l_item, cursor, cursor + 2.0, words, confidence=0.0, beyond_audio=True))
        cursor += 2.5
    stats["trimmed_lines"] = len(dropped)

    word_scores = [w["score"] for l in lines for w in l["words"] if "score" in w]
    stats["words_aligned"] = len(word_scores)
    stats["mean_confidence"] = round(float(np.mean(word_scores)), 3) if word_scores else 0.0
    stats["low_confidence_lines"] = sum(1 for l in lines if not l.get("beyond_audio") and l.get("confidence", 0.0) < FA_LOW_LINE_CONFIDENCE)
    stats["dead_lines"] = [i + 1 for i in health["dead"]]
    return lines, stats


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
ALIGNMENT_MODES = {
    "Auto (Audio → Whisper → ABC)": ("forced", "whisper", "score"),
    "Forced Alignment (audio)": ("forced", "score"),
    "Whisper Segments": ("whisper", "score"),
    "ABC Score Timing": ("score",),
}
_MODE_ALIASES = {"auto": ALIGNMENT_MODES["Auto (Audio → Whisper → ABC)"], "forced": ("forced", "score"),
                 "whisper": ("whisper", "score"), "score": ("score",), "abc": ("score",)}


def _source_order(mode: str) -> tuple[str, ...]:
    if mode in ALIGNMENT_MODES:
        return ALIGNMENT_MODES[mode]
    key = (mode or "auto").strip().lower()
    for alias, order in _MODE_ALIASES.items():
        if key.startswith(alias):
            return order
    return _MODE_ALIASES["auto"]


def estimate_score_offset(lines: list[dict], nominal_lines: list[dict], timeline: dict, max_lines: int = 6) -> float | None:
    """Seconds the audio runs late (+) or early (-) versus the nominal ABC timeline, or None.

    Two independent estimates must agree: the median line-start delta over the first sung
    lines, and the first sung word versus the first ABC vocal note. A lone estimate is only
    trusted when small. Used to shift chords/sections, never the aligned lyrics themselves.
    """
    def collect(min_conf: float) -> tuple[list[float], float | None]:
        deltas = []
        first: float | None = None
        for aligned, nominal in zip(lines, nominal_lines):
            if aligned.get("beyond_audio"):
                break
            if aligned.get("confidence", 1.0) < min_conf:
                continue
            if first is None:
                first = aligned["start"]
            deltas.append(aligned["start"] - nominal["start"])
            if len(deltas) >= max_lines:
                break
        return deltas, first

    deltas, first_word = collect(FA_HIGH_LINE_CONFIDENCE)
    if len(deltas) < 2:
        deltas, first_word = collect(FA_LOW_LINE_CONFIDENCE)
    d_lines = float(statistics.median(deltas)) if deltas else None
    d_first = None
    if first_word is not None and timeline.get("vocal_notes"):
        d_first = first_word - timeline["vocal_notes"][0]["start"]

    candidates = [d for d in (d_lines, d_first) if d is not None and abs(d) <= 20.0]
    if len(candidates) == 2:
        return (candidates[0] + candidates[1]) / 2.0 if abs(candidates[0] - candidates[1]) <= 1.5 else None
    if len(candidates) == 1 and abs(candidates[0]) <= 3.0:
        return candidates[0]
    return None


def shift_lines(lines: list[dict], offset: float) -> None:
    if not offset:
        return
    for line in lines:
        line["start"] = round(max(0.0, line["start"] + offset), 3)
        line["end"] = round(max(line["start"] + 0.05, line["end"] + offset), 3)
        for w in line["words"]:
            w["start"] = round(max(0.0, w["start"] + offset), 3)
            w["end"] = round(max(w["start"] + 0.02, w["end"] + offset), 3)


def make_lrc(lines: list[dict]) -> str:
    out = []
    for item in lines:
        if item.get("beyond_audio"):
            continue
        s = item["start"]
        out.append(f"[{int(s // 60):02d}:{s % 60:05.2f}] {item['text']}")
    return "\n".join(out)


def derive_sections_from_lines(lines: list[dict], duration: float, gap_threshold: float = 4.0) -> list[dict]:
    """Section HUD timeline from the aligned lyric blocks (audio truth beats nominal ABC)."""
    sung = [l for l in lines if not l.get("beyond_audio") and l["words"]]
    if not sung or not any(l.get("section") for l in sung):
        return []
    blocks: list[dict] = []
    for l in sung:
        name = (l.get("section") or "verse").upper()
        if blocks and blocks[-1]["index"] == l["section_index"]:
            blocks[-1]["end"] = max(blocks[-1]["end"], l["end"])
        else:
            blocks.append({"index": l["section_index"], "name": name, "start": l["start"], "end": l["end"]})

    sections: list[dict] = []
    if blocks[0]["start"] > 3.0:
        sections.append({"name": "INTRO", "start": 0.0, "end": blocks[0]["start"]})
    for i, b in enumerate(blocks):
        nxt = blocks[i + 1]["start"] if i + 1 < len(blocks) else duration
        end = b["end"]
        if nxt - end > gap_threshold:
            sections.append({"name": b["name"], "start": b["start"], "end": end + 0.75})
            sections.append({"name": "INTERLUDE" if i + 1 < len(blocks) else "OUTRO", "start": end + 0.75, "end": nxt})
        else:
            sections.append({"name": b["name"], "start": b["start"], "end": nxt})
    if sections and sections[-1]["end"] < duration:
        sections[-1]["end"] = duration
    return sections


def align_lyrics(
    abc_text: str,
    lyrics_text: str,
    *,
    alignment_audio: np.ndarray | None = None,
    alignment_is_stem: bool = False,
    whisper_input: str = "",
    mode: str = "auto",
    device: str = "auto",
    syllable_weighting: str = "ABC Notes",
    time_offset: float = 0.0,
    audio_seconds: float = 0.0,
) -> dict[str, Any]:
    """Time the clean lyrics with the best available source. Never raises for alignment failures.

    Returns a dict with ``lines`` (timed lines with words), ``source`` (forced|whisper|score|none),
    ``timeline`` (ABC score data), ``score_offset`` (audio minus nominal ABC seconds, or None),
    ``stats``, ``warnings``, ``lrc`` and ``report``.
    """
    timeline = extract_score_timeline(abc_text, lyrics_text)
    clean_lines = parse_clean_lyrics(lyrics_text)
    result: dict[str, Any] = {
        "lines": [], "source": "none", "timeline": timeline, "score_offset": None,
        "stats": {}, "warnings": [], "lrc": "", "report": "",
    }
    if not clean_lines:
        result["warnings"].append("No lyric lines found (only headers or empty text).")
        result["report"] = _make_report(result, clean_lines, mode, audio_seconds)
        return result

    nominal_lines = align_lines_by_score(clean_lines, timeline, syllable_weighting)
    whisper_segments = parse_whisper_input(whisper_input)

    for source in _source_order(mode):
        try:
            if source == "forced":
                if alignment_audio is None:
                    result["warnings"].append("Forced alignment skipped: no audio connected.")
                    continue
                lines, stats = forced_align_lines(
                    clean_lines, alignment_audio, device=device, is_stem=alignment_is_stem,
                    expected_seconds=float(timeline.get("score_seconds", 0.0)),
                )
                result["stats"] = stats
                if stats.get("mean_confidence", 0.0) < FA_MIN_MEAN_CONFIDENCE:
                    result["warnings"].append(
                        f"Forced alignment confidence too low ({stats.get('mean_confidence', 0.0):.2f}); "
                        "the audio probably has no sung lyrics or the wrong stem was connected."
                    )
                    continue
            elif source == "whisper":
                if not whisper_segments:
                    result["warnings"].append("Whisper alignment skipped: no segments supplied.")
                    continue
                lines = align_lines_by_whisper(clean_lines, whisper_segments, timeline, syllable_weighting)
            else:
                lines = nominal_lines
        except Exception as exc:
            logger.warning(f"{source} alignment failed: {exc}")
            result["warnings"].append(f"{source} alignment failed: {exc}")
            continue
        if lines:
            result["lines"] = lines
            result["source"] = source
            break

    lines = result["lines"]
    if result["source"] in ("forced", "whisper"):
        result["score_offset"] = estimate_score_offset(lines, nominal_lines, timeline)
    if time_offset:
        shift_lines(lines, float(time_offset))
    result["lrc"] = make_lrc(lines)
    result["report"] = _make_report(result, clean_lines, mode, audio_seconds)
    return result


def _make_report(result: dict[str, Any], clean_lines: list[dict], mode: str, audio_seconds: float) -> str:
    tl = result["timeline"]
    lines = result["lines"]
    stats = result.get("stats", {})
    source_names = {
        "forced": f"forced alignment (MMS_FA, {stats.get('device', '?')})",
        "whisper": "Whisper/LRC segments + ABC note onsets",
        "score": "nominal ABC score timing",
        "none": "none",
    }
    out = [
        "HZ3 YuE2 · Karaoke Lyrics Alignment",
        "===================================",
        f"Mode: {mode} · Source used: {source_names.get(result['source'], result['source'])}",
        f"Tempo: {int(tl['bpm'])} BPM · Meter: {tl['meter']} · Key: {tl['key']} · "
        f"Score: {tl['score_seconds']:.1f}s · Audio: {audio_seconds:.1f}s · ABC parsed: {'yes' if tl['parsed'] else 'no (header fallback)'}",
        f"Lyric lines: {len(clean_lines)} · Vocal notes in ABC: {len(tl['vocal_notes'])}",
    ]
    if result["source"] == "forced":
        out.append(
            f"Words aligned: {stats.get('words_aligned', 0)} · mean confidence: {stats.get('mean_confidence', 0.0):.2f} · "
            f"low-confidence lines: {stats.get('low_confidence_lines', 0)} · passes: {stats.get('passes', 1)}"
        )
        if stats.get("trimmed_lines"):
            out.append(f"Lines beyond the audio (generation shorter than the lyrics): {stats['trimmed_lines']}")
    if result.get("score_offset") is not None:
        out.append(f"Audio vs ABC offset (applied to chords/sections): {result['score_offset']:+.2f}s")
    if result["warnings"]:
        out.append("Warnings:")
        out.extend(f"  - {w}" for w in result["warnings"])
    out.append("")
    out.append("Lines:")
    for idx, item in enumerate(lines):
        tag = f"[{item['section']}] " if item.get("section") else ""
        conf = f" conf {item['confidence']:.2f}" if "confidence" in item else ""
        flag = "  (beyond audio)" if item.get("beyond_audio") else ""
        out.append(f"  {idx + 1:02d} {item['start']:7.2f}–{item['end']:7.2f}{conf}  {tag}{item['text']}{flag}")
    return "\n".join(out)


def align_score_and_lyrics(
    abc_text: str,
    lyrics_text: str,
    whisper_input: str = "",
    alignment_mode: str = "Hybrid (Whisper Audio + ABC Notes)",
    syllable_weighting: str = "ABC Notes",
    intro_silence: float = 0.0,
) -> tuple[list[dict], str, str]:
    """Compatibility wrapper (no audio): Whisper segments when supplied, else nominal ABC timing."""
    mode = "score" if alignment_mode == "ABC Score Timing" else "whisper"
    result = align_lyrics(abc_text, lyrics_text, whisper_input=whisper_input, mode=mode,
                          syllable_weighting=syllable_weighting, time_offset=intro_silence)
    return result["lines"], result["lrc"], result["report"]


NODE_CLASS_MAPPINGS: dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
