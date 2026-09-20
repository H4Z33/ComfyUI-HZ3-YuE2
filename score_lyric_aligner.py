"""HZ3 YuE2 · Score & Lyrics Aligner.

Maps ABC score timings (sections, bars, vocal note attacks) and acoustic Whisper timestamps
to clean, authoritative lyrics text. Generates synchronized timed_lyrics JSON and standard LRC
for HZ3_YuE2_KaraokeVisualizer and external players.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import unicodedata
from typing import Any

try:
    from .score_align import align_and_repair_abc, normalize_section_label
    from .score_analysis import inspect_score, lyric_syllables
except (ImportError, ValueError):
    from score_align import align_and_repair_abc, normalize_section_label
    from score_analysis import inspect_score, lyric_syllables

logger = logging.getLogger("HZ3.ScoreLyricAligner")


def normalize_for_comparison(text: str) -> str:
    """Normalize text for phonetic/fuzzy similarity comparison."""
    if not text:
        return ""
    # Decompose unicode characters into base letters + combining diacritics
    decomposed = unicodedata.normalize("NFD", text.lower())
    # Remove combining diacritics (accents, tildes)
    without_accents = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    # Keep only alphanumeric and spaces
    cleaned = re.sub(r"[^a-z0-9\s]", " ", without_accents)
    return " ".join(cleaned.split())


def text_similarity(s1: str, s2: str) -> float:
    """Compute normalized text similarity combining token overlap and character sequence ratio."""
    n1 = normalize_for_comparison(s1)
    n2 = normalize_for_comparison(s2)
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0

    # Character sequence ratio
    seq_ratio = difflib.SequenceMatcher(None, n1, n2).ratio()

    # Word set Jaccard similarity
    words1 = set(n1.split())
    words2 = set(n2.split())
    if words1 and words2:
        jaccard = len(words1 & words2) / float(len(words1 | words2))
    else:
        jaccard = 0.0

    return 0.6 * seq_ratio + 0.4 * jaccard


def parse_whisper_input(raw_input: str) -> list[dict]:
    """Parse Whisper input from JSON array, JSON object with 'segments', or LRC text."""
    if not raw_input or not str(raw_input).strip():
        return []
    raw = str(raw_input).strip()

    # 1. JSON parse
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
                        words = item.get("words", [])
                        segments.append({"start": s, "end": max(s + 0.1, e), "text": t, "words": words})
                if segments:
                    return segments
        except Exception:
            pass

    # 2. LRC parse
    lrc_pattern = re.compile(r"^\[(\d{1,2}):(\d{2}(?:\.\d+)?)\](.*)$")
    lrc_entries = []
    for line in raw.splitlines():
        line = line.strip()
        m = lrc_pattern.match(line)
        if m:
            mins = int(m.group(1))
            secs = float(m.group(2))
            t_sec = mins * 60.0 + secs
            content = m.group(3).strip()
            if content and not re.match(r"^[a-zA-Z]{2,4}:", content):
                lrc_entries.append((t_sec, content))

    if lrc_entries:
        lrc_entries.sort(key=lambda x: x[0])
        segments = []
        for idx, (t_start, content) in enumerate(lrc_entries):
            if idx + 1 < len(lrc_entries):
                t_end = lrc_entries[idx + 1][0]
            else:
                t_end = t_start + 3.5
            segments.append({"start": t_start, "end": max(t_start + 0.2, t_end), "text": content, "words": []})
        return segments

    # 3. Plain lines with no timestamps
    lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.strip().startswith("[")]
    return [{"start": None, "end": None, "text": l, "words": []} for l in lines]


def parse_clean_lyrics(lyrics_text: str) -> list[dict]:
    """Parse clean lyrics text into structured lines preserving section headers and tags."""
    lines = []
    curr_sec = ""
    sec_header_re = re.compile(r"^\s*\[([a-zA-Z0-9_\s\-]+)\]\s*$")

    for raw in (lyrics_text or "").splitlines():
        line_str = raw.strip()
        if not line_str:
            continue
        m = sec_header_re.match(line_str)
        if m:
            curr_sec = m.group(1).strip()
        else:
            lines.append({
                "text": line_str,
                "section": curr_sec,
                "norm_section": normalize_section_label(curr_sec) if curr_sec else "",
            })
    return lines


def cluster_notes_into_phrases(notes: list[dict], min_pause: float = 0.25) -> list[list[dict]]:
    """Group musical notes into phrases separated by musical rests >= min_pause."""
    phrases = []
    cur_phrase = []
    for n in notes:
        if not cur_phrase:
            cur_phrase.append(n)
        else:
            gap = n["start"] - cur_phrase[-1]["end"]
            if gap >= min_pause:
                phrases.append(cur_phrase)
                cur_phrase = [n]
            else:
                cur_phrase.append(n)
    if cur_phrase:
        phrases.append(cur_phrase)
    return phrases


def adjust_phrases_to_target_count(phrases: list[list[dict]], target_count: int) -> list[list[dict]]:
    """Merge or split phrases until phrase count matches target line count."""
    if not phrases or target_count <= 0:
        return phrases
    phrases = list(phrases)

    # Merge closest adjacent phrases if too many phrases
    while len(phrases) > target_count and len(phrases) > 1:
        min_gap = float("inf")
        min_idx = 0
        for p_i in range(len(phrases) - 1):
            g = phrases[p_i + 1][0]["start"] - phrases[p_i][-1]["end"]
            if g < min_gap:
                min_gap = g
                min_idx = p_i
        merged = phrases[min_idx] + phrases[min_idx + 1]
        phrases[min_idx] = merged
        phrases.pop(min_idx + 1)

    # Split longest phrases at internal note gaps if too few phrases
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
            p_a = target_phrase[:split_idx]
            p_b = target_phrase[split_idx:]
            phrases[best_p_idx] = p_a
            phrases.insert(best_p_idx + 1, p_b)
        else:
            mid_t = (target_phrase[0]["start"] + target_phrase[0]["end"]) / 2.0
            p_a = [{"start": target_phrase[0]["start"], "end": mid_t, "pitch": target_phrase[0].get("pitch", 60)}]
            p_b = [{"start": mid_t, "end": target_phrase[0]["end"], "pitch": target_phrase[0].get("pitch", 60)}]
            phrases[best_p_idx] = p_a
            phrases.insert(best_p_idx + 1, p_b)

    return phrases


def word_syllable_count(w: str) -> int:
    """Count syllables in a word using lyric_syllables analysis."""
    clean = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]", "", w)
    if not clean:
        return 1
    syls = lyric_syllables(clean)
    return max(1, len(syls))


def align_clean_lines_to_whisper(
    clean_lines: list[dict],
    whisper_segments: list[dict],
) -> list[tuple[dict, float, float]]:
    """Monotonically align clean lyric lines to Whisper segment timestamps via Dynamic Programming."""
    N = len(clean_lines)
    M = len(whisper_segments)

    if N == 0:
        return []
    if M == 0:
        return [(l, 0.0, 3.0) for l in clean_lines]

    # Precompute pairwise similarity matrix
    sim_matrix = [[text_similarity(clean_lines[i]["text"], whisper_segments[j]["text"])
                   for j in range(M)] for i in range(N)]

    # Dynamic Programming to find the optimal monotonic alignment
    # Cost = 1.0 - similarity
    # States: dp[i][j] = min cumulative cost aligning clean_lines[:i] with whisper_segments[:j]
    INF = float("inf")
    dp = [[INF] * (M + 1) for _ in range(N + 1)]
    parent = [[None] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0.0

    # Skipping whisper segments at start (e.g. intro/instrumental hallucination)
    for j in range(1, M + 1):
        dp[0][j] = dp[0][j - 1] + 0.35
        parent[0][j] = (0, j - 1, "skip_whisper")

    for i in range(1, N + 1):
        # Skipping clean lines (if Whisper missed beginning of song)
        dp[i][0] = dp[i - 1][0] + 0.8
        parent[i][0] = (i - 1, 0, "skip_clean")

        for j in range(1, M + 1):
            cost_1to1 = 1.0 - sim_matrix[i - 1][j - 1]
            # Transition 1: Match line i with segment j (1-to-1)
            best_cost = dp[i - 1][j - 1] + cost_1to1
            best_op = (i - 1, j - 1, "1to1")

            # Transition 2: Line i spans 2 Whisper segments (j-1 and j)
            if j >= 2:
                merged_whisper = whisper_segments[j - 2]["text"] + " " + whisper_segments[j - 1]["text"]
                cost_1to2 = 1.0 - text_similarity(clean_lines[i - 1]["text"], merged_whisper)
                c = dp[i - 1][j - 2] + cost_1to2
                if c < best_cost:
                    best_cost = c
                    best_op = (i - 1, j - 2, "1to2")

            # Transition 3: Segment j spans 2 Clean lines (i-1 and i)
            if i >= 2:
                merged_clean = clean_lines[i - 2]["text"] + " " + clean_lines[i - 1]["text"]
                cost_2to1 = 1.0 - text_similarity(merged_clean, whisper_segments[j - 1]["text"])
                c = dp[i - 2][j - 1] + cost_2to1
                if c < best_cost:
                    best_cost = c
                    best_op = (i - 2, j - 1, "2to1")

            # Transition 4: Skip whisper segment j (whisper hallucinated or chatter)
            c_skip_w = dp[i][j - 1] + 0.35
            if c_skip_w < best_cost:
                best_cost = c_skip_w
                best_op = (i, j - 1, "skip_whisper")

            # Transition 5: Skip clean line i (whisper completely missed line)
            c_skip_c = dp[i - 1][j] + 0.80
            if c_skip_c < best_cost:
                best_cost = c_skip_c
                best_op = (i - 1, j, "skip_clean")

            dp[i][j] = best_cost
            parent[i][j] = best_op

    # Backtrack alignment path
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

    # Build result timestamps
    results = []
    last_known_end = 0.0

    for idx, l in enumerate(clean_lines):
        w_indices = sorted(set(matched_ranges[idx]))
        if w_indices:
            start_t = whisper_segments[w_indices[0]]["start"]
            end_t = whisper_segments[w_indices[-1]]["end"]
            # Handle cases where multiple clean lines share 1 whisper segment (many-to-1)
            shared_lines = [k for k, v in matched_ranges.items() if set(v) == set(w_indices)]
            if len(shared_lines) > 1:
                pos = shared_lines.index(idx)
                tot = len(shared_lines)
                dur = max(0.2, end_t - start_t)
                part_dur = dur / tot
                line_s = start_t + pos * part_dur
                line_e = line_s + part_dur
                results.append((l, line_s, line_e))
                last_known_end = line_e
            else:
                results.append((l, start_t, end_t))
                last_known_end = end_t
        else:
            # Unmatched line: estimate based on last known end + typical line length (3.0s)
            est_s = last_known_end + 0.3
            est_e = est_s + 3.0
            results.append((l, est_s, est_e))
            last_known_end = est_e

    return results


def map_words_to_timeline(
    clean_line_text: str,
    line_start: float,
    line_end: float,
    notes_in_span: list[dict],
    syllable_mode: str = "ABC Notes",
) -> list[dict]:
    """Generate exact word-level timing for a clean lyric line."""
    words = clean_line_text.split()
    if not words:
        return []

    line_dur = max(0.1, line_end - line_start)
    syl_counts = [word_syllable_count(w) for w in words]
    tot_syl = max(1, sum(syl_counts))

    timed_words = []

    if syllable_mode == "ABC Notes" and len(notes_in_span) >= 2:
        # Map words to ABC note onsets
        num_notes = len(notes_in_span)
        cur_note_idx = 0

        for idx, (w, sc) in enumerate(zip(words, syl_counts)):
            # Determine note slice for this word based on syllable count
            note_slice_len = max(1, round(num_notes * (sc / float(tot_syl))))
            end_note_idx = min(num_notes - 1, cur_note_idx + note_slice_len - 1)

            w_start = notes_in_span[cur_note_idx]["start"]
            w_end = notes_in_span[end_note_idx]["end"]
            if w_end <= w_start:
                w_end = w_start + 0.25

            # Clamp word times inside line bounds with safe margins
            w_start = max(line_start, min(line_end - 0.05, w_start))
            w_end = max(w_start + 0.05, min(line_end, w_end))

            timed_words.append({
                "text": w,
                "start": round(w_start, 3),
                "end": round(w_end, 3),
            })
            cur_note_idx = min(num_notes - 1, end_note_idx + 1)
    else:
        # Phonetic Syllables or Equal Word Split
        cur_t = line_start
        for idx, (w, sc) in enumerate(zip(words, syl_counts)):
            weight = (sc / float(tot_syl)) if syllable_mode != "Equal Word Split" else (1.0 / len(words))
            w_dur = max(0.05, line_dur * weight)
            w_end = cur_t + w_dur if idx < len(words) - 1 else line_end
            timed_words.append({
                "text": w,
                "start": round(cur_t, 3),
                "end": round(w_end, 3),
            })
            cur_t = w_end

    return timed_words


def align_score_and_lyrics(
    abc_text: str,
    lyrics_text: str,
    whisper_input: str = "",
    alignment_mode: str = "Hybrid (Whisper Audio + ABC Notes)",
    syllable_weighting: str = "ABC Notes",
    intro_silence: float = 0.0,
) -> tuple[list[dict], str, str]:
    """Align ABC score and clean lyrics, optionally anchored by Whisper acoustic timestamps.

    Returns:
        (timed_lines: list[dict], lrc_text: str, report: str)
    """
    clean_lines = parse_clean_lyrics(lyrics_text)
    if not clean_lines:
        return [], "", "No lyrics lines found to align."

    # Parse and repair score
    repaired_abc, _ = align_and_repair_abc(abc_text, lyrics_text=lyrics_text)
    try:
        info = inspect_score(repaired_abc, lyrics_text)
    except Exception as exc:
        logger.warning(f"inspect_score strict parse fallback: {exc}")
        info = None

    if info:
        bpm = float(info.get("bpm", 120))
        meter = info.get("meter", "4/4")
        key = info.get("key", "C")
        total_duration = float(info.get("seconds", 0.0)) + intro_silence

        # Extract vocal and instrumental notes
        vocal_raw = info.get("roll", {}).get("tracks", {}).get("Vocal", [])
        ins_raw = info.get("roll", {}).get("tracks", {}).get("Ins", [])
        seconds_per_tick = 60.0 / (bpm * 256.0)

        vocal_notes = []
        for n in vocal_raw:
            s = n["start"] * seconds_per_tick + intro_silence
            dur = n["duration"] * seconds_per_tick
            vocal_notes.append({"start": s, "end": s + dur, "duration": dur, "pitch": n.get("pitch", 60)})

        ins_notes = []
        for n in ins_raw:
            s = n["start"] * seconds_per_tick + intro_silence
            dur = n["duration"] * seconds_per_tick
            ins_notes.append({"start": s, "end": s + dur, "duration": dur, "pitch": n.get("pitch", 48)})

        melody_events = vocal_notes if len(vocal_notes) > 0 else ins_notes

        # Extract score sections
        sections_raw = info.get("roll", {}).get("sections", [])
        bar_ticks = info.get("roll", {}).get("bar_ticks", 1024)
        sections = []
        for s in sections_raw:
            t_start = (s["start"] * bar_ticks / 256.0) * (60.0 / bpm) + intro_silence
            t_end = t_start + (s["bars"] * bar_ticks / 256.0) * (60.0 / bpm)
            sections.append({
                "name": s.get("name", "section").lower().strip(),
                "norm": normalize_section_label(s.get("name", "")),
                "start": t_start,
                "end": t_end,
            })
    else:
        # Fallback metadata from raw ABC header lines
        bpm_m = re.search(r"^Q:1/4=(\d+)", repaired_abc, re.MULTILINE)
        bpm = float(bpm_m.group(1)) if bpm_m else 120.0
        meter_m = re.search(r"^M:(\S+)", repaired_abc, re.MULTILINE)
        meter = meter_m.group(1) if meter_m else "4/4"
        key_m = re.search(r"^K:(\S+)", repaired_abc, re.MULTILINE)
        key = key_m.group(1) if key_m else "C"

        vocal_notes = []
        ins_notes = []
        melody_events = []

        # Extract sections by comment lines
        sections = []
        cur_t = intro_silence
        sec_re = re.compile(r"^%\s*([a-zA-Z0-9_\s\-]+)", re.MULTILINE)
        sec_names = sec_re.findall(repaired_abc)
        approx_sec_dur = 16.0 * (60.0 / bpm)
        for s_name in sec_names:
            sections.append({
                "name": s_name.lower().strip(),
                "norm": normalize_section_label(s_name),
                "start": cur_t,
                "end": cur_t + approx_sec_dur,
            })
            cur_t += approx_sec_dur
        total_duration = cur_t if sections else (intro_silence + 60.0)
    if not sections:
        sections = [{"name": "song", "norm": "song", "start": intro_silence, "end": total_duration}]

    # Check Whisper input
    whisper_segments = parse_whisper_input(whisper_input)
    use_whisper = bool(whisper_segments) and (alignment_mode != "ABC Score Timing")

    timed_lines = []

    if use_whisper:
        # Acoustic Anchoring: Align clean lines to Whisper segment timestamps
        aligned_pairs = align_clean_lines_to_whisper(clean_lines, whisper_segments)

        for l_item, l_start, l_end in aligned_pairs:
            line_text = l_item["text"]
            # Find ABC vocal notes inside or near this time span
            sec_notes = [n for n in melody_events if l_start - 0.15 <= n["start"] <= l_end + 0.15]
            words = map_words_to_timeline(line_text, l_start, l_end, sec_notes, syllable_mode=syllable_weighting)

            timed_lines.append({
                "text": line_text,
                "start": round(l_start, 3),
                "end": round(l_end, 3),
                "section": l_item.get("section", ""),
                "words": words,
            })
    else:
        # Procedural Score Timing: Align clean lines to ABC score sections and phrases
        # Group clean lines by section
        sec_grouped: dict[str, list[dict]] = {}
        order_keys = []
        for l in clean_lines:
            s_name = l["norm_section"] or "main"
            if s_name not in sec_grouped:
                sec_grouped[s_name] = []
                order_keys.append(s_name)
            sec_grouped[s_name].append(l)

        sec_cursor = 0
        for s_key in order_keys:
            lines = sec_grouped[s_key]
            # Match to score section
            target_sec = None
            for idx in range(sec_cursor, len(sections)):
                if s_key in sections[idx]["norm"] or sections[idx]["norm"] in s_key:
                    target_sec = sections[idx]
                    sec_cursor = idx + 1
                    break
            if not target_sec and sec_cursor < len(sections):
                target_sec = sections[sec_cursor]
                sec_cursor += 1

            s_start = target_sec["start"] if target_sec else intro_silence
            s_end = target_sec["end"] if target_sec else total_duration

            sec_notes = [n for n in melody_events if s_start - 0.05 <= n["start"] <= s_end + 0.05]
            phrases = cluster_notes_into_phrases(sec_notes, min_pause=0.25)
            phrases = adjust_phrases_to_target_count(phrases, len(lines))

            for idx, l_item in enumerate(lines):
                line_text = l_item["text"]
                if phrases:
                    p_idx = min(idx, len(phrases) - 1)
                    p = phrases[p_idx]
                    l_start = p[0]["start"]
                    l_end = p[-1]["end"]
                    p_notes = p
                else:
                    line_dur = max(1.0, (s_end - s_start) / max(1, len(lines)))
                    l_start = s_start + idx * line_dur
                    l_end = s_start + (idx + 1) * line_dur
                    p_notes = [n for n in sec_notes if l_start <= n["start"] <= l_end]

                words = map_words_to_timeline(line_text, l_start, l_end, p_notes, syllable_mode=syllable_weighting)
                timed_lines.append({
                    "text": line_text,
                    "start": round(l_start, 3),
                    "end": round(l_end, 3),
                    "section": l_item.get("section", ""),
                    "words": words,
                })

    # Generate standard LRC text
    lrc_lines = []
    for item in timed_lines:
        s = item["start"]
        mins = int(s // 60)
        secs = s % 60
        lrc_lines.append(f"[{mins:02d}:{secs:05.2f}] {item['text']}")
    lrc_text = "\n".join(lrc_lines)

    # Generate diagnostic report
    report_lines = [
        "HZ3 YuE2 · Score & Lyrics Aligner Report",
        "========================================",
        f"Mode: {alignment_mode}",
        f"Syllable Weighting: {syllable_weighting}",
        f"Tempo: {int(bpm)} BPM · Meter: {meter} · Key: {key} · Duration: {total_duration:.1f}s",
        f"Clean Lyrics Lines: {len(clean_lines)}",
        f"Vocal Notes Available: {len(vocal_notes)}",
        f"Whisper Segments Supplied: {len(whisper_segments)} (Acoustic Anchoring: {'Active' if use_whisper else 'Inactive'})",
        "",
        "First 5 Aligned Lines Preview:",
    ]
    for idx, item in enumerate(timed_lines[:5]):
        words_preview = " ".join(f"{w['text']}[{w['start']:.2f}s]" for w in item["words"][:4])
        report_lines.append(f"  Line {idx + 1} ({item['start']:.2f}s - {item['end']:.2f}s): {item['text']}")
        report_lines.append(f"    Words: {words_preview}...")

    report = "\n".join(report_lines)
    return timed_lines, lrc_text, report


class HZ3_YuE2_ScoreLyricAligner:
    CATEGORY = "HZ3 YuE2/Score"
    FUNCTION = "align_lyrics"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("timed_lyrics", "lrc_text", "report")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Reconcile ABC score note attacks, measure timings, and optional Whisper acoustic timestamps "
        "with clean ground-truth lyrics. Produces synchronized timed_lyrics JSON and standard LRC for "
        "the Karaoke Visualizer and external players."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": (
                    "STRING",
                    {
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Clean ground-truth lyrics text. This text is strictly preserved and displayed.",
                    },
                ),
                "abc": (
                    "STRING",
                    {
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "ABC score containing sections, measures, BPM, and vocal/instrumental tracks.",
                    },
                ),
            },
            "optional": {
                "whisper_segments": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional Whisper JSON segments array or LRC text for acoustic time-window anchoring.",
                    },
                ),
                "alignment_mode": (
                    [
                        "Hybrid (Whisper Audio + ABC Notes)",
                        "ABC Score Timing",
                        "Whisper Timestamps Only",
                    ],
                    {
                        "default": "Hybrid (Whisper Audio + ABC Notes)",
                        "tooltip": "Alignment strategy: Hybrid uses Whisper for line bounds and ABC for note onsets.",
                    },
                ),
                "syllable_weighting": (
                    ["ABC Notes", "Phonetic Syllables", "Equal Word Split"],
                    {
                        "default": "ABC Notes",
                        "tooltip": "How to distribute words across the phrase: by ABC vocal note onsets or phonetic syllables.",
                    },
                ),
                "intro_silence": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.1,
                        "tooltip": "Optional audio intro delay offset in seconds to shift score timestamps.",
                    },
                ),
            },
        }

    def align_lyrics(
        self,
        lyrics: str,
        abc: str,
        whisper_segments: str = "",
        alignment_mode: str = "Hybrid (Whisper Audio + ABC Notes)",
        syllable_weighting: str = "ABC Notes",
        intro_silence: float = 0.0,
    ):
        timed_lines, lrc_text, report = align_score_and_lyrics(
            abc_text=abc,
            lyrics_text=lyrics,
            whisper_input=whisper_segments,
            alignment_mode=alignment_mode,
            syllable_weighting=syllable_weighting,
            intro_silence=intro_silence,
        )

        timed_json = json.dumps(timed_lines, ensure_ascii=False, indent=2)
        visible = (
            f"HZ3 YuE2 · Score & Lyrics Aligner\n"
            f"Aligned {len(timed_lines)} lines.\n\n"
            f"LRC PREVIEW:\n{lrc_text[:400]}..."
        )

        return {
            "ui": {"text": [visible]},
            "result": (timed_json, lrc_text, report),
        }


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_ScoreLyricAligner": HZ3_YuE2_ScoreLyricAligner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_ScoreLyricAligner": "HZ3 YuE2 · Score & Lyrics Aligner (ABC / Whisper)",
}
