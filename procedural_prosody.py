"""Deterministic lyric/ABC alignment and controlled conditioning experiments."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .abc_score import parse
from .score_analysis import inspect_score, lyric_syllables


WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|\d+")
SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
VOWELS = "aeiouáéíóúü"
ACCENTED = str.maketrans({"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú",
                          "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú"})
UNSTRESSED_LINKERS = {
    "a", "al", "de", "del", "el", "la", "las", "los", "mi", "mis", "tu", "tus", "su", "sus",
    "un", "una", "unos", "unas", "me", "te", "se", "lo", "le", "les", "que", "y", "o",
}
NUMBER_SYLLABLES = {
    "0": 2, "1": 1, "2": 1, "3": 1, "4": 2, "5": 2, "6": 1, "7": 2, "8": 2, "9": 2,
    "10": 2, "11": 3, "12": 3, "13": 3, "14": 3, "15": 2, "16": 4, "17": 4,
    "18": 4, "19": 4, "20": 3,
}

try:
    import pyphen

    _HYPHENATOR = pyphen.Pyphen(lang="es")
except ImportError:  # Optional; the bundled fallback preserves operation offline.
    _HYPHENATOR = None


def _section_name(value):
    value = re.sub(r"\s+\d+\s*$", "", (value or "").strip().lower())
    aliases = {"coro": "chorus", "estribillo": "chorus", "verso": "verse", "estrofa": "verse",
               "intro": "intro", "interludio": "interlude", "puente": "bridge", "salida": "outro"}
    return aliases.get(value, value)


def _is_sustain_filler(word):
    clean = re.sub(r"[^a-záéíóúüñ]", "", word.lower())
    if not clean:
        return False
    return bool(
        re.fullmatch(r"([aeiouáéíóúü])\1+", clean)
        or (re.fullmatch(r"[aeiouáéíóúü]{2,4}", clean) and len(set(clean)) <= 2)
        or re.fullmatch(r"[mn]{2,}", clean)
        or re.fullmatch(r"([aeiouáéíóúü])\1+[mn]+", clean)
    )


def _word_parts(word):
    if word.isdigit():
        return [word] * NUMBER_SYLLABLES.get(word, 1)
    if _HYPHENATOR is not None:
        inserted = _HYPHENATOR.inserted(word)
        parts = [part for part in inserted.split("-") if part]
        if parts:
            return parts
    return lyric_syllables(word) or [word]


def _line_metrics(text):
    matches = list(WORD_RE.finditer(text))
    words = [match.group(0) for match in matches]
    fillers = [word for word in words if _is_sustain_filler(word)]
    syllables = sum(0 if _is_sustain_filler(word) else len(_word_parts(word)) for word in words)
    vowel_joins = []
    prosodic_groups = []
    for index, (left_match, right_match) in enumerate(zip(matches, matches[1:])):
        left, right = left_match.group(0), right_match.group(0)
        separator = text[left_match.end():right_match.start()]
        if not separator or not separator.isspace():
            continue
        if left[-1].lower() in VOWELS and right[0].lower() in VOWELS + "h":
            vowel_joins.append(f"{left} {right}")
        if left.lower() in UNSTRESSED_LINKERS:
            prosodic_groups.append({
                "boundary": f"{left} {right}", "left_role": "unstressed_linker", "phrase_initial": index == 0,
            })
    return {
        "written_syllables": syllables,
        "connected_syllable_estimate": max(1, syllables - len(vowel_joins)) if words else 0,
        "sustain_fillers": fillers,
        "possible_connected_vowel_boundaries": vowel_joins,
        "possible_prosodic_group_boundaries": prosodic_groups,
    }


def _lyric_lines(lyrics):
    section = ""
    result = []
    for physical_index, raw in enumerate((lyrics or "").splitlines()):
        match = SECTION_RE.fullmatch(raw)
        if match:
            section = _section_name(match.group(1))
        elif raw.strip():
            result.append({
                "line": len(result) + 1, "physical_index": physical_index,
                "section": section, "text": raw.strip(), **_line_metrics(raw.strip()),
            })
    return result


def _vocal_bars(score_abc, min_note_attacks):
    info = inspect_score(score_abc)
    score = parse(info["abc"])
    notes = score.voices["Vocal"].notes
    section_by_bar = {}
    for section in info["roll"]["sections"]:
        for index in range(section["start"], section["start"] + section["bars"]):
            section_by_bar[index] = _section_name(section["name"])
    bars = []
    for index, (start, duration, meter) in enumerate(score.voices["Vocal"].bars):
        end = start + duration
        onsets = [note for note in notes if start <= note[0] < end]
        pitches = [note[1] for note in onsets]
        occupied = sum(min(end, note_start + note_duration) - max(start, note_start)
                       for note_start, _pitch, note_duration in notes
                       if note_start < end and note_start + note_duration > start)
        bar = {
            "bar": index + 1, "section": section_by_bar.get(index, "section"),
            "meter": f"{meter[0]}/{meter[1]}", "attacks": len(onsets),
            "distinct_pitches": len(set(pitches)),
            "adjacent_repeated_pitches": sum(a == b for a, b in zip(pitches, pitches[1:])),
            "pitch_sequence_midi": pitches,
            "longest_note_beats": round(float(max((note[2] for note in onsets), default=0)), 3),
            "rest_beats": round(float(duration - occupied), 3),
            "tie_in": any(note_start < start < note_start + note_duration
                          for note_start, _pitch, note_duration in notes),
            "tie_out": any(note_start < end < note_start + note_duration
                           for note_start, _pitch, note_duration in notes),
        }
        bars.append(bar)
    return [bar for bar in bars if bar["attacks"] >= min_note_attacks], bars, info


def _match_cost(line, bar):
    delta = abs(line["connected_syllable_estimate"] - bar["attacks"])
    section_penalty = 0
    if line["section"] and line["section"] not in {bar["section"], "section"}:
        section_penalty = 8
    return delta + section_penalty


def _align(lines, candidate_bars):
    if len(lines) == len(candidate_bars):
        return list(zip(lines, candidate_bars)), "exact_sequential"
    if not lines or not candidate_bars:
        return [], "unavailable"
    if len(candidate_bars) < len(lines):
        return list(zip(lines, candidate_bars)), "partial_sequential"

    n, m = len(lines), len(candidate_bars)
    infinity = float("inf")
    dp = [[infinity] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for j in range(m):
        if dp[0][j] < infinity:
            dp[0][j + 1] = dp[0][j] + 1.5
            back[0][j + 1] = (0, j, "skip")
    for i in range(n):
        for j in range(m):
            if dp[i][j] == infinity:
                continue
            match = dp[i][j] + _match_cost(lines[i], candidate_bars[j])
            if match < dp[i + 1][j + 1]:
                dp[i + 1][j + 1] = match
                back[i + 1][j + 1] = (i, j, "match")
            skip = dp[i][j] + 1.5
            if skip < dp[i][j + 1]:
                dp[i][j + 1] = skip
                back[i][j + 1] = (i, j, "skip")
    end_j = min(range(n, m + 1), key=lambda j: dp[n][j])
    pairs = []
    i, j = n, end_j
    while i or j:
        previous = back[i][j]
        if previous is None:
            break
        pi, pj, operation = previous
        if operation == "match":
            pairs.append((lines[pi], candidate_bars[pj]))
        i, j = pi, pj
    pairs.reverse()
    return pairs, "dynamic_programming"


def _replace_boundary(text, before):
    words = before.split(" ", 1)
    if len(words) != 2:
        return text
    pattern = re.compile(rf"\b{re.escape(words[0])}\s+{re.escape(words[1])}\b", re.IGNORECASE)
    return pattern.sub(lambda match: re.sub(r"\s+", "", match.group(0)), text, count=1)


def _replace_boundary_with(text, before, after):
    words = before.split(" ", 1)
    if len(words) != 2:
        return text
    pattern = re.compile(rf"\b{re.escape(words[0])}\s+{re.escape(words[1])}\b", re.IGNORECASE)
    return pattern.sub(after, text, count=1)


def _explicit_stress(word):
    """Add a performance accent to the ordinary stressed vowel, if one is absent."""
    if any(char in "áéíóúÁÉÍÓÚ" for char in word):
        return word
    parts = _word_parts(word)
    if not parts:
        return word
    stress_part = max(0, len(parts) - (2 if word[-1:].lower() in VOWELS + "ns" and len(parts) > 1 else 1))
    prefix = "".join(parts[:stress_part])
    start = len(prefix)
    end = start + len(parts[stress_part])
    chars = list(word)
    vowel_positions = [i for i in range(start, min(end, len(chars))) if chars[i].lower() in "aeiou"]
    if not vowel_positions:
        return word
    # Prefer a strong vowel; otherwise mark the last weak vowel in the nucleus.
    position = next((i for i in vowel_positions if chars[i].lower() in "aeo"), vowel_positions[-1])
    chars[position] = chars[position].translate(ACCENTED)
    return "".join(chars)


def _conditioning_forms(left, right, bar):
    """Build escalating, independently labelled text-conditioning hypotheses."""
    forms = []
    joined = left + right
    forms.append({
        "variant_level": "boundary_only", "after": joined,
        "orthographic_change": "remove_space",
        "phoneme_intent": "connected_delivery_without_changing_intended_phonemes",
        "empirical_prior": "mixed: boundary removal changes conditioning but has not always rendered reliably",
        "risk": "low", "rank_bonus": 0,
    })

    respelled = right
    respelling = None
    empirical = "untested_transfer"
    verified_mi_vida = left.lower() == "mi" and right.lower() == "vida"
    if right[:1].lower() == "v":
        replacement = "B" if right[:1].isupper() else "b"
        respelled = replacement + right[1:]
        respelling = "v_to_b_equivalent_phoneme_conditioning"
        empirical = ("repeatedly stable in user renders of the tested mi+vida onset"
                     if verified_mi_vida else
                     "transfer from the successful mi+vida experiment; this boundary is not yet validated")
    elif right.lower() == "que":
        replacement = "K" if right[:1].isupper() else "k"
        respelled = replacement + "e"
        respelling = "que_to_ke_transparent_grapheme_conditioning"
        empirical = "user-proposed hypothesis; not yet validated by rendered audio"
    if respelling:
        forms.append({
            "variant_level": "equivalent_consonant", "after": left + respelled,
            "orthographic_change": f"remove_space+{respelling}",
            "phoneme_intent": "preserve Mexican-Spanish target phonemes while changing tokenizer conditioning",
            "empirical_prior": empirical, "risk": "medium",
            "rank_bonus": 18 if verified_mi_vida else -4 if right[:1].lower() == "v" else 6,
        })
        stressed = _explicit_stress(respelled)
        # A stress mark on a one-syllable v-word (for example voz -> bóz) adds no useful cue.
        # The q->k family remains allowed to test forms such as lo que -> loké.
        allow_stress = len(_word_parts(right)) > 1 or right.lower() == "que"
        if stressed != respelled and allow_stress:
            forms.append({
                "variant_level": "equivalent_consonant_plus_stress", "after": left + stressed,
                "orthographic_change": f"remove_space+{respelling}+explicit_stress",
                "phoneme_intent": "change tokenizer conditioning and explicitly cue the intended lexical stress",
                "empirical_prior": empirical, "risk": "medium",
                "rank_bonus": 30 if verified_mi_vida else -10 if right[:1].lower() == "v" else 10,
            })

    if left[-1:].lower() == "s" and right[:1].lower() in VOWELS + "h":
        support = bar["longest_note_beats"] >= 1.0 or bar["adjacent_repeated_pitches"] > 0
        if support:
            forms.append({
                "variant_level": "sustained_consonant", "after": left[:-1] + "zzz" + right,
                "orthographic_change": "remove_space+s_to_extended_z",
                "phoneme_intent": "test a sustained boundary consonant over score-supported duration",
                "empirical_prior": "user-proposed hypothesis; not yet validated by rendered audio",
                "risk": "high", "rank_bonus": -8,
            })
    return forms


def _load_yue2_tokenizer():
    """Find the tokenizer embedded in an installed native YuE2 checkpoint."""
    try:
        import folder_paths
        from safetensors import safe_open
        from tokenizers import Tokenizer
        names = [name for name in folder_paths.get_filename_list("checkpoints")
                 if "yue2" in name.lower() and name.lower().endswith(".safetensors")]
        names.sort(key=lambda name: ("bf16" not in name.lower(), name.lower()))
        for name in names:
            path = folder_paths.get_full_path("checkpoints", name)
            if not path or not os.path.isfile(path):
                continue
            with safe_open(path, framework="pt", device="cpu") as handle:
                if "text_encoders.yue2_tokenizer_json" not in handle.keys():
                    continue
                data = handle.get_tensor("text_encoders.yue2_tokenizer_json").numpy().tobytes()
            return Tokenizer.from_str(data.decode("utf-8")), path
    except Exception as error:
        return None, f"unavailable: {type(error).__name__}: {error}"
    return None, "unavailable: no native YuE2 checkpoint with embedded tokenizer"


def _token_diagnostics(tokenizer, text):
    if tokenizer is None:
        return None
    encoded = tokenizer.encode(text)
    return {"count": len(encoded.ids), "ids": encoded.ids, "pieces": encoded.tokens, "offsets": encoded.offsets}


def _candidates(line, bar, tokenizer=None):
    delta_written = line["written_syllables"] - bar["attacks"]
    delta_connected = line["connected_syllable_estimate"] - bar["attacks"]
    candidates = []
    for group in line["possible_prosodic_group_boundaries"]:
        left, right = group["boundary"].split(" ", 1)
        score = 120 if line["line"] == 1 and group["phrase_initial"] else 82 if group["phrase_initial"] else 45
        before_tokens = _token_diagnostics(tokenizer, group["boundary"])
        for form in _conditioning_forms(left, right, bar):
            candidates.append({
                "operation": "conditioning_experiment", "before": group["boundary"], "after": form["after"],
                "result_line": _replace_boundary_with(line["text"], group["boundary"], form["after"]),
                "expected_syllable_delta": 0, "conditioning_hypothesis": "alter_boundary_and_subword_conditioning",
                "reason": "Unstressed linker and following content word can form one Mexican-Spanish prosodic group.",
                "confidence": "high" if group["phrase_initial"] else "medium",
                "tokenizer_before": before_tokens, "tokenizer_after": _token_diagnostics(tokenizer, form["after"]),
                "rank_score": score + form.pop("rank_bonus"), **form,
            })
    for boundary in line["possible_connected_vowel_boundaries"]:
        after = boundary.replace(" ", "", 1)
        candidates.append({
            "operation": "join_boundary", "before": boundary, "after": after,
            "result_line": _replace_boundary(line["text"], boundary),
            "expected_syllable_delta": -1, "conditioning_hypothesis": "make_sinalefa_explicit",
            "reason": "Vowel-to-vowel boundary can be sung as one connected articulation.",
            "confidence": "high" if delta_connected > 0 else "medium", "rank_score": 95 + max(0, delta_connected),
        })
    if delta_written < 0:
        splittable = [(match.group(0), _word_parts(match.group(0))) for match in WORD_RE.finditer(line["text"])
                      if not match.group(0).isdigit() and len(_word_parts(match.group(0))) > 1]
        if splittable:
            word, parts = max(splittable, key=lambda item: len(item[1]))
            after = " ".join(parts)
            candidates.append({
                "operation": "split_word", "before": word, "after": after,
                "result_line": re.sub(rf"\b{re.escape(word)}\b", after, line["text"], count=1),
                "expected_syllable_delta": 0, "conditioning_hypothesis": "encourage_rearticulation",
                "reason": "Available attacks may support clearer syllable re-articulation; this does not add sounds.",
                "confidence": "low", "rank_score": 35 + abs(delta_written),
            })
    for candidate in candidates:
        candidate.update({"line": line["line"], "bar": bar["bar"], "section": bar["section"]})
    return candidates


def analyze_prosody(score_abc, lyrics, min_note_attacks=3, mode="analyze_only", max_edits=1):
    lines = _lyric_lines(lyrics)
    tokenizer, tokenizer_source = _load_yue2_tokenizer()
    candidate_bars, all_bars, info = _vocal_bars(score_abc, min_note_attacks)
    pairs, alignment_method = _align(lines, candidate_bars)
    alignment = []
    candidates = []
    for line, bar in pairs:
        written_delta = line["written_syllables"] - bar["attacks"]
        connected_delta = line["connected_syllable_estimate"] - bar["attacks"]
        risk = "compression" if connected_delta > 0 else "melisma_or_rearticulation" if written_delta < -2 else "balanced"
        row = {key: value for key, value in line.items() if key != "physical_index"}
        row.update(bar)
        row.update({
            "written_syllable_attack_delta": written_delta,
            "connected_syllable_attack_delta": connected_delta,
            "risk": risk,
        })
        alignment.append(row)
        candidates.extend(_candidates(line, bar, tokenizer))

    selected = []
    output_lines = (lyrics or "").splitlines()
    if mode in {"apply_best_spacing", "apply_best_conditioning"}:
        used_lines = set()
        eligible = candidates
        if mode == "apply_best_spacing":
            eligible = [item for item in candidates if item.get("variant_level") in {None, "boundary_only"}]
        for candidate in sorted(eligible, key=lambda item: (-item["rank_score"], item["line"])):
            if len(selected) >= max_edits or candidate["line"] in used_lines:
                continue
            source = lines[candidate["line"] - 1]
            if candidate["result_line"] == source["text"]:
                continue
            output_lines[source["physical_index"]] = candidate["result_line"]
            selected.append({key: value for key, value in candidate.items() if key not in {"rank_score", "result_line"}})
            used_lines.add(candidate["line"])

    public_candidates = [{key: value for key, value in candidate.items() if key not in {"rank_score", "result_line"}}
                         for candidate in sorted(candidates, key=lambda item: (-item["rank_score"], item["line"]))]
    evidence = {
        "schema": "hz3-yue2-procedural-prosody/2",
        "mode": mode,
        "alignment_method": alignment_method,
        "score_summary": info["summary"],
        "lyric_line_count": len(lines),
        "candidate_vocal_bar_count": len(candidate_bars),
        "min_note_attacks": min_note_attacks,
        "tokenizer_source": tokenizer_source,
        "alignment": alignment,
        "ranked_conditioning_candidates": public_candidates,
        "selected_edits": selected,
        "limitations": [
            "ABC has no explicit lyric-under-note w: alignment; line-to-bar mapping remains a hypothesis.",
            "Multiple attacks may be a melisma and one attack may carry connected syllables.",
            "Spacing edits change model conditioning, not the linguistic syllable inventory.",
            "Tokenizer pieces describe segmentation, not acoustic quality; rendered audio remains the deciding test.",
            "b/v respelling preserves the usual Mexican-Spanish phoneme target but changes orthography and subword conditioning.",
        ],
    }
    report_lines = [
        f"Procedural Prosody · {alignment_method}",
        f"Lyrics: {len(lines)} lines · Candidate vocal bars: {len(candidate_bars)} · Applied edits: {len(selected)}",
        "",
        "Line  Bar  Section      Syl  Conn  Att  dW  dC  Risk",
        "----  ---  -----------  ---  ----  ---  --  --  ------------------------",
    ]
    for row in alignment:
        report_lines.append(
            f"{row['line']:>4}  {row['bar']:>3}  {row['section'][:11]:<11}  "
            f"{row['written_syllables']:>3}  {row['connected_syllable_estimate']:>4}  {row['attacks']:>3}  "
            f"{row['written_syllable_attack_delta']:>+2}  {row['connected_syllable_attack_delta']:>+2}  {row['risk']}"
        )
    if selected:
        report_lines.extend(["", "Applied conditioning experiments:"])
        report_lines.extend(f"- Line {edit['line']} / bar {edit['bar']}: {edit['before']} -> {edit['after']}"
                            for edit in selected)
    report_lines.extend(["", f"Ranked candidates available to agent: {len(public_candidates)}",
                         f"Tokenizer: {tokenizer_source}"])
    return "\n".join(output_lines), "\n".join(report_lines), json.dumps(evidence, ensure_ascii=False, indent=2)


class HZ3_YuE2_ProceduralProsody:
    CATEGORY = "HZ3 YuE2"
    FUNCTION = "analyze"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("lyrics", "report", "agent_evidence")
    OUTPUT_NODE = True
    DESCRIPTION = "Align lyrics to Vocal bars and compare boundary, consonant-respelling, stress, and sustain hypotheses with the actual YuE2 tokenizer."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["analyze_only", "apply_best_spacing", "apply_best_conditioning"], {"default": "analyze_only"}),
                "min_note_attacks": ("INT", {"default": 3, "min": 1, "max": 16, "step": 1}),
                "max_edits": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
            },
            "optional": {"score_abc": ("STRING", {"forceInput": True})},
        }

    def analyze(self, lyrics, mode, min_note_attacks, max_edits, score_abc=""):
        if not (score_abc or "").strip():
            raise ValueError("Procedural Prosody requires an ABC score.")
        corrected, report, evidence = analyze_prosody(score_abc, lyrics, min_note_attacks, mode, max_edits)
        return {"ui": {"text": [report]}, "result": (corrected, report, evidence)}


NODE_CLASS_MAPPINGS = {"HZ3_YuE2_ProceduralProsody": HZ3_YuE2_ProceduralProsody}
NODE_DISPLAY_NAME_MAPPINGS = {"HZ3_YuE2_ProceduralProsody": "HZ3 YuE2 · Procedural Prosody"}
