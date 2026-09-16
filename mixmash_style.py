"""Multi-source style profiling and LLM-assisted YuE2 mashup prompts."""

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = """You write three alternative [Tags] style prompts for YuE2 music generation.

Evidence and priority:
1. Follow mix_instructions and section_cues exactly.
2. Treat BPM, meter, key, note ranges and chord density measured from a connected score as stronger evidence
   than uncertain classifier labels.
3. Use classifier probabilities comparatively; discard weak, contradictory or redundant labels. Never dump
   the classifier list into the answer and never average every detected genre together.

Compose one intentional mashup and express it at three detail levels. Prefer common musical vocabulary and do
not dump every classifier label. The lyrics are supplied separately and must never be copied or rewritten.
Section cues are soft arrangement guidance rather than exact timestamp control.

Return valid JSON only with exactly these keys:
- style_detailed: 70-130 words. Include vocal delivery, BPM/meter/key when reliable, genre fusion, specific drum
  groove, bass behavior, harmony, lead instruments, matching [Section] changes, dynamics, timbre and production.
- style_balanced: 35-65 words. Preserve the important source roles and only the strongest section contrasts.
- style_compact: 12-30 words. Genre, principal instruments, mood, vocal gender/tone and the most important groove.
Do not include Markdown, commentary, confidence scores or source numbers in any style."""


def lyrics_warnings(lyrics):
    import re
    text = (lyrics or "").replace("\r\n", "\n").strip()
    warnings = []
    if not text:
        return warnings
    blocks = re.split(r"\n\s*\n", text)
    if any(not re.match(r"^\[[^\]\n]+\](?:\n|$)", block) for block in blocks):
        warnings.append("Every lyrics block should begin with a [Section] label and one blank line should separate blocks.")
    first = blocks[0].splitlines()
    if first and first[0].strip().lower() == "[intro]" and any(line.strip() for line in first[1:]):
        warnings.append("A lyric-filled opening [Intro] is unstable; use an empty [Intro] or begin with [Verse]/[Chorus].")
    if re.search(r"\((?:repeat|drums?|pause|instrumental|key change|whisper)[^)]*\)", text, re.I):
        warnings.append("Lyrics contain a production/repeat direction in parentheses; YuE2 may try to sing it.")
    if re.search(r"\[(?:chorus|verse)\s*x\d+\]", text, re.I):
        warnings.append("Write repeated sections out in full instead of using [Section xN].")
    return warnings


def ollama_mixmash(context_1, mix_instructions, lyrics="", context_2="", context_3="", section_cues="",
                   model="deepseek-v4.1-flash:cloud", endpoint="http://127.0.0.1:11434", temperature=0.35,
                   timeout=180):
    contexts = [text.strip() for text in (context_1, context_2, context_3) if (text or "").strip()]
    if not contexts:
        raise ValueError("Connect at least context_1 to an Audio to Style analysis output.")
    user_prompt = json.dumps({
        "sources": [{"source": index, "analysis": text} for index, text in enumerate(contexts, 1)],
        "mix_instructions": mix_instructions,
        "section_cues": section_cues,
        "lyrics": lyrics,
    }, ensure_ascii=False)
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(endpoint.rstrip("/") + "/api/chat",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama is unavailable at {endpoint}: {exc.reason}") from exc
    content = raw.get("message", {}).get("content", "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON: {content[:500]}") from exc
    legacy = result.get("style", "")
    detailed = result.get("style_detailed", legacy).strip().replace("\n", " ")
    balanced = result.get("style_balanced", detailed).strip().replace("\n", " ")
    compact = result.get("style_compact", balanced).strip().replace("\n", " ")
    if not detailed:
        raise RuntimeError("Ollama returned an empty YuE2 style.")
    report = json.dumps(result, ensure_ascii=False, indent=2)
    return detailed, balanced, compact, report


class HZ3_YuE2_MixMashStyle:
    CATEGORY = "HZ3 YuE2"
    FUNCTION = "compose"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("style", "lyrics", "style_compact", "style_balanced")
    OUTPUT_NODE = True
    DESCRIPTION = "Ask an Ollama model to turn multiple measured audio/score profiles into one coherent YuE2 style prompt."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_1": ("STRING", {"forceInput": True, "tooltip": "Connect the analysis output of the first Audio to Style node."}),
                "lyrics": ("STRING", {"multiline": True, "default": "", "tooltip": "Passed unchanged to YuE2 Generate Music; section headers also guide the mashup style."}),
                "mix_instructions": ("STRING", {"multiline": True, "default": "Use source 1 as the foundation and source 2 for instrumentation and groove."}),
                "section_cues": ("STRING", {"multiline": True, "default": "", "tooltip": "Example: verses from source 1; choruses from source 2. Semantic guidance, not exact timing."}),
                "model": ("STRING", {"default": "deepseek-v4.1-flash:cloud"}),
                "endpoint": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "temperature": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05}),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 900}),
            },
            "optional": {
                "context_2": ("STRING", {"forceInput": True}),
                "context_3": ("STRING", {"forceInput": True}),
            },
        }

    def compose(self, context_1, lyrics, mix_instructions, section_cues,
                model, endpoint, temperature, timeout, context_2="", context_3=""):
        style, balanced, compact, report = ollama_mixmash(context_1, mix_instructions, lyrics, context_2, context_3,
                                                          section_cues, model, endpoint, temperature, timeout)
        warnings = lyrics_warnings(lyrics)
        visible = ("DETAILED STYLE (output: style):\n" + style +
                   "\n\nBALANCED STYLE:\n" + balanced +
                   "\n\nCOMPACT STYLE:\n" + compact)
        if warnings:
            visible += "\n\nLYRICS CHECK:\n- " + "\n- ".join(warnings)
        return {"ui": {"text": [visible]},
                "result": (style, lyrics, compact, balanced)}


LYRICS_EDIT_MODES = {"punctuation_only", "light_rewrite", "fit_to_score"}


LYRICS_PROMPT_PATH = Path(__file__).parent / "prompts" / "lyrics_prosody_system.txt"


def _lyrics_system_prompt():
    """Reload the editable prompt on every execution; no ComfyUI restart needed."""
    try:
        prompt = LYRICS_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Could not read Lyrics Prosody system prompt: {LYRICS_PROMPT_PATH}: {exc}") from exc
    if not prompt:
        raise RuntimeError(f"Lyrics Prosody system prompt is empty: {LYRICS_PROMPT_PATH}")
    return prompt


def _score_section_capacity(score_abc):
    if not (score_abc or "").strip():
        return []
    from .score_analysis import inspect_score
    info = inspect_score(score_abc)
    roll = info["roll"]
    if not roll["tracks"]["Vocal"]:
        return []
    seconds_per_tick = 60.0 / info["bpm"] / 256.0
    capacities = []
    for index, section in enumerate(roll["sections"]):
        start = section["start"] * roll["bar_ticks"]
        end = start + section["bars"] * roll["bar_ticks"]
        note_count = sum(start <= note["start"] < end for note in roll["tracks"]["Vocal"])
        capacities.append({"instance": index + 1, "label": section["name"], "bars": section["bars"],
                           "seconds": round((end - start) * seconds_per_tick, 1), "vocal_notes": note_count})
    return capacities


def _prosody_evidence(score_abc, lyrics):
    if not (score_abc or "").strip():
        return {"alignment": [], "note": "No score supplied."}
    from .abc_score import parse
    from .score_analysis import inspect_score, lyric_syllables

    info = inspect_score(score_abc, lyrics)
    score = parse(info["abc"])
    notes = score.voices["Vocal"].notes
    section_by_bar = {}
    for section in info["roll"]["sections"]:
        for bar_index in range(section["start"], section["start"] + section["bars"]):
            section_by_bar[bar_index] = section["name"]

    bars = []
    for index, (start, duration, meter) in enumerate(score.voices["Vocal"].bars):
        end = start + duration
        onsets = [note for note in notes if start <= note[0] < end]
        crossing_in = any(note_start < start < note_start + note_duration for note_start, _pitch, note_duration in notes)
        crossing_out = any(note_start < end < note_start + note_duration for note_start, _pitch, note_duration in notes)
        pitches = [pitch for _note_start, pitch, _note_duration in onsets]
        occupied = sum(min(end, note_start + note_duration) - max(start, note_start)
                       for note_start, _pitch, note_duration in notes
                       if note_start < end and note_start + note_duration > start)
        bars.append({
            "bar": index + 1,
            "section": section_by_bar.get(index, "section"),
            "meter": f"{meter[0]}/{meter[1]}",
            "attacks": len(onsets),
            "distinct_pitches": len(set(pitches)),
            "adjacent_repeated_pitches": sum(left == right for left, right in zip(pitches, pitches[1:])),
            "pitch_sequence_midi": pitches,
            "longest_note_beats": round(float(max((note[2] for note in onsets), default=0)), 3),
            "rest_beats": round(float(duration - occupied), 3),
            "tie_in": crossing_in,
            "tie_out": crossing_out,
        })

    lyric_lines = [line.strip() for line in (lyrics or "").splitlines()
                   if line.strip() and not re.fullmatch(r"\[[^\]]+\]", line.strip())]
    vowels = "aeiouáéíóúü"
    unstressed_linkers = {
        "a", "al", "de", "del", "el", "la", "las", "los", "mi", "mis", "tu", "tus", "su", "sus",
        "un", "una", "unos", "unas", "me", "te", "se", "lo", "le", "les", "que", "y", "o",
    }
    number_syllables = {"0": 2, "1": 1, "2": 1, "3": 1, "4": 2, "5": 2, "6": 1, "7": 2,
                        "8": 2, "9": 2, "10": 2, "11": 3, "12": 3, "13": 3, "14": 3, "15": 2,
                        "16": 4, "17": 4, "18": 4, "19": 4, "20": 3}

    def line_evidence(number, text):
        words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|\d+", text)
        syllables = len(lyric_syllables(text)) + sum(number_syllables.get(word, 1) for word in words if word.isdigit())
        lexical = [word for word in words if not word.isdigit()]
        joins = [f"{left} {right}" for left, right in zip(lexical, lexical[1:])
                 if left[-1].lower() in vowels and right[0].lower() in vowels + "h"]
        prosodic_groups = [
            {
                "boundary": f"{left} {right}",
                "left_role": "unstressed_linker",
                "phrase_initial": index == 0,
            }
            for index, (left, right) in enumerate(zip(lexical, lexical[1:]))
            if left.lower() in unstressed_linkers
        ]
        return {"line": number, "text": text, "written_syllables": syllables,
                "connected_syllable_estimate": max(1, syllables - len(joins)),
                "possible_connected_vowel_boundaries": joins,
                "possible_prosodic_group_boundaries": prosodic_groups}

    lines = [line_evidence(index + 1, text) for index, text in enumerate(lyric_lines)]
    lyric_bars = [bar for bar in bars if bar["attacks"] >= 3]
    alignment = []
    if len(lines) == len(lyric_bars):
        for line, bar in zip(lines, lyric_bars):
            alignment.append({**line, **bar,
                              "written_syllable_attack_delta": line["written_syllables"] - bar["attacks"],
                              "connected_syllable_attack_delta": line["connected_syllable_estimate"] - bar["attacks"]})
        note = "Exact sequential candidate count: one lyric line per vocal bar with at least three attacks."
    else:
        note = (f"No automatic one-to-one mapping: {len(lines)} lyric lines versus {len(lyric_bars)} vocal bars "
                "with at least three attacks. Use sections, rests, and order to refine alignment.")
    return {"alignment": alignment, "all_vocal_bars": bars, "note": note}


def ollama_fix_lyrics(lyrics, score_abc="", edit_strength="light_rewrite", phonetic_assist=False,
                      instructions="", model="deepseek-v4.1-flash:cloud",
                      endpoint="http://127.0.0.1:11434", temperature=0.2, timeout=180,
                      procedural_evidence=""):
    if not (lyrics or "").strip():
        return "", json.dumps({"changes": [], "warnings": ["Lyrics are empty."]})
    if edit_strength not in LYRICS_EDIT_MODES:
        raise ValueError(f"Unknown edit_strength: {edit_strength}")
    score_capacity = _score_section_capacity(score_abc)
    prosody_evidence = _prosody_evidence(score_abc, lyrics)
    # This payload deliberately contains data, not a second layer of prompt instructions.
    # All behavior and mode definitions live in the editable system-prompt file.
    payload = {
        "language_locale": "es-MX",
        "lyrics": lyrics,
        "score_abc": score_abc,
        "score_section_capacity": score_capacity,
        "prosody_evidence": prosody_evidence,
        "edit_strength": edit_strength,
        "phonetic_assist": phonetic_assist,
        "user_instructions": instructions,
    }
    if (procedural_evidence or "").strip():
        try:
            payload["procedural_evidence"] = json.loads(procedural_evidence)
        except json.JSONDecodeError as exc:
            raise ValueError("procedural_evidence must be the JSON output of HZ3 Procedural Prosody.") from exc
    body = {"model": model, "stream": False, "think": False, "format": "json",
            "options": {"temperature": temperature},
            "messages": [{"role": "system", "content": _lyrics_system_prompt()},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
    request = urllib.request.Request(endpoint.rstrip("/") + "/api/chat",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama is unavailable at {endpoint}: {exc.reason}") from exc
    content = raw.get("message", {}).get("content", "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid lyrics JSON: {content[:500]}") from exc
    corrected = result.get("corrected_lyrics", "").replace("\r\n", "\n").strip()
    if not corrected:
        raise RuntimeError("Ollama returned empty corrected lyrics.")
    return corrected, json.dumps(result, ensure_ascii=False, indent=2)


class HZ3_YuE2_LyricsProsody:
    CATEGORY = "HZ3 YuE2"
    FUNCTION = "correct"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lyrics", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Use Ollama to format lyrics and improve syllabic fit against optional ABC section capacity."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lyrics": ("STRING", {"multiline": True, "default": ""}),
            "edit_strength": (["punctuation_only", "light_rewrite", "fit_to_score"], {"default": "light_rewrite"}),
            "phonetic_assist": ("BOOLEAN", {"default": False, "tooltip": "Allows experimental pronunciation guidance such as syllable separation, approximate spelling, contractions, elisions, and extended sounds."}),
            "instructions": ("STRING", {"multiline": True, "default": ""}),
            "model": ("STRING", {"default": "deepseek-v4.1-flash:cloud"}),
            "endpoint": ("STRING", {"default": "http://127.0.0.1:11434"}),
            "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
            "timeout": ("INT", {"default": 180, "min": 10, "max": 900}),
        }, "optional": {
            "score_abc": ("STRING", {"forceInput": True}),
            "procedural_evidence": ("STRING", {
                "forceInput": True,
                "tooltip": "Connect agent_evidence from HZ3 YuE2 · Procedural Prosody.",
            }),
        }}

    def correct(self, lyrics, edit_strength, phonetic_assist, instructions, model, endpoint,
                temperature, timeout, score_abc="", procedural_evidence=""):
        corrected, report = ollama_fix_lyrics(lyrics, score_abc, edit_strength, phonetic_assist,
                                               instructions, model, endpoint, temperature, timeout,
                                               procedural_evidence)
        return {"ui": {"text": [corrected + "\n\nCHANGES:\n" + report]}, "result": (corrected, report)}
