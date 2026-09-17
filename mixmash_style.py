"""Multi-source style profiling and LLM-assisted YuE2 mashup prompts."""

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = """You are an expert music producer and prompt engineer writing prompts for YuE2 music generation.
You adhere strictly to the YuE2 prompt engineering rules (dav.one/how-to-generate-music-with-yue2/):

INPUT DATA:
- structure: Ordered list of song sections (e.g. [{"name": "intro", "start": 0.0, "end": 29.0}, ...]).
- analysis: Measured audio classification profile (genres, instruments, mood, voice, BPM, key from Audio to Style).
- lyrics: The lyrics for the song (raw unsegmented text, or loosely labeled).
- instructions: Creative user direction (tempo, style changes, instrumentation, mood, genre, voice). If empty, infer a creative style fitting the lyrics and structure.
- section_cues: ABC score excerpts or arrangement cues for the sections.
- sources: Any additional audio analysis profiles.

PRIORITY AND MUSICAL HIERARCHY:
1. AUDIO ANALYSIS PRIORITY (when provided):
   If audio analysis (from Audio to Style / Discogs-EffNet) is present, it TAKES ABSOLUTE PRECEDENCE OVER INSTRUCTIONS.
   Extract and strictly follow the measured musical evidence: detected genres, instrumentation, acoustic vs electronic timbre, vocal gender/presence, danceability, and mood.
   Do not override the detected genre or real audio characteristics with conflicting instructions.

2. INSTRUCTIONS ROLE:
   - If audio analysis is present: use instructions only to guide section arrangement, progression, transitions, energy curves, or subtle performance nuances that complement the audio analysis. If instructions contradict the audio analysis, the audio analysis wins.
   - If audio analysis is NOT present: follow the instructions directly for genre, tempo, instrumentation, and mood.
   - If instructions are empty/omitted: creatively invent a rich, fitting arrangement and style based on the lyrics and musical cues.

3. SECTION-BY-SECTION STYLE PROMPT (style_detailed):
   - When a section structure is provided, write ONE unified YuE2 style prompt where EVERY section in the structure is represented in order with its exact bracketed tag:
     [Intro] instrumentation, tempo/BPM, key, room acoustics, no vocals.
     [Verse 1] voice timbre, instrumentation, bassline, groove.
     [Chorus] dynamic lift, energy, drum beat, layered vocal delivery, synths/guitars.
     ... representing all sections from the structure in exact order.
   - Cover the 5 essential ingredients: Genre, Instruments, Mood, Vocal gender, and Vocal tone/delivery.
   - Ensure the vocal description language matches the lyrics' language (e.g. Spanish male lead vocal for Spanish lyrics).
   - Extract any tempo/BPM, key, and meter from section_cues / ABC notation when available.
   - Conclude the style prompt with a concise line of global tags: BPM, meter, key, genres, and production characteristics.
   - If no section structure is provided, compose a rich, detailed YuE2 style prompt (70-130 words).

4. CORRECTED AND SEGMENTED LYRICS RULES (dav.one guide + non-sung cues):
   - Every section must have a [Label] in square brackets (e.g. [Verse], [Verse 2], [Chorus], [Bridge], [Chorus 2]).
   - Leave exactly ONE blank line between sections.
   - STRICT FIDELITY TO USER'S INTENT - DO NOT INVENT SPOKEN/ACAPELLA CUES CREATIVELY:
     * Regular lyrics must ALWAYS remain standard sung lines (without parentheses) by default.
     * Spoken / read / recited lyrics: put lyrics inside parentheses, e.g. (texto de la letra hablada), ONLY if the user explicitly provided them in parentheses in the input lyrics or explicitly instructed that the section be spoken/recited in `instructions`. NEVER convert normal sung lyrics into spoken parentheses on your own.
     * Acapella sections: use (acapella) ONLY if explicitly requested in `instructions` or indicated in the input lyrics.
     * Instrumental sections ([Intro], [Interlude], [Solo], instrumental [Outro]): when a section from the structure has no lyrics assigned from the input, use (instrumental) or leave the section empty underneath.
   - DO NOT open the song with singing lyrics under [Intro]: use an empty [Intro], [Intro] with (instrumental), or begin directly with [Verse]/[Chorus].
   - Keep each section short: roughly a handful of lines (singable in ~30 seconds).
   - Write repeated sections (like choruses) OUT IN FULL: never use [Chorus x2] or (repeat chorus).
   - Prohibited stage directions: do NOT include musical/production directions like (drums build), (pause), (key change), or (whisper). Only valid section cues like (instrumental), (acapella), or explicit spoken lyrics inside parentheses (texto hablado) are permitted.
   - If no structure is provided, clean and format the lyrics into standard [Verse], [Chorus], etc. blocks following the rules above.

5. BALANCED STYLE (style_balanced):
   - A concise 35-65 word version highlighting the core genre, main instruments, key contrast, and vocal character.

6. COMPACT STYLE (style_compact):
   - A quick 15-30 word summary of genre, principal instruments, mood, vocal gender/tone, and BPM/groove.

RETURN FORMAT:
Return valid JSON only with exactly these keys:
{
  "style_detailed": "<the complete multi-section style prompt with [Section] blocks>",
  "corrected_lyrics": "<the cleanly formatted lyrics with [Section] headers, blank lines, and (instrumental) or (spoken text) if applicable>",
  "style_balanced": "<35-65 words summary>",
  "style_compact": "<15-30 words summary>"
}
No markdown formatting around the JSON, no extra keys, no explanatory text."""


def _parse_structure(data):
    """Parse section structure if data is a JSON string or dict/list of sections."""
    if not data:
        return []
    if isinstance(data, (list, tuple)):
        payload = data
    elif isinstance(data, str):
        text = data.strip()
        if not (text.startswith("[") or text.startswith("{")):
            return []
        try:
            payload = json.loads(text)
        except Exception:
            return []
    elif isinstance(data, dict):
        payload = data
    else:
        return []

    if isinstance(payload, dict):
        payload = payload.get("sections") or payload.get("structure") or payload.get("clips") or []
    if not isinstance(payload, list):
        return []
    sections = []
    for item in payload:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("kind") or "section")
            start = item.get("start")
            end = item.get("end")
            sections.append({
                "name": name,
                "start": round(float(start), 3) if start is not None else 0.0,
                "end": round(float(end), 3) if end is not None else 0.0,
            })
        elif isinstance(item, str) and item.strip():
            sections.append({"name": item.strip(), "start": 0.0, "end": 0.0})
    return sections


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
    if first and first[0].strip().lower() == "[intro]":
        content_lines = [l.strip() for l in first[1:] if l.strip()]
        if content_lines and not all(l.lower() in ("(instrumental)", "[instrumental]") for l in content_lines):
            warnings.append("A lyric-filled opening [Intro] is unstable; use an empty [Intro], (instrumental), or begin with [Verse]/[Chorus].")
    # Prohibit stage/repetition directions; allow (instrumental), (acapella), and spoken lyrics in parentheses
    if re.search(r"\((?:repeat|drums?|pause|key change|solo|guitar solo)[^)]*\)", text, re.I):
        warnings.append("Lyrics contain a production/repeat direction in parentheses; YuE2 may try to sing it.")
    if re.search(r"\[(?:chorus|verse)\s*x\d+\]", text, re.I):
        warnings.append("Write repeated sections out in full instead of using [Section xN].")
    return warnings


def ollama_mixmash(context_1="", mix_instructions="", lyrics="", context_2="", context_3="", section_cues="",
                   model="deepseek-v4.1-flash:cloud", endpoint="http://127.0.0.1:11434", temperature=0.35,
                   timeout=180, instructions="", structure="", analysis=""):
    instructions_text = (instructions or mix_instructions or "").strip()

    # Resolve structure: explicit structure argument, or parsed from context_1
    parsed_struct = _parse_structure(structure)
    if not parsed_struct and context_1:
        parsed_struct = _parse_structure(context_1)
        if parsed_struct:
            context_1 = ""  # Consumed as structure

    # Resolve audio analysis: explicit analysis argument, or non-JSON context
    analysis_text = (analysis or "").strip()
    contexts = [text.strip() for text in (context_1, context_2, context_3) if (text or "").strip()]
    if not analysis_text and contexts:
        analysis_text = "\n\n".join(contexts)
        contexts = []

    # Section cues (handle list of ABC strings from SheetSage2 segments_abc or string)
    if isinstance(section_cues, (list, tuple)):
        cues_text = "\n\n".join(str(s) for s in section_cues if (s or "").strip())
    else:
        cues_text = str(section_cues or "").strip()

    lyrics_text = (lyrics or "").strip()

    if not parsed_struct and not analysis_text and not instructions_text and not lyrics_text and not cues_text and not contexts:
        raise ValueError("Provide at least one input (structure, analysis, instructions, lyrics, or section_cues) for MixMash Style.")

    user_payload = {
        "structure": parsed_struct,
        "analysis": analysis_text,
        "instructions": instructions_text,
        "section_cues": cues_text,
        "lyrics": lyrics_text,
        "sources": [{"source": index, "analysis": text} for index, text in enumerate(contexts, 1)],
    }
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
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
    detailed = result.get("style_detailed", legacy).strip()
    detailed = re.sub(r"\n{3,}", "\n\n", detailed)
    balanced = result.get("style_balanced", detailed).strip().replace("\n", " ")
    compact = result.get("style_compact", balanced).strip().replace("\n", " ")
    corrected_lyrics = result.get("corrected_lyrics", result.get("lyrics", lyrics_text)).strip()
    if not detailed:
        raise RuntimeError("Ollama returned an empty YuE2 style.")
    report = json.dumps(result, ensure_ascii=False, indent=2)
    return detailed, balanced, compact, report, corrected_lyrics


class HZ3_YuE2_MixMashStyle:
    CATEGORY = "HZ3 YuE2"
    FUNCTION = "compose"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("style", "lyrics", "style_compact", "style_balanced")
    OUTPUT_NODE = True
    DESCRIPTION = "Ask an Ollama model to turn audio analysis, structure and lyrics into a coherent multi-section YuE2 style prompt and formatted lyrics."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "deepseek-v4.1-flash:cloud"}),
                "endpoint": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "temperature": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05}),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 900}),
            },
            "optional": {
                "structure": ("STRING", {"forceInput": True, "tooltip": "Optional: connect 'structure' or 'report' JSON from SheetSage2 Audio to ABC + Sections."}),
                "analysis": ("STRING", {"forceInput": True, "tooltip": "Optional: connect 'analysis' from Audio to Style (Discogs-EffNet). When present, takes priority over instructions."}),
                "section_cues": ("STRING", {"forceInput": True, "tooltip": "Optional: connect 'segments_abc' from SheetSage2 Sections or ABC score text."}),
                "instructions": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional: speed, changes, style, instruments, mood. If empty, LLM creates a fitting style for the lyrics."}),
                "lyrics": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional: song lyrics (raw or with section labels). Will be formatted to match the sections."}),
                "context": ("STRING", {"forceInput": True, "tooltip": "Optional: single context input (if multiple profiles are needed, join them with String Concatenate)."}),
            },
        }

    def compose(self, model="deepseek-v4.1-flash:cloud", endpoint="http://127.0.0.1:11434",
                temperature=0.35, timeout=180,
                structure="", analysis="", section_cues="", instructions="", lyrics="",
                context="", **kwargs):
        struct = structure or kwargs.get("report", "")
        analys = analysis or kwargs.get("audio_analysis", "")
        mix_inst = (instructions or kwargs.get("mix_instructions", "") or "").strip()

        # Support single context and legacy context_1/2/3/contexto
        ctx = context or kwargs.get("context_1", "") or kwargs.get("contexto", "") or ""
        c2 = kwargs.get("context_2", "") or kwargs.get("contexto_2", "") or ""
        c3 = kwargs.get("context_3", "") or kwargs.get("contexto_3", "") or ""

        lyr = (lyrics or kwargs.get("letra", "") or "").strip()
        cues = section_cues if section_cues is not None else kwargs.get("abc", "")

        style, balanced, compact, report, corrected_lyrics = ollama_mixmash(
            context_1=ctx,
            mix_instructions=mix_inst,
            lyrics=lyr,
            context_2=c2,
            context_3=c3,
            section_cues=cues,
            model=model,
            endpoint=endpoint,
            temperature=temperature,
            timeout=timeout,
            instructions=mix_inst,
            structure=struct,
            analysis=analys,
        )
        final_lyrics = corrected_lyrics if corrected_lyrics else lyr
        warnings = lyrics_warnings(final_lyrics)
        visible = ("DETAILED STYLE (output: style):\n" + style +
                   "\n\nCORRECTED LYRICS (output: lyrics):\n" + final_lyrics +
                   "\n\nBALANCED STYLE:\n" + balanced +
                   "\n\nCOMPACT STYLE:\n" + compact)
        if warnings:
            visible += "\n\nLYRICS CHECK:\n- " + "\n- ".join(warnings)
        return {"ui": {"text": [visible]},
                "result": (style, final_lyrics, compact, balanced)}


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
