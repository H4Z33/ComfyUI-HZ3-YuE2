"""Multi-source style profiling and LLM-assisted YuE2 mashup prompts."""

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = """You are an expert music producer and prompt engineer writing style prompts and formatting lyrics for YuE2 music generation.

You receive JSON with:
- structure: Ordered list of song sections with start/end times and names.
- section_vocal_evidence: Optional alignment of transcribed vocals detected in the audio for each section time window.
- analysis: Measured audio classification profile (genres, instruments, mood, voice, BPM, key).
- instructions: Creative direction (style, tempo, instruments, mood, arrangement changes).
- section_cues: Score notation (ABC) or arrangement cues for sections.
- lyrics: Song lyrics provided by the user.
- sources: Additional reference audio profiles.

1. LANGUAGE RULES:
   - ALL STYLE PROMPTS MUST BE WRITTEN EXCLUSIVELY IN ENGLISH.
   - All instruments, moods, arrangement actions, and vocal descriptors must be in English (e.g. "Spanish male vocal", "intimate delivery", "punchy drums", "walking bass", "electric piano", "brass section arrives").
   - Never write style descriptions in Spanish or other languages, even if lyrics or instructions are in Spanish. YuE2 requires English style conditioning.
   - Lyrics keep their original language (Spanish, English, etc.).

2. MUSICAL HIERARCHY AND ABSOLUTE PRIORITY (analysis > instructions > inference):
   - PRIORITY 1 (KING): AUDIO ANALYSIS (when provided from audio analysis / Discogs-EffNet):
     Takes absolute top priority over everything else. Follow the measured musical evidence strictly: detected genres, instrumentation, acoustic vs electronic timbre, vocal gender/presence, tempo, and mood. Never override the detected genre or real audio characteristics with instructions or creative ideas.
   - PRIORITY 2: USER INSTRUCTIONS:
     * If audio analysis is present: instructions guide arrangement progression, transitions, energy curves, and specific requested instruments/sections (e.g. big band al final, solo de batería) that complement the audio analysis. If instructions contradict the audio analysis, the audio analysis wins.
     * If audio analysis is NOT present: user instructions are KING and must be followed directly and strictly for genre, style, tempo, instrumentation, mood, and vocal delivery.
   - PRIORITY 3: AGENT INFERENCE (ONLY WHEN NEITHER ANALYSIS NOR INSTRUCTIONS EXIST):
     * Inferring or creatively inventing a style is permitted ONLY and EXCLUSIVELY when BOTH analysis AND instructions are completely empty/absent.
     * When analysis OR instructions are provided, DO NOT invent unrelated styles, genres, or unrequested elements. Follow the specified analysis and instructions strictly.

3. SECTION STRUCTURE AND 1:1 MAPPING:
   - Follow the exact sequence and count of sections from the structure.
   - STRICT FIDELITY - NO FABRICATED SECTIONS:
     * Never add an [Intro] if the structure does not begin with an intro. If structure begins with "verse", start directly with [Verse] or [Verse 1].
     * Never append extra sections that do not exist in the structure.
   - FINAL / OUTRO DIRECTIVE:
     * If the user's instructions mention a finale or ending (e.g. "final", "outro", "ending", "cierre"), label the LAST section in the structure as [Outro] and apply the requested finale elements there.
     * Otherwise, keep the original section label for the last section.

4. HYBRID SECTION STYLE PROMPT (style_detailed):
   - START style_detailed with a single concise global summary line for the track:
     BPM: <number>, Meter: <meter>, Key: <key>, <core genre and production tags>, <vocal language and type>.
   - Then, for every section in the structure, write an evocative, musical, and precise 1-2 sentence description with its bracketed tag:
     [Section] Describe what enters, develops, or changes: instrumentation, groove, dynamic texture, and vocal character.
   - DO NOT mindlessly repeat the genre name in every section bracket if the genre does not change. Focus on arrangement progression, dynamic builds, and textural contrast.
   - Examples of desired section phrasing:
     * [Verse 1] Intimate Spanish male lead enters over electric piano, melodic walking bass and subtle tambourine, restrained groove.
     * [Verse 2] Add warm tremolo guitar and subtle string swells, vocal delivery building in emotional intensity.
     * [Chorus] Big band brass section arrives with punchy drums, driving bass and soaring passionate vocal harmonies.
     * [Interlude] Jazz drum solo with energetic brass stabs and walking bass, swinging dynamic break, instrumental no vocals.
     * [Outro] Dynamic jazz drums and full big band brass in crescendo, passionate vocal ad-libs ending on an energetic flourish.
   - Instrumental sections must explicitly specify "instrumental, no vocals".
   - If no structure is provided, write a rich, cohesive YuE2 style prompt (50-80 words in English).

5. CONTRASTING AND RECONCILING LYRICS (corrected_lyrics):
   - Every section must have a bracketed label matching the structure: [Verse], [Verse 2], [Chorus], [Bridge], [Outro], etc.
   - Leave exactly ONE blank line between sections.
   - CONTRAST INPUTS AND PREVENT DUPLICATION:
     * When section_vocal_evidence is provided from audio, use it to determine WHICH lyrics belong to WHICH section. Match the clean user lyrics to the corresponding audio section.
     * When distributing user lyrics across multiple verses, each distinct stanza from the input must go to its own section (e.g. Stanza 1 -> [Verse], Stanza 2 -> [Verse 2], Stanza 4 -> [Verse 3]).
     * NEVER duplicate or repeat Stanza 1 across different verses! Each verse must contain its own unique lyric lines.
     * Instrumental sections ([Interlude], [Solo], wordless [Intro] or [Outro]) MUST contain (instrumental) or be empty. NEVER force vocal lyrics into an instrumental section.
     * Acapella sections: use (acapella) ONLY if explicitly requested in instructions.
     * Spoken/read lyrics: use parentheses (texto de la letra) ONLY if explicitly provided in parentheses in the input or requested as spoken in instructions.
     * Repeated sections (like choruses) must be written out in full; never use [Chorus x2].
     * Prohibited: no stage directions like (drums build), (pause), or (key change).
   - SINGABILITY, PUNCTUATION & SYLLABLE PROSODY:
     * Punctuation marks breath and cadence: use commas (,) for breathing micro-pauses within a line, and periods (.) for full musical phrase cadences.
     * Line breaks: divide lines into natural singing breath phrases (~4-8 syllables per line) so the singer does not rush or drag.
     * Syllables, contractions, and extensions: where words blend smoothly into one musical beat (sinalefa / elision), allow natural locale contractions or underscore joining (e.g. "volvió_ala", "de_este"); where a phrase cadence holds a sustained musical note, support natural held syllables if appropriate.

6. BALANCED AND COMPACT STYLE PROMPTS:
   - style_balanced: 25-45 words in English summarizing genre, key instruments, mood, and vocal character.
   - style_compact: 12-25 words in English summarizing genre, main instruments, mood, and vocal type.

RETURN FORMAT:
Return valid JSON only with exactly these keys:
{
  "style_detailed": "<the complete multi-section style prompt in English with [Section] tags>",
  "corrected_lyrics": "<the cleanly formatted lyrics with matching [Section] tags>",
  "style_balanced": "<25-45 words in English>",
  "style_compact": "<12-25 words in English>"
}
No markdown formatting around the JSON, no extra keys, no explanatory text."""


def _detect_transcript_segments(data):
    """Detect if data is a list of Whisper transcribed segments with timestamps."""
    if not data:
        return None
    if isinstance(data, str):
        text = data.strip()
        if not text.startswith("["):
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], dict):
        if "text" in data[0] and ("start" in data[0] or "end" in data[0]):
            return list(data)
    return None


def _align_segments_to_structure(structure, segments):
    """Correlate timestamped Whisper segments to song sections."""
    section_vocals = []
    for s in structure:
        s_start = s.get("start", 0.0)
        s_end = s.get("end", 0.0)
        matched_texts = []
        for seg in segments:
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", seg_start)
            mid = (seg_start + seg_end) / 2.0
            if s_start <= mid < s_end:
                t = (seg.get("text") or "").strip()
                if t:
                    matched_texts.append(t)
        section_vocals.append({
            "section": s.get("name", "section"),
            "time": f"{s_start:.2f}-{s_end:.2f}s",
            "transcribed_audio": " ".join(matched_texts) if matched_texts else "(no vocals detected in audio)",
        })
    return section_vocals


def _parse_structure(data):
    """Parse section structure from JSON list/dict or SheetSage2 report text."""
    if not data:
        return []
    if isinstance(data, (list, tuple)):
        payload = data
    elif isinstance(data, dict):
        payload = data
    elif isinstance(data, str):
        text = data.strip()
        if not text:
            return []
        payload = None
        if text.startswith("[") or text.startswith("{"):
            try:
                payload = json.loads(text)
            except Exception:
                payload = None
        if payload is None:
            # Match SheetSage2 report format: lines like "verse 0.00-26.00s" or "chorus 2 130.00-168.00s"
            matches = re.findall(r"^\s*([a-zA-Z0-9_ ]+?)\s+([\d\.]+)-([\d\.]+)s\s*$", text, re.M)
            if matches:
                return [{
                    "name": name.strip(),
                    "start": round(float(start), 3),
                    "end": round(float(end), 3),
                } for name, start, end in matches]
            return []
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
                   timeout=180, instructions="", structure="", analysis="", context=""):
    instructions_text = (instructions or mix_instructions or "").strip()

    # Resolve structure: explicit structure argument, or parsed from context / context_1
    parsed_struct = _parse_structure(structure)
    if not parsed_struct and context:
        parsed_struct = _parse_structure(context)
        if parsed_struct:
            context = ""
    if not parsed_struct and context_1:
        parsed_struct = _parse_structure(context_1)
        if parsed_struct:
            context_1 = ""  # Consumed as structure

    # If instructions mention a finale / ending, designate the last section as outro
    if parsed_struct and re.search(r"\b(final|outro|ending|cierre|terminar|finalizar)\b", instructions_text, re.I):
        if parsed_struct[-1]["name"].lower() != "outro":
            parsed_struct[-1]["name"] = "outro"

    # Detect if context, analysis, or legacy context_1 has timestamped Whisper transcribed segments
    transcript_segs = _detect_transcript_segments(context) or _detect_transcript_segments(analysis)
    if not transcript_segs:
        for c in (context_1, context_2, context_3):
            transcript_segs = _detect_transcript_segments(c)
            if transcript_segs:
                break

    section_vocal_evidence = []
    if transcript_segs and parsed_struct:
        section_vocal_evidence = _align_segments_to_structure(parsed_struct, transcript_segs)
        if _detect_transcript_segments(context):
            context = ""
        if _detect_transcript_segments(analysis):
            analysis = ""
        if _detect_transcript_segments(context_1):
            context_1 = ""

    # Resolve audio analysis: explicit analysis argument, or non-JSON context
    analysis_text = (analysis or "").strip()
    contexts = [text.strip() for text in (context, context_1, context_2, context_3) if (text or "").strip()]
    if not analysis_text and contexts:
        analysis_text = "\n\n".join(contexts)
        contexts = []

    # Section cues (handle list of ABC strings from SheetSage2 segments_abc or string)
    if isinstance(section_cues, (list, tuple)):
        cues_text = "\n\n".join(str(s) for s in section_cues if (s or "").strip())
    else:
        cues_text = str(section_cues or "").strip()

    lyrics_text = (lyrics or "").strip()

    if not parsed_struct and not analysis_text and not instructions_text and not lyrics_text and not cues_text and not contexts and not section_vocal_evidence:
        raise ValueError("Provide at least one input (structure, analysis, instructions, lyrics, or section_cues) for MixMash Style.")

    user_payload = {
        "structure": parsed_struct,
        "section_vocal_evidence": section_vocal_evidence,
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
    # Ensure global info line (BPM, Meter, Key, Genre...) is at the beginning of style_detailed
    lines = [l.strip() for l in detailed.splitlines() if l.strip()]
    if lines:
        bpm_idx = None
        for i, line in enumerate(lines):
            if re.match(r"^BPM\s*:\s*\d+", line, re.I):
                bpm_idx = i
                break
        if bpm_idx is not None and bpm_idx > 0:
            bpm_line = lines.pop(bpm_idx)
            lines.insert(0, bpm_line)
            detailed = "\n".join(lines)
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
            context=ctx,
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
