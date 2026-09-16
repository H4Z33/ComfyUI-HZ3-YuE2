import re

from .abc_score import AbcError, KEYS, QUALITIES, parse


def lyric_syllables(lyrics):
    if not isinstance(lyrics, str):
        return []
    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", re.sub(r"\[[^\]]+\]", " ", lyrics))
    result = []
    vowels = "aeiouáéíóúüAEIOUÁÉÍÓÚÜ"
    onset_clusters = {"pr", "br", "tr", "dr", "cr", "gr", "fr", "pl", "bl", "cl", "gl", "fl", "ch", "ll", "rr"}
    for word in words:
        nuclei = list(re.finditer(rf"[{vowels}]+", word))
        if len(nuclei) < 2:
            result.append(word)
            continue
        start = 0
        for left, right in zip(nuclei, nuclei[1:]):
            consonants = word[left.end():right.start()]
            boundary = left.end() if len(consonants) <= 1 or consonants.lower() in onset_clusters else right.start() - 1
            result.append(word[start:boundary])
            start = boundary
        result.append(word[start:])
    return result


def inspect_score(text, lyrics=""):
    if not isinstance(text, str) or len(text) > 200_000:
        raise AbcError("Supply ABC text of at most 200,000 characters.")
    text = text.replace("\r\n", "\n").strip() + "\n"
    score = parse(text)
    lines = text.splitlines()
    sections = []
    current = None
    grid_available = not any(line.startswith("M:") for line in lines[8:])
    for index in range(8, len(lines)):
        line = lines[index]
        if line.startswith("% "):
            current = {"name": line[2:], "vocal": [], "instrumental": []}
            sections.append(current)
        if index not in score.music_lines:
            continue
        if current is None:
            current = {"name": "section", "vocal": [], "instrumental": []}
            sections.append(current)
        voice = "vocal" if score.music_lines[index] == "Vocal" else "instrumental"
        for bar in line[:-1].split("|"):
            bar = bar.strip()
            rest = re.fullmatch(r"Z([2-4])?", bar)
            current[voice].extend(["Z"] * int(rest.group(1) or 1) if rest else [bar])
    sections = [section for section in sections if section["vocal"]]
    bars = len(score.voices["Vocal"].bars)
    seconds = float(score.voices["Vocal"].time * 60 / score.bpm)
    instrumental = not score.voices["Vocal"].notes
    summary = f"Valid native ABC: {bars} bars, {score.bpm} BPM, approximately {seconds:.1f} seconds."
    summary += " Vocal part contains rests only." if instrumental else " Vocal melody present."
    if not grid_available:
        summary += " Meter changes: edit in the ABC tab."
    roll_sections = []
    start = 0
    for section in sections:
        roll_sections.append({"name": section["name"], "bars": len(section["vocal"]), "start": start})
        start += len(section["vocal"])
    roll = {
        "tracks": {name: [{"start": int(t * 256), "duration": int(d * 256), "pitch": pitch} for t, pitch, d in voice.notes] for name, voice in score.voices.items()},
        "chords": [{"start": int(t * 256), "symbol": symbol} for t, symbol in score.voices["Vocal"].chords],
        "sections": roll_sections,
        "bar_ticks": int(score.voices["Vocal"].bars[0][1] * 256),
        "total_ticks": int(score.voices["Vocal"].time * 256),
    }
    syllables = lyric_syllables(lyrics)
    vocal_notes = len(score.voices["Vocal"].notes)
    return {
        "abc": text, "summary": summary, "bpm": score.bpm, "meter": lines[2][2:], "unit": lines[3][2:],
        "key": lines[7][2:], "seconds": seconds, "bars": bars, "instrumental": instrumental,
        "grid_available": grid_available, "sections": sections, "keys": list(KEYS), "qualities": list(QUALITIES),
        "roll": roll, "lyrics": lyrics, "syllables": syllables, "lyric_note_delta": vocal_notes - len(syllables),
    }
