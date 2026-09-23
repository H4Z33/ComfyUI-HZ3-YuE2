"""Deterministic quartet voicing on the native YuE2 ABC timeline."""

from bisect import bisect_left, bisect_right
from fractions import Fraction
from itertools import product
import re

from .abc_score import AbcError, DURATIONS, NATURAL, TOKEN, parse


CHORD_INTERVALS = {
    "": (0, 4, 7), "m": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
    "dim7": (0, 3, 6, 9), "m7b5": (0, 3, 6, 10), "sus4": (0, 5, 7),
    "sus2": (0, 2, 7), "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9),
    "7sus4": (0, 5, 7, 10), "m(maj7)": (0, 3, 7, 11),
}
PARTS = ("tenor", "baritone", "bass")
DEFAULT_RANGES = ((60, 84), (48, 76), (36, 64))
ARRANGEMENTS = ("close_harmony", "barbershop_inspired")


def _pitch_class(name):
    return (NATURAL[name[0]] + name[1:].count("#") - name[1:].count("b")) % 12


def _chord(symbol):
    match = re.fullmatch(r"([A-G](?:bb|##|b|#)?)([^/]*)(?:/([A-G](?:bb|##|b|#)?))?", symbol)
    root_name, quality, slash = match.groups()
    root = _pitch_class(root_name)
    tones = tuple((root + interval) % 12 for interval in CHORD_INTERVALS[quality])
    bass = _pitch_class(slash) if slash else root
    return tones, bass


def _key_triads(key, melody):
    minor = key.endswith("m")
    root = _pitch_class(key[:-1] if minor else key)
    scale = (0, 2, 3, 5, 7, 8, 10) if minor else (0, 2, 4, 5, 7, 9, 11)
    scale = tuple((root + interval) % 12 for interval in scale)
    triads = [tuple(scale[(degree + step) % 7] for step in (0, 2, 4)) for degree in range(7)]
    matching = [tones for tones in triads if melody % 12 in tones]
    return [(tones, tones[0]) for tones in (matching or triads)]


def _section_selected(name, selectors):
    name = name.strip().casefold()
    family = re.sub(r"\s+\d+$", "", name)
    return not selectors or name in selectors or family in selectors


def _layout(score):
    """Keep original groups and section comments, expanding full-measure rests."""
    groups = []
    sections = [(Fraction(0), "section")]
    bar_indices = {"Vocal": 0, "Ins": 0}
    for line_index, line in enumerate(score.text.splitlines()):
        if line.startswith("% "):
            bar_index = bar_indices["Vocal"]
            time = score.voices["Vocal"].bars[bar_index][0] if bar_index < len(score.voices["Vocal"].bars) else score.voices["Vocal"].time
            sections.append((time, line[2:].strip()))
        voice_name = score.music_lines.get(line_index)
        if voice_name is not None:
            bars = []
            for raw in line[:-1].split("|"):
                raw = raw.strip()
                rest = re.fullmatch(r"Z([2-4])?", raw)
                for body in (["Z"] * int(rest.group(1) or 1) if rest else [raw]):
                    start, length, _ = score.voices[voice_name].bars[bar_indices[voice_name]]
                    inline_keys = []
                    offset = start
                    if body != "Z":
                        for token in TOKEN.finditer(body):
                            if token.group("key") is not None:
                                inline_keys.append((offset, token.group("key")))
                            elif token.group("note") is not None:
                                offset += int(token.group("duration") or 1) * score.unit * 4
                    bars.append((start, length, inline_keys))
                    bar_indices[voice_name] += 1
            groups.append((line_index, voice_name, bars))
    return groups, sections


def _choose_voicing(melody, harmonies, ranges, previous, arrangement):
    best = None
    for tones, bass_pc in harmonies:
        candidates = []
        for index, (low, high) in enumerate(ranges):
            palette = set(tones) | ({bass_pc} if index == 2 else set())
            pitches = [pitch for pitch in range(low, high + 1) if pitch % 12 in palette]
            # Very narrow user ranges can exclude every chord tone. Keep the range,
            # make the least conflicting choice, and report the compromise.
            if not pitches:
                pitches = list(range(low, high + 1))
            target = previous[index] if previous else (melody + 4, melody - 4, melody - 19)[index]
            candidates.append(sorted(pitches, key=lambda pitch: (abs(pitch - target), pitch))[:10])
        for voicing in product(*candidates):
            tenor, baritone, bass = voicing
            ordered = tenor > melody and tenor > baritone and bass < min(melody, baritone)
            pcs = {melody % 12, tenor % 12, baritone % 12, bass % 12}
            missing = set(tones) - pcs
            cost = 40 * (not ordered) + 7 * len(missing)
            cost += 5 * (bass % 12 != bass_pc)
            cost += 14 * sum(pitch % 12 not in set(tones) | ({bass_pc} if index == 2 else set())
                             for index, pitch in enumerate(voicing))
            cost += 1.5 * (4 - len(pcs))
            cost += 0.12 * abs(baritone - melody) + 0.08 * abs(tenor - melody)
            cost += (0.7 if arrangement == "barbershop_inspired" else 0.35) * max(0, tenor - min(melody, baritone) - 12)
            if arrangement == "barbershop_inspired" and len(tones) == 4:
                cost += 5 * len(missing & {tones[1], tones[3]})
            if previous:
                moves = [abs(pitch - old) for pitch, old in zip(voicing, previous)]
                cost += 0.4 * sum(moves) + 1.2 * sum(max(0, move - 7) for move in moves)
            else:
                cost += 0.05 * sum(abs(pitch - (low + high) / 2) for pitch, (low, high) in zip(voicing, ranges))
            choice = (cost, voicing, ordered, all(pitch % 12 in set(tones) | ({bass_pc} if index == 2 else set())
                                                for index, pitch in enumerate(voicing)))
            if best is None or choice[:2] < best[:2]:
                best = choice
    return best[1:]


def _note_name(pitch):
    names = ("=C", "^C", "=D", "^D", "=E", "=F", "^F", "=G", "^G", "=A", "^A", "=B")
    octave, pc = divmod(pitch - 60, 12)
    name = names[pc]
    if octave >= 1:
        return name.lower() + "'" * (octave - 1)
    return name + "," * -octave


def _write_span(pitch, duration, unit, continues):
    units = duration / (4 * unit)
    if units.denominator != 1:
        raise AbcError("A harmony boundary cannot be represented by the source L: unit.")
    remaining = int(units)
    result = []
    while remaining:
        size = max(value for value in DURATIONS if value <= remaining)
        remaining -= size
        result.append(("z" if pitch is None else _note_name(pitch)) + str(size)
                      + ("-" if pitch is not None and (remaining or continues) else ""))
    return "".join(result)


def _render_bar(notes, start, length, inline_keys, unit):
    end = start + length
    sounding = [(time, pitch, duration) for time, pitch, duration in notes if time < end and time + duration > start]
    if not sounding and not inline_keys:
        return "Z"
    boundaries = {start, end}
    for time, _, duration in sounding:
        boundaries.update((max(start, time), min(end, time + duration)))
    boundaries.update(time for time, _ in inline_keys)
    boundaries = sorted(boundaries)
    result = []
    note_index = 0
    for left, right in zip(boundaries, boundaries[1:]):
        result.extend(f"[K:{key}]" for time, key in inline_keys if time == left)
        while note_index < len(sounding) and sounding[note_index][0] + sounding[note_index][2] <= left:
            note_index += 1
        event = sounding[note_index] if note_index < len(sounding) and sounding[note_index][0] <= left else None
        result.append(_write_span(event[1] if event else None, right - left, unit,
                                  bool(event and event[0] + event[2] > right)))
    return "".join(result)


def _serialize(score, groups, notes):
    lines = score.text.splitlines()
    for line_index, voice_name, bars in groups:
        voice_notes = notes if voice_name == "Vocal" else []
        lines[line_index] = "|".join(_render_bar(voice_notes, start, length, keys, score.unit)
                                   for start, length, keys in bars) + "|"
    result = "\n".join(lines) + "\n"
    checked = parse(result)
    original = score.voices["Vocal"]
    for voice in checked.voices.values():
        if voice.bars != original.bars or voice.keys != original.keys or voice.time != original.time:
            raise AbcError("Harmony output changed the source timeline.")
    if checked.voices["Vocal"].notes != [list(note) for note in notes]:
        raise AbcError("Harmony output changed note timing or pitch during serialization.")
    return result


def harmonize(score_abc, arrangement="close_harmony", active_sections="", ranges=DEFAULT_RANGES):
    score = parse(score_abc.strip() + "\n")
    if arrangement not in ARRANGEMENTS:
        raise ValueError(f"Unknown harmony arrangement: {arrangement}")
    ranges = tuple(tuple(sorted((max(0, min(127, int(low))), max(0, min(127, int(high)))))) for low, high in ranges)
    groups, sections = _layout(score)
    selectors = {item.strip().casefold() for item in active_sections.split(",") if item.strip()}
    if selectors == {"all"}:
        selectors = set()
    source = score.voices["Vocal"]
    chord_times = [time for time, _ in source.chords]
    key_times = [time for time, _ in source.keys]
    section_times = [time for time, _ in sections]
    cuts = sorted(set(chord_times + key_times + section_times))
    parts = [[], [], []]
    previous = None
    previous_end = None
    inferred = relaxed = outside_chord = slices = 0
    for start, melody, duration in source.notes:
        end = start + duration
        boundaries = [start] + cuts[bisect_right(cuts, start):bisect_left(cuts, end)] + [end]
        for left, right in zip(boundaries, boundaries[1:]):
            section = sections[bisect_right(section_times, left) - 1][1]
            if not _section_selected(section, selectors):
                previous = None
                continue
            if previous_end != left:
                previous = None
            chord_index = bisect_right(chord_times, left) - 1
            key_index = bisect_right(key_times, left) - 1
            key_time, key = source.keys[key_index]
            if chord_index >= 0 and chord_times[chord_index] >= key_time:
                harmonies = [_chord(source.chords[chord_index][1])]
            else:
                harmonies = _key_triads(key, melody)
                inferred += 1
            voicing, ordered, chord_tones = _choose_voicing(melody, harmonies, ranges, previous, arrangement)
            relaxed += not ordered
            outside_chord += not chord_tones
            slices += 1
            for notes, pitch in zip(parts, voicing):
                # Rejoin unchanged harmony across a chord/key boundary within the
                # same lead note, but never merge separate syllable attacks.
                if left > start and notes and notes[-1][0] + notes[-1][2] == left and notes[-1][1] == pitch:
                    notes[-1][2] += right - left
                else:
                    notes.append([left, pitch, right - left])
            previous, previous_end = voicing, right
    outputs = [_serialize(score, groups, notes) for notes in [source.notes] + parts]
    seconds = float(source.time * 60 / score.bpm)
    report = [f"{arrangement}: {len(source.bars)} bars, {seconds:.3f} seconds per part at {score.bpm} BPM.",
              "Lead preserved; instrumental tracks are silent. Harmonies follow lead rests and attacks.",
              f"Voiced spans: {slices}; inferred chords: {inferred}; relaxed voice ordering: {relaxed}; non-chord compromises: {outside_chord}."]
    for name, notes, (low, high) in zip(PARTS, parts, ranges):
        pitches = [note[1] for note in notes]
        actual = f"{min(pitches)}-{max(pitches)}" if pitches else "silent"
        report.append(f"{name}: MIDI range {low}-{high}; actual {actual}.")
    if selectors and not any(_section_selected(name, selectors) for _, name in sections):
        report.append("No section matched; harmony outputs contain rests. Use comma-separated ABC section names, e.g. chorus, outro.")
    if inferred:
        report.append("Missing chords were inferred from the active key and melody; review these harmonies by ear.")
    if not source.notes:
        report.append("No Vocal melody found. Connect abc_repaired; abc_karaoke normally has only Vocal rests.")
    report.append("ABC timing is validated. Separate audio generations still need a phrasing/alignment check.")
    return (*outputs, seconds, "\n".join(report))


class HZ3_YuE2_VocalHarmony:
    CATEGORY = "HZ3 YuE2"
    FUNCTION = "arrange"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("abc_lead", "abc_tenor", "abc_baritone", "abc_bass", "duration_seconds", "report")
    DESCRIPTION = "Create three vocal harmonies from MixMash abc_repaired, with identical score timelines. Local, deterministic; no model required."

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "score_abc": ("STRING", {"forceInput": True, "tooltip": "Connect MixMash abc_repaired, including its original chord symbols."}),
            "arrangement": (list(ARRANGEMENTS), {"default": "close_harmony"}),
            "active_sections": ("STRING", {"default": "", "tooltip": "Blank/all: whole song. Comma-separated section names; chorus matches chorus 1 and chorus 2. Lead is always preserved."}),
        }
        for name, (low, high) in zip(PARTS, DEFAULT_RANGES):
            inputs[f"{name}_low"] = ("INT", {"default": low, "min": 0, "max": 127, "tooltip": "Lowest MIDI note; C4 = 60."})
            inputs[f"{name}_high"] = ("INT", {"default": high, "min": 0, "max": 127, "tooltip": "Highest MIDI note; C4 = 60."})
        return {"required": inputs}

    def arrange(self, score_abc, arrangement="close_harmony", active_sections="",
                tenor_low=60, tenor_high=84, baritone_low=48, baritone_high=76, bass_low=36, bass_high=64):
        return harmonize(score_abc, arrangement, active_sections,
                         ((tenor_low, tenor_high), (baritone_low, baritone_high), (bass_low, bass_high)))


NODE_CLASS_MAPPINGS = {"HZ3_YuE2_VocalHarmony": HZ3_YuE2_VocalHarmony}
NODE_DISPLAY_NAME_MAPPINGS = {"HZ3_YuE2_VocalHarmony": "HZ3 YuE2 · Vocal Harmony"}
