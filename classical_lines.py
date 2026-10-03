"""Instrumental lines borrowed from public-domain classical scores.

Reads MusicXML or MIDI scores (string quartets, chorales, guitar studies), finds 4-bar phrases in their slow
passages, and writes them into the `Ins` voice of a native YuE2 ABC: moved to
the song's key by scale degree, strong beats snapped to the song's chords, and
placed in a register away from the singer. YuE2 ABC has one instrumental line,
so this borrows melodic material (intros, counter-lines, endings); the rest of
the accompaniment still comes from the chords and the style.
"""

from bisect import bisect_right
from fractions import Fraction
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET
import zipfile

from .abc_score import NATURAL, parse
from .vocal_harmony import _chord, _layout, _pitch_class, _render_bar


SLOW = re.compile(r"adagio|lento|largo|larghetto|andante|cantabile|notturno|grave", re.I)
TEMPO = re.compile(r"allegr|vivace|presto|moderato|andante|adagio|lento|largo|larghetto|tempo|scherzo|notturno|grave", re.I)
MAJOR, MINOR = (0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10)
TRIADS = [(root, quality, {root, (root + third) % 12, (root + 7) % 12})
          for root in range(12) for quality, third in (("", 4), ("m", 3))]
GRID = Fraction(1, 8)  # 32nd notes, the finest unit of the YuE2 ABC grid


def _quantize(value):
    return round(Fraction(value).limit_denominator(64) / GRID) * GRID


def read_musicxml(path):
    """{"parts": {name: [[onset, midi, duration]]}, "measures": [(start, length, fifths, words)]} in quarters."""
    path = Path(path)
    if path.suffix == ".mxl":
        with zipfile.ZipFile(path) as package:
            name = next(item for item in package.namelist() if item.endswith((".xml", ".musicxml")) and not item.startswith("META-INF"))
            root = ET.fromstring(package.read(name))
    else:
        root = ET.parse(path).getroot()
    names = {part.get("id"): part.findtext("part-name") or part.get("id") for part in root.iter("score-part")}
    parts, measures = {}, []
    for index, part in enumerate(root.findall("part")):
        notes, time, divisions, fifths = [], Fraction(0), 1, 0
        for measure in part.findall("measure"):
            start, cursor, longest, words = time, Fraction(0), Fraction(0), []
            for element in measure:
                if element.tag == "attributes":
                    divisions = int(element.findtext("divisions") or divisions)
                    fifths = int(element.findtext("key/fifths") or fifths)
                elif element.tag == "direction":
                    words += [text.text.strip() for text in element.iter("words") if text.text and text.text.strip()]
                elif element.tag in ("backup", "forward"):
                    step = Fraction(int(element.findtext("duration") or 0), divisions)
                    cursor += -step if element.tag == "backup" else step
                elif element.tag == "note" and element.find("grace") is None:
                    duration = Fraction(int(element.findtext("duration") or 0), divisions)
                    onset = cursor - (notes[-1][3] if element.find("chord") is not None and notes else 0)
                    pitch = element.find("pitch")
                    tied = any(tie.get("type") == "stop" for tie in element.findall("tie"))
                    if pitch is not None:
                        midi = 12 * (int(pitch.findtext("octave")) + 1) + NATURAL[pitch.findtext("step")] + round(float(pitch.findtext("alter") or 0))
                        if tied and notes and notes[-1][1] == midi and notes[-1][0] + notes[-1][2] == start + onset:
                            notes[-1][2] += duration
                        else:
                            notes.append([start + onset, midi, duration, duration])
                    if element.find("chord") is None:
                        cursor += duration
                    longest = max(longest, cursor)
            time = start + longest
            if index == 0:
                measures.append((start, longest, fifths, " ".join(words)))
        parts[names.get(part.get("id"), str(index))] = [note[:3] for note in notes]
    return {"parts": parts, "measures": measures}


def _monophonic(notes, top):
    """One line from a part that may hold chords: the top (or bottom) note of each onset, cut at the next onset."""
    by_onset = {}
    for onset, midi, duration in sorted(notes):
        if onset not in by_onset or (midi > by_onset[onset][1]) == top:
            by_onset[onset] = [onset, midi, duration]
    line = [by_onset[onset] for onset in sorted(by_onset)]
    for note, following in zip(line, line[1:]):
        note[2] = min(note[2], following[0] - note[0])
    return line


def read_midi(path):
    """Same shape as read_musicxml for a standard MIDI file. Each pitched track becomes one line
    (its top notes); a single-track piece (lute, guitar, piano) is split into its upper and bass lines."""
    data = Path(path).read_bytes()
    division = struct.unpack(">H", data[12:14])[0]
    position, tracks, meters, fifths = 14, [], [], None
    while position + 8 <= len(data):
        chunk, length = struct.unpack(">4sI", data[position:position + 8])
        body, position = data[position + 8:position + 8 + length], position + 8 + length
        if chunk != b"MTrk":
            continue
        tick, index, status, active, notes = 0, 0, 0, {}, []
        while index < len(body):
            delta = 0
            while True:
                byte = body[index]; index += 1
                delta = (delta << 7) | (byte & 0x7F)
                if byte < 0x80:
                    break
            tick += delta
            if body[index] >= 0x80:
                status = body[index]; index += 1
            if status == 0xFF:
                kind, size = body[index], body[index + 1]
                value = body[index + 2:index + 2 + size]
                index += 2 + size
                if kind == 0x58:
                    meters.append((tick, Fraction(value[0] * 4, 2 ** value[1])))
                elif kind == 0x59 and fifths is None:
                    fifths = struct.unpack("b", value[:1])[0]
                continue
            if status in (0xF0, 0xF7):
                size = 0
                while True:
                    byte = body[index]; index += 1
                    size = (size << 7) | (byte & 0x7F)
                    if byte < 0x80:
                        break
                index += size
                continue
            kind, channel = status & 0xF0, status & 0x0F
            if kind in (0xC0, 0xD0):
                index += 1
                continue
            first, second = body[index], body[index + 1]
            index += 2
            if channel == 9 or kind not in (0x80, 0x90):
                continue
            if kind == 0x90 and second:
                active.setdefault(first, []).append(tick)
            elif active.get(first):
                start = active[first].pop(0)
                if tick > start:
                    notes.append([Fraction(start, division), first, Fraction(tick - start, division)])
        if notes:
            tracks.append(notes)
    every = [note for track in tracks for note in track]
    end = max(onset + duration for onset, _, duration in every)
    if fifths is None:
        weight = [0.0] * 12
        for _, midi, duration in every:
            weight[midi % 12] += float(duration)
        tonic = max(range(12), key=lambda root: sum(weight[(root + step) % 12] for step in MAJOR))
        fifths = (tonic * 7 + 5) % 12 - 5
    measures, time = [], Fraction(0)
    meters = sorted((Fraction(tick, division), length) for tick, length in meters) or [(Fraction(0), Fraction(4))]
    while time < end:
        length = [length for start, length in meters if start <= time][-1] if meters[0][0] <= time else meters[0][1]
        measures.append((time, length, fifths, ""))
        time += length
    if len(tracks) == 1:
        parts = {"upper": _monophonic(tracks[0], True), "bass": _monophonic(tracks[0], False)}
    else:
        parts = {f"track {number}": _monophonic(track, True) for number, track in enumerate(tracks, 1)}
    return {"parts": parts, "measures": measures}


def slow_passages(piece):
    """(first_measure, end_measure) ranges whose tempo words are slow; a score without tempo words counts whole."""
    marks = [(index, words) for index, (_, _, _, words) in enumerate(piece["measures"]) if TEMPO.search(words)]
    if not marks:
        return [(0, len(piece["measures"]))]
    ranges = []
    for (index, words), (end, _) in zip(marks, marks[1:] + [(len(piece["measures"]), "")]):
        if SLOW.search(words) and not re.search(r"allegr|presto|vivace", words, re.I):
            ranges.append((index, end))
    return ranges


def _chord_of(notes, start, end):
    """Best-fitting triad (root, quality) of everything sounding in [start, end)."""
    weight = [0.0] * 12
    for onset, midi, duration in notes:
        overlap = min(end, onset + duration) - max(start, onset)
        if overlap > 0:
            weight[midi % 12] += float(overlap)
    root, quality, _ = max(TRIADS, key=lambda triad: sum(weight[pc] for pc in triad[2]) - 0.5 * sum(weight) / 12 * 3)
    return root, quality


def phrases(piece, part, length=16):
    """Windows of `length` quarters of `part` in slow duple-meter passages (2/4, 4/4, 2/2 bars summing
    exactly to the window): dicts with notes relative to the window, tonic and mode."""
    measures = piece["measures"]
    every = [note for line in piece["parts"].values() for note in line]
    result = []
    for first, end in slow_passages(piece):
        for index in range(first, end):
            start, last = measures[index][0], index
            while last < end and measures[last][0] + measures[last][1] - start < length and measures[last][1] in (2, 4):
                last += 1
            stop = measures[last][0] + measures[last][1] if last < end else None
            if stop is None or stop - start != length or measures[last][1] not in (2, 4) or (index - first) % 2:
                continue
            line = [[onset - start, midi, min(duration, stop - onset)] for onset, midi, duration in piece["parts"][part] if start <= onset < stop]
            if len(line) < 6 or sum(note[2] for note in line) < length * 3 / 4:
                continue
            fifths = measures[index][2]
            major_tonic = (7 * fifths) % 12
            first_chord = _chord_of(every, start, start + 4)
            last_chord = _chord_of(every, stop - 4, stop)
            minor = (major_tonic + 9) % 12 in (first_chord[0], last_chord[0]) and "m" in (first_chord[1], last_chord[1])
            result.append({"part": part, "measure": index + 1, "last": last + 1, "notes": line, "minor": minor,
                           "tonic": (major_tonic + 9) % 12 if minor else major_tonic})
    return result


def _degree(pitch, tonic, scale):
    """(octave, scale degree, chromatic offset) of a pitch in a key."""
    relative = pitch - tonic
    octave, pc = divmod(relative, 12)
    degree = max(index for index, step in enumerate(scale) if step <= pc)
    return octave, degree, pc - scale[degree]


def fit_line(phrase, tonic, minor, chords, register):
    """Move a phrase into the song's key and chords; `chords` gives the chord tones per beat."""
    source_scale = MINOR if phrase["minor"] else MAJOR
    scale = MINOR if minor else MAJOR
    moved = []
    for onset, pitch, duration in phrase["notes"]:
        octave, degree, offset = _degree(pitch, phrase["tonic"], source_scale)
        moved.append([onset, tonic + 12 * octave + scale[degree] + offset, duration])
    # One octave shift for the whole line, so its contour stays intact inside the register.
    middle = (register[0] + register[1]) / 2
    shift = 12 * round((middle - sum(note[1] for note in moved) / len(moved)) / 12)
    fitted, clashes = [], 0
    for onset, pitch, duration in moved:
        pitch += shift
        tones = chords[min(int(onset), len(chords) - 1)]
        if onset % 2 == 0 and pitch % 12 not in tones:  # beats 1 and 3; passing tones stay elsewhere
            clashes += 1
            pitch = min((candidate for candidate in range(pitch - 2, pitch + 3) if candidate % 12 in tones), key=lambda candidate: abs(candidate - pitch))
        fitted.append([onset, pitch, duration])
    return fitted, clashes


def _beat_chords(score, start, beats, tonic, minor):
    """Chord tones for every beat of the window; the key's tonic triad before the first chord."""
    source = score.voices["Vocal"]
    times = [time for time, _ in source.chords]
    scale = MINOR if minor else MAJOR
    default = {tonic % 12, (tonic + scale[2]) % 12, (tonic + scale[4]) % 12}
    result = []
    for beat in range(beats):
        index = bisect_right(times, start + beat) - 1
        result.append(set(_chord(source.chords[index][1])[0]) if index >= 0 else default)
    return result


def build_library(paths):
    """Every usable phrase of the given MusicXML or MIDI files, with its role: the first part
    ("high", e.g. Violin 1) or an inner/lower part ("low", e.g. viola, cello, a bass line)."""
    library = []
    for path in paths:
        piece = read_midi(path) if Path(path).suffix == ".mid" else read_musicxml(path)
        names = list(piece["parts"])
        for role, wanted in (("high", names[:1]), ("low", names[1:])):
            for part in wanted:
                for phrase in phrases(piece, part):
                    phrase["notes"] = [[float(onset), pitch, float(duration)] for onset, pitch, duration in phrase["notes"]]
                    library.append({**phrase, "role": role, "source": Path(path).stem})
    return library


def abc_phrases(score_abc, source, length=16):
    """Melody windows of a native YuE2 ABC (e.g. SheetSage2's transcription of a reference
    recording) as "theme" phrases, starting at bar lines of its 2/4, 4/4 or 2/2 bars. Both
    lines count: SheetSage2 writes the tune of an instrumental recording into `Ins`."""
    score = parse(score_abc.strip() + "\n")
    key = score.voices["Vocal"].keys[0][1]
    minor = key.endswith("m")
    result = []
    bars = score.voices["Vocal"].bars
    for voice, part in (("Vocal", "melody"), ("Ins", "instrumental")):
        melody = score.voices[voice].notes
        for index, (start, bar_length, _) in enumerate(bars):
            if bar_length not in (2, 4) or index % 2:
                continue
            line = [[float(onset - start), pitch, float(min(duration, start + length - onset))]
                    for onset, pitch, duration in melody if start <= onset < start + length]
            if (len(line) < 6 or line[0][0] >= bar_length or sum(note[2] for note in line) < length / 2
                    or start + length > score.voices["Vocal"].time):
                continue
            result.append({"part": part, "measure": index + 1, "last": index + int(length / bar_length),
                           "notes": line, "minor": minor, "tonic": _pitch_class(key[:-1] if minor else key),
                           "role": "theme", "source": source})
    return result


def load_library(folder):
    """Phrases of every score under `folder`, kept in folder/phrases.json until a score changes."""
    folder = Path(folder)
    scores = sorted(path for path in folder.rglob("*") if path.suffix in (".mxl", ".musicxml", ".mid"))
    index = folder / "phrases.json"
    if index.is_file() and all(path.stat().st_mtime <= index.stat().st_mtime for path in scores):
        stored = json.loads(index.read_text(encoding="utf-8"))
        if stored["scores"] == [str(path.relative_to(folder)) for path in scores]:
            return stored["phrases"]
    library = build_library(scores)
    index.write_text(json.dumps({"scores": [str(path.relative_to(folder)) for path in scores], "phrases": library}), encoding="utf-8")
    return library


def add_lines(score_abc, library, plan, register_high=(72, 88), register_low=(48, 62)):
    """Write borrowed phrases into the Ins voice.

    plan: {section name: "high" | "low" | "theme" | "theme:<reference name part>"}; "high" suits intros, endings and choruses
    (first violin), "low" a counter-line under the voice (viola / cello), "theme" a melody
    from an analyzed reference recording (see abc_phrases).
    Returns (abc, report lines).
    """
    score = parse(score_abc.strip() + "\n")
    groups, sections = _layout(score)
    section_starts = [time for time, _ in sections]
    key = score.voices["Vocal"].keys[0][1]
    minor = key.endswith("m")
    tonic = _pitch_class(key[:-1] if minor else key)
    candidates = {"high": [], "low": [], "theme": []}
    for phrase in library:
        candidates[phrase["role"]].append(phrase)
    ins = [list(note) for note in score.voices["Ins"].notes]
    used, report = set(), []
    for index, (start, name) in enumerate(sections):
        role = plan.get(name.strip().casefold())
        if not role:
            continue
        # "theme:habanera" narrows the themes to references whose name contains that text.
        role, _, wanted = role.partition(":")
        end = section_starts[index + 1] if index + 1 < len(sections) else score.voices["Vocal"].time
        bar_length = Fraction(score.voices["Vocal"].meter[0] * 4, score.voices["Vocal"].meter[1])
        if bar_length != 4:
            report.append(f"{name}: skipped, only 4/4 songs take 4/4 phrases.")
            continue
        register = register_low if role == "low" else register_high
        ins = [note for note in ins if not start <= note[0] < end]
        cursor = start
        while cursor + 16 <= end:
            chords = _beat_chords(score, cursor, 16, tonic, minor)
            sources = {source for source, _ in used}
            best = None
            for phrase in candidates[role]:
                if wanted and wanted.casefold() not in phrase["source"].casefold():
                    continue
                if any((phrase["source"], measure) in used for measure in range(phrase["measure"], phrase["last"] + 1)):
                    continue
                fitted, clashes = fit_line(phrase, tonic, minor, chords, register)
                pitches = [note[1] for note in fitted]
                if max(pitches) - min(pitches) > 17 or min(pitches) < register[0] - 5 or max(pitches) > register[1] + 5:
                    continue
                # Singable lines over figuration: few notes per beat, mostly steps, and new pieces first.
                density = max(0, len(fitted) / 16 - 2)
                leaps = sum(abs(b[1] - a[1]) > 5 for a, b in zip(fitted, fitted[1:])) / len(fitted)
                cost = 2 * clashes + 3 * density + 4 * leaps + 2 * (phrase["source"] in sources)
                if role == "theme":
                    cost += 0.05 * phrase["measure"]  # a work states its theme early: prefer first statements
                if best is None or cost < best[0]:
                    best = (cost, fitted, phrase, clashes)
            if best is None:
                report.append(f"{name}: no unused {role} phrase left.")
                break
            _, fitted, phrase, clashes = best
            used.update((phrase["source"], measure) for measure in range(phrase["measure"], phrase["last"] + 1))
            for onset, pitch, duration in fitted:
                onset, duration = _quantize(onset), _quantize(duration)
                if duration > 0 and onset < 16:
                    ins.append([cursor + onset, pitch, min(duration, 16 - onset)])
            report.append(f"{name} bars at {float(cursor) / 4 + 1:.0f}: {phrase['source']}, {phrase['part']}, "
                          f"m.{phrase['measure']}-{phrase['last']} ({clashes} strong-beat notes adjusted)")
            cursor += 16
    ins.sort()
    # Overlapping notes would not fit one line: cut each note at the next onset.
    for current, following in zip(ins, ins[1:]):
        current[2] = min(current[2], following[0] - current[0])
    ins = [note for note in ins if note[2] > 0]
    lines = score.text.splitlines()
    for line_index, voice_name, bars in groups:
        if voice_name == "Ins":
            lines[line_index] = "|".join(_render_bar(ins, bar_start, length, keys, score.unit, score.voices["Ins"].keys)
                                         for bar_start, length, keys in bars) + "|"
    result = "\n".join(lines) + "\n"
    parse(result)
    return result, report
