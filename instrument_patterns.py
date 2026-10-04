"""Deterministic instrument parts written from a song's ABC: the notes HZ3 Studio's MIDI tracks play.

Every pattern reads the score's bars and chords (the ABC's own, else the chords heard in a render, else one
guessed per bar from the melody) and returns notes in score ticks; the same inputs always give the same notes.
"""

from bisect import bisect_right
from fractions import Fraction

from .abc_score import parse
from .vocal_harmonizer import _guessed_chords, _heard_chords
from .vocal_harmony import _chord

TICKS_PER_QUARTER = 256
PATTERNS = ("bass", "bass_walk", "pad", "comping", "arpeggio", "drums", "melody", "ins")
KICK, SNARE, CLOSED_HAT, CRASH = 36, 38, 42, 49


def _chords(score, heard_abc):
    written = score.voices["Vocal"].chords
    if written:
        return [(Fraction(time), symbol) for time, symbol in written], "ABC"
    if heard_abc.strip():
        heard = _heard_chords(heard_abc)
        if heard:
            return [(Fraction(seconds * score.bpm / 60).limit_denominator(64), symbol) for seconds, symbol in heard], "heard"
    return _guessed_chords(score.voices["Vocal"]), "guessed"


def _near(pitch_class, center):
    """The pitch with that pitch class closest to center."""
    return center + ((pitch_class - center + 6) % 12) - 6


def pattern_notes(abc, pattern, bars, octave=0, velocity=96, heard_abc=""):
    """Notes {start, duration, pitch, velocity} (ticks) of one pattern over the given bar indices, and the chord source."""
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern {pattern!r}; expected one of {', '.join(PATTERNS)}.")
    score = parse(abc.strip() + "\n")
    chords, source = _chords(score, heard_abc)
    chord_times = [time for time, _ in chords]
    shift = 12 * int(octave)
    notes = []

    def add(start, duration, pitch, level=1.0):
        if duration > 0 and 0 <= pitch + shift <= 127:
            notes.append({"start": round(start * TICKS_PER_QUARTER), "duration": max(1, round(duration * TICKS_PER_QUARTER)),
                          "pitch": pitch + shift, "velocity": max(1, min(127, round(velocity * level)))})

    def chord_at(time):
        index = bisect_right(chord_times, time) - 1
        return _chord(chords[index][1]) if index >= 0 else None

    def segments(start, end):
        """(start, end, chord) pieces of [start, end) split at chord changes."""
        cuts = [start] + [time for time in chord_times if start < time < end] + [end]
        return [(a, b, chord_at(a)) for a, b in zip(cuts, cuts[1:])]

    selected = sorted(set(bars))
    previous_bar = None
    for index in selected:
        start, length, (beats, unit) = score.voices["Vocal"].bars[index]
        beat = Fraction(4, unit)
        end = start + length
        if pattern == "drums":
            if previous_bar is None or index != previous_bar + 1:
                add(start, beat, CRASH, 0.8)
            for step in range(int(length / (beat / 2))):
                at = start + step * beat / 2
                on_beat, count = step % 2 == 0, step // 2
                add(at, beat / 4, CLOSED_HAT, 0.55 if on_beat else 0.4)
                if on_beat and count % 2 == 0:
                    add(at, beat / 2, KICK, 1.0)
                if on_beat and count % 2 == 1:
                    add(at, beat / 2, SNARE, 0.9)
        elif pattern in ("melody", "ins"):
            voice = score.voices["Vocal" if pattern == "melody" else "Ins"]
            for note_start, pitch, duration in voice.notes:
                if start <= note_start < end:
                    add(note_start, min(duration, end - note_start), pitch)
        else:
            for a, b, chord in segments(start, end):
                if chord is None:
                    continue
                tones, bass = chord
                if pattern == "bass":
                    root = _near(bass, 36)
                    half = max(beat, length / 2)
                    at = a
                    while at < b:
                        add(at, min(half, b - at) * Fraction(9, 10), root)
                        at += half
                elif pattern == "bass_walk":
                    root = _near(bass, 36)
                    walk = [root, _near(tones[2 % len(tones)], root + 5), _near(tones[1], root + 3), root + 12]
                    at, step = a, 0
                    while at < b:
                        add(at, min(beat, b - at) * Fraction(9, 10), walk[step % len(walk)])
                        at, step = at + beat, step + 1
                elif pattern == "pad":
                    for tone in sorted(_near(tone, 62) for tone in tones):
                        add(a, b - a, tone, 0.7)
                elif pattern == "comping":
                    at = start + beat
                    while at < end:
                        if a <= at < b:
                            for tone in sorted(_near(tone, 62) for tone in tones):
                                add(at, beat / 2, tone, 0.75)
                        at += 2 * beat
                elif pattern == "arpeggio":
                    ladder = sorted(_near(tone, 60) for tone in tones)
                    ladder += [ladder[0] + 12]
                    at, step = a, 0
                    while at < b:
                        add(at, min(beat / 2, b - at), ladder[step % len(ladder)], 0.8)
                        at, step = at + beat / 2, step + 1
        previous_bar = index
    notes.sort(key=lambda note: (note["start"], note["pitch"]))
    return notes, source
