"""Harmony voices re-pitched from the generated lead vocal itself.

The lead is pitch-tracked, every sung note gets chord tones from the ABC
chords, and TD-PSOLA moves the note there. The voices keep the lead's timing,
words and timbre, so they line up by construction.
"""

from bisect import bisect_right

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal
import torch

from .abc_score import parse
from .vocal_harmony import _chord, _pitch_class


VOICES = ("tenor", "baritone", "low", "bass", "countertenor")
TRACK_RATE = 22050
HOP = 256
FRAME_SECONDS = HOP / TRACK_RATE
MAJOR, MINOR = (0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10)
NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
TRIADS = {(4, 7): "", (3, 7): "m", (3, 6): "dim", (4, 8): "aug"}


def _track(mono, sample_rate):
    """Frame-wise f0 in Hz (NaN when unvoiced)."""
    f0, voiced, probability = librosa.pyin(librosa.resample(mono, orig_sr=sample_rate, target_sr=TRACK_RATE), sr=TRACK_RATE,
                                           fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
                                           frame_length=2048, hop_length=HOP)
    return np.where(voiced & (probability > 0.4), f0, np.nan)


def _note_runs(f0, min_seconds=0.09):
    """(start_frame, end_frame, midi_note, mean_midi) for each stable sung note."""
    midi = librosa.hz_to_midi(f0)
    smooth = scipy.signal.medfilt(np.nan_to_num(midi, nan=-100.0), 5)
    runs = []
    start = current = None
    for index in range(len(smooth) + 1):
        note = round(smooth[index]) if index < len(smooth) and smooth[index] > 0 else None
        if start is not None and note != current:
            if (index - start) * FRAME_SECONDS >= min_seconds:
                runs.append((start, index, current, float(np.nanmean(midi[start:index]))))
            start = None
        if start is None and note is not None:
            start, current = index, note
    return runs


def _time_warp(f0, notes, bpm, window=8.0, hop=2.0, max_lag=10.0, step=0.05, floor=0.45):
    """(abc_seconds, audio_seconds, sure) knots mapping the nominal ABC timeline onto the sung vocal.

    Generated singers drift seconds away from the score. Windowed pitch-class
    cross-correlation of the ABC melody against the pitch track (octave-blind, so
    it also fits a line sung an octave away), Viterbi-smoothed over the lag;
    windows without a confident match interpolate between their neighbours and
    are flagged in `sure`.
    """
    seconds = 60 / bpm
    total = max(len(f0) * FRAME_SECONDS, max((float(start + length) * seconds for start, _, length in notes), default=0))
    count = int(total / step) + 1
    pad = int(max_lag / step)
    score = np.zeros((13, count))  # 12 pitch classes + a sounding flag
    for start, pitch, length in notes:
        first = int(float(start) * seconds / step)
        last = max(int(float(start + length) * seconds / step), first + 1)
        score[pitch % 12, first:last] = 1
        score[12, first:last] = 1
    sung = np.zeros((13, count))
    voiced = np.flatnonzero(~np.isnan(f0))
    midi = 69 + 12 * np.log2(f0[voiced] / 440)
    bins = np.minimum((voiced * FRAME_SECONDS / step).astype(int), count - 1)
    low = np.floor(midi).astype(int)
    np.add.at(sung, (low % 12, bins), 1 - (midi - low))
    np.add.at(sung, ((low + 1) % 12, bins), midi - low)
    sung[:12] /= np.maximum(sung[:12].sum(0), 1e-9)
    sung[12, bins] = 1
    score, sung = (scipy.ndimage.gaussian_filter1d(feature, 0.1 / step, axis=1) for feature in (score, sung))
    sung = np.pad(sung, ((0, 0), (pad, pad + int(window / step))))
    centers = np.arange(0, total + hop, hop)
    lags = np.arange(-pad, pad + 1)
    half = int(window / step / 2)
    corr = np.zeros((len(centers), len(lags)))
    for i, center in enumerate(centers):
        lo, hi = max(int(center / step) - half, 0), min(int(center / step) + half, count)
        if hi - lo < half or score[12, lo:hi].mean() < 0.02:
            continue
        reference = score[:, lo:hi] - score[:, lo:hi].mean(1, keepdims=True)
        weight = min(score[12, lo:hi].mean() / 0.15, 1)
        for j, lag in enumerate(lags):
            segment = sung[:, lo + lag + pad:hi + lag + pad]
            segment = segment - segment.mean(1, keepdims=True)
            norm = np.linalg.norm(reference) * np.linalg.norm(segment)
            corr[i, j] = weight * (reference * segment).sum() / norm if norm > 1e-6 else 0
    cost = -np.maximum(corr - floor, 0)
    jump = 0.06 * np.abs(lags[:, None] - lags[None, :]) * step / hop
    total_cost = cost[0].copy()
    back = np.zeros(cost.shape, dtype=int)
    for i in range(1, len(centers)):
        options = total_cost[None, :] + jump
        back[i] = options.argmin(1)
        total_cost = cost[i] + options.min(1)
    path = np.zeros(len(centers), dtype=int)
    path[-1] = total_cost.argmin()
    for i in range(len(centers) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    lag = lags[path] * step
    sure = corr.max(1) > floor + 0.1
    lag = np.interp(centers, centers[sure], lag[sure]) if sure.any() else np.zeros(len(centers))
    return centers, np.maximum.accumulate(centers + lag), sure


def _guessed_chords(voice):
    """For a melody-only score: per bar, the diatonic triad that covers most of the bar's sung time
    (the tonic, subdominant and dominant win ties). A bar without notes keeps the previous chord."""
    key_times = [time for time, _ in voice.keys]
    chords = []
    for start, length, _meter in voice.bars:
        end = start + length
        weight = np.zeros(12)
        for note_start, pitch, duration in voice.notes:
            overlap = min(end, note_start + duration) - max(start, note_start)
            if overlap > 0:
                weight[pitch % 12] += float(overlap)
        if not weight.any():
            continue
        key = voice.keys[bisect_right(key_times, start) - 1][1]
        minor = key.endswith("m")
        root = _pitch_class(key[:-1] if minor else key)
        scale = [(root + step) % 12 for step in (MINOR if minor else MAJOR)]
        triads = [[scale[(degree + step) % 7] for step in (0, 2, 4)] for degree in range(7)]
        best = max(range(7), key=lambda degree: weight[triads[degree]].sum() + 0.5 * weight[triads[degree][0]]
                   + (0.1 * weight.sum() if degree in (0, 3, 4) else 0))
        tones = triads[best]
        chords.append((start, NAMES[tones[0]] + TRIADS[((tones[1] - tones[0]) % 12, (tones[2] - tones[0]) % 12)]))
    return chords


def _heard_chords(heard_abc):
    """(audio seconds, symbol) at each chord change of an ABC transcribed from the same audio."""
    heard = parse(heard_abc.strip() + "\n")
    chords = []
    for time, symbol in heard.voices["Vocal"].chords:
        if not chords or chords[-1][1] != symbol:
            chords.append((float(time) * 60 / heard.bpm, symbol))
    return chords


def _voice_shifts(runs, score, warp, chords):
    """Per voice, (start_frame, end_frame, semitone shift) for every lead note it sings.
    chords: (audio seconds, symbol), already on the sung timeline."""
    source = score.voices["Vocal"]
    seconds = 60 / score.bpm
    chord_times = [time for time, _ in chords]
    key_times = np.interp([float(time) * seconds for time, _ in source.keys], *warp).tolist()
    parts = {name: [] for name in VOICES}
    held_bass = held_counter = None
    for start, end, note, actual in runs:
        middle = (start + end) / 2 * FRAME_SECONDS
        chord_index = bisect_right(chord_times, middle) - 1
        if chord_index < 0:
            continue
        tones, bass_pc = _chord(chords[chord_index][1])
        tones = set(tones)
        if note % 12 in tones:
            pool = [pitch for pitch in range(note - 15, note + 13) if pitch % 12 in tones]
        else:
            # Passing tone: diatonic thirds and the fifth below.
            key = source.keys[bisect_right(key_times, middle) - 1][1]
            minor = key.endswith("m")
            root = _pitch_class(key[:-1] if minor else key)
            scale = [pitch for pitch in range(note - 15, note + 13) if (pitch - root) % 12 in (MINOR if minor else MAJOR)]
            pool = [scale[scale.index(note) + step] for step in (-4, -2, 2)] if note in scale else []
        upper = min((pitch for pitch in pool if pitch >= note + 3), default=None)
        lower = max((pitch for pitch in pool if pitch <= note - 3), default=None)
        low = max((pitch for pitch in pool if lower is not None and pitch <= lower - 3), default=None)
        held_bass = min((pitch for pitch in range(40, 53) if pitch % 12 == bass_pc), key=lambda pitch: abs(pitch - (held_bass or 46)))
        targets = {"tenor": upper if upper is not None and upper <= 72 else None, "baritone": lower, "low": low, "bass": held_bass}
        choices = [pitch for pitch in range(note + 7, min(note + 12, 76) + 1) if pitch % 12 in tones]
        held_counter = min(choices, key=lambda pitch: abs(pitch - (held_counter or pitch))) if choices else None
        targets["countertenor"] = held_counter
        for name, target in targets.items():
            if target is not None:
                parts[name].append((start, end, target - actual))
    return parts


def _psola(mono, period, marks, frame_index, segments, frames):
    """TD-PSOLA: two-period grains from the lead, placed at the shifted period."""
    shift = np.zeros(frames)
    gain = np.zeros(frames)
    for start, end, semitones in segments:
        shift[start:end] = semitones
        gain[start:end] = 1.0
    sung = gain > 0
    if not sung.any():
        return np.zeros_like(mono)
    # Carry the shift through gaps so glides start from the last note.
    shift = scipy.ndimage.uniform_filter1d(np.interp(np.arange(frames), np.flatnonzero(sung), shift[sung]), 3)
    gain = scipy.ndimage.uniform_filter1d(gain, 4)[frame_index]
    ratio = 2 ** (shift[frame_index] / 12)
    length = len(mono)
    out = np.zeros(length)
    position = 0.0
    while position < length:
        index = int(position)
        if gain[index] <= 0:
            position += 64
            continue
        half = int(period[index])
        mark = marks[min(np.searchsorted(marks, index), len(marks) - 1)]
        if half <= mark < length - half and half <= index < length - half:
            # Grains overlap `ratio` times when raising pitch; keep the level steady.
            out[index - half:index + half + 1] += mono[mark - half:mark + half + 1] * np.hanning(2 * half + 1) / max(1.0, ratio[index])
        position += period[index] / ratio[index]
    return (out * gain).astype(np.float32)


def harmonize_audio(waveform, sample_rate, score_abc, heard_abc=""):
    """Return {voice: mono float32 samples} and a short report."""
    mono = waveform.mean(0).astype(np.float32)
    f0 = _track(mono, sample_rate)
    voiced = ~np.isnan(f0)
    if not voiced.any():
        return {name: np.zeros_like(mono) for name in VOICES}, "No sung notes were detected in the vocal."
    score = parse(score_abc.strip() + "\n")
    runs = _note_runs(f0)
    abc_times, audio_times, _ = _time_warp(f0, score.voices["Vocal"].notes, score.bpm)
    # A melody-only score has no chords to harmonize on: take the ones heard in the song itself, or,
    # without a transcription, guess one per bar from the melody.
    written = score.voices["Vocal"].chords
    heard = [] if written or not heard_abc.strip() else _heard_chords(heard_abc)
    if heard:
        chords = heard
    else:
        # Chords sit where the singer actually is, not at the nominal score time.
        timed = written or _guessed_chords(score.voices["Vocal"])
        chords = list(zip(np.interp([float(time) * 60 / score.bpm for time, _ in timed], abc_times, audio_times).tolist(),
                          [symbol for _, symbol in timed]))
    parts = _voice_shifts(runs, score, (abc_times, audio_times), chords)
    frame_index = np.minimum((np.arange(len(mono)) / sample_rate / FRAME_SECONDS).astype(np.int64), len(f0) - 1)
    period = sample_rate / np.interp(np.arange(len(f0)), np.flatnonzero(voiced), f0[voiced])[frame_index]
    marks = []
    position = 0.0
    while position < len(mono):
        marks.append(int(position))
        position += period[int(position)]
    marks = np.array(marks)
    voices = {name: _psola(mono, period, marks, frame_index, parts[name], len(f0)) for name in VOICES}
    report = [f"{len(runs)} sung notes tracked; {np.mean(voiced):.0%} of frames voiced."]
    if heard:
        report.append(f"The ABC has no chord symbols: {len(chords)} chord changes heard in the song.")
    elif not written:
        report.append(f"The ABC has no chord symbols: {len(chords)} chords guessed from the melody, one per bar.")
    report += [f"{name}: {len(parts[name])} notes." for name in VOICES]
    return voices, "\n".join(report)


def tune_audio(waveform, sample_rate, score_abc, amount):
    """The lead pulled toward the ABC melody: each sung note moves `amount` of the way to the written pitch class
    (in the octave it was sung), or, when the singer chose another note, to the nearest semitone.
    Unvoiced sound (consonants, breaths) is left as it was."""
    mono = waveform.mean(0).astype(np.float32)
    f0 = _track(mono, sample_rate)
    voiced = ~np.isnan(f0)
    if not voiced.any():
        return mono, "No sung notes were detected in the vocal."
    score = parse(score_abc.strip() + "\n")
    runs = _note_runs(f0)
    notes = score.voices["Vocal"].notes
    abc_times, audio_times, _ = _time_warp(f0, notes, score.bpm)
    seconds = 60 / score.bpm
    starts = [float(start) * seconds for start, _, _ in notes]
    segments, toward_score = [], 0
    for start, end, _note, actual in runs:
        middle = float(np.interp((start + end) / 2 * FRAME_SECONDS, audio_times, abc_times))
        index = bisect_right(starts, middle) - 1
        target = round(actual)
        if index >= 0 and middle < starts[index] + float(notes[index][2]) * seconds:
            written = actual + ((notes[index][1] - actual + 6) % 12) - 6
            if abs(written - actual) <= 1.5:
                target, toward_score = written, toward_score + 1
        segments.append((start, end, (target - actual) * amount))
    frame_index = np.minimum((np.arange(len(mono)) / sample_rate / FRAME_SECONDS).astype(np.int64), len(f0) - 1)
    period = sample_rate / np.interp(np.arange(len(f0)), np.flatnonzero(voiced), f0[voiced])[frame_index]
    marks = []
    position = 0.0
    while position < len(mono):
        marks.append(int(position))
        position += period[int(position)]
    tuned = _psola(mono, period, np.array(marks), frame_index, segments, len(f0))
    sung = np.zeros(len(f0))
    for start, end, _shift in segments:
        sung[start:end] = 1.0
    gain = scipy.ndimage.uniform_filter1d(sung, 4)[frame_index]
    report = (f"{len(segments)} sung notes tuned {amount:.0%} of the way: {toward_score} to the written ABC note, "
              f"{len(segments) - toward_score} to the nearest semitone.")
    return (tuned + mono * (1 - gain)).astype(np.float32), report


class HZ3_YuE2_VocalTune:
    CATEGORY = "HZ3 YuE2"
    FUNCTION = "tune"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("vocals", "report")
    DESCRIPTION = ("Pitch-correct the separated lead vocal toward the ABC melody with PSOLA (deterministic): each sung note moves "
                   "toward its written pitch, or to the nearest semitone where the singer chose another note.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vocals": ("AUDIO", {"tooltip": "Separated lead vocal of the generated song."}),
                "score_abc": ("STRING", {"forceInput": True, "tooltip": "The ABC used for the generation."}),
                "amount": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "0 leaves the voice as sung, 1 snaps it fully."}),
            },
        }

    def tune(self, vocals, score_abc, amount):
        sample_rate = vocals["sample_rate"]
        tuned, report = tune_audio(vocals["waveform"][0].float().cpu().numpy(), sample_rate, score_abc, amount)
        return {"waveform": torch.from_numpy(np.stack([tuned] * vocals["waveform"].shape[1]))[None], "sample_rate": sample_rate}, report


class HZ3_YuE2_VocalHarmonizer:
    CATEGORY = "HZ3 YuE2"
    FUNCTION = "harmonize"
    RETURN_TYPES = ("AUDIO",) * len(VOICES) + ("STRING",)
    RETURN_NAMES = VOICES + ("report",)
    DESCRIPTION = ("Sing chord-tone harmonies with the lead vocal's own voice: pitch-track the separated lead, "
                   "pick chord tones from the ABC chords, and re-pitch each note with PSOLA. Each voice is a "
                   "separate track: tenor/baritone/low a third or more above/below, bass on the chord root "
                   "(E2-E3), countertenor a fifth to an octave above (up to E5). The ABC must start where the audio starts.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vocals": ("AUDIO", {"tooltip": "Separated lead vocal of the generated song."}),
                "score_abc": ("STRING", {"forceInput": True, "tooltip": "The ABC used for the generation, with chord symbols."}),
            },
            "optional": {
                "heard_abc": ("STRING", {"forceInput": True, "tooltip": "SheetSage2 (full) transcription of the same song, starting where the vocal starts. "
                                         "Its chords are used when score_abc has none."}),
            },
        }

    def harmonize(self, vocals, score_abc, heard_abc=""):
        sample_rate = vocals["sample_rate"]
        voices, report = harmonize_audio(vocals["waveform"][0].float().cpu().numpy(), sample_rate, score_abc, heard_abc)
        tracks = tuple({"waveform": torch.from_numpy(voices[name])[None, None], "sample_rate": sample_rate} for name in VOICES)
        return tracks + (report,)


NODE_CLASS_MAPPINGS = {"HZ3_YuE2_VocalHarmonizer": HZ3_YuE2_VocalHarmonizer, "HZ3_YuE2_VocalTune": HZ3_YuE2_VocalTune}
NODE_DISPLAY_NAME_MAPPINGS = {"HZ3_YuE2_VocalHarmonizer": "HZ3 YuE2 · Vocal Harmonizer", "HZ3_YuE2_VocalTune": "HZ3 YuE2 · Vocal Tune"}
