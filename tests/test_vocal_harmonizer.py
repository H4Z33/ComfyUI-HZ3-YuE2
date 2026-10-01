"""Run with python -m unittest discover -s tests -v (no ComfyUI/model imports)."""

import importlib
from pathlib import Path
import sys
import types
import unittest

import librosa
import numpy as np


PACKAGE = "harmonizer_test_nodes"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[PACKAGE] = package
harmonizer = importlib.import_module(f"{PACKAGE}.vocal_harmonizer")

SCORE = '''X:1
T:
M:4/4
L:1/32
Q:1/4=60
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% chorus
V: Vocal
"Am"A,32|
V: Ins
Z|
'''
SAMPLE_RATE = 44100


def sung_a3(seconds=2.0):
    """A harmonic-rich A3 (220 Hz) with a little vibrato."""
    time = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    phase = 2 * np.pi * np.cumsum(220 * (1 + 0.003 * np.sin(2 * np.pi * 5 * time))) / SAMPLE_RATE
    wave = sum(np.sin(harmonic * phase) / harmonic for harmonic in range(1, 12))
    return (0.2 * wave).astype(np.float32)


class HarmonizerTests(unittest.TestCase):
    def test_voices_land_on_chord_tones(self):
        voices, report = harmonizer.harmonize_audio(np.stack([sung_a3()] * 2), SAMPLE_RATE, SCORE)
        # Am over a sung A3 (57): C4 above, E3 and C3 below, A2 root, E4 on top.
        expected = {"tenor": 60, "baritone": 52, "low": 48, "bass": 45, "countertenor": 64}
        self.assertEqual(set(voices), set(expected))
        for name, target in expected.items():
            voice = voices[name]
            self.assertEqual(voice.shape, (2 * SAMPLE_RATE,))
            middle = voice[int(0.5 * SAMPLE_RATE):int(1.5 * SAMPLE_RATE)]
            f0, voiced, _ = librosa.pyin(middle, sr=SAMPLE_RATE, fmin=60, fmax=800, frame_length=4096)
            self.assertAlmostEqual(float(np.nanmedian(librosa.hz_to_midi(f0[voiced]))), target, delta=0.5, msg=name)
        self.assertIn("tenor:", report)

    def test_time_warp_follows_a_late_singer(self):
        # One-second notes at 60 BPM; the singer comes in 1.5 s late, an octave up.
        melody = [57, 60, 64, 62, 59, 65, 69, 67, 64, 60] * 4
        notes = [[index, pitch, 1] for index, pitch in enumerate(melody)]
        times = np.arange(int(45 / harmonizer.FRAME_SECONDS)) * harmonizer.FRAME_SECONDS - 1.5
        f0 = np.array([librosa.midi_to_hz(melody[int(time)] + 12) if 0 <= time < len(melody) else np.nan for time in times])
        warp = harmonizer._time_warp(f0, notes, 60)
        for score_time in (10, 20, 30):
            self.assertAlmostEqual(float(np.interp(score_time, *warp[:2])), score_time + 1.5, delta=0.2)

    def test_silence_returns_silent_tracks(self):
        voices, report = harmonizer.harmonize_audio(np.zeros((2, SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE, SCORE)
        self.assertTrue(all(not voice.any() for voice in voices.values()))
        self.assertIn("No sung notes", report)


if __name__ == "__main__":
    unittest.main()
