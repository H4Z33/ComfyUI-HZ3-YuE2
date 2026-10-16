"""Run with the ComfyUI Python environment and its root on PYTHONPATH."""

import importlib
from pathlib import Path
import sys
import types
import unittest

import numpy as np

PACKAGE = "studio_test_nodes"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[PACKAGE] = package
studio = importlib.import_module(f"{PACKAGE}.studio")

SCORE = '''X:1
T:
M:4/4
L:1/32
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Dm
% Intro
V: Vocal
"Dm"z32|
V: Ins
d32|
% Verse 1
V: Vocal
"Bb"z32|"Gm"z16"A"z16|
V: Ins
Z|Z|
% Chorus
V: Vocal
"Dm"a8f8g8z8|
V: Ins
Z|
'''
SAMPLE_RATE = 44100


def tone(hz, seconds):
    phase = 2 * np.pi * hz * np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return 0.2 * sum(np.sin(harmonic * phase) / harmonic for harmonic in range(1, 10))


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE))


class SungMelodyTests(unittest.TestCase):
    def test_recording_becomes_the_section_melody(self):
        # 120 BPM, a quarter is 0.5 s: D4, F4, a rest, a half-note A4, a rest, and a G4 that runs past the section.
        wave = np.concatenate([tone(293.66, .5), tone(349.23, .5), silence(.5), tone(440, 1), silence(.5), tone(392, 1.5)])
        abc, notes = studio._sung_melody(SCORE, "Verse 1", np.stack([wave, wave], 1), SAMPLE_RATE)
        self.assertEqual(notes, 4)
        self.assertIn('"Bb"D8F8z8A8-|"Gm"A8z8"A"G16|', abc)
        # The intro and the chorus keep what they had.
        self.assertIn('"Dm"z32|\nV: Ins\nd32|', abc)
        self.assertIn('"Dm"a8f8g8z8|', abc)

    def test_unknown_section_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Bridge"):
            studio._sung_melody(SCORE, "Bridge", np.zeros((SAMPLE_RATE, 2)), SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
