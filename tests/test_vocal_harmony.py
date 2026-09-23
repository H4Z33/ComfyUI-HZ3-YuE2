"""Run with python -m unittest discover -s tests -v (no ComfyUI/model imports)."""

import importlib
from fractions import Fraction
from pathlib import Path
import sys
import types
import unittest


PACKAGE = "harmony_test_nodes"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[PACKAGE] = package
harmony = importlib.import_module(f"{PACKAGE}.vocal_harmony")
abc = importlib.import_module(f"{PACKAGE}.abc_score")

HEADER = '''X:1
T:
M:4/4
L:1/32
Q:1/4=72
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
'''


def score(vocal, ins="Z|", section="verse 1"):
    return HEADER + f"% {section}\nV: Vocal\n{vocal}\nV: Ins\n{ins}\n"


class HarmonyTests(unittest.TestCase):
    def check_timeline(self, text, **kwargs):
        result = harmony.harmonize(text, **kwargs)
        source = abc.parse(text)
        for output in result[:4]:
            parsed = abc.parse(output)
            for part in parsed.voices.values():
                self.assertEqual(part.bars, source.voices["Vocal"].bars)
                self.assertEqual(part.keys, source.voices["Vocal"].keys)
                self.assertEqual(part.time, source.voices["Vocal"].time)
            self.assertFalse(parsed.voices["Ins"].notes)
            self.assertFalse(parsed.voices["Vocal"].chords)
            self.assertEqual([line for line in output.splitlines() if line.startswith("% ")],
                             [line for line in text.splitlines() if line.startswith("% ")])
        self.assertEqual(abc.parse(result[0]).voices["Vocal"].notes, source.voices["Vocal"].notes)
        self.assertEqual(result[4], float(source.voices["Vocal"].time * 60 / source.bpm))
        return result

    def test_complete_quartet_and_determinism(self):
        text = score('"Cmaj7"E8G8B8c8|')
        result = self.check_timeline(text, arrangement="barbershop_inspired")
        self.assertEqual(result, harmony.harmonize(text, "barbershop_inspired"))
        parts = [abc.parse(output).voices["Vocal"].notes for output in result[:4]]
        for lead, tenor, bari, bass in zip(*parts):
            self.assertEqual({note[1] % 12 for note in (lead, tenor, bari, bass)}, {0, 4, 7, 11})
            self.assertGreater(tenor[1], max(lead[1], bari[1]))
            self.assertLess(bass[1], min(lead[1], bari[1]))
        self.assertIn("inferred chords: 0", result[5])

    def test_all_native_chord_qualities_and_enharmonics(self):
        self.assertEqual(set(harmony.CHORD_INTERVALS), set(abc.QUALITIES))
        for quality in abc.QUALITIES:
            with self.subTest(quality=quality):
                self.check_timeline(score(f'"Db{quality}"_A32|'))
        for symbol in ("D#m/F#", "A#maj7", "F7/Eb", "Cbbm", "F##7"):
            with self.subTest(chord=symbol):
                self.check_timeline(score(f'"{symbol}"G32|'))

    def test_slash_bass(self):
        result = self.check_timeline(score('"C/E"G32|'))
        bass = abc.parse(result[3]).voices["Vocal"].notes
        self.assertEqual(bass[0][1] % 12, 4)

    def test_rests_intro_and_syllable_attacks(self):
        text = score("Z4|", "C32|Z3|", "intro")
        text += '% verse 1\nV: Vocal\n"C"z8E4E4z8G8|\nV: Ins\nZ|\n'
        result = self.check_timeline(text)
        expected = [(time, duration) for time, _, duration in abc.parse(text).voices["Vocal"].notes]
        for output in result[1:4]:
            self.assertEqual([(time, duration) for time, _, duration in abc.parse(output).voices["Vocal"].notes], expected)

    def test_tie_and_accidental_across_barline(self):
        text = score('"F#"^F32-|F8=F8^F8F8|', "Z2|")
        result = self.check_timeline(text)
        self.assertEqual(abc.parse(result[0]).voices["Vocal"].notes[0], [Fraction(0), 66, Fraction(5)])
        for output in result[1:4]:
            self.assertEqual(len(abc.parse(output).voices["Vocal"].notes), 4)

    def test_chord_change_during_tied_note(self):
        text = score('"C"E16-"F#"E16|')
        result = self.check_timeline(text)
        for output in result[1:4]:
            notes = abc.parse(output).voices["Vocal"].notes
            self.assertEqual(notes[0][2], Fraction(2))
            self.assertEqual(notes[1][0], Fraction(2))
            self.assertIn(notes[0][1] % 12, {0, 4, 7})
            self.assertIn(notes[1][1] % 12, {6, 10, 1})

    def test_key_and_meter_changes(self):
        text = score('"C"E16[K:G]"G"F16|', "z16[K:G]z16|")
        text += '% chorus 2\nV: Vocal\nM:3/4\nK:Bb\n"Bb"B24|\nV: Ins\nM:3/4\nK:Bb\nZ|\n'
        self.check_timeline(text)

    def test_section_selection_cuts_ties_without_changing_lead(self):
        text = score('"C"E32-|')
        text += '% chorus 1\nV: Vocal\nE32-|\nV: Ins\nZ|\n'
        text += '% verse 2\nV: Vocal\nE32|\nV: Ins\nZ|\n'
        result = self.check_timeline(text, active_sections="CHORUS")
        for output in result[1:4]:
            notes = abc.parse(output).voices["Vocal"].notes
            self.assertEqual(len(notes), 1)
            self.assertEqual((notes[0][0], notes[0][2]), (Fraction(4), Fraction(4)))

    def test_equivalent_key_fields_written_differently_in_each_track(self):
        text = HEADER + '% verse\nV: Vocal\n[K:G]"G"F32|\nV: Ins\nK:G\nZ|\n'
        self.check_timeline(text)

    def test_nonstandard_meter_and_unit_with_long_notes(self):
        text = score('"C"E48-E8|', 'Z|').replace('M:4/4', 'M:7/8').replace('L:1/32', 'L:1/64')
        self.check_timeline(text)

    def test_unmatched_sections_and_silent_input(self):
        result = self.check_timeline(score('"C"E32|'), active_sections="bridge")
        self.assertIn("No section matched", result[5])
        for output in result[1:4]:
            self.assertFalse(abc.parse(output).voices["Vocal"].notes)
        result = self.check_timeline(score('"C"z32|', "G32|"))
        for output in result[:4]:
            self.assertFalse(abc.parse(output).voices["Vocal"].notes)

    def test_missing_chords_and_stale_chord_after_modulation(self):
        result = self.check_timeline(score("E8G8B8c8|"))
        self.assertIn("inferred chords: 4", result[5])
        text = score('"C"E16[K:F#]F16|', "z16[K:F#]z16|")
        result = self.check_timeline(text)
        self.assertIn("inferred chords: 1", result[5])

    def test_ranges_and_impossible_voicing_report(self):
        result = self.check_timeline(score('"C"E32|'), ranges=((73, 73), (61, 61), (49, 49)))
        for output, pitch in zip(result[1:4], (73, 61, 49)):
            self.assertEqual(abc.parse(output).voices["Vocal"].notes[0][1], pitch)
        self.assertIn("non-chord compromises: 1", result[5])
        result = self.check_timeline(score('"C"c32|'), ranges=((48, 60), (40, 55), (36, 48)))
        self.assertIn("relaxed voice ordering: 1", result[5])

    def test_node_contract(self):
        node = harmony.HZ3_YuE2_VocalHarmony()
        result = node.arrange(score('"C"E32|'))
        self.assertEqual(len(result), len(node.RETURN_TYPES))
        self.assertEqual(node.RETURN_NAMES[:4], ("abc_lead", "abc_tenor", "abc_baritone", "abc_bass"))
        self.assertIsInstance(result[4], float)
        self.assertIs(harmony.NODE_CLASS_MAPPINGS["HZ3_YuE2_VocalHarmony"], type(node))


if __name__ == "__main__":
    unittest.main()
