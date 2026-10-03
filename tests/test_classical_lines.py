"""Run with python -m unittest discover -s tests -v (no ComfyUI/model imports)."""

import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest


PACKAGE = "classical_test_nodes"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[PACKAGE] = package
classical = importlib.import_module(f"{PACKAGE}.classical_lines")
abc = importlib.import_module(f"{PACKAGE}.abc_score")

# A G-major Andante: violin plays a scale over two half-note bass notes per bar.
VIOLIN = "".join(f"<note><pitch><step>{step}</step><octave>{octave}</octave></pitch><duration>1</duration></note>"
                 for step, octave in [("G", 4), ("A", 4), ("B", 4), ("C", 5)] * 1)
CELLO = ("<note><pitch><step>G</step><octave>2</octave></pitch><duration>2</duration></note>"
         "<note><pitch><step>D</step><octave>3</octave></pitch><duration>2</duration></note>")


def musicxml(measures=4):
    def part(identifier, body):
        bars = "".join(
            f"<measure number=\"{index + 1}\">"
            + ("<attributes><divisions>1</divisions><key><fifths>1</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
               "<direction><direction-type><words>Andante</words></direction-type></direction>" if index == 0 else "")
            + body + "</measure>" for index in range(measures))
        return f"<part id=\"{identifier}\">{bars}</part>"
    return ("<?xml version=\"1.0\"?><score-partwise><part-list>"
            "<score-part id=\"P1\"><part-name>Violin 1</part-name></score-part>"
            "<score-part id=\"P2\"><part-name>Violoncello</part-name></score-part></part-list>"
            + part("P1", VIOLIN) + part("P2", CELLO) + "</score-partwise>")


SONG = '''X:1
T:
M:4/4
L:1/32
Q:1/4=80
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% intro
V: Vocal
"C"z32|"F"z32|"G"z32|"C"z32|
V: Ins
Z|Z|Z|Z|
% verse
V: Vocal
"C"c8d8e8f8|"F"f32|"G"g32|"C"c32|
V: Ins
Z|Z|Z|Z|
'''


class ClassicalLinesTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.reference = Path(folder.name) / "quartet.musicxml"
        self.reference.write_text(musicxml(), encoding="utf-8")

    def test_reads_parts_measures_and_slow_passages(self):
        piece = classical.read_musicxml(self.reference)
        self.assertEqual(list(piece["parts"]), ["Violin 1", "Violoncello"])
        self.assertEqual(piece["parts"]["Violin 1"][:2], [[0, 67, 1], [1, 69, 1]])
        self.assertEqual(classical.slow_passages(piece), [(0, 4)])
        phrase = classical.phrases(piece, "Violin 1")[0]
        self.assertEqual((phrase["tonic"], phrase["minor"], phrase["measure"], phrase["last"]), (7, False, 1, 4))

    def test_reads_a_single_track_midi_as_upper_and_bass_lines(self):
        # One track, 480 ticks per quarter: four bars of a G-major quarter-note scale over half-note bass notes.
        events = b""
        for bar in range(4):
            for beat, pitch in enumerate((67, 69, 71, 72)):
                events += (b"\x00\x90" + bytes([pitch, 80]) if beat % 2 else b"\x00\x90" + bytes([pitch, 80]) + b"\x00\x90" + bytes([43 if beat == 0 else 50, 80]))
                events += b"\x83\x60\x80" + bytes([pitch, 0])
                if beat % 2:
                    events += b"\x00\x80" + bytes([43 if beat == 1 else 50, 0])
        track = b"\x00\xff\x58\x04\x04\x02\x18\x08\x00\xff\x59\x02\x01\x00" + events + b"\x00\xff\x2f\x00"
        path = self.reference.with_suffix(".mid")
        path.write_bytes(b"MThd" + (6).to_bytes(4, "big") + b"\x00\x00\x00\x01\x01\xe0" + b"MTrk" + len(track).to_bytes(4, "big") + track)
        piece = classical.read_midi(path)
        self.assertEqual(list(piece["parts"]), ["upper", "bass"])
        self.assertEqual([note[1] for note in piece["parts"]["upper"][:4]], [67, 69, 71, 72])
        self.assertEqual([note[1] for note in piece["parts"]["bass"][:4]], [43, 69, 50, 72])
        self.assertEqual((len(piece["measures"]), piece["measures"][0][1:3]), (4, (4, 1)))
        self.assertEqual(classical.slow_passages(piece), [(0, 4)])
        self.assertEqual([phrase["part"] for phrase in classical.phrases(piece, "bass")], ["bass"])

    def test_writes_a_transposed_line_only_into_the_planned_section(self):
        result, report = classical.add_lines(SONG, classical.build_library([self.reference]), {"intro": "high"})
        song, original = abc.parse(result), abc.parse(SONG)
        self.assertEqual(song.voices["Vocal"].notes, original.voices["Vocal"].notes)
        self.assertEqual(song.voices["Vocal"].chords, original.voices["Vocal"].chords)
        ins = song.voices["Ins"].notes
        self.assertTrue(ins)
        self.assertTrue(all(note[0] < 16 for note in ins))
        # G A B C in G major becomes C D E F in C major (scale degrees 1-4), up to the bar chords.
        self.assertEqual([note[1] % 12 for note in ins[:4]], [0, 2, 4, 5])
        self.assertIn("quartet, Violin 1, m.1-4", report[0])


    def test_reference_melody_becomes_a_theme_in_the_song_key(self):
        reference = SONG.replace("K:C", "K:G").replace('"C"c8d8e8f8|"F"f32|"G"g32|"C"c32|', '"G"G8A8B8c8|"C"c8B8A8G8|"D"F8G8A8B8|"G"G32|')
        themes = classical.abc_phrases(reference, "Referencia (CC0)")
        self.assertEqual([(theme["measure"], theme["tonic"], theme["role"]) for theme in themes], [(5, 7, "theme")])
        result, report = classical.add_lines(SONG, themes, {"verse": "theme"})
        verse = [note for note in abc.parse(result).voices["Ins"].notes if note[0] >= 16]
        # G A B C (degrees 1-4 of G major) arrive as C D E F of C major.
        self.assertEqual([note[1] % 12 for note in verse[:4]], [0, 2, 4, 5])
        self.assertIn("Referencia (CC0), melody, m.5-8", report[0])
        _, filtered = classical.add_lines(SONG, themes, {"verse": "theme:otra"})
        self.assertEqual(filtered, ["verse: no unused theme phrase left."])


if __name__ == "__main__":
    unittest.main()
