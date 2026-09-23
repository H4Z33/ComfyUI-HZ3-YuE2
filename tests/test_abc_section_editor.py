"""Section-editor tests; run with python -m unittest discover -s tests -v."""

import importlib
import json
from pathlib import Path
import sys
import types
import unittest


PACKAGE = "section_editor_test_nodes"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[PACKAGE] = package
editor = importlib.import_module(f"{PACKAGE}.abc_viewer")
abc = importlib.import_module(f"{PACKAGE}.abc_score")
separator = importlib.import_module(f"{PACKAGE}.abc_separator")

HEADER = '''X:1
T:
M:4/4
L:1/32
Q:1/4=72
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
'''


class ABCSectionEditorTests(unittest.TestCase):
    def test_abc_separator_keeps_score_grid_and_splits_the_voices(self):
        source = HEADER + '''% verse
V: Vocal
"C"E32|
V: Ins
G32|
'''
        instrumental, sections, vocals, _report = separator.separate_abc(source)
        original = abc.parse(source)
        instrumental_score = abc.parse(instrumental)
        section_score = abc.parse(sections)
        vocal_score = abc.parse(vocals)

        self.assertEqual(instrumental_score.voices["Ins"].notes, original.voices["Ins"].notes)
        self.assertFalse(instrumental_score.voices["Vocal"].notes)
        self.assertFalse(section_score.voices["Ins"].notes)
        self.assertFalse(section_score.voices["Vocal"].notes)
        self.assertEqual(vocal_score.voices["Vocal"].notes, original.voices["Vocal"].notes)
        self.assertFalse(vocal_score.voices["Ins"].notes)
        for output in (instrumental, sections, vocals):
            self.assertEqual(abc.parse(output).voices["Vocal"].bars, original.voices["Vocal"].bars)
            self.assertIn("% verse", output)

    def test_sections_can_be_moved_independently_and_output_stays_valid(self):
        source = HEADER + '''% intro
V: Vocal
"C"E32-|
V: Ins
Z|
V: Vocal
E32|G32|A32|
V: Ins
Z3|
'''
        lyrics = "[Verse 1]\nfirst lyric\nsecond lyric\n[Chorus]\nlast lyric\n"
        initial = editor.viewer_data(source, lyrics)
        state = {
            "source_hash": initial["source_hash"],
            "lyric_lines": ["first lyric", "second lyric", "last lyric"],
            "sections": [
                {"id": "one", "name": "Intro", "lyrics_end": 2, "abc_end": 1},
                {"id": "two", "name": "Verse", "lyrics_end": 3, "abc_end": 4},
            ],
        }
        result = editor.viewer_data(source, lyrics, json.dumps(state))

        original = abc.parse(source)
        rebuilt = abc.parse(result["edited_abc"])
        self.assertEqual(original.voices["Vocal"].bars, rebuilt.voices["Vocal"].bars)
        self.assertEqual(original.voices["Vocal"].notes, rebuilt.voices["Vocal"].notes)
        self.assertEqual(original.voices["Ins"].bars, rebuilt.voices["Ins"].bars)
        self.assertIn("% Intro", result["edited_abc"])
        self.assertIn("% Verse", result["edited_abc"])
        self.assertEqual(result["edited_lyrics"], "[Intro]\nfirst lyric\nsecond lyric\n[Verse]\nlast lyric\n")
        self.assertIn('"lyrics_lines": {\n        "start": 0,\n        "end": 2', result["section_report"])
        self.assertEqual(result["ranges"]["abc"], [{"start": 0, "end": 1}, {"start": 1, "end": 4}])

    def test_can_split_inside_original_group_and_preserve_meter_changes(self):
        source = HEADER + '''% intro
V: Vocal
z32|z32|
V: Ins
Z2|
V: Vocal
M:3/4
K:G
G24|
V: Ins
M:3/4
K:G
Z|
'''
        result = editor.viewer_data(source, "line one\nline two\n")
        state = json.loads(result["editor_state"])
        state["sections"] = [
            {"id": "a", "name": "Part A", "lyrics_end": 1, "abc_end": 1},
            {"id": "b", "name": "Part B", "lyrics_end": 2, "abc_end": 3},
        ]
        result = editor.viewer_data(source, "line one\nline two\n", json.dumps(state))
        rebuilt = abc.parse(result["edited_abc"])
        self.assertEqual(rebuilt.voices["Vocal"].bars, abc.parse(source).voices["Vocal"].bars)
        self.assertEqual(rebuilt.voices["Vocal"].keys, abc.parse(source).voices["Vocal"].keys)
        self.assertIn("M:3/4", result["edited_abc"])
        self.assertIn("K:G", result["edited_abc"])

    def test_added_sections_can_be_empty_on_one_side(self):
        source = HEADER + "% Verse\nV: Vocal\nz32|z32|\nV: Ins\nZ2|\n"
        result = editor.viewer_data(source, "one lyric\n")
        state = json.loads(result["editor_state"])
        state["sections"] = [
            {"id": "a", "name": "Lyric", "lyrics_end": 0, "abc_end": 1},
            {"id": "b", "name": "Instrumental", "lyrics_end": 1, "abc_end": 2},
        ]
        result = editor.viewer_data(source, "one lyric\n", json.dumps(state))
        self.assertEqual(abc.parse(result["edited_abc"]).voices["Vocal"].bars,
                         abc.parse(source).voices["Vocal"].bars)
        self.assertEqual(result["edited_lyrics"], "[Lyric]\n[Instrumental]\none lyric\n")


if __name__ == "__main__":
    unittest.main()
