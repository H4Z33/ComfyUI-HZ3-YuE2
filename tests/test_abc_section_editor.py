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

    def test_saved_range_state_migrates_to_section_start_boxes(self):
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
        report = json.loads(result["section_report"])
        self.assertEqual([section["start_bar"] for section in report["sections"]], [0, 1])
        self.assertEqual([section["lyrics"] for section in report["sections"]],
                         ["first lyric\nsecond lyric", "last lyric"])

    def test_section_start_and_editable_box_text_are_saved_independently(self):
        source = HEADER + '''% intro
V: Vocal
"C"E32-|E32|G32|A32|
V: Ins
Z4|
'''
        first = editor.viewer_data(source, "[Intro]\nopening\n[Verse]\nold verse\n")
        state = {
            "source_hash": first["source_hash"],
            "editor_version": 2,
            "sections": [
                {"id": "intro", "name": "Intro", "start_bar": 0, "lyrics": "opening"},
                {"id": "verse", "name": "Verse", "start_bar": 2, "lyrics": "new verse\nsecond line"},
            ],
        }
        result = editor.viewer_data(source, "[Intro]\nopening\n[Verse]\nold verse\n", json.dumps(state))

        original = abc.parse(source)
        rebuilt = abc.parse(result["edited_abc"])
        self.assertEqual(rebuilt.voices["Vocal"].notes, original.voices["Vocal"].notes)
        self.assertEqual([section["start_bar"] for section in result["sections"]], [0, 2])
        self.assertEqual(result["sections"][1]["lyrics"], "new verse\nsecond line")
        self.assertEqual(result["edited_lyrics"], "[Intro]\nopening\n[Verse]\nnew verse\nsecond line\n")
        _info, _bars, _directives, markers, *_timing = editor._extract_abc(result["edited_abc"])
        self.assertEqual([(marker["name"], marker["start"]) for marker in markers], [("Intro", 0), ("Verse", 2)])
        reparsed_state = editor.viewer_data(source, "[Intro]\nopening\n[Verse]\nold verse\n", result["editor_state"])
        self.assertEqual(reparsed_state["sections"], result["sections"])

    def test_section_bar_ranges_are_explicit_and_first_section_can_start_later(self):
        source = HEADER + '''% verse
V: Vocal
"C"E32|E32|G32|A32|
V: Ins
z32|z32|z32|z32|
'''
        initial = editor.viewer_data(source, "one\ntwo\n")
        state = {
            "source_hash": initial["source_hash"],
            "editor_version": 2,
            "sections": [
                {"id": "a", "name": "Part A", "start_bar": 1, "end_bar": 2, "lyrics": "one"},
                {"id": "b", "name": "Part B", "start_bar": 3, "end_bar": 3, "lyrics": "two"},
            ],
        }
        result = editor.viewer_data(source, "one\ntwo\n", json.dumps(state))

        self.assertEqual([(section["start_bar"], section["end_bar"]) for section in result["sections"]],
                         [(1, 2), (3, 3)])
        self.assertIn("% Part A", result["edited_abc"])
        self.assertIn("% Part B", result["edited_abc"])
        report = json.loads(result["section_report"])
        self.assertEqual([(section["start_bar"], section["end_bar"]) for section in report["sections"]],
                         [(1, 2), (3, 3)])

    def test_viewer_supplies_resolved_abc_notes_and_timing_for_playback(self):
        source = HEADER + '''% verse
V: Vocal
^F32|G32|
V: Ins
z32|A32|
'''
        result = editor.viewer_data(source)
        parsed = abc.parse(source)
        self.assertEqual(result["tracks"]["Vocal"], [
            {"start": int(start * 256), "pitch": pitch, "duration": int(duration * 256)}
            for start, pitch, duration in parsed.voices["Vocal"].notes
        ])
        self.assertEqual(result["tracks"]["Ins"], [
            {"start": int(start * 256), "pitch": pitch, "duration": int(duration * 256)}
            for start, pitch, duration in parsed.voices["Ins"].notes
        ])
        self.assertEqual(result["bars"][1]["start"], 4 * 256)
        self.assertEqual(result["total_ticks"], 8 * 256)

    def test_loaded_abc_and_lyrics_pair_overrides_connected_inputs(self):
        connected = HEADER + 'V: Vocal\n"C"C32|\nV: Ins\nz32|\n'
        loaded = HEADER + 'V: Vocal\n"G"G32|\nV: Ins\nz32|\n'
        result = editor.HZ3_YuE2_ABCViewer().view(
            connected, lyrics="connected lyric", loaded_abc=loaded, loaded_lyrics="loaded lyric")

        data = result["ui"]["abc_viewer"][0]
        self.assertEqual(data["abc"], loaded)
        self.assertEqual(data["lyrics"], "loaded lyric")
        self.assertEqual(result["result"][1], "[Section 1]\nloaded lyric\n")

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
