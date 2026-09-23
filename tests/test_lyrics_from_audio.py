"""Isolated tests for AcoustID/LRCLIB lyric lookup."""

import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import numpy as np

PACKAGE = "lyrics_from_audio_test_nodes"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[PACKAGE] = package
lyrics_node = importlib.import_module(f"{PACKAGE}.lyrics_from_audio")


class LyricsFromAudioTests(unittest.TestCase):
    def test_acoustid_meta_fields_use_documented_space_encoding(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"status":"ok"}'

        with patch.object(lyrics_node, "urlopen", return_value=FakeResponse()) as urlopen:
            lyrics_node._request_json(
                lyrics_node.ACOUSTID_LOOKUP_URL,
                {"meta": "recordings releasegroups"},
            )
        request = urlopen.call_args.args[0]
        query = parse_qs(urlsplit(request.full_url).query)
        self.assertEqual(query["meta"], ["recordings releasegroups"])

    def test_audio_tensor_is_written_to_wav_and_fingerprinted(self):
        waveform = np.zeros((1, 2, 44100), dtype=np.float32)
        completed = types.SimpleNamespace(stdout=json.dumps({"duration": 1, "fingerprint": "abc123"}))
        with patch.object(lyrics_node.subprocess, "run", return_value=completed) as run:
            duration, fingerprint = lyrics_node._audio_to_fingerprint(
                {"waveform": waveform, "sample_rate": 44100}, "fpcalc-test"
            )
        self.assertEqual((duration, fingerprint), (1.0, "abc123"))
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["fpcalc-test", "-json"])
        self.assertTrue(command[2].endswith("input.wav"))

    def test_selects_best_acoustid_recording(self):
        response = {
            "status": "ok",
            "results": [
                {"score": 0.72, "recordings": [{"title": "Song", "artists": [{"name": "Singer"}]}]},
                {"score": 0.93, "recordings": [{
                    "title": "Better Song",
                    "artists": [{"name": "Artist One"}, {"name": "Artist Two"}],
                    "releasegroups": [{"title": "Album"}],
                }]},
            ],
        }
        self.assertEqual(
            lyrics_node._best_recording(response, 0.65),
            (0.93, "Better Song", "Artist One, Artist Two", "Album"),
        )

    def test_rejects_low_confidence_fingerprint_match(self):
        response = {"status": "ok", "results": [{
            "score": 0.41,
            "recordings": [{"title": "Maybe", "artists": [{"name": "Unknown"}]}],
        }]}
        with self.assertRaisesRegex(RuntimeError, "below the 0.70 threshold"):
            lyrics_node._best_recording(response, 0.70)

    def test_prefers_exact_metadata_and_returns_synced_lyrics(self):
        results = [
            {"trackName": "Song", "artistName": "Singer", "albumName": "Other", "duration": 180,
             "plainLyrics": "wrong version", "syncedLyrics": "[00:01]wrong version"},
            {"trackName": "Song", "artistName": "Singer", "albumName": "Album", "duration": 210,
             "plainLyrics": "right version", "syncedLyrics": "[00:01]right version"},
        ]
        self.assertEqual(
            lyrics_node._select_lyrics(results, "Song", "Singer", "Album", 200),
            ("right version", "[00:01]right version", "Song", "Singer", "Album", "210"),
        )

    def test_extracts_plain_lines_from_synced_lrc_when_plain_text_is_missing(self):
        results = [{
            "trackName": "Song", "artistName": "Singer", "albumName": "Album", "duration": 200,
            "syncedLyrics": "[ar:Singer]\n[00:01.20]First line\n[00:04.50][00:08.50]Repeated line",
        }]
        self.assertEqual(
            lyrics_node._select_lyrics(results, "Song", "Singer", "Album", 200)[:2],
            ("First line\nRepeated line", "[ar:Singer]\n[00:01.20]First line\n[00:04.50][00:08.50]Repeated line"),
        )

    def test_node_uses_audio_lookup_and_exposes_plain_lyrics_output(self):
        acoustid_response = {"status": "ok", "results": [{
            "score": 0.98,
            "recordings": [{"title": "Song", "artists": [{"name": "Singer"}]}],
        }]}
        lrclib_response = [{
            "trackName": "Song", "artistName": "Singer", "albumName": "Album", "duration": 200,
            "plainLyrics": "Verse", "syncedLyrics": "[00:01.00]Verse",
        }]
        with patch.object(lyrics_node, "_audio_to_fingerprint", return_value=(200.0, "fingerprint")) as fingerprint:
            with patch.object(lyrics_node, "_request_json", side_effect=[acoustid_response, lrclib_response]) as request:
                result = lyrics_node.HZ3_YuE2_LyricsFromAudio().find_lyrics(
                    {"waveform": object(), "sample_rate": 44100}, "client-id", 0.65
                )
        outputs = result["result"]
        self.assertEqual(outputs[0], "Verse")
        self.assertEqual(outputs[1], "[00:01.00]Verse")
        self.assertEqual(outputs[2:5], ("Song", "Singer", "Album"))
        self.assertEqual(outputs[5], 0.98)
        self.assertIn("Found LRCLIB lyrics (synced)", outputs[6])
        self.assertEqual(result["ui"]["text"], [outputs[6]])
        fingerprint.assert_called_once()
        self.assertEqual(request.call_count, 2)
        self.assertEqual(lyrics_node.NODE_CLASS_MAPPINGS["HZ3_YuE2_LyricsFromAudio"],
                         lyrics_node.HZ3_YuE2_LyricsFromAudio)

    def test_unrecognized_audio_returns_visible_warning_without_failing(self):
        no_match = {"status": "ok", "results": []}
        with patch.object(lyrics_node, "_audio_to_fingerprint", return_value=(200.0, "fingerprint")):
            with patch.object(lyrics_node, "_request_json", return_value=no_match):
                result = lyrics_node.HZ3_YuE2_LyricsFromAudio().find_lyrics(
                    {"waveform": object(), "sample_rate": 44100}, "client-id"
                )
        outputs = result["result"]
        self.assertEqual(outputs[:6], ("", "", "", "", "", 0.0))
        self.assertTrue(outputs[6].startswith("WARNING:"))
        self.assertEqual(result["ui"]["text"], [outputs[6]])

    def test_missing_lrclib_text_returns_visible_warning(self):
        match = {"status": "ok", "results": [{
            "score": 0.98,
            "recordings": [{"title": "Song", "artists": [{"name": "Singer"}]}],
        }]}
        with patch.object(lyrics_node, "_audio_to_fingerprint", return_value=(200.0, "fingerprint")):
            with patch.object(lyrics_node, "_request_json", side_effect=[match, []]):
                result = lyrics_node.HZ3_YuE2_LyricsFromAudio().find_lyrics(
                    {"waveform": object(), "sample_rate": 44100}, "client-id"
                )
        self.assertTrue(result["result"][6].startswith("WARNING:"))
        self.assertIn("No lyrics were found in LRCLIB", result["result"][6])

    def test_requires_client_id(self):
        with self.assertRaisesRegex(ValueError, "client ID"):
            lyrics_node.HZ3_YuE2_LyricsFromAudio().find_lyrics({}, "")


if __name__ == "__main__":
    unittest.main()
