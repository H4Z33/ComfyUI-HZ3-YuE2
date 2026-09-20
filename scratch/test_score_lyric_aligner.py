"""Unit test for HZ3_YuE2_ScoreLyricAligner and KaraokeVisualizer Whisper-to-Clean-Lyrics reconciliation."""

import json
import sys
from pathlib import Path

# Add parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from score_lyric_aligner import (
    align_score_and_lyrics,
    HZ3_YuE2_ScoreLyricAligner,
    text_similarity,
    parse_whisper_input,
)
from karaoke_visualizer import HZ3_YuE2_KaraokeVisualizer


SAMPLE_ABC = """X:1
T:Yo Queria
M:4/4
L:1/8
Q:1/4=66
K:Eb
% Intro
V:Vocal
z8 | z8 |
V:Ins
"Ebmaj7" [G,B,D]8 | "Cm7" [C,E,G]8 |
% Verse
V:Vocal
"Ebmaj7" G2 G2 G2 F2 | "Cm7" E2 E2 D2 C2 | "Abmaj7" C2 D2 E2 F2 | "Bb7" G4 F4 |
V:Ins
"Ebmaj7" [G,B,D]8 | "Cm7" [C,E,G]8 | "Abmaj7" [A,,C,E]8 | "Bb7" [B,,D,F]8 |
% Chorus
V:Vocal
"Ebmaj7" B4 B4 | "Cm7" G4 G4 | "Abmaj7" A4 A4 | "Bb7" B8 |
V:Ins
"Ebmaj7" [G,B,D]8 | "Cm7" [C,E,G]8 | "Abmaj7" [A,,C,E]8 | "Bb7" [B,,D,F]8 |
"""

CLEAN_LYRICS = """[Verse]
Yo quería tenerte sola
en mi habitación
y poderte abrazar
hasta el amanecer

[Chorus]
Bailando suave
sintiendo tu piel
cerca de mí
otra vez
"""

# Flawed Whisper transcription (misspellings, dropped accents, slight acoustic drift)
WHISPER_SEGMENTS = json.dumps([
    {"start": 7.30, "end": 10.80, "text": "yo queria tenerte sora"},
    {"start": 11.00, "end": 14.50, "text": "en mi habitasion"},
    {"start": 14.80, "end": 18.20, "text": "y poderte abrasar"},
    {"start": 18.50, "end": 21.90, "text": "asta el amanecer"},
    {"start": 22.20, "end": 25.50, "text": "baylando suave"},
    {"start": 25.80, "end": 29.10, "text": "sintiendo tu piel"},
    {"start": 29.40, "end": 32.70, "text": "serca de mi"},
    {"start": 33.00, "end": 36.30, "text": "otra ves"},
])


def test_similarity():
    print("--- Test: Text Similarity ---")
    s1 = "Yo quería tenerte sola"
    s2 = "yo queria tenerte sora"
    sim = text_similarity(s1, s2)
    print(f"Similarity between '{s1}' and '{s2}': {sim:.3f}")
    assert sim > 0.75, f"Expected similarity > 0.75, got {sim}"
    print("PASS: Text similarity test.")


def test_hybrid_alignment():
    print("\n--- Test: Hybrid (Whisper + ABC) Alignment ---")
    timed_lines, lrc, report = align_score_and_lyrics(
        abc_text=SAMPLE_ABC,
        lyrics_text=CLEAN_LYRICS,
        whisper_input=WHISPER_SEGMENTS,
        alignment_mode="Hybrid (Whisper Audio + ABC Notes)",
        syllable_weighting="ABC Notes",
    )

    print(f"Aligned {len(timed_lines)} lines.")
    assert len(timed_lines) == 8, f"Expected 8 lines, got {len(timed_lines)}"

    # Line 1: Must be clean text
    l1 = timed_lines[0]
    print(f"Line 1: '{l1['text']}' [{l1['start']:.2f}s - {l1['end']:.2f}s]")
    assert l1["text"] == "Yo quería tenerte sola", f"Unexpected text: {l1['text']}"
    assert abs(l1["start"] - 7.30) < 0.1, f"Expected start ~7.30, got {l1['start']}"
    assert abs(l1["end"] - 10.80) < 0.1, f"Expected end ~10.80, got {l1['end']}"

    # Verify words are exact substrings of clean line
    print("Line 1 words:")
    for w in l1["words"]:
        print(f"  word '{w['text']}': {w['start']:.2f}s - {w['end']:.2f}s")
        assert w["text"] in l1["text"], f"Word '{w['text']}' not in line text '{l1['text']}'"
        assert w["start"] < w["end"], f"Invalid word duration: {w}"

    # Check LRC output
    print("\nLRC preview:")
    print("\n".join(lrc.splitlines()[:3]))
    assert "[00:07.30] Yo quería tenerte sola" in lrc, "Expected exact LRC line"
    print("PASS: Hybrid alignment test.")


def test_abc_only_alignment():
    print("\n--- Test: ABC Score Only Alignment ---")
    timed_lines, lrc, report = align_score_and_lyrics(
        abc_text=SAMPLE_ABC,
        lyrics_text=CLEAN_LYRICS,
        whisper_input="",
        alignment_mode="ABC Score Timing",
        syllable_weighting="ABC Notes",
    )

    print(f"Aligned {len(timed_lines)} lines from ABC score.")
    assert len(timed_lines) == 8, f"Expected 8 lines, got {len(timed_lines)}"

    l1 = timed_lines[0]
    print(f"Line 1 from ABC: '{l1['text']}' [{l1['start']:.2f}s - {l1['end']:.2f}s]")
    assert l1["text"] == "Yo quería tenerte sola"
    # Intro has 2 bars of 4/4 at BPM 66 = 2 * (4 * 60 / 66) = 7.27 seconds.
    # So Verse starts around ~7.27s!
    assert 6.5 <= l1["start"] <= 8.0, f"Expected Verse start ~7.27s, got {l1['start']}"
    print("PASS: ABC score only alignment test.")


def test_aligner_node():
    print("\n--- Test: HZ3_YuE2_ScoreLyricAligner Node ---")
    node = HZ3_YuE2_ScoreLyricAligner()
    out = node.align_lyrics(
        lyrics=CLEAN_LYRICS,
        abc=SAMPLE_ABC,
        whisper_segments=WHISPER_SEGMENTS,
    )

    assert "ui" in out and "text" in out["ui"]
    assert "result" in out
    timed_json, lrc_text, report = out["result"]

    data = json.loads(timed_json)
    assert isinstance(data, list) and len(data) == 8
    assert data[0]["text"] == "Yo quería tenerte sola"
    print(f"Node output verified successfully ({len(data)} lines in JSON).")
    print("PASS: Aligner node test.")


def test_visualizer_integration():
    print("\n--- Test: Karaoke Visualizer Timeline Integration ---")
    viz = HZ3_YuE2_KaraokeVisualizer()
    timeline = viz._parse_timeline(
        raw_abc=SAMPLE_ABC,
        lyrics_text=CLEAN_LYRICS,
        total_duration=45.0,
        timed_lyrics=WHISPER_SEGMENTS,  # Pass raw Whisper output
    )

    lines = timeline["lines"]
    assert lines is not None and len(lines) == 8
    # Ensure visualizer is displaying CLEAN text, NOT Whisper's text!
    assert lines[0]["text"] == "Yo quería tenerte sola", f"Visualizer text was not clean: {lines[0]['text']}"
    assert lines[1]["text"] == "en mi habitación"
    print(f"Visualizer lines verified: line 0 = '{lines[0]['text']}', line 1 = '{lines[1]['text']}'")

    # Verify all words can be found by line_text.find
    for line in lines:
        lt = line["text"]
        idx_char = 0
        for w in line["words"]:
            pos = lt.find(w["text"], idx_char)
            assert pos != -1, f"Word '{w['text']}' not found in '{lt}' starting at {idx_char}"
            idx_char = pos + len(w["text"])

    print("PASS: Visualizer timeline integration test (all words found, 100% clean text).")


def test_frame_render():
    print("\n--- Test: Frame Render at t = 8.5s (Karaoke Sweep) ---")
    import numpy as np
    from karaoke_visualizer import HZ3_YuE2_KaraokeVisualizer, THEMES, _get_font
    viz = HZ3_YuE2_KaraokeVisualizer()
    timeline = viz._parse_timeline(
        raw_abc=SAMPLE_ABC,
        lyrics_text=CLEAN_LYRICS,
        total_duration=45.0,
        timed_lyrics=WHISPER_SEGMENTS,
    )
    theme = THEMES["Cyberpunk Neon"]
    font_main = _get_font(38, bold=True)
    font_sub = _get_font(24, bold=False)
    font_hud = _get_font(18, bold=True)
    dummy_spec = np.zeros(64, dtype=np.float32)
    dummy_peaks = np.zeros(64, dtype=np.float32)

    img = viz._render_frame(
        t=8.5,
        timeline=timeline,
        spectrum_vals=dummy_spec,
        peaks=dummy_peaks,
        width=1280,
        height=720,
        theme=theme,
        mode="Spectrum Bars",
        font_main=font_main,
        font_sub=font_sub,
        font_hud=font_hud,
    )
    assert img is not None
    assert img.size == (1280, 720)
    print(f"Frame rendered successfully: {img.size}")
    print("PASS: Frame render test.")


if __name__ == "__main__":
    test_similarity()
    test_hybrid_alignment()
    test_abc_only_alignment()
    test_aligner_node()
    test_visualizer_integration()
    test_frame_render()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
