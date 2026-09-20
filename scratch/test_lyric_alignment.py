"""Isolated tests for the lyrics alignment library used by HZ3_YuE2_KaraokeVisualizer.

Run with the ComfyUI venv python:  python scratch/test_lyric_alignment.py

The forced-alignment tests build synthetic CTC emissions on the MMS_FA alphabet, so they
exercise the real torchaudio Viterbi path, the star (wildcard) handling, unalignable words,
and truncated-audio trimming without downloading the 1.2 GB acoustic model.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import score_lyric_aligner as A  # noqa: E402

from torchaudio.pipelines import MMS_FA  # noqa: E402

DICT = MMS_FA.get_dict(star=None)
FPS = int(round(1.0 / A.FA_FRAME_SECONDS))

SAMPLE_ABC = """X:1
T:
M:4/4
L:1/16
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% intro
V: Vocal
z16|z16|
V: Ins
"C"C4E4G4E4|"G"D4G4B4G4|
% verse
V: Vocal
"C"C2D2E2F2G2A2G2F2|"F"z4A2A2G2F2E2D2|"C"C2D2E2F2G2A2G2F2|"G"z4B2B2A2G2F2E2|
V: Ins
Z4|
% chorus
V: Vocal
"C"c4c4d4c4|"F"A4A4G4F4|"C"c4c4d4c4|"G"B4B4A4G4|
V: Ins
Z4|
"""

LYRICS = """[Intro]

[Verse 1]
Me preguntaron, que si aún te extraño
sin titubear, les contesté que sí.

[Chorus]
Y te vi con él, y de la mano.
Un beso te dio y tú le correspondías.
"""


def synth_emission(clean_lines, seconds, sing_start=6.0, adlib_between_sections=True):
    """Blank everywhere; 60 ms per character where each word is 'sung'; optional ad-lib."""
    units = A.build_alignment_units(clean_lines)
    T = int(seconds * FPS)
    C = len(DICT)
    logp = torch.full((T, C), math.log(0.002))
    logp[:, 0] = math.log(0.95)
    plan = {}
    t = int(sing_start * FPS)

    def sing(chars):
        nonlocal t
        for ch in chars:
            logp[t:t + 3, :] = math.log(0.002)
            logp[t:t + 3, DICT[ch]] = math.log(0.9)
            t += 3
            logp[t:t + 1, 0] = math.log(0.95)
            t += 1

    for i, u in enumerate(units):
        if u["kind"] != "word":
            continue
        if u["word"] == 0 and u["line"] > 0:
            if clean_lines[u["line"]]["section_index"] != clean_lines[u["line"] - 1]["section_index"]:
                t += int(1.5 * FPS)
                if adlib_between_sections:
                    sing("ohohohoh")  # sung but absent from the lyrics
                t += int(3.0 * FPS)
            else:
                t += int(0.6 * FPS)
        s = t
        sing(u["rom"])
        plan[i] = (s, t)
        t += int(0.12 * FPS)
    return logp, units, plan, t / FPS


def test_parse_and_romanize():
    cl = A.parse_clean_lyrics(LYRICS)
    assert [l["section"] for l in cl] == ["Verse 1", "Verse 1", "Chorus", "Chorus"], cl
    assert cl[0]["section_index"] == 2 and cl[2]["section_index"] == 3
    assert A.romanize_word("extraño") == "extrano"
    assert A.romanize_word("Después,") == "despues"
    assert A.romanize_word("don’t") == "don't"
    assert A.romanize_word("123") == ""
    assert A.romanize_word("¡Vamos!") == "vamos"
    print("ok  parse_clean_lyrics / romanize_word")


def test_score_timeline_and_nominal():
    tl = A.extract_score_timeline(SAMPLE_ABC, LYRICS)
    assert tl["parsed"], tl
    assert abs(tl["score_seconds"] - 20.0) < 1e-6, tl["score_seconds"]
    assert [s["norm"] for s in tl["sections"]] == ["intro", "verse", "chorus"]
    cl = A.parse_clean_lyrics(LYRICS)
    lines = A.align_lines_by_score(cl, tl)
    assert len(lines) == 4
    assert 4.0 <= lines[0]["start"] < 8.0, lines[0]
    assert 12.0 <= lines[2]["start"] < 16.0, lines[2]
    assert all(l["words"] for l in lines)
    # Two lyric verses must share one long ABC verse instead of jumping to a later section.
    two_verses = LYRICS.replace("[Chorus]", "[Verse 2]")
    cl2 = A.parse_clean_lyrics(two_verses)
    lines2 = A.align_lines_by_score(cl2, tl)
    assert lines2[2]["start"] < 12.5, lines2[2]
    # Unparseable ABC must not raise.
    bad = A.extract_score_timeline("X:1\nT:\nM:4/4\nL:1/8\nQ:1/4=90\nK:C\n% verse\nV: Vocal\nC2 D2 E2 F2 | G8 ||\n", LYRICS)
    assert not bad["parsed"] and bad["bpm"] == 90.0
    print("ok  extract_score_timeline / align_lines_by_score")


def test_whisper_alignment():
    cl = A.parse_clean_lyrics(LYRICS)
    tl = A.extract_score_timeline(SAMPLE_ABC, LYRICS)
    segs = json.dumps([
        {"start": 7.3, "end": 10.8, "text": "me preguntaron que si aun te extrano"},
        {"start": 11.0, "end": 14.5, "text": "sin titubear les conteste que si"},
        {"start": 18.0, "end": 21.0, "text": "y te vi con el y de la mano"},
        {"start": 21.5, "end": 25.0, "text": "un beso te dio y tu le correspondias"},
    ])
    lines = A.align_lines_by_whisper(cl, A.parse_whisper_input(segs), tl)
    assert [round(l["start"], 1) for l in lines] == [7.3, 11.0, 18.0, 21.5], lines
    assert lines[0]["text"] == "Me preguntaron, que si aún te extraño"
    lrc = A.make_lrc(lines)
    assert lrc.startswith("[00:07.30] Me preguntaron"), lrc
    print("ok  align_lines_by_whisper / make_lrc")


def test_forced_alignment_synthetic():
    cl = A.parse_clean_lyrics(LYRICS)
    logp, units, plan, sung_end = synth_emission(cl, seconds=40.0)
    A.run_forced_alignment(logp.clone(), units, DICT)
    for i, u in enumerate(units):
        if u["kind"] == "word":
            ps, pe = plan[i]
            assert abs(u["start_frame"] - ps) <= 1, (u["text"], u["start_frame"], ps)
            assert abs(u["end_frame"] - pe) <= 2, (u["text"], u["end_frame"], pe)
            assert u["score"] > 0.8, u
    health = A._alignment_health(cl, units)
    assert health["healthy"] and not health["dead"], health
    lines = A._lines_from_units(cl, units, 40.0)
    assert len(lines) == 4 and all(l["confidence"] > 0.8 for l in lines)
    assert abs(lines[0]["start"] - 6.0) < 0.05, lines[0]
    assert lines[2]["start"] > lines[1]["end"] + 4.0  # the ad-lib was absorbed by the star token
    # Words keep the exact original spelling and cover every word of the line.
    assert [w["text"] for w in lines[0]["words"]] == lines[0]["text"].split()
    print("ok  forced alignment on synthetic emissions (star absorbs ad-lib)")


def test_forced_alignment_unalignable_words():
    lyrics = "[Verse]\n1, 2, 3 vamos ya\nhasta el amanecer 2024\n"
    cl = A.parse_clean_lyrics(lyrics)
    logp, units, plan, _ = synth_emission(cl, seconds=20.0, adlib_between_sections=False)
    A.run_forced_alignment(logp.clone(), units, DICT)
    lines = A._lines_from_units(cl, units, 20.0)
    w0 = lines[0]["words"]
    assert [w["text"] for w in w0] == ["1,", "2,", "3", "vamos", "ya"]
    assert w0[0]["start"] < w0[1]["start"] < w0[2]["start"] < w0[3]["start"] < w0[4]["start"]
    assert w0[2]["end"] <= w0[3]["start"] + 1e-6
    w1 = lines[1]["words"]
    assert w1[-1]["text"] == "2024" and w1[-1]["start"] >= w1[-2]["end"] - 1e-6
    assert lines[1]["end"] >= w1[-1]["end"]
    print("ok  unalignable words are interpolated in order")


def test_truncated_audio_trims_trailing_lines():
    cl = A.parse_clean_lyrics(LYRICS)
    logp, units, plan, _ = synth_emission(cl, seconds=40.0)
    cut_logp = logp[: int(15.0 * FPS)].clone()  # only the verse is inside the audio
    units2 = A.build_alignment_units(cl)
    A.run_forced_alignment(cut_logp, units2, DICT)
    health = A._alignment_health(cl, units2)
    assert not health["healthy"] and health["tail_dead"], health
    assert A._trailing_dead_cut(health) == 2, health

    # End-to-end through forced_align_lines with a fake acoustic model: the prefix search
    # must keep the two sung verse lines and flag the chorus as beyond the audio.
    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return None, None

    def fake_emissions(model, mono, device):
        return cut_logp.clone()

    saved = (A.load_forced_aligner, A.compute_emissions)
    A.load_forced_aligner = lambda: (FakeModel(), DICT)
    A.compute_emissions = fake_emissions
    try:
        mono = np.zeros(int(15.0 * A.FA_SAMPLE_RATE), dtype=np.float32)
        lines, stats = A.forced_align_lines(cl, mono, device="cpu", expected_seconds=40.0)
    finally:
        A.load_forced_aligner, A.compute_emissions = saved
    assert [l.get("beyond_audio", False) for l in lines] == [False, False, True, True], stats
    assert stats["trimmed_lines"] == 2 and stats.get("truncation_search"), stats
    assert lines[0]["confidence"] > 0.8 and lines[1]["confidence"] > 0.8
    assert lines[2]["start"] >= 15.0
    print("ok  lines beyond a truncated take are detected and flagged")


def test_repeated_verse_not_pulled_onto_wrong_copy():
    """Verse text repeated later in the lyrics (verse 2 == verse 3) must not steal the audio
    of the sung copy when the take is shorter than the score."""
    lyrics = """[Verse 1]
Me preguntaron que si aun te extrano
sin titubear les conteste que si
[Verse 2]
y me aconsejan que vaya a buscarte
y honestamente pedirte perdon
[Chorus]
y te vi con el y de la mano
un beso te dio y tu le correspondias
despues te perdiste en sus brazos bailando
aquella cancion que era mi preferida
[Verse 3]
y me aconsejan que vaya a buscarte
y honestamente pedirte perdon
[Chorus 2]
y te vi con el y de la mano
un beso te dio y tu le correspondias
"""
    cl = A.parse_clean_lyrics(lyrics)
    sung = cl[:5]  # verse 1, verse 2 and the first chorus line were generated
    logp, _, _, sung_end = synth_emission(sung, seconds=60.0, adlib_between_sections=False)
    emission = logp[: int((sung_end + 6.0) * FPS)].clone()  # short instrumental tail

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return None, None

    saved = (A.load_forced_aligner, A.compute_emissions)
    A.load_forced_aligner = lambda: (FakeModel(), DICT)
    A.compute_emissions = lambda model, mono, device: emission.clone()
    try:
        mono = np.zeros(int(emission.shape[0] / FPS * A.FA_SAMPLE_RATE), dtype=np.float32)
        lines, stats = A.forced_align_lines(cl, mono, device="cpu", expected_seconds=150.0)
    finally:
        A.load_forced_aligner, A.compute_emissions = saved
    flags = [l.get("beyond_audio", False) for l in lines]
    assert flags == [False] * 5 + [True] * 7, (flags, stats)
    assert all(l["confidence"] > 0.8 for l in lines[:5]), [l["confidence"] for l in lines[:5]]
    assert lines[2]["section"] == "Verse 2" and lines[2]["start"] > lines[1]["end"]
    print("ok  repeated verse text stays on the sung copy (prefix search)")


def test_align_lyrics_orchestrator_fallbacks():
    # No audio and no whisper -> nominal score timing, never raises.
    res = A.align_lyrics(SAMPLE_ABC, LYRICS, mode="auto", audio_seconds=0.0)
    assert res["source"] == "score" and len(res["lines"]) == 4, res["source"]
    assert "Source used: nominal ABC" in res["report"]
    # Whisper only.
    segs = json.dumps([{"start": 7.3, "end": 10.8, "text": "me preguntaron que si aun te extrano"}])
    res = A.align_lyrics(SAMPLE_ABC, LYRICS, whisper_input=segs, mode="Whisper Segments", audio_seconds=30.0)
    assert res["source"] == "whisper"
    assert res["lines"][0]["start"] == 7.3
    # Manual offset shifts everything and never goes negative.
    res = A.align_lyrics(SAMPLE_ABC, LYRICS, whisper_input=segs, mode="whisper", time_offset=-20.0)
    assert res["lines"][0]["start"] == 0.0 and res["lines"][0]["words"][0]["start"] == 0.0
    # Empty lyrics.
    res = A.align_lyrics(SAMPLE_ABC, "[Intro]\n", mode="auto")
    assert res["source"] == "none" and res["lines"] == []
    # Forced alignment requested but no audio connected -> falls back to score with a warning.
    res = A.align_lyrics(SAMPLE_ABC, LYRICS, mode="Forced Alignment (audio)")
    assert res["source"] == "score" and any("no audio" in w for w in res["warnings"]), res["warnings"]
    print("ok  align_lyrics orchestrator fallbacks")


def test_sections_from_lines_and_offset():
    lines = [
        A._make_line({"text": "a b", "section": "Verse 1", "section_index": 1}, 6.0, 9.0, [{"text": "a", "start": 6.0, "end": 7.0}], confidence=0.9),
        A._make_line({"text": "c d", "section": "Verse 1", "section_index": 1}, 9.5, 12.0, [{"text": "c", "start": 9.5, "end": 10.0}], confidence=0.9),
        A._make_line({"text": "e f", "section": "Chorus", "section_index": 2}, 20.0, 23.0, [{"text": "e", "start": 20.0, "end": 21.0}], confidence=0.9),
    ]
    secs = A.derive_sections_from_lines(lines, 30.0)
    names = [s["name"] for s in secs]
    assert names == ["INTRO", "VERSE 1", "INTERLUDE", "CHORUS", "OUTRO"], names
    assert secs[-1]["end"] == 30.0
    nominal = [dict(l, start=l["start"] - 1.0) for l in lines]  # audio runs 1 s late vs the score
    tl = {"vocal_notes": [{"start": 5.0}]}
    off = A.estimate_score_offset(lines, nominal, tl)
    assert off is not None and abs(off - 1.0) < 1e-6, off
    # Disagreeing estimates (first note far from the first word) -> no offset.
    assert A.estimate_score_offset(lines, nominal, {"vocal_notes": [{"start": 0.5}]}) is None
    print("ok  derive_sections_from_lines / estimate_score_offset")


def test_compute_emissions_frame_clock():
    """The windowed emission pass must return exactly ceil(samples / 320) frames."""

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            n = x.shape[-1]
            frames = (n - 400) // 320 + 1
            return torch.zeros((1, frames, 28)), None

    for seconds in (0.7, 29.99, 30.0, 30.01, 65.3):
        mono = np.zeros(int(seconds * A.FA_SAMPLE_RATE), dtype=np.float32)
        em = A.compute_emissions(FakeModel(), mono, "cpu")
        expected = (len(mono) + 319) // 320
        assert em.shape == (expected, 28), (seconds, em.shape, expected)
    print("ok  compute_emissions frame clock")


def test_stray_fragment_repair():
    """A lone character or a tiny word matched seconds away from its phrase is snapped back."""
    units = [
        {"kind": "star"},
        {"kind": "word", "line": 0, "word": 0, "text": "Y", "rom": "y",
         "char_spans": [(100, 101, 0.9)], "start_frame": 100, "end_frame": 101, "score": 0.9, "min_frames": 1},
        {"kind": "word", "line": 0, "word": 1, "text": "pude", "rom": "pude",
         "char_spans": [(300, 301, 0.95), (850, 852, 0.9), (853, 855, 0.9), (856, 858, 0.9)],
         "start_frame": 300, "end_frame": 858, "score": 0.91, "min_frames": 4},
        {"kind": "word", "line": 0, "word": 2, "text": "verte", "rom": "verte",
         "char_spans": [(860, 862, 0.9)] * 5, "start_frame": 860, "end_frame": 875, "score": 0.9, "min_frames": 5},
        {"kind": "star"},
    ]
    A._repair_stray_fragments(units)
    pude, y = units[2], units[1]
    assert pude["start_frame"] == 849 and pude["end_frame"] == 858, pude
    assert y["end_frame"] == pude["start_frame"] - 1 and y["start_frame"] >= pude["start_frame"] - 3, y
    # A genuine melisma (two characters on each side of a long gap) is left alone.
    melisma = {"kind": "word", "line": 1, "word": 0, "text": "amor", "rom": "amor",
               "char_spans": [(10, 12, 0.9), (13, 15, 0.9), (120, 122, 0.9), (123, 125, 0.9)],
               "start_frame": 10, "end_frame": 125, "score": 0.9, "min_frames": 4}
    A._repair_stray_fragments([melisma])
    assert melisma["start_frame"] == 10 and melisma["end_frame"] == 125
    print("ok  stray character / tiny word repair")


def test_annotation_lines_are_skipped():
    cl = A.parse_clean_lyrics("""[Intro]
(instrumental)
[Verse]
hola mundo
(solo de piano)
[x2]
adios
""")
    assert [l["text"] for l in cl] == ["hola mundo", "adios"], cl
    print("ok  parenthesized annotation lines are not lyrics")


if __name__ == "__main__":
    test_parse_and_romanize()
    test_score_timeline_and_nominal()
    test_whisper_alignment()
    test_forced_alignment_synthetic()
    test_forced_alignment_unalignable_words()
    test_truncated_audio_trims_trailing_lines()
    test_repeated_verse_not_pulled_onto_wrong_copy()
    test_align_lyrics_orchestrator_fallbacks()
    test_sections_from_lines_and_offset()
    test_compute_emissions_frame_clock()
    test_stray_fragment_repair()
    test_annotation_lines_are_skipped()
    print("\nALL LIBRARY TESTS PASSED")
