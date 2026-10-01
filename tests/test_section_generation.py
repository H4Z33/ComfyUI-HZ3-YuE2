"""Run with the ComfyUI Python environment and its root on PYTHONPATH."""

import importlib
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from comfy.text_encoders.llama import FixedKV
from comfy.text_encoders.yue2 import ABC_END, ABC_START, CODEC_OFFSET, CODEC_SIZE, CONTEXT, EOD, MUSIC_START, chunk_ranges

PACKAGE = "section_generation_test_nodes"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[PACKAGE] = package
generation = importlib.import_module(f"{PACKAGE}.section_generation")


HEADER = '''X:1
T:
M:4/4
L:1/32
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
'''


ABC = HEADER + '''% intro
V: Vocal
Z|
V: Ins
Z|
% verse 1
V: Vocal
"C"E32|
V: Ins
G32|
'''


ABC_TIED_ACROSS_SECTION = HEADER + '''% intro
V: Vocal
Z|
V: Ins
G32-|
% verse 1
V: Vocal
"C"E32|
V: Ins
G32|
'''

LYRICS = """[Intro]
(instrumental)

[Verse 1]
first line
second line
"""


class FakeSectionTransformer:
    """Each token's logits peak at a codec id derived from the previous token, so
    sampled tokens form a chain that depends on everything before them."""

    def __init__(self, fixed_cache=False):
        self.prefills = []
        self.fixed_kv = fixed_cache

    def init_kv_cache(self, batch, capacity, device, dtype):
        if self.fixed_kv:
            key = torch.zeros((batch, capacity, 1, 1), device=device, dtype=dtype)
            return [FixedKV(key, torch.zeros_like(key), 0,
                            torch.zeros((batch,), device=device, dtype=torch.int64),
                            torch.zeros((batch,), device=device, dtype=torch.int32))]
        return [(torch.zeros((batch, 1, capacity, 1), device=device, dtype=dtype),
                 torch.zeros((batch, 1, capacity, 1), device=device, dtype=dtype), 0)]

    def __call__(self, ids, past_key_values, dtype, attention_mask=None, position_ids=None, decode_buffers=None):
        result = []
        for cache in past_key_values:
            if isinstance(cache, FixedKV):
                end = cache.index + ids.shape[1]
                cache.key[:, cache.index:end].copy_(ids[:, :, None, None].to(cache.key))
                cache.index = end
                result.append(cache)
            else:
                key, value, past_length = cache
                end = past_length + ids.shape[1]
                key[:, :, past_length:end].copy_(ids[:, None, :, None].to(key))
                result.append((key, value, end))
        hidden = ids.to(dtype).unsqueeze(-1).expand(-1, -1, 4)
        return hidden, None, result

    def lm_head(self, hidden):
        logits = torch.zeros((hidden.shape[0], CODEC_OFFSET + CODEC_SIZE), device=hidden.device)
        for row, last in enumerate(hidden[:, 0].tolist()):
            logits[row, CODEC_OFFSET + int(last) % 97] = 1.0
        return logits

    def compute_freqs_cis(self, position_ids, device):
        return torch.zeros((1, 1, 2), device=device)


class FakeSectionModel:
    def __init__(self, fixed_cache=False):
        self.model = FakeSectionTransformer(fixed_cache)
        self.config = types.SimpleNamespace(
            max_position_embeddings=CONTEXT,
            hidden_size=4,
            num_hidden_layers=1,
            num_key_value_heads=1,
            head_dim=2,
        )

    def _prefill(self, prefixes, capacity, dtype):
        self.model.prefills.append([list(prefix) for prefix in prefixes])
        length = max(map(len, prefixes))
        ids = torch.tensor([[0] * (length - len(prefix)) + prefix for prefix in prefixes], dtype=torch.long)
        mask = None
        if any(len(prefix) != length for prefix in prefixes):
            mask = torch.ones((len(prefixes), capacity), dtype=torch.long)
            for index, prefix in enumerate(prefixes):
                mask[index, :length - len(prefix)] = 0
        cache = self.model.init_kv_cache(len(prefixes), capacity, "cpu", dtype)
        output = self.model(ids, past_key_values=cache, dtype=dtype)
        return self.model.lm_head(output[0][:, -1]), output[2], mask

    def _acoustic_conditioning(self, prefix, tokens, dtype):
        self.acoustic_call = (list(prefix), list(tokens))
        ranges = chunk_ranges(len(tokens), len(prefix), self.config.max_position_embeddings)
        total = sum(len(prefix) + end - start + 1 for start, end in ranges)
        context = torch.tensor(prefix + tokens + [0], dtype=dtype).reshape(1, total, 1).expand(1, total, 4).clone()
        chunks = tuple((start, end, 0, total) for start, end in ranges)
        return context, chunks


class FakeClip:
    def __init__(self):
        self.patcher = types.SimpleNamespace(patches={})
        self.calls = []

    def tokenize(self, style, lyrics, cot, seed, abc, max_tokens, temperature, top_p, top_k,
                 repetition_penalty, cfg_scale):
        self.calls.append({"lyrics": lyrics, "abc": abc, "max_tokens": max_tokens})
        return {
            "prefix": [EOD, 42 if style == "same global style" else 43, ABC_START],
            "negative": [EOD],
            "abc_ids": [71, 72],
            "cot": cot,
            "cfg_scale": cfg_scale,
            "max_tokens": max_tokens,
        }


def run_sections(model, clip=None, **overrides):
    inputs = dict(
        clip=clip or FakeClip(),
        style="same global style",
        lyrics=LYRICS,
        abc=ABC,
        seed=50,
        mode="full",
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
    )
    inputs.update(overrides)
    node = generation.HZ3_YuE2_GenerateMusicSections()
    with mock.patch.object(generation, "_prepare_model", return_value=(model, torch.device("cpu"), torch.float32)), \
            mock.patch.object(generation.comfy.ops, "use_quantized_matmul", return_value=nullcontext()), \
            mock.patch.object(generation.comfy.utils, "model_trange", side_effect=lambda n, **_kwargs: range(n)), \
            mock.patch.object(generation.comfy.model_prefetch, "malloc_graph_begin"), \
            mock.patch.object(generation.comfy.model_prefetch, "malloc_graph_end"):
        return node.generate(**inputs)


POSITIVE = [EOD, 42, ABC_START, 71, 72, ABC_END, MUSIC_START]
NEGATIVE = [EOD, ABC_START, 71, 72, ABC_END, MUSIC_START]


class SectionGenerationTests(unittest.TestCase):
    def setUp(self):
        output = tempfile.TemporaryDirectory()
        self.addCleanup(output.cleanup)
        patcher = mock.patch.object(generation.folder_paths, "get_output_directory", return_value=output.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_builds_matching_sections_and_uses_abc_duration(self):
        sections = generation._build_section_specs(ABC, LYRICS)

        self.assertEqual([section["name"] for section in sections], ["intro", "verse 1"])
        self.assertEqual([section["bars"] for section in sections], [1, 1])
        self.assertEqual([section["frames"] for section in sections], [50, 50])
        self.assertEqual([section["seconds"] for section in sections], [2.0, 2.0])
        self.assertEqual(sections[0]["lyrics"], "(instrumental)")
        self.assertEqual(sections[1]["lyrics"], "first line\nsecond line")
        self.assertEqual(sections[0]["abc"].count("% intro"), 1)

    def test_keeps_a_valid_tie_that_crosses_a_section_boundary(self):
        sections = generation._build_section_specs(ABC_TIED_ACROSS_SECTION, LYRICS)

        self.assertEqual([section["seconds"] for section in sections], [2.0, 2.0])
        self.assertTrue(sections[0]["abc"].rstrip().endswith("G32-|"))
        self.assertEqual(sections[0]["bars"], 1)
        self.assertEqual(sections[1]["bars"], 1)

    def test_rejects_out_of_order_or_mismatched_section_names(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            generation._build_section_specs(ABC, "[Intro]\n(instrumental)\n[Chorus]\nwords")

    def test_generates_sections_as_one_continuous_sequence(self):
        for fixed_cache in (False, True):
            for cfg_scale in (1.0, 2.0):
                with self.subTest(fixed_cache=fixed_cache, cfg_scale=cfg_scale), \
                        tempfile.TemporaryDirectory() as output, \
                        mock.patch.object(generation.folder_paths, "get_output_directory", return_value=output):
                    model = FakeSectionModel(fixed_cache)
                    clip = FakeClip()
                    result = run_sections(model, clip=clip, cfg_scale=cfg_scale)

                    conditioning, seconds, report = result["result"]
                    context, metadata = conditioning[0]
                    self.assertEqual(seconds, 4.0)
                    self.assertEqual(metadata["yue2_frames"], 100)
                    self.assertEqual(metadata["yue2_abc_ids"], [71, 72])
                    self.assertEqual(metadata["hz3_section_layout"][1]["start_frame"], 50)
                    self.assertEqual([(chunk[0], chunk[1]) for chunk in metadata["yue2_chunks"]], [(0, 100)])
                    self.assertIn("2 sampled, 0 reused", report)

                    # The whole song is tokenized once and the model prompt is the full prefix.
                    self.assertEqual([call["max_tokens"] for call in clip.calls], [100])
                    self.assertEqual(clip.calls[0]["lyrics"], LYRICS)
                    prefill_prompts = [prefixes[0][:len(POSITIVE)] for prefixes in model.model.prefills]
                    self.assertEqual(prefill_prompts, [POSITIVE, POSITIVE])
                    if cfg_scale != 1.0:
                        self.assertEqual(model.model.prefills[0][1], NEGATIVE)

                    # Verse 1 starts from the intro's tokens, and the acoustic pass sees the full sequence.
                    acoustic_prefix, acoustic_tokens = model.acoustic_call
                    self.assertEqual(acoustic_prefix, POSITIVE)
                    self.assertEqual(model.model.prefills[1][0], POSITIVE + acoustic_tokens[:50])
                    self.assertEqual(len(acoustic_tokens), 100)
                    self.assertTrue(all(CODEC_OFFSET <= token < CODEC_OFFSET + CODEC_SIZE for token in acoustic_tokens))
                    self.assertEqual(acoustic_tokens[50], CODEC_OFFSET + acoustic_tokens[49] % 97)
                    self.assertEqual(context.shape, (1, len(POSITIVE) + 101, 4))

    def test_reuses_stored_section_tokens_and_rerolls_only_seeded_sections(self):
        first = run_sections(FakeSectionModel())["result"]
        cached_model = FakeSectionModel()
        second = run_sections(cached_model)["result"]
        self.assertEqual(cached_model.model.prefills, [])
        self.assertIn("0 sampled, 2 reused", second[2])
        self.assertTrue(torch.equal(first[0][0][0], second[0][0][0]))

        rerolled_model = FakeSectionModel()
        rerolled = run_sections(rerolled_model, section_seeds="verse 1 = 9")["result"]
        layout = rerolled[0][0][1]["hz3_section_layout"]
        self.assertEqual([section["reused"] for section in layout], [True, False])
        self.assertEqual(layout[1]["seed"], 9)
        # Only the verse was prefilled, on top of the replayed intro tokens.
        intro_tokens = rerolled_model.acoustic_call[1][:50]
        self.assertEqual(rerolled_model.model.prefills, [[POSITIVE + intro_tokens]])

    def test_edited_lyrics_invalidate_only_that_section(self):
        run_sections(FakeSectionModel())
        model = FakeSectionModel()
        edited = LYRICS.replace("second line", "another line")
        report = run_sections(model, lyrics=edited)["result"][2]
        self.assertIn("1 sampled, 1 reused", report)
        self.assertEqual(len(model.model.prefills), 1)

    def test_section_style_samples_that_section_with_its_own_prefix(self):
        run_sections(FakeSectionModel())
        model = FakeSectionModel()
        result = run_sections(model, section_styles="Verse 1: female soprano")["result"]
        self.assertIn("1 sampled, 1 reused", result[2])
        self.assertIn("own style", result[2])
        # The verse continues from the stored intro but under the other style's prefix.
        intro_tokens = model.acoustic_call[1][:50]
        self.assertEqual(model.model.prefills, [[[EOD, 43, ABC_START, 71, 72, ABC_END, MUSIC_START] + intro_tokens]])
        self.assertEqual(model.acoustic_call[0], POSITIVE)

    def test_rejects_seed_overrides_for_unknown_sections(self):
        with self.assertRaisesRegex(ValueError, "unknown section"):
            run_sections(FakeSectionModel(), section_seeds="bridge = 3")
        with self.assertRaisesRegex(ValueError, "unknown section"):
            run_sections(FakeSectionModel(), section_styles="bridge: rock")


if __name__ == "__main__":
    unittest.main()
