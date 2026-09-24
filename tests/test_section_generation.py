"""Run with the ComfyUI Python environment and its root on PYTHONPATH."""

import importlib
from contextlib import nullcontext
from pathlib import Path
import re
import sys
import types
import unittest
from unittest import mock

import torch
from comfy.text_encoders.yue2 import ABC_START, CODEC_OFFSET, CODEC_SIZE, CONTEXT, EOD


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


class FakeModel:
    def __init__(self, cache_factory):
        self.model = types.SimpleNamespace(init_kv_cache=cache_factory)


class FakeFixedKVCache:
    def __init__(self, batch, capacity, device, dtype):
        self.key = torch.zeros((batch, capacity, 1, 1), device=device, dtype=dtype)
        self.value = torch.zeros_like(self.key)
        self.index = 0
        self.position = torch.zeros((batch,), device=device, dtype=torch.int64)
        self.seqlen = torch.zeros((batch,), device=device, dtype=torch.int32)

    def prepare(self, count):
        self.position.copy_(self.seqlen)
        self.seqlen.add_(count)

    def advance(self, count):
        self.index += count


class FakeSectionTransformer:
    def __init__(self, fixed_cache=False):
        self.base_caches = []
        self.fixed_cache = fixed_cache

    def init_kv_cache(self, batch, capacity, device, dtype):
        if self.fixed_cache:
            return [FakeFixedKVCache(batch, capacity, device, dtype)]
        return [(torch.zeros((batch, 1, capacity, 1), device=device, dtype=dtype),
                 torch.zeros((batch, 1, capacity, 1), device=device, dtype=dtype), 0)]

    def __call__(self, ids, past_key_values, dtype, position_ids=None, decode_buffers=None):
        result = []
        for cache in past_key_values:
            if isinstance(cache, FakeFixedKVCache):
                past_length = cache.index
                cache.prepare(ids.shape[1])
                end = past_length + ids.shape[1]
                cache.key[:, past_length:end].copy_(ids[:, :, None, None].to(cache.key))
                cache.value[:, past_length:end].copy_(ids[:, :, None, None].to(cache.value))
                cache.advance(ids.shape[1])
                result.append(cache)
            else:
                key, value, past_length = cache
                end = past_length + ids.shape[1]
                key[:, :, past_length:end].copy_(ids[:, None, :, None].to(key))
                value[:, :, past_length:end].copy_(ids[:, None, :, None].to(value))
                result.append((key, value, end))
        hidden = ids.to(dtype).unsqueeze(-1).expand(-1, -1, 4)
        return hidden, None, result

    def lm_head(self, hidden):
        return torch.zeros((hidden.shape[0], CODEC_OFFSET + CODEC_SIZE), device=hidden.device)

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
        ids = torch.tensor(prefixes, device="cpu", dtype=torch.long)
        cache = self.model.init_kv_cache(1, capacity, "cpu", dtype)
        output = self.model(ids, past_key_values=cache, dtype=dtype)
        self.model.base_caches.append(output[2])
        return self.model.lm_head(output[0][:, -1]), output[2], None

    def _acoustic_conditioning(self, prefix, tokens, dtype):
        ranges = generation.chunk_ranges(len(tokens), len(prefix), self.config.max_position_embeddings)
        total = sum(len(prefix) + end - start + 1 for start, end in ranges)
        context = torch.arange(total * 4, device="cpu", dtype=dtype).reshape(1, total, 4)
        chunks = []
        offset = 0
        for start, end in ranges:
            size = len(prefix) + end - start + 1
            chunks.append((start, end, offset, offset + size))
            offset += size
        return context, tuple(chunks)


class FakeClip:
    def tokenize(self, style, lyrics, cot, seed, abc, max_tokens, temperature, top_p, top_k,
                 repetition_penalty, cfg_scale):
        if lyrics:
            heading = re.search(r"\[([^\]]+)\]", lyrics).group(1)
            section_token = 101 if heading.lower().startswith("intro") else 102
            prefix = [EOD, 42, section_token, ABC_START]
        else:
            prefix = [EOD, 42, ABC_START]
        return {
            "prefix": prefix,
            "negative": [EOD, 42],
            "abc_ids": [71, 72],
            "cot": cot,
            "cfg_scale": cfg_scale,
            "max_tokens": max_tokens,
        }


class SectionGenerationTests(unittest.TestCase):
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

    def test_clones_only_the_used_tuple_cache_prefix(self):
        source_key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3)
        source_value = source_key + 100

        def cache_factory(batch, capacity, device, dtype):
            return [(torch.zeros((batch, 2, capacity, 3), device=device, dtype=dtype),
                     torch.zeros((batch, 2, capacity, 3), device=device, dtype=dtype), 0)]

        model = FakeModel(cache_factory)
        branch = generation._clone_prefix_cache(
            model, [(source_key, source_value, 4)], 3, 7, "cpu", torch.float32
        )

        self.assertEqual(branch[0][2], 3)
        self.assertTrue(torch.equal(branch[0][0][:, :, :3], source_key[:, :, :3]))
        self.assertTrue(torch.equal(branch[0][1][:, :, :3], source_value[:, :, :3]))
        self.assertTrue(torch.equal(branch[0][0][:, :, 3:], torch.zeros_like(branch[0][0][:, :, 3:])))

    def test_clones_fixed_cache_prefix_and_position_state(self):
        source_key = torch.arange(24, dtype=torch.float32).reshape(1, 4, 2, 3)
        source_value = source_key + 100
        source = types.SimpleNamespace(
            key=source_key,
            value=source_value,
            index=4,
            position=torch.tensor([4], dtype=torch.int64),
            seqlen=torch.tensor([4], dtype=torch.int32),
        )

        def cache_factory(batch, capacity, device, dtype):
            return [types.SimpleNamespace(
                key=torch.zeros((batch, capacity, 2, 3), device=device, dtype=dtype),
                value=torch.zeros((batch, capacity, 2, 3), device=device, dtype=dtype),
                index=0,
                position=torch.zeros((batch,), device=device, dtype=torch.int64),
                seqlen=torch.zeros((batch,), device=device, dtype=torch.int32),
            )]

        model = FakeModel(cache_factory)
        branch = generation._clone_prefix_cache(
            model, [source], 3, 7, "cpu", torch.float32
        )

        self.assertEqual(branch[0].index, 3)
        self.assertEqual(branch[0].position.item(), 3)
        self.assertEqual(branch[0].seqlen.item(), 3)
        self.assertTrue(torch.equal(branch[0].key[:, :3], source_key[:, :3]))
        self.assertTrue(torch.equal(branch[0].value[:, :3], source_value[:, :3]))

    def test_generates_sections_serially_into_one_exact_length_conditioning(self):
        for fixed_cache in (False, True):
            for cfg_scale in (1.0, 2.0):
                with self.subTest(fixed_cache=fixed_cache, cfg_scale=cfg_scale):
                    model = FakeSectionModel(fixed_cache)
                    node = generation.HZ3_YuE2_GenerateMusicSections()
                    with mock.patch.object(generation, "_prepare_model", return_value=(model, torch.device("cpu"), torch.float32)), \
                            mock.patch.object(generation.comfy.ops, "use_quantized_matmul", return_value=nullcontext()), \
                            mock.patch.object(generation.comfy.utils, "model_trange", side_effect=lambda n, **_kwargs: range(n)), \
                            mock.patch.object(generation.comfy.model_prefetch, "malloc_graph_begin"), \
                            mock.patch.object(generation.comfy.model_prefetch, "malloc_graph_end"):
                        result = node.generate(
                            clip=FakeClip(),
                            style="same global style",
                            lyrics=LYRICS,
                            abc=ABC,
                            seed=50,
                            mode="full",
                            temperature=0.0,
                            top_p=1.0,
                            top_k=1,
                            repetition_penalty=1.0,
                            cfg_scale=cfg_scale,
                        )

                    conditioning, seconds, report = result["result"]
                    context, metadata = conditioning[0]
                    self.assertEqual(seconds, 4.0)
                    self.assertEqual(metadata["yue2_frames"], 100)
                    self.assertEqual(context.shape[0], 1)
                    self.assertEqual(metadata["hz3_section_layout"][1]["start_frame"], 50)
                    self.assertEqual(
                        [(chunk[0], chunk[1]) for chunk in metadata["yue2_chunks"]],
                        [(0, 50), (50, 100)],
                    )
                    self.assertIn("one final KSampler pass", report)
                    base_layer = model.model.base_caches[0][0]
                    if isinstance(base_layer, FakeFixedKVCache):
                        base_key = base_layer.key[:, :2]
                    else:
                        base_key = base_layer[0][:, :, :2]
                    self.assertTrue(torch.equal(
                        base_key.flatten(),
                        torch.tensor([EOD, 42], dtype=torch.float32),
                    ))


if __name__ == "__main__":
    unittest.main()
