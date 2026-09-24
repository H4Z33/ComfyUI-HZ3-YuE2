"""Generate YuE2 conditioning one ABC/lyrics section at a time.

The global style prefix is prefetched once and kept immutable. Each section gets
a fresh copy of that prefix KV, then appends only its own lyrics and ABC before
semantic sampling. Section conditionings are streamed into one output tensor so
the final KSampler still renders the song in one pass.
"""

from __future__ import annotations

import re
import unicodedata

import torch

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.utils
from comfy.text_encoders.yue2 import (
    ABC_END,
    ABC_START,
    FRAMES_PER_SECOND,
    MUSIC_END,
    MUSIC_START,
    chunk_ranges,
    distribution,
)
from comfy.text_encoders.llama import rope_matrix

from .abc_score import parse as parse_abc
from .score_analysis import inspect_score
from .sheetsage2_sections import split_abc_by_sections
from .token_stream import _prepare_model


def _normalized_section_name(value: str) -> tuple[str, int | None]:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
    words = re.findall(r"[a-z]+|\d+", normalized)
    aliases = {
        "coro": "chorus",
        "estribillo": "chorus",
        "verso": "verse",
        "estrofa": "verse",
        "puente": "bridge",
        "interludio": "interlude",
        "instrumental": "interlude",
        "introduction": "intro",
        "salida": "outro",
        "coda": "outro",
    }
    if words:
        words[0] = aliases.get(words[0], words[0])
    ordinal = int(words[-1]) if words and words[-1].isdigit() else None
    family = " ".join(words[:-1] if ordinal is not None else words)
    return family, ordinal


def _split_lyrics_by_sections(lyrics: str) -> list[dict]:
    sections = []
    current = None
    preamble = []
    for line in (lyrics or "").replace("\r\n", "\n").split("\n"):
        match = re.fullmatch(r"\s*\[([^\]]+)\]\s*", line)
        if match:
            if current is not None:
                current["lyrics"] = "\n".join(current.pop("lines")).strip()
                sections.append(current)
            current = {"name": match.group(1).strip(), "lines": []}
        elif current is None:
            if line.strip():
                preamble.append(line.strip())
        else:
            current["lines"].append(line.rstrip())

    if preamble:
        raise ValueError(
            "Lyrics must use [Section] headings from top to bottom so they can be "
            "matched to the ABC section markers."
        )
    if current is not None:
        current["lyrics"] = "\n".join(current.pop("lines")).strip()
        sections.append(current)
    if not sections:
        raise ValueError("No [Section] headings were found in the lyrics input.")
    return sections


def _build_section_specs(score_abc: str, lyrics: str) -> list[dict]:
    abc_sections = split_abc_by_sections(score_abc)
    lyric_sections = _split_lyrics_by_sections(lyrics)
    if not abc_sections:
        raise ValueError("No ABC sections were found. Add one '% section name' marker per section.")
    if len(abc_sections) != len(lyric_sections):
        raise ValueError(
            f"ABC has {len(abc_sections)} section(s), but lyrics have "
            f"{len(lyric_sections)} [Section] block(s). They must match one-to-one."
        )

    # Validate and time the complete score before splitting it for YuE2. A
    # tied note may legitimately continue across a section marker; parsing each
    # fragment independently would reject that valid tie at the fragment edge.
    parsed_score = parse_abc(score_abc)
    score_sections = inspect_score(score_abc, lyrics)["sections"]
    if len(score_sections) != len(abc_sections):
        raise ValueError(
            "ABC section markers do not match the score's parsed section boundaries."
        )

    specs = []
    bar_cursor = 0
    for index, ((abc_name, raw_fragment), lyric_section) in enumerate(
        zip(abc_sections, lyric_sections), 1
    ):
        abc_family, abc_ordinal = _normalized_section_name(abc_name)
        lyric_family, lyric_ordinal = _normalized_section_name(lyric_section["name"])
        if abc_family != lyric_family or (
            abc_ordinal is not None
            and lyric_ordinal is not None
            and abc_ordinal != lyric_ordinal
        ):
            raise ValueError(
                f"Section {index} does not match: ABC says {abc_name!r}, "
                f"lyrics say {lyric_section['name']!r}. Keep the section names and order aligned."
            )

        section_abc = raw_fragment
        score_section = score_sections[index - 1]
        bars = len(score_section["vocal"])
        section_bars = parsed_score.voices["Vocal"].bars[bar_cursor:bar_cursor + bars]
        if bars < 1 or len(section_bars) != bars:
            raise ValueError(f"ABC section {abc_name!r} has no measurable duration.")
        bar_cursor += bars
        seconds = sum(
            float(length) * 60 / parsed_score.bpm
            for _, length, _meter in section_bars
        )
        target_frames = round(seconds * FRAMES_PER_SECOND)
        if target_frames < 1:
            raise ValueError(f"ABC section {abc_name!r} has no measurable duration.")
        first_meter = section_bars[0][2]
        specs.append(
            {
                "name": abc_name,
                "lyrics_name": lyric_section["name"],
                "lyrics": lyric_section["lyrics"],
                "abc": section_abc,
                "bars": bars,
                "seconds": seconds,
                "frames": target_frames,
                "bpm": int(parsed_score.bpm),
                "meter": f"{first_meter[0]}/{first_meter[1]}",
            }
        )
    return specs


def _common_prefix_length(prefixes: list[list[int]], maximum: int) -> int:
    if not prefixes or maximum < 1:
        raise ValueError("YuE2 produced an empty shared style prefix.")
    limit = min(maximum, *(len(prefix) for prefix in prefixes))
    for index in range(limit):
        token = prefixes[0][index]
        if any(prefix[index] != token for prefix in prefixes[1:]):
            return index
    return limit


def _clone_prefix_cache(model, source_cache, prefix_length: int, capacity: int, device, dtype):
    """Create a writable branch cache containing only the immutable base prefix."""
    if prefix_length < 1:
        raise ValueError("The shared YuE2 prefix must contain at least one token.")
    if capacity < prefix_length:
        raise ValueError("A section KV cache is shorter than its shared prefix.")

    branch = model.model.init_kv_cache(1, capacity, device, dtype)
    if len(branch) != len(source_cache):
        raise ValueError("YuE2 returned incompatible base and section KV caches.")

    for layer_index, (source, target) in enumerate(zip(source_cache, branch)):
        if hasattr(source, "key") and hasattr(source, "value"):
            if not (hasattr(target, "key") and hasattr(target, "value")):
                raise ValueError("YuE2 changed KV cache types while branching a section.")
            if source.key.shape[1] < prefix_length or target.key.shape[1] < prefix_length:
                raise ValueError("YuE2 fixed KV cache cannot hold the frozen prefix.")
            target.key[:, :prefix_length].copy_(source.key[:, :prefix_length])
            target.value[:, :prefix_length].copy_(source.value[:, :prefix_length])
            target.index = prefix_length
            if torch.is_tensor(getattr(target, "seqlen", None)):
                target.seqlen.fill_(prefix_length)
            if torch.is_tensor(getattr(target, "position", None)):
                target.position.fill_(prefix_length)
            continue

        if (
            isinstance(source, (tuple, list))
            and isinstance(target, (tuple, list))
            and len(source) >= 2
            and len(target) >= 2
        ):
            source_key, source_value = source[0], source[1]
            target_key, target_value = target[0], target[1]
            if source_key.shape[2] < prefix_length or target_key.shape[2] < prefix_length:
                raise ValueError("YuE2 tuple KV cache cannot hold the frozen prefix.")
            target_key[:, :, :prefix_length].copy_(source_key[:, :, :prefix_length])
            target_value[:, :, :prefix_length].copy_(source_value[:, :, :prefix_length])
            branch[layer_index] = (
                target_key,
                target_value,
                prefix_length,
            )
            continue

        raise ValueError("YuE2 returned an unsupported KV cache format.")
    return branch


def _forward_with_cache(model, cache, token_ids: list[int], position_start: int, device, dtype):
    ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    positions = torch.arange(
        position_start,
        position_start + len(token_ids),
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)
    output = model.model(ids, past_key_values=cache, dtype=dtype, position_ids=positions)
    return model.model.lm_head(output[0][:, -1]), output[2]


def _new_decode_state(model, logits, first_position: int, cache, device, dtype):
    position_ids = torch.tensor([[first_position]], device=device, dtype=torch.long)
    fixed_kv = hasattr(cache[0], "key") and hasattr(cache[0], "advance")
    decode_buffers = None
    if fixed_kv:
        decode_buffers = (
            torch.empty((1, 1, model.config.hidden_size), device=device, dtype=dtype),
            rope_matrix(model.model.compute_freqs_cis(position_ids, device)),
        )
    return {
        "ids": torch.empty((1, 1), device=device, dtype=torch.long),
        "position_ids": position_ids,
        "logits": torch.empty_like(logits),
        "decode_buffers": decode_buffers,
        "fixed_kv": fixed_kv,
        "dtype": dtype,
    }


def _decode_one(model, cache, token: int, position: int, state):
    state["ids"].fill_(token)
    state["position_ids"].fill_(position)
    fixed_kv = state["fixed_kv"]
    if fixed_kv:
        comfy.model_prefetch.malloc_graph_begin(state["ids"].device)
    try:
        output = model.model(
            state["ids"],
            past_key_values=cache,
            dtype=state["dtype"],
            position_ids=state["position_ids"],
            decode_buffers=state["decode_buffers"],
        )
        state["logits"].copy_(model.model.lm_head(output[0][:, -1]))
        next_cache = output[2]
        del output
        return state["logits"], next_cache
    finally:
        if fixed_kv:
            comfy.model_prefetch.malloc_graph_end()


def _generate_section_tokens(
    model,
    section,
    positive_base_cache,
    positive_base,
    negative_base_cache,
    negative_base,
    positive_tokens,
    cfg_scale,
    section_seed,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    device,
    dtype,
    progress,
    progress_start,
):
    abc_ids = positive_tokens["abc_ids"]
    positive_full = positive_tokens["prefix"] + abc_ids + [ABC_END, MUSIC_START]
    positive_suffix = positive_full[len(positive_base):]
    positive_length = len(positive_full)
    negative_full = None
    negative_suffix = None
    if cfg_scale != 1.0:
        negative_full = negative_base + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]
        negative_suffix = negative_full[len(negative_base):]
    capacity = max(
        positive_length,
        len(negative_full) if negative_full is not None else positive_length,
    ) + section["frames"]
    if capacity > model.config.max_position_embeddings:
        raise ValueError(
            f"Section {section['name']!r} needs {capacity} YuE2 context tokens, "
            f"above the model limit {model.config.max_position_embeddings}. "
            "Shorten this section's lyrics, ABC, or target duration."
        )

    positive_cache = _clone_prefix_cache(
        model, positive_base_cache, len(positive_base), capacity, device, dtype
    )
    positive_logits, positive_cache = _forward_with_cache(
        model, positive_cache, positive_suffix, len(positive_base), device, dtype
    )

    negative_cache = None
    negative_logits = None
    if cfg_scale != 1.0:
        negative_cache = _clone_prefix_cache(
            model, negative_base_cache, len(negative_base), capacity, device, dtype
        )
        negative_logits, negative_cache = _forward_with_cache(
            model, negative_cache, negative_suffix, len(negative_base), device, dtype
        )

    target_frames = section["frames"]
    positive_decode_state = None
    negative_decode_state = None
    if target_frames > 1:
        positive_decode_state = _new_decode_state(
            model, positive_logits, positive_length, positive_cache, device, dtype
        )
        if cfg_scale != 1.0:
            negative_decode_state = _new_decode_state(
                model, negative_logits, len(negative_full), negative_cache, device, dtype
            )

    rng_device = device if torch.device(device).type != "mps" else "cpu"
    generator = torch.Generator(device=rng_device).manual_seed(section_seed)
    history = []

    try:
        for step in comfy.utils.model_trange(
            target_frames,
            desc=f"YuE2 section {section['name']}",
            unit="token",
        ):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if cfg_scale == 1.0:
                guided = positive_logits
            else:
                guided = negative_logits + cfg_scale * (positive_logits - negative_logits)
            scores = distribution(
                guided,
                history,
                step,
                "semantic",
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                50,
                target_frames,
            )
            if temperature == 0:
                next_id = scores.argmax(-1, keepdim=True)
            else:
                probabilities = scores.softmax(-1).to(rng_device)
                next_id = torch.multinomial(probabilities, 1, generator=generator).to(device)
            token = int(next_id.item())
            if token == MUSIC_END:
                raise RuntimeError(
                    f"YuE2 ended section {section['name']!r} before its ABC duration. "
                    "The exact-length guard should prevent this."
                )
            history.append(token)
            progress.update_absolute(progress_start + step + 1)

            if step + 1 < target_frames:
                positive_logits, positive_cache = _decode_one(
                    model, positive_cache, token, positive_length + step, positive_decode_state
                )
                if cfg_scale != 1.0:
                    negative_logits, negative_cache = _decode_one(
                        model,
                        negative_cache,
                        token,
                        len(negative_full) + step,
                        negative_decode_state,
                    )
    finally:
        comfy.model_prefetch.cleanup_prefetch_queues()

    if len(history) != target_frames:
        raise RuntimeError(
            f"YuE2 generated {len(history)} frame token(s) for {section['name']!r}; "
            f"the ABC requires exactly {target_frames}."
        )
    return history, positive_full, abc_ids


def _expected_conditioning_tokens(sections, prefixes, context):
    total = 0
    for section, prefix in zip(sections, prefixes):
        for start, end in chunk_ranges(section["frames"], len(prefix), context):
            total += len(prefix) + end - start + 1
    return total


def _append_section_conditioning(
    output,
    section_context,
    local_chunks,
    global_start,
    kv_cursor,
):
    global_chunks = []
    for chunk_start, chunk_end, local_kv_start, local_kv_end in local_chunks:
        size = local_kv_end - local_kv_start
        output[:, kv_cursor:kv_cursor + size].copy_(
            section_context[:, local_kv_start:local_kv_end]
        )
        global_chunks.append(
            (
                global_start + chunk_start,
                global_start + chunk_end,
                kv_cursor,
                kv_cursor + size,
            )
        )
        kv_cursor += size
    return global_chunks, kv_cursor


class HZ3_YuE2_GenerateMusicSections:
    CATEGORY = "HZ3 YuE2/Generation"
    FUNCTION = "generate"
    RETURN_TYPES = ("CONDITIONING", "FLOAT", "STRING")
    RETURN_NAMES = ("conditioning", "seconds", "report")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Generate each tagged ABC/lyrics section serially from one frozen global "
        "YuE2 prompt prefix, then join the section conditionings for one KSampler pass."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "style": ("STRING", {"multiline": True, "default": ""}),
                "lyrics": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "tooltip": "Full MixMash lyrics with one [Section] heading per ABC section.",
                }),
                "abc": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "tooltip": "Full MixMash ABC with matching '% section' markers.",
                }),
                "seed": ("INT", {
                    "default": 60,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                }),
                "mode": (["full", "melody"], {"default": "full"}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 5.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 100, "min": 1, "max": 32768}),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.01,
                }),
            },
            "optional": {
                "cfg_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                    "advanced": True,
                }),
            },
        }

    def generate(
        self,
        clip,
        style,
        lyrics,
        abc,
        seed,
        mode,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        cfg_scale=1.0,
    ):
        specs = _build_section_specs(abc, lyrics)
        style = str(style or "").strip()
        seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        cfg_scale = float(cfg_scale)

        token_specs = []
        for index, section in enumerate(specs):
            section_seed = (seed + index) & 0xFFFFFFFFFFFFFFFF
            section_lyrics = f"[{section['name']}]\n{section['lyrics']}".strip()
            tokens = clip.tokenize(
                style,
                lyrics=section_lyrics,
                cot=mode,
                seed=section_seed,
                abc=section["abc"],
                max_tokens=section["frames"],
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                cfg_scale=cfg_scale,
            )
            token_specs.append((section_seed, tokens))

        empty_prompt = clip.tokenize(
            style,
            lyrics="",
            cot=mode,
            seed=seed,
            abc="",
            max_tokens=1,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            cfg_scale=cfg_scale,
        )
        # The empty-lyrics prompt ends in a newline plus ABC_START; cap the shared
        # cache before ABC_START and at the common prefix of the actual sections.
        prompt_limit = max(1, len(empty_prompt["prefix"]) - 1)
        positive_prefixes = [tokens["prefix"] for _, tokens in token_specs]
        shared_length = _common_prefix_length(positive_prefixes, prompt_limit)
        positive_base = positive_prefixes[0][:shared_length]
        if not positive_base:
            raise ValueError("Could not find a shared YuE2 style prefix for the sections.")

        negative_base = token_specs[0][1]["negative"]
        if any(tokens["negative"] != negative_base for _, tokens in token_specs[1:]):
            raise ValueError("YuE2 section modes produced different negative prompt prefixes.")

        model, device, dtype = _prepare_model(clip, token_specs[0][1])
        total_frames = sum(section["frames"] for section in specs)
        progress = comfy.utils.ProgressBar(total_frames)
        global_start = 0
        kv_cursor = 0
        all_chunks = []
        all_abc_ids = []
        section_layout = []
        assembled = None
        expected_kv_tokens = None

        try:
            with comfy.model_management.cuda_device_context(device), comfy.ops.use_quantized_matmul(
                model, device
            ):
                _, positive_base_cache, _ = model._prefill(
                    [positive_base], len(positive_base), dtype
                )
                negative_base_cache = None
                if cfg_scale != 1.0:
                    _, negative_base_cache, _ = model._prefill(
                        [negative_base], len(negative_base), dtype
                    )

                positive_full_prefixes = [
                    tokens["prefix"] + tokens["abc_ids"] + [ABC_END, MUSIC_START]
                    for _, tokens in token_specs
                ]
                expected_kv_tokens = _expected_conditioning_tokens(
                    specs, positive_full_prefixes, model.config.max_position_embeddings
                )

                for index, (section, (section_seed, tokens)) in enumerate(
                    zip(specs, token_specs)
                ):
                    semantic, acoustic_prefix, abc_ids = _generate_section_tokens(
                        model=model,
                        section=section,
                        positive_base_cache=positive_base_cache,
                        positive_base=positive_base,
                        negative_base_cache=negative_base_cache,
                        negative_base=negative_base,
                        positive_tokens=tokens,
                        cfg_scale=cfg_scale,
                        section_seed=section_seed,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        top_k=int(top_k),
                        repetition_penalty=float(repetition_penalty),
                        device=device,
                        dtype=dtype,
                        progress=progress,
                        progress_start=global_start,
                    )
                    section_context, local_chunks = model._acoustic_conditioning(
                        acoustic_prefix, semantic, dtype
                    )
                    if assembled is None:
                        assembled = torch.empty(
                            (1, expected_kv_tokens, section_context.shape[-1]),
                            device=section_context.device,
                            dtype=section_context.dtype,
                        )
                    section_chunks, kv_cursor = _append_section_conditioning(
                        assembled,
                        section_context,
                        local_chunks,
                        global_start,
                        kv_cursor,
                    )
                    all_chunks.extend(section_chunks)
                    all_abc_ids.extend(abc_ids)
                    section_layout.append(
                        {
                            "name": section["name"],
                            "start_frame": global_start,
                            "end_frame": global_start + section["frames"],
                            "seconds": section["frames"] / FRAMES_PER_SECOND,
                            "bars": section["bars"],
                            "seed": section_seed,
                        }
                    )
                    global_start += section["frames"]
                    del semantic, section_context
                    comfy.model_prefetch.cleanup_prefetch_queues()
        finally:
            comfy.model_prefetch.cleanup_prefetch_queues()

        if assembled is None or global_start != total_frames or kv_cursor != expected_kv_tokens:
            raise RuntimeError(
                "YuE2 section conditioning assembly did not match the planned score duration."
            )

        metadata = {
            "pooled_output": None,
            "yue2_chunks": tuple(all_chunks),
            "yue2_abc_ids": all_abc_ids,
            "yue2_frames": total_frames,
            "yue2_truncated": False,
            "hz3_section_assembly": True,
            "hz3_section_layout": section_layout,
            "hz3_frozen_prefix_tokens": shared_length,
        }
        conditioning = [[assembled, metadata]]
        seconds = total_frames / FRAMES_PER_SECOND
        report_lines = [
            f"Generated {len(specs)} ABC/lyrics sections · {seconds:.2f} s",
            f"Frozen shared prompt prefix: {shared_length} tokens · one final KSampler pass",
            "Section lengths are locked to the ABC; each block ended at its score boundary.",
        ]
        for section in section_layout:
            start_seconds = section["start_frame"] / FRAMES_PER_SECOND
            end_seconds = section["end_frame"] / FRAMES_PER_SECOND
            report_lines.append(
                f"{section['name']}: {section['bars']} bars · "
                f"{start_seconds:.2f}-{end_seconds:.2f}s · seed {section['seed']}"
            )
        report = "\n".join(report_lines)
        return {
            "ui": {"text": [report]},
            "result": (conditioning, seconds, report),
        }


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_GenerateMusicSections": HZ3_YuE2_GenerateMusicSections,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_GenerateMusicSections": "HZ3 YuE2 · Generate Music Sections",
}
