"""Generate YuE2 conditioning for a whole song one ABC/lyrics section at a time.

The model always sees the complete style, lyrics, and ABC prompt plus every
semantic token that comes before the current section, exactly like a single
whole-song pass, but sampling stops at each section's score boundary. Each
section's tokens are stored on disk keyed by what shaped them, so editing one
section only resamples that section: untouched sections are replayed from
storage through one batched prefill instead of being generated again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata

import torch

import folder_paths
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.utils
from comfy.text_encoders.yue2 import (
    ABC_END,
    ABC_START,
    CODEC_OFFSET,
    CODEC_SIZE,
    FRAMES_PER_SECOND,
    MUSIC_START,
    distribution,
)
from comfy.text_encoders.llama import FixedKV, rope_matrix

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


def _parse_section_seeds(text: str) -> dict[str, int]:
    seeds = {}
    for line in (text or "").replace(",", "\n").splitlines():
        if not line.strip():
            continue
        name, separator, value = line.rpartition("=")
        if not separator or not name.strip() or not value.strip().isdigit():
            raise ValueError(f"Section seed lines must look like 'Chorus 2 = 1234', got {line.strip()!r}.")
        seeds[name.strip().casefold()] = int(value) & 0xFFFFFFFFFFFFFFFF
    return seeds


def _patch_summary(value):
    if torch.is_tensor(value):
        return [list(value.shape), float(value.float().sum())]
    if isinstance(value, (tuple, list)):
        return [_patch_summary(item) for item in value]
    if hasattr(value, "weights"):
        return _patch_summary(value.weights)
    return value if isinstance(value, (int, float, str)) or value is None else repr(value)


def _patches_fingerprint(clip) -> list:
    """LoRA and other weight patches change sampling, so they belong in the token key."""
    return sorted(
        [key, [[float(strength), _patch_summary(patch), float(model_strength)] for strength, patch, model_strength, *_ in patches]]
        for key, patches in clip.patcher.patches.items()
    )


def _section_token_path(fields: dict) -> Path:
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()
    return Path(folder_paths.get_output_directory()) / "HZ3-YuE2" / "section_tokens" / f"{digest}.json"


def _load_section_tokens(path: Path, frames: int):
    if not path.is_file():
        return None
    tokens = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(tokens, list)
        or len(tokens) != frames
        or any(type(token) is not int or not CODEC_OFFSET <= token < CODEC_OFFSET + CODEC_SIZE for token in tokens)
    ):
        return None
    return tokens


def _sample_section(model, prefixes, cfg_scale, frames, seed, history, sampling, device, dtype, progress, progress_start, name):
    """Prefill the prompt plus every earlier token, then sample exactly `frames` codec tokens."""
    rng_device = device if torch.device(device).type != "mps" else "cpu"
    generator = torch.Generator(device=rng_device).manual_seed(seed)
    prefix_length = max(map(len, prefixes))
    logits, cache, mask = model._prefill(prefixes, prefix_length + frames, dtype)
    fixed_kv = isinstance(cache[0], FixedKV)
    decode_tokens = torch.empty((len(prefixes), 1), device=device, dtype=torch.long)
    positions = torch.tensor([[len(prefix)] for prefix in prefixes], device=device, dtype=torch.long)
    decode_buffers = None
    if fixed_kv:
        decode_buffers = (torch.empty((len(prefixes), 1, model.config.hidden_size), device=device, dtype=dtype),
                          rope_matrix(model.model.compute_freqs_cis(positions, device)))
    tokens = []
    try:
        for step in comfy.utils.model_trange(frames, desc=f"YuE2 section {name}", unit="token"):
            comfy.model_management.throw_exception_if_processing_interrupted()
            guided = logits if cfg_scale == 1.0 else logits[1:] + cfg_scale * (logits[:1] - logits[1:])
            scores = distribution(guided, history, step, "semantic", penalty_window=50, min_tokens=frames, **sampling)
            if sampling["temperature"] == 0:
                next_id = scores.argmax(-1, keepdim=True)
            else:
                probabilities = scores.softmax(-1).to(rng_device)
                next_id = torch.multinomial(probabilities, 1, generator=generator).to(device)
            decode_tokens.copy_(next_id)
            token = next_id.item()
            tokens.append(token)
            history.append(token)
            progress.update_absolute(progress_start + step + 1)
            if step + 1 < frames:
                if fixed_kv:
                    comfy.model_prefetch.malloc_graph_begin(device)
                output = model.model(decode_tokens, past_key_values=cache, dtype=dtype, position_ids=positions,
                                     attention_mask=mask[:, :prefix_length + step + 1] if mask is not None and not fixed_kv else None,
                                     decode_buffers=decode_buffers)
                logits.copy_(model.model.lm_head(output[0][:, -1]))
                cache = output[2]
                del output
                if fixed_kv:
                    comfy.model_prefetch.malloc_graph_end()
                positions.add_(1)
    finally:
        comfy.model_prefetch.cleanup_prefetch_queues()
    return tokens


class HZ3_YuE2_GenerateMusicSections:
    CATEGORY = "HZ3 YuE2/Generation"
    FUNCTION = "generate"
    RETURN_TYPES = ("CONDITIONING", "FLOAT", "STRING")
    RETURN_NAMES = ("conditioning", "seconds", "report")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Generate the song as one continuous YuE2 pass that stops at every ABC/lyrics "
        "section boundary and stores each section's tokens, so edited sections are "
        "resampled while the rest is replayed."
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
                "section_seeds": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional per-section seed overrides, one 'Chorus 2 = 1234' per line, "
                               "to re-roll only those sections. Unchanged sections reuse their stored tokens.",
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
        section_seeds="",
    ):
        specs = _build_section_specs(abc, lyrics)
        style = str(style or "").strip()
        seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        cfg_scale = float(cfg_scale)
        sampling = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
        }
        seed_overrides = _parse_section_seeds(section_seeds)
        unknown = set(seed_overrides) - {section["name"].casefold() for section in specs}
        if unknown:
            raise ValueError(f"section_seeds names unknown section(s): {', '.join(sorted(unknown))}.")

        total_frames = sum(section["frames"] for section in specs)
        tokens = clip.tokenize(
            style,
            lyrics=lyrics,
            cot=mode,
            seed=seed,
            abc=abc,
            max_tokens=total_frames,
            cfg_scale=cfg_scale,
            **sampling,
        )
        abc_ids = tokens["abc_ids"]
        positive = tokens["prefix"] + abc_ids + [ABC_END, MUSIC_START]
        negative = tokens["negative"] + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]
        model, device, dtype = _prepare_model(clip, tokens)
        context = model.config.max_position_embeddings
        if max(len(positive), len(negative)) + total_frames > context:
            raise ValueError(
                f"The song needs {max(len(positive), len(negative)) + total_frames} YuE2 context tokens, "
                f"above the model limit {context}. Shorten the lyrics, ABC, or score duration."
            )

        patches = _patches_fingerprint(clip)
        plan = []
        for index, section in enumerate(specs):
            section_seed = seed_overrides.get(section["name"].casefold(), (seed + index) & 0xFFFFFFFFFFFFFFFF)
            path = _section_token_path(
                {
                    "style": style,
                    "mode": mode,
                    "lyrics": section["lyrics"],
                    "abc": section["abc"],
                    "frames": section["frames"],
                    "seed": section_seed,
                    "sampling": [*sampling.values(), cfg_scale],
                    "patches": patches,
                }
            )
            plan.append((section, section_seed, path, _load_section_tokens(path, section["frames"])))

        progress = comfy.utils.ProgressBar(total_frames)
        history = []
        section_layout = []
        try:
            with comfy.model_management.cuda_device_context(device), comfy.ops.use_quantized_matmul(model, device):
                for section, section_seed, path, stored in plan:
                    start = len(history)
                    if stored is None:
                        prefixes = [positive + history]
                        if cfg_scale != 1.0:
                            prefixes.append(negative + history)
                        stored = _sample_section(
                            model, prefixes, cfg_scale, section["frames"], section_seed, history,
                            sampling, device, dtype, progress, start, section["name"],
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(stored), encoding="utf-8")
                        reused = False
                    else:
                        history.extend(stored)
                        progress.update_absolute(len(history))
                        reused = True
                    section_layout.append(
                        {
                            "name": section["name"],
                            "start_frame": start,
                            "end_frame": len(history),
                            "seconds": section["frames"] / FRAMES_PER_SECOND,
                            "bars": section["bars"],
                            "seed": section_seed,
                            "reused": reused,
                        }
                    )
                conditioning, chunks = model._acoustic_conditioning(positive, history, dtype)
        finally:
            comfy.model_prefetch.cleanup_prefetch_queues()

        metadata = {
            "pooled_output": None,
            "yue2_chunks": chunks,
            "yue2_abc_ids": abc_ids,
            "yue2_frames": total_frames,
            "yue2_truncated": False,
            "hz3_section_layout": section_layout,
        }
        seconds = total_frames / FRAMES_PER_SECOND
        resampled = sum(not section["reused"] for section in section_layout)
        report_lines = [
            f"Generated {len(specs)} ABC/lyrics sections · {seconds:.2f} s · "
            f"{resampled} sampled, {len(specs) - resampled} reused from stored tokens",
            "One continuous YuE2 pass over the full prompt; section lengths are locked to the ABC.",
        ]
        for section in section_layout:
            start_seconds = section["start_frame"] / FRAMES_PER_SECOND
            end_seconds = section["end_frame"] / FRAMES_PER_SECOND
            report_lines.append(
                f"{section['name']}: {section['bars']} bars · "
                f"{start_seconds:.2f}-{end_seconds:.2f}s · seed {section['seed']}"
                f"{' · reused' if section['reused'] else ''}"
            )
        report = "\n".join(report_lines)
        return {
            "ui": {"text": [report]},
            "result": ([[conditioning, metadata]], seconds, report),
        }


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_GenerateMusicSections": HZ3_YuE2_GenerateMusicSections,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_GenerateMusicSections": "HZ3 YuE2 · Generate Music Sections",
}
