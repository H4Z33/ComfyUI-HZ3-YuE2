"""Expose YuE2 semantic music tokens so generations can be reused and sliced."""

from __future__ import annotations

import hashlib
import json

import torch

import comfy.model_management
import comfy.ops
import comfy.model_prefetch
from comfy.text_encoders.yue2 import (
    ABC_END,
    ABC_START,
    CODEC_OFFSET,
    CODEC_SIZE,
    CONTEXT,
    FRAMES_PER_SECOND,
    MUSIC_END,
    MUSIC_START,
)


STREAM_TYPE = "HZ3_YUE2_TOKEN_STREAM"


def _prompt_hash(style, lyrics, abc, mode):
    payload = json.dumps(
        {"style": style, "lyrics": lyrics, "abc": abc, "mode": mode},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tokenize(clip, style, lyrics, abc, mode, seed, max_tokens, temperature, top_p, top_k,
              repetition_penalty):
    effective_mode = mode if (abc or "").strip() else "off"
    tokens = clip.tokenize(
        style,
        lyrics=lyrics,
        cot=effective_mode,
        seed=seed,
        abc=abc,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    return tokens, effective_mode


def _prefixes(tokens):
    abc_ids = tokens["abc_ids"] if tokens["cot"] != "off" else []
    prefix = tokens["prefix"] + abc_ids + [ABC_END, MUSIC_START]
    negative = tokens["negative"] + (
        [MUSIC_START] if tokens["cot"] == "off" else [ABC_START] + abc_ids + [ABC_END, MUSIC_START]
    )
    return prefix, negative, abc_ids


def _prepare_model(clip, tokens):
    model = clip.cond_stage_model
    required = ("_generate", "_acoustic_conditioning", "config")
    if not all(hasattr(model, name) for name in required):
        raise TypeError("The connected CLIP is not a native ComfyUI YuE2 text encoder.")
    model.reset_clip_options()
    clip.load_model(tokens)
    device = clip.patcher.load_device
    model.set_clip_options({"layer": None})
    model.set_clip_options({"execution_device": device})
    dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
    return model, device, dtype


def _validate_stream(stream):
    if not isinstance(stream, dict) or stream.get("schema") != "hz3-yue2-token-stream/1":
        raise ValueError("Connect a token_stream produced by HZ3 YuE2 · Generate Token Stream.")
    ids = stream.get("ids")
    if not isinstance(ids, list) or not ids or any(type(token) is not int for token in ids):
        raise ValueError("The YuE2 token stream is empty or malformed.")
    low, high = CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE
    if any(token < low or token >= high for token in ids):
        raise ValueError("The stream contains IDs outside YuE2's semantic codec-token range.")
    return ids


def _slice_stream(stream, start_seconds, duration):
    ids = _validate_stream(stream)
    start = min(len(ids), max(0, round(start_seconds * FRAMES_PER_SECOND)))
    if duration <= 0:
        end = len(ids)
    else:
        end = min(len(ids), start + max(1, round(duration * FRAMES_PER_SECOND)))
    if end <= start:
        raise ValueError("The requested token slice is empty; reduce start_seconds or increase duration.")
    return ids[start:end], start, end


class HZ3_YuE2_GenerateTokenStream:
    CATEGORY = "HZ3 YuE2/Generation"
    FUNCTION = "generate"
    RETURN_TYPES = (STREAM_TYPE, "FLOAT", "STRING")
    RETURN_NAMES = ("token_stream", "seconds", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Generate YuE2 semantic music tokens explicitly so they can be cached, sliced, and reused."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "style": ("STRING", {"multiline": True, "default": ""}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "abc": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "mode": (["full", "melody"], {"default": "full"}),
                "max_duration": ("FLOAT", {"default": 360.0, "min": 0.04, "max": 900.0, "step": 0.04}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 100, "min": 1, "max": 32768}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.01, "max": 10.0, "step": 0.01}),
            }
        }

    def generate(self, clip, style, lyrics, abc, seed, mode, max_duration, temperature, top_p, top_k,
                 repetition_penalty):
        requested = max(1, round(max_duration * FRAMES_PER_SECOND))
        tokens, effective_mode = _tokenize(
            clip, style, lyrics, abc, mode, seed, requested, temperature, top_p, top_k, repetition_penalty
        )
        prefix, negative, abc_ids = _prefixes(tokens)
        max_tokens = min(requested, CONTEXT - max(len(prefix), len(negative)))
        if max_tokens < 1 or len(prefix) + 5 > CONTEXT:
            raise ValueError("YuE2 prompt leaves no room for music tokens; shorten style, lyrics, or ABC.")
        model, device, dtype = _prepare_model(clip, tokens)
        with comfy.model_management.cuda_device_context(device), comfy.ops.use_quantized_matmul(model, device):
            ids, truncated = model._generate(
                prefix,
                seed,
                max_tokens,
                "semantic",
                dtype,
                negative=negative,
                cfg_scale=tokens["cfg_scale"],
                legacy_off=effective_mode == "off",
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                penalty_window=50,
                min_tokens=min(200, max_tokens),
            )
        if not ids:
            raise RuntimeError("YuE2 produced no semantic music tokens.")
        stream = {
            "schema": "hz3-yue2-token-stream/1",
            "ids": ids,
            "frames_per_second": FRAMES_PER_SECOND,
            "seconds": len(ids) / FRAMES_PER_SECOND,
            "seed": seed,
            "mode": effective_mode,
            "truncated": truncated,
            "prompt_hash": _prompt_hash(style, lyrics, abc, effective_mode),
            "abc_token_count": len(abc_ids),
            "sampling": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            },
        }
        report = (
            f"YuE2 semantic stream: {len(ids)} tokens · {len(ids) / FRAMES_PER_SECOND:.2f} s\n"
            f"Mode: {effective_mode} · seed: {seed} · truncated: {truncated}\n"
            f"Prompt SHA-256: {stream['prompt_hash']}"
        )
        return {"ui": {"text": [report]}, "result": (stream, stream["seconds"], report)}


class HZ3_YuE2_MusicFromTokenStream:
    CATEGORY = "HZ3 YuE2/Generation"
    FUNCTION = "condition"
    RETURN_TYPES = ("CONDITIONING", "FLOAT", STREAM_TYPE, "STRING")
    RETURN_NAMES = ("conditioning", "seconds", "sliced_stream", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Build YuE2 acoustic conditioning from an existing semantic stream, optionally using only a timed slice."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "token_stream": (STREAM_TYPE,),
                "style": ("STRING", {"multiline": True, "default": ""}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "abc": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["full", "melody"], {"default": "full"}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 900.0, "step": 0.04}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 900.0, "step": 0.04,
                                      "tooltip": "0 uses the remainder of the stream."}),
            }
        }

    def condition(self, clip, token_stream, style, lyrics, abc, mode, start_seconds, duration):
        ids, start, end = _slice_stream(token_stream, start_seconds, duration)
        # Sampling values are immaterial here: the supplied IDs are never sampled again.
        tokens, effective_mode = _tokenize(clip, style, lyrics, abc, mode, 0, len(ids), 0.0, 1.0, 1, 1.0)
        prefix, _negative, abc_ids = _prefixes(tokens)
        model, device, dtype = _prepare_model(clip, tokens)
        with comfy.model_management.cuda_device_context(device), comfy.ops.use_quantized_matmul(model, device):
            conditioning_tensor, chunks = model._acoustic_conditioning(prefix, ids, dtype)
        comfy.model_prefetch.cleanup_prefetch_queues()
        seconds = len(ids) / FRAMES_PER_SECOND
        source_hash = token_stream.get("prompt_hash", "unknown")
        current_hash = _prompt_hash(style, lyrics, abc, effective_mode)
        sliced = {
            **token_stream,
            "ids": ids,
            "seconds": seconds,
            "slice": {"start_frame": start, "end_frame": end},
            "conditioning_prompt_hash": current_hash,
        }
        metadata = {
            "pooled_output": None,
            "yue2_chunks": chunks,
            "yue2_abc_ids": abc_ids,
            "yue2_frames": len(ids),
            "yue2_truncated": False,
            "hz3_yue2_token_slice": {"start_frame": start, "end_frame": end, "source_prompt_hash": source_hash},
        }
        conditioning = [[conditioning_tensor, metadata]]
        report = (
            f"Reused frames {start}:{end} · {len(ids)} tokens · {seconds:.2f} s\n"
            f"Source prompt: {source_hash}\nConditioning prompt: {current_hash}\n"
            f"Prompt changed: {source_hash != current_hash}"
        )
        return {"ui": {"text": [report]}, "result": (conditioning, seconds, sliced, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_GenerateTokenStream": HZ3_YuE2_GenerateTokenStream,
    "HZ3_YuE2_MusicFromTokenStream": HZ3_YuE2_MusicFromTokenStream,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_GenerateTokenStream": "HZ3 YuE2 · Generate Token Stream",
    "HZ3_YuE2_MusicFromTokenStream": "HZ3 YuE2 · Music From Token Stream",
}
