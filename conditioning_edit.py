"""Time-aware cutting and replacement of native YuE2 conditioning chunks."""

from __future__ import annotations

import copy

import torch

from comfy.text_encoders.yue2 import FRAMES_PER_SECOND


def _entry(conditioning, label):
    if not isinstance(conditioning, list) or len(conditioning) != 1:
        raise ValueError(f"{label} must contain exactly one YuE2 conditioning entry.")
    entry = conditioning[0]
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise ValueError(f"{label} has an invalid ComfyUI CONDITIONING structure.")
    context, metadata = entry
    if not torch.is_tensor(context) or context.ndim != 3:
        raise ValueError(f"{label} must contain a [batch, tokens, features] tensor.")
    if not isinstance(metadata, dict):
        raise ValueError(f"{label} metadata is missing.")
    chunks = metadata.get("yue2_chunks")
    frames = metadata.get("yue2_frames")
    if not isinstance(chunks, (list, tuple)) or not chunks:
        raise ValueError(f"{label} does not contain native yue2_chunks metadata.")
    if type(frames) is not int or frames < 1:
        raise ValueError(f"{label} does not contain a valid yue2_frames value.")
    if chunks[-1][1] != frames:
        raise ValueError(f"{label} frame count disagrees with its final YuE2 chunk.")
    return context, metadata, tuple(tuple(int(value) for value in chunk) for chunk in chunks), frames


def _chunk_layout(context, chunk, label):
    start, end, kv_start, kv_end = chunk
    frame_count = end - start
    kv_count = kv_end - kv_start
    prefix_count = kv_count - frame_count - 1
    if not (0 <= start < end and 0 <= kv_start < kv_end <= context.shape[1] and prefix_count >= 1):
        raise ValueError(f"{label} contains an inconsistent YuE2 chunk: {chunk}.")
    return prefix_count


def _extract_parts(context, chunks, source_start, source_end, output_start, label):
    """Return independent valid chunks for an interval, preserving each source prefix."""
    tensors = []
    output_chunks = []
    kv_cursor = 0
    for chunk in chunks:
        chunk_start, chunk_end, kv_start, kv_end = chunk
        take_start = max(source_start, chunk_start)
        take_end = min(source_end, chunk_end)
        if take_end <= take_start:
            continue
        prefix_count = _chunk_layout(context, chunk, label)
        semantic_start = kv_start + prefix_count + (take_start - chunk_start)
        semantic_end = semantic_start + (take_end - take_start)
        piece = torch.cat((
            context[:, kv_start:kv_start + prefix_count],
            context[:, semantic_start:semantic_end],
            context[:, kv_end - 1:kv_end],
        ), dim=1)
        target_start = output_start + (take_start - source_start)
        target_end = target_start + (take_end - take_start)
        tensors.append(piece)
        output_chunks.append((target_start, target_end, kv_cursor, kv_cursor + piece.shape[1]))
        kv_cursor += piece.shape[1]
    if not tensors:
        raise ValueError(f"{label} interval does not intersect any YuE2 conditioning chunk.")
    return tensors, output_chunks


def _assemble(base_metadata, tensors, chunks, frames):
    context = torch.cat(tensors, dim=1)
    metadata = copy.copy(base_metadata)
    metadata["yue2_chunks"] = tuple(chunks)
    metadata["yue2_frames"] = frames
    metadata["yue2_truncated"] = False
    metadata["hz3_conditioning_edit"] = True
    return [[context, metadata]]


def cut_conditioning(conditioning, start_seconds, duration):
    context, metadata, chunks, frames = _entry(conditioning, "conditioning")
    start = min(frames - 1, max(0, round(start_seconds * FRAMES_PER_SECOND)))
    end = frames if duration <= 0 else min(frames, start + max(1, round(duration * FRAMES_PER_SECOND)))
    tensors, output_chunks = _extract_parts(context, chunks, start, end, 0, "conditioning")
    result = _assemble(metadata, tensors, output_chunks, end - start)
    result[0][1]["hz3_conditioning_cut"] = {"source_start_frame": start, "source_end_frame": end}
    return result, start, end


def paste_conditioning(base, donor, start_seconds, donor_start_seconds, duration):
    base_context, base_metadata, base_chunks, base_frames = _entry(base, "base")
    donor_context, _donor_metadata, donor_chunks, donor_frames = _entry(donor, "donor")
    if base_context.shape[0] != donor_context.shape[0] or base_context.shape[2] != donor_context.shape[2]:
        raise ValueError("Base and donor conditioning tensors have incompatible batch or feature dimensions.")
    base_start = min(base_frames - 1, max(0, round(start_seconds * FRAMES_PER_SECOND)))
    donor_start = min(donor_frames - 1, max(0, round(donor_start_seconds * FRAMES_PER_SECOND)))
    available = min(base_frames - base_start, donor_frames - donor_start)
    replace_frames = available if duration <= 0 else min(available, max(1, round(duration * FRAMES_PER_SECOND)))
    if replace_frames < 1:
        raise ValueError("No overlapping frames are available for the requested paste.")
    base_end = base_start + replace_frames
    donor_end = donor_start + replace_frames

    tensors = []
    chunks = []
    kv_cursor = 0

    def append(source_context, source_chunks, source_start, source_end, target_start, label):
        nonlocal kv_cursor
        if source_end <= source_start:
            return
        parts, part_chunks = _extract_parts(
            source_context, source_chunks, source_start, source_end, target_start, label
        )
        for part, chunk in zip(parts, part_chunks):
            start, end, local_kv_start, local_kv_end = chunk
            tensors.append(part.to(device=base_context.device, dtype=base_context.dtype))
            chunks.append((start, end, kv_cursor + local_kv_start, kv_cursor + local_kv_end))
        kv_cursor += sum(part.shape[1] for part in parts)

    append(base_context, base_chunks, 0, base_start, 0, "base before interval")
    append(donor_context, donor_chunks, donor_start, donor_end, base_start, "donor interval")
    append(base_context, base_chunks, base_end, base_frames, base_end, "base after interval")
    result = _assemble(base_metadata, tensors, chunks, base_frames)
    result[0][1]["hz3_conditioning_paste"] = {
        "base_start_frame": base_start,
        "base_end_frame": base_end,
        "donor_start_frame": donor_start,
        "donor_end_frame": donor_end,
    }
    return result, base_start, base_end, donor_start, donor_end


class HZ3_YuE2_ConditioningCut:
    CATEGORY = "HZ3 YuE2/Conditioning"
    FUNCTION = "cut"
    RETURN_TYPES = ("CONDITIONING", "FLOAT", "STRING")
    RETURN_NAMES = ("conditioning", "seconds", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Cut native YuE2 conditioning by semantic frames while preserving valid per-chunk prefixes."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 900.0, "step": 0.04}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 900.0, "step": 0.04,
                                      "tooltip": "0 uses the remainder of the conditioning."}),
            }
        }

    def cut(self, conditioning, start_seconds, duration):
        result, start, end = cut_conditioning(conditioning, start_seconds, duration)
        seconds = (end - start) / FRAMES_PER_SECOND
        report = f"Cut frames {start}:{end} · {end - start} frames · {seconds:.2f} s"
        return {"ui": {"text": [report]}, "result": (result, seconds, report)}


class HZ3_YuE2_ConditioningPaste:
    CATEGORY = "HZ3 YuE2/Conditioning"
    FUNCTION = "paste"
    RETURN_TYPES = ("CONDITIONING", "FLOAT", "STRING")
    RETURN_NAMES = ("conditioning", "seconds", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Replace a timed interval of one YuE2 conditioning with frames from another generation."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("CONDITIONING",),
                "donor": ("CONDITIONING",),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 900.0, "step": 0.04}),
                "donor_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 900.0, "step": 0.04}),
                "duration": ("FLOAT", {"default": 4.0, "min": 0.04, "max": 900.0, "step": 0.04}),
            }
        }

    def paste(self, base, donor, start_seconds, donor_start_seconds, duration):
        result, base_start, base_end, donor_start, donor_end = paste_conditioning(
            base, donor, start_seconds, donor_start_seconds, duration
        )
        seconds = result[0][1]["yue2_frames"] / FRAMES_PER_SECOND
        report = (
            f"Base frames {base_start}:{base_end} <- donor frames {donor_start}:{donor_end}\n"
            f"Replaced {(base_end - base_start) / FRAMES_PER_SECOND:.2f} s · output remains {seconds:.2f} s"
        )
        return {"ui": {"text": [report]}, "result": (result, seconds, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_ConditioningCut": HZ3_YuE2_ConditioningCut,
    "HZ3_YuE2_ConditioningPaste": HZ3_YuE2_ConditioningPaste,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_ConditioningCut": "HZ3 YuE2 · Conditioning Cut",
    "HZ3_YuE2_ConditioningPaste": "HZ3 YuE2 · Conditioning Paste",
}
