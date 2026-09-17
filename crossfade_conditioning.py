"""Crossfade + concatenate per-section YuE2 conditionings into one.

`HZ3_YuE2_CrossfadeConditioning` takes the LIST of per-section conditionings (from
each per-section YuE2 Generate Music, like ConcatConditionings) plus the MixMash
`report` (which carries the per-section overlap seconds). It concatenates them and,
at every internal seam, CROSSFADES over the overlap length: the transition ramps
linearly from 100% of the previous section's conditioning to 0% (fully into the next),
so the seam blends smoothly instead of cutting hard. The total length shrinks by the
overlaps (each overlap is shared once as the blend).
"""

from __future__ import annotations

import json

import torch

from comfy.text_encoders.yue2 import FRAMES_PER_SECOND

from .conditioning_edit import _entry


def _semantic_frames(cond):
    """Return (semantic [1, frames, C], first_prefix [1, p, C], last_end [1, 1, C]) of a
    conditioning, by slicing each chunk's acoustic semantic region and keeping the first
    chunk's leading prefix and trailing end token (so the wrap can reuse a real prefix/end)."""
    context, _meta, chunks, frames = _entry(cond, "conditioning")
    parts = []
    first_prefix = None
    last_end = None
    for (start, end, kv_start, kv_end) in chunks:
        frame_count = end - start
        prefix_count = (kv_end - kv_start) - frame_count - 1
        if prefix_count < 1:
            raise ValueError("YuE2 chunk contains no prompt prefix tokens.")
        if first_prefix is None:
            first_prefix = context[:, kv_start:kv_start + prefix_count, :]
        last_end = context[:, kv_end - 1:kv_end, :]
        parts.append(context[:, kv_start + prefix_count:kv_end - 1, :])
    semantic = torch.cat(parts, dim=1)
    if semantic.shape[1] != frames:
        raise ValueError(f"Semantic frames ({semantic.shape[1]}) do not match yue2_frames ({frames}).")
    if first_prefix is None:
        first_prefix = context[:, :1, :]
    if last_end is None:
        last_end = context[:, -1:, :]
    return semantic, first_prefix, last_end


def _overlap_frames(report_json, fallback_seconds, n):
    """Per-section LEADING overlap in frames (index 0 is 0). Parsed from the MixMash
    report's conditioning_overlap.overlap_seconds_per_section; else fallback seconds."""
    seconds = [0.0] * n
    if (report_json or "").strip():
        try:
            payload = json.loads(report_json)
            co = payload.get("conditioning_overlap") if isinstance(payload, dict) else None
            if isinstance(co, dict):
                secs = co.get("overlap_seconds_per_section")
                if isinstance(secs, list):
                    for i, s in enumerate(secs[:n]):
                        try:
                            seconds[i] = float(s)
                        except (TypeError, ValueError):
                            seconds[i] = 0.0
        except json.JSONDecodeError:
            seconds = [0.0] * n
    if all(v <= 0 for v in seconds) and fallback_seconds > 0:
        seconds = [fallback_seconds if i > 0 else 0.0 for i in range(n)]
    return [round(abs(s) * FRAMES_PER_SECOND) for s in seconds]


class HZ3_YuE2_CrossfadeConditioning:
    CATEGORY = "HZ3 YuE2/Conditioning"
    FUNCTION = "crossfade"
    RETURN_TYPES = ("CONDITIONING", "FLOAT", "STRING")
    RETURN_NAMES = ("conditioning", "seconds", "report")
    INPUT_IS_LIST = (True, False, False)
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Concatenate a LIST of per-section conditionings with a LINEAR CROSSFADE at each "
        "seam over the overlap length from the MixMash report (100% previous -> 0% previous). "
        "Total length shrinks by the overlaps. Optionally accept a fallback `overlap` (s) "
        "when the report has no overlap values."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditionings": ("CONDITIONING", {
                    "tooltip": "Connect each per-section conditioning (from YuE2 Generate Music) to this list input; order = connection order.",
                }),
            },
            "optional": {
                "mixmash_report": ("STRING", {
                    "forceInput": True, "multiline": True,
                    "tooltip": "The `report` output of MixMash Genius (carries conditioning_overlap.overlap_seconds_per_section).",
                }),
                "fallback_overlap": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 120.0, "step": 0.5,
                                               "tooltip": "Used when the report has no overlap values."}),
            },
        }

    def crossfade(self, conditionings, mixmash_report="", fallback_overlap=4.0):
        conds = [c for c in (conditionings or []) if c is not None]
        if not conds:
            raise ValueError("Connect at least one per-section conditioning to crossfade.")
        if isinstance(mixmash_report, (list, tuple)):
            mixmash_report = mixmash_report[0] if mixmash_report else ""
        if not isinstance(mixmash_report, str):
            mixmash_report = str(mixmash_report or "")
        if isinstance(fallback_overlap, (list, tuple)):
            fallback_overlap = fallback_overlap[0] if fallback_overlap else 4.0
        try:
            fallback_overlap = float(fallback_overlap)
        except (TypeError, ValueError):
            fallback_overlap = 4.0

        n = len(conds)
        if n == 1:
            _ctx, _meta, _chunks, frames_count = _entry(conds[0], "conditioning")
            seconds = frames_count / float(FRAMES_PER_SECOND)
            report = f"Single conditioning · {frames_count} frames · {seconds:.2f} s (no seams to crossfade)"
            return {"ui": {"text": [report]}, "result": (conds[0], seconds, report)}

        overlap_frames_list = _overlap_frames(mixmash_report, fallback_overlap, n)

        sems = []
        prefixes = []
        end_tokens = []
        base_meta = None
        for index, cond in enumerate(conds, 1):
            sem, pref, end = _semantic_frames(cond)
            sems.append(sem)
            prefixes.append(pref)
            end_tokens.append(end)
            if base_meta is None:
                base_meta = dict(_entry(cond, f"conditioning {index}")[1])
            if sem.shape[1] < 1:
                raise ValueError(f"Conditioning {index} has no semantic frames.")

        frames = [s.shape[1] for s in sems]
        device, dtype, feat = sems[0].device, sems[0].dtype, sems[0].shape[2]

        o_used = [0] * n
        for i in range(1, n):
            max_prev = frames[i - 1] // 2 if (i - 1 > 0) else frames[i - 1]
            max_curr = frames[i] // 2 if (i < n - 1) else frames[i]
            o_used[i] = max(0, min(overlap_frames_list[i], max_prev, max_curr))

        # Build blended transitions across each internal seam
        blends_left = [None] * n
        blends_right = [None] * n
        for i in range(1, n):
            o = o_used[i]
            if o > 0:
                tail = sems[i - 1][:, -o:, :].to(device=device, dtype=dtype)
                head = sems[i][:, :o, :].to(device=device, dtype=dtype)
                w = torch.linspace(1.0, 0.0, o, device=device, dtype=dtype).view(1, o, 1)
                blend = w * tail + (1.0 - w) * head
                o_left = o // 2
                o_right = o - o_left
                blends_left[i] = blend[:, :o_left, :]
                blends_right[i] = blend[:, o_left:, :]
            else:
                blends_left[i] = torch.empty((1, 0, feat), device=device, dtype=dtype)
                blends_right[i] = torch.empty((1, 0, feat), device=device, dtype=dtype)

        # Assemble crossfaded acoustic semantic frames for each section
        s_primes = []
        for i in range(n):
            parts = []
            if i > 0:
                parts.append(blends_right[i])
                mid_start = o_used[i]
            else:
                mid_start = 0

            if i < n - 1:
                mid_end = frames[i] - o_used[i + 1]
            else:
                mid_end = frames[i]

            parts.append(sems[i][:, mid_start:mid_end, :].to(device=device, dtype=dtype))

            if i < n - 1:
                parts.append(blends_left[i + 1])

            s_primes.append(torch.cat(parts, dim=1))

        # Build multi-chunk conditioning preserving each section's prompt prefix and end token
        pieces = []
        chunks = []
        start_frame = 0
        kv_cursor = 0

        for i in range(n):
            pref = prefixes[i].to(device=device, dtype=dtype)
            sem = s_primes[i]
            end = end_tokens[i].to(device=device, dtype=dtype)

            piece = torch.cat([pref, sem, end], dim=1)
            pieces.append(piece)

            chunk_start = start_frame
            chunk_end = start_frame + sem.shape[1]
            kv_start = kv_cursor
            kv_end = kv_cursor + piece.shape[1]

            chunks.append((chunk_start, chunk_end, kv_start, kv_end))
            start_frame = chunk_end
            kv_cursor = kv_end

        assembled_context = torch.cat(pieces, dim=1)
        total_frames = chunks[-1][1]

        metadata = dict(base_meta)
        metadata["yue2_chunks"] = tuple(chunks)
        metadata["yue2_frames"] = total_frames
        metadata["yue2_truncated"] = False
        metadata["hz3_crossfade"] = {
            "sections": n,
            "overlap_frames_per_section": o_used,
            "section_frames": [s.shape[1] for s in s_primes],
        }

        conditioning = [[assembled_context, metadata]]
        seconds = total_frames / float(FRAMES_PER_SECOND)

        # Build detailed timeline report
        timeline = []
        for i, ch in enumerate(chunks, 1):
            s_sec = ch[0] / float(FRAMES_PER_SECOND)
            e_sec = ch[1] / float(FRAMES_PER_SECOND)
            ov_info = f"overlap in={o_used[i-1]/float(FRAMES_PER_SECOND):.2f}s" if i > 1 else "no overlap (start)"
            timeline.append(f"  Section {i}: {s_sec:.2f}s - {e_sec:.2f}s ({ch[1]-ch[0]} frames) [{ov_info}]")

        report = (
            f"Crossfaded {n} section conditioning(s) into multi-chunk YuE2 conditioning · "
            f"{total_frames} frames · {seconds:.2f} s\n"
            f"Overlap frames per section: {o_used}\n"
            f"Timeline:\n" + "\n".join(timeline)
        )
        return {"ui": {"text": [report]}, "result": (conditioning, seconds, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_CrossfadeConditioning": HZ3_YuE2_CrossfadeConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_CrossfadeConditioning": "HZ3 YuE2 · Crossfade Audio Conditioning",
}
