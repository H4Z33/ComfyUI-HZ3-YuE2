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
    """Return the contiguous semantic [1, frames, C] tensor of a conditioning,
    and the [1, prefix_len, C] prefix of its first chunk."""
    context, _meta, chunks, frames = _entry(cond, "conditioning")
    parts = []
    first_prefix = None
    for (start, end, kv_start, kv_end) in chunks:
        prefix_count = (kv_end - kv_start) - (end - start) - 1
        if first_prefix is None:
            first_prefix = context[:, kv_start:kv_start + prefix_count, :]
        parts.append(context[:, kv_start + prefix_count:kv_end - 1, :])
    semantic = torch.cat(parts, dim=1)
    if semantic.shape[1] != frames:
        raise ValueError("Semantic frames do not match yue2_frames")
    if first_prefix is None:
        first_prefix = context[:, :1, :]
    return semantic, first_prefix


def _blend(left, right):
    """Linear crossfade: weight goes 1 -> 0 over the overlap from left to right."""
    o = left.shape[1]
    weight = torch.linspace(1.0, 0.0, o, device=left.device, dtype=left.dtype)
    weight = weight.view(1, o, 1)
    return weight * left + (1.0 - weight) * right


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
        try:
            fallback_overlap = float(fallback_overlap[0]) if isinstance(fallback_overlap, (list, tuple)) else float(fallback_overlap)
        except (TypeError, ValueError, IndexError):
            fallback_overlap = 4.0

        seams = len(conds) - 1
        overlap_frames_list = _overlap_frames(mixmash_report, fallback_overlap, len(conds))

        sems = []
        prefixes = []
        for cond in conds:
            sem, pref = _semantic_frames(cond)
            sems.append(sem)
            prefixes.append(pref)
            if sem.shape[1] < 1:
                raise ValueError("A connected conditioning has no semantic frames.")

        # Crossfade: keep section 0, then at each seam blend the previous tail with the
        # next head over `overlap` frames, dropping the old tail and appending the rest.
        out = sems[0]
        used = [0.0] * len(conds)
        for index in range(seams):
            left = out
            right = sems[index + 1]
            o = min(overlap_frames_list[index + 1], left.shape[1], right.shape[1])
            if o <= 0:
                out = torch.cat([out, right], dim=1)
                continue
            blend = _blend(left[:, -o:, :], right[:, :o, :])
            out = torch.cat([left[:, :-o, :], blend, right[:, o:, :]], dim=1)
            used[index + 1] = o

        feat = out.shape[2]
        prefix = prefixes[0]
        prefix_len = prefix.shape[1]
        end_token = prefix[:, :1, :] * 0.0
        context = torch.cat([prefix, out, end_token], dim=1)
        total_frames = out.shape[1]
        base_meta = dict(_entry(conds[0], "first")[1])
        metadata = dict(base_meta)
        metadata["yue2_chunks"] = ((0, total_frames, 0, context.shape[1]),)
        metadata["yue2_frames"] = total_frames
        metadata["yue2_abc_ids"] = base_meta.get("yue2_abc_ids", [])
        metadata["hz3_crossfade"] = {
            "overlap_frames_per_section": used,
            "sections": len(conds),
        }
        conditioning = [[context, metadata]]
        seconds = total_frames / float(FRAMES_PER_SECOND)
        report = (
            f"Crossfaded {len(conds)} conditionings · {total_frames} frames · {seconds:.2f} s\n"
            f"overlap frames per section: {used}\n"
            f"(100% prev -> 0% prev over each overlap; total shrank by {sum(used)} frames)"
        )
        return {"ui": {"text": [report]}, "result": (conditioning, seconds, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_CrossfadeConditioning": HZ3_YuE2_CrossfadeConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_CrossfadeConditioning": "HZ3 YuE2 · Crossfade Audio Conditioning",
}
