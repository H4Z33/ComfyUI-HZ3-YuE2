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

        # RAW tensors: preallocate a final tensor (zeros = padding), place each
        # conditioning's tensor at its offset, and AVERAGE where more than one sits.
        n = len(conds)
        tensors = []
        frames = []
        base_meta = None
        for cond in conds:
            ctx, meta, _chunks, frm = _entry(cond, "conditioning")
            tensors.append(ctx)
            frames.append(frm)
            if base_meta is None:
                base_meta = dict(meta)
            if frm < 1:
                raise ValueError("A connected conditioning has no frames.")

        o_used = [0] * n
        offsets = [0] * n
        for i in range(1, n):
            o_used[i] = min(overlap_frames_list[i], frames[i - 1], frames[i])
            offsets[i] = offsets[i - 1] + tensors[i - 1].shape[1] - o_used[i]
        final_len = offsets[-1] + tensors[-1].shape[1]
        final_frames = sum(frames) - sum(o_used)
        device, dtype, feat = tensors[0].device, tensors[0].dtype, tensors[0].shape[2]

        acc = torch.zeros(1, final_len, feat, device=device, dtype=dtype)
        count = torch.zeros(1, final_len, 1, device=device, dtype=dtype)
        for i, t in enumerate(tensors):
            off = offsets[i]
            acc[:, off:off + t.shape[1], :] += t
            count[:, off:off + t.shape[1], :] += 1
        out = (acc / count.clamp(min=1e-8)) * (count > 0)   # average where overlap, else 0/unit

        metadata = dict(base_meta)
        metadata["yue2_chunks"] = ((0, final_frames, 0, final_len),)
        metadata["yue2_frames"] = final_frames
        metadata["yue2_abc_ids"] = base_meta.get("yue2_abc_ids", [])
        metadata["hz3_crossfade"] = {"overlap_frames_per_section": o_used, "sections": n}
        conditioning = [[out, metadata]]
        seconds = final_frames / float(FRAMES_PER_SECOND)
        report = (
            f"Crossfaded {len(conds)} conditionings · {final_frames} frames · {seconds:.2f} s\n"
            f"overlap frames per section: {o_used}\n"
            f"(raw tensor placement; average where overlap; 0 padding)"
        )
        return {"ui": {"text": [report]}, "result": (conditioning, seconds, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_CrossfadeConditioning": HZ3_YuE2_CrossfadeConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_CrossfadeConditioning": "HZ3 YuE2 · Crossfade Audio Conditioning",
}
