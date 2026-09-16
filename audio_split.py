"""Split one audio clip into segment clips from a whisper/faster-whisper segment list.

The audio is cut at the whisper segment boundaries and the segments are packed into
a single batched AUDIO (batch index per segment), so a downstream SheetSage2
audio-to-ABC node transcribes every segment in one run (it iterates the batch).

Boundary rule (requested): the FIRST output segment spans the whole pre-roll up to
the end of the first segment (it contains everything before the first start), and
the LAST output segment spans from the last segment start to the very end of the
audio (the remainder after the last end). Middle segments are cut exactly to their
whisper start/end. Provided the whisper boundaries are contiguous this tiles the
whole file with no gaps.
"""

from __future__ import annotations

import json

import torch


def _as_seconds(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_segments(data):
    """Return a sorted list of {start, end, text} from a JSON segment payload.

    Accepts either a JSON array of {start, end, text} objects or an object wrapping
    such an array under "segments" / "results".
    """
    if isinstance(data, str):
        payload = json.loads(data) if data.strip() else []
    else:
        payload = data
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = payload.get("segments") or payload.get("results") or []
    else:
        raise ValueError("segments must be a JSON list of {start, end, text} objects.")
    if not isinstance(raw, list):
        raise ValueError("segments must be a JSON list of {start, end, text} objects.")

    segments = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = _as_seconds(item.get("start"), None)
        end = _as_seconds(item.get("end"), None)
        if start is None or end is None:
            raise ValueError("Each segment needs numeric start and end fields.")
        text = str(item.get("text", "") or "").strip()
        if end <= start:
            raise ValueError(f"Segment has end <= start ({start}..{end}); fix the boundaries.")
        segments.append({"start": start, "end": end, "text": text})
    segments.sort(key=lambda segment: segment["start"])
    if not segments:
        raise ValueError("No valid segments were parsed from the segment payload.")
    return segments


def _slice_plan(segments, duration_seconds):
    """Build {(start_s, end_s, text)} covering the whole file with first/last merge.

    - First segment starts at 0 (grabbing the pre-roll before the first start).
    - Last segment ends at the very end of the audio (grabbing the post-roll).
    - Middle segments are the raw whisper intervals.
    """
    last = len(segments) - 1
    plan = []
    prev_end = 0.0
    for index, segment in enumerate(segments):
        start = 0.0 if index == 0 else max(prev_end, segment["start"])
        end = duration_seconds if index == last else segment["end"]
        end = min(duration_seconds, end)
        if end <= start:
            continue
        plan.append((start, end, segment["text"]))
        prev_end = end
    if not plan:
        raise ValueError("No audible interval remains to split; check the segment boundaries.")
    return plan


class HZ3_YuE2_SplitAudioSegments:
    CATEGORY = "HZ3 YuE2/Audio"
    FUNCTION = "split"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("segments_audio", "layout", "report")
    DESCRIPTION = (
        "Slice one audio into segment clips from a fast-whisper segment list and pack them "
        "into a single batched AUDIO (one segment per batch index). The first clip includes "
        "everything before the first start and the last clip the remainder after the last "
        "end, so the whole file is covered. Connect segments_audio to a single SheetSage2 "
        "audio-to-ABC node to transcribe all segments in one pass."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "segments": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "JSON list of {start, end, text} from faster-whisper (or wrap under 'segments').",
                }),
            },
        }

    def split(self, audio, segments):
        waveform = audio["waveform"]          # [batch, channels, samples]
        sample_rate = int(audio["sample_rate"])
        if waveform.shape[0] != 1:
            raise ValueError("Expected a single audio clip (batch of 1) to split.")
        channels = waveform.shape[1]
        total = waveform.shape[-1]
        duration_seconds = total / float(sample_rate)

        parsed = _parse_segments(segments)
        plan = _slice_plan(parsed, duration_seconds)

        clips = []
        layout = []
        for index, (start, end, text) in enumerate(plan):
            i0 = max(0, round(start * sample_rate))
            i1 = min(total, round(end * sample_rate))
            if i1 <= i0:
                continue
            clip = waveform[:, :, i0:i1]      # [1, channels, n_samples]
            clips.append(clip)
            layout.append({
                "index": len(clips) - 1,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round((end - start), 3),
                "text": text,
            })
        if not clips:
            raise ValueError("None of the segments produced an audible audio clip.")

        # Pack into one batched AUDIO: pad to the longest segment so the tensor
        # stays rectangular. SheetSage2 pads its input window anyway; the short
        # trailing silence on shorter segments yields only trailing silence/EOS.
        max_samples = max(clip.shape[-1] for clip in clips)
        batch = len(clips)
        output = torch.zeros(
            (batch, channels, max_samples),
            dtype=waveform.dtype,
            device=waveform.device,
        )
        for index, clip in enumerate(clips):
            output[index:index + 1, :, : clip.shape[-1]].copy_(clip)

        layout_json = json.dumps(layout, ensure_ascii=False, indent=2)
        report = (
            f"Split audio ({duration_seconds:.2f} s) into {batch} segment(s): "
            + "\n".join(
                f"  [{j['index']}] {j['start']:.2f}-{j['end']:.2f}s ({j['duration']:.2f}s) {j['text']}"
                for j in layout
            )
        )
        result_audio = {"waveform": output, "sample_rate": sample_rate}
        return {"ui": {"text": [layout_json]}, "result": (result_audio, layout_json, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_SplitAudioSegments": HZ3_YuE2_SplitAudioSegments,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_SplitAudioSegments": "HZ3 YuE2 · Split Audio by Segments",
}
