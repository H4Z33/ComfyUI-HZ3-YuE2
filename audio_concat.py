"""Collapse a batched AUDIO into one continuous audio clip.

The Split Audio by Segments node emits its segments as a single batched AUDIO
(waveform [N, C, samples]). This node reverses that: it joins all N segments
back together along the time axis into one clip (batch=1), preserving the
sample rate. The result is exposed on the first output slot.
"""

from __future__ import annotations

import torch


class HZ3_YuE2_ConcatAudioSegments:
    CATEGORY = "HZ3 YuE2/Audio"
    FUNCTION = "concat"
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "seconds", "report")
    DESCRIPTION = (
        "Convert a batched AUDIO (e.g. segments_audio from Split Audio by Segments) "
        "into a single continuous audio clip (batch=1), joining the segments along the "
        "time axis and preserving the sample rate."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    def concat(self, audio):
        waveform = audio["waveform"]                 # [N, C, samples]
        sample_rate = int(audio["sample_rate"])
        if waveform.shape[0] < 1:
            raise ValueError("Connect a non-empty AUDIO to concatenate.")
        count = waveform.shape[0]
        if count == 1:
            result = dict(audio)
            seconds = waveform.shape[-1] / float(sample_rate)
            report = f"Single segment already · {seconds:.2f} s (no concatenation needed)"
            return {"ui": {"text": [report]}, "result": (result, seconds, report)}

        channels = waveform.shape[1]
        total = sum(waveform[index].shape[-1] for index in range(count))
        output = torch.zeros(
            (1, channels, total), dtype=waveform.dtype, device=waveform.device
        )
        cursor = 0
        for index in range(count):
            clip = waveform[index]
            length = clip.shape[-1]
            output[0, :, cursor:cursor + length].copy_(clip)
            cursor += length

        seconds = total / float(sample_rate)
        result_audio = {"waveform": output, "sample_rate": sample_rate}
        report = (
            f"Joined {count} segment(s) ({count - 1} join(s)) into 1 continuous "
            f"audio · {seconds:.2f} s"
        )
        return {"ui": {"text": [report]}, "result": (result_audio, seconds, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_ConcatAudioSegments": HZ3_YuE2_ConcatAudioSegments,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_ConcatAudioSegments": "HZ3 YuE2 · Concat Audio Segments",
}
