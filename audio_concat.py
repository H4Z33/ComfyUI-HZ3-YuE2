"""Collapse a batch or list of AUDIO into one continuous audio clip.

Split Audio by Segments now emits its clips as a LIST of AUDIO each at its true
length (no padding). This node accepts either that list, or a single batched
AUDIO, and joins everything along the time axis into one clip (batch=1),
preserving the sample rate. The result is on output [0].
"""

from __future__ import annotations

import torch


def _collect(audio_list):
    """Yield individual [C, samples] clips from a list of AUDIO or one batched AUDIO."""
    items = audio_list if isinstance(audio_list, (list, tuple)) else [audio_list]
    for audio in items:
        if not (isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio):
            raise ValueError("Each element must be an AUDIO dict with 'waveform' and 'sample_rate'.")
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        for index in range(waveform.shape[0]):
            yield audio, waveform[index], sample_rate


class HZ3_YuE2_ConcatAudioSegments:
    CATEGORY = "HZ3 YuE2/Audio"
    FUNCTION = "concat"
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "seconds", "report")
    DESCRIPTION = (
        "Join a LIST of AUDIO (e.g. segments_audio from Split Audio by Segments) or a "
        "single batched AUDIO into one continuous clip (batch=1), preserving sample rate."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    def concat(self, audio):
        entries = list(_collect(audio))
        if not entries:
            raise ValueError("Connect a non-empty AUDIO (or list of AUDIO) to concatenate.")

        sample_rate = entries[0][2]
        for _, _, sr in entries:
            if sr != sample_rate:
                raise ValueError("All audio clips must share the same sample_rate.")

        clips = [clip for _, clip, _ in entries]
        if len(clips) == 1:
            result = dict(entries[0][0])
            seconds = clips[0].shape[-1] / float(sample_rate)
            report = f"Single clip already · {seconds:.2f} s (no concatenation needed)"
            return {"ui": {"text": [report]}, "result": (result, seconds, report)}

        channels = clips[0].shape[0]
        for clip in clips:
            if clip.shape[0] != channels:
                raise ValueError("All audio clips must have the same channel count.")
        total = sum(clip.shape[-1] for clip in clips)
        device, dtype = clips[0].device, clips[0].dtype
        output = torch.zeros((1, channels, total), device=device, dtype=dtype)
        cursor = 0
        for clip in clips:
            length = clip.shape[-1]
            output[0, :, cursor:cursor + length].copy_(clip)
            cursor += length

        seconds = total / float(sample_rate)
        result_audio = {"waveform": output, "sample_rate": sample_rate}
        report = (
            f"Joined {len(clips)} clip(s) ({len(clips) - 1} join(s)) into 1 continuous "
            f"audio · {seconds:.2f} s"
        )
        return {"ui": {"text": [report]}, "result": (result_audio, seconds, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_ConcatAudioSegments": HZ3_YuE2_ConcatAudioSegments,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_ConcatAudioSegments": "HZ3 YuE2 · Concat Audio Segments",
}
