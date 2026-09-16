"""General audio list -> batch / concatenation helper for YuE2 pipelines.

Takes up to 8 AUDIO inputs (each may itself already be a batch) and combines them
either by stacking along the batch dimension ("batch", padding to the longest
sample count) or by joining along the time axis into a single clip
("concatenate"). The result is always exposed on the first output slot.

This lets you bundle several source clips into one batched AUDIO so a single
downstream node (e.g. SheetSage2 audio-to-ABC) processes them in one run.
"""

from __future__ import annotations

import torch


class HZ3_YuE2_ConcatAudio:
    CATEGORY = "HZ3 YuE2/Audio"
    FUNCTION = "concat"
    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "count", "report")
    DESCRIPTION = (
        "Concatenate up to 8 AUDIO inputs (each may already be a batch). Modo 'batch' "
        "stack them into one batched AUDIO (padded to the longest sample count) so a "
        "downstream SheetSage2 node processes them in a single run; modo 'concatenate' "
        "joins them along the time axis into one clip. Result is output [0]."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["batch", "concatenate"], {
                    "default": "batch",
                    "tooltip": "batch: apila en la dimensión de lote (pad al más largo). concatenate: une en el tiempo en un solo clip.",
                }),
            },
            "optional": {
                **{f"audio_{index}": ("AUDIO",) for index in range(1, 9)},
            },
        }

    def concat(self, mode, **kwargs):
        audios = [kwargs[f"audio_{index}"] for index in range(1, 9)
                  if kwargs.get(f"audio_{index}") is not None]
        if not audios:
            raise ValueError("Connect at least one audio input to concatenate.")

        sample_rate = int(audios[0]["sample_rate"])
        for index, audio in enumerate(audios, 1):
            if int(audio["sample_rate"]) != sample_rate:
                raise ValueError(f"All audio inputs must share the same sample_rate (input {index} does not).")

        clips = []
        channels = None
        for audio in audios:
            waveform = audio["waveform"]                 # [batch, C, samples]
            if channels is None:
                channels = waveform.shape[1]
            elif waveform.shape[1] != channels:
                raise ValueError("All audio inputs must have the same channel count.")
            for index in range(waveform.shape[0]):
                clips.append(waveform[index])            # [C, samples]
        if not clips:
            raise ValueError("No audio clips were produced from the connected inputs.")
        device, dtype = clips[0].device, clips[0].dtype
        count = len(clips)

        if mode == "concatenate":
            total = sum(clip.shape[-1] for clip in clips)
            output = torch.zeros((1, channels, total), device=device, dtype=dtype)
            cursor = 0
            for clip in clips:
                length = clip.shape[-1]
                output[0, :, cursor:cursor + length].copy_(clip)
                cursor += length
            report = f"Concatenated {count} clip(s) along time -> 1 audio · {total / sample_rate:.2f} s"
        else:  # batch (stack/pad)
            longest = max(clip.shape[-1] for clip in clips)
            output = torch.zeros((count, channels, longest), device=device, dtype=dtype)
            for index, clip in enumerate(clips):
                output[index, :, : clip.shape[-1]].copy_(clip)
            report = f"Stacked {count} clip(s) into 1 batched AUDIO ({count} segments, padded to {longest / sample_rate:.2f} s)"

        result_audio = {"waveform": output, "sample_rate": sample_rate}
        return {"ui": {"text": [report]}, "result": (result_audio, count, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_ConcatAudio": HZ3_YuE2_ConcatAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_ConcatAudio": "HZ3 YuE2 · Concat Audio List",
}
