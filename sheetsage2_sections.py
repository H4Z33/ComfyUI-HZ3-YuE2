"""Our own SheetSage2 Audio -> ABC node that also exposes SECTION STRUCTURE.

SheetSage2 is run on the whole audio once; in addition to the ABC text we surface
the structure events (Intro/Verse/Chorus/Interlude/Outro...) with their REAL
timestamps in audio seconds, plus the audio cut into per-section clips. This is the
accurate source the segmenter needs (whisper alone cannot place the Interlude that
is hidden inside a long lyric line).
"""

from __future__ import annotations

import json

import torch


# Non-musical boundary labels: adjacent duplicates collapse into one span.
_MERGEABLE_STRUCTURE = frozenset({
    "intro", "outro", "interlude", "silence", "instrumental", "fade-out",
})


def structure_timeline(events, duration):
    """Build [{name, start, end}] section timeline (real seconds) from structure events.

    Every distinct structure event starts a section. Consecutive events with the SAME
    label are merged when that label is a non-musical boundary marker
    (intro/outro/interlude/silence/instrumental/fade-out), so a duplicated "intro"
    becomes a single intro. Repeatable musical labels (verse/chorus/bridge/...) stay
    separate and are numbered (verse, verse 2, ...).
    """
    rows = []
    for event in events:
        values = event.get("values", {})
        if "structure" in values:
            rows.append((float(event.get("time", 0.0)), str(values["structure"])))
    if not rows:
        return [{"name": "section", "start": 0.0, "end": float(duration)}]
    rows.sort(key=lambda item: item[0])

    kept = []
    for time, label in rows:
        if kept and abs(time - kept[-1][0]) < 1e-6:
            continue                              # de-duplicate same-instant events
        if kept and kept[-1][1] == label and label in _MERGEABLE_STRUCTURE:
            continue                              # fold repeated boundary label
        kept.append((time, label))

    timeline = []
    counters = {}
    for index, (time, label) in enumerate(kept):
        end = kept[index + 1][0] if index + 1 < len(kept) else float(duration)
        counters[label] = counters.get(label, 0) + 1
        name = label if counters[label] == 1 else f"{label} {counters[label]}"
        timeline.append({"name": name, "start": round(float(time), 3),
                         "end": round(float(end), 3)})
    return timeline


def cut_sections(audio, structure):
    """Cut one audio (AUDIO dict) into a list of AUDIO clips at section boundaries.

    Ensures full coverage: a leading section (intro) is prepended when the first
    section starts after 0, and a trailing section (outro) is appended when the
    last one stops before the end.
    """
    waveform = audio["waveform"]                       # [batch, C, samples]
    sample_rate = int(audio["sample_rate"])
    total = waveform.shape[-1]
    duration = total / float(sample_rate)
    if waveform.shape[0] != 1:
        raise ValueError("Expected a single audio clip (batch of 1).")

    secs = [s for s in structure if float(s["end"]) > float(s["start"])]
    for s in secs:
        s["start"] = max(0.0, min(duration, float(s["start"])))
        s["end"] = max(0.0, min(duration, float(s["end"])))
    if not secs:
        secs = [{"name": "section", "start": 0.0, "end": duration}]
    if secs and secs[0]["start"] > 1e-3:
        if secs[0]["start"] <= 2.0:
            # Tiny lead-in near 0 -> fold into the first section instead of a phantom clip.
            secs[0]["start"] = 0.0
        else:
            secs.insert(0, {"name": "intro", "start": 0.0, "end": secs[0]["start"]})
    if secs[-1]["end"] < duration - 1e-3:
        secs.append({"name": "outro", "start": secs[-1]["end"], "end": duration})

    clips = []
    for section in secs:
        i0 = max(0, min(total, round(section["start"] * sample_rate)))
        i1 = max(0, min(total, round(section["end"] * sample_rate)))
        if i1 > i0:
            clips.append({"waveform": waveform[:, :, i0:i1], "sample_rate": sample_rate})
    return clips


class HZ3_YuE2_SheetSage2Sections:
    CATEGORY = "HZ3 YuE2/Audio"
    FUNCTION = "transcribe"
    RETURN_TYPES = ("STRING", "STRING", "AUDIO", "STRING")
    RETURN_NAMES = ("abc", "structure", "segments_audio", "report")
    OUTPUT_IS_LIST = (False, False, True, False)
    DESCRIPTION = (
        "Run SheetSage2 on the whole audio once and output the ABC text, the SECTION "
        "STRUCTURE with REAL timestamps (Intro/Verse/Chorus/Interlude/Outro...), and the "
        "audio cut into per-section clips. Connect audio_encoder from 'Load Audio "
        "Encoder'. This is the accurate source for section segmentation."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_encoder": ("AUDIO_ENCODER",),
                "audio": ("AUDIO",),
                "mode": (["melody", "full"], {"default": "melody",
                                              "tooltip": "full: generates melody and chords; melody: melody only, recommended for covers."}),
            },
        }

    def transcribe(self, audio_encoder, audio, mode):
        import torchaudio
        import comfy.model_management
        from comfy.audio_encoders.sheetsage2_abc import events_to_abc

        melody_only = mode == "melody"
        input_waveform = audio["waveform"].float()
        if input_waveform.shape[0] != 1:
            raise ValueError("Expected a single audio clip (batch of 1).")
        mono = input_waveform.mean(dim=1)                    # [1, N]
        encoder_sr = int(audio_encoder.model_sample_rate)
        waveform = torchaudio.functional.resample(mono, int(audio["sample_rate"]), encoder_sr)

        comfy.model_management.load_model_gpu(audio_encoder.patcher)
        events = audio_encoder.model.transcribe(waveform.to(audio_encoder.load_device))
        duration = waveform.shape[-1] / float(encoder_sr)

        structure = structure_timeline(events, duration)
        abc = ""
        abc_error = ""
        try:
            abc = events_to_abc(events, duration, melody_only=melody_only)
        except Exception as error:  # BeatGridError / other -> emit structure anyway
            abc_error = f" ABC failed: {error}"

        segments = cut_sections(audio, structure)
        structure_json = json.dumps(structure, ensure_ascii=False, indent=2)
        if segments:
            # Attach the matching section name to each clip's metadata (index-aligned).
            report_lines = [f"{s['name']} {s['start']:.2f}-{s['end']:.2f}s" for s in structure]
        else:
            report_lines = ["no audible sections"]
        report = (
            f"SheetSage2 · {mode} · {len(structure)} section(s) · "
            f"{len(segments)} audio clip(s){abc_error}\n" + "\n".join(report_lines)
        )
        return {"ui": {"text": [structure_json]},
                "result": (abc, structure_json, segments, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_SheetSage2Sections": HZ3_YuE2_SheetSage2Sections,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_SheetSage2Sections": "HZ3 YuE2 · SheetSage2 Audio to ABC + Sections",
}
