import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_DATASET = r"E:\GEN_AI\YuE2_Training\artist"
DEFAULT_RUNTIME = r"E:\GEN_AI\YuE2_Training\runtime"


def _safe_name(value):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip()).strip("._")
    if not name:
        raise ValueError("track_name must contain at least one letter or number")
    return name


def _audio_array(audio):
    waveform = audio["waveform"].detach().float().cpu()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform.clamp(-1, 1).transpose(0, 1).numpy(), int(audio["sample_rate"])


class HZ3_YuE2_TrainingTrack:
    CATEGORY = "HZ3 YuE2/Tokenizer & LoRA"
    FUNCTION = "save"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("track_path", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Create one upstream-compatible YuE2 real-audio training triplet: FLAC, sectioned lyrics, and trigger-prefixed style caption."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "track_name": ("STRING", {"default": "track_001"}),
            "trigger": ("STRING", {"default": "hz3_voice_01"}),
            "style_caption": ("STRING", {"multiline": True, "default": "Spanish vocal performance"}),
            "lyrics": ("STRING", {"multiline": True, "default": "[instrumental]"}),
            "dataset_folder": ("STRING", {"default": DEFAULT_DATASET}),
            "profile_kind": (["voice", "accompaniment", "full_mix"], {"default": "voice"}),
            "overwrite": ("BOOLEAN", {"default": False}),
        }}

    def save(self, audio, track_name, trigger, style_caption, lyrics, dataset_folder, profile_kind, overwrite):
        name = _safe_name(track_name)
        trigger = _safe_name(trigger)
        root = Path(dataset_folder).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        paths = {"audio": root / f"{name}.flac", "lyrics": root / f"{name}.lyrics.txt", "style": root / f"{name}.txt"}
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing and not overwrite:
            raise FileExistsError("Training files already exist; enable overwrite to replace them: " + ", ".join(existing))
        samples, sample_rate = _audio_array(audio)
        sf.write(paths["audio"], samples, sample_rate, format="FLAC", subtype="PCM_24")
        paths["lyrics"].write_text((lyrics or "[instrumental]").strip() + "\n", encoding="utf-8")
        caption = f"{trigger}, in the style of {trigger}. {(style_caption or '').strip()}".strip()
        paths["style"].write_text(caption + "\n", encoding="utf-8")
        metadata = {"track": name, "trigger": trigger, "profile_kind": profile_kind, "sample_rate": sample_rate,
                    "channels": samples.shape[1], "seconds": round(len(samples) / sample_rate, 3),
                    "files": {key: str(value) for key, value in paths.items()}}
        (root / f"{name}.hz3.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        report = json.dumps(metadata, ensure_ascii=False, indent=2)
        return {"ui": {"text": [report]}, "result": (str(paths["audio"]), report)}


class HZ3_YuE2_DatasetAudit:
    CATEGORY = "HZ3 YuE2/Tokenizer & LoRA"
    FUNCTION = "audit"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("dataset_folder", "valid_tracks", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"dataset_folder": ("STRING", {"default": DEFAULT_DATASET})}}

    def audit(self, dataset_folder):
        root = Path(dataset_folder).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset folder does not exist: {root}")
        audio = {path.stem: path for path in root.glob("*.flac")}
        valid, issues = [], []
        for name, audio_path in sorted(audio.items()):
            lyrics, style = root / f"{name}.lyrics.txt", root / f"{name}.txt"
            missing = [p.name for p in (lyrics, style) if not p.is_file()]
            if missing:
                issues.append(f"{name}: missing {', '.join(missing)}")
                continue
            if not lyrics.read_text(encoding="utf-8").strip(): issues.append(f"{name}: empty lyrics")
            elif not style.read_text(encoding="utf-8").strip(): issues.append(f"{name}: empty style caption")
            else: valid.append(name)
        orphan_text = [p.name for p in root.glob("*.lyrics.txt") if p.name[:-11] not in audio]
        issues.extend(f"{name}: no matching FLAC" for name in orphan_text)
        report = f"Valid tracks: {len(valid)}\n" + ("\n".join(valid) if valid else "(none)")
        if issues: report += "\n\nIssues:\n" + "\n".join(issues)
        return {"ui": {"text": [report]}, "result": (str(root), len(valid), report)}


class HZ3_YuE2_TokenizerLoRARuntime:
    CATEGORY = "HZ3 YuE2/Tokenizer & LoRA"
    FUNCTION = "configure"
    RETURN_TYPES = ("HZ3_YUE2_TRAINING", "STRING")
    RETURN_NAMES = ("training_config", "status")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "runtime_folder": ("STRING", {"default": DEFAULT_RUNTIME}),
            "dataset_folder": ("STRING", {"default": DEFAULT_DATASET}),
            "tokenizer_head": ("STRING", {"default": "tokenizer_head_joint_v4.pt"}),
            "nar_lora": ("STRING", {"default": "nar_lora_joint_v4.pt"}),
            "regularizer_pack": ("STRING", {"default": "minted_regularizer_pack.pt"}),
            "trigger": ("STRING", {"default": "hz3_voice_01"}),
            "profile_kind": (["voice", "accompaniment", "full_mix"], {"default": "voice"}),
        }}

    def configure(self, runtime_folder, dataset_folder, tokenizer_head, nar_lora, regularizer_pack, trigger, profile_kind):
        root = Path(runtime_folder).expanduser().resolve()
        resolve = lambda value: Path(value).expanduser().resolve() if Path(value).is_absolute() else root / value
        config = {"runtime_folder": str(root), "dataset_folder": str(Path(dataset_folder).expanduser().resolve()),
                  "tokenizer_head": str(resolve(tokenizer_head)), "nar_lora": str(resolve(nar_lora)),
                  "regularizer_pack": str(resolve(regularizer_pack)), "trigger": _safe_name(trigger),
                  "profile_kind": profile_kind}
        checks = {key: Path(config[key]).exists() for key in ("dataset_folder", "tokenizer_head", "nar_lora", "regularizer_pack")}
        config["ready"] = all(checks.values())
        status = "Tokenizer/LoRA runtime ready." if config["ready"] else "Missing: " + ", ".join(k for k, ok in checks.items() if not ok)
        status += "\n" + json.dumps(config, ensure_ascii=False, indent=2)
        return {"ui": {"text": [status]}, "result": (config, status)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_TrainingTrack": HZ3_YuE2_TrainingTrack,
    "HZ3_YuE2_DatasetAudit": HZ3_YuE2_DatasetAudit,
    "HZ3_YuE2_TokenizerLoRARuntime": HZ3_YuE2_TokenizerLoRARuntime,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_TrainingTrack": "HZ3 YuE2 · Add Training Track",
    "HZ3_YuE2_DatasetAudit": "HZ3 YuE2 · Audit Training Dataset",
    "HZ3_YuE2_TokenizerLoRARuntime": "HZ3 YuE2 · Tokenizer/LoRA Runtime",
}
