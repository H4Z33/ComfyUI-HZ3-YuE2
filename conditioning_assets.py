"""Persist large YuE2 conditionings as sidecars and assemble a take timeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import folder_paths
from safetensors import safe_open
from safetensors.torch import save_file

from comfy.text_encoders.yue2 import FRAMES_PER_SECOND

from .conditioning_edit import _entry, paste_conditioning


_EMPTY_CHOICE = "(no HZ3 conditioning assets found)"
_LINE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*([1-6])"
    r"(?:\s*@\s*(\d+(?:\.\d+)?))?\s*$"
)


def _output_root():
    return Path(folder_paths.get_output_directory()).resolve()


def _conditioning_choices():
    root = _output_root()
    if not root.exists():
        return [_EMPTY_CHOICE]
    choices = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.safetensors")
        if path.is_file() and "conditioning" in {part.lower() for part in path.relative_to(root).parts[:-1]}
    ]
    return sorted(choices, key=str.lower, reverse=True) or [_EMPTY_CHOICE]


def _resolve_asset(value):
    if not isinstance(value, str) or not value.strip() or value == _EMPTY_CHOICE:
        raise FileNotFoundError("No HZ3 conditioning asset was selected or connected.")
    root = _output_root()
    path = (root / value.strip().replace("\\", "/")).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() != ".safetensors":
        raise ValueError(f"Invalid conditioning asset path: {value}")
    return path


def _target_path(prefix):
    root = _output_root()
    relative = Path((prefix or "conditioning/HZ3-YuE2").replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("filename_prefix must stay inside the ComfyUI output folder.")
    directory = (root / relative.parent).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("filename_prefix escapes the ComfyUI output folder.")
    directory.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", relative.name).strip("._") or "HZ3-YuE2"
    counter = 1
    while True:
        path = directory / f"{stem}_{counter:05d}.safetensors"
        if not path.exists():
            return path
        counter += 1


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _serializable_metadata(metadata, frames):
    return {
        "schema": "hz3-yue2-conditioning/1",
        "yue2_chunks": [list(chunk) for chunk in metadata["yue2_chunks"]],
        "yue2_frames": frames,
        "yue2_abc_ids": list(metadata.get("yue2_abc_ids", [])),
        "yue2_truncated": bool(metadata.get("yue2_truncated", False)),
        "hz3_conditioning_edit": bool(metadata.get("hz3_conditioning_edit", False)),
        "hz3_conditioning_cut": metadata.get("hz3_conditioning_cut"),
        "hz3_conditioning_paste": metadata.get("hz3_conditioning_paste"),
    }


def save_conditioning_asset(conditioning, filename_prefix):
    context, metadata, _chunks, frames = _entry(conditioning, "conditioning")
    path = _target_path(filename_prefix)
    manifest = _serializable_metadata(metadata, frames)
    save_file(
        {"context": context.detach().to("cpu").contiguous()},
        str(path),
        metadata={"hz3_yue2": json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))},
    )
    manifest["sha256"] = _sha256(path)
    relative = path.relative_to(_output_root()).as_posix()
    return relative, manifest, path.stat().st_size


def load_conditioning_asset(value):
    path = _resolve_asset(value)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if "context" not in handle.keys():
            raise ValueError("Conditioning asset has no context tensor.")
        raw = handle.metadata().get("hz3_yue2")
        if not raw:
            raise ValueError("Conditioning asset has no HZ3 manifest.")
        manifest = json.loads(raw)
        context = handle.get_tensor("context")
    if manifest.get("schema") != "hz3-yue2-conditioning/1":
        raise ValueError("Unsupported HZ3 conditioning asset schema.")
    metadata = {
        "pooled_output": None,
        "yue2_chunks": tuple(tuple(int(value) for value in chunk) for chunk in manifest["yue2_chunks"]),
        "yue2_frames": int(manifest["yue2_frames"]),
        "yue2_abc_ids": [int(value) for value in manifest.get("yue2_abc_ids", [])],
        "yue2_truncated": bool(manifest.get("yue2_truncated", False)),
    }
    for key in ("hz3_conditioning_edit", "hz3_conditioning_cut", "hz3_conditioning_paste"):
        if manifest.get(key) is not None:
            metadata[key] = manifest[key]
    conditioning = [[context, metadata]]
    _entry(conditioning, "loaded conditioning")
    return conditioning, manifest, path


class HZ3_YuE2_SaveConditioning:
    CATEGORY = "HZ3 YuE2/Conditioning"
    FUNCTION = "save"
    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "asset_file", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Save a large YuE2 conditioning tensor as a lossless safetensors sidecar."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "filename_prefix": ("STRING", {"default": "conditioning/HZ3-YuE2"}),
            }
        }

    def save(self, conditioning, filename_prefix):
        relative, manifest, size = save_conditioning_asset(conditioning, filename_prefix)
        seconds = manifest["yue2_frames"] / FRAMES_PER_SECOND
        report = f"Saved {relative}\n{seconds:.2f} s · {size / (1024 ** 2):.1f} MiB · SHA-256 {manifest['sha256']}"
        return {"ui": {"text": [report]}, "result": (conditioning, relative, report)}


class HZ3_YuE2_LoadConditioning:
    CATEGORY = "HZ3 YuE2/Conditioning"
    FUNCTION = "load"
    RETURN_TYPES = ("CONDITIONING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "seconds", "asset_file", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Load a lossless HZ3 YuE2 conditioning sidecar from the ComfyUI output folder."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"conditioning_file": (_conditioning_choices(),)},
            "optional": {
                "asset_file": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect Save Conditioning.asset_file or Load Generation.conditioning_asset.",
                })
            },
        }

    @classmethod
    def IS_CHANGED(cls, conditioning_file, asset_file=""):
        try:
            path = _resolve_asset(asset_file or conditioning_file)
        except (FileNotFoundError, ValueError):
            return float("nan")
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def load(self, conditioning_file, asset_file=""):
        selected = asset_file or conditioning_file
        conditioning, manifest, path = load_conditioning_asset(selected)
        seconds = manifest["yue2_frames"] / FRAMES_PER_SECOND
        relative = path.relative_to(_output_root()).as_posix()
        report = f"Loaded {relative}\n{seconds:.2f} s · {path.stat().st_size / (1024 ** 2):.1f} MiB"
        return {"ui": {"text": [report]}, "result": (conditioning, seconds, relative, report)}


class HZ3_YuE2_ConditioningTimeline:
    CATEGORY = "HZ3 YuE2/Conditioning"
    FUNCTION = "compose"
    RETURN_TYPES = ("CONDITIONING", "FLOAT", "STRING")
    RETURN_NAMES = ("conditioning", "seconds", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Assemble a YuE2 production timeline from up to six conditioning takes."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "take_1": ("CONDITIONING",),
                "timeline": ("STRING", {
                    "multiline": True,
                    "default": "0-12: 1@0\n12-18: 2@12\n18-30: 1@18",
                    "tooltip": "One segment per line: baseStart-baseEnd: take@takeStart. Take start defaults to baseStart.",
                }),
            },
            "optional": {f"take_{index}": ("CONDITIONING",) for index in range(2, 7)},
        }

    def compose(self, take_1, timeline, **kwargs):
        takes = {1: take_1}
        for index in range(2, 7):
            value = kwargs.get(f"take_{index}")
            if value is not None:
                takes[index] = value
        _context, _metadata, _chunks, frames = _entry(take_1, "take 1")
        result = take_1
        applied = []
        for line_number, raw in enumerate((timeline or "").splitlines(), 1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = _LINE_RE.fullmatch(stripped)
            if not match:
                raise ValueError(f"Timeline line {line_number} is invalid: {raw!r}")
            base_start, base_end = float(match.group(1)), float(match.group(2))
            take_index = int(match.group(3))
            source_start = float(match.group(4)) if match.group(4) is not None else base_start
            if base_end <= base_start:
                raise ValueError(f"Timeline line {line_number} must have end > start.")
            if take_index not in takes:
                raise ValueError(f"Timeline line {line_number} references unconnected take {take_index}.")
            result, bs, be, ds, de = paste_conditioning(
                result, takes[take_index], base_start, source_start, base_end - base_start
            )
            applied.append(f"{bs / FRAMES_PER_SECOND:.2f}-{be / FRAMES_PER_SECOND:.2f}s <- take {take_index} "
                           f"{ds / FRAMES_PER_SECOND:.2f}-{de / FRAMES_PER_SECOND:.2f}s")
        if not applied:
            raise ValueError("Timeline contains no segments.")
        seconds = frames / FRAMES_PER_SECOND
        report = f"Conditioning timeline · {seconds:.2f} s\n" + "\n".join(applied)
        result[0][1]["hz3_conditioning_timeline"] = applied
        return {"ui": {"text": [report]}, "result": (result, seconds, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_SaveConditioning": HZ3_YuE2_SaveConditioning,
    "HZ3_YuE2_LoadConditioning": HZ3_YuE2_LoadConditioning,
    "HZ3_YuE2_ConditioningTimeline": HZ3_YuE2_ConditioningTimeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_SaveConditioning": "HZ3 YuE2 · Save Conditioning",
    "HZ3_YuE2_LoadConditioning": "HZ3 YuE2 · Load Conditioning",
    "HZ3_YuE2_ConditioningTimeline": "HZ3 YuE2 · Conditioning Timeline",
}
