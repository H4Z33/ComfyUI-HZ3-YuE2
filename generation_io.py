import json
from pathlib import Path

import av
import folder_paths
from comfy_api.latest import IO, UI
from comfy_extras.nodes_audio import load as load_audio

from .conditioning_assets import load_conditioning_asset, save_conditioning_asset


_AUDIO_EXTENSIONS = {".flac", ".mp3", ".opus", ".wav", ".m4a", ".ogg"}
_EMPTY_CHOICE = "(no generated audio found)"


def _output_root():
    return Path(folder_paths.get_output_directory()).resolve()


def _audio_choices():
    root = _output_root()
    if not root.exists():
        return [_EMPTY_CHOICE]
    files = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS
    ]
    return sorted(files, key=lambda item: item.lower(), reverse=True) or [_EMPTY_CHOICE]


def _resolve_output_audio(value):
    if value == _EMPTY_CHOICE:
        raise FileNotFoundError("No generated audio files were found in the ComfyUI output folder.")
    root = _output_root()
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() not in _AUDIO_EXTENSIONS:
        raise ValueError(f"Invalid generated audio path: {value}")
    return path


def _selected_audio_file(audio_file, saved_file=""):
    """Prefer a connected Save Audio path; its batch output is newline-delimited."""
    if isinstance(saved_file, str):
        connected = next((line.strip() for line in saved_file.splitlines() if line.strip()), "")
        if connected:
            return connected
    return audio_file


def _json_value(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _embedded_metadata(path):
    with av.open(str(path)) as container:
        return dict(container.metadata)


def _literal_generation_inputs(prompt):
    if not isinstance(prompt, dict):
        return {}, []
    generators = [
        node for node in prompt.values()
        if isinstance(node, dict) and node.get("class_type") == "YuE2GenerateMusic"
    ]
    if not generators:
        return {}, []
    inputs = generators[-1].get("inputs", {})
    recovered = {}
    missing = []
    for name in ("abc", "style", "lyrics"):
        value = inputs.get(name)
        if isinstance(value, str):
            recovered[name] = value
        else:
            missing.append(name)
    return recovered, missing


def _load_bundle(path):
    metadata = _embedded_metadata(path)
    bundle = _json_value(metadata.get("hz3_yue2"))
    if isinstance(bundle, dict):
        return bundle, "embedded HZ3 metadata", []

    prompt = _json_value(metadata.get("prompt"))
    recovered, missing = _literal_generation_inputs(prompt)
    return recovered, "standard Comfy metadata", missing or ["abc", "style", "lyrics"]


class HZ3_YuE2_SaveAudio:
    CATEGORY = "HZ3 YuE2/Generation"
    FUNCTION = "save"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "saved_file", "conditioning_asset")
    OUTPUT_NODE = True
    DESCRIPTION = "Save audio with its exact YuE2 ABC, style, and lyrics embedded directly in the audio file."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": ("STRING", {"multiline": True, "forceInput": True}),
                "lyrics": ("STRING", {"multiline": True, "forceInput": True}),
                "abc": ("STRING", {"multiline": True, "forceInput": True}),
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/HZ3-YuE2"}),
                "file_format": (["mp3 320k", "mp3 V0", "flac", "opus 128k"], {"default": "mp3 320k"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
            "optional": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Connect the original YuE2 conditioning to archive it automatically as a lossless sidecar."
                }),
                "token_stream": ("HZ3_YUE2_TOKEN_STREAM", {
                    "tooltip": "Optional semantic stream from HZ3 YuE2 · Generate Token Stream or Music From Token Stream."
                }),
                "conditioning_asset": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect HZ3 YuE2 · Save Conditioning.asset_file to associate its sidecar with this audio."
                }),
            },
        }

    def save(self, audio, abc, style, lyrics, filename_prefix, file_format, prompt=None, extra_pnginfo=None,
             token_stream=None, conditioning_asset="", conditioning=None):
        archived_conditioning = conditioning_asset.strip() if isinstance(conditioning_asset, str) else ""
        conditioning_manifest = None
        if conditioning is not None:
            archived_conditioning, conditioning_manifest, _size = save_conditioning_asset(
                conditioning, f"{filename_prefix}-conditioning"
            )
        bundle = {
            "schema": "hz3-yue2-generation/2",
            "abc": abc,
            "style": style,
            "lyrics": lyrics,
        }
        if isinstance(token_stream, dict) and token_stream.get("schema") == "hz3-yue2-token-stream/1":
            bundle["token_stream"] = token_stream
        if archived_conditioning:
            bundle["conditioning_asset"] = archived_conditioning
        if conditioning_manifest is not None:
            bundle["conditioning_sha256"] = conditioning_manifest["sha256"]
            bundle["conditioning_frames"] = conditioning_manifest["yue2_frames"]
        metadata = dict(extra_pnginfo or {})
        metadata["hz3_yue2"] = bundle

        class SaveContext:
            pass

        class Hidden:
            pass

        SaveContext.hidden = Hidden
        Hidden.prompt = prompt
        Hidden.extra_pnginfo = metadata

        format, quality = {
            "mp3 320k": ("mp3", "320k"),
            "mp3 V0": ("mp3", "V0"),
            "flac": ("flac", "128k"),
            "opus 128k": ("opus", "128k"),
        }[file_format]
        results = UI.AudioSaveHelper.save_audio(
            audio,
            filename_prefix=filename_prefix,
            folder_type=IO.FolderType.output,
            cls=SaveContext,
            format=format,
            quality=quality,
        )
        saved = [
            (_output_root() / result.subfolder / result.filename).resolve().relative_to(_output_root()).as_posix()
            for result in results
        ]

        text = list(saved)
        if archived_conditioning:
            text.append(f"conditioning: {archived_conditioning}")

        return {
            "ui": {"audio": results, "text": text},
            "result": (audio, "\n".join(saved), archived_conditioning),
        }


class HZ3_YuE2_LoadGeneration:
    CATEGORY = "HZ3 YuE2/Generation"
    FUNCTION = "load"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING", "STRING", "HZ3_YUE2_TOKEN_STREAM", "STRING",
                    "CONDITIONING", "FLOAT")
    RETURN_NAMES = ("audio", "abc", "style", "lyrics", "report", "token_stream", "conditioning_asset",
                    "conditioning", "conditioning_seconds")
    OUTPUT_NODE = True
    DESCRIPTION = "Load generated audio and recover the ABC, style, and lyrics saved with it."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": (_audio_choices(),),
                "load_conditioning": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Load the potentially large conditioning sidecar referenced by the audio metadata."
                }),
            },
            "optional": {
                "saved_file": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect HZ3 Save Audio's saved_file output to verify the metadata just written.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file, load_conditioning=True, saved_file=""):
        # Linked values may not be available during graph validation. Runtime still
        # resolves and validates the exact path inside the output directory.
        if saved_file is not None and not isinstance(saved_file, str):
            return True
        try:
            _resolve_output_audio(_selected_audio_file(audio_file, saved_file))
        except (FileNotFoundError, ValueError) as error:
            return str(error)
        return True

    @classmethod
    def IS_CHANGED(cls, audio_file, load_conditioning=True, saved_file=""):
        try:
            path = _resolve_output_audio(_selected_audio_file(audio_file, saved_file))
        except (FileNotFoundError, ValueError):
            return float("nan")
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}:{load_conditioning}"

    def load(self, audio_file, load_conditioning=True, saved_file=""):
        selected = _selected_audio_file(audio_file, saved_file)
        path = _resolve_output_audio(selected)
        bundle, source, missing = _load_bundle(path)
        waveform, sample_rate = load_audio(str(path))
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        abc = str(bundle.get("abc", ""))
        style = str(bundle.get("style", ""))
        lyrics = str(bundle.get("lyrics", ""))
        token_stream = bundle.get("token_stream")
        has_stream = isinstance(token_stream, dict) and token_stream.get("schema") == "hz3-yue2-token-stream/1"
        if not has_stream:
            token_stream = {"schema": "hz3-yue2-token-stream/1", "ids": [], "missing": True}
        conditioning_asset = str(bundle.get("conditioning_asset", ""))
        conditioning = []
        conditioning_seconds = 0.0
        conditioning_status = ""
        if conditioning_asset and load_conditioning:
            try:
                conditioning, conditioning_manifest, conditioning_path = load_conditioning_asset(conditioning_asset)
                conditioning_seconds = conditioning_manifest["yue2_frames"] / 25.0
                expected_hash = str(bundle.get("conditioning_sha256", ""))
                conditioning_status = (
                    f" Loaded conditioning sidecar {conditioning_path.name} ({conditioning_seconds:.2f} s)."
                )
                if expected_hash:
                    conditioning_status += f" Expected SHA-256: {expected_hash}."
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as error:
                conditioning_status = f" Conditioning sidecar could not be loaded: {error}"
        elif conditioning_asset:
            conditioning_status = " Conditioning sidecar loading is disabled."
        if missing:
            report = (
                f"Loaded {selected}. Found {source}, but runtime values were not saved: "
                f"{', '.join(missing)}. Use HZ3 YuE2 · Save Audio for future outputs."
            )
        else:
            report = f"Loaded {selected} with ABC, style, and lyrics from {source}."
        report += " Semantic token stream available." if has_stream else " No semantic token stream was stored."
        report += f" Conditioning sidecar: {conditioning_asset}." if conditioning_asset else " No conditioning sidecar is associated."
        report += conditioning_status
        return {
            "ui": {
                "audio": [{"filename": path.name, "subfolder": path.parent.relative_to(_output_root()).as_posix(), "type": "output"}],
                "text": [report],
            },
            "result": (audio, abc, style, lyrics, report, token_stream, conditioning_asset,
                       conditioning, conditioning_seconds),
        }


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_SaveAudio": HZ3_YuE2_SaveAudio,
    "HZ3_YuE2_LoadGeneration": HZ3_YuE2_LoadGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_SaveAudio": "HZ3 YuE2 · Save Audio",
    "HZ3_YuE2_LoadGeneration": "HZ3 YuE2 · Load Generation",
}
