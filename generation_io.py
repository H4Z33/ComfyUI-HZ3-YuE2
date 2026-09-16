import json
from pathlib import Path

import av
import folder_paths
from comfy_api.latest import IO, UI
from comfy_extras.nodes_audio import load as load_audio


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
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "saved_file")
    OUTPUT_NODE = True
    DESCRIPTION = "Save audio with its exact YuE2 ABC, style, and lyrics embedded directly in the audio file."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "abc": ("STRING", {"multiline": True, "forceInput": True}),
                "style": ("STRING", {"multiline": True, "forceInput": True}),
                "lyrics": ("STRING", {"multiline": True, "forceInput": True}),
                "filename_prefix": ("STRING", {"default": "audio/HZ3-YuE2"}),
                "file_format": (["mp3 320k", "mp3 V0", "flac", "opus 128k"], {"default": "mp3 320k"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
            "optional": {
                "token_stream": ("HZ3_YUE2_TOKEN_STREAM", {
                    "tooltip": "Optional semantic stream from HZ3 YuE2 · Generate Token Stream or Music From Token Stream."
                }),
            },
        }

    def save(self, audio, abc, style, lyrics, filename_prefix, file_format, prompt=None, extra_pnginfo=None,
             token_stream=None):
        bundle = {
            "schema": "hz3-yue2-generation/2",
            "abc": abc,
            "style": style,
            "lyrics": lyrics,
        }
        if isinstance(token_stream, dict) and token_stream.get("schema") == "hz3-yue2-token-stream/1":
            bundle["token_stream"] = token_stream
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

        return {
            "ui": {"audio": results, "text": saved},
            "result": (audio, "\n".join(saved)),
        }


class HZ3_YuE2_LoadGeneration:
    CATEGORY = "HZ3 YuE2/Generation"
    FUNCTION = "load"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING", "STRING", "HZ3_YUE2_TOKEN_STREAM")
    RETURN_NAMES = ("audio", "abc", "style", "lyrics", "report", "token_stream")
    OUTPUT_NODE = True
    DESCRIPTION = "Load generated audio and recover the ABC, style, and lyrics saved with it."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"audio_file": (_audio_choices(),)},
            "optional": {
                "saved_file": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect HZ3 Save Audio's saved_file output to verify the metadata just written.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file, saved_file=""):
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
    def IS_CHANGED(cls, audio_file, saved_file=""):
        try:
            path = _resolve_output_audio(_selected_audio_file(audio_file, saved_file))
        except (FileNotFoundError, ValueError):
            return float("nan")
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def load(self, audio_file, saved_file=""):
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
        if missing:
            report = (
                f"Loaded {selected}. Found {source}, but runtime values were not saved: "
                f"{', '.join(missing)}. Use HZ3 YuE2 · Save Audio for future outputs."
            )
        else:
            report = f"Loaded {selected} with ABC, style, and lyrics from {source}."
        report += " Semantic token stream available." if has_stream else " No semantic token stream was stored."
        return {
            "ui": {
                "audio": [{"filename": path.name, "subfolder": path.parent.relative_to(_output_root()).as_posix(), "type": "output"}],
                "text": [report],
            },
            "result": (audio, abc, style, lyrics, report, token_stream),
        }


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_SaveAudio": HZ3_YuE2_SaveAudio,
    "HZ3_YuE2_LoadGeneration": HZ3_YuE2_LoadGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_SaveAudio": "HZ3 YuE2 · Save Audio",
    "HZ3_YuE2_LoadGeneration": "HZ3 YuE2 · Load Generation",
}
