"""Load any audio file from standard or custom folders with dynamic file selection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import folder_paths
import torch

try:
    from comfy_extras.nodes_audio import load as load_audio
except ImportError:
    import av

    def _f32_pcm(wav: torch.Tensor) -> torch.Tensor:
        if wav.dtype.is_floating_point:
            return wav
        if wav.dtype == torch.int16:
            return wav.float() / 32768.0
        if wav.dtype == torch.int32:
            return wav.float() / 2147483648.0
        if wav.dtype == torch.uint8:
            return (wav.float() - 128.0) / 128.0
        return wav.float()

    def load_audio(filepath: str) -> tuple[torch.Tensor, int]:
        with av.open(filepath) as container:
            if not container.streams.audio:
                raise ValueError(f"No audio stream found in '{filepath}'.")
            stream = container.streams.audio[0]
            sr = stream.codec_context.sample_rate
            n_channels = stream.channels
            frames = []
            for frame in container.decode(streams=stream.index):
                buf = torch.from_numpy(frame.to_ndarray())
                if buf.shape[0] != n_channels:
                    buf = buf.view(-1, n_channels).t()
                frames.append(buf)
            if not frames:
                raise ValueError(f"No audio frames decoded from '{filepath}'.")
            wav = torch.cat(frames, dim=1)
            wav = _f32_pcm(wav)
            return wav, sr


_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".ogg", ".flac", ".opus", ".m4a", ".aac", ".webm", ".wma", ".aiff", ".alac"
}
_NO_AUDIO_CHOICE = "(no audio files found)"


def _input_root() -> Path:
    return Path(folder_paths.get_input_directory()).resolve()


def _output_root() -> Path:
    return Path(folder_paths.get_output_directory()).resolve()


class HZ3_YuE2_AudioFolder:
    CATEGORY = "HZ3 YuE2/Audio"
    FUNCTION = "load_audio_file"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("audio", "file_path", "file_name", "duration", "sample_rate")
    DESCRIPTION = (
        "Select a folder (input, output, subfolders, or custom path) and choose any audio file "
        "inside it. Loads the audio into ComfyUI's standard AUDIO tensor format with metadata."
    )

    @classmethod
    def get_folder_choices(cls) -> list[str]:
        choices = ["[input]"]
        try:
            input_subs = folder_paths.get_input_subfolders()
            for sub in input_subs:
                choices.append(f"input/{sub}")
        except Exception:
            pass

        choices.append("[output]")
        try:
            out_root = _output_root()
            if out_root.exists():
                for root, _, _ in os.walk(out_root):
                    rel = os.path.relpath(root, out_root)
                    if rel != ".":
                        choices.append(f"output/{rel.replace(os.sep, '/')}")
        except Exception:
            pass

        choices.append("[custom folder]")
        return choices

    @classmethod
    def _resolve_folder_path(cls, folder: str, custom_folder: str = "") -> Path:
        custom = (custom_folder or "").strip()
        if custom:
            p = Path(custom).resolve()
            if not p.exists() or not p.is_dir():
                raise FileNotFoundError(f"Custom folder does not exist or is not a directory: '{custom}'")
            return p

        folder_clean = (folder or "").strip()
        in_root = _input_root()
        out_root = _output_root()

        if folder_clean in ("[input]", "input", ""):
            return in_root
        elif folder_clean in ("[output]", "output"):
            return out_root
        elif folder_clean.startswith("input/") or folder_clean.startswith("input\\"):
            sub = folder_clean[6:]
            return (in_root / sub).resolve()
        elif folder_clean.startswith("output/") or folder_clean.startswith("output\\"):
            sub = folder_clean[7:]
            return (out_root / sub).resolve()
        elif folder_clean == "[custom folder]":
            if not custom:
                raise ValueError("Selected '[custom folder]' but 'custom_folder' is empty. Please enter a folder path.")
            p = Path(custom).resolve()
            if not p.exists() or not p.is_dir():
                raise FileNotFoundError(f"Custom folder not found: '{custom}'")
            return p
        else:
            p = Path(folder_clean)
            if p.is_dir():
                return p.resolve()
            p_in = (in_root / folder_clean).resolve()
            if p_in.is_dir():
                return p_in
            return in_root

    @classmethod
    def _scan_audio_files(cls, folder_path: Path, subfolders: bool = True) -> list[str]:
        if not folder_path.exists() or not folder_path.is_dir():
            return []

        files: list[str] = []
        if subfolders:
            for root, _, filenames in os.walk(folder_path):
                for fn in filenames:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in _AUDIO_EXTENSIONS:
                        full_p = Path(root) / fn
                        try:
                            rel_p = full_p.relative_to(folder_path).as_posix()
                        except ValueError:
                            rel_p = full_p.name
                        files.append(rel_p)
        else:
            for item in folder_path.iterdir():
                if item.is_file() and item.suffix.lower() in _AUDIO_EXTENSIONS:
                    files.append(item.name)

        files.sort(key=str.lower)
        return files

    @classmethod
    def _resolve_audio_file(cls, folder_path: Path, audio_file: str, subfolders: bool = True) -> Path:
        clean_name = (audio_file or "").strip()
        if not clean_name or clean_name == _NO_AUDIO_CHOICE:
            raise FileNotFoundError(f"No audio file selected or found in '{folder_path}'.")

        # 1. Direct path relative to folder_path
        candidate = (folder_path / clean_name.replace("/", os.sep)).resolve()
        if candidate.is_file():
            return candidate

        # 2. Check if clean_name is an absolute path on disk
        cand_abs = Path(clean_name).resolve()
        if cand_abs.is_file():
            return cand_abs

        # 3. Match case-insensitively or by filename
        target_lower = clean_name.lower().replace("\\", "/")
        target_base_lower = Path(clean_name).name.lower()

        all_files = cls._scan_audio_files(folder_path, subfolders=True)
        for f in all_files:
            if f.lower() == target_lower or Path(f).name.lower() == target_base_lower:
                return (folder_path / f.replace("/", os.sep)).resolve()

        available = cls._scan_audio_files(folder_path, subfolders=subfolders)[:15]
        avail_str = "\n".join(f" - {a}" for a in available) if available else " (no audio files in folder)"
        raise FileNotFoundError(
            f"Audio file '{clean_name}' not found in folder '{folder_path}'.\n"
            f"Available audio files:\n{avail_str}"
        )

    @classmethod
    def INPUT_TYPES(cls):
        folders = cls.get_folder_choices()
        initial_folder = folders[0] if folders else "[input]"
        try:
            initial_dir = cls._resolve_folder_path(initial_folder)
            initial_files = cls._scan_audio_files(initial_dir, subfolders=True)
        except Exception:
            initial_files = []

        if not initial_files:
            initial_files = [_NO_AUDIO_CHOICE]

        return {
            "required": {
                "folder": (folders, {"default": initial_folder}),
                "audio_file": (initial_files, {"default": initial_files[0]}),
            },
            "optional": {
                "custom_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional custom folder path on disk (overrides 'folder' dropdown if specified)."
                }),
                "subfolders": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Scan subdirectories recursively for audio files."
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, folder: str, audio_file: str, custom_folder: str = "", subfolders: bool = True, **kwargs: Any) -> bool | str:
        if not audio_file or audio_file == _NO_AUDIO_CHOICE:
            return "No audio file selected or found."
        try:
            folder_path = cls._resolve_folder_path(folder, custom_folder)
            if not folder_path.exists() or not folder_path.is_dir():
                return f"Folder does not exist: {folder_path}"
            cls._resolve_audio_file(folder_path, audio_file, subfolders=subfolders)
            return True
        except Exception as err:
            return str(err)

    def load_audio_file(self, folder: str, audio_file: str, custom_folder: str = "", subfolders: bool = True):
        folder_path = self._resolve_folder_path(folder, custom_folder)
        file_path = self._resolve_audio_file(folder_path, audio_file, subfolders=subfolders)

        waveform, sample_rate = load_audio(str(file_path))
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        sr = int(sample_rate)
        duration = float(waveform.shape[-1]) / float(sr)
        channels = waveform.shape[1]
        file_name = file_path.name

        audio_dict = {
            "waveform": waveform,
            "sample_rate": sr,
        }

        report = (
            f"Loaded '{file_name}' · {duration:.2f}s · {sr}Hz · {channels}ch\n"
            f"Path: {file_path}"
        )
        return {
            "ui": {"text": [report]},
            "result": (audio_dict, str(file_path), file_name, duration, sr),
        }


_AUDIO_MIME_TYPES: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".opus": "audio/opus",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
    ".wma": "audio/x-ms-wma",
    ".aiff": "audio/x-aiff",
    ".alac": "audio/mp4",
}


import sys

# Silence benign WinError 10054 when clients (browsers) cancel audio streaming requests on Windows
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport

        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

        def _silenced_call_connection_lost(self, exc):
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, OSError):
                if hasattr(self, "_sock") and self._sock:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                self._called_connection_lost = True

        _ProactorBasePipeTransport._call_connection_lost = _silenced_call_connection_lost
    except Exception:
        pass


# Register HTTP API endpoint for dynamic ComfyUI frontend updates
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/hz3/yue2/audio_folder/files")
    async def _api_get_audio_folder_files(request: web.Request) -> web.Response:
        folder = request.rel_url.query.get("folder", "[input]")
        custom_folder = request.rel_url.query.get("custom_folder", "")
        subfolders = request.rel_url.query.get("subfolders", "true").lower() in ("true", "1")

        try:
            folder_path = HZ3_YuE2_AudioFolder._resolve_folder_path(folder, custom_folder)
            files = HZ3_YuE2_AudioFolder._scan_audio_files(folder_path, subfolders=subfolders)
            if not files:
                files = [_NO_AUDIO_CHOICE]
            return web.json_response({
                "success": True,
                "folder_path": str(folder_path),
                "files": files,
                "count": len(files) if files != [_NO_AUDIO_CHOICE] else 0,
            })
        except Exception as err:
            return web.json_response({
                "success": False,
                "error": str(err),
                "files": [_NO_AUDIO_CHOICE],
                "count": 0,
            })

    @PromptServer.instance.routes.get("/hz3/yue2/audio_folder/audio")
    async def _api_get_audio_folder_audio(request: web.Request) -> web.StreamResponse:
        folder = request.rel_url.query.get("folder", "[input]")
        custom_folder = request.rel_url.query.get("custom_folder", "")
        audio_file = request.rel_url.query.get("audio_file", "")
        subfolders = request.rel_url.query.get("subfolders", "true").lower() in ("true", "1")

        if not audio_file or audio_file == _NO_AUDIO_CHOICE:
            return web.Response(status=404, text="No audio file selected.")

        try:
            folder_path = HZ3_YuE2_AudioFolder._resolve_folder_path(folder, custom_folder)
            file_path = HZ3_YuE2_AudioFolder._resolve_audio_file(folder_path, audio_file, subfolders=subfolders)

            if not file_path.exists() or not file_path.is_file():
                return web.Response(status=404, text=f"File not found: {file_path}")

            ext = file_path.suffix.lower()
            if ext not in _AUDIO_EXTENSIONS:
                return web.Response(status=403, text="Selected file is not an allowed audio format.")

            content_type = _AUDIO_MIME_TYPES.get(ext, "application/octet-stream")

            return web.FileResponse(
                file_path,
                headers={
                    "Content-Type": content_type,
                    "Accept-Ranges": "bytes",
                    "Cache-Control": "private, max-age=3600",
                },
            )
        except Exception as err:
            return web.Response(status=400, text=str(err))
except Exception:
    pass


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_AudioFolder": HZ3_YuE2_AudioFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_AudioFolder": "HZ3 YuE2 · Audio Folder",
}
