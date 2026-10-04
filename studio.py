"""HZ3 Studio: a local song editor page served by ComfyUI at /hz3/studio.

Analysis and generation go through ComfyUI's own /prompt queue from the page.
This module serves the page, the section timeline of a song, the harmony
arranger, the studio assistant, albums and .mixmash packages: a zip with the
source audio, the corrected lyrics/ABC/style and the whole studio project,
under output/HZ3-YuE2/studio. An album groups songs and keeps their shared
style catalog (singers, groups, genres), the assistant chat and an action log.
"""

import asyncio
import io
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from aiohttp import web
import numpy as np
from safetensors import safe_open
import soundfile
import torch
import torchaudio

import folder_paths
from comfy.text_encoders.yue2 import FRAMES_PER_SECOND

from .abc_score import parse
from .classical_lines import abc_phrases, add_lines, load_library
from .section_generation import _build_section_specs
from .vocal_harmonizer import _time_warp, _track
from .vocal_harmony import PARTS, harmonize


STATIC = Path(__file__).parent / "studio"
REFERENCES = Path(__file__).parent / "references"
ARRANGER_PROMPT = Path(__file__).parent / "prompts" / "harmony_arranger_system.txt"
ASSISTANT_PROMPT = Path(__file__).parent / "prompts" / "studio_assistant_system.txt"
VOICES = ("tenor", "baritone", "low", "bass", "countertenor")
OLLAMA = "http://127.0.0.1:11434"
PROJECT_NAME = re.compile(r"[\w \-()]{1,64}")
AUDIO_TYPES = (".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac")
SOURCE_FILE = re.compile(r"hz3studio_[\w \-()]{1,64}_[\w \-]{1,80}(" + "|".join(re.escape(ext) for ext in AUDIO_TYPES) + ")")
PACKAGE_FORMAT = "hz3-mixmash/1"
DEFAULT_ALBUM = "General"


def _studio_folder():
    return Path(folder_paths.get_output_directory()) / "HZ3-YuE2" / "studio"


def _package_path(name):
    if not PROJECT_NAME.fullmatch(name):
        raise web.HTTPBadRequest(text="Project names may use letters, numbers, spaces, '-', '_' and parentheses (64 max).")
    return _studio_folder() / f"{name}.mixmash"


def _album_path(name):
    if not PROJECT_NAME.fullmatch(name):
        raise web.HTTPBadRequest(text="Album names may use letters, numbers, spaces, '-', '_' and parentheses (64 max).")
    return _studio_folder() / "albums" / f"{name}.json"


def _source_path(filename):
    """Source audio of a project, kept in the input folder so LoadAudio can read it."""
    if not SOURCE_FILE.fullmatch(filename):
        raise web.HTTPBadRequest(text="Invalid source audio name.")
    return Path(folder_paths.get_input_directory()) / filename


def _write_package(project):
    path = _package_path(str(project.get("name", "")))
    filename = (project.get("source") or {}).get("filename")
    audio = None
    if filename:
        source = _source_path(filename)
        if source.is_file():
            audio = source.read_bytes()
        elif path.is_file():
            with zipfile.ZipFile(path) as old:
                if f"source/{filename}" in old.namelist():
                    audio = old.read(f"source/{filename}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps({"format": PACKAGE_FORMAT, "project": project}, ensure_ascii=False, indent=1))
        package.writestr("lyrics.txt", str(project.get("lyrics", "")))
        package.writestr("score.abc", str(project.get("abc", "")))
        package.writestr("style.txt", str(project.get("style", "")))
        if audio is not None:
            package.writestr(f"source/{filename}", audio, compress_type=zipfile.ZIP_STORED)
    temporary.replace(path)


def _read_package(package):
    """Return the project of an open .mixmash zip, restoring its source audio to the input folder."""
    try:
        manifest = json.loads(package.read("manifest.json"))
    except (KeyError, json.JSONDecodeError) as error:
        raise web.HTTPBadRequest(text="Not a .mixmash package.") from error
    if manifest.get("format") != PACKAGE_FORMAT or not isinstance(manifest.get("project"), dict):
        raise web.HTTPBadRequest(text="Unsupported .mixmash format.")
    project = manifest["project"]
    filename = (project.get("source") or {}).get("filename")
    if filename:
        source = _source_path(filename)
        if not source.is_file() and f"source/{filename}" in package.namelist():
            source.write_bytes(package.read(f"source/{filename}"))
    return project


def _ollama(body):
    """The reply message of the local Ollama chat endpoint."""
    request = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps({"stream": False, "think": False, **body}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))["message"]
    except urllib.error.URLError as error:
        raise RuntimeError(f"Ollama is unavailable at {OLLAMA}: {error.reason}") from error


def _ollama_json(model, messages, temperature):
    """One JSON answer from the local Ollama chat endpoint."""
    content = _ollama({"model": model, "format": "json", "options": {"temperature": temperature}, "messages": messages})["content"]
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"The model returned invalid JSON: {content[:300]}") from error


def _arrange(style, instructions, sections, model):
    """Ask the local Ollama for the harmony voices of each section."""
    payload = {"style": style, "instructions": instructions, "sections": sections}
    answer = _ollama_json(model, [{"role": "system", "content": ARRANGER_PROMPT.read_text(encoding="utf-8")},
                                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], 0.3)
    try:
        plan = answer["sections"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"The arranger returned no sections: {json.dumps(answer)[:300]}") from error
    names = {section["name"] for section in sections}
    return {entry["name"]: {"voices": [voice for voice in entry.get("voices", []) if voice in VOICES],
                            "reason": str(entry.get("reason", ""))}
            for entry in plan if isinstance(entry, dict) and entry.get("name") in names}


def _assist(model, state, messages, tools):
    """One step of the studio assistant: a reply, or the page tools it calls, given the open song and its album."""
    context = "Current studio state (JSON):\n" + json.dumps(state, ensure_ascii=False)
    message = _ollama({"model": model, "options": {"temperature": 0.4}, "tools": tools,
                       "messages": [{"role": "system", "content": ASSISTANT_PROMPT.read_text(encoding="utf-8")},
                                    {"role": "system", "content": context}, *messages]})
    return {"content": message.get("content", ""),
            "tool_calls": [call["function"] for call in message.get("tool_calls") or []]}


def _project_of(path):
    with zipfile.ZipFile(path) as package:
        return json.loads(package.read("manifest.json"))["project"]


def _move_album_songs(old, new):
    """Rewrite every song of album `old` into album `new`."""
    for path in _studio_folder().glob("*.mixmash"):
        project = _project_of(path)
        if (project.get("album") or DEFAULT_ALBUM) == old:
            project["album"] = new
            _write_package(project)


def _discard(path):
    """Deleted songs and albums go to studio/deleted, so a mistake can be undone by hand."""
    target = _studio_folder() / "deleted" / f"{path.stem} {time.strftime('%Y%m%d-%H%M%S')}{path.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    path.replace(target)


def _output_file(file):
    """A ComfyUI output file named by the page, kept inside the output folder."""
    output = Path(folder_paths.get_output_directory()).resolve()
    path = (output / file["subfolder"] / file["filename"]).resolve()
    if not path.is_relative_to(output) or not path.is_file():
        raise web.HTTPBadRequest(text="Audio file not found in the output folder.")
    return path


def _yue2_loras():
    """The YuE2 LoRAs ComfyUI can load, with the trigger and kind written into them when they were trained."""
    loras = []
    for name in folder_paths.get_filename_list("loras"):
        with safe_open(folder_paths.get_full_path("loras", name), framework="pt") as handle:
            metadata = handle.metadata() or {}
        if metadata.get("format") == "comfyui-yue2-lora":
            loras.append({"name": name, "trigger": metadata.get("trigger", ""), "type": metadata.get("lora_type", "")})
    return loras


def _stage_audio(paths, name):
    """Join generated clips into one stereo FLAC in the input folder, where LoadAudio reads it for LoRA training.

    Mixes come out of YuE2 at 48 kHz and separated vocals at the separator's rate; all follow the first clip's rate.
    """
    parts, rate = [], None
    for path in paths:
        wave, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        wave = torch.from_numpy(np.repeat(wave, 2, axis=1) if wave.shape[1] == 1 else wave[:, :2]).T
        rate = rate or sample_rate
        parts.append(torchaudio.functional.resample(wave, sample_rate, rate) if sample_rate != rate else wave)
    filename = f"hz3studio_lora_{name}.flac"
    soundfile.write(str(Path(folder_paths.get_input_directory()) / filename), torch.cat(parts, dim=1).T.numpy(), rate, format="FLAC")
    return filename


def _reference_themes():
    """Melody phrases of every analyzed reference recording in the catalog."""
    folder = _studio_folder()
    themes = []
    for path in sorted(folder.glob("*.mixmash")):
        project = _project_of(path)
        if project.get("kind") == "reference" and project.get("abc", "").strip():
            license = (project.get("license") or {}).get("name", "?")
            themes += abc_phrases(project["abc"], f"{path.stem} ({license})")
    return themes


def _warp(path, score_abc, line):
    """ABC-to-audio time warp of one sung line (the lead, or a Vocal Harmony part)."""
    wave, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    abc = score_abc if line == "lead" else harmonize(score_abc)[1 + PARTS.index(line)]
    score = parse(abc.strip() + "\n")
    return _time_warp(_track(wave.mean(1), sample_rate), score.voices["Vocal"].notes, score.bpm)


def register(routes):
    @routes.get("/hz3/studio")
    async def index(request):
        # Asset URLs carry their modification time, so a new page is never paired with a cached script or stylesheet.
        page = (STATIC / "index.html").read_text(encoding="utf-8")
        for name in ("studio.css", "studio.js"):
            page = page.replace(f"/static/{name}", f"/static/{name}?v={int((STATIC / name).stat().st_mtime)}")
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-cache"})

    routes.static("/hz3/studio/static", STATIC)

    @routes.post("/hz3/studio/sections")
    async def sections(request):
        body = await request.json()
        try:
            specs = _build_section_specs(body["abc"], body["lyrics"])
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)
        timeline = []
        frame = 0
        for spec in specs:
            timeline.append({"name": spec["name"], "lyrics": spec["lyrics"], "lyrics_index": spec["lyrics_index"], "abc": spec["abc"], "bars": spec["bars"],
                             "start": frame / FRAMES_PER_SECOND, "end": (frame + spec["frames"]) / FRAMES_PER_SECOND})
            frame += spec["frames"]
        return web.json_response({"sections": timeline})

    @routes.get("/hz3/studio/projects")
    async def projects(request):
        folder = _studio_folder()
        files = sorted(folder.glob("*.mixmash"), key=lambda path: path.stat().st_mtime, reverse=True) if folder.is_dir() else []
        catalog = []
        for path in files:
            project = _project_of(path)
            style = next((line.strip() for line in str(project.get("style", "")).splitlines() if line.strip()), "")
            catalog.append({
                "name": path.stem,
                "updated": path.stat().st_mtime,
                "source": (project.get("source") or {}).get("original"),
                "source_file": (project.get("source") or {}).get("filename"),
                "analyzed": bool(project.get("analysis")),
                "takes": len(project.get("takes", [])),
                "voices": len(project.get("voices", [])),
                "sections": len(re.findall(r"^\s*\[[^\]\n]+\]\s*$", str(project.get("lyrics", "")), re.M)),
                "style": style[:160],
                "kind": project.get("kind", "song"),
                "license": project.get("license"),
                "album": project.get("album") or DEFAULT_ALBUM,
                "notes": project.get("notes", ""),
                "archived": bool(project.get("archived")),
                "styles": [{"id": style.get("id"), "name": style.get("name"), "kind": style.get("kind")}
                           for style in project.get("styles") or [] if isinstance(style, dict)],
            })
        return web.json_response({"projects": catalog})

    @routes.get("/hz3/studio/albums")
    async def albums(request):
        folder = _studio_folder() / "albums"
        names = {path.stem for path in folder.glob("*.json")} if folder.is_dir() else set()
        return web.json_response({"albums": sorted(names | {DEFAULT_ALBUM}, key=str.casefold)})

    @routes.get("/hz3/studio/album")
    async def load_album(request):
        path = _album_path(request.rel_url.query.get("name", ""))
        if not path.is_file():
            return web.json_response({"name": path.stem, "styles": [], "chat": [], "log": []})
        return web.json_response(json.loads(path.read_text(encoding="utf-8")))

    @routes.post("/hz3/studio/album")
    async def save_album(request):
        album = await request.json()
        path = _album_path(str(album.get("name", "")))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(album, ensure_ascii=False, indent=1), encoding="utf-8")
        temporary.replace(path)
        return web.json_response({"saved": path.stem})

    @routes.get("/hz3/studio/project")
    async def load_project(request):
        path = _package_path(request.rel_url.query.get("name", ""))
        if not path.is_file():
            raise web.HTTPNotFound(text="Project not found.")
        with zipfile.ZipFile(path) as package:
            return web.json_response(_read_package(package))

    @routes.post("/hz3/studio/project")
    async def save_project(request):
        project = await request.json()
        await asyncio.to_thread(_write_package, project)
        return web.json_response({"saved": project["name"]})

    @routes.post("/hz3/studio/rename")
    async def rename_project(request):
        body = await request.json()
        source, target = _package_path(str(body.get("from", ""))), _package_path(str(body.get("to", "")))
        if not source.is_file():
            raise web.HTTPNotFound(text="Project not found.")
        if target.exists():
            raise web.HTTPConflict(text=f"A project named {target.stem!r} already exists.")
        source.replace(target)
        # The manifest carries the name too; a stale one would write the song back under its old name.
        project = _project_of(target)
        project["name"] = target.stem
        await asyncio.to_thread(_write_package, project)
        return web.json_response({"name": target.stem})

    @routes.post("/hz3/studio/delete")
    async def delete_project(request):
        path = _package_path(str((await request.json()).get("name", "")))
        if not path.is_file():
            raise web.HTTPNotFound(text="Project not found.")
        _discard(path)
        return web.json_response({"deleted": path.stem})

    @routes.post("/hz3/studio/album/rename")
    async def rename_album(request):
        body = await request.json()
        source, target = _album_path(str(body.get("from", ""))), _album_path(str(body.get("to", "")))
        if target.exists():
            raise web.HTTPConflict(text=f"An album named {target.stem!r} already exists.")
        await asyncio.to_thread(_move_album_songs, source.stem, target.stem)
        album = json.loads(source.read_text(encoding="utf-8")) if source.is_file() else {"styles": [], "chat": [], "log": []}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({**album, "name": target.stem}, ensure_ascii=False, indent=1), encoding="utf-8")
        source.unlink(missing_ok=True)
        return web.json_response({"name": target.stem})

    @routes.post("/hz3/studio/album/delete")
    async def delete_album(request):
        path = _album_path(str((await request.json()).get("name", "")))
        if path.stem == DEFAULT_ALBUM:
            raise web.HTTPBadRequest(text=f"The {DEFAULT_ALBUM!r} album cannot be deleted.")
        await asyncio.to_thread(_move_album_songs, path.stem, DEFAULT_ALBUM)
        if path.is_file():
            _discard(path)
        return web.json_response({"deleted": path.stem})

    @routes.post("/hz3/studio/source")
    async def upload_source(request):
        name = request.rel_url.query.get("project", "")
        if not PROJECT_NAME.fullmatch(name):
            raise web.HTTPBadRequest(text="Name the project before adding its audio.")
        field = (await request.post()).get("file")
        original = Path(getattr(field, "filename", "") or "")
        if original.suffix.lower() not in AUDIO_TYPES:
            raise web.HTTPBadRequest(text=f"Use an audio file ({', '.join(AUDIO_TYPES)}).")
        stem = re.sub(r"[^\w \-]+", "_", original.stem)[:80] or "audio"
        filename = f"hz3studio_{name}_{stem}{original.suffix.lower()}"
        data = field.file.read()
        if not data:
            raise web.HTTPBadRequest(text="The audio file is empty.")
        _source_path(filename).write_bytes(data)
        return web.json_response({"filename": filename, "original": original.name})

    @routes.get("/hz3/studio/package")
    async def download_package(request):
        path = _package_path(request.rel_url.query.get("name", ""))
        if not path.is_file():
            raise web.HTTPNotFound(text="Project not found.")
        return web.FileResponse(path, headers={"Content-Disposition": f"attachment; filename*=UTF-8''{urllib.parse.quote(path.name)}"})

    @routes.post("/hz3/studio/package")
    async def import_package(request):
        field = (await request.post()).get("file")
        data = field.file.read()
        try:
            package = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as error:
            raise web.HTTPBadRequest(text="Not a .mixmash package.") from error
        with package:
            project = _read_package(package)
        path = _package_path(str(project.get("name", "")))
        if path.exists():
            raise web.HTTPConflict(text=f"A project named {path.stem!r} already exists.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return web.json_response({"name": path.stem})

    @routes.post("/hz3/studio/warp")
    async def warp(request):
        body = await request.json()
        knots = await asyncio.to_thread(_warp, _output_file(body["file"]), body["abc"], body.get("line", "lead"))
        return web.json_response({"abc": knots[0].tolist(), "audio": knots[1].tolist(), "sure": knots[2].tolist()})

    @routes.get("/hz3/studio/loras")
    async def loras(request):
        return web.json_response({"loras": await asyncio.to_thread(_yue2_loras)})

    @routes.post("/hz3/studio/stage")
    async def stage(request):
        body = await request.json()
        name = str(body.get("name", ""))
        if not PROJECT_NAME.fullmatch(name):
            raise web.HTTPBadRequest(text="Invalid LoRA name.")
        if not body.get("files"):
            raise web.HTTPBadRequest(text="Choose at least one audio.")
        paths = [_output_file(file) for file in body["files"]]
        return web.json_response({"filename": await asyncio.to_thread(_stage_audio, paths, name)})

    @routes.post("/hz3/studio/classical")
    async def classical(request):
        body = await request.json()
        library = await asyncio.to_thread(load_library, REFERENCES) if REFERENCES.is_dir() else []
        library += await asyncio.to_thread(_reference_themes)
        if not library:
            return web.json_response({"error": f"Add analyzed reference recordings or MusicXML scores in {REFERENCES}."}, status=400)
        abc, report = add_lines(body["abc"], library, {name.casefold(): role for name, role in body["plan"].items()})
        return web.json_response({"abc": abc, "report": report})

    @routes.post("/hz3/studio/arrange")
    async def arrange(request):
        body = await request.json()
        try:
            plan = await asyncio.to_thread(_arrange, body.get("style", ""), body.get("instructions", ""),
                                           body["sections"], body.get("model") or "deepseek-v4.1-flash:cloud")
        except RuntimeError as error:
            return web.json_response({"error": str(error)}, status=502)
        return web.json_response({"plan": plan})

    @routes.post("/hz3/studio/assistant")
    async def assistant(request):
        body = await request.json()
        try:
            answer = await asyncio.to_thread(_assist, body.get("model") or "deepseek-v4.1-flash:cloud",
                                             body["state"], body["messages"], body["tools"])
        except RuntimeError as error:
            return web.json_response({"error": str(error)}, status=502)
        return web.json_response(answer)
