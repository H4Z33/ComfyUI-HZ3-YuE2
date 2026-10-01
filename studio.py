"""HZ3 Studio: a local song editor page served by ComfyUI at /hz3/studio.

Analysis and generation go through ComfyUI's own /prompt queue from the page.
This module serves the page, the section timeline of a song, the harmony
arranger, and .mixmash packages: a zip with the source audio, the corrected
lyrics/ABC/style and the whole studio project, under output/HZ3-YuE2/studio.
"""

import asyncio
import io
import json
from pathlib import Path
import re
import urllib.error
import urllib.request
import zipfile

from aiohttp import web

import folder_paths
from comfy.text_encoders.yue2 import FRAMES_PER_SECOND

from .section_generation import _build_section_specs


STATIC = Path(__file__).parent / "studio"
ARRANGER_PROMPT = Path(__file__).parent / "prompts" / "harmony_arranger_system.txt"
VOICES = ("tenor", "baritone", "low", "bass", "countertenor")
OLLAMA = "http://127.0.0.1:11434"
PROJECT_NAME = re.compile(r"[\w \-]{1,64}")
AUDIO_TYPES = (".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac")
SOURCE_FILE = re.compile(r"hz3studio_[\w \-]{1,64}_[\w \-]{1,80}(" + "|".join(re.escape(ext) for ext in AUDIO_TYPES) + ")")
PACKAGE_FORMAT = "hz3-mixmash/1"


def _package_path(name):
    if not PROJECT_NAME.fullmatch(name):
        raise web.HTTPBadRequest(text="Project names may use letters, numbers, spaces, '-' and '_' (64 max).")
    return Path(folder_paths.get_output_directory()) / "HZ3-YuE2" / "studio" / f"{name}.mixmash"


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


def _arrange(style, instructions, sections, model):
    """Ask the local Ollama for the harmony voices of each section."""
    payload = {"style": style, "instructions": instructions, "sections": sections}
    body = {"model": model, "stream": False, "think": False, "format": "json", "options": {"temperature": 0.3},
            "messages": [{"role": "system", "content": ARRANGER_PROMPT.read_text(encoding="utf-8")},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
    request = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            content = json.loads(response.read().decode("utf-8"))["message"]["content"]
    except urllib.error.URLError as error:
        raise RuntimeError(f"Ollama is unavailable at {OLLAMA}: {error.reason}") from error
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        plan = json.loads(content)["sections"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"The arranger returned invalid JSON: {content[:300]}") from error
    names = {section["name"] for section in sections}
    return {entry["name"]: {"voices": [voice for voice in entry.get("voices", []) if voice in VOICES],
                            "reason": str(entry.get("reason", ""))}
            for entry in plan if isinstance(entry, dict) and entry.get("name") in names}


def register(routes):
    @routes.get("/hz3/studio")
    async def index(request):
        return web.FileResponse(STATIC / "index.html")

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
            timeline.append({"name": spec["name"], "lyrics": spec["lyrics"], "bars": spec["bars"],
                             "start": frame / FRAMES_PER_SECOND, "end": (frame + spec["frames"]) / FRAMES_PER_SECOND})
            frame += spec["frames"]
        return web.json_response({"sections": timeline})

    @routes.get("/hz3/studio/projects")
    async def projects(request):
        folder = Path(folder_paths.get_output_directory()) / "HZ3-YuE2" / "studio"
        files = sorted(folder.glob("*.mixmash"), key=lambda path: path.stat().st_mtime, reverse=True) if folder.is_dir() else []
        return web.json_response({"projects": [path.stem for path in files]})

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
        return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="{path.name}"'})

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

    @routes.post("/hz3/studio/arrange")
    async def arrange(request):
        body = await request.json()
        try:
            plan = await asyncio.to_thread(_arrange, body.get("style", ""), body.get("instructions", ""),
                                           body["sections"], body.get("model") or "deepseek-v4.1-flash:cloud")
        except RuntimeError as error:
            return web.json_response({"error": str(error)}, status=502)
        return web.json_response({"plan": plan})
