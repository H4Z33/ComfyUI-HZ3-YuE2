"""HZ3 Studio: a local song editor page served by ComfyUI at /hz3/studio.

Generation goes through ComfyUI's own /prompt queue from the page; this module
only serves the page, the section timeline of a song, and project files under
output/HZ3-YuE2/studio.
"""

import json
from pathlib import Path
import re

from aiohttp import web

import folder_paths
from comfy.text_encoders.yue2 import FRAMES_PER_SECOND

from .section_generation import _build_section_specs


STATIC = Path(__file__).parent / "studio"
PROJECT_NAME = re.compile(r"[\w \-]{1,64}")


def _project_path(name):
    if not PROJECT_NAME.fullmatch(name):
        raise web.HTTPBadRequest(text="Project names may use letters, numbers, spaces, '-' and '_' (64 max).")
    return Path(folder_paths.get_output_directory()) / "HZ3-YuE2" / "studio" / f"{name}.json"


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
        files = sorted(folder.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True) if folder.is_dir() else []
        return web.json_response({"projects": [path.stem for path in files]})

    @routes.get("/hz3/studio/project")
    async def load_project(request):
        path = _project_path(request.rel_url.query.get("name", ""))
        if not path.is_file():
            raise web.HTTPNotFound(text="Project not found.")
        return web.json_response(json.loads(path.read_text(encoding="utf-8")))

    @routes.post("/hz3/studio/project")
    async def save_project(request):
        project = await request.json()
        path = _project_path(str(project.get("name", "")))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(project, ensure_ascii=False, indent=1), encoding="utf-8")
        return web.json_response({"saved": path.stem})
