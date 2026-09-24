"""ABC piano roll with editable lyrics pinned to manually chosen section bars."""

from __future__ import annotations

import hashlib
import json
import re

try:
    from .abc_score import parse
    from .score_analysis import inspect_score
except (ImportError, ValueError):
    from abc_score import parse
    from score_analysis import inspect_score


_REST_BARS = re.compile(r"Z([2-4])?")
_LYRIC_HEADER = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_MAX_STATE_CHARS = 300_000
_MAX_SECTIONS = 64


def _raw_bars(music_line: str) -> list[str]:
    """Split a native music line into actual measures, expanding Z2-Z4."""
    measures = []
    for part in music_line[:-1].split("|"):
        part = part.strip()
        rest = _REST_BARS.fullmatch(part)
        measures.extend(["Z"] * int(rest.group(1) or "1") if rest else [part])
    return measures


def _extract_abc(score_abc: str):
    info = inspect_score(score_abc)
    score = parse(info["abc"])
    lines = info["abc"].splitlines()
    bars = [{"number": i + 1, "vocal": "Z", "ins": "Z", "meter": ""}
            for i in range(len(score.voices["Vocal"].bars))]
    directives: dict[int, dict[str, list[str]]] = {}
    seen = {"Vocal": 0, "Ins": 0}

    for line_index, voice in sorted(score.music_lines.items()):
        measures = _raw_bars(lines[line_index])
        voice_line = max(i for i in range(line_index) if lines[i] == f"V: {voice}")
        fields = [line for line in lines[voice_line + 1:line_index]
                  if line.startswith(("M:", "K:"))]
        if fields:
            directives.setdefault(seen[voice], {})[voice] = fields
        for offset, measure in enumerate(measures):
            index = seen[voice] + offset
            bars[index]["vocal" if voice == "Vocal" else "ins"] = measure
            if voice == "Vocal":
                bar_start, bar_duration, bar_meter = score.voices[voice].bars[index]
                bars[index]["start"] = int(bar_start * 256)
                bars[index]["duration"] = int(bar_duration * 256)
                bars[index]["meter"] = f"{score.voices[voice].bars[index][2][0]}/{score.voices[voice].bars[index][2][1]}"
        seen[voice] += len(measures)

    markers = []
    pending_marker = None
    vocal_bar_index = 0
    for line_index, line in enumerate(lines):
        if line.startswith("% "):
            pending_marker = line[2:].strip()
        if score.music_lines.get(line_index) == "Vocal":
            if pending_marker:
                markers.append({"name": pending_marker, "start": vocal_bar_index})
            pending_marker = None
            vocal_bar_index += len(_raw_bars(line))

    tracks = {
        name: [{"start": int(start * 256), "pitch": int(pitch), "duration": int(duration * 256)}
               for start, pitch, duration in voice.notes]
        for name, voice in score.voices.items()
    }
    chords = [{"start": int(start * 256), "symbol": symbol}
              for start, symbol in score.voices["Vocal"].chords]
    total_ticks = int(score.voices["Vocal"].time * 256)
    return info, bars, directives, markers, tracks, chords, total_ticks


def _parse_lyrics(lyrics: str):
    rows, markers = [], []
    for raw_line in str(lyrics or "").replace("\r\n", "\n").replace("\r", "\n").splitlines():
        match = _LYRIC_HEADER.fullmatch(raw_line)
        if match:
            name = match.group(1).strip()
            if name and not re.fullmatch(r"\d{1,2}:\d{2}(?:\.\d+)?", name):
                markers.append({"name": name, "start": len(rows)})
            continue
        if raw_line.strip():
            rows.append(raw_line.strip())
    return rows, markers


def _name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def _seed_starts(markers, section_names, total):
    """Use matching source labels as anchors and interpolate only missing cuts."""
    count = len(section_names)
    if count <= 1:
        return [0]
    anchors = {0: 0, count: total}
    search_from = 0
    for marker in markers:
        key = _name_key(marker["name"])
        section_index = next((i for i in range(search_from, count)
                              if _name_key(section_names[i]) == key), None)
        if section_index is None:
            continue
        search_from = section_index + 1
        if section_index > 0:
            anchors[section_index] = max(0, min(total, int(marker["start"])))
    ordered = sorted(anchors.items())
    starts = [0] * (count + 1)
    for (left_i, left_pos), (right_i, right_pos) in zip(ordered, ordered[1:]):
        right_pos = max(left_pos, right_pos)
        for index in range(left_i, right_i + 1):
            fraction = (index - left_i) / (right_i - left_i)
            starts[index] = round(left_pos + (right_pos - left_pos) * fraction)
    return starts[:-1]


def _initial_sections(lyric_markers, abc_markers, lyric_lines, bar_count):
    # Prefer the richer labeled source; use labels from the other source to seed
    # missing names, but never compute lyric-to-note/syllable alignment.
    primary = lyric_markers if len(lyric_markers) >= len(abc_markers) else abc_markers
    secondary = abc_markers if primary is lyric_markers else lyric_markers
    names = [marker["name"] for marker in primary]
    if not names:
        names = [marker["name"] for marker in secondary]
    if not names:
        names = ["Section 1"]
    for marker in secondary:
        if len(names) >= max(len(lyric_markers), len(abc_markers)):
            break
        if _name_key(marker["name"]) not in {_name_key(name) for name in names}:
            names.append(marker["name"])
    desired = min(max(1, bar_count), max(1, len(names), len(lyric_markers), len(abc_markers)))
    names = names[:desired]
    while len(names) < desired:
        names.append(f"Section {len(names) + 1}")

    lyric_starts = _seed_starts(lyric_markers, names, len(lyric_lines))
    abc_starts = _seed_starts(abc_markers, names, bar_count)
    if names:
        abc_starts[0] = 0
        for index in range(1, len(names)):
            lower = abc_starts[index - 1] + 1
            upper = bar_count - (len(names) - index)
            abc_starts[index] = max(lower, min(upper, abc_starts[index]))
    sections = []
    for index, name in enumerate(names):
        lyric_start = lyric_starts[index]
        lyric_end = lyric_starts[index + 1] if index + 1 < len(names) else len(lyric_lines)
        end_bar = abc_starts[index + 1] - 1 if index + 1 < len(names) else bar_count - 1
        sections.append({
            "id": f"section-{index + 1}",
            "name": name,
            "start_bar": abc_starts[index],
            "end_bar": end_bar,
            "lyrics": "\n".join(lyric_lines[lyric_start:lyric_end]),
        })
    return sections


def _normalize_state(state_text, source_hash, lyric_lines, bars, lyric_markers, abc_markers):
    state = None
    if isinstance(state_text, str) and len(state_text) <= _MAX_STATE_CHARS:
        try:
            candidate = json.loads(state_text)
            if isinstance(candidate, dict) and candidate.get("source_hash") == source_hash:
                state = candidate
        except (json.JSONDecodeError, TypeError):
            pass

    if state is None:
        sections = _initial_sections(lyric_markers, abc_markers, lyric_lines, len(bars))
        return {"source_hash": source_hash, "editor_version": 3, "sections": sections}

    raw_sections = state.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections or len(raw_sections) > _MAX_SECTIONS:
        raw_sections = _initial_sections(lyric_markers, abc_markers, lyric_lines, len(bars))

    # Migrate saved state from the old independent lyrics/ABC range editor.
    is_new_state = all(isinstance(raw, dict) and "start_bar" in raw and "lyrics" in raw
                       for raw in raw_sections)
    if not is_new_state:
        old_lines = state.get("lyric_lines", lyric_lines)
        if not isinstance(old_lines, list) or len(old_lines) > 20_000:
            old_lines = lyric_lines
        old_lines = [str(line).replace("\r", " ").replace("\n", " ")[:2_000] for line in old_lines]
        migrated, lyric_cursor, bar_cursor = [], 0, 0
        for index, raw in enumerate(raw_sections):
            if not isinstance(raw, dict):
                continue
            try:
                lyric_end = max(lyric_cursor, min(len(old_lines), int(raw.get("lyrics_end", len(old_lines)))))
            except (TypeError, ValueError):
                lyric_end = len(old_lines)
            try:
                old_end = max(bar_cursor, min(len(bars), int(raw.get("abc_end", len(bars)))))
            except (TypeError, ValueError):
                old_end = len(bars)
            migrated.append({
                "id": str(raw.get("id") or f"section-{index + 1}")[:100],
                "name": raw.get("name") or f"Section {index + 1}",
                "start_bar": bar_cursor,
                "end_bar": max(bar_cursor, old_end - 1),
                "lyrics": "\n".join(old_lines[lyric_cursor:lyric_end]),
            })
            lyric_cursor, bar_cursor = lyric_end, old_end
        raw_sections = migrated or _initial_sections(lyric_markers, abc_markers, lyric_lines, len(bars))

    raw_sections = [raw for raw in raw_sections if isinstance(raw, dict)]
    if not raw_sections:
        raw_sections = _initial_sections(lyric_markers, abc_markers, lyric_lines, len(bars))
    raw_sections.sort(key=lambda raw: _safe_int(raw.get("start_bar"), 0))
    raw_sections = raw_sections[:min(_MAX_SECTIONS, len(bars))]
    starts, previous_start = [], -1
    for index, raw in enumerate(raw_sections):
        requested = max(0, min(len(bars) - 1, _safe_int(raw.get("start_bar"), 0)))
        upper = len(bars) - (len(raw_sections) - index)
        start_bar = max(0, min(upper, requested)) if index == 0 else max(previous_start + 1, min(upper, requested))
        starts.append(start_bar)
        previous_start = start_bar

    sections = []
    for index, raw in enumerate(raw_sections):
        name = re.sub(r"[\r\n\[\]]+", " ", str(raw.get("name") or f"Section {index + 1}")).strip()[:100]
        name = name or f"Section {index + 1}"
        start_bar = starts[index]
        end_limit = starts[index + 1] - 1 if index + 1 < len(starts) else len(bars) - 1
        end_bar = max(start_bar, min(end_limit, _safe_int(raw.get("end_bar"), end_limit)))
        text = str(raw.get("lyrics") or "").replace("\r\n", "\n").replace("\r", "\n")[:50_000]
        sections.append({"id": str(raw.get("id") or f"section-{index + 1}")[:100],
                         "name": name, "start_bar": start_bar, "end_bar": end_bar, "lyrics": text})

    if not sections:
        sections = _initial_sections(lyric_markers, abc_markers, lyric_lines, len(bars))
    return {"source_hash": source_hash, "editor_version": 3, "sections": sections}


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_abc(score_abc, bars, directives, sections):
    header = score_abc.replace("\r\n", "\n").replace("\r", "\n").strip().splitlines()[:8]
    out = list(header)
    sections_by_start = {section["start_bar"]: section for section in sections}
    cursor = 0
    while cursor < len(bars):
        section = sections_by_start.get(cursor)
        if section:
            label = section["name"].replace("\n", " ").replace("\r", " ").strip() or "Section"
            out.append(f"% {label}")
        next_section = min((start for start in sections_by_start if start > cursor), default=len(bars))
        next_change = min((index for index in directives if index > cursor), default=len(bars))
        chunk_end = min(len(bars), cursor + 4, next_section, next_change)
        fields = directives.get(cursor, {})
        out.append("V: Vocal")
        out.extend(fields.get("Vocal", []))
        out.append("|".join(bar["vocal"] for bar in bars[cursor:chunk_end]) + "|")
        out.append("V: Ins")
        out.extend(fields.get("Ins", []))
        out.append("|".join(bar["ins"] for bar in bars[cursor:chunk_end]) + "|")
        cursor = chunk_end
    return "\n".join(out) + "\n"


def _build_lyrics(sections):
    out = []
    for section in sections:
        name = section["name"].replace("\n", " ").replace("\r", " ").replace("]", " ").strip()
        out.append(f"[{name or 'Section'}]")
        text = section["lyrics"].strip()
        if text:
            out.extend(text.splitlines())
    return "\n".join(out) + "\n"


def _state_json(state):
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def viewer_data(score_abc, lyrics="", editor_state=""):
    info, bars, directives, abc_markers, tracks, chords, total_ticks = _extract_abc(score_abc)
    lyric_lines, lyric_markers = _parse_lyrics(lyrics)
    source_hash = hashlib.sha256((info["abc"] + "\0" + str(lyrics or "")).encode("utf-8")).hexdigest()[:20]
    state = _normalize_state(editor_state, source_hash, lyric_lines, bars, lyric_markers, abc_markers)

    edited_abc = _build_abc(info["abc"], bars, directives, state["sections"])
    # Reparse the result so boundary placement cannot silently damage valid ABC.
    parse(edited_abc)
    edited_lyrics = _build_lyrics(state["sections"])
    report = {
        "sections": [{
            "name": section["name"],
            "start_bar": section["start_bar"],
            "end_bar": section["end_bar"],
            "lyrics": section["lyrics"],
        } for section in state["sections"]],
        "note": "Each section has manually adjustable starting and ending bars. Lyrics belong to the color-coded section range.",
    }
    return {
        "abc": info["abc"],
        "lyrics": str(lyrics or ""),
        "summary": info["summary"],
        "bpm": info["bpm"],
        "meter": info["meter"],
        "key": info["key"],
        "bars": bars,
        "tracks": tracks,
        "chords": chords,
        "total_ticks": total_ticks,
        "sections": state["sections"],
        "source_hash": source_hash,
        "editor_state": _state_json(state),
        "edited_abc": edited_abc,
        "edited_lyrics": edited_lyrics,
        "section_report": json.dumps(report, ensure_ascii=False, indent=2),
        "rough_layout": not (len(lyric_markers) == len(abc_markers) and
                              [_name_key(item["name"]) for item in lyric_markers] ==
                              [_name_key(item["name"]) for item in abc_markers]),
    }


class HZ3_YuE2_ABCViewer:
    CATEGORY = "HZ3 YuE2/Score"
    FUNCTION = "view"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("abc_with_sections", "lyrics_with_sections", "section_map")
    OUTPUT_NODE = True
    DESCRIPTION = "View resolved ABC notes in a piano roll and edit lyrics with explicit section bar ranges."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "score_abc": ("STRING", {"forceInput": True}),
                "lyrics": ("STRING", {"forceInput": True}),
                "editor_state": ("STRING", {"default": "", "multiline": True}),
                "loaded_abc": ("STRING", {"default": "", "multiline": True}),
                "loaded_lyrics": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def view(self, score_abc="", lyrics="", editor_state="", loaded_abc="", loaded_lyrics=""):
        # A pair loaded in the editor takes precedence until cleared by the user.
        effective_abc = loaded_abc or score_abc
        effective_lyrics = loaded_lyrics if loaded_abc else lyrics
        if not str(effective_abc or "").strip():
            raise ValueError("Connect an ABC input or load an ABC + lyrics pair in the editor.")
        data = viewer_data(effective_abc, effective_lyrics, editor_state)
        return {
            "ui": {"abc_viewer": [data]},
            "result": (data["edited_abc"], data["edited_lyrics"], data["section_report"]),
        }


NODE_CLASS_MAPPINGS = {"HZ3_YuE2_ABCViewer": HZ3_YuE2_ABCViewer}
NODE_DISPLAY_NAME_MAPPINGS = {"HZ3_YuE2_ABCViewer": "HZ3 YuE2 · ABC Piano Roll + Section Lyrics"}


# The editor uses this route to preview imported ABC/lyrics pairs and to export
# current edits without requiring a workflow execution first.
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/hz3/yue2/abc_viewer/data")
    async def _api_abc_viewer_data(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            score_abc = payload.get("abc", "")
            lyrics = payload.get("lyrics", "")
            if not isinstance(score_abc, str) or not score_abc.strip():
                return web.json_response({"error": "The ABC file is empty or invalid."}, status=400)
            if not isinstance(lyrics, str):
                return web.json_response({"error": "The lyrics must be text."}, status=400)

            editor_state = payload.get("editor_state", "")
            sections = payload.get("sections")
            if isinstance(sections, list):
                base = viewer_data(score_abc, lyrics)
                editor_state = _state_json({
                    "source_hash": base["source_hash"],
                    "editor_version": 3,
                    "sections": sections,
                })
            result = viewer_data(score_abc, lyrics, editor_state)
            return web.json_response(result)
        except Exception as error:
            return web.json_response({"error": str(error)}, status=400)
except (ImportError, AttributeError):
    # Unit tests and non-ComfyUI imports do not have the server API available.
    pass
