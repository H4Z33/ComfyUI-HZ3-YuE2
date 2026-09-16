"""Split a whole-song ABC into per-section ABC fragments.

`HZ3_YuE2_SheetSage2Sections` already transcribes the FULL audio once, so there is
no need to re-transcribe each short clip (which is what triggered the coarse-grid
MelodyVoiceError). This node cuts the existing ABC text at its `% [section]`
markers and returns one independent ABC fragment per section, index-aligned with
`segments_audio[i]` / `structure[i]`. Each fragment keeps the shared header
(X:/T:/M:/L:/Q:/V:/K:) plus that section's measures for both voices.
"""

from __future__ import annotations

import json

import torch  # noqa: F401 (kept for parity; node is text-only)


def split_abc_by_sections(abc):
    """Return a list of (name, abc_fragment) in section order from the raw ABC text.

    The header is the text up to the `K:` line. After it, `% [section]` lines and the
    `V: Vocal` / `V: Ins` measure lines follow; a new section starts at each `% label`.
    """
    lines = (abc or "").replace("\r\n", "\n").split("\n")
    header_end = None
    for index, line in enumerate(lines):
        if line.startswith("K:"):
            header_end = index
            break
    if header_end is None:
        header_end = 0
    header = lines[:header_end + 1]
    body = lines[header_end + 1:]

    sections = []
    current = None
    for line in body:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("% "):
            name = stripped[2:].strip() or "section"
            current = {"name": name, "lines": []}
            sections.append(current)
        elif current is None:
            name = "section"
            current = {"name": name, "lines": []}
            sections.append(current)
        current["lines"].append(line.rstrip())

    fragments = []
    for section in sections:
        text = "\n".join(header + section["lines"]).strip() + "\n"
        if text.strip():
            fragments.append((section["name"], text))
    return fragments


class HZ3_YuE2_ABCPerSections:
    CATEGORY = "HZ3 YuE2/Score"
    FUNCTION = "split"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("abc_list", "names", "report")
    OUTPUT_IS_LIST = (True, False, False)
    DESCRIPTION = (
        "Split the whole-song ABC into one ABC fragment per section by its '% [section]' "
        "markers. Returns a LIST index-aligned with segments_audio/structure (abc_list[i] "
        "covers the i-th section), so you don't re-transcribe short clips."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc": ("STRING", {"multiline": True, "forceInput": True,
                                   "tooltip": "The whole-song ABC from 'HZ3 YuE2 · SheetSage2 Audio to ABC + Sections'."}),
            },
            "optional": {
                "structure": ("STRING", {"multiline": True, "forceInput": True,
                                         "tooltip": "Optional structure JSON: validates that the section count matches."}),
            },
        }

    def split(self, abc, structure=""):
        fragments = split_abc_by_sections(abc)
        if not fragments:
            raise ValueError("`abc` produced no section fragments; check the ABC text.")
        names = [name for name, _ in fragments]
        abc_list = [text for _, text in fragments]

        structure_count = None
        if (structure or "").strip():
            try:
                payload = json.loads(structure)
                if isinstance(payload, dict):
                    payload = payload.get("sections") or payload.get("structure") or payload.get("clips") or []
                if isinstance(payload, list):
                    structure_count = len([s for s in payload if isinstance(s, dict)
                                            and s.get("start") is not None and s.get("end") is not None])
            except json.JSONDecodeError:
                structure_count = None

        names_json = json.dumps(names, ensure_ascii=False, indent=2)
        note = ""
        if structure_count is not None and structure_count != len(fragments):
            note = (f" WARNING: ABC produced {len(fragments)} sections but structure has "
                    f"{structure_count}; alignment by index may drift.")
        report = (f"Split ABC into {len(fragments)} section fragment(s):\n"
                  + "\n".join(f"  [{i}] {name}" for i, name in enumerate(names))
                  + note)
        return {"ui": {"text": [report]}, "result": (abc_list, names_json, report)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_ABCPerSections": HZ3_YuE2_ABCPerSections,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_ABCPerSections": "HZ3 YuE2 · ABC Per Sections",
}
