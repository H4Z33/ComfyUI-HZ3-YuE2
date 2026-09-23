"""Split the native two-voice YuE2 ABC format into instrumental, section, and vocal scores."""

from __future__ import annotations

import re

try:
    from .abc_score import TOKEN, parse
except (ImportError, ValueError):
    from abc_score import TOKEN, parse


def _expanded_bars(music_line: str) -> list[str]:
    if not music_line.endswith("|"):
        raise ValueError("Every ABC music line must end with a barline.")
    bars = [bar.strip() for bar in music_line[:-1].split("|")]
    if any(not bar for bar in bars):
        raise ValueError("Empty measures are not supported in native ABC.")
    expanded = []
    for bar in bars:
        rest = re.fullmatch(r"Z([2-4])?", bar)
        expanded.extend(["Z"] * int(rest.group(1) or 1) if rest else [bar])
    return expanded


def _rest_line(music_line: str) -> str:
    count = len(_expanded_bars(music_line))
    return f"Z{count}|" if count > 1 else "Z|"


def _notes_to_rests_keep_chords(music_line: str) -> str:
    transformed = []
    for bar in _expanded_bars(music_line):
        if re.fullmatch(r"Z", bar):
            transformed.append("Z")
            continue

        def replace_token(match):
            if match.group("chord") is not None:
                return match.group(0)
            return f"z{match.group('duration')}"

        blanked = TOKEN.sub(replace_token, bar).strip()
        transformed.append(blanked or "Z")
    return "|".join(transformed) + "|"


def _remove_chords(music_line: str) -> str:
    transformed = []
    for bar in _expanded_bars(music_line):
        if re.fullmatch(r"Z", bar):
            transformed.append("Z")
            continue
        clean = re.sub(r'"[^"\n]*"', "", bar).strip()
        transformed.append(clean or "Z")
    return "|".join(transformed) + "|"


def _transform(abc_text: str, target: str) -> str:
    text = str(abc_text or "").replace("\r\n", "\n").strip() + "\n"
    if not text.strip():
        raise ValueError("Connect a non-empty native ABC score.")
    if len(text) > 200_000:
        raise ValueError("ABC input must be 200,000 characters or less.")

    score = parse(text)
    lines = text.splitlines()
    output = []
    for index, line in enumerate(lines):
        voice = score.music_lines.get(index)
        if voice is None:
            output.append(line)
            continue

        if target == "sections":
            output.append(_rest_line(line))
        elif target == "instrumental" and voice == "Vocal":
            output.append(_notes_to_rests_keep_chords(line))
        elif target == "vocals" and voice == "Ins":
            output.append(_rest_line(line))
        elif target == "vocals" and voice == "Vocal":
            output.append(_remove_chords(line))
        else:
            output.append(line)

    result = "\n".join(output) + "\n"
    # Validate the transformed grid and durations before returning any outputs.
    parse(result)
    return result


def separate_abc(abc_text: str) -> tuple[str, str, str, str]:
    """Return ABC with instrumental content, section scaffold, and vocal melody."""
    instrumental = _transform(abc_text, "instrumental")
    sections = _transform(abc_text, "sections")
    vocals = _transform(abc_text, "vocals")
    score = parse(abc_text.replace("\r\n", "\n").strip() + "\n")
    section_count = sum(line.startswith("% ") for line in abc_text.replace("\r\n", "\n").splitlines())
    report = (
        f"ABC Separator · {len(score.voices['Vocal'].bars)} bars · {section_count} section markers · "
        f"{score.bpm} BPM\n"
        "ABC_Instrumental keeps the Ins part and chord symbols while resting the Vocal notes.\n"
        "ABC_Sections keeps the timing grid and section comments as rests.\n"
        "ABC_Vocals keeps the Vocal melody without chord symbols and rests the Ins part."
    )
    return instrumental, sections, vocals, report


class HZ3_YuE2_ABCSeparator:
    CATEGORY = "HZ3 YuE2/Score"
    FUNCTION = "separate"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("ABC_Instrumental", "ABC_Sections", "ABC_Vocals", "report")
    DESCRIPTION = (
        "Split native two-voice ABC while preserving headers, section markers, bar counts, and timing. "
        "Instrumental keeps the Ins part and chord symbols; Sections emits a rest-only timing scaffold; "
        "Vocals keeps only the Vocal melody."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"abc": ("STRING", {"forceInput": True, "multiline": True})}}

    def separate(self, abc):
        return {"result": separate_abc(abc)}


NODE_CLASS_MAPPINGS = {"HZ3_YuE2_ABCSeparator": HZ3_YuE2_ABCSeparator}
NODE_DISPLAY_NAME_MAPPINGS = {"HZ3_YuE2_ABCSeparator": "ABC Separator"}
