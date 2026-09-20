"""HZ3 YuE2 · Karaoke & Audio Visualizer.

Generates synchronized karaoke video with progressive text sweep,
chords, song section HUD, and audio-reactive spectrum or oscilloscope.
Outputs ComfyUI IMAGE batch [F, H, W, 3], passthrough AUDIO, and saved MP4 video.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
import time
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from .score_align import align_and_repair_abc
    from .score_analysis import inspect_score
except (ImportError, ValueError):
    from score_align import align_and_repair_abc
    from score_analysis import inspect_score

logger = logging.getLogger("HZ3.KaraokeVisualizer")


# --- COLOR THEMES ---
THEMES: dict[str, dict[str, Any]] = {
    "Cyberpunk Neon": {
        "bg_top": (10, 8, 22),
        "bg_bottom": (22, 14, 42),
        "bar_low": (0, 242, 254),
        "bar_high": (255, 42, 133),
        "peak_color": (255, 230, 100),
        "pill_bg": (35, 25, 65, 200),
        "pill_border": (0, 242, 254),
        "section_text": (255, 200, 80),
        "chord_text": (0, 242, 254),
        "info_text": (180, 185, 215),
        "lyrics_base": (200, 205, 220),
        "lyrics_highlight": (0, 255, 240),
        "lyrics_glow": (0, 200, 255, 120),
        "progress_fill": (0, 242, 254),
    },
    "Midnight Gold": {
        "bg_top": (12, 12, 16),
        "bg_bottom": (26, 24, 20),
        "bar_low": (245, 158, 11),
        "bar_high": (253, 224, 71),
        "peak_color": (255, 255, 240),
        "pill_bg": (40, 35, 25, 200),
        "pill_border": (245, 158, 11),
        "section_text": (251, 191, 36),
        "chord_text": (253, 224, 71),
        "info_text": (210, 200, 180),
        "lyrics_base": (215, 210, 200),
        "lyrics_highlight": (253, 224, 71),
        "lyrics_glow": (245, 158, 11, 120),
        "progress_fill": (245, 158, 11),
    },
    "Sunset Horizon": {
        "bg_top": (20, 10, 32),
        "bg_bottom": (45, 18, 38),
        "bar_low": (250, 112, 154),
        "bar_high": (254, 225, 64),
        "peak_color": (255, 255, 255),
        "pill_bg": (50, 25, 55, 200),
        "pill_border": (250, 112, 154),
        "section_text": (254, 225, 64),
        "chord_text": (250, 112, 154),
        "info_text": (225, 190, 210),
        "lyrics_base": (220, 210, 220),
        "lyrics_highlight": (255, 130, 92),
        "lyrics_glow": (250, 112, 154, 120),
        "progress_fill": (250, 112, 154),
    },
    "Matrix Green": {
        "bg_top": (6, 18, 12),
        "bg_bottom": (10, 32, 20),
        "bar_low": (16, 185, 129),
        "bar_high": (110, 231, 183),
        "peak_color": (220, 255, 235),
        "pill_bg": (15, 45, 30, 200),
        "pill_border": (16, 185, 129),
        "section_text": (110, 231, 183),
        "chord_text": (52, 211, 153),
        "info_text": (165, 215, 190),
        "lyrics_base": (195, 220, 205),
        "lyrics_highlight": (52, 211, 153),
        "lyrics_glow": (16, 185, 129, 120),
        "progress_fill": (16, 185, 129),
    },
    "Clean Studio Dark": {
        "bg_top": (15, 23, 42),
        "bg_bottom": (30, 41, 59),
        "bar_low": (56, 189, 248),
        "bar_high": (129, 140, 248),
        "peak_color": (255, 255, 255),
        "pill_bg": (35, 48, 70, 200),
        "pill_border": (56, 189, 248),
        "section_text": (129, 140, 248),
        "chord_text": (56, 189, 248),
        "info_text": (200, 215, 230),
        "lyrics_base": (226, 232, 240),
        "lyrics_highlight": (56, 189, 248),
        "lyrics_glow": (56, 189, 248, 120),
        "progress_fill": (56, 189, 248),
    },
}

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1280x720 (16:9 HD)": (1280, 720),
    "1920x1080 (16:9 FHD)": (1920, 1080),
    "854x480 (16:9 SD)": (854, 480),
    "720x1280 (9:16 Vertical)": (720, 1280),
    "1080x1920 (9:16 Shorts/Reels)": (1080, 1920),
}


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a crisp system TrueType font with fallbacks."""
    font_candidates = (
        ["arialbd.ttf", "segoeuib.ttf", "calibrib.ttf", "arial.ttf"]
        if bold
        else ["arial.ttf", "segoeui.ttf", "calibri.ttf"]
    )
    for name in font_candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Linear interpolate between two RGB colors."""
    t = max(0.0, min(1.0, float(t)))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


class HZ3_YuE2_KaraokeVisualizer:
    CATEGORY = "HZ3 YuE2/Score"
    FUNCTION = "generate_karaoke"
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "video_path")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Render a synchronized karaoke video with progressive text highlighting, active chord tracking, "
        "musical section HUD, and an audio-reactive visualizer (spectrum bars / oscilloscope)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc": ("STRING", {"multiline": True, "forceInput": True}),
                "lyrics": ("STRING", {"multiline": True, "forceInput": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "resolution": (list(RESOLUTIONS.keys()), {"default": "1280x720 (16:9 HD)"}),
                "fps": ("INT", {"default": 30, "min": 15, "max": 60, "step": 1}),
                "theme": (list(THEMES.keys()), {"default": "Cyberpunk Neon"}),
                "visualizer_mode": (
                    ["Spectrum Bars", "Mirrored Spectrum", "Waveform Oscilloscope"],
                    {"default": "Spectrum Bars"},
                ),
                "font_size": ("INT", {"default": 38, "min": 20, "max": 72, "step": 2}),
                "filename_prefix": ("STRING", {"default": "video/HZ3-Karaoke"}),
                "save_video": ("BOOLEAN", {"default": True}),
                "render_image_batch": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Output IMAGE tensor to ComfyUI. Recommended True for clips (<60s).",
                    },
                ),
                "max_duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 1.0,
                        "tooltip": "Optional maximum duration in seconds (0 = full length of audio/score).",
                    },
                ),
            },
        }

    def _parse_timeline(self, raw_abc: str, lyrics_text: str, total_duration: float):
        """Align score and lyrics into timed sections, lines, words, chords, and notes."""
        repaired_abc, structure = align_and_repair_abc(raw_abc, lyrics_text=lyrics_text)
        info = inspect_score(repaired_abc, lyrics_text)

        bpm = float(info.get("bpm", 120))
        meter = info.get("meter", "4/4")
        key = info.get("key", "C")
        score_seconds = float(info.get("seconds", 0.0))

        duration = max(total_duration, score_seconds)
        if duration <= 0:
            duration = 10.0

        # Vocal and Instrumental note events
        vocal_notes_raw = info.get("roll", {}).get("tracks", {}).get("Vocal", [])
        ins_notes_raw = info.get("roll", {}).get("tracks", {}).get("Ins", [])

        vocal_events = []
        for n in vocal_notes_raw:
            t_start = float(n["start"] / 256.0) * (60.0 / bpm)
            t_dur = float(n["duration"] / 256.0) * (60.0 / bpm)
            vocal_events.append({
                "start": t_start,
                "end": t_start + t_dur,
                "duration": t_dur,
                "pitch": int(n.get("pitch", 60)),
            })

        ins_events = []
        for n in ins_notes_raw:
            t_start = float(n["start"] / 256.0) * (60.0 / bpm)
            t_dur = float(n["duration"] / 256.0) * (60.0 / bpm)
            ins_events.append({
                "start": t_start,
                "end": t_start + t_dur,
                "duration": t_dur,
                "pitch": int(n.get("pitch", 48)),
            })

        # Chords timeline
        chords_raw = info.get("roll", {}).get("chords", [])
        chords = []
        for c in chords_raw:
            t_start = float(c["start"] / 256.0) * (60.0 / bpm)
            chords.append({"start": t_start, "symbol": c.get("symbol", "")})
        chords.sort(key=lambda x: x["start"])

        # Sections timeline
        sections_raw = info.get("roll", {}).get("sections", [])
        sections = []
        bar_ticks = info.get("roll", {}).get("bar_ticks", 1024)
        for s in sections_raw:
            t_start = float(s["start"] * bar_ticks / 256.0) * (60.0 / bpm)
            t_end = t_start + float(s["bars"] * bar_ticks / 256.0) * (60.0 / bpm)
            sections.append({
                "name": s.get("name", "section").upper(),
                "start": t_start,
                "end": t_end,
            })
        if not sections:
            sections = [{"name": "MUSIC", "start": 0.0, "end": duration}]

        # Flat lines fallback if no headers or structure
        flat_lines = [l.strip() for l in lyrics_text.splitlines() if l.strip() and not (l.startswith("[") and l.endswith("]"))]

        # Group vocal notes into phrases (cluster notes separated by pauses > 0.4s)
        vocal_phrases = []
        curr_phrase = []
        for n in vocal_events:
            if not curr_phrase:
                curr_phrase.append(n)
            else:
                gap = n["start"] - curr_phrase[-1]["end"]
                if gap > 0.40:
                    vocal_phrases.append(curr_phrase)
                    curr_phrase = [n]
                else:
                    curr_phrase.append(n)
        if curr_phrase:
            vocal_phrases.append(curr_phrase)

        # Build timed karaoke lines
        timed_lines = []
        if flat_lines and vocal_phrases:
            num_lines = len(flat_lines)
            num_phrases = len(vocal_phrases)
            for idx, text in enumerate(flat_lines):
                # Map line index to phrase
                p_idx = int(idx * num_phrases / num_lines)
                phrase = vocal_phrases[min(p_idx, num_phrases - 1)]
                l_start = phrase[0]["start"]
                l_end = phrase[-1]["end"]
                if l_end <= l_start:
                    l_end = l_start + 2.0

                # Compute word timestamps
                words = text.split()
                timed_words = []
                w_dur = (l_end - l_start) / max(1, len(words))
                for w_i, w in enumerate(words):
                    timed_words.append({
                        "text": w,
                        "start": l_start + w_i * w_dur,
                        "end": l_start + (w_i + 1) * w_dur,
                    })

                timed_lines.append({
                    "text": text,
                    "start": l_start,
                    "end": l_end,
                    "words": timed_words,
                })
        elif flat_lines:
            # Fallback: distribute evenly over duration
            line_dur = duration / max(1, len(flat_lines))
            for idx, text in enumerate(flat_lines):
                l_start = idx * line_dur
                l_end = (idx + 1) * line_dur
                words = text.split()
                w_dur = (l_end - l_start) / max(1, len(words))
                timed_words = [
                    {"text": w, "start": l_start + w_i * w_dur, "end": l_start + (w_i + 1) * w_dur}
                    for w_i, w in enumerate(words)
                ]
                timed_lines.append({
                    "text": text,
                    "start": l_start,
                    "end": l_end,
                    "words": timed_words,
                })

        return {
            "bpm": int(bpm),
            "meter": meter,
            "key": key,
            "duration": duration,
            "vocal_events": vocal_events,
            "ins_events": ins_events,
            "chords": chords,
            "sections": sections,
            "lines": timed_lines,
        }

    def _get_active_chord(self, chords: list[dict], t: float) -> str:
        """Find the chord sounding at time t."""
        active = ""
        for c in chords:
            if c["start"] <= t:
                active = c["symbol"]
            else:
                break
        return active

    def _get_active_section(self, sections: list[dict], t: float) -> str:
        """Find the musical section active at time t."""
        for s in sections:
            if s["start"] <= t <= s["end"]:
                return s["name"]
        if sections:
            return sections[-1]["name"] if t > sections[-1]["end"] else sections[0]["name"]
        return "SONG"

    def _get_active_line_index(self, lines: list[dict], t: float) -> int:
        """Find the line index currently active or nearest."""
        if not lines:
            return -1
        for idx, line in enumerate(lines):
            if line["start"] <= t <= line["end"]:
                return idx
        # If in a gap between lines
        for idx in range(len(lines) - 1):
            if lines[idx]["end"] < t < lines[idx + 1]["start"]:
                # If closer to next line, show next line upcoming
                if t - lines[idx]["end"] > lines[idx + 1]["start"] - t:
                    return idx + 1
                return idx
        if t < lines[0]["start"]:
            return 0
        return len(lines) - 1

    def _render_frame(
        self,
        t: float,
        timeline: dict,
        spectrum_vals: np.ndarray,
        peaks: np.ndarray,
        width: int,
        height: int,
        theme: dict,
        mode: str,
        font_main: ImageFont.FreeTypeFont,
        font_sub: ImageFont.FreeTypeFont,
        font_hud: ImageFont.FreeTypeFont,
    ) -> Image.Image:
        """Render a single high-quality video frame with Pillow."""
        # 1. Background Gradient
        frame_img = Image.new("RGB", (width, height), theme["bg_top"])
        draw = ImageDraw.Draw(frame_img, "RGBA")

        # Fast 4-pixel vertical gradient
        for y_step in range(0, height, 4):
            frac = y_step / float(height)
            col = _lerp_color(theme["bg_top"], theme["bg_bottom"], frac)
            draw.rectangle([0, y_step, width, y_step + 4], fill=col)

        # Subtle ambient spotlight glow behind lyrics
        spot_center_y = int(height * 0.36)
        spot_r = int(min(width, height) * 0.40)
        draw.ellipse(
            [width // 2 - spot_r, spot_center_y - spot_r // 2, width // 2 + spot_r, spot_center_y + spot_r // 2],
            fill=(theme["bar_low"][0] // 10, theme["bar_low"][1] // 10, theme["bar_low"][2] // 10, 45),
        )

        # 2. Top HUD Bar
        hud_y = int(height * 0.04)
        active_sec = self._get_active_section(timeline["sections"], t)
        active_chord = self._get_active_chord(timeline["chords"], t)

        # Section pill (Top-Left)
        sec_text = f"  {active_sec}  "
        sec_bbox = font_hud.getbbox(sec_text)
        sec_w = sec_bbox[2] - sec_bbox[0] + 16
        sec_h = sec_bbox[3] - sec_bbox[1] + 10
        draw.rounded_rectangle(
            [40, hud_y, 40 + sec_w, hud_y + sec_h],
            radius=6,
            fill=theme["pill_bg"],
            outline=theme["pill_border"],
            width=1,
        )
        draw.text((48, hud_y + 4), sec_text.strip(), fill=theme["section_text"], font=font_hud)

        # Song info (Top-Center)
        info_str = f"Key: {timeline['key']}  ·  {timeline['meter']}  ·  {timeline['bpm']} BPM"
        draw.text((width // 2, hud_y + sec_h // 2), info_str, fill=theme["info_text"], font=font_hud, anchor="mm")

        # Chord pill (Top-Right)
        if active_chord:
            chord_str = f"  {active_chord}  "
            c_bbox = font_hud.getbbox(chord_str)
            c_w = c_bbox[2] - c_bbox[0] + 16
            c_h = c_bbox[3] - c_bbox[1] + 10
            draw.rounded_rectangle(
                [width - 40 - c_w, hud_y, width - 40, hud_y + c_h],
                radius=6,
                fill=theme["pill_bg"],
                outline=theme["chord_text"],
                width=1,
            )
            draw.text((width - 40 - c_w + 8, hud_y + 4), chord_str.strip(), fill=theme["chord_text"], font=font_hud)

        # 3. Karaoke Lyrics Display
        lines = timeline["lines"]
        active_idx = self._get_active_line_index(lines, t)
        lyric_center_y = int(height * 0.38)
        line_spacing = int(font_main.size * 1.35)

        if 0 <= active_idx < len(lines):
            active_line = lines[active_idx]

            # (a) Previous line (above, dimmed)
            if active_idx > 0:
                prev_text = lines[active_idx - 1]["text"]
                p_font = font_sub
                p_bbox = p_font.getbbox(prev_text)
                p_w = p_bbox[2] - p_bbox[0]
                if p_w > width - 80:
                    p_scale = max(14, int(font_sub.size * (width - 100) / max(1, p_w)))
                    p_font = _get_font(p_scale, bold=False)
                draw.text(
                    (width // 2, lyric_center_y - line_spacing),
                    prev_text,
                    fill=(theme["lyrics_base"][0], theme["lyrics_base"][1], theme["lyrics_base"][2], 75),
                    font=p_font,
                    anchor="mm",
                )

            # (b) Next line (below, upcoming)
            if active_idx + 1 < len(lines):
                next_text = lines[active_idx + 1]["text"]
                n_font = font_sub
                n_bbox = n_font.getbbox(next_text)
                n_w = n_bbox[2] - n_bbox[0]
                if n_w > width - 80:
                    n_scale = max(14, int(font_sub.size * (width - 100) / max(1, n_w)))
                    n_font = _get_font(n_scale, bold=False)
                draw.text(
                    (width // 2, lyric_center_y + line_spacing),
                    next_text,
                    fill=(theme["lyrics_base"][0], theme["lyrics_base"][1], theme["lyrics_base"][2], 115),
                    font=n_font,
                    anchor="mm",
                )

            # (c) Active line with progressive two-tone karaoke sweep!
            line_text = active_line["text"]
            line_start = active_line["start"]
            line_end = active_line["end"]

            cur_font = font_main
            bbox = cur_font.getbbox(line_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            if text_w > width - 100:
                scaled_size = max(18, int(font_main.size * (width - 120) / max(1, text_w)))
                cur_font = _get_font(scaled_size, bold=True)
                bbox = cur_font.getbbox(line_text)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

            x_start = max(20, (width - text_w) // 2)
            y_start = lyric_center_y - text_h // 2

            # Base unsung text (off-white)
            draw.text((x_start, y_start), line_text, fill=theme["lyrics_base"], font=cur_font)

            # Progressive Highlight Sweep
            if t >= line_start:
                prog = 1.0 if t >= line_end else (t - line_start) / max(0.01, (line_end - line_start))
                prog = max(0.0, min(1.0, prog))
                fill_w = int(text_w * prog)
                right_bound = x_start + fill_w
                left_bound = x_start

                if right_bound > left_bound and fill_w > 0:
                    hl_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                    hl_draw = ImageDraw.Draw(hl_img)

                    # Soft glow pass behind highlight text
                    for offset in (-1, 0, 1):
                        hl_draw.text(
                            (x_start + offset, y_start),
                            line_text,
                            fill=theme["lyrics_glow"],
                            font=cur_font,
                        )

                    # Crisp main highlight text
                    hl_draw.text((x_start, y_start), line_text, fill=theme["lyrics_highlight"], font=cur_font)

                    # Crop safely with strict left <= right bounds
                    crop_left = max(0, min(width - 1, left_bound))
                    crop_right = max(crop_left + 1, min(width, right_bound))
                    crop_box = (crop_left, 0, crop_right, height)
                    hl_clipped = hl_img.crop(crop_box)
                    frame_img.paste(hl_clipped, (crop_left, 0), hl_clipped)

                    # Glowing karaoke bouncy pointer at singing head
                    if t < line_end:
                        ball_x = max(10, min(width - 10, right_bound))
                        ball_y = y_start - 6 + int(3.0 * math.sin(t * 12.0))
                        draw.ellipse(
                            [ball_x - 4, ball_y - 4, ball_x + 4, ball_y + 4],
                            fill=theme["lyrics_highlight"],
                            outline=(255, 255, 255),
                        )
        else:
            # Instrumental or waiting text
            draw.text(
                (width // 2, lyric_center_y),
                "( Instrumental )",
                fill=(theme["lyrics_base"][0], theme["lyrics_base"][1], theme["lyrics_base"][2], 120),
                font=font_main,
                anchor="mm",
            )

        # 4. Audio Reactive Visualizer
        num_bars = len(spectrum_vals)
        margin_x = int(width * 0.08)
        avail_w = width - 2 * margin_x
        bar_w = max(2, int(avail_w / num_bars))
        gap = max(1, int(bar_w * 0.20))
        actual_bar_w = bar_w - gap

        if mode == "Spectrum Bars":
            base_y = int(height * 0.90)
            max_bar_h = int(height * 0.28)
            for b in range(num_bars):
                val = float(spectrum_vals[b])
                h = int(val * max_bar_h)
                bx = margin_x + b * bar_w
                by = base_y - h
                bar_col = _lerp_color(theme["bar_low"], theme["bar_high"], val)

                if h > 2:
                    draw.rounded_rectangle([bx, by, bx + actual_bar_w, base_y], radius=3, fill=bar_col)

                # Floating peak hold dot
                pk = float(peaks[b])
                pk_y = base_y - int(pk * max_bar_h) - 4
                draw.rectangle([bx, pk_y, bx + actual_bar_w, pk_y + 2], fill=theme["peak_color"])

        elif mode == "Mirrored Spectrum":
            center_y = int(height * 0.78)
            max_half_h = int(height * 0.14)
            for b in range(num_bars):
                val = float(spectrum_vals[b])
                h = int(val * max_half_h)
                bx = margin_x + b * bar_w
                bar_col = _lerp_color(theme["bar_low"], theme["bar_high"], val)

                if h > 2:
                    draw.rounded_rectangle(
                        [bx, center_y - h, bx + actual_bar_w, center_y + h], radius=3, fill=bar_col
                    )
                pk = float(peaks[b])
                pk_h = int(pk * max_half_h)
                draw.rectangle([bx, center_y - pk_h - 3, bx + actual_bar_w, center_y - pk_h - 1], fill=theme["peak_color"])
                draw.rectangle([bx, center_y + pk_h + 1, bx + actual_bar_w, center_y + pk_h + 3], fill=theme["peak_color"])

        elif mode == "Waveform Oscilloscope":
            center_y = int(height * 0.78)
            amp = int(height * 0.12)
            pts = []
            for b in range(num_bars):
                val = float(spectrum_vals[b])
                bx = margin_x + b * bar_w + actual_bar_w // 2
                by = center_y + int(math.sin(b * 0.4 + t * 6.0) * val * amp)
                pts.append((bx, by))

            if len(pts) > 1:
                draw.line(pts, fill=theme["lyrics_glow"], width=6)
                draw.line(pts, fill=theme["bar_low"], width=3)
                draw.line(pts, fill=(255, 255, 255, 220), width=1)

        # 5. Bottom Progress Bar & Timecodes
        prog_y = height - 16
        tot_dur = max(0.1, timeline["duration"])
        p_frac = max(0.0, min(1.0, t / tot_dur))

        # Track
        draw.rectangle([margin_x, prog_y, width - margin_x, prog_y + 3], fill=(50, 55, 75))
        # Filled progress
        draw.rectangle([margin_x, prog_y, margin_x + int((width - 2 * margin_x) * p_frac), prog_y + 3], fill=theme["progress_fill"])

        # Time labels
        cur_min, cur_sec = int(t // 60), int(t % 60)
        tot_min, tot_sec = int(tot_dur // 60), int(tot_dur % 60)
        draw.text((margin_x, prog_y - 14), f"{cur_min:02d}:{cur_sec:02d}", fill=theme["info_text"], font=font_hud)
        draw.text(
            (width - margin_x, prog_y - 14),
            f"{tot_min:02d}:{tot_sec:02d}",
            fill=theme["info_text"],
            font=font_hud,
            anchor="ra",
        )

        return frame_img

    def generate_karaoke(
        self,
        abc: str,
        lyrics: str,
        audio: dict | None = None,
        resolution: str = "1280x720 (16:9 HD)",
        fps: int = 30,
        theme: str = "Cyberpunk Neon",
        visualizer_mode: str = "Spectrum Bars",
        font_size: int = 38,
        filename_prefix: str = "video/HZ3-Karaoke",
        save_video: bool = True,
        render_image_batch: bool = True,
        max_duration: float = 0.0,
    ):
        t_start = time.time()
        width, height = RESOLUTIONS.get(resolution, (1280, 720))
        theme_cfg = THEMES.get(theme, THEMES["Cyberpunk Neon"])

        # 1. Resolve Audio Information
        has_audio = audio is not None and isinstance(audio, dict) and "waveform" in audio
        audio_duration = 0.0
        audio_sr = 44100
        waveform_mono = None
        waveform_tensor = None

        if has_audio:
            waveform_tensor = audio["waveform"]
            audio_sr = int(audio.get("sample_rate", 44100))
            if waveform_tensor.dim() == 3:
                w_2d = waveform_tensor[0]
            else:
                w_2d = waveform_tensor
            audio_duration = float(w_2d.shape[-1]) / float(audio_sr)
            waveform_mono = w_2d.mean(dim=0).cpu().numpy()

        # 2. Build Timeline
        timeline = self._parse_timeline(abc, lyrics, audio_duration)
        total_duration = timeline["duration"]
        if max_duration > 0.0:
            total_duration = min(total_duration, float(max_duration))
        timeline["duration"] = total_duration
        total_frames = max(1, int(total_duration * fps))

        # Fonts
        font_main = _get_font(font_size, bold=True)
        font_sub = _get_font(int(font_size * 0.65), bold=False)
        font_hud = _get_font(max(14, int(font_size * 0.45)), bold=True)

        # Visualizer frequency bands
        num_bands = 48
        freq_bands = np.geomspace(40, 14000, num_bands + 1)
        smooth_bars = np.zeros(num_bands, dtype=np.float32)
        peaks = np.zeros(num_bands, dtype=np.float32)

        # 3. Setup Video Output File
        output_dir = (
            folder_paths.get_output_directory() if folder_paths else os.path.join(os.getcwd(), "output")
        )
        if folder_paths:
            full_output_folder, file_basename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, output_dir
            )
        else:
            full_output_folder = os.path.join(output_dir, "video")
            file_basename = "HZ3-Karaoke"
            counter = 1
            subfolder = "video"

        os.makedirs(full_output_folder, exist_ok=True)
        video_filename = f"{file_basename}_{counter:05}_.mp4"
        video_full_path = os.path.join(full_output_folder, video_filename)

        # Setup PyAV Container
        container = av.open(video_full_path, mode="w")

        # Video stream: CPU-only libx264 ensures ZERO GPU VRAM usage and zero contention with active CUDA renders
        v_stream = None
        for codec_name in ("libx264", "mpeg4"):
            try:
                v_stream = container.add_stream(codec_name, rate=fps)
                v_stream.width = width
                v_stream.height = height
                v_stream.pix_fmt = "yuv420p"
                if codec_name == "libx264":
                    v_stream.options = {"preset": "fast", "crf": "21"}
                break
            except Exception:
                continue

        if v_stream is None:
            raise RuntimeError("Failed to initialize video encoder with libx264 or mpeg4.")

        # Audio stream
        a_stream = None
        if has_audio:
            try:
                a_stream = container.add_stream("aac", rate=audio_sr)
                a_stream.layout = "mono" if w_2d.shape[0] == 1 else "stereo"
            except Exception as e:
                logger.warning(f"Could not initialize AAC audio stream: {e}")

        # 4. Render & Encode Loop
        image_batch_list = []
        fft_win = 2048
        half_win = fft_win // 2
        hanning = np.hanning(fft_win)

        for frame_idx in range(total_frames):
            cur_time = float(frame_idx) / float(fps)

            # Compute Spectrum Bars
            if has_audio and waveform_mono is not None:
                center_sample = int(cur_time * audio_sr)
                start_s = max(0, center_sample - half_win)
                end_s = min(len(waveform_mono), center_sample + half_win)
                chunk = waveform_mono[start_s:end_s]
                if len(chunk) < fft_win:
                    chunk = np.pad(chunk, (0, fft_win - len(chunk)))

                fft_mag = np.abs(np.fft.rfft(chunk * hanning))
                fft_freqs = np.fft.rfftfreq(fft_win, 1.0 / audio_sr)

                raw_bars = np.zeros(num_bands, dtype=np.float32)
                for b_i in range(num_bands):
                    f_lo, f_hi = freq_bands[b_i], freq_bands[b_i + 1]
                    mask = (fft_freqs >= f_lo) & (fft_freqs < f_hi)
                    if np.any(mask):
                        band_energy = np.mean(fft_mag[mask])
                        dB = 20.0 * np.log10(band_energy + 1e-5)
                        norm_val = np.clip((dB + 50.0) / 48.0, 0.0, 1.0)
                        raw_bars[b_i] = float(norm_val)

                smooth_bars = smooth_bars * 0.60 + raw_bars * 0.40
            else:
                raw_bars = np.zeros(num_bands, dtype=np.float32)
                for vn in timeline["vocal_events"]:
                    if vn["start"] <= cur_time <= vn["end"]:
                        p_bin = int((vn["pitch"] - 40) / (84 - 40) * num_bands)
                        p_bin = np.clip(p_bin, 0, num_bands - 1)
                        raw_bars[p_bin] = 1.0
                        if p_bin > 0:
                            raw_bars[p_bin - 1] = max(raw_bars[p_bin - 1], 0.6)
                        if p_bin < num_bands - 1:
                            raw_bars[p_bin + 1] = max(raw_bars[p_bin + 1], 0.6)

                for in_n in timeline["ins_events"]:
                    if in_n["start"] <= cur_time <= in_n["end"]:
                        i_bin = int((in_n["pitch"] - 36) / (72 - 36) * num_bands)
                        i_bin = np.clip(i_bin, 0, num_bands - 1)
                        raw_bars[i_bin] = max(raw_bars[i_bin], 0.75)

                beat_phase = (cur_time * timeline["bpm"] / 60.0) % 1.0
                kick = math.exp(-beat_phase * 6.0)
                raw_bars[0:4] = np.maximum(raw_bars[0:4], float(kick * 0.90))

                smooth_bars = smooth_bars * 0.70 + raw_bars * 0.30

            peaks = np.maximum(peaks - 0.022, smooth_bars)

            frame_img = self._render_frame(
                cur_time,
                timeline,
                smooth_bars,
                peaks,
                width,
                height,
                theme_cfg,
                visualizer_mode,
                font_main,
                font_sub,
                font_hud,
            )

            v_frame = av.VideoFrame.from_image(frame_img)
            for p in v_stream.encode(v_frame):
                container.mux(p)

            if render_image_batch and (total_frames <= 1200 or frame_idx % 2 == 0):
                arr = np.array(frame_img, dtype=np.float32) / 255.0
                image_batch_list.append(arr)

        for p in v_stream.encode():
            container.mux(p)

        # 5. Mux Audio Stream
        if has_audio and a_stream is not None and waveform_tensor is not None:
            chunk_size = 1024
            w_2d_cpu = w_2d.cpu()
            total_samples = min(w_2d_cpu.shape[-1], int(total_duration * audio_sr))
            layout = "mono" if w_2d_cpu.shape[0] == 1 else "stereo"
            pts = 0

            for a_start in range(0, total_samples, chunk_size):
                a_end = min(a_start + chunk_size, total_samples)
                chunk = w_2d_cpu[:, a_start:a_end]
                if chunk.shape[1] < chunk_size:
                    chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[1]))

                a_frame = av.AudioFrame.from_ndarray(
                    chunk.movedim(0, 1).reshape(1, -1).float().numpy(),
                    format="flt",
                    layout=layout,
                )
                a_frame.sample_rate = audio_sr
                a_frame.pts = pts
                pts += chunk_size
                for p in a_stream.encode(a_frame):
                    container.mux(p)

            for p in a_stream.encode():
                container.mux(p)

        container.close()

        # 6. Format Outputs for ComfyUI
        elapsed = time.time() - t_start
        fps_rendered = float(total_frames) / max(0.01, elapsed)

        if image_batch_list:
            images_tensor = torch.from_numpy(np.stack(image_batch_list, axis=0))
        else:
            first_frame = self._render_frame(
                0.0,
                timeline,
                smooth_bars,
                peaks,
                width,
                height,
                theme_cfg,
                visualizer_mode,
                font_main,
                font_sub,
                font_hud,
            )
            images_tensor = torch.from_numpy(np.array(first_frame, dtype=np.float32) / 255.0).unsqueeze(0)

        if not has_audio:
            silent_wav = torch.zeros((1, 2, int(audio_sr * min(total_duration, 1.0))))
            audio_out = {"waveform": silent_wav, "sample_rate": audio_sr}
        else:
            audio_out = audio

        report = (
            f"Rendered Karaoke & Visualizer · {total_frames} frames ({total_duration:.1f}s) @ {fps}fps\n"
            f"Resolution: {width}x{height} · Speed: {fps_rendered:.1f} fps ({elapsed:.2f}s total)\n"
            f"Saved MP4: {video_full_path}"
        )

        ui_output = {
            "images": [
                {
                    "filename": video_filename,
                    "subfolder": subfolder,
                    "type": "output",
                    "format": "video/mp4",
                }
            ],
            "animated": (True,),
            "text": [report],
        }

        return {
            "ui": ui_output,
            "result": (images_tensor, audio_out, str(video_full_path)),
        }


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_KaraokeVisualizer": HZ3_YuE2_KaraokeVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_KaraokeVisualizer": "HZ3 YuE2 · Karaoke & Audio Visualizer",
}
