"""HZ3 YuE2 · Karaoke & Audio Visualizer.

Single node: times the clean lyrics against the generated song (audio forced alignment,
with Whisper-segment and ABC-score fallbacks) and renders a synchronized karaoke video
with progressive word sweep, chords, song section HUD, and an audio-reactive spectrum or
oscilloscope. Outputs a preview IMAGE, passthrough AUDIO, the saved MP4 path, and the
timed lyrics (JSON + LRC) plus an alignment report.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections import OrderedDict, deque
from functools import lru_cache
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from .score_lyric_aligner import (
        ALIGNMENT_MODES,
        align_lyrics,
        audio_to_mono16k,
        derive_sections_from_lines,
    )
except (ImportError, ValueError):
    from score_lyric_aligner import (
        ALIGNMENT_MODES,
        align_lyrics,
        audio_to_mono16k,
        derive_sections_from_lines,
    )

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

BACKGROUND_TRANSITIONS = ["Cut", "Crossfade", "Fade Through Black", "Wipe Left"]


class _BackgroundCarousel:
    """Load, sequence, transition, and composite a folder of images for video frames."""

    IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

    def __init__(
        self,
        folder: str,
        width: int,
        height: int,
        base: Image.Image,
        interval: float,
        mode: str,
        transition: str,
        transition_duration: float,
        transparency: float,
    ):
        self.width = int(width)
        self.height = int(height)
        self.base = base
        self.interval = max(0.1, float(interval))
        self.transition = transition if transition in BACKGROUND_TRANSITIONS else "Crossfade"
        self.transition_duration = min(self.interval, max(0.0, float(transition_duration)))
        self.opacity = 1.0 - max(0.0, min(100.0, float(transparency))) / 100.0
        self.paths = sorted(
            (os.path.join(folder, name) for name in os.listdir(folder)
             if os.path.splitext(name)[1].lower() in self.IMAGE_EXTENSIONS
             and os.path.isfile(os.path.join(folder, name))),
            key=lambda path: os.path.basename(path).casefold(),
        )
        if mode == "Random":
            random.SystemRandom().shuffle(self.paths)
            if len(self.paths) > 1 and self.paths[0] == self.paths[-1]:
                self.paths[0], self.paths[1] = self.paths[1], self.paths[0]
        self.cache: OrderedDict[str, Image.Image] = OrderedDict()
        self.cache_limit = 3
        self.black = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 255))

    def _load(self, path: str) -> Image.Image:
        cached = self.cache.get(path)
        if cached is not None:
            self.cache.move_to_end(path)
            return cached
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGBA")
                image = ImageOps.fit(image, (self.width, self.height), method=Image.Resampling.LANCZOS)
        except Exception as exc:
            logger.warning("Could not load visualizer background image '%s': %s", path, exc)
            image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        self.cache[path] = image
        self.cache.move_to_end(path)
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return image

    def frame(self, seconds: float) -> Image.Image:
        if not self.paths or self.opacity <= 0.0:
            return self.base.copy()

        slot = max(0, int(max(0.0, float(seconds)) // self.interval))
        current_index = slot % len(self.paths)
        current = self._load(self.paths[current_index])
        previous = None
        phase = max(0.0, float(seconds)) % self.interval
        in_transition = (
            slot > 0
            and self.transition != "Cut"
            and self.transition_duration > 0.0
            and phase < self.transition_duration
        )
        if in_transition:
            previous = self._load(self.paths[(current_index - 1) % len(self.paths)])
            amount = max(0.0, min(1.0, phase / self.transition_duration))
            if self.transition == "Crossfade":
                image = Image.blend(previous, current, amount)
            elif self.transition == "Fade Through Black":
                if amount < 0.5:
                    image = Image.blend(previous, self.black, amount * 2.0)
                else:
                    image = Image.blend(self.black, current, (amount - 0.5) * 2.0)
            else:  # Wipe Left
                mask = Image.new("L", (self.width, self.height), 0)
                ImageDraw.Draw(mask).rectangle((0, 0, int(self.width * amount), self.height), fill=255)
                image = Image.composite(current, previous, mask)
        else:
            image = current

        # The configured transparency is applied over the existing theme gradient.
        alpha = image.getchannel("A").point(lambda value: round(value * self.opacity))
        result = self.base.copy()
        result.paste(image.convert("RGB"), (0, 0), alpha)
        return result


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


class _RenderResources:
    """Bounded drawing resources owned by one video render."""

    def __init__(self, width: int, height: int, theme: dict):
        background = Image.new("RGB", (width, height), theme["bg_top"])
        draw = ImageDraw.Draw(background, "RGBA")

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

        self.background = background
        self.font = lru_cache(maxsize=128)(_get_font)
        unit_circle = [(math.cos(i * math.tau / 120), math.sin(i * math.tau / 120)) for i in range(121)]
        rotations = {angle: (math.cos(math.radians(angle)), math.sin(math.radians(angle))) for angle in (-30, 30)}

        @lru_cache(maxsize=512)
        def ellipse(cx, cy, rx, ry, angle):
            cos_a, sin_a = rotations[angle]
            return [(cx + rx * cosine * cos_a - ry * sine * sin_a,
                     cy + rx * cosine * sin_a + ry * sine * cos_a)
                    for cosine, sine in unit_circle]

        self.ellipse = ellipse
        self.trails = deque(maxlen=16)
        self.last_visual_time = None
        self.last_trail_time = -math.inf


class HZ3_YuE2_KaraokeVisualizer:
    CATEGORY = "HZ3 YuE2/Score"
    FUNCTION = "generate_karaoke"
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "video_path", "timed_lyrics", "lrc_text", "report")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Time the clean lyrics against the generated song (forced alignment on the vocal stem/audio, "
        "with Whisper-segment and ABC-score fallbacks) and render a synchronized karaoke video with "
        "progressive word highlighting, chord tracking, section HUD, and an audio-reactive visualizer. "
        "Cycles optional folder backgrounds with selectable timing, order, transparency, and transitions. "
        "Returns a first-frame IMAGE preview, timed lyrics as JSON and LRC, and the rendered MP4."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc": ("STRING", {"multiline": True, "forceInput": True}),
                "lyrics": ("STRING", {"multiline": True, "forceInput": True}),
            },
            "optional": {
                "audio": (
                    "AUDIO",
                    {"tooltip": "Generated song (full mix). Used for the video soundtrack, the visualizer, and as alignment audio when no vocal stem is connected."},
                ),
                "vocals": (
                    "AUDIO",
                    {"tooltip": "Optional separated vocal stem of the SAME audio (e.g. AudioSeparation.Vocals). Drives the central vocal waveform oscilloscope in Rolling Mode. Strongly recommended for accurate forced alignment."},
                ),
                "instrumental": (
                    "AUDIO",
                    {"tooltip": "Optional separated instrumental stem (e.g. AudioSeparation.Instrumental). Used as the audio fallback when neither audio nor vocals is connected."},
                ),
                "whisper_segments": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Optional Whisper JSON segments (HZ3 Transcribe.segments) or LRC text. Fallback line anchors when forced alignment is unavailable.",
                    },
                ),
                "alignment": (
                    list(ALIGNMENT_MODES.keys()),
                    {
                        "default": list(ALIGNMENT_MODES.keys())[0],
                        "tooltip": "Auto: forced alignment of the lyrics onto the audio (MMS_FA), then Whisper segments, then nominal ABC timing.",
                    },
                ),
                "alignment_device": (
                    ["auto", "cuda", "cpu"],
                    {"default": "auto", "tooltip": "Device for the forced-alignment model (~1.2 GB, downloaded once to models/mms_fa)."},
                ),
                "lyrics_offset": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -5.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Manual fine-tune in seconds applied to every lyric timestamp (negative = highlight earlier).",
                    },
                ),
                "resolution": (list(RESOLUTIONS.keys()), {"default": "1280x720 (16:9 HD)"}),
                "fps": ("INT", {"default": 30, "min": 15, "max": 60, "step": 1}),
                "theme": (list(THEMES.keys()), {"default": "Cyberpunk Neon"}),
                "background_folder": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Optional folder of background images. Images are listed alphabetically, or shuffled in Random mode."},
                ),
                "background_interval": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.5, "max": 300.0, "step": 0.5, "tooltip": "Seconds between background changes."},
                ),
                "background_mode": (
                    ["Alphabetical", "Random"],
                    {"default": "Alphabetical", "tooltip": "Cycle through the folder alphabetically or use a shuffled order."},
                ),
                "background_transition": (
                    BACKGROUND_TRANSITIONS,
                    {"default": "Crossfade", "tooltip": "Transition between background images."},
                ),
                "background_transition_duration": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Length of each transition in seconds (up to the change interval)."},
                ),
                "background_transparency": (
                    "FLOAT",
                    {"default": 25.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Transparency of the image over the visualizer's theme background (0 = opaque, 100 = invisible)."},
                ),
                "visualizer_mode": (
                    ["Spectrum Bars", "Mirrored Spectrum", "Waveform Oscilloscope", "Rolling Mode"],
                    {"default": "Spectrum Bars"},
                ),
                "font_size": ("INT", {"default": 38, "min": 20, "max": 72, "step": 2}),
                "filename_prefix": ("STRING", {"default": "video/HZ3-Karaoke"}),
                "save_video": ("BOOLEAN", {"default": True}),
                "encoder": (
                    ["Auto (NVENC / CPU)", "h264_nvenc (GPU)", "libx264 (CPU)"],
                    {
                        "default": "Auto (NVENC / CPU)",
                        "tooltip": "Video encoder. Auto uses Nvidia GPU hardware NVENC if available (~3x faster), with automatic fallback to CPU libx264.",
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

    def _parse_timeline(
        self,
        raw_abc: str,
        lyrics_text: str,
        audio_duration: float,
        *,
        align_mono16k: np.ndarray | None = None,
        align_is_stem: bool = False,
        whisper_segments: str = "",
        alignment_mode: str = "auto",
        alignment_device: str = "auto",
        lyrics_offset: float = 0.0,
    ) -> dict[str, Any]:
        """Time lyrics, chords, notes and sections for rendering.

        Lyrics come from the best available source (see score_lyric_aligner.align_lyrics).
        Chords and note events keep their nominal ABC clock, shifted by the measured
        audio-vs-score offset when the lyrics were aligned on audio. The section HUD then
        follows the aligned lyric blocks instead of the nominal ABC sections.
        """
        result = align_lyrics(
            raw_abc,
            lyrics_text,
            alignment_audio=align_mono16k,
            alignment_is_stem=align_is_stem,
            whisper_input=whisper_segments,
            mode=alignment_mode,
            device=alignment_device,
            time_offset=lyrics_offset,
            audio_seconds=audio_duration,
        )
        score = result["timeline"]
        lines = result["lines"]
        audio_based = result["source"] in ("forced", "whisper")

        if audio_duration > 0:
            duration = audio_duration
        else:
            last_line_end = max((l["end"] for l in lines if not l.get("beyond_audio")), default=0.0)
            duration = max(float(score["score_seconds"]), last_line_end + 2.0)
        if duration <= 0:
            duration = 10.0

        score_shift = float(result["score_offset"] or 0.0) if audio_based else 0.0

        def shifted(events: list[dict]) -> list[dict]:
            if not score_shift:
                return [dict(e) for e in events]
            out = []
            for e in events:
                item = dict(e)
                item["start"] = max(0.0, item["start"] + score_shift)
                if "end" in item:
                    item["end"] = max(item["start"], item["end"] + score_shift)
                out.append(item)
            return out

        vocal_events = shifted(score["vocal_notes"])
        ins_events = shifted(score["ins_notes"])
        chords = shifted(score["chords"])
        chords.sort(key=lambda x: x["start"])

        sections = derive_sections_from_lines(lines, duration) if audio_based else []
        if not sections:
            sections = [
                {"name": str(s["name"]).upper(), "start": s["start"], "end": s["end"]}
                for s in shifted(score["sections"])
            ]
        if not sections:
            sections = [{"name": "MUSIC", "start": 0.0, "end": duration}]

        return {
            "bpm": int(score["bpm"]),
            "meter": score["meter"],
            "key": score["key"],
            "duration": duration,
            "vocal_events": vocal_events,
            "ins_events": ins_events,
            "chords": chords,
            "sections": sections,
            "lines": [l for l in lines if not l.get("beyond_audio")],
            "alignment": result,
        }


    def _band_levels(
        self,
        waveform_mono: np.ndarray,
        audio_sr: int,
        fps: int,
        total_frames: int,
        freq_bands: np.ndarray,
        fft_win: int = 2048,
    ) -> np.ndarray:
        """Per-frame band magnitudes in dB, shape [total_frames, len(freq_bands) - 1]."""
        half_win = fft_win // 2
        num_bands = len(freq_bands) - 1
        padded = np.pad(np.asarray(waveform_mono, dtype=np.float32), (half_win, fft_win))
        centers = (np.arange(total_frames) / float(fps) * audio_sr).astype(np.int64)
        centers = np.clip(centers, 0, max(0, len(padded) - fft_win - 1))
        window = np.hanning(fft_win).astype(np.float32)
        fft_freqs = np.fft.rfftfreq(fft_win, 1.0 / audio_sr)
        membership = np.zeros((len(fft_freqs), num_bands), dtype=np.float32)
        for b_i in range(num_bands):
            mask = (fft_freqs >= freq_bands[b_i]) & (fft_freqs < freq_bands[b_i + 1])
            if not np.any(mask):  # very narrow low bands: take the nearest bin
                mask = np.zeros_like(mask)
                mask[int(np.argmin(np.abs(fft_freqs - freq_bands[b_i])))] = True
            membership[mask, b_i] = 1.0 / float(mask.sum())

        levels = np.zeros((total_frames, num_bands), dtype=np.float32)
        batch = 512
        for start in range(0, total_frames, batch):
            idx = centers[start:start + batch]
            chunks = padded[idx[:, None] + np.arange(fft_win)[None, :]] * window
            mags = np.abs(np.fft.rfft(chunks, axis=1)).astype(np.float32)
            levels[start:start + len(idx)] = 20.0 * np.log10(mags @ membership + 1e-5)
        return levels

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
        # Before first line starts
        if t < lines[0]["start"] - 2.5:
            return -1
        # After last line ends
        if t > lines[-1]["end"] + 3.0:
            return -1

        for idx, line in enumerate(lines):
            if line["start"] <= t <= line["end"]:
                return idx

        # If in a gap between lines
        for idx in range(len(lines) - 1):
            if lines[idx]["end"] < t < lines[idx + 1]["start"]:
                gap = lines[idx + 1]["start"] - lines[idx]["end"]
                # Long instrumental break between sections (> 4.5s)
                if gap > 4.5 and lines[idx]["end"] + 2.0 < t < lines[idx + 1]["start"] - 2.0:
                    return -1
                # Short pause: show previous line until halfway, then switch to next line
                if t - lines[idx]["end"] > (lines[idx + 1]["start"] - t):
                    return idx + 1
                return idx

        if t < lines[0]["start"]:
            return 0
        return len(lines) - 1

    def _frame_rms(
        self,
        mono: np.ndarray,
        audio_sr: int,
        fps: int,
        total_frames: int,
        win_ms: float = 50.0,
    ) -> np.ndarray:
        """Fast vectorized RMS amplitude envelope per video frame."""
        win_samples = max(64, int(audio_sr * (win_ms / 1000.0)))
        half = win_samples // 2
        padded = np.pad(np.asarray(mono, dtype=np.float32), (half, win_samples))
        centers = (np.arange(total_frames) / float(fps) * audio_sr).astype(np.int64)
        centers = np.clip(centers, 0, max(0, len(padded) - win_samples - 1))
        rms = np.zeros(total_frames, dtype=np.float32)
        batch = 1024
        for start in range(0, total_frames, batch):
            idx = centers[start:start + batch]
            chunks = padded[idx[:, None] + np.arange(win_samples)[None, :]]
            rms[start:start + len(idx)] = np.sqrt(np.mean(chunks ** 2, axis=1))
        return rms

    def _draw_karaoke_sweep(
        self,
        frame_img: Image.Image,
        draw: ImageDraw.ImageDraw,
        active_line: dict,
        x_start: int,
        y_start: int,
        text_w: int,
        cur_font: ImageFont.FreeTypeFont,
        theme: dict,
        t: float,
        width: int,
        height: int,
    ) -> None:
        """Draw two-tone progressive highlight sweep with glow and bouncing singing pointer."""
        line_text = active_line["text"]
        line_start = active_line["start"]
        line_end = active_line["end"]
        timed_words = active_line.get("words", [])

        if t < line_start:
            return

        if t >= line_end or not timed_words:
            prog = 1.0 if t >= line_end else (t - line_start) / max(0.01, (line_end - line_start))
            fill_w = int(text_w * max(0.0, min(1.0, prog)))
        else:
            idx_char = 0
            word_spans = []
            for tw in timed_words:
                w_str = tw["text"]
                pos = line_text.find(w_str, idx_char)
                if pos != -1:
                    w_left = cur_font.getlength(line_text[:pos])
                    w_right = cur_font.getlength(line_text[:pos + len(w_str)])
                    word_spans.append({
                        "start": tw["start"],
                        "end": tw["end"],
                        "left": w_left,
                        "right": w_right,
                        "width": max(1.0, w_right - w_left),
                    })
                    idx_char = pos + len(w_str)

            if word_spans:
                fill_w = 0
                for i, ws in enumerate(word_spans):
                    if t < ws["start"]:
                        if i > 0:
                            prev_ws = word_spans[i - 1]
                            gap_dur = ws["start"] - prev_ws["end"]
                            if gap_dur > 0.01:
                                gap_frac = min(1.0, (t - prev_ws["end"]) / gap_dur)
                                fill_w = int(prev_ws["right"] + gap_frac * (ws["left"] - prev_ws["right"]))
                            else:
                                fill_w = int(prev_ws["right"])
                        else:
                            fill_w = 0
                        break
                    elif ws["start"] <= t <= ws["end"]:
                        dur = max(0.01, ws["end"] - ws["start"])
                        w_prog = min(1.0, max(0.0, (t - ws["start"]) / dur))
                        fill_w = int(ws["left"] + w_prog * ws["width"])
                        break
                    else:
                        fill_w = int(ws["right"])
            else:
                prog = (t - line_start) / max(0.01, (line_end - line_start))
                fill_w = int(text_w * max(0.0, min(1.0, prog)))

        fill_w = max(0, min(text_w, fill_w))
        right_bound = x_start + fill_w
        left_bound = x_start

        if right_bound > left_bound and fill_w > 0:
            text_bbox = cur_font.getbbox(line_text)
            highlight_top = y_start + text_bbox[1]
            highlight_height = max(1, text_bbox[3] - text_bbox[1])
            hl_img = Image.new("RGBA", (width, highlight_height), (0, 0, 0, 0))
            hl_draw = ImageDraw.Draw(hl_img)

            # Soft glow pass behind highlight text
            for offset in (-1, 0, 1):
                hl_draw.text(
                    (x_start + offset, -text_bbox[1]),
                    line_text,
                    fill=theme["lyrics_glow"],
                    font=cur_font,
                )

            # Crisp main highlight text
            hl_draw.text((x_start, -text_bbox[1]), line_text, fill=theme["lyrics_highlight"], font=cur_font)

            # Crop safely with strict bounds
            crop_left = max(0, min(width - 1, left_bound))
            crop_right = max(crop_left + 1, min(width, right_bound))
            crop_top = max(0, highlight_top)
            crop_bottom = min(height, highlight_top + highlight_height)
            if crop_bottom > crop_top:
                crop_box = (crop_left, crop_top - highlight_top, crop_right, crop_bottom - highlight_top)
                hl_clipped = hl_img.crop(crop_box)
                frame_img.paste(hl_clipped, (crop_left, crop_top), hl_clipped)

            # Glowing karaoke bouncy pointer at singing head
            if t < line_end:
                ball_x = max(10, min(width - 10, right_bound))
                ball_y = y_start - 6 + int(3.0 * math.sin(t * 12.0))
                draw.ellipse(
                    [ball_x - 4, ball_y - 4, ball_x + 4, ball_y + 4],
                    fill=theme["lyrics_highlight"],
                    outline=(255, 255, 255),
                )

    def _render_speakers(
        self,
        draw: ImageDraw.ImageDraw,
        t: float,
        width: int,
        height: int,
        theme: dict,
        energies: tuple[float, float],
        render_resources: _RenderResources,
        bass_pulse: float = 0.0,
        opacity: float = 1.0,
    ) -> None:
        """Render mirrored waves expanding from lower audio-reactive centers."""
        xl = int(width * 0.105)
        ys = int(height * 0.73)
        base_rx = max(26, int(width * 0.045))
        base_ry = max(60, int(height * 0.125))
        energy = max(0.0, min(1.0, float(energies[0])))
        color = theme["bar_low"]

        for cx, angle in ((xl, -30), (width - xl, 30)):
            if bass_pulse > 0.01:
                scale = 0.35 + 0.75 * bass_pulse
                points = render_resources.ellipse(cx, ys, base_rx * scale, base_ry * scale, angle)
                draw.polygon(points, fill=(*color[:3], int(150 * bass_pulse * opacity)))
            for wave in range(4):
                phase = (t * 1.5 + wave * 0.25) % 1.0
                fade = ((1.0 - phase) ** 1.6) * energy
                if fade > 0.02:
                    rx = int(phase * base_rx * 2.35)
                    ry = int(phase * base_ry * 2.35)
                    points = render_resources.ellipse(cx, ys, rx, ry, angle)
                    draw.line(
                        points,
                        fill=(*color[:3], int(210 * fade * opacity)),
                        width=2 if phase < 0.55 else 1,
                        joint="curve",
                    )

    def _vocal_waveform_points(self, t, samples, sample_rate, width, height):
        x_left, x_right = int(width * 0.22), int(width * 0.78)
        y_center = int(height * 0.79)
        max_height = height * 0.075
        count = max(2, x_right - x_left + 1)
        amplitudes = np.zeros(count, dtype=np.float32)
        if samples is not None and len(samples):
            positions = (t + np.linspace(-0.02, 0.02, count)) * sample_rate
            window_start = max(0, int(math.floor(positions[0])))
            window_end = min(len(samples), int(math.ceil(positions[-1])) + 1)
            if window_end > window_start:
                amplitudes = np.interp(
                    positions, np.arange(window_start, window_end),
                    samples[window_start:window_end], left=0.0, right=0.0,
                )
        xs = np.linspace(x_left, x_right, count)
        points = list(zip(xs.tolist(), (y_center - amplitudes * max_height).tolist()))
        return points

    def _render_audio_visuals(self, draw, t, width, height, theme, energies, samples,
                              sample_rate, resources, bass_pulse):
        if resources.last_visual_time is not None and t <= resources.last_visual_time:
            resources.trails.clear()
            resources.last_trail_time = -math.inf
        resources.last_visual_time = t
        while resources.trails and t - resources.trails[0][0] >= 2.0:
            resources.trails.popleft()

        color = theme["lyrics_highlight"][:3]
        for old_time, old_energy, old_bass, old_points in resources.trails:
            opacity = 0.18 * (1.0 - (t - old_time) / 2.0) ** 2
            self._render_speakers(draw, old_time, width, height, theme, old_energy, resources,
                                  old_bass, opacity)
            draw.line(old_points, fill=(*color, int(180 * opacity)), width=1)

        self._render_speakers(draw, t, width, height, theme, energies, resources, bass_pulse)
        points = self._vocal_waveform_points(t, samples, sample_rate, width, height)
        draw.line(points, fill=(*color, 40), width=5)
        draw.line(points, fill=(*color, 230), width=2)
        # Keep geometry snapshots at 8 Hz, rather than retaining full image frames.
        if t - resources.last_trail_time >= 0.125 - 1e-9:
            resources.trails.append((t, energies, bass_pulse, points))
            resources.last_trail_time = t

    def _bass_pulses(self, mono, sample_rate, fps, total_frames):
        bass_db = self._band_levels(mono, sample_rate, fps, total_frames, np.array([30.0, 180.0]))[:, 0]
        amplitude = np.power(10.0, bass_db / 20.0)
        level = np.clip(amplitude / max(1e-4, float(np.percentile(amplitude, 95))), 0.0, 1.0)
        onset = np.maximum(0.0, np.diff(level, prepend=0.0))
        threshold = max(0.025, float(np.percentile(onset, 90)))
        pulses = np.zeros(total_frames, dtype=np.float32)
        decay = math.exp(-1.0 / (fps * 0.24))
        last_beat = -fps
        pulse = 0.0
        for index in range(total_frames):
            pulse *= decay
            if (onset[index] >= threshold and level[index] > 0.12
                    and index - last_beat >= max(1, round(fps * 0.18))):
                pulse = max(pulse, float(level[index]))
                last_beat = index
            pulses[index] = pulse
        return pulses

    def _get_rolling_lambda(self, lines: list[dict], t: float) -> float:
        """Compute smooth continuous virtual line index lambda(t).
        Remains stationary at integer i while line i is sung; smoothly eases to i+1 during handover."""
        n = len(lines)
        if n == 0:
            return 0.0
        if n == 1 or t < lines[0]["start"]:
            return 0.0
        if t >= lines[-1]["end"]:
            return float(n - 1)

        for i in range(n - 1):
            cur_line = lines[i]
            next_line = lines[i + 1]
            t_end = cur_line["end"]
            t_next_start = next_line["start"]

            gap = t_next_start - t_end
            if gap > 0.6:
                t0 = t_end + 0.05
                t1 = min(t_next_start - 0.05, t0 + 0.45)
            else:
                t1 = max(t_end, t_next_start)
                t0 = max(cur_line["start"] + 0.10, t1 - 0.45)

            if t < t0:
                return float(i)
            elif t0 <= t < t1:
                u = (t - t0) / max(0.01, (t1 - t0))
                return float(i) + (u * u * (3.0 - 2.0 * u))
            # If t >= t1, proceed to next line i+1 in loop

        return float(n - 1)

    def _draw_rolling_line(self, draw, text, d, width, height, theme, font_main,
                           render_resources, highlight=False):
        y_focus = int(height * 0.38)
        line_spacing = int(font_main.size * 1.42)
        yk = y_focus + int(d * line_spacing)
        if yk < int(height * 0.10) or yk > int(height * 0.58):
            return None

        dist = abs(d)
        scale = 0.65 + 0.35 * max(0.0, 1.0 - (dist ** 1.3))
        scaled_size = max(16, int(font_main.size * scale))
        cur_font = render_resources.font(scaled_size, bold=(dist < 0.45))

        if d <= 0:
            alpha = int(255 * max(0.0, 1.0 - (dist * 0.65)))
        else:
            alpha = int(255 * max(0.20, 1.0 - (dist * 0.50)))

        bbox = cur_font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w > width - 140:
            cur_font = render_resources.font(max(14, int(scaled_size * (width - 160) / max(1, text_w))), bold=False)
            bbox = cur_font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

        x_start = max(20, (width - text_w) // 2)
        y_start = yk - text_h // 2

        col = (*theme["lyrics_highlight" if highlight else "lyrics_base"][:3], alpha)
        draw.text((x_start, y_start), text, fill=col, font=cur_font)

        return x_start, y_start, text_w, cur_font

    def _render_rolling_lyrics(
        self,
        frame_img: Image.Image,
        draw: ImageDraw.ImageDraw,
        t: float,
        timeline: dict,
        width: int,
        height: int,
        theme: dict,
        font_main: ImageFont.FreeTypeFont,
        font_sub: ImageFont.FreeTypeFont,
        render_resources: _RenderResources,
    ) -> None:
        """Render rolling lyrics: stays steady during line singing; smoothly scrolls upward during transitions."""
        lines = timeline.get("lines", [])
        if not lines:
            draw.text(
                (width // 2, int(height * 0.38)),
                "( Instrumental )",
                fill=(theme["lyrics_base"][0], theme["lyrics_base"][1], theme["lyrics_base"][2], 120),
                font=font_main,
                anchor="mm",
            )
            return

        lambda_line = self._get_rolling_lambda(lines, t)

        min_k = max(0, int(lambda_line) - 2)
        max_k = min(len(lines), int(lambda_line) + 3)

        for k in range(min_k, max_k):
            k_line = lines[k]
            text = k_line["text"]
            d = k - lambda_line

            layout = self._draw_rolling_line(draw, text, d, width, height, theme, font_main, render_resources)
            if layout is None:
                continue
            x_start, y_start, text_w, cur_font = layout

            # Karaoke sweep: active while line is being sung or during handover before next line starts
            should_sweep = (k_line["start"] <= t <= k_line["end"]) or (
                k == int(lambda_line) and t >= k_line["start"] and (k + 1 >= len(lines) or t < lines[k + 1]["start"])
            )
            if should_sweep:
                self._draw_karaoke_sweep(
                    frame_img, draw, k_line, x_start, y_start, text_w, cur_font, theme, t, width, height
                )

    def _get_lyric_break(self, timeline: dict, t: float) -> tuple[str, int | None] | None:
        """Use aligned lyric times to distinguish instrumental breaks from short breaths."""
        previous_end = None
        next_start = None
        for line in timeline["lines"]:
            if line["start"] <= t < line["end"]:
                return None
            if line["end"] <= t:
                previous_end = max(previous_end or 0.0, line["end"])
            elif line["start"] > t:
                next_start = line["start"] if next_start is None else min(next_start, line["start"])

        label = None
        for section in timeline["sections"]:
            if section["start"] <= t < section["end"]:
                name = section["name"].strip().lower()
                if name.startswith("intro"):
                    label = "intro"
                elif name.startswith("interlude"):
                    label = "interlude"
                elif name.startswith(("instrumental", "solo", "break", "outro")):
                    label = "instrumental"
                break

        if label is None:
            if previous_end is None and next_start is not None:
                label = "intro"
            elif next_start is None:
                label = "instrumental"
            elif previous_end is not None and next_start - previous_end >= 4.0:
                label = "interlude"
            else:
                return None

        countdown = None
        if next_start is not None and 0.0 < next_start - t <= 3.0:
            countdown = math.ceil(next_start - t)
        return label, countdown

    def _render_lyric_break(self, draw, lyric_break, t, timeline, width, height, theme,
                           font_main, render_resources):
        label, countdown = lyric_break
        next_line = next((line for line in timeline["lines"] if line["start"] > t), None)
        shift = 0.0
        rise = 0.0
        if next_line is not None:
            remaining = next_line["start"] - t
            progress = max(0.0, min(1.0, 1.0 - remaining / 0.45))
            shift = progress * progress * (3.0 - 2.0 * progress)
            if countdown is not None:
                # The section tag only leaves the center once the countdown starts.
                approach = max(0.0, min(1.0, (3.0 - remaining) / 0.45))
                rise = approach * approach * (3.0 - 2.0 * approach)

        self._draw_rolling_line(draw, f"({label})", -rise - shift, width, height,
                                theme, font_main, render_resources)
        if countdown is not None:
            self._draw_rolling_line(draw, str(countdown), -shift, width, height,
                                    theme, font_main, render_resources, highlight=True)
            if next_line is not None:
                self._draw_rolling_line(draw, next_line["text"], 1.0 - shift, width, height,
                                        theme, font_main, render_resources)

    def _render_progress_frame(self, draw, t, duration, width, height, theme):
        inset = max(8, int(min(width, height) * 0.018))
        left, top, right, bottom = inset, inset, width - inset, height - inset
        path = [(width / 2, top), (left, top), (left, bottom),
                (right, bottom), (right, top), (width / 2, top)]
        draw.line(path, fill=(*theme["pill_border"][:3], 25), width=1)
        perimeter = 2 * (right - left + bottom - top)
        remaining = perimeter * max(0.0, min(1.0, t / max(0.001, duration)))
        completed = [path[0]]
        for start, end in zip(path, path[1:]):
            length = abs(end[0] - start[0]) + abs(end[1] - start[1])
            if remaining <= 0:
                break
            fraction = min(1.0, remaining / length)
            completed.append((start[0] + (end[0] - start[0]) * fraction,
                              start[1] + (end[1] - start[1]) * fraction))
            remaining -= length
        if len(completed) > 1:
            color = theme["progress_fill"][:3]
            draw.line(completed, fill=(*color, 40), width=7, joint="curve")
            draw.line(completed, fill=(*color, 230), width=3, joint="curve")

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
        speaker_energies: tuple[float, float] = (0.0, 0.0),
        vocal_samples: np.ndarray | None = None,
        vocal_sample_rate: float = 1.0,
        render_resources: _RenderResources | None = None,
        bass_pulse: float = 0.0,
        background_carousel: _BackgroundCarousel | None = None,
    ) -> Image.Image:
        """Render a single high-quality video frame with Pillow."""
        if render_resources is None:
            render_resources = _RenderResources(width, height, theme)
        frame_img = (
            background_carousel.frame(t)
            if background_carousel is not None
            else render_resources.background.copy()
        )
        draw = ImageDraw.Draw(frame_img, "RGBA")

        # 2. Top HUD Bar
        hud_y = int(height * 0.04)
        active_sec = self._get_active_section(timeline["sections"], t)
        active_chord = self._get_active_chord(timeline["chords"], t)

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

        info_str = f"Key: {timeline['key']}  ·  {timeline['meter']}  ·  {timeline['bpm']} BPM"
        draw.text((width // 2, hud_y + sec_h // 2), info_str, fill=theme["info_text"], font=font_hud, anchor="mm")

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

        lyric_break = self._get_lyric_break(timeline, t)
        if lyric_break is not None:
            self._render_lyric_break(draw, lyric_break, t, timeline, width, height, theme, font_main, render_resources)

        # 3. Rolling Mode vs Classic Visualizers
        if mode == "Rolling Mode":
            self._render_audio_visuals(draw, t, width, height, theme, speaker_energies,
                                       vocal_samples, vocal_sample_rate, render_resources, bass_pulse)
            if lyric_break is None:
                self._render_rolling_lyrics(frame_img, draw, t, timeline, width, height, theme, font_main, font_sub, render_resources)
            self._render_progress_frame(draw, t, timeline["duration"], width, height, theme)
            return frame_img

        # Classic Karaoke Display
        lines = timeline["lines"]
        active_idx = self._get_active_line_index(lines, t)
        lyric_center_y = int(height * 0.38)
        line_spacing = int(font_main.size * 1.35)

        if lyric_break is None and 0 <= active_idx < len(lines):
            active_line = lines[active_idx]

            if active_idx > 0:
                prev_text = lines[active_idx - 1]["text"]
                p_font = font_sub
                p_bbox = p_font.getbbox(prev_text)
                p_w = p_bbox[2] - p_bbox[0]
                if p_w > width - 80:
                    p_scale = max(14, int(font_sub.size * (width - 100) / max(1, p_w)))
                    p_font = render_resources.font(p_scale, bold=False)
                draw.text(
                    (width // 2, lyric_center_y - line_spacing),
                    prev_text,
                    fill=(theme["lyrics_base"][0], theme["lyrics_base"][1], theme["lyrics_base"][2], 75),
                    font=p_font,
                    anchor="mm",
                )

            if active_idx + 1 < len(lines):
                next_text = lines[active_idx + 1]["text"]
                n_font = font_sub
                n_bbox = n_font.getbbox(next_text)
                n_w = n_bbox[2] - n_bbox[0]
                if n_w > width - 80:
                    n_scale = max(14, int(font_sub.size * (width - 100) / max(1, n_w)))
                    n_font = render_resources.font(n_scale, bold=False)
                draw.text(
                    (width // 2, lyric_center_y + line_spacing),
                    next_text,
                    fill=(theme["lyrics_base"][0], theme["lyrics_base"][1], theme["lyrics_base"][2], 115),
                    font=n_font,
                    anchor="mm",
                )

            line_text = active_line["text"]
            cur_font = font_main
            bbox = cur_font.getbbox(line_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            if text_w > width - 100:
                scaled_size = max(18, int(font_main.size * (width - 120) / max(1, text_w)))
                cur_font = render_resources.font(scaled_size, bold=True)
                bbox = cur_font.getbbox(line_text)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

            x_start = max(20, (width - text_w) // 2)
            y_start = lyric_center_y - text_h // 2
            draw.text((x_start, y_start), line_text, fill=theme["lyrics_base"], font=cur_font)

            self._draw_karaoke_sweep(
                frame_img, draw, active_line, x_start, y_start, text_w, cur_font, theme, t, width, height
            )
        elif lyric_break is None:
            draw.text(
                (width // 2, lyric_center_y),
                "( Instrumental )",
                fill=(theme["lyrics_base"][0], theme["lyrics_base"][1], theme["lyrics_base"][2], 120),
                font=font_main,
                anchor="mm",
            )

        # 4. Audio Reactive Visualizer (Spectrum Bars, Mirrored, Oscilloscope)
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

        self._render_progress_frame(draw, t, timeline["duration"], width, height, theme)
        return frame_img

    def generate_karaoke(
        self,
        abc: str,
        lyrics: str,
        audio: dict | None = None,
        vocals: dict | None = None,
        instrumental: dict | None = None,
        whisper_segments: str = "",
        alignment: str = "auto",
        alignment_device: str = "auto",
        lyrics_offset: float = 0.0,
        resolution: str = "1280x720 (16:9 HD)",
        fps: int = 30,
        theme: str = "Cyberpunk Neon",
        background_folder: str = "",
        background_interval: float = 8.0,
        background_mode: str = "Alphabetical",
        background_transition: str = "Crossfade",
        background_transition_duration: float = 1.0,
        background_transparency: float = 25.0,
        visualizer_mode: str = "Spectrum Bars",
        font_size: int = 38,
        filename_prefix: str = "video/HZ3-Karaoke",
        save_video: bool = True,
        encoder: str = "Auto (NVENC / CPU)",
        max_duration: float = 0.0,
        **kwargs,
    ):
        t_start = time.time()
        width, height = RESOLUTIONS.get(resolution, (1280, 720))
        theme_cfg = THEMES.get(theme, THEMES["Cyberpunk Neon"])

        # 1. Resolve Audio Information (the mix drives the video; the stem, if any, drives alignment)
        has_vocals = isinstance(vocals, dict) and "waveform" in vocals
        has_ins = isinstance(instrumental, dict) and "waveform" in instrumental
        if not (isinstance(audio, dict) and "waveform" in audio):
            if has_ins and has_vocals:
                audio = vocals
            elif has_vocals:
                audio = vocals
            elif has_ins:
                audio = instrumental
        has_audio = isinstance(audio, dict) and "waveform" in audio
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

        vocals_mono = None
        if has_vocals:
            v_tensor = vocals["waveform"]
            if v_tensor.dim() == 3:
                v_tensor = v_tensor[0]
            vocals_mono = v_tensor.mean(dim=0).cpu().numpy()

        align_mono16k = audio_to_mono16k(vocals if has_vocals else audio) if has_audio else None

        # 2. Build Timeline (lyrics alignment + score events)
        timeline = self._parse_timeline(
            abc,
            lyrics,
            audio_duration,
            align_mono16k=align_mono16k,
            align_is_stem=has_vocals,
            whisper_segments=whisper_segments,
            alignment_mode=alignment,
            alignment_device=alignment_device,
            lyrics_offset=lyrics_offset,
        )
        alignment_result = timeline["alignment"]
        logger.info(
            "Karaoke lyrics alignment: source=%s lines=%d %s",
            alignment_result["source"],
            len(alignment_result["lines"]),
            "; ".join(alignment_result["warnings"]) if alignment_result["warnings"] else "",
        )
        total_duration = timeline["duration"]
        if max_duration > 0.0:
            total_duration = min(total_duration, float(max_duration))
        timeline["duration"] = total_duration
        total_frames = max(1, int(total_duration * fps))

        render_resources = _RenderResources(width, height, theme_cfg)
        background_carousel = None
        background_folder = os.path.abspath(os.path.expandvars(os.path.expanduser(str(background_folder or "").strip())))
        if background_folder:
            if os.path.isdir(background_folder):
                try:
                    candidate = _BackgroundCarousel(
                        background_folder,
                        width,
                        height,
                        render_resources.background,
                        background_interval,
                        background_mode,
                        background_transition,
                        background_transition_duration,
                        background_transparency,
                    )
                    if candidate.paths:
                        background_carousel = candidate
                        logger.info("Visualizer backgrounds: %d images from %s", len(candidate.paths), background_folder)
                    else:
                        logger.warning("No supported images found in visualizer background folder '%s'. Using theme background.", background_folder)
                except Exception as exc:
                    logger.warning("Could not read visualizer background folder '%s': %s. Using theme background.", background_folder, exc)
            else:
                logger.warning("Visualizer background folder does not exist: '%s'. Using theme background.", background_folder)

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

        # Video stream: Auto/GPU uses h264_nvenc for ~3x faster encoding with negligible VRAM (~50MB)
        if "nvenc" in encoder.lower():
            candidate_codecs = ("h264_nvenc",)
        elif "libx264" in encoder.lower():
            candidate_codecs = ("libx264", "mpeg4")
        else:  # "Auto (NVENC / CPU)"
            candidate_codecs = ("h264_nvenc", "libx264", "mpeg4")

        v_stream = None
        chosen_codec = None
        for codec_name in candidate_codecs:
            try:
                v_stream = container.add_stream(codec_name, rate=fps)
                v_stream.width = width
                v_stream.height = height
                v_stream.pix_fmt = "yuv420p"
                if codec_name == "h264_nvenc":
                    v_stream.options = {"preset": "fast", "cq": "22"}
                elif codec_name == "libx264":
                    v_stream.options = {"preset": "fast", "crf": "21"}
                chosen_codec = codec_name
                break
            except Exception as e:
                logger.warning(f"Could not initialize video codec '{codec_name}': {e}")
                continue

        if v_stream is None:
            raise RuntimeError(f"Failed to initialize video encoder with any candidate from {candidate_codecs}.")

        # Audio stream
        a_stream = None
        if has_audio:
            try:
                a_stream = container.add_stream("aac", rate=audio_sr)
                a_stream.layout = "mono" if w_2d.shape[0] == 1 else "stereo"
            except Exception as e:
                logger.warning(f"Could not initialize AAC audio stream: {e}")

        # 4. Render & Encode Loop
        band_db_all = None
        db_floor, db_span = -50.0, 48.0
        if visualizer_mode != "Rolling Mode" and has_audio and waveform_mono is not None:
            # Band levels for every frame at once; the display range adapts to the track's
            # own loudness so a hot master does not pin every bar to the ceiling.
            band_db_all = self._band_levels(waveform_mono, audio_sr, fps, total_frames, freq_bands)
            db_floor = float(np.percentile(band_db_all, 12.0))
            db_span = max(18.0, float(np.percentile(band_db_all, 99.5)) - db_floor)

        speaker_rms_all = None
        bass_pulses = np.zeros(total_frames, dtype=np.float32)
        smooth_energy = 0.0
        vocal_samples = None
        vocal_sample_rate = 1.0
        if visualizer_mode == "Rolling Mode":
            if waveform_mono is not None:
                bass_pulses = self._bass_pulses(waveform_mono, audio_sr, fps, total_frames)
                rms_raw = self._frame_rms(waveform_mono, audio_sr, fps, total_frames)
                p95 = float(np.percentile(rms_raw, 95)) if len(rms_raw) else 1.0
                speaker_rms_all = np.clip(rms_raw / max(1e-4, p95), 0.0, 1.0)
            if vocals_mono is not None and len(vocals_mono):
                vocal_sample_rate = int(vocals.get("sample_rate", 44100))
                vocal_peak = max(1e-4, float(np.max(np.abs(vocals_mono))))
                vocal_samples = vocals_mono / vocal_peak

        render_seconds = 0.0
        encode_seconds = 0.0
        for frame_idx in range(total_frames):
            cur_time = float(frame_idx) / float(fps)

            if visualizer_mode != "Rolling Mode":
                # Compute Spectrum Bars
                if band_db_all is not None:
                    raw_bars = np.clip((band_db_all[frame_idx] - db_floor) / db_span, 0.0, 1.0).astype(np.float32)
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

            speaker_energies = (0.0, 0.0)
            if visualizer_mode == "Rolling Mode" and speaker_rms_all is not None:
                smooth_energy = smooth_energy * 0.55 + float(speaker_rms_all[frame_idx]) * 0.45
                speaker_energies = (smooth_energy, smooth_energy)

            render_start = time.perf_counter()
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
                speaker_energies=speaker_energies,
                bass_pulse=float(bass_pulses[frame_idx]),
                vocal_samples=vocal_samples,
                vocal_sample_rate=vocal_sample_rate,
                render_resources=render_resources,
                background_carousel=background_carousel,
            )

            render_seconds += time.perf_counter() - render_start
            encode_start = time.perf_counter()
            v_frame = av.VideoFrame.from_image(frame_img)
            for p in v_stream.encode(v_frame):
                container.mux(p)

            encode_seconds += time.perf_counter() - encode_start

        encode_start = time.perf_counter()
        for p in v_stream.encode():
            container.mux(p)
        encode_seconds += time.perf_counter() - encode_start

        # 5. Mux Audio Stream
        audio_mux_start = time.perf_counter()
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
        audio_mux_seconds = time.perf_counter() - audio_mux_start

        # 6. Format Outputs for ComfyUI
        elapsed = time.time() - t_start
        fps_rendered = float(total_frames) / max(0.01, elapsed)

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
            speaker_energies=(0.0, 0.0),
            vocal_samples=vocal_samples,
            vocal_sample_rate=vocal_sample_rate,
            render_resources=render_resources,
            background_carousel=background_carousel,
        )
        images_tensor = torch.from_numpy(np.array(first_frame, dtype=np.float32) / 255.0).unsqueeze(0)

        if not has_audio:
            silent_wav = torch.zeros((1, 2, int(audio_sr * min(total_duration, 1.0))))
            audio_out = {"waveform": silent_wav, "sample_rate": audio_sr}
        else:
            audio_out = audio

        timed_lines = alignment_result["lines"]
        timed_json = json.dumps(timed_lines, ensure_ascii=False, indent=2)
        lrc_text = alignment_result["lrc"]
        sung_lines = sum(1 for l in timed_lines if not l.get("beyond_audio"))
        alignment_label = {
            "forced": f"forced alignment on {'vocal stem' if has_vocals else 'mix'} ({alignment_result['stats'].get('device', '?')})",
            "whisper": "Whisper segments",
            "score": "nominal ABC timing",
            "none": "none",
        }.get(alignment_result["source"], alignment_result["source"])
        render_summary = (
            f"Rendered Karaoke & Visualizer · {total_frames} frames ({total_duration:.1f}s) @ {fps}fps\n"
            f"Encoder: {chosen_codec} · Resolution: {width}x{height} · Speed: {fps_rendered:.1f} fps ({elapsed:.2f}s total)\n"
            f"Timing: drawing {render_seconds:.2f}s · video encoding {encode_seconds:.2f}s · audio/finalize {audio_mux_seconds:.2f}s\n"
            f"Lyrics: {sung_lines}/{len(timed_lines)} lines timed via {alignment_label}\n"
            f"Saved MP4: {video_full_path}"
        )
        report = render_summary + "\n\n" + alignment_result["report"]

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
            "result": (images_tensor, audio_out, str(video_full_path), timed_json, lrc_text, report),
        }


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_KaraokeVisualizer": HZ3_YuE2_KaraokeVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_KaraokeVisualizer": "HZ3 YuE2 · Karaoke & Audio Visualizer",
}
