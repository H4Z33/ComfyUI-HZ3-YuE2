"""Focused tests for image sequencing used by the karaoke visualizer."""

import os
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
from PIL import Image

from karaoke_visualizer import (
    BACKGROUND_TRANSITIONS,
    HZ3_YuE2_KaraokeVisualizer,
    _BackgroundCarousel,
    _rolling_scope_source,
)


class RollingScopeSourceTests(unittest.TestCase):
    def test_uses_mix_when_no_vocal_stem_is_connected(self):
        mix = object()
        self.assertIs(_rolling_scope_source(None, mix), mix)

    def test_prefers_connected_vocal_stem(self):
        vocals = object()
        self.assertIs(_rolling_scope_source(vocals, object()), vocals)


class RollingLyricsWindowTests(unittest.TestCase):
    def test_rolling_lyrics_show_only_previous_current_and_next(self):
        visualizer = HZ3_YuE2_KaraokeVisualizer()
        lines = [
            {"text": f"line {index}", "start": index * 10.0, "end": index * 10.0 + 5.0}
            for index in range(6)
        ]
        visible_text = []

        def record_line(_draw, text, *_args, **_kwargs):
            visible_text.append(text)
            return None

        with (
            patch.object(visualizer, "_get_rolling_lambda", return_value=2.75),
            patch.object(visualizer, "_draw_rolling_line", side_effect=record_line),
        ):
            visualizer._render_rolling_lyrics(
                frame_img=object(),
                draw=object(),
                t=22.0,
                timeline={"lines": lines},
                width=1280,
                height=720,
                theme={},
                font_main=object(),
                font_sub=object(),
                render_resources=object(),
            )

        self.assertEqual(visible_text, ["line 1", "line 2", "line 3"])


class BackgroundCarouselTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base = Image.new("RGB", (4, 4), (0, 0, 0))
        Image.new("RGB", (4, 4), (255, 0, 0)).save(os.path.join(self.temp_dir.name, "01-red.png"))
        Image.new("RGB", (4, 4), (0, 0, 255)).save(os.path.join(self.temp_dir.name, "02-blue.png"))

    def carousel(self, **overrides):
        options = {
            "folder": self.temp_dir.name,
            "width": 4,
            "height": 4,
            "base": self.base,
            "interval": 5.0,
            "mode": "Alphabetical",
            "transition": "Crossfade",
            "transition_duration": 2.0,
            "transparency": 0.0,
            "motion_speed": 0.25,
        }
        options.update(overrides)
        return _BackgroundCarousel(**options)

    def pixel(self, image):
        return image.getpixel((0, 0))

    @staticmethod
    def write_test_video(path):
        container = av.open(path, mode="w", format="mp4")
        stream = container.add_stream("mpeg4", rate=2)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        colors = ((255, 0, 0), (0, 0, 255)) + ((0, 255, 0),) * 10
        for index, color in enumerate(colors):
            pixels = np.empty((32, 32, 3), dtype=np.uint8)
            pixels[:] = color
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    def test_alphabetical_order_and_interval_rotation(self):
        carousel = self.carousel()
        self.assertEqual([os.path.basename(path) for path in carousel.paths], ["01-red.png", "02-blue.png"])
        self.assertEqual(self.pixel(carousel.frame(0)), (255, 0, 0))
        self.assertEqual(self.pixel(carousel.frame(4.99)), (255, 0, 0))
        self.assertEqual(self.pixel(carousel.frame(5)), (255, 0, 0))
        self.assertEqual(self.pixel(carousel.frame(7)), (0, 0, 255))

    def test_transparency_composites_over_theme_background(self):
        carousel = self.carousel(transparency=50.0, transition="Cut")
        self.assertEqual(self.pixel(carousel.frame(0)), (128, 0, 0))

    def test_still_images_get_slow_zoom_and_pan(self):
        gradient = np.zeros((120, 120, 3), dtype=np.uint8)
        gradient[:, :, 0] = np.arange(120, dtype=np.uint8)[None, :]
        gradient[:, :, 1] = np.arange(120, dtype=np.uint8)[:, None]
        Image.fromarray(gradient).save(os.path.join(self.temp_dir.name, "03-gradient.png"))
        carousel = self.carousel(width=40, height=40, base=Image.new("RGB", (40, 40)), transition="Cut")
        first = carousel.frame(10.0)
        moved = carousel.frame(12.5)
        self.assertNotEqual(first.tobytes(), moved.tobytes())

    def test_motion_speed_zero_freezes_still_backgrounds(self):
        gradient = np.zeros((120, 120, 3), dtype=np.uint8)
        gradient[:, :, 0] = np.arange(120, dtype=np.uint8)[None, :]
        gradient[:, :, 1] = np.arange(120, dtype=np.uint8)[:, None]
        Image.fromarray(gradient).save(os.path.join(self.temp_dir.name, "03-gradient.png"))
        carousel = self.carousel(motion_speed=0.0, transition="Cut")
        first = carousel.frame(10.0)
        later = carousel.frame(12.5)
        self.assertEqual(first.tobytes(), later.tobytes())

    def test_overlapping_images_share_one_continuous_motion_phase(self):
        carousel = self.carousel()
        with patch.object(carousel, "_image_frame", wraps=carousel._image_frame) as render_image:
            carousel.frame(5.5)
        self.assertEqual([call.args[1] for call in render_image.call_args_list], [5.5, 5.5])

    def test_video_backgrounds_play_in_order_with_transparency(self):
        video_path = os.path.join(self.temp_dir.name, "03-colors.mp4")
        self.write_test_video(video_path)
        carousel = self.carousel(transition="Cut", transparency=50.0)
        self.addCleanup(carousel.close)

        first_video_frame = carousel.frame(10.1).getpixel((2, 2))
        second_video_frame = carousel.frame(10.6).getpixel((2, 2))
        late_video_frame = carousel.frame(15.2).getpixel((2, 2))
        self.assertGreater(first_video_frame[0], 100)
        self.assertLess(first_video_frame[2], 40)
        self.assertGreater(second_video_frame[2], 100)
        self.assertLess(second_video_frame[0], 40)
        self.assertGreater(late_video_frame[1], 100)
        self.assertLess(late_video_frame[0], 40)

    def test_video_backgrounds_use_selected_transition(self):
        video_path = os.path.join(self.temp_dir.name, "03-colors.mp4")
        self.write_test_video(video_path)
        carousel = self.carousel(transition="Crossfade", transition_duration=0.4)
        self.addCleanup(carousel.close)

        blended = carousel.frame(10.2).getpixel((2, 2))
        self.assertGreater(blended[0], 90)
        self.assertGreater(blended[2], 90)

    def test_fade_through_black_and_wipe_are_renderable(self):
        for transition in ("Fade Through Black", "Wipe Left"):
            with self.subTest(transition=transition):
                carousel = self.carousel(transition=transition)
                self.assertEqual(carousel.frame(5).size, (4, 4))
                self.assertEqual(carousel.frame(6).size, (4, 4))
                self.assertEqual(carousel.frame(7).size, (4, 4))

    def test_node_exposes_background_controls_and_removes_image_batch_toggle(self):
        inputs = HZ3_YuE2_KaraokeVisualizer.INPUT_TYPES()["optional"]
        self.assertTrue({
            "background_folder",
            "background_interval",
            "background_motion_speed",
            "background_mode",
            "background_transition",
            "background_transition_duration",
            "background_transparency",
        }.issubset(inputs))
        self.assertEqual(inputs["background_transition"][0], BACKGROUND_TRANSITIONS)
        self.assertNotIn("render_image_batch", inputs)


if __name__ == "__main__":
    unittest.main()
