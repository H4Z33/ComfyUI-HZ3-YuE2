"""Focused tests for image sequencing used by the karaoke visualizer."""

import os
import tempfile
import unittest

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
        }
        options.update(overrides)
        return _BackgroundCarousel(**options)

    def pixel(self, image):
        return image.getpixel((0, 0))

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
            "background_mode",
            "background_transition",
            "background_transition_duration",
            "background_transparency",
        }.issubset(inputs))
        self.assertEqual(inputs["background_transition"][0], BACKGROUND_TRANSITIONS)
        self.assertNotIn("render_image_batch", inputs)


if __name__ == "__main__":
    unittest.main()
