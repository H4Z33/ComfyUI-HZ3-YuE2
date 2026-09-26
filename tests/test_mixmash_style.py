"""MixMash cache-control tests; run with python -m unittest discover -s tests -v."""

import importlib
import math
from pathlib import Path
import sys
import types
import unittest


PACKAGE = "mixmash_test_nodes"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[PACKAGE] = package
mixmash = importlib.import_module(f"{PACKAGE}.mixmash_style")


class MixMashCacheTests(unittest.TestCase):
    def test_force_redo_switch_defaults_off(self):
        node = mixmash.HZ3_YuE2_MixMashStyle
        setting = node.INPUT_TYPES()["required"]["force_redo"]
        self.assertEqual(setting[0], "BOOLEAN")
        self.assertFalse(setting[1]["default"])
        self.assertIs(node.IS_CHANGED(), False)

    def test_force_redo_bypasses_comfy_cache_only_when_enabled(self):
        node = mixmash.HZ3_YuE2_MixMashStyle
        self.assertIs(node.IS_CHANGED(force_redo=False), False)
        self.assertTrue(math.isnan(node.IS_CHANGED(force_redo=True)))


if __name__ == "__main__":
    unittest.main()
