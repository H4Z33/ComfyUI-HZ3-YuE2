"""Run with the ComfyUI Python environment and its root on PYTHONPATH."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

PACKAGE = "studio_history_test_nodes"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[PACKAGE] = package
studio = importlib.import_module(f"{PACKAGE}.studio")


class HistorySnapshotTests(unittest.TestCase):
    def setUp(self):
        output = tempfile.TemporaryDirectory()
        self.addCleanup(output.cleanup)
        patcher = mock.patch.object(studio.folder_paths, "get_output_directory", return_value=output.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def entries(self, name):
        with zipfile.ZipFile(studio._package_path(name)) as package:
            return {entry: json.loads(package.read(entry)) for entry in package.namelist() if entry.startswith("history/")}

    def test_snapshots_are_written_and_carried_over_by_later_saves(self):
        first, second = "a" * 26, "b" * 26
        studio._write_package({"name": "Song", "takes": [], "_snapshots": {first: {"lyrics": "one"}}})
        studio._write_package({"name": "Song", "takes": [], "_snapshots": {second: {"lyrics": "two"}}})
        studio._write_package({"name": "Song", "takes": []})
        self.assertEqual(self.entries("Song"), {f"history/{first}.json": {"lyrics": "one"}, f"history/{second}.json": {"lyrics": "two"}})
        with zipfile.ZipFile(studio._package_path("Song")) as package:
            self.assertNotIn("_snapshots", json.loads(package.read("manifest.json"))["project"])

    def test_snapshot_ids_that_are_not_hashes_are_ignored(self):
        studio._write_package({"name": "Song", "takes": [], "_snapshots": {"../../evil": {"x": 1}, "c" * 26: {"x": 2}}})
        self.assertEqual(list(self.entries("Song")), [f"history/{'c' * 26}.json"])


if __name__ == "__main__":
    unittest.main()
