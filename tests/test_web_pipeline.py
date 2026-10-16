import inspect
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import cv2

import app
from tests.test_grid_detector import make_fixture, make_incomplete_fixture


class WebPipelineTests(unittest.TestCase):
    def assert_fixed_pipeline_signature(self):
        self.assertEqual(
            list(inspect.signature(app.process_pipeline).parameters),
            ["image", "filename"],
        )

    def test_process_pipeline_builds_complete_fixed_grid_bundle(self):
        self.assert_fixed_pipeline_signature()
        fixture_bgr, _ = make_fixture(filled=(1, 5, 9))
        fixture_bgr[0, 0] = (10, 20, 30)
        fixture_rgb = cv2.cvtColor(fixture_bgr, cv2.COLOR_BGR2RGB)

        overlay, heatmap, table, bundle_path, message = app.process_pipeline(
            fixture_rgb, "fixture.png"
        )

        self.assertIsNotNone(overlay)
        self.assertIsNotNone(heatmap)
        self.assertEqual(len(table), 9)
        self.assertEqual(len(table[0]), len(app.TABLE_HEADERS))
        self.assertEqual(len(table[0]), 24)
        self.assertEqual(tuple(overlay[0, 0]), tuple(fixture_rgb[0, 0]))
        self.assertTrue(os.path.isfile(bundle_path))
        self.assertIn("9 cells", message)
        with zipfile.ZipFile(bundle_path) as archive:
            names = set(archive.namelist())
            expected = {
                *(f"cell_{idx:02d}.png" for idx in range(1, 10)),
                "grid_overlay.png",
                "heatmap.png",
                "results.csv",
                "results.json",
            }
            self.assertEqual(names, expected)
            payload = json.loads(archive.read("results.json"))
            self.assertEqual(payload["grid"], "3x3")
            self.assertEqual(len(payload["cells"]), 9)
        self.assertEqual(
            os.listdir(os.path.dirname(bundle_path)),
            ["coagulation_results.zip"],
        )
        shutil.rmtree(os.path.dirname(bundle_path))

    def test_missing_upload_returns_no_partial_results(self):
        self.assert_fixed_pipeline_signature()
        result = app.process_pipeline(None, "upload.png")

        self.assertEqual(result[:4], (None, None, [], None))
        self.assertIn("upload", result[4].lower())

    def test_incomplete_fixture_returns_no_partial_results(self):
        self.assert_fixed_pipeline_signature()
        fixture_rgb = cv2.cvtColor(
            make_incomplete_fixture(), cv2.COLOR_BGR2RGB
        )

        result = app.process_pipeline(fixture_rgb, "incomplete.png")

        self.assertEqual(result[:4], (None, None, [], None))
        self.assertIn("incomplete", result[4].lower())

    def test_undersized_image_returns_no_partial_results(self):
        fixture_bgr, _ = make_fixture(size=500)
        fixture_rgb = cv2.cvtColor(fixture_bgr, cv2.COLOR_BGR2RGB)

        result = app.process_pipeline(fixture_rgb, "small.png")

        self.assertEqual(result[:4], (None, None, [], None))
        self.assertIn("at least 600x600", result[4].lower())

    def test_bundle_paths_are_isolated_between_requests(self):
        fixture_bgr, _ = make_fixture()
        fixture_rgb = cv2.cvtColor(fixture_bgr, cv2.COLOR_BGR2RGB)

        first = app.process_pipeline(fixture_rgb, "fixture.png")[3]
        second = app.process_pipeline(fixture_rgb, "fixture.png")[3]

        try:
            self.assertNotEqual(first, second)
        finally:
            shutil.rmtree(os.path.dirname(first))
            shutil.rmtree(os.path.dirname(second))

    def test_failed_bundle_creation_removes_request_directory(self):
        fixture_bgr, _ = make_fixture()
        fixture_rgb = cv2.cvtColor(fixture_bgr, cv2.COLOR_BGR2RGB)

        with tempfile.TemporaryDirectory() as parent:
            real_mkdtemp = tempfile.mkdtemp

            def request_directory(prefix):
                return real_mkdtemp(prefix=prefix, dir=parent)

            with patch.object(app.tempfile, "mkdtemp", side_effect=request_directory):
                with patch.object(app, "_write_png", side_effect=OSError("disk full")):
                    result = app.process_pipeline(fixture_rgb, "fixture.png")

            self.assertEqual(result[:4], (None, None, [], None))
            self.assertEqual(os.listdir(parent), [])

    def test_module_exposes_gradio_demo(self):
        self.assertTrue(hasattr(app, "demo"))


if __name__ == "__main__":
    unittest.main()
