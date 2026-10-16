import inspect
import json
import os
import unittest
import zipfile

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
        fixture_rgb = cv2.cvtColor(fixture_bgr, cv2.COLOR_BGR2RGB)

        overlay, heatmap, table, bundle_path, message = app.process_pipeline(
            fixture_rgb, "fixture.png"
        )

        self.assertIsNotNone(overlay)
        self.assertIsNotNone(heatmap)
        self.assertEqual(len(table), 9)
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

    def test_module_exposes_gradio_demo(self):
        self.assertTrue(hasattr(app, "demo"))


if __name__ == "__main__":
    unittest.main()
