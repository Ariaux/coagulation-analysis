import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import cv2

import analysis_service
import grid_detector
from analysis_service import (
    DEEP_RED_RGB,
    LIGHT_RED_RGB,
    NO_CLOT_BLUE_RGB,
    AnalysisSettings,
    analyze_image,
    heatmap_color_rgb,
    inset_bbox,
)
from tests.test_grid_detector import make_fixture


class SingleImageServiceTests(unittest.TestCase):
    def test_analyze_image_publishes_inset_cells_and_metadata(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            self.assertTrue(cv2.imwrite(str(source), image))

            result = analyze_image(
                source,
                AnalysisSettings(5.0, 60.0, root / "results"),
            )

            self.assertEqual(9, len(result["cells"]))
            self.assertTrue(Path(result["zip_path"]).is_file())
            for cell in result["cells"]:
                detected_x1, detected_y1, detected_x2, detected_y2 = cell[
                    "detected_bbox"
                ]
                final_x1, final_y1, final_x2, final_y2 = cell["final_bbox"]
                self.assertGreater(final_x1, detected_x1)
                self.assertGreater(final_y1, detected_y1)
                self.assertLess(final_x2, detected_x2)
                self.assertLess(final_y2, detected_y2)

            with Path(result["json_path"]).open(encoding="utf-8") as results_file:
                payload = json.load(results_file)
            self.assertEqual(
                "ImageJ-equivalent inverted 8-bit grayscale mean",
                payload["measurement_method"],
            )
            self.assertEqual(5.0, payload["settings"]["inset_percent"])
            self.assertEqual(60.0, payload["settings"]["no_clot_threshold"])
            self.assertEqual("publication-blue-red-v1", payload["palette_version"])
            self.assertEqual(
                "publication-blue-red-v1",
                payload["settings"]["palette_version"],
            )

            with Path(result["csv_path"]).open(
                newline="", encoding="utf-8"
            ) as results_file:
                rows = list(csv.DictReader(results_file))
            self.assertEqual(9, len(rows))
            self.assertIn("final_x1", rows[0])
            self.assertEqual("5.0", rows[0]["inset_percent"])

    def test_backup_cleanup_failure_restores_the_prior_complete_result(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            results_root = root / "results"
            self.assertTrue(cv2.imwrite(str(source), image))
            prior = analyze_image(
                source,
                AnalysisSettings(5.0, 60.0, results_root),
            )
            prior_crop = cv2.imread(prior["crop_paths"][0])
            self.assertIsNotNone(prior_crop)

            real_remove_directory = analysis_service._remove_known_directory

            def fail_previous_directory_cleanup(path):
                if ".previous-" in Path(path).name:
                    raise OSError("simulated backup cleanup failure")
                return real_remove_directory(path)

            with mock.patch.object(
                analysis_service,
                "_remove_known_directory",
                side_effect=fail_previous_directory_cleanup,
            ):
                with self.assertRaisesRegex(
                    OSError, "simulated backup cleanup failure"
                ):
                    analyze_image(
                        source,
                        AnalysisSettings(10.0, 60.0, results_root),
                    )

            with Path(prior["json_path"]).open(encoding="utf-8") as results_file:
                metadata = json.load(results_file)
            self.assertEqual(5.0, metadata["settings"]["inset_percent"])
            restored_crop = cv2.imread(prior["crop_paths"][0])
            self.assertIsNotNone(restored_crop)
            self.assertEqual(prior_crop.shape, restored_crop.shape)
            with zipfile.ZipFile(prior["zip_path"]) as archive:
                json_name = next(
                    name
                    for name in archive.namelist()
                    if name.endswith("_results.json")
                )
                archived_metadata = json.loads(archive.read(json_name))
            self.assertEqual(
                5.0,
                archived_metadata["settings"]["inset_percent"],
            )
            self.assertFalse(
                any(
                    ".previous-" in path.name or "_staging_" in path.name
                    for path in results_root.iterdir()
                )
            )


class AnalysisSettingsTests(unittest.TestCase):
    def test_defaults(self):
        settings = AnalysisSettings()

        self.assertEqual(5.0, settings.inset_percent)
        self.assertEqual(60.0, settings.no_clot_threshold)

    def test_validate_rejects_out_of_range_inset_percent(self):
        for inset_percent in (-0.1, 15.1):
            with self.subTest(inset_percent=inset_percent):
                with self.assertRaises(ValueError):
                    AnalysisSettings(inset_percent=inset_percent).validate()

    def test_validate_rejects_out_of_range_no_clot_threshold(self):
        for no_clot_threshold in (-0.1, 255.1):
            with self.subTest(no_clot_threshold=no_clot_threshold):
                with self.assertRaises(ValueError):
                    AnalysisSettings(no_clot_threshold=no_clot_threshold).validate()

    def test_inset_bbox_returns_inner_bbox_and_rejects_too_small_crops(self):
        self.assertEqual(
            (15, 25, 195, 115),
            inset_bbox((10, 20, 200, 120), inset_percent=5.0),
        )

        with self.assertRaisesRegex(
            grid_detector.DetectionError, "too small after the inner inset"
        ):
            inset_bbox((10, 10, 48, 48), 15.0, minimum_side=32)


class PublicationPaletteTests(unittest.TestCase):
    def test_values_at_or_below_threshold_are_exact_no_clot_blue(self):
        for value in (0.0, 59.9, 60.0):
            with self.subTest(value=value):
                self.assertEqual(NO_CLOT_BLUE_RGB, heatmap_color_rgb(value, 60.0))

    def test_value_above_threshold_uses_red_color_family(self):
        red, green, blue = heatmap_color_rgb(60.1, 60.0)

        self.assertNotEqual(NO_CLOT_BLUE_RGB, (red, green, blue))
        self.assertGreaterEqual(red, green)
        self.assertGreaterEqual(red, blue)

    def test_above_threshold_gradient_has_stable_endpoints(self):
        near_threshold = heatmap_color_rgb(60.000001, 60.0)

        for actual, expected in zip(near_threshold, LIGHT_RED_RGB):
            self.assertLessEqual(abs(actual - expected), 1)
        self.assertEqual(DEEP_RED_RGB, heatmap_color_rgb(255, 60.0))
        self.assertEqual(heatmap_color_rgb(100, 60.0), heatmap_color_rgb(100, 60.0))
