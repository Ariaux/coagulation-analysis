import csv
import json
import os
import tempfile
import unittest
from unittest import mock

import cv2

import app_standalone
from app_standalone import load_image, process_image
from grid_detector import DetectionError
from tests.test_grid_detector import make_fixture


class StandalonePipelineTests(unittest.TestCase):
    def test_process_image_crops_detected_inner_squares_and_saves_outputs(self):
        image, _ = make_fixture(filled=(1, 5, 9))

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "fixture.png")
            self.assertTrue(cv2.imwrite(image_path, image))

            outputs = process_image(
                image_path,
                show_windows=False,
                open_folder=False,
            )

            self.assertEqual(9, len(outputs["cells"]))
            crop_widths = []
            crop_heights = []
            for idx in range(1, 10):
                crop_path = os.path.join(
                    outputs["output_dir"], f"cell_{idx:02d}.png"
                )
                self.assertTrue(os.path.exists(crop_path))
                crop = cv2.imread(crop_path)
                self.assertIsNotNone(crop)
                height, width = crop.shape[:2]
                crop_widths.append(width)
                crop_heights.append(height)
                self.assertLessEqual(abs(width - height), 4)

            full_grid_cell_side = image.shape[0] / 3
            self.assertLess(max(crop_widths), full_grid_cell_side * 0.75)
            self.assertLess(max(crop_heights), full_grid_cell_side * 0.75)

            for output_key in (
                "overlay_path",
                "heatmap_path",
                "csv_path",
                "json_path",
            ):
                self.assertTrue(os.path.exists(outputs[output_key]))

            with open(outputs["json_path"], encoding="utf-8") as results_file:
                payload = json.load(results_file)
            self.assertEqual("fixture.png", payload["image"])
            self.assertEqual("3x3", payload["grid"])
            self.assertEqual(9, len(payload["cells"]))
            self.assertIn("crop_quad", payload["cells"][0])
            self.assertEqual(4, len(payload["cells"][0]["crop_quad"]))

            with open(
                outputs["csv_path"], newline="", encoding="utf-8"
            ) as results_file:
                header = next(csv.reader(results_file))
            self.assertEqual(
                [
                    "cell",
                    "row",
                    "col",
                    "mean",
                    "median",
                    "std",
                    "min",
                    "max",
                    "int_den",
                    "area_px",
                    "confidence",
                    "recovered",
                ],
                header,
            )

    def test_pipeline_rejects_images_with_either_dimension_below_600(self):
        images = [
            make_fixture(size=size)[0]
            for size in (180, 300, 599)
        ]
        images.append(
            cv2.resize(make_fixture(size=800)[0], (800, 599))
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for idx, image in enumerate(images):
                with self.subTest(shape=image.shape):
                    image_path = os.path.join(temp_dir, f"small_{idx}.png")
                    self.assertTrue(cv2.imwrite(image_path, image))

                    with self.assertRaises(DetectionError) as raised:
                        process_image(
                            image_path,
                            show_windows=False,
                            open_folder=False,
                        )

                    self.assertIn("600", str(raised.exception))

    def test_pipeline_accepts_600_fixture_without_frame_bias_or_overcrop(self):
        image, expected = make_fixture(size=600, brightness=210)
        expected_width = expected[0][2] - expected[0][0]
        expected_height = expected[0][3] - expected[0][1]
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "boundary.png")
            self.assertTrue(cv2.imwrite(image_path, image))

            outputs = process_image(
                image_path,
                show_windows=False,
                open_folder=False,
            )

            self.assertEqual(9, len(outputs["cells"]))
            for cell in outputs["cells"]:
                self.assertLessEqual(abs(cell["mean"] - 45.0), 3.0)
                crop = load_image(
                    os.path.join(
                        outputs["output_dir"],
                        f"cell_{cell['idx']:02d}.png",
                    )
                )
                height, width = crop.shape[:2]
                self.assertGreaterEqual(width / expected_width, 0.75)
                self.assertGreaterEqual(height / expected_height, 0.75)
                self.assertLessEqual(width / expected_width, 1.10)
                self.assertLessEqual(height / expected_height, 1.10)

    def test_same_stem_different_extensions_keep_distinct_outputs(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = os.path.join(temp_dir, "run.png")
            jpg_path = os.path.join(temp_dir, "run.jpg")
            self.assertTrue(cv2.imwrite(png_path, image))
            self.assertTrue(cv2.imwrite(jpg_path, image))

            png_outputs = process_image(
                png_path,
                show_windows=False,
                open_folder=False,
            )
            jpg_outputs = process_image(
                jpg_path,
                show_windows=False,
                open_folder=False,
            )

            self.assertNotEqual(
                png_outputs["output_dir"],
                jpg_outputs["output_dir"],
            )
            self.assertNotEqual(
                png_outputs["json_path"],
                jpg_outputs["json_path"],
            )
            for outputs in (png_outputs, jpg_outputs):
                self.assertTrue(os.path.exists(outputs["json_path"]))
                self.assertTrue(
                    os.path.exists(
                        os.path.join(outputs["output_dir"], "cell_01.png")
                    )
                )
            with open(png_outputs["json_path"], encoding="utf-8") as result_file:
                self.assertEqual("run.png", json.load(result_file)["image"])
            with open(jpg_outputs["json_path"], encoding="utf-8") as result_file:
                self.assertEqual("run.jpg", json.load(result_file)["image"])

    def test_unicode_paths_use_encoded_image_writes(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            unicode_dir = os.path.join(temp_dir, "中文结果")
            os.makedirs(unicode_dir)
            image_path = os.path.join(unicode_dir, "样本图片.png")
            self.assertTrue(cv2.imwrite(image_path, image))

            with mock.patch.object(
                app_standalone.cv2,
                "imwrite",
                side_effect=AssertionError("cv2.imwrite is not Unicode-safe"),
            ):
                outputs = process_image(
                    image_path,
                    show_windows=False,
                    open_folder=False,
                )

            for idx in range(1, 10):
                crop_path = os.path.join(
                    outputs["output_dir"], f"cell_{idx:02d}.png"
                )
                self.assertIsNotNone(load_image(crop_path))
            self.assertIsNotNone(load_image(outputs["overlay_path"]))
            self.assertIsNotNone(load_image(outputs["heatmap_path"]))
            with open(outputs["json_path"], encoding="utf-8") as result_file:
                self.assertEqual("样本图片.png", json.load(result_file)["image"])
            with open(outputs["csv_path"], encoding="utf-8") as result_file:
                self.assertTrue(result_file.readline().startswith("cell,"))


if __name__ == "__main__":
    unittest.main()
