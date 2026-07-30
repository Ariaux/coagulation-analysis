import csv
import json
import os
import tempfile
import unittest
from unittest import mock

import cv2

import app_standalone
from app_standalone import load_image, process_image
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

    def test_low_resolution_empty_cells_exclude_the_inner_frame(self):
        for size in (200, 300):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temp_dir:
                image, _ = make_fixture(size=size, brightness=210)
                image_path = os.path.join(temp_dir, f"empty_{size}.png")
                self.assertTrue(cv2.imwrite(image_path, image))

                outputs = process_image(
                    image_path,
                    show_windows=False,
                    open_folder=False,
                )

                self.assertEqual(9, len(outputs["cells"]))
                for cell in outputs["cells"]:
                    self.assertLessEqual(cell["mean"], 50.0)
                    crop = load_image(
                        os.path.join(
                            outputs["output_dir"],
                            f"cell_{cell['idx']:02d}.png",
                        )
                    )
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    boundary = cv2.hconcat(
                        [
                            gray[0:1, :],
                            gray[-1:, :],
                            gray[:, 0:1].T,
                            gray[:, -1:].T,
                        ]
                    )
                    self.assertLessEqual(
                        float((boundary < 100).mean()),
                        0.05,
                    )

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
