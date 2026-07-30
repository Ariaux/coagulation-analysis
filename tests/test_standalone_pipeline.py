import csv
import json
import os
import tempfile
import unittest

import cv2

from app_standalone import process_image
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


if __name__ == "__main__":
    unittest.main()
