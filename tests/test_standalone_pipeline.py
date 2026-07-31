import csv
import contextlib
import io
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
    def test_process_image_writes_complete_detection_audit_log(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "audit.png")
            log_path = os.path.join(temp_dir, "audit.log")
            self.assertTrue(cv2.imwrite(image_path, image))

            with mock.patch.object(app_standalone, "LOG_FILE", log_path):
                outputs = process_image(
                    image_path,
                    show_windows=False,
                    open_folder=False,
                )

            with open(log_path, encoding="utf-8") as audit_file:
                audit = audit_file.read()
            self.assertIn("Grid detection confidence=", audit)
            self.assertIn("outer_quad=", audit)
            for cell in outputs["cells"]:
                expected = (
                    f"Cell #{cell['idx']} confidence={cell['confidence']:.3f} "
                    f"recovered={str(cell['recovered']).lower()} "
                    f"source_bbox={cell['source_bbox']}"
                )
                self.assertIn(expected, audit)

    def test_process_image_logs_detection_failure_before_reraising(self):
        image = cv2.cvtColor(
            255 * cv2.getStructuringElement(cv2.MORPH_RECT, (900, 900)),
            cv2.COLOR_GRAY2BGR,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "invalid.png")
            log_path = os.path.join(temp_dir, "failure.log")
            self.assertTrue(cv2.imwrite(image_path, image))

            with mock.patch.object(app_standalone, "LOG_FILE", log_path):
                with self.assertRaises(DetectionError) as raised:
                    process_image(
                        image_path,
                        show_windows=False,
                        open_folder=False,
                    )

            with open(log_path, encoding="utf-8") as audit_file:
                audit = audit_file.read()
            self.assertIn("Grid detection failed:", audit)
            self.assertIn(str(raised.exception), audit)

    def test_log_warns_on_stderr_when_log_file_is_unavailable(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch("builtins.open", side_effect=OSError("read-only")), (
            contextlib.redirect_stdout(stdout)
        ), contextlib.redirect_stderr(stderr):
            app_standalone.log("original audit message")

        self.assertIn("original audit message", stdout.getvalue())
        self.assertIn("Log file unavailable", stderr.getvalue())
        self.assertIn("read-only", stderr.getvalue())

    def test_write_failure_does_not_publish_partial_output(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "write-failure.png")
            self.assertTrue(cv2.imwrite(image_path, image))
            real_write = app_standalone._write_image
            calls = 0

            def fail_third(path, content):
                nonlocal calls
                calls += 1
                if calls == 3:
                    return False
                return real_write(path, content)

            with mock.patch.object(
                app_standalone, "_write_image", side_effect=fail_third
            ):
                with self.assertRaises(OSError):
                    process_image(
                        image_path,
                        show_windows=False,
                        open_folder=False,
                    )

            final_dir = os.path.join(
                temp_dir,
                app_standalone._artifact_key("write-failure.png") + "_analysis",
            )
            self.assertFalse(os.path.exists(final_dir))
            self.assertFalse(
                any(name.startswith(".write-failure") for name in os.listdir(temp_dir))
            )

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
            self.assertIn("source_bbox", payload["cells"][0])

            with open(
                outputs["csv_path"], newline="", encoding="utf-8"
            ) as results_file:
                rows = list(csv.DictReader(results_file))
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
                    "source_bbox_x1",
                    "source_bbox_y1",
                    "source_bbox_x2",
                    "source_bbox_y2",
                    "quad_1_x",
                    "quad_1_y",
                    "quad_2_x",
                    "quad_2_y",
                    "quad_3_x",
                    "quad_3_y",
                    "quad_4_x",
                    "quad_4_y",
                ],
                list(rows[0]),
            )
            first_json = payload["cells"][0]
            first_csv = rows[0]
            self.assertEqual(
                first_json["source_bbox"],
                [
                    int(first_csv["source_bbox_x1"]),
                    int(first_csv["source_bbox_y1"]),
                    int(first_csv["source_bbox_x2"]),
                    int(first_csv["source_bbox_y2"]),
                ],
            )
            self.assertEqual(
                first_json["crop_quad"],
                [
                    [
                        int(first_csv[f"quad_{point}_x"]),
                        int(first_csv[f"quad_{point}_y"]),
                    ]
                    for point in range(1, 5)
                ],
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

    def test_extensionless_name_cannot_collide_with_extension_key(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            jpg_path = os.path.join(temp_dir, "sample.jpg")
            extensionless_path = os.path.join(temp_dir, "sample_jpg")
            self.assertTrue(cv2.imwrite(jpg_path, image))
            success, encoded = cv2.imencode(".png", image)
            self.assertTrue(success)
            encoded.tofile(extensionless_path)

            jpg_outputs = process_image(
                jpg_path, show_windows=False, open_folder=False
            )
            extensionless_outputs = process_image(
                extensionless_path, show_windows=False, open_folder=False
            )

            self.assertNotEqual(
                jpg_outputs["output_dir"], extensionless_outputs["output_dir"]
            )
            for outputs, filename in (
                (jpg_outputs, "sample.jpg"),
                (extensionless_outputs, "sample_jpg"),
            ):
                self.assertTrue(os.path.isfile(outputs["json_path"]))
                with open(outputs["json_path"], encoding="utf-8") as result_file:
                    self.assertEqual(filename, json.load(result_file)["image"])

    def test_unicode_paths_use_encoded_image_writes(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            unicode_dir = os.path.join(temp_dir, "中文结果")
            os.makedirs(unicode_dir)
            image_path = os.path.join(unicode_dir, "样本图片.png")
            encoded_ok, encoded_image = cv2.imencode(".png", image)
            self.assertTrue(encoded_ok)
            with open(image_path, "wb") as image_file:
                image_file.write(encoded_image.tobytes())

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
