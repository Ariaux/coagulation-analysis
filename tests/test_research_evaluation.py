import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import research.evaluate_cropping as evaluation
from research.annotate_inner_squares import (
    save_annotations,
    validate_annotations,
)
from research.evaluate_cropping import (
    boundary_error,
    box_iou,
    evaluate_methods,
    fixed_ratio_detector,
    make_perturbations,
)
from tests.test_grid_detector import make_fixture


class ResearchEvaluationTests(unittest.TestCase):
    def test_box_metrics_use_xyxy_coordinates(self):
        truth = (10, 20, 110, 120)
        shifted = (15, 20, 115, 120)

        self.assertEqual(1.0, box_iou(truth, truth))
        self.assertEqual(2.5, boundary_error(truth, shifted))
        self.assertAlmostEqual(95 / 105, box_iou(truth, shifted))

    def test_perturbations_are_deterministic(self):
        image, truth = make_fixture(filled=(1, 5, 9))

        first = make_perturbations(image, tuple(truth), seed=20260730)
        second = make_perturbations(image, tuple(truth), seed=20260730)

        self.assertGreaterEqual(len(first), 14)
        self.assertEqual(
            [(case.name, case.truth, case.condition, case.level) for case in first],
            [(case.name, case.truth, case.condition, case.level) for case in second],
        )
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.image, right.image)
            self.assertGreaterEqual(min(left.image.shape[:2]), 600)

    def test_fixed_ratio_uses_safe_content_inset(self):
        image, truth = make_fixture()

        predicted = fixed_ratio_detector(image)

        mean_error = np.mean(
            [boundary_error(wanted, found) for wanted, found in zip(truth, predicted)]
        )
        self.assertLessEqual(mean_error, 4.0)

    def test_evaluation_writes_metrics_and_artifacts(self):
        image, truth = make_fixture(filled=(2, 5, 8))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            rows = evaluate_methods(
                [("synthetic-001", image, tuple(truth))],
                output,
            )

            self.assertEqual(
                {"fixed_ratio", "contour_only", "hybrid"},
                {row["method"] for row in rows},
            )
            for row in rows:
                self.assertIn("mean_iou", row)
                self.assertIn("runtime_ms", row)
            for relative in (
                "per_image_results.csv",
                "summary.json",
                "method_comparison.png",
                "robustness_by_condition.png",
            ):
                artifact = output / relative
                self.assertTrue(artifact.is_file(), relative)
                self.assertGreater(artifact.stat().st_size, 0, relative)
            summary = json.loads((output / "summary.json").read_text("utf-8"))
            self.assertIn("by_method", summary)
            self.assertIn("by_condition", summary)
            required_metrics = {
                "mean_iou",
                "mean_boundary_error",
                "cell_success_rate",
                "all_nine_success_rate",
                "measurement_mae",
                "mean_runtime_ms",
                "counts",
            }
            for method in ("fixed_ratio", "contour_only", "hybrid"):
                self.assertTrue(
                    required_metrics <= set(summary["by_method"][method])
                )
                baseline = summary["method_by_condition"][method]["baseline"]
                self.assertTrue(required_metrics <= set(baseline))

    def test_contour_baseline_selects_largest_qualifying_square(self):
        smaller_closer_to_template = np.asarray(
            [[[20, 20]], [[78, 20]], [[78, 78]], [[20, 78]]], np.int32
        )
        larger = np.asarray(
            [[[10, 10]], [[80, 10]], [[80, 80]], [[10, 80]]], np.int32
        )

        selected = evaluation._choose_contour(
            [smaller_closer_to_template, larger], cell_side=100
        )

        self.assertEqual(cv2.boundingRect(larger), selected)

    def test_runtime_times_detector_only(self):
        image, truth = make_fixture()
        events = []
        clock_values = iter((10.0, 11.0, 20.0, 21.0, 30.0, 31.0))

        def clock():
            events.append("clock")
            return next(clock_values)

        def detector(_image):
            events.append("detector")
            if events.count("detector") == 2:
                raise evaluation.DetectionError("controlled failure")
            return tuple(truth)

        original_mean = evaluation._mean_inverted

        def measured_mean(*args):
            events.append("metric")
            return original_mean(*args)

        with tempfile.TemporaryDirectory() as directory, mock.patch.multiple(
            evaluation,
            fixed_ratio_detector=detector,
            contour_only_detector=detector,
            hybrid_detector=detector,
            _mean_inverted=measured_mean,
        ):
            rows = evaluate_methods(
                [("synthetic-001", image, tuple(truth))],
                directory,
                clock=clock,
            )
            summary = json.loads(
                (Path(directory) / "summary.json").read_text("utf-8")
            )

        first_detector = events.index("detector")
        self.assertEqual("clock", events[first_detector - 1])
        self.assertEqual("clock", events[first_detector + 1])
        self.assertEqual([1000.0, 1000.0, 1000.0], [row["runtime_ms"] for row in rows])
        self.assertTrue(rows[1]["failed"])
        failed_summary = summary["by_method"]["contour_only"]
        self.assertEqual(0.0, failed_summary["mean_iou"])
        self.assertEqual(
            {"total": 1, "detected": 0, "failed": 1},
            failed_summary["counts"],
        )
        self.assertIsNone(failed_summary["mean_boundary_error"])
        self.assertIsNone(failed_summary["measurement_mae"])


class AnnotationValidationTests(unittest.TestCase):
    def setUp(self):
        self.boxes = [
            (10 + col * 100, 20 + row * 100, 80 + col * 100, 90 + row * 100)
            for row in range(3)
            for col in range(3)
        ]

    def test_accepts_exactly_nine_row_major_nonempty_boxes(self):
        payload = validate_annotations("sample.png", self.boxes)

        self.assertEqual("sample.png", payload["image"])
        self.assertEqual("manual", payload["annotator"])
        self.assertEqual(list(range(1, 10)), [item["idx"] for item in payload["boxes"]])
        self.assertEqual(list(self.boxes[0]), payload["boxes"][0]["bbox"])

        with tempfile.TemporaryDirectory() as directory:
            path = save_annotations(
                Path(directory) / "sample.json", "sample.png", self.boxes
            )
            self.assertEqual(payload, json.loads(path.read_text("utf-8")))

    def test_rejects_incomplete_degenerate_or_out_of_order_boxes(self):
        with self.assertRaises(ValueError):
            validate_annotations("sample.png", self.boxes[:8])
        degenerate = list(self.boxes)
        degenerate[4] = (100, 100, 100, 200)
        with self.assertRaises(ValueError):
            validate_annotations("sample.png", degenerate)
        out_of_order = list(self.boxes)
        out_of_order[0], out_of_order[8] = out_of_order[8], out_of_order[0]
        with self.assertRaises(ValueError):
            validate_annotations("sample.png", out_of_order)

    def test_rejects_overlapping_rows_even_when_average_y_increases(self):
        boxes = list(self.boxes)
        boxes[2] = (210, 230, 280, 270)

        with self.assertRaises(ValueError):
            validate_annotations("sample.png", boxes)


if __name__ == "__main__":
    unittest.main()
