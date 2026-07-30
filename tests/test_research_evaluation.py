import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

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


if __name__ == "__main__":
    unittest.main()
