import json
import tempfile
import unittest
from collections import Counter
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

    def test_plot_iou_values_include_failed_rows_as_zero(self):
        rows = [
            {
                "method": "hybrid",
                "condition": "noise",
                "mean_iou": 1.0,
                "failed": False,
            },
            {
                "method": "hybrid",
                "condition": "noise",
                "mean_iou": 0.0,
                "failed": True,
            },
        ]

        values = evaluation._plot_iou_values(rows, "hybrid", "noise")

        self.assertEqual([1.0, 0.0], values)
        self.assertEqual(0.5, float(np.mean(values)))

    def test_failure_overlays_cannot_escape_output_directory(self):
        image, truth = make_fixture()
        image[:] = 255
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"

            evaluate_methods(
                [("../../escaped", image, tuple(truth))],
                output,
            )

            self.assertFalse(any(root.glob("escaped*.png")))
            failures = output / "failures"
            overlays = list(failures.glob("*.png"))
            self.assertEqual(3, len(overlays))
            for overlay in overlays:
                overlay.resolve().relative_to(failures.resolve())
                self.assertTrue(overlay.name.startswith("000_"))
                self.assertNotIn("..", overlay.name)

    def test_nonempty_output_directory_is_rejected_without_deleting_files(self):
        image, truth = make_fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            stale = output / "failures" / "old.png"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"old-result")

            with self.assertRaisesRegex(ValueError, "empty"):
                evaluate_methods(
                    [("synthetic-001", image, tuple(truth))],
                    output,
                )

            self.assertEqual(b"old-result", stale.read_bytes())
            self.assertEqual([stale], [path for path in output.rglob("*") if path.is_file()])

    def test_runtime_is_repeated_detector_only_and_order_is_rotated(self):
        image, truth = make_fixture()
        events = []
        durations = iter([0.003, 0.001, 0.002] * 9)
        timeline = 10.0
        pending_duration = None

        def clock():
            nonlocal timeline, pending_duration
            if pending_duration is None:
                pending_duration = next(durations)
                events.append(("clock_start",))
                return timeline
            timeline += pending_duration
            pending_duration = None
            events.append(("clock_end",))
            result = timeline
            timeline += 0.01
            return result

        def make_detector(method):
            def detector(case_image):
                case_index = int(case_image[0, 0, 0])
                events.append(("detector", case_index, method))
                return tuple(truth)

            return detector

        original_mean = evaluation._mean_inverted

        def measured_mean(*args):
            events.append(("metric",))
            return original_mean(*args)

        cases = []
        for case_index in range(3):
            case_image = image.copy()
            case_image[0, 0] = case_index
            cases.append((f"case-{case_index}", case_image, tuple(truth)))

        with tempfile.TemporaryDirectory() as directory, mock.patch.multiple(
            evaluation,
            fixed_ratio_detector=make_detector("fixed_ratio"),
            contour_only_detector=make_detector("contour_only"),
            hybrid_detector=make_detector("hybrid"),
            _mean_inverted=measured_mean,
        ):
            rows = evaluate_methods(
                cases,
                directory,
                clock=clock,
            )

        calls = Counter(
            event[2] for event in events if event[0] == "detector"
        )
        self.assertEqual(
            {"fixed_ratio": 12, "contour_only": 12, "hybrid": 12},
            dict(calls),
        )
        for row in rows:
            np.testing.assert_allclose([3.0, 1.0, 2.0], row["runtime_samples_ms"])
            self.assertAlmostEqual(2.0, row["runtime_ms"])
        for index, event in enumerate(events):
            if event[0] == "clock_start":
                self.assertEqual("detector", events[index + 1][0])
                self.assertEqual("clock_end", events[index + 2][0])

        first_methods = {}
        for event in events:
            if event[0] == "detector":
                first_methods.setdefault(event[1], [])
                if event[2] not in first_methods[event[1]]:
                    first_methods[event[1]].append(event[2])
        self.assertEqual(
            {
                0: ["fixed_ratio", "contour_only", "hybrid"],
                1: ["contour_only", "hybrid", "fixed_ratio"],
                2: ["hybrid", "fixed_ratio", "contour_only"],
            },
            first_methods,
        )

    def test_inconsistent_detector_repetitions_are_recorded_as_failure(self):
        image, truth = make_fixture()
        call_count = 0

        def changing_detector(_image):
            nonlocal call_count
            call_count += 1
            method_call = (call_count - 1) % 4
            if method_call == 1:
                shifted = list(truth)
                x1, y1, x2, y2 = shifted[0]
                shifted[0] = (x1 + 1, y1, x2 + 1, y2)
                return tuple(shifted)
            return tuple(truth)

        with tempfile.TemporaryDirectory() as directory, mock.patch.multiple(
            evaluation,
            fixed_ratio_detector=changing_detector,
            contour_only_detector=changing_detector,
            hybrid_detector=changing_detector,
        ):
            rows = evaluate_methods(
                [("synthetic-001", image, tuple(truth))],
                directory,
            )

        self.assertEqual(3, len(rows))
        self.assertTrue(all(row["failed"] for row in rows))
        self.assertTrue(
            all("non-deterministic" in row["error"].lower() for row in rows)
        )


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
