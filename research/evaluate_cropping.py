"""Reproducible evaluation of nine-grid cropping methods."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid_detector import (
    DetectionError,
    DetectorOptions,
    _find_outer_quad,
    _rectify,
    detect_inner_squares,
)

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    image: np.ndarray
    truth: tuple[BBox, ...]
    condition: str = "baseline"
    level: str = "reference"

    def __post_init__(self) -> None:
        if len(self.truth) != 9:
            raise ValueError("Evaluation cases require exactly nine truth boxes.")


def box_iou(left: BBox, right: BBox) -> float:
    """Return intersection over union for two half-open xyxy boxes."""
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(0, lx2 - lx1) * max(0, ly2 - ly1)
    right_area = max(0, rx2 - rx1) * max(0, ry2 - ry1)
    union = left_area + right_area - intersection
    return float(intersection / union) if union else 0.0


def boundary_error(truth: BBox, predicted: BBox) -> float:
    """Return mean absolute error across the four xyxy boundaries."""
    return float(np.mean(np.abs(np.asarray(truth) - np.asarray(predicted))))


def _boxes_from_detection(image: np.ndarray, options: DetectorOptions) -> tuple[BBox, ...]:
    detection = detect_inner_squares(image, options)
    return tuple(square.source_bbox for square in detection.squares)


def fixed_ratio_detector(image: np.ndarray) -> tuple[BBox, ...]:
    """Apply the calibrated 22/78 percent template without edge refinement."""
    if image is None or image.ndim != 3 or min(image.shape[:2]) < 600:
        raise DetectionError("A color image of at least 600x600 pixels is required.")
    outer = _find_outer_quad(image)
    rectified, _, inverse = _rectify(image, outer, True)
    height, width = rectified.shape[:2]
    cell_width = width / 3.0
    cell_height = height / 3.0
    safe_x = max(4, round(cell_width * 0.025))
    safe_y = max(4, round(cell_height * 0.025))
    boxes = []
    for row in range(3):
        for col in range(3):
            box = (
                round((col + 0.22) * cell_width) + safe_x,
                round((row + 0.22) * cell_height) + safe_y,
                round((col + 0.78) * cell_width) - safe_x,
                round((row + 0.78) * cell_height) - safe_y,
            )
            boxes.append(_map_box(box, inverse, image.shape))
    return tuple(boxes)


def _map_box(box: BBox, inverse: np.ndarray, shape: tuple[int, ...]) -> BBox:
    x1, y1, x2, y2 = box
    corners = np.asarray([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], np.float32)
    mapped = cv2.perspectiveTransform(corners, inverse)[0]
    height, width = shape[:2]
    return (
        int(np.clip(np.floor(mapped[:, 0].min()), 0, width - 1)),
        int(np.clip(np.floor(mapped[:, 1].min()), 0, height - 1)),
        int(np.clip(np.ceil(mapped[:, 0].max()), 1, width)),
        int(np.clip(np.ceil(mapped[:, 1].max()), 1, height)),
    )


def contour_only_detector(image: np.ndarray) -> tuple[BBox, ...]:
    """Find one plausible near-square dark contour independently per cell."""
    if image is None or image.ndim != 3 or min(image.shape[:2]) < 600:
        raise DetectionError("A color image of at least 600x600 pixels is required.")
    outer = _find_outer_quad(image)
    rectified, _, inverse = _rectify(image, outer, True)
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    height, width = gray.shape
    boxes: list[BBox] = []
    for row in range(3):
        for col in range(3):
            cell_x1, cell_x2 = round(col * width / 3), round((col + 1) * width / 3)
            cell_y1, cell_y2 = round(row * height / 3), round((row + 1) * height / 3)
            cell_mask = dark[cell_y1:cell_y2, cell_x1:cell_x2]
            contours, _ = cv2.findContours(
                cell_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            cell_side = min(cell_mask.shape)
            candidates: list[tuple[float, tuple[int, int, int, int]]] = []
            for contour in contours:
                x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
                ratio = candidate_width / max(candidate_height, 1)
                relative = np.sqrt(candidate_width * candidate_height) / cell_side
                if 0.8 <= ratio <= 1.25 and 0.42 <= relative <= 0.72:
                    score = abs(relative - 0.58) + abs(1.0 - ratio) * 0.25
                    candidates.append(
                        (score, (x, y, x + candidate_width, y + candidate_height))
                    )
            if not candidates:
                raise DetectionError(f"No plausible inner contour found in cell {len(boxes)+1}.")
            _, (x1, y1, x2, y2) = min(candidates, key=lambda item: item[0])
            safe_inset = max(3, round(cell_side * 0.043))
            rectified_box = (
                cell_x1 + x1 + safe_inset,
                cell_y1 + y1 + safe_inset,
                cell_x1 + x2 - safe_inset,
                cell_y1 + y2 - safe_inset,
            )
            if rectified_box[2] <= rectified_box[0] or rectified_box[3] <= rectified_box[1]:
                raise DetectionError(f"Degenerate inner contour found in cell {len(boxes)+1}.")
            boxes.append(_map_box(rectified_box, inverse, image.shape))
    return tuple(boxes)


def hybrid_detector(image: np.ndarray) -> tuple[BBox, ...]:
    """Run the normal production detector."""
    return _boxes_from_detection(image, DetectorOptions())


def _transform_boxes(boxes: Sequence[BBox], matrix: np.ndarray, shape: tuple[int, int]) -> tuple[BBox, ...]:
    output: list[BBox] = []
    height, width = shape
    for box in boxes:
        x1, y1, x2, y2 = box
        points = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
        transformed = cv2.transform(points[None, :, :], matrix)[0]
        output.append(
            (
                int(np.clip(np.floor(transformed[:, 0].min()), 0, width - 1)),
                int(np.clip(np.floor(transformed[:, 1].min()), 0, height - 1)),
                int(np.clip(np.ceil(transformed[:, 0].max()), 1, width)),
                int(np.clip(np.ceil(transformed[:, 1].max()), 1, height)),
            )
        )
    return tuple(output)


def make_perturbations(
    image: np.ndarray, truth: tuple[BBox, ...], seed: int = 20260730
) -> list[EvaluationCase]:
    """Construct the deterministic robustness matrix used by the study."""
    cases: list[EvaluationCase] = []
    height, width = image.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    for angle in (-5, -3, 3, 5):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (width, height), borderValue=tuple(map(int, image[0, 0]))
        )
        cases.append(
            EvaluationCase(
                f"rotation-{angle:+d}",
                rotated,
                _transform_boxes(truth, matrix, rotated.shape[:2]),
                "rotation",
                str(angle),
            )
        )
    for multiplier in (0.7, 0.85, 1.15, 1.3):
        adjusted = np.clip(image.astype(np.float32) * multiplier, 0, 255).astype(np.uint8)
        cases.append(
            EvaluationCase(
                f"brightness-{multiplier:g}", adjusted, truth, "brightness", f"{multiplier:g}"
            )
        )
    for scale in (0.85, 1.15):
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_truth = tuple(
            tuple(int(round(value * scale)) for value in box) for box in truth
        )
        target_height = max(600, scaled.shape[0])
        target_width = max(600, scaled.shape[1])
        top = (target_height - scaled.shape[0]) // 2
        left = (target_width - scaled.shape[1]) // 2
        if top or left:
            scaled = cv2.copyMakeBorder(
                scaled,
                top,
                target_height - scaled.shape[0] - top,
                left,
                target_width - scaled.shape[1] - left,
                cv2.BORDER_CONSTANT,
                value=tuple(map(int, image[0, 0])),
            )
            scaled_truth = tuple(
                (x1 + left, y1 + top, x2 + left, y2 + top)
                for x1, y1, x2, y2 in scaled_truth
            )
        cases.append(
            EvaluationCase(f"scale-{scale:g}", scaled, scaled_truth, "scale", f"{scale:g}")
        )
    generator = np.random.default_rng(seed)
    for standard_deviation in (8, 16):
        noise = generator.normal(0, standard_deviation, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        cases.append(
            EvaluationCase(
                f"noise-{standard_deviation}",
                noisy,
                truth,
                "gaussian_noise",
                str(standard_deviation),
            )
        )
    empty = image.copy()
    alternating = image.copy()
    background = int(np.median(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
    for index, (x1, y1, x2, y2) in enumerate(truth):
        empty[y1:y2, x1:x2] = background
        alternating[y1:y2, x1:x2] = 65 if index % 2 == 0 else background
    cases.extend(
        [
            EvaluationCase("content-empty", empty, truth, "content", "empty"),
            EvaluationCase(
                "content-alternating", alternating, truth, "content", "alternating"
            ),
        ]
    )
    return cases


def _mean_inverted(image: np.ndarray, box: BBox) -> float:
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if not crop.size:
        raise DetectionError("A predicted crop is empty.")
    b, g, r = [crop[:, :, channel].astype(np.float32) for channel in range(3)]
    grayscale = np.clip(0.114 * b + 0.587 * g + 0.299 * r, 0, 255).astype(np.uint8)
    return float(np.mean(255 - grayscale))


def _normalise_case(item: EvaluationCase | tuple[str, np.ndarray, Sequence[BBox]]) -> EvaluationCase:
    if isinstance(item, EvaluationCase):
        return item
    name, image, truth = item
    return EvaluationCase(name, image, tuple(truth))


def _annotate_failure(image: np.ndarray, truth: Sequence[BBox], predicted: Sequence[BBox]) -> np.ndarray:
    annotated = image.copy()
    for box in truth:
        cv2.rectangle(annotated, box[:2], box[2:], (0, 255, 0), 2)
    for box in predicted:
        cv2.rectangle(annotated, box[:2], box[2:], (0, 0, 255), 2)
    return annotated


def _aggregate(rows: Sequence[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    result = {}
    for name, group in groups.items():
        successful = [row for row in group if not row["failed"]]
        result[name] = {
            "n": len(group),
            "detections": len(successful),
            "mean_iou": float(np.mean([row["mean_iou"] for row in successful]))
            if successful
            else None,
            "all_nine_success_rate": float(
                np.mean([row["all_nine_success"] for row in group])
            ),
            "mean_runtime_ms": float(np.mean([row["runtime_ms"] for row in group])),
        }
    return result


def _write_plots(rows: Sequence[dict], output_dir: Path) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    method_values = [
        [float(row["mean_iou"]) for row in rows if row["method"] == method and not row["failed"]]
        for method in methods
    ]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.boxplot([values or [0.0] for values in method_values], tick_labels=methods)
    axis.set_ylabel("Mean IoU")
    axis.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "method_comparison.png", dpi=150)
    plt.close(fig)

    conditions = sorted({str(row["condition"]) for row in rows})
    fig, axis = plt.subplots(figsize=(9, 4.8))
    x_values = np.arange(len(conditions))
    width = 0.8 / max(1, len(methods))
    for offset, method in enumerate(methods):
        values = []
        for condition in conditions:
            selected = [
                float(row["mean_iou"])
                for row in rows
                if row["method"] == method
                and row["condition"] == condition
                and not row["failed"]
            ]
            values.append(float(np.mean(selected)) if selected else 0.0)
        axis.bar(x_values + offset * width, values, width, label=method)
    axis.set_xticks(x_values + width * (len(methods) - 1) / 2, conditions)
    axis.set_ylabel("Mean IoU")
    axis.set_ylim(0, 1.02)
    axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_by_condition.png", dpi=150)
    plt.close(fig)


def evaluate_methods(
    cases: Iterable[EvaluationCase | tuple[str, np.ndarray, Sequence[BBox]]],
    output_dir: str | Path,
    ablations: bool = False,
) -> list[dict]:
    """Evaluate methods, write tabular results, plots, and failure overlays."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    failures = output / "failures"
    failures.mkdir(exist_ok=True)
    methods: list[tuple[str, Callable[[np.ndarray], tuple[BBox, ...]]]] = [
        ("fixed_ratio", fixed_ratio_detector),
        ("contour_only", contour_only_detector),
        ("hybrid", hybrid_detector),
    ]
    if ablations:
        methods.extend(
            [
                (
                    "hybrid_no_rectification",
                    lambda image: _boxes_from_detection(
                        image, DetectorOptions(rectify=False)
                    ),
                ),
                (
                    "hybrid_no_refinement",
                    lambda image: _boxes_from_detection(
                        image, DetectorOptions(refine_edges=False)
                    ),
                ),
                (
                    "hybrid_no_grid_validation",
                    lambda image: _boxes_from_detection(
                        image, DetectorOptions(validate_grid=False)
                    ),
                ),
            ]
        )
    rows: list[dict] = []
    for case_item in cases:
        case = _normalise_case(case_item)
        truth_means = [_mean_inverted(case.image, box) for box in case.truth]
        for method_name, detector in methods:
            start = time.perf_counter()
            predicted: tuple[BBox, ...] = ()
            error = ""
            try:
                predicted = detector(case.image)
                if len(predicted) != 9:
                    raise DetectionError("Detector did not return exactly nine boxes.")
                ious = [box_iou(truth, found) for truth, found in zip(case.truth, predicted)]
                errors = [
                    boundary_error(truth, found)
                    for truth, found in zip(case.truth, predicted)
                ]
                predicted_means = [_mean_inverted(case.image, box) for box in predicted]
                measurement_error = float(
                    np.mean(np.abs(np.asarray(truth_means) - np.asarray(predicted_means)))
                )
            except DetectionError as exception:
                error = str(exception)
                ious = []
                errors = []
                measurement_error = float("nan")
            runtime_ms = (time.perf_counter() - start) * 1000.0
            cell_success = sum(iou >= 0.85 for iou in ious)
            all_nine = cell_success == 9
            row = {
                "case": case.name,
                "condition": case.condition,
                "level": case.level,
                "method": method_name,
                "failed": bool(error),
                "error": error,
                "mean_iou": float(np.mean(ious)) if ious else 0.0,
                "mean_boundary_error": float(np.mean(errors)) if errors else None,
                "cell_success_count": cell_success,
                "cell_success_rate": cell_success / 9.0,
                "all_nine_success": all_nine,
                "measurement_mae": measurement_error if not np.isnan(measurement_error) else None,
                "runtime_ms": runtime_ms,
            }
            rows.append(row)
            if error or not all_nine:
                annotated = _annotate_failure(case.image, case.truth, predicted)
                cv2.imwrite(str(failures / f"{case.name}__{method_name}.png"), annotated)

    fields = list(rows[0]) if rows else []
    with (output / "per_image_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "thresholds": {"cell_success_iou": 0.85, "all_nine_cells_required": 9},
        "by_method": _aggregate(rows, "method"),
        "by_condition": _aggregate(rows, "condition"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_plots(rows, output)
    return rows


def make_synthetic_fixture(
    size: int = 900, filled: Sequence[int] = (2, 5, 8)
) -> tuple[np.ndarray, tuple[BBox, ...]]:
    """Create a study fixture without depending on the test package."""
    image = np.full((size, size, 3), 210, np.uint8)
    margin = round(size * 0.04)
    cell = (size - 2 * margin) // 3
    line = max(8, size // 90)
    end = margin + 3 * cell
    cv2.rectangle(image, (margin, margin), (end, end), (25, 25, 25), line)
    truth = []
    for row in range(3):
        for col in range(3):
            x1, y1 = margin + col * cell, margin + row * cell
            x2, y2 = x1 + cell, y1 + cell
            cv2.rectangle(image, (x1, y1), (x2, y2), (25, 25, 25), line)
            inset = round(cell * 0.22)
            inner = (x1 + inset, y1 + inset, x2 - inset, y2 - inset)
            cv2.rectangle(image, inner[:2], inner[2:], (25, 25, 25), line)
            safe = line // 2 + 2
            box = (inner[0] + safe, inner[1] + safe, inner[2] - safe, inner[3] - safe)
            truth.append(box)
            if row * 3 + col + 1 in filled:
                cv2.rectangle(image, box[:2], box[2:], (65, 65, 65), -1)
    return image, tuple(truth)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="run the synthetic matrix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ablations", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.synthetic:
        parser.error("Select --synthetic; real data require an explicit labeled manifest.")
    image, truth = make_synthetic_fixture()
    cases = [EvaluationCase("synthetic-reference", image, truth)]
    cases.extend(make_perturbations(image, truth))
    rows = evaluate_methods(cases, arguments.output, arguments.ablations)
    print(f"Evaluated {len(cases)} cases and wrote {len(rows)} method rows.")
    for name in (
        "per_image_results.csv",
        "summary.json",
        "method_comparison.png",
        "robustness_by_condition.png",
    ):
        print(arguments.output / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
