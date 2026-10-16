# Nine-Grid Inner-Square Cropping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows offline drag-and-drop workflow that reliably crops the nine innermost squares of the fixed 3×3 fixture before measurement.

**Architecture:** Add a focused `grid_detector.py` module that locates and rectifies the connected outer fixture, predicts nine inner-square locations from the fixed template, refines each boundary against nearby dark edges, validates the nine results as one coherent grid, and maps coordinates back to the source image. Keep image measurement and desktop interaction in `app_standalone.py`, replacing its variable grid selection with the detector and explicit failure reporting.

**Tech Stack:** Python 3.12, OpenCV, NumPy, standard-library `unittest`, PyInstaller, GitHub Actions

---

## File Structure

- Create `grid_detector.py`: pure detection, validation, coordinate mapping, and overlay helpers; no UI or file I/O.
- Create `tests/test_grid_detector.py`: synthetic fixture generation plus geometry, robustness, empty/sample-content, and failure tests.
- Modify `app_standalone.py`: fixed 3×3 workflow, detector integration, inner-square cropping, result metadata, and user-facing errors.
- Modify `README.md`: offline Windows usage, photography constraints, output semantics, and troubleshooting.
- Modify `.github/workflows/build.yml`: run detector tests before packaging and explicitly include the detector module.
- Create `research/evaluate_cropping.py`: reproducible baseline, ablation, perturbation, metric, CSV/JSON, and figure generation.
- Create `research/annotate_inner_squares.py`: local real-photo ground-truth annotation without network access.
- Create `research/EXPERIMENTS.md`: paper-ready protocol, commands, metric definitions, and non-fabrication rules.
- Create `requirements_research.txt`: research-only plotting dependency kept out of the Windows app.

### Task 1: Detect and Number the Nine Inner Squares

**Files:**
- Create: `grid_detector.py`
- Create: `tests/__init__.py`
- Create: `tests/test_grid_detector.py`

- [ ] **Step 1: Write the failing geometry test**

Create `tests/__init__.py` as an empty file. In `tests/test_grid_detector.py`, add a synthetic fixed-fixture generator and assert that detection returns nine squares in row-major order:

```python
import unittest

import cv2
import numpy as np

from grid_detector import DetectionError, detect_inner_squares


def make_fixture(size=900, angle=0.0, brightness=210, filled=()):
    image = np.full((size, size, 3), brightness, dtype=np.uint8)
    margin = int(size * 0.04)
    cell = (size - 2 * margin) // 3
    line = max(8, size // 90)
    expected = []

    cv2.rectangle(
        image,
        (margin, margin),
        (margin + 3 * cell, margin + 3 * cell),
        (25, 35, 50),
        line * 2,
    )
    for row in range(3):
        for col in range(3):
            x0 = margin + col * cell
            y0 = margin + row * cell
            cv2.rectangle(
                image,
                (x0, y0),
                (x0 + cell, y0 + cell),
                (25, 35, 50),
                line * 2,
            )
            inset = int(cell * 0.22)
            x1, y1 = x0 + inset, y0 + inset
            x2, y2 = x0 + cell - inset, y0 + cell - inset
            cv2.rectangle(image, (x1, y1), (x2, y2), (20, 25, 35), line)
            if row * 3 + col + 1 in filled:
                cv2.rectangle(
                    image,
                    (x1 + line, y1 + line),
                    (x2 - line, y2 - line),
                    (95, 110, 135),
                    -1,
                )
            expected.append((x1 + line, y1 + line, x2 - line, y2 - line))

    if angle:
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        image = cv2.warpAffine(
            image,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderValue=(245, 245, 245),
        )
    return image, expected


class InnerSquareDetectionTests(unittest.TestCase):
    def test_detects_nine_squares_in_row_major_order(self):
        image, expected = make_fixture()
        result = detect_inner_squares(image)

        self.assertEqual([square.idx for square in result.squares], list(range(1, 10)))
        self.assertEqual(
            [(square.row, square.col) for square in result.squares],
            [(row, col) for row in range(1, 4) for col in range(1, 4)],
        )
        self.assertEqual(len(result.squares), 9)
        for square, target in zip(result.squares, expected):
            actual = square.source_bbox
            self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, target)), 12)
            self.assertGreaterEqual(square.confidence, 0.55)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
.venv/bin/python -m unittest tests.test_grid_detector.InnerSquareDetectionTests.test_detects_nine_squares_in_row_major_order -v
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'grid_detector'`.

- [ ] **Step 3: Implement the detector data contract and geometry pipeline**

Create `grid_detector.py` with immutable result types and these public signatures:

```python
from dataclasses import dataclass

import cv2
import numpy as np


class DetectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class InnerSquare:
    idx: int
    row: int
    col: int
    source_quad: np.ndarray
    source_bbox: tuple[int, int, int, int]
    rectified_bbox: tuple[int, int, int, int]
    confidence: float
    recovered: bool = False


@dataclass(frozen=True)
class GridDetection:
    squares: tuple[InnerSquare, ...]
    rectified: np.ndarray
    source_to_rectified: np.ndarray
    rectified_to_source: np.ndarray
    outer_quad: np.ndarray
    confidence: float


@dataclass(frozen=True)
class DetectorOptions:
    rectify: bool = True
    refine_edges: bool = True
    validate_grid: bool = True


def detect_inner_squares(
    image: np.ndarray,
    options: DetectorOptions | None = None,
) -> GridDetection:
    options = options or DetectorOptions()
    if image is None or image.ndim != 3 or min(image.shape[:2]) < 180:
        raise DetectionError("Image is missing or too small.")
    outer_quad = _find_outer_quad(image)
    rectified, source_to_rectified, rectified_to_source = _rectify(
        image,
        outer_quad,
        enabled=options.rectify,
    )
    raw = _refine_template_squares(rectified, enabled=options.refine_edges)
    validated = _validate_grid(
        raw,
        rectified.shape[:2],
        enabled=options.validate_grid,
    )
    squares = _map_squares(validated, rectified_to_source, image.shape[:2])
    confidence = float(np.mean([square.confidence for square in squares]))
    return GridDetection(
        squares=tuple(squares),
        rectified=rectified,
        source_to_rectified=source_to_rectified,
        rectified_to_source=rectified_to_source,
        outer_quad=outer_quad,
        confidence=confidence,
    )
```

Implement the private helpers in the same file:

- `_find_outer_quad(image)`: convert to grayscale, Otsu-threshold the dark fixture, close gaps with a kernel sized to 1.5% of the shorter image side, select the largest connected contour with area between 20% and 95% of the image, and return ordered `cv2.boxPoints(cv2.minAreaRect(contour))`.
- `_rectify(image, quad)`: order points top-left/top-right/bottom-right/bottom-left; warp to a square whose side is the mean of the four edge lengths; return the warped image and both perspective matrices.
- `_refine_template_squares(rectified)`: for each of nine cells, predict edges at 22% and 78% of cell width/height; search within ±7% of cell size using dark-pixel projections measured only in a narrow band around the expected perpendicular span; move each predicted edge to the strongest local dark ridge; then inset past the ridge by half its estimated thickness plus two pixels.
- `_validate_grid(raw, shape)`: require nine candidates; require width/height ratios from 0.85 to 1.15; require each width and height to be within 15% of the median; require row centers and column centers to be monotonically ordered; recover at most one weak edge per candidate from the row/column medians; otherwise raise `DetectionError`.
- `_map_squares(validated, inverse, source_shape)`: perspective-transform each rectangle’s corners back to the source image, clamp them to image bounds, derive an axis-aligned `source_bbox`, and assign fixed `idx`, `row`, and `col`.

Use a confidence score in `[0, 1]` composed of 50% local edge strength, 25% square aspect agreement, and 25% agreement with the median grid size. Reject a final square below `0.55`.

- [ ] **Step 4: Run the geometry test to verify it passes**

Run:

```bash
.venv/bin/python -m unittest tests.test_grid_detector.InnerSquareDetectionTests.test_detects_nine_squares_in_row_major_order -v
```

Expected: `OK`.

- [ ] **Step 5: Commit the detector baseline**

```bash
git add grid_detector.py tests/__init__.py tests/test_grid_detector.py
git commit -m "feat: detect fixed nine-grid inner squares"
```

### Task 2: Make Detection Robust to Content and Normal Photo Variation

**Files:**
- Modify: `tests/test_grid_detector.py`
- Modify: `grid_detector.py`

- [ ] **Step 1: Add failing robustness and failure tests**

Append these tests to `InnerSquareDetectionTests`:

```python
    def test_sample_content_does_not_move_inner_square_edges(self):
        empty, _ = make_fixture()
        filled, _ = make_fixture(filled=(1, 2, 4, 5, 8))
        empty_result = detect_inner_squares(empty)
        filled_result = detect_inner_squares(filled)

        for empty_square, filled_square in zip(empty_result.squares, filled_result.squares):
            delta = max(
                abs(a - b)
                for a, b in zip(empty_square.source_bbox, filled_square.source_bbox)
            )
            self.assertLessEqual(delta, 5)

    def test_tolerates_small_rotation_and_brightness_change(self):
        for angle in (-4.0, 3.0):
            for brightness in (165, 235):
                with self.subTest(angle=angle, brightness=brightness):
                    image, _ = make_fixture(
                        angle=angle,
                        brightness=brightness,
                        filled=(1, 2, 4, 5, 8),
                    )
                    result = detect_inner_squares(image)
                    self.assertEqual(len(result.squares), 9)
                    self.assertGreaterEqual(result.confidence, 0.55)

    def test_rejects_image_without_complete_fixture(self):
        image = np.full((900, 900, 3), 225, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (430, 430), (20, 25, 35), 20)
        with self.assertRaisesRegex(DetectionError, "complete 3x3 fixture"):
            detect_inner_squares(image)
```

- [ ] **Step 2: Run the new tests to verify at least one fails**

Run:

```bash
.venv/bin/python -m unittest tests.test_grid_detector -v
```

Expected: one or more `FAIL` results for rotation/content invariance or the required failure message.

- [ ] **Step 3: Separate structural evidence from sample content**

Update `grid_detector.py` so edge refinement:

```python
def _edge_profile(gray, axis, span_start, span_end):
    band = gray[span_start:span_end, :] if axis == 0 else gray[:, span_start:span_end]
    dark_cutoff = min(120, int(np.percentile(gray, 35)))
    dark_fraction = np.mean(band < dark_cutoff, axis=axis)
    gradient = np.abs(np.gradient(np.mean(band.astype(np.float32), axis=axis)))
    if gradient.max() > 0:
        gradient /= gradient.max()
    return 0.7 * dark_fraction + 0.3 * gradient
```

Restrict each edge’s perpendicular sampling span to the outer 25% strips of the predicted square rather than its content-heavy center. Add `_has_three_by_three_structure(binary, quad)` that projects the rectified dark mask across both axes and requires four repeated major divider bands in each direction (two outer sides plus two internal dividers). Raise:

```python
raise DetectionError(
    "Could not detect a complete 3x3 fixture. Keep the full grid in frame and photograph it straight on."
)
```

when the structural check fails.

- [ ] **Step 4: Run the complete detector suite**

Run:

```bash
.venv/bin/python -m unittest tests.test_grid_detector -v
```

Expected: all four tests report `ok`.

- [ ] **Step 5: Commit robustness**

```bash
git add grid_detector.py tests/test_grid_detector.py
git commit -m "test: harden nine-grid detection"
```

### Task 3: Integrate Inner-Square Cropping into the Offline Desktop App

**Files:**
- Modify: `app_standalone.py`
- Create: `tests/test_standalone_pipeline.py`

- [ ] **Step 1: Write a failing end-to-end processing test**

Create `tests/test_standalone_pipeline.py`:

```python
import json
import os
import tempfile
import unittest

import cv2

from app_standalone import process_image
from tests.test_grid_detector import make_fixture


class StandalonePipelineTests(unittest.TestCase):
    def test_process_image_writes_nine_inner_square_crops_and_metadata(self):
        image, _ = make_fixture(filled=(1, 2, 4, 5, 8))
        with tempfile.TemporaryDirectory() as folder:
            image_path = os.path.join(folder, "fixture.png")
            self.assertTrue(cv2.imwrite(image_path, image))

            outputs = process_image(image_path, show_windows=False, open_folder=False)

            self.assertEqual(len(outputs["cells"]), 9)
            for idx in range(1, 10):
                crop = cv2.imread(os.path.join(outputs["output_dir"], f"cell_{idx:02d}.png"))
                self.assertIsNotNone(crop)
                self.assertLessEqual(abs(crop.shape[1] - crop.shape[0]), 4)
            with open(outputs["json_path"], encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["grid"], "3x3")
            self.assertEqual(len(payload["cells"]), 9)
            self.assertIn("crop_quad", payload["cells"][0])
            self.assertTrue(os.path.exists(outputs["overlay_path"]))
```

- [ ] **Step 2: Run the pipeline test to verify it fails**

Run:

```bash
.venv/bin/python -m unittest tests.test_standalone_pipeline -v
```

Expected: `ERROR` because `app_standalone.process_image` does not exist.

- [ ] **Step 3: Extract a testable fixed-grid processing function**

In `app_standalone.py`:

- import `DetectionError`, `GridDetection`, and `detect_inner_squares` from `grid_detector`;
- remove `pick_grid()` and `confirm_grid()` from the runtime path;
- add `process_image(path, show_windows=True, open_folder=True)`;
- keep `_main()` responsible only for command-line argument handling and friendly fatal-error display.

Use the rectified detection image for exact square crops so a slight camera rotation does not make the crop content diagonal:

```python
def process_image(path, show_windows=True, open_folder=True):
    img = load_image(path)
    if img is None:
        raise ValueError("Cannot open image.")

    detection = detect_inner_squares(img)
    base_name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(os.path.dirname(path) or ".", f"{base_name}_analysis")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for square in detection.squares:
        x1, y1, x2, y2 = square.rectified_bbox
        cell = detection.rectified[y1:y2, x1:x2].copy()
        if cell.size == 0:
            raise DetectionError(f"Inner square #{square.idx} produced an empty crop.")
        inv = 255 - to_8bit(cell)
        measured = measure(inv)
        measured.update({
            "idx": square.idx,
            "row": square.row,
            "col": square.col,
            "confidence": round(square.confidence, 3),
            "recovered": square.recovered,
            "crop_quad": np.rint(square.source_quad).astype(int).tolist(),
        })
        results.append(measured)
        cv2.imwrite(os.path.join(out_dir, f"cell_{square.idx:02d}.png"), cell)

    overlay = draw_detection_overlay(img, detection)
    overlay_path = os.path.join(out_dir, f"{base_name}_grid_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    heatmap = heatmap_image(results, 3, 3)
    heatmap_path = os.path.join(out_dir, f"{base_name}_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap)
    csv_path, json_path = save_results(out_dir, base_name, results)

    if show_windows:
        show_results(base_name, overlay, heatmap)
    if open_folder:
        open_output_folder(out_dir)
    return {
        "cells": results,
        "output_dir": out_dir,
        "overlay_path": overlay_path,
        "heatmap_path": heatmap_path,
        "csv_path": csv_path,
        "json_path": json_path,
    }
```

Add focused helpers `draw_detection_overlay`, `save_results`, `show_results`, and `open_output_folder`. Draw each source quadrilateral green when confidence is at least `0.55`; include `#1` through `#9` at polygon centers. Add `confidence`, `recovered`, and `crop_quad` to JSON, and add `confidence` and `recovered` columns to CSV.

Catch `DetectionError` in `_main()` and show the exact actionable message before waiting for Enter:

```python
    except DetectionError as exc:
        log(f"DETECTION FAILED: {exc}")
        input(f"\nDetection failed: {exc}\nPress Enter to exit...")
```

- [ ] **Step 4: Run desktop pipeline and regression tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_standalone_pipeline tests.test_grid_detector -v
```

Expected: all tests report `ok`.

- [ ] **Step 5: Run the provided photo without UI and inspect the outputs**

Run:

```bash
.venv/bin/python - <<'PY'
from app_standalone import process_image

result = process_image(
    "/var/folders/vl/tw5qk42j12q95ll5tnj2bmy00000gn/T/codex-clipboard-aa3c00d9-4b04-4f79-ada8-62795a8577de.png",
    show_windows=False,
    open_folder=False,
)
print(result["output_dir"])
print([(cell["idx"], cell["confidence"]) for cell in result["cells"]])
PY
```

Expected: nine entries numbered 1–9, every confidence at least `0.55`, and an analysis directory containing nine cell images plus overlay, heatmap, CSV, and JSON. Visually inspect the overlay and a 3×3 contact sheet of the crops; no crop may contain the surrounding long slots or thick black frame.

- [ ] **Step 6: Commit desktop integration**

```bash
git add app_standalone.py tests/test_standalone_pipeline.py
git commit -m "feat: crop inner squares in offline desktop app"
```

### Task 4: Add Reproducible Paper Experiments

**Files:**
- Create: `research/__init__.py`
- Create: `research/evaluate_cropping.py`
- Create: `research/annotate_inner_squares.py`
- Create: `research/EXPERIMENTS.md`
- Create: `requirements_research.txt`
- Create: `tests/test_research_evaluation.py`

- [ ] **Step 1: Write failing metric and benchmark tests**

Create `tests/test_research_evaluation.py` with tests for exact-match IoU, known pixel displacement, deterministic perturbations, and three-method output:

```python
import tempfile
import unittest

import numpy as np

from research.evaluate_cropping import (
    boundary_error,
    box_iou,
    evaluate_methods,
    make_perturbations,
)
from tests.test_grid_detector import make_fixture


class ResearchEvaluationTests(unittest.TestCase):
    def test_metrics_have_known_values(self):
        truth = (10, 20, 110, 120)
        shifted = (15, 20, 115, 120)
        self.assertEqual(box_iou(truth, truth), 1.0)
        self.assertAlmostEqual(boundary_error(truth, shifted), 2.5)
        self.assertAlmostEqual(box_iou(truth, shifted), 95 / 105)

    def test_perturbations_are_deterministic(self):
        image, truth = make_fixture(filled=(1, 2, 4, 5, 8))
        first = make_perturbations(image, truth, seed=20260730)
        second = make_perturbations(image, truth, seed=20260730)
        self.assertEqual([case.name for case in first], [case.name for case in second])
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.image, right.image)
            self.assertEqual(left.truth, right.truth)

    def test_benchmark_reports_all_three_methods(self):
        image, truth = make_fixture(filled=(1, 2, 4, 5, 8))
        with tempfile.TemporaryDirectory() as output_dir:
            rows = evaluate_methods(
                [("synthetic-001", image, truth)],
                output_dir=output_dir,
            )
        self.assertEqual(
            {row["method"] for row in rows},
            {"fixed_ratio", "contour_only", "hybrid"},
        )
        self.assertTrue(all("mean_iou" in row for row in rows))
        self.assertTrue(all("runtime_ms" in row for row in rows))
```

- [ ] **Step 2: Run the research tests and verify they fail**

Run:

```bash
/Users/aria/Downloads/chenmeixi/.venv/bin/python -m unittest tests.test_research_evaluation -v
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'research'`.

- [ ] **Step 3: Implement metrics, baselines, perturbations, and ablations**

In `research/evaluate_cropping.py`, define:

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    image: np.ndarray
    truth: tuple[tuple[int, int, int, int], ...]
    condition: str
    level: float


def box_iou(a, b) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def boundary_error(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
```

Add `fixed_ratio_detector(image)`, `contour_only_detector(image)`, `hybrid_detector(image)`, `make_perturbations(image, truth, seed=20260730)`, and `evaluate_methods(cases, output_dir, ablations=False)` with these exact behaviors:

- `box_iou`: intersection area divided by union area, returning `0.0` for no overlap.
- `boundary_error`: mean absolute difference of left, top, right, and bottom coordinates.
- `fixed_ratio_detector`: detect only the outer fixture and return template boxes at the calibrated 22%/78% insets without local refinement.
- `contour_only_detector`: divide the detected outer fixture into nine cells and choose the largest near-square contour inside each cell without shared-grid recovery.
- `hybrid_detector`: call `detect_inner_squares`.
- `make_perturbations`: create deterministic cases for rotations `[-5, -3, 3, 5]`, brightness multipliers `[0.7, 0.85, 1.15, 1.3]`, scales `[0.85, 1.15]`, Gaussian noise standard deviations `[8, 16]`, and content/empty patterns; transform ground-truth boxes with the same affine matrices.
- `evaluate_methods`: time each method with `time.perf_counter`, compute per-image mean IoU, mean boundary error, single-cell success at IoU ≥ 0.85, all-nine success, and measurement mean absolute error; catch `DetectionError` as a recorded failed result.
- when `ablations=True`, evaluate hybrid variants by passing `DetectorOptions(rectify=False)`, `DetectorOptions(refine_edges=False)`, and `DetectorOptions(validate_grid=False)` to `detect_inner_squares`.

Write `per_image_results.csv`, `summary.json`, `method_comparison.png`, `robustness_by_condition.png`, and a `failures/` directory. Use Matplotlib only in the research module, never in the desktop runtime.

- [ ] **Step 4: Add local ground-truth annotation**

Implement `research/annotate_inner_squares.py` as a local OpenCV tool. The annotator opens one photo at a time, asks the researcher to click top-left and bottom-right corners for cells 1–9 in row-major order, permits Backspace to redo the current cell, draws numbered rectangles, and writes:

```json
{
  "image": "sample.jpg",
  "annotator": "manual",
  "boxes": [
    {"idx": 1, "bbox": [x1, y1, x2, y2]}
  ]
}
```

Require exactly nine non-empty boxes before saving. Do not offer automatic pre-labels, because real-photo ground truth must remain independent of the evaluated detector.

- [ ] **Step 5: Document the paper protocol**

Create `research/EXPERIMENTS.md` with:

- the research question and three compared methods;
- independent real-photo annotation protocol;
- synthetic perturbation matrix and fixed random seed;
- definitions for IoU, boundary error, cell success, all-nine success, measurement error, and runtime;
- ablation definitions;
- exact commands to annotate and evaluate;
- a results-table template containing blank cells marked `—`, never invented values;
- a warning that at least two people should independently review a subset of manual labels before reporting real-image accuracy.

Create `requirements_research.txt`:

```text
opencv-python
numpy
matplotlib
```

- [ ] **Step 6: Run research tests and smoke-test generated artifacts**

Run:

```bash
/Users/aria/Downloads/chenmeixi/.venv/bin/python -m unittest tests.test_research_evaluation -v
/Users/aria/Downloads/chenmeixi/.venv/bin/python -m research.evaluate_cropping --synthetic --output research/results/smoke
```

Expected: all research tests report `ok`; the smoke command writes non-empty CSV, JSON, two PNG charts, and exits with code 0.

- [ ] **Step 7: Commit the research evaluation tooling**

```bash
git add research requirements_research.txt tests/test_research_evaluation.py
git commit -m "feat: add reproducible cropping evaluation"
```

### Task 5: Document and Verify the Windows Offline Build

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/build.yml`

- [ ] **Step 1: Update usage and troubleshooting documentation**

In `README.md`, replace the standalone usage text with:

```markdown
### Windows Offline App — Fixed 3×3 Fixture

1. Download and unzip `CoagulationAnalysis-Windows.zip`.
2. Keep the full `CoagulationAnalysis` folder together.
3. Photograph the complete 3×3 fixture straight on with all four outer edges visible.
4. Drag the photo onto `CoagulationAnalysis.exe`.
5. Open `<photo-name>_analysis` to review the numbered overlay, nine inner-square crops, heatmap, CSV, and JSON.

The app always analyzes the nine innermost squares in row-major order. Empty squares are retained. If the fixture is incomplete or detection confidence is too low, the app stops and asks for a new photo rather than silently producing incorrect measurements. No internet connection or Python installation is required.
```

Add troubleshooting entries for an incomplete fixture, strong side angle, reflections over black edges, and moving/deleting files from the unzipped application folder.

- [ ] **Step 2: Make CI test before packaging**

Update `.github/workflows/build.yml`:

```yaml
      - run: pip install pyinstaller opencv-python numpy
      - name: Run detector tests
        run: python -m unittest tests.test_grid_detector tests.test_standalone_pipeline -v
      - run: pyinstaller --onedir --name "CoagulationAnalysis" app_standalone.py
```

PyInstaller will discover `grid_detector.py` through the direct import; do not add a hidden import unless the build log reports it missing.

- [ ] **Step 3: Run all local tests and syntax checks**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile grid_detector.py app_standalone.py
git diff --check
```

Expected: all tests pass, compilation emits no output, and `git diff --check` emits no output.

- [ ] **Step 4: Build a local distributable smoke test**

Run:

```bash
.venv/bin/python -m pip install pyinstaller
.venv/bin/pyinstaller --clean --noconfirm --onedir --name CoagulationAnalysis app_standalone.py
```

Expected: `dist/CoagulationAnalysis/CoagulationAnalysis` exists on macOS. This validates module collection; the Windows `.exe` remains the responsibility of the `windows-latest` GitHub Actions job.

- [ ] **Step 5: Final visual verification**

Run the non-UI sample command from Task 3 again, open the generated overlay, and inspect all nine crops as a contact sheet. Confirm:

- nine green polygons follow the innermost boundaries;
- cells are numbered left-to-right, top-to-bottom;
- all crops exclude the black fixture;
- blank cells remain present;
- output CSV and JSON each contain nine records.

- [ ] **Step 6: Commit documentation and build checks**

```bash
git add README.md .github/workflows/build.yml
git commit -m "docs: describe fixed nine-grid Windows workflow"
```

### Task 6: Completion Review

**Files:**
- Review only: `grid_detector.py`
- Review only: `app_standalone.py`
- Review only: `tests/test_grid_detector.py`
- Review only: `tests/test_standalone_pipeline.py`
- Review only: `README.md`
- Review only: `.github/workflows/build.yml`
- Review only: `research/evaluate_cropping.py`
- Review only: `research/annotate_inner_squares.py`
- Review only: `research/EXPERIMENTS.md`

- [ ] **Step 1: Compare implementation against the design specification**

Read `docs/superpowers/specs/2026-07-30-nine-grid-inner-square-cropping-design.md` and confirm every in-scope requirement maps to code or a test: fixed 3×3, inner-edge inset, row-major numbering, empty-cell retention, confidence failure, preview, nine crops, heatmap, CSV/JSON coordinates, offline operation, Windows build, three-method comparison, ablations, deterministic perturbations, independent real-photo annotation, metrics, and paper artifacts.

- [ ] **Step 2: Check repository state and commit scope**

Run:

```bash
git status --short
git log --oneline -6
```

Expected: only the pre-existing untracked handoff documents may remain; no implementation file is uncommitted.

- [ ] **Step 3: Re-run final evidence**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile grid_detector.py app_standalone.py
git diff --check
```

Expected: all tests pass and both verification commands are silent.
