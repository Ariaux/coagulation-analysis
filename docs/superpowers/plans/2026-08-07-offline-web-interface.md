# Offline Web Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a packaged Windows-only local website that reuses the fixed nine-grid detector, supports single and batch analysis, excludes black frame pixels with an adjustable inset, and produces a thresholded blue/red publication heatmap.

**Architecture:** Move UI-independent analysis behavior into a shared `analysis_service.py` used by both the existing desktop entry point and the new Gradio controller. Keep batch orchestration, browser-facing value conversion, UI construction, and Windows startup in separate focused modules. All durable outputs are staged atomically under a local `results/` root, while the packaged server binds only to `127.0.0.1` with Gradio sharing disabled.

**Tech Stack:** Python 3.12, OpenCV, NumPy, Gradio, standard-library CSV/JSON/ZIP/file APIs, `unittest`, PyInstaller, GitHub Actions.

---

## File map

- Create `analysis_service.py`: validated analysis settings, inward crop geometry,
  publication palette, shared single-image pipeline, atomic per-image artifacts.
- Create `batch_service.py`: isolated per-image batch execution, summaries,
  failures, and complete batch ZIP creation.
- Create `web_controller.py`: stable values returned to Gradio for single and
  batch actions; no layout code.
- Replace `app.py`: English one-page Gradio application with single and batch
  tabs, shared sliders, previews, tables, and downloads.
- Create `web_launcher.py`: packaged localhost-only startup, browser opening,
  command-line smoke-test switches, and result-root resolution.
- Create `web_styles.css`: self-contained local visual styling; no CDN assets.
- Modify `app_standalone.py`: retain the desktop drag-and-drop entry point while
  delegating analysis and artifact generation to `analysis_service.py`.
- Create `tests/test_analysis_service.py`: inset, palette, metadata, and atomic
  single-image behavior.
- Create `tests/test_batch_service.py`: partial failure, duplicate names,
  summaries, and ZIP behavior.
- Create `tests/test_web_controller.py`: single and batch browser-facing return
  contracts and failure messages.
- Create `tests/test_web_launcher.py`: localhost, no-share, browser, and result
  path startup behavior.
- Modify `tests/test_standalone_pipeline.py`: desktop/shared-service parity and
  backward-compatibility assertions.
- Modify `requirements.txt`: pin the runtime dependency set needed by the
  packaged local site.
- Modify `.github/workflows/build.yml`: test, compile, package, smoke-test, ZIP,
  and release both Windows launchers.
- Modify `README.md`: offline website download, startup, analysis, batch,
  results, privacy, and troubleshooting instructions with no emoji.

## Task 1: Add validated settings and inward crop geometry

**Files:**
- Create: `analysis_service.py`
- Create: `tests/test_analysis_service.py`

- [ ] **Step 1: Write failing settings and geometry tests**

```python
# tests/test_analysis_service.py
import unittest

from analysis_service import AnalysisSettings, inset_bbox
from grid_detector import DetectionError


class AnalysisSettingsTests(unittest.TestCase):
    def test_defaults_match_approved_web_controls(self):
        settings = AnalysisSettings()
        self.assertEqual(5.0, settings.inset_percent)
        self.assertEqual(60.0, settings.no_clot_threshold)

    def test_settings_reject_values_outside_slider_ranges(self):
        for inset in (-0.1, 15.1):
            with self.subTest(inset=inset), self.assertRaises(ValueError):
                AnalysisSettings(inset_percent=inset).validate()
        for threshold in (-0.1, 255.1):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                AnalysisSettings(no_clot_threshold=threshold).validate()

    def test_inset_bbox_uses_shorter_side_and_preserves_half_open_box(self):
        self.assertEqual((15, 25, 195, 115), inset_bbox((10, 20, 200, 120), 5.0))

    def test_inset_bbox_rejects_an_analysis_inadequate_final_crop(self):
        with self.assertRaisesRegex(DetectionError, "too small after the inner inset"):
            inset_bbox((10, 10, 48, 48), 15.0, minimum_side=32)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m unittest tests.test_analysis_service.AnalysisSettingsTests -v
```

Expected: import error because `analysis_service.py` does not exist.

- [ ] **Step 3: Implement settings validation and inset geometry**

```python
# analysis_service.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from grid_detector import DetectionError

BBox: TypeAlias = tuple[int, int, int, int]
MIN_FINAL_CROP_SIDE = 32


@dataclass(frozen=True)
class AnalysisSettings:
    inset_percent: float = 5.0
    no_clot_threshold: float = 60.0
    results_root: Path | None = None

    def validate(self) -> "AnalysisSettings":
        if not 0.0 <= float(self.inset_percent) <= 15.0:
            raise ValueError("Inner crop inset must be between 0 and 15 percent.")
        if not 0.0 <= float(self.no_clot_threshold) <= 255.0:
            raise ValueError("No-clot threshold must be between 0 and 255.")
        return self


def inset_bbox(
    bbox: BBox,
    inset_percent: float,
    minimum_side: int = MIN_FINAL_CROP_SIDE,
) -> BBox:
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    inset = round(min(width, height) * float(inset_percent) / 100.0)
    final = (x1 + inset, y1 + inset, x2 - inset, y2 - inset)
    if final[2] - final[0] < minimum_side or final[3] - final[1] < minimum_side:
        raise DetectionError(
            "A detected cell is too small after the inner inset. "
            "Reduce the inset or use a higher-resolution image."
        )
    return final
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_analysis_service.AnalysisSettingsTests -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit the settings boundary**

```bash
git add analysis_service.py tests/test_analysis_service.py
git commit -m "feat: add validated inner crop settings"
```

## Task 2: Implement the fixed publication heatmap palette

**Files:**
- Modify: `analysis_service.py`
- Modify: `tests/test_analysis_service.py`

- [ ] **Step 1: Write failing threshold and interpolation tests**

```python
# append inside tests/test_analysis_service.py
from analysis_service import (
    DEEP_RED_RGB,
    LIGHT_RED_RGB,
    NO_CLOT_BLUE_RGB,
    heatmap_color_rgb,
)


class PublicationPaletteTests(unittest.TestCase):
    def test_threshold_and_lower_values_are_exactly_blue(self):
        for value in (0.0, 59.9, 60.0):
            self.assertEqual(NO_CLOT_BLUE_RGB, heatmap_color_rgb(value, 60.0))

    def test_first_clot_value_enters_only_the_red_ramp(self):
        color = heatmap_color_rgb(60.1, 60.0)
        self.assertNotEqual(NO_CLOT_BLUE_RGB, color)
        self.assertGreaterEqual(color[0], color[1])
        self.assertGreaterEqual(color[0], color[2])

    def test_red_ramp_has_fixed_cross_image_endpoints(self):
        near_threshold = heatmap_color_rgb(60.000001, 60.0)
        self.assertTrue(all(abs(a - b) <= 1 for a, b in zip(LIGHT_RED_RGB, near_threshold)))
        self.assertEqual(DEEP_RED_RGB, heatmap_color_rgb(255.0, 60.0))
        self.assertEqual(heatmap_color_rgb(140.0, 60.0), heatmap_color_rgb(140.0, 60.0))
```

- [ ] **Step 2: Run the palette tests and verify RED**

Run:

```bash
python -m unittest tests.test_analysis_service.PublicationPaletteTests -v
```

Expected: import error for the palette constants and function.

- [ ] **Step 3: Implement the approved blue/red mapping**

```python
# append to analysis_service.py
NO_CLOT_BLUE_RGB = (63, 120, 181)
LIGHT_RED_RGB = (246, 210, 207)
MEDIUM_RED_RGB = (212, 95, 98)
DEEP_RED_RGB = (126, 16, 36)
PALETTE_VERSION = "publication-blue-red-v1"


def _lerp_rgb(start: tuple[int, int, int], end: tuple[int, int, int], amount: float):
    return tuple(round(a + (b - a) * amount) for a, b in zip(start, end))


def heatmap_color_rgb(value: float, threshold: float) -> tuple[int, int, int]:
    if value <= threshold:
        return NO_CLOT_BLUE_RGB
    if threshold >= 255.0:
        return LIGHT_RED_RGB
    position = min(1.0, max(0.0, (value - threshold) / (255.0 - threshold)))
    if position <= 0.5:
        return _lerp_rgb(LIGHT_RED_RGB, MEDIUM_RED_RGB, position * 2.0)
    return _lerp_rgb(MEDIUM_RED_RGB, DEEP_RED_RGB, (position - 0.5) * 2.0)
```

- [ ] **Step 4: Run palette and settings tests**

Run:

```bash
python -m unittest tests.test_analysis_service -v
```

Expected: all tests in `test_analysis_service.py` pass.

- [ ] **Step 5: Commit the publication palette**

```bash
git add analysis_service.py tests/test_analysis_service.py
git commit -m "feat: add thresholded publication heatmap palette"
```

## Task 3: Build the shared single-image analysis service

**Files:**
- Modify: `analysis_service.py`
- Modify: `app_standalone.py:46-346`
- Modify: `tests/test_analysis_service.py`
- Modify: `tests/test_standalone_pipeline.py:131-234`

- [ ] **Step 1: Write a failing shared-service artifact test**

```python
# append inside tests/test_analysis_service.py
import csv
import json
import tempfile
from pathlib import Path

import cv2

from analysis_service import analyze_image
from tests.test_grid_detector import make_fixture


class SingleImageServiceTests(unittest.TestCase):
    def test_analysis_uses_inset_for_crops_measurements_and_metadata(self):
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
                detected = cell["detected_bbox"]
                final = cell["final_bbox"]
                self.assertGreater(final[0], detected[0])
                self.assertGreater(final[1], detected[1])
                self.assertLess(final[2], detected[2])
                self.assertLess(final[3], detected[3])

            metadata = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
            self.assertEqual(
                "ImageJ-equivalent inverted 8-bit grayscale mean",
                metadata["measurement_method"],
            )
            self.assertEqual(5.0, metadata["settings"]["inset_percent"])
            self.assertEqual(60.0, metadata["settings"]["no_clot_threshold"])
            self.assertEqual("publication-blue-red-v1", metadata["settings"]["palette_version"])

            with Path(result["csv_path"]).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(9, len(rows))
            self.assertIn("final_x1", rows[0])
            self.assertEqual("5.0", rows[0]["inset_percent"])
```

- [ ] **Step 2: Run the service test and verify RED**

Run:

```bash
python -m unittest tests.test_analysis_service.SingleImageServiceTests -v
```

Expected: import error because `analyze_image` is not defined.

- [ ] **Step 3: Move UI-independent helpers into `analysis_service.py`**

Move the existing Unicode-safe `load_image`, `_write_image`, grayscale
conversion, measurement, filename-key, atomic staging, CSV, and JSON behavior
from `app_standalone.py` into `analysis_service.py`. Preserve their tested
contracts. Add these public types and signature:

```python
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from grid_detector import GridDetection, detect_inner_squares

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def analyze_image(path: str | Path, settings: AnalysisSettings) -> dict[str, Any]:
    settings.validate()
    source = Path(path)
    if source.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError("Supported image types are PNG, JPG, JPEG, BMP, and TIFF.")
    image = load_image(source)
    if image is None:
        raise DetectionError(f"Could not read image: {source.name}")
    detection = detect_inner_squares(image)
    return _publish_analysis(source, image, detection, settings)
```

The cell loop inside `_publish_analysis` must be exactly one source of crop
geometry and measurement:

```python
cells = []
for detected_cell in detection.squares:
    detected_bbox = tuple(detected_cell.source_bbox)
    final_bbox = inset_bbox(detected_bbox, settings.inset_percent)
    x1, y1, x2, y2 = final_bbox
    crop = image[y1:y2, x1:x2].copy()
    inverted = 255 - to_8bit(crop)
    cell = {
        "idx": detected_cell.idx,
        "row": detected_cell.row,
        "col": detected_cell.col,
        "confidence": detected_cell.confidence,
        "recovered": detected_cell.recovered,
        "detected_bbox": list(detected_bbox),
        "final_bbox": list(final_bbox),
        "inset_percent": float(settings.inset_percent),
        **measure(inverted),
    }
    cells.append((cell, crop))
```

Write crops from these `(cell, crop)` pairs and reuse the same `cell` mappings
for heatmap, overlay, CSV, JSON, and the returned response. Resolve the durable
root as `Path(settings.results_root)` when supplied and `source.parent`
otherwise, preserving desktop output location compatibility. The returned
mapping must also include `image=source.name`, `grid_confidence`, `outer_quad`,
and the paths to every published artifact.

- [ ] **Step 4: Generate the fixed publication heatmap and dual-boundary overlay**

Replace the old per-image min/max heatmap normalization with:

```python
def heatmap_image(results: list[dict], threshold: float) -> np.ndarray:
    canvas = np.full((430, 384, 3), 248, dtype=np.uint8)
    for result in results:
        row, col = result["row"] - 1, result["col"] - 1
        rgb = heatmap_color_rgb(result["mean"], threshold)
        bgr = (rgb[2], rgb[1], rgb[0])
        x1, y1 = 8 + col * 126, 42 + row * 126
        cv2.rectangle(canvas, (x1, y1), (x1 + 118, y1 + 118), bgr, -1)
        cv2.rectangle(canvas, (x1, y1), (x1 + 118, y1 + 118), (255, 255, 255), 2)
        cv2.putText(canvas, f"#{result['idx']}", (x1 + 5, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(canvas, f"{result['mean']:.1f}", (x1 + 28, y1 + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    _draw_publication_legend(canvas, threshold)
    return canvas
```

The overlay must draw detected boxes dashed in cyan and final boxes solid in
red, both mapped to the source image, with row-major labels on final boxes.

Implement the legend directly rather than relying on image-relative min/max:

```python
def _draw_publication_legend(canvas: np.ndarray, threshold: float) -> None:
    cv2.putText(canvas, "No clot", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (70, 70, 70), 1)
    cv2.rectangle(canvas, (76, 7), (106, 20), NO_CLOT_BLUE_RGB[::-1], -1)
    cv2.putText(canvas, f"Threshold {threshold:.0f}", (116, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
    for offset in range(120):
        value = threshold + (255.0 - threshold) * offset / 119.0
        rgb = heatmap_color_rgb(value, threshold)
        canvas[405:418, 250 + offset] = rgb[::-1]
    cv2.putText(canvas, "More clot", (292, 399), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (70, 70, 70), 1)
```

- [ ] **Step 5: Make result publication atomic and include the complete ZIP**

Build all artifacts in a sibling staging directory, create the ZIP outside the
directory being zipped, then atomically replace the final output directory. The
ZIP must contain nine crops, overlay, heatmap, CSV, and JSON. On any exception,
remove only that known staging directory and leave any prior complete result
unchanged.

- [ ] **Step 6: Reduce the desktop entry point to a shared-service adapter**

Keep `process_image` backward compatible:

```python
# app_standalone.py
from analysis_service import AnalysisSettings, analyze_image, load_image


def process_image(
    path,
    show_windows=True,
    open_folder=True,
    inset_percent=5.0,
    no_clot_threshold=60.0,
    results_root=None,
):
    result = analyze_image(
        path,
        AnalysisSettings(inset_percent, no_clot_threshold, results_root),
    )
    if show_windows:
        show_results(
            load_image(result["overlay_path"]),
            load_image(result["heatmap_path"]),
            Path(path).name,
        )
    if open_folder:
        open_output_folder(result["output_dir"])
    return result
```

Keep console logging, the drag-and-drop CLI, OpenCV display, and folder opening
in `app_standalone.py`; remove duplicate measurement and artifact code.

- [ ] **Step 7: Add desktop parity assertions**

Extend `test_process_image_crops_detected_inner_squares_and_saves_outputs` to
assert that default desktop results contain `detected_bbox`, `final_bbox`,
`inset_percent == 5.0`, and `palette_version == publication-blue-red-v1`.

- [ ] **Step 8: Run shared and desktop tests**

Run:

```bash
python -m unittest tests.test_analysis_service tests.test_standalone_pipeline -v
```

Expected: all shared-service and desktop tests pass.

- [ ] **Step 9: Commit the shared pipeline**

```bash
git add analysis_service.py app_standalone.py tests/test_analysis_service.py tests/test_standalone_pipeline.py
git commit -m "refactor: share fixed-grid analysis pipeline"
```

## Task 4: Add resilient batch processing and archives

**Files:**
- Create: `batch_service.py`
- Create: `tests/test_batch_service.py`

- [ ] **Step 1: Write a failing mixed-success batch test**

```python
# tests/test_batch_service.py
import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from analysis_service import AnalysisSettings
from batch_service import analyze_batch
from tests.test_grid_detector import make_fixture


class BatchServiceTests(unittest.TestCase):
    def test_batch_continues_after_failure_and_archives_reports(self):
        good, _ = make_fixture(filled=(1, 5, 9))
        bad = np.full((900, 900, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            good_path, bad_path = root / "good.png", root / "bad.png"
            self.assertTrue(cv2.imwrite(str(good_path), good))
            self.assertTrue(cv2.imwrite(str(bad_path), bad))

            result = analyze_batch(
                [good_path, bad_path],
                AnalysisSettings(results_root=root / "results"),
            )

            self.assertEqual(1, result["success_count"])
            self.assertEqual(1, result["failure_count"])
            with Path(result["failures_csv"]).open(encoding="utf-8", newline="") as handle:
                failures = list(csv.DictReader(handle))
            self.assertEqual("bad.png", failures[0]["image"])
            with zipfile.ZipFile(result["zip_path"]) as archive:
                names = archive.namelist()
            self.assertTrue(any(name.endswith("batch-summary.csv") for name in names))
            self.assertTrue(any(name.endswith("failures.csv") for name in names))
            self.assertTrue(any(name.endswith("cell_09.png") for name in names))
```

- [ ] **Step 2: Run the batch test and verify RED**

Run:

```bash
python -m unittest tests.test_batch_service -v
```

Expected: import error because `batch_service.py` does not exist.

- [ ] **Step 3: Implement deterministic batch orchestration**

```python
# batch_service.py
from __future__ import annotations

import csv
import json
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from analysis_service import AnalysisSettings, analyze_image


Progress = Callable[[int, int, str], None]


def analyze_batch(
    paths: Iterable[str | Path],
    settings: AnalysisSettings,
    progress: Progress | None = None,
) -> dict:
    settings.validate()
    sources = [Path(path) for path in paths]
    if not sources:
        raise ValueError("Select at least one image for batch processing.")
    root = Path(settings.results_root or Path.cwd() / "results")
    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    batch_dir = root / batch_name
    batch_dir.mkdir(parents=True, exist_ok=False)
    successes, failures = [], []
    for index, source in enumerate(sources, start=1):
        if progress:
            progress(index, len(sources), source.name)
        per_image = AnalysisSettings(
            settings.inset_percent,
            settings.no_clot_threshold,
            batch_dir,
        )
        try:
            successes.append(analyze_image(source, per_image))
        except Exception as exception:
            failures.append({"image": source.name, "reason": str(exception)})
    return _publish_batch(batch_dir, settings, successes, failures)
```

Use one explicit publisher for the reports and ZIP:

```python
def _publish_batch(batch_dir, settings, successes, failures):
    summary_path = batch_dir / "batch-summary.csv"
    failures_path = batch_dir / "failures.csv"
    metadata_path = batch_dir / "batch-metadata.json"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "status", "cells", "result"])
        writer.writeheader()
        for result in successes:
            writer.writerow({
                "image": result["image"],
                "status": "Success",
                "cells": len(result["cells"]),
                "result": Path(result["output_dir"]).name,
            })
        for failure in failures:
            writer.writerow({"image": failure["image"], "status": "Failed", "cells": "", "result": failure["reason"]})
    with failures_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    metadata_path.write_text(json.dumps({
        "settings": {
            "inset_percent": settings.inset_percent,
            "no_clot_threshold": settings.no_clot_threshold,
        },
        "success_count": len(successes),
        "failure_count": len(failures),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    zip_path = batch_dir / "batch-results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(batch_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                archive.write(path, path.relative_to(batch_dir))
    return {
        "batch_dir": str(batch_dir),
        "success_count": len(successes),
        "failure_count": len(failures),
        "summary_csv": str(summary_path),
        "failures_csv": str(failures_path),
        "zip_path": str(zip_path),
        "successes": successes,
        "failures": failures,
    }
```

- [ ] **Step 4: Add duplicate and Unicode filename coverage**

Add a test that passes two sources named `sample.png` from different Unicode
parent directories and asserts their result directories differ and both appear
in the ZIP. Use the same stable filename-key helper as single-image analysis.

- [ ] **Step 5: Run batch and shared-service tests**

Run:

```bash
python -m unittest tests.test_batch_service tests.test_analysis_service -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit batch processing**

```bash
git add batch_service.py tests/test_batch_service.py
git commit -m "feat: add resilient batch analysis"
```

## Task 5: Add browser-facing controller contracts

**Files:**
- Create: `web_controller.py`
- Create: `tests/test_web_controller.py`

- [ ] **Step 1: Write failing single and batch controller tests**

```python
# tests/test_web_controller.py
import tempfile
import unittest
from pathlib import Path

import cv2

from tests.test_grid_detector import make_fixture
from web_controller import run_batch_analysis, run_single_analysis


class WebControllerTests(unittest.TestCase):
    def test_single_response_contains_nine_previews_and_downloads(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            response = run_single_analysis(source, 5.0, 60.0, root / "results")
            self.assertTrue(response.ok)
            self.assertEqual(9, len(response.crops))
            self.assertTrue(Path(response.csv_path).is_file())
            self.assertTrue(Path(response.zip_path).is_file())
            self.assertEqual("9 cells detected", response.status)

    def test_batch_response_exposes_successes_and_failures(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            response = run_batch_analysis(
                [source, root / "missing.png"], 5.0, 60.0, root / "results"
            )
            self.assertEqual(1, response.success_count)
            self.assertEqual(1, response.failure_count)
            self.assertTrue(Path(response.zip_path).is_file())
```

- [ ] **Step 2: Run controller tests and verify RED**

Run:

```bash
python -m unittest tests.test_web_controller -v
```

Expected: import error because `web_controller.py` does not exist.

- [ ] **Step 3: Implement typed controller responses**

```python
# web_controller.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys

from analysis_service import AnalysisSettings, analyze_image
from batch_service import analyze_batch


@dataclass
class SingleWebResponse:
    ok: bool
    status: str
    crops: list[str] = field(default_factory=list)
    overlay_path: str | None = None
    heatmap_path: str | None = None
    rows: list[list] = field(default_factory=list)
    csv_path: str | None = None
    zip_path: str | None = None
    output_dir: str | None = None


def run_single_analysis(path, inset, threshold, results_root) -> SingleWebResponse:
    try:
        result = analyze_image(
            path,
            AnalysisSettings(float(inset), float(threshold), Path(results_root)),
        )
    except Exception as exception:
        return SingleWebResponse(False, str(exception))
    return SingleWebResponse(
        True,
        "9 cells detected",
        [cell["crop_path"] for cell in result["cells"]],
        result["overlay_path"],
        result["heatmap_path"],
        _table_rows(result["cells"]),
        result["csv_path"],
        result["zip_path"],
        result["output_dir"],
    )
```

Add a `BatchWebResponse` dataclass and `run_batch_analysis` adapter that expose
counts, table rows, batch summary CSV, failure CSV, batch ZIP, and batch
directory. Controller functions return explicit failure responses instead of
raising into Gradio.

```python
def _table_rows(cells):
    return [[
        cell["idx"], cell["row"], cell["col"], cell["mean"],
        cell["confidence"], cell["recovered"],
    ] for cell in cells]


@dataclass
class BatchWebResponse:
    ok: bool
    status: str
    success_count: int = 0
    failure_count: int = 0
    rows: list[list] = field(default_factory=list)
    summary_csv: str | None = None
    failures_csv: str | None = None
    zip_path: str | None = None
    batch_dir: str | None = None


def run_batch_analysis(paths, inset, threshold, results_root) -> BatchWebResponse:
    try:
        result = analyze_batch(
            paths,
            AnalysisSettings(float(inset), float(threshold), Path(results_root)),
        )
    except Exception as exception:
        return BatchWebResponse(False, str(exception))
    rows = [
        [item["image"], 9, "Success", "", item["output_dir"]]
        for item in result["successes"]
    ] + [
        [item["image"], "", "Failed", item["reason"], ""]
        for item in result["failures"]
    ]
    return BatchWebResponse(
        True,
        f"{result['success_count']} succeeded, {result['failure_count']} failed",
        result["success_count"],
        result["failure_count"],
        rows,
        result["summary_csv"],
        result["failures_csv"],
        result["zip_path"],
        result["batch_dir"],
    )
```

- [ ] **Step 4: Add invalid slider and unreadable image tests**

Assert that invalid inset/threshold values and an unreadable path return
`ok == False`, contain the actionable service message, and expose no artifact
paths.

- [ ] **Step 5: Add a contained result-folder opener**

Implement and test this controller action:

```python
def open_result_folder(path: str, results_root: str | Path) -> str:
    root = Path(results_root).resolve()
    candidate = Path(path).resolve()
    if not candidate.is_dir() or not candidate.is_relative_to(root):
        return "Result folder is unavailable."
    if sys.platform == "win32":
        command = ["explorer", str(candidate)]
    elif sys.platform == "darwin":
        command = ["open", str(candidate)]
    else:
        command = ["xdg-open", str(candidate)]
    subprocess.Popen(command)
    return f"Opened result folder: {candidate.name}"
```

Patch `subprocess.Popen` in tests. Assert a directory inside `results_root`
opens and a sibling directory is rejected without invoking a process.

- [ ] **Step 6: Run controller, batch, and analysis tests**

Run:

```bash
python -m unittest tests.test_web_controller tests.test_batch_service tests.test_analysis_service -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit controller contracts**

```bash
git add web_controller.py tests/test_web_controller.py
git commit -m "feat: add offline web controller"
```

## Task 6: Build the English one-page Gradio interface

**Files:**
- Replace: `app.py`
- Create: `web_styles.css`
- Modify: `tests/test_web_controller.py`

- [ ] **Step 1: Write a failing application-construction smoke test**

```python
# append inside tests/test_web_controller.py
from app import create_app


class WebApplicationTests(unittest.TestCase):
    def test_create_app_builds_without_starting_a_server(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(Path(temp_dir) / "results")
        self.assertIsNotNone(app)
        config = app.get_config_file()
        serialized = str(config)
        for label in (
            "Single Image",
            "Batch Processing",
            "Inner crop inset",
            "No-clot threshold",
            "Analyze Image",
            "Analyze Batch",
        ):
            self.assertIn(label, serialized)
```

- [ ] **Step 2: Run the UI smoke test and verify RED**

Run:

```bash
python -m unittest tests.test_web_controller.WebApplicationTests -v
```

Expected: failure because the current `app.py` has no `create_app` and builds
the old variable-grid interface at import time.

- [ ] **Step 3: Replace the old hosted app with a factory**

```python
# app.py
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr

from web_controller import open_result_folder, run_batch_analysis, run_single_analysis


def create_app(results_root: str | Path) -> gr.Blocks:
    root = Path(results_root)
    css = Path(__file__).with_name("web_styles.css").read_text(encoding="utf-8")
    with gr.Blocks(
        title="Coagulation Analysis",
        css=css,
        analytics_enabled=False,
    ) as application:
        gr.Markdown("# Coagulation Analysis\nLocal and offline")
        with gr.Tabs():
            with gr.Tab("Single Image"):
                _build_single_tab(root)
            with gr.Tab("Batch Processing"):
                _build_batch_tab(root)
    return application


if __name__ == "__main__":
    create_app(Path.cwd() / "results").launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
    )
```

- [ ] **Step 4: Build the Single Image tab**

Add the adapter and builder below. It includes the input, sliders, nine-crop
gallery, overlay, heatmap, table, status, downloads, result path, and contained
folder-open action:

```python
def _single_values(path, inset, threshold, root):
    response = run_single_analysis(path, inset, threshold, root)
    return (
        response.crops,
        response.overlay_path,
        response.heatmap_path,
        response.rows,
        response.csv_path,
        response.zip_path,
        response.output_dir,
        response.status,
    )


def _build_single_tab(root: Path) -> None:
    with gr.Row():
        with gr.Column(scale=2):
            source = gr.File(label="Complete 3×3 fixture image", type="filepath")
            inset = gr.Slider(0, 15, value=5, step=0.5, label="Inner crop inset")
            threshold = gr.Slider(0, 255, value=60, step=1, label="No-clot threshold")
            analyze = gr.Button("Analyze Image", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=3):
            crops = gr.Gallery(label="Final inner crops", columns=3, rows=3)
            with gr.Row():
                overlay = gr.Image(label="Detected and final boundaries", type="filepath")
                heatmap = gr.Image(label="Publication heatmap", type="filepath")
    table = gr.Dataframe(
        headers=["Cell", "Row", "Column", "Mean", "Confidence", "Recovered"],
        interactive=False,
        label="Per-cell results",
    )
    with gr.Row():
        csv_file = gr.File(label="Download CSV")
        zip_file = gr.File(label="Download result ZIP")
    result_dir = gr.Textbox(label="Saved result folder", interactive=False)
    open_folder = gr.Button("Open result folder")
    analyze.click(
        lambda path, value, cutoff: _single_values(path, value, cutoff, root),
        [source, inset, threshold],
        [crops, overlay, heatmap, table, csv_file, zip_file, result_dir, status],
    )
    open_folder.click(
        lambda path: open_result_folder(path, root),
        result_dir,
        status,
    )
```

- [ ] **Step 5: Build the Batch Processing tab**

Use a separate component set and explicit adapter:

```python
def _batch_values(paths, inset, threshold, root):
    response = run_batch_analysis(paths or [], inset, threshold, root)
    return (
        response.rows,
        response.summary_csv,
        response.failures_csv,
        response.zip_path,
        response.batch_dir,
        response.status,
    )


def _build_batch_tab(root: Path) -> None:
    sources = gr.File(
        label="Complete 3×3 fixture images",
        file_count="multiple",
        type="filepath",
    )
    with gr.Row():
        inset = gr.Slider(0, 15, value=5, step=0.5, label="Inner crop inset")
        threshold = gr.Slider(0, 255, value=60, step=1, label="No-clot threshold")
    analyze = gr.Button("Analyze Batch", variant="primary")
    status = gr.Textbox(label="Batch status", interactive=False)
    table = gr.Dataframe(
        headers=["Image", "Cells", "Status", "Reason", "Result"],
        interactive=False,
        label="Batch results",
    )
    with gr.Row():
        summary = gr.File(label="Batch summary CSV")
        failures = gr.File(label="Failure report CSV")
        archive = gr.File(label="Download batch ZIP")
    batch_dir = gr.Textbox(label="Saved batch folder", interactive=False)
    open_folder = gr.Button("Open batch folder")
    analyze.click(
        lambda paths, value, cutoff: _batch_values(paths, value, cutoff, root),
        [sources, inset, threshold],
        [table, summary, failures, archive, batch_dir, status],
    )
    open_folder.click(lambda path: open_result_folder(path, root), batch_dir, status)
```

Do not share mutable Gradio component state between tabs; the visible settings
are passed to each callback.

- [ ] **Step 6: Add self-contained styling**

```css
/* web_styles.css */
:root {
  --navy: #182a45;
  --blue: #3f78b5;
  --red: #a42e3d;
  --surface: #ffffff;
  --background: #f5f7fa;
}
.gradio-container { max-width: 1440px !important; margin: 0 auto; }
.offline-status { color: #ffffff; background: var(--navy); border-radius: 999px; }
.primary { background: var(--red) !important; border-color: var(--red) !important; }
```

Keep all styles local. Do not reference external fonts, images, scripts, or
CDNs.

- [ ] **Step 7: Run UI and controller tests**

Run:

```bash
python -m unittest tests.test_web_controller -v
```

Expected: all tests pass and importing `app` does not start a server.

- [ ] **Step 8: Commit the web interface**

```bash
git add app.py web_styles.css tests/test_web_controller.py
git commit -m "feat: add offline analysis website"
```

## Task 7: Add the localhost-only Windows launcher

**Files:**
- Create: `web_launcher.py`
- Create: `tests/test_web_launcher.py`

- [ ] **Step 1: Write failing launch-configuration tests**

```python
# tests/test_web_launcher.py
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import web_launcher


class WebLauncherTests(unittest.TestCase):
    def test_launch_is_loopback_only_and_never_shared(self):
        fake_app = mock.Mock()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher, "create_app", return_value=fake_app
        ):
            web_launcher.launch_site(
                results_root=Path(temp_dir) / "results",
                port=7860,
                open_browser=False,
            )
        fake_app.launch.assert_called_once_with(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=False,
            show_error=True,
            allowed_paths=[str((Path(temp_dir) / "results").resolve())],
        )

    def test_default_results_directory_is_beside_packaged_launcher(self):
        with mock.patch.object(web_launcher.sys, "frozen", True, create=True), mock.patch.object(
            web_launcher.sys, "executable", "/opt/LabApp/StartWebsite.exe"
        ):
            self.assertEqual(
                Path("/opt/LabApp/results"),
                web_launcher.default_results_root(),
            )

    def test_automatic_port_is_available_on_loopback(self):
        port = web_launcher.available_loopback_port()
        self.assertGreater(port, 0)
        self.assertLessEqual(port, 65535)

    def test_main_reports_unwritable_results_root(self):
        with mock.patch.object(web_launcher, "launch_site", side_effect=OSError("read-only")), mock.patch.object(
            web_launcher, "_print_console"
        ) as print_console:
            exit_code = web_launcher.main(["--no-browser"])
        self.assertEqual(1, exit_code)
        print_console.assert_called_once_with("Website startup failed: read-only")
```

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
python -m unittest tests.test_web_launcher -v
```

Expected: import error because `web_launcher.py` does not exist.

- [ ] **Step 3: Implement startup and test-friendly CLI switches**

```python
# web_launcher.py
from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

from app import create_app


def default_results_root() -> Path:
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    return base / "results"


def available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _print_console(message: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None)
    text = str(message)
    if encoding:
        text = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(text)


def launch_site(results_root: Path, port: int | None, open_browser: bool) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    application = create_app(results_root)
    selected_port = port if port is not None else available_loopback_port()
    application.launch(
        server_name="127.0.0.1",
        server_port=selected_port,
        share=False,
        inbrowser=open_browser,
        show_error=True,
        allowed_paths=[str(results_root.resolve())],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Start the offline coagulation website.")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    arguments = parser.parse_args(argv)
    try:
        launch_site(arguments.results_root, arguments.port, not arguments.no_browser)
    except (OSError, RuntimeError) as exception:
        _print_console(f"Website startup failed: {exception}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify startup failures are contained**

Run the `test_main_reports_unwritable_results_root` test and confirm `main`
returns 1, emits one Unicode-safe message, and never allows the exception to
escape into the packaged launcher.

- [ ] **Step 5: Run launcher and web tests**

Run:

```bash
python -m unittest tests.test_web_launcher tests.test_web_controller -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the launcher**

```bash
git add web_launcher.py tests/test_web_launcher.py
git commit -m "feat: add offline website launcher"
```

## Task 8: Package and smoke-test the Windows website

**Files:**
- Modify: `requirements.txt`
- Modify: `.github/workflows/build.yml:1-43`
- Create: `tests/test_runtime_dependencies.py`

- [ ] **Step 1: Pin runtime dependencies and test their availability**

```text
# requirements.txt
gradio>=5.0,<7
opencv-python>=4.10,<5
numpy>=2.0,<3
```

```python
# tests/test_runtime_dependencies.py
import unittest


class RuntimeDependencyTests(unittest.TestCase):
    def test_web_runtime_imports(self):
        import cv2
        import gradio
        import numpy

        self.assertTrue(cv2.__version__)
        self.assertTrue(gradio.__version__)
        self.assertTrue(numpy.__version__)
```

- [ ] **Step 2: Run the dependency and full test suites**

Run:

```bash
python -m unittest tests.test_runtime_dependencies -v
python -m unittest discover -s tests -q
```

Expected: dependency test passes and the full suite reports zero failures.

- [ ] **Step 3: Compile every packaged module in CI**

Set the compile step to:

```yaml
- name: Compile Python modules
  run: >-
    python -m py_compile
    app.py app_standalone.py analysis_service.py batch_service.py
    grid_detector.py web_controller.py web_launcher.py
    research/__init__.py research/annotate_inner_squares.py
    research/evaluate_cropping.py
```

- [ ] **Step 4: Build both Windows entry points**

Add Windows-only steps:

```yaml
- name: Build Windows web launcher
  if: runner.os == 'Windows'
  run: pyinstaller --clean --noconfirm --onedir --collect-all gradio --name StartWebsite web_launcher.py

- name: Build Windows desktop launcher
  if: runner.os == 'Windows'
  run: pyinstaller --clean --noconfirm --onedir --name CoagulationAnalysis app_standalone.py
```

Keep the existing macOS desktop build as a separate conditional step.

- [ ] **Step 5: Smoke-test the packaged localhost website on Windows**

```yaml
- name: Smoke-test Windows website
  if: runner.os == 'Windows'
  shell: pwsh
  run: |
    $process = Start-Process -FilePath "dist/StartWebsite/StartWebsite.exe" `
      -ArgumentList "--port", "7860", "--no-browser", "--results-root", "$env:RUNNER_TEMP/results" `
      -PassThru
    try {
      $ready = $false
      for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
          $response = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7860/
          if ($response.StatusCode -eq 200 -and $response.Content -match "Coagulation Analysis") {
            $ready = $true
            break
          }
        } catch {}
        Start-Sleep -Seconds 1
      }
      if (-not $ready) { throw "Packaged website did not become ready on localhost." }
    } finally {
      Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
```

- [ ] **Step 6: Assemble the Windows release ZIP**

Create `release/CoagulationAnalysis-Windows/Web` from `dist/StartWebsite` and
`release/CoagulationAnalysis-Windows/Desktop` from
`dist/CoagulationAnalysis`. Copy `README.md` to the release root, then run:

```powershell
Compress-Archive -Path release/CoagulationAnalysis-Windows/* -DestinationPath CoagulationAnalysis-Windows.zip
```

Assert both `Web/StartWebsite.exe` and
`Desktop/CoagulationAnalysis.exe` exist before compression.

- [ ] **Step 7: Upload the verified ZIP to the latest release**

Keep `softprops/action-gh-release@v2`, upload
`CoagulationAnalysis-Windows.zip`, and update its release body to state that
`Web/StartWebsite.exe` is the recommended offline entry point.

- [ ] **Step 8: Run workflow syntax and local verification**

Run:

```bash
python -m unittest discover -s tests -q
python -m py_compile app.py app_standalone.py analysis_service.py batch_service.py grid_detector.py web_controller.py web_launcher.py
git diff --check
```

Expected: tests pass, compilation exits zero, and `git diff --check` prints
nothing.

- [ ] **Step 9: Commit packaging and CI**

```bash
git add requirements.txt .github/workflows/build.yml tests/test_runtime_dependencies.py
git commit -m "build: package offline website for Windows"
```

## Task 9: Document the website and preserve research provenance

**Files:**
- Modify: `README.md:76-145`
- Modify: `research/EXPERIMENTS.md:73-135`

- [ ] **Step 1: Rewrite the Windows quick start around the website**

Document this exact path after extraction:

```text
CoagulationAnalysis-Windows/
├── Web/
│   ├── StartWebsite.exe
│   └── _internal/
├── Desktop/
│   ├── CoagulationAnalysis.exe
│   └── _internal/
└── README.md
```

Explain double-click startup, localhost privacy, accepted images, 600x600
minimum, straight-on full-fixture photography, single and batch tabs, default
5% inset, default threshold 60, `results/`, batch failure reports, and closing
the launcher window. State that the whole extracted folder must remain intact.

- [ ] **Step 2: Document heatmap and scientific interpretation limits**

State plainly that blue means measurement at or below the selected threshold,
red means above it, color is not a diagnosis, and the chosen inset and threshold
are recorded for reproducibility. Update `research/EXPERIMENTS.md` so manual and
synthetic evaluations state whether metrics use the detected or final inset
boxes and require the setting values in reported experiments.

- [ ] **Step 3: Verify documentation contains no emoji and no stale web claims**

Run:

```bash
rg -n --pcre2 '[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}]' README.md research/EXPERIMENTS.md
rg -n 'Hugging Face|public website|share=True|0\.0\.0\.0' README.md app.py web_launcher.py
```

Expected: both commands print nothing.

- [ ] **Step 4: Run provenance and full tests**

Run:

```bash
python -m unittest tests.test_analysis_service -v
python -m unittest discover -s tests -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md research/EXPERIMENTS.md
git commit -m "docs: explain offline website workflow"
```

## Task 10: Perform physical-sample and release acceptance

**Files:**
- Create locally but do not commit: `acceptance/<run-id>/`
- Modify only if an acceptance failure requires a tested fix: the owning
  service, controller, launcher, or workflow file and its matching test.

- [ ] **Step 1: Run the complete local verification gate**

Run:

```bash
python -m unittest discover -s tests -q
python -m py_compile app.py app_standalone.py analysis_service.py batch_service.py grid_detector.py web_controller.py web_launcher.py
git diff --check
git status --short
```

Expected: all tests pass, compilation and diff checks exit zero, and status
contains only intentional tracked changes.

- [ ] **Step 2: Analyze the supplied straight-on fixture in single mode**

Resolve the exact original straight-on fixture photograph supplied by the user.
If the temporary attachment no longer exists, stop and request that photograph
again; do not substitute the CAD screenshot, a synthetic fixture, or a different
sample for physical acceptance.

Run a script that calls `run_single_analysis` with inset 5, threshold 60, and a
temporary acceptance results root. Assert `ok`, nine crops, nine table rows,
and complete CSV, JSON, overlay, heatmap, and ZIP paths.

- [ ] **Step 3: Verify final geometry excludes every detected boundary**

Read JSON and assert for every cell:

```python
detected = cell["detected_bbox"]
final = cell["final_bbox"]
assert final[0] > detected[0]
assert final[1] > detected[1]
assert final[2] < detected[2]
assert final[3] < detected[3]
```

Visually inspect the overlay and all nine crops. Record failure if any black
fixture border remains at the 5% default; do not increase the default without a
new regression test and explicit design review.

- [ ] **Step 4: Verify heatmap category colors and fixed mapping**

Read each measurement and sample the corresponding heatmap cell center. Assert
measurements at or below 60 are exactly `#3F78B5` and measurements above 60 are
within the approved red ramp. Analyze the same source beside another image and
assert an equal measurement maps to the same RGB value.

- [ ] **Step 5: Verify batch parity and partial failure**

Run a batch containing the physical image, the same image under a Unicode name,
and an invalid fixture. Assert the two valid results have the same final boxes
and measurements as single mode, the invalid fixture appears in `failures.csv`,
and `batch-results.zip` contains both successful result directories and both
reports.

- [ ] **Step 6: Push the branch and open a ready pull request**

Use a PR title without assistant attribution. The body must summarize the local
website, inset rule, heatmap rule, batch behavior, privacy boundary, and exact
test commands.

- [ ] **Step 7: Wait for Windows Actions and inspect every job**

Use:

```bash
gh pr checks --watch
```

Expected: test, compile, Windows PyInstaller, packaged localhost smoke test,
ZIP assembly, and release upload all succeed. If a check fails, use the
`github:gh-fix-ci` and `systematic-debugging` skills before changing code.

- [ ] **Step 8: Verify the published Windows ZIP**

Download the newly created asset, run `unzip -t`, calculate SHA-256, and list
the archive to confirm both entry points:

```bash
unzip -t CoagulationAnalysis-Windows.zip
shasum -a 256 CoagulationAnalysis-Windows.zip
unzip -l CoagulationAnalysis-Windows.zip | rg 'Web/StartWebsite\.exe|Desktop/CoagulationAnalysis\.exe'
```

Expected: no ZIP errors and exactly one web and one desktop executable path.

- [ ] **Step 9: Report the verified delivery**

Provide the direct release asset link, local downloaded ZIP link, byte size,
SHA-256, successful Actions run link, and concise Windows startup instructions.
