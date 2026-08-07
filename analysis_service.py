"""UI-independent single-image analysis and publication pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np

from grid_detector import DetectionError, GridDetection, detect_inner_squares


BBox: TypeAlias = tuple[int, int, int, int]
MIN_FINAL_CROP_SIDE = 32
NO_CLOT_BLUE_RGB = (63, 120, 181)
LIGHT_RED_RGB = (246, 210, 207)
MEDIUM_RED_RGB = (212, 95, 98)
DEEP_RED_RGB = (126, 16, 36)
PALETTE_VERSION = "publication-blue-red-v1"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MEASUREMENT_METHOD = "ImageJ-equivalent inverted 8-bit grayscale mean"


@dataclass(frozen=True)
class AnalysisSettings:
    inset_percent: float = 5.0
    no_clot_threshold: float = 60.0
    results_root: Path | None = None

    def validate(self) -> AnalysisSettings:
        if not 0.0 <= self.inset_percent <= 15.0:
            raise ValueError("inset_percent must be between 0 and 15.")
        if not 0.0 <= self.no_clot_threshold <= 255.0:
            raise ValueError("no_clot_threshold must be between 0 and 255.")
        return self


def inset_bbox(
    bbox: BBox,
    inset_percent: float,
    minimum_side: int = MIN_FINAL_CROP_SIDE,
) -> BBox:
    """Return a uniformly inset half-open bounding box."""
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    inset = round(min(width, height) * float(inset_percent) / 100)
    inner_bbox = (left + inset, top + inset, right - inset, bottom - inset)
    inner_width = inner_bbox[2] - inner_bbox[0]
    inner_height = inner_bbox[3] - inner_bbox[1]
    if inner_width < minimum_side or inner_height < minimum_side:
        raise DetectionError(
            "A detected cell is too small after the inner inset. Reduce the inset "
            "or use a higher-resolution image."
        )
    return inner_bbox


def _lerp_rgb(
    start: tuple[int, int, int], end: tuple[int, int, int], amount: float
) -> tuple[int, int, int]:
    """Return the rounded per-channel interpolation between two RGB colors."""
    return tuple(
        round(start_channel + (end_channel - start_channel) * amount)
        for start_channel, end_channel in zip(start, end)
    )


def heatmap_color_rgb(value: float, threshold: float) -> tuple[int, int, int]:
    """Map a clot intensity to the publication blue-to-red palette."""
    if value <= threshold:
        return NO_CLOT_BLUE_RGB
    if threshold >= 255:
        return LIGHT_RED_RGB

    position = max(0.0, min(1.0, (value - threshold) / (255 - threshold)))
    if position <= 0.5:
        return _lerp_rgb(LIGHT_RED_RGB, MEDIUM_RED_RGB, position * 2)
    return _lerp_rgb(MEDIUM_RED_RGB, DEEP_RED_RGB, (position - 0.5) * 2)


def load_image(path: str | Path) -> np.ndarray | None:
    """Load an image while supporting non-ASCII paths on Windows."""
    path_string = os.fspath(path)
    image = cv2.imread(path_string)
    if image is None:
        try:
            data = np.fromfile(path_string, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except (OSError, ValueError, cv2.error):
            return None
    return image


def _write_image(path: str | Path, image: np.ndarray) -> bool:
    """Write an image through encoded bytes for Unicode-safe Windows paths."""
    destination = Path(path)
    success, encoded = cv2.imencode(destination.suffix, image)
    if not success:
        return False
    try:
        encoded.tofile(os.fspath(destination))
    except OSError:
        return False
    return destination.is_file() and destination.stat().st_size > 0


def to_8bit(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR pixels to ImageJ-equivalent 8-bit grayscale."""
    blue = bgr[:, :, 0].astype(np.float32)
    green = bgr[:, :, 1].astype(np.float32)
    red = bgr[:, :, 2].astype(np.float32)
    return np.clip(0.114 * blue + 0.587 * green + 0.299 * red, 0, 255).astype(np.uint8)


def measure(inverted: np.ndarray) -> dict[str, float | int]:
    """Return the established measurement fields for an inverted crop."""
    return {
        "mean": round(float(np.mean(inverted)), 2),
        "median": round(float(np.median(inverted)), 2),
        "std": round(float(np.std(inverted)), 2),
        "min": int(np.min(inverted)),
        "max": int(np.max(inverted)),
        "int_den": round(float(np.sum(inverted)), 2),
        "area_px": int(inverted.size),
    }


def _artifact_key(filename: str) -> str:
    stem, extension = os.path.splitext(filename)
    readable = "_".join(part for part in (stem, extension.lstrip(".")) if part)
    readable = re.sub(r"[^\w-]+", "_", readable, flags=re.UNICODE).strip("_")
    readable = readable[:80] or "image"
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{digest}"


def _draw_publication_legend(canvas: np.ndarray, threshold: float) -> None:
    cv2.putText(
        canvas,
        "No clot",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (70, 70, 70),
        1,
    )
    cv2.rectangle(canvas, (76, 7), (106, 20), NO_CLOT_BLUE_RGB[::-1], -1)
    cv2.putText(
        canvas,
        f"Threshold {threshold:.0f}",
        (116, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (70, 70, 70),
        1,
    )
    for offset in range(120):
        value = threshold + (255.0 - threshold) * offset / 119.0
        rgb = heatmap_color_rgb(value, threshold)
        canvas[405:418, 250 + offset] = rgb[::-1]
    cv2.putText(
        canvas,
        "More clot",
        (292, 399),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (70, 70, 70),
        1,
    )


def heatmap_image(results: list[dict[str, Any]], threshold: float) -> np.ndarray:
    """Render a fixed-scale publication heatmap for the nine cells."""
    canvas = np.full((430, 384, 3), 248, dtype=np.uint8)
    for result in results:
        row, col = result["row"] - 1, result["col"] - 1
        rgb = heatmap_color_rgb(result["mean"], threshold)
        bgr = (rgb[2], rgb[1], rgb[0])
        x1, y1 = 8 + col * 126, 42 + row * 126
        cv2.rectangle(canvas, (x1, y1), (x1 + 118, y1 + 118), bgr, -1)
        cv2.rectangle(canvas, (x1, y1), (x1 + 118, y1 + 118), (255, 255, 255), 2)
        cv2.putText(
            canvas,
            f"#{result['idx']}",
            (x1 + 5, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            canvas,
            f"{result['mean']:.1f}",
            (x1 + 28, y1 + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
    _draw_publication_legend(canvas, threshold)
    return canvas


def _dashed_rectangle(
    image: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 10,
) -> None:
    left, top, right, bottom = (int(value) for value in bbox)
    for start in range(left, right, dash_length * 2):
        cv2.line(
            image,
            (start, top),
            (min(start + dash_length, right), top),
            color,
            thickness,
        )
        cv2.line(
            image,
            (start, bottom),
            (min(start + dash_length, right), bottom),
            color,
            thickness,
        )
    for start in range(top, bottom, dash_length * 2):
        cv2.line(
            image,
            (left, start),
            (left, min(start + dash_length, bottom)),
            color,
            thickness,
        )
        cv2.line(
            image,
            (right, start),
            (right, min(start + dash_length, bottom)),
            color,
            thickness,
        )


def draw_detection_overlay(
    image: np.ndarray,
    cells: list[dict[str, Any]],
) -> np.ndarray:
    """Draw detected and final crop boundaries in source-image coordinates."""
    overlay = image.copy()
    for cell in cells:
        _dashed_rectangle(overlay, cell["detected_bbox"], (255, 255, 0))
        x1, y1, x2, y2 = (int(value) for value in cell["final_bbox"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"#{cell['idx']}",
            (x1 + 4, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    return overlay


def _csv_header() -> list[str]:
    return [
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
        "detected_x1",
        "detected_y1",
        "detected_x2",
        "detected_y2",
        "final_x1",
        "final_y1",
        "final_x2",
        "final_y2",
        "inset_percent",
    ]


def save_results(
    out_dir: str | Path,
    artifact_key: str,
    original_filename: str,
    results: list[dict[str, Any]],
    settings: AnalysisSettings,
    detection: GridDetection,
) -> tuple[Path, Path]:
    """Write the CSV and JSON audit artifacts into a staging directory."""
    destination = Path(out_dir)
    csv_path = destination / f"{artifact_key}_results.csv"
    json_path = destination / f"{artifact_key}_results.json"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(_csv_header())
        for result in results:
            writer.writerow(
                [
                    result["idx"],
                    result["row"],
                    result["col"],
                    result["mean"],
                    result["median"],
                    result["std"],
                    result["min"],
                    result["max"],
                    result["int_den"],
                    result["area_px"],
                    result["confidence"],
                    result["recovered"],
                    *result["source_bbox"],
                    *[
                        coordinate
                        for point in result["crop_quad"]
                        for coordinate in point
                    ],
                    *result["detected_bbox"],
                    *result["final_bbox"],
                    result["inset_percent"],
                ]
            )

    metadata = {
        "image": original_filename,
        "grid": "3x3",
        "grid_confidence": float(detection.confidence),
        "outer_quad": np.rint(detection.outer_quad).astype(int).tolist(),
        "measurement_method": MEASUREMENT_METHOD,
        "palette_version": PALETTE_VERSION,
        "settings": {
            "inset_percent": float(settings.inset_percent),
            "no_clot_threshold": float(settings.no_clot_threshold),
            "palette_version": PALETTE_VERSION,
        },
        "cells": results,
    }
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(metadata, json_file, indent=2)
    return csv_path, json_path


def _create_zip(source_dir: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for artifact in sorted(source_dir.iterdir(), key=lambda path: path.name):
            if artifact.is_file():
                archive.write(artifact, artifact.name)


def _remove_known_directory(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)


def _remove_known_file(path: Path) -> None:
    if path.is_file():
        path.unlink()


def _atomic_publish(
    staging_dir: Path,
    staging_zip: Path,
    final_dir: Path,
    final_zip: Path,
) -> None:
    token = uuid.uuid4().hex
    backup_dir = final_dir.with_name(f".{final_dir.name}.previous-{token}")
    backup_zip = final_zip.with_name(f".{final_zip.name}.previous-{token}")
    backed_up_dir = False
    backed_up_zip = False
    published_dir = False
    published_zip = False
    try:
        if final_dir.exists():
            os.replace(final_dir, backup_dir)
            backed_up_dir = True
        if final_zip.exists():
            os.replace(final_zip, backup_zip)
            backed_up_zip = True
        os.replace(staging_dir, final_dir)
        published_dir = True
        os.replace(staging_zip, final_zip)
        published_zip = True
    except Exception:
        if published_zip:
            _remove_known_file(final_zip)
        if published_dir:
            _remove_known_directory(final_dir)
        if backed_up_dir:
            os.replace(backup_dir, final_dir)
        if backed_up_zip:
            os.replace(backup_zip, final_zip)
        raise
    else:
        if backed_up_dir:
            _remove_known_directory(backup_dir)
        if backed_up_zip:
            _remove_known_file(backup_zip)


def _publish_analysis(
    source: Path,
    image: np.ndarray,
    detection: GridDetection,
    settings: AnalysisSettings,
) -> dict[str, Any]:
    results_root = (
        Path(settings.results_root)
        if settings.results_root is not None
        else source.parent
    )
    results_root.mkdir(parents=True, exist_ok=True)
    artifact_key = _artifact_key(source.name)
    final_dir = results_root / f"{artifact_key}_analysis"
    final_zip = results_root / f"{artifact_key}_analysis.zip"
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{artifact_key}_staging_", dir=results_root)
    )
    staging_zip = staging_dir.with_suffix(".zip")

    try:
        cell_crops: list[tuple[dict[str, Any], np.ndarray]] = []
        for detected_cell in detection.squares:
            detected_bbox = tuple(detected_cell.source_bbox)
            final_bbox = inset_bbox(detected_bbox, settings.inset_percent)
            x1, y1, x2, y2 = final_bbox
            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                raise DetectionError(
                    f"Detected crop #{detected_cell.idx} is empty. Retake the photo "
                    "with the full grid in frame."
                )
            inverted = 255 - to_8bit(crop)
            cell = {
                "idx": detected_cell.idx,
                "row": detected_cell.row,
                "col": detected_cell.col,
                "confidence": float(detected_cell.confidence),
                "recovered": bool(detected_cell.recovered),
                "detected_bbox": list(detected_bbox),
                "final_bbox": list(final_bbox),
                "inset_percent": float(settings.inset_percent),
                **measure(inverted),
            }
            # Keep the desktop CSV/JSON audit aliases for existing consumers.
            cell["source_bbox"] = cell["detected_bbox"]
            cell["crop_quad"] = np.rint(detected_cell.source_quad).astype(int).tolist()
            cell_crops.append((cell, crop))

        cells = [cell for cell, _crop in cell_crops]
        crop_names: list[str] = []
        for cell, crop in cell_crops:
            crop_name = f"cell_{cell['idx']:02d}.png"
            crop_path = staging_dir / crop_name
            if not _write_image(crop_path, crop):
                raise OSError(f"Could not write crop: {crop_path}")
            crop_names.append(crop_name)

        overlay = draw_detection_overlay(image, cells)
        overlay_name = f"{artifact_key}_grid_overlay.png"
        if not _write_image(staging_dir / overlay_name, overlay):
            raise OSError(f"Could not write overlay: {staging_dir / overlay_name}")

        heatmap = heatmap_image(cells, settings.no_clot_threshold)
        heatmap_name = f"{artifact_key}_heatmap.png"
        if not _write_image(staging_dir / heatmap_name, heatmap):
            raise OSError(f"Could not write heatmap: {staging_dir / heatmap_name}")

        csv_path, json_path = save_results(
            staging_dir,
            artifact_key,
            source.name,
            cells,
            settings,
            detection,
        )
        _create_zip(staging_dir, staging_zip)
        _atomic_publish(staging_dir, staging_zip, final_dir, final_zip)
    except Exception:
        _remove_known_directory(staging_dir)
        _remove_known_file(staging_zip)
        raise

    outer_quad = np.rint(detection.outer_quad).astype(int).tolist()
    return {
        "image": source.name,
        "grid_confidence": float(detection.confidence),
        "outer_quad": outer_quad,
        "cells": cells,
        "palette_version": PALETTE_VERSION,
        "output_dir": str(final_dir),
        "crop_paths": [str(final_dir / name) for name in crop_names],
        "overlay_path": str(final_dir / overlay_name),
        "heatmap_path": str(final_dir / heatmap_name),
        "csv_path": str(final_dir / csv_path.name),
        "json_path": str(final_dir / json_path.name),
        "zip_path": str(final_zip),
    }


def analyze_image(
    path: str | Path,
    settings: AnalysisSettings,
) -> dict[str, Any]:
    """Analyze one supported image and atomically publish all artifacts."""
    settings.validate()
    source = Path(path)
    if source.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError("Supported image types are PNG, JPG, JPEG, BMP, and TIFF.")
    image = load_image(source)
    if image is None:
        raise DetectionError(f"Could not read image: {source.name}")
    detection = detect_inner_squares(image)
    return _publish_analysis(source, image, detection, settings)
