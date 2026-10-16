"""UI-independent single-image analysis and publication pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np

from grid_detector import DetectionError, GridDetection, detect_inner_squares


BBox: TypeAlias = tuple[int, int, int, int]
CellResult: TypeAlias = dict[str, Any]
MIN_FINAL_CROP_SIDE = 32
NO_CLOT_BLUE_RGB = (63, 120, 181)
LIGHT_RED_RGB = (246, 210, 207)
MEDIUM_RED_RGB = (212, 95, 98)
DEEP_RED_RGB = (126, 16, 36)
PALETTE_VERSION = "publication-blue-red-v1"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MEASUREMENT_METHOD = "ImageJ-equivalent inverted 8-bit grayscale mean"
TRANSACTION_MARKER_NAME = ".analysis-transaction.json"
TRANSACTION_KIND = "coagulation-analysis-result"
TRANSACTION_VERSION = 1
_PUBLICATION_LOCKS: dict[Path, threading.Lock] = {}
_PUBLICATION_LOCKS_GUARD = threading.Lock()


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


def _artifact_key(filename: str, source_path: str | Path | None = None) -> str:
    stem, extension = os.path.splitext(filename)
    readable = "_".join(part for part in (stem, extension.lstrip(".")) if part)
    readable = re.sub(r"[^\w-]+", "_", readable, flags=re.UNICODE).strip("_")
    readable = readable[:80] or "image"
    if source_path is None:
        digest_input = filename
    else:
        identity = os.path.normcase(os.fspath(Path(source_path).resolve()))
        digest_input = f"{filename}\0{identity}"
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{digest}"


def _publication_lock(target: Path) -> threading.Lock:
    with _PUBLICATION_LOCKS_GUARD:
        return _PUBLICATION_LOCKS.setdefault(target, threading.Lock())


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
        canvas[7:20, 250 + offset] = rgb[::-1]
    cv2.putText(
        canvas,
        "More clot",
        (292, 37),
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
    right -= 1
    bottom -= 1
    for start in range(left, right + 1, dash_length * 2):
        cv2.line(
            image,
            (start, top),
            (min(start + dash_length - 1, right), top),
            color,
            thickness,
        )
        cv2.line(
            image,
            (start, bottom),
            (min(start + dash_length - 1, right), bottom),
            color,
            thickness,
        )
    for start in range(top, bottom + 1, dash_length * 2):
        cv2.line(
            image,
            (left, start),
            (left, min(start + dash_length - 1, bottom)),
            color,
            thickness,
        )
        cv2.line(
            image,
            (right, start),
            (right, min(start + dash_length - 1, bottom)),
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
        cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
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
        "crop_file",
    ]


def _durable_cell(cell: CellResult) -> CellResult:
    durable = {key: value for key, value in cell.items() if key != "crop_path"}
    durable["crop_file"] = Path(cell["crop_path"]).name
    return durable


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
    durable_results = [_durable_cell(result) for result in results]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(_csv_header())
        for result in durable_results:
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
                    result["crop_file"],
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
        "cells": durable_results,
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


def _transaction_prefix(final_dir: Path) -> str:
    return f".{final_dir.name}.previous-"


def _transaction_manifest(final_dir: Path) -> dict[str, str | int]:
    return {
        "kind": TRANSACTION_KIND,
        "version": TRANSACTION_VERSION,
        "target": os.fspath(final_dir.resolve()),
    }


def _create_transaction_root(final_dir: Path) -> Path:
    transaction = final_dir.parent / (
        f"{_transaction_prefix(final_dir)}{uuid.uuid4().hex}"
    )
    transaction.mkdir()
    marker = transaction / TRANSACTION_MARKER_NAME
    try:
        with marker.open("x", encoding="utf-8") as marker_file:
            json.dump(_transaction_manifest(final_dir), marker_file)
            marker_file.flush()
            os.fsync(marker_file.fileno())
    except Exception:
        try:
            marker.unlink(missing_ok=True)
            transaction.rmdir()
        except OSError:
            pass
        raise
    return transaction


def _is_owned_transaction(transaction: Path, final_dir: Path) -> bool:
    if transaction.parent.resolve() != final_dir.parent.resolve():
        return False
    if transaction.is_symlink() or not transaction.is_dir():
        return False
    name_pattern = re.compile(
        rf"{re.escape(_transaction_prefix(final_dir))}[0-9a-f]{{32}}"
    )
    if name_pattern.fullmatch(transaction.name) is None:
        return False
    marker = transaction / TRANSACTION_MARKER_NAME
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        if {child.name for child in transaction.iterdir()} - {
            TRANSACTION_MARKER_NAME,
            "output",
        }:
            return False
    except OSError:
        return False
    try:
        with marker.open(encoding="utf-8") as marker_file:
            manifest = json.load(marker_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return manifest == _transaction_manifest(final_dir)


def _remove_owned_transaction(transaction: Path, final_dir: Path) -> None:
    if _is_owned_transaction(transaction, final_dir):
        _remove_known_directory(transaction)


def _recover_previous_bundle(final_dir: Path) -> None:
    """Recover or discard only transaction bundles belonging to this target."""
    transactions = sorted(
        (
            path
            for path in final_dir.parent.iterdir()
            if _is_owned_transaction(path, final_dir)
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not transactions:
        return

    if not final_dir.exists():
        for transaction in transactions:
            previous_bundle = transaction / "output"
            if previous_bundle.is_dir() and not previous_bundle.is_symlink():
                os.replace(previous_bundle, final_dir)
                try:
                    _remove_owned_transaction(transaction, final_dir)
                except OSError:
                    pass
                break

    if final_dir.exists():
        for transaction in transactions:
            try:
                _remove_owned_transaction(transaction, final_dir)
            except OSError:
                pass


def _atomic_publish_bundle(staging_dir: Path, final_dir: Path) -> None:
    """Commit a complete immutable result bundle with one publication rename."""
    backup_root: Path | None = None
    backup_bundle: Path | None = None
    backup_root_created = False
    backed_up_bundle = False
    try:
        if final_dir.exists():
            backup_root = _create_transaction_root(final_dir)
            backup_bundle = backup_root / "output"
            backup_root_created = True
            os.replace(final_dir, backup_bundle)
            backed_up_bundle = True
        os.replace(staging_dir, final_dir)
    except Exception as original_exception:
        if backed_up_bundle and backup_bundle is not None:
            try:
                os.replace(backup_bundle, final_dir)
            except Exception as rollback_exception:
                original_exception.add_note(
                    f"Could not restore the previous result bundle: {rollback_exception}"
                )
        if (
            backup_root_created
            and backup_root is not None
            and backup_root.exists()
            and (backup_bundle is None or not backup_bundle.exists())
        ):
            try:
                _remove_owned_transaction(backup_root, final_dir)
            except OSError as cleanup_exception:
                original_exception.add_note(
                    f"Could not remove transaction directory: {cleanup_exception}"
                )
        raise

    # The single rename above publishes the directory and its ZIP together.
    # Deleting the prior bundle is an irrevocable post-commit cleanup, so a
    # partial cleanup failure must leave the new complete bundle authoritative.
    if backup_root_created and backup_root is not None:
        try:
            _remove_owned_transaction(backup_root, final_dir)
        except Exception:
            pass


def _publish_analysis(
    source: Path,
    image: np.ndarray,
    detection: GridDetection,
    settings: AnalysisSettings,
) -> dict[str, Any]:
    results_root = (
        Path(settings.results_root).expanduser().resolve()
        if settings.results_root is not None
        else source.parent
    )
    results_root.mkdir(parents=True, exist_ok=True)
    artifact_key = _artifact_key(source.name, source)
    final_dir = results_root / f"{artifact_key}_analysis"
    with _publication_lock(final_dir):
        _recover_previous_bundle(final_dir)
        return _publish_analysis_locked(
            source,
            image,
            detection,
            settings,
            results_root,
            artifact_key,
            final_dir,
        )


def _publish_analysis_locked(
    source: Path,
    image: np.ndarray,
    detection: GridDetection,
    settings: AnalysisSettings,
    results_root: Path,
    artifact_key: str,
    final_dir: Path,
) -> dict[str, Any]:
    zip_name = f"{artifact_key}_analysis.zip"
    final_zip = final_dir / zip_name
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{artifact_key}_staging_", dir=results_root)
    )
    staging_zip = staging_dir.with_suffix(".zip")

    try:
        cell_crops: list[tuple[CellResult, np.ndarray]] = []
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
            crop_name = f"cell_{detected_cell.idx:02d}.png"
            cell: CellResult = {
                "idx": detected_cell.idx,
                "row": detected_cell.row,
                "col": detected_cell.col,
                "confidence": float(detected_cell.confidence),
                "recovered": bool(detected_cell.recovered),
                "detected_bbox": list(detected_bbox),
                "final_bbox": list(final_bbox),
                "inset_percent": float(settings.inset_percent),
                "crop_path": str(final_dir / crop_name),
                **measure(inverted),
            }
            # Keep the desktop CSV/JSON audit aliases for existing consumers.
            cell["source_bbox"] = cell["detected_bbox"]
            cell["crop_quad"] = np.rint(detected_cell.source_quad).astype(int).tolist()
            cell_crops.append((cell, crop))

        cells = [cell for cell, _crop in cell_crops]
        crop_names: list[str] = []
        for cell, crop in cell_crops:
            crop_name = Path(cell["crop_path"]).name
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
        os.replace(staging_zip, staging_dir / zip_name)
        _atomic_publish_bundle(staging_dir, final_dir)
    except Exception as original_exception:
        for label, path, cleanup in (
            ("staging directory", staging_dir, _remove_known_directory),
            ("staging ZIP", staging_zip, _remove_known_file),
        ):
            try:
                cleanup(path)
            except Exception as cleanup_exception:
                original_exception.add_note(
                    f"Could not remove {label} {path}: {cleanup_exception}"
                )
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
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError("Supported image types are PNG, JPG, JPEG, BMP, and TIFF.")
    image = load_image(source)
    if image is None:
        raise DetectionError(f"Could not read image: {source.name}")
    detection = detect_inner_squares(image)
    return _publish_analysis(source, image, detection, settings)
