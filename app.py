#!/usr/bin/env python3
"""Fixed 3x3 coagulation analysis web app for Hugging Face Spaces."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
import zipfile

import cv2
import gradio as gr
import numpy as np

from grid_detector import DetectionError, GridDetection, detect_inner_squares


TABLE_HEADERS = [
    "Cell", "Row", "Col", "Mean", "Median", "Std", "Min", "Max",
    "IntDen", "Area(px)", "Confidence", "Recovered",
    "BBox X1", "BBox Y1", "BBox X2", "BBox Y2",
    "Quad 1 X", "Quad 1 Y", "Quad 2 X", "Quad 2 Y",
    "Quad 3 X", "Quad 3 Y", "Quad 4 X", "Quad 4 Y",
]

CSV_HEADERS = [
    "cell", "row", "col", "mean", "median", "std", "min", "max",
    "int_den", "area_px", "confidence", "recovered",
    "source_bbox_x1", "source_bbox_y1", "source_bbox_x2", "source_bbox_y2",
    "quad_1_x", "quad_1_y", "quad_2_x", "quad_2_y",
    "quad_3_x", "quad_3_y", "quad_4_x", "quad_4_y",
]


def to_8bit_grayscale(bgr: np.ndarray) -> np.ndarray:
    """Return ImageJ-equivalent 8-bit grayscale values."""
    blue = bgr[:, :, 0].astype(np.float32)
    green = bgr[:, :, 1].astype(np.float32)
    red = bgr[:, :, 2].astype(np.float32)
    return np.clip(
        0.114 * blue + 0.587 * green + 0.299 * red, 0, 255
    ).astype(np.uint8)


def measure(inverted: np.ndarray) -> dict[str, float | int]:
    return {
        "mean": round(float(np.mean(inverted)), 2),
        "median": round(float(np.median(inverted)), 2),
        "std": round(float(np.std(inverted)), 2),
        "min": int(np.min(inverted)),
        "max": int(np.max(inverted)),
        "int_den": round(float(np.sum(inverted)), 2),
        "area_px": int(inverted.size),
    }


def draw_detection_overlay(image: np.ndarray, detection: GridDetection) -> np.ndarray:
    overlay = image.copy()
    for square in detection.squares:
        color = (0, 255, 0) if square.confidence >= 0.55 else (0, 0, 255)
        quad = np.rint(square.source_quad).astype(np.int32)
        cv2.polylines(overlay, [quad], True, color, 3)
        center = tuple(np.rint(quad.mean(axis=0)).astype(int))
        cv2.putText(
            overlay, f"#{square.idx}", (center[0] - 15, center[1] + 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
        )
    return overlay


def heatmap_image(results: list[dict]) -> np.ndarray:
    means = [result["mean"] for result in results]
    minimum, maximum = min(means), max(means)
    cell_size, padding = 120, 6
    height = 3 * (cell_size + padding) + padding + 60
    width = 3 * (cell_size + padding) + padding
    heatmap = np.full((height, width, 3), 36, dtype=np.uint8)

    for result in results:
        row, col = result["row"] - 1, result["col"] - 1
        normalized = (
            (result["mean"] - minimum) / (maximum - minimum)
            if maximum > minimum else 0.5
        )
        blue = int(255 * max(0, 1 - normalized * 2))
        green = int(255 * min(1, abs(normalized - 0.5) * 2))
        red = int(255 * min(1, normalized * 2))
        left = padding + col * (cell_size + padding)
        top = padding + row * (cell_size + padding) + 40
        cv2.rectangle(
            heatmap, (left, top), (left + cell_size, top + cell_size),
            (blue, green, red), -1,
        )
        cv2.rectangle(
            heatmap, (left, top), (left + cell_size, top + cell_size),
            (255, 255, 255), 1,
        )
        value = f"{result['mean']:.1f}"
        (text_width, text_height), _ = cv2.getTextSize(
            value, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        text_color = (0, 0, 0) if normalized > 0.5 else (255, 255, 255)
        cv2.putText(
            heatmap, value,
            (left + (cell_size - text_width) // 2,
             top + (cell_size + text_height) // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2,
        )
        cv2.putText(
            heatmap, f"#{result['idx']}", (left + 4, top + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1,
        )
    return heatmap


def _write_png(path: str, image: np.ndarray) -> None:
    if not cv2.imwrite(path, image):
        raise OSError(f"Could not write image artifact: {os.path.basename(path)}")


def _csv_row(result: dict) -> list:
    return [
        result["idx"], result["row"], result["col"], result["mean"],
        result["median"], result["std"], result["min"], result["max"],
        result["int_den"], result["area_px"], result["confidence"],
        result["recovered"], *result["source_bbox"],
        *[coordinate for point in result["source_quad"] for coordinate in point],
    ]


def _build_bundle(
    detection: GridDetection,
    crops: list[np.ndarray],
    overlay: np.ndarray,
    heatmap: np.ndarray,
    results: list[dict],
    filename: str,
) -> str:
    _cleanup_stale_bundles()
    output_dir = tempfile.mkdtemp(prefix="coagulation_web_")
    artifact_names = [
        *(f"cell_{index:02d}.png" for index in range(1, 10)),
        "grid_overlay.png", "heatmap.png", "results.csv", "results.json",
    ]
    bundle_path = os.path.join(output_dir, "coagulation_results.zip")
    try:
        for index, crop in enumerate(crops, start=1):
            _write_png(os.path.join(output_dir, f"cell_{index:02d}.png"), crop)
        _write_png(os.path.join(output_dir, "grid_overlay.png"), overlay)
        _write_png(os.path.join(output_dir, "heatmap.png"), heatmap)

        with open(
            os.path.join(output_dir, "results.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CSV_HEADERS)
            writer.writerows(_csv_row(result) for result in results)

        with open(
            os.path.join(output_dir, "results.json"), "w", encoding="utf-8"
        ) as json_file:
            json.dump(
                {
                    "image": os.path.basename(filename),
                    "grid": "3x3",
                    "detection_confidence": round(float(detection.confidence), 3),
                    "cells": results,
                },
                json_file,
                indent=2,
            )

        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for artifact_name in artifact_names:
                archive.write(
                    os.path.join(output_dir, artifact_name), arcname=artifact_name
                )
        for artifact_name in artifact_names:
            os.remove(os.path.join(output_dir, artifact_name))
        return bundle_path
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def _cleanup_stale_bundles(max_age_seconds: int = 24 * 60 * 60) -> None:
    """Remove web result directories older than the download grace period."""
    cutoff = time.time() - max_age_seconds
    try:
        entries = os.scandir(tempfile.gettempdir())
    except OSError:
        return
    with entries:
        for entry in entries:
            if not entry.name.startswith("coagulation_web_"):
                continue
            try:
                if entry.is_dir(follow_symlinks=False) and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry.path, ignore_errors=True)
            except OSError:
                continue


def _empty_result(message: str) -> tuple:
    return None, None, [], None, f"**Analysis failed:** {message}"


def process_pipeline(
    image: np.ndarray | None, filename: str = "uploaded_image.png"
) -> tuple:
    """Analyze a Gradio RGB image with the fixed 3x3 detector."""
    if image is None:
        return _empty_result("Please upload an image first.")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        return _empty_result("Upload a color photo of the complete fixture.")

    rgb = image[:, :, :3].astype(np.uint8, copy=False)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        detection = detect_inner_squares(bgr)
        results = []
        crops = []
        for square in detection.squares:
            left, top, right, bottom = square.rectified_bbox
            crop = detection.rectified[top:bottom, left:right]
            if crop.size == 0:
                raise DetectionError(
                    f"Detected crop #{square.idx} is empty. Retake the photo "
                    "with the complete fixture visible."
                )
            result = measure(255 - to_8bit_grayscale(crop))
            result.update(
                {
                    "idx": square.idx,
                    "row": square.row,
                    "col": square.col,
                    "confidence": round(float(square.confidence), 3),
                    "recovered": bool(square.recovered),
                    "source_bbox": [int(value) for value in square.source_bbox],
                    "source_quad": np.rint(square.source_quad).astype(int).tolist(),
                }
            )
            results.append(result)
            crops.append(crop)

        overlay_bgr = draw_detection_overlay(bgr, detection)
        heatmap_bgr = heatmap_image(results)
        bundle_path = _build_bundle(
            detection, crops, overlay_bgr, heatmap_bgr, results, filename
        )
    except (DetectionError, OSError, ValueError, cv2.error) as exception:
        return _empty_result(str(exception))

    table = [
        [
            result["idx"], result["row"], result["col"], result["mean"],
            result["median"], result["std"], result["min"], result["max"],
            result["int_den"], result["area_px"], result["confidence"],
            result["recovered"], *result["source_bbox"],
            *[
                coordinate
                for point in result["source_quad"]
                for coordinate in point
            ],
        ]
        for result in results
    ]
    message = (
        "**Success:** 9 cells analyzed as a fixed 3×3 fixture. "
        f"Overall detection confidence: {detection.confidence:.3f}."
    )
    return (
        cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB),
        table,
        bundle_path,
        message,
    )


HEADER = """
# 🧫 Coagulation Quantification

Upload a photo of the **complete fixed 3×3 fixture**, then click **Analyze**.
The app corrects small rotation or perspective differences, finds the nine
innermost squares, and measures each square using the ImageJ-equivalent
8-bit → Invert workflow.

For reliable detection, keep all outer edges in the photo and use an image
that is at least **600 × 600 pixels**. If detection is uncertain, the app stops
without producing partial measurements.
"""


with gr.Blocks(title="Coagulation Quantification") as demo:
    gr.Markdown(HEADER)
    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(
                label="1. Upload complete 3×3 fixture photo", type="numpy"
            )
            analyze_button = gr.Button(
                "2. Analyze fixed 3×3 fixture", variant="primary", size="lg"
            )
            status_output = gr.Markdown("")
            bundle_output = gr.File(label="Download complete results (ZIP)")
        with gr.Column(scale=3):
            overlay_output = gr.Image(
                label="Detected inner squares (verify alignment)"
            )
            heatmap_output = gr.Image(
                label="3×3 heatmap (blue=low, red=high coagulation)"
            )

    table_output = gr.DataFrame(label="Per-cell results", headers=TABLE_HEADERS)

    analyze_button.click(
        process_pipeline,
        inputs=[image_input],
        outputs=[
            overlay_output, heatmap_output, table_output, bundle_output,
            status_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
