#!/usr/bin/env python3
"""
Coagulation Quantification — Standalone Desktop App
====================================================
Usage: drag an image file onto the app icon, or:
       CoagulationAnalysis.exe image.jpg
"""
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid

import numpy as np
import cv2

from grid_detector import DetectionError, GridDetection, detect_inner_squares

LOG_FILE = os.path.join(os.path.expanduser("~"), "Desktop", "coagulation_log.txt")


def _print_console(message, file=None):
    stream = file if file is not None else sys.stdout
    text = str(message)
    encoding = getattr(stream, "encoding", None)
    if encoding:
        text = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(text, file=stream)


def log(msg):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
    except OSError as exception:
        _print_console(f"Log file unavailable: {exception}", file=sys.stderr)
    _print_console(msg)


def to_8bit(bgr):
    b, g, r = bgr[:,:,0].astype(np.float32), bgr[:,:,1].astype(np.float32), bgr[:,:,2].astype(np.float32)
    return np.clip(0.114*b + 0.587*g + 0.299*r, 0, 255).astype(np.uint8)


def measure(inverted):
    return {
        "mean": round(float(np.mean(inverted)), 2),
        "median": round(float(np.median(inverted)), 2),
        "std": round(float(np.std(inverted)), 2),
        "min": int(np.min(inverted)),
        "max": int(np.max(inverted)),
        "int_den": round(float(np.sum(inverted)), 2),
        "area_px": int(inverted.size),
    }


def heatmap_image(results):
    means = [r["mean"] for r in results]
    vmin, vmax = min(means), max(means)
    n_rows, n_cols = 3, 3
    cs, pad = 100, 5
    hh, ww = n_rows*(cs+pad)+pad+50, n_cols*(cs+pad)+pad
    hm = np.full((hh, ww, 3), 35, dtype=np.uint8)
    for r in results:
        row, col = r["row"]-1, r["col"]-1
        norm = (r["mean"]-vmin)/(vmax-vmin) if vmax>vmin else 0.5
        b = int(255*max(0,1-norm*2)); g = int(255*min(1,abs(norm-0.5)*2)); red = int(255*min(1,norm*2))
        x1, y1 = pad+col*(cs+pad), pad+row*(cs+pad)+40
        cv2.rectangle(hm, (x1,y1), (x1+cs,y1+cs), (b,g,red), -1)
        cv2.rectangle(hm, (x1,y1), (x1+cs,y1+cs), (255,255,255), 1)
        txt = f"{r['mean']:.1f}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(hm, txt, (x1+(cs-tw)//2, y1+(cs+th)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0,0,0) if norm>0.5 else (255,255,255), 2)
        cv2.putText(hm, f"#{r['idx']}", (x1+3,y1+14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1)
    return hm


def load_image(path):
    """Load image, handling non-ASCII paths on Windows."""
    img = cv2.imread(path)
    if img is None:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except:
            pass
    return img


def _write_image(path, image):
    extension = os.path.splitext(path)[1]
    success, encoded = cv2.imencode(extension, image)
    if not success:
        return False
    try:
        encoded.tofile(path)
    except OSError:
        return False
    return os.path.isfile(path) and os.path.getsize(path) > 0


def draw_detection_overlay(img, detection: GridDetection):
    overlay = img.copy()
    for square in detection.squares:
        color = (0, 255, 0) if square.confidence >= 0.55 else (0, 0, 255)
        quad = np.rint(square.source_quad).astype(np.int32)
        cv2.polylines(overlay, [quad], True, color, 2)
        center = tuple(np.rint(quad.mean(axis=0)).astype(int))
        cv2.putText(
            overlay,
            f"#{square.idx}",
            (center[0] - 12, center[1] + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return overlay


def save_results(out_dir, artifact_key, original_filename, results):
    csv_path = os.path.join(out_dir, f"{artifact_key}_results.csv")
    json_path = os.path.join(out_dir, f"{artifact_key}_results.json")
    csv_header = [
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
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)
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
                ]
            )

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(
            {"image": original_filename, "grid": "3x3", "cells": results},
            json_file,
            indent=2,
        )
    return csv_path, json_path


def _artifact_key(filename):
    stem, extension = os.path.splitext(filename)
    readable = "_".join(part for part in (stem, extension.lstrip(".")) if part)
    readable = re.sub(r"[^\w-]+", "_", readable, flags=re.UNICODE).strip("_")
    readable = readable[:80] or "image"
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{digest}"


def show_results(overlay, heatmap, base_name):
    height, width = overlay.shape[:2]
    if max(width, height) > 700:
        display_width = min(width, 700)
        display_overlay = cv2.resize(
            overlay, (display_width, int(display_width / width * height))
        )
    else:
        display_overlay = overlay.copy()
    display_heatmap = cv2.resize(
        heatmap,
        (
            display_overlay.shape[1],
            display_overlay.shape[1] * heatmap.shape[0] // heatmap.shape[1],
        ),
    )
    cv2.imshow(
        f"Results - {base_name}  (any key to close)",
        np.vstack([display_overlay, display_heatmap]),
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def open_output_folder(out_dir):
    if sys.platform == "win32":
        os.startfile(out_dir)
    elif sys.platform == "darwin":
        subprocess.run(["open", out_dir], check=False)
    else:
        subprocess.run(["xdg-open", out_dir], check=False)


def process_image(path, show_windows=True, open_folder=True):
    img = load_image(path)
    if img is None:
        raise ValueError(
            "Cannot open image. Try renaming it to a simple filename and try again."
        )

    try:
        detection = detect_inner_squares(img)
    except DetectionError as exception:
        log(f"Grid detection failed: {exception}")
        raise
    outer_quad = np.rint(detection.outer_quad).astype(int).tolist()
    log(
        f"Grid detection confidence={detection.confidence:.3f} "
        f"outer_quad={outer_quad}"
    )
    original_filename = os.path.basename(path)
    artifact_key = _artifact_key(original_filename)
    parent_dir = os.path.dirname(os.path.abspath(path))
    out_dir = os.path.join(parent_dir, f"{artifact_key}_analysis")
    staging_dir = tempfile.mkdtemp(
        prefix=f".{artifact_key}_staging_",
        dir=parent_dir,
    )
    try:
        results = []
        for square in detection.squares:
            x1, y1, x2, y2 = square.rectified_bbox
            cell = detection.rectified[y1:y2, x1:x2]
            if cell.size == 0:
                raise DetectionError(
                    f"Detected crop #{square.idx} is empty. Retake the photo "
                    "with the full grid in frame."
                )
            inverted = 255 - to_8bit(cell)
            result = measure(inverted)
            result.update(
                {
                    "idx": square.idx,
                    "row": square.row,
                    "col": square.col,
                    "confidence": round(float(square.confidence), 3),
                    "recovered": bool(square.recovered),
                    "source_bbox": list(square.source_bbox),
                    "crop_quad": np.rint(square.source_quad).astype(int).tolist(),
                }
            )
            results.append(result)
            log(
                f"Cell #{square.idx} confidence={result['confidence']:.3f} "
                f"recovered={str(result['recovered']).lower()} "
                f"source_bbox={result['source_bbox']}"
            )
            crop_path = os.path.join(staging_dir, f"cell_{square.idx:02d}.png")
            if not _write_image(crop_path, cell):
                raise OSError(f"Could not write crop: {crop_path}")

        overlay = draw_detection_overlay(img, detection)
        overlay_name = f"{artifact_key}_grid_overlay.png"
        overlay_path = os.path.join(staging_dir, overlay_name)
        if not _write_image(overlay_path, overlay):
            raise OSError(f"Could not write overlay: {overlay_path}")

        heatmap = heatmap_image(results)
        heatmap_name = f"{artifact_key}_heatmap.png"
        heatmap_path = os.path.join(staging_dir, heatmap_name)
        if not _write_image(heatmap_path, heatmap):
            raise OSError(f"Could not write heatmap: {heatmap_path}")

        csv_path, json_path = save_results(
            staging_dir,
            artifact_key,
            original_filename,
            results,
        )
        backup_dir = None
        if os.path.exists(out_dir):
            backup_dir = f"{out_dir}.previous-{uuid.uuid4().hex}"
            os.replace(out_dir, backup_dir)
        try:
            os.replace(staging_dir, out_dir)
        except Exception:
            if backup_dir is not None and not os.path.exists(out_dir):
                os.replace(backup_dir, out_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
    except Exception:
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir)
        raise

    overlay_path = os.path.join(out_dir, overlay_name)
    heatmap_path = os.path.join(out_dir, heatmap_name)
    csv_path = os.path.join(out_dir, os.path.basename(csv_path))
    json_path = os.path.join(out_dir, os.path.basename(json_path))
    log(f"Done. {len(results)} cells analyzed. Output: {out_dir}")

    if show_windows:
        show_results(overlay, heatmap, original_filename)
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


def main():
    try:
        _main()
    except DetectionError as exc:
        message = f"Grid detection failed: {exc}"
        print(message)
        input(f"{message}\nPress Enter to exit...")
    except Exception:
        msg = traceback.format_exc()
        log(f"FATAL ERROR:\n{msg}")
        print(msg)
        input("\nPress Enter to exit...")


def _main():
    log("=== Coagulation Analysis App ===")

    # ── Get image path ──
    if len(sys.argv) < 2:
        log("No image provided. Usage: drag an image file onto this app icon.")
        print("\nDrag an image file onto this app icon.")
        print("Or run: CoagulationAnalysis.exe image.jpg\n")
        input("Press Enter to exit...")
        sys.exit(0)

    path = sys.argv[1].strip('"').strip("'")
    log(f"Image path: {path}")

    if not os.path.exists(path):
        log(f"File not found: {path}")
        input("File not found. Press Enter.")
        sys.exit(1)

    process_image(path)


if __name__ == "__main__":
    main()
