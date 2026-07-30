#!/usr/bin/env python3
"""
Coagulation Quantification — Standalone Desktop App
====================================================
Usage: drag an image file onto the app icon, or:
       CoagulationAnalysis.exe image.jpg
"""
import csv
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import cv2

from grid_detector import DetectionError, GridDetection, detect_inner_squares

LOG_FILE = os.path.join(os.path.expanduser("~"), "Desktop", "coagulation_log.txt")

def log(msg):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
    except:
        pass
    print(msg)


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


def save_results(out_dir, base_name, results):
    csv_path = os.path.join(out_dir, f"{base_name}_results.csv")
    json_path = os.path.join(out_dir, f"{base_name}_results.json")
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
                ]
            )

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(
            {"image": base_name, "grid": "3x3", "cells": results},
            json_file,
            indent=2,
        )
    return csv_path, json_path


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

    detection = detect_inner_squares(img)
    base_name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(path)), f"{base_name}_analysis"
    )
    os.makedirs(out_dir, exist_ok=True)

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
                "crop_quad": np.rint(square.source_quad).astype(int).tolist(),
            }
        )
        results.append(result)
        crop_path = os.path.join(out_dir, f"cell_{square.idx:02d}.png")
        if not cv2.imwrite(crop_path, cell):
            raise OSError(f"Could not write crop: {crop_path}")

    overlay = draw_detection_overlay(img, detection)
    overlay_path = os.path.join(out_dir, f"{base_name}_grid_overlay.png")
    if not cv2.imwrite(overlay_path, overlay):
        raise OSError(f"Could not write overlay: {overlay_path}")

    heatmap = heatmap_image(results)
    heatmap_path = os.path.join(out_dir, f"{base_name}_heatmap.png")
    if not cv2.imwrite(heatmap_path, heatmap):
        raise OSError(f"Could not write heatmap: {heatmap_path}")

    csv_path, json_path = save_results(out_dir, base_name, results)
    log(f"Done. {len(results)} cells analyzed. Output: {out_dir}")

    if show_windows:
        show_results(overlay, heatmap, base_name)
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
        log(message)
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
