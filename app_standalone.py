#!/usr/bin/env python3
"""
Coagulation Quantification — Standalone Desktop App
====================================================
Usage: drag an image file onto the app icon, or:
       CoagulationAnalysis.exe image.jpg
"""

import os
import subprocess
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

import analysis_service as _analysis_service
from analysis_service import AnalysisSettings, analyze_image, load_image
from grid_detector import DetectionError


LOG_FILE = os.path.join(os.path.expanduser("~"), "Desktop", "coagulation_log.txt")

# These aliases preserve the helper surface used by existing desktop integrations.
_write_image = _analysis_service._write_image
_artifact_key = _analysis_service._artifact_key


def _print_console(message, file=None):
    stream = file if file is not None else sys.stdout
    text = str(message)
    encoding = getattr(stream, "encoding", None)
    if encoding:
        text = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(text, file=stream)


def log(msg):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(f"{msg}\n")
    except OSError as exception:
        _print_console(f"Log file unavailable: {exception}", file=sys.stderr)
    _print_console(msg)


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


def process_image(
    path,
    show_windows=True,
    open_folder=True,
    inset_percent=5.0,
    no_clot_threshold=60.0,
    results_root=None,
):
    """Run the shared analysis service with desktop display and audit logging."""
    service_writer = _analysis_service._write_image
    _analysis_service._write_image = _write_image
    try:
        result = analyze_image(
            path,
            AnalysisSettings(inset_percent, no_clot_threshold, results_root),
        )
    except DetectionError as exception:
        log(f"Grid detection failed: {exception}")
        raise
    finally:
        _analysis_service._write_image = service_writer

    log(
        f"Grid detection confidence={result['grid_confidence']:.3f} "
        f"outer_quad={result['outer_quad']}"
    )
    for cell in result["cells"]:
        log(
            f"Cell #{cell['idx']} confidence={cell['confidence']:.3f} "
            f"recovered={str(cell['recovered']).lower()} "
            f"source_bbox={cell['source_bbox']}"
        )
    log(
        f"Done. {len(result['cells'])} cells analyzed. "
        f"Output: {result['output_dir']}"
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
