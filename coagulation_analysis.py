#!/usr/bin/env python3
"""
Coagulation Quantification Pipeline — ImageJ Workflow Automation
=================================================================
Exact ImageJ workflow replicated:
  1. Image > Type > 8-bit        →  convert to grayscale (0-255)
  2. Edit > Invert               →  255 - gray (coagulation = bright)
  3. Rectangle ROI on glass slide →  auto-detect or manual
  4. Analyze > Measure           →  Mean, Area, StdDev, Min, Max, IntDen
  5. Optional: divide into grid  →  per-square analysis (e.g. 4×3 = 12)

Usage:
  python3 coagulation_analysis.py <image.jpg>
  python3 coagulation_analysis.py <image.jpg> --rows 4 --cols 3
  python3 coagulation_analysis.py <image.jpg> --manual     # interactive ROI
  python3 coagulation_analysis.py <folder/> --batch        # all images

Output matches ImageJ terminology: Mean, Area, IntDen, StdDev
"""

import sys, os, json, argparse, glob
import numpy as np
import cv2
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
#  SLIDE DETECTION — Adaptive: intensity first, texture fallback
# ═══════════════════════════════════════════════════════════════

def _intensity_detect(gray):
    """
    For HIGH-CONTRAST images: slide is clearly brighter/darker than background.
    Uses Otsu thresholding to find the slide as the largest connected region.
    Returns (box, method) or None if slide not clearly separable.
    """
    h, w = gray.shape
    img_area = h * w

    for thresh_type, invert in [("otsu", False), ("triangle", False)]:
        try:
            if thresh_type == "otsu":
                t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

            # Only flip if slide is clearly the minority (slide should be 3-95% of image)
            white_frac = np.mean(binary) / 255
            if white_frac < 0.03:
                binary = cv2.bitwise_not(binary)

            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            area_ratio = cv2.contourArea(largest) / img_area

            # Slide should be 3-95% of image
            if 0.03 < area_ratio < 0.95:
                x, y, rw, rh = cv2.boundingRect(largest)
                # Check: slide should be roughly centered and large enough
                center_x, center_y = x + rw//2, y + rh//2
                if (rw > w*0.10 and rh > h*0.10):
                    box = np.array([[x, y], [x+rw, y], [x+rw, y+rh], [x, y+rh]], dtype=np.int32)
                    return box, f"intensity_{thresh_type}"
        except Exception:
            continue

    return None


def _texture_detect(image):
    """
    For LOW-CONTRAST images: slide and background have similar intensity.
    Uses multi-scale texture (local std) scanning to find slide edges.
    Returns (box, method) or None.
    """
    h, w = image.shape[:2]
    blue = image[:, :, 0].astype(np.float32)
    cx, cy = w // 2, h // 2

    search_half = int(min(w, h) * 0.35)
    refine_half = int(min(w, h) * 0.10)

    edges = {}
    for scale_name, ksize in [('fine', 15), ('medium', 31), ('coarse', 61)]:
        mean = cv2.blur(blue, (ksize, ksize))
        sq_mean = cv2.blur(blue**2, (ksize, ksize))
        std_map = np.sqrt(np.maximum(sq_mean - mean**2, 0))
        std_map = cv2.GaussianBlur(std_map, (ksize|1, ksize|1), 0)

        edges[scale_name] = {
            'left':   _scan_edge(blue, std_map, 'left',   cx, search_half, refine_half),
            'right':  _scan_edge(blue, std_map, 'right',  cx, search_half, refine_half),
            'top':    _scan_edge(blue, std_map, 'top',    cy, search_half, refine_half),
            'bottom': _scan_edge(blue, std_map, 'bottom', cy, search_half, refine_half),
        }

    left_x   = int(np.median([edges[s]['left']   for s in edges]))
    right_x  = int(np.median([edges[s]['right']  for s in edges]))
    top_y    = int(np.median([edges[s]['top']    for s in edges]))
    bottom_y = int(np.median([edges[s]['bottom'] for s in edges]))

    if right_x - left_x < 50 or bottom_y - top_y < 50:
        return None

    box = np.array([[left_x, top_y], [right_x, top_y],
                     [right_x, bottom_y], [left_x, bottom_y]], dtype=np.int32)
    return box, "texture_multiscale"


def detect_slide(image, slide_fraction=0.30):
    """
    Adaptive slide detection:
      1. Try intensity-based (Otsu/Triangle) — for high-contrast images
      2. Fall back to multi-scale texture scanning — for low-contrast images
      3. Keep natural shape (rectangular OR square — no forced square)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Strategy 1: Intensity-based (best for high-contrast)
    result = _intensity_detect(gray)
    if result is not None:
        return result

    # Strategy 2: Texture-based (best for low-contrast)
    result = _texture_detect(image)
    if result is not None:
        return result

    # Strategy 3: Centered fallback
    h, w = image.shape[:2]
    size = int(min(w, h) * slide_fraction)
    cx, cy = w // 2, h // 2
    half = size // 2
    box = np.array([[cx-half, cy-half], [cx+half, cy-half],
                     [cx+half, cy+half], [cx-half, cy+half]], dtype=np.int32)
    return box, "centered_fallback"


def manual_roi_gui(image):
    """Interactive 4-corner ROI selection."""
    points = []
    img_copy = image.copy()

    def click_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img_copy, (x, y), 6, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(img_copy, points[-2], points[-1], (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(img_copy, points[3], points[0], (0, 255, 0), 2)
            cv2.imshow("Click 4 corners (top-left, clockwise), SPACE=done", img_copy)

    print("\n  Click 4 corners of the glass slide (top-left first, clockwise)")
    print("  Press SPACE when done, ESC to cancel")
    cv2.namedWindow("Click 4 corners (top-left, clockwise), SPACE=done")
    cv2.setMouseCallback("Click 4 corners (top-left, clockwise), SPACE=done", click_handler)

    while True:
        cv2.imshow("Click 4 corners (top-left, clockwise), SPACE=done", img_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        if key == 32 and len(points) >= 4: break
    cv2.destroyAllWindows()

    if len(points) < 4:
        raise ValueError("Need 4 corners")
    return np.array(points[:4], dtype=np.int32)


# ═══════════════════════════════════════════════════════════════
#  GRID ANALYSIS
# ═══════════════════════════════════════════════════════════════

def _scan_edge(channel, std_map, axis, center, search_half, refine_half):
    """Scan perpendicular to an edge to find background→texture transition."""
    h, w = channel.shape
    positions = []
    if axis == 'left':
        start, end = max(0, center-search_half), min(w, center+refine_half)
        bg = np.median(std_map[:, :max(1,start//2)])
        for row in range(0, h, max(1, h//60)):
            cross = np.where(std_map[row, start:end] > bg*2.5)[0]
            if len(cross): positions.append(start + cross[0])
    elif axis == 'right':
        start, end = max(0, center-refine_half), min(w, center+search_half)
        bg = np.median(std_map[:, max(0, w-50):w]) if end >= w-10 else np.median(std_map[:, end+(w-end)//2:min(w, end+(w-end))])
        for row in range(0, h, max(1, h//60)):
            cross = np.where(std_map[row, start:end][::-1] > bg*2.5)[0]
            if len(cross): positions.append(end - cross[0])
    elif axis == 'top':
        start, end = max(0, center-search_half), min(h, center+refine_half)
        bg = np.median(std_map[:max(1,start//2), :])
        for col in range(0, w, max(1, w//60)):
            cross = np.where(std_map[start:end, col] > bg*2.5)[0]
            if len(cross): positions.append(start + cross[0])
    elif axis == 'bottom':
        start, end = max(0, center-refine_half), min(h, center+search_half)
        bg = np.median(std_map[max(0, h-50):h, :]) if end >= h-10 else np.median(std_map[end+(h-end)//2:min(h, end+(h-end)), :])
        for col in range(0, w, max(1, w//60)):
            cross = np.where(std_map[start:end, col][::-1] > bg*2.5)[0]
            if len(cross): positions.append(end - cross[0])
    return int(np.median(positions)) if positions else (start + end)//2 if axis in ('left','top') else (start + end)//2


def extract_slide_roi(image, box):
    """Extract slide region — keeps natural shape (rectangular OR square)."""
    x, y, rw, rh = cv2.boundingRect(box)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image.shape[1], x+rw), min(image.shape[0], y+rh)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def find_grid_area(roi_gray):
    """
    Find the textured sub-region (the actual grid of squares) within the slide.
    Crops out blank margins around the grid for more precise cell division.
    """
    h, w = roi_gray.shape
    ksize = max(21, min(w, h) // 30) | 1
    local_mean = cv2.blur(roi_gray.astype(np.float32), (ksize, ksize))
    local_sq_mean = cv2.blur(roi_gray.astype(np.float32)**2, (ksize, ksize))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    std_smooth = cv2.GaussianBlur(local_std, (51, 51), 0)

    # Threshold at 50th percentile to get textured region
    t = np.percentile(std_smooth, 50)
    mask = (std_smooth > t).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        gx, gy, gw, gh = cv2.boundingRect(largest)
        # Ensure reasonable size (use smaller dimension ratio)
        min_dim = min(w, h)
        if gw > min_dim * 0.3 and gh > min_dim * 0.3:
            return gx, gy, gw, gh
    # Fallback: use entire ROI with 10% margin trimmed
    return int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9)


def analyze_grid(roi_bgr, n_rows, n_cols, output_dir, image_name):
    """
    Auto-detect textured grid area within slide, then divide into
    n_rows × n_cols grid. Analyze each cell individually.
    For each cell: invert colors, measure gray values.
    """
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = roi_gray.shape

    # Find the grid area (crops blank margins)
    gx, gy, gw, gh = find_grid_area(roi_gray)
    cell_h = gh // n_rows
    cell_w = gw // n_cols
    print(f"  Grid area: ({gx},{gy}) {gw}×{gh} px, cells: {cell_w}×{cell_h} px")

    cells = []
    cell_images = {}
    grid_roi = roi_gray[gy:gy+gh, gx:gx+gw]  # the actual grid region

    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col + 1
            ys, ye = row * cell_h, (row + 1) * cell_h
            xs, xe = col * cell_w, (col + 1) * cell_w

            cell_gray = grid_roi[ys:ye, xs:xe]
            cell_bgr = roi_bgr[gy+ys:gy+ye, gx+xs:gx+xe]

            # Invert: coagulation (dark) → bright/high value
            cell_inv = cv2.bitwise_not(cell_gray)

            # Statistics
            mean_val = float(np.mean(cell_inv))
            median_val = float(np.median(cell_inv))
            std_val = float(np.std(cell_inv))
            min_val = int(np.min(cell_inv))
            max_val = int(np.max(cell_inv))
            int_den = float(np.sum(cell_inv))

            cell_data = {
                "cell": idx,
                "row": row + 1,
                "col": col + 1,
                "position_px": {"x": xs, "y": ys, "width": cell_w, "height": cell_h},
                "gray_value_statistics": {
                    "mean": round(mean_val, 2),
                    "median": round(median_val, 2),
                    "std": round(std_val, 2),
                    "min": min_val,
                    "max": max_val,
                    "integrated_density": round(int_den, 2),
                    "mean_normalized": round(mean_val / 255.0, 4),
                },
            }
            cells.append(cell_data)
            cell_images[idx] = {
                "original": cell_bgr,
                "gray": cell_gray,
                "inverted": cell_inv,
            }

    # ── Build grid visualization ──
    # Grid overlay with auto-detected grid area (yellow) + cell boundaries (green)
    grid_img = roi_bgr.copy()
    cv2.rectangle(grid_img, (gx, gy), (gx+gw, gy+gh), (0, 255, 255), 3)
    for row in range(1, n_rows):
        y = gy + row * cell_h
        cv2.line(grid_img, (gx, y), (gx+gw, y), (0, 255, 0), 2)
    for col in range(1, n_cols):
        x = gx + col * cell_w
        cv2.line(grid_img, (x, gy), (x, gy+gh), (0, 255, 0), 2)
    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col + 1
            xs, ys = gx + col * cell_w, gy + row * cell_h
            cv2.putText(grid_img, str(idx), (xs + cell_w//2 - 20, ys + cell_h//2 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_grid_overlay.png"), grid_img)

    # Grid heatmap: tile the 12 inverted cells in a montage
    inv_rows = []
    for row in range(n_rows):
        row_cells = []
        for col in range(n_cols):
            idx = row * n_cols + col + 1
            cell_inv = cell_images[idx]["inverted"]
            # Colorize
            cell_heat = cv2.applyColorMap(cell_inv, cv2.COLORMAP_INFERNO)
            # Add label
            cv2.putText(cell_heat, f"#{idx}", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            row_cells.append(cell_heat)
        inv_rows.append(np.hstack(row_cells))
    grid_heatmap = np.vstack(inv_rows)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_grid_heatmap.png"), grid_heatmap)

    # Per-cell detail images
    for idx, imgs in cell_images.items():
        # Small comparison per cell: original | gray | inverted
        gray_3ch = cv2.cvtColor(imgs["gray"], cv2.COLOR_GRAY2BGR)
        inv_3ch = cv2.cvtColor(imgs["inverted"], cv2.COLOR_GRAY2BGR)
        comp = np.hstack([imgs["original"], gray_3ch, inv_3ch])
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_cell_{idx:02d}.png"), comp)

    return cells, grid_img


# ═══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def process_image(image_path, args):
    """Full pipeline for one image."""
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.dirname(os.path.abspath(image_path)) or "."
    output_dir = args.output_dir or os.path.join(image_dir, f"{image_name}_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'─'*55}")
    print(f"  {os.path.basename(image_path)}")
    print(f"{'─'*55}")

    # Step 1: Load
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: Cannot load")
        return None
    print(f"  [1/4] Loaded: {image.shape[1]}×{image.shape[0]} px")

    # Step 2: ROI
    if args.manual:
        print(f"  [2/4] Manual ROI selection...")
        try:
            box = manual_roi_gui(image)
            method = "manual"
        except Exception as e:
            print(f"        Fallback to auto-detect: {e}")
            box, method = detect_slide(image, args.size)
    else:
        box, method = detect_slide(image, args.size)
    xb, yb, rwb, rhb = cv2.boundingRect(box)
    print(f"  [2/4] Slide ROI: {rwb}×{rhb} px ({method})")

    # Step 3: Extract + Grid
    roi, bounds = extract_slide_roi(image, box)
    n_rows, n_cols = args.rows, args.cols
    print(f"  [3/4] Dividing into {n_rows}×{n_cols} grid ({n_rows*n_cols} squares)...")

    # Save detection image
    det_img = image.copy()
    cv2.drawContours(det_img, [box], 0, (0, 255, 0), 3)
    cv2.putText(det_img, f"Slide: {rwb}x{rhb}px | Grid: {n_rows}x{n_cols}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_detection.png"), det_img)

    # Step 4: Analyze each cell
    print(f"  [4/4] Analyzing {n_rows*n_cols} squares...")
    cells, grid_img = analyze_grid(roi, n_rows, n_cols, output_dir, image_name)

    # ── Save data ──
    results = {
        "image": image_name,
        "timestamp": datetime.now().isoformat(),
        "grid": {"rows": n_rows, "cols": n_cols, "total_cells": n_rows * n_cols},
        "slide_roi_size_px": f"{roi.shape[1]}x{roi.shape[0]}",
        "detection_method": method,
        "cells": cells,
    }

    json_path = os.path.join(output_dir, f"{image_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, f"{image_name}_results.csv")
    with open(csv_path, "w") as f:
        f.write("cell,row,col,mean,median,std,min,max,integrated_density,mean_norm\n")
        for c in cells:
            s = c["gray_value_statistics"]
            f.write(f"{c['cell']},{c['row']},{c['col']},{s['mean']},{s['median']},"
                    f"{s['std']},{s['min']},{s['max']},{s['integrated_density']},{s['mean_normalized']}\n")

    # ── Print summary table ──
    print(f"\n  {'Cell':>5s} {'Row':>4s} {'Col':>4s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} {'IntDen':>12s}")
    print(f"  {'─'*5} {'─'*4} {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*12}")
    for c in cells:
        s = c["gray_value_statistics"]
        print(f"  {c['cell']:>5d} {c['row']:>4d} {c['col']:>4d} {s['mean']:>8.1f} {s['median']:>8.1f} {s['std']:>8.1f} {s['integrated_density']:>12.0f}")

    print(f"\n  Output: {output_dir}/")
    print(f"    - {image_name}_grid_overlay.png   (ROI + grid overlay)")
    print(f"    - {image_name}_grid_heatmap.png   (12-cell heatmap montage)")
    print(f"    - {image_name}_cell_XX.png        (per-cell detail)")
    print(f"    - {image_name}_results.json       (full data)")
    print(f"    - {image_name}_results.csv        (Excel-ready)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Coagulation Quantification — 12-Square Grid Analysis")
    parser.add_argument("image", nargs="?", help="Image path or directory (with --batch)")
    parser.add_argument("--rows", type=int, default=3, help="Grid rows (default: 3)")
    parser.add_argument("--cols", type=int, default=4, help="Grid columns (default: 4)")
    parser.add_argument("--size", type=float, default=0.30, help="Slide size as fraction (default: 0.30)")
    parser.add_argument("--manual", action="store_true", help="Manual ROI selection")
    parser.add_argument("--batch", action="store_true", help="Process all images in directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    if not args.image:
        parser.print_help()
        sys.exit(1)

    # Collect images
    if args.batch:
        if not os.path.isdir(args.image):
            print("Error: --batch requires a directory")
            sys.exit(1)
        images = sorted(glob.glob(os.path.join(args.image, "*.jpg")) +
                        glob.glob(os.path.join(args.image, "*.JPG")) +
                        glob.glob(os.path.join(args.image, "*.jpeg")) +
                        glob.glob(os.path.join(args.image, "*.png")) +
                        glob.glob(os.path.join(args.image, "*.PNG")))
        if not images:
            print(f"No images in {args.image}")
            sys.exit(1)
        print(f"Batch: {len(images)} images, {args.rows}×{args.cols} grid")
    else:
        if not os.path.exists(args.image):
            print(f"Error: {args.image} not found")
            sys.exit(1)
        images = [args.image]

    all_results = {}
    for img_path in images:
        result = process_image(img_path, args)
        if result:
            all_results[os.path.basename(img_path)] = result

    if len(all_results) > 1:
        # Batch summary
        print(f"\n{'='*70}")
        print(f"  BATCH SUMMARY — {len(all_results)} images, {args.rows}×{args.cols} grid")
        print(f"{'='*70}")
        # Per-cell mean across all images
        cell_means = {i+1: [] for i in range(args.rows * args.cols)}
        for name, r in all_results.items():
            for c in r["cells"]:
                cell_means[c["cell"]].append(c["gray_value_statistics"]["mean"])
        print(f"  {'Cell':>5s} {'Mean':>8s} {'Min':>8s} {'Max':>8s} {'Std':>8s}")
        for idx in sorted(cell_means):
            vals = cell_means[idx]
            print(f"  {idx:>5d} {np.mean(vals):>8.1f} {np.min(vals):>8.1f} {np.max(vals):>8.1f} {np.std(vals):>8.1f}")

    print(f"\nDone — {len(all_results)} image(s) processed.")


if __name__ == "__main__":
    main()
