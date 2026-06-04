#!/usr/bin/env python3
"""
Coagulation Full Workflow — 半自动裁切 + 分析 + 可视化 + 多组对比
=================================================================
1. 交互式网格裁切（拖矩形 → 自动分格）
2. ImageJ 精度对齐 (8-bit → Invert → Measure)
3. 热力图可视化
4. 多组实验结果汇总对比

用法:
  python3 full_workflow.py 玻片照.jpg                  # 交互裁切+分析
  python3 full_workflow.py 玻片照.jpg --rows 3 --cols 6
  python3 full_workflow.py 文件夹/ --compare            # 多组对比
"""
import sys, os, json, glob, time, argparse
import numpy as np
import cv2

# ═══════════════════════════════════════════════════════════════
#  ImageJ 精度对齐
# ═══════════════════════════════════════════════════════════════

def to_8bit_grayscale(bgr_image):
    """
    ImageJ 'Image > Type > 8-bit' exact formula:
    gray = (0.299*R + 0.587*G + 0.114*B)
    OpenCV uses same coefficients, but we compute explicitly
    to guarantee pixel-perfect match with ImageJ.
    """
    b, g, r = bgr_image[:, :, 0].astype(np.float32), \
              bgr_image[:, :, 1].astype(np.float32), \
              bgr_image[:, :, 2].astype(np.float32)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(gray, 0, 255).astype(np.uint8)


def invert(gray_image):
    """ImageJ 'Edit > Invert': 255 - pixel."""
    return 255 - gray_image


def measure(inverted_image):
    """ImageJ 'Analyze > Measure': Mean, StdDev, Min, Max, IntDen."""
    return {
        "mean":   round(float(np.mean(inverted_image)), 2),
        "median": round(float(np.median(inverted_image)), 2),
        "std":    round(float(np.std(inverted_image)), 2),
        "min":    int(np.min(inverted_image)),
        "max":    int(np.max(inverted_image)),
        "int_den": round(float(np.sum(inverted_image)), 2),
        "area_px": inverted_image.size,
    }


# ═══════════════════════════════════════════════════════════════
#  交互式网格裁切
# ═══════════════════════════════════════════════════════════════

def interactive_grid_crop(image_path, n_rows, n_cols):
    """
    Show image, let user drag a rectangle around all squares,
    then auto-divide into n_rows × n_cols grid cells.
    Returns: list of (cell_idx, cropped_bgr_image) sorted by position.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load: {image_path}")

    # Resize for display if too large
    h, w = img.shape[:2]
    max_display = 1600
    scale = 1.0
    if max(w, h) > max_display:
        scale = max_display / max(w, h)
        display = cv2.resize(img, (int(w*scale), int(h*scale)))
    else:
        display = img.copy()

    # ROI selection state
    roi = {'drawing': False, 'start': None, 'end': None, 'done': False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi['drawing'] = True
            roi['start'] = (x, y)
            roi['end'] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and roi['drawing']:
            roi['end'] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            roi['drawing'] = False
            roi['end'] = (x, y)

    win_name = "Draw rectangle around all squares — SPACE to confirm, ESC to cancel"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)

    print("\n  ┌──────────────────────────────────────┐")
    print("  │  鼠标拖一个矩形，框住所有方格        │")
    print("  │  按 空格键 确认   按 ESC 取消        │")
    print(f"  │  网格: {n_rows}行 × {n_cols}列 = {n_rows*n_cols}格  │")
    print("  └──────────────────────────────────────┘")

    while True:
        frame = display.copy()
        if roi['start'] and roi['end']:
            sx, sy = roi['start']
            ex, ey = roi['end']
            cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
            # Draw grid preview
            rw, rh = ex - sx, ey - sy
            if rw > 10 and rh > 10:
                cell_w, cell_h = rw // n_cols, rh // n_rows
                for r in range(1, n_rows):
                    y = sy + r * cell_h
                    cv2.line(frame, (sx, y), (ex, y), (0, 255, 0), 1)
                for c in range(1, n_cols):
                    x = sx + c * cell_w
                    cv2.line(frame, (x, sy), (x, ey), (0, 255, 0), 1)
        key = cv2.waitKey(1) & 0xFF
        cv2.imshow(win_name, frame)
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return None
        if key == 32 and roi['start'] and roi['end']:  # SPACE
            roi['done'] = True
            break

    cv2.destroyAllWindows()

    # Map back to original coordinates
    sx, sy = roi['start']
    ex, ey = roi['end']
    sx, sy = int(sx / scale), int(sy / scale)
    ex, ey = int(ex / scale), int(ey / scale)

    # Ensure sx < ex, sy < ey
    sx, ex = min(sx, ex), max(sx, ex)
    sy, ey = min(sy, ey), max(sy, ey)

    # Crop each cell
    grid_w, grid_h = ex - sx, ey - sy
    cell_w, cell_h = grid_w // n_cols, grid_h // n_rows
    cells = []
    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col + 1
            xs = sx + col * cell_w
            ys = sy + row * cell_h
            xe = xs + cell_w
            ye = ys + cell_h
            cell_img = img[ys:ye, xs:xe].copy()
            cells.append({
                "idx": idx,
                "row": row + 1,
                "col": col + 1,
                "image": cell_img,
                "position": (xs, ys, cell_w, cell_h),
            })

    # ── Fine-tuning preview with arrow keys ──
    shift = 3  # pixels per arrow-key nudge (at original resolution)
    while True:
        # Recompute cells with current sx,sy,ex,ey
        grid_w, grid_h = ex - sx, ey - sy
        cell_w, cell_h = grid_w // n_cols, grid_h // n_rows
        cells = []
        for row in range(n_rows):
            for col in range(n_cols):
                idx = row * n_cols + col + 1
                xs = sx + col * cell_w
                ys = sy + row * cell_h
                xe = xs + cell_w
                ye = ys + cell_h
                cell_img = img[ys:ye, xs:xe].copy()
                cells.append({"idx": idx, "row": row+1, "col": col+1,
                              "image": cell_img, "position": (xs, ys, cell_w, cell_h)})

        preview = img.copy()
        cv2.rectangle(preview, (sx, sy), (ex, ey), (0, 255, 255), 2)
        for c in cells:
            xs_p, ys_p, cw, ch = c["position"]
            cv2.rectangle(preview, (xs_p, ys_p), (xs_p+cw, ys_p+ch), (0, 255, 0), 1)
            cv2.putText(preview, str(c["idx"]), (xs_p + cw//2 - 12, ys_p + ch//2 + 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Resize preview for display
        s2 = 1.0
        if max(w, h) > 1200:
            s2 = 1200 / max(w, h)
            preview_disp = cv2.resize(preview, (int(w*s2), int(h*s2)))
        else:
            preview_disp = preview.copy()

        msg = "Check grid! Arrows=nudge  +/-=zoom  ENTER=confirm  ESC=redo"
        cv2.putText(preview_disp, msg, (10, preview_disp.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(preview_disp, f"Rect: ({sx},{sy})-({ex},{ey})  Cell: {cell_w}x{cell_h}px",
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Preview — adjust with arrow keys, ENTER to confirm", preview_disp)
        key = cv2.waitKey(0) & 0xFF

        if key == 13:  # ENTER
            cv2.destroyAllWindows()
            return cells
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return None
        elif key == 81:  # left arrow
            sx = max(0, sx - shift)
        elif key == 83:  # right arrow
            ex = min(w, ex + shift)
        elif key == 82:  # up arrow
            sy = max(0, sy - shift)
        elif key == 84:  # down arrow
            ey = min(h, ey + shift)
        elif key == 43 or key == 61:  # + key
            shift = min(shift * 2, 50)
        elif key == 45:  # - key
            shift = max(shift // 2, 1)
        cv2.destroyWindow("Preview — adjust with arrow keys, ENTER to confirm")


# ═══════════════════════════════════════════════════════════════
#  可视化热力图
# ═══════════════════════════════════════════════════════════════

def generate_heatmap(cells_results, output_path, title=""):
    """
    Generate a heatmap grid showing the Mean value for each cell.
    Color: low (blue) → high (red), with value labels.
    """
    n_rows = max(c["row"] for c in cells_results)
    n_cols = max(c["col"] for c in cells_results)
    means = [c["mean"] for c in cells_results]
    vmin, vmax = min(means), max(means)

    cell_size = 100
    pad = 5
    img_h = n_rows * (cell_size + pad) + pad + 60
    img_w = n_cols * (cell_size + pad) + pad

    canvas = np.full((img_h, img_w, 3), 40, dtype=np.uint8)

    for c in cells_results:
        row, col = c["row"] - 1, c["col"] - 1
        val = c["mean"]
        # Normalize to 0-1 for colormap
        norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        # Inferno-like: dark blue → purple → red → yellow
        r = int(255 * min(1, norm * 2))
        g = int(255 * min(1, abs(norm - 0.5) * 2))
        b = int(255 * max(0, 1 - norm * 2))
        color = (b, g, r)  # BGR

        x1 = pad + col * (cell_size + pad)
        y1 = pad + row * (cell_size + pad) + 40
        x2, y2 = x1 + cell_size, y1 + cell_size
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Value label
        text = f"{val:.1f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        tx = x1 + (cell_size - text_size[0]) // 2
        ty = y1 + (cell_size + text_size[1]) // 2
        # White text on dark cells, black on bright
        text_color = (0, 0, 0) if norm > 0.5 else (255, 255, 255)
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        # Index label
        cv2.putText(canvas, f"#{c['idx']}", (x1 + 3, y1 + 14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Title
    if title:
        cv2.putText(canvas, title, (pad, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    # Color scale bar
    bar_y = img_h - 30
    bar_h = 15
    for i in range(img_w - 2*pad):
        norm_i = i / (img_w - 2*pad)
        r = int(255 * min(1, norm_i * 2))
        g = int(255 * min(1, abs(norm_i - 0.5) * 2))
        b = int(255 * max(0, 1 - norm_i * 2))
        canvas[bar_y:bar_y+bar_h, pad+i] = (b, g, r)
    cv2.putText(canvas, f"{vmin:.0f}", (pad, bar_y - 3),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(canvas, f"{vmax:.0f}", (img_w - pad - 30, bar_y - 3),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imwrite(output_path, canvas)
    return output_path


# ═══════════════════════════════════════════════════════════════
#  单张图片处理
# ═══════════════════════════════════════════════════════════════

def process_single(image_path, n_rows, n_cols, output_dir=None):
    """Interactive grid crop + analyze + visualize for one slide image."""
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_path) or ".", f"{image_name}_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Interactive grid crop
    cells = interactive_grid_crop(image_path, n_rows, n_cols)
    if cells is None:
        print("Cancelled.")
        return None

    # ── Save grid overlay (框选识别结果) ──
    original = cv2.imread(image_path)
    overlay = original.copy()
    # Yellow rectangle = grid area
    xs0, ys0 = cells[0]["position"][0], cells[0]["position"][1]
    last = cells[-1]["position"]
    xe0, ye0 = last[0] + last[2], last[1] + last[3]
    cv2.rectangle(overlay, (xs0, ys0), (xe0, ye0), (0, 255, 255), 3)
    # Green grid lines + cell numbers
    for c in cells:
        xs, ys, cw, ch = c["position"]
        cv2.rectangle(overlay, (xs, ys), (xs+cw, ys+ch), (0, 255, 0), 1)
        cv2.putText(overlay, str(c["idx"]), (xs + cw//2 - 12, ys + ch//2 + 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    overlay_path = os.path.join(output_dir, f"{image_name}_grid_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"  Grid overlay: {overlay_path}")

    # Step 2: Analyze each cell (ImageJ-precise)
    results = []
    for c in cells:
        gray = to_8bit_grayscale(c["image"])
        inv = invert(gray)
        m = measure(inv)
        m.update({"idx": c["idx"], "row": c["row"], "col": c["col"]})
        results.append(m)

        # Save cropped cell images
        cv2.imwrite(os.path.join(output_dir, f"cell_{c['idx']:02d}.png"), c["image"])

    # Step 3: Generate heatmap
    heatmap_path = os.path.join(output_dir, f"{image_name}_heatmap.png")
    generate_heatmap(results, heatmap_path, title=image_name)

    # Step 4: Save data
    csv_path = os.path.join(output_dir, f"{image_name}_results.csv")
    with open(csv_path, "w") as f:
        f.write("cell,row,col,mean,median,std,min,max,int_den,area_px\n")
        for r in results:
            f.write(f"{r['idx']},{r['row']},{r['col']},{r['mean']},{r['median']},"
                    f"{r['std']},{r['min']},{r['max']},{r['int_den']},{r['area_px']}\n")

    json_path = os.path.join(output_dir, f"{image_name}_results.json")
    with open(json_path, "w") as f:
        json.dump({"image": image_name, "grid": f"{n_rows}x{n_cols}", "cells": results}, f, indent=2)

    # Step 5: Print
    print(f"\n{'='*60}")
    print(f"  {image_name} — {n_rows}×{n_cols} = {n_rows*n_cols} cells")
    print(f"{'='*60}")
    print(f"  {'Cell':>5s} {'Row':>4s} {'Col':>4s} {'Mean':>8s} {'Median':>8s} {'Std':>8s}")
    print(f"  {'─'*5} {'─'*4} {'─'*4} {'─'*8} {'─'*8} {'─'*8}")
    for r in results:
        print(f"  {r['idx']:>5d} {r['row']:>4d} {r['col']:>4d} {r['mean']:>8.1f} {r['median']:>8.1f} {r['std']:>8.1f}")
    print(f"\n  Heatmap: {heatmap_path}")
    print(f"  Data:    {csv_path}")
    print(f"  Cells:   {output_dir}/cell_*.png")

    # ── Pop up results window: grid overlay + heatmap side by side ──
    heatmap_img = cv2.imread(heatmap_path)
    ov_img = cv2.imread(overlay_path)
    # Resize heatmap to match overlay height
    if ov_img is not None and heatmap_img is not None:
        scale_h = ov_img.shape[0] / heatmap_img.shape[0]
        new_w = int(heatmap_img.shape[1] * scale_h)
        hm_resized = cv2.resize(heatmap_img, (new_w, ov_img.shape[0]))
        side_by_side = np.hstack([ov_img, hm_resized])
        cv2.imshow(f"RESULTS — {image_name}  Left: grid overlay  Right: heatmap  (any key to close)", side_by_side)
    else:
        cv2.imshow(f"RESULTS — {image_name} (press any key to close)", heatmap_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results


# ═══════════════════════════════════════════════════════════════
#  多组对比
# ═══════════════════════════════════════════════════════════════

def compare_groups(folder_path, n_rows, n_cols):
    """
    Process all slide images in a folder, then generate a comparison table
    showing per-cell means across all slides side-by-side.
    """
    images = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) +
                   glob.glob(os.path.join(folder_path, "*.JPG")) +
                   glob.glob(os.path.join(folder_path, "*.png")) +
                   glob.glob(os.path.join(folder_path, "*.PNG")))

    if len(images) < 2:
        print("Need at least 2 images for comparison")
        return

    print(f"\n{'='*60}")
    print(f"  MULTI-GROUP COMPARISON — {len(images)} slides, {n_rows}×{n_cols}")
    print(f"{'='*60}")

    all_slides = {}
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n  Processing: {name}")
        cells = interactive_grid_crop(img_path, n_rows, n_cols)
        if cells is None:
            print(f"  Skipped {name}")
            continue

        slide_results = []
        for c in cells:
            gray = to_8bit_grayscale(c["image"])
            inv = invert(gray)
            m = measure(inv)
            m.update({"idx": c["idx"], "row": c["row"], "col": c["col"]})
            slide_results.append(m)
        all_slides[name] = slide_results

        # Individual heatmap
        out_dir = os.path.join(folder_path, f"{name}_analysis")
        os.makedirs(out_dir, exist_ok=True)
        generate_heatmap(slide_results, os.path.join(out_dir, f"{name}_heatmap.png"), title=name)

    if len(all_slides) < 2:
        print("Not enough slides processed for comparison")
        return

    # Build comparison table
    n_cells = n_rows * n_cols
    slide_names = list(all_slides.keys())

    # Print table
    header = f"  {'Cell':>5s}"
    for name in slide_names:
        header += f" {name[:12]:>12s}"
    print(f"\n{'='*60}")
    print(f"  COMPARISON TABLE (Mean values)")
    print(f"{'='*60}")
    print(header)
    print(f"  {'─'*5} " + "─" * (13 * len(slide_names)))

    for idx in range(1, n_cells + 1):
        row_str = f"  {idx:>5d}"
        for name in slide_names:
            val = all_slides[name][idx-1]["mean"]
            row_str += f" {val:>12.1f}"
        print(row_str)

    # Save comparison CSV
    csv_path = os.path.join(folder_path, "comparison_results.csv")
    with open(csv_path, "w") as f:
        f.write("cell," + ",".join(slide_names) + "\n")
        for idx in range(1, n_cells + 1):
            vals = [str(all_slides[name][idx-1]["mean"]) for name in slide_names]
            f.write(f"{idx}," + ",".join(vals) + "\n")

    # Generate comparison heatmap (stacked vertically)
    heatmaps = []
    for name in slide_names:
        hmap_path = os.path.join(folder_path, f"{name}_analysis", f"{name}_heatmap.png")
        if os.path.exists(hmap_path):
            hm = cv2.imread(hmap_path)
            # Add slide name label
            cv2.putText(hm, name[:30], (5, hm.shape[0]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            heatmaps.append(hm)

    if heatmaps:
        max_w = max(h.shape[1] for h in heatmaps)
        padded = []
        for h in heatmaps:
            if h.shape[1] < max_w:
                pad = np.zeros((h.shape[0], max_w - h.shape[1], 3), dtype=np.uint8)
                h = np.hstack([h, pad])
            padded.append(h)
        comparison_img = np.vstack(padded)
        cv2.imwrite(os.path.join(folder_path, "comparison_heatmaps.png"), comparison_img)
        print(f"\n  Comparison heatmaps: {folder_path}/comparison_heatmaps.png")

    print(f"\n  Comparison CSV: {csv_path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Coagulation Full Workflow")
    parser.add_argument("path", help="Image file or folder (with --compare)")
    parser.add_argument("--rows", type=int, default=3, help="Grid rows")
    parser.add_argument("--cols", type=int, default=6, help="Grid columns")
    parser.add_argument("--compare", action="store_true",
                        help="Multi-group comparison mode (path = folder)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    if args.compare:
        if not os.path.isdir(args.path):
            print("--compare requires a folder path")
            sys.exit(1)
        compare_groups(args.path, args.rows, args.cols)
    elif os.path.isdir(args.path):
        print("For folders, use --compare for multi-group comparison")
        print("Or pass a single image file for interactive grid crop")
    else:
        process_single(args.path, args.rows, args.cols, args.output_dir)


if __name__ == "__main__":
    main()
