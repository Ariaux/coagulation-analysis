#!/usr/bin/env python3
"""
Coagulation Quantification — Standalone Desktop App
====================================================
Usage: drag an image file onto the app icon, or:
       CoagulationAnalysis.exe image.jpg
"""
import sys, os, json, traceback
import numpy as np
import cv2

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


def heatmap_image(results, n_rows, n_cols):
    means = [r["mean"] for r in results]
    vmin, vmax = min(means), max(means)
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


def pick_grid(n_rows, n_cols, w, h, sw, sh):
    """Show a small OpenCV window to let user pick grid rows/cols."""
    canvas = np.full((400, 500, 3), 40, dtype=np.uint8)
    cv2.putText(canvas, "Grid Settings", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
    cv2.putText(canvas, f"Image: {w}x{h}  ROI: {sw}x{sh}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    cv2.putText(canvas, f"Rows: {n_rows}    Cols: {n_cols}    Cells: {n_rows*n_cols}", (30, 130),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    cv2.putText(canvas, "UP/DOWN: change rows", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(canvas, "LEFT/RIGHT: change cols", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(canvas, "ENTER: confirm   ESC: quit", (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    while True:
        disp = canvas.copy()
        cv2.putText(disp, f"  {n_rows} x {n_cols}  =  {n_rows*n_cols} cells", (120, 350),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Grid Settings", disp)
        key = cv2.waitKey(0) & 0xFF
        if key == 13: break  # Enter
        if key == 27: cv2.destroyAllWindows(); sys.exit(0)  # Esc
        if key == 82 and n_rows > 1: n_rows -= 1  # Up
        if key == 84 and n_rows < 10: n_rows += 1  # Down
        if key == 81 and n_cols > 1: n_cols -= 1  # Left
        if key == 83 and n_cols < 10: n_cols += 1  # Right
    cv2.destroyAllWindows()
    return n_rows, n_cols


def confirm_grid(img, sx, sy, sw, sh, n_rows, n_cols):
    """Show grid overlay, let user nudge with arrow keys, Enter to confirm."""
    shift = 2
    h, w = img.shape[:2]
    while True:
        cell_w, cell_h = sw//n_cols, sh//n_rows
        overlay = img.copy()
        cv2.rectangle(overlay, (sx,sy), (sx+sw,sy+sh), (0,255,255), 3)
        for r in range(n_rows):
            for c in range(n_cols):
                xs, ys = sx+c*cell_w, sy+r*cell_h
                cv2.rectangle(overlay, (xs,ys), (xs+cell_w,ys+cell_h), (0,255,0), 1)
                cv2.putText(overlay, str(r*n_cols+c+1), (xs+cell_w//2-12, ys+cell_h//2+6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        max_dim = 1200
        scale = max_dim/max(w,h) if max(w,h)>max_dim else 1.0
        if scale != 1.0:
            disp = cv2.resize(overlay, (int(w*scale), int(h*scale)))
        else:
            disp = overlay.copy()

        cv2.putText(disp, f"ROI:({sx},{sy}) {sw}x{sh}  Grid:{n_rows}x{n_cols}  Step:{shift}px",
                   (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        cv2.putText(disp, "Arrow=nudge  +/-=step  Enter=confirm  Esc=cancel",
                   (10, disp.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

        cv2.imshow("Confirm Grid - Arrows: nudge, Enter: confirm", disp)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == 13: return sx, sy, sw, sh, n_rows, n_cols  # Enter
        if key == 27: sys.exit(0)  # Esc
        if key == 81: sx = max(0, sx-shift)
        if key == 83: sx = min(w-sw, sx+shift)
        if key == 82: sy = max(0, sy-shift)
        if key == 84: sy = min(h-sh, sy+shift)
        if key in (43,61): shift = min(shift*2, 50)
        if key == 45: shift = max(shift//2, 1)


def main():
    try:
        _main()
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

    img = load_image(path)
    if img is None:
        log("Cannot open image - try renaming to English filename with no spaces")
        input("Cannot open image. Try renaming to a simple name. Press Enter.")
        sys.exit(1)

    h, w = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(os.path.dirname(path) or ".", f"{base_name}_analysis")
    os.makedirs(out_dir, exist_ok=True)
    log(f"Loaded: {w}x{h}px  Output: {out_dir}")

    # ── Auto-detect slide ROI ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary)/255 < 0.5: binary = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sx, sy, sw, sh = 0, 0, w, h
    if contours:
        largest = max(contours, key=cv2.contourArea)
        sx, sy, sw, sh = cv2.boundingRect(largest)
        if sw < w*0.1 or sh < h*0.1: sx, sy, sw, sh = 0, 0, w, h
    log(f"Auto-detected ROI: ({sx},{sy}) {sw}x{sh}")

    # ── Pick grid ──
    n_rows, n_cols = pick_grid(3, 6, w, h, sw, sh)
    log(f"Grid: {n_rows}x{n_cols}")

    # ── Confirm with nudge ──
    sx, sy, sw, sh, n_rows, n_cols = confirm_grid(img, sx, sy, sw, sh, n_rows, n_cols)

    # ── Analyze ──
    cell_w, cell_h = sw//n_cols, sh//n_rows
    results = []
    for r in range(n_rows):
        for c in range(n_cols):
            xs, ys = sx+c*cell_w, sy+r*cell_h
            cell = img[ys:ys+cell_h, xs:xs+cell_w]
            inv = 255 - to_8bit(cell)
            m = measure(inv)
            m.update({"idx": r*n_cols+c+1, "row": r+1, "col": c+1})
            results.append(m)
            cv2.imwrite(os.path.join(out_dir, f"cell_{m['idx']:02d}.png"), cell)

    # Save outputs
    overlay = img.copy()
    cv2.rectangle(overlay, (sx,sy), (sx+sw,sy+sh), (0,255,255), 3)
    for r in range(n_rows):
        for c in range(n_cols):
            xs, ys = sx+c*cell_w, sy+r*cell_h
            cv2.rectangle(overlay, (xs,ys), (xs+cell_w,ys+cell_h), (0,255,0), 1)
            cv2.putText(overlay, str(r*n_cols+c+1), (xs+cell_w//2-12, ys+cell_h//2+6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_grid_overlay.png"), overlay)

    hm = heatmap_image(results, n_rows, n_cols)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_heatmap.png"), hm)

    with open(os.path.join(out_dir, f"{base_name}_results.csv"), "w") as f:
        f.write("cell,row,col,mean,median,std,min,max,int_den,area_px\n")
        for r in results:
            f.write(f"{r['idx']},{r['row']},{r['col']},{r['mean']},{r['median']},"
                    f"{r['std']},{r['min']},{r['max']},{r['int_den']},{r['area_px']}\n")

    with open(os.path.join(out_dir, f"{base_name}_results.json"), "w") as f:
        json.dump({"image": base_name, "cells": results}, f, indent=2)

    log(f"Done. {len(results)} cells analyzed.")

    # Show results
    d1 = cv2.resize(overlay, (min(w,700), int(min(w,700)/w*h))) if max(w,h)>700 else overlay.copy()
    d2 = cv2.resize(hm, (d1.shape[1], d1.shape[1]*hm.shape[0]//hm.shape[1]))
    cv2.imshow(f"Results - {base_name}  (any key to close)", np.vstack([d1,d2]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Open output folder
    if sys.platform == "win32":
        os.startfile(out_dir)
    else:
        os.system(f'open "{out_dir}"')


if __name__ == "__main__":
    main()
