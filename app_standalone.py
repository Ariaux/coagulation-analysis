#!/usr/bin/env python3
"""
Coagulation Quantification — Standalone Desktop App
====================================================
Double-click to launch.
"""
import sys, os, json, traceback
import numpy as np
import cv2
from datetime import datetime

LOG_FILE = os.path.join(os.path.expanduser("~"), "Desktop", "coagulation_log.txt")

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
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


def main():
    try:
        _main()
    except Exception:
        log(f"FATAL ERROR:\n{traceback.format_exc()}")
        input("Press Enter to exit...")
        sys.exit(1)


def _main():
    log("App started")

    # ── Get image path ──
    if len(sys.argv) < 2:
        log("No image path provided, opening file dialog...")
        try:
            import tkinter.filedialog, tkinter
            root = tkinter.Tk(); root.withdraw()
            path = tkinter.filedialog.askopenfilename(
                title="Select a slide image",
                filetypes=[("Images", "*.jpg *.JPG *.jpeg *.png *.PNG")]
            )
            root.destroy()
        except Exception as e:
            log(f"File dialog failed: {e}")
            path = input("Drag an image file here and press Enter:\n").strip().strip('"').strip("'")
        if not path:
            log("No file selected, exiting")
            sys.exit(0)
    else:
        path = sys.argv[1]

    log(f"Image: {path}")
    if not os.path.exists(path):
        log(f"File not found: {path}")
        input("File not found. Press Enter to exit.")
        sys.exit(1)

    # imread fails on Windows with non-ASCII paths; use numpy fallback
    img = cv2.imread(path)
    if img is None:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            pass
    if img is None:
        log(f"Cannot open image (try renaming file to English path): {path}")
        input("Cannot open image. Try moving it to Desktop and renaming to a simple English name. Press Enter to exit.")
        sys.exit(1)

    h, w = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(os.path.dirname(path) or ".", f"{base_name}_analysis")
    os.makedirs(out_dir, exist_ok=True)
    log(f"Output: {out_dir}")

    # ── Auto-detect slide ──
    log("Detecting slide...")
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
    log(f"Detected ROI: ({sx},{sy}) {sw}x{sh}")

    # ── Grid settings ──
    n_rows, n_cols = 3, 6
    try:
        import tkinter
        root = tkinter.Tk(); root.title("Grid Settings"); root.geometry("300x150")
        tkinter.Label(root, text=f"Image: {w}x{h}  |  ROI: {sw}x{sh}", font=("Arial",10)).pack(pady=5)
        tkinter.Label(root, text="Grid dimensions:", font=("Arial",12)).pack()
        f = tkinter.Frame(root); f.pack(pady=5)
        tkinter.Label(f, text="Rows:").pack(side="left")
        sr = tkinter.Spinbox(f, from_=1, to=10, width=5); sr.pack(side="left", padx=5)
        tkinter.Label(f, text="Cols:").pack(side="left")
        sc = tkinter.Spinbox(f, from_=1, to=10, width=5); sc.pack(side="left", padx=5)
        sr.delete(0); sr.insert(0,"3"); sc.delete(0); sc.insert(0,"6")
        def ok(): root.quit()
        tkinter.Button(root, text="OK", command=ok, width=10).pack(pady=5)
        root.mainloop()
        n_rows = int(sr.get()); n_cols = int(sc.get())
        root.destroy()
        log(f"Grid: {n_rows}x{n_cols}")
    except Exception as e:
        log(f"Grid dialog failed, using defaults: {e}")

    # ── Interactive confirmation ──
    shift = 2
    cell_w, cell_h = sw//n_cols, sh//n_rows
    log("Waiting for user to confirm grid...")

    while True:
        overlay = img.copy()
        cv2.rectangle(overlay, (sx,sy), (sx+sw,sy+sh), (0,255,255), 3)
        for r in range(n_rows):
            for c in range(n_cols):
                xs, ys = sx+c*cell_w, sy+r*cell_h
                cv2.rectangle(overlay, (xs,ys), (xs+cell_w,ys+cell_h), (0,255,0), 1)
                cv2.putText(overlay, str(r*n_cols+c+1), (xs+cell_w//2-12, ys+cell_h//2+6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        max_dim = 1200
        if max(w,h) > max_dim:
            scale = max_dim / max(w,h)
            disp = cv2.resize(overlay, (int(w*scale), int(h*scale)))
        else:
            disp = overlay.copy()

        cv2.putText(disp, "Arrows=nudge  +/-=step  Enter=confirm  Esc=cancel",
                   (10, disp.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        cv2.putText(disp, f"ROI: ({sx},{sy}) {sw}x{sh}px  Grid: {n_rows}x{n_cols}",
                   (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

        cv2.imshow("Grid Confirmation - Arrows: nudge, Enter: confirm", disp)
        log("Grid confirmation window shown, waiting for key...")
        key = cv2.waitKey(0) & 0xFF
        log(f"Key pressed: {key}")
        cv2.destroyAllWindows()

        if key == 13:  # Enter
            log("User confirmed grid")
            break
        if key == 27:  # Esc
            log("User cancelled")
            sys.exit(0)
        if key == 81: sx = max(0, sx-shift)
        if key == 83: sx = min(w-sw, sx+shift)
        if key == 82: sy = max(0, sy-shift)
        if key == 84: sy = min(h-sh, sy+shift)
        if key in (43, 61): shift = min(shift*2, 50)
        if key == 45: shift = max(shift//2, 1)
        cell_w, cell_h = sw//n_cols, sh//n_rows

    # ── Analysis ──
    log("Analyzing cells...")
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

    # Save overlay
    overlay_final = img.copy()
    cv2.rectangle(overlay_final, (sx,sy), (sx+sw,sy+sh), (0,255,255), 3)
    for r in range(n_rows):
        for c in range(n_cols):
            xs, ys = sx+c*cell_w, sy+r*cell_h
            cv2.rectangle(overlay_final, (xs,ys), (xs+cell_w,ys+cell_h), (0,255,0), 1)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_grid_overlay.png"), overlay_final)

    # Heatmap
    hm = heatmap_image(results, n_rows, n_cols)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_heatmap.png"), hm)

    # CSV
    csv_path = os.path.join(out_dir, f"{base_name}_results.csv")
    with open(csv_path, "w") as f:
        f.write("cell,row,col,mean,median,std,min,max,int_den,area_px\n")
        for r in results:
            f.write(f"{r['idx']},{r['row']},{r['col']},{r['mean']},{r['median']},"
                    f"{r['std']},{r['min']},{r['max']},{r['int_den']},{r['area_px']}\n")

    # JSON
    with open(os.path.join(out_dir, f"{base_name}_results.json"), "w") as f:
        json.dump({"image": base_name, "cells": results}, f, indent=2)

    log(f"Analysis done. {len(results)} cells analyzed.")

    # Show results
    disp2 = cv2.resize(overlay_final, (min(w,700), int(min(w,700)/w*h))) if max(w,h)>700 else overlay_final.copy()
    hm2 = cv2.resize(hm, (disp2.shape[1], disp2.shape[1]*hm.shape[0]//hm.shape[1])) if hm.shape[1]!=disp2.shape[1] else hm.copy()
    combined = np.vstack([disp2, hm2])

    cv2.imshow(f"Results - {base_name} (any key to close)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Open folder
    if sys.platform == "darwin":
        os.system(f'open "{out_dir}"')
    elif sys.platform == "win32":
        os.system(f'explorer "{out_dir}"')

    log("Done!")


if __name__ == "__main__":
    main()
