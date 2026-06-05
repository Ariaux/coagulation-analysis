#!/usr/bin/env python3
"""
Coagulation Quantification — Standalone Desktop App
====================================================
Double-click to launch. No browser, no network, no installation needed.
"""
import sys, os, json, tempfile, webbrowser
import numpy as np
import cv2

# ═══════════════════════════════════════════════════
#  ImageJ-Precision Analysis
# ═══════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: drag an image onto this app icon, or:")
        print("  coagulation_app image.jpg")
        # Also try opening a file dialog
        from tkinter import Tk, filedialog
        root = Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title="Select a slide image",
            filetypes=[("Images", "*.jpg *.JPG *.jpeg *.png *.PNG")]
        )
        root.destroy()
        if not path:
            print("No file selected.")
            sys.exit(1)
    else:
        path = sys.argv[1]

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    img = cv2.imread(path)
    if img is None:
        print(f"Cannot open: {path}")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(os.path.dirname(path) or ".", f"{base_name}_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Auto-detect slide ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary)/255 < 0.5: binary = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    sx, sy, sw, sh = 0, 0, w, h
    if contours:
        largest = max(contours, key=cv2.contourArea)
        sx, sy, sw, sh = cv2.boundingRect(largest)
        if sw < w*0.1 or sh < h*0.1: sx, sy, sw, sh = 0, 0, w, h

    # ── Step 2: Grid settings dialog ──
    n_rows, n_cols = 3, 6
    try:
        from tkinter import Tk, Frame, Label, Spinbox, Button
        root = Tk(); root.title("Grid Settings")
        root.geometry("250x120")
        Label(root, text="Grid dimensions:").pack(pady=5)
        f = Frame(root); f.pack()
        Label(f, text="Rows:").pack(side="left")
        sb_r = Spinbox(f, from_=1, to=10, width=5); sb_r.pack(side="left", padx=5)
        Label(f, text="Cols:").pack(side="left")
        sb_c = Spinbox(f, from_=1, to=10, width=5); sb_c.pack(side="left", padx=5)
        sb_r.delete(0); sb_r.insert(0, "3")
        sb_c.delete(0); sb_c.insert(0, "6")
        def ok(): root.quit()
        Button(root, text="OK", command=ok).pack(pady=5)
        root.mainloop()
        n_rows = int(sb_r.get())
        n_cols = int(sb_c.get())
        root.destroy()
    except Exception:
        pass  # use defaults if tkinter fails

    # ── Step 3: Interactive crop confirmation ──
    shift = 2
    cell_w, cell_h = sw//n_cols, sh//n_rows
    while True:
        overlay = img.copy()
        cv2.rectangle(overlay, (sx,sy), (sx+sw,sy+sh), (0,255,255), 3)
        for r in range(n_rows):
            for c in range(n_cols):
                xs, ys = sx+c*cell_w, sy+r*cell_h
                idx = r*n_cols + c + 1
                cv2.rectangle(overlay, (xs,ys), (xs+cell_w,ys+cell_h), (0,255,0), 1)
                cv2.putText(overlay, str(idx), (xs+cell_w//2-12, ys+cell_h//2+6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        disp = cv2.resize(overlay, (min(w,1400), int(min(w,1400)/w*h))) if max(w,h)>1400 else overlay.copy()
        cv2.putText(disp, "Arrows=nudge  +/-=zoom  Enter=confirm  Esc=cancel",
                   (10, disp.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
        cv2.imshow("Confirm Grid — Arrows: nudge, Enter: confirm, Esc: cancel", disp)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == 13: break  # Enter
        if key == 27: sys.exit(0)  # Esc
        if key == 81: sx = max(0, sx-shift)  # left
        if key == 83: sx = min(w-sw, sx+shift)  # right
        if key == 82: sy = max(0, sy-shift)  # up
        if key == 84: sy = min(h-sh, sy+shift)  # down
        if key in (43,61): shift = min(shift*2, 50)  # +
        if key == 45: shift = max(shift//2, 1)  # -
        cell_w, cell_h = sw//n_cols, sh//n_rows

    # ── Step 4: Analysis ──
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

    # Overlay
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

    # ── Step 5: Show results ──
    overlay_small = cv2.resize(overlay_final, (min(w,800), int(min(w,800)/w*h))) if max(w,h)>800 else overlay_final.copy()
    hm_small = cv2.resize(hm, (overlay_small.shape[1], int(overlay_small.shape[1]/hm.shape[1]*hm.shape[0]))) if hm.shape[1] != overlay_small.shape[1] else hm.copy()
    combined = np.vstack([overlay_small, hm_small]) if overlay_small.shape[1] == hm_small.shape[1] else overlay_small

    cv2.imshow(f"Results — {base_name} (press any key to close)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Open output folder
    os.system(f'open "{out_dir}"')

    print(f"Done! Results: {out_dir}")
    print(f"  {base_name}_grid_overlay.png")
    print(f"  {base_name}_heatmap.png")
    print(f"  {base_name}_results.csv")
    for r in results:
        print(f"  Cell {r['idx']:2d}: mean={r['mean']:.1f}  std={r['std']:.1f}")


if __name__ == "__main__":
    main()
