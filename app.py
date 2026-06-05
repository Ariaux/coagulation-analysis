#!/usr/bin/env python3
"""
Coagulation Quantification — Gradio Web App
===========================================
Browser-based interface: upload → draw ROI → analyze → download results.

Usage:
  pip3 install gradio opencv-python numpy
  python3 app.py
  # Open http://127.0.0.1:7860 in browser
  # Share with others: add --share flag
"""
import gradio as gr
import numpy as np
import cv2
import json
import os
from datetime import datetime
import tempfile


def to_8bit_grayscale(bgr):
    """ImageJ-equivalent 8-bit grayscale conversion."""
    b, g, r = bgr[:, :, 0].astype(np.float32), \
              bgr[:, :, 1].astype(np.float32), \
              bgr[:, :, 2].astype(np.float32)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(gray, 0, 255).astype(np.uint8)


def analyze_cell(cell_bgr):
    """Invert and measure a single cell."""
    gray = to_8bit_grayscale(cell_bgr)
    inv = 255 - gray
    return {
        "mean": round(float(np.mean(inv)), 2),
        "median": round(float(np.median(inv)), 2),
        "std": round(float(np.std(inv)), 2),
        "min": int(np.min(inv)),
        "max": int(np.max(inv)),
        "int_den": round(float(np.sum(inv)), 2),
        "area_px": int(inv.size),
    }


def generate_heatmap(results, n_rows, n_cols):
    """Generate a heatmap image with cell mean values."""
    means = [r["mean"] for r in results]
    vmin, vmax = min(means), max(means)
    cell_size = 120
    pad = 6
    h = n_rows * (cell_size + pad) + pad + 60
    w = n_cols * (cell_size + pad) + pad
    canvas = np.full((h, w, 3), 36, dtype=np.uint8)

    for r in results:
        row, col = r["row"] - 1, r["col"] - 1
        norm = (r["mean"] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        # Inferno-like colormap: blue → red → yellow
        red   = int(255 * min(1, norm * 2))
        green = int(255 * min(1, abs(norm - 0.5) * 2))
        blue  = int(255 * max(0, 1 - norm * 2))
        color = (blue, green, red)

        x1, y1 = pad + col * (cell_size + pad), pad + row * (cell_size + pad) + 40
        cv2.rectangle(canvas, (x1, y1), (x1 + cell_size, y1 + cell_size), color, -1)
        cv2.rectangle(canvas, (x1, y1), (x1 + cell_size, y1 + cell_size), (255, 255, 255), 1)

        text = f"{r['mean']:.1f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx, ty = x1 + (cell_size - tw) // 2, y1 + (cell_size + th) // 2
        txt_color = (0, 0, 0) if norm > 0.5 else (255, 255, 255)
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2)
        cv2.putText(canvas, f"#{r['idx']}", (x1 + 4, y1 + 16),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Scale bar
    bar_y = h - 30
    bar_h = 14
    bar_w = w - 2 * pad
    for i in range(bar_w):
        ni = i / bar_w
        r_c = int(255 * min(1, ni * 2))
        g_c = int(255 * min(1, abs(ni - 0.5) * 2))
        b_c = int(255 * max(0, 1 - ni * 2))
        canvas[bar_y:bar_y + bar_h, pad + i] = (b_c, g_c, r_c)
    cv2.putText(canvas, f"{vmin:.0f}", (pad, bar_y - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(canvas, f"{vmax:.0f}", (w - pad - 35, bar_y - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return canvas


def process_image(image, n_rows, n_cols, x, y, w, h):
    """Main analysis pipeline."""
    if image is None:
        return None, None, "Error: no image uploaded"

    img_h, img_w = image.shape[:2]

    # Clamp coordinates
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(10, min(w, img_w - x))
    h = max(10, min(h, img_h - y))

    cell_w = w // n_cols
    cell_h = h // n_rows

    # Generate preview overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 3)
    for row in range(n_rows):
        for col in range(n_cols):
            xs = x + col * cell_w
            ys = y + row * cell_h
            idx = row * n_cols + col + 1
            cv2.rectangle(overlay, (xs, ys), (xs + cell_w, ys + cell_h), (0, 255, 0), 1)
            cv2.putText(overlay, str(idx), (xs + cell_w // 2 - 12, ys + cell_h // 2 + 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Analyze each cell
    results = []
    for row in range(n_rows):
        for col in range(n_cols):
            xs = x + col * cell_w
            ys = y + row * cell_h
            cell_img = image[ys:ys + cell_h, xs:xs + cell_w]
            m = analyze_cell(cell_img)
            m.update({"idx": row * n_cols + col + 1, "row": row + 1, "col": col + 1})
            results.append(m)

    # Generate heatmap
    heatmap = generate_heatmap(results, n_rows, n_cols)

    # Build data table
    table_data = [[r["idx"], r["row"], r["col"], r["mean"], r["median"], r["std"],
                   r["min"], r["max"], r["int_den"], r["area_px"]] for r in results]
    headers = ["Cell", "Row", "Col", "Mean", "Median", "Std", "Min", "Max", "IntDen", "Area(px)"]

    # Build CSV
    csv_lines = ["cell,row,col,mean,median,std,min,max,int_den,area_px"]
    for r in results:
        csv_lines.append(f"{r['idx']},{r['row']},{r['col']},{r['mean']},{r['median']},"
                         f"{r['std']},{r['min']},{r['max']},{r['int_den']},{r['area_px']}")
    csv_content = "\n".join(csv_lines)

    # Save CSV to temp file for download
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(csv_content.encode())
    tmp.close()

    return overlay, heatmap, table_data, headers, tmp.name


# ═══════════════════════════════════════════════════════════════
#  Gradio Interface
# ═══════════════════════════════════════════════════════════════

with gr.Blocks(title="Coagulation Quantification") as demo:
    gr.Markdown("""
    # Coagulation Quantification Pipeline

    Automated ImageJ workflow: **8-bit → Invert → Measure Mean**

    1. Upload your slide image
    2. Set grid dimensions (rows × cols)
    3. Adjust the ROI bounding box via sliders
    4. Preview the grid overlay, then click **Analyze**
    5. Download results as CSV
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(label="Upload Slide Image", type="numpy")

            with gr.Row():
                rows_slider = gr.Slider(1, 10, value=3, step=1, label="Grid Rows")
                cols_slider = gr.Slider(1, 10, value=6, step=1, label="Grid Columns")

            with gr.Accordion("ROI Bounding Box", open=True):
                x_slider = gr.Slider(0, 5000, value=0, step=1, label="X (left)")
                y_slider = gr.Slider(0, 5000, value=0, step=1, label="Y (top)")
                w_slider = gr.Slider(100, 5000, value=1000, step=1, label="Width")
                h_slider = gr.Slider(100, 5000, value=800, step=1, label="Height")

            analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Column(scale=1):
            overlay_output = gr.Image(label="Grid Overlay Preview")
            heatmap_output = gr.Image(label="Heatmap (Blue=low, Red=high coagulation)")

    with gr.Row():
        table_output = gr.DataFrame(label="Per-Cell Results", headers=["Cell", "Row", "Col",
            "Mean", "Median", "Std", "Min", "Max", "IntDen", "Area(px)"])

    csv_output = gr.File(label="Download CSV")

    # Update slider max values when image is uploaded
    def update_sliders(image):
        if image is None:
            return gr.Slider(maximum=5000), gr.Slider(maximum=5000), \
                   gr.Slider(value=1000, maximum=5000), gr.Slider(value=800, maximum=5000)
        h, w = image.shape[:2]
        return gr.Slider(maximum=w), gr.Slider(maximum=h), \
               gr.Slider(value=w, maximum=w), gr.Slider(value=h, maximum=h)

    img_input.change(update_sliders, img_input, [x_slider, y_slider, w_slider, h_slider])

    # Analyze
    analyze_btn.click(
        process_image,
        [img_input, rows_slider, cols_slider, x_slider, y_slider, w_slider, h_slider],
        [overlay_output, heatmap_output, table_output, csv_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", theme=gr.themes.Soft())
