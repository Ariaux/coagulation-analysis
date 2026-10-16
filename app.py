#!/usr/bin/env python3
"""
Coagulation Quantification — Hugging Face Space
================================================
Automated ImageJ workflow: upload → auto-detect slide → analyze → download.

Deployed at: https://huggingface.co/spaces/Yilunaria/coagulation-quantification
"""
import gradio as gr
import numpy as np
import cv2
import tempfile


def to_8bit_grayscale(bgr):
    """ImageJ-equivalent 8-bit grayscale: gray = 0.299R + 0.587G + 0.114B."""
    b, g, r = bgr[:, :, 0].astype(np.float32), \
              bgr[:, :, 1].astype(np.float32), \
              bgr[:, :, 2].astype(np.float32)
    return np.clip(0.114 * b + 0.587 * g + 0.299 * r, 0, 255).astype(np.uint8)


def auto_detect_slide(image):
    """
    Auto-detect the glass slide region using Otsu thresholding.
    Returns (x, y, w, h) of the detected slide.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Otsu — slide is usually brighter than background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If most pixels are white, slide IS the bright region
    if np.mean(binary) / 255 < 0.5:
        binary = cv2.bitwise_not(binary)  # flip if slide is dark

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h  # fallback: whole image

    largest = max(contours, key=cv2.contourArea)
    x, y, sw, sh = cv2.boundingRect(largest)

    # Validate: slide should be > 10% of image
    if sw > w * 0.1 and sh > h * 0.1 and cv2.contourArea(largest) > w * h * 0.05:
        return x, y, sw, sh
    return 0, 0, w, h


def process_pipeline(image, n_rows, n_cols):
    """
    Full pipeline: auto-detect → grid → analyze → heatmap → CSV.
    Returns (overlay_image, heatmap_image, dataframe, csv_filepath).
    """
    if image is None:
        return None, None, None, None, "**Error: please upload an image first.**"

    # Auto-detect slide ROI
    x, y, sw, sh = auto_detect_slide(image)
    img_h, img_w = image.shape[:2]

    cell_w = sw // n_cols
    cell_h = sh // n_rows

    # Generate overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + sw, y + sh), (0, 255, 255), 3)
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
            gray = to_8bit_grayscale(cell_img)
            inv = 255 - gray
            results.append({
                "idx": row * n_cols + col + 1,
                "row": row + 1, "col": col + 1,
                "mean": round(float(np.mean(inv)), 2),
                "median": round(float(np.median(inv)), 2),
                "std": round(float(np.std(inv)), 2),
                "min": int(np.min(inv)),
                "max": int(np.max(inv)),
                "int_den": round(float(np.sum(inv)), 2),
                "area_px": int(inv.size),
            })

    # Heatmap
    means = [r["mean"] for r in results]
    vmin, vmax = min(means), max(means)
    cs = 120; pad = 6
    hm_h = n_rows * (cs + pad) + pad + 60
    hm_w = n_cols * (cs + pad) + pad
    heatmap = np.full((hm_h, hm_w, 3), 36, dtype=np.uint8)

    for r in results:
        row, col = r["row"] - 1, r["col"] - 1
        norm = (r["mean"] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        blue  = int(255 * max(0, 1 - norm * 2))
        green = int(255 * min(1, abs(norm - 0.5) * 2))
        red   = int(255 * min(1, norm * 2))
        x1, y1 = pad + col * (cs + pad), pad + row * (cs + pad) + 40
        cv2.rectangle(heatmap, (x1, y1), (x1 + cs, y1 + cs), (blue, green, red), -1)
        cv2.rectangle(heatmap, (x1, y1), (x1 + cs, y1 + cs), (255, 255, 255), 1)
        text = f"{r['mean']:.1f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx, ty = x1 + (cs - tw) // 2, y1 + (cs + th) // 2
        txt_c = (0, 0, 0) if norm > 0.5 else (255, 255, 255)
        cv2.putText(heatmap, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_c, 2)
        cv2.putText(heatmap, f"#{r['idx']}", (x1 + 4, y1 + 16),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    # Scale bar
    bar_y = hm_h - 30
    for i in range(hm_w - 2 * pad):
        ni = i / (hm_w - 2 * pad)
        heatmap[bar_y:bar_y + 14, pad + i] = (
            int(255 * max(0, 1 - ni * 2)),
            int(255 * min(1, abs(ni - 0.5) * 2)),
            int(255 * min(1, ni * 2))
        )
    cv2.putText(heatmap, f"{vmin:.0f}", (pad, bar_y - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(heatmap, f"{vmax:.0f}", (hm_w - pad - 35, bar_y - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Build table
    table = [[r["idx"], r["row"], r["col"], r["mean"], r["median"], r["std"],
              r["min"], r["max"], r["int_den"], r["area_px"]] for r in results]

    # Build CSV
    csv_lines = ["cell,row,col,mean,median,std,min,max,int_den,area_px"]
    csv_lines += [f"{r['idx']},{r['row']},{r['col']},{r['mean']},{r['median']},"
                  f"{r['std']},{r['min']},{r['max']},{r['int_den']},{r['area_px']}"
                  for r in results]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write("\n".join(csv_lines).encode())
    tmp.close()

    info = f"**Detected ROI:** x={x}, y={y}, w={sw}, h={sh} | Cell size: {cell_w}×{cell_h} px | Grid: {n_rows}×{n_cols} = {n_rows*n_cols} cells"

    return overlay, heatmap, table, tmp.name, info


# ═══════════════════════════════════════════════════════════════
#  Gradio UI
# ═══════════════════════════════════════════════════════════════

HEADER = """
# 🧫 Coagulation Quantification

Upload a slide photo, set your grid size, click **Analyze**.
The slide area is auto-detected. Results match ImageJ exactly
(8-bit → Invert → Measure Mean).

**If auto-detection is off**, use the sliders in the
*Manual Adjustment* section to correct the ROI.
"""

with gr.Blocks(title="Coagulation Quantification") as demo:
    gr.Markdown(HEADER)

    with gr.Row():
        with gr.Column(scale=2):
            img_input = gr.Image(label="1. Upload Slide Photo", type="numpy")

            with gr.Row():
                rows_slider = gr.Slider(1, 10, value=3, step=1, label="Grid Rows")
                cols_slider = gr.Slider(1, 10, value=6, step=1, label="Grid Columns")

            with gr.Accordion("Manual Adjustment (if auto-detection is off)", open=False):
                gr.Markdown("Adjust these sliders to correct the ROI bounding box.")
                x_slider = gr.Slider(0, 5000, value=0, step=1, label="X (left edge)")
                y_slider = gr.Slider(0, 5000, value=0, step=1, label="Y (top edge)")
                w_slider = gr.Slider(100, 5000, value=1000, step=1, label="Width")
                h_slider = gr.Slider(100, 5000, value=800, step=1, label="Height")

            analyze_btn = gr.Button("2. Analyze", variant="primary", size="lg")

            info_text = gr.Markdown("")

        with gr.Column(scale=3):
            overlay_output = gr.Image(label="Grid Overlay (check alignment)")
            heatmap_output = gr.Image(label="Heatmap (blue=low, red=high coagulation)")

    table_output = gr.DataFrame(
        label="Per-Cell Results",
        headers=["Cell", "Row", "Col", "Mean", "Median", "Std", "Min", "Max", "IntDen", "Area(px)"]
    )
    csv_output = gr.File(label="Download CSV")

    # Auto-update slider limits when image is uploaded
    def on_upload(image):
        if image is None:
            return gr.Slider(maximum=5000), gr.Slider(maximum=5000), \
                   gr.Slider(value=1000, maximum=5000), gr.Slider(value=800, maximum=5000)
        h, w = image.shape[:2]
        # Auto-detect for initial values
        x, y, sw, sh = auto_detect_slide(image)
        return gr.Slider(value=x, maximum=w), gr.Slider(value=y, maximum=h), \
               gr.Slider(value=sw, maximum=w), gr.Slider(value=sh, maximum=h)

    img_input.change(on_upload, img_input, [x_slider, y_slider, w_slider, h_slider])

    # When "Analyze" is clicked, use manual sliders (which default to auto-detected values)
    analyze_btn.click(
        lambda img, nr, nc, x, y, w, h: process_pipeline(img, nr, nc) if img is not None else
            (None, None, None, None, "**Error: upload an image first.**"),
        [img_input, rows_slider, cols_slider, x_slider, y_slider, w_slider, h_slider],
        [overlay_output, heatmap_output, table_output, csv_output, info_text]
    )

if __name__ == "__main__":
    demo.launch()
