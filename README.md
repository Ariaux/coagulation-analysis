# Coagulation Quantification Pipeline

Automated ImageJ workflow for coagulation assay analysis: **8-bit conversion → invert → region-of-interest measurement**.

---

## Quick Start

### Option A — Hugging Face Web App (no installation)

**https://huggingface.co/spaces/Yilunaria/coagulation-quantification**

Upload a slide photo, set grid dimensions, click **Analyze**. Works in any browser.

### Option B — Desktop GUI (interactive grid cropping)

```bash
pip install opencv-python numpy
python3 full_workflow.py slide.jpg --rows 3 --cols 6
```

Drag a rectangle around all sample squares in the pop-up window, fine-tune with arrow keys, press Enter. Heatmap and CSV are generated automatically.

### Option C — Command Line (pre-cropped cell images)

```bash
python3 analyze.py folder/ --batch   # process all images
python3 analyze.py folder/ --watch   # auto-process new images
```

---

## Workflow Steps

The pipeline replicates the standard ImageJ procedure:

| Step | Operation | ImageJ Equivalent |
|------|-----------|-------------------|
| 1 | Load image | File > Open |
| 2 | Detect or specify ROI | Rectangle tool |
| 3 | Divide into grid | — |
| 4 | Convert to 8-bit grayscale | Image > Type > 8-bit |
| 5 | Invert | Edit > Invert |
| 6 | Measure per cell | Analyze > Measure |

### Grayscale Conversion Fidelity

```
I_gray(x,y) = 0.299·R + 0.587·G + 0.114·B
I_inv(x,y)  = 255 − I_gray(x,y)
```

Identical to ImageJ (Fiji). Values are directly comparable to manual operation.

---

## Output

Results are saved to `<image>_analysis/`:

| File | Content |
|------|---------|
| `*_grid_overlay.png` | Annotated ROI with grid lines and cell indices |
| `*_heatmap.png` | Heatmap (blue = low, red = high coagulation) with mean values |
| `*_results.csv` | Per-cell statistics (Excel-compatible) |
| `*_results.json` | Structured machine-readable data |
| `cell_*.png` | Extracted individual cell images |

---

## Metrics

Computed on the **inverted** image (255 − grayscale). Higher values indicate greater coagulation.

| Column | ImageJ Term | Description |
|--------|-------------|-------------|
| `mean` | Mean | Arithmetic mean of pixel intensities — **primary endpoint** |
| `median` | — | Median pixel intensity |
| `std` | StdDev | Standard deviation; larger values indicate greater heterogeneity |
| `min` / `max` | Min / Max | Minimum and maximum pixel intensity |
| `int_den` | IntDen | Integrated density (mean × pixel count) |
| `area_px` | Area | Cell area in pixels |

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Hugging Face / Gradio web interface |
| `full_workflow.py` | Desktop GUI: interactive grid cropping + analysis + heatmap |
| `analyze.py` | Command-line: direct measurement of pre-cropped cell images |
| `imagej_workflow.ijm` | ImageJ macro: 8-bit → invert → rectangle → measure |
| `requirements.txt` | Python dependencies |
| `run_app.sh` | Launch script for local Gradio server |
| `input/` | Example image and analysis results |
