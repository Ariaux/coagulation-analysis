# Coagulation Quantification Pipeline

An automated implementation of the standard ImageJ workflow for coagulation assay analysis: **8-bit conversion → invert → region-of-interest measurement**.

---

## Installation

```bash
pip3 install opencv-python numpy
```

## Quick Start

```bash
cd /Users/aria/Downloads/chenmeixi
python3 full_workflow.py input/image.JPG --rows 3 --cols 6
```

---

## Workflow

### Step 1 — ROI Specification

A window displaying the slide image will appear.

| Action | Result |
|--------|--------|
| Click and drag | Draw a bounding rectangle enclosing all sample squares (exclude scale bars or extraneous markings) |
| Release mouse | Grid overlay preview is rendered |
| **Space** | Confirm and proceed |

### Step 2 — Grid Alignment Refinement

A preview window displays the grid overlay (green lines) superimposed on the original image, with each cell numbered.

| Key | Function |
|-----|----------|
| **↑ ↓ ← →** | Nudge the bounding rectangle by n pixels |
| **+ / −** | Increase / decrease the nudge step size |
| **Enter** | Confirm and begin analysis |
| **Esc** | Discard and return to Step 1 |

### Step 3 — Automated Analysis

Upon confirmation, the following operations are executed automatically:

1. Extraction of each grid cell as an individual image
2. Conversion to 8-bit grayscale using the ImageJ-equivalent formula:  
   `gray = 0.299·R + 0.587·G + 0.114·B`
3. Inversion: `I′ = 255 − I` (coagulation signal becomes positive)
4. Computation of descriptive statistics per cell

### Step 4 — Results Display

A results window opens showing the grid overlay alongside the heatmap. Press **any key** to dismiss.

---

## Output

All results are saved to a subdirectory named `<image>_analysis/`, co-located with the input image:

```
input/
├── image.JPG                    # Original micrograph
└── image_analysis/
    ├── image_grid_overlay.png   # Annotated ROI: bounding rectangle, grid lines, cell indices
    ├── image_heatmap.png        # Heatmap (blue = low, red = high coagulation), with per-cell mean values
    ├── image_results.csv        # Tabular data (comma-separated, Excel-compatible)
    ├── image_results.json       # Structured data (machine-readable)
    └── cell_01.png … cell_N.png # Extracted individual cell images
```

## Metrics

All statistics are computed on the **inverted** image (255 − grayscale). Higher values indicate greater coagulation.

| Column | ImageJ Equivalent | Description |
|--------|-------------------|-------------|
| `mean` | Mean | Arithmetic mean of pixel intensities within the cell — **primary endpoint** |
| `median` | — | Median pixel intensity |
| `std` | StdDev | Standard deviation; larger values indicate greater heterogeneity |
| `min` | Min | Minimum pixel intensity |
| `max` | Max | Maximum pixel intensity |
| `int_den` | IntDen | Integrated density (mean × pixel count) |
| `area_px` | Area | Cell area in pixels |

## Group Comparison

For multi-slide experiments (e.g., treatment vs. control):

```bash
python3 full_workflow.py input/ --compare --rows 3 --cols 6
```

Each slide is processed interactively in sequence. The script then generates:

- `comparison_results.csv` — per-cell mean values across all slides in a single table
- `comparison_heatmaps.png` — vertically stacked heatmaps for visual comparison

## Processing Individual Cell Images

If cells have already been manually cropped:

```bash
python3 analyze.py input/ --batch   # Process all images in directory
python3 analyze.py input/ --watch   # Monitor directory; auto-process new images
```

## Grayscale Conversion Fidelity

The grayscale conversion formula is identical to ImageJ (Fiji):

```
I_gray(x,y) = 0.299 × R(x,y) + 0.587 × G(x,y) + 0.114 × B(x,y)
I_inv(x,y)  = 255 − I_gray(x,y)
```

Values produced by this pipeline are directly comparable to those obtained through manual ImageJ operation.

---

## Repository Structure

| File | Purpose |
|------|---------|
| `full_workflow.py` | **Primary tool**: interactive grid cropping + analysis + heatmap visualization |
| `analyze.py` | Direct measurement of pre-cropped single-cell images |
| `coagulation_analysis.py` | Legacy: fully automated slide detection and grid division |
| `imagej_workflow.ijm` | ImageJ macro (alternative to the Python pipeline) |
