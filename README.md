# Coagulation Quantification Pipeline

Automated coagulation assay image analysis with **classical computer vision** and **deep learning** methods.

---

## Project Architecture

```
                          ┌─────────────────┐
                          │   Slide Image    │
                          └────────┬────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼                             ▼
          ┌─────────────────┐           ┌─────────────────┐
          │  Classical CV    │           │  CoagNet DL      │
          │  (Otsu + Morph)  │           │  (Multi-Task)    │
          └────────┬────────┘           └────────┬────────┘
                   │                             │
          ┌────────┼────────┐          ┌─────────┼─────────┐
          ▼        ▼        ▼          ▼         ▼         ▼
       Grid     Heatmap   CSV       Seg Mask   Intensity  Grade
      Overlay                        (pixel)   (0-255)   (3-class)
```

### CoagNet Architecture

```
┌───────────────────────────────────────────────────────┐
│   Input: Cell Image (224×224×3)                       │
│          │                                            │
│   ┌──────▼──────────────────────────────────────┐     │
│   │     Shared Encoder (ResNet-50, ImageNet)     │     │
│   │     C1 → C2 → C3 → C4 → C5                  │     │
│   │     (skip connections to U-Net decoder)      │     │
│   └──────┬──────────────┬──────────────┬────────┘     │
│          │              │              │              │
│   ┌──────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐      │
│   │ U-Net       │ │ Regression│ │Classification│      │
│   │ Decoder     │ │ Head (MLP)│ │ Head (MLP)    │      │
│   └──────┬──────┘ └─────┬─────┘ └──────┬──────┘      │
│          │              │              │              │
│   ┌──────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐      │
│   │ Coag Mask   │ │ Intensity │ │ Severity    │      │
│   │ (binary)    │ │ (scalar)  │ │ (mild/mod/  │      │
│   │             │ │           │ │  severe)    │      │
│   └─────────────┘ └───────────┘ └─────────────┘      │
│                                                       │
│   Loss: L = L_seg/2σ₁² + L_reg/2σ₂²                  │
│           + L_cls/σ₃² + log(σ₁σ₂σ₃)                  │
│   (Kendall Uncertainty Weighting, CVPR 2018)          │
└───────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
cd coagulation-analysis

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Lightweight: classical CV only
pip install opencv-python numpy gradio

# Full: including deep learning
pip install -r requirements_dl.txt
```

### 2. Classical CV Pipeline (no GPU needed)

```bash
# Interactive desktop GUI — drag a rectangle, auto grid analysis
python3 full_workflow.py slide.jpg --rows 3 --cols 6

# Batch process pre-cropped cell images
python3 analyze.py folder/ --batch

# Watch mode — auto-analyze new images as they appear
python3 analyze.py folder/ --watch
```

### 3. Deep Learning Training (GPU recommended)

```bash
# Basic training — needs cell_*.png images from the classical pipeline
python3 train_dl.py --cell-dir input/slide1_analysis

# Full training with custom settings
python3 train_dl.py \
  --cell-dir input/slide1_analysis \
  --encoder resnet50 \         # or: efficientnet-b3, convnext_tiny
  --image-size 224 \
  --phase1-epochs 10 \         # stage 1: frozen encoder, train decoder only
  --phase2-epochs 50 \         # stage 2: unfreeze all, joint multi-task
  --batch-size 8 \
  --lr 3e-4 \
  --device cuda
```

Models are saved to `dl/checkpoints/best_model.pt`. TensorBoard logs in `dl/logs/`.

### 4. Deep Learning Inference

```python
from dl.inference import CoagInference
import cv2

# Load trained model
infer = CoagInference("dl/checkpoints/best_model.pt")

# Single image prediction
img = cv2.imread("cell_01.png")
results = infer.predict(img)
print(results["cls_name"])      # Severity: Mild / Moderate / Severe
print(results["reg_value"])     # Coagulation intensity (0-255)
print(results["coag_ratio"])    # Fraction of pixels classified as coagulation

# 4-panel visualization
infer.visualize(img, save_path="output.png", show=True)

# DL vs classical CV comparison
comparison = infer.compare_with_classical(img)
```

---

## Training Pipeline

### Zero-Annotation Pseudo-Label Generation

```
Raw Cell Image
      │
      ▼
┌─────────────────────┐
│ 1. 8-bit Grayscale   │  gray = 0.299R + 0.587G + 0.114B
│    (ImageJ formula)  │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 2. Invert            │  inverted = 255 - gray
│    (coag → bright)   │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 3. Otsu Auto-Thresh  │  Binary mask (segmentation label)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 4. Morph Cleanup     │  Close + Open (9×9 elliptical kernel)
└─────────┬───────────┘
          ▼
┌─────────────────────────────────────────┐
│          Three Pseudo-Labels             │
│  • seg_mask:  binary (0/1)              │
│  • reg_value: mean intensity (0-255)    │
│  • cls_label: tertile → mild/mod/severe  │
└─────────────────────────────────────────┘
```

### Two-Phase Training Strategy

```
Phase 1: Segmentation Pre-training (10 epochs)
┌────────────────────────────────────┐
│  Encoder (frozen ❄️)               │
│       │                            │
│       ├── Decoder (training 🔥)     │
│       └── Seg Head (training 🔥)    │
│                                    │
│  Loss: Dice + BCE (segmentation)   │
│  LR:   1e-3                        │
│  Goal: decoder learns coagulation  │
│        feature hierarchy           │
└────────────────────────────────────┘
                │
                ▼
Phase 2: Joint Multi-Task Training (50 epochs)
┌────────────────────────────────────┐
│  Encoder (unfrozen 🔥)             │
│       │                            │
│       ├── Decoder + Seg Head 🔥    │
│       ├── Regression Head 🔥       │
│       └── Classification Head 🔥   │
│                                    │
│  Loss: uncertainty-weighted        │
│  LR:   3e-4, cosine warmup         │
│  Goal: all three tasks co-optimize │
└────────────────────────────────────┘
```

### Data Augmentation

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| `HorizontalFlip` | p=0.5 | Mirror invariance |
| `VerticalFlip` | p=0.5 | Rotation invariance |
| `RandomRotate90` | p=0.5 | Orientation generalization |
| `ElasticTransform` | α=120, σ=15 | Simulate tissue deformation |
| `ColorJitter` | 0.2 | Lighting variation robustness |
| `GaussNoise` | σ∈[0.01, 0.05] | Sensor noise simulation |
| `Normalize` | ImageNet μ,σ | Transfer learning |

---

## Workflow Steps (Classical CV)

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

### Classical CV

Results are saved to `<image>_analysis/`:

| File | Content |
|------|---------|
| `*_grid_overlay.png` | Annotated ROI with grid lines and cell indices |
| `*_heatmap.png` | Heatmap (blue = low, red = high coagulation) with mean values |
| `*_results.csv` | Per-cell statistics (Excel-compatible) |
| `*_results.json` | Structured machine-readable data |
| `cell_*.png` | Extracted individual cell images |

### Deep Learning

| File | Content |
|------|---------|
| `dl/checkpoints/best_model.pt` | Best model weights (by val Dice) |
| `dl/checkpoints/phase1_final.pt` | After Phase 1 completion |
| `dl/checkpoints/final_model.pt` | Final model after Phase 2 |
| `dl/logs/<timestamp>/` | TensorBoard training logs |

---

## Metrics

Computed on the **inverted** image (255 − grayscale). Higher values = greater coagulation.

| Metric | Formula / Method | Description |
|--------|-----------------|-------------|
| **Mean** | Σpixel / N | Average coagulation intensity — primary endpoint |
| **Median** | Pixel median | Robust to outliers |
| **Std** | Pixel std dev | Coagulation heterogeneity |
| **IntDen** | Mean × Area | Total integrated density |
| **Dice Score** (DL) | 2\|P∩T\|/(\|P\|+\|T\|) | Segmentation accuracy (0-1) |
| **Coag Ratio** (DL) | Segmented pixels / total | Coagulation area fraction |
| **Grade** (DL) | mild / moderate / severe | 3-class severity classification |

---

## Project Files

```
coagulation-analysis/
├── README.md                  # This document
├── requirements.txt           # Classical CV dependencies (lightweight)
├── requirements_dl.txt        # Full dependencies (including deep learning)
├── run_app.sh                 # Gradio web app launcher
│
├── full_workflow.py           # Classical CV: interactive GUI + analysis + heatmap
├── analyze.py                 # Classical CV: CLI batch processing
├── app.py                     # Gradio web interface (Hugging Face Space)
├── app_standalone.py          # Standalone desktop app (PyInstaller)
├── imagej_workflow.ijm        # ImageJ/Fiji macro
│
├── dl/                        # Deep learning module
│   ├── config.py              #   Centralized config (data/model/loss/training)
│   ├── data.py                #   Pseudo-label generation + augmentation + Dataset
│   ├── model.py               #   CoagNet multi-task network architecture
│   ├── losses.py              #   Dice Loss + Kendall uncertainty weighting
│   ├── train.py               #   Two-phase training loop
│   └── inference.py           #   Inference + visualization + CV comparison
│
├── train_dl.py                # DL training entry point
│
└── input/                     # Example data
    └── *_analysis/            #   Analysis output + cell images
```

---

## CLI Reference

### `train_dl.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--cell-dir` | `input/` | Directory containing cell_*.png images |
| `--encoder` | `resnet50` | Backbone: resnet34/50/101, efficientnet-b0/b3 |
| `--image-size` | `224` | Input resolution (pixels) |
| `--phase1-epochs` | `10` | Segmentation pre-training epochs |
| `--phase2-epochs` | `50` | Joint multi-task training epochs |
| `--batch-size` | `4` | Training batch size |
| `--lr` | `3e-4` | Learning rate (Phase 2) |
| `--device` | `auto` | cuda / cpu / auto |
| `--no-amp` | `False` | Disable automatic mixed precision |
| `--save-dir` | `dl/checkpoints` | Model save path |
| `--log-dir` | `dl/logs` | TensorBoard log path |

### `full_workflow.py`

| Argument | Description |
|----------|-------------|
| `path` | Image file path |
| `--rows` | Grid rows (default 3) |
| `--cols` | Grid columns (default 6) |
| `--compare` | Multi-group comparison mode (path = folder) |
| `--output-dir` | Output directory |

### `analyze.py`

| Argument | Description |
|----------|-------------|
| `path` | Image file or folder |
| `--batch` | Process all images in folder |
| `--watch` | Monitor folder, auto-process new images |

---

## References

- Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. *CVPR 2018*.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
- Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Trans. Sys. Man. Cyber.*
