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

## Windows 离线网站使用说明

推荐使用网站版入口 `Web/StartWebsite.exe`。它不需安装 Python，也不需
连接外网。下载 Windows ZIP 后必须先完整解压，正确的目录结构是：

```text
CoagulationAnalysis-Windows/
├── Web/
│   ├── StartWebsite.exe
│   └── _internal/
├── Desktop/
│   ├── CoagulationAnalysis.exe
│   └── _internal/
└── README.md
```

整个 `CoagulationAnalysis-Windows` 目录必须保持完整。不要单独移动 EXE，
不要删除或重命名 `_internal` 目录。

### 启动网站

1. 把手机和 Windows 电脑连接到同一个可信的私人 Wi-Fi。
2. 双击 `CoagulationAnalysis-Windows\Web\StartWebsite.exe`。
3. 若 Windows 防火墙询问，只允许“专用网络”，不要允许公共网络。
4. 电脑浏览器会自动打开；手机可扫描页面二维码或输入窗口中的 Phone 地址。
5. 保留启动器窗口。关闭窗口后，电脑和手机都会停止访问。

网站没有密码。同一 Wi-Fi 中知道地址的设备都可能打开，因此不要在公共网络
使用。手机和电脑可同时操作，所有计算及永久结果仍在 Windows 电脑完成并保存，
图片不会上传到云端或互联网。更换 Wi-Fi 后地址可能改变，请关闭后重新启动。

### 图片要求

- 支持 PNG、JPG/JPEG、BMP 和 TIFF。
- 图片宽度和高度都必须至少为 600 像素；例如 600×800 可以使用。
- 从正上方正对拍摄，保持画面清晰，尽量减少反光和大角度透视。
- 必须拍入完整的 3×3 九宫格和四条外边，不要预先裁掉外框。

程序固定识别 3×3 九宫格，位置 1–9 按从左到右、从上到下编号，
空白方格也会保留。每格先识别黑色框内最里层的正方形，再按设定向里
内缩，避免把黑色边框算入测量区域。

### 单张与批量分析

- `Single Image`：手机可用 `Choose from gallery or files` 从相册选择，或用
  `Take photo` 调用后置相机直接拍摄，然后点击 `Analyze Image`。
  页面会显示 9 张最终裁图、边界叠加图、热图和逐格数据，并可下载
  CSV 和单张结果 ZIP。
- `Batch Processing`：一次选择多张完整九宫格图片，然后点击
  `Analyze Batch`。某张图片识别失败时，其他图片仍会继续处理。
  `failures.csv` 会列出失败文件和原因，`batch-summary.csv` 是完整汇总，
  `batch-results.zip` 包含成功结果和两份报告。

默认 `Inner crop inset` 为 5%，表示在已识别的最里层正方形基础上再
向内缩 5%。默认 `No-clot threshold` 为 60。如果修改这两项，应在同一组
实验中保持一致。实际使用的内缩值和阈值会记录到结果 JSON；内缩值也会写入
单张 CSV 的每一行，批量设置则会写入 `batch-metadata.json`，便于复现。

### 热图解读

- 测量值小于或等于所选阈值时，方格为固定蓝色。
- 测量值高于所选阈值时，方格使用从浅红到深红的固定色阶。

热图颜色只是选定阈值下的定量数据可视化，不是医学或临床诊断。论文或报告
必须同时写明内缩百分比和无凝血阈值，不要仅根据颜色下结论。

### 结果保存位置

网站版结果默认保存在 `CoagulationAnalysis-Windows\Web\results\`。单张目录
包含 9 张裁图、边界叠加图、热图、CSV、JSON 和结果 ZIP。批量目录包含
每张成功图片的完整结果、汇总表、失败报告和批量 ZIP。页面中的
`Open folder on Windows PC` 可在 Windows 主机上打开对应目录；手机端请使用
CSV 或 ZIP 下载按钮。

`Desktop/CoagulationAnalysis.exe` 是备用桌面入口，一次处理一张图片，不提供
批量网页。网站无法启动时，请先确认已完整解压且 `_internal` 仍位于同一目录。
中文和其他 Unicode 文件名可以使用。

## Developer Quick Start

```bash
cd coagulation-analysis

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Lightweight local website and classical CV pipeline
pip install -r requirements.txt

# Optional deep-learning modules
pip install -r requirements_dl.txt
```

## Classical CV Pipeline (Variable-Grid CLI, no GPU needed)

`full_workflow.py` is the separate, developer-oriented variable-grid workflow.
Unlike the fixed desktop app above, it accepts `--rows` and `--cols`.

```bash
# Interactive desktop GUI — drag a rectangle, auto grid analysis
python3 full_workflow.py slide.jpg --rows 3 --cols 6

# Batch process pre-cropped cell images
python3 analyze.py folder/ --batch

# Watch mode — auto-analyze new images as they appear
python3 analyze.py folder/ --watch
```

## Deep Learning Training (GPU recommended)

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

## Deep Learning Inference

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
│  Encoder (frozen)                  │
│       │                            │
│       ├── Decoder (training)        │
│       └── Seg Head (training)       │
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
│  Encoder (unfrozen)                │
│       │                            │
│       ├── Decoder + Seg Head        │
│       ├── Regression Head           │
│       └── Classification Head       │
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

### Variable-Grid Classical CV CLI

The `full_workflow.py` and `analyze.py` CLI tools save results to
`<image>_analysis/`. These outputs are separate from the fixed desktop app
described above.

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
├── requirements_research.txt  # Reproducible cropping evaluation dependencies
├── requirements_dl.txt        # Full dependencies (including deep learning)
├── run_app.sh                 # Gradio web app launcher
│
├── full_workflow.py           # Classical CV: interactive GUI + analysis + heatmap
├── analyze.py                 # Classical CV: CLI batch processing
├── app.py                     # Local offline web interface
├── app_standalone.py          # Fixed 3x3 offline desktop app (PyInstaller)
├── grid_detector.py           # Fixed-fixture and inner-square detector
├── imagej_workflow.ijm        # ImageJ/Fiji macro
│
├── research/                  # Cropping evaluation and annotation tools
│   ├── EXPERIMENTS.md         # Protocol, commands, metrics, and artifact map
│   ├── annotate_inner_squares.py
│   └── evaluate_cropping.py
│
├── tests/                     # Production and research unit tests
│   ├── test_grid_detector.py
│   ├── test_standalone_pipeline.py
│   └── test_research_evaluation.py
│
├── dl/                        # Deep learning module
│   ├── config.py              #   Centralized config (data/model/loss/training)
│   ├── data.py                #   Pseudo-label generation + augmentation + Dataset
│   ├── model.py               #   CoagNet multi-task network architecture
│   ├── attention.py           #   Attention gates, SE blocks, CBAM
│   ├── losses.py              #   Dice Loss + Kendall uncertainty weighting
│   ├── train.py               #   Two-phase training loop
│   ├── inference.py           #   Inference + visualization + CV comparison
│   ├── advanced.py            #   TTA, MC Dropout, Ensemble inference
│   ├── evaluate.py            #   K-fold CV, ablation study, encoder benchmark
│   └── visualize.py           #   Grad-CAM, t-SNE, confusion matrix, ROC curves
│
├── train_dl.py                # DL training entry point
│
└── input/                     # Example data
    └── *_analysis/            #   Analysis output + cell images
```

---

## Research Evaluation

The reproducible fixed-fixture cropping study is documented in
[`research/EXPERIMENTS.md`](research/EXPERIMENTS.md). Install its dependencies
separately:

```bash
pip install -r requirements_research.txt
```

Run the synthetic comparison into a new or empty output directory:

```bash
python -m research.evaluate_cropping --synthetic \
  --output research/results/primary-new
```

Run the synthetic ablation study into another new or empty directory:

```bash
python -m research.evaluate_cropping --synthetic \
  --output research/results/ablations-new --ablations
```

Create independent manual annotations for a real image with:

```bash
python -m research.annotate_inner_squares path/to/image.png \
  --output path/to/image.annotations.json
```

Create a UTF-8 manifest with the exact columns
`case,image,annotations,condition,level`. Image and annotation paths are
relative to the manifest:

```csv
case,image,annotations,condition,level
photo-001,images/photo-001.png,labels/photo-001.json,lighting,normal
```

Then run the labeled evaluation:

```bash
python -m research.evaluate_cropping \
  --manifest path/to/manifest.csv \
  --output research/results/manual-new
```

Real-image results must come from reviewed manual annotations. The repository
does not provide or fabricate real-data measurements; synthetic runs verify
software behavior and reproducibility only.

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

### `dl/evaluate.py`

| Argument | Description |
|----------|-------------|
| `--cell-dir` | Directory with cell images |
| `--k-folds` | Run k-fold cross-validation (e.g. 5) |
| `--ablation` | Run ablation study |
| `--benchmark` | Run multi-encoder benchmark |
| `--quick` | Quick mode (fewer epochs) |
| `--output` | JSON output path |

---

## Advanced Features

### Grad-CAM Visualization

Generates saliency maps showing which image regions the model focuses on for each class prediction. Uses Grad-CAM (Selvaraju et al., ICCV 2017) and Grad-CAM++ (Chattopadhyay et al., WACV 2018) for fine-grained localization.

```python
from dl.visualize import GradCAM
cam = GradCAM(model)
heatmap = cam.generate(cell_img, target_class=0)
```

### Test-Time Augmentation (TTA)

Averages predictions across 8 augmented views (flips, rotations) for robust, uncertainty-aware inference. Provides per-pixel prediction variance.

```python
from dl.advanced import TTAInference
tta = TTAInference("best_model.pt", num_tta_views=8)
result = tta.predict(cell_img)
# result includes seg_uncertainty, reg_uncertainty, cls_uncertainty
```

### MC Dropout Uncertainty

Estimates epistemic (model) and aleatoric (data) uncertainty via 30 stochastic forward passes with dropout active at test time (Gal & Ghahramani, ICML 2016).

```python
from dl.advanced import MCDropoutInference
mc = MCDropoutInference("best_model.pt", num_samples=30)
result = mc.predict(cell_img)
mc.visualize_uncertainty(cell_img, save_path="uncertainty.png")
```

### Ensemble Prediction

Combines multiple model checkpoints (different architectures, seeds, or training epochs) following Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017).

```python
from dl.advanced import EnsembleInference
ensemble = EnsembleInference(["resnet50.pt", "efficientnet_b3.pt", "convnext.pt"])
result = ensemble.predict(cell_img)
# result includes seg_agreement (inter-model consensus)
```

### K-Fold Cross-Validation

Stratified k-fold CV with bootstrap confidence intervals for all metrics.

```bash
python3 -m dl.evaluate --cell-dir input/cells --k-folds 5
```

### Ablation Study

Systematically removes each component (segmentation head, regression head, classification head, uncertainty weighting, data augmentation, encoder fine-tuning) to quantify contribution.

```bash
python3 -m dl.evaluate --cell-dir input/cells --ablation
```

### Multi-Encoder Benchmark

Compares ResNet-34/50/101, EfficientNet-B0/B3, and ConvNeXt-Tiny under identical settings.

```bash
python3 -m dl.evaluate --cell-dir input/cells --benchmark
```

### Attention Mechanisms

Attention U-Net gates (Oktay et al., MIDL 2018), Squeeze-and-Excitation (Hu et al., CVPR 2018), and CBAM (Woo et al., ECCV 2018) for enhanced feature refinement.

```python
from dl.attention import AttentionGate, SEBlock, CBAM
```

---

## References

- Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. *CVPR 2018*.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
- Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Trans. Sys. Man. Cyber.*
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV 2017*.
- Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. *MIDL 2018*.
- Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML 2016*.
- Lakshminarayanan, B., et al. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS 2017*.
- Hu, J., et al. (2018). Squeeze-and-Excitation Networks. *CVPR 2018*.
- Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.
- Chattopadhyay, A., et al. (2018). Grad-CAM++: Generalized Gradient-Based Visual Explanations. *WACV 2018*.
- Moshkov, N., Mathe, B., Kertesz-Farkas, A., Hollandi, R., & Horvath, P. (2020). Test-time augmentation for deep learning-based cell segmentation on microscopy images. *Scientific Reports*, 10, 5068.
