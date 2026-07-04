"""
Paper-ready visualization suite for CoagNet.

Generates:
  - Grad-CAM / Grad-CAM++ heatmaps
  - t-SNE feature embedding plots
  - Confusion matrices
  - ROC & Precision-Recall curves
  - Feature map montages
  - Attention gate activation maps
  - Training curve comparison plots
  - Per-sample prediction vs ground truth comparison
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)

from .config import Config
from .model import CoagNet
from .data import to_8bit_grayscale

CLASS_NAMES = ["Mild", "Moderate", "Severe"]
CLASS_COLORS = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]


# ═══════════════════════════════════════════════════════════════
#  Grad-CAM
# ═══════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al., ICCV 2017).

    Highlights image regions the model uses for its prediction. Works with
    any CNN by hooking the final convolutional layer.
    """

    def __init__(self, model: CoagNet, target_layer_name: str = None):
        self.model = model
        self.model.eval()
        self.activations = {}
        self.gradients = {}

        # Auto-find the last conv layer of the encoder
        if target_layer_name is None:
            target_layer_name = self._find_last_conv()

        self._register_hooks(target_layer_name)

    def _find_last_conv(self) -> str:
        """Find the deepest Conv2d layer in the encoder."""
        last_name = None
        for name, module in self.model.encoder.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_name = f"encoder.{name}"
        return last_name or "encoder.layer4.2.conv3"

    def _register_hooks(self, target_name: str):
        """Register forward and backward hooks on the target layer."""
        target_module = None
        parts = target_name.split(".")
        obj = self.model
        for part in parts:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        target_module = obj

        def forward_hook(module, inp, out):
            self.activations["value"] = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients["value"] = grad_out[0]

        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess to ImageNet-normalized tensor."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_rgb = (image_rgb - mean) / std
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def generate(
        self,
        image_bgr: np.ndarray,
        target_class: Optional[int] = None,
        use_gradcam_pp: bool = True,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            image_bgr: Input BGR image (any size).
            target_class: Class index for the heatmap. If None, uses predicted class.
            use_gradcam_pp: If True, use Grad-CAM++ for better localization.

        Returns:
            Heatmap overlay (BGR), same size as input.
        """
        tensor = self._preprocess(image_bgr)
        device = next(self.model.parameters()).device
        tensor = tensor.to(device)
        tensor.requires_grad = True

        # Forward
        self.model.zero_grad()
        output = self.model(tensor)

        if target_class is None:
            target_class = output["cls"].argmax(dim=1).item()

        # Backward for target class
        score = output["cls"][0, target_class]
        score.backward()

        # Get activations and gradients
        activations = self.activations["value"].detach()  # (1, C, H', W')
        gradients = self.gradients["value"].detach()       # (1, C, H', W')

        # Compute weights
        if use_gradcam_pp:
            weights = self._gradcam_pp_weights(gradients, activations)
        else:
            weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to original image size
        cam = F.interpolate(
            cam,
            size=(image_bgr.shape[0], image_bgr.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0].cpu().numpy()

        # Colorize and overlay
        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(image_bgr, 0.5, heatmap, 0.5, 0)

        # Add label
        cv2.putText(
            overlay, f"Grad-CAM: {CLASS_NAMES[target_class]}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        return overlay

    def _gradcam_pp_weights(self, gradients, activations):
        """Grad-CAM++ weighting (Chattopadhyay et al., WACV 2018)."""
        grads_power_2 = gradients ** 2
        grads_power_3 = grads_power_2 * gradients
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)

        eps = 1e-8
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
        aij = aij.sum(dim=(2, 3), keepdim=True)

        weights = F.relu(gradients * aij).sum(dim=(2, 3), keepdim=True)
        return weights


# ═══════════════════════════════════════════════════════════════
#  Confusion Matrix
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str | Path] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Render a confusion matrix as an OpenCV image.

    Returns (H, W, 3) BGR image.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    if normalize:
        cm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    cell_size = 100
    pad = 50
    h = pad * 2 + cell_size * 3
    w = pad * 2 + cell_size * 3
    img = np.full((h + 60, w + 120, 3), 255, dtype=np.uint8)

    for i in range(3):
        for j in range(3):
            x1 = pad + j * cell_size + 120
            y1 = pad + i * cell_size + 40
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            # Color by value
            val = cm[i, j]
            intensity = int(255 * (1 - val))
            color = (intensity, intensity, 255) if i == j else (intensity, 255, intensity)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)

            # Value text
            text = f"{val:.2f}" if normalize else f"{val:.0f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(img, text, (x1 + (cell_size - tw) // 2, y1 + (cell_size + th) // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Labels
    for i, name in enumerate(CLASS_NAMES):
        cv2.putText(img, name, (10, pad + 40 + i * cell_size + cell_size // 2 + 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, name, (pad + 120 + i * cell_size + cell_size // 2 - 30, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(img, "True", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Predicted", (w // 2, h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    if save_path:
        cv2.imwrite(str(save_path), img)

    return img


# ═══════════════════════════════════════════════════════════════
#  ROC & PR Curves
# ═══════════════════════════════════════════════════════════════

def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str | Path] = None,
) -> np.ndarray:
    """
    Plot multi-class ROC curves (one-vs-rest).

    Args:
        y_true: (N,) class labels {0, 1, 2}
        y_scores: (N, 3) class probabilities
    """
    w, h = 500, 500
    img = np.full((h + 60, w + 120, 3), 255, dtype=np.uint8)

    # Plot area
    margin = 60
    plot_x = 80
    plot_y = 20
    plot_w = w - plot_x - margin
    plot_h = h - plot_y - margin

    # Axes
    cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (0, 0, 0), 1)
    cv2.line(img, (plot_x, plot_y), (plot_x, plot_y + plot_h), (0, 0, 0), 1)
    cv2.line(img, (plot_x, plot_y + plot_h), (plot_x + plot_w, plot_y + plot_h), (0, 0, 0), 1)

    # Diagonal
    cv2.line(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (180, 180, 180), 1)
    cv2.putText(img, "ROC Curves", (w // 2 - 50, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "FPR", (w // 2, h + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "TPR", (10, h // 2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR
    auc_values = []

    for cls_idx in range(3):
        y_bin = (y_true == cls_idx).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_scores[:, cls_idx])
        auc_val = auc(fpr, tpr)
        auc_values.append(auc_val)

        # Map to pixel coords
        for i in range(len(fpr) - 1):
            x1 = int(plot_x + fpr[i] * plot_w)
            y1 = int(plot_y + (1 - tpr[i]) * plot_h)
            x2 = int(plot_x + fpr[i + 1] * plot_w)
            y2 = int(plot_y + (1 - tpr[i + 1]) * plot_h)
            cv2.line(img, (x1, y1), (x2, y2), colors[cls_idx], 2)

        cv2.putText(img, f"{CLASS_NAMES[cls_idx]} (AUC={auc_val:.3f})",
                   (w - 180, 35 + cls_idx * 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[cls_idx], 1)

    if save_path:
        cv2.imwrite(str(save_path), img)

    return img


# ═══════════════════════════════════════════════════════════════
#  t-SNE Feature Visualization
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(
    model: CoagNet,
    dataloader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract encoder features from all samples in the dataloader."""
    model.eval()
    all_features = []
    all_labels = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["cls_label"].numpy()

        features = model.encoder(images)[-1]  # deepest features
        features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels)

    return np.vstack(all_features), np.concatenate(all_labels)


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str | Path] = None,
    perplexity: float = 5.0,
) -> np.ndarray:
    """
    Generate t-SNE 2D embedding plot.

    Args:
        features: (N, D) feature vectors from encoder.
        labels: (N,) class labels {0, 1, 2}.
        perplexity: t-SNE perplexity. Use min(5, N/3) for small datasets.

    Returns:
        (H, W, 3) BGR image.
    """
    n_samples = len(features)
    perplexity = min(perplexity, max(1, n_samples / 3 - 1))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embedded = tsne.fit_transform(features)

    # Normalize to [0, 1]
    embedded = (embedded - embedded.min(axis=0)) / (embedded.max(axis=0) - embedded.min(axis=0) + 1e-8)

    w, h = 600, 500
    img = np.full((h + 50, w + 80, 3), 255, dtype=np.uint8)

    margin = 50
    for i in range(n_samples):
        x = int(margin + 40 + embedded[i, 0] * (w - margin * 2))
        y = int(margin + embedded[i, 1] * (h - margin * 2))
        color = CLASS_COLORS[labels[i]]
        cv2.circle(img, (x, y), 8, color, -1)
        cv2.circle(img, (x, y), 8, (0, 0, 0), 1)

    cv2.putText(img, "t-SNE Feature Embedding", (w // 2 - 100, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Legend
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        cv2.circle(img, (w - 150, 50 + i * 30), 6, color, -1)
        cv2.putText(img, name, (w - 135, 55 + i * 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if save_path:
        cv2.imwrite(str(save_path), img)

    return img


# ═══════════════════════════════════════════════════════════════
#  Comprehensive Prediction Visualization
# ═══════════════════════════════════════════════════════════════

def plot_prediction_grid(
    images_bgr: list[np.ndarray],
    results: list[dict],
    ground_truth: Optional[list[dict]] = None,
    save_path: Optional[str | Path] = None,
    max_cols: int = 4,
) -> np.ndarray:
    """
    Create a grid of predictions with mask overlays.

    Each cell shows: original image | predicted mask | metrics text.

    Args:
        images_bgr: list of input BGR images.
        results: list of dicts from CoagInference.predict().
        ground_truth: optional list of dicts with 'cls_label' and 'reg_value'.
    """
    n = len(images_bgr)
    n_cols = min(max_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    cell_w, cell_h = 300, 350
    img_w = n_cols * cell_w
    img_h = n_rows * cell_h
    canvas = np.full((img_h, img_w, 3), 245, dtype=np.uint8)

    for idx, (img, res) in enumerate(zip(images_bgr, results)):
        row, col = divmod(idx, n_cols)
        x0, y0 = col * cell_w, row * cell_h

        # Resize original
        h, w = img.shape[:2]
        scale = min(280 / w, 200 / h)
        dh, dw = int(h * scale), int(w * scale)
        thumb = cv2.resize(img, (dw, dh))
        canvas[y0 + 5:y0 + 5 + dh, x0 + 5:x0 + 5 + dw] = thumb

        # Mask overlay
        mask_resized = cv2.resize(
            (res["seg_mask"] * 255).astype(np.uint8), (dw, dh),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_color = np.zeros((dh, dw, 3), dtype=np.uint8)
        mask_color[mask_resized > 128] = (0, 0, 255)
        canvas[y0 + 210:y0 + 210 + dh, x0 + 5:x0 + 5 + dw] = cv2.addWeighted(
            thumb, 0.5, mask_color, 0.5, 0
        )

        # Text
        lines = [
            f"Grade: {res['cls_name']} ({res['cls_probs'][res['cls_name']]:.2f})",
            f"Intensity: {res['reg_value']:.1f}",
            f"Coag Ratio: {res['coag_ratio']:.1%}",
        ]
        if ground_truth:
            gt = ground_truth[idx]
            lines.append(f"GT: {CLASS_NAMES[gt.get('cls_label', 0)]}")
            lines.append(f"GT Reg: {gt.get('reg_value', 0):.1f}")

        y_text = 215 + dh
        for line in lines:
            cv2.putText(canvas, line, (x0 + 5, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            y_text += 16

    if save_path:
        cv2.imwrite(str(save_path), canvas)

    return canvas


# ═══════════════════════════════════════════════════════════════
#  Training Curve Plot
# ═══════════════════════════════════════════════════════════════

def plot_training_curves(
    metrics_history: dict,
    save_path: Optional[str | Path] = None,
) -> np.ndarray:
    """
    Plot training/validation curves from logged metrics.

    Args:
        metrics_history: dict with keys like 'train_loss_seg', 'val_dice', etc.
                         Each value is a list of (epoch, value) tuples.
    """
    w, h = 800, 600
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Find global ranges
    all_epochs = {}
    for key, values in metrics_history.items():
        if values:
            epochs, vals = zip(*values)
            all_epochs[key] = (epochs, vals)

    if not all_epochs:
        return img

    max_epoch = max(max(e) for e, _ in all_epochs.values())

    # Plot params
    margin = 70
    plot_x, plot_y = margin + 40, margin
    plot_w = w - plot_x - margin
    plot_h = h - plot_y - margin

    # Axes
    cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (0, 0, 0), 1)

    # Grid lines
    for i in range(1, 5):
        y = plot_y + int(i * plot_h / 4)
        cv2.line(img, (plot_x, y), (plot_x + plot_w, y), (220, 220, 220), 1, cv2.LINE_AA)

    # Dual y-axis: left for loss, right for dice/accuracy
    colors = {
        "train_loss_seg": ((255, 0, 0), "Seg Loss (train)"),
        "val_loss_seg": ((0, 0, 255), "Seg Loss (val)"),
        "val_dice": ((0, 150, 0), "Dice Score"),
        "val_accuracy": ((200, 100, 0), "Accuracy"),
    }

    for key, (color, label) in colors.items():
        if key not in all_epochs:
            continue
        epochs, vals = all_epochs[key]
        vals = np.array(vals)

        # Normalize to plot space
        if "loss" in key:
            y_vals = vals / max(vals.max(), 1)  # [0, 1]
        elif "dice" in key:
            y_vals = vals  # already [0, 1]
        elif "accuracy" in key:
            y_vals = vals  # [0, 1]

        # Draw line
        for i in range(len(epochs) - 1):
            x1 = plot_x + int(epochs[i] / max(max_epoch, 1) * plot_w)
            y1 = plot_y + int((1 - y_vals[i]) * plot_h)
            x2 = plot_x + int(epochs[i + 1] / max(max_epoch, 1) * plot_w)
            y2 = plot_y + int((1 - y_vals[i + 1]) * plot_h)
            cv2.line(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label
        cv2.putText(img, label, (w - 220, 30 + list(colors.keys()).index(key) * 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(img, "Training Curves", (w // 2 - 80, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Epoch", (w // 2, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Value", (10, h // 2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if save_path:
        cv2.imwrite(str(save_path), img)

    return img
