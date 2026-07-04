"""
Inference and visualization for CoagNet.

Usage:
    from dl.inference import CoagInference
    infer = CoagInference("dl/checkpoints/best_model.pt")
    results = infer.predict(cell_image_bgr)
    infer.visualize(cell_image_bgr, save_path="output.png")
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import Config, ModelConfig
from .model import CoagNet
from .data import to_8bit_grayscale


CLASS_NAMES = ["Mild", "Moderate", "Severe"]
CLASS_COLORS = {
    0: (0, 255, 0),    # green — mild
    1: (0, 255, 255),  # yellow — moderate
    2: (0, 0, 255),    # red — severe
}


class CoagInference:
    """
    Inference wrapper for trained CoagNet model.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        config: Config for model architecture. Uses defaults if None.
        device: 'cuda', 'cpu', or None for auto-detect.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if config is None:
            config = Config()
        self.config = config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model
        self.model = CoagNet(config.model).to(self.device)
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.image_size = config.data.image_size
        print(f"Loaded CoagNet from {self.checkpoint_path}")
        print(f"  Best val Dice: {checkpoint.get('best_dice', 'N/A')}")
        print(f"  Device: {self.device}")

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess BGR image to ImageNet-normalized tensor."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size))
        image_rgb = image_rgb.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_rgb = (image_rgb - mean) / std

        # HWC → CHW → BCHW
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> dict:
        """
        Run inference on a single cell image.

        Args:
            image_bgr: BGR image (any size, will be resized).

        Returns:
            dict with:
              - seg_mask: (H, W) binary mask uint8
              - seg_prob: (H, W) probability map float32
              - coag_ratio: float, fraction of pixels classified as coagulation
              - reg_value: float, predicted mean intensity (0-255 scale)
              - cls_grade: int, severity grade (0=mild, 1=moderate, 2=severe)
              - cls_probs: dict, class probabilities
        """
        tensor = self._preprocess(image_bgr)
        result = self.model.predict(tensor)

        seg_prob = result["seg_prob"][0, 0].cpu().numpy()  # (H, W)
        seg_mask = result["seg_mask"][0, 0].cpu().numpy().astype(np.uint8)
        coag_ratio = float(seg_mask.mean())

        reg_value = float(result["reg_value"][0, 0].cpu().numpy())
        # Scale reg_value to 0-255 range (output is in normalized space)
        # Clamp to reasonable range
        reg_value = max(0.0, min(255.0, reg_value))

        cls_grade = int(result["cls_grade"][0].cpu().numpy())
        cls_probs = result["cls_probs"][0].cpu().numpy()
        cls_dict = {CLASS_NAMES[i]: float(cls_probs[i]) for i in range(len(cls_probs))}

        return {
            "seg_mask": seg_mask,
            "seg_prob": seg_prob,
            "coag_ratio": coag_ratio,
            "reg_value": reg_value,
            "cls_grade": cls_grade,
            "cls_name": CLASS_NAMES[cls_grade],
            "cls_probs": cls_dict,
        }

    def visualize(
        self,
        image_bgr: np.ndarray,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> np.ndarray:
        """
        Create a 4-panel visualization:
          (a) Original cell image
          (b) Segmentation probability heatmap
          (c) Binary mask overlay
          (d) Summary panel with metrics

        Args:
            image_bgr: Input BGR image.
            save_path: Path to save the visualization PNG.
            show: If True, display via OpenCV.

        Returns:
            4-panel visualization image (BGR).
        """
        results = self.predict(image_bgr)

        # Resize original for display
        h_orig, w_orig = image_bgr.shape[:2]
        display_size = 400
        scale = display_size / max(h_orig, w_orig)
        dh, dw = int(h_orig * scale), int(w_orig * scale)
        original = cv2.resize(image_bgr, (dw, dh))

        # Resize mask/prob to match
        seg_mask = cv2.resize(
            results["seg_mask"].astype(np.uint8) * 255, (dw, dh),
            interpolation=cv2.INTER_NEAREST,
        )
        seg_prob = cv2.resize(results["seg_prob"], (dw, dh))

        # Panel A: Original
        panel_a = original.copy()
        cv2.putText(panel_a, "(a) Original", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Panel B: Probability heatmap
        prob_colored = cv2.applyColorMap(
            (seg_prob * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        panel_b = cv2.addWeighted(original, 0.5, prob_colored, 0.5, 0)
        cv2.putText(panel_b, "(b) Coagulation Probability", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Panel C: Binary mask overlay
        mask_color = np.zeros((dh, dw, 3), dtype=np.uint8)
        mask_color[seg_mask > 128] = (0, 0, 255)  # Red overlay
        panel_c = cv2.addWeighted(original, 0.6, mask_color, 0.4, 0)
        cv2.putText(panel_c, f"(c) Mask (coag ratio: {results['coag_ratio']:.1%})", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Panel D: Summary
        panel_d = np.full((dh, dw, 3), 30, dtype=np.uint8)
        lines = [
            f"Grade: {results['cls_name']}",
            f"Intensity: {results['reg_value']:.1f}",
            f"Coag Ratio: {results['coag_ratio']:.1%}",
            "",
            "Class Probabilities:",
        ]
        for name, prob in results["cls_probs"].items():
            lines.append(f"  {name}: {prob:.2%}")

        y = 30
        for line in lines:
            cv2.putText(panel_d, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25

        cv2.putText(panel_d, "(d) Summary", (5, dh - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Concatenate into 2x2 grid
        top = np.hstack([panel_a, panel_b])
        bottom = np.hstack([panel_c, panel_d])
        grid = np.vstack([top, bottom])

        if save_path:
            cv2.imwrite(str(save_path), grid)
            print(f"Visualization saved to {save_path}")

        if show:
            cv2.imshow("CoagNet Inference", grid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return grid

    def compare_with_classical(
        self,
        image_bgr: np.ndarray,
        save_path: Optional[str | Path] = None,
    ) -> dict:
        """
        Run both CoagNet and classical CV (Otsu) on the same image,
        returning a side-by-side comparison.

        Returns:
            dict with keys: dl_*, classical_* for each metric.
        """
        # Deep learning prediction
        dl_results = self.predict(image_bgr)

        # Classical CV (same pipeline as current project)
        from .data import generate_pseudo_mask, to_8bit_grayscale

        gray = to_8bit_grayscale(image_bgr)
        inverted = 255 - gray
        classical_mask = generate_pseudo_mask(image_bgr)
        classical_coag_ratio = float(classical_mask.mean())
        classical_intensity = float(np.mean(inverted))

        comparison = {
            "dl_coag_ratio": dl_results["coag_ratio"],
            "classical_coag_ratio": classical_coag_ratio,
            "dl_intensity": dl_results["reg_value"],
            "classical_intensity": classical_intensity,
            "dl_grade": dl_results["cls_name"],
            "dl_cls_probs": dl_results["cls_probs"],
        }

        return comparison
