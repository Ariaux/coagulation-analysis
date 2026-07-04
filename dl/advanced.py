"""
Advanced inference techniques for CoagNet.

Features:
  - Test-Time Augmentation (TTA): average predictions across augmented views
  - MC Dropout: epistemic uncertainty estimation via repeated stochastic forward passes
  - Ensemble prediction: combine multiple model checkpoints
  - Multi-scale inference: sliding window across image scales
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import albumentations as A
    HAS_ALB = True
except ImportError:
    HAS_ALB = False

from .config import Config
from .model import CoagNet

CLASS_NAMES = ["Mild", "Moderate", "Severe"]


# ═══════════════════════════════════════════════════════════════
#  Test-Time Augmentation (TTA)
# ═══════════════════════════════════════════════════════════════

class TTAInference:
    """
    Test-Time Augmentation: average predictions across multiple augmented
    views of the same input for more robust predictions.

    Augmentations used at test time:
      - Original (no augmentation)
      - Horizontal flip
      - Vertical flip
      - Rotation 90° / 180° / 270°
      - Combined flips + rotations

    Reference: Wang et al., "Test-Time Augmentation for Semantic Segmentation", 2020.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: Optional[Config] = None,
        device: Optional[str] = None,
        num_tta_views: int = 8,
    ):
        if config is None:
            config = Config()
        self.config = config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = CoagNet(config.model).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.image_size = config.data.image_size
        self.num_tta_views = num_tta_views

        # Define TTA transforms
        self.tta_transforms = self._build_tta_transforms()

    def _build_tta_transforms(self) -> list[dict]:
        """Build list of TTA transforms: each is (pre_transform, post_inverse_fn)."""
        tta = [
            # 0: Original
            {"name": "original", "flip_h": False, "flip_v": False, "rotate": 0},
            # 1: Horizontal flip
            {"name": "hflip", "flip_h": True, "flip_v": False, "rotate": 0},
            # 2: Vertical flip
            {"name": "vflip", "flip_h": False, "flip_v": True, "rotate": 0},
            # 3: 90° rotation
            {"name": "rot90", "flip_h": False, "flip_v": False, "rotate": 90},
            # 4: 180° rotation
            {"name": "rot180", "flip_h": False, "flip_v": False, "rotate": 180},
            # 5: 270° rotation
            {"name": "rot270", "flip_h": False, "flip_v": False, "rotate": 270},
            # 6: HFlip + Rot90
            {"name": "hflip_rot90", "flip_h": True, "flip_v": False, "rotate": 90},
            # 7: VFlip + Rot90
            {"name": "vflip_rot90", "flip_h": False, "flip_v": True, "rotate": 90},
        ]
        return tta[:self.num_tta_views]

    def _apply_transform(self, img: np.ndarray, tta_params: dict) -> np.ndarray:
        """Apply TTA transform to image."""
        result = img.copy()
        if tta_params["flip_h"]:
            result = cv2.flip(result, 1)
        if tta_params["flip_v"]:
            result = cv2.flip(result, 0)
        if tta_params["rotate"] != 0:
            k = tta_params["rotate"] // 90
            result = np.rot90(result, k, axes=(0, 1))
        return np.ascontiguousarray(result)

    def _inverse_transform(
        self, mask: np.ndarray, tta_params: dict
    ) -> np.ndarray:
        """Inverse TTA transform to bring mask back to original orientation."""
        result = mask.copy()
        if tta_params["rotate"] != 0:
            k = tta_params["rotate"] // 90
            result = np.rot90(result, -k, axes=(0, 1))
        if tta_params["flip_v"]:
            result = cv2.flip(result, 0)
        if tta_params["flip_h"]:
            result = cv2.flip(result, 1)
        return result

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """Preprocess to ImageNet-normalized numpy array (H, W, 3)."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size))
        image_rgb = image_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (image_rgb - mean) / std

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> dict:
        """
        TTA inference: average predictions across all augmented views.

        Returns same dict format as CoagInference.predict().
        """
        image_np = self._preprocess(image_bgr)

        seg_probs = []
        reg_values = []
        cls_probs_list = []

        for tta_params in self.tta_transforms:
            # Apply transform
            transformed = self._apply_transform(image_np, tta_params)

            # To tensor
            tensor = torch.from_numpy(transformed).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.float().to(self.device)

            # Forward
            result = self.model.predict(tensor)

            # Inverse transform for segmentation mask
            seg_prob = result["seg_prob"][0, 0].cpu().numpy()
            seg_prob = self._inverse_transform(seg_prob, tta_params)
            seg_probs.append(seg_prob)

            reg_values.append(result["reg_value"][0, 0].cpu().item())
            cls_probs_list.append(result["cls_probs"][0].cpu().numpy())

        # Average across TTA views
        seg_prob_avg = np.mean(seg_probs, axis=0)
        seg_mask = (seg_prob_avg > 0.5).astype(np.uint8)
        reg_value_avg = np.mean(reg_values)
        cls_probs_avg = np.mean(cls_probs_list, axis=0)

        coag_ratio = float(seg_mask.mean())
        cls_grade = int(np.argmax(cls_probs_avg))

        # TTA uncertainty (agreement across views)
        seg_uncertainty = np.std(seg_probs, axis=0)

        return {
            "seg_mask": seg_mask,
            "seg_prob": seg_prob_avg,
            "seg_uncertainty": seg_uncertainty,
            "coag_ratio": coag_ratio,
            "reg_value": max(0.0, min(255.0, reg_value_avg)),
            "reg_uncertainty": float(np.std(reg_values)),
            "cls_grade": cls_grade,
            "cls_name": CLASS_NAMES[cls_grade],
            "cls_probs": {CLASS_NAMES[i]: float(cls_probs_avg[i]) for i in range(len(cls_probs_avg))},
            "cls_uncertainty": float(np.std([p[cls_grade] for p in cls_probs_list])),
        }


# ═══════════════════════════════════════════════════════════════
#  MC Dropout — Epistemic Uncertainty
# ═══════════════════════════════════════════════════════════════

class MCDropoutInference:
    """
    Monte Carlo Dropout: estimate epistemic uncertainty by running N
    stochastic forward passes with dropout ENABLED at test time.

    The variance across passes captures model uncertainty — where the
    model is unsure, predictions will vary more.

    Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: Optional[Config] = None,
        device: Optional[str] = None,
        num_samples: int = 30,
    ):
        if config is None:
            config = Config()
        self.config = config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = CoagNet(config.model).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.train()  # Keep dropout active!

        self.image_size = config.data.image_size
        self.num_samples = num_samples

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess to tensor."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size))
        image_rgb = image_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_rgb = (image_rgb - mean) / std
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.float().to(self.device)

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> dict:
        """
        MC Dropout inference with uncertainty estimation.

        Returns:
            dict with mean predictions and uncertainty maps/metrics.
        """
        tensor = self._preprocess(image_bgr)

        seg_samples = []
        reg_samples = []
        cls_samples = []

        for _ in range(self.num_samples):
            result = self.model.predict(tensor)
            seg_samples.append(result["seg_prob"][0, 0].cpu().numpy())
            reg_samples.append(result["reg_value"][0, 0].cpu().item())
            cls_samples.append(result["cls_probs"][0].cpu().numpy())

        seg_samples = np.stack(seg_samples, axis=0)  # (N, H, W)
        reg_samples = np.array(reg_samples)
        cls_samples = np.stack(cls_samples, axis=0)   # (N, 3)

        # Mean predictions
        seg_mean = seg_samples.mean(axis=0)
        seg_mask = (seg_mean > 0.5).astype(np.uint8)
        reg_mean = reg_samples.mean()
        cls_mean = cls_samples.mean(axis=0)

        # Uncertainty (per-pixel for segmentation, scalar for reg/cls)
        seg_epistemic = seg_samples.std(axis=0)  # per-pixel model uncertainty
        seg_aleatoric = np.mean(seg_samples * (1 - seg_samples), axis=0)  # data uncertainty

        cls_grade = int(np.argmax(cls_mean))
        coag_ratio = float(seg_mask.mean())

        return {
            "seg_mask": seg_mask,
            "seg_prob": seg_mean,
            "seg_epistemic_uncertainty": seg_epistemic,  # model uncertainty
            "seg_aleatoric_uncertainty": seg_aleatoric,  # data uncertainty
            "coag_ratio": coag_ratio,
            "reg_value": max(0.0, min(255.0, reg_mean)),
            "reg_uncertainty": float(reg_samples.std()),
            "cls_grade": cls_grade,
            "cls_name": CLASS_NAMES[cls_grade],
            "cls_probs": {CLASS_NAMES[i]: float(cls_mean[i]) for i in range(len(cls_mean))},
            "cls_uncertainty": float(cls_samples[:, cls_grade].std()),
            "num_mc_samples": self.num_samples,
        }

    def visualize_uncertainty(
        self,
        image_bgr: np.ndarray,
        save_path: Optional[str | Path] = None,
    ) -> np.ndarray:
        """
        Generate a 3-panel visualization:
          (a) Original + mask overlay
          (b) Epistemic uncertainty map (model uncertainty)
          (c) Aleatoric uncertainty map (data uncertainty)
        """
        results = self.predict(image_bgr)

        h, w = image_bgr.shape[:2]
        scale = min(400 / max(h, w), 1.0)
        dh, dw = int(h * scale), int(w * scale)
        thumb = cv2.resize(image_bgr, (dw, dh))

        # Panel A: mask overlay
        mask_disp = cv2.resize(
            (results["seg_mask"] * 255).astype(np.uint8), (dw, dh),
            interpolation=cv2.INTER_NEAREST,
        )
        overlay_color = np.zeros((dh, dw, 3), dtype=np.uint8)
        overlay_color[mask_disp > 128] = (0, 0, 255)
        panel_a = cv2.addWeighted(thumb, 0.5, overlay_color, 0.5, 0)
        cv2.putText(panel_a, "(a) Prediction", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Panel B: epistemic uncertainty
        epi = cv2.resize(results["seg_epistemic_uncertainty"], (dw, dh))
        epi_norm = ((epi / (epi.max() + 1e-8)) * 255).astype(np.uint8)
        panel_b = cv2.applyColorMap(epi_norm, cv2.COLORMAP_HOT)
        cv2.putText(panel_b, "(b) Epistemic Uncertainty (model)", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Panel C: aleatoric uncertainty
        ale = cv2.resize(results["seg_aleatoric_uncertainty"], (dw, dh))
        ale_norm = ((ale / (ale.max() + 1e-8)) * 255).astype(np.uint8)
        panel_c = cv2.applyColorMap(ale_norm, cv2.COLORMAP_HOT)
        cv2.putText(panel_c, "(c) Aleatoric Uncertainty (data)", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        grid = np.hstack([panel_a, panel_b, panel_c])

        if save_path:
            cv2.imwrite(str(save_path), grid)

        return grid


# ═══════════════════════════════════════════════════════════════
#  Ensemble Prediction
# ═══════════════════════════════════════════════════════════════

class EnsembleInference:
    """
    Ensemble predictions across multiple model checkpoints.

    Combines:
      - Different encoder architectures (ResNet, EfficientNet)
      - Different training epochs (snapshot ensemble)
      - Different random seeds

    Reference: Lakshminarayanan et al., "Simple and Scalable Predictive
    Uncertainty Estimation using Deep Ensembles", NeurIPS 2017.
    """

    def __init__(
        self,
        checkpoint_paths: list[str | Path],
        configs: Optional[list[Config]] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if configs is None:
            configs = [Config() for _ in checkpoint_paths]

        self.models = []
        for ckpt_path, cfg in zip(checkpoint_paths, configs):
            model = CoagNet(cfg.model).to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.models.append(model)

        self.image_size = Config().data.image_size
        print(f"Ensemble of {len(self.models)} models loaded.")

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size))
        image_rgb = image_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_rgb = (image_rgb - mean) / std
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.float().to(self.device)

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> dict:
        """
        Ensemble prediction: average outputs from all models.

        Returns same dict format as CoagInference.predict(), with
        added inter-model agreement metrics.
        """
        tensor = self._preprocess(image_bgr)

        all_seg = []
        all_reg = []
        all_cls = []

        for model in self.models:
            result = model.predict(tensor)
            all_seg.append(result["seg_prob"][0, 0].cpu().numpy())
            all_reg.append(result["reg_value"][0, 0].cpu().item())
            all_cls.append(result["cls_probs"][0].cpu().numpy())

        seg_mean = np.mean(all_seg, axis=0)
        seg_mask = (seg_mean > 0.5).astype(np.uint8)
        reg_mean = np.mean(all_reg)
        cls_mean = np.mean(all_cls, axis=0)

        # Ensemble agreement (how many models agree with the majority)
        seg_votes = np.mean([(s > 0.5).astype(np.float32) for s in all_seg], axis=0)
        seg_agreement = np.maximum(seg_votes, 1 - seg_votes)  # majority ratio

        cls_grade = int(np.argmax(cls_mean))
        cls_votes = [int(np.argmax(c)) for c in all_cls]
        cls_agreement = cls_votes.count(cls_grade) / len(cls_votes)

        return {
            "seg_mask": seg_mask,
            "seg_prob": seg_mean,
            "seg_agreement": seg_agreement,
            "coag_ratio": float(seg_mask.mean()),
            "reg_value": max(0.0, min(255.0, reg_mean)),
            "reg_std": float(np.std(all_reg)),
            "cls_grade": cls_grade,
            "cls_name": CLASS_NAMES[cls_grade],
            "cls_probs": {CLASS_NAMES[i]: float(cls_mean[i]) for i in range(len(cls_mean))},
            "cls_agreement": cls_agreement,
            "num_models": len(self.models),
        }
