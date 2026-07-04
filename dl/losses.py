"""
Loss functions for multi-task coagulation quantification.

Key components:
  - DiceLoss: region overlap for segmentation
  - MultiTaskLoss: combines segmentation (Dice+BCE), regression (MSE),
    and classification (CrossEntropy) with learned uncertainty weighting
    (Kendall, Gal, & Cipolla, CVPR 2018).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig


# ═══════════════════════════════════════════════════════════════════
#  Dice Loss
# ═══════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Soft Dice coefficient loss for binary segmentation.

    Dice = 2·|pred ∩ target| / (|pred| + |target|)
    L_dice = 1 - Dice

    Numerically stable implementation with optional logits input.
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits or probabilities
            target: (B, 1, H, W) binary mask {0, 1}
        """
        if self.from_logits:
            pred = torch.sigmoid(pred)

        batch_size = pred.shape[0]
        pred = pred.view(batch_size, -1)
        target = target.view(batch_size, -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


# ═══════════════════════════════════════════════════════════════════
#  Segmentation Loss (Dice + BCE)
# ═══════════════════════════════════════════════════════════════════

class SegmentationLoss(nn.Module):
    """
    Combined segmentation loss: weighted sum of Dice and BCE.

    L_seg = λ_dice · DiceLoss + λ_bce · BCELoss
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(smooth=1.0, from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(pred, target)
            + self.bce_weight * self.bce(pred, target)
        )


# ═══════════════════════════════════════════════════════════════════
#  Multi-Task Loss with Uncertainty Weighting
# ═══════════════════════════════════════════════════════════════════

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learned homoscedastic uncertainty weighting.

    Reference:
      Kendall, Gal, Cipolla. "Multi-Task Learning Using Uncertainty to
      Weigh Losses for Scene Geometry and Semantics." CVPR 2018.

    Core idea: each task's loss is weighted by a learned precision
    parameter log(σ²). Tasks with higher inherent noise automatically
    receive lower weight. This avoids manual tuning of loss weights.

    L_total = Σ (1/(2σᵢ²)) · Lᵢ + log(σᵢ)

    For classification, the scaling uses 1/σ² instead of 1/(2σ²)
    because CrossEntropy is a log-likelihood with different temperature.
    """

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.seg_loss_fn = SegmentationLoss(cfg.dice_weight, cfg.bce_weight)
        self.reg_loss_fn = nn.MSELoss()
        self.cls_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=cfg.label_smoothing
        )

        if cfg.uncertainty_weighting:
            # Learnable log variances (one per task)
            self.log_var_seg = nn.Parameter(torch.tensor(0.0))
            self.log_var_reg = nn.Parameter(torch.tensor(0.0))
            self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("log_var_seg", torch.tensor(0.0))
            self.register_buffer("log_var_reg", torch.tensor(0.0))
            self.register_buffer("log_var_cls", torch.tensor(0.0))

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            outputs: {'seg': logits, 'reg': values, 'cls': logits}
            targets: {'mask': (B,1,H,W), 'reg_value': (B,1), 'cls_label': (B,)}

        Returns:
            total_loss: scalar tensor for backprop
            loss_dict: detached per-task losses for logging
        """
        # Per-task losses
        loss_seg = self.seg_loss_fn(outputs["seg"], targets["mask"])
        loss_reg = self.reg_loss_fn(outputs["reg"], targets["reg_value"])
        loss_cls = self.cls_loss_fn(outputs["cls"], targets["cls_label"])

        if self.cfg.uncertainty_weighting:
            # Kendall uncertainty weighting
            precision_seg = torch.exp(-self.log_var_seg)
            precision_reg = torch.exp(-self.log_var_reg)
            precision_cls = torch.exp(-self.log_var_cls)

            total = (
                0.5 * precision_seg * loss_seg + 0.5 * self.log_var_seg
                + 0.5 * precision_reg * loss_reg + 0.5 * self.log_var_reg
                + precision_cls * loss_cls + 0.5 * self.log_var_cls
            )
        else:
            total = (
                self.cfg.seg_loss_weight * loss_seg
                + self.cfg.reg_loss_weight * loss_reg
                + self.cfg.cls_loss_weight * loss_cls
            )

        loss_dict = {
            "total": total.detach().item(),
            "seg": loss_seg.detach().item(),
            "reg": loss_reg.detach().item(),
            "cls": loss_cls.detach().item(),
        }

        if self.cfg.uncertainty_weighting:
            loss_dict["sigma_seg"] = torch.exp(self.log_var_seg).detach().item()
            loss_dict["sigma_reg"] = torch.exp(self.log_var_reg).detach().item()
            loss_dict["sigma_cls"] = torch.exp(self.log_var_cls).detach().item()

        return total, loss_dict


# ═══════════════════════════════════════════════════════════════════
#  Metrics (for validation logging, not backprop)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Compute validation metrics:
      - dice_score: segmentation Dice coefficient
      - reg_mae: mean absolute error for regression
      - cls_accuracy: classification accuracy
    """
    # Dice score
    seg_prob = torch.sigmoid(outputs["seg"])
    seg_pred = (seg_prob > 0.5).float()
    intersection = (seg_pred * targets["mask"]).sum(dim=(1, 2, 3))
    union = seg_pred.sum(dim=(1, 2, 3)) + targets["mask"].sum(dim=(1, 2, 3))
    dice = ((2.0 * intersection + 1.0) / (union + 1.0)).mean().item()

    # MAE
    mae = F.l1_loss(outputs["reg"], targets["reg_value"]).item()

    # Accuracy
    cls_pred = outputs["cls"].argmax(dim=1)
    acc = (cls_pred == targets["cls_label"]).float().mean().item()

    return {"dice_score": dice, "reg_mae": mae, "cls_accuracy": acc}
