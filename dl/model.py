"""
CoagNet: Multi-Task Deep Coagulation Quantification Network.

Architecture:
    Shared Encoder (ResNet-50/EfficientNet, ImageNet pretrained)
    ├── U-Net Decoder (skip connections) → Binary segmentation mask
    ├── Regression Head → Coagulation intensity (scalar)
    └── Classification Head → Severity grade (mild/moderate/severe)

The encoder runs once; all three heads share the same feature hierarchy.
Skip connections feed multi-scale features to the U-Net decoder, while
only the deepest features feed the regression and classification MLPs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

from .config import ModelConfig


# ═══════════════════════════════════════════════════════════════════
#  Regression Head
# ═══════════════════════════════════════════════════════════════════

class RegressionHead(nn.Module):
    """MLP head that regresses coagulation intensity from pooled features."""

    def __init__(
        self,
        in_channels: int,
        hidden_dims: tuple = (512, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_dim = in_channels

        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hd

        layers.append(nn.Linear(prev_dim, 1))  # scalar output
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            *layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ═══════════════════════════════════════════════════════════════════
#  Classification Head
# ═══════════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """MLP head for coagulation severity grading."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 3,
        hidden_dims: tuple = (256,),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_dim = in_channels

        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hd

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            *layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ═══════════════════════════════════════════════════════════════════
#  CoagNet
# ═══════════════════════════════════════════════════════════════════

class CoagNet(nn.Module):
    """
    Multi-task coagulation quantification network.

    Shared encoder → 3 heads:
      - U-Net decoder → binary segmentation mask
      - Regression MLP → continuous coagulation intensity
      - Classification MLP → severity grade

    Args:
        cfg: ModelConfig with architecture choices.

    Usage:
        model = CoagNet(cfg)
        out = model(image_batch)  # {'seg': ..., 'reg': ..., 'cls': ...}

    The encoder runs exactly once per forward pass. Features are routed
    to all three heads: skip connections (C1→C4) to the U-Net decoder,
    and the deepest features (C5) to the task MLPs via GAP.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        if not HAS_SMP:
            raise ImportError(
                "segmentation-models-pytorch is required. "
                "Install with: pip install segmentation-models-pytorch"
            )

        # ── U-Net (encoder + decoder + seg head) ──
        self.unet = smp.Unet(
            encoder_name=cfg.encoder,
            encoder_weights=cfg.encoder_weights,
            in_channels=3,
            classes=cfg.seg_num_classes,
            decoder_channels=cfg.decoder_channels,
        )

        # Expose encoder/decoder for fine-grained control during phased training
        self.encoder = self.unet.encoder
        self.decoder = self.unet.decoder
        self.segmentation_head = self.unet.segmentation_head

        # ── Determine encoder output channels ──
        enc_channels = self._get_encoder_channels()

        # ── Task heads on deepest features ──
        self.reg_head = RegressionHead(
            in_channels=enc_channels,
            hidden_dims=cfg.reg_hidden_dims,
            dropout=cfg.reg_dropout,
        )
        self.cls_head = ClassificationHead(
            in_channels=enc_channels,
            num_classes=cfg.cls_num_classes,
            hidden_dims=cfg.cls_hidden_dims,
            dropout=cfg.cls_dropout,
        )

        self._init_weights()

    def _get_encoder_channels(self) -> int:
        """Infer encoder output channels with a dry run."""
        with torch.no_grad():
            dummy = torch.randn(1, 3, 64, 64)
            features = self.encoder(dummy)
            return features[-1].shape[1]

    def _init_weights(self):
        """Kaiming init for MLP heads (encoder/decoder already initialized by smp)."""
        for head in [self.reg_head, self.cls_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters (phase 1: seg-only pretraining)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen.")

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters (phase 2: joint training)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen.")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, 3, H, W) RGB image tensor, ImageNet-normalized.

        Returns:
            dict with:
              'seg': (B, 1, H, W) logits for binary segmentation
              'reg': (B, 1) coagulation intensity prediction
              'cls': (B, num_classes) classification logits
        """
        # ── Shared encoder (runs once) ──
        features = self.encoder(x)  # list of [C1, C2, C3, C4, C5]

        # ── Segmentation branch ──
        decoder_output = self.decoder(features)
        seg_logits = self.segmentation_head(decoder_output)

        # ── Task heads on deepest features ──
        deepest = features[-1]  # (B, C, H/32, W/32)
        reg_out = self.reg_head(deepest)
        cls_out = self.cls_head(deepest)

        return {
            "seg": seg_logits,
            "reg": reg_out,
            "cls": cls_out,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Inference mode: returns probabilities and rounded predictions.

        Returns:
            dict with:
              'seg_prob': (B, 1, H, W) sigmoid probabilities
              'seg_mask': (B, 1, H, W) binary mask (threshold=0.5)
              'reg_value': (B, 1) predicted intensity
              'cls_probs': (B, num_classes) softmax probabilities
              'cls_grade': (B,) predicted class index
        """
        self.eval()
        out = self.forward(x)

        seg_prob = torch.sigmoid(out["seg"])
        seg_mask = (seg_prob > 0.5).float()

        cls_probs = F.softmax(out["cls"], dim=1)
        cls_grade = cls_probs.argmax(dim=1)

        return {
            "seg_prob": seg_prob,
            "seg_mask": seg_mask,
            "reg_value": out["reg"],
            "cls_probs": cls_probs,
            "cls_grade": cls_grade,
        }
