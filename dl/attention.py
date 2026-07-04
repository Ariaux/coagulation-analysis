"""
Attention mechanisms for U-Net decoder.

Implements:
  - Attention Gate (AG): from Attention U-Net (Oktay et al., 2018)
  - SE Block: Squeeze-and-Excitation channel attention
  - CBAM: Convolutional Block Attention Module (channel + spatial)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate from "Attention U-Net: Learning Where to Look for the Pancreas"
    (Oktay et al., MIDL 2018).

    Filters skip-connection features using a gating signal from the coarser scale.
    The gate suppresses irrelevant regions and highlights salient features.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int = None):
        """
        Args:
            F_g: channels in gating signal (from decoder)
            F_l: channels in skip connection (from encoder)
            F_int: intermediate channels (default: F_l // 2)
        """
        super().__init__()
        if F_int is None:
            F_int = F_l // 2

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: gating signal from decoder (B, F_g, H, W)
            x: skip connection from encoder (B, F_l, H, W)

        Returns:
            Attention-weighted skip features (B, F_l, H, W)
        """
        # Align gating signal spatial size to match skip
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)  # (B, 1, H, W) attention map
        return x * alpha


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., CVPR 2018).

    Recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., ECCV 2018).

    Combines channel attention (what to attend to) with spatial attention
    (where to attend).
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        # Channel attention
        avg_out = self.channel_fc(self.avg_pool(x).view(b, c))
        max_out = self.channel_fc(self.max_pool(x).view(b, c))
        channel_attn = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_attn

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attn

        return x
