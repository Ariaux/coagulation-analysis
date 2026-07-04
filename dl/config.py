"""
Centralized configuration for CoagNet training and inference.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    image_size: int = 224
    cell_dir: Path = Path("input")
    val_split: float = 0.2
    num_workers: int = 4

    # Pseudo-label generation (Otsu on inverted grayscale)
    otsu_kernel_size: int = 9
    cls_num_bins: int = 3  # mild / moderate / severe

    # Augmentation
    augmentation: bool = True
    elastic_alpha: int = 120
    elastic_sigma: int = 15
    color_jitter: float = 0.2


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    encoder: Literal[
        "resnet34", "resnet50", "resnet101",
        "efficientnet-b0", "efficientnet-b3",
        "timm-resnest50d", "timm-convnext_tiny",
    ] = "resnet50"
    encoder_weights: str = "imagenet"
    decoder_channels: tuple = (256, 128, 64, 32, 16)
    seg_num_classes: int = 1  # binary segmentation head

    # Regression head architecture
    reg_hidden_dims: tuple = (512, 128)
    reg_dropout: float = 0.3

    # Classification head architecture
    cls_hidden_dims: tuple = (256,)
    cls_dropout: float = 0.3
    cls_num_classes: int = 3

    # Frozen encoder stages during phase 1 (seg-only pretraining)
    freeze_encoder_phase1: bool = True


@dataclass
class LossConfig:
    """Loss function configuration."""

    # Segmentation losses
    dice_weight: float = 0.5
    bce_weight: float = 0.5

    # Multi-task weights (initial values before uncertainty learning)
    seg_loss_weight: float = 1.0
    reg_loss_weight: float = 0.3
    cls_loss_weight: float = 0.5

    # Whether to use learned uncertainty weighting (Kendall et al. 2018)
    uncertainty_weighting: bool = True

    # Dice loss smooth factor
    dice_smooth: float = 1.0

    # Label smoothing for classification
    label_smoothing: float = 0.1


@dataclass
class TrainConfig:
    """Training loop configuration."""

    # Phase 1: segmentation pretraining
    phase1_epochs: int = 10
    phase1_lr: float = 1e-3
    phase1_batch_size: int = 8

    # Phase 2: multi-task joint training
    phase2_epochs: int = 50
    phase2_lr: float = 3e-4
    phase2_batch_size: int = 4

    # Optimizer
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    weight_decay: float = 1e-4
    adam_betas: tuple = (0.9, 0.999)

    # LR scheduler
    scheduler: Literal["cosine", "cosine_warmup", "plateau", "step"] = "cosine_warmup"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Mixed precision
    use_amp: bool = True

    # Gradient clipping
    grad_clip_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 15

    # Checkpointing
    save_dir: Path = Path("dl/checkpoints")
    save_best: bool = True
    save_latest: bool = True

    # Logging
    log_dir: Path = Path("dl/logs")
    log_interval: int = 10  # steps between log prints
    val_interval: int = 1   # epochs between validation

    # Hardware
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    seed: int = 42


@dataclass
class Config:
    """Top-level config aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        self.train.save_dir.mkdir(parents=True, exist_ok=True)
        self.train.log_dir.mkdir(parents=True, exist_ok=True)
