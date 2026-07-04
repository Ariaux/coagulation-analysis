"""
CoagNet: Multi-Task Deep Learning for Coagulation Quantification.

Semantic segmentation + regression + classification with shared encoder.

Modules:
  - config: Centralized dataclass-based configuration
  - data: Pseudo-label generation, augmentation, PyTorch Dataset
  - model: CoagNet multi-task architecture (U-Net + MLP heads)
  - attention: Attention gates, SE blocks, CBAM
  - losses: Dice loss, Kendall uncertainty-weighted multi-task loss
  - train: Two-phase training pipeline
  - inference: Single-model inference with visualization
  - advanced: TTA, MC Dropout, and Ensemble inference
  - evaluate: K-fold CV, ablation studies, encoder benchmarks
  - visualize: Grad-CAM, t-SNE, confusion matrices, ROC curves
"""
from .config import Config
from .model import CoagNet
from .losses import MultiTaskLoss
from .data import CoagDataset, generate_pseudo_labels
