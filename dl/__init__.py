"""
CoagNet: Multi-Task Deep Learning for Coagulation Quantification.

Semantic segmentation + regression + classification with shared encoder.
"""
from .config import Config
from .model import CoagNet
from .losses import MultiTaskLoss
from .data import CoagDataset, generate_pseudo_labels
