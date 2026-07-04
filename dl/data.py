"""
Data pipeline: pseudo-label generation, augmentation, and PyTorch Dataset.

Pseudo-label strategy for zero-annotation setup:
  1. Load cell image → 8-bit grayscale (ImageJ formula) → invert
  2. Otsu thresholding → binary coagulation mask (segmentation label)
  3. Mean inverted intensity → regression target
  4. Tertile binning of mean intensity → classification label
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

try:
    import albumentations as A
    HAS_ALB = True
except ImportError:
    HAS_ALB = False

from .config import DataConfig


# ═══════════════════════════════════════════════════════════════════
#  Pseudo-Label Generation
# ═══════════════════════════════════════════════════════════════════

def to_8bit_grayscale(bgr: np.ndarray) -> np.ndarray:
    """ImageJ-exact grayscale: 0.299·R + 0.587·G + 0.114·B."""
    b, g, r = bgr[:, :, 0].astype(np.float32), \
              bgr[:, :, 1].astype(np.float32), \
              bgr[:, :, 2].astype(np.float32)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(gray, 0, 255).astype(np.uint8)


def generate_pseudo_mask(
    cell_image: np.ndarray,
    kernel_size: int = 9,
) -> np.ndarray:
    """
    Generate binary coagulation mask via Otsu thresholding on inverted grayscale.

    Steps:
      1. 8-bit grayscale (ImageJ formula)
      2. Invert (255 - gray) → coagulation becomes bright
      3. Otsu adaptive threshold → binary mask
      4. Morphological close + open to denoise

    Returns:
        Binary mask (H, W), dtype=uint8, values {0, 1}.
        1 = coagulated region, 0 = background/clear.
    """
    gray = to_8bit_grayscale(cell_image)
    inverted = 255 - gray

    # Otsu threshold on inverted image
    _, binary = cv2.threshold(
        inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return (binary > 0).astype(np.uint8)


def generate_pseudo_labels(
    cell_image: np.ndarray,
    cfg: DataConfig,
) -> dict:
    """
    Generate all pseudo-labels for one cell image.

    Returns:
        dict with:
          - 'mask': (H, W) uint8 binary coagulation mask
          - 'reg_value': float, mean intensity of inverted image (0-255)
          - 'cls_label': int, coagulation grade {0: mild, 1: moderate, 2: severe}
    """
    gray = to_8bit_grayscale(cell_image)
    inverted = 255 - gray

    # Segmentation mask
    mask = generate_pseudo_mask(cell_image, cfg.otsu_kernel_size)

    # Regression target: mean inverted intensity
    reg_value = float(np.mean(inverted))

    return {
        "mask": mask,
        "reg_value": reg_value,
        # cls_label assigned later via global binning
    }


def assign_class_labels(samples: list, num_bins: int = 3) -> list:
    """
    Assign class labels via tertile binning of regression values across
    all samples, ensuring balanced class distribution.

    Labels: 0 = mild, 1 = moderate, 2 = severe.
    """
    reg_values = np.array([s["reg_value"] for s in samples])
    thresholds = np.percentile(reg_values, np.linspace(0, 100, num_bins + 1)[1:-1])

    for sample in samples:
        val = sample["reg_value"]
        if val <= thresholds[0]:
            sample["cls_label"] = 0
        elif val <= thresholds[1]:
            sample["cls_label"] = 1
        else:
            sample["cls_label"] = 2

    return samples, thresholds


# ═══════════════════════════════════════════════════════════════════
#  Augmentation Pipeline
# ═══════════════════════════════════════════════════════════════════

def build_augmentation(cfg: DataConfig, is_train: bool = True):
    """
    Build albumentations Compose pipeline.

    Train: heavy augmentation (elastic transform, flips, color jitter).
    Val/Test: only resize + normalize.
    """
    # albumentations 2.x: output is HWC; ToTensorV2 converts to CHW
    base_transforms = [
        A.Resize(cfg.image_size, cfg.image_size),
    ]

    if is_train and cfg.augmentation and HAS_ALB:
        base_transforms += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(
                alpha=cfg.elastic_alpha,
                sigma=cfg.elastic_sigma,
                approximate=True,
                p=0.5,
            ),
            A.ColorJitter(
                brightness=cfg.color_jitter,
                contrast=cfg.color_jitter,
                saturation=cfg.color_jitter,
                hue=0.05,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        ]

    base_transforms += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ]

    return A.Compose(base_transforms)


# ═══════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════

class CoagDataset(Dataset):
    """
    PyTorch Dataset for coagulation cell images.

    Each sample returns:
        - image: (3, H, W) float32 tensor, ImageNet-normalized
        - mask: (1, H, W) float32 tensor, binary coagulation mask
        - reg_value: (1,) float32 tensor, mean inverted intensity
        - cls_label: (1,) int64 tensor, coagulation grade
        - name: str, cell image filename
    """

    def __init__(
        self,
        samples: list,
        cfg: DataConfig,
        is_train: bool = True,
    ):
        self.samples = samples
        self.cfg = cfg
        self.is_train = is_train
        self.transform = build_augmentation(cfg, is_train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(str(sample["path"]))
        if image is None:
            raise FileNotFoundError(f"Cannot load: {sample['path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = sample["mask"]

        # Albumentations: image (H,W,C) → (C,H,W) tensor, mask (H,W) → (H,W) tensor
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]  # already (3, H, W) float32 tensor
        mask = transformed["mask"].unsqueeze(0).float()  # (H, W) → (1, H, W)

        reg_value = torch.tensor([sample["reg_value"]], dtype=torch.float32)
        cls_label = torch.tensor(sample["cls_label"], dtype=torch.long)

        return {
            "image": image,
            "mask": mask,
            "reg_value": reg_value,
            "cls_label": cls_label,
            "name": sample["name"],
        }


# ═══════════════════════════════════════════════════════════════════
#  Data Loading Helpers
# ═══════════════════════════════════════════════════════════════════

def collect_cell_images(cell_dir: Path) -> list[dict]:
    """
    Scan directory for cell images and generate pseudo-labels.

    Returns list of dicts with keys: path, name, mask, reg_value, cls_label.
    """
    patterns = ["cell_*.png", "cell_*.jpg", "cell_*.jpeg", "cell_*.PNG", "cell_*.JPG"]
    paths = []
    for pat in patterns:
        paths.extend(cell_dir.glob(pat))

    if not paths:
        raise FileNotFoundError(
            f"No cell images found in {cell_dir}. "
            f"Run the classical pipeline first to generate cell_*.png files."
        )

    paths = sorted(set(paths))
    print(f"Found {len(paths)} cell images in {cell_dir}")

    return [{"path": p, "name": p.name} for p in paths]


def prepare_data(cfg: DataConfig) -> tuple[DataLoader, DataLoader, dict]:
    """
    Full data preparation pipeline:

    1. Collect cell images
    2. Generate pseudo-labels (mask + reg_value) via Otsu
    3. Assign classification labels via tertile binning
    4. Split into train/val
    5. Create DataLoaders

    Returns:
        train_loader, val_loader, metadata dict with thresholds and stats
    """
    # Collect image paths
    raw_samples = collect_cell_images(cfg.cell_dir)

    # Generate pseudo-labels for each image
    print("Generating pseudo-labels via Otsu thresholding...")
    labeled_samples = []
    for s in raw_samples:
        cell_img = cv2.imread(str(s["path"]))
        if cell_img is None:
            print(f"  WARNING: cannot read {s['path']}, skipping")
            continue
        labels = generate_pseudo_labels(cell_img, cfg)
        labeled_samples.append({**s, **labels})

    if len(labeled_samples) < 4:
        raise RuntimeError(
            f"Need at least 4 cell images for train/val split, "
            f"got {len(labeled_samples)}. Run the classical pipeline on more slides."
        )

    # Assign classification labels via global tertile binning
    labeled_samples, cls_thresholds = assign_class_labels(
        labeled_samples, cfg.cls_num_bins
    )
    print(f"Classification thresholds: {cls_thresholds}")

    # Train/val split (stratified by class)
    paths = [s["path"] for s in labeled_samples]
    cls_labels = [s["cls_label"] for s in labeled_samples]
    train_idx, val_idx = train_test_split(
        range(len(labeled_samples)),
        test_size=cfg.val_split,
        stratify=cls_labels if len(set(cls_labels)) > 1 else None,
        random_state=42,
    )

    train_samples = [labeled_samples[i] for i in train_idx]
    val_samples = [labeled_samples[i] for i in val_idx]

    print(f"Train: {len(train_samples)} cells, Val: {len(val_samples)} cells")

    # Create datasets
    train_dataset = CoagDataset(train_samples, cfg, is_train=True)
    val_dataset = CoagDataset(val_samples, cfg, is_train=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.image_size,  # will be overridden by train config
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.image_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    metadata = {
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "cls_thresholds": cls_thresholds.tolist(),
        "class_names": ["mild", "moderate", "severe"],
        "reg_range": (
            float(min(s["reg_value"] for s in labeled_samples)),
            float(max(s["reg_value"] for s in labeled_samples)),
        ),
    }

    return train_dataset, val_dataset, metadata


def create_dataloaders(
    train_dataset: CoagDataset,
    val_dataset: CoagDataset,
    train_batch_size: int = 8,
    val_batch_size: int = 4,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create DataLoaders with configured batch sizes."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
