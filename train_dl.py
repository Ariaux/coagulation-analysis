#!/usr/bin/env python3
"""
CoagNet Training Entry Point.

Usage:
    python train_dl.py                          # train with defaults
    python train_dl.py --cell-dir input/        # specify cell image directory
    python train_dl.py --encoder efficientnet-b3 --epochs 80
"""
import sys
import argparse
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dl.config import Config
from dl.train import train


def main():
    parser = argparse.ArgumentParser(
        description="CoagNet: Multi-Task Deep Learning for Coagulation Quantification"
    )
    # Data
    parser.add_argument("--cell-dir", type=Path, default=Path("input"),
                        help="Directory containing cell_*.png images")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio")

    # Model
    parser.add_argument("--encoder", type=str, default="resnet50",
                        choices=["resnet34", "resnet50", "resnet101",
                                 "efficientnet-b0", "efficientnet-b3",
                                 "timm-resnest50d", "timm-convnext_tiny"],
                        help="Encoder backbone")

    # Training
    parser.add_argument("--phase1-epochs", type=int, default=10,
                        help="Phase 1 (seg pretraining) epochs")
    parser.add_argument("--phase2-epochs", type=int, default=50,
                        help="Phase 2 (joint training) epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (phase 2)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")

    # Hardware
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, cpu, or auto")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output
    parser.add_argument("--save-dir", type=Path, default=Path("dl/checkpoints"),
                        help="Checkpoint save directory")
    parser.add_argument("--log-dir", type=Path, default=Path("dl/logs"),
                        help="TensorBoard log directory")

    args = parser.parse_args()

    # Build config
    config = Config()

    config.data.cell_dir = args.cell_dir
    config.data.image_size = args.image_size
    config.data.val_split = args.val_split
    config.data.num_workers = args.num_workers

    config.model.encoder = args.encoder

    config.train.phase1_epochs = args.phase1_epochs
    config.train.phase2_epochs = args.phase2_epochs
    config.train.phase2_batch_size = args.batch_size
    config.train.phase2_lr = args.lr
    config.train.use_amp = not args.no_amp
    config.train.device = args.device if args.device != "auto" else config.train.device
    config.train.seed = args.seed
    config.train.save_dir = args.save_dir
    config.train.log_dir = args.log_dir

    # Run training
    model = train(config)
    return model


if __name__ == "__main__":
    main()
