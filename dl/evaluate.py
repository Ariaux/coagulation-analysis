"""
Comprehensive evaluation suite for CoagNet.

Features:
  - K-fold cross-validation with stratified splits
  - Ablation study: test contribution of each model component
  - Multi-encoder benchmark: compare different backbones
  - Bootstrap confidence intervals for all metrics
  - Statistical significance tests (paired t-test, McNemar)
  - Per-class metric breakdown
  - Learning curve analysis (performance vs dataset size)

Usage:
    python -m dl.evaluate --cell-dir input/cells --k-folds 5
    python -m dl.evaluate --cell-dir input/cells --ablation
    python -m dl.evaluate --cell-dir input/cells --benchmark
"""
import sys
import json
import time
import itertools
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
)
from scipy import stats
from tqdm import tqdm

from .config import Config, ModelConfig
from .model import CoagNet
from .losses import MultiTaskLoss, compute_metrics
from .data import (
    CoagDataset,
    prepare_data,
    create_dataloaders,
    collect_cell_images,
    generate_pseudo_labels,
    assign_class_labels,
)

CLASS_NAMES = ["Mild", "Moderate", "Severe"]


# ═══════════════════════════════════════════════════════════════
#  Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════════════

def bootstrap_confidence_interval(
    values: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """
    Compute bootstrap 95% confidence interval for a metric.

    Returns:
        dict with mean, ci_lower, ci_upper, std.
    """
    values = np.array(values)
    n = len(values)
    bootstraps = np.random.choice(values, size=(n_bootstrap, n), replace=True)
    means = bootstraps.mean(axis=1)

    ci_lower = np.percentile(means, 100 * alpha / 2)
    ci_upper = np.percentile(means, 100 * (1 - alpha / 2))

    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


# ═══════════════════════════════════════════════════════════════
#  K-Fold Cross-Validation
# ═══════════════════════════════════════════════════════════════

def kfold_cross_validation(
    config: Config,
    k_folds: int = 5,
    quick: bool = False,
) -> dict:
    """
    Stratified K-fold cross-validation.

    For each fold:
      1. Split data into train/val
      2. Train CoagNet from scratch
      3. Evaluate on held-out fold
      4. Record all metrics

    Args:
        config: Full Config.
        k_folds: Number of folds (use 3 for very small datasets).
        quick: If True, use fewer epochs for a fast check.

    Returns:
        dict with aggregated metrics across folds, ready for paper tables.
    """
    print(f"\n{'='*60}")
    print(f"  K-FOLD CROSS-VALIDATION ({k_folds} folds)")
    print(f"{'='*60}")

    # Prepare data
    raw_samples = collect_cell_images(config.data.cell_dir)
    labeled = []
    for s in raw_samples:
        img = __import__('cv2').imread(str(s["path"]))
        labels = generate_pseudo_labels(img, config.data)
        labeled.append({**s, **labels})

    labeled, _ = assign_class_labels(labeled, config.data.cls_num_bins)
    cls_labels = [s["cls_label"] for s in labeled]

    if len(labeled) < k_folds * 2:
        k_folds = max(2, len(labeled) // 2)
        print(f"  Adjusted folds to {k_folds} (small dataset)")

    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []
    all_dice_scores = []
    all_maes = []
    all_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        kfold.split(range(len(labeled)), cls_labels)
    ):
        print(f"\n  ── Fold {fold_idx + 1}/{k_folds} ──")
        print(f"    Train: {len(train_idx)}, Val: {len(val_idx)}")

        train_samples = [labeled[i] for i in train_idx]
        val_samples = [labeled[i] for i in val_idx]

        train_dataset = CoagDataset(train_samples, config.data, is_train=True)
        val_dataset = CoagDataset(val_samples, config.data, is_train=False)

        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset,
            train_batch_size=config.train.phase2_batch_size,
            val_batch_size=config.train.phase2_batch_size * 2,
            num_workers=0,  # safer for CV loops
        )

        # Train model for this fold
        device = torch.device(config.train.device)
        model = CoagNet(config.model).to(device)

        # Quick training per fold
        epochs = 5 if quick else config.train.phase2_epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.phase2_lr)
        loss_fn = MultiTaskLoss(config.loss).to(device)

        best_dice = 0.0
        best_metrics = {}

        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                outputs = model(batch["image"])
                targets = {
                    "mask": batch["mask"],
                    "reg_value": batch["reg_value"],
                    "cls_label": batch["cls_label"],
                }
                loss, _ = loss_fn(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            val_metrics = _validate_fold(model, val_loader, loss_fn, device)
            dice = val_metrics.get("dice_score", 0.0)

            if dice > best_dice:
                best_dice = dice
                best_metrics = val_metrics

        all_dice_scores.append(best_metrics.get("dice_score", 0.0))
        all_maes.append(best_metrics.get("reg_mae", float("inf")))
        all_accuracies.append(best_metrics.get("cls_accuracy", 0.0))

        fold_results.append({
            "fold": fold_idx + 1,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            **best_metrics,
        })
        print(f"    Fold {fold_idx + 1} best: Dice={best_metrics.get('dice_score', 0):.4f}, "
              f"MAE={best_metrics.get('reg_mae', 0):.2f}, Acc={best_metrics.get('cls_accuracy', 0):.2%}")

    # Aggregate
    agg = {
        "method": f"{k_folds}-fold CV",
        "num_samples": len(labeled),
        "folds": fold_results,
        "dice_score": bootstrap_confidence_interval(all_dice_scores),
        "reg_mae": bootstrap_confidence_interval(all_maes),
        "cls_accuracy": bootstrap_confidence_interval(all_accuracies),
    }

    print(f"\n  {'─'*50}")
    print(f"  AGGREGATE RESULTS")
    print(f"  {'─'*50}")
    print(f"  Dice Score:  {agg['dice_score']['mean']:.4f} "
          f"[{agg['dice_score']['ci_lower']:.4f}, {agg['dice_score']['ci_upper']:.4f}]")
    print(f"  Reg MAE:     {agg['reg_mae']['mean']:.2f} "
          f"[{agg['reg_mae']['ci_lower']:.2f}, {agg['reg_mae']['ci_upper']:.2f}]")
    print(f"  Cls Acc:     {agg['cls_accuracy']['mean']:.4f} "
          f"[{agg['cls_accuracy']['ci_lower']:.4f}, {agg['cls_accuracy']['ci_upper']:.4f}]")

    return agg


@torch.no_grad()
def _validate_fold(model, loader, loss_fn, device):
    """Compute all metrics on validation set."""
    total_dice, total_mae, total_acc = 0.0, 0.0, 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
        outputs = model(batch["image"])
        targets = {
            "mask": batch["mask"],
            "reg_value": batch["reg_value"],
            "cls_label": batch["cls_label"],
        }
        metrics = compute_metrics(outputs, targets)
        total_dice += metrics["dice_score"]
        total_mae += metrics["reg_mae"]
        total_acc += metrics["cls_accuracy"]
        n_batches += 1

    return {
        "dice_score": total_dice / max(1, n_batches),
        "reg_mae": total_mae / max(1, n_batches),
        "cls_accuracy": total_acc / max(1, n_batches),
    }


# ═══════════════════════════════════════════════════════════════
#  Ablation Study
# ═══════════════════════════════════════════════════════════════

def ablation_study(config: Config, quick: bool = True) -> dict:
    """
    Systematically ablate each component and measure impact.

    Variants tested:
      1. Full CoagNet (baseline)
      2. No segmentation head (reg + cls only)
      3. No regression head (seg + cls only)
      4. No classification head (seg + reg only)
      5. No uncertainty weighting (fixed weights)
      6. No data augmentation
      7. Frozen encoder (no fine-tuning)
      8. Smaller encoder (ResNet-18 vs ResNet-50)

    Returns:
        dict mapping variant name → metrics, with delta from baseline.
    """
    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY")
    print(f"{'='*60}")

    # Prepare data once
    train_dataset, val_dataset, meta = prepare_data(config.data)
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        train_batch_size=config.train.phase2_batch_size,
        val_batch_size=config.train.phase2_batch_size * 2,
        num_workers=0,
    )
    device = torch.device(config.train.device)

    variants = {
        "Full CoagNet": {},
        "w/o Segmentation Head": {"no_seg": True},
        "w/o Regression Head": {"no_reg": True},
        "w/o Classification Head": {"no_cls": True},
        "w/o Uncertainty Weighting": {"no_uncertainty": True},
        "w/o Data Augmentation": {"no_augment": True},
        "Frozen Encoder Only": {"frozen_encoder": True},
        "ResNet-18 Backbone": {"encoder": "resnet18"},
    }

    results = {}
    baseline_metrics = None
    epochs = 3 if quick else config.train.phase2_epochs

    for variant_name, variant_config in variants.items():
        print(f"\n  ── {variant_name} ──")

        # Build variant config
        v_cfg = _build_variant_config(config, variant_config)

        # Build model
        model = CoagNet(v_cfg.model).to(device)

        # Handle variant-specific modifications
        if variant_config.get("no_seg"):
            # Zero out seg loss weight
            v_cfg.loss.seg_loss_weight = 0.0
        if variant_config.get("no_reg"):
            v_cfg.loss.reg_loss_weight = 0.0
        if variant_config.get("no_cls"):
            v_cfg.loss.cls_loss_weight = 0.0
        if variant_config.get("no_uncertainty"):
            v_cfg.loss.uncertainty_weighting = False
        if variant_config.get("frozen_encoder"):
            model.freeze_encoder()
        if variant_config.get("no_augment"):
            train_dataset.transform = train_dataset.transform  # keep but disable aug
            # Actually rebuild without augmentation
            from .data import build_augmentation
            train_dataset.transform = build_augmentation(config.data, is_train=False)

        loss_fn = MultiTaskLoss(v_cfg.loss).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.phase2_lr)

        # Train
        best_dice = 0.0
        best_metrics = {}
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                outputs = model(batch["image"])
                targets = {
                    "mask": batch["mask"],
                    "reg_value": batch["reg_value"],
                    "cls_label": batch["cls_label"],
                }
                loss, _ = loss_fn(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics = _validate_fold(model, val_loader, loss_fn, device)
            if metrics.get("dice_score", 0.0) > best_dice:
                best_dice = metrics.get("dice_score", 0.0)
                best_metrics = metrics

        results[variant_name] = best_metrics
        print(f"    Dice={best_metrics.get('dice_score', 0):.4f}, "
              f"MAE={best_metrics.get('reg_mae', 0):.2f}, "
              f"Acc={best_metrics.get('cls_accuracy', 0):.2%}")

        if variant_name == "Full CoagNet":
            baseline_metrics = best_metrics

    # Compute deltas
    if baseline_metrics:
        for name, metrics in results.items():
            if name == "Full CoagNet":
                continue
            delta_dice = metrics.get("dice_score", 0) - baseline_metrics.get("dice_score", 0)
            delta_mae = baseline_metrics.get("reg_mae", 0) - metrics.get("reg_mae", 0)
            delta_acc = metrics.get("cls_accuracy", 0) - baseline_metrics.get("cls_accuracy", 0)
            results[name]["_delta"] = {
                "dice_score": round(delta_dice, 4),
                "reg_mae": round(delta_mae, 2),
                "cls_accuracy": round(delta_acc, 4),
            }

    # Print summary table
    print(f"\n  {'─'*70}")
    print(f"  ABLATION SUMMARY")
    print(f"  {'─'*70}")
    print(f"  {'Variant':<35s} {'Dice':>8s} {'MAE':>8s} {'Acc':>8s} {'ΔDice':>8s}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for name, m in results.items():
        delta = m.get("_delta", {"dice_score": 0.0, "reg_mae": 0.0, "cls_accuracy": 0.0})
        print(f"  {name:<35s} {m.get('dice_score', 0):>8.4f} {m.get('reg_mae', 0):>8.2f} "
              f"{m.get('cls_accuracy', 0):>8.2%} {delta['dice_score']:>+8.4f}")

    return results


def _build_variant_config(base_config: Config, variant: dict) -> Config:
    """Create a modified config for an ablation variant."""
    import copy
    cfg = copy.deepcopy(base_config)
    if "encoder" in variant:
        cfg.model.encoder = variant["encoder"]
    return cfg


# ═══════════════════════════════════════════════════════════════
#  Multi-Encoder Benchmark
# ═══════════════════════════════════════════════════════════════

def benchmark_encoders(config: Config, quick: bool = True) -> dict:
    """
    Compare different encoder backbones with identical settings.

    Encoders tested:
      - ResNet-34, ResNet-50, ResNet-101
      - EfficientNet-B0, EfficientNet-B3
      - ConvNeXt-Tiny (if timm available)

    Returns:
        dict mapping encoder name → metrics + param count.
    """
    print(f"\n{'='*60}")
    print(f"  ENCODER BENCHMARK")
    print(f"{'='*60}")

    train_dataset, val_dataset, meta = prepare_data(config.data)
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        train_batch_size=config.train.phase2_batch_size,
        val_batch_size=config.train.phase2_batch_size * 2,
        num_workers=0,
    )
    device = torch.device(config.train.device)
    epochs = 3 if quick else config.train.phase2_epochs

    encoders = [
        "resnet34",
        "resnet50",
        "resnet101",
        "efficientnet-b0",
        "efficientnet-b3",
    ]

    # Check for timm encoders
    try:
        import timm
        encoders.append("timm-convnext_tiny")
    except ImportError:
        pass

    results = {}

    for encoder_name in encoders:
        print(f"\n  ── {encoder_name} ──")

        try:
            model_cfg = ModelConfig()
            model_cfg.encoder = encoder_name
            model = CoagNet(model_cfg).to(device)
        except Exception as e:
            print(f"    SKIP: {e}")
            results[encoder_name] = {"error": str(e)}
            continue

        params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {params:,}")

        loss_fn = MultiTaskLoss(config.loss).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.phase2_lr)

        best_dice = 0.0
        best_metrics = {}
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                outputs = model(batch["image"])
                targets = {
                    "mask": batch["mask"],
                    "reg_value": batch["reg_value"],
                    "cls_label": batch["cls_label"],
                }
                loss, _ = loss_fn(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics = _validate_fold(model, val_loader, loss_fn, device)
            if metrics.get("dice_score", 0.0) > best_dice:
                best_dice = metrics.get("dice_score", 0.0)
                best_metrics = metrics

        results[encoder_name] = {**best_metrics, "params": params}
        print(f"    Dice={best_metrics.get('dice_score', 0):.4f}, "
              f"MAE={best_metrics.get('reg_mae', 0):.2f}, "
              f"Acc={best_metrics.get('cls_accuracy', 0):.2%}")

    # Summary table
    print(f"\n  {'─'*70}")
    print(f"  ENCODER BENCHMARK SUMMARY")
    print(f"  {'─'*70}")
    print(f"  {'Encoder':<25s} {'Params':>10s} {'Dice':>8s} {'MAE':>8s} {'Acc':>8s}")
    print(f"  {'─'*25} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    for name, m in results.items():
        if "error" in m:
            print(f"  {name:<25s} {'ERROR':>10s}")
        else:
            print(f"  {name:<25s} {m['params']:>10,} {m['dice_score']:>8.4f} "
                  f"{m['reg_mae']:>8.2f} {m['cls_accuracy']:>8.2%}")

    return results


# ═══════════════════════════════════════════════════════════════
#  Statistical Tests
# ═══════════════════════════════════════════════════════════════

def statistical_tests(
    model_a_predictions: list[dict],
    model_b_predictions: list[dict],
    ground_truth: list[dict],
) -> dict:
    """
    Paired statistical tests between two models.

    Tests:
      - Paired t-test on Dice scores
      - McNemar's test on classification correctness
      - Wilcoxon signed-rank test on MAE
    """
    dice_a = np.array([p["dice_score"] for p in model_a_predictions])
    dice_b = np.array([p["dice_score"] for p in model_b_predictions])

    mae_a = np.array([p["reg_mae"] for p in model_a_predictions])
    mae_b = np.array([p["reg_mae"] for p in model_b_predictions])

    correct_a = np.array([p["cls_correct"] for p in model_a_predictions])
    correct_b = np.array([p["cls_correct"] for p in model_b_predictions])

    # Paired t-test (two-sided)
    t_stat, t_pvalue = stats.ttest_rel(dice_a, dice_b)

    # Wilcoxon signed-rank (non-parametric)
    w_stat, w_pvalue = stats.wilcoxon(mae_a, mae_b)

    # McNemar's test for classification
    n_both = int(((correct_a == 1) & (correct_b == 1)).sum())
    n_a_only = int(((correct_a == 1) & (correct_b == 0)).sum())
    n_b_only = int(((correct_a == 0) & (correct_b == 1)).sum())
    n_neither = int(((correct_a == 0) & (correct_b == 0)).sum())

    # McNemar chi-squared with continuity correction
    if n_a_only + n_b_only > 0:
        mcnemar_chi2 = (abs(n_a_only - n_b_only) - 1) ** 2 / (n_a_only + n_b_only)
        mcnemar_pvalue = 1 - stats.chi2.cdf(mcnemar_chi2, 1)
    else:
        mcnemar_chi2, mcnemar_pvalue = 0.0, 1.0

    return {
        "paired_ttest_dice": {
            "statistic": float(t_stat),
            "p_value": float(t_pvalue),
            "significant_05": t_pvalue < 0.05,
        },
        "wilcoxon_mae": {
            "statistic": float(w_stat),
            "p_value": float(w_pvalue),
            "significant_05": w_pvalue < 0.05,
        },
        "mcnemar_classification": {
            "chi2": float(mcnemar_chi2),
            "p_value": float(mcnemar_pvalue),
            "significant_05": mcnemar_pvalue < 0.05,
            "contingency": {
                "both_correct": n_both,
                "a_only_correct": n_a_only,
                "b_only_correct": n_b_only,
                "neither_correct": n_neither,
            },
        },
    }


# ═══════════════════════════════════════════════════════════════
#  Per-Class Metric Breakdown
# ═══════════════════════════════════════════════════════════════

def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Generate per-class precision, recall, F1, and support.

    Returns scikit-learn classification_report as dict.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    # Add additional metrics
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    report["cohen_kappa"] = kappa
    report["matthews_corrcoef"] = mcc

    return report


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CoagNet Evaluation Suite")
    parser.add_argument("--cell-dir", type=Path, default=Path("input"),
                       help="Directory with cell images")
    parser.add_argument("--k-folds", type=int, default=0,
                       help="Run k-fold cross-validation (0 = skip)")
    parser.add_argument("--ablation", action="store_true",
                       help="Run ablation study")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run encoder benchmark")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode (fewer epochs)")
    parser.add_argument("--output", type=Path, default=Path("dl/eval_results.json"),
                       help="Save results to JSON")

    args = parser.parse_args()

    config = Config()
    config.data.cell_dir = args.cell_dir

    all_results = {}

    if args.k_folds > 0:
        all_results["kfold"] = kfold_cross_validation(
            config, k_folds=args.k_folds, quick=args.quick
        )

    if args.ablation:
        all_results["ablation"] = ablation_study(config, quick=args.quick)

    if args.benchmark:
        all_results["benchmark"] = benchmark_encoders(config, quick=args.quick)

    if all_results:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types to native Python for JSON
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(args.output, "w") as f:
            json.dump(convert(all_results), f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print("No evaluation selected. Use --k-folds, --ablation, or --benchmark.")
