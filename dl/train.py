"""
Training pipeline for CoagNet.

Two-phase training:
  Phase 1 (Seg pretraining): Freeze encoder, train only decoder + seg head.
  Phase 2 (Joint multi-task): Unfreeze all, train all three heads jointly.

Features:
  - Automatic mixed precision (AMP)
  - Cosine annealing with linear warmup
  - TensorBoard logging
  - Best-model checkpointing (by val Dice)
  - Early stopping
  - Gradient clipping
"""
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
    LambdaLR,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from .config import Config, TrainConfig
from .model import CoagNet
from .losses import MultiTaskLoss, compute_metrics
from .data import CoagDataset, create_dataloaders


# ═══════════════════════════════════════════════════════════════════
#  Learning Rate Scheduler Builder
# ═══════════════════════════════════════════════════════════════════

def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """Build LR scheduler with optional linear warmup."""
    total_steps = cfg.phase2_epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    if cfg.scheduler == "cosine_warmup":
        # Cosine with linear warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(cfg.min_lr / cfg.phase2_lr, 0.5 * (1 + np.cos(np.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    elif cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.min_lr)

    elif cfg.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=cfg.min_lr
        )

    elif cfg.scheduler == "step":
        return StepLR(optimizer, step_size=15, gamma=0.5)

    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


# ═══════════════════════════════════════════════════════════════════
#  Optimizer Builder
# ═══════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, cfg: TrainConfig, lr: float):
    """Build optimizer with per-parameter weight decay exclusion for norms/biases."""
    # Separate params that should not have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "norm" in name or "log_var" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if cfg.optimizer == "adamw":
        return AdamW(param_groups, lr=lr, betas=cfg.adam_betas)
    elif cfg.optimizer == "adam":
        return Adam(param_groups, lr=lr, betas=cfg.adam_betas)
    elif cfg.optimizer == "sgd":
        return SGD(param_groups, lr=lr, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


# ═══════════════════════════════════════════════════════════════════
#  Training Loop Utilities
# ═══════════════════════════════════════════════════════════════════

def to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a batch dict to the target device."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_dice: float,
    metrics: dict,
    path: Path,
):
    """Save full training state for resumption."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_dice": best_dice,
            "metrics": metrics,
        },
        path,
    )


# ═══════════════════════════════════════════════════════════════════
#  Validation Loop
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
) -> dict[str, float]:
    """Run validation on full val set."""
    model.eval()
    total_losses = {}
    total_metrics = {}
    n_batches = 0

    for batch in val_loader:
        batch = to_device(batch, device)
        outputs = model(batch["image"])

        # Targets
        targets = {
            "mask": batch["mask"],
            "reg_value": batch["reg_value"],
            "cls_label": batch["cls_label"],
        }

        _, loss_dict = loss_fn(outputs, targets)
        metrics = compute_metrics(outputs, targets)

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

    return {
        **{f"val_loss/{k}": v / n_batches for k, v in total_losses.items()},
        **{f"val_metric/{k}": v / n_batches for k, v in total_metrics.items()},
    }


# ═══════════════════════════════════════════════════════════════════
#  Phase 1: Segmentation Pretraining
# ═══════════════════════════════════════════════════════════════════

def train_phase1(
    model: CoagNet,
    train_loader,
    val_loader,
    loss_fn: MultiTaskLoss,
    cfg: TrainConfig,
    device: torch.device,
    writer: SummaryWriter,
) -> Path:
    """
    Phase 1: Freeze encoder, train decoder + segmentation head only.
    Uses only segmentation loss. Quick pretraining to stabilize decoder.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: Segmentation Pretraining (encoder frozen)")
    print("=" * 60)

    model.freeze_encoder()
    model.train()

    # Only optimize parameters that require gradients
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.phase1_lr,
        weight_decay=cfg.weight_decay,
    )
    scaler = GradScaler(enabled=cfg.use_amp)

    best_dice = 0.0
    global_step = 0

    for epoch in range(1, cfg.phase1_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Phase1 E{epoch}/{cfg.phase1_epochs}")

        for batch in pbar:
            batch = to_device(batch, device)

            with autocast(enabled=cfg.use_amp):
                outputs = model(batch["image"])
                targets = {
                    "mask": batch["mask"],
                    "reg_value": batch["reg_value"],
                    "cls_label": batch["cls_label"],
                }
                loss, loss_dict = loss_fn(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss_dict["seg"]
            pbar.set_postfix({"loss_seg": f"{loss_dict['seg']:.4f}"})

            writer.add_scalar("phase1/train_loss_seg", loss_dict["seg"], global_step)
            global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch} avg seg loss: {avg_loss:.4f}")

        # Validation
        if epoch % cfg.val_interval == 0:
            val_metrics = validate(model, val_loader, loss_fn, device)
            dice = val_metrics.get("val_metric/dice_score", 0.0)
            for k, v in val_metrics.items():
                writer.add_scalar(f"phase1/{k}", v, epoch)

            print(f"  Val dice: {dice:.4f}  (best: {best_dice:.4f})")

            if dice > best_dice:
                best_dice = dice
                ckpt_path = cfg.save_dir / "phase1_best.pt"
                save_checkpoint(model, optimizer, None, epoch, best_dice, val_metrics, ckpt_path)
                print(f"  → Saved best model to {ckpt_path}")

    # Save phase 1 final
    ckpt_path = cfg.save_dir / "phase1_final.pt"
    save_checkpoint(model, optimizer, None, cfg.phase1_epochs, best_dice, {}, ckpt_path)
    print(f"  Phase 1 complete. Final model: {ckpt_path}")
    return ckpt_path


# ═══════════════════════════════════════════════════════════════════
#  Phase 2: Joint Multi-Task Training
# ═══════════════════════════════════════════════════════════════════

def train_phase2(
    model: CoagNet,
    train_loader,
    val_loader,
    loss_fn: MultiTaskLoss,
    cfg: TrainConfig,
    device: torch.device,
    writer: SummaryWriter,
) -> Path:
    """
    Phase 2: Unfreeze all layers, joint multi-task training with
    uncertainty-weighted loss, cosine warmup LR, and early stopping.
    """
    print("\n" + "=" * 60)
    print("  PHASE 2: Joint Multi-Task Training (all layers)")
    print("=" * 60)

    model.unfreeze_encoder()
    model.train()

    optimizer = build_optimizer(model, cfg, lr=cfg.phase2_lr)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler(enabled=cfg.use_amp)

    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    global_step = 0

    for epoch in range(1, cfg.phase2_epochs + 1):
        model.train()
        epoch_losses = {}
        pbar = tqdm(train_loader, desc=f"Phase2 E{epoch}/{cfg.phase2_epochs}")

        for batch in pbar:
            batch = to_device(batch, device)

            with autocast(enabled=cfg.use_amp):
                outputs = model(batch["image"])
                targets = {
                    "mask": batch["mask"],
                    "reg_value": batch["reg_value"],
                    "cls_label": batch["cls_label"],
                }
                loss, loss_dict = loss_fn(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            # Cosine warmup: step every batch
            if cfg.scheduler in ("cosine_warmup", "cosine"):
                scheduler.step()

            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            pbar.set_postfix({
                "seg": f"{loss_dict.get('seg', 0):.3f}",
                "reg": f"{loss_dict.get('reg', 0):.3f}",
                "cls": f"{loss_dict.get('cls', 0):.3f}",
            })

            if global_step % cfg.log_interval == 0:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"phase2/train_loss/{k}", v, global_step)
                writer.add_scalar(
                    "phase2/lr", optimizer.param_groups[0]["lr"], global_step
                )

            global_step += 1

        # Epoch summary
        n_batches = len(train_loader)
        summary = "  ".join(
            f"{k}: {v/n_batches:.4f}" for k, v in epoch_losses.items()
        )
        print(f"  Epoch {epoch}  {summary}  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Validation
        val_metrics = validate(model, val_loader, loss_fn, device)
        for k, v in val_metrics.items():
            writer.add_scalar(f"phase2/{k}", v, epoch)

        current_dice = val_metrics.get("val_metric/dice_score", 0.0)
        current_mae = val_metrics.get("val_metric/reg_mae", float("inf"))
        current_acc = val_metrics.get("val_metric/cls_accuracy", 0.0)

        print(
            f"  Val → Dice: {current_dice:.4f}  "
            f"MAE: {current_mae:.2f}  Acc: {current_acc:.2%}  "
            f"(best Dice: {best_dice:.4f} @ epoch {best_epoch})"
        )

        # Plateau scheduler
        if cfg.scheduler == "plateau":
            scheduler.step(current_dice)

        # Save best by Dice
        if current_dice > best_dice:
            best_dice = current_dice
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = cfg.save_dir / "best_model.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_dice, val_metrics, ckpt_path
            )
            print(f"  → Saved best model to {ckpt_path}")
        else:
            patience_counter += 1

        # Save latest
        if cfg.save_latest:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_dice, val_metrics,
                cfg.save_dir / "latest.pt"
            )

        # Early stopping
        if patience_counter >= cfg.early_stopping_patience:
            print(
                f"\n  Early stopping triggered after {cfg.early_stopping_patience} "
                f"epochs without improvement."
            )
            break

    final_path = cfg.save_dir / "final_model.pt"
    save_checkpoint(
        model, optimizer, scheduler, epoch, best_dice, val_metrics, final_path
    )
    print(f"\n  Phase 2 complete. Best Dice: {best_dice:.4f} @ epoch {best_epoch}")
    print(f"  Final model: {final_path}")
    return final_path


# ═══════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════

def train(config: Optional[Config] = None) -> CoagNet:
    """
    Run the full two-phase training pipeline.

    Args:
        config: Config object. If None, uses defaults from Config().

    Returns:
        Trained CoagNet model (in eval mode, on CPU).
    """
    if config is None:
        config = Config()

    cfg = config.train
    device = torch.device(cfg.device)

    # ── Reproducibility ──
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    # ── TensorBoard ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = cfg.log_dir / timestamp
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard: {log_dir}")

    # ── Data ──
    from .data import prepare_data

    train_dataset, val_dataset, meta = prepare_data(config.data)
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        train_batch_size=cfg.phase2_batch_size,
        val_batch_size=cfg.phase2_batch_size * 2,
        num_workers=config.data.num_workers,
    )

    print(f"\nData summary:")
    print(f"  Train: {meta['num_train']} cells")
    print(f"  Val:   {meta['num_val']} cells")
    print(f"  Class thresholds: {meta['cls_thresholds']}")
    print(f"  Reg range: [{meta['reg_range'][0]:.1f}, {meta['reg_range'][1]:.1f}]")

    # ── Model ──
    print(f"\nBuilding CoagNet with encoder: {config.model.encoder}")
    model = CoagNet(config.model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable:    {trainable_params:,}")

    # ── Loss ──
    loss_fn = MultiTaskLoss(config.loss).to(device)
    print(f"  Uncertainty weighting: {config.loss.uncertainty_weighting}")

    # ── Phase 1 ──
    p1_batch_size = min(cfg.phase1_batch_size, meta["num_train"])
    p1_train_loader, _ = create_dataloaders(
        train_dataset, val_dataset,
        train_batch_size=p1_batch_size,
        val_batch_size=p1_batch_size * 2,
        num_workers=config.data.num_workers,
    )
    train_phase1(model, p1_train_loader, val_loader, loss_fn, cfg, device, writer)

    # ── Phase 2 ──
    p2_batch_size = min(cfg.phase2_batch_size, meta["num_train"] // 2)
    p2_train_loader, p2_val_loader = create_dataloaders(
        train_dataset, val_dataset,
        train_batch_size=max(2, p2_batch_size),
        val_batch_size=max(2, p2_batch_size * 2),
        num_workers=config.data.num_workers,
    )
    train_phase2(model, p2_train_loader, p2_val_loader, loss_fn, cfg, device, writer)

    # ── Cleanup ──
    writer.close()
    model.eval()
    model.cpu()

    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Logs:    {log_dir}")
    print(f"  Model:   {cfg.save_dir / 'best_model.pt'}")
    print(f"{'=' * 60}")

    return model


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
