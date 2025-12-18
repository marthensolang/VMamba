#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title
-----
Gradient Masking Fine-Tuning for Vision Mamba (Research / Negative Control)

Purpose
-------
This script fine-tunes a Vision Mamba (Vim) classifier while applying techniques that
*obfuscate or dampen gradients* (commonly referred to as "gradient masking").
It is intended for research analysis (e.g., as a negative control) to illustrate that
apparent robustness can be misleading under adaptive attacks.

Important note
--------------
Gradient masking is NOT a recommended defense. Many masking-style techniques can create
"false robustness" that breaks under stronger or adaptive attacks. Use this script to
reproduce experiments and document results transparently.

What the script does
--------------------
1) Finds the latest baseline Vim training directory and loads its checkpoint.
2) Detects the architecture (dim/depth) from the checkpoint state_dict to rebuild Vim.
3) Wraps the model with:
   - optional gradient noise injection (on parameter gradients)
   - optional temperature scaling on logits (to soften gradients)
   - optional gradient clipping (to limit gradient norms)
   - label smoothing in the loss
4) Fine-tunes on the TSR dataset and saves:
   - best checkpoint
   - training history (CSV)
   - config + architecture (JSON)
   - plots and gradient distribution stats

Pseudocode
----------
base_dir = find_latest_save_dir()
ckpt     = load(base_dir/best_vim_rambu_small.pth)
classes  = read(base_dir/class_mapping.json)
vim_kwargs = build_kwargs_matching_checkpoint(ckpt, num_classes=len(classes))

model = Vim(**vim_kwargs); load_state_dict(ckpt)
trainer = Trainer(model, config)

for epoch in epochs:
    train_epoch():
        forward -> logits
        loss = CrossEntropyWithLabelSmoothing(logits / T, y)
        backward
        optionally add noise to parameter gradients
        optionally clip gradients
        optimizer step
    validate_epoch()

save best model + plots + gradient distribution

Usage
-----
DATA_ROOT=dataset_rambu_lalu_lintas \
BASE_PATTERN=outputs_vim_rambu_small_* \
python gradient_masking_vim_github.py

Environment variables
---------------------
DATA_ROOT      : dataset root containing train/valid/test (default: dataset_rambu_lalu_lintas)
BASE_PATTERN   : glob pattern to find baseline run folder (default: outputs_vim_rambu_small_*)
EPOCHS         : number of fine-tuning epochs (default: 30)
BATCH_SIZE     : batch size (default: 32)
LR             : learning rate (default: 1e-5)
WEIGHT_DECAY   : AdamW weight decay (default: 1e-4)
TEMP           : temperature scaling T for logits (default: 20)
LABEL_SMOOTH   : label smoothing epsilon (default: 0.1)
GRAD_NOISE_STD : std for gradient noise injection (default: 0.01)
GRAD_CLIP      : max norm for gradient clipping (default: 1.0)

Output
------
Creates: outputs_gradient_masking_YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import glob
import inspect
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from vision_mamba import Vim


# -----------------------------------------------------------------------------
# 1) Utilities: find baseline run directory and detect architecture
# -----------------------------------------------------------------------------
def find_latest_save_dir(pattern: str) -> Path:
    """Find the latest baseline training output directory that contains required files."""
    candidates = sorted([Path(p) for p in glob.glob(pattern) if Path(p).is_dir()])
    if not candidates:
        raise FileNotFoundError(
            f"Could not find any baseline output directory matching pattern: {pattern}"
        )

    for p in reversed(candidates):
        if (p / "best_vim_rambu_small.pth").exists() and (p / "class_mapping.json").exists():
            return p

    raise FileNotFoundError(
        f"Found directories for pattern '{pattern}', but none contain both "
        f"'best_vim_rambu_small.pth' and 'class_mapping.json'."
    )


def _extract_state_dict(checkpoint: dict) -> Dict[str, torch.Tensor]:
    """Return a state_dict from a checkpoint that may wrap it under 'model'."""
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        # Might already be a state_dict-shaped dict
        return checkpoint
    raise TypeError("Unsupported checkpoint format; expected a dict or a dict-like state_dict.")


def detect_model_architecture_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """
    Heuristically detect (dim, depth) from a Vim-like state_dict.

    - dim: tries cls_token or head.weight or patch-embed projection weight shapes
    - depth: inferred from max layer index found in keys (layers./blocks.)
    """
    dim: Optional[int] = None
    depth: int = 0

    # 1) dim via cls_token
    for k, v in state_dict.items():
        if "cls_token" in k and hasattr(v, "shape") and len(v.shape) >= 1:
            dim = int(v.shape[-1])
            break

    # 2) dim via classifier head
    if dim is None:
        for k, v in state_dict.items():
            if k.endswith("head.weight") and hasattr(v, "shape") and len(v.shape) == 2:
                dim = int(v.shape[1])
                break

    # 3) dim via patch embedding projection weight
    if dim is None:
        for k, v in state_dict.items():
            if any(s in k.lower() for s in ["patch", "embed", "to_patch"]) and k.endswith("weight"):
                if hasattr(v, "shape") and len(v.shape) in (2, 4):
                    dim = int(v.shape[0])
                    break

    # Depth via layer indices
    for key in state_dict.keys():
        for token in ["layers.", "blocks."]:
            if token in key:
                try:
                    idx = int(key.split(token)[1].split(".")[0])
                    depth = max(depth, idx + 1)  # zero-indexed
                except Exception:
                    pass

    if dim is None or depth <= 0:
        raise RuntimeError(
            f"Failed to detect architecture from state_dict (dim={dim}, depth={depth}). "
            "Please ensure you are using a compatible Vim checkpoint."
        )

    print(f"[ARCH DETECT] dim={dim}, depth={depth}")
    return {"dim": dim, "depth": depth}


def build_vim_kwargs_from_checkpoint(num_classes: int, checkpoint_path: Path) -> Dict:
    """Build Vim kwargs that (best-effort) match the checkpoint architecture."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)

    arch = detect_model_architecture_from_state_dict(state_dict)

    sig = inspect.signature(Vim).parameters

    # Conservative defaults; adjust only what we can infer reliably.
    kwargs = {
        "dim": arch["dim"],
        "dt_rank": max(1, arch["dim"] // 8),  # heuristic used in prior scripts
        "dim_inner": arch["dim"],
        "d_state": 64,
        "num_classes": num_classes,
        "image_size": 224,
        "patch_size": 32,
        "channels": 3,
        "dropout": 0.1,
        "depth": arch["depth"],
    }

    if "heads" in sig:
        kwargs["heads"] = 4

    # Keep only parameters supported by the installed Vim signature
    filtered = {k: v for k, v in kwargs.items() if k in sig}
    print(f"[MODEL KWARGS] {filtered}")
    return filtered


# -----------------------------------------------------------------------------
# 2) Configuration
# -----------------------------------------------------------------------------
class GradientMaskingConfig:
    """
    Configuration for gradient masking style fine-tuning.

    This intentionally combines multiple "masking-like" ingredients:
    - temperature scaling on logits (TEMP)
    - label smoothing in CE loss (LABEL_SMOOTH)
    - gradient noise injection after backward (GRAD_NOISE_STD)
    - gradient clipping (GRAD_CLIP)
    """

    def __init__(self) -> None:
        # Training hyperparameters
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self.epochs = int(os.getenv("EPOCHS", "30"))
        self.learning_rate = float(os.getenv("LR", "1e-5"))
        self.weight_decay = float(os.getenv("WEIGHT_DECAY", "1e-4"))

        # Masking / obfuscation knobs
        self.temperature = float(os.getenv("TEMP", "20"))
        self.label_smoothing = float(os.getenv("LABEL_SMOOTH", "0.1"))
        self.grad_noise_std = float(os.getenv("GRAD_NOISE_STD", "0.01"))
        self.grad_clip = float(os.getenv("GRAD_CLIP", "1.0"))

        self.use_grad_noise = True
        self.use_temp_scaling = True
        self.use_grad_clip = True


# -----------------------------------------------------------------------------
# 3) Wrapper and loss
# -----------------------------------------------------------------------------
class MaskingWrapper(nn.Module):
    """
    A thin wrapper around the base model.

    We keep the forward output as *logits* to avoid incorrect loss usage
    (e.g., passing probabilities into CrossEntropyLoss).
    """

    def __init__(self, base_model: nn.Module, config: GradientMaskingConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        if self.config.use_temp_scaling and self.config.temperature > 0:
            logits = logits / self.config.temperature
        return logits


class MaskingLoss(nn.Module):
    """Cross-entropy with label smoothing (operates on logits)."""

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


# -----------------------------------------------------------------------------
# 4) Trainer
# -----------------------------------------------------------------------------
class GradientMaskingTrainer:
    def __init__(self, model: nn.Module, config: GradientMaskingConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        self.criterion = MaskingLoss(label_smoothing=config.label_smoothing)

        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "grad_norm": [],
        }

    def _grad_norm_l2(self) -> float:
        total_sq = 0.0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            total_sq += float(torch.sum(g * g).cpu().item())
        return float(np.sqrt(total_sq))

    def _apply_grad_noise(self) -> None:
        """Inject Gaussian noise into parameter gradients (post-backward)."""
        if not (self.config.use_grad_noise and self.config.grad_noise_std > 0):
            return
        std = self.config.grad_noise_std
        for p in self.model.parameters():
            if p.grad is None:
                continue
            p.grad.add_(torch.randn_like(p.grad) * std)

    def _apply_grad_clip(self) -> None:
        if not (self.config.use_grad_clip and self.config.grad_clip > 0):
            return
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        grad_norms: List[float] = []

        for b, (images, targets) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(images)  # logits (possibly temp-scaled)
            loss = self.criterion(logits, targets)

            loss.backward()

            # Gradient masking ingredients
            self._apply_grad_noise()
            self._apply_grad_clip()

            grad_norm = self._grad_norm_l2()
            grad_norms.append(grad_norm)

            self.optimizer.step()

            running_loss += float(loss.item())

            preds = torch.argmax(logits, dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.size(0))

            if b % 10 == 0:
                print(
                    f"[TRAIN] epoch={epoch} batch={b}/{len(train_loader)} "
                    f"loss={loss.item():.4f} grad_norm={grad_norm:.4f}"
                )

        epoch_loss = running_loss / max(1, len(train_loader))
        epoch_acc = 100.0 * correct / max(1, total)
        avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        return epoch_loss, epoch_acc, avg_grad_norm

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = F.cross_entropy(logits, targets)

            running_loss += float(loss.item())

            preds = torch.argmax(logits, dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.size(0))

        epoch_loss = running_loss / max(1, len(val_loader))
        epoch_acc = 100.0 * correct / max(1, total)
        return epoch_loss, epoch_acc


# -----------------------------------------------------------------------------
# 5) Visualization and gradient analysis
# -----------------------------------------------------------------------------
def save_training_plots(history: dict, output_dir: Path) -> None:
    """Save loss/accuracy/grad-norm plots to disk."""
    epochs = history["epoch"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    plt.plot(epochs, history["val_acc"], label="Val Acc", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["grad_norm"], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm (L2)")
    plt.title("Gradient Norm During Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "grad_norm_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def perform_gradient_distribution_sample(model: nn.Module, dataloader: DataLoader, device: torch.device, output_dir: Path) -> None:
    """
    Collect a small sample of gradients (few batches) and save:
    - histogram plot
    - summary statistics JSON
    """
    print("[INFO] Gradient distribution sampling...")

    model.train()
    grads: List[np.ndarray] = []

    for i, (images, targets) in enumerate(dataloader):
        if i >= 5:
            break
        images = images.to(device)
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)

        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        for p in model.parameters():
            if p.grad is None:
                continue
            grads.append(p.grad.detach().cpu().flatten().numpy())

    if not grads:
        print("[WARN] No gradients collected.")
        return

    g = np.concatenate(grads, axis=0)

    plt.figure(figsize=(10, 6))
    plt.hist(g, bins=60, alpha=0.8, edgecolor="black")
    plt.xlabel("Gradient Values")
    plt.ylabel("Frequency")
    plt.title("Gradient Value Distribution (Sample)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "gradient_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    stats = {
        "mean": float(np.mean(g)),
        "std": float(np.std(g)),
        "min": float(np.min(g)),
        "max": float(np.max(g)),
        "p95": float(np.percentile(g, 95)),
        "p99": float(np.percentile(g, 99)),
        "count": int(g.size),
    }

    with open(output_dir / "gradient_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[GRAD STATS] mean={stats['mean']:.6f} std={stats['std']:.6f} p99={stats['p99']:.6f}")


# -----------------------------------------------------------------------------
# 6) Main
# -----------------------------------------------------------------------------
def main() -> None:
    config = GradientMaskingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    data_root = Path(os.getenv("DATA_ROOT", "dataset_rambu_lalu_lintas"))
    base_pattern = os.getenv("BASE_PATTERN", "outputs_vim_rambu_small_*")

    # Find baseline run and checkpoint
    base_dir = find_latest_save_dir(base_pattern)
    ckpt_path = base_dir / "best_vim_rambu_small.pth"
    print(f"[BASE DIR] {base_dir}")
    print(f"[CKPT] {ckpt_path}")

    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs_gradient_masking_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] {output_dir}")

    # Load class mapping to determine num_classes
    with open(base_dir / "class_mapping.json", "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    # Support common formats:
    # - {"idx_to_class": {...}}
    # - {"class_to_idx": {...}}
    if "idx_to_class" in class_mapping:
        num_classes = len(class_mapping["idx_to_class"])
    elif "class_to_idx" in class_mapping:
        num_classes = len(class_mapping["class_to_idx"])
    else:
        raise RuntimeError("class_mapping.json must contain idx_to_class or class_to_idx.")

    print(f"[INFO] num_classes={num_classes}")

    # Build model that matches the checkpoint
    vim_kwargs = build_vim_kwargs_from_checkpoint(num_classes, ckpt_path)
    base_model = Vim(**vim_kwargs).to(device)

    # Load weights (allow minor mismatches if wrapper changed)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)

    missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
    print(f"[LOAD] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing:
        print("[LOAD] sample missing:", missing[:5])
    if unexpected:
        print("[LOAD] sample unexpected:", unexpected[:5])

    # Wrap with masking behaviors (logits preserved)
    model = MaskingWrapper(base_model, config).to(device)

    # Dataset transforms
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dir = data_root / "train"
    valid_dir = data_root / "valid"
    test_dir = data_root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    eval_dir = valid_dir if valid_dir.exists() else test_dir
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    print(f"[DATA] train={train_dir} eval={eval_dir}")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    eval_ds = datasets.ImageFolder(eval_dir, transform=eval_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[DATA] train_samples={len(train_ds)} eval_samples={len(eval_ds)}")

    trainer = GradientMaskingTrainer(model, config, device)

    best_val_acc = 0.0
    best_epoch = 0

    print("[INFO] Starting gradient masking fine-tuning...")
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc, grad_norm = trainer.train_epoch(train_loader, epoch)
        val_loss, val_acc = trainer.validate(eval_loader)

        trainer.scheduler.step()

        trainer.history["epoch"].append(epoch)
        trainer.history["train_loss"].append(train_loss)
        trainer.history["train_acc"].append(train_acc)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["val_acc"].append(val_acc)
        trainer.history["grad_norm"].append(grad_norm)

        print(
            f"[EPOCH {epoch:03d}/{config.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
            f"grad_norm={grad_norm:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "config": config.__dict__,
                    "vim_kwargs": vim_kwargs,
                    "history": trainer.history,
                    "base_dir": str(base_dir),
                },
                output_dir / "best_gradient_masking_model.pth",
            )
            print(f"  -> [BEST] saved checkpoint (val_acc={best_val_acc:.2f}%)")

    # Save history and metadata
    pd.DataFrame(trainer.history).to_csv(output_dir / "training_history.csv", index=False)

    with open(output_dir / "gradient_masking_config.json", "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)

    with open(output_dir / "model_architecture.json", "w", encoding="utf-8") as f:
        json.dump(vim_kwargs, f, indent=2)

    with open(output_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)

    save_training_plots(trainer.history, output_dir)
    perform_gradient_distribution_sample(model, eval_loader, device, output_dir)

    summary = {
        "training_completed": True,
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "total_epochs": int(config.epochs),
        "final_train_accuracy": float(trainer.history["train_acc"][-1]) if trainer.history["train_acc"] else None,
        "final_val_accuracy": float(trainer.history["val_acc"][-1]) if trainer.history["val_acc"] else None,
        "model_parameters": int(sum(p.numel() for p in model.parameters())),
        "output_directory": str(output_dir),
        "timestamp": timestamp,
    }

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[SUCCESS] Gradient masking fine-tuning completed.")
    print(f"  best_val_acc : {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  outputs      : {output_dir}")


if __name__ == "__main__":
    main()
