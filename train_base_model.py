# ============================================================
# file: train_vim_v3_balance.py
#  - NO ATTACK
#  - Training + visualization (loss & accuracy line charts)
#  - Increased model capacity, moderate regularization
# ============================================================
"""
Vision Mamba Training Script (Balanced v3) — English-documented version

This file preserves the original training logic and structure.
Only documentation (comments/docstrings/user-facing messages) was translated/added
to improve readability for an international GitHub audience.

What this script does
- Trains a Vision Mamba (Vim) model for Indonesian Traffic Sign Recognition (TSR)
  using ImageFolder datasets under:
    dataset_rambu_lalu_lintas/{train, valid, test}
- Uses moderately strong augmentations + optional Mixup.
- Tracks training/validation loss and accuracy per epoch.
- Saves:
    - config.json
    - class_mapping.json
    - best_vim_rambu_small.pth
    - train_history.json / train_history.csv
    - loss_curve.png / accuracy_curve.png

High-level pseudocode
1) Build dataloaders (train + valid; fallback to test if valid is missing/empty)
2) Build Vim model kwargs and instantiate model
3) Train for up to MAX_EPOCHS with:
     - AdamW optimizer
     - CosineAnnealingLR scheduler
     - Early stopping based on validation accuracy
4) Save best checkpoint when validation accuracy improves
5) Export history to JSON/CSV + generate plots
"""

from __future__ import annotations
import os
import json
import time
import inspect
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import pandas as pd

from vision_mamba import Vim   # make sure it is installed

# ------------------------------------------------------------
# 1. CONFIG & UTILS
# ------------------------------------------------------------
DATA_ROOT = Path("dataset_rambu_lalu_lintas")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "valid"  # if missing/empty, fallback to test
TEST_DIR  = DATA_ROOT / "test"

BATCH_SIZE = 64
MAX_EPOCHS = 150             # longer training, but with early stopping
TARGET_ACC = 0.70            # minimum target validation accuracy
EARLY_STOP_PATIENCE = 20     # stop if no improvement for N epochs

BASE_LR = 5e-4               # slightly higher learning rate
WEIGHT_DECAY = 3e-4          # moderate weight decay (lower than 5e-4)
LABEL_SMOOTHING = 0.0        # 0 because Mixup is used

# Mixup
USE_MIXUP = True
MIXUP_ALPHA = 0.2            # reduced from 0.4

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


def create_save_dir(prefix: str = "outputs_vim_rambu_small") -> Path:
    """Create an output directory with a timestamp suffix."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"{prefix}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_vim_kwargs(num_classes: int, img_size: int = 224) -> Dict:
    """
    Slightly larger model than v1:
      - dim=192, depth=6
      - dropout=0.20 (moderate)
    """
    sig = inspect.signature(Vim).parameters
    kwargs = {
        "dim": 192,
        "dt_rank": 24,
        "dim_inner": 192,
        "d_state": 64,
        "num_classes": num_classes,
        "image_size": img_size,
        "patch_size": 32,
        "channels": 3,
        "dropout": 0.20,
        "depth": 6,
    }
    if "heads" in sig:
        kwargs["heads"] = 4
    return {k: v for k, v in kwargs.items() if k in sig}


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy from logits and integer targets."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


# ------------------------------------------------------------
# 2. DATASET & DATALOADER
# ------------------------------------------------------------
def get_dataloaders() -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    # Augmentations are still fairly strong but not as strong as v2 (RandomErasing removed)
    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25,
                               saturation=0.25, hue=0.08),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train folder not found: {TRAIN_DIR}")

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)

    use_val_dir = VAL_DIR.exists() and any(VAL_DIR.rglob("*.*"))
    if use_val_dir:
        val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tfms)
        print("[INFO] Using 'valid' as the validation set.")
    else:
        val_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tfms)
        print("[INFO] 'valid' folder is missing/empty. Using 'test' as the validation set.")

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"[INFO] Number of classes: {len(idx_to_class)}")
    print("[INFO] idx_to_class mapping:")
    for idx, name in idx_to_class.items():
        print(f"  {idx}: {name}")

    num_workers = min(4, (os.cpu_count() // 2) if os.cpu_count() else 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    return train_loader, val_loader, idx_to_class


# ------------------------------------------------------------
# 3. MIXUP HELPER
# ------------------------------------------------------------
def mixup_data(x, y, alpha: float):
    """Return a Mixup batch: x_mix, y_a, y_b, lam."""
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ------------------------------------------------------------
# 4. TRAIN & EVAL LOOP (NO ATTACK)
# ------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_mixup: bool = False,
    mixup_alpha: float = 0.0,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_mixup:
            mixed_x, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
            logits = model(mixed_x)

            loss_a = criterion(logits, targets_a)
            loss_b = criterion(logits, targets_b)
            loss = (lam * loss_a + (1 - lam) * loss_b).mean()

            # Approx accuracy: use the original targets (targets)
            acc = accuracy_from_logits(logits, targets)
        else:
            logits = model(images)
            loss_vec = criterion(logits, targets)   # [B]
            loss = loss_vec.mean()
            acc = accuracy_from_logits(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += acc * batch_size
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss_vec = criterion(logits, targets)
        loss = loss_vec.mean()
        acc = accuracy_from_logits(logits, targets)
        batch_size = targets.size(0)

        running_loss += loss.item() * batch_size
        running_correct += acc * batch_size
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


# ------------------------------------------------------------
# 5. MAIN TRAINING + VISUALIZATION
# ------------------------------------------------------------
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    print(f"[DEVICE] Using: {device}")

    train_loader, val_loader, idx_to_class = get_dataloaders()
    num_classes = len(idx_to_class)

    save_dir = create_save_dir("outputs_vim_rambu_small")
    print(f"[INFO] Save dir: {save_dir}")

    config = {
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "target_acc": TARGET_ACC,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "base_lr": BASE_LR,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "use_mixup": USE_MIXUP,
        "mixup_alpha": MIXUP_ALPHA,
        "num_classes": num_classes,
        "train_dir": str(TRAIN_DIR),
        "val_dir": str(VAL_DIR),
        "test_dir": str(TEST_DIR),
    }
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    mapping = {"idx_to_class": {str(k): v for k, v in idx_to_class.items()}}
    with open(save_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    vim_kwargs = build_vim_kwargs(num_classes=num_classes, img_size=224)
    print("[INFO] Vim kwargs:", vim_kwargs)
    model = Vim(**vim_kwargs).to(device)

    criterion = nn.CrossEntropyLoss(
        reduction="none",
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    patience_counter = 0
    best_ckpt_path = save_dir / "best_vim_rambu_small.pth"

    print("[INFO] Starting training (NO ATTACK, larger model, moderate regularization) ...")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_mixup=USE_MIXUP,
            mixup_alpha=MIXUP_ALPHA,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "vim_kwargs": vim_kwargs,
                },
                best_ckpt_path,
            )
            print(f"  -> [BEST] Saved model (val_acc={best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1

        if best_val_acc >= TARGET_ACC:
            print(
                f"[STOP] Target accuracy >= {TARGET_ACC*100:.0f}% reached "
                f"(best_val_acc={best_val_acc*100:.2f}%)."
            )
            break

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(
                f"[EARLY STOP] Validation accuracy did not improve for "
                f"{EARLY_STOP_PATIENCE} epochs."
            )
            break

    history_path_json = save_dir / "train_history.json"
    with open(history_path_json, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    df_hist = pd.DataFrame(history)
    history_path_csv = save_dir / "train_history.csv"
    df_hist.to_csv(history_path_csv, index=False)

    # Loss curve
    plt.figure()
    plt.plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss")
    plt.plot(df_hist["epoch"], df_hist["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    loss_curve_path = save_dir / "loss_curve.png"
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(df_hist["epoch"], df_hist["train_acc"] * 100, label="Train Acc")
    plt.plot(df_hist["epoch"], df_hist["val_acc"] * 100,   label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    acc_curve_path = save_dir / "accuracy_curve.png"
    plt.savefig(acc_curve_path, dpi=200)
    plt.close()

    print("\n[INFO] Training finished.")
    print(f"  Best Val Acc : {best_val_acc*100:.2f}%")
    print(f"  History JSON : {history_path_json}")
    print(f"  History CSV  : {history_path_csv}")
    print(f"  Loss curve   : {loss_curve_path}")
    print(f"  Acc curve    : {acc_curve_path}")
    print(f"  Best ckpt    : {best_ckpt_path}")


if __name__ == "__main__":
    main()
