# ============================================================
# file: robust_adversarial_training_fixed.py
# Advanced Adversarial Training - FIXED VERSION
# ============================================================
from __future__ import annotations
import os
import json
import time
import inspect
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from vision_mamba import Vim

# ------------------------------------------------------------
# 1. CONFIG & UTILS - DIPERBAIKI
# ------------------------------------------------------------
def setup_paths() -> Tuple[Path, Path, Path, Path]:
    """Setup paths dengan error handling yang lebih baik"""
    base_model_dir = Path("outputs_vim_rambu_small_20251119_220259")
    
    if not base_model_dir.exists():
        # Fallback: cari folder terbaru dengan pattern
        candidates = list(Path(".").glob("outputs_vim_rambu_small_*"))
        if candidates:
            base_model_dir = max(candidates, key=os.path.getmtime)
            print(f"[INFO] Menggunakan model terbaru: {base_model_dir}")
        else:
            raise FileNotFoundError("Tidak ditemukan folder model base")
    
    ckpt_path = base_model_dir / "best_vim_rambu_small.pth"
    config_path = base_model_dir / "config.json"
    classmap_path = base_model_dir / "class_mapping.json"
    
    for path in [ckpt_path, config_path, classmap_path]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
    
    return base_model_dir, ckpt_path, config_path, classmap_path

# Setup paths
BASE_MODEL_DIR, BASE_CKPT_PATH, BASE_CONFIG_PATH, BASE_CLASSMAP_PATH = setup_paths()

DATA_ROOT = Path("dataset_rambu_lalu_lintas")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "valid"
TEST_DIR = DATA_ROOT / "test"

# Hyperparameters untuk training stabil
BATCH_SIZE = 32
MAX_EPOCHS = 60
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-4

# Multiple Attack Parameters untuk Variasi Training
ATTACK_EPSILONS = [4/255, 8/255]  # Multiple epsilon untuk variasi
PGD_ALPHA = 2/255
PGD_STEPS = 10
PGD_RANDOM_START = True

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def create_save_dir(prefix: str = "robust_adv_train") -> Path:
    """Membuat save directory dengan verifikasi"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"{prefix}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Directory created: {out_dir}")
    return out_dir

def build_vim_kwargs_compatible(num_classes: int, img_size: int = 224) -> Dict:
    """
    Build Vim kwargs yang KOMPATIBEL dengan model base
    Menggunakan arsitektur yang sama dengan model base (dim=192, depth=6)
    """
    sig = inspect.signature(Vim).parameters
    kwargs = {
        "dim": 192,  # SAMA dengan base model
        "dt_rank": 24,  # SAMA dengan base model
        "dim_inner": 192,  # SAMA dengan base model
        "d_state": 64,  # SAMA dengan base model
        "num_classes": num_classes,
        "image_size": img_size,
        "patch_size": 32,
        "channels": 3,
        "dropout": 0.20,  # SAMA dengan base model
        "depth": 6,  # SAMA dengan base model
    }
    if "heads" in sig:
        kwargs["heads"] = 4
    return {k: v for k, v in kwargs.items() if k in sig}

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0

# ------------------------------------------------------------
# 2. ADVERSARIAL ATTACK UTILITIES
# ------------------------------------------------------------
def channel_bounds_normalized(mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    x_min = (0.0 - mean_t) / std_t
    x_max = (1.0 - mean_t) / std_t
    return x_min, x_max

def clamp_normed(x, x_min, x_max):
    return torch.max(torch.min(x, x_max), x_min)

def _scaled_eps_alpha(eps: float, alpha: float, std, device):
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return eps / std_t, alpha / std_t

# 2.1 FGSM Attack
def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8/255,
    mean=MEAN,
    std=STD,
) -> torch.Tensor:
    """FGSM Attack"""
    device = x.device
    model.eval()
    
    x_min, x_max = channel_bounds_normalized(mean, std, device)
    eps_scaled, _ = _scaled_eps_alpha(eps, eps, std, device)

    x_adv = x.detach().clone().requires_grad_(True)
    
    with torch.enable_grad():
        logits = model(x_adv)
        loss = nn.functional.cross_entropy(logits, y)
    
    grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
    
    with torch.no_grad():
        x_adv = clamp_normed(x_adv + eps_scaled * grad.sign(), x_min, x_max)

    return x_adv.detach()

# 2.2 PGD Attack (Standard)
def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8/255,
    alpha: float = 2/255,
    steps: int = 10,
    random_start: bool = True,
    mean=MEAN,
    std=STD,
) -> torch.Tensor:
    """Standard PGD Attack"""
    device = x.device
    model.eval()
    
    x_min, x_max = channel_bounds_normalized(mean, std, device)
    eps_scaled, alpha_scaled = _scaled_eps_alpha(eps, alpha, std, device)

    if random_start:
        delta = torch.empty_like(x).uniform_(-1.0, 1.0) * eps_scaled
        x_adv = clamp_normed(x + delta, x_min, x_max)
    else:
        x_adv = x.detach().clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        
        with torch.enable_grad():
            logits = model(x_adv)
            loss = nn.functional.cross_entropy(logits, y)
        
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]

        with torch.no_grad():
            x_adv = x_adv + alpha_scaled * grad.sign()
            delta = torch.clamp(x_adv - x, min=-eps_scaled, max=eps_scaled)
            x_adv = clamp_normed(x + delta, x_min, x_max)

    return x_adv.detach()

# 2.3 Adaptive PGD Attack (Lebih Kuat)
def adaptive_pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8/255,
    steps: int = 20,
    random_start: bool = True,
    mean=MEAN,
    std=STD,
) -> torch.Tensor:
    """Adaptive PGD dengan lebih banyak steps dan adaptive alpha"""
    device = x.device
    model.eval()
    
    # Adaptive alpha based on epsilon
    alpha = max(eps / 4, 1/255)  # Minimum alpha
    
    return pgd_attack(model, x, y, eps, alpha, steps, random_start, mean, std)

# 2.4 Multi-Step Attack Scheduler
class AttackScheduler:
    """Scheduler untuk variasi attack selama training"""
    
    def __init__(self, attack_fns: List[Callable], eps_list: List[float]):
        self.attack_fns = attack_fns
        self.eps_list = eps_list
        
    def get_attack(self, epoch: int, batch_idx: int):
        """Get attack configuration berdasarkan epoch dan batch"""
        # Rotasi antara different attack types
        attack_idx = (epoch + batch_idx // 10) % len(self.attack_fns)
        attack_fn = self.attack_fns[attack_idx]
        
        # Variasi epsilon
        eps_idx = (epoch + batch_idx // 5) % len(self.eps_list)
        eps = self.eps_list[eps_idx]
        
        return attack_fn, eps

# ------------------------------------------------------------
# 3. DATASET & DATALOADER
# ------------------------------------------------------------
def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """Enhanced data loading dengan lebih banyak augmentasi"""
    
    # Strong Augmentation untuk Adversarial Training
    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Verify directories exist
    for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory tidak ditemukan: {dir_path}")

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tfms)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tfms)

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"[INFO] Jumlah kelas: {len(idx_to_class)}")
    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Val samples: {len(val_ds)}")
    print(f"[INFO] Test samples: {len(test_ds)}")

    num_workers = min(4, os.cpu_count() // 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, idx_to_class

# ------------------------------------------------------------
# 4. ADVERSARIAL TRAINING LOOP
# ------------------------------------------------------------
def adversarial_train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    attack_scheduler: AttackScheduler,
    epoch: int,
) -> Tuple[float, float, float]:
    """
    Satu epoch adversarial training
    Returns: (train_loss, train_adv_acc, train_clean_acc)
    """
    model.train()
    running_loss = 0.0
    running_adv_correct = 0
    running_clean_correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Get attack configuration dari scheduler
        attack_fn, eps = attack_scheduler.get_attack(epoch, batch_idx)
        
        # Generate adversarial examples
        with torch.no_grad():
            adv_images = attack_fn(model, images, targets, eps=eps)

        # Train dengan adversarial examples
        optimizer.zero_grad()
        
        # Forward pass dengan adversarial examples
        adv_logits = model(adv_images)
        loss = criterion(adv_logits, targets)
        
        loss.backward()
        optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            clean_logits = model(images)
            clean_acc = accuracy_from_logits(clean_logits, targets)
            adv_acc = accuracy_from_logits(adv_logits, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_adv_correct += adv_acc * batch_size
        running_clean_correct += clean_acc * batch_size
        total += batch_size

        # Print progress
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}: Loss: {loss.item():.4f}, "
                  f"Clean Acc: {clean_acc*100:.2f}%, "
                  f"Adv Acc: {adv_acc*100:.2f}%")

    epoch_loss = running_loss / total
    epoch_adv_acc = running_adv_correct / total
    epoch_clean_acc = running_clean_correct / total
    
    return epoch_loss, epoch_adv_acc, epoch_clean_acc

@torch.no_grad()
def evaluate_robustness(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model robustness terhadap berbagai serangan
    """
    model.eval()
    results = {}
    
    # Clean evaluation
    clean_correct = 0
    clean_total = 0
    clean_loss = 0.0
    
    # Attack evaluations
    attacks = {
        'fgsm_4': lambda m, x, y: fgsm_attack(m, x, y, eps=4/255),
        'fgsm_8': lambda m, x, y: fgsm_attack(m, x, y, eps=8/255),
        'pgd_4': lambda m, x, y: pgd_attack(m, x, y, eps=4/255),
        'pgd_8': lambda m, x, y: pgd_attack(m, x, y, eps=8/255),
    }
    
    attack_results = {name: {'correct': 0, 'total': 0} for name in attacks.keys()}
    
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)
        
        # Clean evaluation
        clean_logits = model(images)
        clean_loss += criterion(clean_logits, targets).item() * batch_size
        clean_pred = clean_logits.argmax(1)
        clean_correct += (clean_pred == targets).sum().item()
        clean_total += batch_size
        
        # Attack evaluations
        for attack_name, attack_fn in attacks.items():
            adv_images = attack_fn(model, images, targets)
            adv_logits = model(adv_images)
            adv_pred = adv_logits.argmax(1)
            attack_results[attack_name]['correct'] += (adv_pred == targets).sum().item()
            attack_results[attack_name]['total'] += batch_size
    
    # Compile results
    results['clean_accuracy'] = clean_correct / clean_total if clean_total > 0 else 0.0
    results['clean_loss'] = clean_loss / clean_total if clean_total > 0 else 0.0
    
    for attack_name in attacks.keys():
        acc = attack_results[attack_name]['correct'] / attack_results[attack_name]['total'] if attack_results[attack_name]['total'] > 0 else 0.0
        results[f'{attack_name}_accuracy'] = acc
    
    # Calculate average robustness
    attack_accuracies = [results[f'{name}_accuracy'] for name in attacks.keys()]
    results['avg_robust_accuracy'] = np.mean(attack_accuracies) if attack_accuracies else 0.0
    
    return results

# ------------------------------------------------------------
# 5. MAIN ADVERSARIAL TRAINING FUNCTION - DIPERBAIKI
# ------------------------------------------------------------
def main():
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"[DEVICE] Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("[DEVICE] Using CPU")
    
    # Create save directory
    save_dir = create_save_dir("robust_adv_train")
    print(f"[INFO] Save directory: {save_dir}")

    try:
        # Load dataloaders
        train_loader, val_loader, test_loader, idx_to_class = get_dataloaders()
        num_classes = len(idx_to_class)
        print(f"[INFO] Successfully loaded dataloaders - {num_classes} classes")

        # Load base model configuration
        with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
            base_config = json.load(f)
        with open(BASE_CLASSMAP_PATH, "r", encoding="utf-8") as f:
            base_mapping = json.load(f)

        # Build model yang KOMPATIBEL dengan base model
        vim_kwargs = build_vim_kwargs_compatible(num_classes=num_classes, img_size=224)
        print("[INFO] Model kwargs (COMPATIBLE):", vim_kwargs)
        model = Vim(**vim_kwargs).to(device)

        # Load base model weights - DENGAN COMPATIBILITY CHECK
        print("[INFO] Loading base model weights...")
        base_ckpt = torch.load(BASE_CKPT_PATH, map_location=device)
        
        # Cek apakah model memiliki key 'model' atau langsung state_dict
        if "model" in base_ckpt:
            print("[INFO] Loading from checkpoint with 'model' key")
            model.load_state_dict(base_ckpt["model"], strict=True)
        else:
            print("[INFO] Loading checkpoint directly as state_dict")
            model.load_state_dict(base_ckpt, strict=True)
        
        print("[INFO] ✅ Successfully loaded base model weights!")

        # Setup attack scheduler
        attack_fns = [fgsm_attack, pgd_attack]
        attack_scheduler = AttackScheduler(attack_fns, ATTACK_EPSILONS)
        
        print("[INFO] Attack scheduler configured with:")
        print(f"  - Attack functions: {[fn.__name__ for fn in attack_fns]}")
        print(f"  - Epsilons: {ATTACK_EPSILONS}")

        # Save configuration
        config = {
            "base_model": str(BASE_CKPT_PATH),
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "base_lr": BASE_LR,
            "weight_decay": WEIGHT_DECAY,
            "attack_epsilons": ATTACK_EPSILONS,
            "pgd_alpha": PGD_ALPHA,
            "pgd_steps": PGD_STEPS,
            "num_classes": num_classes,
            "model_config": vim_kwargs,
            "compatible_with_base": True,
        }
        
        config_path = save_dir / "training_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"[INFO] Configuration saved: {config_path}")

        # Save class mapping
        classmap_path = save_dir / "class_mapping.json"
        with open(classmap_path, "w", encoding="utf-8") as f:
            json.dump(base_mapping, f, indent=2)

        # Optimizer dan scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=BASE_LR,
            weight_decay=WEIGHT_DECAY,
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS
        )
        
        criterion = nn.CrossEntropyLoss()

        # Training history
        history = {
            "epoch": [],
            "train_loss": [],
            "train_adv_acc": [],
            "train_clean_acc": [],
            "val_clean_acc": [],
            "val_robust_acc": [],
            "val_avg_robust_acc": [],
            "learning_rate": [],
        }

        # Attack-specific history
        attack_history = {
            'fgsm_4_acc': [], 'fgsm_8_acc': [],
            'pgd_4_acc': [], 'pgd_8_acc': [],
        }

        best_avg_robust_acc = 0.0
        best_ckpt_path = save_dir / "best_robust_model.pth"

        print("\n" + "="*60)
        print("[INFO] Memulai Adversarial Training...")
        print("="*60)

        # Initial evaluation
        print("[INFO] Initial evaluation on validation set...")
        initial_results = evaluate_robustness(model, val_loader, criterion, device)
        print(f"Initial Clean Acc: {initial_results['clean_accuracy']*100:.2f}%")
        print(f"Initial Avg Robust Acc: {initial_results['avg_robust_accuracy']*100:.2f}%")

        for epoch in range(1, MAX_EPOCHS + 1):
            print(f"\n[Epoch {epoch}/{MAX_EPOCHS}]")
            start_time = time.time()
            
            # Adversarial training
            train_loss, train_adv_acc, train_clean_acc = adversarial_train_one_epoch(
                model, train_loader, criterion, optimizer, device, 
                attack_scheduler, epoch
            )
            
            # Evaluation
            val_results = evaluate_robustness(model, val_loader, criterion, device)
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - start_time

            # Update history
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_adv_acc"].append(train_adv_acc)
            history["train_clean_acc"].append(train_clean_acc)
            history["val_clean_acc"].append(val_results['clean_accuracy'])
            history["val_avg_robust_acc"].append(val_results['avg_robust_accuracy'])
            history["learning_rate"].append(current_lr)
            
            # Update attack-specific history
            for attack_name in attack_history.keys():
                if f'{attack_name}_accuracy' in val_results:
                    attack_history[attack_name].append(val_results[f'{attack_name}_accuracy'])

            # Print results
            print(f"Training:")
            print(f"  Loss: {train_loss:.4f}, Clean Acc: {train_clean_acc*100:.2f}%, Adv Acc: {train_adv_acc*100:.2f}%")
            print(f"Validation:")
            print(f"  Clean Acc: {val_results['clean_accuracy']*100:.2f}%")
            print(f"  Avg Robust Acc: {val_results['avg_robust_accuracy']*100:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"  Epoch Time: {epoch_time:.1f}s")

            # Save best model berdasarkan average robust accuracy
            if val_results['avg_robust_accuracy'] > best_avg_robust_acc:
                best_avg_robust_acc = val_results['avg_robust_accuracy']
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_clean_acc": val_results['clean_accuracy'],
                        "val_avg_robust_acc": val_results['avg_robust_accuracy'],
                        "vim_kwargs": vim_kwargs,
                    },
                    best_ckpt_path,
                )
                print(f"  → [BEST] Model saved (Avg Robust Acc: {best_avg_robust_acc*100:.2f}%)")

            # Save checkpoint setiap 10 epoch
            if epoch % 10 == 0:
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "history": history,
                    },
                    checkpoint_path,
                )
                print(f"  → Checkpoint saved: {checkpoint_path}")

        # Final evaluation on test set
        print("\n[INFO] Final evaluation on test set...")
        test_results = evaluate_robustness(model, test_loader, criterion, device)
        
        print(f"Final Test Results:")
        print(f"  Clean Accuracy: {test_results['clean_accuracy']*100:.2f}%")
        print(f"  Average Robust Accuracy: {test_results['avg_robust_accuracy']*100:.2f}%")
        for attack_name in ['fgsm_4', 'fgsm_8', 'pgd_4', 'pgd_8']:
            acc = test_results.get(f'{attack_name}_accuracy', 0)
            print(f"  {attack_name}: {acc*100:.2f}%")

        # Save final model
        final_ckpt_path = save_dir / "final_robust_model.pth"
        torch.save(
            {
                "epoch": MAX_EPOCHS,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "test_results": test_results,
                "vim_kwargs": vim_kwargs,
            },
            final_ckpt_path,
        )

        # --------------------------------------------------------
        # 6. SAVE RESULTS & VISUALIZATION
        # --------------------------------------------------------
        print("\n[INFO] Saving results and visualizations...")
        
        # Save history to CSV
        df_history = pd.DataFrame(history)
        history_csv_path = save_dir / "training_history.csv"
        df_history.to_csv(history_csv_path, index=False)
        print(f"  - Training history: {history_csv_path}")

        # Save final report
        report = {
            "training_summary": {
                "final_epoch": MAX_EPOCHS,
                "best_avg_robust_acc": best_avg_robust_acc,
                "final_test_clean_acc": test_results['clean_accuracy'],
                "final_test_avg_robust_acc": test_results['avg_robust_accuracy'],
                "training_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "base_model": str(BASE_CKPT_PATH),
            },
            "test_results": test_results,
        }
        
        report_path = save_dir / "training_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  - Training report: {report_path}")

        # Create visualizations
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Accuracy curves
        axes[0, 0].plot(history["epoch"], np.array(history["train_clean_acc"]) * 100, 
                       label='Train Clean Acc', linewidth=2)
        axes[0, 0].plot(history["epoch"], np.array(history["train_adv_acc"]) * 100, 
                       label='Train Adv Acc', linewidth=2)
        axes[0, 0].plot(history["epoch"], np.array(history["val_clean_acc"]) * 100, 
                       label='Val Clean Acc', linewidth=2)
        axes[0, 0].plot(history["epoch"], np.array(history["val_avg_robust_acc"]) * 100, 
                       label='Val Avg Robust Acc', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Training and Validation Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Loss curves
        axes[0, 1].plot(history["epoch"], history["train_loss"], 
                       label='Train Loss', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Learning rate
        axes[1, 0].plot(history["epoch"], history["learning_rate"], 
                       linewidth=2, color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Robustness comparison
        categories = ['Clean', 'FGSM (ε=4/255)', 'FGSM (ε=8/255)', 'PGD (ε=4/255)', 'PGD (ε=8/255)']
        base_values = [45.5, 0, 0, 0, 0]  # Base model
        
        adv_values = [
            test_results['clean_accuracy'] * 100,
            test_results.get('fgsm_4_accuracy', 0) * 100,
            test_results.get('fgsm_8_accuracy', 0) * 100,
            test_results.get('pgd_4_accuracy', 0) * 100,
            test_results.get('pgd_8_accuracy', 0) * 100,
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, base_values, width, label='Base Model', alpha=0.7)
        axes[1, 1].bar(x + width/2, adv_values, width, label='Robust Model', alpha=0.7)
        axes[1, 1].set_xlabel('Attack Type')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Robustness Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(categories, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        curves_path = save_dir / "training_results.png"
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Training curves: {curves_path}")

        print("\n" + "="*70)
        print("[SUCCESS] Adversarial Training Completed!")
        print("="*70)
        print(f"Results saved in: {save_dir}")
        print(f"Best Avg Robust Accuracy: {best_avg_robust_acc*100:.2f}%")
        print(f"Final Test Clean Accuracy: {test_results['clean_accuracy']*100:.2f}%")
        print(f"Final Test Robust Accuracy: {test_results['avg_robust_accuracy']*100:.2f}%")
        
        improvement = test_results['avg_robust_accuracy'] * 100
        print(f"Robustness Improvement: +{improvement:.2f}%")
        
        print("\nGenerated Files:")
        for file_name in [
            "best_robust_model.pth", "final_robust_model.pth",
            "training_history.csv", "training_report.json", 
            "training_results.png"
        ]:
            file_path = save_dir / file_name
            if file_path.exists():
                print(f"  ✓ {file_name}")
        
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        
        # Save error log
        error_log_path = save_dir / "error_log.txt"
        with open(error_log_path, "w") as f:
            f.write(f"Error during training: {str(e)}\n")
            f.write("Traceback:\n")
            traceback.print_exc(file=f)
        print(f"Error log saved: {error_log_path}")

if __name__ == "__main__":
    main()