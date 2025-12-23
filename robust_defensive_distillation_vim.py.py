# ============================================================
# file: defensive_distillation_vim.py
# Fine-tuning Vision Mamba dengan Defensive Distillation
# Referensi: Papernot et al. (2016)
# ============================================================
from __future__ import annotations
import os
import json
import time
import inspect
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from vision_mamba import Vim

# ------------------------------------------------------------
# 1. CONFIG & UTILS
# ------------------------------------------------------------
DATA_ROOT = Path("dataset_rambu_lalu_lintas")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "valid"
TEST_DIR = DATA_ROOT / "test"

# Hyperparameters Defensive Distillation
BATCH_SIZE = 64
MAX_EPOCHS_TEACHER = 50
MAX_EPOCHS_STUDENT = 50
TEMPERATURE_TEACHER = 20.0  # Temperature tinggi untuk teacher
TEMPERATURE_STUDENT = 20.0  # Sama untuk student training
TEMPERATURE_INFERENCE = 1.0  # Temperature normal untuk inference

BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4
TARGET_ACC = 0.70
EARLY_STOP_PATIENCE = 10

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def create_save_dir(prefix: str = "defensive_distillation_vim") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"{prefix}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def build_vim_kwargs(num_classes: int, img_size: int = 224) -> Dict:
    """Build Vim kwargs yang kompatibel dengan pretrained model"""
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
        "dropout": 0.10,  # Dropout lebih rendah untuk distillation
        "depth": 6,
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
# 2. TEMPERATURE SCALED SOFTMAX & LOSS
# ------------------------------------------------------------
class TemperatureScaledSoftmax(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.temperature, dim=1)

class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 1.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kldiv = nn.KLDivLoss(reduction='batchmean')
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        # Soft targets dari teacher dengan temperature
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss untuk distillation
        distillation_loss = self.kldiv(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Cross entropy loss dengan hard labels
        student_loss = self.cross_entropy(student_logits, targets)
        
        # Kombinasi kedua loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        return total_loss

# ------------------------------------------------------------
# 3. DATA LOADER
# ------------------------------------------------------------
def get_dataloaders() -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    # Augmentasi untuk defensive distillation (moderat)
    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
        raise FileNotFoundError(f"Folder train tidak ditemukan: {TRAIN_DIR}")

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    
    use_val_dir = VAL_DIR.exists() and any(VAL_DIR.rglob("*.*"))
    if use_val_dir:
        val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tfms)
        print("[INFO] Menggunakan 'valid' sebagai validation set.")
    else:
        val_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tfms)
        print("[INFO] Folder 'valid' kosong/tidak ada. Menggunakan 'test' sebagai validation set.")

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"[INFO] Jumlah kelas: {len(idx_to_class)}")

    num_workers = min(4, (os.cpu_count() // 2) if os.cpu_count() else 2)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, idx_to_class

# ------------------------------------------------------------
# 4. TRAINING LOOP UNTUK TEACHER MODEL
# ------------------------------------------------------------
def train_teacher_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    num_classes: int
) -> nn.Module:
    
    print("\n" + "="*50)
    print("TRAINING TEACHER MODEL (Temperature Scaling)")
    print("="*50)
    
    # Temperature-scaled softmax untuk teacher
    temp_softmax = TemperatureScaledSoftmax(temperature=TEMPERATURE_TEACHER)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS_TEACHER)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS_TEACHER + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            
            # Teacher menggunakan temperature-scaled softmax selama training
            probs = temp_softmax(logits)
            loss = criterion(probs, targets)
            
            loss.backward()
            optimizer.step()

            acc = accuracy_from_logits(logits, targets)
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += acc * batch_size
            total += batch_size

        train_loss = running_loss / total
        train_acc = running_correct / total

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[Teacher Epoch {epoch:03d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}%")

        # Save best teacher model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_acc": best_val_acc,
                "temperature": TEMPERATURE_TEACHER,
            }, save_dir / "best_teacher_model.pth")
            print(f"  -> [BEST TEACHER] Model disimpan (val_acc={best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"[EARLY STOP] Teacher training stopped at epoch {epoch}")
            break

    # Save teacher training history
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(save_dir / "teacher_training_history.csv", index=False)
    
    return model

# ------------------------------------------------------------
# 5. TRAINING LOOP UNTUK STUDENT MODEL (DISTILLATION)
# ------------------------------------------------------------
def train_student_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: Path
) -> nn.Module:
    
    print("\n" + "="*50)
    print("TRAINING STUDENT MODEL (Defensive Distillation)")
    print("="*50)
    
    # Freeze teacher model
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Distillation loss
    criterion = DistillationLoss(
        temperature=TEMPERATURE_STUDENT, 
        alpha=0.7  # Weight untuk distillation loss vs hard labels
    )
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS_STUDENT)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS_STUDENT + 1):
        # Training phase dengan distillation
        student_model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # Forward pass student
            student_logits = student_model(images)
            
            # Forward pass teacher (no gradients)
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            
            # Distillation loss
            loss = criterion(student_logits, teacher_logits, targets)
            
            loss.backward()
            optimizer.step()

            acc = accuracy_from_logits(student_logits, targets)
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += acc * batch_size
            total += batch_size

        train_loss = running_loss / total
        train_acc = running_correct / total

        # Validation phase (student saja)
        val_criterion = nn.CrossEntropyLoss()
        val_loss, val_acc = evaluate_model(student_model, val_loader, val_criterion, device)

        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[Student Epoch {epoch:03d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}%")

        # Save best student model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model": student_model.state_dict(),
                "val_acc": best_val_acc,
                "temperature_train": TEMPERATURE_STUDENT,
                "temperature_inference": TEMPERATURE_INFERENCE,
            }, save_dir / "best_student_model.pth")
            print(f"  -> [BEST STUDENT] Model disimpan (val_acc={best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"[EARLY STOP] Student training stopped at epoch {epoch}")
            break

    # Save student training history
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(save_dir / "student_training_history.csv", index=False)
    
    return student_model

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)
            acc = accuracy_from_logits(logits, targets)
            batch_size = targets.size(0)

            running_loss += loss.item() * batch_size
            running_correct += acc * batch_size
            total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc

# ------------------------------------------------------------
# 6. EVALUASI ROBUSTNESS
# ------------------------------------------------------------
def evaluate_robustness(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    attack_fn=None,
    attack_kwargs=None
) -> float:
    """Evaluasi robust accuracy terhadap serangan adversarial"""
    model.eval()
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if attack_fn:
            x_adv = attack_fn(model, images, targets, **attack_kwargs)
        else:
            x_adv = images

        with torch.no_grad():
            logits = model(x_adv)
            preds = logits.argmax(dim=1)

        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return correct / total if total > 0 else 0.0

# ------------------------------------------------------------
# 7. MAIN EXECUTION
# ------------------------------------------------------------
def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")
    
    # Load data
    train_loader, val_loader, idx_to_class = get_dataloaders()
    num_classes = len(idx_to_class)
    
    # Create save directory
    save_dir = create_save_dir()
    print(f"[INFO] Save dir: {save_dir}")
    
    # Save configuration
    config = {
        "num_classes": num_classes,
        "temperature_teacher": TEMPERATURE_TEACHER,
        "temperature_student": TEMPERATURE_STUDENT,
        "temperature_inference": TEMPERATURE_INFERENCE,
        "batch_size": BATCH_SIZE,
        "base_lr": BASE_LR,
        "weight_decay": WEIGHT_DECAY,
        "defense_method": "defensive_distillation",
        "reference": "Papernot et al. (2016) - Distillation as a Defense to Adversarial Perturbations"
    }
    
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    with open(save_dir / "class_mapping.json", "w") as f:
        json.dump({"idx_to_class": {str(k): v for k, v in idx_to_class.items()}}, f, indent=2)

    # Step 1: Train Teacher Model dengan Temperature Scaling
    print("\n🚀 Step 1: Training Teacher Model with Temperature Scaling")
    teacher_kwargs = build_vim_kwargs(num_classes=num_classes)
    teacher_model = Vim(**teacher_kwargs).to(device)
    
    teacher_model = train_teacher_model(
        teacher_model, train_loader, val_loader, device, save_dir, num_classes
    )

    # Step 2: Train Student Model dengan Defensive Distillation
    print("\n🚀 Step 2: Training Student Model with Defensive Distillation")
    student_kwargs = build_vim_kwargs(num_classes=num_classes)
    student_model = Vim(**student_kwargs).to(device)
    
    student_model = train_student_model(
        teacher_model, student_model, train_loader, val_loader, device, save_dir
    )

    # Step 3: Evaluasi Final Model
    print("\n📊 Step 3: Final Evaluation")
    
    # Load best student model
    student_ckpt = torch.load(save_dir / "best_student_model.pth", map_location=device)
    student_model.load_state_dict(student_ckpt["model"])
    
    # Clean accuracy
    clean_acc = evaluate_robustness(student_model, val_loader, device)
    print(f"✅ Clean Accuracy: {clean_acc*100:.2f}%")
    
    print(f"\n🎯 Defensive Distillation Training Completed!")
    print(f"📍 Results saved to: {save_dir}")
    print(f"📁 Model files:")
    print(f"   - Teacher model: {save_dir / 'best_teacher_model.pth'}")
    print(f"   - Student model: {save_dir / 'best_student_model.pth'}")
    print(f"   - Training history: {save_dir / 'student_training_history.csv'}")

if __name__ == "__main__":
    main()