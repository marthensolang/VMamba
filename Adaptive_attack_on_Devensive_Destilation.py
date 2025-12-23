#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""TRUE Adaptive PGD Attack Evaluation (Defensive Distillation - Teacher Model)
==========================================================================

This script evaluates adversarial robustness of a *teacher* model trained with
Defensive Distillation, using a **defense-aware (true adaptive) PGD** attack
across multiple epsilon budgets.

What you get (artifacts)
------------------------
- adaptive_attack_TRUE_results.csv
- robustness_curve_TRUE_defdistill_teacher.png
- attack_progression_eps_*.png
- adv_examples_grid_eps_*.png
- perturbation_heatmap_eps_*.png
- attack_configuration_TRUE_defdistill_teacher.pth

Key ideas (high-level)
----------------------
- Inputs are **ImageNet-normalized** (MEAN/STD). Epsilon is defined in pixel
  space (e.g., 2/255), then converted to normalized space (eps_norm = eps / STD).
- "True adaptive" means the attack adapts its gradient to the assumed defense:
  - defdistill  -> CW-style margin loss (better probe for distillation behavior)
  - rs          -> EOT (Expectation over Transformation) with Gaussian noise
  - gradmask    -> optional BPDA straight-through (if a non-diff preprocess exists)
  - plain       -> standard cross-entropy

Pseudocode (pipeline)
---------------------
1) Resolve model paths (checkpoint/config/class mapping), create output folder.
2) Load teacher checkpoint safely (Vision Mamba if available; otherwise CNN fallback).
3) Build test dataloader with Resize->CenterCrop->Normalize.
4) Measure clean accuracy on the same subset used for attacks.
5) For each epsilon in {0.5..8}/255:
     - Run adaptive PGD (restarts + steps) and compute robust accuracy.
     - Save per-epsilon diagnostic plots (progression, grid, heatmap).
6) Save CSV + robustness curve plot.

Reproducibility notes
---------------------
- CUDA_LAUNCH_BLOCKING=1 is enabled for easier debugging (can slow execution).
- cuDNN deterministic mode is enabled.

Safety / research note
----------------------
This code is intended for robustness evaluation and research reproducibility.

"""

from __future__ import annotations
import os, json, time, sys, warnings, inspect
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

# Debug CUDA (opsional; bisa dipercepat dengan mematikan)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

warnings.filterwarnings("ignore")
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("=== ADAPTIVE PGD ATTACK EVALUATION (TRUE ADAPTIVE, ULTRA-LIGHT READY) ===")

# ------------------------------------------------------------
# 0. CONFIGURATION & PATHS (DEFENSIVE DISTILLATION - TEACHER MODEL)
# ------------------------------------------------------------
# English note:
# - This script targets the *teacher* checkpoint (best_teacher_model.pth).
# - Folder spelling is intentionally handled with multiple fallbacks
#   (e.g., 'Devensive Destilation' vs 'Defensive Distillation').
# ------------------------------------------------------------
# 0. KONFIGURASI & PATH (DEFENSIVE DISTILLATION - TEACHER MODEL)
# ------------------------------------------------------------
print(f"[DEBUG] CWD = {Path.cwd()}")

# sesuai path yang Anda berikan (ejaannya: "Devensive Destilation")
BASE_DIR = Path("FINAL RESULTS") / "Devensive Destilation"
CKPT_PATH = BASE_DIR / "best_teacher_model.pth"
CONFIG_PATH = BASE_DIR / "config.json"
CLASSMAP_PATH = BASE_DIR / "class_mapping.json"
HISTORY_CSV_PATH = BASE_DIR / "teacher_training_history.csv"

# fallback otomatis kalau CWD/path beda atau ejaan folder berbeda
if not CKPT_PATH.exists():
    print(f"⚠ CKPT tidak ketemu: {CKPT_PATH}")

    alt_dirs = [
        Path("FINAL RESULTS") / "Defensive Distillation",
        Path("FINAL RESULTS") / "Defensive Destillation",
        Path("FINAL RESULTS") / "Devensive Distillation",
        Path("FINAL RESULTS") / "DevensiveDestilation",
        Path("FINAL RESULTS") / "defensive_distillation",
        Path("FINAL RESULTS") / "distillation",
        Path("FINAL RESULTS") / "DD",
    ]
    found = False
    for d in alt_dirs:
        cand = d / "best_teacher_model.pth"
        if cand.exists():
            BASE_DIR = d
            CKPT_PATH = cand
            CONFIG_PATH = BASE_DIR / "config.json"
            CLASSMAP_PATH = BASE_DIR / "class_mapping.json"
            HISTORY_CSV_PATH = BASE_DIR / "teacher_training_history.csv"
            print(f"✅ Mengubah BASE_DIR ke: {BASE_DIR}")
            print(f"✅ CKPT ditemukan: {CKPT_PATH}")
            found = True
            break

    if not found:
        hits = list(Path.cwd().rglob("best_teacher_model.pth"))
        if hits:
            CKPT_PATH = hits[0]
            BASE_DIR = CKPT_PATH.parent
            CONFIG_PATH = BASE_DIR / "config.json"
            CLASSMAP_PATH = BASE_DIR / "class_mapping.json"
            HISTORY_CSV_PATH = BASE_DIR / "teacher_training_history.csv"
            print(f"✅ CKPT ditemukan via search: {CKPT_PATH}")
            print(f"✅ BASE_DIR diset ke: {BASE_DIR}")
        else:
            print("❌ Tidak menemukan best_teacher_model.pth di path yang diberikan maupun di bawah CWD.")
            print("📁 Daftar direktori yang ada di CWD:")
            for item in Path.cwd().iterdir():
                if item.is_dir():
                    print(f"  - {item.name}")
            raise FileNotFoundError("Checkpoint Defensive Distillation (teacher) tidak ditemukan (cek CWD/path).")

print(f"✓ BASE_DIR         : {BASE_DIR} (exists: {BASE_DIR.exists()})")
print(f"✓ CKPT_PATH        : {CKPT_PATH} (exists: {CKPT_PATH.exists()})")
print(f"✓ CONFIG_PATH      : {CONFIG_PATH} (exists: {CONFIG_PATH.exists()})")
print(f"✓ CLASSMAP_PATH    : {CLASSMAP_PATH} (exists: {CLASSMAP_PATH.exists()})")
print(f"✓ HISTORY_CSV_PATH : {HISTORY_CSV_PATH} (exists: {HISTORY_CSV_PATH.exists()})")

ADAPTIVE_DIR = BASE_DIR.parent / f"{BASE_DIR.name}_adaptive_attack_complete_TRUE"
ADAPTIVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Output akan disimpan di: {ADAPTIVE_DIR}")

# Path dataset
DATA_ROOT = Path("dataset_rambu_lalu_lintas")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "valid"
TEST_DIR = DATA_ROOT / "test"
EVAL_DIR = TEST_DIR  # menggunakan test directory untuk evaluasi

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

ULTRA_LIGHT = True  # set False kalau mau full

BATCH_SIZE = 4
MAX_SAMPLES = 64
ADAPTIVE_STEPS = 10
NUM_RESTARTS = 2

EPS_PX_LIST = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
EPS_LIST = [e / 255.0 for e in EPS_PX_LIST]

USE_ALPHA_RATIO = False
ADAPTIVE_ALPHA_RATIO = 0.01

# Akurasi base model full test set untuk perbandingan (isi sesuai angka Anda)
BASE_GLOBAL_CLEAN_ACC = 45.5  # %

# Defense type: Defensive Distillation (gunakan CW-margin loss di attack)
DEFENSE_TYPE = "defdistill"

# Parameter RS (biarkan ada; tidak dipakai untuk defdistill)
RS_SIGMA = 0.25
RS_EOT_SAMPLES = 8

USE_BPDA = False
def defense_preprocess(x: torch.Tensor) -> torch.Tensor:
    return x


# ------------------------------------------------------------
# Helper: filter kwargs to match a class constructor signature
# ------------------------------------------------------------
# Why:
# - Checkpoints/configs may contain extra keys not accepted by Vim.__init__.
# - We filter kwargs to avoid unexpected-argument errors.
# ------------------------------------------------------------
# Helper: filter kwargs sesuai signature class
# ------------------------------------------------------------
def filter_kwargs_for_class(kwargs: Dict, cls) -> Dict:
    try:
        sig = inspect.signature(cls.__init__)
        allowed = set(sig.parameters.keys()) - {"self"}
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


# ------------------------------------------------------------
# 1. SAFE MODEL WRAPPER WITH LABEL VALIDATION
# ------------------------------------------------------------
class SafeVimWrapper(nn.Module):
    """
    Safe wrapper around the target model.

    - Tries to instantiate Vision Mamba (Vim) with the provided configuration.
    - Falls back to a small CNN if Vim is unavailable (keeps the script runnable).
    - Provides validate_labels() to clamp invalid targets into [0, num_classes-1].

    Catatan (ID): Wrapper aman + validasi label.
    """
    def __init__(self, model_config: Dict, num_classes: int):
        super().__init__()
        self.num_classes = int(num_classes)

        try:
            from vision_mamba import Vim
            model_config = dict(model_config)
            model_config["num_classes"] = self.num_classes
            model_config = filter_kwargs_for_class(model_config, Vim)
            self.model = Vim(**model_config)
            print(f"✓ Vim loaded with {self.num_classes} classes")
        except Exception as e:
            print(f"⚠ Using safe CNN fallback (Vim unavailable/failed): {e}")
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, self.num_classes),
            )

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def validate_labels(self, targets: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        targets = targets.to(device)
        if (targets < 0).any() or (targets >= self.num_classes).any():
            print(f"⚠ LABEL INVALID: min={int(targets.min())}, max={int(targets.max())}, num_classes={self.num_classes}")
            original = targets.clone()
            targets = torch.clamp(targets, 0, self.num_classes - 1)
            changed = (original != targets).sum().item()
            if changed:
                print(f"  Fixed {changed} invalid labels by clamping.")
        return targets

    def forward(self, x: torch.Tensor):
        return self.model(x)


# ------------------------------------------------------------
# 2. ADAPTIVE PGD (TRUE ADAPTIVE PER DEFENSE)
# ------------------------------------------------------------
class RobustAdaptivePGDAttack:
    """
    Adaptive PGD untargeted (maximize loss).
    - defdistill: pakai CW-style margin loss (lebih cocok untuk distillation defense).
    - rs       : EOT atas gaussian noise (randomized smoothing).
    - plain    : cross-entropy biasa.
    """
    def __init__(
        self,
        model: nn.Module,
        eps: float,
        steps: int = 10,
        restarts: int = 2,
        mean=MEAN,
        std=STD,
        use_alpha_ratio: bool = False,
        alpha_ratio: float = 0.01,
        defense_type: str = "plain",
        rs_sigma: float = 0.25,
        rs_eot_samples: int = 8,
        use_bpda: bool = False,
        preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.model = model
        self.steps = int(steps)
        self.restarts = int(restarts)
        self.device = next(model.parameters()).device

        self.defense_type = defense_type.lower()
        self.rs_sigma = float(rs_sigma)
        self.rs_eot_samples = int(rs_eot_samples)
        self.use_bpda = bool(use_bpda)
        self.preprocess_fn = preprocess_fn

        self.mean_t = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std_t  = torch.tensor(std,  device=self.device).view(1, 3, 1, 1)

        self.x_min = (0.0 - self.mean_t) / self.std_t
        self.x_max = (1.0 - self.mean_t) / self.std_t

        eps = float(eps)
        self.eps = eps
        self.eps_norm = eps / self.std_t

        if use_alpha_ratio:
            self.alpha = self.eps_norm * float(alpha_ratio)
        else:
            self.alpha = (2.0 * self.eps_norm) / max(self.steps, 1)

        print(f"[Attack] defense_type={self.defense_type}, eps={self.eps} (~{self.eps*255:.1f}/255), "
              f"steps={self.steps}, restarts={self.restarts}")

    def _project(self, x_adv: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        delta = x_adv - x_orig
        delta = torch.max(torch.min(delta, self.eps_norm), -self.eps_norm)
        x_adv = x_orig + delta
        x_adv = torch.max(torch.min(x_adv, self.x_max), self.x_min)
        return x_adv

    def _forward_defense(self, x_adv: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Defense-aware forward pass:
        # - rs: EOT over Gaussian noise (averaging logits)
        # - gradmask + BPDA: straight-through estimator for non-diff preprocess
        # - otherwise: standard forward
        if self.defense_type == "rs":
            logits_sum = 0.0
            for _ in range(self.rs_eot_samples):
                noise = torch.randn_like(x_adv) * self.rs_sigma
                noisy = x_adv + noise
                noisy = torch.max(torch.min(noisy, self.x_max), self.x_min)
                logits_sum = logits_sum + self.model(noisy)
            return logits_sum / float(self.rs_eot_samples)

        elif self.defense_type == "gradmask" and self.use_bpda and self.preprocess_fn is not None:
            with torch.no_grad():
                x_def = self.preprocess_fn(x_adv)
            x_st = x_adv + (x_def - x_adv).detach()
            return self.model(x_st)

        else:
            return self.model(x_adv)

    def _cw_margin_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        true_logit = logits[torch.arange(logits.size(0), device=self.device), y]
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, y.unsqueeze(1), True)
        logits_other = logits.masked_fill(mask, float("-inf"))
        max_other, _ = logits_other.max(dim=1)
        return -(true_logit - max_other)

    def attack(self, x: torch.Tensor, y: torch.Tensor, return_trajectory: bool = False):
        self.model.eval()
        x = x.to(self.device)

        if hasattr(self.model, "validate_labels"):
            y = self.model.validate_labels(y)
        else:
            y = y.to(self.device)

        B = x.size(0)
        best_adv = x.detach().clone()
        best_loss = torch.full((B,), -1e9, device=self.device)

        traj = None
        if return_trajectory:
            traj = {"step": [], "avg_loss": [], "acc": []}
        global_step = 0

        with torch.enable_grad():
            for _r in range(self.restarts):
                noise = (2.0 * torch.rand_like(x) - 1.0) * self.eps_norm
                x_adv = x + noise
                x_adv = torch.max(torch.min(x_adv, self.x_max), self.x_min)
                x_adv = self._project(x_adv, x)

                for _ in range(self.steps):
                    global_step += 1
                    x_adv = x_adv.detach().requires_grad_(True)

                    logits = self._forward_defense(x_adv, y)

                    if self.defense_type == "defdistill":
                        loss_vec = self._cw_margin_loss(logits, y)
                    else:
                        loss_vec = F.cross_entropy(logits, y, reduction="none")

                    if return_trajectory:
                        with torch.no_grad():
                            preds = logits.argmax(1)
                            acc = (preds == y).float().mean().item()
                            traj["step"].append(global_step)
                            traj["avg_loss"].append(loss_vec.mean().item())
                            traj["acc"].append(acc)

                    with torch.no_grad():
                        upd = loss_vec > best_loss
                        if upd.any():
                            best_loss[upd] = loss_vec[upd]
                            best_adv[upd] = x_adv[upd].detach()

                    grad = torch.autograd.grad(loss_vec.sum(), x_adv, only_inputs=True)[0]
                    with torch.no_grad():
                        x_adv = x_adv + self.alpha * torch.sign(grad)
                        x_adv = self._project(x_adv, x)

        if return_trajectory:
            return best_adv.detach(), traj
        return best_adv.detach()


# ------------------------------------------------------------
# 3. EVALUASI (CLEAN)
# ------------------------------------------------------------
@torch.no_grad()
def evaluate_clean_acc(model: nn.Module, loader: DataLoader, device: torch.device, max_samples: int) -> float:
    model.eval()
    correct, total = 0, 0
    for images, targets in loader:
        if total >= max_samples:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if hasattr(model, "validate_labels"):
            targets = model.validate_labels(targets)

        remain = max_samples - total
        if images.size(0) > remain:
            images = images[:remain]
            targets = targets[:remain]

        outputs = model(images)
        preds = outputs.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return correct / max(total, 1)


# ------------------------------------------------------------
# 4. VISUALIZER
# ------------------------------------------------------------
def _unnormalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=img.device).view(3, 1, 1)
    std = torch.tensor(STD, device=img.device).view(3, 1, 1)
    x = img * std + mean
    return torch.clamp(x, 0.0, 1.0)


class AttackVisualizer:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def create_attack_progression_plot(self, eps: float, all_metrics: List[Dict]) -> Path:
        if not all_metrics:
            return Path("")
        df = pd.DataFrame(all_metrics)

        plt.figure(figsize=(8, 4))
        plt.plot(df["batch_idx"], df["overall_acc"] * 100.0, marker="o", label="Overall Acc")
        plt.plot(df["batch_idx"], df["batch_acc"] * 100.0, marker="x", linestyle="--", label="Batch Acc")
        plt.xlabel("Batch index")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Attack Progression (ε={eps*255:.1f}/255)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = self.save_dir / f"attack_progression_eps_{int(eps*255)}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return out_path

    def create_adversarial_examples_grid(
        self,
        clean_imgs: torch.Tensor,
        adv_imgs: torch.Tensor,
        clean_preds: torch.Tensor,
        adv_preds: torch.Tensor,
        targets: torch.Tensor,
        eps: float,
        max_visual_samples: int = 6,
    ) -> Path:
        n_show = min(max_visual_samples, clean_imgs.size(0))
        if n_show == 0:
            return Path("")

        fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
        if n_show == 1:
            axes = axes.reshape(1, 3)

        for i in range(n_show):
            o = _unnormalize(clean_imgs[i])
            a = _unnormalize(adv_imgs[i])
            t = int(targets[i].item())
            cp = int(clean_preds[i].item())
            ap = int(adv_preds[i].item())

            axes[i, 0].imshow(o.permute(1, 2, 0).numpy())
            axes[i, 0].set_title(f"Original\nGT={t}, Pred={cp}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(a.permute(1, 2, 0).numpy())
            axes[i, 1].set_title(f"Adversarial\nGT={t}, Pred={ap}")
            axes[i, 1].axis("off")

            d = (adv_imgs[i] - clean_imgs[i])
            d_vis = d / (d.abs().max() + 1e-8)
            d_vis = 0.5 + 0.5 * d_vis
            axes[i, 2].imshow(d_vis.permute(1, 2, 0).numpy())
            axes[i, 2].set_title("Perturbation (scaled)")
            axes[i, 2].axis("off")

        out_path = self.save_dir / f"adv_examples_grid_eps_{int(eps*255)}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return out_path

    def create_perturbation_heatmap(self, clean_imgs: torch.Tensor, adv_imgs: torch.Tensor, eps: float) -> Path:
        if clean_imgs.size(0) == 0:
            return Path("")
        delta = (adv_imgs - clean_imgs).abs()
        mean_delta = delta.mean(dim=0)
        heatmap = mean_delta.mean(dim=0)

        plt.figure(figsize=(4, 4))
        plt.imshow(heatmap.numpy(), cmap="hot")
        plt.colorbar(label="Mean |perturbation|")
        plt.title(f"Perturbation Heatmap (ε={eps*255:.1f}/255)")
        plt.axis("off")

        out_path = self.save_dir / f"perturbation_heatmap_eps_{int(eps*255)}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return out_path


# ------------------------------------------------------------
# 5. TRUE ADAPTIVE EVALUATION + VISUALS
# ------------------------------------------------------------
# For each epsilon:
#   - Generate adversarial examples (adaptive PGD)
#   - Measure robust accuracy
#   - Save diagnostic plots:
#       * attack_progression_eps_*.png
#       * adv_examples_grid_eps_*.png
#       * perturbation_heatmap_eps_*.png
# ------------------------------------------------------------
# 5. EVALUASI ADAPTIVE + VISUALIZER
# ------------------------------------------------------------
def evaluate_adaptive_attack_safe(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float,
    max_samples: int,
) -> float:
    model.eval()
    attacker = RobustAdaptivePGDAttack(
        model=model,
        eps=eps,
        steps=ADAPTIVE_STEPS,
        restarts=NUM_RESTARTS,
        mean=MEAN,
        std=STD,
        use_alpha_ratio=USE_ALPHA_RATIO,
        alpha_ratio=ADAPTIVE_ALPHA_RATIO,
        defense_type=DEFENSE_TYPE,
        rs_sigma=RS_SIGMA,
        rs_eot_samples=RS_EOT_SAMPLES,
        use_bpda=USE_BPDA,
        preprocess_fn=defense_preprocess if USE_BPDA else None,
    )

    correct, total = 0, 0
    failed_batches = 0
    all_metrics: List[Dict] = []
    saved_examples = None

    visualizer = AttackVisualizer(ADAPTIVE_DIR)

    print(f"\n[AdaptivePGD-TRUE] Evaluating ε={eps*255:.1f}/255 on max {max_samples} samples...")
    pbar = tqdm(loader, desc=f"ε={eps*255:.1f}/255", leave=False)

    for batch_idx, (images, targets) in enumerate(pbar):
        if total >= max_samples:
            break
        try:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if hasattr(model, "validate_labels"):
                targets = model.validate_labels(targets)

            remain = max_samples - total
            if images.size(0) > remain:
                images = images[:remain]
                targets = targets[:remain]

            with torch.no_grad():
                clean_outputs = model(images)
                clean_preds = clean_outputs.argmax(dim=1)

            adv_images = attacker.attack(images, targets, return_trajectory=False)

            with torch.no_grad():
                delta_max = (adv_images - images).abs().max().item()
                outputs = model(adv_images)
                preds = outputs.argmax(dim=1)

            batch_correct = (preds == targets).sum().item()
            batch_total = targets.size(0)

            correct += batch_correct
            total += batch_total

            batch_acc = batch_correct / max(batch_total, 1)
            overall_acc = correct / max(total, 1)

            pbar.set_postfix({
                "batch_acc": f"{batch_acc*100.0:.1f}%",
                "overall_acc": f"{overall_acc*100.0:.1f}%",
                "Δmax": f"{delta_max:.4f}",
                "n": f"{total}/{max_samples}",
            })

            all_metrics.append({
                "batch_idx": batch_idx,
                "batch_acc": batch_acc,
                "overall_acc": overall_acc,
                "delta_max": delta_max,
                "batch_size": batch_total,
                "n_seen": total,
            })

            if saved_examples is None and batch_total > 0:
                saved_examples = (
                    images.detach().cpu(),
                    adv_images.detach().cpu(),
                    clean_preds.detach().cpu(),
                    preds.detach().cpu(),
                    targets.detach().cpu(),
                )

        except Exception as e:
            print(f"    ⚠ Batch {batch_idx} failed: {e}")
            failed_batches += 1
            continue
        finally:
            if torch.cuda.is_available() and batch_idx % 2 == 0:
                torch.cuda.empty_cache()

    pbar.close()

    if failed_batches:
        print(f"⚠ {failed_batches} batches failed during evaluation")

    robust_acc = correct / max(total, 1)
    print(f"✅ [AdaptivePGD-TRUE ε={eps*255:.1f}/255] Robust Acc: {robust_acc*100:.2f}% (Correct: {correct}/{total})")

    prog = visualizer.create_attack_progression_plot(eps, all_metrics)
    if prog and prog.exists():
        print(f"  ✓ Attack progression: {prog}")

    if saved_examples is not None:
        clean_imgs, adv_imgs, clean_preds, adv_preds, targets = saved_examples
        grid = visualizer.create_adversarial_examples_grid(clean_imgs, adv_imgs, clean_preds, adv_preds, targets, eps)
        if grid and grid.exists():
            print(f"  ✓ Adversarial examples grid: {grid}")

        heat = visualizer.create_perturbation_heatmap(clean_imgs, adv_imgs, eps)
        if heat and heat.exists():
            print(f"  ✓ Perturbation heatmap: {heat}")

    return robust_acc


# ------------------------------------------------------------
# 6. LOAD MODEL & DATA (AMAN) - DEFENSIVE DISTILLATION TEACHER
# ------------------------------------------------------------
def safe_load_model_and_data(device: torch.device) -> Tuple[nn.Module, DataLoader, Dict[int, str]]:
    print("🔍 Scanning for required files in Defensive Distillation directory...")

    # Load config.json
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            print(f"✓ Loaded config from {CONFIG_PATH.name}")
        except Exception as e:
            print(f"⚠ Failed to load config: {e}")
            cfg = {}
    else:
        print("⚠ No config.json found, using default config {}")

    # (opsional) baca sekilas history csv untuk memastikan file terbaca
    if HISTORY_CSV_PATH.exists():
        try:
            _dfh = pd.read_csv(HISTORY_CSV_PATH)
            print(f"✓ Loaded teacher history CSV: {HISTORY_CSV_PATH.name} (rows={len(_dfh)})")
        except Exception as e:
            print(f"⚠ Failed to read teacher history CSV: {e}")

    # Load class mapping
    mapping_raw = {}
    if CLASSMAP_PATH.exists():
        try:
            mapping_raw = json.loads(CLASSMAP_PATH.read_text(encoding="utf-8"))
            print(f"✓ Loaded class mapping from {CLASSMAP_PATH.name}")
        except Exception as e:
            print(f"⚠ Failed to load class mapping: {e}")

    def build_idx_to_class(mapping: Dict) -> Dict[int, str]:
        if not mapping:
            return {}
        if "idx_to_class" in mapping:
            sub = mapping["idx_to_class"]
            return {int(k): v for k, v in sub.items()}
        if "class_to_idx" in mapping:
            sub = mapping["class_to_idx"]
            return {int(v): k for k, v in sub.items()}

        all_keys_int = True
        for k in mapping.keys():
            try:
                int(k)
            except Exception:
                all_keys_int = False
                break
        if all_keys_int:
            return {int(k): v for k, v in mapping.items()}

        try:
            return {int(v): k for k, v in mapping.items()}
        except Exception:
            print("❌ Gagal membangun idx_to_class dari mapping:")
            print("   Contoh isi mapping:", list(mapping.items())[:5])
            return {}

    idx_to_class = build_idx_to_class(mapping_raw)
    if not idx_to_class:
        print("⚠ Empty class mapping, will infer from dataset")

    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    print("✓ Checkpoint loaded successfully")

    # Ambil vim_kwargs kalau ada
    vim_kwargs = {}
    if isinstance(ckpt, dict):
        for k in ["vim_kwargs", "model_kwargs", "config", "cfg"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                vim_kwargs = dict(ckpt[k])
                print(f"Using {k} from checkpoint")
                break

    # gabung dari config.json jika ada
    if isinstance(cfg, dict):
        if "vim_kwargs" in cfg and isinstance(cfg["vim_kwargs"], dict):
            vim_kwargs.update(cfg["vim_kwargs"])
        elif "model_kwargs" in cfg and isinstance(cfg["model_kwargs"], dict):
            vim_kwargs.update(cfg["model_kwargs"])

    # Determine num_classes
    if idx_to_class:
        num_classes = len(idx_to_class)
        print(f"Number of classes from mapping: {num_classes}")
    else:
        num_classes = int(vim_kwargs.get("num_classes", 10))
        print(f"⚠ num_classes fallback: {num_classes}")

    vim_kwargs["num_classes"] = num_classes
    model = SafeVimWrapper(vim_kwargs, num_classes).to(device)

    # Load weights
    loaded = False
    if isinstance(ckpt, dict):
        for key in ["model", "model_state", "state_dict", "model_state_dict", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                try:
                    missing_keys, unexpected_keys = model.model.load_state_dict(ckpt[key], strict=False)
                    loaded = True
                    print(f"✓ Loaded weights from '{key}'")
                    if missing_keys:
                        print(f"  Missing keys: {missing_keys}")
                    if unexpected_keys:
                        print(f"  Unexpected keys: {unexpected_keys}")
                    break
                except Exception as e:
                    print(f"⚠ Failed load from {key}: {e}")

    if not loaded:
        try:
            if isinstance(ckpt, dict):
                missing_keys, unexpected_keys = model.model.load_state_dict(ckpt, strict=False)
                print("✓ Loaded checkpoint directly into model (dict)")
                if missing_keys:
                    print(f"  Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"  Unexpected keys: {unexpected_keys}")
            else:
                model.model.load_state_dict(ckpt, strict=False)
                print("✓ Loaded checkpoint directly into model")
            loaded = True
        except Exception as e:
            print(f"⚠ Could not load model weights: {e}")
            print("⚠ Using random initialization")

    model.eval()

    # Setup dataset
    eval_dir = EVAL_DIR
    if not eval_dir.exists():
        for p in [DATA_ROOT/"test", DATA_ROOT/"val", DATA_ROOT/"valid", DATA_ROOT/"validation", DATA_ROOT/"eval"]:
            if p.exists():
                eval_dir = p
                print(f"Using evaluation directory: {eval_dir}")
                break
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    eval_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    print("Loading dataset...")
    eval_ds = datasets.ImageFolder(eval_dir, transform=eval_tfms)
    print(f"✓ Dataset loaded with {len(eval_ds)} samples")

    if not idx_to_class and hasattr(eval_ds, "class_to_idx"):
        idx_to_class = {v: k for k, v in eval_ds.class_to_idx.items()}
        print(f"✓ Inferred {len(idx_to_class)} classes from dataset")
    elif not idx_to_class:
        idx_to_class = {i: f"class_{i}" for i in range(num_classes)}
        print(f"⚠ Created generic mapping for {num_classes} classes")

    actual_max = min(MAX_SAMPLES, len(eval_ds))
    if actual_max < len(eval_ds):
        indices = torch.randperm(len(eval_ds))[:actual_max].tolist()
        eval_ds = Subset(eval_ds, indices)
        print(f"Using subset of {actual_max} samples")

    eval_loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    return model, eval_loader, idx_to_class


# ------------------------------------------------------------
# 7. ROBUSTNESS CURVE (CSV + PLOT)
# ------------------------------------------------------------
def create_visualizations(results: List[Dict], save_dir: Path):
    df = pd.DataFrame(results)
    df["epsilon_255"] = df["epsilon"] * 255.0

    csv_path = save_dir / "adaptive_attack_TRUE_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")

    plt.figure(figsize=(10, 6))
    plt.axhline(y=df["clean_acc_percent"].iloc[0], linestyle="--", linewidth=2,
                label="Clean Accuracy (subset, teacher)")
    plt.axhline(y=BASE_GLOBAL_CLEAN_ACC, linestyle=":", linewidth=2,
                label=f"Base Model Clean (full) {BASE_GLOBAL_CLEAN_ACC:.1f}%")

    plt.plot(df["epsilon_255"], df["robust_acc_percent"], marker="o", linewidth=2,
             label="Robust Accuracy (adv)")

    plt.xlabel("Epsilon (pixel)")
    plt.ylabel("Accuracy (%)")
    plt.title("Robustness vs Epsilon (Adaptive PGD - Defensive Distillation Teacher)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = save_dir / f"robustness_curve_TRUE_{DEFENSE_TYPE}_teacher.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Curve saved to: {out_path}")

    return csv_path, out_path


# ------------------------------------------------------------
# 8. MAIN (end-to-end pipeline)
# ------------------------------------------------------------
# Steps:
#   1) Load teacher model + data safely
#   2) Compute clean accuracy on the same subset
#   3) Save attack configuration for reproducibility
#   4) Run adaptive PGD across epsilons
#   5) Save CSV + robustness curve plot
# ------------------------------------------------------------
# 8. MAIN
# ------------------------------------------------------------
def main():
    global MAX_SAMPLES, ADAPTIVE_STEPS, NUM_RESTARTS, RS_EOT_SAMPLES

    print("=" * 70)
    print("COMPLETE TRUE ADAPTIVE PGD ATTACK EVALUATION")
    print("Target model      : Defensive Distillation (Teacher Model)")
    print(f"Model dir         : {BASE_DIR}")
    print(f"Checkpoint        : {CKPT_PATH.name}")
    print(f"Defense type      : {DEFENSE_TYPE}")
    print(f"Testing epsilons (px): {EPS_PX_LIST}")
    print(f"ULTRA_LIGHT mode  : {ULTRA_LIGHT}")
    print(f"Base model clean accuracy (full test set) = {BASE_GLOBAL_CLEAN_ACC:.2f}%")
    print("=" * 70)

    if ULTRA_LIGHT:
        MAX_SAMPLES = min(MAX_SAMPLES, 64)
        ADAPTIVE_STEPS = min(ADAPTIVE_STEPS, 5)
        NUM_RESTARTS = min(NUM_RESTARTS, 1)
        RS_EOT_SAMPLES = min(RS_EOT_SAMPLES, 4)
        print(f"[ULTRA_LIGHT] MAX_SAMPLES={MAX_SAMPLES}, STEPS={ADAPTIVE_STEPS}, "
              f"RESTARTS={NUM_RESTARTS}, RS_EOT_SAMPLES={RS_EOT_SAMPLES}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   Total memory: {mem_gb:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚡ Using CPU")

    try:
        print("\n📥 Step 1: Loading teacher model and data...")
        model, eval_loader, class_mapping = safe_load_model_and_data(device)
        print(f"   ✓ Evaluation samples: {len(eval_loader.dataset)}")
        print(f"   ✓ Number of classes: {len(class_mapping)}")
        print(f"   ✓ Batch size: {BATCH_SIZE}")

        print("\n🧪 Step 2: Clean accuracy baseline on same subset...")
        clean_acc = evaluate_clean_acc(model, eval_loader, device, MAX_SAMPLES)
        clean_acc_percent = clean_acc * 100.0
        print(f"   ✓ Clean Acc (subset): {clean_acc_percent:.2f}%")

        print("\n💾 Step 3: Saving attack configuration...")
        try:
            torch.save({
                "model_state": model.model.state_dict() if hasattr(model, "model") else model.state_dict(),
                "class_mapping": class_mapping,
                "attack_params": {
                    "eps_px_list": EPS_PX_LIST,
                    "eps_list": EPS_LIST,
                    "steps": ADAPTIVE_STEPS,
                    "restarts": NUM_RESTARTS,
                    "use_alpha_ratio": USE_ALPHA_RATIO,
                    "alpha_ratio": ADAPTIVE_ALPHA_RATIO,
                    "clean_acc_percent_subset": clean_acc_percent,
                    "base_global_clean_acc": BASE_GLOBAL_CLEAN_ACC,
                    "max_samples": MAX_SAMPLES,
                    "defense_type": DEFENSE_TYPE,
                    "rs_sigma": RS_SIGMA,
                    "rs_eot_samples": RS_EOT_SAMPLES,
                    "use_bpda": USE_BPDA,
                    "ultra_light": ULTRA_LIGHT,
                    "target_model": str(CKPT_PATH),
                }
            }, ADAPTIVE_DIR / "attack_configuration_TRUE_defdistill_teacher.pth")
            print(f"   ✓ Configuration saved to: {ADAPTIVE_DIR / 'attack_configuration_TRUE_defdistill_teacher.pth'}")
        except Exception as e:
            print(f"   ⚠ Could not save configuration: {e}")

        print("\n🎯 Step 4: Running TRUE Adaptive PGD ...")
        results = []
        start_time = time.time()

        eps_pbar = tqdm(EPS_LIST, desc="Overall Progress")
        for eps in eps_pbar:
            eps_pbar.set_description(f"ε={eps*255:.1f}/255")

            robust_acc = evaluate_adaptive_attack_safe(model, eval_loader, device, eps, MAX_SAMPLES)
            robust_acc_percent = robust_acc * 100.0
            accuracy_drop_subset = clean_acc_percent - robust_acc_percent
            gain_vs_base = robust_acc_percent - BASE_GLOBAL_CLEAN_ACC

            results.append({
                "epsilon": eps,
                "epsilon_255": eps * 255.0,
                "clean_acc_percent": clean_acc_percent,
                "base_clean_acc_percent": BASE_GLOBAL_CLEAN_ACC,
                "robust_acc_percent": robust_acc_percent,
                "accuracy_drop_subset": accuracy_drop_subset,
                "gain_vs_base_clean": gain_vs_base,
                "attack_steps": ADAPTIVE_STEPS,
                "num_restarts": NUM_RESTARTS,
                "samples_tested": MAX_SAMPLES,
                "defense_type": DEFENSE_TYPE,
                "ultra_light": ULTRA_LIGHT,
                "target_model": str(CKPT_PATH),
            })

            eps_pbar.set_postfix({
                "robust": f"{robust_acc_percent:.1f}%",
                "drop(subset)": f"{accuracy_drop_subset:.1f}%",
                "Δvs_base": f"{gain_vs_base:+.1f}pt",
            })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        eps_pbar.close()
        total_time = time.time() - start_time

        print("\n📊 Step 5: Saving robustness reports & plots...")
        csv_path, curve_path = create_visualizations(results, ADAPTIVE_DIR)

        df = pd.DataFrame(results)
        print("\n" + "=" * 70)
        print("SUMMARY (TRUE ADAPTIVE - DEFENSIVE DISTILLATION TEACHER)")
        print("=" * 70)
        cols_show = [
            "epsilon_255",
            "clean_acc_percent",
            "base_clean_acc_percent",
            "robust_acc_percent",
            "accuracy_drop_subset",
            "gain_vs_base_clean",
        ]
        print(df[cols_show].round(2).to_string(index=False))
        print(f"\nOutput:")
        print(f" - CSV   : {csv_path}")
        print(f" - Curve : {curve_path}")
        print(f" - Time  : {total_time:.1f}s")
        print(f" - Attack progression : attack_progression_eps_*.png")
        print(f" - Adv examples grid  : adv_examples_grid_eps_*.png")
        print(f" - Perturbation heatmap: perturbation_heatmap_eps_*.png")

        print("\n✅ DONE.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
