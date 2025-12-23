#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRUE Adaptive PGD Attack for a Gradient Masking Defense (Evaluation Script)

This script evaluates adversarial robustness of a trained traffic-sign classifier under a
"true adaptive" (defense-aware) untargeted PGD attack across multiple epsilon budgets.

Key idea
- The defended model (GradientMaskingWrapper) outputs PROBABILITIES (softmax), not logits.
  Therefore, the attack loss uses negative log-likelihood on probabilities:  -log(p_y).

Outputs (saved under OUT_DIR)
- adaptive_attack_TRUE_results.csv
- robustness_curve_TRUE_gradmask.png
- adv_examples_grid_eps_<eps255>.png  (one representative batch per epsilon)

Assumptions
- Dataset structure: dataset_rambu_lalu_lintas/{train,valid,test}
- Inputs are normalized with MEAN/STD (ImageNet style). The attack runs in normalized space.
  Epsilon is defined in pixel scale (e.g., 2/255) and converted to normalized-space bounds.

High-level pipeline (pseudocode)
1) Resolve paths (checkpoint/config/architecture/class mapping) and create OUT_DIR
2) Load dataset (EVAL_DIR) with transforms: Resize -> CenterCrop -> ToTensor -> Normalize
3) Build label remapping (dataset labels -> training labels) using class names
4) Rebuild model architecture from model_architecture.json and wrap with GradientMaskingWrapper
5) Load checkpoint weights (strict=False)
6) Evaluate clean accuracy on a subset (MAX_SAMPLES)
7) For each epsilon in EPS_LIST:
     - Run RobustAdaptivePGDAttack (steps, restarts) to generate adversarial examples
     - Compute robust accuracy and save a representative visualization grid
8) Export CSV + robustness curve plot

Safety / research note
- This code is intended for robustness evaluation and reproducible research.
"""


from __future__ import annotations
import os, json, time, warnings, inspect, re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

# =========================
# SPEED / DEBUG
# =========================
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # jangan 1 (lambat)
warnings.filterwarnings("ignore")
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("=== ADAPTIVE PGD ATTACK (MATCH GRADIENT MASKING FINETUNE SCRIPT) ===")
print(f"[DEBUG] CWD={Path.cwd()}")

# ============================================================
# =============================================================================
# Section 0 — Paths & artifacts
# - Defines where the trained Gradient Masking model artifacts live (ckpt/config/arch/classmap).
# - Creates OUT_DIR for all evaluation outputs.
# =============================================================================

# 0) PATH (ACUAN ANDA)
# ============================================================
BASE_DIR = Path("FINAL RESULTS") / "Gradiend Masking"
CKPT_PATH = BASE_DIR / "best_gradient_masking_model.pth"
CONFIG_PATH = BASE_DIR / "gradient_masking_config.json"
ARCH_PATH = BASE_DIR / "model_architecture.json"
CLASSMAP_PATH = BASE_DIR / "class_mapping.json"

for p in [BASE_DIR, CKPT_PATH, CONFIG_PATH, ARCH_PATH, CLASSMAP_PATH]:
    print(f"[PATH] {p} (exists={p.exists()})")

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint tidak ditemukan: {CKPT_PATH}")

OUT_DIR = BASE_DIR.parent / f"{BASE_DIR.name}_adaptive_attack_complete_TRUE"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ OUT_DIR: {OUT_DIR}")

# ============================================================
# =============================================================================
# Section 1 — Dataset selection
# - Uses TEST_DIR if available; otherwise falls back to VAL_DIR.
# =============================================================================

# 1) DATASET PATH
# ============================================================
DATA_ROOT = Path("dataset_rambu_lalu_lintas")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "valid"
TEST_DIR = DATA_ROOT / "test"
EVAL_DIR = TEST_DIR if TEST_DIR.exists() else (VAL_DIR if VAL_DIR.exists() else None)
if EVAL_DIR is None:
    raise FileNotFoundError(f"Tidak ada folder test/ atau valid/ di {DATA_ROOT}")
print(f"✓ EVAL_DIR: {EVAL_DIR}")

# ============================================================
# =============================================================================
# Section 2 — Evaluation settings
# - ULTRA_LIGHT reduces samples/steps/restarts for quick sanity checks.
# - EPS_PX_LIST is in pixel scale; EPS_LIST converts to eps in [0,1] scale.
# =============================================================================

# 2) SETTINGS
# ============================================================
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

ULTRA_LIGHT = True
BATCH_SIZE = 4
MAX_SAMPLES = 64
ADAPTIVE_STEPS = 10
NUM_RESTARTS = 2

EPS_PX_LIST = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
EPS_LIST = [e / 255.0 for e in EPS_PX_LIST]

USE_ALPHA_RATIO = False
ADAPTIVE_ALPHA_RATIO = 0.01

BASE_GLOBAL_CLEAN_ACC = 45.5  # garis pembanding

# Defense type sesuai fine-tuning
DEFENSE_TYPE = "gradmask"
USE_BPDA = False  # gradient masking Anda di wrapper (diff), biasanya BPDA tidak perlu

def defense_preprocess(x: torch.Tensor) -> torch.Tensor:
    # kalau Anda punya preprocess non-diff (mis. quantization), taruh di sini
    return x

# ============================================================
# 3) UTIL JSON
# ============================================================
def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Gagal membaca JSON: {path} | {e}")

def load_json_optional(path: Optional[Path]) -> Dict[str, Any]:
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠ Gagal load {path.name}: {e}")
    return {}

def normalize_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# ============================================================
# =============================================================================
# Section 4 — Class mapping & label remapping (critical for non-zero accuracy)
# - Matches dataset class indices to training-time class indices using normalized class names.
# - If mapping is wrong, clean accuracy can collapse to ~0%.
# =============================================================================

# 4) CLASS MAPPING + LABEL REMAP (PENTING!)
# ============================================================
def parse_class_mapping(mapping: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Mendukung format:
    - {"idx_to_class": {"0": "a", ...}}
    - {"class_to_idx": {"a": 0, ...}}
    - {"0":"a","1":"b"} atau kebalikannya
    """
    if not mapping:
        return {}, {}

    if isinstance(mapping.get("idx_to_class"), dict):
        idx_to_class = {int(k): str(v) for k, v in mapping["idx_to_class"].items()}
        return idx_to_class, {v: k for k, v in idx_to_class.items()}

    if isinstance(mapping.get("class_to_idx"), dict):
        class_to_idx = {str(k): int(v) for k, v in mapping["class_to_idx"].items()}
        return {v: k for k, v in class_to_idx.items()}, class_to_idx

    # flat dict
    keys = list(mapping.keys())
    all_int_keys = True
    for k in keys:
        try:
            int(k)
        except Exception:
            all_int_keys = False
            break

    if all_int_keys:
        idx_to_class = {int(k): str(v) for k, v in mapping.items()}
        return idx_to_class, {v: k for k, v in idx_to_class.items()}

    # assume name->idx
    try:
        class_to_idx = {str(k): int(v) for k, v in mapping.items()}
        return {v: k for k, v in class_to_idx.items()}, class_to_idx
    except Exception:
        return {}, {}

def build_label_mapper(dataset_class_to_idx: Dict[str, int], train_name_to_idx: Dict[str, int]) -> Tuple[torch.Tensor, bool]:
    """
    Buat tensor map: y_train = map[y_dataset]
    Cocokkan berdasarkan nama class (dinormalisasi).
    """
    n_ds = len(dataset_class_to_idx)
    map_arr = np.full((n_ds,), -1, dtype=np.int64)

    train_norm = {normalize_name(k): v for k, v in train_name_to_idx.items()}
    ok = True
    missing = []

    # dataset idx->name
    ds_idx_to_name = {v: k for k, v in dataset_class_to_idx.items()}
    for ds_idx in range(n_ds):
        ds_name = ds_idx_to_name[ds_idx]
        key = normalize_name(ds_name)
        if key in train_norm:
            map_arr[ds_idx] = int(train_norm[key])
        else:
            ok = False
            missing.append(ds_name)

    if not ok:
        print("⚠ LABEL REMAP: Ada class dataset yang tidak ketemu di class_mapping training.")
        print("  Contoh missing (maks 10):", missing[:10])
        print("  Akurasi bisa 0% kalau label tidak match.")
    else:
        print("✓ LABEL REMAP: Nama class dataset cocok dengan mapping training.")

    return torch.tensor(map_arr, dtype=torch.long), ok

# ============================================================
# =============================================================================
# Section 5 — Model reconstruction (must match training)
# - Rebuilds Vim from model_architecture.json, then wraps it with GradientMaskingWrapper.
# - Wrapper outputs probabilities (softmax), not logits.
# =============================================================================

# 5) MODEL: Vim + GradientMaskingWrapper (SAMA DENGAN TRAINING)
# ============================================================
class GradientNoiseLayer(nn.Module):
    def __init__(self, std=0.01):
        super().__init__()
        self.std = float(std)

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x

class SmoothSoftmax(nn.Module):
    def __init__(self, temperature=1.0, epsilon=0.1):
        super().__init__()
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)

    def forward(self, x):
        x = x / self.temperature
        if self.training and self.epsilon > 0:
            probs = F.softmax(x, dim=1)
            smooth_probs = (1 - self.epsilon) * probs + self.epsilon / x.size(1)
            return smooth_probs
        return F.softmax(x, dim=1)

class GradientMaskingWrapper(nn.Module):
    """
    Sesuai training: outputnya PROBABILITAS (softmax), bukan logits.
    Untuk attack, loss harus pakai -log(p_y).
    """
    def __init__(self, base_model: nn.Module, temperature: float, epsilon: float, grad_noise_std: float, use_grad_noise: bool, use_softmax_smoothing: bool):
        super().__init__()
        self.base_model = base_model
        self.use_grad_noise = bool(use_grad_noise)
        self.use_softmax_smoothing = bool(use_softmax_smoothing)
        self.gradient_noise = GradientNoiseLayer(grad_noise_std)
        self.smooth_softmax = SmoothSoftmax(temperature=temperature, epsilon=epsilon)

    def forward(self, x):
        if self.training and self.use_grad_noise:
            x = self.gradient_noise(x)
        logits = self.base_model(x)
        if self.use_softmax_smoothing:
            return self.smooth_softmax(logits)
        return F.softmax(logits, dim=1)

def filter_kwargs_for_class(kwargs: Dict[str, Any], cls) -> Dict[str, Any]:
    try:
        sig = inspect.signature(cls).parameters
        return {k: v for k, v in kwargs.items() if k in sig}
    except Exception:
        return kwargs

def build_vim_from_arch(num_classes: int, arch_kwargs: Dict[str, Any]) -> nn.Module:
    try:
        from vision_mamba import Vim
    except Exception as e:
        raise RuntimeError(f"vision_mamba/Vim tidak bisa diimport. Pastikan modul tersedia. Error: {e}")

    kw = dict(arch_kwargs)
    kw["num_classes"] = int(num_classes)

    # fallback bila ada key berbeda
    kw.setdefault("image_size", kw.get("img_size", 224))
    kw.setdefault("patch_size", 32)
    kw.setdefault("channels", 3)

    kw = filter_kwargs_for_class(kw, Vim)
    model = Vim(**kw)
    return model

# ============================================================
# =============================================================================
# Section 6 — True adaptive PGD attack (probability-based loss)
# - Runs PGD in normalized space.
# - Loss: NLL on probabilities (because wrapper outputs softmax probabilities).
# - Optional BPDA is supported but disabled by default for this wrapper.
# =============================================================================

# 6) ATTACK: PGD pada ruang NORMALIZED, loss = -log(p_y)
# ============================================================
class RobustAdaptivePGDAttack:
    def __init__(
        self,
        model: nn.Module,
        eps: float,
        steps: int,
        restarts: int,
        mean=MEAN,
        std=STD,
        use_alpha_ratio: bool = False,
        alpha_ratio: float = 0.01,
        defense_type: str = "gradmask",
        use_bpda: bool = False,
        preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.model = model
        self.eps = float(eps)
        self.steps = int(steps)
        self.restarts = int(restarts)
        self.device = next(model.parameters()).device
        self.defense_type = defense_type.lower()
        self.use_bpda = bool(use_bpda)
        self.preprocess_fn = preprocess_fn

        self.mean_t = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std_t  = torch.tensor(std,  device=self.device).view(1, 3, 1, 1)

        self.x_min = (0.0 - self.mean_t) / self.std_t
        self.x_max = (1.0 - self.mean_t) / self.std_t

        self.eps_norm = self.eps / self.std_t
        if use_alpha_ratio:
            self.alpha = self.eps_norm * float(alpha_ratio)
        else:
            self.alpha = (2.0 * self.eps_norm) / max(self.steps, 1)

        print(f"[Attack] eps={self.eps} (~{self.eps*255:.1f}/255) steps={self.steps} restarts={self.restarts} "
              f"loss=NLL(on probs) defense={self.defense_type}")

    def _project(self, x_adv: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        delta = x_adv - x0
        delta = torch.max(torch.min(delta, self.eps_norm), -self.eps_norm)
        x_adv = x0 + delta
        x_adv = torch.max(torch.min(x_adv, self.x_max), self.x_min)
        return x_adv

    def _forward_defense(self, x_adv: torch.Tensor) -> torch.Tensor:
        if self.defense_type == "gradmask" and self.use_bpda and self.preprocess_fn is not None:
            with torch.no_grad():
                x_def = self.preprocess_fn(x_adv)
            x_st = x_adv + (x_def - x_adv).detach()
            return self.model(x_st)
        return self.model(x_adv)

    def _nll_on_probs(self, probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # probs: (B,C) sudah softmax
        p = probs[torch.arange(probs.size(0), device=self.device), y]
        return -torch.log(p.clamp_min(1e-12))

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)

        B = x.size(0)
        best_adv = x.detach().clone()
        best_loss = torch.full((B,), -1e9, device=self.device)

        with torch.enable_grad():
            for _ in range(self.restarts):
                noise = (2.0 * torch.rand_like(x) - 1.0) * self.eps_norm
                x_adv = torch.max(torch.min(x + noise, self.x_max), self.x_min)
                x_adv = self._project(x_adv, x)

                for _ in range(self.steps):
                    x_adv = x_adv.detach().requires_grad_(True)
                    probs = self._forward_defense(x_adv)  # probs
                    loss_vec = self._nll_on_probs(probs, y)

                    with torch.no_grad():
                        upd = loss_vec > best_loss
                        best_loss[upd] = loss_vec[upd]
                        best_adv[upd] = x_adv[upd].detach()

                    grad = torch.autograd.grad(loss_vec.sum(), x_adv, only_inputs=True)[0]
                    with torch.no_grad():
                        x_adv = x_adv + self.alpha * torch.sign(grad)
                        x_adv = self._project(x_adv, x)

        return best_adv.detach()

# ============================================================
# 7) VISUALIZER (opsional ringan)
# ============================================================
def _unnormalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=img.device).view(3, 1, 1)
    std = torch.tensor(STD, device=img.device).view(3, 1, 1)
    x = img * std + mean
    return torch.clamp(x, 0.0, 1.0)

class AttackVisualizer:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_grid(self, clean: torch.Tensor, adv: torch.Tensor, y: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, eps: float) -> None:
        n = min(6, clean.size(0))
        if n <= 0:
            return
        fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))
        if n == 1:
            axes = axes.reshape(1, 3)
        for i in range(n):
            o = _unnormalize(clean[i])
            a = _unnormalize(adv[i])
            d = (adv[i] - clean[i])
            d_vis = d / (d.abs().max() + 1e-8)
            d_vis = 0.5 + 0.5 * d_vis

            axes[i, 0].imshow(o.permute(1,2,0).cpu().numpy()); axes[i, 0].axis("off")
            axes[i, 0].set_title(f"Original\nGT={int(y[i])}, Pred={int(p0[i])}")

            axes[i, 1].imshow(a.permute(1,2,0).cpu().numpy()); axes[i, 1].axis("off")
            axes[i, 1].set_title(f"Adversarial\nGT={int(y[i])}, Pred={int(p1[i])}")

            axes[i, 2].imshow(d_vis.permute(1,2,0).cpu().numpy()); axes[i, 2].axis("off")
            axes[i, 2].set_title("Perturbation (scaled)")

        out = self.save_dir / f"adv_examples_grid_eps_{int(eps*255)}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()

# ============================================================
# =============================================================================
# Section 8 — Evaluation helpers
# - eval_clean: clean accuracy on a subset (after optional label remap).
# - eval_adaptive: robust accuracy under the adaptive PGD attack + one grid visualization.
# =============================================================================

# 8) EVAL
# ============================================================
@torch.no_grad()
def eval_clean(model: nn.Module, loader: DataLoader, device: torch.device, max_samples: int, label_map: Optional[torch.Tensor]) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        if total >= max_samples:
            break
        x = x.to(device)
        y = y.to(device)

        if label_map is not None:
            y = label_map[y]

        remain = max_samples - total
        if x.size(0) > remain:
            x, y = x[:remain], y[:remain]

        # defense preprocess (kalau dipakai)
        x_eval = defense_preprocess(x) if (DEFENSE_TYPE == "gradmask" and USE_BPDA) else x
        probs = model(x_eval)
        pred = probs.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

def eval_adaptive(model: nn.Module, loader: DataLoader, device: torch.device, eps: float, max_samples: int, label_map: Optional[torch.Tensor]) -> float:
    model.eval()
    atk = RobustAdaptivePGDAttack(
        model=model,
        eps=eps,
        steps=ADAPTIVE_STEPS,
        restarts=NUM_RESTARTS,
        mean=MEAN,
        std=STD,
        use_alpha_ratio=USE_ALPHA_RATIO,
        alpha_ratio=ADAPTIVE_ALPHA_RATIO,
        defense_type=DEFENSE_TYPE,
        use_bpda=USE_BPDA,
        preprocess_fn=defense_preprocess if USE_BPDA else None,
    )
    correct, total = 0, 0
    visual = AttackVisualizer(OUT_DIR)
    saved = False

    for bidx, (x, y) in enumerate(tqdm(loader, desc=f"ε={eps*255:.1f}/255", leave=False)):
        if total >= max_samples:
            break
        x = x.to(device)
        y = y.to(device)

        if label_map is not None:
            y = label_map[y]

        remain = max_samples - total
        if x.size(0) > remain:
            x, y = x[:remain], y[:remain]

        # pred clean (untuk grid)
        with torch.no_grad():
            probs0 = model(x)
            p0 = probs0.argmax(1)

        adv = atk.attack(x, y)

        with torch.no_grad():
            probs1 = model(adv)
            p1 = probs1.argmax(1)

        correct += (p1 == y).sum().item()
        total += y.numel()

        # simpan grid batch pertama saja
        if (not saved) and x.size(0) > 0:
            visual.save_grid(clean=x.detach().cpu(), adv=adv.detach().cpu(), y=y.detach().cpu(), p0=p0.detach().cpu(), p1=p1.detach().cpu(), eps=eps)
            saved = True

        if device.type == "cuda" and (bidx % 2 == 0):
            torch.cuda.empty_cache()

    return correct / max(total, 1)

def save_curve(rows: List[Dict[str, Any]]) -> Tuple[Path, Path]:
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "adaptive_attack_TRUE_results.csv"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.axhline(y=df["clean_acc_percent"].iloc[0], linestyle="--", linewidth=2, label="Clean (subset)")
    plt.axhline(y=BASE_GLOBAL_CLEAN_ACC, linestyle=":", linewidth=2, label=f"Base clean(full) {BASE_GLOBAL_CLEAN_ACC:.1f}%")
    plt.plot(df["epsilon_255"], df["robust_acc_percent"], marker="o", linewidth=2, label="Robust (adv)")
    plt.xlabel("Epsilon (pixel)")
    plt.ylabel("Accuracy (%)")
    plt.title("Robustness vs Epsilon (Adaptive PGD - Gradient Masking Finetune)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    png_path = OUT_DIR / "robustness_curve_TRUE_gradmask.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✓ CSV  : {csv_path}")
    print(f"✓ CURVE: {png_path}")
    return csv_path, png_path

# ============================================================
# =============================================================================
# Section 9 — Main entrypoint
# - Loads artifacts, builds model, runs clean + adaptive evaluations, saves reports.
# =============================================================================

# 9) MAIN
# ============================================================
def main():
    global MAX_SAMPLES, ADAPTIVE_STEPS, NUM_RESTARTS
    if ULTRA_LIGHT:
        MAX_SAMPLES = min(MAX_SAMPLES, 32)
        ADAPTIVE_STEPS = min(ADAPTIVE_STEPS, 5)
        NUM_RESTARTS = min(NUM_RESTARTS, 1)
        print(f"[ULTRA_LIGHT] MAX_SAMPLES={MAX_SAMPLES}, STEPS={ADAPTIVE_STEPS}, RESTARTS={NUM_RESTARTS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # load files
    cfg = load_json_optional(CONFIG_PATH)
    arch = load_json_optional(ARCH_PATH)
    cmap_raw = load_json_optional(CLASSMAP_PATH)
    idx_to_class, name_to_idx_train = parse_class_mapping(cmap_raw)

    # dataset (normalized, sesuai training)
    img_size = int(arch.get("image_size", cfg.get("image_size", 224)))
    eval_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_ds_full = datasets.ImageFolder(EVAL_DIR, transform=eval_tfms)
    print(f"✓ Dataset loaded: N={len(eval_ds_full)}, classes={len(eval_ds_full.classes)}, img={img_size}")

    # subset
    actual_max = min(MAX_SAMPLES, len(eval_ds_full))
    if actual_max < len(eval_ds_full):
        idx = torch.randperm(len(eval_ds_full))[:actual_max].tolist()
        eval_ds = Subset(eval_ds_full, idx)
        print(f"✓ Using subset: {actual_max} samples")
    else:
        eval_ds = eval_ds_full

    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )

    # label remap (dataset -> training)
    label_map = None
    if name_to_idx_train:
        map_tensor, ok = build_label_mapper(eval_ds_full.class_to_idx, name_to_idx_train)
        label_map = map_tensor.to(device)
        if not ok:
            print("⚠ LABEL REMAP tidak perfect. Tapi saya tetap jalankan, hasil bisa rendah/0 kalau mismatch parah.")
    else:
        print("⚠ class_mapping.json tidak terbaca / kosong. Saya tidak bisa remap label (raw dataset idx dipakai).")

    # num_classes dipakai dari mapping bila ada, else dataset
    num_classes = len(name_to_idx_train) if name_to_idx_train else len(eval_ds_full.classes)
    print(f"✓ num_classes for model: {num_classes}")

    # build Vim from model_architecture.json (sesuai training: OUTPUT_DIR/model_architecture.json = vim_kwargs)
    if not arch:
        raise RuntimeError("model_architecture.json kosong/tidak terbaca. Ini wajib untuk build Vim yang match.")
    base_model = build_vim_from_arch(num_classes=num_classes, arch_kwargs=arch).to(device)

    # wrapper hyperparams (ambil dari config kalau ada)
    temp = float(cfg.get("distillation_temperature", cfg.get("temperature", 20)))
    sm_eps = float(cfg.get("softmax_smoothing", cfg.get("epsilon", 0.1)))
    gns = float(cfg.get("gradient_noise_std", 0.01))
    use_gn = bool(cfg.get("use_gradient_noise", True))
    use_ss = bool(cfg.get("use_softmax_smoothing", True))

    model = GradientMaskingWrapper(
        base_model=base_model,
        temperature=temp,
        epsilon=sm_eps,
        grad_noise_std=gns,
        use_grad_noise=use_gn,
        use_softmax_smoothing=use_ss,
    ).to(device)

    # load checkpoint: training menyimpan 'model_state_dict'
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            sd = ckpt["model_state_dict"]
            print("✓ Using ckpt['model_state_dict']")
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
            print("✓ Using ckpt['state_dict']")
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
            print("✓ Using ckpt['model']")
        else:
            # fallback: jika ckpt langsung state_dict
            sd = ckpt
            print("⚠ Using ckpt as state_dict directly")
    else:
        sd = ckpt
        print("⚠ Using checkpoint object as state_dict directly")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("  missing sample:", missing[:10])
    if len(unexpected) > 0:
        print("  unexpected sample:", unexpected[:10])

    # penting: eval mode (gradient noise OFF)
    model.eval()

    print("\n[1/3] Clean evaluation (subset)...")
    clean_acc = eval_clean(model, eval_loader, device, MAX_SAMPLES, label_map)
    clean_pct = clean_acc * 100.0
    print(f"✅ Clean Acc (subset) = {clean_pct:.2f}%")

    if clean_pct == 0.0:
        print("\n❌ Clean masih 0%. Penyebab paling sering:")
        print("1) LABEL MISMATCH: folder dataset class tidak match class_mapping.json (lihat warning remap).")
        print("2) model_architecture.json bukan vim_kwargs yang dipakai saat training (dim/depth/patch_size beda).")
        print("3) checkpoint bukan model yang benar (best_gradient_masking_model.pth salah folder).")
        print("Saya stop agar tidak buang waktu attack.")
        return

    print("\n[2/3] Adaptive PGD evaluation...")
    rows = []
    t0 = time.time()

    for eps in tqdm(EPS_LIST, desc="Overall"):
        rob = eval_adaptive(model, eval_loader, device, eps, MAX_SAMPLES, label_map)
        rows.append({
            "epsilon": eps,
            "epsilon_255": eps * 255.0,
            "clean_acc_percent": clean_pct,
            "base_clean_acc_percent": BASE_GLOBAL_CLEAN_ACC,
            "robust_acc_percent": rob * 100.0,
            "attack_steps": ADAPTIVE_STEPS,
            "num_restarts": NUM_RESTARTS,
            "samples_tested": MAX_SAMPLES,
            "defense_type": DEFENSE_TYPE,
            "use_bpda": USE_BPDA,
            "target_model": str(CKPT_PATH),
        })

        if device.type == "cuda":
            torch.cuda.empty_cache()

    dt = time.time() - t0

    print("\n[3/3] Save reports...")
    csv_path, png_path = save_curve(rows)

    df = pd.DataFrame(rows)
    print("\n=== SUMMARY ===")
    print(df[["epsilon_255", "clean_acc_percent", "robust_acc_percent"]].round(2).to_string(index=False))
    print(f"\nDONE ✅ time={dt:.1f}s")
    print(f"Output:")
    print(f" - {csv_path}")
    print(f" - {png_path}")
    print(f" - adv_examples_grid_eps_*.png (contoh visual)")

if __name__ == "__main__":
    main()
