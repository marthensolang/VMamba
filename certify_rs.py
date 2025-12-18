#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# file: certify_rs_vim_fixed.py
# title: certified accuracy evaluation for randomized smoothing (fixed)
# purpose: certify l2 robustness (randomized smoothing) and report certified accuracy vs radius
# ============================================================

"""
Certified Accuracy Evaluation for Randomized Smoothing (Fixed)

Goal
----
Evaluate a Vision Mamba classifier using randomized smoothing certification and report:
- clean accuracy (top-1 under the smoothed predictor)
- certified accuracy (correct + non-abstained + radius > 0)
- certified accuracy curve vs. certified radius thresholds
- summary JSON + curve PNG saved to an output folder

Method (high-level)
-------------------
Given an input image x, the smoothed classifier considers noisy samples:
    x_noisy = clip(x + N(0, sigma^2 I), 0, 1)
We estimate the top class A and runner-up class B via Monte Carlo sampling, then
use Clopper–Pearson confidence bounds to compute a certified L2 radius:

    R = max(0, sigma * (Phi^{-1}(pA_lower) - Phi^{-1}(pB_upper)) / 2)

If pA_lower <= pB_upper, we abstain (no certificate / radius = 0).

Pseudocode
----------
1) load config + class mapping + checkpoint
2) build Vision Mamba model and robustly load weights from various checkpoint formats
3) wrap model with input normalization
4) build dataset from test folder (class subfolders)
5) for each batch:
      a) sample N0 noisy predictions -> select candidate class A
      b) sample N noisy predictions  -> count votes for all classes
      c) compute pA_lower and pB_upper via Clopper–Pearson bounds
      d) if separable -> compute certified radius; else abstain
6) compute aggregate clean accuracy and certified accuracy
7) write results.json and plot certified accuracy vs radius curve

Reproducibility
---------------
The script sets seeds for numpy/torch and uses deterministic sampling patterns where possible.
"""

from __future__ import annotations
import os
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
torch.set_printoptions(precision=4, sci_mode=False, linewidth=140)

# ============================================================
# 1. CONFIGURATION - UPDATE THESE PATHS
# ============================================================
BASE_DIR = Path("outputs_vim_rambu_small_20251119_220259")  # update to your output folder
CKPT_PATH = BASE_DIR / "best_vim_rambu_small.pth"           # update to your checkpoint
CONFIG_PATH = BASE_DIR / "config.json"
CLASSMAP_PATH = BASE_DIR / "class_mapping.json"

DATA_ROOT = Path("dataset_rambu_lalu_lintas")
TEST_DIR = DATA_ROOT / "test"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# randomized smoothing config
RS_SIGMA = 0.25
CERT_N0 = 100
CERT_N = 1000
CERT_ALPHA = 0.001
BATCH_SIZE = 8

# ============================================================
# 2. ROBUST CHECKPOINT LOADING (HANDLES MULTIPLE FORMATS)
# ============================================================
def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Robustly extract a PyTorch state_dict from multiple checkpoint formats.

    Supported formats (common in the wild)
    -------------------------------------
    - {"model": state_dict}
    - {"model_state_dict": state_dict}
    - {"state_dict": state_dict}
    - {"net": state_dict}
    - {"weights": state_dict}
    - direct state_dict (no wrapper dict)

    Returns
    -------
    Dict[str, Tensor]
        A valid state_dict mapping parameter/buffer names to tensors.

    Raises
    ------
    KeyError
        If no state_dict-like object can be found.
    """
    if isinstance(ckpt, dict):
        # try common wrapper keys first
        for key in ["model_state_dict", "state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                print(f"[INFO] Found state_dict under key: '{key}'")
                return ckpt[key]

        # if this dict looks like a raw state_dict (all values are tensors)
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            print("[INFO] Checkpoint appears to be a raw state_dict (no wrapper key).")
            return ckpt

        # show keys to help debugging
        print(f"[DEBUG] Checkpoint keys: {list(ckpt.keys())}")

        # search nested dicts that look like a state_dict
        for key in ckpt.keys():
            if isinstance(ckpt[key], dict):
                inner = ckpt[key]
                if inner and all(isinstance(v, torch.Tensor) for v in inner.values()):
                    print(f"[INFO] Found state_dict under nested key: '{key}'")
                    return inner

    raise KeyError(
        "Cannot locate a state_dict in the checkpoint. "
        f"Container type: {type(ckpt)} | "
        f"Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}"
    )


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove 'module.' prefix from keys (typical output of torch.nn.DataParallel).

    This keeps parameter names compatible when loading into a non-DataParallel model.
    """
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================
def read_json(path: Path) -> Any:
    """Read a JSON file and return the decoded object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int = 42):
    """Set RNG seeds for reproducibility across python/numpy/torch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_class_mapping(classmap_obj: Any) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Parse multiple possible class-mapping formats into consistent outputs.

    Returns
    -------
    classes : List[str]
        Ordered list of class names (index is the numeric label).
    class_to_idx : Dict[str, int]
        Mapping from class name to numeric label.
    idx_to_class : Dict[int, str]
        Mapping from numeric label to class name.
    """
    # list of strings: ["classA", "classB", ...]
    if isinstance(classmap_obj, list) and all(isinstance(c, str) for c in classmap_obj):
        classes = list(classmap_obj)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = {i: c for i, c in enumerate(classes)}
        return classes, class_to_idx, idx_to_class

    if isinstance(classmap_obj, dict):
        # common keys: {"classes": [...]} or {"class_names": [...]}
        for key in ["classes", "class_names", "labels", "names"]:
            if key in classmap_obj and isinstance(classmap_obj[key], list):
                classes = list(classmap_obj[key])
                class_to_idx = {c: i for i, c in enumerate(classes)}
                idx_to_class = {i: c for i, c in enumerate(classes)}
                return classes, class_to_idx, idx_to_class

        # id2label / idx_to_class formats
        for key in ["idx_to_class", "id2label"]:
            if key in classmap_obj:
                val = classmap_obj[key]
                if isinstance(val, dict):
                    idx_to_class = {int(k): v for k, v in val.items()}
                    classes = [idx_to_class[i] for i in sorted(idx_to_class)]
                    class_to_idx = {c: i for i, c in enumerate(classes)}
                    return classes, class_to_idx, idx_to_class

        # label2id / class_to_idx formats
        for key in ["class_to_idx", "label2id"]:
            if key in classmap_obj and isinstance(classmap_obj[key], dict):
                val = classmap_obj[key]
                classes = [c for c, _ in sorted(val.items(), key=lambda x: x[1])]
                class_to_idx = {c: i for i, c in enumerate(classes)}
                idx_to_class = {i: c for i, c in enumerate(classes)}
                return classes, class_to_idx, idx_to_class

        # direct idx->class dict
        try:
            idx_to_class = {int(k): str(v) for k, v in classmap_obj.items()}
            classes = [idx_to_class[i] for i in sorted(idx_to_class)]
            class_to_idx = {c: i for i, c in enumerate(classes)}
            return classes, class_to_idx, idx_to_class
        except Exception:
            pass

    raise ValueError(f"Cannot parse class mapping object of type: {type(classmap_obj)}")


# ============================================================
# 4. DATASET
# ============================================================
class FolderDataset(Dataset):
    """
    Minimal folder-based dataset:
    - expects test_dir/<class_name>/*.jpg|png|...
    - returns (tensor_image, int_label)
    """

    def __init__(self, root_dir: Path, classes: List[str], transform=None, img_size=(224, 224)):
        self.root_dir = Path(root_dir)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.img_size = img_size
        self.samples: List[Tuple[Path, int]] = []

        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    self.samples.append((img_path, self.class_to_idx[cls]))

        print(f"[DATA] {root_dir} -> {len(self.samples)} images | {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB").resize(self.img_size, Image.LANCZOS)
            x = self.transform(img) if self.transform else transforms.ToTensor()(img)
            return x, label
        except Exception:
            # keep behavior unchanged: return a random tensor as a fallback
            return torch.rand(3, *self.img_size), label


# ============================================================
# 5. NORMALIZATION WRAPPER
# ============================================================
class NormalizeWrapper(nn.Module):
    """
    Wrap a core model with input normalization.

    Input is expected in [0,1]. The wrapper clamps and normalizes using mean/std,
    then calls the core model.
    """

    def __init__(self, core: nn.Module, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.core = core
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        x = (x - self.mean) / self.std
        return self.core(x)


# ============================================================
# 6. RANDOMIZED SMOOTHING CERTIFICATION
# ============================================================
def norm_ppf(p: float) -> float:
    """
    Inverse CDF (percent-point function) of the standard normal distribution.

    Implementation uses rational approximations (no external scipy dependency).
    """
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def beta_icdf(q: float, a: float, b: float, device: torch.device) -> float:
    """
    Inverse CDF for Beta(a, b) using torch.distributions.

    This is used to compute Clopper–Pearson binomial confidence bounds.
    """
    q = float(min(max(q, 1e-12), 1 - 1e-12))
    dist = torch.distributions.Beta(
        torch.tensor([a], device=device),
        torch.tensor([b], device=device)
    )
    try:
        return dist.icdf(torch.tensor([q], device=device)).item()
    except Exception:
        # fallback: mean of Beta(a,b)
        return a / (a + b)


def lower_confidence_bound(k: int, n: int, alpha: float, device: torch.device) -> float:
    """Clopper–Pearson lower confidence bound for binomial proportion."""
    if k <= 0:
        return 0.0
    return beta_icdf(alpha, k, n - k + 1, device)


def upper_confidence_bound(k: int, n: int, alpha: float, device: torch.device) -> float:
    """Clopper–Pearson upper confidence bound for binomial proportion."""
    if k >= n:
        return 1.0
    return beta_icdf(1 - alpha, k + 1, n - k, device)


class RSCertifier:
    """
    Randomized smoothing certifier for L2 robustness.

    Parameters
    ----------
    sigma : float
        Standard deviation of Gaussian noise used in smoothing.
    """

    def __init__(self, sigma: float):
        self.sigma = float(sigma)

    @torch.no_grad()
    def _predict_counts(self, model: nn.Module, x: torch.Tensor, n_samples: int,
                        num_classes: int, chunk: int = 16) -> torch.Tensor:
        """
        Draw n_samples noisy predictions for each element in a batch and count votes.

        Notes
        -----
        - x has shape (B, C, H, W)
        - internally, the function repeats inputs and adds Gaussian noise
        - counts has shape (B, num_classes) and stores vote counts as int32
        """
        model.eval()
        B = x.size(0)
        counts = torch.zeros(B, num_classes, device=x.device, dtype=torch.int32)

        left = n_samples
        while left > 0:
            m = min(chunk, left)
            left -= m

            x_rep = x.repeat(m, 1, 1, 1)
            noise = torch.randn_like(x_rep) * self.sigma
            x_noisy = torch.clamp(x_rep + noise, 0.0, 1.0)

            logits = model(x_noisy)
            pred = logits.argmax(dim=1).view(m, B)
            for i in range(m):
                counts.scatter_add_(1, pred[i].unsqueeze(1),
                                   torch.ones(B, 1, device=x.device, dtype=torch.int32))

        return counts

    @torch.no_grad()
    def certify_batch(self, model: nn.Module, x: torch.Tensor, num_classes: int,
                      n0: int, n: int, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Certify a batch and return (predicted_class, certified_radius).

        Behavior
        --------
        - first use n0 samples to choose class A
        - then use n samples to estimate class probabilities
        - abstain (pred=-1, radius=0) if pA_lower <= pB_upper
        """
        device = x.device
        model.eval()

        counts0 = self._predict_counts(model, x, n_samples=n0, num_classes=num_classes)
        cA = counts0.argmax(dim=1)

        counts = self._predict_counts(model, x, n_samples=n, num_classes=num_classes)

        B = x.size(0)
        pred = torch.full((B,), -1, device=device, dtype=torch.long)
        radius = torch.zeros((B,), device=device, dtype=torch.float32)

        for i in range(B):
            ca = int(cA[i].item())
            nA = int(counts[i, ca].item())

            counts_i = counts[i].clone()
            counts_i[ca] = -1
            cb = int(counts_i.argmax().item())
            nB = int(counts[i, cb].item())

            pA_lower = lower_confidence_bound(nA, n, alpha, device)
            pB_upper = upper_confidence_bound(nB, n, alpha, device)

            if pA_lower <= pB_upper:
                continue

            ra = norm_ppf(pA_lower)
            rb = norm_ppf(pB_upper)
            r = max(0.0, self.sigma * (ra - rb) / 2.0)
            pred[i] = ca
            radius[i] = r

        return pred, radius


# ============================================================
# 7. MODEL LOADING
# ============================================================
def load_model(ckpt_path: Path, config_path: Path, classmap_path: Path,
               device: torch.device) -> Tuple[nn.Module, List[str], Tuple[int, int]]:
    """
    Load a Vision Mamba model and its class mapping with robust checkpoint handling.

    Returns
    -------
    model : nn.Module
        Normalized model ready for inference.
    classes : List[str]
        Ordered list of class names.
    img_size : Tuple[int, int]
        Image size used for resizing in the dataset.
    """

    print(f"[LOAD] Loading checkpoint: {ckpt_path}")

    # load config + class mapping + checkpoint
    config = read_json(config_path)
    classmap = read_json(classmap_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # parse classes
    classes, _, _ = parse_class_mapping(classmap)
    num_classes = len(classes)
    print(f"[LOAD] Detected {num_classes} classes")

    # locate vim_kwargs (model constructor kwargs)
    vim_kwargs = None
    for key in ["vim_kwargs", "model_kwargs", "model_cfg"]:
        if isinstance(ckpt, dict) and key in ckpt:
            vim_kwargs = ckpt[key]
            break
        if isinstance(config, dict) and key in config:
            vim_kwargs = config[key]
            break

    if vim_kwargs is None:
        raise RuntimeError("Cannot find 'vim_kwargs' in checkpoint or config.")

    # image size (supports int or [H,W])
    img_size_raw = vim_kwargs.get("image_size", vim_kwargs.get("img_size", 224))
    if isinstance(img_size_raw, (list, tuple)):
        img_size = (int(img_size_raw[0]), int(img_size_raw[1]))
    else:
        img_size = (int(img_size_raw), int(img_size_raw))

    print(f"[LOAD] Image size: {img_size}")
    print(f"[LOAD] vim_kwargs: {vim_kwargs}")

    # import and create model
    try:
        from vision_mamba import Vim
    except ImportError:
        raise ImportError("Cannot import Vim. Ensure 'vision_mamba.py' is available in your project.")

    model_core = Vim(**vim_kwargs)

    # robust state_dict extraction
    state_dict = extract_state_dict(ckpt)
    state_dict = strip_module_prefix(state_dict)

    # load weights
    try:
        model_core.load_state_dict(state_dict, strict=True)
        print("[LOAD] state_dict loaded successfully (strict=True).")
    except RuntimeError as e:
        incompatible = model_core.load_state_dict(state_dict, strict=False)
        print(f"[WARN] Loaded with strict=False. Missing: {len(incompatible.missing_keys)}, "
              f"Unexpected: {len(incompatible.unexpected_keys)}")

    model = NormalizeWrapper(model_core).to(device)
    model.eval()

    return model, classes, img_size


# ============================================================
# 8. MAIN CERTIFICATION
# ============================================================
def main():
    """
    Entry point: validate paths, load model, certify the test set, and save outputs.
    """
    print("=" * 70)
    print("CERTIFIED ACCURACY EVALUATION (RANDOMIZED SMOOTHING)")
    print("=" * 70)

    set_seed(42)

    # validate paths
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    if not CLASSMAP_PATH.exists():
        raise FileNotFoundError(f"Class mapping not found: {CLASSMAP_PATH}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # output directory
    out_dir = Path(f"outputs_certified_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] {out_dir.resolve()}")

    # load model
    model, classes, img_size = load_model(CKPT_PATH, CONFIG_PATH, CLASSMAP_PATH, device)
    num_classes = len(classes)

    # create data loader
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = FolderDataset(TEST_DIR, classes=classes, transform=transform, img_size=img_size)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # certify
    certifier = RSCertifier(sigma=RS_SIGMA)

    print(f"\n[CERTIFY] sigma={RS_SIGMA}, n0={CERT_N0}, n={CERT_N}, alpha={CERT_ALPHA}")
    print(f"[CERTIFY] Total samples: {len(test_dataset)}")

    all_radii = []
    all_correct = []
    all_certified = []

    for batch_idx, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        targets = targets.to(device)

        pred, radius = certifier.certify_batch(model, images, num_classes,
                                               n0=CERT_N0, n=CERT_N, alpha=CERT_ALPHA)

        correct = (pred == targets)
        certified = correct & (pred != -1) & (radius > 0)

        all_radii.extend(radius.cpu().tolist())
        all_correct.extend(correct.cpu().tolist())
        all_certified.extend(certified.cpu().tolist())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx + 1) * BATCH_SIZE} / {len(test_dataset)} samples...")

    # results
    total = len(all_correct)
    correct_count = sum(all_correct)
    certified_count = sum(all_certified)

    clean_acc = 100.0 * correct_count / total
    certified_acc = 100.0 * certified_count / total
    mean_radius = np.mean(all_radii)
    median_radius = np.median(all_radii)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Total samples:        {total}")
    print(f"Clean accuracy:       {clean_acc:.2f}%")
    print(f"Certified accuracy:   {certified_acc:.2f}%")
    print(f"Mean radius:          {mean_radius:.4f}")
    print(f"Median radius:        {median_radius:.4f}")

    # save results
    results = {
        "sigma": RS_SIGMA,
        "n0": CERT_N0,
        "n": CERT_N,
        "alpha": CERT_ALPHA,
        "total_samples": total,
        "clean_accuracy": clean_acc,
        "certified_accuracy": certified_acc,
        "mean_radius": mean_radius,
        "median_radius": median_radius,
    }

    with open(out_dir / "certified_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # plot certified accuracy curve vs radius thresholds
    radii_thresholds = np.linspace(0, max(all_radii) if max(all_radii) > 0 else 1.0, 50)
    cert_accs = []
    for r in radii_thresholds:
        count = sum(1 for i in range(len(all_radii)) if all_certified[i] and all_radii[i] >= r)
        cert_accs.append(100.0 * count / total)

    plt.figure(figsize=(10, 6))
    plt.plot(radii_thresholds, cert_accs, 'b-', linewidth=2)
    plt.xlabel("Certified Radius")
    plt.ylabel("Certified Accuracy (%)")
    plt.title(f"Certified Accuracy vs Radius (sigma={RS_SIGMA})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "certified_accuracy_curve.png", dpi=300)
    plt.close()

    print(f"\nSaved outputs:")
    print(f" - {out_dir / 'certified_results.json'}")
    print(f" - {out_dir / 'certified_accuracy_curve.png'}")


if __name__ == "__main__":
    main()
