# ============================================================
# file: gpu_rs_training_fixed.py
# True Randomized Smoothing Fine-tuning (Vision Mamba ONLY)
# ============================================================
"""
Title
-----
Randomized Smoothing Fine-Tuning for Vision Mamba (GPU-Oriented) + Optional Certification

Purpose
-------
This script fine-tunes a pre-trained Vision Mamba (Vim) model using Randomized Smoothing (Gaussian noise
in pixel space) and periodically evaluates both:
1) smoothed (Monte Carlo) accuracy, and
2) certified accuracy / certified radius under L2 randomized smoothing (via Clopper–Pearson bounds).

Pseudocode
----------
1. Validate all required paths (checkpoint, config, class mapping, dataset folders).
2. Load checkpoint + config, parse class mapping, and build Vim with the original vim_kwargs.
3. Wrap the model with input normalization (ImageNet mean/std).
4. Build train/val loaders with lightweight augmentations.
5. Train for MAX_EPOCHS:
   - compute RS training loss using n_train noisy samples per batch,
   - update with gradient accumulation,
   - log clean accuracy for sanity checks.
6. Validate smoothed accuracy using n_pred samples (Monte Carlo voting).
7. Every CERTIFY_EVERY epochs (or at the end), compute certified accuracy/radius on a subset.
8. Save checkpoints, best models, training history JSON, and simple plots.

Notes
-----
- Only comments, docstrings, and user-facing messages are edited for GitHub readability.
- Model logic, math, and control-flow are intentionally preserved to avoid introducing new runtime errors.
"""


from __future__ import annotations
import os
import json
import time
import math
import warnings
import gc
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
torch.set_printoptions(precision=4, sci_mode=False, linewidth=140)

# ============================================================
# 1. CONFIGURATION AND PATHS
# ============================================================
BASE_DIR = Path("outputs_vim_rambu_small_20251119_220259")
CKPT_PATH = BASE_DIR / "best_vim_rambu_small.pth"
CONFIG_PATH = BASE_DIR / "config.json"
CLASSMAP_PATH = BASE_DIR / "class_mapping.json"

DATA_ROOT = Path("dataset_rambu_lalu_lintas")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "valid"
TEST_DIR = DATA_ROOT / "test"

print("=== PATH VALIDATION ===")
print(f"BASE_DIR exists: {BASE_DIR.exists()}")
print(f"CKPT_PATH exists: {CKPT_PATH.exists()} -> {CKPT_PATH}")
print(f"CONFIG_PATH exists: {CONFIG_PATH.exists()} -> {CONFIG_PATH}")
print(f"CLASSMAP_PATH exists: {CLASSMAP_PATH.exists()} -> {CLASSMAP_PATH}")
print(f"DATA_ROOT exists: {DATA_ROOT.exists()}")
print(f"TRAIN_DIR exists: {TRAIN_DIR.exists()}")
print(f"VAL_DIR exists: {VAL_DIR.exists()}")
print(f"TEST_DIR exists: {TEST_DIR.exists()}")

if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
if not CLASSMAP_PATH.exists():
    raise FileNotFoundError(f"Class mapping not found: {CLASSMAP_PATH}")

# -----------------------
# Training hyperparams
# -----------------------
RS_SIGMA = 0.25
RS_NUM_SAMPLES_TRAIN = 2
RS_NUM_SAMPLES_PRED = 16

CERT_N0 = 32
CERT_N = 256
CERT_ALPHA = 0.001

BATCH_SIZE = 8
ACCUMULATION_STEPS = 2
MAX_EPOCHS = 15
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4
USE_AMP = False

EVAL_MAX_SAMPLES = 256
CERTIFY_EVERY = 3

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

print("\n=== TRAINING CONFIG ===")
print(f"Batch size: {BATCH_SIZE}")
print(f"Accumulation steps: {ACCUMULATION_STEPS}")
print(f"Epochs: {MAX_EPOCHS}")
print(f"Learning rate: {BASE_LR}")
print(f"Weight decay: {WEIGHT_DECAY}")
print(f"Randomized Smoothing Sigma: {RS_SIGMA}  (pixel space [0,1])")
print(f"Train noise samples (MC): {RS_NUM_SAMPLES_TRAIN}")
print(f"Pred noise samples (val quick): {RS_NUM_SAMPLES_PRED}")
print(f"Certify: N0={CERT_N0}, N={CERT_N}, alpha={CERT_ALPHA}")
print(f"EVAL_MAX_SAMPLES: {EVAL_MAX_SAMPLES}")
print(f"USE_AMP: {USE_AMP}")

# ============================================================
# 2. HELPERS
# ============================================================
def read_json(path: Path) -> Any:
    """Load a JSON file from disk and return the parsed Python object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def set_seed(seed: int = 42):
    """Set Python/NumPy/PyTorch RNG seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove the 'module.' prefix from state_dict keys (DataParallel compatibility)."""
    if not state_dict:
        return state_dict
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Extract a model state_dict from common checkpoint wrapper formats."""
    for key in ["model_state_dict", "state_dict", "model", "net", "weights"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    if isinstance(ckpt, dict) and ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # type: ignore
    raise KeyError("Cannot find state_dict. Please check your checkpoint format.")

def _is_list_of_str(x: Any) -> bool:
    """Return True if x is a non-empty list of strings."""
    return isinstance(x, list) and len(x) > 0 and all(isinstance(i, str) for i in x)

def _try_int(x: Any) -> Optional[int]:
    """Best-effort int conversion; return None if conversion fails."""
    try:
        return int(x)
    except Exception:
        return None

def parse_class_mapping(classmap_obj: Any) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Flexible class-mapping parser.

    Supported inputs
    ----------------
    - ["classA", "classB", ...]
    - {"classes":[...]} / {"class_names":[...]} / {"labels":[...]}
    - {"idx_to_class": {...}} or {"id2label": {...}} (dict/list)
    - {"class_to_idx": {...}} or {"label2id": {...}}
    - direct dict idx->class or class->idx
    - nested structures such as {"mapping": {...}}, etc.
    """
    # 1) list[str]
    if _is_list_of_str(classmap_obj):
        classes = list(classmap_obj)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = {i: c for i, c in enumerate(classes)}
        return classes, class_to_idx, idx_to_class

    # 2) dict
    if isinstance(classmap_obj, dict):
        # single nested object
        if len(classmap_obj) == 1:
            only_val = next(iter(classmap_obj.values()))
            try:
                return parse_class_mapping(only_val)
            except Exception:
                pass

        # keys list[str]
        for key in ["classes", "class_names", "labels", "names"]:
            if key in classmap_obj and _is_list_of_str(classmap_obj[key]):
                classes = list(classmap_obj[key])
                class_to_idx = {c: i for i, c in enumerate(classes)}
                idx_to_class = {i: c for i, c in enumerate(classes)}
                return classes, class_to_idx, idx_to_class

        # idx_to_class / id2label
        for key in ["idx_to_class", "id2label", "index_to_class"]:
            if key in classmap_obj:
                val = classmap_obj[key]
                if _is_list_of_str(val):
                    classes = list(val)
                    class_to_idx = {c: i for i, c in enumerate(classes)}
                    idx_to_class = {i: c for i, c in enumerate(classes)}
                    return classes, class_to_idx, idx_to_class
                if isinstance(val, dict):
                    tmp: Dict[int, str] = {}
                    ok = True
                    for k, v in val.items():
                        ik = _try_int(k)
                        if ik is None or not isinstance(v, str):
                            ok = False
                            break
                        tmp[ik] = v
                    if ok and tmp:
                        classes = [tmp[i] for i in sorted(tmp)]
                        class_to_idx = {c: i for i, c in enumerate(classes)}
                        idx_to_class = {i: c for i, c in enumerate(classes)}
                        return classes, class_to_idx, idx_to_class

        # class_to_idx
        for key in ["class_to_idx", "label2id", "class2idx"]:
            if key in classmap_obj and isinstance(classmap_obj[key], dict):
                val = classmap_obj[key]
                tmp: Dict[str, int] = {}
                ok = True
                for k, v in val.items():
                    if not isinstance(k, str):
                        ok = False
                        break
                    iv = _try_int(v)
                    if iv is None:
                        ok = False
                        break
                    tmp[k] = iv
                if ok and tmp:
                    classes = [c for c, _ in sorted(tmp.items(), key=lambda x: x[1])]
                    class_to_idx = {c: i for i, c in enumerate(classes)}
                    idx_to_class = {i: c for i, c in enumerate(classes)}
                    return classes, class_to_idx, idx_to_class

        # dict idx->class?
        tmp_idx_to_class: Dict[int, str] = {}
        ok_int_keys = True
        for k, v in classmap_obj.items():
            ik = _try_int(k)
            if ik is None or not isinstance(v, str):
                ok_int_keys = False
                break
            tmp_idx_to_class[ik] = v

        if ok_int_keys and tmp_idx_to_class:
            classes = [tmp_idx_to_class[i] for i in sorted(tmp_idx_to_class)]
            class_to_idx = {c: i for i, c in enumerate(classes)}
            idx_to_class = {i: c for i, c in enumerate(classes)}
            return classes, class_to_idx, idx_to_class

        # dict class->idx?
        tmp_class_to_idx: Dict[str, int] = {}
        ok_class_keys = True
        for k, v in classmap_obj.items():
            if not isinstance(k, str):
                ok_class_keys = False
                break
            iv = _try_int(v)
            if iv is None:
                ok_class_keys = False
                break
            tmp_class_to_idx[k] = iv

        if ok_class_keys and tmp_class_to_idx:
            classes = [c for c, _ in sorted(tmp_class_to_idx.items(), key=lambda x: x[1])]
            class_to_idx = {c: i for i, c in enumerate(classes)}
            idx_to_class = {i: c for i, c in enumerate(classes)}
            return classes, class_to_idx, idx_to_class

        # other nested containers
        for key in ["mapping", "map", "meta", "data"]:
            if key in classmap_obj:
                try:
                    return parse_class_mapping(classmap_obj[key])
                except Exception:
                    pass

        keys = list(classmap_obj.keys())[:20]
        sample_kv = None
        if keys:
            k0 = keys[0]
            sample_kv = (k0, classmap_obj.get(k0))
        raise ValueError(
            "Unrecognized class_mapping.json format. "
            f"Type={type(classmap_obj).__name__}, keys(sample)={keys}, sample_kv={sample_kv}"
        )

    raise ValueError(f"Unrecognized class_mapping.json format. Type={type(classmap_obj).__name__}")

def find_classes_from_sources(
    ckpt_obj: Any,
    cfg_obj: Any,
    classmap_obj: Any,
) -> Optional[List[str]]:
    """Try to locate class names from checkpoint/config/classmap objects (best effort)."""
    # 1) ckpt
    if isinstance(ckpt_obj, dict):
        for key in ["classes", "class_names", "labels", "idx_to_class"]:
            if key in ckpt_obj:
                try:
                    classes, _, _ = parse_class_mapping(ckpt_obj[key])
                    return classes
                except Exception:
                    pass

    # 2) config
    if isinstance(cfg_obj, dict):
        for key in ["classes", "class_names", "labels", "idx_to_class", "class_mapping"]:
            if key in cfg_obj:
                try:
                    classes, _, _ = parse_class_mapping(cfg_obj[key])
                    return classes
                except Exception:
                    pass

    # 3) file classmap
    try:
        classes, _, _ = parse_class_mapping(classmap_obj)
        return classes
    except Exception:
        return None


# ============================================================
# 3. DATASET
# ============================================================
class FolderDataset(Dataset):
    """A simple folder-based image dataset: root/class_name/*.jpg|png|... -> (tensor, label)."""
    def __init__(self, root_dir: Path, classes: List[str], transform=None, img_size: Tuple[int, int]=(224,224)):
        self.root_dir = Path(root_dir)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.img_size = img_size
        self.samples: List[Tuple[Path, int]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {self.root_dir}")

        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                print(f"[WARN] class folder missing: {cls_dir}")
                continue
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    self.samples.append((img_path, self.class_to_idx[cls]))

        print(f"[DATA] {root_dir} -> {len(self.samples)} images | {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize(self.img_size, Image.LANCZOS)
            x = self.transform(img) if self.transform else transforms.ToTensor()(img)
            return x, label
        except Exception as e:
            print(f"[WARN] failed to load {img_path}: {e}")
            return torch.rand(3, self.img_size[0], self.img_size[1]), label


def create_data_loaders(classes: List[str], img_size: Tuple[int,int], device_is_cuda: bool):
    """Create training and validation DataLoaders with minimal augmentations."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),  # [0,1]
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
    ])

    train_ds = FolderDataset(TRAIN_DIR, classes=classes, transform=train_transform, img_size=img_size)
    if VAL_DIR.exists():
        val_ds = FolderDataset(VAL_DIR, classes=classes, transform=val_transform, img_size=img_size)
    elif TEST_DIR.exists():
        print("[INFO] VAL_DIR is missing; using TEST_DIR for validation.")
        val_ds = FolderDataset(TEST_DIR, classes=classes, transform=val_transform, img_size=img_size)
    else:
        raise FileNotFoundError("No valid/ or test/ directory available for validation.")

    num_workers = 0 if "ipykernel" in str(type(getattr(__builtins__, "__IPYTHON__", ""))).lower() else 2

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device_is_cuda,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_is_cuda,
        drop_last=False,
    )
    return train_loader, val_loader


# ============================================================
# 4. VISION MAMBA LOADING
# ============================================================
def pick_vim_kwargs(config_obj: Any, ckpt_obj: Any) -> Dict[str, Any]:
    """Select vim_kwargs from checkpoint/config in a backward-compatible way."""
    if isinstance(ckpt_obj, dict):
        for k in ["vim_kwargs", "model_kwargs", "model_cfg"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return dict(ckpt_obj[k])

    if isinstance(config_obj, dict):
        for k in ["vim_kwargs", "model_kwargs", "model_cfg"]:
            if k in config_obj and isinstance(config_obj[k], dict):
                return dict(config_obj[k])

    if isinstance(config_obj, dict):
        candidate_keys = [
            "image_size", "img_size", "patch_size", "channels", "num_classes",
            "dim", "depth", "dropout", "dt_rank", "d_state", "dim_inner",
        ]
        found = {k: config_obj[k] for k in candidate_keys if k in config_obj}
        if len(found) >= 3:
            return found

    raise RuntimeError(
        "Cannot find vim_kwargs in checkpoint/config. "
        "Ensure the base training run saved a 'vim_kwargs' dict."
    )


class NormalizeWrapper(nn.Module):
    """Wrap a model with ImageNet-style normalization applied in forward()."""
    def __init__(self, core: nn.Module, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.core = core
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std", torch.tensor(std).view(1,3,1,1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        x = (x - self.mean) / self.std
        return self.core(x)


def load_vim_from_checkpoint(
    ckpt_path: Path,
    config_path: Path,
    classmap_path: Path,
    device: torch.device
) -> Tuple[nn.Module, Dict[str, Any], List[str], Tuple[int,int]]:
    """Load Vim using vim_kwargs and checkpoint weights, returning (model, kwargs, classes, img_size)."""

    run_cfg = read_json(config_path)
    classmap_obj = read_json(classmap_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # --- cari classes dari ckpt/config/classmap ---
    classes = find_classes_from_sources(ckpt, run_cfg, classmap_obj)

    # fallback terakhir: scan folder train/
    if classes is None:
        print("[WARN] Unable to parse class mapping from file/config/ckpt. Fallback: scan train/ folder (risk: label mismatch).")
        classes = sorted([p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])

    if not classes or not all(isinstance(c, str) for c in classes):
        raise RuntimeError("Invalid classes after parsing/fallback.")

    num_classes = len(classes)

    vim_kwargs = pick_vim_kwargs(run_cfg, ckpt)

    # --- IMG SIZE dari 'image_size' atau 'img_size' ---
    img_size_raw = vim_kwargs.get("image_size", vim_kwargs.get("img_size", 224))
    if isinstance(img_size_raw, (list, tuple)) and len(img_size_raw) == 2:
        img_h, img_w = int(img_size_raw[0]), int(img_size_raw[1])
    else:
        img_h = img_w = int(img_size_raw)

    # Do not inject new arguments like 'in_chans' (can break compatibility).
    # Only check num_classes; if different, keep the value from vim_kwargs to preserve head compatibility.
    if "num_classes" in vim_kwargs and vim_kwargs["num_classes"] != num_classes:
        print(
            f"[WARN] num_classes di vim_kwargs ({vim_kwargs['num_classes']}) "
            f"≠ jumlah kelas dari mapping ({num_classes}). "
            "To preserve head compatibility, the value from vim_kwargs will be used."
        )
        num_classes = int(vim_kwargs["num_classes"])
    else:
        # If missing, use len(classes) but DO NOT add a new key (keep kwargs identical to the base run).
        if "num_classes" not in vim_kwargs:
            print(f"[INFO] vim_kwargs has no 'num_classes'; using len(classes)={num_classes} (not adding a new key).")

    try:
        from vision_mamba import Vim
    except Exception as e:
        raise ImportError(
            "Failed to import Vim. Ensure `vision_mamba.py` or the `vision_mamba` package is available.\n"
            "Example: from vision_mamba import Vim"
        ) from e

    # Bangun model dengan kwargs persis seperti saat base training
    model_core = Vim(**vim_kwargs)

    sd = extract_state_dict(ckpt)
    sd = _strip_module_prefix(sd)

    try:
        model_core.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        incompat = model_core.load_state_dict(sd, strict=False)
        missing = list(incompat.missing_keys)
        unexpected = list(incompat.unexpected_keys)

        allowed_prefixes = ("head.", "classifier.", "fc.", "mlp_head.", "proj_head.")
        bad_missing = [k for k in missing if not k.startswith(allowed_prefixes)]
        bad_unexp = [k for k in unexpected if not k.startswith(allowed_prefixes)]

        print("[ERROR] State_dict mismatch detected!")
        print("  missing_keys (example):", missing[:15])
        print("  unexpected_keys (example):", unexpected[:15])

        if bad_missing or bad_unexp:
            raise RuntimeError(
                "STOP to avoid misleading results: mismatch is not limited to the classifier/head. "
                "Verify that vim_kwargs exactly match the previous training run."
            ) from e

        print("[WARN] Mismatch appears limited to head/classifier. Continuing with strict=False.")

    model = NormalizeWrapper(model_core).to(device)
    return model, vim_kwargs, classes, (img_h, img_w)


# ============================================================
# 5. TRUE RS TRAINING + TRUE CERTIFICATION
# ============================================================
def norm_ppf(p: float) -> float:
    """Approximate the inverse CDF (percent-point function) of the standard normal distribution."""
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
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

def beta_icdf(q: float, a: float, b: float, device: torch.device) -> float:
    """Compute the inverse CDF for a Beta(a, b) distribution (used for Clopper–Pearson bounds)."""
    q = float(min(max(q, 1e-12), 1 - 1e-12))
    qa = torch.tensor([q], device=device, dtype=torch.float32)
    A = torch.tensor([a], device=device, dtype=torch.float32)
    B = torch.tensor([b], device=device, dtype=torch.float32)
    dist = torch.distributions.Beta(A, B)
    try:
        v = dist.icdf(qa).item()
    except Exception:
        v = (a / (a + b))
    return float(v)

def lower_confidence_bound(k: int, n: int, alpha: float, device: torch.device) -> float:
    """Lower Clopper–Pearson confidence bound for a binomial proportion."""
    if k <= 0:
        return 0.0
    return beta_icdf(alpha, k, n - k + 1, device)

def upper_confidence_bound(k: int, n: int, alpha: float, device: torch.device) -> float:
    """Upper Clopper–Pearson confidence bound for a binomial proportion."""
    if k >= n:
        return 1.0
    return beta_icdf(1 - alpha, k + 1, n - k, device)

class RSTrainer:
    """Utilities for RS training, smoothed prediction, and randomized smoothing certification."""
    def __init__(self, sigma: float, n_train: int):
        self.sigma = float(sigma)
        self.n_train = int(n_train)

    @torch.no_grad()
    def _predict_counts(self, model: nn.Module, x: torch.Tensor, n_samples: int, num_classes: int, chunk: int = 16) -> torch.Tensor:
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
                counts.scatter_add_(1, pred[i].unsqueeze(1), torch.ones(B, 1, device=x.device, dtype=torch.int32))

        return counts

    def smooth_train_loss(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        total = 0.0
        for _ in range(self.n_train):
            noise = torch.randn_like(x) * self.sigma
            x_noisy = torch.clamp(x + noise, 0.0, 1.0)
            logits = model(x_noisy)
            total = total + criterion(logits, y)
        return total / float(self.n_train)

    @torch.no_grad()
    def smooth_predict(self, model: nn.Module, x: torch.Tensor, n_samples: int, num_classes: int) -> torch.Tensor:
        counts = self._predict_counts(model, x, n_samples=n_samples, num_classes=num_classes, chunk=16)
        return counts.argmax(dim=1)

    @torch.no_grad()
    def certify_batch(self, model: nn.Module, x: torch.Tensor, num_classes: int, n0: int, n: int, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        model.eval()

        counts0 = self._predict_counts(model, x, n_samples=n0, num_classes=num_classes, chunk=16)
        cA = counts0.argmax(dim=1)

        counts = self._predict_counts(model, x, n_samples=n, num_classes=num_classes, chunk=16)

        B = x.size(0)
        pred = torch.full((B,), -1, device=device, dtype=torch.long)
        radius = torch.zeros((B,), device=device, dtype=torch.float32)
        pA_lower_t = torch.zeros((B,), device=device, dtype=torch.float32)

        for i in range(B):
            ca = int(cA[i].item())
            nA = int(counts[i, ca].item())

            counts_i = counts[i].clone()
            counts_i[ca] = -1
            cb = int(counts_i.argmax().item())
            nB = int(counts[i, cb].item())

            pA_lower = lower_confidence_bound(nA, n, alpha, device)
            pB_upper = upper_confidence_bound(nB, n, alpha, device)
            pA_lower_t[i] = float(pA_lower)

            if pA_lower <= pB_upper:
                pred[i] = -1
                radius[i] = 0.0
                continue

            ra = norm_ppf(pA_lower)
            rb = norm_ppf(pB_upper)
            r = (self.sigma * (ra - rb) / 2.0)
            r = max(0.0, float(r))
            pred[i] = ca
            radius[i] = r

        return pred, radius, pA_lower_t


# ============================================================
# 6. TRAIN / EVAL
# ============================================================
@dataclass
class TrainStats:
    """Lightweight container for loss/accuracy statistics."""
    loss: float
    acc: float

@torch.no_grad()
def eval_smoothed_accuracy(model: nn.Module, rs: RSTrainer, loader: DataLoader, num_classes: int, n_pred: int, device: torch.device, max_samples: Optional[int] = None) -> TrainStats:
    """Evaluate smoothed accuracy (Monte Carlo voting) on a loader (optionally truncated)."""
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        noise = torch.randn_like(images) * rs.sigma
        logits = model(torch.clamp(images + noise, 0.0, 1.0))
        loss = criterion(logits, targets)

        preds = rs.smooth_predict(model, images, n_samples=n_pred, num_classes=num_classes)
        total += targets.size(0)
        correct += (preds == targets).sum().item()
        total_loss += loss.item() * targets.size(0)

        if max_samples is not None and total >= max_samples:
            break

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return TrainStats(loss=avg_loss, acc=acc)

@torch.no_grad()
def eval_certified(model: nn.Module, rs: RSTrainer, loader: DataLoader, num_classes: int, device: torch.device,
                   n0: int, n: int, alpha: float, max_samples: Optional[int] = None) -> Dict[str, float]:
    """Evaluate certified accuracy and radius stats on a loader (optionally truncated)."""
    model.eval()
    total = 0
    correct = 0
    certified_correct = 0
    radii: List[float] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        pred, radius, _pA = rs.certify_batch(model, images, num_classes, n0=n0, n=n, alpha=alpha)

        ok = (pred == targets)
        correct += ok.sum().item()
        total += targets.size(0)

        cert_ok = ok & (pred != -1) & (radius > 0)
        certified_correct += cert_ok.sum().item()

        radii.extend([float(r) for r in radius.detach().cpu().tolist()])

        if max_samples is not None and total >= max_samples:
            break

    acc = 100.0 * correct / max(total, 1)
    cert_acc = 100.0 * certified_correct / max(total, 1)
    mean_r = float(np.mean(radii)) if radii else 0.0
    med_r = float(np.median(radii)) if radii else 0.0

    return {
        "smoothed_acc(%)": acc,
        "certified_acc(%)": cert_acc,
        "mean_radius": mean_r,
        "median_radius": med_r,
        "total_eval": float(total),
    }


def run_rs_finetune_vim():
    """Main training loop for RS fine-tuning + periodic certification + artifact saving."""
    print("\n" + "="*70)
    print("TRUE RS FINE-TUNING (VISION MAMBA ONLY)")
    print("="*70)

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

    out_dir = Path(f"outputs_rs_finetune_vim_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] {out_dir.resolve()}")

    model, vim_kwargs, classes, img_size = load_vim_from_checkpoint(CKPT_PATH, CONFIG_PATH, CLASSMAP_PATH, device=device)
    num_classes = len(classes)
    print(f"[MODEL] num_classes: {num_classes} | img_size: {img_size}")
    print(f"[MODEL] classes(sample): {classes[:10]}")

    with open(out_dir / "vim_kwargs.json", "w", encoding="utf-8") as f:
        json.dump(vim_kwargs, f, indent=2)
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    train_loader, val_loader = create_data_loaders(classes=classes, img_size=img_size, device_is_cuda=torch.cuda.is_available())
    print(f"[DATA] train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and torch.cuda.is_available()))
    rs = RSTrainer(sigma=RS_SIGMA, n_train=RS_NUM_SAMPLES_TRAIN)

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [], "train_acc_clean": [],
        "val_loss_noisy1": [], "val_acc_smoothed": [],
        "cert_acc": [], "mean_radius": [],
        "lr": [],
    }

    best_val_smoothed = 0.0
    best_cert_acc = 0.0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        run_loss = 0.0
        run_total = 0
        run_correct_clean = 0

        ep_t0 = time.time()
        print(f"\n[Epoch {epoch}/{MAX_EPOCHS}] TRAIN...")

        for it, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    loss = rs.smooth_train_loss(model, images, targets, criterion) / ACCUMULATION_STEPS
                scaler.scale(loss).backward()
            else:
                loss = rs.smooth_train_loss(model, images, targets, criterion) / ACCUMULATION_STEPS
                loss.backward()

            with torch.no_grad():
                logits_clean = model(images)
                pred_clean = logits_clean.argmax(dim=1)
                run_correct_clean += (pred_clean == targets).sum().item()
                run_total += targets.size(0)
                run_loss += (loss.item() * ACCUMULATION_STEPS) * targets.size(0)

            if it % ACCUMULATION_STEPS == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if it % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                acc_clean = 100.0 * run_correct_clean / max(run_total, 1)
                print(f"  it {it:4d}/{len(train_loader)} | loss {run_loss/max(run_total,1):.4f} | clean_acc {acc_clean:.2f}% | lr {lr:.2e}")

        if len(train_loader) % ACCUMULATION_STEPS != 0:
            if scaler.is_enabled():
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        train_loss = run_loss / max(run_total, 1)
        train_acc_clean = 100.0 * run_correct_clean / max(run_total, 1)

        print(f"[Epoch {epoch}/{MAX_EPOCHS}] VAL (smoothed pred)...")
        val_stats = eval_smoothed_accuracy(
            model=model, rs=rs, loader=val_loader, num_classes=num_classes,
            n_pred=RS_NUM_SAMPLES_PRED, device=device, max_samples=EVAL_MAX_SAMPLES
        )

        cert_acc = 0.0
        mean_radius = 0.0
        if (epoch % CERTIFY_EVERY == 0) or (epoch == MAX_EPOCHS):
            print(f"[Epoch {epoch}/{MAX_EPOCHS}] CERTIFY (true RS)...")
            cert_res = eval_certified(
                model=model, rs=rs, loader=val_loader, num_classes=num_classes, device=device,
                n0=CERT_N0, n=CERT_N, alpha=CERT_ALPHA, max_samples=min(EVAL_MAX_SAMPLES, 128)
            )
            cert_acc = float(cert_res["certified_acc(%)"])
            mean_radius = float(cert_res["mean_radius"])
            print(f"  certify: acc={cert_acc:.2f}% | mean_radius={mean_radius:.4f} | eval={int(cert_res['total_eval'])}")

        lr = optimizer.param_groups[0]["lr"]
        ep_time = time.time() - ep_t0

        history["epoch"].append(float(epoch))
        history["train_loss"].append(float(train_loss))
        history["train_acc_clean"].append(float(train_acc_clean))
        history["val_loss_noisy1"].append(float(val_stats.loss))
        history["val_acc_smoothed"].append(float(val_stats.acc))
        history["cert_acc"].append(float(cert_acc))
        history["mean_radius"].append(float(mean_radius))
        history["lr"].append(float(lr))

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} | clean_acc={train_acc_clean:.2f}% | "
              f"val_loss(noisy1)={val_stats.loss:.4f} | val_acc(smoothed)={val_stats.acc:.2f}% | "
              f"cert_acc={cert_acc:.2f}% | lr={lr:.2e} | time={ep_time:.1f}s")

        if val_stats.acc > best_val_smoothed:
            best_val_smoothed = val_stats.acc
            best_path = out_dir / "best_clean_vim_rs.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "vim_kwargs": vim_kwargs,
                "classes": classes,
                "rs_config": {"sigma": RS_SIGMA, "n_train": RS_NUM_SAMPLES_TRAIN},
                "metrics": {"val_acc_smoothed": val_stats.acc, "train_acc_clean": train_acc_clean},
                "history": history,
            }, best_path)
            print(f"  ✅ save best (val_acc_smoothed): {best_path}")

        if cert_acc > best_cert_acc:
            best_cert_acc = cert_acc
            bestc_path = out_dir / "best_certified_vim_rs.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "vim_kwargs": vim_kwargs,
                "classes": classes,
                "rs_config": {"sigma": RS_SIGMA, "n_train": RS_NUM_SAMPLES_TRAIN},
                "certify_config": {"n0": CERT_N0, "n": CERT_N, "alpha": CERT_ALPHA},
                "metrics": {"cert_acc": cert_acc, "mean_radius": mean_radius},
                "history": history,
            }, bestc_path)
            print(f"  🛡️ save best certified: {bestc_path}")

        ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "vim_kwargs": vim_kwargs,
            "classes": classes,
            "rs_config": {"sigma": RS_SIGMA, "n_train": RS_NUM_SAMPLES_TRAIN},
            "history": history,
        }, ckpt_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - t0

    final_path = out_dir / "final_certified_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "vim_kwargs": vim_kwargs,
        "classes": classes,
        "rs_config": {"sigma": RS_SIGMA, "n_train": RS_NUM_SAMPLES_TRAIN},
        "certify_config": {"n0": CERT_N0, "n": CERT_N, "alpha": CERT_ALPHA},
        "history": history,
        "total_time_sec": total_time,
        "best_val_smoothed": best_val_smoothed,
        "best_cert_acc": best_cert_acc,
    }, final_path)

    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*70)
    print("DONE ✅ TRUE RS FINE-TUNING COMPLETE")
    print("="*70)
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Best val smoothed acc: {best_val_smoothed:.2f}%")
    print(f"Best certified acc: {best_cert_acc:.2f}%")
    print(f"Final model: {final_path}")
    print(f"Total time: {total_time/60:.1f} min")

    try:
        ep = history["epoch"]

        plt.figure(figsize=(10,6))
        plt.plot(ep, history["train_loss"], label="train_loss")
        plt.plot(ep, history["val_loss_noisy1"], label="val_loss(noisy1)")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss"); plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(ep, history["train_acc_clean"], label="train_acc_clean")
        plt.plot(ep, history["val_acc_smoothed"], label="val_acc_smoothed")
        plt.xlabel("epoch"); plt.ylabel("acc (%)"); plt.title("Accuracy"); plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "acc_curve.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(ep, history["cert_acc"], label="cert_acc(%)")
        plt.xlabel("epoch"); plt.ylabel("cert acc (%)"); plt.title("Certified Accuracy (Cohen)"); plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "cert_acc_curve.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(ep, history["mean_radius"], label="mean_radius")
        plt.xlabel("epoch"); plt.ylabel("radius"); plt.title("Mean Certified Radius"); plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "mean_radius_curve.png", dpi=150)
        plt.close()

        print("[PLOT] saved: loss_curve.png, acc_curve.png, cert_acc_curve.png, mean_radius_curve.png")
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    return out_dir


# ============================================================
# 7. QUICK TEST
# ============================================================
def quick_test():
    """Quick sanity test: load model, run forward pass, RS loss, smooth predict, and certify on a tiny batch."""
    print("\n" + "="*60)
    print("QUICK SYSTEM TEST (VIM + RS)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ CUDA not available (CPU mode)")

    model, vim_kwargs, classes, img_size = load_vim_from_checkpoint(CKPT_PATH, CONFIG_PATH, CLASSMAP_PATH, device=device)
    print(f"✓ Loaded Vim | classes={len(classes)} | img_size={img_size}")
    num_classes = len(classes)

    train_loader, _val_loader = create_data_loaders(classes=classes, img_size=img_size, device_is_cuda=torch.cuda.is_available())
    images, targets = next(iter(train_loader))
    images = images.to(device)
    targets = targets.to(device)

    logits = model(images)
    assert logits.shape[1] == num_classes, "Output num_classes mismatch!"
    print(f"✓ Forward OK | logits: {logits.shape}")

    rs = RSTrainer(sigma=RS_SIGMA, n_train=RS_NUM_SAMPLES_TRAIN)
    loss = rs.smooth_train_loss(model, images, targets, nn.CrossEntropyLoss())
    print(f"✓ RS train loss OK: {loss.item():.4f}")

    preds = rs.smooth_predict(model, images, n_samples=8, num_classes=num_classes)
    print(f"✓ Smooth predict OK | preds: {preds[:5].tolist()}")

    pred_c, radius, pA = rs.certify_batch(model, images[:2], num_classes, n0=8, n=32, alpha=0.01)
    print(f"✓ Certify OK | pred={pred_c.tolist()} | radius={radius.tolist()} | pA_lower={pA.tolist()}")

    print("SYSTEM TEST PASSED ✅")
    return True


# ============================================================
# 8. MAIN
# ============================================================
if __name__ == "__main__":
    print("="*70)
    print("TRUE RANDOMIZED SMOOTHING FINE-TUNING - VISION MAMBA ONLY")
    print("OPTIMIZED FOR 16GB GPU")
    print("="*70)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    try:
        if quick_test():
            run_rs_finetune_vim()
        else:
            print("❌ Quick test failed.")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nTraining process completed!")