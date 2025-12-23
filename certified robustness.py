#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
certify_rs_vim_fixed.py
True certification randomized smoothing (L2) dengan Clopper–Pearson bounds.

FIX:
- Tidak lagi tergantung folder outputs_rs_finetune_*.
- Bisa set env RS_CKPT ke path checkpoint hasil fine-tuning RS.
- Kalau env tidak di-set, auto-search checkpoint secara rekursif.
- bincount pakai num_classes tetap (bukan max(pred)+1) agar stabil.

Cara pakai:
  RS_CKPT=outputs_rs_finetune_20251202_123456/best_certified_vim_rs.pth python certify_rs_vim_fixed.py

Atur hyperparameter:
  N0=100 N=5000 ALPHA=0.001 CHUNK=64 MAX_SAMPLES=500 python certify_rs_vim_fixed.py
"""

from __future__ import annotations
import os, json
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from scipy.stats import beta
from scipy import special  # ndtri
from vision_mamba import Vim


# ====== USER PATHS (sesuai kamu) ======
DATA_ROOT = Path(os.getenv("DATA_ROOT", "dataset_rambu_lalu_lintas"))
TEST_DIR = DATA_ROOT / "test"


# ====== Statistik ======
def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    if k <= 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))

def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


# ====== Wrapper model: input pixel [0,1] -> normalize -> core ======
class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std).view(1, 3, 1, 1))
    def forward(self, x):
        return (x - self.mean) / self.std

class SmoothedModel(nn.Module):
    def __init__(self, core: nn.Module, mean, std):
        super().__init__()
        self.core = core
        self.norm = NormalizeLayer(mean, std)
    def forward_noisy(self, x01: torch.Tensor, sigma: float) -> torch.Tensor:
        noise = torch.randn_like(x01) * sigma
        x = torch.clamp(x01 + noise, 0.0, 1.0)
        return self.core(self.norm(x))


@torch.inference_mode()
def sample_predictions_counts(
    smodel: SmoothedModel,
    x01: torch.Tensor,
    sigma: float,
    n: int,
    chunk: int,
    num_classes: int,
) -> np.ndarray:
    """Return counts per class (length=num_classes)."""
    preds_all = []
    remaining = n
    while remaining > 0:
        m = min(chunk, remaining)
        remaining -= m
        x_rep = x01.repeat_interleave(m, dim=0)
        logits = smodel.forward_noisy(x_rep, sigma=sigma)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        preds_all.append(preds)
    preds_all = np.concatenate(preds_all, axis=0)  # (n,)
    counts = np.bincount(preds_all, minlength=num_classes)
    return counts


def certify_one(
    smodel: SmoothedModel,
    x01: torch.Tensor,
    sigma: float,
    N0: int,
    N: int,
    alpha: float,
    chunk: int,
    num_classes: int,
) -> Tuple[int, float, bool]:
    """
    Cohen-style certify (multi-class):
      - sample N0 to select class A (top)
      - sample N to estimate pA, pB with confidence bounds
      - if pA_lower <= pB_upper -> abstain
      - else radius = (sigma/2)*(Phi^-1(pA_lower)-Phi^-1(pB_upper))

    Agar overall confidence ~ (1-alpha), kita pakai Bonferroni: alpha/2 untuk pA dan pB.
    """
    counts0 = sample_predictions_counts(smodel, x01, sigma, N0, chunk, num_classes)
    A0 = int(np.argmax(counts0))

    counts = sample_predictions_counts(smodel, x01, sigma, N, chunk, num_classes)

    # ambil top-2 dari counts
    top2 = np.argsort(-counts)[:2].tolist()
    A = int(top2[0])
    B = int(top2[1]) if len(top2) > 1 else int(top2[0])

    nA = int(counts[A])
    nB = int(counts[B])

    a2 = alpha / 2.0
    pA_lower = clopper_pearson_lower(nA, N, a2)
    pB_upper = clopper_pearson_upper(nB, N, a2)

    if pA_lower <= pB_upper:
        return A, 0.0, True  # abstain (tidak bisa certified)

    # avoid inf
    pA_lower = float(np.clip(pA_lower, 1e-12, 1 - 1e-12))
    pB_upper = float(np.clip(pB_upper, 1e-12, 1 - 1e-12))
    radius = (sigma / 2.0) * (special.ndtri(pA_lower) - special.ndtri(pB_upper))
    radius = float(max(0.0, radius))
    return A, radius, False


# ====== Cari checkpoint tanpa bergantung nama folder ======
def find_rs_checkpoint() -> Path:
    """
    Prioritas:
    1) env RS_CKPT (path langsung)
    2) cari rekursif file checkpoint yang umum
    """
    env_path = os.getenv("RS_CKPT", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"RS_CKPT diset tapi file tidak ada: {p}")

    # auto search
    patterns = [
        "best_certified_vim_rs.pth",
        "certified_vim_rs.pth",
        "final_certified_model.pth",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(list(Path(".").rglob(pat)))

    if not candidates:
        raise FileNotFoundError(
            "Tidak menemukan checkpoint RS.\n"
            "Jalankan dulu Script 1 (fine-tuning RS), atau set env RS_CKPT ke file .pth hasilnya."
        )

    # pilih yang terbaru berdasarkan modified time
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    print("[INFO] cwd   :", Path(".").resolve())

    ckpt_path = find_rs_checkpoint()
    out_dir = ckpt_path.parent  # simpan output di folder yang sama dengan checkpoint
    print("[INFO] RS checkpoint:", ckpt_path)

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Folder test tidak ditemukan: {TEST_DIR}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # Harus ada vim_kwargs + model + rs_config (mean/std/sigma)
    vim_kwargs = ckpt.get("vim_kwargs", None)
    if vim_kwargs is None:
        raise RuntimeError("Checkpoint tidak punya 'vim_kwargs' (arsitektur).")

    model_sd = ckpt.get("model", None)
    if model_sd is None:
        raise RuntimeError("Checkpoint tidak punya state_dict 'model'.")

    rs_cfg = ckpt.get("rs_config", {})
    sigma = float(rs_cfg.get("sigma", os.getenv("RS_SIGMA", "0.25")))
    mean = tuple(rs_cfg.get("mean", (0.485, 0.456, 0.406)))
    std  = tuple(rs_cfg.get("std",  (0.229, 0.224, 0.225)))

    num_classes = int(vim_kwargs.get("num_classes", 0))
    if num_classes <= 1:
        raise RuntimeError("num_classes tidak valid pada vim_kwargs. Pastikan checkpoint RS benar.")

    core = Vim(**vim_kwargs).to(device)
    core.load_state_dict(model_sd, strict=True)
    core.eval()
    smodel = SmoothedModel(core, mean, std).to(device)

    img_size = int(vim_kwargs.get("image_size", vim_kwargs.get("img_size", 224)))
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # pixel [0,1]
    ])
    ds = datasets.ImageFolder(TEST_DIR, transform=tfm)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # Hyperparameters certify (via env)
    N0 = int(os.getenv("N0", "100"))
    N  = int(os.getenv("N", "5000"))
    alpha = float(os.getenv("ALPHA", "0.001"))
    chunk = int(os.getenv("CHUNK", "64"))
    max_samples = int(os.getenv("MAX_SAMPLES", "0"))  # 0 = all

    rmax = float(os.getenv("RMAX", "1.0"))
    rpoints = int(os.getenv("RPOINTS", "21"))
    radii_grid = np.linspace(0.0, rmax, rpoints)

    print(f"[INFO] sigma={sigma} | N0={N0} N={N} alpha={alpha} chunk={chunk} img={img_size} classes={num_classes}")
    print(f"[INFO] test samples={len(ds)} | max_samples={max_samples or 'ALL'}")

    records = []
    for i, (x01, y) in enumerate(loader):
        if max_samples > 0 and i >= max_samples:
            break

        x01 = x01.to(device)
        y_true = int(y.item())

        pred, radius, abstain = certify_one(
            smodel, x01,
            sigma=sigma, N0=N0, N=N, alpha=alpha, chunk=chunk,
            num_classes=num_classes
        )

        cert_correct = (not abstain) and (pred == y_true)
        records.append({
            "idx": int(i),
            "true": int(y_true),
            "pred": int(pred),
            "abstain": bool(abstain),
            "radius": float(radius),
            "cert_correct": bool(cert_correct),
        })

        if (i + 1) % 50 == 0:
            print(f"[{i+1}] radius={radius:.4f} abstain={abstain}")

    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "certify_per_sample.csv", index=False)

    cert_acc = []
    for r in radii_grid:
        cert_acc.append(float(np.mean((df["cert_correct"].values == True) & (df["radius"].values >= r))))

    with open(out_dir / "certify_curve.json", "w", encoding="utf-8") as f:
        json.dump(
            {"radii": radii_grid.tolist(), "certified_accuracy": cert_acc, "N0": N0, "N": N,
             "alpha": alpha, "sigma": sigma, "checkpoint": str(ckpt_path)},
            f, indent=2
        )

    plt.figure(figsize=(10, 6))
    plt.plot(radii_grid, cert_acc, marker="o", linewidth=2)
    plt.xlabel("Certified Radius (L2)")
    plt.ylabel("Certified Accuracy")
    plt.title(f"Certified Accuracy vs Radius (sigma={sigma}, N={N}, alpha={alpha})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "certified_accuracy_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\n[DONE] Saved:")
    print(" -", out_dir / "certify_per_sample.csv")
    print(" -", out_dir / "certify_curve.json")
    print(" -", out_dir / "certified_accuracy_curve.png")


if __name__ == "__main__":
    main()
