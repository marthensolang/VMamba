"""
TRUE Adaptive Attack (Adversarial Training checkpoint resolver)
==============================================================

Purpose
-------
This script locates the checkpoint/config files of the *Adversarial Training* fine-tuned model
and prepares an output directory for storing TRUE adaptive-attack evaluation artifacts.

Scope (important)
-----------------
The uploaded file mainly contains **path discovery + fallback logic**:
- Prefer:  FINAL RESULTS/Adversarial Training/best_robust_model.pth
- Fallback: checkpoint_epoch_60.pth
- If not found, scan recursively under current working directory (CWD).

If you plan to run a full TRUE adaptive-attack evaluation, you typically import/use the
resolved variables (CKPT_PATH, CONFIG_PATH, CLASSMAP_PATH, ADAPTIVE_DIR) from this file
or paste the same block into the main evaluation script.

Assumptions
-----------
- This file expects `Path` (from `pathlib`) to be available in the runtime environment.
  If you run this file as a standalone script, ensure you have:
      from pathlib import Path

Outputs
-------
- Creates (if missing) the output directory:
    <BASE_DIR.parent>/<BASE_DIR.name>_adaptive_attack_complete_TRUE

Pseudocode
----------
1) Print current working directory
2) Set BASE_DIR and expected file paths (checkpoint/config/class mapping)
3) If best checkpoint not found:
     a) use epoch-60 checkpoint if present
     b) else recursively search under CWD for either checkpoint
     c) if still missing -> raise FileNotFoundError
4) Print resolved paths (and whether they exist)
5) Create output directory for adaptive-attack results

Commenting style
----------------
- Use clear “what/why” comments for file discovery and fallback behavior.
- Log resolved paths explicitly to help other users reproduce your setup.
"""

# =============================================================================
# NOTE: The code below is unchanged (only comments/docstrings were added).
# =============================================================================

# ------------------------------------------------------------
# 0. KONFIGURASI & PATH (ADVERSARIAL TRAINING FINE-TUNED MODEL)
# ------------------------------------------------------------
print(f"[DEBUG] CWD = {Path.cwd()}")

BASE_DIR = Path("FINAL RESULTS") / "Adversarial Training"
CKPT_PATH = BASE_DIR / "best_robust_model.pth"
CONFIG_PATH = BASE_DIR / "training_config.json"
CLASSMAP_PATH = BASE_DIR / "class_mapping.json"
CKPT_FALLBACK = BASE_DIR / "checkpoint_epoch_60.pth"

# cek file utama
if not CKPT_PATH.exists():
    print(f"⚠ best_robust_model.pth tidak ketemu: {CKPT_PATH}")

    # fallback ke checkpoint_epoch_60.pth kalau ada
    if CKPT_FALLBACK.exists():
        CKPT_PATH = CKPT_FALLBACK
        print(f"✅ Pakai fallback checkpoint: {CKPT_PATH}")
    else:
        # cari cepat di bawah CWD untuk jaga-jaga kalau CWD beda
        hits_best = list(Path.cwd().rglob("best_robust_model.pth"))
        hits_ep60 = list(Path.cwd().rglob("checkpoint_epoch_60.pth"))

        if hits_best:
            CKPT_PATH = hits_best[0]
            BASE_DIR = CKPT_PATH.parent
            CONFIG_PATH = BASE_DIR / "training_config.json"
            CLASSMAP_PATH = BASE_DIR / "class_mapping.json"
            CKPT_FALLBACK = BASE_DIR / "checkpoint_epoch_60.pth"
            print(f"✅ best_robust_model.pth ditemukan: {CKPT_PATH}")
            print(f"✅ BASE_DIR diset ke: {BASE_DIR}")
        elif hits_ep60:
            CKPT_PATH = hits_ep60[0]
            BASE_DIR = CKPT_PATH.parent
            CONFIG_PATH = BASE_DIR / "training_config.json"
            CLASSMAP_PATH = BASE_DIR / "class_mapping.json"
            CKPT_FALLBACK = BASE_DIR / "checkpoint_epoch_60.pth"
            print(f"✅ checkpoint_epoch_60.pth ditemukan: {CKPT_PATH}")
            print(f"✅ BASE_DIR diset ke: {BASE_DIR}")
        else:
            raise FileNotFoundError(
                "Tidak menemukan best_robust_model.pth maupun checkpoint_epoch_60.pth "
                "di path yang diberikan atau di bawah CWD."
            )

print(f"✓ BASE_DIR: {BASE_DIR} (exists: {BASE_DIR.exists()})")
print(f"✓ CKPT_PATH: {CKPT_PATH} (exists: {CKPT_PATH.exists()})")
print(f"✓ CONFIG_PATH: {CONFIG_PATH} (exists: {CONFIG_PATH.exists()})")
print(f"✓ CLASSMAP_PATH: {CLASSMAP_PATH} (exists: {CLASSMAP_PATH.exists()})")

ADAPTIVE_DIR = BASE_DIR.parent / f"{BASE_DIR.name}_adaptive_attack_complete_TRUE"
ADAPTIVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Output akan disimpan di: {ADAPTIVE_DIR}")
