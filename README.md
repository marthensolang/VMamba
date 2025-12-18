# DSRM Research Artifact Package — Vision Mamba Robustness on Indonesian Traffic Sign Recognition

This repository is the **research artifact** for a Design Science Research Methodology (DSRM) study on improving and evaluating the robustness of **Vision Mamba** against adversarial attacks in an Indonesian Traffic Sign Recognition (TSR) setting.

It contains **executable scripts**, optional **model checkpoints (.pth)**, and a clear **reproduction protocol** to regenerate the **tables/figures in Chapter IV/V (Bab IV/V)** of the thesis/paper.

---

## 0) Citation and identity

**Title (thesis/paper):** `<YOUR TITLE HERE>`  
**Author:** `<YOUR NAME>`  
**Affiliation:** `<YOUR INSTITUTION>`  
**Date:** `<YYYY-MM-DD>`  
**Repository version used in Bab IV/V:** `git commit <COMMIT_HASH>`  

If you create a DOI (recommended), add it here:
- **DOI:** `<ZENODO/OSF DOI LINK>`

---

## 1) DSRM mapping (how this repo qualifies as an artifact)

This repository supports DSRM as follows:

- **Design & Development:** implementation of defense methods (scripts in `src/`)
- **Demonstration:** training/finetuning to produce defense-specific `.pth` models
- **Evaluation:** producing quantitative robustness metrics (CSV/JSON) and plots (PNG)
- **Communication:** mapping outputs into thesis tables/figures (Bab IV/V), with traceable sources

**Core artifact forms:**
1) Source code (reproducible pipeline)  
2) Checkpoints (`.pth`) — optional in GitHub, can be stored in Releases/Zenodo  
3) Experimental outputs (CSV/JSON + PNG) for auditing and replication  

---

## 2) What is included (manifest)

### Main scripts
Place the following scripts under `src/`:

- `robust_adversarial_training.py`  
  **Adversarial Training** (e.g., FGSM/PGD variants) to produce robust model checkpoints and training reports.

- `gpu_rs_training.py`  
  **Randomized Smoothing fine-tuning** (Gaussian noise based training) optimized for GPU.

- `certify_rs.py`  
  **Randomized Smoothing certification** (L2 certification, typically with statistical bounds) to produce certified accuracy/radius results.

- `defensive_distillation.py`  
  **Defensive Distillation** (teacher–student training) to produce teacher/student checkpoints and histories.

- `gradient_masking.py`  
  **Gradient masking–style defense** included as a **negative control** (to detect/illustrate false robustness).

> Note: If you also have evaluation scripts for AutoAttack/adaptive attacks, add them under `src/` and list them here.

---

## 3) Expected repository layout (recommended)

You can use any structure, but this layout is strongly recommended for review/sidang and replication:

