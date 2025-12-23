# ============================================================
# file: statistical_analysis_clean_fixed.py
# Statistical analysis (clean + robust against key mismatch)
# ============================================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def _to_py(x):
    """Convert numpy types to plain Python for JSON serialization."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _effect_interpretation(d):
    ad = abs(d)
    if ad >= 0.8:
        return "large"
    if ad >= 0.5:
        return "medium"
    if ad >= 0.2:
        return "small"
    return "negligible"


def extract_curve(results: dict, model_key: str):
    """
    Return:
      eps_keys_sorted: list[str] (original JSON keys, sorted by float)
      eps_vals: list[float]
      robust: np.array (N,)
    """
    if model_key not in results:
        raise KeyError(f"Key '{model_key}' tidak ditemukan di JSON.")

    d = results[model_key]
    eps_keys_sorted = sorted(d.keys(), key=lambda k: float(k))
    eps_vals = [float(k) for k in eps_keys_sorted]
    robust = np.array([float(d[k]["robust_accuracy"]) for k in eps_keys_sorted], dtype=float)
    return eps_keys_sorted, eps_vals, robust


def align_by_intersection(results: dict, key_a: str, key_b: str):
    """Align curves by intersection of epsilon keys (string exact)."""
    da = results.get(key_a, {})
    db = results.get(key_b, {})
    common = sorted(set(da.keys()) & set(db.keys()), key=lambda k: float(k))
    if not common:
        raise ValueError("Tidak ada epsilon yang sama antara base_model dan certified_model.")
    eps_vals = [float(k) for k in common]
    a = np.array([float(da[k]["robust_accuracy"]) for k in common], dtype=float)
    b = np.array([float(db[k]["robust_accuracy"]) for k in common], dtype=float)
    return common, eps_vals, a, b


def perform_statistical_analysis(results_path: Path, output_dir: Path):
    print("📊 Performing Statistical Analysis")
    print("=" * 50)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Align epsilon keys safely
    eps_keys, eps_vals, base_robust, cert_robust = align_by_intersection(
        results, "base_model", "certified_model"
    )

    n = len(eps_vals)
    if n < 2:
        raise ValueError("Butuh minimal 2 titik epsilon untuk uji statistik.")

    # Robust accuracy biasanya 0..1. Kalau ternyata sudah 0..100, ini tetap jalan,
    # tapi interpretasi improvement (%) akan berbeda. (Disarankan simpan 0..1.)
    diffs = cert_robust - base_robust

    analysis_results = {}

    # 1) Paired t-test
    t_stat, p_value = stats.ttest_rel(cert_robust, base_robust)
    analysis_results["paired_ttest"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_0p05": bool(p_value < 0.05),
        "n_pairs": int(n),
    }

    # 1b) Wilcoxon (non-parametrik, lebih aman kalau n kecil)
    # Note: wilcoxon butuh tidak semua diffs == 0
    try:
        w_stat, w_p = stats.wilcoxon(diffs)
        analysis_results["wilcoxon"] = {
            "w_statistic": float(w_stat),
            "p_value": float(w_p),
            "significant_0p05": bool(w_p < 0.05),
        }
    except Exception as e:
        analysis_results["wilcoxon"] = {"error": str(e)}

    print("📈 Statistical Significance Test:")
    print(f"   • Paired t-test: t={t_stat:.4f}, p={p_value:.6f} -> {'Significant' if p_value < 0.05 else 'Not significant'}")
    if "p_value" in analysis_results["wilcoxon"]:
        print(f"   • Wilcoxon     : W={analysis_results['wilcoxon']['w_statistic']:.4f}, p={analysis_results['wilcoxon']['p_value']:.6f}")

    # 2) Effect size (paired Cohen's dz)
    diff_mean = float(np.mean(diffs))
    diff_sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    cohens_dz = (diff_mean / diff_sd) if diff_sd > 0 else 0.0
    analysis_results["effect_size"] = {
        "cohens_dz": float(cohens_dz),
        "interpretation": _effect_interpretation(cohens_dz),
        "mean_diff": float(diff_mean),
        "std_diff": float(diff_sd),
    }

    print("\n📏 Effect Size (Paired):")
    print(f"   • Cohen's dz: {cohens_dz:.4f} ({analysis_results['effect_size']['interpretation']})")

    # 3) Improvement analysis (dalam poin persentase jika robust_accuracy 0..1)
    improvements_pp = diffs * 100.0
    analysis_results["improvement"] = {
        "average_pp": float(np.mean(improvements_pp)),
        "maximum_pp": float(np.max(improvements_pp)),
        "minimum_pp": float(np.min(improvements_pp)),
        "all_improvements_pp": [float(x) for x in improvements_pp.tolist()],
    }

    print("\n📈 Improvement Analysis:")
    print(f"   • Avg improvement: {analysis_results['improvement']['average_pp']:.2f} pp")
    print(f"   • Max improvement: {analysis_results['improvement']['maximum_pp']:.2f} pp")
    print(f"   • Min improvement: {analysis_results['improvement']['minimum_pp']:.2f} pp")

    # 4) Save a tidy table
    df = pd.DataFrame({
        "eps_key": eps_keys,
        "epsilon": eps_vals,
        "epsilon_255": [e * 255.0 for e in eps_vals],
        "base_robust": base_robust,
        "certified_robust": cert_robust,
        "diff": diffs,
        "improvement_pp": improvements_pp,
    })
    df.to_csv(output_dir / "statistical_summary_table.csv", index=False)

    # 5) Plots
    create_statistical_plots(df, analysis_results, output_dir)

    # 6) Save JSON (safe conversion)
    safe = json.loads(json.dumps(analysis_results, default=_to_py))
    with open(output_dir / "statistical_analysis.json", "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2)

    print("\n✅ Statistical analysis completed")
    print(f"📁 Saved:")
    print(f"   - {output_dir / 'statistical_analysis.json'}")
    print(f"   - {output_dir / 'statistical_summary_table.csv'}")
    print(f"   - plots (*.png)")


def create_statistical_plots(df: pd.DataFrame, analysis_results: dict, output_dir: Path):
    # Plot 1: Improvement bar chart (x = eps*255)
    plt.figure(figsize=(10, 6))
    x = df["epsilon_255"].values
    y = df["improvement_pp"].values

    colors = ["lightgreen" if v >= 0 else "lightcoral" for v in y]
    plt.bar(x, y, color=colors)

    plt.xlabel("Epsilon (scale 1/255)")
    plt.ylabel("Improvement in Robust Accuracy (percentage points)")
    plt.title("Improvement from Certified Fine-tuning")
    plt.grid(True, alpha=0.3, axis="y")

    # labels at correct x
    for xi, yi in zip(x, y):
        plt.text(xi, yi + (0.4 if yi >= 0 else -0.6), f"{yi:.1f}",
                 ha="center", va="bottom" if yi >= 0 else "top", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "improvement_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Robust accuracy curves (base vs certified)
    plt.figure(figsize=(10, 6))
    plt.plot(df["epsilon_255"], df["base_robust"] * 100.0, marker="o", linewidth=2, label="Base")
    plt.plot(df["epsilon_255"], df["certified_robust"] * 100.0, marker="o", linewidth=2, label="Certified")
    plt.xlabel("Epsilon (scale 1/255)")
    plt.ylabel("Robust Accuracy (%)")
    plt.title("Robust Accuracy vs Epsilon (Base vs Certified)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "robust_curve_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Mean comparison + significance note
    plt.figure(figsize=(8, 6))
    base_mean = float(np.mean(df["base_robust"])) * 100.0
    cert_mean = float(np.mean(df["certified_robust"])) * 100.0
    means = [base_mean, cert_mean]

    plt.bar(["Base Model", "Certified Model"], means, color=["lightblue", "lightgreen"])
    plt.ylabel("Average Robust Accuracy (%)")

    sig = analysis_results["paired_ttest"]["significant_0p05"]
    p = analysis_results["paired_ttest"]["p_value"]
    plt.title(f"Average Robustness Comparison\n(p={p:.4g} {'<0.05' if sig else '>=0.05'})")
    plt.grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(means):
        plt.text(i, v + 0.8, f"{v:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_dir / "statistical_significance.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    results_path = Path("outputs_vim_rambu_small_20251119_220259_certified_finetuned/comprehensive_analysis.json")
    output_dir = Path("outputs_vim_rambu_small_20251119_220259_certified_finetuned")

    if results_path.exists():
        perform_statistical_analysis(results_path, output_dir)
    else:
        print("❌ Results file not found:", results_path)
        print("   Run certified fine-tuning / comprehensive analysis first.")
