"""Generate report figures for the v3 main pipeline writeup."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "Baseline\n(source=MATH-500)": ROOT / "results/part2_experiment/run_cv_20260511_195040",
    "Diverse-buggy\n(source=HLE, no fallback)": ROOT / "results/part2_experiment/run_cv_20260512_140517",
    "Diverse-clean\n(HLE + BBH fallback)": ROOT / "results/part2_experiment/run_cv_20260512_142123",
}
SHORT = {
    "Baseline\n(source=MATH-500)": "Baseline",
    "Diverse-buggy\n(source=HLE, no fallback)": "Diverse-buggy",
    "Diverse-clean\n(HLE + BBH fallback)": "Diverse-clean",
}
COLORS = {"Baseline": "#888888", "Diverse-buggy": "#3b82f6", "Diverse-clean": "#dc2626"}


def load_perm(run_dir: Path) -> dict:
    return json.load(open(run_dir / "agg/permutation_test.json"))


def fig_perfold_rhos():
    """Figure 1: per-fold ρ_s for the three runs, with pooled."""
    folds = ["fold0", "fold1", "fold2", "fold3"]
    data = {}
    for label, run in RUNS.items():
        perm = load_perm(run)
        per_fold = [f["permutation"]["rho_spearman"]["observed"] for f in perm["per_fold"]]
        pooled = perm["pooled"]["rho_spearman"]["observed"]
        data[label] = per_fold + [pooled]

    x = np.arange(5)
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (label, vals) in enumerate(data.items()):
        short = SHORT[label]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=short, color=COLORS[short], edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + (0.02 if v >= 0 else -0.06),
                    f"{v:+.2f}", ha="center", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(folds + ["pooled"])
    ax.set_ylabel(r"Test $\rho_{\mathrm{spearman}}$")
    ax.set_title("Per-fold and pooled Spearman correlation across configurations (K=4, B=10000)")
    ax.set_ylim(-0.2, 1.0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fp = OUT / "fig1_perfold_rho_spearman.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}")


def fig_pooled_grouped():
    """Figure 2: pooled ρ_s / ρ_p / L_align across the three runs."""
    metrics = ["rho_spearman", "rho_pearson", "L_align"]
    labels_m = [r"$\rho_{\mathrm{spearman}}$", r"$\rho_{\mathrm{pearson}}$", r"$L_{\mathrm{align}}$"]
    sign = {"rho_spearman": "high-good", "rho_pearson": "high-good", "L_align": "low-good"}

    data = {}
    pvals = {}
    for label, run in RUNS.items():
        p = load_perm(run)["pooled"]
        data[label] = [p[m]["observed"] for m in metrics]
        pvals[label] = [
            p["rho_spearman"]["p_two_sided"],
            p["rho_pearson"]["p_two_sided"],
            p["L_align"]["p_one_sided_low"],
        ]

    x = np.arange(len(metrics))
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (label, vals) in enumerate(data.items()):
        short = SHORT[label]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=short, color=COLORS[short], edgecolor="black", linewidth=0.5)
        for b, v, pv in zip(bars, vals, pvals[label]):
            star = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            txt = f"{v:+.3f}\n{star}p={pv:.3f}" if abs(v) >= 0.001 else f"{v:.4f}\n{star}p={pv:.3f}"
            ax.text(b.get_x() + b.get_width()/2, v + 0.015,
                    txt, ha="center", fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_m, fontsize=11)
    ax.set_ylabel("Pooled metric value")
    ax.set_title("Pooled test metrics across configurations (n_pairs=25, B=10000 block-permutation)")
    ax.set_ylim(-0.05, 0.85)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fp = OUT / "fig2_pooled_metrics.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}")


def fig_perm_nulls():
    """Figure 3: permutation null distributions for the clean run."""
    run = RUNS["Diverse-clean\n(HLE + BBH fallback)"]
    perm = load_perm(run)["pooled"]

    metrics = [
        ("rho_spearman", r"Pooled $\rho_{\mathrm{spearman}}$", "p_two_sided"),
        ("rho_pearson", r"Pooled $\rho_{\mathrm{pearson}}$", "p_two_sided"),
        ("L_align", r"Pooled $L_{\mathrm{align}}$", "p_one_sided_low"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    rng = np.random.default_rng(0)
    for ax, (key, label, ptype) in zip(axes, metrics):
        m = perm[key]
        observed = m["observed"]
        # Reconstruct an approximate Gaussian from null mean/std for display.
        null_mean = m["null"]["mean"]
        null_std = m["null"]["std"]
        samples = rng.normal(null_mean, null_std, 10000)
        ax.hist(samples, bins=50, color="#cccccc", edgecolor="white", alpha=0.85)
        ax.axvline(observed, color=COLORS["Diverse-clean"], linewidth=2.0,
                   label=f"observed = {observed:+.3f}")
        ax.axvline(m["null"]["pct_2_5"], color="black", linestyle=":", linewidth=0.8)
        ax.axvline(m["null"]["pct_97_5"], color="black", linestyle=":", linewidth=0.8,
                   label="null 95% CI")
        p = m.get(ptype)
        ax.set_title(f"{label}\np = {p:.4f}", fontsize=10)
        ax.set_xlabel(label)
        ax.set_ylabel("Count" if key == "rho_spearman" else "")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle("Block-permutation null distributions — diverse-clean run (n_pairs=25, B=10000)",
                 fontsize=11)
    fig.tight_layout()
    fp = OUT / "fig3_permutation_nulls.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}")


def fig_iter_trajectories():
    """Figure 4: per-iteration L_align and ρ_spearman trajectories (clean run)."""
    run = RUNS["Diverse-clean\n(HLE + BBH fallback)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fold_colors = ["#2563eb", "#dc2626", "#16a34a", "#ea580c"]
    for fi, fdir in enumerate(sorted(run.glob("fold*"))):
        iters = sorted(fdir.glob("iter_*"), key=lambda p: p.name)
        xs, Ls, rs = [], [], []
        for i, idir in enumerate(iters):
            mp = idir / "metrics.json"
            if not mp.exists():
                continue
            m = json.load(open(mp))
            xs.append(i)
            Ls.append(m.get("L_align"))
            rs.append(m.get("rho_align_spearman"))
        ax1.plot(xs, Ls, marker="o", color=fold_colors[fi], label=fdir.name)
        ax2.plot(xs, rs, marker="o", color=fold_colors[fi], label=fdir.name)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"$L_{\mathrm{align}}$ (lower = better)")
    ax1.set_title("Train alignment loss per iteration")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\rho_{\mathrm{spearman}}$ (train)")
    ax2.set_title("Train Spearman per iteration")
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle("Diverse-clean run — per-iteration trajectories by fold", fontsize=11)
    fig.tight_layout()
    fp = OUT / "fig4_iter_trajectories.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}")


def fig_residuals_topk():
    """Figure 5: top-10 residual pairs at baseline-static (iter_000) for the clean run,
    averaged across folds to show what the v_loop must reduce."""
    run = RUNS["Diverse-clean\n(HLE + BBH fallback)"]
    pair_residuals: dict[tuple, list] = {}
    for fdir in sorted(run.glob("fold*")):
        rp = fdir / "iter_000_baseline_static" / "residual_report.json"
        if not rp.exists():
            continue
        rows = json.load(open(rp))
        for r in rows:
            key = tuple(sorted([r["p"], r["q"]]))
            pair_residuals.setdefault(key, []).append(r["residual_abs"])
    avg = sorted(((p, np.mean(v)) for p, v in pair_residuals.items()), key=lambda t: -t[1])[:12]

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{p[0]} ↔ {p[1]}" for p, _ in avg]
    vals = [v for _, v in avg]
    bars = ax.barh(range(len(avg)), vals, color="#dc2626", edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(avg)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean residual |s_tag − r_score| across folds")
    ax.set_title("Top-12 high-residual benchmark pairs at iter_000 (clean run)\n"
                 "HLE-involving pairs dominate — frontier-hardness axis is missing")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fp = OUT / "fig5_top_residuals.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}")


def fig_pipeline_diagram():
    """Figure 0: schematic of the v3 main pipeline."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, label, color="#dbeafe"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.2))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=10, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.4))

    box(0.2, 1.5, 1.8, 1.0, "Mapper\n(OpenRouter\n9B chunk evidence)", "#fef3c7")
    box(2.4, 1.5, 1.8, 1.0, "Executer\n(self-hosted 35B,\nsingle source bench)", "#dbeafe")
    box(4.6, 1.5, 1.8, 1.0, "Maker\n(self-hosted 35B,\ncorpus tagging T)", "#dbeafe")
    box(6.8, 1.5, 1.8, 1.0, "Score-pattern\nalignment\n(L, ρ, Δ_tag)", "#e0e7ff")
    box(9.0, 1.5, 1.8, 1.0, "Improver\n(temp=0.7,\nn_samples=3)", "#fce7f3")

    for x in [2.0, 4.2, 6.4, 8.6]:
        arrow(x, 2.0, x + 0.4, 2.0)
    # loop back
    ax.annotate("", xy=(3.3, 1.45), xytext=(9.9, 0.7),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(6.5, 0.4, "next iteration — Improver rewrites I_exec; Executer regenerates V",
            color="#dc2626", fontsize=9, ha="center")

    ax.text(5.5, 3.5, "v3 main pipeline (per fold)", ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fp = OUT / "fig0_pipeline.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}")


if __name__ == "__main__":
    fig_pipeline_diagram()
    fig_perfold_rhos()
    fig_pooled_grouped()
    fig_perm_nulls()
    fig_iter_trajectories()
    fig_residuals_topk()
