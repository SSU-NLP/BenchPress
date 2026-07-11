"""Generate v_loop result figures.

Outputs to results/figures/:
  fig1_radar_small_multiples.png — 6 benchmarks (math/code/knowledge) on small radars
  fig2_radar_overlay.png — math+code+knowledge category averages overlaid
  fig3_pooled_forest.png — 4-run pooled ρ_s + L_align vs null
  fig4_per_fold_heatmap.png — 4 runs × 4 folds, stored_test_rho_spearman
  fig5_run_d_fold1_trajectory.png — train L_align + Δtag per iter, Run D fold1
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "Baseline": ROOT / "results/part2_experiment/run_cv_20260512_142123",
    "A": ROOT / "results/part2_experiment/run_cv_20260512_151736",
    "B": ROOT / "results/part2_experiment/run_cv_20260512_152429",
    "C": ROOT / "results/part2_experiment/run_cv_20260512_153924",
}
RUN_D = ROOT / "results/part2_experiment/run_cv_20260512_205704"
OUT = ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Paul Tol qualitative palette (colorblind-safe)
TOL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]


def load_tstar(run_dir: Path, fold: int = 0):
    p = run_dir / f"fold{fold}/final/T_star.json"
    return json.load(open(p))


def load_vocab(run_dir: Path, fold: int = 0):
    p = run_dir / f"fold{fold}/final/vocab_star.json"
    return json.load(open(p))


# ---- categories ----
CATEGORIES = {
    "math": ["AIME 2024", "MATH-500", "GSM8K"],
    "code": ["HumanEval", "MBPP"],
    "knowledge": ["MMLU", "GPQA", "HLE", "SimpleQA", "SuperGPQA"],
    "reasoning": ["BBH", "ARC Challenge", "HellaSwag", "WinoGrande"],
}
CAT_COLOR = {"math": TOL[0], "code": TOL[1], "knowledge": TOL[2], "reasoning": TOL[3]}


def fig1_radar_small_multiples():
    """6 representative benchmarks on small radar plots (2x3 grid)."""
    T = load_tstar(RUNS["Baseline"])
    V = load_vocab(RUNS["Baseline"])
    tag_ids = [v["id"] for v in V]
    tag_labels = [v["abbr"] for v in V]
    angles = np.linspace(0, 2 * np.pi, len(tag_ids), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    picks = [
        ("AIME 2024", "math"),
        ("MATH-500", "math"),
        ("HumanEval", "code"),
        ("MMLU", "knowledge"),
        ("HLE", "knowledge"),
        ("BBH", "reasoning"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.06, wspace=0.25, hspace=0.35)

    for ax, (bench, cat) in zip(axes.flat, picks):
        vec = [T[bench].get(tid, 0.0) for tid in tag_ids]
        vec_closed = vec + [vec[0]]
        color = CAT_COLOR[cat]
        ax.plot(angles_closed, vec_closed, color=color, lw=2, zorder=3)
        ax.fill(angles_closed, vec_closed, color=color, alpha=0.22, zorder=2)
        ax.set_xticks(angles)
        ax.set_xticklabels(tag_labels, fontsize=9)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="grey")
        ax.set_ylim(0, 1.0)
        ax.text(
            0.5, 1.16,
            f"{bench}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color,
        )
        ax.text(
            0.5, 1.08,
            f"[{cat}]",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color="0.4",
        )
        ax.grid(True, alpha=0.4)

    out = OUT / "fig1_radar_small_multiples.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def fig2_radar_overlay():
    """One radar with math / code / knowledge / reasoning category averages."""
    T = load_tstar(RUNS["Baseline"])
    V = load_vocab(RUNS["Baseline"])
    tag_ids = [v["id"] for v in V]
    tag_labels = [v["abbr"] for v in V]
    angles = np.linspace(0, 2 * np.pi, len(tag_ids), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig = plt.figure(figsize=(9, 8.5))
    ax = fig.add_subplot(111, polar=True)
    fig.subplots_adjust(left=0.10, right=0.78, top=0.92, bottom=0.06)

    handles, labels = [], []
    for cat, members in CATEGORIES.items():
        vecs = []
        for b in members:
            if b not in T:
                continue
            vecs.append([T[b].get(tid, 0.0) for tid in tag_ids])
        if not vecs:
            continue
        mean_vec = np.mean(vecs, axis=0)
        mean_closed = mean_vec.tolist() + [mean_vec[0]]
        color = CAT_COLOR[cat]
        (ln,) = ax.plot(angles_closed, mean_closed, color=color, lw=2.6, zorder=3,
                        label=f"{cat} (n={len(vecs)})", marker="o", markersize=6)
        ax.fill(angles_closed, mean_closed, color=color, alpha=0.06, zorder=2)
        handles.append(ln)
        labels.append(f"{cat} (n={len(vecs)})")

    ax.set_xticks(angles)
    ax.set_xticklabels(tag_labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8, color="grey")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.45)

    ax.legend(
        handles, labels,
        loc="center left", bbox_to_anchor=(1.18, 0.5),
        fontsize=10, framealpha=0.95,
        title="benchmark family", title_fontsize=10,
    )

    out = OUT / "fig2_radar_overlay.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def fig3_pooled_forest():
    """4-run pooled ρ_spearman + L_align observed vs null 95% CI."""
    rows = []
    for run_name, run_dir in RUNS.items():
        p = json.load(open(run_dir / "agg/permutation_test.json"))
        pool = p["pooled"]
        rows.append({
            "run": run_name,
            "rho_s_obs": pool["rho_spearman"]["observed"],
            "rho_s_null_lo": pool["rho_spearman"]["null"]["pct_2_5"],
            "rho_s_null_hi": pool["rho_spearman"]["null"]["pct_97_5"],
            "rho_s_p": pool["rho_spearman"]["p_two_sided"],
            "L_obs": pool["L_align"]["observed"],
            "L_null_lo": pool["L_align"]["null"]["pct_2_5"],
            "L_null_hi": pool["L_align"]["null"]["pct_97_5"],
            "L_p": pool["L_align"]["p_one_sided_low"],
        })

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.86, bottom=0.18, wspace=0.32)

    y = np.arange(len(rows))[::-1]
    names = [r["run"] for r in rows]

    # Left: ρ_spearman
    ax = axes[0]
    for i, r in enumerate(rows):
        yp = y[i]
        ax.plot([r["rho_s_null_lo"], r["rho_s_null_hi"]], [yp, yp],
                color="#bbbbbb", lw=8, solid_capstyle="butt", zorder=1, alpha=0.9)
        ax.scatter([r["rho_s_obs"]], [yp], marker="D", s=120,
                   color=TOL[0], edgecolor="black", lw=1, zorder=3)
        ax.annotate(
            f"  ρ_s={r['rho_s_obs']:+.3f}  p={r['rho_s_p']:.4f}",
            (r["rho_s_obs"], yp), xytext=(10, 0), textcoords="offset points",
            fontsize=9.5, va="center",
        )
    ax.axvline(0, color="grey", lw=0.8, ls=":", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel(r"pooled $\rho_{spearman}$ (observed = diamond, null 95% = grey bar)", fontsize=10)
    ax.set_xlim(-0.55, 0.95)
    ax.grid(axis="x", alpha=0.3)

    # Right: L_align
    ax = axes[1]
    for i, r in enumerate(rows):
        yp = y[i]
        ax.plot([r["L_null_lo"], r["L_null_hi"]], [yp, yp],
                color="#bbbbbb", lw=8, solid_capstyle="butt", zorder=1, alpha=0.9)
        ax.scatter([r["L_obs"]], [yp], marker="D", s=120,
                   color=TOL[1], edgecolor="black", lw=1, zorder=3)
        ax.annotate(
            f"  L={r['L_obs']:.3f}  p={r['L_p']:.4f}",
            (r["L_obs"], yp), xytext=(10, 0), textcoords="offset points",
            fontsize=9.5, va="center",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel(r"pooled $L_{align}$ (observed = diamond, null 95% = grey bar; lower is better)", fontsize=10)
    ax.set_xlim(0.05, 0.36)
    ax.grid(axis="x", alpha=0.3)

    out = OUT / "fig3_pooled_forest.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def fig4_per_fold_heatmap():
    """4 runs × 4 folds matrix of held-test ρ_spearman (stored_test_rho_spearman)."""
    runs = list(RUNS.keys())
    M = np.zeros((len(runs), 4))
    for i, name in enumerate(runs):
        p = json.load(open(RUNS[name] / "agg/permutation_test.json"))
        for fold_obj in p["per_fold"]:
            f = int(fold_obj["fold_dir"].rsplit("fold", 1)[-1])
            M[i, f] = fold_obj["permutation"]["stored_test_rho_spearman"]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    fig.subplots_adjust(left=0.10, right=0.92, top=0.84, bottom=0.16)

    vmax = 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(4))
    ax.set_xticklabels([f"fold{k}" for k in range(4)], fontsize=11)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(runs, fontsize=11)
    ax.set_xlabel("CV fold (held-test split)", fontsize=10)
    ax.set_ylabel("run", fontsize=10)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt_color = "white" if abs(v) > 0.55 else "black"
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=11, color=txt_color,
                    fontweight="bold" if abs(v) > 0.4 else "normal")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"held-test $\rho_{spearman}$", fontsize=10)

    out = OUT / "fig4_per_fold_heatmap.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def fig5_run_d_fold1_trajectory():
    """Run D fold1: train L_align + train Δ_tag per iter, mark cascade onset."""
    base = RUN_D / "fold1"
    iters, train_L, train_delta_tag, accepted = [], [], [], []
    for it_dir in sorted(os.listdir(base)):
        mp = base / it_dir / "metrics.json"
        ip = base / it_dir / "improver_response.json"
        if not mp.is_file():
            continue
        m = json.load(open(mp))
        if it_dir == "iter_000_baseline_static":
            label = "static"
        elif it_dir == "iter_000_baseline_random":
            label = "random"
        elif it_dir.startswith("iter_"):
            label = it_dir[len("iter_"):]
        else:
            label = it_dir
        iters.append(label)
        train_L.append(m.get("L_align"))
        train_delta_tag.append(m.get("delta_tag"))
        if ip.is_file():
            try:
                ipd = json.load(open(ip))
                accepted.append(ipd.get("accepted"))
            except Exception:
                accepted.append(None)
        else:
            accepted.append(None)

    # Skip random baseline for cleaner story; keep static + iter_001..N
    show_idx = [i for i, n in enumerate(iters) if n != "random"]
    iters = [iters[i] for i in show_idx]
    train_L = [train_L[i] for i in show_idx]
    train_delta_tag = [train_delta_tag[i] for i in show_idx]
    accepted = [accepted[i] for i in show_idx]

    x = np.arange(len(iters))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.88, top=0.90, bottom=0.12, hspace=0.18)

    ax = axes[0]
    ax.plot(x, train_L, marker="o", color=TOL[0], lw=2, markersize=8, zorder=3)
    ax.axhline(train_L[0], ls=":", color="grey", lw=1, zorder=1)
    ax.annotate(
        f"static seed = {train_L[0]:.3f}",
        (x[0], train_L[0]), xytext=(8, -14),
        textcoords="offset points", fontsize=9, color="grey",
    )
    # Highlight iter_003 — best train L before cascade
    best_idx = int(np.argmin([v for v in train_L if v is not None]))
    ax.scatter([x[best_idx]], [train_L[best_idx]],
               marker="*", s=320, color=TOL[3], edgecolor="black", lw=1.2, zorder=4)
    ax.annotate(
        f"best: train L={train_L[best_idx]:.3f}\n(-{(1-train_L[best_idx]/train_L[0])*100:.0f}% vs static)",
        (x[best_idx], train_L[best_idx]),
        xytext=(15, -10), textcoords="offset points",
        fontsize=9.5, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff4d4", edgecolor="#cc9000"),
    )
    ax.set_ylabel(r"train $L_{align}$", fontsize=11)
    ax.set_ylim(0, max([v for v in train_L if v is not None]) * 1.25)
    ax.grid(alpha=0.3)

    ax = axes[1]
    colors = [TOL[2] if (v is not None and v > -0.10) else TOL[1] for v in train_delta_tag]
    ax.bar(x, train_delta_tag, color=colors, edgecolor="black", lw=0.8, zorder=3)
    ax.axhline(0, color="black", lw=0.8, zorder=2)
    ax.axhline(-0.10, ls="--", color=TOL[1], lw=1.2, zorder=2,
               label=r"$\Delta_{tag}$ gate = -0.10")
    ax.set_ylabel(r"train $\Delta_{tag}$", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3, axis="y")

    # Detect cascade onset: first iter whose (L, Δtag) duplicates an earlier iter
    # → confirms K2.6 empty cascade caused vocab fallback to a prior iter.
    cascade_start = None
    seen = {}
    for i, (name, L, dt) in enumerate(zip(iters, train_L, train_delta_tag)):
        if name in ("static", "random") or L is None:
            continue
        key = (round(L, 4), round(dt, 4))
        if key in seen:
            cascade_start = i
            break
        seen[key] = i
    if cascade_start is not None:
        for a in axes:
            a.axvspan(cascade_start - 0.5, len(iters) - 0.5,
                      color="#ffd5d5", alpha=0.45, zorder=0)
        axes[0].text(
            cascade_start + (len(iters) - cascade_start - 1) / 2,
            axes[0].get_ylim()[1] * 0.92,
            "K2.6 gateway empty-response cascade\n(API returned '{}')",
            ha="center", va="top", fontsize=9.5, color="#a33333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#a33333", alpha=0.9),
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [n if n in ("static", "random") else f"iter_{n}" for n in iters],
        rotation=30, ha="right", fontsize=9,
    )

    out = OUT / "fig5_run_d_fold1_trajectory.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    fig1_radar_small_multiples()
    fig2_radar_overlay()
    fig3_pooled_forest()
    fig4_per_fold_heatmap()
    fig5_run_d_fold1_trajectory()
