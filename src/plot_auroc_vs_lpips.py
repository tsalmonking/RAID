#!/usr/bin/env python3
"""Plot AUROC vs LPIPS for the PGD ablation grid.

Walks RESULTS_DIR (src/results/evaluate_detector/) for results.json +
lpips_{net}.json pairs, parses (epsilon, steps, alpha, seed, held_out) from the
path, and plots one point per (epsilon, steps, alpha) triple averaged over all
available (seed, held_out) rows.

Same epsilon → same colour + dashed connector sorted by ascending LPIPS.
No GPU, no model loading.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import shutil as _shutil
if _shutil.which("latex"):
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsfonts}",
    })
else:
    # No LaTeX binary — Computer Modern mathtext gives an equivalent look.
    matplotlib.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
    })

# ────────────────────────── configurable constants ────────────────────────────

RESULTS_DIR = "src/results/evaluate_detector"
OUTPUT_DIR  = "src/results/ablation"

EPSILON_COLORS = {
    "8/255":  "#9ecae1",
    "16/255": "#4292c6",
    "32/255": "#08519c",
}
DEFAULT_COLOR  = "#888888"
EPSILON_ORDER  = ["8/255", "16/255", "32/255"]

ALL_EPSILONS    = ["8/255", "16/255", "32/255"]
ALL_STEP_COUNTS = [10, 20, 30]
ALL_STEP_SIZES  = [0.01, 0.03, 0.05]
ALL_SEEDS       = [42, 234, 123]
ALL_DETECTORS   = [
    "ojha2023", "corvi2023", "cavia2024",
    "chen2024_convnext", "chen2024_clip", "koutlis2024", "wang2020",
]
EXPECTED_ROWS = len(ALL_SEEDS) * len(ALL_DETECTORS)   # 21


# ───────────────────────── tree walking + parsing ─────────────────────────────

def parse_path_components(results_json: Path):
    """Return (eps, steps, alpha, seed, held_out) or None if path doesn't match."""
    parts    = results_json.parts
    held_out = results_json.parent.name
    adv_dir  = next((p for p in parts if p.startswith("adv_raw_")), None)
    if adv_dir is None:
        return None
    m = re.search(r'_(\d+)_255_(\d+)_([\d.]+)$', adv_dir)
    if m is None:
        return None
    eps_num, steps_str, alpha_str = m.groups()
    seed_part = next((p for p in parts if p.startswith("output_seed")), None)
    if seed_part is None:
        return None
    return (f"{eps_num}/255", int(steps_str), float(alpha_str),
            int(seed_part[len("output_seed"):]), held_out)


def load_data(results_dir: str, lpips_net: str, epsilons=None, seeds=None):
    root   = Path(results_dir)
    sidecar_name = f"lpips_{lpips_net}.json"
    rows   = []
    skipped = 0

    for rj in sorted(root.rglob("results.json")):
        parsed = parse_path_components(rj)
        if parsed is None:
            skipped += 1
            continue
        eps, steps, alpha, seed, held_out = parsed

        if epsilons and eps not in epsilons:
            continue
        if seeds and seed not in seeds:
            continue

        sidecar = rj.parent / sidecar_name
        if not sidecar.exists():
            continue

        with open(rj) as f:
            rd = json.load(f)
        with open(sidecar) as f:
            ld = json.load(f)

        rows.append({
            "epsilon": eps, "steps": steps, "alpha": alpha,
            "seed": seed, "held_out": held_out,
            "auroc": rd["metrics"]["All"]["auc"],
            "lpips": ld["lpips"],
        })

    if skipped:
        print(f"Skipped {skipped} results.json files (path didn't match pattern).")
    return rows


def aggregate(rows: list, min_runs: int = 0):
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        key = (r["epsilon"], r["steps"], r["alpha"])
        buckets[key].append(r)

    agg = []
    warned = False
    for (eps, steps, alpha), bucket in sorted(buckets.items()):
        n = len(bucket)
        if n < EXPECTED_ROWS:
            if not warned:
                print(f"WARNING: some triples have < {EXPECTED_ROWS} rows "
                      f"(7 detectors × {len(ALL_SEEDS)} seeds).")
                warned = True
            print(f"  eps={eps} steps={steps} alpha={alpha:.2f} → {n}/{EXPECTED_ROWS}")
        if min_runs > 0 and n < min_runs:
            continue
        lpips_vals = [r["lpips"] for r in bucket]
        auroc_vals = [r["auroc"] for r in bucket]
        agg.append({
            "epsilon":    eps,
            "steps":      steps,
            "alpha":      alpha,
            "lpips_mean": float(np.mean(lpips_vals)),
            "lpips_std":  float(np.std(lpips_vals)),
            "auroc_mean": float(np.mean(auroc_vals)),
            "auroc_std":  float(np.std(auroc_vals)),
            "n_rows":     n,
        })
    return agg


# ─────────────────────────────── plotting ────────────────────────────────────

def plot(agg: list, args: argparse.Namespace, output_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Print table
    print(f"\n{'epsilon':<8} {'steps':>5} {'alpha':>6}  "
          f"{'LPIPS':>8} {'±':>7}  {'AUROC':>7} {'±':>7}  {'n':>4}")
    print("-" * 62)
    for eps in EPSILON_ORDER:
        sub = [r for r in agg if r["epsilon"] == eps]
        sub.sort(key=lambda r: r["lpips_mean"])
        for row in sub:
            print(f"{row['epsilon']:<8} {row['steps']:>5} {row['alpha']:>6.2f}  "
                  f"{row['lpips_mean']:>8.4f} {row['lpips_std']:>7.4f}  "
                  f"{row['auroc_mean']:>7.4f} {row['auroc_std']:>7.4f}  "
                  f"{row['n_rows']:>4}")

    epsilons_in_data = {r["epsilon"] for r in agg}
    for eps in ALL_EPSILONS:
        if eps not in epsilons_in_data:
            print(f"WARNING: eps={eps} absent — not generated yet on this machine.")

    for eps in EPSILON_ORDER:
        sub = [r for r in agg if r["epsilon"] == eps]
        if not sub:
            continue
        sub.sort(key=lambda r: r["lpips_mean"])
        color = EPSILON_COLORS.get(eps, DEFAULT_COLOR)

        xs  = [r["lpips_mean"] for r in sub]
        ys  = [r["auroc_mean"] for r in sub]
        std = [r["auroc_std"]  for r in sub]
        ax.plot(xs, ys, linestyle="--", color=color, linewidth=1, zorder=1)
        ax.fill_between(
            xs,
            [y - 0.5 * s for y, s in zip(ys, std)],
            [y + 0.5 * s for y, s in zip(ys, std)],
            color=color, alpha=0.15, zorder=0,
        )

        for row in sub:
            if args.errorbars:
                ax.errorbar(
                    row["lpips_mean"], row["auroc_mean"],
                    xerr=row["lpips_std"], yerr=row["auroc_std"],
                    fmt="o", markersize=6.5, color=color,
                    ecolor=color, elinewidth=0.8, capsize=2, zorder=3,
                )
            else:
                ax.plot(row["lpips_mean"], row["auroc_mean"],
                        "o", markersize=6.5, color=color, zorder=3)

            if args.annotate:
                ax.annotate(
                    f"{row['steps']}/{row['alpha']:.2f}",
                    xy=(row["lpips_mean"], row["auroc_mean"]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=6, color=color, zorder=4,
                )

        ax.plot([], [], "o--", color=color, markersize=6.5, linewidth=1,
                label=rf"$\varepsilon$={eps}")

    ax.set_xlabel("LPIPS")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


# ─────────────────────────── arg parsing ─────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot AUROC vs LPIPS for the PGD ablation grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir", default=RESULTS_DIR,
                   help="Root of the git-tracked evaluate_detector results tree.")
    p.add_argument("--epsilons", nargs="+", default=None)
    p.add_argument("--seeds",    type=int, nargs="+", default=None)
    p.add_argument("--min_runs", type=int, default=0,
                   help="Drop triples with fewer rows.")
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--errorbars",  action="store_true")
    p.add_argument("--annotate",     dest="annotate", action="store_true",  default=True)
    p.add_argument("--no-annotate",  dest="annotate", action="store_false")
    p.add_argument("--output", default=str(Path(OUTPUT_DIR) / "auroc_vs_lpips"),
                   help="Output path stem (no extension; .png and .pdf written).")
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_data(args.results_dir, args.lpips_net, args.epsilons, args.seeds)
    if not rows:
        print(f"ERROR: no data found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)
    agg = aggregate(rows, args.min_runs)
    if not agg:
        print("ERROR: no triples survived aggregation.", file=sys.stderr)
        sys.exit(1)
    plot(agg, args, args.output)


if __name__ == "__main__":
    main()
