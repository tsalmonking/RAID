# Ablation metrics — RAID CVIU revision

This directory is **git-tracked** and designed to be machine-merged.

## Directory layout

```
ablation/
├── README.md              — this file
├── ablation_metrics.csv   — derived artefact: concatenation of all runs/*.json,
│                            sorted by (epsilon, steps, alpha, seed, held_out).
│                            Rebuild with: python src/compute_ablation_metrics.py --rebuild_csv_only
├── auroc_vs_lpips.png     — figure: AUROC vs LPIPS, one series per epsilon
├── auroc_vs_lpips.pdf     — vector version for LaTeX/Overleaf
└── runs/                  — one JSON per (epsilon, steps, alpha, seed, held_out) run
    └── eps<E>_steps<S>_alpha<A>_seed<SEED>_<held_out>.json
```

## Cross-machine workflow

Two machines each generate adversarial pkls for different epsilon budgets and compute
metrics locally. Only this `results/ablation/` tree is committed and synced via git.
The pkl files live under `/storageC/` and are never committed.

After a `git pull` from the other machine, rebuild the merged CSV:

```bash
python src/compute_ablation_metrics.py --rebuild_csv_only
```

This reads only `runs/*.json` — no GPU, no pkl access needed.

## Run JSON schema

Each `runs/*.json` file contains exactly the columns of `ablation_metrics.csv` for
that one run, plus metadata:

```json
{
  "epsilon": "32/255",
  "steps": 10,
  "alpha": 0.01,
  "seed": 42,
  "held_out": "ojha2023",
  "auroc": 0.723,
  "f1": 0.812,
  "acc": 0.754,
  "lpips": 0.183,
  "lpips_std": 0.042,
  "lpips_net": "alex",
  "n_images": 1000,
  "pkl_path": "/storageC/..."
}
```

## Attack grid

- Epsilons: 8/255, 16/255, 32/255
- Steps: 10, 20, 30  (PGD iterations)
- Alphas: 0.01, 0.03, 0.05  (PGD step size)
- Seeds: 42, 234, 123
- Held-out detectors: ojha2023, corvi2023, cavia2024, chen2024_convnext,
  chen2024_clip, koutlis2024, wang2020

Total: 3 × 3 × 3 × 7 × 3 = 567 runs (27 hyper-param configs × 7 LOO × 3 seeds).
