#!/usr/bin/env python3
"""Compute per-run metrics (AUROC + LPIPS) for the PGD ablation grid.

For each (epsilon, steps, alpha, seed, held_out) cell:
  1. Locate adv_dataset.pkl produced by run_ablation_attacks.sh.
  2. Run raid/evaluate_detector.py on the held-out detector (LOO protocol).
     Results land in EVAL_OUTPUT_DIR mirroring the pkl tree → results.json.
  3. Compute LPIPS; write a lpips_{net}.json sidecar next to results.json.
  4. Rebuild ablation_metrics.csv by walking the EVAL_OUTPUT_DIR tree.

Both results.json and lpips_{net}.json are git-tracked (EVAL_OUTPUT_DIR defaults
to src/results/evaluate_detector/).  Cross-machine sync: commit those files on
each machine and `git pull` on the other — no pkl transfer needed.

Deps: lpips==0.1.4.  GPU defaults to cuda:0 (cuda:1 is busy).
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

# ─────────────────────── configurable path constants ────────────────────────

ADV_ROOT      = "/storageC/heddoubi/raid_revision"
CLEAN_DATASET = "/storageC/heddoubi/raid/data/data/RAID/ELSA_TEST"

# Git-tracked directory for results.json + lpips_<net>.json sidecars.
EVAL_OUTPUT_DIR = "src/results/evaluate_detector"

# CSV output (also git-tracked).
METRICS_DIR = "src/results/ablation"

# ─────────────────────────── grid constants ──────────────────────────────────
EPSILONS    = ["8/255", "16/255", "32/255"]
STEP_COUNTS = [10, 20, 30]
STEP_SIZES  = [0.01, 0.03, 0.05]
SEEDS       = [42, 234, 123]
DETECTORS   = [
    "ojha2023", "corvi2023", "cavia2024",
    "chen2024_convnext", "chen2024_clip", "koutlis2024", "wang2020",
]

CSV_COLUMNS = [
    "epsilon", "steps", "alpha", "seed", "held_out",
    "auroc", "f1", "acc",
    "lpips", "lpips_std", "lpips_net", "n_images",
    "pkl_path",
]


# ─────────────────────────── path helpers ────────────────────────────────────

def eps_str(eps: str) -> str:
    return eps.replace("/", "_")


def alpha_str(alpha: float) -> str:
    return str(alpha)


def build_pkl_path(adv_root: str, seed: int, eps: str, steps: int,
                   alpha: float, held_out: str) -> Path:
    ensemble  = [d for d in DETECTORS if d != held_out]
    list_repr = repr(ensemble)
    dirname   = f"adv_raw_{list_repr}_{eps_str(eps)}_{steps}_{alpha_str(alpha)}"
    return (
        Path(adv_root)
        / f"output_seed{seed}"
        / "ADV" / "ELSA_TEST_200"
        / dirname
        / "adv_dataset.pkl"
    )


def build_results_json_path(eval_output_dir: str, pkl: Path, model: str) -> Path:
    """Mirror evaluate_detector.py's path derivation for dataset_type='dataset'."""
    path_name = pkl.parts[1:-1]
    return Path(eval_output_dir) / Path(*path_name) / model / "results.json"


def build_lpips_sidecar_path(results_json: Path, net: str) -> Path:
    return results_json.parent / f"lpips_{net}.json"


# ─────────────────────────── path parser ─────────────────────────────────────

def parse_path_components(results_json: Path):
    """Extract (eps, steps, alpha, seed, held_out) from a results.json path."""
    parts = results_json.parts
    held_out = results_json.parent.name

    adv_dir = next((p for p in parts if p.startswith("adv_raw_")), None)
    if adv_dir is None:
        return None
    m = re.search(r'_(\d+)_255_(\d+)_([\d.]+)$', adv_dir)
    if m is None:
        return None
    eps_num_str, steps_str, alpha_str_v = m.groups()
    eps   = f"{eps_num_str}/255"
    steps = int(steps_str)
    alpha = float(alpha_str_v)

    seed_part = next((p for p in parts if p.startswith("output_seed")), None)
    if seed_part is None:
        return None
    seed = int(seed_part[len("output_seed"):])

    return eps, steps, alpha, seed, held_out


# ─────────────────────── AUROC evaluation (subprocess) ──────────────────────

def run_evaluate_detector(pkl: Path, model: str, eval_output_dir: str,
                          device: str, repo_src: str) -> tuple:
    result_path = build_results_json_path(eval_output_dir, pkl, model)
    cmd = [
        sys.executable, "raid/evaluate_detector.py",
        "--model",           model,
        "--device",          device,
        "--path_to_dataset", str(pkl),
        "--dataset_type",    "dataset",
        "--output_dir",      eval_output_dir,
        "--batch_size",      "32",
        "--num_workers",     "4",
        "--subset",          "-1",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_src}:{repo_src}/raid:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(cmd, cwd=repo_src, env=env, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"  ERROR: evaluate_detector.py failed (rc={proc.returncode})")
        print(proc.stderr[-2000:])
        return None, "error"
    if not result_path.exists():
        print(f"  ERROR: results.json not written at {result_path}")
        return None, "error"

    with open(result_path) as f:
        data = json.load(f)
    return data["metrics"]["All"], "computed"


# ───────────────────────── LPIPS computation ─────────────────────────────────

def compute_lpips(adv_data: dict, clean_images: dict, eps_num: float,
                  loss_fn, device: str, batch_size: int = 32) -> tuple:
    per_image = []
    for cls_key in sorted(clean_images.keys()):
        clean = clean_images[cls_key].to(device)
        adv   = adv_data[cls_key]["images"].to(device)
        if clean.shape != adv.shape:
            raise ValueError(f"Shape mismatch '{cls_key}': {clean.shape} vs {adv.shape}")
        max_diff = (adv - clean).abs().max().item()
        if max_diff > eps_num + 1e-4:
            raise RuntimeError(
                f"Sanity FAILED '{cls_key}': max|adv-clean|={max_diff:.6f} > "
                f"eps={eps_num:.6f}+1e-4"
            )
        clean_s = clean * 2 - 1
        adv_s   = adv   * 2 - 1
        with torch.no_grad():
            for i in range(0, clean.shape[0], batch_size):
                vals = loss_fn(clean_s[i:i+batch_size], adv_s[i:i+batch_size])
                v = vals.squeeze().cpu().float()
                per_image.extend(v.tolist() if v.dim() > 0 else [v.item()])
    t = torch.tensor(per_image)
    return t.mean().item(), t.std().item(), len(per_image)


# ───────────────────────── clean image loading ───────────────────────────────

def load_clean_images(clean_dataset: str, subset: int = 200) -> dict:
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader, Subset
    from torchvision.transforms import CenterCrop, ToTensor, Compose

    class PathDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths, self.labels, self.transform = paths, labels, transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            return self.transform(Image.open(self.paths[idx]).convert("RGB")), self.labels[idx]

    transform = Compose([CenterCrop((256, 256)), ToTensor()])
    suffixes  = {".png", ".jpg", ".jpeg"}
    result    = {}
    root      = Path(clean_dataset)
    for folder in sorted(root.iterdir()):
        if folder.name == "modern_generator":
            continue
        paths  = [str(f) for f in sorted(folder.iterdir()) if f.suffix.lower() in suffixes]
        labels = [0 if folder.name == "real" else 1] * len(paths)
        ds     = Subset(PathDataset(paths, labels, transform), list(range(subset)))
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        result[folder.name] = torch.cat([b for b, _ in loader], dim=0)
    return result


# ─────────────────────────── CSV rebuild ─────────────────────────────────────

def rebuild_csv(eval_output_dir: str, output_csv: str, lpips_net: str = "alex") -> int:
    """Walk eval_output_dir tree, pair results.json + lpips_{net}.json → CSV."""
    root   = Path(eval_output_dir)
    rows   = []
    sidecar_name = f"lpips_{lpips_net}.json"

    for rj in sorted(root.rglob("results.json")):
        parsed = parse_path_components(rj)
        if parsed is None:
            continue
        eps, steps, alpha, seed, held_out = parsed

        with open(rj) as f:
            rd = json.load(f)
        m = rd["metrics"]["All"]

        lpips_path = rj.parent / sidecar_name
        if not lpips_path.exists():
            continue
        with open(lpips_path) as f:
            ld = json.load(f)

        rows.append({
            "epsilon":   eps,
            "steps":     steps,
            "alpha":     alpha,
            "seed":      seed,
            "held_out":  held_out,
            "auroc":     m["auc"],
            "f1":        m["f1"],
            "acc":       m["acc"],
            "lpips":     ld["lpips"],
            "lpips_std": ld["lpips_std"],
            "lpips_net": ld.get("lpips_net", lpips_net),
            "n_images":  ld["n_images"],
            "pkl_path":  "",
        })

    eps_order = {e: i for i, e in enumerate(EPSILONS)}
    rows.sort(key=lambda r: (
        eps_order.get(str(r["epsilon"]), 99),
        int(r["steps"]),
        float(r["alpha"]),
        int(r["seed"]),
        str(r["held_out"]),
    ))

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out_path)
    print(f"CSV written: {out_path} ({len(rows)} rows)")
    return len(rows)


# ──────────────────────────────── main ───────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute AUROC + LPIPS for the PGD ablation grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adv_root",        default=ADV_ROOT)
    p.add_argument("--clean_dataset",   default=CLEAN_DATASET)
    p.add_argument("--eval_output_dir", default=EVAL_OUTPUT_DIR,
                   help="Git-tracked dir for results.json + lpips sidecars.")
    p.add_argument("--metrics_dir",     default=METRICS_DIR,
                   help="Dir for the merged CSV.")
    p.add_argument("--epsilons", nargs="+", default=EPSILONS)
    p.add_argument("--steps",    type=int,   nargs="+", default=STEP_COUNTS)
    p.add_argument("--alphas",   type=float, nargs="+", default=STEP_SIZES)
    p.add_argument("--seeds",    type=int,   nargs="+", default=SEEDS)
    p.add_argument("--force",            action="store_true")
    p.add_argument("--dry_run",          action="store_true")
    p.add_argument("--rebuild_csv_only", action="store_true",
                   help="Walk eval_output_dir and rebuild CSV. No GPU, no pkl access.")
    p.add_argument("--output",    default=None)
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--device",    default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve all relative paths to absolute up front.
    if not Path(args.metrics_dir).is_absolute():
        args.metrics_dir = str(Path.cwd() / args.metrics_dir)
    if not Path(args.eval_output_dir).is_absolute():
        args.eval_output_dir = str(Path.cwd() / args.eval_output_dir)

    output_csv = args.output or str(Path(args.metrics_dir) / "ablation_metrics.csv")
    if not Path(output_csv).is_absolute():
        output_csv = str(Path.cwd() / output_csv)

    if args.rebuild_csv_only:
        n = rebuild_csv(args.eval_output_dir, output_csv, args.lpips_net)
        print(f"rebuild_csv_only: {n} rows from {args.eval_output_dir}")
        return

    repo_src = str(Path(__file__).parent)

    clean_images  = None
    lpips_loss_fn = None

    n_total = n_skip_no_pkl = 0
    n_eval_computed = n_eval_cached = n_eval_error = 0
    n_lpips_computed = n_lpips_cached = n_rows_written = 0

    for eps in args.epsilons:
        eps_num = float(eval(eps))

        for steps in args.steps:
            for alpha in args.alphas:
                for seed in args.seeds:
                    for held_out in DETECTORS:
                        n_total += 1
                        pkl = build_pkl_path(args.adv_root, seed, eps, steps, alpha, held_out)
                        tag = (f"eps={eps} steps={steps} alpha={alpha} "
                               f"seed={seed} held_out={held_out}")

                        if not pkl.exists():
                            print(f"SKIP (no pkl): {tag}")
                            n_skip_no_pkl += 1
                            continue

                        result_path  = build_results_json_path(args.eval_output_dir, pkl, held_out)
                        sidecar_path = build_lpips_sidecar_path(result_path, args.lpips_net)

                        # ── AUROC ─────────────────────────────────────────────
                        if result_path.exists() and not args.force:
                            with open(result_path) as f:
                                auroc_metrics = json.load(f)["metrics"]["All"]
                            auroc_status = "auroc-cached"
                            n_eval_cached += 1
                        else:
                            if args.dry_run:
                                print(f"[dry] {tag}: would evaluate + compute LPIPS")
                                continue
                            metrics_all, status = run_evaluate_detector(
                                pkl, held_out, args.eval_output_dir, args.device, repo_src)
                            if metrics_all is None:
                                n_eval_error += 1
                                continue
                            auroc_metrics = metrics_all
                            auroc_status  = "auroc-computed"
                            n_eval_computed += 1

                        # ── LPIPS ─────────────────────────────────────────────
                        if sidecar_path.exists() and not args.force:
                            with open(sidecar_path) as f:
                                sd = json.load(f)
                            lpips_mean, lpips_std = sd["lpips"], sd["lpips_std"]
                            lpips_net_v, n_images = sd.get("lpips_net", args.lpips_net), sd["n_images"]
                            lpips_status = "lpips-cached"
                            n_lpips_cached += 1
                        else:
                            if args.dry_run:
                                continue
                            if clean_images is None:
                                print("Loading clean images…")
                                clean_images = load_clean_images(args.clean_dataset, 200)
                                print(f"  {sum(v.shape[0] for v in clean_images.values())} images loaded.")
                            if lpips_loss_fn is None:
                                import lpips as lpips_lib
                                print(f"Building LPIPS model (net={args.lpips_net})…")
                                lpips_loss_fn = lpips_lib.LPIPS(net=args.lpips_net).to(args.device)
                                lpips_loss_fn.eval()

                            import pickle
                            with open(pkl, "rb") as f:
                                adv_data = pickle.load(f)

                            lpips_mean, lpips_std, n_images = compute_lpips(
                                adv_data, clean_images, eps_num, lpips_loss_fn, args.device)
                            lpips_net_v  = args.lpips_net
                            lpips_status = "lpips-computed"
                            n_lpips_computed += 1

                            # Write LPIPS sidecar atomically.
                            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                            tmp = sidecar_path.with_suffix(".tmp")
                            with open(tmp, "w") as f:
                                json.dump({
                                    "lpips":     lpips_mean,
                                    "lpips_std": lpips_std,
                                    "lpips_net": lpips_net_v,
                                    "n_images":  n_images,
                                }, f, indent=2)
                            tmp.replace(sidecar_path)

                        n_rows_written += 1
                        print(
                            f"  OK [{n_rows_written}] {tag} | "
                            f"AUROC={auroc_metrics['auc']:.4f} "
                            f"LPIPS={lpips_mean:.4f} "
                            f"({auroc_status}, {lpips_status})",
                            flush=True,
                        )

    if not args.dry_run:
        rebuild_csv(args.eval_output_dir, output_csv, args.lpips_net)
    else:
        print(f"\n[dry_run] {n_total} cells | skip: {n_skip_no_pkl} | "
              f"processable: {n_total - n_skip_no_pkl}")

    print("\n" + "=" * 60)
    print(f"  Grid cells:        {n_total}")
    print(f"  Skipped (no pkl):  {n_skip_no_pkl}")
    print(f"  Rows written:      {n_rows_written}")
    print(f"  AUROC computed:    {n_eval_computed}  cached: {n_eval_cached}  errors: {n_eval_error}")
    print(f"  LPIPS computed:    {n_lpips_computed}  cached: {n_lpips_cached}")


if __name__ == "__main__":
    main()
