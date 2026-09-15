#!/usr/bin/env python3
"""Generate gaussian noise control baselines for the same epsilon budgets.

For each epsilon, loads the same clean dataset with the exact same pipeline
as the ablation attacks (sorted folders, shuffle=False, Subset(200),
CenterCrop(256), ToTensor), adds clamped gaussian noise, and saves to the
same pkl format: {class_name: {'images': FloatTensor, 'labels': LongTensor}}.

Output path:
  <output_dir>/noise_control_<eps_num>_255/adv_dataset.pkl

Usage:
  python3 src/generate_noise_control.py --dataset /path/to/ELSA_TEST --output_dir /path/to/output
  python3 src/generate_noise_control.py --dataset /path/to/ELSA_TEST --output_dir /path/to/output --epsilons 32/255
"""

import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import CenterCrop, ToTensor, Compose

EPSILONS = ["8/255", "16/255", "32/255"]
SUBSET = 200


class PathDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB")), self.labels[idx]


def load_clean_images(dataset_path: str, subset: int) -> dict:
    transform = Compose([CenterCrop((256, 256)), ToTensor()])
    suffixes = {".png", ".jpg", ".jpeg"}
    result = {}
    root = Path(dataset_path)
    for folder in sorted(root.iterdir()):
        if folder.name == "modern_generator":
            continue
        paths = [str(f) for f in sorted(folder.iterdir()) if f.suffix.lower() in suffixes]
        labels = [0 if folder.name == "real" else 1] * len(paths)
        ds = Subset(PathDataset(paths, labels, transform), list(range(subset)))
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        imgs = torch.cat([b for b, _ in loader], dim=0)
        labs = torch.cat([l for _, l in loader], dim=0)
        result[folder.name] = {"images": imgs, "labels": labs}
    return result


def add_gaussian_noise(clean_data: dict, eps: float, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    noisy = {}
    for cls_key, entry in clean_data.items():
        images = entry["images"].clone()
        noise = torch.randn(images.shape, generator=gen) * eps
        noise = noise.clamp(-eps, eps)
        images = (images + noise).clamp(0.0, 1.0)
        noisy[cls_key] = {"images": images, "labels": entry["labels"].clone()}
    return noisy


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate gaussian noise control for epsilon budgets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="Path to clean dataset (e.g. ELSA_TEST)")
    p.add_argument("--output_dir", required=True, help="Output directory for noise control pkls")
    p.add_argument("--epsilons", nargs="+", default=EPSILONS)
    p.add_argument("--subset", type=int, default=SUBSET)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading clean images from {args.dataset} (subset={args.subset})...")
    clean_data = load_clean_images(args.dataset, args.subset)
    n_images = sum(v["images"].shape[0] for v in clean_data.values())
    print(f"  {n_images} images across {len(clean_data)} classes")

    for eps_str in args.epsilons:
        eps_num = float(eval(eps_str))
        eps_tag = eps_str.replace("/", "_")
        out_dir = Path(args.output_dir) / f"noise_control_{eps_tag}"
        out_path = out_dir / "adv_dataset.pkl"

        if out_path.exists():
            print(f"SKIP (exists): {out_path}")
            continue

        print(f"Generating noise control eps={eps_str} (eps={eps_num:.6f})...")
        noisy_data = add_gaussian_noise(clean_data, eps_num, args.seed)

        # Sanity check
        for cls_key in sorted(noisy_data.keys()):
            diff = (noisy_data[cls_key]["images"] - clean_data[cls_key]["images"]).abs().max().item()
            assert diff <= eps_num + 1e-6, f"Noise exceeds eps for {cls_key}: {diff}"

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(noisy_data, f)
        print(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    main()
