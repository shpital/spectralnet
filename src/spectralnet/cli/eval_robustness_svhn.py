"""
eval_robustness_svhn.py — SVHN Robustness Suite (blur + AWGN only)

Targeted robustness pass for the SVHN shape-check.
Only blur and AWGN are evaluated — these are the two most informative
corruptions on SVHN given the spectral line's known profile:
  blur: good (spectral filtering helps)
  AWGN: weak (68.1% drop for GELU-B n=4)

Checkpoint discovery mirrors the CIFAR-100 script layout:
  results/rmsb_svhn/<tag>_seed<N>/<timestamp>/lineage.json + best_checkpoint.pt

Usage:
  python src/spectralnet/cli/eval_robustness_svhn.py \
      --output ./results/rmsb_svhn/robustness_blur_awgn.json
"""

import argparse
import glob
import json
import math
import os
import statistics
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

SVHN_META = {
    "mean": (0.4377, 0.4438, 0.4728),
    "std":  (0.1980, 0.2010, 0.1970),
    "num_classes": 10,
}

BLUR_SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
AWGN_SNRS   = [20, 15, 10, 5, 0, -5]


class AWGNDataset(Dataset):
    def __init__(self, base_dataset, snr_db: float):
        self.base   = base_dataset
        self.snr_db = snr_db

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        sp  = img.pow(2).mean().item()
        sp  = max(sp, 1e-10)
        std = math.sqrt(sp / (10 ** (self.snr_db / 10.0)))
        return img + torch.randn_like(img) * std, label


def _clean_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_META["mean"], SVHN_META["std"]),
    ])

def _blur_transform(sigma: float):
    return transforms.Compose([
        transforms.GaussianBlur(kernel_size=5, sigma=sigma),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_META["mean"], SVHN_META["std"]),
    ])

def _svhn_test(transform, data_dir: str):
    return datasets.SVHN(root=data_dir, split="test", download=True, transform=transform)

def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def build_model(lineage: dict) -> nn.Module:
    cfg = lineage["config"]
    mc  = cfg.get("model", {})
    nc  = SVHN_META["num_classes"]
    from omegaconf import OmegaConf
    from src.spectralnet.models.spectralnet_s import SpectralNetS
    mc["num_classes"] = nc
    return SpectralNetS(OmegaConf.create(mc))


def load_checkpoint(run_dir: str, device: str):
    lp = os.path.join(run_dir, "lineage.json")
    cp = os.path.join(run_dir, "best_checkpoint.pt")
    if not (os.path.exists(lp) and os.path.exists(cp)):
        return None, None
    with open(lp) as f:
        lineage = json.load(f)
    model = build_model(lineage)
    ckpt  = torch.load(cp, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), lineage


def discover_checkpoints(base_dir: str, seeds: list[int]) -> dict:
    """Scan rmsb_svhn/<tag>_seed<N>/<timestamp>/ for checkpoints."""
    branches = {"spectral": [], "rmsb_r1": []}

    for tag, branch_key in [("spectral_rank4_sh8", "spectral"),
                            ("rmsb_r1_n4_e16", "rmsb_r1")]:
        for seed in seeds:
            seed_dir = os.path.join(base_dir, f"{tag}_seed{seed}")
            if not os.path.isdir(seed_dir):
                continue
            candidates = glob.glob(os.path.join(seed_dir, "*/lineage.json"))
            for lp in candidates:
                cp = os.path.join(os.path.dirname(lp), "best_checkpoint.pt")
                if not os.path.exists(cp):
                    continue
                with open(lp) as f:
                    lin = json.load(f)
                acc = lin.get("best_val_acc", 0.0)
                if acc < 1.0:
                    continue
                branches[branch_key].append({
                    "seed": seed,
                    "run_dir": os.path.dirname(lp),
                    "acc": acc,
                    "lineage": lin,
                })
    return branches


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += pred.eq(y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


def eval_branch(runs: list[dict], data_dir: str, batch_size: int,
                num_workers: int, device: str) -> dict:
    if not runs:
        return {"error": "no valid runs found"}

    seed_results = []
    for r in sorted(runs, key=lambda x: x["seed"]):
        seed    = r["seed"]
        run_dir = r["run_dir"]
        print(f"    seed={seed}  run={run_dir}  frozen_acc={r['acc']:.2f}%")

        model, _ = load_checkpoint(run_dir, device)
        if model is None:
            print(f"      ⚠  checkpoint missing, skip")
            continue

        sr = {"seed": seed, "frozen_acc": r["acc"], "clean": None,
              "blur": {}, "awgn": {}}

        clean_ds   = _svhn_test(_clean_transform(), data_dir)
        sr["clean"] = evaluate(model, make_loader(clean_ds, batch_size, num_workers), device)
        print(f"      clean: {sr['clean']:.2f}%")

        for sigma in BLUR_SIGMAS:
            ds  = _svhn_test(_blur_transform(sigma), data_dir)
            acc = evaluate(model, make_loader(ds, batch_size, num_workers), device)
            sr["blur"][sigma] = acc
        print(f"      blur:  " +
              "  ".join(f"σ={s}: {sr['blur'][s]:.1f}%" for s in BLUR_SIGMAS))

        clean_ds2 = _svhn_test(_clean_transform(), data_dir)
        for snr in AWGN_SNRS:
            ds  = AWGNDataset(clean_ds2, snr)
            acc = evaluate(model, make_loader(ds, batch_size, num_workers), device)
            sr["awgn"][snr] = acc
        print(f"      awgn:  " +
              "  ".join(f"{snr}dB: {sr['awgn'][snr]:.1f}%" for snr in AWGN_SNRS))

        del model
        torch.cuda.empty_cache()
        seed_results.append(sr)

    return aggregate(seed_results)


def aggregate(seed_results: list[dict]) -> dict:
    n = len(seed_results)
    if n == 0:
        return {"error": "no results"}

    out = {"n_seeds": n}
    cleans = [sr["clean"] for sr in seed_results]
    out["clean"] = {
        "mean": statistics.mean(cleans),
        "std":  statistics.stdev(cleans) if n > 1 else 0.0,
        "per_seed": cleans,
    }

    for ct, levels in [("blur", BLUR_SIGMAS), ("awgn", AWGN_SNRS)]:
        abs_acc = {}
        drops   = {}
        for lv in levels:
            lv_key = str(lv)
            vals = [sr[ct][lv] for sr in seed_results]
            abs_acc[lv_key] = {
                "mean": statistics.mean(vals),
                "std":  statistics.stdev(vals) if n > 1 else 0.0,
            }
            drop_vals = [sr["clean"] - sr[ct][lv] for sr in seed_results]
            drops[lv_key] = {
                "mean": statistics.mean(drop_vals),
                "std":  statistics.stdev(drop_vals) if n > 1 else 0.0,
            }

        all_drops = []
        for lv in levels:
            for sr in seed_results:
                all_drops.append(sr["clean"] - sr[ct][lv])
        auc = statistics.mean(all_drops) if all_drops else 0.0

        out[ct] = {
            "absolute": abs_acc,
            "drop_vs_clean": drops,
            "auc_drop": {"mean": auc},
        }

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir",    default="./results/rmsb_svhn")
    p.add_argument("--data_dir",    default="./data")
    p.add_argument("--output",      default="./results/rmsb_svhn/robustness_blur_awgn.json")
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seeds",       default="42,123,777")
    p.add_argument("--branches",    nargs="+", default=["spectral", "rmsb_r1"])
    args = p.parse_args()

    seeds  = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"  SVHN Robustness Suite (blur + AWGN)")
    print(f"  Seeds: {seeds}")
    print(f"  Branches: {args.branches}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    all_runs = discover_checkpoints(args.base_dir, seeds)
    for br in args.branches:
        runs = all_runs.get(br, [])
        print(f"  {br}: found {len(runs)} runs — seeds {sorted(r['seed'] for r in runs)}")
    print()

    results = {}
    for br in args.branches:
        runs = all_runs.get(br, [])
        if not runs:
            print(f"  ⚠  {br}: no checkpoints found, skipping")
            continue
        print(f"  ── {br} ({len(runs)} seeds) ──")
        t0 = time.time()
        results[br] = eval_branch(runs, args.data_dir, args.batch_size,
                                  args.num_workers, device)
        elapsed = time.time() - t0
        print(f"  ── {br} done in {elapsed:.0f}s ──\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")

    print(f"\n{'='*80}")
    print(f"  SVHN Robustness Summary (blur + AWGN)")
    print(f"{'='*80}")
    hdr = f"  {'Branch':<12} {'Clean':>10} {'BlurAUC':>10} {'AWGNAUC':>10} {'Blur@σ=2':>10} {'AWGN@0dB':>10}"
    print(hdr)
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for br, res in results.items():
        if "error" in res:
            print(f"  {br:<12} ERROR: {res['error']}")
            continue
        cl = res["clean"]["mean"]
        ba = res.get("blur", {}).get("auc_drop", {}).get("mean", 0)
        aa = res.get("awgn", {}).get("auc_drop", {}).get("mean", 0)
        b2 = res.get("blur", {}).get("drop_vs_clean", {}).get("2.0", {}).get("mean", 0)
        a0 = res.get("awgn", {}).get("drop_vs_clean", {}).get("0", {}).get("mean", 0)
        print(f"  {br:<12} {cl:>9.2f}% {ba:>9.2f}pp {aa:>9.2f}pp {b2:>9.2f}pp {a0:>9.2f}pp")
    print(f"{'='*80}")
    print(f"\n  Historical ref: GELU-B n=4 SVHN blur=good, AWGN=68.1% drop")
    print(f"  CIFAR-10 ref: spectral blur=28.5pp, RMSB-R1 blur=49.7pp")
    print(f"  CIFAR-100 ref: spectral blur=26.1pp, RMSB-R1 blur=42.7pp")


if __name__ == "__main__":
    main()
