"""
eval_robustness_cifar100.py — CIFAR-100 Robustness Suite

Evaluates spectral-hybrid and RMSB-R1 branches on CIFAR-100 with
blur / AWGN / contrast corruptions across all frozen seeds.

Checkpoint discovery handles the mixed layout:
  seed=42 (Phase 1): saved under results/runs/ and results/rmsb/
  seeds 123/777/1234/2024 (Phase 2/3): saved under results/rmsb_cifar100/<tag>_seed<N>/<timestamp>/

Usage:
  python src/spectralnet/cli/eval_robustness_cifar100.py \
      --output ./results/rmsb_cifar100/robustness_full.json
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

CIFAR100_META = {
    "mean": (0.5071, 0.4867, 0.4408),
    "std":  (0.2675, 0.2565, 0.2761),
    "num_classes": 100,
}

BLUR_SIGMAS      = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
AWGN_SNRS        = [20, 15, 10, 5, 0, -5]
CONTRAST_FACTORS = [0.9, 0.7, 0.5, 0.3, 0.15, 0.1]


# ── Corruption datasets ──────────────────────────────────────────────────────

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
        transforms.Normalize(CIFAR100_META["mean"], CIFAR100_META["std"]),
    ])

def _blur_transform(sigma: float):
    return transforms.Compose([
        transforms.GaussianBlur(kernel_size=5, sigma=sigma),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_META["mean"], CIFAR100_META["std"]),
    ])

def _contrast_transform(factor: float):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.adjust_contrast(img, factor)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_META["mean"], CIFAR100_META["std"]),
    ])

def _cifar100_test(transform, data_dir: str):
    return datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ── Model loading ────────────────────────────────────────────────────────────

def build_model(lineage: dict) -> nn.Module:
    cfg = lineage["config"]
    mc  = cfg.get("model", {})
    nc  = CIFAR100_META["num_classes"]

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


# ── Checkpoint discovery ─────────────────────────────────────────────────────

def discover_checkpoints(base_dir: str, runs_main: str, runs_rmsb: str,
                         seeds: list[int]) -> dict:
    """
    Returns {branch: [{seed, run_dir, acc}, ...]} for both branches.
    Handles the mixed layout where seed=42 is in old dirs.
    """
    branches = {
        "spectral": {"exp_name": "rmsb_spectral_cifar100", "runs": []},
        "rmsb_r1":  {"exp_name": "rmsb_r1_cifar100",       "runs": []},
    }

    found_seeds = {"spectral": set(), "rmsb_r1": set()}

    # 1) Scan rmsb_cifar100 base dir (Phase 2/3 runs with timestamp subdirs)
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
                branches[branch_key]["runs"].append({
                    "seed": seed,
                    "run_dir": os.path.dirname(lp),
                    "acc": acc,
                    "lineage": lin,
                })
                found_seeds[branch_key].add(seed)

    # 2) For missing seeds, scan old dirs (runs_main for spectral, runs_rmsb for rmsb_r1)
    for branch_key, scan_dir, exp_name in [
        ("spectral", runs_main, "rmsb_spectral_cifar100"),
        ("rmsb_r1",  runs_rmsb, "rmsb_r1_cifar100"),
    ]:
        missing = set(seeds) - found_seeds[branch_key]
        if not missing or not os.path.isdir(scan_dir):
            continue
        for d in sorted(os.listdir(scan_dir)):
            lp = os.path.join(scan_dir, d, "lineage.json")
            cp = os.path.join(scan_dir, d, "best_checkpoint.pt")
            if not (os.path.exists(lp) and os.path.exists(cp)):
                continue
            with open(lp) as f:
                lin = json.load(f)
            name = lin.get("config", {}).get("name", "")
            ds   = lin.get("config", {}).get("dataset", {}).get("name", "")
            seed = lin.get("seed", -1)
            acc  = lin.get("best_val_acc", 0.0)
            if name != exp_name or ds != "cifar100" or seed not in missing or acc < 1.0:
                continue
            if seed in found_seeds[branch_key]:
                continue
            branches[branch_key]["runs"].append({
                "seed": seed,
                "run_dir": os.path.join(scan_dir, d),
                "acc": acc,
                "lineage": lin,
            })
            found_seeds[branch_key].add(seed)

    return {k: v["runs"] for k, v in branches.items()}


# ── Evaluation ───────────────────────────────────────────────────────────────

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

    corr_types = ["blur", "awgn", "contrast"]
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
              "blur": {}, "awgn": {}, "contrast": {}}

        clean_ds   = _cifar100_test(_clean_transform(), data_dir)
        sr["clean"] = evaluate(model, make_loader(clean_ds, batch_size, num_workers), device)
        print(f"      clean: {sr['clean']:.2f}%")

        for sigma in BLUR_SIGMAS:
            ds  = _cifar100_test(_blur_transform(sigma), data_dir)
            acc = evaluate(model, make_loader(ds, batch_size, num_workers), device)
            sr["blur"][sigma] = acc
        print(f"      blur:  " +
              "  ".join(f"σ={s}: {sr['blur'][s]:.1f}%" for s in BLUR_SIGMAS))

        clean_ds2 = _cifar100_test(_clean_transform(), data_dir)
        for snr in AWGN_SNRS:
            ds  = AWGNDataset(clean_ds2, snr)
            acc = evaluate(model, make_loader(ds, batch_size, num_workers), device)
            sr["awgn"][snr] = acc
        print(f"      awgn:  " +
              "  ".join(f"{snr}dB: {sr['awgn'][snr]:.1f}%" for snr in AWGN_SNRS))

        for factor in CONTRAST_FACTORS:
            ds  = _cifar100_test(_contrast_transform(factor), data_dir)
            acc = evaluate(model, make_loader(ds, batch_size, num_workers), device)
            sr["contrast"][factor] = acc
        print(f"      contrast: " +
              "  ".join(f"{f}: {sr['contrast'][f]:.1f}%" for f in CONTRAST_FACTORS))

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

    for ct, levels in [("blur", BLUR_SIGMAS), ("awgn", AWGN_SNRS),
                       ("contrast", CONTRAST_FACTORS)]:
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir",   default="./results/rmsb_cifar100")
    p.add_argument("--runs_main",  default="./results/runs")
    p.add_argument("--runs_rmsb",  default="./results/rmsb")
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--output",     default="./results/rmsb_cifar100/robustness_full.json")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--seeds",      default="42,123,777,1234,2024")
    p.add_argument("--branches",   nargs="+", default=["spectral", "rmsb_r1"])
    args = p.parse_args()

    seeds  = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"  CIFAR-100 Robustness Suite")
    print(f"  Seeds: {seeds}")
    print(f"  Branches: {args.branches}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    all_runs = discover_checkpoints(args.base_dir, args.runs_main, args.runs_rmsb, seeds)

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

    # Print summary
    print(f"\n{'='*90}")
    print(f"  CIFAR-100 Robustness Summary")
    print(f"{'='*90}")
    hdr = f"  {'Branch':<12} {'Clean':>10} {'BlurAUC':>10} {'AWGNAUC':>10} {'ContAUC':>10} {'Blur@σ=2':>10} {'AWGN@0dB':>10}"
    print(hdr)
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for br, res in results.items():
        if "error" in res:
            print(f"  {br:<12} ERROR: {res['error']}")
            continue
        cl = res["clean"]["mean"]
        ba = res.get("blur", {}).get("auc_drop", {}).get("mean", 0)
        aa = res.get("awgn", {}).get("auc_drop", {}).get("mean", 0)
        ca = res.get("contrast", {}).get("auc_drop", {}).get("mean", 0)
        b2 = res.get("blur", {}).get("drop_vs_clean", {}).get("2.0", {}).get("mean", 0)
        a0 = res.get("awgn", {}).get("drop_vs_clean", {}).get("0", {}).get("mean", 0)
        print(f"  {br:<12} {cl:>9.2f}% {ba:>9.2f}pp {aa:>9.2f}pp {ca:>9.2f}pp {b2:>9.2f}pp {a0:>9.2f}pp")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
