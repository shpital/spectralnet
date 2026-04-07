"""
eval_robustness_rmsb.py — RMSB Practical Metrics: Robustness Suite

Two launch modes:
  --mode smoke   Pass A: one seed (best_val_acc) per branch, blur only.
                 Fast; verifies the pipeline and corruption scales before the final run.
  --mode full    Pass B: all seeds (mean ± std), all three corruption types.

Three branches:
  spectral  → rmsb_spectral_cifar10  (5 seeds, results/runs/)
  pure_shift→ rmsb_pure_shift_cifar10 (5 seeds, results/runs/)
  rmsb_r1   → rmsb_r1_cifar10        (5 seeds, results/rmsb/)

Severity scales (stepped, identical for all models):
  blur:     σ ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
  awgn:     SNR ∈ {20, 15, 10, 5, 0, -5} dB
  contrast: factor ∈ {0.9, 0.7, 0.5, 0.3, 0.15, 0.1}

Output per branch:
  - clean accuracy (mean ± std over seeds)
  - per-severity absolute accuracy (mean ± std)
  - drop_vs_clean per severity (mean ± std)
  - AUC: mean drop across all severity levels

Usage:
  # Pass A — smoke test:
  python src/spectralnet/cli/eval_robustness_rmsb.py \\
      --mode smoke \\
      --runs_main  ./results/runs \\
      --runs_rmsb  ./results/rmsb \\
      --data_dir   ./data \\
      --output     ./results/rmsb/robustness_smoke.json

  # Pass B — full:
  python src/spectralnet/cli/eval_robustness_rmsb.py \\
      --mode full \\
      --runs_main  ./results/runs \\
      --runs_rmsb  ./results/rmsb \\
      --data_dir   ./data \\
      --output     ./results/rmsb/robustness_full.json \\
      --batch_size 256
"""

import argparse
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
import torchvision.models as tvm

# ──────────────────────────────────────────────────────────────────────────────
# Dataset constants
# ──────────────────────────────────────────────────────────────────────────────

CIFAR10_META = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std":  (0.2023, 0.1994, 0.2010),
    "num_classes": 10,
}

# Severity grids — 6 levels each, uniformly spaced from mild to severe
BLUR_SIGMAS      = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
AWGN_SNRS        = [20, 15, 10, 5, 0, -5]   # dB; lower = more noise
CONTRAST_FACTORS = [0.9, 0.7, 0.5, 0.3, 0.15, 0.1]  # lower = more degraded

# Branch definitions: (exp_name_in_lineage, runs_dir_key)
BRANCH_DEFS = {
    "spectral":   ("rmsb_spectral_cifar10",   "main"),
    "pure_shift": ("rmsb_pure_shift_cifar10", "main"),
    "rmsb_r1":    ("rmsb_r1_cifar10",         "rmsb"),
}

# ──────────────────────────────────────────────────────────────────────────────
# Corruption datasets
# ──────────────────────────────────────────────────────────────────────────────

class AWGNDataset(Dataset):
    """Wraps a normalised tensor dataset; adds AWGN in normalised space."""
    def __init__(self, base_dataset, snr_db: float):
        self.base   = base_dataset
        self.snr_db = snr_db

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        sp = img.pow(2).mean().item()
        sp = max(sp, 1e-10)
        std = math.sqrt(sp / (10 ** (self.snr_db / 10.0)))
        return img + torch.randn_like(img) * std, label


def _clean_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_META["mean"], CIFAR10_META["std"]),
    ])


def _blur_transform(sigma: float):
    return transforms.Compose([
        transforms.GaussianBlur(kernel_size=5, sigma=sigma),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_META["mean"], CIFAR10_META["std"]),
    ])


def _contrast_transform(factor: float):
    return transforms.Compose([
        transforms.Lambda(lambda img: TF.adjust_contrast(img, factor)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_META["mean"], CIFAR10_META["std"]),
    ])


def _cifar10_test(transform, data_dir: str):
    return datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def build_model(lineage: dict) -> nn.Module:
    cfg = lineage["config"]
    mc  = cfg.get("model", {})
    bt  = mc.get("baseline_type", None)
    nc  = CIFAR10_META["num_classes"]
    if bt == "resnet18":
        return tvm.resnet18(weights=None, num_classes=nc)
    if bt == "mobilenetv2":
        return tvm.mobilenet_v2(weights=None, num_classes=nc)
    if bt == "shufflenetv2":
        return tvm.shufflenet_v2_x0_5(weights=None, num_classes=nc)
    # SpectralNet family
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
    ckpt  = torch.load(cp, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), lineage


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += pred.eq(y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


def auc_drop(drops: list[float]) -> float:
    """Mean drop across severity levels (area under corruption curve, linear)."""
    return statistics.mean(drops) if drops else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Run scanning
# ──────────────────────────────────────────────────────────────────────────────

def scan_branch(runs_dir: str, exp_name: str) -> list[dict]:
    """Return list of {seed, run_dir, acc} for all valid runs of exp_name."""
    hits = []
    if not os.path.isdir(runs_dir):
        return hits
    for d in sorted(os.listdir(runs_dir)):
        lp = os.path.join(runs_dir, d, "lineage.json")
        cp = os.path.join(runs_dir, d, "best_checkpoint.pt")
        if not (os.path.exists(lp) and os.path.exists(cp)):
            continue
        with open(lp) as f:
            lin = json.load(f)
        name = lin.get("config", {}).get("name", "")
        if name != exp_name:
            continue
        acc  = lin.get("best_val_acc", 0.0)
        seed = lin.get("seed", -1)
        if acc < 1.0:   # skip crashed / empty runs
            print(f"  ⚠  skip {d} (acc={acc:.2f}, likely crashed)")
            continue
        hits.append({"seed": seed, "run_dir": os.path.join(runs_dir, d),
                     "acc": acc, "lineage": lin})
    return hits


# ──────────────────────────────────────────────────────────────────────────────
# Per-branch evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_branch(runs: list[dict], data_dir: str, batch_size: int,
                num_workers: int, mode: str, device: str) -> dict:
    """
    Evaluate one branch.
    mode='smoke' → only blur, one seed (best acc).
    mode='full'  → all three corruptions, all seeds.
    Returns dict with aggregated statistics.
    """
    if not runs:
        return {"error": "no valid runs found"}

    if mode == "smoke":
        # pick seed with highest frozen acc
        runs = [max(runs, key=lambda r: r["acc"])]
        corr_types = ["blur"]
    else:
        corr_types = ["blur", "awgn", "contrast"]

    seed_results = []
    for r in runs:
        seed   = r["seed"]
        run_dir= r["run_dir"]
        print(f"    seed={seed}  run={os.path.basename(run_dir)}  frozen_acc={r['acc']:.2f}%")

        model, _ = load_checkpoint(run_dir, device)
        if model is None:
            print(f"      ⚠  checkpoint missing, skip")
            continue

        sr = {"seed": seed, "frozen_acc": r["acc"], "clean": None,
              "blur": {}, "awgn": {}, "contrast": {}}

        # clean
        clean_ds  = _cifar10_test(_clean_transform(), data_dir)
        sr["clean"]= evaluate(model, make_loader(clean_ds, batch_size, num_workers), device)
        print(f"      clean: {sr['clean']:.2f}%")

        if "blur" in corr_types:
            for sigma in BLUR_SIGMAS:
                ds  = _cifar10_test(_blur_transform(sigma), data_dir)
                acc = evaluate(model, make_loader(ds, batch_size, num_workers), device)
                sr["blur"][sigma] = acc
            print(f"      blur:  " +
                  "  ".join(f"σ={s}: {sr['blur'][s]:.1f}%" for s in BLUR_SIGMAS))

        if "awgn" in corr_types:
            clean_ds2 = _cifar10_test(_clean_transform(), data_dir)
            base_ds   = clean_ds2
            for snr in AWGN_SNRS:
                ds  = AWGNDataset(base_ds, snr)
                acc = evaluate(model, DataLoader(ds, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers, pin_memory=True), device)
                sr["awgn"][snr] = acc
            print(f"      awgn:  " +
                  "  ".join(f"{s}dB: {sr['awgn'][s]:.1f}%" for s in AWGN_SNRS))

        if "contrast" in corr_types:
            for factor in CONTRAST_FACTORS:
                ds  = _cifar10_test(_contrast_transform(factor), data_dir)
                acc = evaluate(model, make_loader(ds, batch_size, num_workers), device)
                sr["contrast"][factor] = acc
            print(f"      contrast: " +
                  "  ".join(f"{f}: {sr['contrast'][f]:.1f}%" for f in CONTRAST_FACTORS))

        seed_results.append(sr)

    return aggregate_seeds(seed_results, corr_types)


def aggregate_seeds(seed_results: list[dict], corr_types: list[str]) -> dict:
    """Aggregate per-seed results into mean ± std + drop + AUC."""
    if not seed_results:
        return {"error": "no seed results"}

    out = {
        "n_seeds": len(seed_results),
        "seeds":   [r["seed"] for r in seed_results],
        "frozen_acc": {
            "per_seed": [r["frozen_acc"] for r in seed_results],
            "mean": statistics.mean(r["frozen_acc"] for r in seed_results),
            "std":  statistics.stdev(r["frozen_acc"] for r in seed_results)
                    if len(seed_results) > 1 else 0.0,
        },
        "clean": _agg_scalar([r["clean"] for r in seed_results]),
    }

    for ct in corr_types:
        severities = {
            "blur":     BLUR_SIGMAS,
            "awgn":     AWGN_SNRS,
            "contrast": CONTRAST_FACTORS,
        }[ct]

        ct_out = {"absolute": {}, "drop_vs_clean": {}, "auc_drop": None}
        clean_means = [r["clean"] for r in seed_results]

        for sev in severities:
            accs  = [r[ct].get(sev, None) for r in seed_results]
            accs  = [a for a in accs if a is not None]
            drops = [c - a for c, a in zip(clean_means, accs)]

            ct_out["absolute"][str(sev)]      = _agg_scalar(accs)
            ct_out["drop_vs_clean"][str(sev)] = _agg_scalar(drops)

        # AUC: mean drop across all severity levels (per seed, then aggregate)
        per_seed_aucs = []
        for r in seed_results:
            drops = [r["clean"] - r[ct].get(sev, r["clean"]) for sev in severities]
            per_seed_aucs.append(statistics.mean(drops))
        ct_out["auc_drop"] = _agg_scalar(per_seed_aucs)

        out[ct] = ct_out

    return out


def _agg_scalar(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    return {
        "per_seed": vals,
        "mean": statistics.mean(vals) if vals else None,
        "std":  statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Summary table printer
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict, mode: str):
    corr_types = ["blur"] if mode == "smoke" else ["blur", "awgn", "contrast"]

    print(f"\n{'='*90}")
    print(f"  ROBUSTNESS SUMMARY  (mode={mode})")
    print(f"{'='*90}")
    hdr = f"  {'Branch':<14} {'n':>3} {'Clean':>8}"
    for ct in corr_types:
        hdr += f"  {ct.upper()+' AUCdrop':>14}"
    for ct in corr_types:
        sev = {"blur": 2.0, "awgn": 0, "contrast": 0.3}[ct]
        hdr += f"  {ct[:4]+f'@{sev}':>10}"
    print(hdr)
    print(f"  {'-'*14} {'-'*3} {'-'*8}" + f"  {'-'*14}"*len(corr_types)
          + f"  {'-'*10}"*len(corr_types))

    for branch, res in results.items():
        if "error" in res:
            print(f"  {branch:<14} ERROR: {res['error']}")
            continue
        n     = res.get("n_seeds", "?")
        clean = res.get("clean", {}).get("mean", 0) or 0
        row   = f"  {branch:<14} {n:>3} {clean:>8.2f}%"
        for ct in corr_types:
            ct_res = res.get(ct, {})
            auc    = ct_res.get("auc_drop", {}).get("mean")
            row   += f"  {auc:>13.2f}pp" if auc is not None else f"  {'—':>13}"
        for ct in corr_types:
            ct_res = res.get(ct, {})
            sev    = {"blur": "2.0", "awgn": "0", "contrast": "0.3"}[ct]
            drop   = ct_res.get("drop_vs_clean", {}).get(sev, {}).get("mean")
            row   += f"  {drop:>9.2f}pp" if drop is not None else f"  {'—':>9}"
        print(row)
    print(f"{'='*90}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="RMSB Robustness Suite — Pass A/B")
    p.add_argument("--mode",       choices=["smoke", "full"], default="smoke",
                   help="smoke=Pass A (blur only, best seed); full=Pass B (all seeds, all corruptions)")
    p.add_argument("--runs_main",  default="./results/runs",
                   help="Directory containing rmsb_spectral and rmsb_pure_shift runs")
    p.add_argument("--runs_rmsb",  default="./results/rmsb",
                   help="Directory containing rmsb_r1 runs")
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--output",     required=True, help="Output JSON path")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--branches",   nargs="+",
                   default=["spectral", "pure_shift", "rmsb_r1"],
                   help="Which branches to evaluate")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  mode: {args.mode}")

    dir_map = {"main": args.runs_main, "rmsb": args.runs_rmsb}
    results = {}

    for branch in args.branches:
        if branch not in BRANCH_DEFS:
            print(f"Unknown branch: {branch}, skipping"); continue
        exp_name, dir_key = BRANCH_DEFS[branch]
        runs_dir = dir_map[dir_key]

        print(f"\n{'─'*60}")
        print(f"  Branch: {branch}  ({exp_name}  →  {runs_dir})")
        print(f"{'─'*60}")

        runs = scan_branch(runs_dir, exp_name)
        print(f"  Found {len(runs)} valid seeds: "
              + ", ".join(str(r["seed"]) for r in runs))

        t0 = time.time()
        results[branch] = eval_branch(
            runs, args.data_dir, args.batch_size, args.num_workers,
            args.mode, device
        )
        elapsed = time.time() - t0
        print(f"  Branch done in {elapsed:.1f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")

    print_summary(results, args.mode)


if __name__ == "__main__":
    main()
