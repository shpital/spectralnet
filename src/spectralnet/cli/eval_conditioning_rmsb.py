"""
eval_conditioning_rmsb.py — RMSB Practical Metrics: Conditioning Analysis

Three sets of metrics:

1. Spectral W analysis (spectral branch only)
   - singular-value spread of the spectral operator W = D + UV*
   - per-frequency condition number (sigma_max / sigma_min)
   - global condition number of W as a C²-matrix per spatial-freq bin
   - UV* contribution norm vs diagonal norm

2. Resolvent aggregation analysis (all branches)
   - effective lambda = softplus(raw_lambda)
   - weight vector w_k = exp(-λ·k·τ)·τ (normalised)
   - weight decay ratio w_0 / w_last
   - "resolvent norm": sum of weights (= approximate L1 mass)
   - entropy of weight distribution

3. Block norm proxy / feature amplification (all branches)
   - ||x_out|| / ||x_in|| via forward hooks on each evolution block
   - per-block mean ± std over a batch of 512 test images
   - norm ratio > 1.0 → amplification, < 1.0 → contraction
   - "conditioning proxy": std(||x||) over time (depth) — low = stable

Usage:
  python src/spectralnet/cli/eval_conditioning_rmsb.py \\
      --runs_main  ./results/runs \\
      --runs_rmsb  ./results/rmsb \\
      --data_dir   ./data \\
      --output     ./results/rmsb/conditioning.json
"""

import argparse
import json
import math
import os
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as tvm

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BRANCH_DEFS = {
    "spectral":   ("rmsb_spectral_cifar10",   "main"),
    "pure_shift": ("rmsb_pure_shift_cifar10", "main"),
    "rmsb_r1":    ("rmsb_r1_cifar10",         "rmsb"),
}
CIFAR10_META  = {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)}
N_PROBE_IMGS  = 512    # images for block norm proxy
BATCH_SIZE    = 128


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def build_model(lineage: dict) -> nn.Module:
    cfg = lineage["config"]
    mc  = cfg.get("model", {})
    bt  = mc.get("baseline_type", None)
    nc  = 10
    if bt == "resnet18":    return tvm.resnet18(weights=None, num_classes=nc)
    if bt == "mobilenetv2": return tvm.mobilenet_v2(weights=None, num_classes=nc)
    if bt == "shufflenetv2":return tvm.shufflenet_v2_x0_5(weights=None, num_classes=nc)
    from omegaconf import OmegaConf
    from src.spectralnet.models.spectralnet_s import SpectralNetS
    mc["num_classes"] = nc
    return SpectralNetS(OmegaConf.create(mc))


def load_best_checkpoint(runs_dir: str, exp_name: str, device: str):
    best = None
    for d in sorted(os.listdir(runs_dir)):
        lp = os.path.join(runs_dir, d, "lineage.json")
        cp = os.path.join(runs_dir, d, "best_checkpoint.pt")
        if not (os.path.exists(lp) and os.path.exists(cp)):
            continue
        with open(lp) as f:
            lin = json.load(f)
        if lin.get("config", {}).get("name", "") != exp_name:
            continue
        acc = lin.get("best_val_acc", 0.0)
        if acc < 1.0:
            continue
        if best is None or acc > best[0]:
            best = (acc, os.path.join(runs_dir, d), lin)
    if best is None:
        return None, None
    _, run_dir, lineage = best
    model = build_model(lineage)
    ckpt  = torch.load(os.path.join(run_dir, "best_checkpoint.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), lineage


# ──────────────────────────────────────────────────────────────────────────────
# 1. Spectral W conditioning
# ──────────────────────────────────────────────────────────────────────────────

def analyze_spectral_w(model: nn.Module) -> dict:
    """Analyse W = D + UV* for the spectral branch.

    W resides in model.evolution_step (SpectralRemizovLayer).
    At each spatial-frequency bin (m, n) W is a (C_out, C_in) complex matrix.
    We compute SVD per (m,n) bin and aggregate statistics.
    """
    from src.spectralnet.core.layers.spectral_remizov_layer import SpectralRemizovLayer
    layer = getattr(model, "evolution_step", None)
    if not isinstance(layer, SpectralRemizovLayer):
        return {"note": "evolution_step is not SpectralRemizovLayer — spectral W analysis skipped"}

    with torch.no_grad():
        # Build W = D + UV* in complex form
        w_c = torch.complex(layer.w_real, layer.w_imag)   # (C, C, M, M)
        C, _, M, _ = w_c.shape

        if hasattr(layer, "u_real"):
            u_c = torch.complex(layer.u_real, layer.u_imag)  # (C, rank, M, M)
            v_c = torch.complex(layer.v_real, layer.v_imag)  # (C, rank, M, M)
            # UV* contribution: for each (m,n) → (C, rank) @ (C, rank).conj().T = (C, C)
            # Using einsum over all (m,n) bins
            uv_star = torch.einsum("irMN,jrMN->ijMN", u_c, v_c.conj())  # (C, C, M, M)
            w_full  = w_c + uv_star
            uv_norm = uv_star.abs().pow(2).mean().sqrt().item()
            d_norm  = w_c.abs().pow(2).mean().sqrt().item()
            rank    = layer.rank
        else:
            w_full = w_c
            uv_norm, d_norm, rank = 0.0, w_c.abs().pow(2).mean().sqrt().item(), 0

        # Per-(m,n) bin: SVD → singular values
        # w_full: (C_out, C_in, M, M) → reshape to (M*M, C_out, C_in)
        W_bins = w_full.permute(2, 3, 0, 1).reshape(M * M, C, C)  # (M², C, C)
        sv_list = []
        cond_list = []
        for i in range(M * M):
            try:
                sv = torch.linalg.svdvals(W_bins[i])  # (C,) real
                sv_list.append(sv.cpu())
                cond = (sv.max() / (sv.min() + 1e-12)).item()
                cond_list.append(cond)
            except Exception:
                continue

        all_sv = torch.stack(sv_list)   # (M², C)
        sv_max  = all_sv.max().item()
        sv_min  = all_sv.min().item()
        sv_mean = all_sv.mean().item()
        sv_std  = all_sv.std().item()
        cond_global = sv_max / (sv_min + 1e-12)
        cond_median = statistics.median(cond_list) if cond_list else None
        cond_p95    = sorted(cond_list)[int(0.95 * len(cond_list))] if cond_list else None

    return {
        "rank":          rank,
        "modes":         M,
        "channels":      C,
        "sv_max":        sv_max,
        "sv_min":        sv_min,
        "sv_mean":       sv_mean,
        "sv_std":        sv_std,
        "cond_global":   cond_global,
        "cond_median":   cond_median,
        "cond_p95":      cond_p95,
        "d_norm_rms":    d_norm,
        "uv_norm_rms":   uv_norm,
        "uv_vs_d_ratio": (uv_norm / (d_norm + 1e-12)) if rank > 0 else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Resolvent aggregation analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_resolvent(model: nn.Module) -> dict:
    """Analyse the resolvent aggregation weights."""
    agg = getattr(model, "resolvent_agg", None)
    if agg is None:
        return {"note": "no resolvent_agg found"}

    with torch.no_grad():
        lam_raw = agg.raw_lambda.item()
        lam     = F.softplus(agg.raw_lambda).item()
        tau     = agg.tau
        n       = agg.n_steps

        device  = agg.raw_lambda.device
        weights = agg.get_weights(n + 1, device).cpu()

        w_list = weights.tolist()
        w_norm  = sum(w_list)
        entropy = -sum((w / (w_norm + 1e-12)) * math.log(w / (w_norm + 1e-12) + 1e-15)
                       for w in w_list if w > 0)
        decay_ratio = w_list[0] / (w_list[-1] + 1e-12)

    return {
        "raw_lambda":        lam_raw,
        "effective_lambda":  lam,
        "tau":               tau,
        "n_steps":           n,
        "weight_vector":     [round(w, 6) for w in w_list],
        "weight_sum":        w_norm,
        "weight_entropy":    entropy,
        "weight_decay_ratio": decay_ratio,    # w_0 / w_last; >1 = decaying (expected)
        "note_stability": (
            "stable: λ > 0, strong decay"  if lam > 0.5 and decay_ratio > 5 else
            "moderate decay"               if decay_ratio > 2 else
            "⚠ weak decay — check λ value"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Block norm proxy (feature amplification)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_block_norms(model: nn.Module, data_dir: str, device: str) -> dict:
    """Measure ||x_out|| / ||x_in|| per evolution block using forward hooks.

    Works for all layer_type variants:
      spectral  → single SpectralRemizovLayer registered as model.evolution_step
      shift     → single RemizovShiftLayer registered as model.evolution_step
      shift_rich→ nn.ModuleList model.evolution_steps (ShiftMixerBlock × N)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_META["mean"], CIFAR10_META["std"]),
    ])
    ds = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)

    # Identify evolution blocks to hook
    blocks = []
    if hasattr(model, "evolution_steps"):   # shift_rich
        blocks = list(model.evolution_steps)
        block_names = [f"ShiftMixerBlock[{i}]" for i in range(len(blocks))]
    elif hasattr(model, "evolution_step"):  # spectral / shift
        blocks = [model.evolution_step]
        lt = model.layer_type
        block_names = [f"evolution_step({lt})"]
    else:
        return {"note": "No evolution blocks found in model"}

    # Storage for norms: list[block_idx] → list of (in_norm, out_norm)
    norms_in  = [[] for _ in blocks]
    norms_out = [[] for _ in blocks]
    handles   = []

    def make_hook(idx):
        def hook(module, inp, out):
            x_in  = inp[0].detach()
            x_out = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
            ni = x_in.pow(2).sum(dim=(1, 2, 3)).sqrt()   # (B,)
            no = x_out.pow(2).sum(dim=(1, 2, 3)).sqrt()  # (B,)
            norms_in[idx].extend(ni.cpu().tolist())
            norms_out[idx].extend(no.cpu().tolist())
        return hook

    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))

    # Run N_PROBE_IMGS through the model
    n_done = 0
    with torch.no_grad():
        for x, _ in loader:
            if n_done >= N_PROBE_IMGS:
                break
            model(x.to(device))
            n_done += x.size(0)

    for h in handles:
        h.remove()

    results = {}
    all_ratios = []
    for i, name in enumerate(block_names):
        ni = norms_in[i]
        no = norms_out[i]
        if not ni:
            results[name] = {"note": "no data"}
            continue
        ratios = [o / (n + 1e-12) for n, o in zip(ni, no)]
        all_ratios.append(statistics.mean(ratios))
        results[name] = {
            "mean_in_norm":  statistics.mean(ni),
            "mean_out_norm": statistics.mean(no),
            "mean_ratio":    statistics.mean(ratios),
            "std_ratio":     statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
            "p95_ratio":     sorted(ratios)[int(0.95 * len(ratios))],
            "interpretation": (
                "amplifying"  if statistics.mean(ratios) > 1.05 else
                "contracting" if statistics.mean(ratios) < 0.95 else
                "near-isometric"
            ),
        }

    # Depth conditioning proxy: std of mean ratio across blocks
    depth_std = statistics.stdev(all_ratios) if len(all_ratios) > 1 else 0.0
    return {
        "n_probe_imgs":   n_done,
        "n_blocks":       len(blocks),
        "blocks":         results,
        "depth_norm_std": depth_std,
        "depth_note": (
            "stable depth" if depth_std < 0.1 else
            "moderate variation across blocks" if depth_std < 0.3 else
            "⚠ high norm variation — check training stability"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="RMSB Conditioning Analysis")
    p.add_argument("--runs_main",  default="./results/runs")
    p.add_argument("--runs_rmsb",  default="./results/rmsb")
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--output",     required=True)
    p.add_argument("--branches",   nargs="+",
                   default=["spectral", "pure_shift", "rmsb_r1"])
    args = p.parse_args()

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    dir_map = {"main": args.runs_main, "rmsb": args.runs_rmsb}
    results = {}

    for branch in args.branches:
        if branch not in BRANCH_DEFS:
            print(f"Unknown branch: {branch}"); continue
        exp_name, dir_key = BRANCH_DEFS[branch]
        runs_dir = dir_map[dir_key]

        print(f"\n{'─'*60}")
        print(f"  Branch: {branch}  ({exp_name})")
        print(f"{'─'*60}")

        model, lineage = load_best_checkpoint(runs_dir, exp_name, device)
        if model is None:
            results[branch] = {"error": "no checkpoint found"}
            continue

        print(f"  Loaded seed={lineage.get('seed','?')}, "
              f"frozen_acc={lineage.get('best_val_acc',0):.2f}%")

        br = {
            "seed":        lineage.get("seed"),
            "frozen_acc":  lineage.get("best_val_acc"),
            "layer_type":  lineage.get("config", {}).get("model", {})
                               .get("evolution", {}).get("layer_type", "?"),
        }

        print("  Resolvent analysis...")
        br["resolvent"] = analyze_resolvent(model)
        r = br["resolvent"]
        print(f"    λ_eff={r.get('effective_lambda',0):.4f}  "
              f"decay_ratio={r.get('weight_decay_ratio',0):.2f}  "
              f"entropy={r.get('weight_entropy',0):.3f}  "
              f"→ {r.get('note_stability','')}")

        if branch == "spectral":
            print("  Spectral W analysis...")
            br["spectral_w"] = analyze_spectral_w(model)
            sw = br["spectral_w"]
            print(f"    cond_global={sw.get('cond_global',0):.2f}  "
                  f"cond_median={sw.get('cond_median',0):.2f}  "
                  f"UV/D_ratio={sw.get('uv_vs_d_ratio',0):.4f}")

        print("  Block norm proxy...")
        br["block_norms"] = analyze_block_norms(model, args.data_dir, device)
        bn = br["block_norms"]
        print(f"    n_blocks={bn.get('n_blocks',0)}  "
              f"depth_norm_std={bn.get('depth_norm_std',0):.4f}  "
              f"→ {bn.get('depth_note','')}")
        for name, binfo in bn.get("blocks", {}).items():
            if "mean_ratio" in binfo:
                print(f"      {name}: ratio={binfo['mean_ratio']:.4f} "
                      f"±{binfo['std_ratio']:.4f}  ({binfo['interpretation']})")

        results[branch] = br

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")

    # Compact summary
    print(f"\n{'='*80}")
    print("  CONDITIONING SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Branch':<14} {'lt':<12} {'λ_eff':>7} {'decay_r':>8} "
          f"{'depth_std':>10} {'cond_med':>10} {'UV/D':>7}")
    print(f"  {'-'*14} {'-'*12} {'-'*7} {'-'*8} {'-'*10} {'-'*10} {'-'*7}")
    for branch, r in results.items():
        if "error" in r:
            print(f"  {branch:<14} ERROR"); continue
        lt   = r.get("layer_type", "?")
        rv   = r.get("resolvent", {})
        lam  = rv.get("effective_lambda", 0)
        dcr  = rv.get("weight_decay_ratio", 0)
        bn   = r.get("block_norms", {})
        dstd = bn.get("depth_norm_std", 0)
        sw   = r.get("spectral_w", {})
        cmed = sw.get("cond_median") or 0
        uvd  = sw.get("uv_vs_d_ratio") or 0
        print(f"  {branch:<14} {lt:<12} {lam:>7.4f} {dcr:>8.2f} "
              f"{dstd:>10.4f} {cmed:>10.2f} {uvd:>7.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
