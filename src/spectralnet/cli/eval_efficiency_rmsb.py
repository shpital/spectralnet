"""
eval_efficiency_rmsb.py — RMSB Practical Metrics: Efficiency Block

Считает для каждой ветки (spectral / pure_shift / rmsb_r1):
  - params           : total + trainable parameter count
  - MACs             : multiply-accumulate ops на одно изображение (ptflops → thop → ручной fallback)
  - single-image latency (ms): медиана по 200 прогонам после разогрева 20 прогонами
  - batch throughput (img/s): батч из 64 изображений, 50 итераций
  - peak GPU memory (MB): torch.cuda.max_memory_allocated после одного прямого прохода

Использование:
  python src/spectralnet/cli/eval_efficiency_rmsb.py \\
      --runs_main  ./results/runs \\
      --runs_rmsb  ./results/rmsb \\
      --output     ./results/rmsb/efficiency.json

  # Ограничить ветки:
  python src/spectralnet/cli/eval_efficiency_rmsb.py \\
      --branches spectral rmsb_r1 \\
      --runs_main ./results/runs --runs_rmsb ./results/rmsb \\
      --output    ./results/rmsb/efficiency.json
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn as nn
import torchvision.models as tvm

# ──────────────────────────────────────────────────────────────────────────────
# Shared branch definitions (same as eval_robustness_rmsb.py)
# ──────────────────────────────────────────────────────────────────────────────

BRANCH_DEFS = {
    "spectral":   ("rmsb_spectral_cifar10",   "main"),
    "pure_shift": ("rmsb_pure_shift_cifar10", "main"),
    "rmsb_r1":    ("rmsb_r1_cifar10",         "rmsb"),
}

CIFAR10_NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)          # CIFAR-10 single-image shape
WARMUP_ITERS    = 20
LATENCY_ITERS   = 200
THROUGHPUT_BATCH= 64
THROUGHPUT_ITERS= 50


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (mirrors eval_robustness_rmsb)
# ──────────────────────────────────────────────────────────────────────────────

def build_model(lineage: dict) -> nn.Module:
    cfg = lineage["config"]
    mc  = cfg.get("model", {})
    bt  = mc.get("baseline_type", None)
    nc  = CIFAR10_NUM_CLASSES
    if bt == "resnet18":
        return tvm.resnet18(weights=None, num_classes=nc)
    if bt == "mobilenetv2":
        return tvm.mobilenet_v2(weights=None, num_classes=nc)
    if bt == "shufflenetv2":
        return tvm.shufflenet_v2_x0_5(weights=None, num_classes=nc)
    from omegaconf import OmegaConf
    from src.spectralnet.models.spectralnet_s import SpectralNetS
    mc["num_classes"] = nc
    return SpectralNetS(OmegaConf.create(mc))


def load_best_checkpoint(runs_dir: str, exp_name: str, device: str):
    """Return (model, lineage) for the seed with highest best_val_acc."""
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
# Parameter counting
# ──────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ──────────────────────────────────────────────────────────────────────────────
# MACs counting (ptflops → thop → fallback note)
# ──────────────────────────────────────────────────────────────────────────────

def count_macs(model: nn.Module, device: str) -> dict:
    """Attempt MACs count with ptflops, then thop, then return None with note."""
    cpu_model = model.cpu()

    # ── ptflops ──
    try:
        from ptflops import get_model_complexity_info
        macs_str, params_str = get_model_complexity_info(
            cpu_model, INPUT_SHAPE,
            as_strings=True, print_per_layer_stat=False, verbose=False
        )
        macs_num, _ = get_model_complexity_info(
            cpu_model, INPUT_SHAPE,
            as_strings=False, print_per_layer_stat=False, verbose=False
        )
        model.to(device)
        return {"macs": macs_num, "macs_str": macs_str, "tool": "ptflops"}
    except ImportError:
        pass
    except Exception as e:
        print(f"    ptflops failed: {e}")

    # ── thop ──
    try:
        from thop import profile
        dummy = torch.zeros(1, *INPUT_SHAPE)
        macs, _ = profile(cpu_model, inputs=(dummy,), verbose=False)
        model.to(device)
        return {"macs": macs, "macs_str": f"{macs/1e6:.2f}M", "tool": "thop"}
    except ImportError:
        pass
    except Exception as e:
        print(f"    thop failed: {e}")

    model.to(device)
    return {"macs": None, "macs_str": "n/a (install ptflops or thop)",
            "tool": "none", "note": "pip install ptflops  OR  pip install thop"}


# ──────────────────────────────────────────────────────────────────────────────
# Latency and throughput
# ──────────────────────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latency(model: nn.Module, device: str) -> dict:
    """Single-image latency (ms), median over LATENCY_ITERS after warmup."""
    dummy = torch.zeros(1, *INPUT_SHAPE, device=device)
    with torch.no_grad():
        # warmup
        for _ in range(WARMUP_ITERS):
            _ = model(dummy)
        _sync()
        # measure
        times = []
        for _ in range(LATENCY_ITERS):
            t0 = time.perf_counter()
            _ = model(dummy)
            _sync()
            times.append((time.perf_counter() - t0) * 1000)  # ms
    times.sort()
    return {
        "median_ms": statistics.median(times),
        "p95_ms":    times[int(0.95 * len(times))],
        "mean_ms":   statistics.mean(times),
    }


def measure_throughput(model: nn.Module, device: str) -> dict:
    """Batch throughput (images/sec), batch=THROUGHPUT_BATCH."""
    dummy = torch.zeros(THROUGHPUT_BATCH, *INPUT_SHAPE, device=device)
    with torch.no_grad():
        # warmup
        for _ in range(5):
            _ = model(dummy)
        _sync()
        t0 = time.perf_counter()
        for _ in range(THROUGHPUT_ITERS):
            _ = model(dummy)
        _sync()
        elapsed = time.perf_counter() - t0
    total_imgs = THROUGHPUT_BATCH * THROUGHPUT_ITERS
    return {
        "imgs_per_sec": total_imgs / elapsed,
        "batch_size":   THROUGHPUT_BATCH,
        "n_iters":      THROUGHPUT_ITERS,
    }


def measure_peak_memory(model: nn.Module, device: str) -> dict:
    """Peak GPU memory (MB) for a single forward pass."""
    if not torch.cuda.is_available():
        return {"peak_mb": None, "note": "CUDA not available"}
    dummy = torch.zeros(1, *INPUT_SHAPE, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {"peak_mb": peak}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="RMSB Efficiency Block")
    p.add_argument("--runs_main",  default="./results/runs")
    p.add_argument("--runs_rmsb",  default="./results/rmsb")
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

        seed = lineage.get("seed", "?")
        acc  = lineage.get("best_val_acc", 0)
        print(f"  Loaded seed={seed}, frozen_acc={acc:.2f}%")

        br = {"seed": seed, "frozen_acc": acc}

        print("  Counting params...")
        br["params"] = count_params(model)
        print(f"    total={br['params']['total']:,}  "
              f"trainable={br['params']['trainable']:,}")

        print("  Counting MACs...")
        br["macs"] = count_macs(model, device)
        print(f"    {br['macs']['macs_str']}  (tool: {br['macs']['tool']})")

        print("  Measuring single-image latency...")
        br["latency"] = measure_latency(model, device)
        print(f"    median={br['latency']['median_ms']:.3f}ms  "
              f"p95={br['latency']['p95_ms']:.3f}ms")

        print(f"  Measuring batch throughput (bs={THROUGHPUT_BATCH})...")
        br["throughput"] = measure_throughput(model, device)
        print(f"    {br['throughput']['imgs_per_sec']:.1f} img/s")

        print("  Measuring peak memory...")
        br["memory"] = measure_peak_memory(model, device)
        print(f"    {br['memory'].get('peak_mb', 'N/A')} MB")

        results[branch] = br

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")

    # Summary table
    print(f"\n{'='*80}")
    print("  EFFICIENCY SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Branch':<14} {'Params':>10} {'MACs':>12} "
          f"{'Lat(ms)':>9} {'Tput(i/s)':>11} {'Mem(MB)':>9} {'CleanAcc':>9}")
    print(f"  {'-'*14} {'-'*10} {'-'*12} {'-'*9} {'-'*11} {'-'*9} {'-'*9}")
    for branch, r in results.items():
        if "error" in r:
            print(f"  {branch:<14} ERROR: {r['error']}"); continue
        params  = r.get("params", {}).get("total", 0)
        macs_s  = r.get("macs", {}).get("macs_str", "?")
        lat     = r.get("latency", {}).get("median_ms", 0)
        tput    = r.get("throughput", {}).get("imgs_per_sec", 0)
        mem     = r.get("memory", {}).get("peak_mb")
        acc     = r.get("frozen_acc", 0)
        print(f"  {branch:<14} {params/1e6:>9.2f}M {macs_s:>12} "
              f"{lat:>8.3f}ms {tput:>10.0f} "
              f"{(str(round(mem,1)) if mem else 'N/A'):>9} {acc:>8.2f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
