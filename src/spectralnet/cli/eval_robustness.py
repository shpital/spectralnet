"""
eval_robustness.py — Line B: Robustness evaluation on corrupted data.

Supported datasets: CIFAR-10, CIFAR-100, SVHN.
The dataset is determined automatically from the lineage.json of each run,
or specified manually via --dataset.

Protocol:
  - Train clean, test corrupted (architectural robustness, not data aug effect).
  - Three corruption families:
      * AWGN:     SNR ∈ {20, 10, 5, 0, -5} dB  (noise in normalized space)
      * Blur:     σ ∈ {0.5, 1.0, 2.0, 3.0}     (Gaussian blur before normalization)
      * Contrast: factor ∈ {0.8, 0.5, 0.3, 0.1} (contrast reduction before normalization)

Usage:
  python src/spectralnet/cli/eval_robustness.py \
      --runs_dir ./results/runs \
      --exp_names exp005_modes8_multiseed exp006_resnet18 exp006_mobilenetv2 exp006_shufflenetv2 \
      --data_dir ./data \
      --output ./results/lineb_summary/robustness.json \
      --batch_size 256

  # Forcing dataset (for runs with different datasets in the same folder):
  python src/spectralnet/cli/eval_robustness.py \
      --runs_dir ./results/lineb_extended/tmp_svhn \
      --data_dir ./data \
      --dataset svhn \
      --output ./results/lineb_extended/robustness_svhn.json

For each experiment, the best seed is taken (highest best_val_acc).
"""

import argparse
import json
import os
import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tvm

# -------------------------------------------------------------------
# Dataset metadata
# -------------------------------------------------------------------
_DATASET_META = {
    "cifar10":  {"num_classes": 10,  "mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)},
    "cifar100": {"num_classes": 100, "mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
    "svhn":     {"num_classes": 10,  "mean": (0.4377, 0.4438, 0.4728), "std": (0.1980, 0.2010, 0.1970)},
}

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)


def _get_test_dataset_raw(dataset_name: str, data_dir: str):
    """Return raw test dataset (without transform) and its metadata."""
    name = dataset_name.lower()
    meta = _DATASET_META.get(name, _DATASET_META["cifar10"])
    if name == "cifar100":
        return datasets.CIFAR100, meta
    elif name == "svhn":
        return datasets.SVHN, meta
    else:
        return datasets.CIFAR10, meta


def _detect_dataset(lineage: dict) -> str:
    """Detect dataset name from lineage config."""
    cfg = lineage.get("config", {})
    ds = cfg.get("dataset", {}).get("name", "cifar10")
    return ds.lower()

# -------------------------------------------------------------------
# Corruption datasets
# -------------------------------------------------------------------

class AWGNDataset(Dataset):
    """CIFAR-10 test set with additive white Gaussian noise in normalized space."""
    def __init__(self, base_dataset, snr_db: float):
        self.base = base_dataset
        self.snr_db = snr_db

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        # img is already normalized tensor (C,H,W)
        signal_power = img.pow(2).mean().item()
        if signal_power < 1e-10:
            signal_power = 1e-10
        noise_std = math.sqrt(signal_power / (10 ** (self.snr_db / 10.0)))
        noise = torch.randn_like(img) * noise_std
        return img + noise, label


class BlurDataset(Dataset):
    """Test set with Gaussian blur applied before normalization."""
    def __init__(self, data_dir: str, sigma: float, dataset_name: str = "cifar10"):
        kernel_size = 5
        meta = _DATASET_META.get(dataset_name.lower(), _DATASET_META["cifar10"])
        transform = transforms.Compose([
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
            transforms.ToTensor(),
            transforms.Normalize(meta["mean"], meta["std"]),
        ])
        ds_cls, _ = _get_test_dataset_raw(dataset_name, data_dir)
        if dataset_name.lower() == "svhn":
            self.base = ds_cls(root=data_dir, split="test", download=False, transform=transform)
        else:
            self.base = ds_cls(root=data_dir, train=False, download=False, transform=transform)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx]


class ContrastDataset(Dataset):
    """Test set with reduced contrast applied before normalization."""
    def __init__(self, data_dir: str, factor: float, dataset_name: str = "cifar10"):
        meta = _DATASET_META.get(dataset_name.lower(), _DATASET_META["cifar10"])
        transform = transforms.Compose([
            transforms.Lambda(lambda img: TF.adjust_contrast(img, factor)),
            transforms.ToTensor(),
            transforms.Normalize(meta["mean"], meta["std"]),
        ])
        ds_cls, _ = _get_test_dataset_raw(dataset_name, data_dir)
        if dataset_name.lower() == "svhn":
            self.base = ds_cls(root=data_dir, split="test", download=False, transform=transform)
        else:
            self.base = ds_cls(root=data_dir, train=False, download=False, transform=transform)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx]


# -------------------------------------------------------------------
# Model reconstruction from lineage
# -------------------------------------------------------------------

def _resolve_num_classes(cfg: dict) -> int:
    """Determine num_classes from dataset.name, falling back to model.num_classes."""
    ds_name = cfg.get("dataset", {}).get("name", "cifar10").lower()
    ds_meta = _DATASET_META.get(ds_name)
    if ds_meta:
        return ds_meta["num_classes"]
    return cfg.get("model", {}).get("num_classes", 10)


def build_model_from_lineage(lineage: dict):
    """Reconstruct model from lineage config."""
    cfg = lineage["config"]
    model_cfg = cfg.get("model", {})
    num_classes = _resolve_num_classes(cfg)

    baseline_type = model_cfg.get("baseline_type", None)
    if baseline_type is not None:
        if baseline_type == "resnet18":
            return tvm.resnet18(weights=None, num_classes=num_classes)
        elif baseline_type == "mobilenetv2":
            return tvm.mobilenet_v2(weights=None, num_classes=num_classes)
        elif baseline_type == "shufflenetv2":
            return tvm.shufflenet_v2_x0_5(weights=None, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type}")
    else:
        from omegaconf import OmegaConf
        from src.spectralnet.models.spectralnet_s import SpectralNetS
        model_cfg["num_classes"] = num_classes
        model_conf = OmegaConf.create(model_cfg)
        return SpectralNetS(model_conf)


def load_best_checkpoint(run_dir: str, device: str):
    """Load model from best_checkpoint.pt in run_dir."""
    lineage_path = os.path.join(run_dir, "lineage.json")
    ckpt_path    = os.path.join(run_dir, "best_checkpoint.pt")

    if not os.path.exists(lineage_path) or not os.path.exists(ckpt_path):
        return None, None

    with open(lineage_path) as f:
        lineage = json.load(f)

    model = build_model_from_lineage(lineage)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model, lineage


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    correct = 0
    total   = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total   += target.size(0)
    return 100.0 * correct / total


def get_clean_loader(data_dir: str, batch_size: int, num_workers: int,
                     dataset_name: str = "cifar10") -> DataLoader:
    meta = _DATASET_META.get(dataset_name.lower(), _DATASET_META["cifar10"])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(meta["mean"], meta["std"]),
    ])
    ds_cls, _ = _get_test_dataset_raw(dataset_name, data_dir)
    if dataset_name.lower() == "svhn":
        ds = ds_cls(root=data_dir, split="test", download=False, transform=transform)
    else:
        ds = ds_cls(root=data_dir, train=False, download=False, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_awgn_loader(data_dir: str, snr_db: float, batch_size: int, num_workers: int,
                    dataset_name: str = "cifar10") -> DataLoader:
    meta = _DATASET_META.get(dataset_name.lower(), _DATASET_META["cifar10"])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(meta["mean"], meta["std"]),
    ])
    ds_cls, _ = _get_test_dataset_raw(dataset_name, data_dir)
    if dataset_name.lower() == "svhn":
        base = ds_cls(root=data_dir, split="test", download=False, transform=transform)
    else:
        base = ds_cls(root=data_dir, train=False, download=False, transform=transform)
    ds = AWGNDataset(base, snr_db)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_blur_loader(data_dir: str, sigma: float, batch_size: int, num_workers: int,
                    dataset_name: str = "cifar10") -> DataLoader:
    ds = BlurDataset(data_dir, sigma, dataset_name=dataset_name)
    return DataLoader(ds.base, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_contrast_loader(data_dir: str, factor: float, batch_size: int, num_workers: int,
                        dataset_name: str = "cifar10") -> DataLoader:
    ds = ContrastDataset(data_dir, factor, dataset_name=dataset_name)
    return DataLoader(ds.base, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def scan_runs(runs_dir: str, exp_names: list) -> dict:
    """Return {exp_name: (best_run_dir, best_acc, lineage)} — one run per exp (highest best_val_acc)."""
    result = {}
    for run_id in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, run_id)
        lineage_path = os.path.join(run_dir, "lineage.json")
        if not os.path.exists(lineage_path):
            continue
        with open(lineage_path) as f:
            lineage = json.load(f)
        exp_name = lineage.get("config", {}).get("name", "")
        if exp_names and exp_name not in exp_names:
            continue
        acc = lineage.get("best_val_acc", 0.0)
        if exp_name not in result or acc > result[exp_name][1]:
            result[exp_name] = (run_dir, acc, lineage)
    return result


def main():
    parser = argparse.ArgumentParser(description="Line B: Robustness evaluation")
    parser.add_argument("--runs_dir",   required=True)
    parser.add_argument("--exp_names",  nargs="+", default=[])
    parser.add_argument("--data_dir",   default="./data")
    parser.add_argument("--dataset",    default=None,
                        help="Force dataset (cifar10/cifar100/svhn). "
                             "If not set, auto-detected from lineage.")
    parser.add_argument("--output",     required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers",type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    awgn_snrs       = [20, 10, 5, 0, -5]
    blur_sigmas     = [0.5, 1.0, 2.0, 3.0]
    contrast_factors= [0.8, 0.5, 0.3, 0.1]

    runs = scan_runs(args.runs_dir, args.exp_names)
    print(f"Found {len(runs)} experiments: {list(runs.keys())}")

    results = {}

    for exp_name, (run_dir, best_acc, lineage) in runs.items():
        print(f"\n{'='*50}")
        print(f"  {exp_name} | best_val_acc={best_acc:.2f}% | {run_dir}")
        print(f"{'='*50}")

        model, _ = load_best_checkpoint(run_dir, device)
        if model is None:
            print(f"  ⚠️  Skipping {exp_name} — checkpoint not found")
            continue

        ds_name = args.dataset if args.dataset else _detect_dataset(lineage)
        print(f"  Dataset: {ds_name}")

        exp_result = {"best_val_acc": best_acc, "run_dir": run_dir, "dataset": ds_name}

        # --- Clean ---
        clean_loader = get_clean_loader(args.data_dir, args.batch_size, args.num_workers,
                                        dataset_name=ds_name)
        clean_acc = evaluate(model, clean_loader, device)
        exp_result["clean"] = clean_acc
        print(f"  Clean:          {clean_acc:.2f}%")

        # --- AWGN ---
        exp_result["awgn"] = {}
        for snr in awgn_snrs:
            loader = get_awgn_loader(args.data_dir, snr, args.batch_size, args.num_workers,
                                     dataset_name=ds_name)
            acc = evaluate(model, loader, device)
            exp_result["awgn"][snr] = acc
            print(f"  AWGN {snr:+3d} dB:   {acc:.2f}%")

        # --- Blur ---
        exp_result["blur"] = {}
        for sigma in blur_sigmas:
            loader = get_blur_loader(args.data_dir, sigma, args.batch_size, args.num_workers,
                                     dataset_name=ds_name)
            acc = evaluate(model, loader, device)
            exp_result["blur"][sigma] = acc
            print(f"  Blur σ={sigma}:      {acc:.2f}%")

        # --- Contrast ---
        exp_result["contrast"] = {}
        for factor in contrast_factors:
            loader = get_contrast_loader(args.data_dir, factor, args.batch_size, args.num_workers,
                                         dataset_name=ds_name)
            acc = evaluate(model, loader, device)
            exp_result["contrast"][factor] = acc
            print(f"  Contrast {factor}: {acc:.2f}%")

        results[exp_name] = exp_result

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Results saved: {args.output}")

    print(f"\n{'='*70}")
    print(f"  ROBUSTNESS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<30} {'Clean':>7} {'AWGN0':>7} {'Blur2':>7} {'Cont0.3':>8}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for exp_name, r in results.items():
        clean   = r.get("clean", 0)
        awgn0   = r.get("awgn", {}).get(0, 0)
        blur2   = r.get("blur", {}).get(2.0, 0)
        cont03  = r.get("contrast", {}).get(0.3, 0)
        print(f"  {exp_name:<30} {clean:>7.2f} {awgn0:>7.2f} {blur2:>7.2f} {cont03:>8.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
