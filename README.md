# SpectralNet

**SpectralNet: A Resolvent-Inspired Neural Architecture Based on Chernoff Approximations and Photonic Motivation**

*Sergey V. Shpital*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19452600.svg)](https://doi.org/10.5281/zenodo.19452600)

## Overview

This repository contains the official PyTorch implementation of **SpectralNet**, a digital neural architecture motivated by the Chernoff-Remizov constructive line for operator evolution. 

The core idea is to interpret a learnable spectral layer as a discrete approximation of a small-step family whose repeated composition approximates a semigroup, while dense aggregation of intermediate states with decaying weights approximates the resolvent as a Laplace transform of the evolution.

## Installation

```bash
git clone https://github.com/shpital/spectralnet.git
cd spectralnet
pip install -r requirements.txt
```

## Usage

The repository uses [Hydra](https://hydra.cc/) for configuration management. All configurations are located in the `configs/` directory. The main execution scripts are located in `src/spectralnet/cli/`.

### 1. Training Models

**Train SpectralNet (e.g., EXP-004 on CIFAR-10):**
```bash
python src/spectralnet/cli/train.py \
    --config-name experiment/exp004_vision_cifar10_multiseed \
    training.seed=42
```

**Train Baselines (ResNet-18, MobileNetV2, ShuffleNetV2):**
```bash
python src/spectralnet/cli/train_baseline.py \
    --config-name experiment/exp_baseline_cifar10 \
    model.baseline_type=resnet18 \
    training.seed=42
```

*Logs, checkpoints, and a `lineage.json` (containing the exact configuration and metrics) will be saved in `results/runs/<timestamp>_...`.*

### 2. Evaluation Scripts

The repository includes several scripts to evaluate pre-trained models across different dimensions. These scripts automatically scan the `results/` directory, find the best checkpoints using `lineage.json`, and compute metrics.

**Robustness Evaluation:**
Evaluates model robustness against AWGN, Gaussian Blur, and Contrast reduction.
```bash
python src/spectralnet/cli/eval_robustness_rmsb.py \
    --mode full \
    --runs_main ./results/runs \
    --runs_rmsb ./results/rmsb \
    --output ./results/robustness_summary.json
```

**Efficiency Metrics:**
Computes parameters, MACs (requires `ptflops` or `thop`), latency, throughput, and peak GPU memory.
```bash
python src/spectralnet/cli/eval_efficiency_rmsb.py \
    --runs_main ./results/runs \
    --runs_rmsb ./results/rmsb \
    --output ./results/efficiency_summary.json
```

**Conditioning Analysis:**
Analyzes the spectral operator $W = D + UV^*$, resolvent weights, and block norm proxy (feature amplification).
```bash
python src/spectralnet/cli/eval_conditioning_rmsb.py \
    --runs_main ./results/runs \
    --runs_rmsb ./results/rmsb \
    --output ./results/conditioning_summary.json
```

### 3. Utilities

**Aggregate Multi-seed Results:**
Computes mean and standard deviation across multiple seeds for a given experiment.
```bash
python src/spectralnet/cli/collect_results.py \
    --runs_dir ./results/runs \
    --exp_name exp004_vision_cifar10_multiseed \
    --output ./results/exp004_summary.json
```

**Numerical Audit (Gradcheck):**
Runs an isolated PyTorch `gradcheck` in `float64` to verify the differentiability of the complex FFT math and spectral masking.
```bash
python src/spectralnet/cli/run_gradcheck.py
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{shpital2026spectralnet,
      title={SpectralNet: A Resolvent-Inspired Neural Architecture Based on Chernoff Approximations and Photonic Motivation}, 
      author={Sergey V. Shpital},
      year={2026},
      doi={10.5281/zenodo.19452600},
      url={https://doi.org/10.5281/zenodo.19452600},
      publisher={Zenodo}
}
```