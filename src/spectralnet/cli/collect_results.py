"""
collect_results.py
==================
Aggregates results of multi-seed runs from lineage.json files.
Computes mean ± std for reporting in the paper.

Usage:
    python src/spectralnet/cli/collect_results.py \
        --runs_dir ./results/runs \
        --output   ./results/exp004_summary/exp004_summary.json

    # Or with an experiment filter:
    python src/spectralnet/cli/collect_results.py \
        --runs_dir ./results/runs \
        --exp_name exp004_vision_cifar10_multiseed \
        --output   ./results/exp004_summary/exp004_summary.json
"""

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import List, Dict, Any


def load_lineages(runs_dir: str, exp_name: str = None, exp_names: str = None) -> List[Dict[str, Any]]:
    """
    Loads lineage.json from subfolders of runs_dir.

    Filtering:
      exp_name  — single name (string), substring match
      exp_names — multiple names comma-separated (for ablation scripts)
    """
    # Normalize filter into a set of strings
    name_filters: set = set()
    if exp_name:
        name_filters.add(exp_name.strip())
    if exp_names:
        for n in exp_names.split(","):
            n = n.strip()
            if n:
                name_filters.add(n)

    lineages = []
    runs_path = Path(runs_dir)

    for run_dir in sorted(runs_path.iterdir()):
        lineage_path = run_dir / "lineage.json"
        if not lineage_path.exists():
            continue
        with open(lineage_path) as f:
            data = json.load(f)

        # Filter: config name must contain at least one of the filter strings
        if name_filters:
            cfg_name = data.get("config", {}).get("name", "")
            if not any(f in cfg_name for f in name_filters):
                continue

        # Take only runs with a recorded best_val_acc
        if "best_val_acc" not in data or data["best_val_acc"] == 0.0:
            continue

        lineages.append(data)

    return lineages


def aggregate(lineages: List[Dict[str, Any]], label: str = "Summary") -> Dict[str, Any]:
    """Computes mean ± std for best_val_acc."""
    if not lineages:
        return {"error": "No completed runs found"}

    accs   = [r["best_val_acc"]   for r in lineages]
    epochs = [r["best_val_epoch"] for r in lineages]
    seeds  = [r.get("seed", "?")  for r in lineages]

    mean_acc = statistics.mean(accs)
    std_acc  = statistics.stdev(accs) if len(accs) > 1 else 0.0

    return {
        "label":             label,
        "n_runs":            len(lineages),
        "seeds":             seeds,
        "best_val_accs":     [round(a, 4) for a in accs],
        "mean_val_acc":      round(mean_acc, 4),
        "std_val_acc":       round(std_acc, 4),
        "report":            f"{mean_acc:.2f}% ± {std_acc:.2f}%",
        "best_val_epochs":   epochs,
        "mean_best_epoch":   round(statistics.mean(epochs), 1),
        "run_ids":           [r["run_id"] for r in lineages],
    }


def print_summary(result: Dict[str, Any]) -> None:
    label = result.get("label", "Results")
    print("\n" + "=" * 50)
    print(f"  {label}")
    print("=" * 50)
    if "error" in result:
        print(f"  ❌ {result['error']}")
    else:
        print(f"  Runs:          {result['n_runs']}")
        print(f"  Seeds:         {result['seeds']}")
        print(f"  Per-seed accs: {result['best_val_accs']}")
        print(f"  REPORT:        {result['report']}")
        print(f"  Mean best ep:  {result['mean_best_epoch']}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results")
    parser.add_argument("--runs_dir",  required=True, help="Папка с run-директориями")
    parser.add_argument("--output",    required=True, help="Путь для сохранения summary JSON")
    parser.add_argument("--exp_name",  default=None,  help="Фильтр по одному имени эксперимента")
    parser.add_argument("--exp_names", default=None,  help="Фильтры через запятую: name1,name2,...")
    parser.add_argument("--per_exp",   action="store_true",
                        help="Группировать по config.name и отчитываться отдельно по каждому. "
                             "Игнорирует --exp_name/--exp_names. Удобно для ablation-скриптов.")
    args = parser.parse_args()

    print(f"🔍 Сканирование: {args.runs_dir}")

    if args.per_exp:
        # Загружаем все прогоны без фильтра
        all_lineages = load_lineages(args.runs_dir)
        print(f"   Найдено прогонов: {len(all_lineages)}")

        if not all_lineages:
            print("❌ Прогоны не найдены.")
            return

        # Группируем по config.name
        groups: Dict[str, List] = {}
        for lg in all_lineages:
            name = lg.get("config", {}).get("name", "unknown")
            groups.setdefault(name, []).append(lg)

        results_per_exp = {}
        for name, lineages in sorted(groups.items()):
            result = aggregate(lineages, label=name)
            print_summary(result)
            results_per_exp[name] = result

        # Сохраняем словарь {exp_name: summary}
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results_per_exp, f, indent=4)
        print(f"\n💾 Per-experiment summary сохранён: {args.output}")

    else:
        lineages = load_lineages(args.runs_dir, args.exp_name, args.exp_names)
        print(f"   Найдено прогонов: {len(lineages)}")

        if not lineages:
            print("❌ Прогоны не найдены. Проверьте runs_dir и exp_name.")
            return

        label = args.exp_name or args.exp_names or "Results Summary"
        result = aggregate(lineages, label=label)
        print_summary(result)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=4)
        print(f"\n💾 Summary сохранён: {args.output}")


if __name__ == "__main__":
    main()
