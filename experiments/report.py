"""
Turn experiments/results/*.json into a comparison table + graphs
for the README.

Usage:
    python -m experiments.report
"""
import json
import os
import glob

import matplotlib.pyplot as plt

RESULTS_DIR = "experiments/results"
GRAPHS_DIR = "experiments/results/graphs"


def load_all_results() -> list[dict]:
    reports = []
    for path in sorted(glob.glob(f"{RESULTS_DIR}/*.json")):
        with open(path) as f:
            reports.append(json.load(f))
    return reports


def print_comparison_table(reports: list[dict]):
    print(f"\n{'Experiment':<22}{'Hit Rate':<12}{'MRR':<10}{'Recall':<10}")
    print("-" * 54)
    for r in reports:
        m = r["aggregate_metrics"]
        print(f"{r['experiment_name']:<22}{m['hit_rate']:<12.2%}{m['mrr']:<10.3f}{m['avg_recall']:<10.2%}")


def plot_comparison(reports: list[dict]):
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    names = [r["experiment_name"] for r in reports]
    hit_rates = [r["aggregate_metrics"]["hit_rate"] for r in reports]
    mrrs = [r["aggregate_metrics"]["mrr"] for r in reports]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(names, hit_rates, color="#4C72B0")
    axes[0].set_title("Hit Rate by Experiment")
    axes[0].set_ylabel("Hit Rate")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, mrrs, color="#55A868")
    axes[1].set_title("MRR by Experiment")
    axes[1].set_ylabel("MRR")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = f"{GRAPHS_DIR}/experiment_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved graph -> {out_path}")


if __name__ == "__main__":
    reports = load_all_results()
    if not reports:
        print(f"No results found in {RESULTS_DIR}. Run experiments first.")
    else:
        print_comparison_table(reports)
        plot_comparison(reports)
