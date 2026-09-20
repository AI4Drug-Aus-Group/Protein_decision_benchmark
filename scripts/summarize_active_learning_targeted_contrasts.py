from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "active_learning_external" / "active_learning_external_common_pool_metrics.csv"
OUTPUT = ROOT / "results" / "active_learning_external" / "active_learning_common27_targeted_contrasts.csv"

CONTRASTS = [
    ("ensemble_ucb:top_methods", "ensemble_mean:top_methods"),
    ("ensemble_rank_mean_std:top_methods", "ensemble_mean:top_methods"),
    ("ensemble_rank_mean_disagreement:top_methods", "ensemble_mean:top_methods"),
    ("diverse_ensemble:top_methods", "ensemble_mean:top_methods"),
]

METRICS = ["top1pct_found", "best_observed_percentile"]


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> float:
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(repetitions, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (repetitions + 1))


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    observed = values.to_numpy(float)
    order = np.argsort(observed)
    ranked = observed[order] * len(observed) / np.arange(1, len(observed) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty(len(observed))
    restored[order] = np.minimum(ranked, 1.0)
    return pd.Series(restored, index=values.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    metrics = pd.read_csv(INPUT)
    final = metrics[metrics["round"].eq(metrics["round"].max())]
    rng = np.random.default_rng(args.seed)
    rows = []
    for reference, comparison in CONTRASTS:
        for metric in METRICS:
            pivot = final[final["policy"].isin([reference, comparison])].pivot_table(
                index=["assay", "seed"], columns="policy", values=metric, aggfunc="first"
            )
            paired = pivot[[reference, comparison]].dropna()
            assay_delta = (paired[reference] - paired[comparison]).groupby(level="assay").mean().to_numpy(float)
            low, high = bootstrap_mean(assay_delta, rng, args.bootstrap)
            rows.append(
                {
                    "reference": reference,
                    "comparison": comparison,
                    "metric": metric,
                    "n_assays": len(assay_delta),
                    "mean_paired_difference_reference_minus_comparison": float(assay_delta.mean()),
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "two_sided_sign_flip_p": sign_flip(assay_delta, rng, args.permutations),
                }
            )
    table = pd.DataFrame(rows)
    table["benjamini_hochberg_q_targeted_contrasts"] = benjamini_hochberg(table["two_sided_sign_flip_p"])
    table.to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()
