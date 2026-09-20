from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> float:
    values = values[np.isfinite(values)]
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(repetitions, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (repetitions + 1))


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    p_values = pd.to_numeric(values, errors="coerce").to_numpy(float)
    valid = np.isfinite(p_values)
    adjusted = np.full(len(p_values), np.nan)
    if valid.any():
        observed = p_values[valid]
        order = np.argsort(observed)
        ranked = observed[order] * len(observed) / np.arange(1, len(observed) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        restored = np.empty(len(observed))
        restored[order] = np.minimum(ranked, 1.0)
        adjusted[valid] = restored
    return pd.Series(adjusted, index=values.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "few_shot_external")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = root / "external_repos" / "kermut" / "results" / "merged_scores.csv"
    data = pd.read_csv(source)
    data = data.loc[data["fold_variable_name"].isin(["fold_random_5", "fold_modulo_5", "fold_contiguous_5"])]
    data["Spearman_fitness"] = pd.to_numeric(data["Spearman_fitness"], errors="coerce")

    assay_split = data[["model_name", "assay_id", "UniProt_id", "fold_variable_name", "Spearman_fitness"]].dropna()
    assay_mean = (
        assay_split.groupby(["model_name", "assay_id", "UniProt_id"], as_index=False)["Spearman_fitness"]
        .mean()
        .rename(columns={"Spearman_fitness": "mean_spearman_across_split_schemes"})
    )
    protein_mean = (
        assay_mean.groupby(["model_name", "UniProt_id"], as_index=False)["mean_spearman_across_split_schemes"]
        .mean()
    )
    summary = (
        protein_mean.groupby("model_name")["mean_spearman_across_split_schemes"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "n_proteins",
                "mean": "mean_protein_aggregated_spearman",
                "median": "median_protein_aggregated_spearman",
                "std": "sd_protein_aggregated_spearman",
            }
        )
    )
    summary["n_assays"] = summary["model_name"].map(assay_mean.groupby("model_name")["assay_id"].nunique())
    summary = summary.sort_values("mean_protein_aggregated_spearman", ascending=False)

    pivot = protein_mean.pivot(index="UniProt_id", columns="model_name", values="mean_spearman_across_split_schemes")
    rng = np.random.default_rng(args.seed)
    comparisons: list[dict[str, object]] = []
    for reference in ["ProteinNPT", "OHE - Not augmented"]:
        if reference not in pivot:
            continue
        for model in pivot.columns:
            if model == reference:
                continue
            paired = pivot[[model, reference]].dropna()
            delta = (paired[model] - paired[reference]).to_numpy(float)
            low, high = bootstrap_interval(delta, rng, args.bootstrap)
            comparisons.append(
                {
                    "model": model,
                    "reference": reference,
                    "n_paired_proteins": len(delta),
                    "mean_paired_spearman_difference": float(delta.mean()),
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "two_sided_sign_flip_p": sign_flip(delta, rng, args.permutations),
                }
            )

    assay_split.to_csv(out / "native_supervised_assay_split_scores.csv", index=False)
    assay_mean.to_csv(out / "native_supervised_assay_means.csv", index=False)
    protein_mean.to_csv(out / "native_supervised_protein_means.csv", index=False)
    summary.to_csv(out / "native_supervised_model_summary.csv", index=False)
    comparison_table = pd.DataFrame(comparisons)
    comparison_table["benjamini_hochberg_q_all_tests"] = benjamini_hochberg(comparison_table["two_sided_sign_flip_p"])
    comparison_table.to_csv(out / "native_supervised_paired_tests.csv", index=False)
    metadata = {
        "source": str(source.relative_to(root)),
        "setting": "ProteinGym supervised single-substitution benchmark",
        "split_schemes": ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"],
        "important_non_equivalence": "These models use the native five-fold supervised setting and are not matched to the 20, 50 or 100 measurement budgets in the decision benchmark.",
        "statistical_unit": "Assays were averaged within UniProt protein before bootstrap confidence intervals and paired sign-flip tests.",
        "multiple_testing": "Benjamini-Hochberg correction across all paired model comparisons.",
        "official_repository_commits": {"Kermut": "e6500221b5159a1896272bfc4a3c6e3896eb0027", "ProteinNPT": "ba50f15671870039aff8cff97081073dfcc73eb2"},
        "bootstrap_replicates": args.bootstrap,
        "sign_flip_permutations": args.permutations,
        "seed": args.seed,
    }
    (out / "native_supervised_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
