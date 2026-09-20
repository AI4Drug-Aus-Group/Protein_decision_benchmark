from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "decision_map" / "assay_strategy_recommendations.csv"
OUT_DIR = ROOT / "results" / "decision_map_validation"
STRATEGIES = [
    "calibrate_strong_zero_shot_plm",
    "measure_or_select_double_mutants",
    "use_structure_aware_scores",
    "use_msa_or_evolutionary_models",
    "additive_or_simple_ridge_first",
]


def parse_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def nearest_rank_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def thresholds_from_rows(rows: pd.DataFrame) -> dict[str, float]:
    epistasis = [value for raw in rows["normalized_epistasis_strength"] if (value := parse_float(raw)) is not None]
    msa_depth = [value for raw in rows["official_MSA_Neff_L"] if (value := parse_float(raw)) is not None]
    return {
        "low_n_gain": 0.02,
        "epistasis": nearest_rank_quantile(epistasis, 2 / 3),
        "structure_gain": 0.10,
        "msa_depth": nearest_rank_quantile(msa_depth, 2 / 3),
    }


def recommend(row: pd.Series, thresholds: dict[str, float]) -> str:
    low_n_gain = parse_float(row.get("low_n20_best_plm_gain_over_additive"))
    epistasis = parse_float(row.get("normalized_epistasis_strength"))
    doubles = parse_float(row.get("n_doubles")) or 0.0
    structure_gain = parse_float(row.get("structure_gain"))
    msa_depth = parse_float(row.get("official_MSA_Neff_L"))
    if low_n_gain is not None and low_n_gain >= thresholds["low_n_gain"]:
        return "calibrate_strong_zero_shot_plm"
    if epistasis is not None and epistasis >= thresholds["epistasis"] and doubles > 0:
        return "measure_or_select_double_mutants"
    if structure_gain is not None and structure_gain >= thresholds["structure_gain"]:
        return "use_structure_aware_scores"
    if msa_depth is not None and msa_depth >= thresholds["msa_depth"]:
        return "use_msa_or_evolutionary_models"
    return "additive_or_simple_ridge_first"


def assign_all(rows: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    return rows.apply(lambda row: recommend(row, thresholds), axis=1)


def summarize_retention(detail: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return (
        detail.groupby("base_strategy", as_index=False)
        .agg(
            n_assays=("assay", "size"),
            mean_base_retention=(value_column, "mean"),
            q10_base_retention=(value_column, lambda values: float(np.quantile(values, 0.10))),
            median_base_retention=(value_column, "median"),
            minimum_base_retention=(value_column, "min"),
        )
        .sort_values(["mean_base_retention", "n_assays"], ascending=[False, False])
    )


def leave_one_protein(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    for protein in sorted(rows["protein_key"].dropna().unique()):
        training = rows[rows["protein_key"].ne(protein)]
        held_out = rows[rows["protein_key"].eq(protein)]
        thresholds = thresholds_from_rows(training)
        assigned = assign_all(held_out, thresholds)
        for (_, row), strategy in zip(held_out.iterrows(), assigned):
            detail_rows.append(
                {
                    "assay": row["assay"],
                    "protein_key": protein,
                    "base_strategy": row["recommended_strategy"],
                    "heldout_threshold_strategy": strategy,
                    "retained": int(strategy == row["recommended_strategy"]),
                    "low_n_gain_threshold_train": thresholds["low_n_gain"],
                    "epistasis_threshold_train": thresholds["epistasis"],
                    "structure_gain_threshold_train": thresholds["structure_gain"],
                    "msa_depth_threshold_train": thresholds["msa_depth"],
                    "n_train_assays": len(training),
                    "n_test_assays_for_protein": len(held_out),
                }
            )
    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby("base_strategy", as_index=False)
        .agg(n_assays=("assay", "size"), mean_retention=("retained", "mean"), min_retention=("retained", "min"))
        .sort_values(["mean_retention", "n_assays"], ascending=[False, False])
    )
    overall = pd.DataFrame(
        [
            {
                "base_strategy": "overall",
                "n_assays": len(detail),
                "mean_retention": detail["retained"].mean(),
                "min_retention": detail["retained"].min(),
            }
        ]
    )
    return detail, pd.concat([summary, overall], ignore_index=True)


def clustered_joint_sensitivity(
    rows: pd.DataFrame,
    iterations: int,
    seed: int,
    fixed_threshold_relative_range: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    assays = rows["assay"].tolist()
    proteins = sorted(rows["protein_key"].dropna().unique())
    counters = {assay: Counter() for assay in assays}
    threshold_rows: list[dict[str, object]] = []
    base_thresholds = thresholds_from_rows(rows)
    for iteration in range(iterations):
        sampled_proteins = rng.choice(proteins, size=len(proteins), replace=True)
        sampled_blocks = [rows[rows["protein_key"].eq(protein)] for protein in sampled_proteins]
        training = pd.concat(sampled_blocks, ignore_index=True)
        thresholds = thresholds_from_rows(training)
        thresholds["low_n_gain"] = base_thresholds["low_n_gain"] * rng.uniform(
            1.0 - fixed_threshold_relative_range, 1.0 + fixed_threshold_relative_range
        )
        thresholds["structure_gain"] = base_thresholds["structure_gain"] * rng.uniform(
            1.0 - fixed_threshold_relative_range, 1.0 + fixed_threshold_relative_range
        )
        assigned = assign_all(rows, thresholds)
        for assay, strategy in zip(assays, assigned):
            counters[assay][strategy] += 1
        threshold_rows.append({"iteration": iteration, **thresholds})
    base_map = dict(zip(rows["assay"], rows["recommended_strategy"]))
    protein_map = dict(zip(rows["assay"], rows["protein_key"]))
    detail_rows: list[dict[str, object]] = []
    for assay in assays:
        base_strategy = base_map[assay]
        counts = counters[assay]
        most_common = counts.most_common(1)[0]
        detail_rows.append(
            {
                "assay": assay,
                "protein_key": protein_map[assay],
                "base_strategy": base_strategy,
                "base_retention_fraction": counts[base_strategy] / iterations,
                "most_common_strategy": most_common[0],
                "most_common_fraction": most_common[1] / iterations,
                **{f"fraction_{strategy}": counts[strategy] / iterations for strategy in STRATEGIES},
            }
        )
    detail = pd.DataFrame(detail_rows)
    return detail, summarize_retention(detail, "base_retention_fraction"), pd.DataFrame(threshold_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--fixed-threshold-relative-range", type=float, default=0.20)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(SOURCE)
    base_thresholds = thresholds_from_rows(rows)
    recomputed = assign_all(rows, base_thresholds)
    reproduction = pd.DataFrame(
        {
            "assay": rows["assay"],
            "protein_key": rows["protein_key"],
            "base_strategy": rows["recommended_strategy"],
            "recomputed_strategy": recomputed,
            "matched": recomputed.eq(rows["recommended_strategy"]).astype(int),
        }
    )
    leave_detail, leave_summary = leave_one_protein(rows)
    sensitivity_detail, sensitivity_summary, threshold_values = clustered_joint_sensitivity(
        rows,
        args.n_bootstrap,
        args.seed,
        args.fixed_threshold_relative_range,
    )
    reproduction.to_csv(OUT_DIR / "decision_map_rule_reproduction.csv", index=False)
    leave_detail.to_csv(OUT_DIR / "decision_map_leave_one_protein_detail.csv", index=False)
    leave_summary.to_csv(OUT_DIR / "decision_map_leave_one_protein_summary.csv", index=False)
    sensitivity_detail.to_csv(OUT_DIR / "decision_map_bootstrap_threshold_detail.csv", index=False)
    sensitivity_summary.to_csv(OUT_DIR / "decision_map_bootstrap_threshold_summary.csv", index=False)
    threshold_values.to_csv(OUT_DIR / "decision_map_bootstrap_threshold_values.csv", index=False)
    metadata = {
        "analysis_label": "retrospective decision-rule assignment stability",
        "source": str(SOURCE.relative_to(ROOT)),
        "base_thresholds": base_thresholds,
        "rule_reproduction_rate": reproduction["matched"].mean(),
        "leave_one_protein_overall_assignment_retention": leave_detail["retained"].mean(),
        "joint_sensitivity_iterations": args.n_bootstrap,
        "joint_sensitivity_seed": args.seed,
        "cluster_resampling_unit": "protein",
        "data_driven_thresholds_recomputed": ["epistasis", "msa_depth"],
        "fixed_thresholds_perturbed": ["low_n_gain", "structure_gain"],
        "fixed_threshold_relative_range": args.fixed_threshold_relative_range,
        "analysis_scope": "Retrospective assignment retention under leave-one-protein threshold estimation and protein-clustered threshold perturbations.",
    }
    (OUT_DIR / "decision_map_validation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
