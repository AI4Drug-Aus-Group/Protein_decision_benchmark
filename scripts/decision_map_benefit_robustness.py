import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(os.environ.get(
    "PROTEIN_DECISION_PACKAGE", ROOT / "analysis_package"
)).resolve()
PACKAGE = PACKAGE_ROOT
LOW_N_METRICS = PACKAGE / "low_measurement_metrics.csv"
METHOD_INVENTORY = PACKAGE / "benchmark_definition/methods/method_inventory.csv"
ASSIGNMENTS = (
    PACKAGE
    / "decision_map/final_assignments/assay_strategy_recommendations.csv"
)
OUTPUT_DIR = ROOT / "results" / "experiments"

BUDGET = 20
REGIME = "mixed_random"
ROUTING_METRIC = "best_true_in_pred_top_100_percentile"
METRICS = {
    "top1pct_recall_at_100": "top-1% recall among the top 100 predictions",
    "spearman_all": "global Spearman correlation",
    "ndcg_at_100": "NDCG at 100",
    ROUTING_METRIC: "best-top-100 percentile, the routing metric",
}

CATEGORY_TO_FAMILY = {
    "calibrate_strong_zero_shot_plm": "ANY",
    "use_structure_aware_scores": "structure_aware",
    "use_msa_or_evolutionary_models": "evolution_or_msa",
    "additive_or_simple_ridge_first": "ADDITIVE",
}
EXCLUDED_CATEGORY = "measure_or_select_double_mutants"
ADDITIVE_LABEL = "additive control"

SHUFFLES = 10_000
RANDOM_SEED = 20260916


def load_wide_tables() -> dict:
    usecols = ["assay", "seed", "budget", "regime", "model", "zero_shot_method", *METRICS]
    raw = pd.read_csv(LOW_N_METRICS, usecols=usecols)
    raw = raw[(raw.budget == BUDGET) & (raw.regime == REGIME)]

    calibrated = raw[raw.model == "zscore_linear"].copy()
    calibrated["strategy"] = calibrated.zero_shot_method
    additive = raw[raw.model == "additive_lookup"].copy()
    additive["strategy"] = ADDITIVE_LABEL

    combined = pd.concat([calibrated, additive], ignore_index=True)
    averaged = combined.groupby(["assay", "strategy"], as_index=False)[list(METRICS)].mean()
    return {metric: averaged.pivot(index="assay", columns="strategy", values=metric) for metric in METRICS}


def family_members(families: dict, wide: pd.DataFrame) -> dict:
    methods = [column for column in wide.columns if column != ADDITIVE_LABEL]
    return {
        "ANY": methods,
        "structure_aware": [m for m in methods if families.get(m) == "structure_aware"],
        "evolution_or_msa": [m for m in methods if families.get(m) == "evolution_or_msa"],
        "ADDITIVE": [ADDITIVE_LABEL],
    }


def generous_map_values(wide: pd.DataFrame, assignments: pd.DataFrame, members: dict) -> np.ndarray:
    values = []
    for row in assignments.itertuples():
        family = CATEGORY_TO_FAMILY[row.recommended_strategy]
        values.append(np.nanmax(wide.loc[row.assay, members[family]].to_numpy(dtype=float)))
    return np.array(values, dtype=float)


def run_metric(wide: pd.DataFrame, assignments: pd.DataFrame, families: dict, metric: str, label: str) -> list:
    members = family_members(families, wide)

    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")

    complete = wide.notna().all(axis=1)
    dropped = int((~complete).sum())
    wide = wide[complete]
    if dropped:
        print(f"   dropped {dropped} of {len(complete)} assays where at least one strategy is undefined")
        if dropped > len(complete) / 2:
            print("   WARNING: more than half the assays are gone, so this metric is not representative")

    assignments = assignments[assignments.assay.isin(wide.index)].reset_index(drop=True)

    generous = generous_map_values(wide, assignments, members)
    all_methods = members["ANY"] + [ADDITIVE_LABEL]
    unrouted_oracle = np.nanmax(wide.loc[assignments.assay, all_methods].to_numpy(dtype=float), axis=1)

    print("Map at its theoretical best, against the same oracle without routing")
    print(f"   generous map, best in the recommended family   {np.nanmean(generous):.4f}")
    print(f"   no routing, best of all strategies per assay   {np.nanmean(unrouted_oracle):.4f}")
    print(f"   difference                                     {np.nanmean(generous - unrouted_oracle):+.4f}")
    print("   The per-assay oracle selects the best outcome among the same strategy representatives.")
    print("   The gap is the cost of routing when every family is played perfectly.")

    rng = np.random.default_rng(RANDOM_SEED)
    method_means = wide.drop(columns=[ADDITIVE_LABEL]).mean()
    representatives = {}
    for category, family in CATEGORY_TO_FAMILY.items():
        if family == "ADDITIVE":
            representatives[category] = ADDITIVE_LABEL
        elif family == "ANY":
            representatives[category] = method_means.idxmax()
        else:
            representatives[category] = method_means.loc[members[family]].idxmax()

    deployable = np.array(
        [wide.loc[row.assay, representatives[row.recommended_strategy]] for row in assignments.itertuples()],
        dtype=float,
    )
    observed_mean = float(np.nanmean(deployable))

    labels = assignments.recommended_strategy.to_numpy()
    shuffled_means = np.empty(SHUFFLES)
    for i in range(SHUFFLES):
        permuted = rng.permutation(labels)
        shuffled_means[i] = np.nanmean(
            [wide.loc[assay, representatives[category]] for assay, category in zip(assignments.assay, permuted)]
        )

    p_value = float((np.sum(shuffled_means >= observed_mean) + 1) / (SHUFFLES + 1))
    print("\nReal routing against random routing with the same category sizes")
    print(f"   real assignment                {observed_mean:.4f}")
    print(f"   random assignment, mean        {shuffled_means.mean():.4f}")
    print(f"   random assignment, 95th pct    {np.percentile(shuffled_means, 95):.4f}")
    print(f"   one-sided permutation P        {p_value:.4g}")
    outcome = "routing carries signal" if p_value < 0.05 else "routing indistinguishable from random"
    print(f"   outcome                        {outcome}")

    return [
        {
            "evaluation_metric": label,
            "check": "generous map minus unrouted oracle",
            "value": float(np.nanmean(generous - unrouted_oracle)),
            "generous_map": float(np.nanmean(generous)),
            "unrouted_oracle": float(np.nanmean(unrouted_oracle)),
            "p_value": np.nan,
        },
        {
            "evaluation_metric": label,
            "check": "real routing versus random routing",
            "value": observed_mean - float(shuffled_means.mean()),
            "generous_map": observed_mean,
            "unrouted_oracle": float(shuffled_means.mean()),
            "p_value": p_value,
        },
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    families = pd.read_csv(METHOD_INVENTORY).set_index("method").category.to_dict()
    tables = load_wide_tables()

    assignments = pd.read_csv(ASSIGNMENTS)[["assay", "recommended_strategy"]]
    assignments = assignments[assignments.recommended_strategy != EXCLUDED_CATEGORY]

    rows = []
    for metric, label in METRICS.items():
        rows.extend(run_metric(tables[metric], assignments, families, metric, label))

    destination = OUTPUT_DIR / "decision_map_benefit_robustness.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
