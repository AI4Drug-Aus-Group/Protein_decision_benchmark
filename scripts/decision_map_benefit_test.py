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
EVALUATION_METRICS = ["top1pct_recall_at_100", "spearman_all", "ndcg_at_100"]

BOOTSTRAP_RESAMPLES = 5000
PERMUTATIONS = 100_000
RANDOM_SEED = 20260916

CATEGORY_TO_FAMILY = {
    "calibrate_strong_zero_shot_plm": "ANY",
    "use_structure_aware_scores": "structure_aware",
    "use_msa_or_evolutionary_models": "evolution_or_msa",
    "additive_or_simple_ridge_first": "ADDITIVE",
}
EXCLUDED_CATEGORY = "measure_or_select_double_mutants"

ADDITIVE_LABEL = "additive control"


def load_per_assay_table() -> pd.DataFrame:
    usecols = [
        "assay",
        "seed",
        "budget",
        "regime",
        "model",
        "zero_shot_method",
        ROUTING_METRIC,
        *EVALUATION_METRICS,
    ]
    raw = pd.read_csv(LOW_N_METRICS, usecols=usecols)
    raw = raw[(raw.budget == BUDGET) & (raw.regime == REGIME)]

    calibrated = raw[raw.model == "zscore_linear"].copy()
    calibrated["strategy"] = calibrated.zero_shot_method

    additive = raw[raw.model == "additive_lookup"].copy()
    additive["strategy"] = ADDITIVE_LABEL

    combined = pd.concat([calibrated, additive], ignore_index=True)
    metrics = [ROUTING_METRIC, *EVALUATION_METRICS]
    return combined.groupby(["assay", "strategy"], as_index=False)[metrics].mean()


def choose_representatives(per_assay: pd.DataFrame, families: dict, metric: str) -> dict:
    method_means = (
        per_assay[per_assay.strategy != ADDITIVE_LABEL].groupby("strategy")[metric].mean().sort_values(ascending=False)
    )

    representatives = {}
    for category, family in CATEGORY_TO_FAMILY.items():
        if family == "ADDITIVE":
            representatives[category] = ADDITIVE_LABEL
        elif family == "ANY":
            representatives[category] = method_means.index[0]
        else:
            members = [m for m in method_means.index if families.get(m) == family]
            representatives[category] = method_means.loc[members].idxmax()
    return representatives


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple:
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_permutation_p(differences: np.ndarray, rng: np.random.Generator) -> float:
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return float("nan")
    observed = abs(differences.mean())
    if len(differences) <= 20:
        signs = np.array(np.meshgrid(*[[-1, 1]] * len(differences))).T.reshape(-1, len(differences))
    else:
        signs = rng.choice([-1, 1], size=(PERMUTATIONS, len(differences)))
    null = np.abs((signs * differences).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(p_values)
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def run_for_metric(
    per_assay: pd.DataFrame,
    assignments: pd.DataFrame,
    families: dict,
    metric: str,
    label: str,
) -> pd.DataFrame:
    representatives = choose_representatives(per_assay, families, metric)
    wide = per_assay.pivot(index="assay", columns="strategy", values=metric)

    print(f"\n{'=' * 78}\nEvaluation metric: {label}\n{'=' * 78}")
    print("Globally chosen representative for each decision-map category:")
    for category, method in representatives.items():
        print(f"   {category:34s} -> {method}")
    if len(set(representatives.values())) < len(representatives):
        print("   NOTE: two or more categories collapse to the same concrete method.")

    needed = sorted(set(representatives.values()))
    complete = wide[needed].notna().all(axis=1)
    dropped = int((~complete).sum())
    wide = wide[complete]
    if dropped:
        print(f"\n   dropped {dropped} of {len(complete)} assays where at least one strategy is undefined")
        if dropped > len(complete) / 2:
            print("   WARNING: more than half the assays are gone, so this metric is not representative")

    usable = assignments[assignments.assay.isin(wide.index)].copy()
    usable["representative"] = usable.recommended_strategy.map(representatives)

    print(f"\nAssays in the comparison: {len(usable)}")
    print(usable.recommended_strategy.value_counts().to_string())

    map_values = np.array(
        [wide.loc[row.assay, row.representative] for row in usable.itertuples()],
        dtype=float,
    )

    candidates = sorted(set(representatives.values()))
    fixed = {method: wide.loc[usable.assay, method].to_numpy(dtype=float) for method in candidates}
    oracle = np.max(np.vstack(list(fixed.values())), axis=0)

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    comparisons = [(f"always {method}", values) for method, values in fixed.items()]
    comparisons.append(("per-assay oracle over the same four strategies", oracle))

    for name, values in comparisons:
        difference = map_values - values
        finite = np.isfinite(difference)
        low, high = bootstrap_mean_ci(difference[finite], rng)
        rows.append(
            {
                "evaluation_metric": label,
                "comparison": f"decision map minus {name}",
                "n_assays": int(finite.sum()),
                "mean_decision_map": float(np.nanmean(map_values)),
                "mean_comparator": float(np.nanmean(values)),
                "mean_difference": float(np.nanmean(difference)),
                "ci_low": low,
                "ci_high": high,
                "raw_p": sign_permutation_p(difference[finite], rng),
            }
        )

    results = pd.DataFrame(rows)
    results["bh_adjusted_p"] = benjamini_hochberg(results.raw_p.to_numpy())

    print("\nDoes the map beat each alternative?")
    for row in results.itertuples():
        outcome = "map better" if row.mean_difference > 0 else "map worse"
        print(
            f"   {row.comparison:58s} {row.mean_difference:+.4f}"
            f"  95% CI [{row.ci_low:+.4f}, {row.ci_high:+.4f}]"
            f"  adj P = {row.bh_adjusted_p:.3g}   {outcome}"
        )

    oracle_choice = np.array(candidates)[np.argmax(np.vstack([fixed[m] for m in candidates]), axis=0)]
    agreement = float((oracle_choice == usable.representative.to_numpy()).mean())
    print(f"\n   Map picks the per-assay best of the four strategies in {agreement:.1%} of assays.")
    print(f"   Picking one strategy at random would agree {1 / len(candidates):.1%} of the time.")

    best_fixed = max(fixed, key=lambda m: np.nanmean(fixed[m]))
    headroom = float(np.nanmean(oracle) - np.nanmean(fixed[best_fixed]))
    captured = float(np.nanmean(map_values) - np.nanmean(fixed[best_fixed]))
    print(f"\n   Best fixed strategy is '{best_fixed}' at {np.nanmean(fixed[best_fixed]):.4f}.")
    print(f"   Oracle headroom above it: {headroom:+.4f}")
    print(f"   Headroom the map captures: {captured:+.4f}", end="")
    print(f"  ({captured / headroom:.1%} of available headroom)" if headroom > 0 else "")

    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    families = pd.read_csv(METHOD_INVENTORY).set_index("method").category.to_dict()
    per_assay = load_per_assay_table()
    print(f"loaded seed-averaged metrics for {per_assay.assay.nunique()} assays "
          f"and {per_assay.strategy.nunique()} strategies at budget {BUDGET}, {REGIME} design")

    assignments = pd.read_csv(ASSIGNMENTS)[["assay", "recommended_strategy"]]
    excluded = assignments[assignments.recommended_strategy == EXCLUDED_CATEGORY]
    assignments = assignments[assignments.recommended_strategy != EXCLUDED_CATEGORY]
    print(f"excluded {len(excluded)} assays recommended for double-mutant measurement, "
          "whose recommendation is an experimental action rather than a predictor")

    missing = set(assignments.assay) - set(per_assay.assay.unique())
    if missing:
        print(f"{len(missing)} further assays have no full-candidate-pool low-N metrics and drop out")

    all_results = [
        run_for_metric(per_assay, assignments, families, metric, label)
        for metric, label in [
            ("top1pct_recall_at_100", "top-1% recall among the top 100 predictions"),
            ("spearman_all", "global Spearman correlation"),
            ("ndcg_at_100", "NDCG at 100"),
            (ROUTING_METRIC, "best-top-100 percentile, the routing metric, shown for reference only"),
        ]
    ]

    combined = pd.concat(all_results, ignore_index=True)
    destination = OUTPUT_DIR / "decision_map_benefit_test.csv"
    combined.to_csv(destination, index=False)
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
