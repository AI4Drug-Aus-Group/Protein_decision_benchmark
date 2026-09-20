import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(os.environ.get(
    "PROTEIN_DECISION_PACKAGE", ROOT / "analysis_package"
)).resolve()
GLOBAL_DATA = PACKAGE_ROOT / "global_data"
LOW_N_METRICS = PACKAGE_ROOT / "low_measurement_metrics.csv"
DELTA_DETAIL = (
    GLOBAL_DATA / "double_mutant_measurements/measurement_value/single_plus_double_delta_detail.csv"
)
ASSIGNMENTS = GLOBAL_DATA / "decision_map/final_assignments/assay_strategy_recommendations.csv"
FEATURES = GLOBAL_DATA / "decision_map/assay_features/assay_decision_features.csv"
OUTPUT_DIR = ROOT / "results" / "experiments"

DOUBLE_CATEGORY = "measure_or_select_double_mutants"
EVALUATION_METRIC = "top1pct_recall_at_100"
BUDGET = 20
REGIME = "mixed_random"
ADDITIVE_LABEL = "additive control"

MUTATION_AWARE_MODELS = ["onehot_site_ridge", "additive_plus_site_ridge"]

CANDIDATE_ACTIONS = ["ProSST-2048", "VenusREM", "S3F_MSA", "GEMME", "RSALOR", ADDITIVE_LABEL]

ROUTER_FEATURES = [
    "zero_abs_spearman_ProSST-2048",
    "zero_abs_spearman_Site_Independent",
    "low_n20_best_plm_gain_over_additive",
    "official_MSA_Neff_L",
    "n_doubles",
    "n_singles",
    "n_variants",
    "zero_abs_spearman_GEMME",
    "zero_abs_spearman_VenusREM",
    "zero_abs_spearman_S3F_MSA",
]

BOOTSTRAP_RESAMPLES = 5000
PERMUTATIONS = 100_000
RANDOM_SEED = 20260916


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple:
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def group_permutation_p(first: np.ndarray, second: np.ndarray, rng: np.random.Generator) -> float:
    pooled = np.concatenate([first, second])
    observed = abs(first.mean() - second.mean())
    n_first = len(first)
    count = 0
    for _ in range(10_000):
        shuffled = rng.permutation(pooled)
        if abs(shuffled[:n_first].mean() - shuffled[n_first:].mean()) >= observed:
            count += 1
    return (count + 1) / 10_001


def sign_permutation_p(differences: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(differences.mean())
    signs = rng.choice([-1, 1], size=(PERMUTATIONS, len(differences)))
    null = np.abs((signs * differences).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def part_a(rng: np.random.Generator) -> pd.DataFrame:
    print(f"\n{'=' * 78}\nRule 2 selection, gain among selected assays against assays passed over\n{'=' * 78}")

    assignments = pd.read_csv(ASSIGNMENTS)
    with_doubles = set(assignments[assignments.n_doubles > 0].assay)
    recommended = set(assignments[assignments.recommended_strategy == DOUBLE_CATEGORY].assay)

    delta = pd.read_csv(DELTA_DETAIL)
    delta = delta[(delta.budget == BUDGET) & delta.assay.isin(with_doubles)]

    rows = []
    for model in MUTATION_AWARE_MODELS:
        subset = delta[delta.model == model]
        per_assay = subset.groupby("assay").delta_spearman_doubles.mean().dropna()

        selected = per_assay[per_assay.index.isin(recommended)]
        passed_over = per_assay[~per_assay.index.isin(recommended)]
        if len(selected) < 3 or len(passed_over) < 3:
            continue

        difference = float(selected.mean() - passed_over.mean())
        low, high = bootstrap_ci(selected.to_numpy(), rng)
        p_value = group_permutation_p(selected.to_numpy(), passed_over.to_numpy(), rng)

        print(f"\n   model: {model}")
        print(f"      assays rule 2 selected      n={len(selected):3d}  mean gain {selected.mean():+.4f}"
              f"  95% CI [{low:+.4f}, {high:+.4f}]")
        print(f"      assays rule 2 passed over   n={len(passed_over):3d}  mean gain {passed_over.mean():+.4f}")
        print(f"      difference                  {difference:+.4f}   permutation P = {p_value:.3g}")
        outcome = "rule 2 selects better assays" if p_value < 0.05 and difference > 0 else "no evidence rule 2 selects better assays"
        print(f"      outcome                     {outcome}")

        rows.append(
            {
                "part": "A",
                "model": model,
                "n_selected": len(selected),
                "n_passed_over": len(passed_over),
                "mean_gain_selected": float(selected.mean()),
                "mean_gain_passed_over": float(passed_over.mean()),
                "difference": difference,
                "ci_low": low,
                "ci_high": high,
                "p_value": p_value,
            }
        )

    print("\n   Both groups gain from measuring double mutants. The question is only whether")
    print("   the test is whether rule 2 concentrates that gain.")
    return pd.DataFrame(rows)


def load_strategy_matrix() -> pd.DataFrame:
    usecols = ["assay", "seed", "budget", "regime", "model", "zero_shot_method", EVALUATION_METRIC]
    raw = pd.read_csv(LOW_N_METRICS, usecols=usecols)
    raw = raw[(raw.budget == BUDGET) & (raw.regime == REGIME)]

    calibrated = raw[raw.model == "zscore_linear"].copy()
    calibrated["strategy"] = calibrated.zero_shot_method
    additive = raw[raw.model == "additive_lookup"].copy()
    additive["strategy"] = ADDITIVE_LABEL

    combined = pd.concat([calibrated, additive], ignore_index=True)
    averaged = combined.groupby(["assay", "strategy"], as_index=False)[EVALUATION_METRIC].mean()
    wide = averaged.pivot(index="assay", columns="strategy", values=EVALUATION_METRIC)
    return wide[CANDIDATE_ACTIONS].dropna()


def part_b(rng: np.random.Generator) -> pd.DataFrame:
    print(f"\n{'=' * 78}\nRouting against the best fixed strategy\n{'=' * 78}")

    wide = load_strategy_matrix()
    features = pd.read_csv(FEATURES).set_index("assay")
    assignments = pd.read_csv(ASSIGNMENTS).set_index("assay")

    shared = wide.index.intersection(features.index).intersection(assignments.index)
    wide = wide.loc[shared]
    matrix = features.loc[shared, ROUTER_FEATURES].copy()
    matrix = matrix.fillna(matrix.median())
    groups = assignments.loc[shared, "protein_key"]

    best_action = wide.idxmax(axis=1)
    print(f"   {len(shared)} assays, {len(CANDIDATE_ACTIONS)} candidate strategies, "
          f"{groups.nunique()} proteins for grouped cross-validation")
    print("\n   how often each strategy is the per-assay best:")
    for action, count in best_action.value_counts().items():
        print(f"      {action:18s} {count:3d}  ({count / len(shared):.1%})")

    predicted = pd.Series(index=shared, dtype=object)
    splitter = GroupKFold(n_splits=min(10, groups.nunique()))
    for train_index, test_index in splitter.split(matrix, best_action, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, multi_class="multinomial", C=1.0),
        )
        model.fit(matrix.iloc[train_index], best_action.iloc[train_index])
        predicted.iloc[test_index] = model.predict(matrix.iloc[test_index])

    router_values = np.array([wide.loc[assay, predicted[assay]] for assay in shared], dtype=float)
    oracle_values = wide.max(axis=1).to_numpy()
    fixed_means = wide.mean()
    best_fixed_name = fixed_means.idxmax()
    best_fixed_values = wide[best_fixed_name].to_numpy()

    accuracy = float((predicted == best_action).mean())
    print(f"\n   router picks the per-assay best strategy in {accuracy:.1%} of assays")
    print(f"   always picking {best_fixed_name} would be best in "
          f"{(best_action == best_fixed_name).mean():.1%} of assays")

    print(f"\n   mean {EVALUATION_METRIC}")
    print(f"      learned router, cross-validated by protein   {router_values.mean():.4f}")
    print(f"      best fixed strategy, always {best_fixed_name:14s}  {best_fixed_values.mean():.4f}")
    print(f"      per-assay oracle                             {oracle_values.mean():.4f}")

    rows = []
    for name, comparator in [
        (f"always {best_fixed_name}", best_fixed_values),
        ("per-assay oracle", oracle_values),
    ]:
        difference = router_values - comparator
        low, high = bootstrap_ci(difference, rng)
        p_value = sign_permutation_p(difference, rng)
        print(f"\n      router minus {name:26s} {difference.mean():+.4f}"
              f"  95% CI [{low:+.4f}, {high:+.4f}]  P = {p_value:.3g}")
        rows.append(
            {
                "part": "B",
                "comparison": f"learned router minus {name}",
                "n_assays": len(shared),
                "mean_router": float(router_values.mean()),
                "mean_comparator": float(comparator.mean()),
                "difference": float(difference.mean()),
                "ci_low": low,
                "ci_high": high,
                "p_value": p_value,
            }
        )

    headroom = oracle_values.mean() - best_fixed_values.mean()
    captured = router_values.mean() - best_fixed_values.mean()
    print(f"\n   oracle headroom above the best fixed strategy   {headroom:+.4f}")
    print(f"   headroom the learned router captures            {captured:+.4f}"
          f"  ({captured / headroom:.1%})" if headroom > 0 else "")
    print("\n   The router is given the same evidence features the map uses, including a")
    print("   calibration gain computed from held-out performance, so this is an upper")
    print("   bound on what routing from these features can deliver.")

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    results = pd.concat([part_a(rng), part_b(rng)], ignore_index=True)
    destination = OUTPUT_DIR / "decision_map_completion.csv"
    results.to_csv(destination, index=False)
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
