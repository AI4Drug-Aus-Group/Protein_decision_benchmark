import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(os.environ.get(
    "PROTEIN_DECISION_PACKAGE", ROOT / "analysis_package"
)).resolve()
RESIDUALS = (
    PACKAGE_ROOT / "multi_mutant/double_residuals"
    / "strict_double_epistasis_residuals.csv"
)
OUTPUT_DIR = ROOT / "results" / "experiments"

REFERENCE_LINES = {
    "ProSST-2048": 0.419,
    "ESM3": 0.355,
    "GEMME": 0.316,
}
PUBLISHED_FULL_ADDITIVE = 0.730

BUDGETS = [20, 50, 100, 200, 500, 1000, 2000, 5000]
SEEDS = [0, 1, 2, 3, 4]
RANDOM_SEED = 20260916


def recover_singles(group: pd.DataFrame) -> pd.Series:
    first = group[["single_a", "score_single_a"]].rename(
        columns={"single_a": "mutant", "score_single_a": "score"}
    )
    second = group[["single_b", "score_single_b"]].rename(
        columns={"single_b": "mutant", "score_single_b": "score"}
    )
    both = pd.concat([first, second], ignore_index=True)
    return both.groupby("mutant").score.first()


def additive_metrics(group: pd.DataFrame, singles: pd.Series, measured: np.ndarray) -> dict:
    known = singles.loc[measured]
    baseline = float(known.mean())
    effects = known - baseline

    effect_a = group.single_a.map(effects).fillna(0.0).to_numpy()
    effect_b = group.single_b.map(effects).fillna(0.0).to_numpy()
    prediction = baseline + effect_a + effect_b
    observed = group.score_double.to_numpy()

    if np.std(prediction) == 0 or np.std(observed) == 0:
        return {"abs_spearman": np.nan, "top1pct_recall_at_100": np.nan, "best_top100_percentile": np.nan}

    result = {"abs_spearman": abs(spearmanr(prediction, observed).statistic)}

    order = np.argsort(-prediction, kind="mergesort")
    top_list = order[: min(100, len(order))]
    cut = np.quantile(observed, 0.99)
    true_top = observed >= cut
    n_true_top = int(true_top.sum())
    result["top1pct_recall_at_100"] = (
        float(true_top[top_list].sum() / n_true_top) if n_true_top > 0 else np.nan
    )
    best_in_list = float(observed[top_list].max())
    result["best_top100_percentile"] = float((observed <= best_in_list).mean())
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        RESIDUALS,
        usecols=["assay", "single_a", "single_b", "score_double", "score_single_a", "score_single_b"],
    )
    print(f"loaded {len(data):,} component-resolved double mutants from {data.assay.nunique()} assays")

    rng = np.random.default_rng(RANDOM_SEED)
    records = []

    for assay, group in data.groupby("assay", sort=False):
        singles = recover_singles(group)
        available = singles.index.to_numpy()
        full = additive_metrics(group, singles, available)
        records.append({"assay": assay, "budget": "all", "n_singles_used": len(available), **full})

        for budget in BUDGETS:
            if budget >= len(available):
                continue
            for seed in SEEDS:
                sampled = rng.choice(available, size=budget, replace=False)
                records.append(
                    {
                        "assay": assay,
                        "budget": budget,
                        "n_singles_used": budget,
                        **additive_metrics(group, singles, sampled),
                    }
                )

    frame = pd.DataFrame(records)
    per_assay = frame.groupby(["assay", "budget"], as_index=False).agg(
        abs_spearman=("abs_spearman", "mean"),
        top1pct_recall_at_100=("top1pct_recall_at_100", "mean"),
        best_top100_percentile=("best_top100_percentile", "mean"),
        n_singles_used=("n_singles_used", "max"),
    )

    singles_per_assay = per_assay[per_assay.budget == "all"].set_index("assay").n_singles_used

    reference = pd.read_csv(
        PACKAGE_ROOT / "multi_mutant"
        / "single_to_multi/single_to_multi_representative_metrics.csv"
    )
    reference = reference[reference.order_bucket == "double"]

    reported = reference[reference.method == "additive_all_singles"].set_index("assay").abs_spearman
    mine = per_assay[per_assay.budget == "all"].set_index("assay").abs_spearman
    shared = reported.index.intersection(mine.index)
    agreement = float(np.corrcoef(reported[shared], mine[shared])[0, 1])
    print(f"\nagreement between the recomputed all-singles additive control and the reported one:")
    print(f"   {len(shared)} shared assays, correlation {agreement:.4f}, "
          f"mean difference {float((mine[shared] - reported[shared]).mean()):+.4f}")

    print(f"\nreported value with all singles measured, 66 assays: {PUBLISHED_FULL_ADDITIVE:.3f}")

    metrics = [
        ("abs_spearman", "abs_spearman", "global ranking, absolute Spearman"),
        ("top1pct_recall_at_100", "top1pct_recall_at_100", "discovery, top-1% recall among the top 100"),
    ]

    for minimum in (100, 500):
        cohort = singles_per_assay[singles_per_assay >= minimum].index
        swept = [b for b in BUDGETS if b <= minimum]
        subset = per_assay[per_assay.assay.isin(cohort)]

        print(f"\n{'=' * 78}")
        print(f"Fixed cohort: {len(cohort)} assays with at least {minimum} measured single mutants")
        print(f"{'=' * 78}")

        for column, reference_column, label in metrics:
            curve = subset.groupby("budget")[column].mean()
            ordered = [b for b in swept if b in curve.index] + (["all"] if "all" in curve.index else [])

            print(f"\n   {label}")
            print(f"   {'singles measured':>18s} {'additive control':>18s}")
            for budget in ordered:
                print(f"   {str(budget):>18s} {curve.loc[budget]:>18.3f}")

            lines = {}
            for method in list(REFERENCE_LINES) + ["additive_all_singles"]:
                values = reference[(reference.method == method) & (reference.assay.isin(cohort))]
                if len(values):
                    lines[method] = float(values[reference_column].mean())
            print("\n   same cohort, reported values on double mutants:")
            for method, value in lines.items():
                print(f"      {method:22s} {value:.3f}")

            numeric = curve.loc[[b for b in ordered if b != "all"]]
            print("\n   crossover, smallest swept budget where the additive control overtakes each predictor")
            for method in REFERENCE_LINES:
                if method not in lines:
                    continue
                above = numeric[numeric >= lines[method]]
                if len(above):
                    print(f"      {method:22s} about {above.index[0]} measured single mutants")
                else:
                    print(f"      {method:22s} not reached by {numeric.index[-1]}, best swept {numeric.max():.3f}")

    destination = OUTPUT_DIR / "additive_control_crossover.csv"
    per_assay.to_csv(destination, index=False)
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
