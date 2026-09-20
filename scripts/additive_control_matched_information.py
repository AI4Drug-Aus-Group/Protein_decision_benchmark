import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(os.environ.get(
    "PROTEIN_DECISION_PACKAGE", ROOT / "analysis_package"
)).resolve()
LOW_N_METRICS = (
    PACKAGE_ROOT / "low_measurement_metrics.csv"
)
OUTPUT_DIR = ROOT / "results" / "experiments"

REGIME = "single_only"
OUTCOME = "spearman_doubles"
BUDGETS = [20, 50, 100]
BOOTSTRAP_RESAMPLES = 5000
PERMUTATIONS = 100_000
RANDOM_SEED = 20260916

LEADING_METHODS = [
    "ProSST-2048",
    "VenusREM",
    "S3F_MSA",
    "S2F_MSA",
    "ESM3",
    "GEMME",
    "TranceptEVE_L",
    "PoET",
]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple:
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_permutation_p(differences: np.ndarray, rng: np.random.Generator) -> float:
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    usecols = ["assay", "seed", "budget", "regime", "model", "zero_shot_method", "n_test_doubles", OUTCOME]
    raw = pd.read_csv(LOW_N_METRICS, usecols=usecols)
    raw = raw[(raw.regime == REGIME) & raw[OUTCOME].notna()]

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for budget in BUDGETS:
        at_budget = raw[raw.budget == budget]

        additive = (
            at_budget[at_budget.model == "additive_lookup"]
            .groupby("assay")[OUTCOME]
            .mean()
            .rename("additive")
        )

        print(f"\n{'=' * 78}\nBudget {budget} measured single mutants, evaluated on held-out double mutants")
        print(f"{'=' * 78}")
        print(f"additive control defined on {len(additive)} assays, mean {additive.mean():.3f}")

        budget_rows = []
        for method in LEADING_METHODS:
            alone = (
                at_budget[(at_budget.model == "zscore_linear") & (at_budget.zero_shot_method == method)]
                .groupby("assay")[OUTCOME]
                .mean()
                .rename("alone")
            )
            combined = (
                at_budget[(at_budget.model == "additive_plus_zscore") & (at_budget.zero_shot_method == method)]
                .groupby("assay")[OUTCOME]
                .mean()
                .rename("combined")
            )
            joined = pd.concat([additive, alone, combined], axis=1).dropna()
            if len(joined) < 5:
                continue

            for contrast, values in [
                ("model alone minus additive alone", joined.alone - joined.additive),
                ("additive plus model minus additive alone", joined.combined - joined.additive),
                ("additive plus model minus model alone", joined.combined - joined.alone),
            ]:
                array = values.to_numpy(dtype=float)
                low, high = bootstrap_ci(array, rng)
                budget_rows.append(
                    {
                        "budget": budget,
                        "method": method,
                        "contrast": contrast,
                        "n_assays": len(joined),
                        "mean_additive": float(joined.additive.mean()),
                        "mean_alone": float(joined.alone.mean()),
                        "mean_combined": float(joined.combined.mean()),
                        "mean_difference": float(array.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "raw_p": sign_permutation_p(array, rng),
                    }
                )

        frame = pd.DataFrame(budget_rows)
        frame["bh_adjusted_p"] = benjamini_hochberg(frame.raw_p.to_numpy())
        rows.append(frame)

        for contrast in frame.contrast.unique():
            print(f"\n   {contrast}")
            block = frame[frame.contrast == contrast].sort_values("mean_difference", ascending=False)
            for row in block.itertuples():
                flag = "significant" if row.bh_adjusted_p < 0.05 else ""
                print(
                    f"      {row.method:16s} n={row.n_assays:3d}  {row.mean_difference:+.4f}"
                    f"  95% CI [{row.ci_low:+.4f}, {row.ci_high:+.4f}]"
                    f"  adj P = {row.bh_adjusted_p:.3g}  {flag}"
                )

    combined = pd.concat(rows, ignore_index=True)
    destination = OUTPUT_DIR / "additive_control_matched_information.csv"
    combined.to_csv(destination, index=False)
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
