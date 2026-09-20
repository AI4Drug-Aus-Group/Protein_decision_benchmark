import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(os.environ.get(
    "PROTEIN_DECISION_PACKAGE", ROOT / "analysis_package"
)).resolve()
PACKAGE = PACKAGE_ROOT
RESIDUALS = (
    PACKAGE
    / "multi_mutant/double_residuals"
    / "strict_double_epistasis_residuals.csv"
)
ASSIGNMENTS = (
    PACKAGE
    / "decision_map/final_assignments/assay_strategy_recommendations.csv"
)
OUTPUT_DIR = ROOT / "results" / "experiments"

ITERATIONS = 2000
RANDOM_SEED = 20260916

CALIBRATION_GAIN = 0.02
STRUCTURE_GAIN = 0.10
THRESHOLD_FACTOR_RANGE = (0.8, 1.2)

PROXY_SHIFT_RANGE = (-0.25, 0.25)
PROXY_GRID = np.linspace(PROXY_SHIFT_RANGE[0], PROXY_SHIFT_RANGE[1], 101)

CATEGORIES = [
    "calibrate_strong_zero_shot_plm",
    "measure_or_select_double_mutants",
    "use_structure_aware_scores",
    "use_msa_or_evolutionary_models",
    "additive_or_simple_ridge_first",
]


def build_epistasis_grid() -> pd.DataFrame:
    data = pd.read_csv(
        RESIDUALS,
        usecols=["assay", "score_double", "score_single_a", "score_single_b", "wt_proxy"],
    )
    print(f"loaded {len(data):,} component-resolved double mutants from {data.assay.nunique()} assays")

    rows = {}
    for assay, group in data.groupby("assay", sort=False):
        double = group.score_double.to_numpy()
        first = group.score_single_a.to_numpy()
        second = group.score_single_b.to_numpy()
        proxy = group.wt_proxy.to_numpy()

        iqr = float(np.subtract(*np.percentile(double, [75, 25])))
        if iqr <= 0:
            continue
        base_residual = double - (first + second - proxy)
        rows[assay] = np.array(
            [float(np.median(np.abs(base_residual - shift * iqr)) / iqr) for shift in PROXY_GRID]
        )

    grid = pd.DataFrame(rows, index=PROXY_GRID).T
    grid.index.name = "assay"
    print(f"built normalized-epistasis curves for {len(grid)} assays over {len(PROXY_GRID)} proxy shifts")
    return grid


def assign(features: pd.DataFrame, epistasis: pd.Series, thresholds: dict) -> pd.Series:
    gain = features.low_n20_best_plm_gain_over_additive
    structure = features.structure_gain
    depth = features.official_MSA_Neff_L
    doubles = features.n_doubles

    result = pd.Series("additive_or_simple_ridge_first", index=features.index, dtype=object)
    remaining = pd.Series(True, index=features.index)

    hit = remaining & gain.notna() & (gain >= thresholds["calibration"])
    result[hit] = "calibrate_strong_zero_shot_plm"
    remaining &= ~hit

    epi = epistasis.reindex(features.index)
    hit = remaining & epi.notna() & (epi >= thresholds["epistasis"]) & doubles.notna() & (doubles > 0)
    result[hit] = "measure_or_select_double_mutants"
    remaining &= ~hit

    hit = remaining & structure.notna() & (structure >= thresholds["structure"])
    result[hit] = "use_structure_aware_scores"
    remaining &= ~hit

    hit = remaining & depth.notna() & (depth >= thresholds["depth"])
    result[hit] = "use_msa_or_evolutionary_models"
    remaining &= ~hit

    return result


def run(features: pd.DataFrame, grid: pd.DataFrame, perturb_proxy: bool, label: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    baseline = features.recommended_strategy
    proteins = features.protein_key.unique()
    base_column = int(np.argmin(np.abs(PROXY_GRID)))

    retained = pd.DataFrame(0, index=features.index, columns=["hits", "draws"], dtype=float)

    for _ in range(ITERATIONS):
        sampled = rng.choice(proteins, size=len(proteins), replace=True)
        in_sample = features.protein_key.isin(set(sampled))

        if perturb_proxy:
            shift = rng.uniform(*PROXY_SHIFT_RANGE)
            column = int(np.argmin(np.abs(PROXY_GRID - shift)))
        else:
            column = base_column
        epistasis = grid.iloc[:, column]

        resampled_epistasis = epistasis.reindex(features.index[in_sample]).dropna()
        resampled_depth = features.official_MSA_Neff_L[in_sample].dropna()
        if len(resampled_epistasis) < 3 or len(resampled_depth) < 3:
            continue

        thresholds = {
            "calibration": CALIBRATION_GAIN * rng.uniform(*THRESHOLD_FACTOR_RANGE),
            "structure": STRUCTURE_GAIN * rng.uniform(*THRESHOLD_FACTOR_RANGE),
            "epistasis": float(np.quantile(resampled_epistasis.to_numpy(), 2 / 3, method="lower")),
            "depth": float(np.quantile(resampled_depth.to_numpy(), 2 / 3, method="lower")),
        }

        assigned = assign(features, epistasis, thresholds)
        retained["hits"] += (assigned == baseline).astype(float)
        retained["draws"] += 1.0

    features = features.copy()
    features["retention"] = retained.hits / retained.draws
    summary = (
        features.groupby("recommended_strategy")
        .retention.agg(["count", "mean", "median", "min"])
        .reindex(CATEGORIES)
    )
    print(f"\n{label}")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"   overall mean retention {features.retention.mean():.3f}")
    return features[["assay", "protein_key", "recommended_strategy", "retention"]].assign(analysis=label)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(ASSIGNMENTS).set_index("assay", drop=False)
    grid = build_epistasis_grid()

    stored = features.normalized_epistasis_strength.dropna()
    reconstructed = grid.iloc[:, int(np.argmin(np.abs(PROXY_GRID)))]
    shared = stored.index.intersection(reconstructed.index)
    difference = (stored[shared] - reconstructed[shared]).abs().max()
    print(f"\nreconstruction check against the reported normalized epistasis: max abs difference {difference:.2e}")
    if difference > 1e-6:
        print("   WARNING: the reconstruction does not match, treat the numbers below with caution")

    published = run(features, grid, perturb_proxy=False, label="published axes only, proxy held fixed")
    extended = run(features, grid, perturb_proxy=True, label="published axes plus a perturbed wild-type proxy")

    combined = pd.concat([published, extended], ignore_index=True)
    destination = OUTPUT_DIR / "decision_map_proxy_sensitivity.csv"
    combined.to_csv(destination, index=False)
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
