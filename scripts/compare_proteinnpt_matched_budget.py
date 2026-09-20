from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETAIL = ROOT / "results" / "local_matched_reproduction" / "proteinnpt" / "proteinnpt_matched_budget_detail.csv"
DEFAULT_OUT = ROOT / "results" / "local_matched_reproduction" / "proteinnpt" / "comparison"
KEYS = ["assay", "seed", "budget", "regime"]
METRICS = ["spearman_all", "top1pct_recall_at_100", "best_top100_percentile"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, iterations: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def holm_adjust(values: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(values, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    if valid.size == 0:
        return adjusted
    order = valid[np.argsort(pvalues[valid])]
    running = 0.0
    total = order.size
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def load_score_distributions(dms_dir: Path) -> dict[str, np.ndarray]:
    distributions = {}
    for path in sorted(dms_dir.glob("*.csv")):
        scores = pd.to_numeric(pd.read_csv(path, usecols=["DMS_score"])["DMS_score"], errors="coerce").to_numpy(float)
        scores = np.sort(scores[np.isfinite(scores)])
        distributions[path.name] = scores
    return distributions


def assign_full_assay_percentile(
    frame: pd.DataFrame,
    score_distributions: dict[str, np.ndarray],
    best_value_column: str,
) -> pd.DataFrame:
    if best_value_column not in frame.columns:
        raise ValueError(f"missing best selected value column: {best_value_column}")
    output = frame.copy()
    percentiles = []
    for assay, best_value in output[["assay", best_value_column]].itertuples(index=False, name=None):
        scores = score_distributions.get(str(assay))
        try:
            best = float(best_value)
        except (TypeError, ValueError):
            best = np.nan
        if scores is None or scores.size == 0 or not np.isfinite(best):
            percentiles.append(np.nan)
        else:
            percentiles.append(float(np.searchsorted(scores, best, side="right") / scores.size))
    output["best_top100_percentile"] = percentiles
    return output


def load_comparators(
    root: Path,
    score_distributions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, list[Path]]:
    frames = []
    sources = []

    embedding_path = root / "results" / "few_shot_external" / "matched_budget_esm2_embedding_baselines_detail.csv"
    if embedding_path.exists():
        embedding = pd.read_csv(embedding_path)
        embedding = embedding[(embedding["regime"] == "mixed_random") & (embedding["status"] == "ok")].copy()
        embedding = assign_full_assay_percentile(embedding, score_distributions, "best_true_in_pred_top100")
        embedding["comparison_method"] = embedding["model"].map({
            "esm2_embedding_gp": "Gaussian process on ESM2 embeddings",
            "esm2_embedding_ridge": "Ridge regression on ESM2 embeddings",
            "esm2_embedding_hist_gradient_boosting": "Gradient boosting on ESM2 embeddings",
            "esm2_embedding_xgboost": "XGBoost on ESM2 embeddings",
        })
        frames.append(embedding[KEYS + ["comparison_method", "n_test"] + METRICS])
        sources.append(embedding_path)

    calibration_path = root / "results" / "low_n" / "low_n_calibration_representative_metrics.percentile.csv"
    calibration = pd.read_csv(calibration_path)
    calibration = calibration[
        (calibration["regime"] == "mixed_random")
        & (calibration["model"] == "zscore_linear")
        & (calibration["zero_shot_method"] == "ProSST-2048")
    ].copy()
    calibration = assign_full_assay_percentile(calibration, score_distributions, "best_true_in_pred_top_100")
    calibration["comparison_method"] = "Linear calibration of ProSST-2048"
    frames.append(calibration[KEYS + ["comparison_method", "n_test"] + METRICS])
    sources.append(calibration_path)

    multi_path = root / "results" / "low_n" / "low_n_multi_zscore_ridge_nongiant_metrics.percentile.csv"
    multi = pd.read_csv(multi_path)
    multi = multi[multi["regime"] == "mixed_random"].copy()
    multi = assign_full_assay_percentile(multi, score_distributions, "best_true_in_pred_top_100")
    multi["comparison_method"] = "Ridge regression on multiple zero-shot scores"
    frames.append(multi[KEYS + ["comparison_method", "n_test"] + METRICS])
    sources.append(multi_path)

    linear_path = root / "results" / "low_n" / "low_n_sklearn_linear_nongiant_metrics.percentile.csv"
    linear = pd.read_csv(linear_path)
    linear = linear[(linear["regime"] == "mixed_random") & (linear["model"] == "ridge")].copy()
    linear = assign_full_assay_percentile(linear, score_distributions, "best_true_in_pred_top_100")
    linear["comparison_method"] = "Ridge regression on selected zero-shot scores"
    frames.append(linear[KEYS + ["comparison_method", "n_test"] + METRICS])
    sources.append(linear_path)

    kermut_path = root / "results" / "local_matched_reproduction" / "kermut" / "runs"
    kermut_records = []
    if kermut_path.exists():
        for path in sorted(kermut_path.glob("*/*.json")):
            with path.open() as handle:
                record = json.load(handle)
            if record.get("status") == "ok":
                record["comparison_method"] = "Kermut"
                record["source_json"] = str(path)
                kermut_records.append(record)
        if kermut_records:
            kermut = pd.DataFrame(kermut_records)
            kermut = assign_full_assay_percentile(kermut, score_distributions, "best_true_in_pred_top_100")
            frames.append(kermut[KEYS + ["comparison_method", "n_test"] + METRICS])

    return pd.concat(frames, ignore_index=True), sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare collected ProteinNPT matched-budget results with existing project baselines.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    score_distributions = load_score_distributions(root / "proteingym" / "extracted" / "DMS_ProteinGym_substitutions")
    pnpt_all = pd.read_csv(args.detail)
    pnpt_all = assign_full_assay_percentile(pnpt_all, score_distributions, "best_true_in_pred_top_100")
    pnpt = pnpt_all[pnpt_all["status"] == "ok"].copy()
    pnpt["comparison_method"] = "ProteinNPT"
    expected = {(seed, budget) for seed in range(5) for budget in (20, 50, 100)}
    observed = pnpt.groupby("assay").apply(
        lambda frame: set(zip(frame["seed"].astype(int), frame["budget"].astype(int))),
        include_groups=False,
    )
    complete_assays = sorted(observed[observed.map(lambda value: value == expected)].index)
    pnpt_complete = pnpt[pnpt["assay"].isin(complete_assays)].copy()
    comparators, sources = load_comparators(root, score_distributions)
    comparators = comparators[comparators["assay"].isin(complete_assays)].copy()

    paired = pnpt_complete[KEYS + ["n_test"] + METRICS].merge(
        comparators,
        on=KEYS,
        how="inner",
        suffixes=("_proteinnpt", "_comparator"),
        validate="one_to_many",
    )
    paired["same_candidate_count"] = paired["n_test_proteinnpt"] == paired["n_test_comparator"]
    n_mismatches = int((~paired["same_candidate_count"]).sum())
    paired = paired[paired["same_candidate_count"]].copy()
    for metric in METRICS:
        paired[f"delta_{metric}"] = paired[f"{metric}_proteinnpt"] - paired[f"{metric}_comparator"]

    long = pd.concat(
        [
            pnpt_complete[KEYS + ["comparison_method", "n_test"] + METRICS],
            comparators[KEYS + ["comparison_method", "n_test"] + METRICS],
        ],
        ignore_index=True,
    )
    methods = sorted(long["comparison_method"].dropna().unique())
    counts = long.groupby(["assay", "budget", "comparison_method"]).size().unstack(fill_value=0)
    valid_index = counts.index[(counts.reindex(columns=methods, fill_value=0) == 5).all(axis=1)]
    valid_pairs = set(valid_index.tolist())
    long = long[long.apply(lambda row: (row["assay"], row["budget"]) in valid_pairs, axis=1)].copy()

    assay_means = long.groupby(["assay", "budget", "comparison_method"], as_index=False)[METRICS].mean()
    rng = np.random.default_rng(args.seed)
    summary_rows = []
    for (budget, method), frame in assay_means.groupby(["budget", "comparison_method"], sort=True):
        record = {"budget": int(budget), "method": method, "n_assays": int(frame["assay"].nunique())}
        for metric in METRICS:
            values = frame[metric].to_numpy(float)
            low, high = bootstrap_mean(values, rng, args.bootstrap_iterations)
            record[f"mean_{metric}"] = float(np.nanmean(values)) if values.size else np.nan
            record[f"ci95_low_{metric}"] = low
            record[f"ci95_high_{metric}"] = high
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows)

    paired_assay = paired.groupby(["assay", "budget", "comparison_method"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        **{f"mean_delta_{metric}": (f"delta_{metric}", "mean") for metric in METRICS},
    )
    paired_assay = paired_assay[paired_assay["n_seeds"] == 5].copy()
    test_rows = []
    for (budget, method), frame in paired_assay.groupby(["budget", "comparison_method"], sort=True):
        for metric in METRICS:
            values = frame[f"mean_delta_{metric}"].dropna().to_numpy(float)
            if values.size and np.any(values != 0):
                statistic, pvalue = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
            else:
                statistic, pvalue = np.nan, np.nan
            low, high = bootstrap_mean(values, rng, args.bootstrap_iterations)
            test_rows.append({
                "budget": int(budget),
                "comparator": method,
                "metric": metric,
                "n_assays": int(values.size),
                "mean_delta_proteinnpt_minus_comparator": float(np.nanmean(values)) if values.size else np.nan,
                "median_delta_proteinnpt_minus_comparator": float(np.nanmedian(values)) if values.size else np.nan,
                "ci95_low_mean_delta": low,
                "ci95_high_mean_delta": high,
                "wilcoxon_statistic": statistic,
                "p_value": pvalue,
            })
    tests = pd.DataFrame(test_rows)
    tests["p_value_holm"] = holm_adjust(tests["p_value"].to_numpy(float)) if not tests.empty else []

    coverage = pd.DataFrame([
        {"quantity": "saved_proteinnpt_status_records", "value": int(len(pnpt_all))},
        {"quantity": "saved_successful_proteinnpt_runs", "value": int(len(pnpt))},
        {"quantity": "assays_with_at_least_one_successful_proteinnpt_run", "value": int(pnpt["assay"].nunique()) if not pnpt.empty else 0},
        {"quantity": "assays_complete_for_all_five_seeds_and_three_budgets", "value": int(len(complete_assays))},
        {"quantity": "candidate_count_mismatches_in_joined_rows", "value": n_mismatches},
    ])

    pnpt_all.to_csv(out / "proteinnpt_status_records.csv", index=False)
    pd.DataFrame({"assay": complete_assays}).to_csv(out / "complete_assays_proteinnpt.csv", index=False)
    paired.to_csv(out / "paired_run_level_proteinnpt.csv", index=False)
    assay_means.to_csv(out / "method_assay_means_proteinnpt.csv", index=False)
    summary.to_csv(out / "method_summary_proteinnpt.csv", index=False)
    paired_assay.to_csv(out / "paired_assay_differences_proteinnpt.csv", index=False)
    tests.to_csv(out / "paired_tests_proteinnpt.csv", index=False)
    coverage.to_csv(out / "coverage_proteinnpt.csv", index=False)
    metadata = {
        "status": "comparison_from_collected_proteinnpt_results",
        "n_complete_assays": len(complete_assays),
        "complete_assays": complete_assays,
        "bootstrap_iterations": args.bootstrap_iterations,
        "analysis_seed": args.seed,
        "comparison_protocol": "ProteinNPT and comparators are joined by assay, seed, budget and regime; candidate counts must match; best-top-100 percentiles are recomputed against the complete assay candidate pool; seed-level values are averaged within assay before assay-level summaries.",
        "best_top100_percentile_reference": "complete_assay_candidate_pool",
        "source_sha256": {str(path.relative_to(root)): sha256(path) for path in [args.detail] + sources if path.exists()},
    }
    with (out / "metadata_proteinnpt.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
