import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


KEYS = ["assay", "seed", "budget", "regime"]
RAW_METRICS = ["spearman_all", "top1pct_recall_at_100", "best_top100_percentile"]
METRICS = ["abs_spearman_all", "spearman_all", "top1pct_recall_at_100", "best_top100_percentile"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean(values, rng, iterations):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    if valid.size == 0:
        return adjusted
    order = valid[np.argsort(pvalues[valid])]
    running = 0.0
    m = order.size
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def load_kermut(run_root):
    records = []
    files = sorted(run_root.glob("*/*.json"))
    for path in files:
        with open(path) as handle:
            record = json.load(handle)
        record["result_file"] = str(path)
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError(f"No Kermut result records found under {run_root}")
    duplicated = frame.duplicated(KEYS, keep=False)
    if duplicated.any():
        duplicate_keys = frame.loc[duplicated, KEYS].drop_duplicates().to_dict("records")
        raise RuntimeError(f"Duplicate Kermut result keys: {duplicate_keys[:10]}")
    return frame, files


def load_comparators(root):
    frames = []
    source_paths = []

    embedding_path = root / "results/few_shot_external/matched_budget_esm2_embedding_baselines_detail.csv"
    embedding = pd.read_csv(embedding_path)
    embedding = embedding[(embedding["regime"] == "mixed_random") & (embedding["status"] == "ok")].copy()
    embedding["comparison_method"] = embedding["model"].map({
        "esm2_embedding_gp": "Gaussian process on ESM2 embeddings",
        "esm2_embedding_ridge": "Ridge regression on ESM2 embeddings",
        "esm2_embedding_hist_gradient_boosting": "Gradient boosting on ESM2 embeddings",
        "esm2_embedding_xgboost": "XGBoost on ESM2 embeddings",
    })
    embedding = embedding[embedding["comparison_method"].notna()].copy()
    frames.append(embedding[KEYS + ["comparison_method", "n_test"] + RAW_METRICS])
    source_paths.append(embedding_path)

    calibration_path = root / "results/low_n/low_n_calibration_representative_metrics.percentile.csv"
    calibration = pd.read_csv(calibration_path)
    calibration = calibration[
        (calibration["regime"] == "mixed_random")
        & (calibration["model"] == "zscore_linear")
        & (calibration["zero_shot_method"] == "ProSST-2048")
    ].copy()
    calibration["comparison_method"] = "Linear calibration of ProSST-2048"
    calibration["best_top100_percentile"] = calibration["best_true_in_pred_top_100_percentile"]
    frames.append(calibration[KEYS + ["comparison_method", "n_test"] + RAW_METRICS])
    source_paths.append(calibration_path)

    multi_path = root / "results/low_n/low_n_multi_zscore_ridge_nongiant_metrics.percentile.csv"
    multi = pd.read_csv(multi_path)
    multi = multi[multi["regime"] == "mixed_random"].copy()
    multi["comparison_method"] = "Ridge regression on multiple zero-shot scores"
    multi["best_top100_percentile"] = multi["best_true_in_pred_top_100_percentile"]
    frames.append(multi[KEYS + ["comparison_method", "n_test"] + RAW_METRICS])
    source_paths.append(multi_path)

    linear_path = root / "results/low_n/low_n_sklearn_linear_nongiant_metrics.percentile.csv"
    linear = pd.read_csv(linear_path)
    linear = linear[(linear["regime"] == "mixed_random") & (linear["model"] == "ridge")].copy()
    linear["comparison_method"] = "Ridge regression on selected zero-shot scores"
    linear["best_top100_percentile"] = linear["best_true_in_pred_top_100_percentile"]
    frames.append(linear[KEYS + ["comparison_method", "n_test"] + RAW_METRICS])
    source_paths.append(linear_path)

    frame = pd.concat(frames, ignore_index=True)
    frame["abs_spearman_all"] = frame["spearman_all"].abs()
    duplicated = frame.duplicated(KEYS + ["comparison_method"], keep=False)
    if duplicated.any():
        duplicate_keys = frame.loc[duplicated, KEYS + ["comparison_method"]].drop_duplicates().to_dict("records")
        raise RuntimeError(f"Duplicate comparator keys: {duplicate_keys[:10]}")
    return frame, source_paths


def summarize_kermut(kermut_ok, rng, iterations):
    assay_budget = kermut_ok.groupby(["assay", "budget"], as_index=False).agg(
        n_successful_seeds=("seed", "nunique"),
        **{f"mean_{metric}": (metric, "mean") for metric in METRICS},
    )
    records = []
    for budget, frame in assay_budget.groupby("budget", sort=True):
        record = {
            "budget": int(budget),
            "n_assays": int(frame["assay"].nunique()),
            "n_successful_runs": int(frame["n_successful_seeds"].sum()),
            "n_assays_with_five_seeds": int((frame["n_successful_seeds"] == 5).sum()),
        }
        for metric in METRICS:
            values = frame[f"mean_{metric}"].to_numpy(float)
            low, high = bootstrap_mean(values, rng, iterations)
            record[f"mean_{metric}"] = np.nanmean(values)
            record[f"ci95_low_{metric}"] = low
            record[f"ci95_high_{metric}"] = high
        records.append(record)
    return assay_budget, pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-root", default="results/local_matched_reproduction/kermut/runs")
    parser.add_argument("--coverage", default="results/local_matched_reproduction/kermut_resource_coverage.csv")
    parser.add_argument("--out", default="results/local_matched_reproduction/comparison_successful_kermut")
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    run_root = root / args.run_root
    output = root / args.out
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    kermut_all, result_files = load_kermut(run_root)
    kermut_ok = kermut_all[kermut_all["status"] == "ok"].copy()
    kermut_ok["abs_spearman_all"] = kermut_ok["spearman_all"].abs()
    kermut_ok["comparison_method"] = "Kermut"
    comparators, source_paths = load_comparators(root)

    paired = kermut_ok[KEYS + ["n_test"] + METRICS].merge(
        comparators,
        on=KEYS,
        how="inner",
        suffixes=("_kermut", "_comparator"),
        validate="one_to_many",
    )
    paired["same_candidate_count"] = paired["n_test_kermut"] == paired["n_test_comparator"]
    mismatches = paired[~paired["same_candidate_count"]].copy()
    paired = paired[paired["same_candidate_count"]].copy()
    for metric in METRICS:
        paired[f"delta_{metric}"] = paired[f"{metric}_kermut"] - paired[f"{metric}_comparator"]

    aggregation = {
        "n_matched_seeds": ("seed", "nunique"),
        "mean_n_test": ("n_test_kermut", "mean"),
    }
    for metric in METRICS:
        aggregation[f"mean_kermut_{metric}"] = (f"{metric}_kermut", "mean")
        aggregation[f"mean_comparator_{metric}"] = (f"{metric}_comparator", "mean")
        aggregation[f"mean_delta_{metric}"] = (f"delta_{metric}", "mean")
    assay_budget = paired.groupby(
        ["assay", "budget", "regime", "comparison_method"], as_index=False
    ).agg(**aggregation)

    summary_records = []
    test_records = []
    for (budget, method), frame in assay_budget.groupby(["budget", "comparison_method"], sort=True):
        summary = {
            "budget": int(budget),
            "comparator": method,
            "n_assays": int(frame["assay"].nunique()),
            "n_matched_runs": int(frame["n_matched_seeds"].sum()),
            "n_assays_with_five_matched_seeds": int((frame["n_matched_seeds"] == 5).sum()),
        }
        for metric in METRICS:
            k_values = frame[f"mean_kermut_{metric}"].to_numpy(float)
            c_values = frame[f"mean_comparator_{metric}"].to_numpy(float)
            deltas = frame[f"mean_delta_{metric}"].to_numpy(float)
            summary[f"mean_kermut_{metric}"] = np.nanmean(k_values)
            summary[f"mean_comparator_{metric}"] = np.nanmean(c_values)
            summary[f"mean_delta_{metric}"] = np.nanmean(deltas)
            low, high = bootstrap_mean(deltas, rng, args.bootstrap_iterations)
            valid = deltas[np.isfinite(deltas)]
            if valid.size and np.any(valid != 0):
                statistic, pvalue = wilcoxon(valid, alternative="two-sided", zero_method="wilcox")
            else:
                statistic, pvalue = np.nan, np.nan
            test_records.append({
                "budget": int(budget),
                "comparator": method,
                "metric": metric,
                "n_assays": int(valid.size),
                "mean_delta_kermut_minus_comparator": np.nanmean(valid) if valid.size else np.nan,
                "median_delta_kermut_minus_comparator": np.nanmedian(valid) if valid.size else np.nan,
                "ci95_low_mean_delta": low,
                "ci95_high_mean_delta": high,
                "wilcoxon_statistic": statistic,
                "p_value": pvalue,
            })
        summary_records.append(summary)
    summary = pd.DataFrame(summary_records)
    tests = pd.DataFrame(test_records)
    tests["p_value_holm"] = holm_adjust(tests["p_value"].to_numpy(float))

    kermut_assay_budget, kermut_summary = summarize_kermut(kermut_ok, rng, args.bootstrap_iterations)

    status_counts = kermut_all["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="n_records")
    coverage = pd.read_csv(root / args.coverage)
    coverage_records = [{
        "quantity": "assay_budget_seed_combinations_expected",
        "value": int(len(coverage)),
    }, {
        "quantity": "combinations_with_complete_official_kermut_resources",
        "value": int(coverage["complete"].sum()),
    }, {
        "quantity": "saved_status_records",
        "value": int(len(kermut_all)),
    }, {
        "quantity": "successful_kermut_runs",
        "value": int(len(kermut_ok)),
    }, {
        "quantity": "assays_with_at_least_one_successful_run",
        "value": int(kermut_ok["assay"].nunique()),
    }, {
        "quantity": "candidate_count_mismatches_excluded",
        "value": int(len(mismatches)),
    }]
    coverage_summary = pd.DataFrame(coverage_records)
    comparator_coverage = assay_budget.groupby(["budget", "comparison_method"], as_index=False).agg(
        n_assays=("assay", "nunique"),
        n_matched_runs=("n_matched_seeds", "sum"),
        n_assays_with_five_matched_seeds=("n_matched_seeds", lambda values: int((values == 5).sum())),
        minimum_matched_seeds_per_assay=("n_matched_seeds", "min"),
        median_matched_seeds_per_assay=("n_matched_seeds", "median"),
        maximum_matched_seeds_per_assay=("n_matched_seeds", "max"),
    )

    kermut_all.to_csv(output / "kermut_status_records.csv", index=False)
    status_counts.to_csv(output / "kermut_status_counts.csv", index=False)
    coverage_summary.to_csv(output / "coverage_summary.csv", index=False)
    comparator_coverage.to_csv(output / "comparator_coverage.csv", index=False)
    mismatches.to_csv(output / "candidate_count_mismatches.csv", index=False)
    paired.to_csv(output / "paired_run_level.csv", index=False)
    assay_budget.to_csv(output / "paired_assay_budget_means.csv", index=False)
    summary.to_csv(output / "paired_method_summary.csv", index=False)
    tests.to_csv(output / "paired_statistical_tests.csv", index=False)
    kermut_assay_budget.to_csv(output / "kermut_assay_budget_means.csv", index=False)
    kermut_summary.to_csv(output / "kermut_successful_run_summary.csv", index=False)

    metadata = {
        "status": "successful_case_matched_analysis",
        "analysis_seed": args.seed,
        "bootstrap_iterations": args.bootstrap_iterations,
        "analysis_unit": "assay",
        "protocol": (
            "Only successful Kermut runs were analyzed. Comparisons were matched by assay, measurement budget, "
            "random seed and split regime, required identical held-out candidate counts, averaged available "
            "matched seeds within each assay and budget, and treated assays as independent units."
        ),
        "missing_data_handling": "Effect estimates use observed outcomes from successful Kermut runs with available comparator predictions.",
        "analysis_population": "Successful Kermut runs with available comparator predictions.",
        "n_result_json_files": len(result_files),
        "n_successful_kermut_runs": int(len(kermut_ok)),
        "n_assays_with_successful_kermut_runs": int(kermut_ok["assay"].nunique()),
        "source_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in source_paths + [root / args.coverage]
        },
    }
    with open(output / "analysis_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
