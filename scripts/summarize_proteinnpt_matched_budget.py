from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "local_matched_reproduction" / "comparison_proteinnpt_sampled_2000_len400"
DETAIL = ROOT / "results" / "local_matched_reproduction" / "proteinnpt_sampled_2000_len400" / "proteinnpt_matched_budget_detail.csv"


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def main() -> None:
    detail = pd.read_csv(DETAIL)
    broad = pd.read_csv(OUT / "method_summary_proteinnpt.csv")
    strict = pd.read_csv(OUT / "strict_paired_summary_by_comparator.csv")
    tests = pd.read_csv(OUT / "paired_tests_proteinnpt.csv")
    complete = pd.read_csv(OUT / "complete_assays_proteinnpt.csv")
    paired = pd.read_csv(OUT / "paired_run_level_proteinnpt.csv")

    metrics = ["spearman_all", "top1pct_recall_at_100", "best_top100_percentile"]
    self_assay = detail[detail["status"] == "ok"].groupby(["assay", "budget"], as_index=False)[metrics].mean()
    self_summary = self_assay.groupby("budget", as_index=False).agg(
        assays=("assay", "nunique"),
        mean_spearman=("spearman_all", "mean"),
        mean_top1pct_recall_at_100=("top1pct_recall_at_100", "mean"),
        mean_best_top100_percentile=("best_top100_percentile", "mean"),
    )
    broad_rank_rows = []
    for budget, group in broad.groupby("budget"):
        for metric, label in [
            ("mean_spearman_all", "Spearman"),
            ("mean_top1pct_recall_at_100", "Top-1% recall at 100"),
            ("mean_best_top100_percentile", "Best-top-100 percentile"),
        ]:
            top = group.sort_values(metric, ascending=False).head(3)
            broad_rank_rows.append({
                "Budget": int(budget),
                "Metric": label,
                "Top methods": "; ".join(f"{row.method} ({fmt(getattr(row, metric))})" for row in top.itertuples(index=False)),
            })
    broad_rank_table = pd.DataFrame(broad_rank_rows)

    key_comparators = [
        "Linear calibration of ProSST-2048",
        "Ridge regression on selected zero-shot scores",
        "Ridge regression on multiple zero-shot scores",
        "Kermut",
        "Gaussian process on ESM2 embeddings",
        "XGBoost on ESM2 embeddings",
    ]
    strict_key = strict[strict["comparator"].isin(key_comparators)].copy()
    complete_set = set(complete["assay"])
    strict_set = set(paired["assay"])
    common_set = set(pd.read_csv(OUT / "method_assay_means_proteinnpt.csv")["assay"])

    outputs = {
        "proteinnpt_assay_mean_metrics.csv": self_assay,
        "proteinnpt_summary.csv": self_summary,
        "proteinnpt_method_rank_summary.csv": broad_rank_table,
        "proteinnpt_comparator_effects.csv": strict_key,
        "proteinnpt_spearman_tests.csv": tests[
            tests["metric"].eq("spearman_all") & tests["comparator"].isin(key_comparators)
        ],
        "proteinnpt_comparison_cohorts.csv": pd.DataFrame({
            "assay": sorted(complete_set),
            "all_methods_available": [assay in common_set for assay in sorted(complete_set)],
            "paired_comparison_available": [assay in strict_set for assay in sorted(complete_set)],
        }),
    }
    for name, frame in outputs.items():
        destination = OUT / name
        frame.to_csv(destination, index=False)
        print(destination)



if __name__ == "__main__":
    main()
