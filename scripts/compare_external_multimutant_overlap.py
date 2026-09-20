from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_detail(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["assay"] = df["assay"].astype(str)
    df["seed"] = df["seed"].astype(int)
    df["budget"] = df["budget"].astype(int)
    df["spearman"] = pd.to_numeric(df["spearman"], errors="coerce")
    df["abs_spearman"] = pd.to_numeric(df["abs_spearman"], errors="coerce")
    return df


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = df[df["budget"].eq(20) & df["spearman"].notna() & df["n_variants"].ge(3)].copy()
    assay_means = (
        usable.groupby(["method", "order_bucket", "assay"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_n_variants=("n_variants", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_abs_spearman=("abs_spearman", "mean"),
            mean_top1pct_recall_at_100=("top1pct_recall_at_100", "mean"),
            mean_best_top100_percentile=("best_top100_percentile", "mean"),
        )
    )
    summary = (
        assay_means.groupby(["method", "order_bucket"], as_index=False)
        .agg(
            n_assays=("assay", "nunique"),
            n_assays_with_five_seeds=("n_seeds", lambda values: int((values == 5).sum())),
            mean_n_variants=("mean_n_variants", "mean"),
            mean_spearman=("mean_spearman", "mean"),
            mean_abs_spearman=("mean_abs_spearman", "mean"),
            mean_top1pct_recall_at_100=("mean_top1pct_recall_at_100", "mean"),
            mean_best_top100_percentile=("mean_best_top100_percentile", "mean"),
        )
    )
    return usable, assay_means, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proteinnpt-detail", type=Path, required=True)
    parser.add_argument("--kermut-detail", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pnpt = read_detail(args.proteinnpt_detail)
    kermut = read_detail(args.kermut_detail)
    assays = sorted(pnpt["assay"].unique().tolist())
    combined = pd.concat(
        [
            pnpt[pnpt["assay"].isin(assays)].copy(),
            kermut[kermut["assay"].isin(assays)].copy(),
        ],
        ignore_index=True,
    )
    detail, assay_means, summary = summarize(combined)
    args.out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out / "external_multimutant_overlap_detail.csv", index=False)
    assay_means.to_csv(args.out / "external_multimutant_overlap_assay_means.csv", index=False)
    summary.to_csv(args.out / "external_multimutant_overlap_summary.csv", index=False)
    metadata = {
        "status": "ok",
        "assays": assays,
        "n_detail_rows": int(len(detail)),
        "methods": sorted(detail["method"].unique().tolist()),
        "order_buckets": sorted(detail["order_bucket"].unique().tolist()),
    }
    (args.out / "external_multimutant_overlap_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
