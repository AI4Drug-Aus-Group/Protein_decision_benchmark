from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
KERMUT_SCRIPT_DIR = ROOT / "results" / "local_matched_reproduction" / "scripts"
KERMUT_REFERENCE = ROOT / "external_repos" / "kermut" / "data" / "DMS_substitutions.csv"
DMS_DIR = ROOT / "proteingym" / "extracted" / "DMS_ProteinGym_substitutions"


def mutation_order(mutant: str) -> int:
    if not isinstance(mutant, str) or mutant in {"", "WT", "wildtype", "_wt"}:
        return 0
    return mutant.count(":") + 1


def order_bucket(order: int) -> str:
    if order == 1:
        return "single"
    if order == 2:
        return "double"
    if order >= 3:
        return "higher"
    return "wt_or_unknown"


def top1_recall(y_true: np.ndarray, y_pred: np.ndarray, k: int = 100) -> float:
    if len(y_true) < 3:
        return math.nan
    k = min(k, len(y_true))
    predicted = set(np.argsort(-y_pred)[:k].tolist())
    true_top = set(np.argsort(-y_true)[: max(1, math.ceil(len(y_true) * 0.01))].tolist())
    return len(predicted & true_top) / len(true_top)


def best_top100_percentile(y_true: np.ndarray, y_pred: np.ndarray, assay: str) -> float:
    if len(y_true) == 0:
        return math.nan
    selected = np.argsort(-y_pred)[: min(100, len(y_pred))]
    best = float(np.max(y_true[selected]))
    ref = pd.read_csv(DMS_DIR / assay, usecols=["DMS_score"])
    values = pd.to_numeric(ref["DMS_score"], errors="coerce").dropna().sort_values().to_numpy(float)
    return float(np.searchsorted(values, best, side="right") / len(values)) if len(values) else math.nan


def metric_record(method: str, record: dict, mutants: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, bucket: str) -> dict[str, object]:
    mask = np.array([order_bucket(mutation_order(str(m))) == bucket for m in mutants], dtype=bool)
    yt = y_true[mask]
    yp = y_pred[mask]
    rho = spearmanr(yp, yt).statistic if len(yt) >= 3 else math.nan
    return {
        "method": method,
        "assay": record["assay"],
        "seed": int(record["seed"]),
        "budget": int(record["budget"]),
        "regime": record["regime"],
        "order_bucket": bucket,
        "n_variants": int(len(yt)),
        "spearman": float(rho) if np.isfinite(rho) else math.nan,
        "abs_spearman": float(abs(rho)) if np.isfinite(rho) else math.nan,
        "top1pct_recall_at_100": top1_recall(yt, yp),
        "best_top100_percentile": best_top100_percentile(yt, yp, record["assay"]),
    }


def summarize_assay_level(detail: pd.DataFrame) -> pd.DataFrame:
    assay_means = (
        detail.groupby(["method", "budget", "order_bucket", "assay"], as_index=False)
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
        assay_means.groupby(["method", "budget", "order_bucket"], as_index=False)
        .agg(
            n_assays=("assay", "nunique"),
            n_assays_with_five_seeds=("n_seeds", lambda x: int((x == 5).sum())),
            mean_n_variants=("mean_n_variants", "mean"),
            mean_spearman=("mean_spearman", "mean"),
            mean_abs_spearman=("mean_abs_spearman", "mean"),
            mean_top1pct_recall_at_100=("mean_top1pct_recall_at_100", "mean"),
            mean_best_top100_percentile=("mean_best_top100_percentile", "mean"),
        )
    )
    return summary, assay_means


def load_kermut_records(root: Path) -> pd.DataFrame:
    if str(KERMUT_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(KERMUT_SCRIPT_DIR))
    from run_kermut_matched_budget import load_assay

    reference = pd.read_csv(KERMUT_REFERENCE)
    rows = []
    assay_cache = {}
    for path in sorted((root / "runs").glob("*/*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "ok":
            continue
        npz_path = path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        assay = record["assay"]
        if assay not in assay_cache:
            table, _, _ = load_assay(assay, reference)
            assay_cache[assay] = table
        table = assay_cache[assay]
        data = np.load(npz_path)
        test_indices = data["test_indices"].astype(int)
        mutants = table.loc[test_indices, "mutant"].astype(str).to_numpy()
        y_true = data["y_true"].astype(float)
        y_pred = data["y_pred"].astype(float)
        for bucket in ["double", "higher"]:
            rows.append(metric_record("Kermut", record, mutants, y_true, y_pred, bucket))
    return pd.DataFrame(rows)


def load_proteinnpt_records(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "runs").glob("*/*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "ok":
            continue
        npz_path = Path(record.get("npz_file", path.with_suffix(".npz")))
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        mutants = data["mutant"].astype(str)
        y_true = data["y_true"].astype(float)
        y_pred = data["y_pred"].astype(float)
        for bucket in ["double", "higher"]:
            rows.append(metric_record("ProteinNPT", record, mutants, y_true, y_pred, bucket))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kermut-root", type=Path, default=None)
    parser.add_argument("--proteinnpt-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    if args.kermut_root is not None and args.kermut_root.exists():
        frames.append(load_kermut_records(args.kermut_root.resolve()))
    if args.proteinnpt_root is not None and args.proteinnpt_root.exists():
        frames.append(load_proteinnpt_records(args.proteinnpt_root.resolve()))
    if not frames:
        raise RuntimeError("no input records found")
    detail = pd.concat(frames, ignore_index=True)
    summary, assay_means = summarize_assay_level(detail)
    args.out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out / "external_single_only_to_multi_detail.csv", index=False)
    assay_means.to_csv(args.out / "external_single_only_to_multi_assay_means.csv", index=False)
    summary.to_csv(args.out / "external_single_only_to_multi_summary.csv", index=False)
    metadata = {
        "status": "ok",
        "n_detail_rows": int(len(detail)),
        "methods": sorted(detail["method"].unique().tolist()),
        "regime": sorted(detail["regime"].unique().tolist()),
        "order_buckets": sorted(detail["order_bucket"].unique().tolist()),
    }
    (args.out / "external_single_only_to_multi_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
