from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
KERMUT_SCRIPT_DIR = ROOT / "results" / "local_matched_reproduction" / "scripts"
DMS_DIR = ROOT / "proteingym" / "extracted" / "DMS_ProteinGym_substitutions"
KERMUT_REFERENCE = ROOT / "external_repos" / "kermut" / "data" / "DMS_substitutions.csv"


POLICY_LABELS = {
    "random_doubles": "random double-mutant selection",
    "mutation_coverage": "mutation-coverage selection",
    "extreme_additive_prediction": "extreme single-mutant-sum selection",
    "additive_versus_calibrated_prosst_disagreement": "additive-versus-calibrated-score disagreement",
    "oracle_absolute_epistatic_residual": "oracle epistatic-residual selection",
}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return math.nan
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else math.nan


def top1_recall(y_true: np.ndarray, y_pred: np.ndarray, k: int = 100) -> float:
    if len(y_true) < 3:
        return math.nan
    k = min(k, len(y_true))
    predicted = set(np.argsort(-y_pred)[:k].tolist())
    true_top = set(np.argsort(-y_true)[: max(1, math.ceil(len(y_true) * 0.01))].tolist())
    return float(len(predicted & true_top) / len(true_top))


def best_holdout_percentile(y_true: np.ndarray, y_pred: np.ndarray, k: int = 100) -> float:
    if len(y_true) == 0:
        return math.nan
    selected = np.argsort(-y_pred)[: min(k, len(y_pred))]
    best = float(np.max(y_true[selected]))
    return float(np.searchsorted(np.sort(y_true), best, side="right") / len(y_true))


def best_full_assay_percentile(assay: str, y_true: np.ndarray, y_pred: np.ndarray, k: int = 100) -> float:
    if len(y_true) == 0:
        return math.nan
    selected = np.argsort(-y_pred)[: min(k, len(y_pred))]
    best = float(np.max(y_true[selected]))
    reference = pd.read_csv(DMS_DIR / assay, usecols=["DMS_score"])
    values = pd.to_numeric(reference["DMS_score"], errors="coerce").dropna().sort_values().to_numpy(float)
    if len(values) == 0:
        return math.nan
    return float(np.searchsorted(values, best, side="right") / len(values))


def parse_selected_regime(regime: str) -> dict[str, object]:
    match = re.fullmatch(r"single20_plus_(.+)_double(\d+)", str(regime))
    if match is None:
        return {
            "training_design": str(regime),
            "selection_policy": str(regime),
            "selection_policy_label": str(regime),
            "initial_single_mutant_budget": math.nan,
            "selected_double_mutant_budget": math.nan,
        }
    policy = match.group(1)
    double_budget = int(match.group(2))
    return {
        "training_design": str(regime),
        "selection_policy": policy,
        "selection_policy_label": POLICY_LABELS.get(policy, policy.replace("_", " ")),
        "initial_single_mutant_budget": 20,
        "selected_double_mutant_budget": double_budget,
    }


def holdout_lookup(path: Path) -> dict[tuple[str, int], set[str]]:
    holdout = pd.read_csv(path)
    required = {"assay", "seed", "mutant"}
    missing = required - set(holdout.columns)
    if missing:
        raise ValueError(f"holdout table missing columns: {sorted(missing)}")
    holdout["assay"] = holdout["assay"].astype(str)
    holdout["seed"] = holdout["seed"].astype(int)
    holdout["mutant"] = holdout["mutant"].astype(str)
    return {
        (assay, int(seed)): set(block["mutant"].tolist())
        for (assay, seed), block in holdout.groupby(["assay", "seed"], sort=False)
    }


def metric_row(
    method: str,
    record: dict,
    mutants: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    holdout: dict[tuple[str, int], set[str]],
) -> dict[str, object] | None:
    assay = str(record["assay"])
    seed = int(record["seed"])
    expected = holdout.get((assay, seed), set())
    if not expected:
        return None
    mask = np.array([str(mutant) in expected for mutant in mutants], dtype=bool)
    yt = y_true[mask]
    yp = y_pred[mask]
    parsed = parse_selected_regime(str(record["regime"]))
    rho = spearman(yp, yt)
    return {
        "method": method,
        "assay": assay,
        "seed": seed,
        "budget": int(record["budget"]),
        "regime": record["regime"],
        **parsed,
        "n_holdout_expected": int(len(expected)),
        "n_holdout_evaluated": int(len(yt)),
        "holdout_coverage": float(len(yt) / len(expected)) if expected else math.nan,
        "spearman": rho,
        "abs_spearman": abs(rho) if np.isfinite(rho) else math.nan,
        "top1pct_recall_at_100": top1_recall(yt, yp),
        "best_top100_holdout_percentile": best_holdout_percentile(yt, yp),
        "best_top100_full_assay_percentile": best_full_assay_percentile(assay, yt, yp),
    }


def load_kermut_records(root: Path, holdout: dict[tuple[str, int], set[str]]) -> pd.DataFrame:
    if str(KERMUT_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(KERMUT_SCRIPT_DIR))
    from run_kermut_matched_budget import load_assay

    reference = pd.read_csv(KERMUT_REFERENCE)
    assay_cache = {}
    rows = []
    for path in sorted((root / "runs").glob("*/*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "ok":
            continue
        npz_path = path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        assay = str(record["assay"])
        if assay not in assay_cache:
            table, _, _ = load_assay(assay, reference)
            assay_cache[assay] = table
        data = np.load(npz_path)
        test_indices = data["test_indices"].astype(int)
        mutants = assay_cache[assay].loc[test_indices, "mutant"].astype(str).to_numpy()
        row = metric_row(
            "Kermut",
            record,
            mutants,
            data["y_true"].astype(float),
            data["y_pred"].astype(float),
            holdout,
        )
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def load_proteinnpt_records(root: Path, holdout: dict[tuple[str, int], set[str]]) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "runs").glob("*/*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "ok":
            continue
        npz_path = Path(record.get("npz_file", path.with_suffix(".npz")))
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        row = metric_row(
            "ProteinNPT",
            record,
            data["mutant"].astype(str),
            data["y_true"].astype(float),
            data["y_pred"].astype(float),
            holdout,
        )
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def add_gain_vs_single_baseline(detail: pd.DataFrame, single_detail: pd.DataFrame) -> pd.DataFrame:
    baseline = single_detail[single_detail["budget"].eq(20)].copy()
    baseline = baseline[
        [
            "method",
            "assay",
            "seed",
            "spearman",
            "abs_spearman",
            "top1pct_recall_at_100",
            "best_top100_holdout_percentile",
            "best_top100_full_assay_percentile",
        ]
    ].rename(
        columns={
            "spearman": "single20_spearman",
            "abs_spearman": "single20_abs_spearman",
            "top1pct_recall_at_100": "single20_top1pct_recall_at_100",
            "best_top100_holdout_percentile": "single20_best_top100_holdout_percentile",
            "best_top100_full_assay_percentile": "single20_best_top100_full_assay_percentile",
        }
    )
    merged = detail.merge(baseline, on=["method", "assay", "seed"], how="left", validate="many_to_one")
    for metric in [
        "spearman",
        "abs_spearman",
        "top1pct_recall_at_100",
        "best_top100_holdout_percentile",
        "best_top100_full_assay_percentile",
    ]:
        merged[f"gain_vs_single20_{metric}"] = merged[metric] - merged[f"single20_{metric}"]
    return merged


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assay_means = (
        detail.groupby(["method", "selection_policy", "selection_policy_label", "selected_double_mutant_budget", "assay"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_holdout_coverage=("holdout_coverage", "mean"),
            mean_n_holdout_evaluated=("n_holdout_evaluated", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_abs_spearman=("abs_spearman", "mean"),
            mean_top1pct_recall_at_100=("top1pct_recall_at_100", "mean"),
            mean_best_top100_holdout_percentile=("best_top100_holdout_percentile", "mean"),
            mean_best_top100_full_assay_percentile=("best_top100_full_assay_percentile", "mean"),
        )
    )
    gain_columns = [column for column in detail.columns if column.startswith("gain_vs_single20_")]
    aggregations = {
        "n_assays": ("assay", "nunique"),
        "n_assays_with_five_seeds": ("seed", lambda x: int(x.groupby(detail.loc[x.index, "assay"]).nunique().eq(5).sum())),
        "mean_holdout_coverage": ("holdout_coverage", "mean"),
        "mean_n_holdout_evaluated": ("n_holdout_evaluated", "mean"),
        "mean_spearman": ("spearman", "mean"),
        "mean_abs_spearman": ("abs_spearman", "mean"),
        "mean_top1pct_recall_at_100": ("top1pct_recall_at_100", "mean"),
        "mean_best_top100_holdout_percentile": ("best_top100_holdout_percentile", "mean"),
        "mean_best_top100_full_assay_percentile": ("best_top100_full_assay_percentile", "mean"),
    }
    for column in gain_columns:
        aggregations[f"mean_{column}"] = (column, "mean")
    summary = (
        detail.groupby(["method", "selection_policy", "selection_policy_label", "selected_double_mutant_budget"], as_index=False)
        .agg(**aggregations)
    )
    return summary, assay_means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--kermut-root", type=Path, default=None)
    parser.add_argument("--proteinnpt-root", type=Path, default=None)
    parser.add_argument("--single-kermut-root", type=Path, default=None)
    parser.add_argument("--single-proteinnpt-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    holdout = holdout_lookup(args.holdout.resolve())
    frames = []
    if args.kermut_root is not None and args.kermut_root.exists():
        frames.append(load_kermut_records(args.kermut_root.resolve(), holdout))
    if args.proteinnpt_root is not None and args.proteinnpt_root.exists():
        frames.append(load_proteinnpt_records(args.proteinnpt_root.resolve(), holdout))
    if not frames:
        raise RuntimeError("no selected-double records found")
    detail = pd.concat(frames, ignore_index=True)

    single_frames = []
    if args.single_kermut_root is not None and args.single_kermut_root.exists():
        single_frames.append(load_kermut_records(args.single_kermut_root.resolve(), holdout))
    if args.single_proteinnpt_root is not None and args.single_proteinnpt_root.exists():
        single_frames.append(load_proteinnpt_records(args.single_proteinnpt_root.resolve(), holdout))
    if single_frames:
        detail = add_gain_vs_single_baseline(detail, pd.concat(single_frames, ignore_index=True))

    summary, assay_means = summarize(detail)
    args.out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out / "selected_double_external_detail.csv", index=False)
    assay_means.to_csv(args.out / "selected_double_external_assay_means.csv", index=False)
    summary.to_csv(args.out / "selected_double_external_summary.csv", index=False)
    metadata = {
        "status": "ok",
        "n_detail_rows": int(len(detail)),
        "methods": sorted(detail["method"].unique().tolist()),
        "selection_policies": sorted(detail["selection_policy"].dropna().unique().tolist()),
        "selected_double_budgets": sorted(
            [int(value) for value in detail["selected_double_mutant_budget"].dropna().unique().tolist()]
        ),
        "holdout_file": str(args.holdout.resolve()),
        "baseline_gain_available": bool(single_frames),
    }
    (args.out / "selected_double_external_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
