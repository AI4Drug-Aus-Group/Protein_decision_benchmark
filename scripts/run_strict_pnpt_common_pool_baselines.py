from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
PNPT = ROOT / "results" / "local_matched_reproduction" / "proteinnpt_sampled_2000_len400"
ZERO_DIR = ROOT / "proteingym" / "extracted" / "zero_shot_substitutions_scores"
DMS_DIR = ROOT / "proteingym" / "extracted" / "DMS_ProteinGym_substitutions"
FEATURES = ["VenusREM", "ProSST-2048", "S3F_MSA", "S2F_MSA", "PoET", "RSALOR", "ESM3", "GEMME", "SaProt_650M_AF2"]


def mutation_order(mutant: str) -> int:
    if not isinstance(mutant, str) or mutant in {"", "WT", "wildtype", "_wt"}:
        return 0
    return mutant.count(":") + 1


def parts(mutant: str) -> list[str]:
    return [] if mutation_order(mutant) == 0 else str(mutant).split(":")


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray | None:
    if len(y) < x.shape[1] + 1:
        return None
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    try:
        return np.linalg.pinv(x.T @ x + alpha * penalty) @ x.T @ y
    except np.linalg.LinAlgError:
        return None


def top_metrics(y_true: np.ndarray, y_pred: np.ndarray, full_scores: np.ndarray) -> dict[str, float]:
    if len(y_true) < 3:
        return {"top1pct_recall_at_100": math.nan, "best_top100_percentile": math.nan}
    k = min(100, len(y_true))
    selected = np.argsort(-y_pred)[:k]
    true_top = set(np.argsort(-y_true)[: max(1, math.ceil(len(y_true) * 0.01))].tolist())
    best = float(np.max(y_true[selected]))
    return {
        "top1pct_recall_at_100": len(set(selected.tolist()) & true_top) / len(true_top),
        "best_top100_percentile": float(np.searchsorted(np.sort(full_scores), best, side="right") / len(full_scores)),
    }


def additive_predictions(frame: pd.DataFrame, train_mutants: set[str]) -> pd.Series:
    scores = frame.set_index("mutant")["DMS_score"].to_dict()
    singles = [m for m in train_mutants if mutation_order(m) == 1 and m in scores]
    if singles:
        baseline = float(np.mean([scores[m] for m in singles]))
        effects = {m: scores[m] - baseline for m in singles}
    else:
        observed = [scores[m] for m in train_mutants if m in scores]
        baseline = float(np.mean(observed)) if observed else 0.0
        effects = {}
    return frame["mutant"].map(lambda m: baseline + sum(effects.get(part, 0.0) for part in parts(str(m))))


def evaluate_predictions(frame: pd.DataFrame, prediction: np.ndarray, method: str, row: pd.Series, full_scores: np.ndarray) -> dict[str, object]:
    test = frame["matched_split"].eq(1).to_numpy()
    y_true = frame.loc[test, "DMS_score"].to_numpy(float)
    y_pred = prediction[test]
    rho = spearmanr(y_pred, y_true).statistic if len(y_true) >= 3 else math.nan
    out = {
        "assay": row["assay"],
        "seed": int(row["seed"]),
        "budget": int(row["budget"]),
        "regime": row["regime"],
        "method": method,
        "n_train": int((frame["matched_split"] == 0).sum()),
        "n_test": int(test.sum()),
        "spearman_all": float(rho) if np.isfinite(rho) else math.nan,
    }
    out.update(top_metrics(y_true, y_pred, full_scores))
    return out


def process_run(row: pd.Series) -> list[dict[str, object]]:
    assay = row["assay"]
    frame = pd.read_csv(row["assay_data_location"], usecols=["mutant", "DMS_score", "matched_split"])
    frame["mutant"] = frame["mutant"].astype(str)
    frame["DMS_score"] = pd.to_numeric(frame["DMS_score"], errors="coerce")
    zero = pd.read_csv(ZERO_DIR / assay, usecols=lambda name: name == "mutant" or name in FEATURES)
    zero["mutant"] = zero["mutant"].astype(str)
    frame = frame.merge(zero, on="mutant", how="left", validate="one_to_one")
    full_scores = pd.to_numeric(pd.read_csv(DMS_DIR / assay, usecols=["DMS_score"])["DMS_score"], errors="coerce").dropna().to_numpy(float)
    train_mask = frame["matched_split"].eq(0).to_numpy()
    train_mutants = set(frame.loc[train_mask, "mutant"].astype(str))
    add = additive_predictions(frame, train_mutants).to_numpy(float)
    rows = [evaluate_predictions(frame, add, "single-mutant-effect sum", row, full_scores)]
    if "ProSST-2048" in frame:
        use = train_mask & frame["ProSST-2048"].notna().to_numpy()
        x = np.column_stack([np.ones(use.sum()), frame.loc[use, "ProSST-2048"].to_numpy(float)])
        y = frame.loc[use, "DMS_score"].to_numpy(float)
        weights = fit_ridge(x, y, 1e-8)
        if weights is not None:
            all_pred = np.full(len(frame), np.nan)
            ok = frame["ProSST-2048"].notna().to_numpy()
            all_pred[ok] = np.column_stack([np.ones(ok.sum()), frame.loc[ok, "ProSST-2048"].to_numpy(float)]) @ weights
            if np.isfinite(all_pred[frame["matched_split"].eq(1).to_numpy()]).all():
                rows.append(evaluate_predictions(frame, all_pred, "linear calibration of ProSST-2048", row, full_scores))
    available = [name for name in FEATURES if name in frame.columns and frame[name].notna().all()]
    if available:
        x_train = np.column_stack([np.ones(train_mask.sum())] + [frame.loc[train_mask, name].to_numpy(float) for name in available])
        y_train = frame.loc[train_mask, "DMS_score"].to_numpy(float)
        weights = fit_ridge(x_train, y_train, 1.0)
        if weights is not None:
            all_x = np.column_stack([np.ones(len(frame))] + [frame[name].to_numpy(float) for name in available])
            rows.append(evaluate_predictions(frame, all_x @ weights, "ridge on multiple zero-shot scores", row, full_scores))
    selected = [name for name in ["VenusREM", "ProSST-2048", "S3F_MSA", "ESM3", "GEMME", "TranceptEVE_L"] if name in frame.columns and frame[name].notna().all()]
    if selected:
        x_train = np.column_stack([np.ones(train_mask.sum()), frame.loc[train_mask, "mutant"].map(mutation_order).to_numpy(float), add[train_mask]] + [frame.loc[train_mask, name].to_numpy(float) for name in selected])
        y_train = frame.loc[train_mask, "DMS_score"].to_numpy(float)
        weights = fit_ridge(x_train, y_train, 1.0)
        if weights is not None:
            all_x = np.column_stack([np.ones(len(frame)), frame["mutant"].map(mutation_order).to_numpy(float), add] + [frame[name].to_numpy(float) for name in selected])
            rows.append(evaluate_predictions(frame, all_x @ weights, "ridge on mutation order, single-mutant sum and zero-shot scores", row, full_scores))
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    assay_means = detail.groupby(["method", "budget", "assay"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        mean_spearman_all=("spearman_all", "mean"),
        mean_top1pct_recall_at_100=("top1pct_recall_at_100", "mean"),
        mean_best_top100_percentile=("best_top100_percentile", "mean"),
    )
    return assay_means.groupby(["method", "budget"], as_index=False).agg(
        n_assays=("assay", "nunique"),
        n_assays_with_five_seeds=("n_seeds", lambda x: int((x == 5).sum())),
        n_assays_spearman_all=("mean_spearman_all", "count"),
        mean_spearman_all=("mean_spearman_all", "mean"),
        n_assays_top1pct_recall_at_100=("mean_top1pct_recall_at_100", "count"),
        mean_top1pct_recall_at_100=("mean_top1pct_recall_at_100", "mean"),
        n_assays_best_top100_percentile=("mean_best_top100_percentile", "count"),
        mean_best_top100_percentile=("mean_best_top100_percentile", "mean"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "sota_extension" / "strict_pnpt_common_pool")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(PNPT / "proteinnpt_run_manifest.csv")
    detail = pd.read_csv(PNPT / "proteinnpt_matched_budget_detail.csv")
    complete = set(detail[detail["status"].eq("ok")].groupby("assay").filter(lambda x: len(x) == 15)["assay"])
    manifest = manifest[manifest["assay"].isin(complete)].copy()
    rows = []
    for _, row in manifest.sort_values(["assay", "seed", "budget"]).iterrows():
        rows.extend(process_run(row))
    baseline_detail = pd.DataFrame(rows)
    pnpt = detail[detail["assay"].isin(complete) & detail["status"].eq("ok")].copy()
    pnpt["method"] = "ProteinNPT"
    pnpt = pnpt[["assay", "seed", "budget", "regime", "method", "n_train", "n_test", "spearman_all", "top1pct_recall_at_100", "best_top100_percentile"]]
    combined = pd.concat([pnpt, baseline_detail], ignore_index=True)
    combined.to_csv(out / "strict_pnpt_common_pool_detail.csv", index=False)
    summarize(combined).to_csv(out / "strict_pnpt_common_pool_summary.csv", index=False)
    metadata = {
        "status": "ok",
        "n_assays": int(len(complete)),
        "n_runs": int(manifest[["assay", "seed", "budget"]].drop_duplicates().shape[0]),
        "candidate_pool": "ProteinNPT candidate-capped run tables with exact matched_split labels",
        "methods": sorted(combined["method"].unique().tolist()),
    }
    (out / "strict_pnpt_common_pool_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
