from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "results" / "splits" / "low_n_splits.csv"
ZERO_DIR = ROOT / "proteingym" / "extracted" / "zero_shot_substitutions_scores"
DMS_DIR = ROOT / "proteingym" / "extracted" / "DMS_ProteinGym_substitutions"
EMB_DIR = ROOT / "external_repos" / "kermut" / "data" / "embeddings"
OUT_DIR = ROOT / "results" / "few_shot_external"
DEFAULT_MODELS = [
    "esm2_embedding_ridge",
    "esm2_embedding_gp",
    "esm2_embedding_hist_gradient_boosting",
    "esm2_embedding_xgboost",
]


def mutation_order(mutant: str) -> int:
    if not mutant or mutant in {"WT", "wildtype", "_wt"}:
        return 0
    return mutant.count(":") + 1


def read_h5(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        embeddings = np.asarray(handle["embeddings"][:], dtype=np.float32)
        if embeddings.ndim == 3:
            embeddings = embeddings.mean(axis=1)
        mutants = handle["mutants"][:]
    for mutant, embedding in zip(mutants, embeddings):
        key = mutant.decode("utf-8") if isinstance(mutant, bytes) else str(mutant)
        out[key] = embedding
    return out


def load_embeddings(assay: str) -> dict[str, np.ndarray]:
    stem = assay.removesuffix(".csv")
    merged: dict[str, np.ndarray] = {}
    merged.update(read_h5(EMB_DIR / "substitutions_singles" / "ESM2" / f"{stem}.h5"))
    merged.update(read_h5(EMB_DIR / "substitutions_multiples" / "ESM2" / f"{stem}.h5"))
    return merged


def load_split_groups(regimes: set[str], budgets: set[int], seeds: set[int]) -> dict[str, list[tuple[tuple[str, int, int, str], list[str]]]]:
    splits = pd.read_csv(SPLITS)
    splits = splits[splits["regime"].isin(regimes)]
    splits = splits[splits["budget"].isin(budgets)]
    splits = splits[splits["seed"].isin(seeds)]
    groups: dict[str, list[tuple[tuple[str, int, int, str], list[str]]]] = {}
    for key, block in splits.groupby(["assay", "seed", "budget", "regime"], sort=False):
        assay, seed, budget, regime = key
        groups.setdefault(str(assay), []).append(((str(assay), int(seed), int(budget), str(regime)), block["mutant"].tolist()))
    return groups


def top_metrics(y_true: np.ndarray, y_pred: np.ndarray, percentile_reference: np.ndarray, k: int = 100) -> dict[str, float]:
    if len(y_true) < 3:
        return {"top1pct_recall_at_100": np.nan, "best_true_in_pred_top100": np.nan, "best_top100_percentile": np.nan}
    k = min(k, len(y_true))
    pred_idx = np.argsort(-y_pred)[:k]
    true_idx = np.argsort(-y_true)
    top_n = max(1, math.ceil(len(y_true) * 0.01))
    true_top = set(true_idx[:top_n].tolist())
    best = float(np.max(y_true[pred_idx]))
    percentile = float(np.mean(percentile_reference <= best))
    return {
        "top1pct_recall_at_100": len(set(pred_idx.tolist()) & true_top) / len(true_top),
        "best_true_in_pred_top100": best,
        "best_top100_percentile": percentile,
    }


def make_model(name: str, seed: int):
    if name == "esm2_embedding_ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "esm2_embedding_gp":
        kernel = ConstantKernel(1.0) * RBF(length_scale=math.sqrt(1280.0), length_scale_bounds="fixed") + WhiteKernel(noise_level=0.1)
        return make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, optimizer=None, random_state=seed),
        )
    if name == "esm2_embedding_hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=60,
            max_leaf_nodes=15,
            min_samples_leaf=2,
            learning_rate=0.05,
            random_state=seed,
        )
    if name == "esm2_embedding_xgboost":
        return XGBRegressor(
            n_estimators=80,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(name)


def evaluate_prediction(
    meta: tuple[str, int, int, str],
    model_name: str,
    table: pd.DataFrame,
    train: set[str],
    y_pred: np.ndarray,
    percentile_reference: np.ndarray,
) -> dict[str, object]:
    assay, seed, budget, regime = meta
    y_true = table["DMS_score"].to_numpy(float)
    mutants = table["mutant"].tolist()
    orders = np.array([mutation_order(m) for m in mutants], dtype=int)
    train_mask = table["mutant"].isin(train).to_numpy()
    test_mask = ~train_mask
    out: dict[str, object] = {
        "assay": assay,
        "seed": seed,
        "budget": budget,
        "regime": regime,
        "model": model_name,
        "n_train_requested": len(train),
        "n_train_used": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "embedding_source": "official_kermut_esm2_650m_mean_pooled",
    }
    test_prediction = y_pred[test_mask]
    if test_mask.sum() >= 3 and np.unique(test_prediction[np.isfinite(test_prediction)]).size >= 2:
        rho = spearmanr(test_prediction, y_true[test_mask]).statistic
        out["spearman_all"] = float(rho) if np.isfinite(rho) else np.nan
        out.update(top_metrics(y_true[test_mask], y_pred[test_mask], percentile_reference))
        out["best_top100_percentile_reference"] = "complete_assay_candidate_pool"
    else:
        out["spearman_all"] = np.nan
        out.update(top_metrics(np.array([]), np.array([]), percentile_reference))
    for label, mask in [
        ("singles", orders == 1),
        ("doubles", orders == 2),
        ("higher", orders >= 3),
    ]:
        subset = test_mask & mask
        out[f"n_test_{label}"] = int(subset.sum())
        subset_prediction = y_pred[subset]
        if subset.sum() >= 3 and np.unique(subset_prediction[np.isfinite(subset_prediction)]).size >= 2:
            rho = spearmanr(subset_prediction, y_true[subset]).statistic
            out[f"spearman_{label}"] = float(rho) if np.isfinite(rho) else np.nan
        else:
            out[f"spearman_{label}"] = np.nan
    return out


def process_assay(args: tuple[str, list[tuple[tuple[str, int, int, str], list[str]]], list[str]]) -> list[dict[str, object]]:
    assay, groups, models = args
    csv_path = ZERO_DIR / assay
    if not csv_path.exists():
        return []
    table = pd.read_csv(csv_path, usecols=["mutant", "DMS_score"])
    table = table.dropna(subset=["mutant", "DMS_score"]).copy()
    percentile_reference = pd.read_csv(DMS_DIR / assay, usecols=["DMS_score"])
    full_assay_scores = pd.to_numeric(percentile_reference["DMS_score"], errors="coerce").to_numpy(float)
    full_assay_scores = full_assay_scores[np.isfinite(full_assay_scores)]
    table["mutant"] = table["mutant"].astype(str)
    embeddings = load_embeddings(assay)
    if not embeddings:
        return []
    table = table[table["mutant"].isin(embeddings)].reset_index(drop=True)
    if len(table) < 50:
        return []
    mutants = table["mutant"].tolist()
    x_all = np.vstack([embeddings[mutant] for mutant in mutants]).astype(np.float32)
    y_all = table["DMS_score"].to_numpy(float)
    rows: list[dict[str, object]] = []
    for meta, train_mutants in groups:
        requested = list(dict.fromkeys(str(mutant) for mutant in train_mutants))
        train = set(mutant for mutant in requested if mutant in embeddings)
        train_mask = table["mutant"].isin(train).to_numpy()
        if len(requested) != int(meta[2]):
            status = "excluded_non_nominal_split_budget"
        elif len(train) != len(requested) or int(train_mask.sum()) != len(requested):
            status = "excluded_missing_training_embedding"
        elif int((~train_mask).sum()) < 20:
            status = "excluded_insufficient_test_candidates"
        else:
            status = "eligible"
        if status != "eligible":
            for model_name in models:
                rows.append(
                    {
                        "assay": meta[0],
                        "seed": meta[1],
                        "budget": meta[2],
                        "regime": meta[3],
                        "model": model_name,
                        "status": status,
                        "n_train_requested": len(requested),
                        "n_train_used": int(train_mask.sum()),
                        "n_test": int((~train_mask).sum()),
                        "n_candidates_with_embeddings": len(table),
                        "embedding_source": "official_kermut_esm2_650m_mean_pooled",
                    }
                )
            continue
        for model_name in models:
            model = make_model(model_name, int(meta[1]))
            try:
                model.fit(x_all[train_mask], y_all[train_mask])
                y_pred = np.asarray(model.predict(x_all), dtype=float)
            except Exception as exc:
                rows.append(
                    {
                        "assay": meta[0],
                        "seed": meta[1],
                        "budget": meta[2],
                        "regime": meta[3],
                        "model": model_name,
                        "status": f"failed:{type(exc).__name__}",
                        "n_train_requested": len(requested),
                        "n_train_used": int(train_mask.sum()),
                        "n_test": int((~train_mask).sum()),
                        "n_candidates_with_embeddings": len(table),
                        "embedding_source": "official_kermut_esm2_650m_mean_pooled",
                    }
                )
                continue
            rec = evaluate_prediction(meta, model_name, table, train, y_pred, full_assay_scores)
            rec["n_train_requested"] = len(requested)
            rec["n_candidates_with_embeddings"] = len(table)
            test_prediction = y_pred[~train_mask]
            if np.unique(test_prediction[np.isfinite(test_prediction)]).size < 2:
                rec["status"] = "constant_prediction"
                for metric in [
                    "spearman_all",
                    "top1pct_recall_at_100",
                    "best_true_in_pred_top100",
                    "best_top100_percentile",
                    "spearman_singles",
                    "spearman_doubles",
                    "spearman_higher",
                ]:
                    rec[metric] = np.nan
            else:
                rec["status"] = "ok"
            rows.append(rec)
    return rows


def boot_ci(values: Iterable[float], seed: int = 17, n_boot: int = 5000) -> tuple[float, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    valid = np.isfinite(p)
    q = np.full(len(p), np.nan)
    if valid.any():
        observed = p[valid]
        order = np.argsort(observed)
        ranked = observed[order] * len(observed) / np.arange(1, len(observed) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        restored = np.empty(len(observed))
        restored[order] = np.minimum(ranked, 1.0)
        q[valid] = restored
    return pd.Series(q, index=values.index)


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = detail[detail["status"].eq("ok")].copy()
    metric_cols = [
        "spearman_all",
        "top1pct_recall_at_100",
        "best_top100_percentile",
        "spearman_singles",
        "spearman_doubles",
        "spearman_higher",
    ]
    group_columns = ["model", "budget", "regime"]
    assay_means = ok.groupby(group_columns + ["assay"], as_index=False)[metric_cols].mean()
    run_counts = ok.groupby(group_columns).size().rename("n_runs").reset_index()
    summary_rows: list[dict[str, object]] = []
    for key, group in assay_means.groupby(group_columns, sort=True):
        rec: dict[str, object] = {
            "model": key[0],
            "budget": int(key[1]),
            "regime": key[2],
            "n_runs": int(run_counts.query("model == @key[0] and budget == @key[1] and regime == @key[2]")["n_runs"].iloc[0]),
            "n_assays": int(group["assay"].nunique()),
        }
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            rec[f"n_assays_{metric}"] = len(values)
            rec[f"mean_{metric}"] = float(values.mean()) if len(values) else np.nan
            low, high = boot_ci(values)
            rec[f"ci95_low_{metric}"] = low
            rec[f"ci95_high_{metric}"] = high
        summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)
    pair_rows: list[dict[str, object]] = []
    baseline = "esm2_embedding_ridge"
    for budget in sorted(ok["budget"].unique()):
        for regime in sorted(ok["regime"].unique()):
            subset = ok[(ok["budget"].eq(budget)) & (ok["regime"].eq(regime))]
            for metric in ["spearman_all", "top1pct_recall_at_100", "best_top100_percentile"]:
                assay_metric = subset.groupby(["assay", "model"], as_index=False)[metric].mean()
                pivot = assay_metric.pivot(index="assay", columns="model", values=metric)
                if baseline not in pivot:
                    continue
                for model in pivot.columns:
                    if model == baseline:
                        continue
                    valid_pairs = pivot[[baseline, model]].dropna()
                    if len(valid_pairs) < 5:
                        continue
                    difference = valid_pairs[model] - valid_pairs[baseline]
                    p_value = wilcoxon(difference, zero_method="wilcox", alternative="two-sided").pvalue if (difference != 0).any() else 1.0
                    low, high = boot_ci(difference)
                    pair_rows.append(
                        {
                            "budget": int(budget),
                            "regime": regime,
                            "metric": metric,
                            "model": model,
                            "baseline": baseline,
                            "n_assays": len(difference),
                            "mean_delta": float(difference.mean()),
                            "ci95_low_delta": low,
                            "ci95_high_delta": high,
                            "wilcoxon_p_assay_level": float(p_value),
                        }
                    )
    tests = pd.DataFrame(pair_rows)
    if not tests.empty:
        tests["benjamini_hochberg_q_all_tests"] = benjamini_hochberg(tests["wilcoxon_p_assay_level"])
    return summary, tests


def parse_set(raw: str, cast):
    return {cast(x) for x in raw.split(",") if x}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", default="mixed_random,single_only,single_plus_double")
    parser.add_argument("--budgets", default="20,50,100")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--assays", default="")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = load_split_groups(parse_set(args.regimes, str), parse_set(args.budgets, int), parse_set(args.seeds, int))
    if args.assays:
        keep = {a if a.endswith(".csv") else f"{a}.csv" for a in args.assays.split(",") if a}
        groups = {assay: value for assay, value in groups.items() if assay in keep}
    models = [m for m in args.models.split(",") if m]
    tasks = [(assay, assay_groups, models) for assay, assay_groups in groups.items()]
    detail_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_assay, task) for task in tasks]
        for future in as_completed(futures):
            detail_rows.extend(future.result())
    detail = pd.DataFrame(detail_rows)
    detail_path = out_dir / "matched_budget_esm2_embedding_baselines_detail.csv"
    summary_path = out_dir / "matched_budget_esm2_embedding_baselines_summary.csv"
    tests_path = out_dir / "matched_budget_esm2_embedding_baselines_paired_tests.csv"
    detail.to_csv(detail_path, index=False)
    summary, tests = summarize(detail)
    summary.to_csv(summary_path, index=False)
    tests.to_csv(tests_path, index=False)
    metadata = {
        "input_splits": str(SPLITS.relative_to(ROOT)),
        "assay_tables": str(ZERO_DIR.relative_to(ROOT)),
        "embedding_sources": [
            str((EMB_DIR / "substitutions_singles" / "ESM2").relative_to(ROOT)),
            str((EMB_DIR / "substitutions_multiples" / "ESM2").relative_to(ROOT)),
        ],
        "models": models,
        "budgets": sorted(parse_set(args.budgets, int)),
        "seeds": sorted(parse_set(args.seeds, int)),
        "regimes": sorted(parse_set(args.regimes, str)),
        "workers": args.workers,
        "kermut_repository_commit": "e6500221b5159a1896272bfc4a3c6e3896eb0027",
        "important_scope": "Strict nominal-budget evaluation on candidates with official Kermut ESM2-650M mean-pooled embeddings. Runs with fewer than the stated 20, 50 or 100 training measurements, missing training embeddings or constant predictions were excluded. This is not the full Kermut mutation kernel.",
        "gaussian_process_kernel": "Constant times RBF with fixed length scale sqrt(1280), plus white noise; this is a generic embedding GP control rather than Kermut.",
        "statistical_unit": "Seeds were averaged within assay before assay-level confidence intervals and paired tests.",
        "constant_prediction_handling": "All ranking and top-candidate metrics were set to missing when held-out predictions were constant.",
        "detail_rows": int(len(detail)),
        "status_counts": {str(key): int(value) for key, value in detail["status"].value_counts(dropna=False).items()},
        "ok_rows": int(detail["status"].eq("ok").sum()) if "status" in detail.columns else 0,
    }
    (out_dir / "matched_budget_esm2_embedding_baselines_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
