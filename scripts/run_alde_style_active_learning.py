from __future__ import annotations

import argparse
import bisect
import hashlib
import csv
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / "proteingym" / "extracted" / "zero_shot_substitutions_scores"
ASSAYS = ROOT / "results" / "active_learning" / "representative_assays.csv"
OUT_DIR = ROOT / "results" / "active_learning_external"
POLICIES = ["ALDE boosting greedy", "ALDE boosting UCB", "ALDE boosting TS"]


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def mutation_order(mutant: str) -> int:
    if not mutant or mutant in {"WT", "wildtype", "_wt"}:
        return 0
    return mutant.count(":") + 1


def mutation_parts(mutant: str) -> list[str]:
    if mutation_order(mutant) == 0:
        return []
    return [part for part in mutant.split(":") if part]


def load_table(assay: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with (ZERO_DIR / assay).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            score = parse_float(row.get("DMS_score"))
            if score is not None:
                out[row["mutant"]] = score
    return out


def candidate_pool(table: dict[str, float], assay: str, max_candidates: int, retain_top: bool) -> list[str]:
    mutants = list(table)
    if len(mutants) <= max_candidates:
        return mutants
    rng = random.Random(f"{assay}|candidate_pool|retain_top={retain_top}")
    if not retain_top:
        return rng.sample(mutants, max_candidates)
    top_true = sorted(mutants, key=lambda m: table[m], reverse=True)[: max(500, max_candidates // 100)]
    remaining = [m for m in mutants if m not in set(top_true)]
    return top_true + rng.sample(remaining, max_candidates - len(top_true))


def initial_observed(pool: Sequence[str], assay: str, seed: int, n: int) -> set[str]:
    rng = random.Random(f"{assay}|{seed}|initial")
    return set(rng.sample(list(pool), min(n, len(pool))))


def build_binary_features(pool: Sequence[str]) -> csr_matrix:
    vocabulary = sorted({part for mutant in pool for part in mutation_parts(mutant)})
    col = {part: i for i, part in enumerate(vocabulary)}
    rows: list[int] = []
    cols: list[int] = []
    for i, mutant in enumerate(pool):
        for part in mutation_parts(mutant):
            if part in col:
                rows.append(i)
                cols.append(col[part])
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(len(pool), len(vocabulary)), dtype=np.float32)


def scaled_targets(y: np.ndarray) -> np.ndarray:
    shifted = y - np.min(y) + 1e-6
    normalizer = np.max(shifted)
    if normalizer <= 0:
        return np.zeros_like(shifted)
    return shifted / normalizer


def ensemble_predictions(
    x_all: csr_matrix,
    observed_idx: list[int],
    y_all: np.ndarray,
    seed: int,
    n_splits: int,
    bootstrap_size: float,
) -> np.ndarray:
    x_train_all = x_all[observed_idx]
    y_train_all = scaled_targets(y_all[observed_idx])
    preds = np.zeros((x_all.shape[0], n_splits), dtype=float)
    if len(observed_idx) < 5:
        preds[:, :] = float(np.mean(y_train_all)) if len(y_train_all) else 0.0
        return preds
    for i in range(n_splits):
        if bootstrap_size < 1 and len(observed_idx) >= 10:
            x_train, x_valid, y_train, y_valid = train_test_split(
                x_train_all,
                y_train_all,
                test_size=1 - bootstrap_size,
                random_state=seed + i,
            )
            eval_set = [(x_valid, y_valid)]
            early_stopping_rounds = 10
        else:
            x_train, y_train = x_train_all, y_train_all
            eval_set = None
            early_stopping_rounds = None
        model = XGBRegressor(
            n_estimators=100,
            objective="reg:tweedie",
            random_state=seed + i,
            n_jobs=1,
            tree_method="hist",
            early_stopping_rounds=early_stopping_rounds,
            verbosity=0,
        )
        model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
        preds[:, i] = model.predict(x_all)
    return preds


def select_batch(
    policy: str,
    preds: np.ndarray,
    observed_idx: set[int],
    batch: int,
    rng: random.Random,
    xi: float,
) -> list[int]:
    available = np.array([index for index in range(preds.shape[0]) if index not in observed_idx], dtype=int)
    if len(available) == 0:
        return []
    n_select = min(batch, len(available))
    if policy == "ALDE boosting greedy":
        score = np.nan_to_num(preds.mean(axis=1), nan=-np.inf)
        order = np.argsort(-score[available], kind="mergesort")
        return available[order[:n_select]].tolist()
    if policy == "ALDE boosting UCB":
        score = preds.mean(axis=1) + math.sqrt(xi) * preds.std(axis=1, ddof=0)
        score = np.nan_to_num(score, nan=-np.inf)
        order = np.argsort(-score[available], kind="mergesort")
        return available[order[:n_select]].tolist()
    if policy == "ALDE boosting TS":
        remaining = available.tolist()
        selected: list[int] = []
        for _ in range(n_select):
            column = rng.randrange(preds.shape[1])
            values = np.nan_to_num(preds[remaining, column], nan=-np.inf)
            local = int(np.argmax(values))
            selected.append(remaining.pop(local))
        return selected
    raise ValueError(policy)


def simulate_assay(
    assay: str,
    seeds: Sequence[int],
    rounds: int,
    batch: int,
    max_candidates: int,
    retain_top: bool,
    n_splits: int,
    bootstrap_size: float,
    xi: float,
) -> list[dict[str, object]]:
    table = load_table(assay)
    pool = candidate_pool(table, assay, max_candidates, retain_top)
    x_all = build_binary_features(pool)
    y_all = np.asarray([table[m] for m in pool], dtype=float)
    pool_hash = hashlib.sha256("\n".join(pool).encode("utf-8")).hexdigest()
    true_sorted = sorted(range(len(pool)), key=lambda i: y_all[i], reverse=True)
    true_scores_ascending = sorted(y_all.tolist())
    true_top1 = set(true_sorted[: max(1, math.ceil(len(pool) * 0.01))])
    rows: list[dict[str, object]] = []
    for seed in seeds:
        init = initial_observed(pool, assay, seed, 20)
        init_idx = {pool.index(m) for m in init}
        for policy in POLICIES:
            observed_idx = set(init_idx)
            rng = random.Random(f"{assay}|{seed}|{policy}")
            for round_idx in range(rounds + 1):
                observed_scores = y_all[list(observed_idx)]
                best = float(np.max(observed_scores))
                best_percentile = bisect.bisect_right(true_scores_ascending, best) / len(true_scores_ascending)
                rows.append({
                    "assay": assay,
                    "seed": seed,
                    "policy": policy,
                    "round": round_idx,
                    "n_observed": len(observed_idx),
                    "best_observed_fitness": best,
                    "best_observed_percentile": best_percentile,
                    "top1pct_found": int(bool(observed_idx & true_top1)),
                    "pool_size": len(pool),
                    "candidate_pool_sha256": pool_hash,
                    "observed_set_sha256": hashlib.sha256("\n".join(sorted(pool[index] for index in observed_idx)).encode("utf-8")).hexdigest(),
                    "retain_top_candidates": int(retain_top),
                    "model_family": "ALDE BOOSTING_ENSEMBLE adapted to ProteinGym mutation-identity features",
                })
                if round_idx == rounds:
                    break
                preds = ensemble_predictions(x_all, sorted(observed_idx), y_all, seed * 1000 + round_idx * 31, n_splits, bootstrap_size)
                selected = select_batch(policy, preds, observed_idx, batch, rng, xi)
                observed_idx.update(selected)
    return rows


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics.groupby(["policy", "round"], as_index=False)
        .agg(
            n_runs=("top1pct_found", "size"),
            n_assays=("assay", "nunique"),
            mean_top1pct_found=("top1pct_found", "mean"),
            mean_best_observed_percentile=("best_observed_percentile", "mean"),
            mean_n_observed=("n_observed", "mean"),
        )
        .sort_values(["round", "mean_top1pct_found", "mean_best_observed_percentile"], ascending=[True, False, False])
    )
    final = metrics[metrics["round"].eq(metrics["round"].max())].copy()
    wins: list[dict[str, object]] = []
    for key, block in final.groupby(["assay", "seed"], sort=False):
        best = block["best_observed_fitness"].max()
        for _, row in block[block["best_observed_fitness"].eq(best)].iterrows():
            wins.append({"assay": key[0], "seed": key[1], "policy": row["policy"], "final_best_win": 1})
    final_wins = pd.DataFrame(wins).groupby("policy", as_index=False)["final_best_win"].sum()
    hit_rows: list[dict[str, object]] = []
    for key, block in metrics.groupby(["assay", "seed", "policy"], sort=False):
        hit = block[block["top1pct_found"].eq(1)]
        first = int(hit["n_observed"].min()) if not hit.empty else math.nan
        hit_rows.append({"assay": key[0], "seed": key[1], "policy": key[2], "first_top1pct_measurement": first})
    hit_time = pd.DataFrame(hit_rows)
    return summary, final_wins, hit_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=50000)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--no-retain-top-candidates", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--bootstrap-size", type=float, default=0.9)
    parser.add_argument("--xi", type=float, default=4.0)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    with ASSAYS.open(newline="", encoding="utf-8") as handle:
        assays = [row["assay"] for row in csv.DictReader(handle)]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                simulate_assay,
                assay,
                seeds,
                args.rounds,
                args.batch,
                args.max_candidates,
                not args.no_retain_top_candidates,
                args.n_splits,
                args.bootstrap_size,
                args.xi,
            )
            for assay in assays
        ]
        for future in as_completed(futures):
            rows.extend(future.result())
    metrics = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "alde_style_active_learning_metrics.csv"
    summary_path = OUT_DIR / "alde_style_active_learning_summary.csv"
    wins_path = OUT_DIR / "alde_style_active_learning_final_wins.csv"
    hit_path = OUT_DIR / "alde_style_active_learning_hit_time.csv"
    metrics.to_csv(metrics_path, index=False)
    summary, wins, hit_time = summarize(metrics)
    summary.to_csv(summary_path, index=False)
    wins.to_csv(wins_path, index=False)
    hit_time.to_csv(hit_path, index=False)
    metadata = {
        "source_method": "ALDE BOOSTING_ENSEMBLE acquisition structure: GREEDY, UCB with xi=4, and Thompson sampling from ensemble members.",
        "adaptation": "ProteinGym variants were encoded by mutation-identity binary features. DMS scores were shifted and rescaled within each observed set before XGBoost training because ProteinGym assay scores can be negative.",
        "official_repository_commit": "d0b3593dd17987c9864830d36f5976b1ad04a619",
        "setting": "Retrospective acquisition across 28 selected assays, with 20 initial measurements, five rounds and batches of ten variants; eligibility and candidate-pool capping follow the specified model inputs.",
        "n_assays": len(assays),
        "seeds": seeds,
        "rounds": args.rounds,
        "batch": args.batch,
        "workers": args.workers,
        "policies": POLICIES,
        "retain_top_candidates": not args.no_retain_top_candidates,
        "candidate_pool_hash_recorded": True,
        "discovery_target": "A measured variant in the top 1% of the label-independent assay candidate pool.",
    }
    (OUT_DIR / "alde_style_active_learning_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
