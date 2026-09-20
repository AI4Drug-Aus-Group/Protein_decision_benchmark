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

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / "proteingym" / "extracted" / "zero_shot_substitutions_scores"
ASSAYS = ROOT / "results" / "active_learning" / "representative_assays.csv"
EMB_DIR = ROOT / "external_repos" / "kermut" / "data" / "embeddings"
OUT_DIR = ROOT / "results" / "active_learning_external"
POLICY = "EVOLVEpro random-forest top-n"


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


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
    out: dict[str, np.ndarray] = {}
    out.update(read_h5(EMB_DIR / "substitutions_singles" / "ESM2" / f"{stem}.h5"))
    out.update(read_h5(EMB_DIR / "substitutions_multiples" / "ESM2" / f"{stem}.h5"))
    return out


def load_table(assay: str) -> dict[str, float]:
    table: dict[str, float] = {}
    with (ZERO_DIR / assay).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            score = parse_float(row.get("DMS_score"))
            if score is not None:
                table[row["mutant"]] = score
    return table


def candidate_pool(table: dict[str, float], embeddings: dict[str, np.ndarray], assay: str, max_candidates: int, retain_top: bool) -> list[str]:
    mutants = [m for m in table if m in embeddings]
    if len(mutants) <= max_candidates:
        return mutants
    rng = random.Random(f"{assay}|evolvepro_candidate_pool|retain_top={retain_top}")
    if not retain_top:
        return rng.sample(mutants, max_candidates)
    top_true = sorted(mutants, key=lambda m: table[m], reverse=True)[: max(500, max_candidates // 100)]
    remaining = [m for m in mutants if m not in set(top_true)]
    return top_true + rng.sample(remaining, max_candidates - len(top_true))


def initial_observed(pool: Sequence[str], assay: str, seed: int, n: int) -> set[str]:
    rng = random.Random(f"{assay}|{seed}|initial")
    return set(rng.sample(list(pool), min(n, len(pool))))


def model(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        criterion="friedman_mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=1,
        random_state=1,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    )


def simulate_assay(assay: str, seeds: Sequence[int], rounds: int, batch: int, max_candidates: int, retain_top: bool) -> list[dict[str, object]]:
    table = load_table(assay)
    embeddings = load_embeddings(assay)
    pool = candidate_pool(table, embeddings, assay, max_candidates, retain_top)
    if len(pool) < 50:
        return []
    x_all = np.vstack([embeddings[m] for m in pool]).astype(np.float32)
    y_all = np.asarray([table[m] for m in pool], dtype=float)
    pool_hash = hashlib.sha256("\n".join(pool).encode("utf-8")).hexdigest()
    true_sorted = sorted(range(len(pool)), key=lambda i: y_all[i], reverse=True)
    true_scores_ascending = sorted(y_all.tolist())
    true_top1 = set(true_sorted[: max(1, math.ceil(len(pool) * 0.01))])
    index = {mutant: i for i, mutant in enumerate(pool)}
    rows: list[dict[str, object]] = []
    for seed in seeds:
        observed = {index[m] for m in initial_observed(pool, assay, seed, 20)}
        for round_idx in range(rounds + 1):
            observed_scores = y_all[list(observed)]
            best = float(np.max(observed_scores))
            rows.append({
                "assay": assay,
                "seed": seed,
                "policy": POLICY,
                "round": round_idx,
                "n_observed": len(observed),
                "best_observed_fitness": best,
                "best_observed_percentile": bisect.bisect_right(true_scores_ascending, best) / len(true_scores_ascending),
                "top1pct_found": int(bool(observed & true_top1)),
                "pool_size": len(pool),
                "candidate_pool_sha256": pool_hash,
                "retain_top_candidates": int(retain_top),
                "embedding_source": "official_kermut_esm2_650m_mean_pooled",
            })
            if round_idx == rounds:
                break
            available = np.array([i for i in range(len(pool)) if i not in observed], dtype=int)
            rf = model(seed * 1000 + round_idx)
            train_idx = np.array(sorted(observed), dtype=int)
            rf.fit(x_all[train_idx], y_all[train_idx])
            pred = rf.predict(x_all[available])
            selected = available[np.argsort(-pred)[: min(batch, len(available))]]
            observed.update(selected.tolist())
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
    )
    final = metrics[metrics["round"].eq(metrics["round"].max())]
    wins: list[dict[str, object]] = []
    for key, block in final.groupby(["assay", "seed"], sort=False):
        best = block["best_observed_fitness"].max()
        for _, row in block[block["best_observed_fitness"].eq(best)].iterrows():
            wins.append({"assay": key[0], "seed": key[1], "policy": row["policy"], "final_best_win": 1})
    final_wins = pd.DataFrame(wins).groupby("policy", as_index=False)["final_best_win"].sum()
    hit_rows: list[dict[str, object]] = []
    for key, block in metrics.groupby(["assay", "seed", "policy"], sort=False):
        hit = block[block["top1pct_found"].eq(1)]
        hit_rows.append({"assay": key[0], "seed": key[1], "policy": key[2], "first_top1pct_measurement": int(hit["n_observed"].min()) if not hit.empty else math.nan})
    return summary, final_wins, pd.DataFrame(hit_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=50000)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--no-retain-top-candidates", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    with ASSAYS.open(newline="", encoding="utf-8") as handle:
        assays = [row["assay"] for row in csv.DictReader(handle)]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(simulate_assay, assay, seeds, args.rounds, args.batch, args.max_candidates, not args.no_retain_top_candidates) for assay in assays]
        for future in as_completed(futures):
            rows.extend(future.result())
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "evolvepro_style_active_learning_metrics.csv", index=False)
    summary, wins, hit_time = summarize(metrics)
    summary.to_csv(OUT_DIR / "evolvepro_style_active_learning_summary.csv", index=False)
    wins.to_csv(OUT_DIR / "evolvepro_style_active_learning_final_wins.csv", index=False)
    hit_time.to_csv(OUT_DIR / "evolvepro_style_active_learning_hit_time.csv", index=False)
    metadata = {
        "source_method": "EVOLVEpro-style active learning using the official random-forest top-n model settings from evolvepro.src.model.top_layer.",
        "adaptation": "The EVOLVEpro random-forest acquisition rule uses Kermut ESM2-650M mean-pooled embeddings.",
        "setting": "Retrospective acquisition across 28 selected assays, with 20 initial measurements, five rounds and batches of ten variants; eligibility and candidate-pool capping follow the specified model inputs.",
        "n_assays_requested": len(assays),
        "n_assays_evaluable": int(metrics["assay"].nunique()) if not metrics.empty else 0,
        "seeds": seeds,
        "rounds": args.rounds,
        "batch": args.batch,
        "retain_top_candidates": not args.no_retain_top_candidates,
        "candidate_pool_hash_recorded": True,
        "official_random_forest_random_state": 1,
    }
    (OUT_DIR / "evolvepro_style_active_learning_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
