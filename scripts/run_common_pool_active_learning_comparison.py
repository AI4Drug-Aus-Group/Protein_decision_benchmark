from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(SCRIPT_DIR))

import run_active_learning_simulation as manuscript
import run_alde_style_active_learning as alde
import run_evolvepro_style_active_learning as evolve


OUT_DIR = ROOT / "results" / "active_learning_external"
ASSAY_FILE = ROOT / "results" / "active_learning" / "representative_assays.csv"
MAIN_POLICIES = [
    "random",
    "ensemble_mean:top_methods",
    "ensemble_ucb:top_methods",
    "diverse_ensemble:top_methods",
    "ensemble_rank_mean_std:top_methods",
    "ensemble_rank_mean_disagreement:top_methods",
]
ALDE_POLICIES = list(alde.POLICIES)
EVOLVEPRO_POLICY = evolve.POLICY
POLICIES = [EVOLVEPRO_POLICY, *ALDE_POLICIES, *MAIN_POLICIES]
METRICS = ["top1pct_found", "best_observed_percentile"]


def label_independent_pool(
    table: dict[str, dict[str, float]],
    embeddings: dict[str, np.ndarray],
    assay: str,
    maximum: int,
) -> list[str]:
    candidates = [mutant for mutant in table if mutant in embeddings]
    if len(candidates) <= maximum:
        return candidates
    rng = random.Random(f"{assay}|common_embedding_pool|no_label_retention")
    return rng.sample(candidates, maximum)


def record_state(
    rows: list[dict[str, object]],
    assay: str,
    seed: int,
    policy: str,
    source: str,
    round_index: int,
    observed: set[str],
    fitness: dict[str, float],
    sorted_scores: list[float],
    true_top1: set[str],
    pool_size: int,
    pool_hash: str,
) -> None:
    best = max(fitness[mutant] for mutant in observed)
    rows.append(
        {
            "assay": assay,
            "seed": seed,
            "policy": policy,
            "source": source,
            "round": round_index,
            "n_observed": len(observed),
            "best_observed_fitness": best,
            "best_observed_percentile": bisect.bisect_right(sorted_scores, best) / len(sorted_scores),
            "top1pct_found": int(bool(observed & true_top1)),
            "pool_size": pool_size,
            "candidate_pool_sha256": pool_hash,
            "observed_set_sha256": hashlib.sha256("\n".join(sorted(observed)).encode("utf-8")).hexdigest(),
            "retain_top_candidates": 0,
        }
    )


def simulate_policy(
    assay: str,
    seed: int,
    policy: str,
    table: dict[str, dict[str, float]],
    pool: list[str],
    fitness: dict[str, float],
    embeddings: dict[str, np.ndarray],
    binary_features,
    index: dict[str, int],
    y_all: np.ndarray,
    rounds: int,
    batch: int,
    n_splits: int,
    bootstrap_size: float,
    xi: float,
    pool_hash: str,
    true_top1: set[str],
    sorted_scores: list[float],
) -> list[dict[str, object]]:
    observed = manuscript.initial_observed(pool, assay, seed, 20)
    rows: list[dict[str, object]] = []
    rng = random.Random(f"{assay}|{seed}|{policy}|common_embedding_pool")
    if policy == EVOLVEPRO_POLICY:
        source = "EVOLVEpro-style"
    elif policy in ALDE_POLICIES:
        source = "ALDE-style"
    else:
        source = "decision-benchmark policy"
    for round_index in range(rounds + 1):
        record_state(
            rows,
            assay,
            seed,
            policy,
            source,
            round_index,
            observed,
            fitness,
            sorted_scores,
            true_top1,
            len(pool),
            pool_hash,
        )
        if round_index == rounds:
            break
        if policy == EVOLVEPRO_POLICY:
            observed_idx = np.array(sorted(index[mutant] for mutant in observed), dtype=int)
            available_idx = np.array([i for i in range(len(pool)) if pool[i] not in observed], dtype=int)
            model = evolve.model(1)
            model.fit(np.vstack([embeddings[pool[i]] for i in observed_idx]), y_all[observed_idx])
            prediction = model.predict(np.vstack([embeddings[pool[i]] for i in available_idx]))
            selected_idx = available_idx[np.argsort(-prediction, kind="mergesort")[: min(batch, len(available_idx))]]
            selected = [pool[i] for i in selected_idx]
        elif policy in ALDE_POLICIES:
            observed_idx = {index[mutant] for mutant in observed}
            predictions = alde.ensemble_predictions(
                binary_features,
                sorted(observed_idx),
                y_all,
                seed * 1000 + round_index * 31,
                n_splits,
                bootstrap_size,
            )
            selected_idx = alde.select_batch(policy, predictions, observed_idx, batch, rng, xi)
            selected = [pool[i] for i in selected_idx]
        else:
            selected = manuscript.select_batch(policy, table, observed, pool, batch, rng)
        observed.update(selected)
    return rows


def simulate_assay(
    assay: str,
    seeds: list[int],
    rounds: int,
    batch: int,
    maximum: int,
    n_splits: int,
    bootstrap_size: float,
    xi: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    table = manuscript.load_table(assay)
    embeddings = evolve.load_embeddings(assay)
    pool = label_independent_pool(table, embeddings, assay, maximum)
    if len(pool) < 70:
        return [], {
            "assay": assay,
            "status": "not_evaluable",
            "reason": "fewer than 70 candidates with ESM2-650M embeddings",
            "pool_size": len(pool),
        }
    fitness = {mutant: table[mutant]["DMS_score"] for mutant in pool}
    y_all = np.asarray([fitness[mutant] for mutant in pool], dtype=float)
    binary_features = alde.build_binary_features(pool)
    index = {mutant: i for i, mutant in enumerate(pool)}
    pool_hash = hashlib.sha256("\n".join(pool).encode("utf-8")).hexdigest()
    true_order = sorted(pool, key=lambda mutant: fitness[mutant], reverse=True)
    true_top1 = set(true_order[: max(1, math.ceil(len(pool) * 0.01))])
    sorted_scores = sorted(y_all.tolist())
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for policy in POLICIES:
            rows.extend(
                simulate_policy(
                    assay,
                    seed,
                    policy,
                    table,
                    pool,
                    fitness,
                    embeddings,
                    binary_features,
                    index,
                    y_all,
                    rounds,
                    batch,
                    n_splits,
                    bootstrap_size,
                    xi,
                    pool_hash,
                    true_top1,
                    sorted_scores,
                )
            )
    return rows, {
        "assay": assay,
        "status": "ok",
        "reason": "",
        "pool_size": len(pool),
        "candidate_pool_sha256": pool_hash,
        "n_zero_shot_candidates": len(table),
        "embedding_coverage_fraction": len(pool) / len(table),
    }


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(repetitions, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (repetitions + 1))


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


def summary_table(metrics: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (policy, source, round_index), group in metrics.groupby(["policy", "source", "round"], sort=True):
        assay_means = group.groupby("assay")[["top1pct_found", "best_observed_percentile", "n_observed"]].mean()
        hit_low, hit_high = bootstrap_mean(assay_means["top1pct_found"].to_numpy(float), rng, bootstrap)
        best_low, best_high = bootstrap_mean(assay_means["best_observed_percentile"].to_numpy(float), rng, bootstrap)
        rows.append(
            {
                "source": source,
                "policy": policy,
                "round": int(round_index),
                "n_runs": len(group),
                "n_assays": group["assay"].nunique(),
                "top1pct_found_rate": assay_means["top1pct_found"].mean(),
                "bootstrap_95ci_low_top1pct_found": hit_low,
                "bootstrap_95ci_high_top1pct_found": hit_high,
                "mean_best_observed_percentile": assay_means["best_observed_percentile"].mean(),
                "bootstrap_95ci_low_best_percentile": best_low,
                "bootstrap_95ci_high_best_percentile": best_high,
                "mean_n_observed": assay_means["n_observed"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["round", "top1pct_found_rate", "mean_best_observed_percentile"], ascending=[True, False, False])


def paired_tests(
    metrics: pd.DataFrame,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    final = metrics[metrics["round"].eq(metrics["round"].max())]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for reference in [EVOLVEPRO_POLICY, "random"]:
        for comparison in sorted(final["policy"].unique()):
            if comparison == reference:
                continue
            for metric in METRICS:
                pivot = final[final["policy"].isin([reference, comparison])].pivot_table(
                    index=["assay", "seed"], columns="policy", values=metric, aggfunc="first"
                )
                if reference not in pivot or comparison not in pivot:
                    continue
                paired = pivot[[reference, comparison]].dropna()
                assay_delta = (paired[reference] - paired[comparison]).groupby(level="assay").mean().to_numpy(float)
                low, high = bootstrap_mean(assay_delta, rng, bootstrap)
                rows.append(
                    {
                        "reference": reference,
                        "comparison": comparison,
                        "metric": metric,
                        "n_assays": len(assay_delta),
                        "mean_paired_difference_reference_minus_comparison": float(assay_delta.mean()),
                        "bootstrap_95ci_low": low,
                        "bootstrap_95ci_high": high,
                        "two_sided_sign_flip_p": sign_flip(assay_delta, rng, permutations),
                    }
                )
    tests = pd.DataFrame(rows)
    tests["benjamini_hochberg_q_all_tests"] = benjamini_hochberg(tests["two_sided_sign_flip_p"])
    return tests


def hit_time_and_wins(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = metrics[metrics["round"].eq(metrics["round"].max())]
    hit_rows: list[dict[str, object]] = []
    for key, block in metrics.groupby(["assay", "seed", "policy"], sort=False):
        hit = block[block["top1pct_found"].eq(1)]
        hit_rows.append(
            {
                "assay": key[0],
                "seed": key[1],
                "policy": key[2],
                "first_top1pct_measurement": int(hit["n_observed"].min()) if not hit.empty else math.nan,
            }
        )
    win_rows: list[dict[str, object]] = []
    for key, block in final.groupby(["assay", "seed"], sort=False):
        best = block["best_observed_fitness"].max()
        for policy in block.loc[block["best_observed_fitness"].eq(best), "policy"]:
            win_rows.append({"assay": key[0], "seed": key[1], "policy": policy, "final_best_win": 1})
    wins = pd.DataFrame(win_rows).groupby("policy", as_index=False)["final_best_win"].sum()
    return pd.DataFrame(hit_rows), wins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=50000)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--bootstrap-size", type=float, default=0.9)
    parser.add_argument("--xi", type=float, default=4.0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--statistical-seed", type=int, default=20260822)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    with ASSAY_FILE.open(newline="", encoding="utf-8") as handle:
        assays = [row["assay"] for row in csv.DictReader(handle)]
    rows: list[dict[str, object]] = []
    pool_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                simulate_assay,
                assay,
                seeds,
                args.rounds,
                args.batch,
                args.max_candidates,
                args.n_splits,
                args.bootstrap_size,
                args.xi,
            ): assay
            for assay in assays
        }
        for future in as_completed(futures):
            assay_rows, pool_row = future.result()
            rows.extend(assay_rows)
            pool_rows.append(pool_row)
            print(f"{futures[future]}\t{pool_row['status']}\t{len(assay_rows)}", flush=True)
    metrics = pd.DataFrame(rows)
    pool_table = pd.DataFrame(pool_rows).sort_values("assay")
    summary = summary_table(metrics, args.bootstrap, args.statistical_seed)
    tests = paired_tests(metrics, args.bootstrap, args.permutations, args.statistical_seed)
    hit_time, wins = hit_time_and_wins(metrics)
    metrics.to_csv(OUT_DIR / "active_learning_external_common_pool_metrics.csv", index=False)
    pool_table.to_csv(OUT_DIR / "active_learning_external_common_pool_assays.csv", index=False)
    summary.to_csv(OUT_DIR / "active_learning_external_common27_summary.csv", index=False)
    tests.to_csv(OUT_DIR / "active_learning_external_common27_paired_tests.csv", index=False)
    hit_time.to_csv(OUT_DIR / "active_learning_external_common_pool_hit_time.csv", index=False)
    wins.to_csv(OUT_DIR / "active_learning_external_common_pool_final_wins.csv", index=False)
    evolve_metrics = metrics[metrics["policy"].eq(EVOLVEPRO_POLICY)].copy()
    evolve_summary = summary[summary["policy"].eq(EVOLVEPRO_POLICY)].copy()
    evolve_hit = hit_time[hit_time["policy"].eq(EVOLVEPRO_POLICY)].copy()
    evolve_wins = wins[wins["policy"].eq(EVOLVEPRO_POLICY)].copy()
    evolve_metrics.to_csv(OUT_DIR / "evolvepro_style_active_learning_metrics.csv", index=False)
    evolve_summary.to_csv(OUT_DIR / "evolvepro_style_active_learning_summary.csv", index=False)
    evolve_hit.to_csv(OUT_DIR / "evolvepro_style_active_learning_hit_time.csv", index=False)
    evolve_wins.to_csv(OUT_DIR / "evolvepro_style_active_learning_final_wins.csv", index=False)
    evolve_metadata = {
        "source_method": "EVOLVEpro-style active learning using the official random-forest top-n model logic.",
        "adaptation": "The EVOLVEpro random-forest acquisition rule uses Kermut ESM2-650M mean-pooled embeddings on the shared ProteinGym candidate pools.",
        "candidate_pool": "The same embedding-covered, label-independent candidate pool used by every policy in the common-pool comparison.",
        "retain_true_top_candidates": False,
        "discovery_target": "A measured variant in the top 1% of the shared embedding-covered candidate pool.",
        "shared_initial_measurements_across_policies": True,
        "measurement_schedule": {"initial": 20, "rounds": args.rounds, "batch": args.batch},
        "workers": args.workers,
        "n_assays_requested": len(assays),
        "n_assays_evaluable": int(evolve_metrics["assay"].nunique()),
        "seeds": seeds,
        "official_repository_commit": "1c77697d0c09bf6989a1562a55da99301a12e2cd",
        "embedding_repository_commit": "e6500221b5159a1896272bfc4a3c6e3896eb0027",
        "representation": "ESM2-650M mean-pooled embeddings",
    }
    (OUT_DIR / "evolvepro_style_active_learning_metadata.json").write_text(
        json.dumps(evolve_metadata, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "candidate_pool": "Intersection of ProteinGym scored candidates and available official Kermut ESM2-650M embeddings, with label-independent random capping if needed.",
        "candidate_pool_hash_recorded": True,
        "retain_true_top_candidates": False,
        "discovery_target": "A measured variant in the top 1% of the shared embedding-covered candidate pool.",
        "shared_initial_measurements_across_policies": True,
        "measurement_schedule": {"initial": 20, "rounds": args.rounds, "batch": args.batch},
        "workers": args.workers,
        "policies": POLICIES,
        "statistical_unit": "Seeds were averaged within assay before confidence intervals and paired tests.",
        "multiple_testing": "Benjamini-Hochberg correction across all common-pool paired tests.",
        "assays_requested": len(assays),
        "assays_evaluable": int(metrics["assay"].nunique()),
        "official_repository_commits": {
            "ALDE": "d0b3593dd17987c9864830d36f5976b1ad04a619",
            "EVOLVEpro": "1c77697d0c09bf6989a1562a55da99301a12e2cd",
            "Kermut embedding source": "e6500221b5159a1896272bfc4a3c6e3896eb0027",
        },
        "external_method_boundaries": {
            "EVOLVEpro-style": "Official random-forest top-n logic with ESM2-650M rather than ESM2-15B embeddings.",
            "ALDE-style": "Official boosting-ensemble acquisition forms with mutation-identity features and shifted, scaled ProteinGym targets.",
        },
    }
    (OUT_DIR / "active_learning_external_common_pool_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
