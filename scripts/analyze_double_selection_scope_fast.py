from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr


BUDGETS = [20, 50, 100]
SEEDS = [0, 1, 2, 3, 4]
POLICIES = [
    "random doubles",
    "mutation coverage",
    "extreme additive prediction",
    "additive versus calibrated ProSST disagreement",
    "oracle absolute epistatic residual",
]
MODELS = [
    "additive plus global residual mean",
    "additive plus site-pair residual",
    "mutation-identity ridge",
    "additive plus mutation-identity ridge",
]
METRICS = [
    "spearman",
    "delta_spearman_over_additive",
    "top1pct_recall_at_100",
    "best_top100_percentile",
]


def mutation_parts(mutant: str) -> list[str]:
    return [part for part in mutant.split(":") if len(part) >= 3]


def mutation_sites(mutant: str) -> tuple[str, ...]:
    return tuple(sorted(part[1:-1] for part in mutation_parts(mutant)))


def stable_seed(*parts: object) -> int:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def calibrated_prosst(rows: pd.DataFrame, score_table: pd.DataFrame) -> pd.Series:
    singles = pd.concat(
        [
            rows[["single_a", "score_single_a"]].rename(columns={"single_a": "mutant", "score_single_a": "DMS_score"}),
            rows[["single_b", "score_single_b"]].rename(columns={"single_b": "mutant", "score_single_b": "DMS_score"}),
        ],
        ignore_index=True,
    ).drop_duplicates("mutant")
    singles = singles.merge(score_table[["mutant", "ProSST-2048"]], on="mutant", how="inner").dropna()
    doubles = rows[["mutant"]].merge(score_table[["mutant", "ProSST-2048"]], on="mutant", how="left")
    if len(singles) < 3 or singles["ProSST-2048"].nunique() < 2:
        return pd.Series(np.nan, index=rows.index)
    design = np.column_stack([np.ones(len(singles)), singles["ProSST-2048"].to_numpy(float)])
    weights = np.linalg.pinv(design.T @ design + np.diag([0.0, 1e-8])) @ design.T @ singles["DMS_score"].to_numpy(float)
    return pd.Series(weights[0] + weights[1] * doubles["ProSST-2048"].to_numpy(float), index=rows.index)


def token_matrix(token_lists: list[list[str]]) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    vocabulary = sorted({token for tokens in token_lists for token in tokens})
    lookup = {token: index for index, token in enumerate(vocabulary)}
    row_indices: list[int] = []
    col_indices: list[int] = []
    for row_index, tokens in enumerate(token_lists):
        for token in tokens:
            row_indices.append(row_index)
            col_indices.append(lookup[token])
    values = np.ones(len(row_indices), dtype=np.float64)
    matrix = sparse.csr_matrix((values, (row_indices, col_indices)), shape=(len(token_lists), len(vocabulary)))
    first = np.array([lookup[tokens[0]] for tokens in token_lists], dtype=np.int32)
    second = np.array([lookup[tokens[1]] for tokens in token_lists], dtype=np.int32)
    return matrix, first, second


def categorical_codes(values: list[tuple[str, ...]]) -> tuple[np.ndarray, int]:
    lookup = {value: index for index, value in enumerate(sorted(set(values)))}
    return np.array([lookup[value] for value in values], dtype=np.int32), len(lookup)


def fixed_holdout(
    n_rows: int,
    assay: str,
    seed: int,
    fraction: float,
    minimum: int,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    feasible_budgets = [budget for budget in BUDGETS if n_rows - minimum >= budget]
    if not feasible_budgets:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    largest_feasible_budget = max(feasible_budgets)
    n_holdout = max(minimum, int(math.ceil(n_rows * fraction)))
    n_holdout = min(maximum, n_holdout, n_rows - largest_feasible_budget)
    rng = np.random.default_rng(stable_seed(assay, "common_holdout", seed))
    holdout = np.sort(rng.choice(n_rows, size=n_holdout, replace=False))
    mask = np.ones(n_rows, dtype=bool)
    mask[holdout] = False
    return holdout, np.flatnonzero(mask)


def coverage_selection(cache: dict[str, object], candidates: np.ndarray, budget: int) -> np.ndarray:
    token_a = cache["token_a"][candidates]
    token_b = cache["token_b"][candidates]
    site_a = cache["site_a"][candidates]
    site_b = cache["site_b"][candidates]
    selected = np.zeros(len(candidates), dtype=bool)
    covered_tokens = np.zeros(int(cache["n_tokens"]), dtype=bool)
    covered_sites = np.zeros(int(cache["n_sites"]), dtype=bool)
    output: list[int] = []
    local_indices = np.arange(len(candidates), dtype=np.int64)
    for _ in range(budget):
        token_gain = (~covered_tokens[token_a]).astype(np.int8) + (~covered_tokens[token_b]).astype(np.int8)
        site_gain = (~covered_sites[site_a]).astype(np.int8) + (~covered_sites[site_b]).astype(np.int8)
        token_gain[selected] = -1
        best_token = token_gain.max()
        eligible = token_gain == best_token
        best_site = site_gain[eligible].max()
        eligible &= site_gain == best_site
        chosen = int(local_indices[eligible].min())
        output.append(chosen)
        selected[chosen] = True
        covered_tokens[token_a[chosen]] = True
        covered_tokens[token_b[chosen]] = True
        covered_sites[site_a[chosen]] = True
        covered_sites[site_b[chosen]] = True
    return candidates[np.asarray(output, dtype=np.int64)]


def select_indices(
    cache: dict[str, object],
    candidates: np.ndarray,
    policy: str,
    budget: int,
    assay: str,
    seed: int,
) -> np.ndarray:
    if policy == "random doubles":
        rng = np.random.default_rng(stable_seed(assay, "random_selection", seed))
        ordering = rng.permutation(candidates)
        return ordering[:budget]
    if policy == "mutation coverage":
        return coverage_selection(cache, candidates, budget)
    if policy == "extreme additive prediction":
        values = cache["additive"][candidates]
        center = float(np.median(values))
        order = np.argsort(np.abs(values - center), kind="mergesort")
        return candidates[order[-budget:]]
    if policy == "additive versus calibrated ProSST disagreement":
        values = np.abs(cache["additive"][candidates] - cache["calibrated_prosst"][candidates])
        values = np.nan_to_num(values, nan=-np.inf)
        order = np.argsort(values, kind="mergesort")
        return candidates[order[-budget:]]
    if policy == "oracle absolute epistatic residual":
        order = np.argsort(cache["abs_residual"][candidates], kind="mergesort")
        return candidates[order[-budget:]]
    raise ValueError(policy)


def ridge_prediction(
    cache: dict[str, object],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    include_additive: bool,
) -> np.ndarray:
    matrix = cache["token_matrix"]
    columns = np.unique(matrix[train_idx].nonzero()[1])
    train_base = [np.ones(len(train_idx)), cache["order"][train_idx]]
    test_base = [np.ones(len(test_idx)), cache["order"][test_idx]]
    if include_additive:
        train_base.append(cache["additive"][train_idx])
        test_base.append(cache["additive"][test_idx])
    train_x = np.column_stack(train_base + [matrix[train_idx][:, columns].toarray()])
    penalty = np.eye(train_x.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(train_x.T @ train_x + penalty) @ train_x.T @ cache["truth"][train_idx]
    base_width = 3 if include_additive else 2
    prediction = np.column_stack(test_base) @ weights[:base_width]
    if len(columns):
        prediction = prediction + matrix[test_idx][:, columns] @ weights[base_width:]
    return np.asarray(prediction).reshape(-1)


def predict(cache: dict[str, object], train_idx: np.ndarray, test_idx: np.ndarray, model: str) -> np.ndarray:
    if model == "additive plus global residual mean":
        return cache["additive"][test_idx] + float(cache["residual"][train_idx].mean())
    if model == "additive plus site-pair residual":
        pair = cache["site_pair"]
        counts = np.bincount(pair[train_idx], minlength=int(cache["n_site_pairs"]))
        sums = np.bincount(pair[train_idx], weights=cache["residual"][train_idx], minlength=int(cache["n_site_pairs"]))
        fallback = float(cache["residual"][train_idx].mean())
        means = np.divide(sums, counts, out=np.full_like(sums, fallback, dtype=float), where=counts > 0)
        return cache["additive"][test_idx] + means[pair[test_idx]]
    if model == "mutation-identity ridge":
        return ridge_prediction(cache, train_idx, test_idx, False)
    if model == "additive plus mutation-identity ridge":
        return ridge_prediction(cache, train_idx, test_idx, True)
    raise ValueError(model)


def top_one_percent_recall(prediction: np.ndarray, truth: np.ndarray, k: int = 100) -> float:
    n_top = max(1, math.ceil(len(truth) * 0.01))
    true_top = set(np.argsort(truth)[-n_top:])
    predicted_top = set(np.argsort(prediction)[-min(k, len(prediction)):])
    return len(true_top & predicted_top) / len(true_top)


def best_selected_percentile(prediction: np.ndarray, truth: np.ndarray, k: int = 100) -> float:
    selected = np.argsort(prediction)[-min(k, len(prediction)):]
    best = float(np.max(truth[selected]))
    return float(np.searchsorted(np.sort(truth), best, side="right") / len(truth))


def evaluate(
    cache: dict[str, object],
    assay: str,
    budget: int,
    seed: int,
    policy: str,
    model: str,
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    acquisition_size: int,
) -> dict[str, object]:
    prediction = predict(cache, train_idx, holdout_idx, model)
    truth = cache["truth"][holdout_idx]
    additive = cache["additive"][holdout_idx]
    model_spearman = float(spearmanr(prediction, truth).statistic)
    additive_spearman = float(spearmanr(additive, truth).statistic)
    return {
        "assay": assay,
        "budget": budget,
        "seed": seed,
        "selection_policy": policy,
        "downstream_model": model,
        "n_train_doubles": len(train_idx),
        "n_acquisition_candidates": acquisition_size,
        "n_common_holdout_doubles": len(holdout_idx),
        "component_single_mutants_used_as_side_information": True,
        "spearman": model_spearman,
        "additive_spearman": additive_spearman,
        "delta_spearman_over_additive": model_spearman - additive_spearman,
        "top1pct_recall_at_100": top_one_percent_recall(prediction, truth),
        "best_top100_percentile": best_selected_percentile(prediction, truth),
    }


def build_cache(rows: pd.DataFrame) -> dict[str, object]:
    token_lists = [mutation_parts(mutant) for mutant in rows["mutant"]]
    matrix, token_a, token_b = token_matrix(token_lists)
    site_lists = [mutation_sites(mutant) for mutant in rows["mutant"]]
    site_pair, n_site_pairs = categorical_codes(site_lists)
    sites = sorted({site for pair in site_lists for site in pair})
    site_lookup = {site: index for index, site in enumerate(sites)}
    site_a = np.array([site_lookup[pair[0]] for pair in site_lists], dtype=np.int32)
    site_b = np.array([site_lookup[pair[1]] for pair in site_lists], dtype=np.int32)
    return {
        "n_rows": len(rows),
        "truth": rows["score_double"].to_numpy(float),
        "additive": rows["additive_prediction"].to_numpy(float),
        "residual": rows["epistasis_residual"].to_numpy(float),
        "abs_residual": rows["abs_epistasis_residual"].to_numpy(float),
        "calibrated_prosst": rows["calibrated_prosst"].to_numpy(float),
        "order": rows["mutant"].str.count(":").to_numpy(float) + 1.0,
        "token_matrix": matrix,
        "token_a": token_a,
        "token_b": token_b,
        "n_tokens": matrix.shape[1],
        "site_a": site_a,
        "site_b": site_b,
        "n_sites": len(sites),
        "site_pair": site_pair,
        "n_site_pairs": n_site_pairs,
    }


def process_assay(
    task: tuple[str, pd.DataFrame, Path, float, int, int],
) -> list[dict[str, object]]:
    assay, rows, score_dir, holdout_fraction, holdout_minimum, holdout_maximum = task
    rows = rows.reset_index(drop=True)
    score_path = score_dir / assay
    if not score_path.exists():
        return []
    score_table = pd.read_csv(score_path, usecols=lambda name: name in {"mutant", "ProSST-2048"})
    rows["calibrated_prosst"] = calibrated_prosst(rows, score_table)
    cache = build_cache(rows)
    output: list[dict[str, object]] = []
    for seed in SEEDS:
        holdout_idx, acquisition_idx = fixed_holdout(
            len(rows), assay, seed, holdout_fraction, holdout_minimum, holdout_maximum
        )
        if len(holdout_idx) < holdout_minimum:
            continue
        for budget in BUDGETS:
            if len(acquisition_idx) < budget:
                continue
            for policy in POLICIES:
                if policy == "additive versus calibrated ProSST disagreement" and rows["calibrated_prosst"].isna().all():
                    continue
                train_idx = select_indices(cache, acquisition_idx, policy, budget, assay, seed)
                for model in MODELS:
                    output.append(
                        evaluate(
                            cache,
                            assay,
                            budget,
                            seed,
                            policy,
                            model,
                            train_idx,
                            holdout_idx,
                            len(acquisition_idx),
                        )
                    )
    return output


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


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["budget", "selection_policy", "downstream_model", "epistasis_quartile"]
    assay_means = detail.groupby(group_columns + ["assay"], observed=True, dropna=False, as_index=False)[METRICS].mean()
    run_counts = detail.groupby(group_columns, observed=True, dropna=False).size().rename("n_runs").reset_index()
    summary = (
        assay_means.groupby(group_columns, observed=True, dropna=False)
        .agg(
            n_assays=("assay", "nunique"),
            mean_spearman=("spearman", "mean"),
            mean_delta_spearman_over_additive=("delta_spearman_over_additive", "mean"),
            mean_top1pct_recall_at_100=("top1pct_recall_at_100", "mean"),
            mean_best_top100_percentile=("best_top100_percentile", "mean"),
        )
        .reset_index()
        .merge(run_counts, on=group_columns, how="left", validate="one_to_one")
    )
    return summary


def paired_policy_tests(
    detail: pd.DataFrame,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    keys = ["assay", "budget", "seed", "downstream_model", "epistasis_quartile"]
    baseline = detail.loc[detail["selection_policy"].eq("random doubles"), keys + METRICS]
    baseline = baseline.rename(columns={metric: f"random_{metric}" for metric in METRICS})
    compared = detail.loc[~detail["selection_policy"].eq("random doubles")].merge(
        baseline, on=keys, how="inner", validate="many_to_one"
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    group_columns = ["budget", "downstream_model", "selection_policy", "epistasis_quartile"]
    for group_key, group in compared.groupby(group_columns, observed=True, dropna=False, sort=True):
        budget, model, policy, quartile = group_key
        for metric in METRICS:
            group = group.copy()
            group["paired_delta"] = group[metric] - group[f"random_{metric}"]
            assay_values = group.groupby("assay")["paired_delta"].mean().to_numpy(float)
            low, high = bootstrap_mean(assay_values, rng, bootstrap)
            rows.append(
                {
                    "budget": int(budget),
                    "downstream_model": model,
                    "selection_policy": policy,
                    "epistasis_quartile": quartile,
                    "metric": metric,
                    "n_assays": len(assay_values),
                    "mean_paired_delta_vs_random": float(np.mean(assay_values)),
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "two_sided_sign_flip_p": sign_flip(assay_values, rng, permutations),
                }
            )
    tests = pd.DataFrame(rows)
    tests["benjamini_hochberg_q_all_tests"] = benjamini_hochberg(tests["two_sided_sign_flip_p"])
    return tests.sort_values(group_columns + ["metric"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "double_selection_scope")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-minimum", type=int, default=50)
    parser.add_argument("--holdout-maximum", type=int, default=10000)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    residual_path = root / "results" / "epistasis" / "strict_double_epistasis_residuals.csv"
    score_dir = root / "proteingym" / "extracted" / "zero_shot_substitutions_scores"
    epistasis_path = root / "results" / "decision_map" / "assay_epistasis_strength.csv"
    residuals = pd.read_csv(residual_path)
    tasks = [
        (assay, rows.copy(), score_dir, args.holdout_fraction, args.holdout_minimum, args.holdout_maximum)
        for assay, rows in residuals.groupby("assay", sort=True)
    ]
    output_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_assay, task): task[0] for task in tasks}
        for future in as_completed(futures):
            assay = futures[future]
            rows = future.result()
            output_rows.extend(rows)
            print(f"{assay}\t{len(rows)}", flush=True)
    detail = pd.DataFrame(output_rows)
    epistasis = pd.read_csv(epistasis_path)
    epistasis["epistasis_quartile"] = pd.qcut(
        epistasis["normalized_epistasis_strength"],
        q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
        duplicates="drop",
    )
    epistasis_lookup = epistasis.set_index("assay")[["normalized_epistasis_strength", "epistasis_quartile"]]
    detail = detail.join(epistasis_lookup, on="assay")
    summary = summarize(detail)
    all_tests = paired_policy_tests(detail, args.bootstrap, args.permutations, args.seed)
    primary_tests = all_tests[all_tests["metric"].isin(["spearman", "delta_spearman_over_additive"])].copy()
    detail.to_csv(out / "double_selection_scope_detail.csv", index=False)
    summary.to_csv(out / "double_selection_scope_summary.csv", index=False)
    primary_tests.to_csv(out / "double_selection_policy_paired_tests.csv", index=False)
    all_tests.to_csv(out / "double_selection_policy_paired_tests_all_metrics.csv", index=False)
    metadata = {
        "runner": "fixed_common_holdout_sparse",
        "residual_source": str(residual_path.relative_to(root)),
        "score_source": str(score_dir.relative_to(root)),
        "epistasis_source": str(epistasis_path.relative_to(root)),
        "budgets": BUDGETS,
        "seeds": SEEDS,
        "selection_policies": POLICIES,
        "downstream_models": MODELS,
        "common_holdout": {
            "fraction": args.holdout_fraction,
            "minimum": args.holdout_minimum,
            "maximum": args.holdout_maximum,
            "shared_across_policies_and_budgets_within_each_assay_seed": True,
        },
        "component_single_mutants": "All available component single mutants were used as side information for the additive control and ProSST calibration. They were not counted in the double-mutant acquisition budget.",
        "oracle_policy": "Retrospective selection by measured absolute epistatic residual in the acquisition pool; reserved holdout variants are used for evaluation.",
        "statistical_unit": "Seeds were averaged within assay before assay-level bootstrap and sign-flip tests.",
        "multiple_testing": "Benjamini-Hochberg correction across all paired policy tests.",
        "workers": args.workers,
        "bootstrap_replicates": args.bootstrap,
        "sign_flip_permutations": args.permutations,
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
