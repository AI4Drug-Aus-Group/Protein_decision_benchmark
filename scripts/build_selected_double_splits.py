from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESIDUALS = ROOT / "results" / "epistasis" / "strict_double_epistasis_residuals.csv"
LOW_N_SPLITS = ROOT / "results" / "splits" / "low_n_splits.csv"
SCORE_DIR = ROOT / "proteingym" / "extracted" / "zero_shot_substitutions_scores"
OUT = ROOT / "results" / "sota_extension" / "selected_double_splits"
SEEDS = [0, 1, 2, 3, 4]
DOUBLE_BUDGETS = [20, 50, 100]
INITIAL_SINGLE_BUDGET = 20
POLICIES = [
    "random_doubles",
    "mutation_coverage",
    "extreme_additive_prediction",
    "additive_versus_calibrated_prosst_disagreement",
    "oracle_absolute_epistatic_residual",
]


def stable_seed(*parts: object) -> int:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def mutation_parts(mutant: str) -> list[str]:
    return [part for part in str(mutant).split(":") if len(part) >= 3]


def mutation_sites(mutant: str) -> tuple[str, ...]:
    return tuple(sorted(part[1:-1] for part in mutation_parts(mutant)))


def fixed_holdout(n_rows: int, assay: str, seed: int, fraction: float, minimum: int, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    feasible = [budget for budget in DOUBLE_BUDGETS if n_rows - minimum >= budget]
    if not feasible:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    largest = max(feasible)
    n_holdout = max(minimum, int(math.ceil(n_rows * fraction)))
    n_holdout = min(maximum, n_holdout, n_rows - largest)
    rng = np.random.default_rng(stable_seed(assay, "common_holdout", seed))
    holdout = np.sort(rng.choice(n_rows, size=n_holdout, replace=False))
    mask = np.ones(n_rows, dtype=bool)
    mask[holdout] = False
    return holdout, np.flatnonzero(mask)


def calibrated_prosst(rows: pd.DataFrame, assay: str) -> np.ndarray:
    score_path = SCORE_DIR / assay
    if not score_path.exists():
        return np.full(len(rows), np.nan)
    scores = pd.read_csv(score_path, usecols=lambda name: name in {"mutant", "ProSST-2048"})
    singles = pd.concat(
        [
            rows[["single_a", "score_single_a"]].rename(columns={"single_a": "mutant", "score_single_a": "DMS_score"}),
            rows[["single_b", "score_single_b"]].rename(columns={"single_b": "mutant", "score_single_b": "DMS_score"}),
        ],
        ignore_index=True,
    ).drop_duplicates("mutant")
    singles = singles.merge(scores, on="mutant", how="inner").dropna()
    doubles = rows[["mutant"]].merge(scores, on="mutant", how="left")
    if len(singles) < 3 or singles["ProSST-2048"].nunique() < 2:
        return np.full(len(rows), np.nan)
    design = np.column_stack([np.ones(len(singles)), singles["ProSST-2048"].to_numpy(float)])
    penalty = np.diag([0.0, 1e-8])
    weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ singles["DMS_score"].to_numpy(float)
    return weights[0] + weights[1] * doubles["ProSST-2048"].to_numpy(float)


def coverage_selection(rows: pd.DataFrame, candidates: np.ndarray, budget: int) -> np.ndarray:
    tokens = [mutation_parts(value) for value in rows["mutant"]]
    sites = [mutation_sites(value) for value in rows["mutant"]]
    token_lookup = {value: i for i, value in enumerate(sorted({x for pair in tokens for x in pair}))}
    site_lookup = {value: i for i, value in enumerate(sorted({x for pair in sites for x in pair}))}
    token_a = np.array([token_lookup[pair[0]] for pair in tokens])
    token_b = np.array([token_lookup[pair[1]] for pair in tokens])
    site_a = np.array([site_lookup[pair[0]] for pair in sites])
    site_b = np.array([site_lookup[pair[1]] for pair in sites])
    selected = np.zeros(len(candidates), dtype=bool)
    covered_tokens = np.zeros(len(token_lookup), dtype=bool)
    covered_sites = np.zeros(len(site_lookup), dtype=bool)
    local = np.arange(len(candidates))
    output = []
    for _ in range(budget):
        tg = (~covered_tokens[token_a[candidates]]).astype(int) + (~covered_tokens[token_b[candidates]]).astype(int)
        sg = (~covered_sites[site_a[candidates]]).astype(int) + (~covered_sites[site_b[candidates]]).astype(int)
        tg[selected] = -1
        best_token = tg.max()
        keep = tg == best_token
        best_site = sg[keep].max()
        keep &= sg == best_site
        chosen_local = int(local[keep].min())
        selected[chosen_local] = True
        chosen = int(candidates[chosen_local])
        output.append(chosen)
        covered_tokens[token_a[chosen]] = True
        covered_tokens[token_b[chosen]] = True
        covered_sites[site_a[chosen]] = True
        covered_sites[site_b[chosen]] = True
    return np.array(output, dtype=np.int64)


def select_doubles(rows: pd.DataFrame, candidates: np.ndarray, budget: int, assay: str, seed: int, policy: str) -> np.ndarray:
    if policy == "random_doubles":
        rng = np.random.default_rng(stable_seed(assay, "selected_double_random", seed))
        return rng.permutation(candidates)[:budget]
    if policy == "mutation_coverage":
        return coverage_selection(rows, candidates, budget)
    if policy == "extreme_additive_prediction":
        values = rows["additive_prediction"].to_numpy(float)[candidates]
        center = float(np.median(values))
        order = np.argsort(np.abs(values - center), kind="mergesort")
        return candidates[order[-budget:]]
    if policy == "additive_versus_calibrated_prosst_disagreement":
        calibrated = rows["calibrated_prosst"].to_numpy(float)[candidates]
        values = np.abs(rows["additive_prediction"].to_numpy(float)[candidates] - calibrated)
        values = np.nan_to_num(values, nan=-np.inf)
        order = np.argsort(values, kind="mergesort")
        return candidates[order[-budget:]]
    if policy == "oracle_absolute_epistatic_residual":
        values = rows["abs_epistasis_residual"].to_numpy(float)[candidates]
        order = np.argsort(values, kind="mergesort")
        return candidates[order[-budget:]]
    raise ValueError(policy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-minimum", type=int, default=50)
    parser.add_argument("--holdout-maximum", type=int, default=10000)
    parser.add_argument("--assays", default="")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    residuals = pd.read_csv(RESIDUALS)
    low = pd.read_csv(LOW_N_SPLITS)
    allowed = {value if value.endswith(".csv") else f"{value}.csv" for value in args.assays.split(",") if value}
    if allowed:
        residuals = residuals[residuals["assay"].isin(allowed)].copy()
    rows = []
    holdout_rows = []
    skipped = []
    for assay, block in residuals.groupby("assay", sort=True):
        block = block.reset_index(drop=True).copy()
        block["calibrated_prosst"] = calibrated_prosst(block, assay)
        single_block = low[
            low["assay"].eq(assay)
            & low["regime"].eq("single_only")
            & low["budget"].eq(INITIAL_SINGLE_BUDGET)
        ].copy()
        if single_block.empty:
            skipped.append({"assay": assay, "reason": "missing_initial_single_only_split"})
            continue
        for seed in SEEDS:
            initial = single_block[single_block["seed"].eq(seed)].sort_values("rank")["mutant"].astype(str).tolist()
            if len(initial) != INITIAL_SINGLE_BUDGET:
                skipped.append({"assay": assay, "seed": seed, "reason": "bad_initial_single_only_split_size"})
                continue
            holdout, candidates = fixed_holdout(len(block), assay, seed, args.holdout_fraction, args.holdout_minimum, args.holdout_maximum)
            if len(holdout) < args.holdout_minimum:
                skipped.append({"assay": assay, "seed": seed, "reason": "insufficient_common_holdout"})
                continue
            for idx in holdout:
                holdout_rows.append({"assay": assay, "seed": seed, "mutant": block.loc[idx, "mutant"], "role": "held_out_double"})
            for double_budget in DOUBLE_BUDGETS:
                if len(candidates) < double_budget:
                    skipped.append({"assay": assay, "seed": seed, "double_budget": double_budget, "reason": "insufficient_acquisition_candidates"})
                    continue
                total_budget = INITIAL_SINGLE_BUDGET + double_budget
                for policy in POLICIES:
                    if policy == "additive_versus_calibrated_prosst_disagreement" and block["calibrated_prosst"].isna().all():
                        continue
                    selected = select_doubles(block, candidates, double_budget, assay, seed, policy)
                    regime = f"single20_plus_{policy}_double{double_budget}"
                    for rank, mutant in enumerate(initial + block.loc[selected, "mutant"].astype(str).tolist(), start=1):
                        rows.append({
                            "assay": assay,
                            "seed": seed,
                            "budget": total_budget,
                            "regime": regime,
                            "rank": rank,
                            "mutant": mutant,
                            "initial_single_budget": INITIAL_SINGLE_BUDGET,
                            "selected_double_budget": double_budget,
                            "selection_policy": policy,
                        })
    split = pd.DataFrame(rows)
    holdout_frame = pd.DataFrame(holdout_rows)
    skipped_frame = pd.DataFrame(skipped)
    split.to_csv(out / "selected_double_training_splits.csv", index=False)
    holdout_frame.to_csv(out / "selected_double_common_holdout.csv", index=False)
    skipped_frame.to_csv(out / "selected_double_skipped_groups.csv", index=False)
    metadata = {
        "status": "ok",
        "n_training_rows": int(len(split)),
        "n_assays": int(split["assay"].nunique()) if not split.empty else 0,
        "total_budgets": sorted(split["budget"].dropna().astype(int).unique().tolist()) if not split.empty else [],
        "selected_double_budgets": DOUBLE_BUDGETS,
        "initial_single_budget": INITIAL_SINGLE_BUDGET,
        "policies": POLICIES,
        "holdout_file": str((out / "selected_double_common_holdout.csv").relative_to(ROOT)),
    }
    (out / "selected_double_split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
