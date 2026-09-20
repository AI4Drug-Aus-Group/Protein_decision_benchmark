from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESIDUALS = ROOT / "results" / "epistasis" / "strict_double_epistasis_residuals.csv"
LOW_N_SPLITS = ROOT / "results" / "splits" / "low_n_splits.csv"
SCORE_DIR = ROOT / "proteingym" / "extracted" / "zero_shot_substitutions_scores"
DMS_DIR = ROOT / "proteingym" / "extracted" / "DMS_ProteinGym_substitutions"
KERMUT_DATA = ROOT / "external_repos" / "kermut" / "data"
KERMUT_REFERENCE = KERMUT_DATA / "DMS_substitutions.csv"
KERMUT_ZERO = KERMUT_DATA / "zero_shot_fitness_predictions" / "ESM2" / "650M"
KERMUT_EMBEDDINGS = KERMUT_DATA / "embeddings"
KERMUT_CONDITIONAL = KERMUT_DATA / "conditional_probs" / "ProteinMPNN"
KERMUT_COORDS = KERMUT_DATA / "structures" / "coords"
DEFAULT_OUT = ROOT / "results" / "sota_extension" / "kermut_r4_resource_compatible_splits"
SEEDS = [0, 1, 2, 3, 4]
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


def read_h5_mutants(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with h5py.File(path, "r") as handle:
        values = handle["mutants"][:]
    return {value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values}


def kermut_resource_set(assay: str, reference_ids: set[str]) -> tuple[set[str], dict[str, object]]:
    assay_id = assay.removesuffix(".csv")
    required = {
        "dms": DMS_DIR / assay,
        "zero": KERMUT_ZERO / f"{assay_id}.csv",
        "single_embedding": KERMUT_EMBEDDINGS / "substitutions_singles" / "ESM2" / f"{assay_id}.h5",
        "multiple_embedding": KERMUT_EMBEDDINGS / "substitutions_multiples" / "ESM2" / f"{assay_id}.h5",
        "conditional_probabilities": KERMUT_CONDITIONAL / f"{assay_id}.npy",
        "coordinates": KERMUT_COORDS / f"{assay_id}.npy",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if assay_id not in reference_ids:
        missing.append("reference_sequence")
    if missing:
        return set(), {"assay": assay, "eligible": False, "reason": "missing_kermut_files", "missing_files": ";".join(missing)}
    dms = set(pd.read_csv(required["dms"], usecols=["mutant"])["mutant"].dropna().astype(str))
    zero = set(pd.read_csv(required["zero"], usecols=["mutant"])["mutant"].dropna().astype(str))
    embeddings = read_h5_mutants(required["single_embedding"]) | read_h5_mutants(required["multiple_embedding"])
    complete = dms & zero & embeddings
    return complete, {
        "assay": assay,
        "eligible": True,
        "n_dms_mutants": len(dms),
        "n_kermut_zero_shot_mutants": len(zero),
        "n_kermut_embedding_mutants": len(embeddings),
        "n_complete_resource_mutants": len(complete),
        "complete_resource_sha256": hashlib.sha256("\n".join(sorted(complete)).encode("utf-8")).hexdigest(),
    }


def calibrated_prosst(rows: pd.DataFrame, assay: str) -> np.ndarray:
    score_path = SCORE_DIR / assay
    if not score_path.exists():
        return np.full(len(rows), np.nan)
    scores = pd.read_csv(score_path, usecols=lambda name: name in {"mutant", "ProSST-2048"})
    if "ProSST-2048" not in scores.columns:
        return np.full(len(rows), np.nan)
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


def fixed_holdout(n_rows: int, assay: str, seed: int, selected_budget: int, fraction: float, minimum: int, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    if n_rows - minimum < selected_budget:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    n_holdout = max(minimum, int(math.ceil(n_rows * fraction)))
    n_holdout = min(maximum, n_holdout, n_rows - selected_budget)
    rng = np.random.default_rng(stable_seed(assay, "kermut_resource_compatible_holdout", seed))
    holdout = np.sort(rng.choice(n_rows, size=n_holdout, replace=False))
    mask = np.ones(n_rows, dtype=bool)
    mask[holdout] = False
    return holdout, np.flatnonzero(mask)


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
        token_gain = (~covered_tokens[token_a[candidates]]).astype(int) + (~covered_tokens[token_b[candidates]]).astype(int)
        site_gain = (~covered_sites[site_a[candidates]]).astype(int) + (~covered_sites[site_b[candidates]]).astype(int)
        token_gain[selected] = -1
        keep = token_gain == token_gain.max()
        site_best = site_gain[keep].max()
        keep &= site_gain == site_best
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
        rng = np.random.default_rng(stable_seed(assay, "kermut_selected_double_random", seed))
        return rng.permutation(candidates)[:budget]
    if policy == "mutation_coverage":
        return coverage_selection(rows, candidates, budget)
    if policy == "extreme_additive_prediction":
        values = rows["additive_prediction"].to_numpy(float)[candidates]
        center = float(np.median(values))
        order = np.argsort(np.abs(values - center), kind="mergesort")
        return candidates[order[-budget:]]
    if policy == "additive_versus_calibrated_prosst_disagreement":
        values = np.abs(rows["additive_prediction"].to_numpy(float)[candidates] - rows["calibrated_prosst"].to_numpy(float)[candidates])
        values = np.nan_to_num(values, nan=-np.inf)
        order = np.argsort(values, kind="mergesort")
        return candidates[order[-budget:]]
    if policy == "oracle_absolute_epistatic_residual":
        values = rows["abs_epistasis_residual"].to_numpy(float)[candidates]
        order = np.argsort(values, kind="mergesort")
        return candidates[order[-budget:]]
    raise ValueError(policy)


def parse_assays(value: str) -> set[str]:
    return {item if item.endswith(".csv") else f"{item}.csv" for item in value.split(",") if item}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--assays", default="")
    parser.add_argument("--max-assays", type=int, default=0)
    parser.add_argument("--max-resource-mutants", type=int, default=0)
    parser.add_argument("--min-resource-doubles", type=int, default=70)
    parser.add_argument("--selected-double-budget", type=int, default=20)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-minimum", type=int, default=50)
    parser.add_argument("--holdout-maximum", type=int, default=100)
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    residuals = pd.read_csv(RESIDUALS)
    low = pd.read_csv(LOW_N_SPLITS)
    reference_ids = set(pd.read_csv(KERMUT_REFERENCE, usecols=["DMS_id"])["DMS_id"].astype(str))
    allowed = parse_assays(args.assays)
    if allowed:
        residuals = residuals[residuals["assay"].isin(allowed)].copy()

    assay_blocks = {}
    manifest_rows = []
    skipped_rows = []
    for assay, block in residuals.groupby("assay", sort=True):
        complete, manifest = kermut_resource_set(assay, reference_ids)
        block = block[block["mutant"].astype(str).isin(complete)].reset_index(drop=True).copy()
        n_resource_doubles = int(len(block))
        single_block = low[
            low["assay"].eq(assay)
            & low["regime"].eq("single_only")
            & low["budget"].eq(INITIAL_SINGLE_BUDGET)
        ].copy()
        complete_seeds = 0
        for seed in SEEDS:
            initial = single_block[single_block["seed"].eq(seed)].sort_values("rank")["mutant"].astype(str).tolist()
            if len(initial) == INITIAL_SINGLE_BUDGET and set(initial).issubset(complete):
                complete_seeds += 1
        manifest.update(
            n_resource_doubles=n_resource_doubles,
            n_total_component_resolved_doubles=int((residuals["assay"] == assay).sum()),
            single20_complete_seeds=complete_seeds,
        )
        if manifest.get("eligible") and n_resource_doubles >= args.min_resource_doubles and complete_seeds == len(SEEDS):
            if args.max_resource_mutants and manifest["n_complete_resource_mutants"] > args.max_resource_mutants:
                manifest["eligible"] = False
                manifest["reason"] = "above_max_resource_mutants"
            else:
                assay_blocks[assay] = block
        elif manifest.get("eligible"):
            manifest["eligible"] = False
            manifest["reason"] = "insufficient_resource_doubles_or_initial_singles"
        manifest_rows.append(manifest)

    selected_manifest = pd.DataFrame([row for row in manifest_rows if row.get("eligible")]).sort_values(
        ["n_resource_doubles", "assay"], ascending=[False, True]
    )
    if args.max_assays > 0:
        selected_manifest = selected_manifest.head(args.max_assays)
    selected_assays = set(selected_manifest["assay"].tolist())

    training_rows = []
    baseline_rows = []
    holdout_rows = []
    selection_rows = []
    for assay in selected_manifest["assay"]:
        block = assay_blocks[assay].reset_index(drop=True).copy()
        block["calibrated_prosst"] = calibrated_prosst(block, assay)
        single_block = low[
            low["assay"].eq(assay)
            & low["regime"].eq("single_only")
            & low["budget"].eq(INITIAL_SINGLE_BUDGET)
        ].copy()
        for seed in SEEDS:
            initial = single_block[single_block["seed"].eq(seed)].sort_values("rank")["mutant"].astype(str).tolist()
            if len(initial) != INITIAL_SINGLE_BUDGET:
                skipped_rows.append({"assay": assay, "seed": seed, "reason": "bad_initial_single_only_split_size"})
                continue
            holdout, candidates = fixed_holdout(
                len(block),
                assay,
                seed,
                args.selected_double_budget,
                args.holdout_fraction,
                args.holdout_minimum,
                args.holdout_maximum,
            )
            if len(holdout) < args.holdout_minimum or len(candidates) < args.selected_double_budget:
                skipped_rows.append({"assay": assay, "seed": seed, "reason": "insufficient_resource_compatible_holdout_or_candidates"})
                continue
            if np.isfinite(block["calibrated_prosst"].to_numpy(float)[candidates]).sum() < args.selected_double_budget:
                skipped_rows.append({"assay": assay, "seed": seed, "reason": "insufficient_calibrated_prosst_scores"})
                continue
            for rank, mutant in enumerate(initial, start=1):
                baseline_rows.append({
                    "assay": assay,
                    "seed": seed,
                    "budget": INITIAL_SINGLE_BUDGET,
                    "regime": "single20_baseline",
                    "rank": rank,
                    "mutant": mutant,
                })
            for idx in holdout:
                holdout_rows.append({"assay": assay, "seed": seed, "mutant": block.loc[idx, "mutant"], "role": "held_out_double"})
            for policy in POLICIES:
                selected = select_doubles(block, candidates, args.selected_double_budget, assay, seed, policy)
                regime = f"single20_plus_{policy}_double{args.selected_double_budget}"
                training_mutants = initial + block.loc[selected, "mutant"].astype(str).tolist()
                for rank, mutant in enumerate(training_mutants, start=1):
                    training_rows.append({
                        "assay": assay,
                        "seed": seed,
                        "budget": INITIAL_SINGLE_BUDGET + args.selected_double_budget,
                        "regime": regime,
                        "rank": rank,
                        "mutant": mutant,
                        "initial_single_budget": INITIAL_SINGLE_BUDGET,
                        "selected_double_budget": args.selected_double_budget,
                        "selection_policy": policy,
                    })
                for idx in selected:
                    selection_rows.append({
                        "assay": assay,
                        "seed": seed,
                        "selection_policy": policy,
                        "mutant": block.loc[idx, "mutant"],
                        "additive_prediction": block.loc[idx, "additive_prediction"],
                        "epistasis_residual": block.loc[idx, "epistasis_residual"],
                        "abs_epistasis_residual": block.loc[idx, "abs_epistasis_residual"],
                        "calibrated_prosst": block.loc[idx, "calibrated_prosst"],
                    })

    manifest = pd.DataFrame(manifest_rows)
    manifest["selected_for_kermut_r4"] = manifest["assay"].isin(selected_assays)
    manifest.to_csv(out / "kermut_resource_manifest.csv", index=False)
    pd.DataFrame(training_rows).to_csv(out / "selected_double_training_splits.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(out / "single20_baseline_splits.csv", index=False)
    pd.DataFrame(holdout_rows).to_csv(out / "selected_double_common_holdout.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out / "selected_double_selected_mutants.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(out / "selected_double_skipped_groups.csv", index=False)
    metadata = {
        "status": "ok",
        "n_candidate_assays_passing_resource_filters": int(len(selected_manifest)),
        "n_training_rows": int(len(training_rows)),
        "n_baseline_rows": int(len(baseline_rows)),
        "n_holdout_rows": int(len(holdout_rows)),
        "n_selected_mutant_rows": int(len(selection_rows)),
        "selected_assays": selected_manifest["assay"].tolist(),
        "initial_single_budget": INITIAL_SINGLE_BUDGET,
        "selected_double_budget": args.selected_double_budget,
        "policies": POLICIES,
        "resource_filter": {
            "min_resource_doubles": args.min_resource_doubles,
            "max_resource_mutants": args.max_resource_mutants if args.max_resource_mutants else None,
            "complete_single20_seeds_required": len(SEEDS),
        },
        "holdout": {
            "fraction": args.holdout_fraction,
            "minimum": args.holdout_minimum,
            "maximum": args.holdout_maximum,
        },
    }
    (out / "selected_double_split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
