from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOTA = ROOT / "results" / "sota_extension"
DEFAULT_OUT = SOTA / "sample_sets_expanded"
LOW_SPLITS = ROOT / "results" / "splits" / "low_n_splits.csv"
SELECTED_SPLITS = SOTA / "selected_double_splits" / "selected_double_training_splits.csv"
SELECTED_HOLDOUT = SOTA / "selected_double_splits" / "selected_double_common_holdout.csv"
DMS_DIR = ROOT / "proteingym" / "extracted" / "DMS_ProteinGym_substitutions"
PNPT = ROOT / "external_repos" / "ProteinNPT"
PNPT_REFERENCE = PNPT / "proteinnpt" / "utils" / "proteingym" / "DMS_substitutions.csv"
PNPT_DATA = PNPT / "ProteinNPT_data"
ZERO_DIR = PNPT_DATA / "data" / "zero_shot_fitness_predictions" / "substitutions"
MSA_FILES = PNPT_DATA / "data" / "MSA" / "MSA_files"
MSA_WEIGHTS = PNPT_DATA / "data" / "MSA" / "MSA_weights"
KERMUT_R3_ASSAY_MEANS = SOTA / "single_only_to_multi_kermut_partial" / "external_single_only_to_multi_assay_means.csv"
PREVIOUS_R3 = {
    "GFP_AEQVI_Sarkisyan_2016.csv",
    "GRB2_HUMAN_Faure_2021.csv",
    "HIS7_YEAST_Pokusaeva_2019.csv",
}
PREVIOUS_R4 = {
    "F7YBW8_MESOW_Aakre_2015.csv",
    "GFP_AEQVI_Sarkisyan_2016.csv",
    "GRB2_HUMAN_Faure_2021.csv",
    "HIS7_YEAST_Pokusaeva_2019.csv",
}
SELECTED_DOUBLE_POLICIES = [
    "random_doubles",
    "mutation_coverage",
    "extreme_additive_prediction",
    "additive_versus_calibrated_prosst_disagreement",
    "oracle_absolute_epistatic_residual",
]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item != ""]


def mutation_order(mutant: str) -> int:
    mutant = str(mutant)
    if mutant in {"", "WT", "wildtype", "_wt"}:
        return 0
    return mutant.count(":") + 1


def read_assay_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, usecols=["mutant", "mutated_sequence", "DMS_score"])
    table = table.dropna(subset=["mutant", "mutated_sequence", "DMS_score"]).copy()
    table["mutant"] = table["mutant"].astype(str)
    table["mutated_sequence"] = table["mutated_sequence"].astype(str)
    table["DMS_score"] = pd.to_numeric(table["DMS_score"], errors="coerce")
    table = table.dropna(subset=["DMS_score"]).drop_duplicates("mutant", keep="first")
    table["mutation_order"] = table["mutant"].map(mutation_order)
    return table


def read_zero_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, usecols=["mutant", "mutated_sequence", "MSA_Transformer_ensemble"])
    table = table.dropna(subset=["mutant", "mutated_sequence", "MSA_Transformer_ensemble"]).copy()
    table["mutant"] = table["mutant"].astype(str)
    table["mutated_sequence"] = table["mutated_sequence"].astype(str)
    table = table.drop_duplicates("mutant", keep="first")
    return table


def complete_low_split(low: pd.DataFrame, assay: str, seeds: list[int], budget: int) -> bool:
    block = low[
        low["assay"].eq(assay)
        & low["regime"].eq("single_only")
        & low["budget"].eq(budget)
        & low["seed"].isin(seeds)
    ]
    expected = {(seed, budget, "single_only") for seed in seeds}
    observed = {
        (int(seed), int(bgt), str(regime))
        for seed, bgt, regime in block[["seed", "budget", "regime"]].drop_duplicates().itertuples(index=False, name=None)
    }
    sizes = block.groupby(["seed", "budget", "regime"])["mutant"].nunique()
    return expected.issubset(observed) and bool((sizes == budget).all())


def complete_selected_split(selected: pd.DataFrame, assay: str, seeds: list[int], selected_double_budget: int) -> bool:
    budget = 20 + selected_double_budget
    regimes = [f"single20_plus_{policy}_double{selected_double_budget}" for policy in SELECTED_DOUBLE_POLICIES]
    block = selected[
        selected["assay"].eq(assay)
        & selected["budget"].eq(budget)
        & selected["seed"].isin(seeds)
        & selected["regime"].isin(regimes)
    ]
    expected = {(seed, budget, regime) for seed in seeds for regime in regimes}
    observed = {
        (int(seed), int(bgt), str(regime))
        for seed, bgt, regime in block[["seed", "budget", "regime"]].drop_duplicates().itertuples(index=False, name=None)
    }
    sizes = block.groupby(["seed", "budget", "regime"])["mutant"].nunique()
    return expected.issubset(observed) and bool((sizes == budget).all())


def seed_specific_holdout(holdout: pd.DataFrame, assay: str, seeds: list[int], per_seed: int) -> tuple[int, int, list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    required = []
    counts = []
    for seed in seeds:
        mutants = sorted(holdout.loc[holdout["assay"].eq(assay) & holdout["seed"].eq(seed), "mutant"].astype(str).tolist())
        chosen = mutants[: min(per_seed, len(mutants))]
        counts.append(len(chosen))
        rows.extend({"assay": assay, "seed": seed, "mutant": mutant, "role": "held_out_double"} for mutant in chosen)
        required.extend({"assay": assay, "mutant": mutant} for mutant in chosen)
    required_frame = pd.DataFrame(required).drop_duplicates(["assay", "mutant"]) if required else pd.DataFrame(columns=["assay", "mutant"])
    return (min(counts) if counts else 0), int(len(required_frame)), required_frame.to_dict("records"), rows


def kermut_success_lookup() -> set[str]:
    if not KERMUT_R3_ASSAY_MEANS.exists():
        return set()
    table = pd.read_csv(KERMUT_R3_ASSAY_MEANS)
    keep = table[
        table["method"].eq("Kermut")
        & table["budget"].eq(20)
        & table["order_bucket"].isin(["double", "higher"])
        & table["mean_n_variants"].gt(0)
        & table["mean_abs_spearman"].notna()
    ]
    return set(keep["assay"].astype(str))


def write_single_column(values: list[str], path: Path) -> None:
    pd.DataFrame({"assay": values}).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--multimutant-seeds", default="0,1,2,3,4")
    parser.add_argument("--selected-double-seeds", default="0,1,2,3,4")
    parser.add_argument("--multimutant-n-assays", type=int, default=10)
    parser.add_argument("--selected-double-n-assays", type=int, default=8)
    parser.add_argument("--multimutant-eval-per-bucket", type=int, default=120)
    parser.add_argument("--selected-double-holdout-per-seed", type=int, default=100)
    parser.add_argument("--max-msa-len", type=int, default=400)
    parser.add_argument("--min-double", type=int, default=20)
    parser.add_argument("--min-higher", type=int, default=0)
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    multimutant_seeds = parse_csv_ints(args.multimutant_seeds)
    selected_double_seeds = parse_csv_ints(args.selected_double_seeds)
    low = pd.read_csv(LOW_SPLITS)
    selected = pd.read_csv(SELECTED_SPLITS)
    holdout = pd.read_csv(SELECTED_HOLDOUT)
    reference = pd.read_csv(PNPT_REFERENCE)
    kermut_success = kermut_success_lookup()

    rows = []
    candidate_cache = {}
    holdout_cache = {}
    for _, ref in reference.iterrows():
        assay = str(ref["DMS_filename"])
        assay_id = str(ref["DMS_id"])
        dms_path = DMS_DIR / assay
        zero_path = ZERO_DIR / assay
        msa_path = MSA_FILES / str(ref["MSA_filename"])
        weight_path = MSA_WEIGHTS / str(ref["weight_file_name"])
        row = {
            "assay": assay,
            "assay_id": assay_id,
            "msa_len": int(ref["MSA_len"]),
            "resource_files_exist": all(path.exists() for path in [dms_path, zero_path, msa_path, weight_path]),
            "multimutant_split_complete": complete_low_split(low, assay, multimutant_seeds, 20),
            "selected_double_split_complete": complete_selected_split(selected, assay, selected_double_seeds, 20),
            "previous_r3_assay": assay in PREVIOUS_R3,
            "previous_selected_double_assay": assay in PREVIOUS_R4,
            "kermut_r3_evaluable": assay in kermut_success,
            "n_total": 0,
            "n_single": 0,
            "n_double": 0,
            "n_higher": 0,
            "n_r4_holdout_min_per_seed": 0,
            "n_selected_double_required_holdout_union": 0,
            "eligible_r3": False,
            "eligible_r4": False,
            "reason": "",
        }
        try:
            if row["msa_len"] > args.max_msa_len:
                raise ValueError("MSA length above configured limit")
            if not row["resource_files_exist"]:
                raise ValueError("required ProteinNPT resource file missing")
            assay_table = read_assay_table(dms_path)
            zero_table = read_zero_table(zero_path)
            merged = assay_table.merge(
                zero_table[["mutant", "mutated_sequence"]],
                on=["mutant", "mutated_sequence"],
                how="inner",
                validate="one_to_one",
            )
            if len(merged) != len(assay_table):
                raise ValueError("project candidates do not exactly match ProteinNPT auxiliary table")
            counts = merged["mutation_order"].value_counts()
            row["n_total"] = int(len(merged))
            row["n_single"] = int(counts.get(1, 0))
            row["n_double"] = int(counts.get(2, 0))
            row["n_higher"] = int((merged["mutation_order"] >= 3).sum())
            min_holdout, union_holdout, required_rows, seeded_rows = seed_specific_holdout(
                holdout,
                assay,
                selected_double_seeds,
                args.selected_double_holdout_per_seed,
            )
            row["n_r4_holdout_min_per_seed"] = int(min_holdout)
            row["n_selected_double_required_holdout_union"] = int(union_holdout)
            row["eligible_r3"] = bool(
                row["multimutant_split_complete"]
                and row["n_double"] >= args.min_double
                and row["n_higher"] >= args.min_higher
            )
            row["eligible_r4"] = bool(
                row["selected_double_split_complete"]
                and row["n_r4_holdout_min_per_seed"] >= args.selected_double_holdout_per_seed
            )
            row["reason"] = "ok"
            candidate_cache[assay] = merged
            holdout_cache[assay] = (required_rows, seeded_rows)
        except Exception as exc:
            row["reason"] = str(exc)
        rows.append(row)

    feasibility = pd.DataFrame(rows)
    feasibility.to_csv(out / "sota_expanded_feasibility_table.csv", index=False)

    multimutant_pool = feasibility[feasibility["eligible_r3"]].copy()
    multimutant_pool["rank_group"] = 2
    multimutant_pool.loc[multimutant_pool["previous_r3_assay"], "rank_group"] = 0
    multimutant_pool.loc[multimutant_pool["kermut_r3_evaluable"] & ~multimutant_pool["previous_r3_assay"], "rank_group"] = 1
    multimutant_pool = multimutant_pool.sort_values(
        ["rank_group", "msa_len", "n_higher", "n_double", "assay"],
        ascending=[True, True, False, False, True],
    )
    multimutant_assays = multimutant_pool["assay"].head(args.multimutant_n_assays).tolist()
    multimutant_required = []
    for assay in multimutant_assays:
        merged = candidate_cache[assay]
        for bucket, predicate in [
            ("double", merged["mutation_order"].eq(2)),
            ("higher", merged["mutation_order"].ge(3)),
        ]:
            mutants = sorted(merged.loc[predicate, "mutant"].astype(str).tolist())[: args.multimutant_eval_per_bucket]
            multimutant_required.extend({"assay": assay, "mutant": mutant, "evaluation_bucket": bucket} for mutant in mutants)

    selected_double_pool = feasibility[feasibility["eligible_r4"]].copy()
    selected_double_pool["rank_group"] = 1
    selected_double_pool.loc[selected_double_pool["previous_selected_double_assay"], "rank_group"] = 0
    selected_double_pool = selected_double_pool.sort_values(
        ["rank_group", "msa_len", "n_selected_double_required_holdout_union", "n_double", "assay"],
        ascending=[True, True, False, False, True],
    )
    selected_double_assays = selected_double_pool["assay"].head(args.selected_double_n_assays).tolist()
    selected_double_required = []
    selected_double_holdout_rows = []
    for assay in selected_double_assays:
        required_rows, seeded_rows = holdout_cache[assay]
        selected_double_required.extend(required_rows)
        selected_double_holdout_rows.extend(seeded_rows)

    selected_double_required_df = pd.DataFrame(selected_double_required).drop_duplicates(["assay", "mutant"]) if selected_double_required else pd.DataFrame(columns=["assay", "mutant"])
    selected_double_holdout_df = pd.DataFrame(selected_double_holdout_rows) if selected_double_holdout_rows else pd.DataFrame(columns=["assay", "seed", "mutant", "role"])
    write_single_column(multimutant_assays, out / "proteinnpt_multimutant_assays_expanded.csv")
    pd.DataFrame(multimutant_required).to_csv(out / "proteinnpt_multimutant_evaluation_candidates_expanded.csv", index=False)
    write_single_column(selected_double_assays, out / "selected_double_assays_expanded.csv")
    selected_double_required_df.to_csv(out / "selected_double_evaluation_candidates_expanded.csv", index=False)
    selected_double_holdout_df.to_csv(out / "selected_double_evaluation_holdout_by_seed_expanded.csv", index=False)

    metadata = {
        "status": "ok",
        "multimutant_seeds": multimutant_seeds,
        "selected_double_seeds": selected_double_seeds,
        "max_msa_len": args.max_msa_len,
        "multimutant_selection": {
            "n_requested": args.multimutant_n_assays,
            "n_selected": len(multimutant_assays),
            "minimum_double_mutants": args.min_double,
            "minimum_higher_order_variants": args.min_higher,
            "evaluation_candidates_per_bucket_per_assay": args.multimutant_eval_per_bucket,
            "priority": "previous successful ProteinNPT assays, then assays evaluable by Kermut, then shorter MSA length and more higher-order candidates",
            "assays": multimutant_assays,
        },
        "selected_double_selection": {
            "n_requested": args.selected_double_n_assays,
            "n_selected": len(selected_double_assays),
            "holdout_double_mutants_per_seed": args.selected_double_holdout_per_seed,
            "seeds": selected_double_seeds,
            "policies": SELECTED_DOUBLE_POLICIES,
            "priority": "previous successful ProteinNPT selected-double assays, then shorter MSA length and larger seed-specific held-out double-mutant pools",
            "assays": selected_double_assays,
        },
        "feasibility_file": str((out / "sota_expanded_feasibility_table.csv").resolve()),
    }
    (out / "sota_expanded_sample_set_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
