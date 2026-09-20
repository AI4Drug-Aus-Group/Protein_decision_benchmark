from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PNPT = ROOT / "external_repos" / "ProteinNPT"
PNPT_DATA = PNPT / "ProteinNPT_data"
DEFAULT_SPLITS = ROOT / "results" / "splits" / "low_n_splits.csv"
DEFAULT_DMS_DIR = ROOT / "proteingym" / "extracted" / "DMS_ProteinGym_substitutions"
DEFAULT_REFERENCE = PNPT / "proteinnpt" / "utils" / "proteingym" / "DMS_substitutions.csv"
DEFAULT_ZERO_SHOT = PNPT_DATA / "data" / "zero_shot_fitness_predictions" / "substitutions"
DEFAULT_MSA_FILES = PNPT_DATA / "data" / "MSA" / "MSA_files"
DEFAULT_MSA_WEIGHTS = PNPT_DATA / "data" / "MSA" / "MSA_weights"
DEFAULT_HHFILTER = PNPT_DATA / "utils" / "hhfilter"
DEFAULT_MODEL = PNPT_DATA / "ESM" / "MSA_Transformer" / "esm_msa1b_t12_100M_UR50S.pt"
DEFAULT_MODEL_CONFIG = PNPT / "proteinnpt" / "proteinnpt" / "model_configs" / "PNPT_final.json"
DEFAULT_TARGET_CONFIG = PNPT / "proteinnpt" / "utils" / "target_configs" / "fitness.json"
DEFAULT_OUT = ROOT / "results" / "local_matched_reproduction" / "proteinnpt"


def parse_set(value: str, cast):
    return {cast(item) for item in value.split(",") if item}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_table_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    text = frame[columns].astype(str).to_csv(index=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataframe_file_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def atomic_write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_or_validate_table(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    expected_hash = dataframe_file_hash(frame)
    if path.exists() and not overwrite:
        if sha256(path) != expected_hash:
            raise ValueError(f"existing prepared file differs from current inputs; rerun with --overwrite: {path}")
        return
    atomic_write_table(frame, path)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def read_csv_checked(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, usecols=usecols)


def sanitize_assay_id(assay: str) -> str:
    return assay.removesuffix(".csv")


def clean_project_assay(path: Path) -> pd.DataFrame:
    table = read_csv_checked(path, usecols=["mutant", "mutated_sequence", "DMS_score"])
    table = table.dropna(subset=["mutant", "mutated_sequence", "DMS_score"]).copy()
    table["mutant"] = table["mutant"].astype(str)
    table["mutated_sequence"] = table["mutated_sequence"].astype(str)
    table["DMS_score"] = pd.to_numeric(table["DMS_score"], errors="coerce")
    table = table.dropna(subset=["DMS_score"]).copy()
    duplicated = table[table.duplicated("mutant", keep=False)]
    if not duplicated.empty:
        grouped = duplicated.groupby("mutant", sort=False)
        bad = []
        for mutant, block in grouped:
            if block[["mutated_sequence", "DMS_score"]].drop_duplicates().shape[0] > 1:
                bad.append(mutant)
        if bad:
            raise ValueError(f"conflicting duplicate mutants: {len(bad)}")
        table = table.drop_duplicates("mutant", keep="first")
    return table.sort_values("mutant").reset_index(drop=True)


def clean_zero_shot(path: Path) -> pd.DataFrame:
    zero = read_csv_checked(path)
    needed = ["mutant", "mutated_sequence", "MSA_Transformer_ensemble"]
    missing = [column for column in needed if column not in zero.columns]
    if missing:
        raise ValueError(f"zero-shot file missing columns: {missing}")
    zero = zero[needed].dropna(subset=needed).copy()
    zero["mutant"] = zero["mutant"].astype(str)
    zero["mutated_sequence"] = zero["mutated_sequence"].astype(str)
    zero["MSA_Transformer_ensemble"] = pd.to_numeric(zero["MSA_Transformer_ensemble"], errors="coerce")
    zero = zero.dropna(subset=["MSA_Transformer_ensemble"]).copy()
    duplicated = zero[zero.duplicated("mutant", keep=False)]
    if not duplicated.empty:
        grouped = duplicated.groupby("mutant", sort=False)
        bad = []
        for mutant, block in grouped:
            if block[["mutated_sequence", "MSA_Transformer_ensemble"]].drop_duplicates().shape[0] > 1:
                bad.append(mutant)
        if bad:
            raise ValueError(f"conflicting duplicate zero-shot mutants: {len(bad)}")
        zero = zero.drop_duplicates("mutant", keep="first")
    return zero.sort_values("mutant").reset_index(drop=True)


def exact_merge_assay_and_zero(assay: pd.DataFrame, zero: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = assay.merge(
        zero,
        on=["mutant", "mutated_sequence"],
        how="left",
        validate="one_to_one",
    )
    missing = merged["MSA_Transformer_ensemble"].isna()
    if missing.any():
        missing_mutants = merged.loc[missing, "mutant"].head(5).tolist()
        raise ValueError(f"official ProteinNPT zero-shot missing project mutants: {int(missing.sum())}; examples={missing_mutants}")
    assay_out = merged[["mutant", "mutated_sequence", "DMS_score"]].copy()
    zero_out = merged[["mutant", "mutated_sequence", "MSA_Transformer_ensemble"]].copy()
    return assay_out, zero_out


def infer_required_splits(splits: pd.DataFrame, assay_file: str, regimes: set[str], budgets: set[int], seeds: set[int]) -> pd.DataFrame:
    block = splits[
        splits["assay"].eq(assay_file)
        & splits["regime"].isin(regimes)
        & splits["budget"].isin(budgets)
        & splits["seed"].isin(seeds)
    ].copy()
    return block


def deterministic_candidate_subset(
    assay: pd.DataFrame,
    zero: pd.DataFrame,
    split_block: pd.DataFrame,
    assay_id: str,
    max_candidates: int,
    sampling_seed: int,
    extra_required_mutants: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required_mutants = set(split_block["mutant"].astype(str))
    if extra_required_mutants:
        required_mutants |= set(extra_required_mutants)
    full_count = len(assay)
    if max_candidates <= 0 or full_count <= max_candidates:
        selected_mutants = set(assay["mutant"].astype(str))
        strategy = "complete_candidate_pool"
    else:
        if len(required_mutants) > max_candidates:
            raise ValueError(
                f"candidate cap is smaller than the union of required training variants: "
                f"{max_candidates}<{len(required_mutants)}"
            )
        remaining = assay.loc[~assay["mutant"].isin(required_mutants), "mutant"].astype(str).to_numpy()
        assay_seed = int.from_bytes(
            hashlib.sha256(f"{sampling_seed}:{assay_id}".encode("utf-8")).digest()[:8],
            byteorder="little",
            signed=False,
        )
        rng = np.random.default_rng(assay_seed)
        sampled = rng.choice(remaining, size=max_candidates - len(required_mutants), replace=False)
        selected_mutants = required_mutants | set(sampled.tolist())
        strategy = "all_required_training_variants_plus_uniform_unlabelled_candidates"
    assay_out = assay[assay["mutant"].isin(selected_mutants)].sort_values("mutant").reset_index(drop=True)
    zero_out = zero[zero["mutant"].isin(selected_mutants)].sort_values("mutant").reset_index(drop=True)
    if len(assay_out) != len(zero_out):
        raise ValueError("sampled assay and auxiliary-score tables have different sizes")
    return assay_out, zero_out, {
        "candidate_sampling_strategy": strategy,
        "candidate_sampling_seed": sampling_seed,
        "max_candidates_per_assay": max_candidates,
        "n_candidates_full": full_count,
        "n_candidates_sampled": len(assay_out),
        "n_required_training_candidates": len(required_mutants),
        "sampled_candidate_identity_sha256": stable_table_hash(assay_out, ["mutant", "mutated_sequence"]),
    }


def write_reference(reference_rows: list[dict], output_path: Path) -> None:
    columns = [
        "DMS_id",
        "DMS_filename",
        "UniProt_ID",
        "target_seq",
        "seq_len",
        "MSA_filename",
        "MSA_start",
        "MSA_end",
        "MSA_len",
        "weight_file_name",
    ]
    atomic_write_table(pd.DataFrame(reference_rows, columns=columns), output_path)


def prepare(args: argparse.Namespace) -> dict:
    output = args.out.resolve()
    runtime = output / "runtime"
    shared_assays = runtime / "assay_data" / "substitutions"
    run_inputs = runtime / "run_inputs"
    zero_out = runtime / "zero_shot_fitness_predictions" / "substitutions"
    embeddings = runtime / "embeddings" / "MSA_Transformer"
    logs = runtime / "logs"
    results = runtime / "results"
    for folder in [shared_assays, run_inputs, zero_out, embeddings, logs, results]:
        folder.mkdir(parents=True, exist_ok=True)
    for folder in [
        args.msa_files / "preprocessed",
        args.msa_files / "hhfiltered",
        args.msa_weights / "hhfiltered",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    splits = read_csv_checked(args.splits)
    required_candidates = None
    if args.required_candidates:
        required_candidates = read_csv_checked(args.required_candidates)
        missing = {"assay", "mutant"} - set(required_candidates.columns)
        if missing:
            raise ValueError(f"required-candidates table missing columns: {sorted(missing)}")
        required_candidates["assay"] = required_candidates["assay"].astype(str)
        required_candidates["mutant"] = required_candidates["mutant"].astype(str)
    reference = read_csv_checked(args.reference)
    regimes = parse_set(args.regimes, str)
    budgets = parse_set(args.budgets, int)
    seeds = parse_set(args.seeds, int)
    expected_runs = {(seed, budget, regime) for seed in seeds for budget in budgets for regime in regimes}
    model_config_hash = sha256(args.model_config)
    target_config_hash = sha256(DEFAULT_TARGET_CONFIG)
    embedding_model_hash = sha256(args.embedding_model)
    assays = sorted(splits.loc[splits["regime"].isin(regimes) & splits["budget"].isin(budgets) & splits["seed"].isin(seeds), "assay"].unique())
    if args.assays:
        selected = {value if value.endswith(".csv") else f"{value}.csv" for value in parse_set(args.assays, str)}
        assays = [assay for assay in assays if assay in selected]

    manifest_rows = []
    run_rows = []
    reference_rows = []
    for assay_file in assays:
        assay_id = sanitize_assay_id(assay_file)
        record = {
            "assay": assay_file,
            "assay_id": assay_id,
            "eligible": False,
            "reason": "",
        }
        try:
            ref = reference.loc[reference["DMS_id"].eq(assay_id)]
            if ref.empty:
                raise ValueError("assay absent from ProteinNPT official reference")
            ref_row = ref.iloc[0].to_dict()
            dms_path = args.dms_dir / assay_file
            zero_path = args.zero_shot / assay_file
            msa_path = args.msa_files / str(ref_row["MSA_filename"])
            weight_path = args.msa_weights / str(ref_row["weight_file_name"])
            for label, path in [
                ("project_dms", dms_path),
                ("official_zero_shot", zero_path),
                ("msa_file", msa_path),
                ("msa_weight", weight_path),
                ("msa_transformer_checkpoint", args.embedding_model),
            ]:
                if not path.exists():
                    raise FileNotFoundError(f"{label} missing: {path}")
            if not (args.hhfilter / "bin" / "hhfilter").exists():
                raise FileNotFoundError(f"hhfilter binary missing: {args.hhfilter / 'bin' / 'hhfilter'}")
            msa_len = int(ref_row["MSA_len"])
            if msa_len > args.max_positions:
                raise ValueError(f"MSA length exceeds max_positions: {msa_len}>{args.max_positions}")

            assay_table = clean_project_assay(dms_path)
            zero_table = clean_zero_shot(zero_path)
            assay_table, zero_table = exact_merge_assay_and_zero(assay_table, zero_table)
            split_block = infer_required_splits(splits, assay_file, regimes, budgets, seeds)
            observed_runs = {
                (int(seed), int(budget), str(regime))
                for seed, budget, regime in split_block[["seed", "budget", "regime"]].drop_duplicates().itertuples(index=False, name=None)
            }
            missing_runs = sorted(expected_runs - observed_runs)
            if missing_runs:
                raise ValueError(f"missing split groups: {len(missing_runs)}")
            assay_mutants = set(assay_table["mutant"].astype(str))
            missing_split_mutants = sorted(set(split_block["mutant"].astype(str)) - assay_mutants)
            if missing_split_mutants:
                raise ValueError(f"split mutants absent from assay table: {len(missing_split_mutants)}")
            extra_required = None
            if required_candidates is not None:
                extra_required = set(required_candidates.loc[required_candidates["assay"].eq(assay_file), "mutant"])
            assay_table, zero_table, sampling_record = deterministic_candidate_subset(
                assay_table,
                zero_table,
                split_block,
                assay_id,
                args.max_candidates_per_assay,
                args.candidate_sampling_seed,
                extra_required,
            )

            shared_assay_path = shared_assays / assay_file
            zero_file_path = zero_out / assay_file
            ref_index = len(reference_rows)
            reference_entry = {
                "DMS_id": assay_id,
                "DMS_filename": assay_file,
                "UniProt_ID": ref_row.get("UniProt_ID", assay_id),
                "target_seq": ref_row["target_seq"],
                "seq_len": int(ref_row.get("seq_len", len(str(ref_row["target_seq"])))),
                "MSA_filename": ref_row["MSA_filename"],
                "MSA_start": int(ref_row["MSA_start"]),
                "MSA_end": int(ref_row["MSA_end"]),
                "MSA_len": msa_len,
                "weight_file_name": ref_row["weight_file_name"],
            }

            candidate_hash = stable_table_hash(assay_table, ["mutant", "mutated_sequence", "DMS_score"])
            embedding_input_hash = stable_table_hash(assay_table, ["mutant", "mutated_sequence"])
            zero_hash = stable_table_hash(zero_table, ["mutant", "mutated_sequence", "MSA_Transformer_ensemble"])
            local_runs = []
            for (seed, budget, regime), block in split_block.groupby(["seed", "budget", "regime"], sort=True):
                seed = int(seed)
                budget = int(budget)
                train_mutants = block.sort_values("rank")["mutant"].astype(str).tolist()
                if len(train_mutants) != budget or len(set(train_mutants)) != budget:
                    raise ValueError(f"bad training split size for seed={seed}, budget={budget}, regime={regime}")
                run_dir = run_inputs / assay_id / f"seed{seed}_budget{budget}_{regime}"
                run_assay_path = run_dir / assay_file
                run_table = assay_table.copy()
                run_table["matched_split"] = np.where(run_table["mutant"].isin(train_mutants), 0, 1)
                if int((run_table["matched_split"] == 0).sum()) != budget:
                    raise ValueError(f"training split did not map exactly for seed={seed}, budget={budget}, regime={regime}")
                run_table_hash = stable_table_hash(
                    run_table,
                    ["mutant", "mutated_sequence", "DMS_score", "matched_split"],
                )
                fingerprint_payload = "|".join(
                    [
                        run_table_hash,
                        zero_hash,
                        model_config_hash,
                        target_config_hash,
                        embedding_model_hash,
                    ]
                )
                run_fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
                run_id = f"{assay_id}__seed{seed}__budget{budget}__{regime}"
                suffix = f"pnpt_{assay_id[:16]}_s{seed}_b{budget}_{run_fingerprint[:10]}"
                local_runs.append(
                    (
                        run_assay_path,
                        run_table,
                        {
                            "run_id": run_id,
                            "assay": assay_file,
                            "assay_id": assay_id,
                            "seed": seed,
                            "budget": budget,
                            "regime": regime,
                            "n_train": budget,
                            "n_test": int((run_table["matched_split"] == 1).sum()),
                            "n_candidates": int(len(run_table)),
                            **sampling_record,
                            "model_name_suffix": suffix,
                            "reference_index": ref_index,
                            "assay_data_location": str(run_assay_path.resolve()),
                            "shared_assay_location": str(shared_assay_path.resolve()),
                            "percentile_reference_location": str(dms_path.resolve()),
                            "zero_shot_fitness_predictions_location": str(zero_out.resolve()),
                            "sequence_embeddings_folder": str(embeddings.resolve()),
                            "embedding_file": str((embeddings / f"{assay_id}.h5").resolve()),
                            "target_seq": str(ref_row["target_seq"]),
                            "MSA_location": str(msa_path.resolve()),
                            "MSA_sequence_weights_filename": str(ref_row["weight_file_name"]),
                            "MSA_start": int(ref_row["MSA_start"]),
                            "MSA_end": int(ref_row["MSA_end"]),
                            "MSA_len": msa_len,
                            "candidate_table_sha256": candidate_hash,
                            "embedding_input_sha256": embedding_input_hash,
                            "embedding_model_sha256": embedding_model_hash,
                            "zero_auxiliary_table_sha256": zero_hash,
                            "run_table_sha256": run_table_hash,
                            "run_fingerprint_sha256": run_fingerprint,
                        },
                    )
                )

            if len(local_runs) != len(expected_runs):
                raise ValueError(f"unexpected number of fully validated runs: {len(local_runs)}")
            write_or_validate_table(assay_table, shared_assay_path, args.overwrite)
            write_or_validate_table(zero_table, zero_file_path, args.overwrite)
            for run_assay_path, run_table, _ in local_runs:
                write_or_validate_table(run_table, run_assay_path, args.overwrite)

            reference_rows.append(reference_entry)
            run_rows.extend(run_record for _, _, run_record in local_runs)
            record.update({
                "eligible": True,
                "reason": "ok",
                "n_candidates": int(len(assay_table)),
                **sampling_record,
                "n_runs": len(expected_runs),
                "MSA_len": msa_len,
                "MSA_filename": ref_row["MSA_filename"],
                "weight_file_name": ref_row["weight_file_name"],
                "project_dms_sha256": sha256(dms_path),
                "official_zero_shot_sha256": sha256(zero_path),
                "candidate_table_sha256": candidate_hash,
                "embedding_input_sha256": embedding_input_hash,
                "embedding_model_sha256": embedding_model_hash,
                "zero_auxiliary_table_sha256": zero_hash,
                "reference_index": ref_index,
            })
        except Exception as exc:
            record["reason"] = str(exc)
        manifest_rows.append(record)

    assay_manifest = pd.DataFrame(manifest_rows)
    run_manifest = pd.DataFrame(run_rows)
    reference_path = runtime / "protein_npt_matched_reference.csv"
    write_reference(reference_rows, reference_path)
    atomic_write_table(assay_manifest, output / "proteinnpt_assay_resource_manifest.csv")
    atomic_write_table(run_manifest, output / "proteinnpt_run_manifest.csv")
    metadata = {
        "status": "prepared_without_running_proteinnpt",
        "protocol": "ProteinNPT official model under project matched low-measurement train-test splits.",
        "regimes": sorted(regimes),
        "budgets": sorted(budgets),
        "seeds": sorted(seeds),
        "candidate_sampling": {
            "max_candidates_per_assay": args.max_candidates_per_assay,
            "seed": args.candidate_sampling_seed,
            "policy": "Retain the union of all requested training variants, then uniformly sample without replacement from the remaining candidates using an assay-specific seed derived from the recorded global seed. Fitness values are not used for candidate selection.",
        },
        "n_assays_requested": len(assays),
        "n_assays_eligible": int(assay_manifest["eligible"].sum()) if not assay_manifest.empty else 0,
        "n_runs_eligible": int(len(run_manifest)),
        "reason_for_exclusion_policy": "Assays without official ProteinNPT MSA-Transformer auxiliary scores, matching MSA resources, matching project candidate sequences, or required split groups are excluded rather than filled from other project tables.",
        "paths": {
            "runtime": str(runtime.resolve()),
            "reference": str(reference_path.resolve()),
            "shared_assays": str(shared_assays.resolve()),
            "run_inputs": str(run_inputs.resolve()),
            "zero_shot_fitness_predictions": str(zero_out.resolve()),
            "embeddings": str(embeddings.resolve()),
            "logs": str(logs.resolve()),
            "results": str(results.resolve()),
        },
        "official_paths": {
            "protein_npt_repo": str(PNPT.resolve()),
            "model_config": str(args.model_config.resolve()),
            "model_config_sha256": model_config_hash,
            "target_config": str(DEFAULT_TARGET_CONFIG.resolve()),
            "target_config_sha256": target_config_hash,
            "embedding_model": str(args.embedding_model.resolve()),
            "embedding_model_sha256": embedding_model_hash,
            "msa_files": str(args.msa_files.resolve()),
            "msa_weights": str(args.msa_weights.resolve()),
            "hhfilter": str(args.hhfilter.resolve()),
        },
    }
    atomic_write_json(metadata, output / "proteinnpt_prepare_metadata.json")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ProteinNPT inputs for matched low-measurement splits.")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--dms-dir", type=Path, default=DEFAULT_DMS_DIR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--zero-shot", type=Path, default=DEFAULT_ZERO_SHOT)
    parser.add_argument("--msa-files", type=Path, default=DEFAULT_MSA_FILES)
    parser.add_argument("--msa-weights", type=Path, default=DEFAULT_MSA_WEIGHTS)
    parser.add_argument("--hhfilter", type=Path, default=DEFAULT_HHFILTER)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--regimes", default="mixed_random")
    parser.add_argument("--budgets", default="20,50,100")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--assays", default="")
    parser.add_argument("--required-candidates", type=Path, default=None)
    parser.add_argument("--max-positions", type=int, default=1024)
    parser.add_argument("--max-candidates-per-assay", type=int, default=0)
    parser.add_argument("--candidate-sampling-seed", type=int, default=20260826)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = prepare(args)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
