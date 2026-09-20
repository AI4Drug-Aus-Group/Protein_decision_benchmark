from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
PNPT = ROOT / "external_repos" / "ProteinNPT"
DEFAULT_ENV_PYTHON = ROOT / "external_repos" / "envs" / "proteinnpt" / "bin" / "python3.10"
DEFAULT_OUT = ROOT / "results" / "local_matched_reproduction" / "proteinnpt"
DEFAULT_MANIFEST = DEFAULT_OUT / "proteinnpt_run_manifest.csv"
DEFAULT_ASSAY_MANIFEST = DEFAULT_OUT / "proteinnpt_assay_resource_manifest.csv"
DEFAULT_REFERENCE = DEFAULT_OUT / "runtime" / "protein_npt_matched_reference.csv"
MODEL_CONFIG = PNPT / "proteinnpt" / "proteinnpt" / "model_configs" / "PNPT_final.json"
TARGET_CONFIG = PNPT / "proteinnpt" / "utils" / "target_configs" / "fitness.json"
TRAIN_SCRIPT = PNPT / "scripts" / "train.py"
EMBEDDING_SCRIPT = PNPT / "scripts" / "embeddings.py"
SEEDED_EMBEDDING_SCRIPT = Path(__file__).resolve().with_name("run_seeded_proteinnpt_embeddings.py")
MSA_FILES = PNPT / "ProteinNPT_data" / "data" / "MSA" / "MSA_files"
MSA_WEIGHTS = PNPT / "ProteinNPT_data" / "data" / "MSA" / "MSA_weights"
HHFILTER = PNPT / "ProteinNPT_data" / "utils" / "hhfilter"
EMBEDDING_MODEL = PNPT / "ProteinNPT_data" / "ESM" / "MSA_Transformer" / "esm_msa1b_t12_100M_UR50S.pt"
CLUSTALOMEGA_ROOT = ROOT / "external_repos" / "tools" / "clustalo"
CLUSTALOMEGA = CLUSTALOMEGA_ROOT / "bin" / "clustalo"
KEYS = ["assay", "seed", "budget", "regime"]
MAX_MSA_SEQUENCES = 384


def runtime_root(args: argparse.Namespace) -> Path:
    return args.out / "runtime"


def parse_shard(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    left, right = value.split("/")
    index = int(left)
    count = int(right)
    if index < 0 or count <= 0 or index >= count:
        raise ValueError("shard must be formatted as index/count with 0 <= index < count")
    return index, count


def shard_frame(frame: pd.DataFrame, shard: tuple[int, int] | None, key_columns: list[str]) -> pd.DataFrame:
    if shard is None:
        return frame
    index, count = shard
    keys = frame[key_columns].drop_duplicates().reset_index(drop=True)
    keys["_shard_keep"] = [i % count == index for i in range(len(keys))]
    keep = keys[keys["_shard_keep"]].drop(columns="_shard_keep")
    return frame.merge(keep, on=key_columns, how="inner")


def run_command(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {"status": "dry_run", "returncode": None, "command": command, "log": str(log_path)}
    started = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PNPT) + os.pathsep + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = str(CLUSTALOMEGA_ROOT / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    with log_path.open("w") as log:
        process = subprocess.run(command, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return {
        "status": "ok" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "elapsed_seconds": time.time() - started,
        "command": command,
        "log": str(log_path),
    }


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def mutant_list_hash(mutants: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(mutants).encode("utf-8")).hexdigest()


def embedding_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".metadata.json")


@contextmanager
def msa_resource_lock(msa_filename: str):
    lock_dir = MSA_FILES / ".proteinnpt_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{Path(msa_filename).stem}.lock"
    with lock_path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def filtered_msa_configuration(msa_filename: str, max_msa_sequences: int, prepare: bool = True, weight_file_name: str | None = None) -> dict:
    raw_path = MSA_FILES / msa_filename
    filtered_path = MSA_FILES / "hhfiltered" / f"{raw_path.stem}_hhfiltered_cov_75_maxid_90_minid_0.a2m"
    processed_msa_depth = None
    if prepare and weight_file_name is not None:
        if str(PNPT) not in sys.path:
            sys.path.insert(0, str(PNPT))
        from proteinnpt.utils.msa_utils import process_MSA

        msa_sequences, _ = process_MSA(
            MSA_data_folder=str(MSA_FILES),
            MSA_weight_data_folder=str(MSA_WEIGHTS),
            MSA_filename=msa_filename,
            MSA_weights_filename=weight_file_name,
            path_to_hhfilter=str(HHFILTER),
        )
        processed_msa_depth = len(msa_sequences)
    elif not filtered_path.exists() and prepare:
        if str(PNPT) not in sys.path:
            sys.path.insert(0, str(PNPT))
        from proteinnpt.utils.msa_utils import filter_msa

        filtered_path = Path(filter_msa(filename=str(raw_path), path_to_hhfilter=str(HHFILTER)))
    if not filtered_path.exists():
        return {
            "filtered_msa_file": str(filtered_path),
            "filtered_msa_depth": None,
            "filtered_fasta_depth": None,
            "num_msa_sequences": max_msa_sequences,
        }
    with filtered_path.open() as handle:
        filtered_fasta_depth = sum(line.startswith(">") for line in handle)
    effective_depth = processed_msa_depth if processed_msa_depth is not None else filtered_fasta_depth
    if effective_depth < 2:
        raise ValueError(f"Filtered MSA contains fewer than two usable sequences: {filtered_path}")
    return {
        "filtered_msa_file": str(filtered_path),
        "filtered_msa_depth": effective_depth,
        "filtered_fasta_depth": filtered_fasta_depth,
        "num_msa_sequences": min(max_msa_sequences, effective_depth),
    }


def current_collected_result(
    json_path: Path,
    default_npz_path: Path,
    expected_fingerprint: str,
) -> dict | None:
    if not json_path.exists():
        return None
    try:
        with json_path.open() as handle:
            record = json.load(handle)
    except Exception:
        return None
    npz_value = record.get("npz_file")
    npz_path = Path(npz_value) if npz_value else default_npz_path
    if (
        record.get("status") == "ok"
        and npz_path.exists()
        and record.get("run_fingerprint_sha256") == expected_fingerprint
    ):
        return record
    return None


def validate_h5(
    path: Path,
    expected_mutants: list[str],
    expected_seed: int | None = None,
    expected_input_hash: str | None = None,
    expected_model_hash: str | None = None,
    expected_fast_msa_mode: bool | None = None,
    expected_num_msa_sequences: int | None = None,
) -> dict:
    if not path.exists():
        return {"valid": False, "reason": "missing_embedding_file"}
    try:
        with h5py.File(path, "r") as handle:
            for dataset in ["embeddings", "mutants", "sequences", "pseudo_likelihoods"]:
                if dataset not in handle:
                    return {"valid": False, "reason": f"missing_dataset_{dataset}"}
            mutants = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in handle["mutants"][:]
            ]
            shape = tuple(handle["embeddings"].shape)
    except Exception as exc:
        return {"valid": False, "reason": str(exc)}
    if mutants != expected_mutants:
        return {"valid": False, "reason": "mutant_order_or_identity_mismatch", "n_mutants": len(mutants)}
    if expected_seed is not None:
        metadata_path = embedding_metadata_path(path)
        if not metadata_path.exists():
            return {"valid": False, "reason": "missing_embedding_metadata", "n_mutants": len(mutants)}
        try:
            with metadata_path.open() as handle:
                metadata = json.load(handle)
        except Exception as exc:
            return {"valid": False, "reason": f"invalid_embedding_metadata:{exc}", "n_mutants": len(mutants)}
        if int(metadata.get("embedding_seed", -1)) != expected_seed:
            return {"valid": False, "reason": "embedding_seed_mismatch", "n_mutants": len(mutants)}
        if metadata.get("mutant_order_sha256") != mutant_list_hash(expected_mutants):
            return {"valid": False, "reason": "embedding_metadata_mutant_hash_mismatch", "n_mutants": len(mutants)}
        if expected_input_hash is not None and metadata.get("embedding_input_sha256") != expected_input_hash:
            return {"valid": False, "reason": "embedding_input_hash_mismatch", "n_mutants": len(mutants)}
        if expected_model_hash is not None and metadata.get("embedding_model_sha256") != expected_model_hash:
            return {"valid": False, "reason": "embedding_model_hash_mismatch", "n_mutants": len(mutants)}
        if expected_fast_msa_mode is not None and bool(metadata.get("fast_msa_mode")) != expected_fast_msa_mode:
            return {"valid": False, "reason": "embedding_fast_msa_mode_mismatch", "n_mutants": len(mutants)}
        if expected_num_msa_sequences is not None and int(metadata.get("num_msa_sequences", -1)) != expected_num_msa_sequences:
            return {"valid": False, "reason": "embedding_num_msa_sequences_mismatch", "n_mutants": len(mutants)}
    return {"valid": True, "reason": "ok", "n_mutants": len(mutants), "embedding_shape": shape}


def assay_table_for_embedding(row: pd.Series) -> pd.DataFrame:
    return pd.read_csv(row["shared_assay_location"], usecols=["mutant", "mutated_sequence"])


def embed_command(args: argparse.Namespace, assay_row: pd.Series, num_msa_sequences: int) -> list[str]:
    data_location = runtime_root(args)
    command = [
        str(args.python),
        str(SEEDED_EMBEDDING_SCRIPT),
        "--wrapper-seed",
        str(args.embedding_seed),
        "--official-script",
        str(EMBEDDING_SCRIPT),
        "--assay_reference_file_location",
        str(args.reference),
        "--assay_index",
        str(int(assay_row["reference_index"])),
        "--input_data_location",
        str(data_location / "assay_data" / "substitutions"),
        "--output_data_location",
        str(data_location / "embeddings" / "MSA_Transformer"),
        "--model_type",
        "MSA_Transformer",
        "--model_location",
        str(EMBEDDING_MODEL),
        "--max_positions",
        "1024",
        "--batch_size",
        str(args.embedding_batch_size),
        "--num_MSA_sequences",
        str(num_msa_sequences),
        "--MSA_data_folder",
        str(MSA_FILES),
        "--MSA_weight_data_folder",
        str(MSA_WEIGHTS),
        "--path_to_hhfilter",
        str(HHFILTER),
    ]
    if args.fast_msa_mode:
        command.extend(["--path_to_clustalomega", str(CLUSTALOMEGA), "--fast_MSA_mode"])
    return command


def train_command(args: argparse.Namespace, row: pd.Series) -> list[str]:
    data_location = runtime_root(args)
    return [
        str(args.python),
        str(TRAIN_SCRIPT),
        "--data_location",
        str(data_location),
        "--assay_data_location",
        row["assay_data_location"],
        "--model_config_location",
        str(args.model_config),
        "--target_config_location",
        str(TARGET_CONFIG),
        "--fold_variable_name",
        "matched_split",
        "--test_fold_index",
        "1",
        "--zero_shot_fitness_predictions_location",
        row["zero_shot_fitness_predictions_location"],
        "--training_fp16",
        "--sequence_embeddings_folder",
        row["sequence_embeddings_folder"],
        "--embedding_model_location",
        str(EMBEDDING_MODEL),
        "--MSA_data_folder",
        str(MSA_FILES),
        "--MSA_location",
        row["MSA_location"],
        "--MSA_weight_data_folder",
        str(MSA_WEIGHTS),
        "--path_to_hhfilter",
        str(HHFILTER),
        "--MSA_sequence_weights_filename",
        row["MSA_sequence_weights_filename"],
        "--target_seq",
        row["target_seq"],
        "--MSA_start",
        str(int(row["MSA_start"])),
        "--MSA_end",
        str(int(row["MSA_end"])),
        "--model_name_suffix",
        row["model_name_suffix"],
        "--output_results_location",
        str(data_location / "results"),
        "--seed",
        str(int(row["seed"])),
        "--do_not_save_model_checkpoint",
    ]


def top_metrics(y_true: np.ndarray, y_pred: np.ndarray, percentile_reference: np.ndarray, k: int = 100) -> dict[str, float]:
    k = min(k, len(y_true))
    predicted = np.argsort(-y_pred)[:k]
    true_order = np.argsort(-y_true)
    top_count = max(1, math.ceil(len(y_true) * 0.01))
    true_top = set(true_order[:top_count].tolist())
    minimum = float(np.min(y_true))
    maximum = float(np.max(y_true))
    if maximum == minimum:
        relevance = np.zeros(len(y_true), dtype=float)
    else:
        relevance = (y_true - minimum) / (maximum - minimum)
    discounts = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum((np.power(2.0, relevance[predicted]) - 1.0) / discounts))
    ideal = float(np.sum((np.power(2.0, relevance[true_order[:k]]) - 1.0) / discounts))
    best = float(np.max(y_true[predicted]))
    return {
        "ndcg_at_100": dcg / ideal if ideal else 0.0,
        "top1pct_recall_at_100": len(set(predicted.tolist()) & true_top) / len(true_top),
        "best_true_in_pred_top_100": best,
        "best_top100_percentile": float(np.mean(percentile_reference <= best)),
    }


def locate_prediction_file(row: pd.Series, data_location: Path) -> Path | None:
    assay_id = row["assay_id"]
    suffix = row["model_name_suffix"]
    folder = data_location / "model_predictions" / suffix
    if not folder.exists():
        return None
    candidates = sorted(folder.glob(f"ProteinNPT_{assay_id}_fitness_zero_shot_fitness_predictions_matched_split_*.csv"))
    if not candidates:
        candidates = sorted(folder.glob("*.csv"))
    return candidates[0] if candidates else None


def collect_one(row: pd.Series, out_dir: Path, data_location: Path, overwrite: bool) -> dict:
    assay_id = row["assay_id"]
    result_dir = out_dir / "runs" / assay_id
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seed{int(row['seed'])}_budget{int(row['budget'])}_{row['regime']}"
    json_path = result_dir / f"{stem}.json"
    npz_path = result_dir / f"{stem}.npz"
    if not overwrite:
        existing = current_collected_result(
            json_path, npz_path, str(row.get("run_fingerprint_sha256"))
        )
        if existing is not None:
            return existing
    prediction_path = locate_prediction_file(row, data_location)
    if prediction_path is None:
        record = dict(row[KEYS].to_dict())
        record.update({
            "assay_id": assay_id,
            "run_fingerprint_sha256": row.get("run_fingerprint_sha256"),
            "status": "missing_prediction_file",
            "result_file": str(json_path),
        })
        with json_path.open("w") as handle:
            json.dump(record, handle, indent=2)
        return record
    prediction = pd.read_csv(prediction_path)
    run_table = pd.read_csv(row["assay_data_location"], usecols=["mutant", "DMS_score", "matched_split"])
    test_table = run_table[run_table["matched_split"] == 1][["mutant", "DMS_score"]].copy()
    merged = prediction.merge(test_table, on="mutant", how="inner", validate="one_to_one")
    if len(merged) != len(test_table):
        status = "test_mutant_identity_mismatch"
    elif "predictions_fitness" not in merged.columns:
        status = "missing_predictions_fitness_column"
    else:
        status = "ok"
    record = dict(row[KEYS].to_dict())
    record.update({
        "assay_id": assay_id,
        "method": "ProteinNPT",
        "run_fingerprint_sha256": row.get("run_fingerprint_sha256"),
        "status": status,
        "n_train": int(row["n_train"]),
        "n_test": int(len(test_table)),
        "prediction_file": str(prediction_path),
        "result_file": str(json_path),
        "npz_file": str(npz_path),
    })
    if status == "ok":
        y_true = merged["DMS_score"].to_numpy(float)
        y_pred = merged["predictions_fitness"].to_numpy(float)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[finite]
        y_pred = y_pred[finite]
        mutants = merged.loc[finite, "mutant"].astype(str).to_numpy()
        rho = spearmanr(y_true, y_pred).correlation if len(y_true) > 1 else np.nan
        record["spearman_all"] = float(rho) if np.isfinite(rho) else np.nan
        percentile_reference = pd.read_csv(row["percentile_reference_location"], usecols=["DMS_score"])
        full_assay_scores = pd.to_numeric(percentile_reference["DMS_score"], errors="coerce").to_numpy(float)
        full_assay_scores = full_assay_scores[np.isfinite(full_assay_scores)]
        record.update(top_metrics(y_true, y_pred, full_assay_scores))
        record["best_top100_percentile_reference"] = "complete_assay_candidate_pool"
        np.savez_compressed(npz_path, mutant=mutants, y_true=y_true, y_pred=y_pred)
    with json_path.open("w") as handle:
        json.dump(record, handle, indent=2)
    return record


def stage_check(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    data_location = runtime_root(args)
    rows = []
    for _, row in manifest.iterrows():
        run_json = args.out / "runs" / row["assay_id"] / f"seed{int(row['seed'])}_budget{int(row['budget'])}_{row['regime']}.json"
        rows.append({
            "assay": row["assay"],
            "assay_id": row["assay_id"],
            "seed": int(row["seed"]),
            "budget": int(row["budget"]),
            "regime": row["regime"],
            "embedding_file_exists": Path(row["embedding_file"]).exists(),
            "prediction_file_exists": locate_prediction_file(row, data_location) is not None,
            "collected_json_exists": run_json.exists(),
        })
    check = pd.DataFrame(rows)
    check_path = args.out / "proteinnpt_execution_check.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    check.to_csv(check_path, index=False)
    print(json.dumps({"check_file": str(check_path), "n_rows": int(len(check))}, indent=2))


def stage_embed(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    assay_manifest = pd.read_csv(args.assay_manifest)
    eligible = assay_manifest[assay_manifest["eligible"].astype(str).str.lower().isin(["true", "1"])].copy()
    eligible = shard_frame(eligible, parse_shard(args.assay_shard), ["assay_id"])
    data_location = runtime_root(args)
    status_rows = []
    for _, assay_row in eligible.sort_values("assay_id").iterrows():
        assay_id = assay_row["assay_id"]
        run_rows = manifest[manifest["assay_id"].eq(assay_id)]
        if run_rows.empty:
            continue
        shared = pd.read_csv(run_rows.iloc[0]["shared_assay_location"], usecols=["mutant"])
        expected_mutants = shared["mutant"].astype(str).tolist()
        embedding_file = Path(run_rows.iloc[0]["embedding_file"])
        embedding_input_hash = str(run_rows.iloc[0]["embedding_input_sha256"])
        embedding_model_hash = str(run_rows.iloc[0]["embedding_model_sha256"])
        with msa_resource_lock(str(assay_row["MSA_filename"])):
            msa_configuration = filtered_msa_configuration(
                str(assay_row["MSA_filename"]),
                args.num_msa_sequences,
                prepare=not args.dry_run,
                weight_file_name=str(assay_row["weight_file_name"]),
            )
        validity = validate_h5(
            embedding_file,
            expected_mutants,
            args.embedding_seed,
            embedding_input_hash,
            embedding_model_hash,
            args.fast_msa_mode,
            msa_configuration["num_msa_sequences"],
        )
        if validity["valid"] and not args.overwrite:
            status = {"status": "ok_existing", "assay_id": assay_id, "embedding_file": str(embedding_file), **msa_configuration, **validity}
        else:
            command = embed_command(args, assay_row, msa_configuration["num_msa_sequences"])
            status = run_command(command, PNPT, data_location / "logs" / "embeddings" / f"{assay_id}.log", args.dry_run)
            status.update({"assay_id": assay_id, "embedding_file": str(embedding_file)})
            status.update(msa_configuration)
            if status["status"] == "ok":
                initial_validity = validate_h5(embedding_file, expected_mutants)
                if initial_validity["valid"]:
                    atomic_write_json(
                        {
                            "embedding_seed": args.embedding_seed,
                            "mutant_order_sha256": mutant_list_hash(expected_mutants),
                            "embedding_input_sha256": embedding_input_hash,
                            "embedding_model_sha256": embedding_model_hash,
                            "fast_msa_mode": args.fast_msa_mode,
                            "embedding_batch_size": args.embedding_batch_size,
                            "num_msa_sequences": msa_configuration["num_msa_sequences"],
                            "filtered_msa_depth": msa_configuration["filtered_msa_depth"],
                            "filtered_msa_file": msa_configuration["filtered_msa_file"],
                            "n_mutants": len(expected_mutants),
                            "embedding_model": str(EMBEDDING_MODEL.resolve()),
                            "official_embedding_script": str(EMBEDDING_SCRIPT.resolve()),
                        },
                        embedding_metadata_path(embedding_file),
                    )
                status.update(
                    validate_h5(
                        embedding_file,
                        expected_mutants,
                        args.embedding_seed,
                        embedding_input_hash,
                        embedding_model_hash,
                        args.fast_msa_mode,
                        msa_configuration["num_msa_sequences"],
                    )
                )
        status_rows.append(status)
    status_path = shard_status_path(args.out, "embedding", args.assay_shard)
    args.out.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(status_rows).to_csv(status_path, index=False)
    print(json.dumps({"status_file": str(status_path), "n_assays": len(status_rows)}, indent=2))


def stage_train(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    manifest = shard_frame(manifest, parse_shard(args.run_shard), ["run_id"])
    assay_manifest = pd.read_csv(args.assay_manifest).set_index("assay_id")
    msa_configurations = {}
    data_location = runtime_root(args)
    status_rows = []
    for _, row in manifest.sort_values(["assay_id", "seed", "budget", "regime"]).iterrows():
        result_json = args.out / "runs" / row["assay_id"] / f"seed{int(row['seed'])}_budget{int(row['budget'])}_{row['regime']}.json"
        if not args.overwrite:
            collected = current_collected_result(
                result_json,
                result_json.with_suffix(".npz"),
                str(row.get("run_fingerprint_sha256")),
            )
            if collected is not None:
                status_rows.append({"run_id": row["run_id"], "status": "ok_existing_collected", "result_file": str(result_json)})
                continue
        embedding_file = Path(row["embedding_file"])
        shared = pd.read_csv(row["shared_assay_location"], usecols=["mutant"])
        assay_id = str(row["assay_id"])
        if assay_id not in msa_configurations:
            msa_configurations[assay_id] = filtered_msa_configuration(
                str(assay_manifest.loc[assay_id, "MSA_filename"]),
                args.num_msa_sequences,
                weight_file_name=str(assay_manifest.loc[assay_id, "weight_file_name"]),
            )
        validity = validate_h5(
            embedding_file,
            shared["mutant"].astype(str).tolist(),
            args.embedding_seed,
            str(row["embedding_input_sha256"]),
            str(row["embedding_model_sha256"]),
            args.fast_msa_mode,
            msa_configurations[assay_id]["num_msa_sequences"],
        )
        if not validity["valid"]:
            status_rows.append({"run_id": row["run_id"], "status": "missing_or_invalid_embedding", **validity})
            continue
        prediction_path = locate_prediction_file(row, data_location)
        if prediction_path is not None and not args.overwrite:
            status_rows.append({"run_id": row["run_id"], "status": "ok_existing_prediction", "prediction_file": str(prediction_path)})
            continue
        command = train_command(args, row)
        log_name = f"{row['assay_id']}_seed{int(row['seed'])}_budget{int(row['budget'])}_{row['regime']}.log"
        status = run_command(command, PNPT, data_location / "logs" / "training" / log_name, args.dry_run)
        status.update({"run_id": row["run_id"], "assay_id": row["assay_id"], "seed": int(row["seed"]), "budget": int(row["budget"]), "regime": row["regime"]})
        status_rows.append(status)
    status_path = shard_status_path(args.out, "training", args.run_shard)
    args.out.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(status_rows).to_csv(status_path, index=False)
    print(json.dumps({"status_file": str(status_path), "n_runs": len(status_rows)}, indent=2))


def stage_collect(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    manifest = shard_frame(manifest, parse_shard(args.run_shard), ["run_id"])
    data_location = runtime_root(args)
    records = []
    for _, row in manifest.sort_values(["assay_id", "seed", "budget", "regime"]).iterrows():
        records.append(collect_one(row, args.out, data_location, args.overwrite))
    detail = pd.DataFrame(records)
    detail_path = args.out / "proteinnpt_matched_budget_detail.csv"
    detail.to_csv(detail_path, index=False)
    metadata = {
        "status": "collected_existing_predictions",
        "n_records": int(len(detail)),
        "n_ok": int((detail["status"] == "ok").sum()) if "status" in detail.columns else 0,
        "detail_file": str(detail_path),
    }
    with (args.out / "proteinnpt_collect_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


def shard_status_path(output: Path, stage: str, shard: str) -> Path:
    if not shard:
        return output / f"proteinnpt_{stage}_status.csv"
    shard_name = shard.replace("/", "_of_")
    return output / f"proteinnpt_{stage}_status" / f"shard_{shard_name}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or collect ProteinNPT under matched project low-measurement splits.")
    parser.add_argument("--stage", choices=["check", "embed", "train", "collect"], required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--assay-manifest", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=MODEL_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--assay-shard", default="")
    parser.add_argument("--run-shard", default="")
    parser.add_argument("--embedding-batch-size", type=int, default=1)
    parser.add_argument("--num-msa-sequences", type=int, default=MAX_MSA_SEQUENCES)
    parser.add_argument("--embedding-seed", type=int, default=2023)
    parser.add_argument("--fast-msa-mode", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.manifest = args.manifest.resolve() if args.manifest is not None else args.out / "proteinnpt_run_manifest.csv"
    args.assay_manifest = args.assay_manifest.resolve() if args.assay_manifest is not None else args.out / "proteinnpt_assay_resource_manifest.csv"
    args.reference = args.reference.resolve() if args.reference is not None else args.out / "runtime" / "protein_npt_matched_reference.csv"
    args.model_config = args.model_config.resolve()
    if args.stage == "check":
        stage_check(args)
    elif args.stage == "embed":
        stage_embed(args)
    elif args.stage == "train":
        stage_train(args)
    elif args.stage == "collect":
        stage_collect(args)


if __name__ == "__main__":
    main()
