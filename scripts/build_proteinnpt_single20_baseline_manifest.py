from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOW_SPLITS = ROOT / "results" / "splits" / "low_n_splits.csv"
TARGET_CONFIG = ROOT / "external_repos" / "ProteinNPT" / "proteinnpt" / "utils" / "target_configs" / "fitness.json"
DEFAULT_MODEL_CONFIG = ROOT / "results" / "sota_extension" / "configs" / "PNPT_sampled_1000_steps.json"
DEFAULT_ROOT = ROOT / "results" / "sota_extension" / "proteinnpt_selected_double20_expanded4_seed5_sample800_1000steps"


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item != ""]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_table_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    text = frame[columns].astype(str).to_csv(index=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--low-splits", type=Path, default=LOW_SPLITS)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--out-manifest", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    selected_manifest = pd.read_csv(root / "proteinnpt_run_manifest.csv")
    low = pd.read_csv(args.low_splits)
    seeds = parse_csv_ints(args.seeds)
    model_config_hash = sha256(args.model_config.resolve())
    target_config_hash = sha256(TARGET_CONFIG)
    baseline_rows = []
    for assay, assay_runs in selected_manifest.groupby("assay", sort=True):
        example = assay_runs.iloc[0]
        assay_id = str(example["assay_id"])
        shared = pd.read_csv(example["shared_assay_location"])
        for seed in seeds:
            block = low[
                low["assay"].eq(assay)
                & low["seed"].eq(seed)
                & low["budget"].eq(args.budget)
                & low["regime"].eq("single_only")
            ].sort_values("rank")
            train_mutants = block["mutant"].astype(str).tolist()
            if len(train_mutants) != args.budget or len(set(train_mutants)) != args.budget:
                raise ValueError(f"bad single-only split for {assay} seed {seed}")
            run_table = shared.copy()
            run_table["matched_split"] = np.where(run_table["mutant"].astype(str).isin(train_mutants), 0, 1)
            if int((run_table["matched_split"] == 0).sum()) != args.budget:
                raise ValueError(f"training split did not map exactly for {assay} seed {seed}")
            run_dir = root / "runtime" / "run_inputs" / assay_id / f"seed{seed}_budget{args.budget}_single20_baseline"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_assay_path = run_dir / assay
            run_table.to_csv(run_assay_path, index=False)
            candidate_hash = stable_table_hash(shared, ["mutant", "mutated_sequence", "DMS_score"])
            embedding_input_hash = stable_table_hash(shared, ["mutant", "mutated_sequence"])
            zero_table = pd.read_csv(root / "runtime" / "zero_shot_fitness_predictions" / "substitutions" / assay)
            zero_hash = stable_table_hash(zero_table, ["mutant", "mutated_sequence", "MSA_Transformer_ensemble"])
            run_table_hash = stable_table_hash(run_table, ["mutant", "mutated_sequence", "DMS_score", "matched_split"])
            fingerprint_payload = "|".join([
                run_table_hash,
                zero_hash,
                model_config_hash,
                target_config_hash,
                str(example["embedding_model_sha256"]),
            ])
            run_fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
            row = example.to_dict()
            row.update({
                "run_id": f"{assay_id}__seed{seed}__budget{args.budget}__single20_baseline",
                "assay": assay,
                "assay_id": assay_id,
                "seed": seed,
                "budget": args.budget,
                "regime": "single20_baseline",
                "n_train": args.budget,
                "n_test": int((run_table["matched_split"] == 1).sum()),
                "n_candidates": int(len(run_table)),
                "model_name_suffix": f"pnpt_{assay_id[:16]}_s{seed}_b{args.budget}_{run_fingerprint[:10]}",
                "assay_data_location": str(run_assay_path.resolve()),
                "shared_assay_location": str(Path(example["shared_assay_location"]).resolve()),
                "zero_shot_fitness_predictions_location": str((root / "runtime" / "zero_shot_fitness_predictions" / "substitutions").resolve()),
                "sequence_embeddings_folder": str((root / "runtime" / "embeddings" / "MSA_Transformer").resolve()),
                "embedding_file": str((root / "runtime" / "embeddings" / "MSA_Transformer" / f"{assay_id}.h5").resolve()),
                "candidate_table_sha256": candidate_hash,
                "embedding_input_sha256": embedding_input_hash,
                "zero_auxiliary_table_sha256": zero_hash,
                "run_table_sha256": run_table_hash,
                "run_fingerprint_sha256": run_fingerprint,
            })
            baseline_rows.append(row)
    output = args.out_manifest.resolve() if args.out_manifest else root / "proteinnpt_single20_baseline_run_manifest.csv"
    pd.DataFrame(baseline_rows).to_csv(output, index=False)
    metadata = {
        "status": "ok",
        "root": str(root),
        "out_manifest": str(output),
        "n_runs": len(baseline_rows),
        "assays": sorted(pd.DataFrame(baseline_rows)["assay"].unique().tolist()) if baseline_rows else [],
        "seeds": seeds,
        "budget": args.budget,
        "model_config": str(args.model_config.resolve()),
        "model_config_sha256": model_config_hash,
        "shared_candidate_pool": "same shared candidate tables and embeddings as the selected-double ProteinNPT root",
    }
    (root / "proteinnpt_single20_baseline_manifest_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
