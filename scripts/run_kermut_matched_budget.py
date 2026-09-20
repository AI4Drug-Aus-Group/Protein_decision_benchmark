from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import gpytorch
import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
KERMUT = ROOT / "external_repos" / "kermut"
SPLITS = ROOT / "results" / "splits" / "low_n_splits.csv"
DMS_DIR = ROOT / "proteingym" / "extracted" / "DMS_ProteinGym_substitutions"
REFERENCE = KERMUT / "data" / "DMS_substitutions.csv"
EMBEDDINGS = KERMUT / "data" / "embeddings"
ZERO_SHOT = KERMUT / "data" / "zero_shot_fitness_predictions" / "ESM2" / "650M"
CONDITIONAL = KERMUT / "data" / "conditional_probs" / "ProteinMPNN"
COORDINATES = KERMUT / "data" / "structures" / "coords"
DEFAULT_OUT = ROOT / "results" / "local_matched_reproduction" / "kermut"

sys.path.insert(0, str(KERMUT))
from src.data.data_utils import Tokenizer
from src.model.gp import ExactGPKermut


def parse_set(value: str, cast):
    return {cast(item) for item in value.split(",") if item}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        return [], np.empty((0, 1280), dtype=np.float32)
    with h5py.File(path, "r") as handle:
        mutants = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in handle["mutants"][:]]
        embeddings = np.asarray(handle["embeddings"][:], dtype=np.float32)
    if embeddings.ndim == 3:
        embeddings = embeddings.mean(axis=1)
    return mutants, embeddings


def embedding_table(assay_id: str) -> pd.DataFrame:
    pieces = []
    for group in ["substitutions_singles", "substitutions_multiples"]:
        mutants, embeddings = read_embeddings(EMBEDDINGS / group / "ESM2" / f"{assay_id}.h5")
        if mutants:
            pieces.append(pd.DataFrame({"mutant": mutants, "embedding_index": np.arange(len(mutants)), "embedding_group": group}))
    if not pieces:
        return pd.DataFrame(columns=["mutant", "embedding_index", "embedding_group"])
    return pd.concat(pieces, ignore_index=True).drop_duplicates("mutant", keep="last")


def load_embedding_matrix(assay_id: str, records: pd.DataFrame) -> np.ndarray:
    matrices = {}
    for group in records["embedding_group"].unique():
        _, matrices[group] = read_embeddings(EMBEDDINGS / group / "ESM2" / f"{assay_id}.h5")
    return np.vstack([matrices[group][int(index)] for group, index in zip(records["embedding_group"], records["embedding_index"])]).astype(np.float32)


def load_assay(assay: str, reference: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, str]:
    assay_id = assay.removesuffix(".csv")
    table = pd.read_csv(DMS_DIR / assay, usecols=["mutant", "mutated_sequence", "DMS_score"])
    table = table.dropna(subset=["mutant", "mutated_sequence", "DMS_score"]).copy()
    table["mutant"] = table["mutant"].astype(str)
    table["mutated_sequence"] = table["mutated_sequence"].astype(str)
    zero = pd.read_csv(ZERO_SHOT / f"{assay_id}.csv", usecols=["mutant", "esm2_t33_650M_UR50D"])
    zero = zero.groupby("mutant", as_index=False)["esm2_t33_650M_UR50D"].mean()
    embedding_records = embedding_table(assay_id)
    table = table.merge(zero, on="mutant", how="inner").merge(embedding_records, on="mutant", how="inner")
    table = table.drop_duplicates("mutant", keep="first").reset_index(drop=True)
    embeddings = load_embedding_matrix(assay_id, table[["embedding_group", "embedding_index"]])
    target = reference.loc[reference["DMS_id"].eq(assay_id), "target_seq"]
    if target.empty:
        raise ValueError("assay absent from Kermut reference file")
    return table, embeddings, str(target.iloc[0])


def top_metrics(y_true: np.ndarray, y_pred: np.ndarray, percentile_reference: np.ndarray, k: int = 100) -> dict[str, float]:
    k = min(k, len(y_true))
    predicted = np.argsort(-y_pred)[:k]
    true_order = np.argsort(-y_true)
    top_count = max(1, math.ceil(len(y_true) * 0.01))
    true_top = set(true_order[:top_count].tolist())
    minimum = float(np.min(y_true))
    maximum = float(np.max(y_true))
    relevance = np.zeros(len(y_true), dtype=float) if maximum == minimum else (y_true - minimum) / (maximum - minimum)
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


def build_features(tokenizer: Tokenizer, sequences: list[str], zero: np.ndarray, embeddings: np.ndarray, device: torch.device) -> torch.Tensor:
    tokens = tokenizer(np.asarray(sequences)).float()
    features = torch.cat([
        tokens,
        torch.from_numpy(zero.astype(np.float32)).unsqueeze(-1),
        torch.from_numpy(embeddings.astype(np.float32)),
    ], dim=-1)
    return features.to(device)


def fit_predict(
    table: pd.DataFrame,
    embeddings: np.ndarray,
    target_sequence: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    assay_id: str,
    seed: int,
    device: torch.device,
    prediction_batch_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tokenizer = Tokenizer(flatten=True)
    zero = table["esm2_t33_650M_UR50D"].to_numpy(dtype=np.float32)
    x_train = build_features(
        tokenizer,
        table.loc[train_indices, "mutated_sequence"].tolist(),
        zero[train_indices],
        embeddings[train_indices],
        device,
    )
    y_raw = table.loc[train_indices, "DMS_score"].to_numpy(dtype=np.float32)
    y_mean = float(y_raw.mean())
    y_std = float(y_raw.std(ddof=1))
    if not np.isfinite(y_std) or y_std <= 0:
        raise ValueError("constant training phenotype")
    y_train = torch.from_numpy((y_raw - y_mean) / y_std).to(device)

    km_cfg = OmegaConf.load(KERMUT / "configs" / "gp" / "kermut.yaml")["mutation_kernel"]
    conditional = torch.from_numpy(np.load(CONDITIONAL / f"{assay_id}.npy")).float()
    coords = torch.from_numpy(np.load(COORDINATES / f"{assay_id}.npy")).float()
    wild_type = tokenizer(target_sequence).squeeze()
    noise_prior = gpytorch.priors.HalfCauchyPrior(scale=0.1)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior)
    model = ExactGPKermut(
        train_x=x_train,
        train_y=y_train,
        likelihood=likelihood,
        km_cfg=km_cfg,
        use_zero_shot=True,
        embedding_dim=1280,
        conditional_probs=conditional,
        wt_sequence=wild_type,
        coords=coords,
        use_global_kernel=True,
    ).to(device)
    likelihood = likelihood.to(device)
    model.train()
    likelihood.train()
    objective = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    final_loss = float("nan")
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        output = model(x_train)
        loss = -objective(output, y_train)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    model.eval()
    likelihood.eval()
    predictions = np.empty(len(test_indices), dtype=np.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in range(0, len(test_indices), prediction_batch_size):
            block = test_indices[start:start + prediction_batch_size]
            x_test = build_features(
                tokenizer,
                table.loc[block, "mutated_sequence"].tolist(),
                zero[block],
                embeddings[block],
                device,
            )
            standardized = likelihood(model(x_test)).mean.detach().cpu().numpy()
            predictions[start:start + len(block)] = standardized * y_std + y_mean
            del x_test
    diagnostics = {
        "final_training_loss": final_loss,
        "learned_noise": float(likelihood.noise.detach().cpu()),
        "learned_zero_shot_scale": float(model.zero_shot_scale.detach().cpu()),
        "learned_kernel_weight": float(torch.sigmoid(model.alpha).detach().cpu()),
    }
    del model, likelihood, x_train, y_train
    torch.cuda.empty_cache()
    return predictions, diagnostics


def split_groups(splits_path: Path, regimes: set[str], budgets: set[int], seeds: set[int], assays: set[str]) -> dict[str, list[tuple[tuple[int, int, str], list[str]]]]:
    splits = pd.read_csv(splits_path)
    splits = splits[splits["regime"].isin(regimes) & splits["budget"].isin(budgets) & splits["seed"].isin(seeds)]
    if assays:
        normalized = {value if value.endswith(".csv") else f"{value}.csv" for value in assays}
        splits = splits[splits["assay"].isin(normalized)]
    output = {}
    for (assay, seed, budget, regime), block in splits.groupby(["assay", "seed", "budget", "regime"], sort=True):
        output.setdefault(str(assay), []).append(((int(seed), int(budget), str(regime)), block["mutant"].astype(str).tolist()))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", default="mixed_random")
    parser.add_argument("--budgets", default="20,50,100")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--assays", default="")
    parser.add_argument("--assay-shard", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prediction-batch-size", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.splits = args.splits.resolve()
    groups = split_groups(
        args.splits,
        parse_set(args.regimes, str),
        parse_set(args.budgets, int),
        parse_set(args.seeds, int),
        parse_set(args.assays, str),
    )
    if args.assay_shard:
        shard_index, shard_count = [int(value) for value in args.assay_shard.split("/")]
        groups = {assay: value for index, (assay, value) in enumerate(sorted(groups.items())) if index % shard_count == shard_index}

    args.out.mkdir(parents=True, exist_ok=True)
    run_dir = args.out / "runs"
    run_dir.mkdir(exist_ok=True)
    reference = pd.read_csv(REFERENCE)
    summary_rows = []
    for assay, assay_groups in sorted(groups.items()):
        assay_id = assay.removesuffix(".csv")
        try:
            table, embeddings, target = load_assay(assay, reference)
            mutant_to_index = {mutant: index for index, mutant in enumerate(table["mutant"])}
            candidate_hash = hashlib.sha256("\n".join(table["mutant"]).encode()).hexdigest()
        except Exception as error:
            summary_rows.append({"assay": assay, "status": f"assay_load_failed:{type(error).__name__}", "message": str(error)})
            continue

        for (seed, budget, regime), requested in assay_groups:
            output = run_dir / assay_id / f"seed{seed}_budget{budget}_{regime}.json"
            prediction_output = output.with_suffix(".npz")
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and prediction_output.exists() and not args.overwrite:
                summary_rows.append(json.loads(output.read_text()))
                continue
            started = time.time()
            record = {
                "assay": assay,
                "seed": seed,
                "budget": budget,
                "regime": regime,
                "model": "Kermut",
                "repository_commit": "e6500221b5159a1896272bfc4a3c6e3896eb0027",
                "n_train_requested": len(dict.fromkeys(requested)),
                "n_candidates_with_complete_resources": len(table),
                "candidate_order_sha256": candidate_hash,
            }
            unique_requested = list(dict.fromkeys(requested))
            missing = [mutant for mutant in unique_requested if mutant not in mutant_to_index]
            if len(unique_requested) != budget:
                record.update(status="excluded_non_nominal_budget", missing_training_mutants=missing)
            elif missing:
                record.update(status="excluded_missing_training_resources", n_missing_training_mutants=len(missing), missing_training_mutants=missing[:20])
            elif len(table) - budget < 20:
                record.update(status="excluded_insufficient_test_candidates")
            else:
                train_indices = np.asarray([mutant_to_index[mutant] for mutant in unique_requested], dtype=int)
                test_mask = np.ones(len(table), dtype=bool)
                test_mask[train_indices] = False
                test_indices = np.flatnonzero(test_mask)
                try:
                    prediction, diagnostics = fit_predict(
                        table,
                        embeddings,
                        target,
                        train_indices,
                        test_indices,
                        assay_id,
                        seed,
                        device,
                        args.prediction_batch_size,
                    )
                    y_true = table.loc[test_indices, "DMS_score"].to_numpy(dtype=float)
                    full_assay_scores = pd.to_numeric(
                        pd.read_csv(DMS_DIR / assay, usecols=["DMS_score"])["DMS_score"],
                        errors="coerce",
                    ).to_numpy(float)
                    full_assay_scores = full_assay_scores[np.isfinite(full_assay_scores)]
                    rho = spearmanr(prediction, y_true).statistic
                    record.update(
                        status="ok",
                        n_train_used=len(train_indices),
                        n_test=len(test_indices),
                        spearman_all=float(rho),
                        **top_metrics(y_true, prediction, full_assay_scores),
                        best_top100_percentile_reference="complete_assay_candidate_pool",
                        **diagnostics,
                    )
                    np.savez_compressed(
                        prediction_output,
                        test_indices=test_indices.astype(np.int32),
                        y_pred=prediction.astype(np.float32),
                        y_true=y_true.astype(np.float32),
                    )
                except Exception as error:
                    record.update(status=f"failed:{type(error).__name__}", message=str(error))
                    torch.cuda.empty_cache()
            record["wall_time_seconds"] = time.time() - started
            output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            summary_rows.append(record)
            print(json.dumps({key: record.get(key) for key in ["assay", "seed", "budget", "regime", "status", "spearman_all", "wall_time_seconds"]}), flush=True)

    shard_name = args.assay_shard.replace("/", "_of_") if args.assay_shard else "all"
    pd.DataFrame(summary_rows).to_csv(args.out / f"kermut_matched_budget_detail_{shard_name}.csv", index=False)
    metadata = {
        "model": "Kermut",
        "repository": str(KERMUT.relative_to(ROOT)),
        "repository_commit": "e6500221b5159a1896272bfc4a3c6e3896eb0027",
        "official_configuration": "composite mutation kernel plus ESM2-650M global kernel and ESM2 zero-shot mean function",
        "training_steps": 150,
        "learning_rate": 0.1,
        "input_splits": str(args.splits.relative_to(ROOT)) if args.splits.is_relative_to(ROOT) else str(args.splits),
        "regimes": sorted(parse_set(args.regimes, str)),
        "budgets": sorted(parse_set(args.budgets, int)),
        "seeds": sorted(parse_set(args.seeds, int)),
        "gpu": args.gpu,
        "assay_shard": args.assay_shard,
        "split_file_sha256": file_sha256(args.splits),
    }
    (args.out / f"kermut_matched_budget_metadata_{shard_name}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
