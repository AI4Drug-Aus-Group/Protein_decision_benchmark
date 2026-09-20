from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ["ProSST-2048", "ESM3", "S2F_MSA", "S3F_MSA", "PoET", "VenusREM"]
STRUCTURE_METHODS = ["ProSST-2048", "S2F_MSA", "S3F_MSA"]
SEQUENCE_METHODS = ["ESM3", "PoET", "VenusREM"]
SCORE_NAMES = [
    "ensemble_std",
    "ensemble_rank_std",
    "additive_ensemble_disagreement",
    "additive_rank_disagreement",
    "sequence_structure_disagreement",
    "ensemble_mean",
    "ensemble_rank_mean",
]


def finite_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)


def percentile_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def fit_linear_calibration(score: np.ndarray, target: np.ndarray) -> tuple[float, float] | None:
    valid = np.isfinite(score) & np.isfinite(target)
    score = score[valid]
    target = target[valid]
    if len(score) < 3 or np.unique(score).size < 2:
        return None
    design = np.column_stack([np.ones(len(score)), score])
    penalty = np.diag([0.0, 1e-8])
    weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ target
    return float(weights[0]), float(weights[1])


def component_single_table(rows: pd.DataFrame) -> pd.DataFrame:
    first = rows[["single_a", "score_single_a"]].rename(
        columns={"single_a": "mutant", "score_single_a": "DMS_score"}
    )
    second = rows[["single_b", "score_single_b"]].rename(
        columns={"single_b": "mutant", "score_single_b": "DMS_score"}
    )
    return pd.concat([first, second], ignore_index=True).drop_duplicates("mutant")


def load_assay_scores(path: Path) -> pd.DataFrame:
    usecols = ["mutant", *METHODS]
    frame = pd.read_csv(path, usecols=lambda name: name in usecols)
    return finite_numeric(frame, METHODS)


def calibrated_score_table(residuals: pd.DataFrame, score_path: Path) -> pd.DataFrame:
    scores = load_assay_scores(score_path)
    singles = component_single_table(residuals).merge(scores, on="mutant", how="inner", validate="one_to_one")
    doubles = residuals.merge(scores, on="mutant", how="inner", validate="one_to_one")
    calibrated: list[str] = []
    calibration_sizes: list[int] = []
    for method in METHODS:
        if method not in doubles or method not in singles:
            continue
        valid = singles[[method, "DMS_score"]].dropna()
        fit = fit_linear_calibration(valid[method].to_numpy(float), valid["DMS_score"].to_numpy(float))
        if fit is None:
            continue
        intercept, slope = fit
        doubles[method] = intercept + slope * doubles[method].to_numpy(float)
        calibrated.append(method)
        calibration_sizes.append(len(valid))
    if len(calibrated) < 2:
        return pd.DataFrame()
    predictions = doubles[calibrated]
    ranks = predictions.apply(percentile_rank)
    doubles["ensemble_mean"] = predictions.mean(axis=1, skipna=True)
    doubles["ensemble_std"] = predictions.std(axis=1, skipna=True, ddof=1)
    doubles["ensemble_rank_mean"] = ranks.mean(axis=1, skipna=True)
    doubles["ensemble_rank_std"] = ranks.std(axis=1, skipna=True, ddof=1)
    doubles["additive_ensemble_disagreement"] = (
        doubles["additive_prediction"] - doubles["ensemble_mean"]
    ).abs()
    doubles["additive_rank_disagreement"] = (
        percentile_rank(doubles["additive_prediction"]) - doubles["ensemble_rank_mean"]
    ).abs()
    structure = [method for method in STRUCTURE_METHODS if method in calibrated]
    sequence = [method for method in SEQUENCE_METHODS if method in calibrated]
    if structure and sequence:
        doubles["sequence_structure_disagreement"] = (
            ranks[structure].mean(axis=1, skipna=True) - ranks[sequence].mean(axis=1, skipna=True)
        ).abs()
    else:
        doubles["sequence_structure_disagreement"] = np.nan
    doubles["n_calibrated_methods"] = len(calibrated)
    doubles["minimum_calibration_singles"] = min(calibration_sizes)
    return doubles


def tail_events(frame: pd.DataFrame, tail_fraction: float) -> dict[str, pd.Series]:
    residual = frame["epistasis_residual"]
    lower = residual.quantile(tail_fraction)
    upper = residual.quantile(1.0 - tail_fraction)
    absolute_cut = frame["abs_epistasis_residual"].quantile(1.0 - tail_fraction)
    return {
        "positive_residual_tail": residual >= upper,
        "negative_residual_tail": residual <= lower,
        "absolute_residual_tail": frame["abs_epistasis_residual"] >= absolute_cut,
        "sign_epistasis_candidate": frame["sign_epistasis_candidate"] > 0,
    }


def assay_enrichment(
    frame: pd.DataFrame,
    assay: str,
    k_values: list[int],
    tail_fraction: float,
) -> list[dict[str, object]]:
    events = tail_events(frame, tail_fraction)
    rows: list[dict[str, object]] = []
    for score_name in SCORE_NAMES:
        evaluable = frame.loc[frame[score_name].notna()].copy()
        if len(evaluable) < 100:
            continue
        ranked = evaluable.sort_values(score_name, ascending=False, kind="mergesort")
        for k in k_values:
            selected = ranked.head(min(k, len(ranked)))
            for event_name, event in events.items():
                base_event = event.loc[evaluable.index]
                selected_event = event.loc[selected.index]
                base_hits = int(base_event.sum())
                selected_hits = int(selected_event.sum())
                base_rate = base_hits / len(evaluable)
                selected_rate = selected_hits / len(selected)
                enrichment = selected_rate / base_rate if base_rate > 0 else math.nan
                smoothed_base = (base_hits + 0.5) / (len(evaluable) + 1.0)
                smoothed_selected = (selected_hits + 0.5) / (len(selected) + 1.0)
                rows.append(
                    {
                        "assay": assay,
                        "score": score_name,
                        "k": k,
                        "event": event_name,
                        "n_variants": len(evaluable),
                        "n_selected": len(selected),
                        "n_calibrated_methods": int(evaluable["n_calibrated_methods"].iloc[0]),
                        "minimum_calibration_singles": int(evaluable["minimum_calibration_singles"].iloc[0]),
                        "base_hits": base_hits,
                        "selected_hits": selected_hits,
                        "base_rate": base_rate,
                        "selected_rate": selected_rate,
                        "enrichment": enrichment,
                        "log2_enrichment_smoothed": math.log2(smoothed_selected / smoothed_base),
                    }
                )
    return rows


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_p(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(repetitions, len(values)))
    permuted = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (repetitions + 1))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    valid = np.isfinite(values)
    adjusted = np.full(len(values), np.nan)
    if valid.any():
        observed = values[valid]
        order = np.argsort(observed)
        ranked = observed[order] * len(observed) / np.arange(1, len(observed) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        restored = np.empty(len(observed))
        restored[order] = np.minimum(ranked, 1.0)
        adjusted[valid] = restored
    return pd.Series(adjusted, index=p_values.index)


def summarize(detail: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (score, k, event), group in detail.groupby(["score", "k", "event"], sort=True):
        enrichment = group["enrichment"].to_numpy(float)
        enrichment = enrichment[np.isfinite(enrichment)]
        log_enrichment = group["log2_enrichment_smoothed"].to_numpy(float)
        log_enrichment = log_enrichment[np.isfinite(log_enrichment)]
        raw_low, raw_high = bootstrap_mean(enrichment, rng, n_boot)
        log_low, log_high = bootstrap_mean(log_enrichment, rng, n_boot)
        rows.append(
            {
                "score": score,
                "k": int(k),
                "event": event,
                "n_assays": len(log_enrichment),
                "mean_enrichment": float(enrichment.mean()) if len(enrichment) else math.nan,
                "median_enrichment": float(np.median(enrichment)) if len(enrichment) else math.nan,
                "bootstrap_95ci_low_mean_enrichment": raw_low,
                "bootstrap_95ci_high_mean_enrichment": raw_high,
                "mean_log2_enrichment_smoothed": float(log_enrichment.mean()) if len(log_enrichment) else math.nan,
                "median_log2_enrichment_smoothed": float(np.median(log_enrichment)) if len(log_enrichment) else math.nan,
                "bootstrap_95ci_low_mean_log2_enrichment": log_low,
                "bootstrap_95ci_high_mean_log2_enrichment": log_high,
                "assays_enrichment_above_one": int(np.count_nonzero(enrichment > 1)),
                "assays_enrichment_below_one": int(np.count_nonzero(enrichment < 1)),
                "two_sided_sign_flip_p_log2_enrichment": sign_flip_p(log_enrichment, rng, n_perm),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["event", "k", "score"]).reset_index(drop=True)
    summary["benjamini_hochberg_q_all_tests"] = benjamini_hochberg(
        summary["two_sided_sign_flip_p_log2_enrichment"]
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "negative_residual")
    parser.add_argument("--k", default="10,50,100")
    parser.add_argument("--tail-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    residual_path = root / "results" / "epistasis" / "strict_double_epistasis_residuals.csv"
    score_dir = root / "proteingym" / "extracted" / "zero_shot_substitutions_scores"
    residuals = pd.read_csv(residual_path)
    residuals = finite_numeric(
        residuals,
        ["score_single_a", "score_single_b", "additive_prediction", "epistasis_residual", "abs_epistasis_residual", "sign_epistasis_candidate"],
    )
    detail_rows: list[dict[str, object]] = []
    k_values = [int(value) for value in args.k.split(",") if value]
    for assay, assay_residuals in residuals.groupby("assay", sort=True):
        score_path = score_dir / assay
        if not score_path.exists():
            continue
        table = calibrated_score_table(assay_residuals, score_path)
        if len(table) >= 100:
            detail_rows.extend(assay_enrichment(table, assay, k_values, args.tail_fraction))
    detail = pd.DataFrame(detail_rows)
    summary = summarize(detail, args.bootstrap, args.permutations, args.seed)
    detail.to_csv(out / "signed_residual_enrichment_detail.csv", index=False)
    summary.to_csv(out / "signed_residual_enrichment_summary.csv", index=False)
    metadata = {
        "residual_source": str(residual_path.relative_to(root)),
        "score_source": str(score_dir.relative_to(root)),
        "methods": METHODS,
        "calibration": "Each zero-shot score was linearly mapped to the assay measurement scale using unique component single mutants only.",
        "tail_definition": f"Assay-wise lower and upper {args.tail_fraction:.0%} of the full epistatic-residual distribution.",
        "k_values": k_values,
        "minimum_variants_per_assay": 100,
        "bootstrap_unit": "assay",
        "hypothesis_test": "two-sided sign-flip test of smoothed log2 enrichment against zero",
        "multiple_testing": "Benjamini-Hochberg correction across all score, k and event combinations",
        "bootstrap_replicates": args.bootstrap,
        "sign_flip_permutations": args.permutations,
        "random_seed": args.seed,
        "n_assays": int(detail["assay"].nunique()),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\\n", encoding="utf-8")


if __name__ == "__main__":
    main()
