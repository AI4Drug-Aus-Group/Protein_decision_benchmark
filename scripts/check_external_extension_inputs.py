from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sota_extension" / "input_checks"
DMS_DIR = ROOT / "proteingym" / "extracted" / "DMS_ProteinGym_substitutions"
SPLITS = ROOT / "results" / "splits" / "low_n_splits.csv"
PNPT = ROOT / "results" / "local_matched_reproduction" / "proteinnpt_sampled_2000_len400"
KERMUT_RUNS = ROOT / "results" / "local_matched_reproduction" / "kermut" / "runs"
STRICT_DOUBLES = ROOT / "results" / "epistasis" / "strict_double_epistasis_residuals.csv"
STM_SUMMARY = ROOT / "results" / "single_to_multi" / "single_to_multi_summary.csv"


def sha256_text(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def mutation_order(mutant: str) -> int:
    if not isinstance(mutant, str) or mutant in {"", "WT", "wildtype", "_wt"}:
        return 0
    return mutant.count(":") + 1


def assay_features(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, usecols=["mutant", "mutated_sequence", "DMS_score"])
    frame = frame.dropna(subset=["mutant", "mutated_sequence", "DMS_score"]).copy()
    frame["mutant"] = frame["mutant"].astype(str)
    frame["DMS_score"] = pd.to_numeric(frame["DMS_score"], errors="coerce")
    frame = frame.dropna(subset=["DMS_score"]).drop_duplicates("mutant", keep="first")
    orders = frame["mutant"].map(mutation_order)
    lengths = frame["mutated_sequence"].astype(str).str.len()
    return {
        "assay": path.name,
        "n_variants": int(len(frame)),
        "sequence_length_median": float(lengths.median()) if len(lengths) else math.nan,
        "n_single_variants": int((orders == 1).sum()),
        "n_double_variants": int((orders == 2).sum()),
        "n_higher_order_variants": int((orders >= 3).sum()),
        "fitness_mean": float(frame["DMS_score"].mean()) if len(frame) else math.nan,
        "fitness_sd": float(frame["DMS_score"].std(ddof=1)) if len(frame) > 1 else math.nan,
        "fitness_iqr": float(frame["DMS_score"].quantile(0.75) - frame["DMS_score"].quantile(0.25)) if len(frame) else math.nan,
    }


def summarize_numeric(frame: pd.DataFrame, group: str, columns: list[str]) -> pd.DataFrame:
    rows = []
    for value, block in frame.groupby(group, dropna=False):
        row = {group: value, "n_assays": int(len(block))}
        for column in columns:
            data = pd.to_numeric(block[column], errors="coerce")
            row[f"mean_{column}"] = float(data.mean()) if data.notna().any() else math.nan
            row[f"median_{column}"] = float(data.median()) if data.notna().any() else math.nan
            row[f"q25_{column}"] = float(data.quantile(0.25)) if data.notna().any() else math.nan
            row[f"q75_{column}"] = float(data.quantile(0.75)) if data.notna().any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def existing_scope() -> pd.DataFrame:
    rows = []
    kermut_files = sorted(KERMUT_RUNS.glob("*/*.json"))
    kermut = []
    for path in kermut_files:
        with path.open() as handle:
            record = json.load(handle)
        kermut.append(record)
    if kermut:
        frame = pd.DataFrame(kermut)
        ok = frame[frame["status"].eq("ok")]
        for regime, block in ok.groupby("regime", dropna=False):
            rows.append({
                "method": "Kermut",
                "evidence_object": "low-measurement supervised prediction",
                "regime": regime,
                "n_runs": int(len(block)),
                "n_assays": int(block["assay"].nunique()),
                "budgets": ",".join(str(x) for x in sorted(block["budget"].dropna().astype(int).unique())),
                "seeds": ",".join(str(x) for x in sorted(block["seed"].dropna().astype(int).unique())),
                "status": "completed",
            })
    pnpt_detail = PNPT / "proteinnpt_matched_budget_detail.csv"
    if pnpt_detail.exists():
        frame = pd.read_csv(pnpt_detail)
        ok = frame[frame["status"].eq("ok")]
        for regime, block in ok.groupby("regime", dropna=False):
            rows.append({
                "method": "ProteinNPT",
                "evidence_object": "low-measurement supervised prediction",
                "regime": regime,
                "n_runs": int(len(block)),
                "n_assays": int(block["assay"].nunique()),
                "budgets": ",".join(str(x) for x in sorted(block["budget"].dropna().astype(int).unique())),
                "seeds": ",".join(str(x) for x in sorted(block["seed"].dropna().astype(int).unique())),
                "status": "completed",
            })
    for method in ["EVOLVEpro-style", "ALDE-style"]:
        rows.append({
            "method": method,
            "evidence_object": "active experimental acquisition",
            "regime": "common active-learning candidate pool",
            "n_runs": math.nan,
            "n_assays": math.nan,
            "budgets": "20 initial measurements plus five acquisition rounds",
            "seeds": "0,1,2,3,4",
            "status": "completed as matched-setting policy",
        })
    return pd.DataFrame(rows)


def pnpt_subset_check(all_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.read_csv(PNPT / "proteinnpt_matched_budget_detail.csv")
    complete = (
        detail[detail["status"].eq("ok")]
        .groupby("assay")
        .agg(n_runs=("seed", "size"), n_seeds=("seed", "nunique"), budgets=("budget", lambda x: ",".join(str(v) for v in sorted(set(x)))))
        .reset_index()
    )
    complete["in_proteinnpt_completed_subset"] = complete["n_runs"].eq(15) & complete["n_seeds"].eq(5) & complete["budgets"].eq("20,50,100")
    selected = set(complete.loc[complete["in_proteinnpt_completed_subset"], "assay"])
    features = all_features.copy()
    features["group"] = np.where(features["assay"].isin(selected), "ProteinNPT completed candidate-capped subset", "All other ProteinGym assays")
    summary = summarize_numeric(
        features,
        "group",
        ["n_variants", "sequence_length_median", "n_single_variants", "n_double_variants", "n_higher_order_variants", "fitness_sd", "fitness_iqr"],
    )
    return features, summary


def single_only_candidate_coverage() -> pd.DataFrame:
    splits = pd.read_csv(SPLITS)
    splits = splits[splits["regime"].eq("single_only")].copy()
    pnpt_manifest = pd.read_csv(PNPT / "proteinnpt_run_manifest.csv")
    rows = []
    for assay, runs in pnpt_manifest.groupby("assay", sort=True):
        shared_path = Path(str(runs.iloc[0]["shared_assay_location"]))
        if not shared_path.exists():
            continue
        pool = set(pd.read_csv(shared_path, usecols=["mutant"])["mutant"].astype(str))
        needed = set(splits.loc[splits["assay"].eq(assay), "mutant"].astype(str))
        missing = sorted(needed - pool)
        rows.append({
            "assay": assay,
            "n_candidate_pool": int(len(pool)),
            "n_single_only_training_mutants_required": int(len(needed)),
            "n_missing_from_existing_proteinnpt_pool": int(len(missing)),
            "existing_pool_can_reuse_embeddings_for_single_only": len(missing) == 0,
            "missing_examples": ";".join(missing[:10]),
            "candidate_identity_sha256": sha256_text(sorted(pool)),
        })
    return pd.DataFrame(rows)


def multi_evolve_feasibility(all_features: pd.DataFrame) -> pd.DataFrame:
    residuals = pd.read_csv(STRICT_DOUBLES)
    double_counts = residuals.groupby("assay").agg(
        n_component_resolved_doubles=("mutant", "size"),
        mean_abs_epistasis_residual=("abs_epistasis_residual", "mean"),
        median_abs_epistasis_residual=("abs_epistasis_residual", "median"),
    ).reset_index()
    rows = []
    for path in sorted(DMS_DIR.glob("*.csv")):
        frame = pd.read_csv(path, usecols=["mutant", "DMS_score"])
        frame = frame.dropna(subset=["mutant", "DMS_score"]).copy()
        frame["mutant"] = frame["mutant"].astype(str)
        frame["DMS_score"] = pd.to_numeric(frame["DMS_score"], errors="coerce")
        frame = frame.dropna(subset=["DMS_score"])
        orders = frame["mutant"].map(mutation_order)
        singles = frame[orders.eq(1)]
        threshold = float(frame["DMS_score"].quantile(0.75)) if len(frame) else math.nan
        rows.append({
            "assay": path.name,
            "n_single_mutants": int(len(singles)),
            "n_single_mutants_above_assay_q75": int((singles["DMS_score"] > threshold).sum()) if len(frame) else 0,
            "n_higher_order_variants": int((orders >= 3).sum()),
        })
    feasibility = pd.DataFrame(rows).merge(double_counts, on="assay", how="left")
    feasibility["n_component_resolved_doubles"] = feasibility["n_component_resolved_doubles"].fillna(0).astype(int)
    feasibility["has_multi_evolve_like_design_space"] = (
        (feasibility["n_single_mutants_above_assay_q75"] >= 10)
        & (feasibility["n_component_resolved_doubles"] >= 100)
        & (feasibility["n_higher_order_variants"] >= 20)
    )
    return feasibility


def strict_common_pool_status() -> pd.DataFrame:
    detail = pd.read_csv(PNPT / "proteinnpt_matched_budget_detail.csv")
    rows = []
    for assay, block in detail[detail["status"].eq("ok")].groupby("assay", sort=True):
        pnpt_n = sorted(block["n_test"].dropna().astype(int).unique())
        kermut_paths = sorted((KERMUT_RUNS / assay.removesuffix(".csv")).glob("seed*_budget*_mixed_random.json"))
        kermut_n = []
        for path in kermut_paths:
            with path.open() as handle:
                record = json.load(handle)
            if record.get("status") == "ok":
                kermut_n.append(int(record.get("n_test", -1)))
        rows.append({
            "assay": assay,
            "proteinnpt_test_counts": ",".join(str(x) for x in pnpt_n),
            "kermut_test_counts": ",".join(str(x) for x in sorted(set(kermut_n))),
            "has_kermut_mixed_random_results": bool(kermut_n),
            "strict_variant_identity_from_existing_metrics": False,
            "reason": "Existing summarized low-measurement comparator files do not retain every per-variant prediction, so exact common-pool metrics must be recomputed or explicitly labelled by candidate count.",
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_features = pd.DataFrame([assay_features(path) for path in sorted(DMS_DIR.glob("*.csv"))])
    scope = existing_scope()
    subset_detail, subset_summary = pnpt_subset_check(all_features)
    single_coverage = single_only_candidate_coverage()
    feasibility = multi_evolve_feasibility(all_features)
    common_pool = strict_common_pool_status()
    stm = pd.read_csv(STM_SUMMARY)
    scope.to_csv(OUT / "existing_external_evidence_scope.csv", index=False)
    all_features.to_csv(OUT / "all_assay_basic_features.csv", index=False)
    subset_detail.to_csv(OUT / "proteinnpt_subset_representativeness_detail.csv", index=False)
    subset_summary.to_csv(OUT / "proteinnpt_subset_representativeness_summary.csv", index=False)
    single_coverage.to_csv(OUT / "proteinnpt_existing_pool_single_only_coverage.csv", index=False)
    feasibility.to_csv(OUT / "multi_evolve_like_feasibility_matrix.csv", index=False)
    common_pool.to_csv(OUT / "strict_common_pool_status.csv", index=False)
    metadata = {
        "status": "ok",
        "n_assays_total": int(all_features["assay"].nunique()),
        "n_proteinnpt_completed_assays": int((subset_detail["group"] == "ProteinNPT completed candidate-capped subset").sum()),
        "n_proteinnpt_assays_reusing_existing_embedding_pool_for_single_only": int(single_coverage["existing_pool_can_reuse_embeddings_for_single_only"].sum()),
        "n_multi_evolve_like_feasible_assays": int(feasibility["has_multi_evolve_like_design_space"].sum()),
        "single_to_multi_existing_methods": sorted(stm["method"].dropna().unique().tolist()),
        "outputs": [str(path.relative_to(ROOT)) for path in sorted(OUT.glob("*.csv"))],
    }
    (OUT / "check_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
