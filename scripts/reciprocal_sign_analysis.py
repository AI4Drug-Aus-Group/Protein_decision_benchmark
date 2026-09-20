from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(os.environ.get(
    "PROTEIN_DECISION_PACKAGE", ROOT / "analysis_package"
)).resolve()
DATA_ROOT = Path(
    PACKAGE_ROOT / "multi_mutant"
)
CAND_PATH = DATA_ROOT / "regime_summaries" / "reciprocal_sign_epistasis_candidates.csv"
SUMMARY_PATH = DATA_ROOT / "regime_summaries" / "reciprocal_sign_epistasis_summary.csv"
RESID_PATH = DATA_ROOT / "double_residuals" / "strict_double_epistasis_residuals.csv"

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "experiments"
PREFIX = "reciprocal_sign_"

METHODS = [
    "VenusREM",
    "ProSST-2048",
    "ESM3",
    "S3F_MSA",
    "S2F_MSA",
    "PoET",
    "GEMME",
    "TranceptEVE_L",
]
ADDITIVE = "additive_prediction"
PREDICTORS = [ADDITIVE] + METHODS

BOOTSTRAPS = 5000
PERMUTATIONS = 100000
MIN_N_PRIMARY = 20
MIN_N_STRICT = 50
SHIFTS = [-0.50, -0.25, 0.0, 0.25, 0.50]
SEED = 20240518


def abs_spearman(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 3:
        return np.nan
    if np.nanstd(pred) == 0 or np.nanstd(target) == 0:
        return np.nan
    rho = spearmanr(pred, target).statistic
    return float(abs(rho)) if np.isfinite(rho) else np.nan


def abs_spearman_ordinal(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 3:
        return np.nan
    rx = np.empty(len(pred), float)
    rx[np.argsort(pred, kind="stable")] = np.arange(len(pred), dtype=float)
    ry = np.empty(len(target), float)
    ry[np.argsort(target, kind="stable")] = np.arange(len(target), dtype=float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom <= 0:
        return np.nan
    return float(abs((rx * ry).sum() / denom))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, values.size, size=(BOOTSTRAPS, values.size))
    sims = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))


def sign_flip_p(values: np.ndarray, rng: np.random.Generator) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    observed = abs(float(values.mean()))
    extreme = 0
    block = 10000
    done = 0
    while done < PERMUTATIONS:
        take = min(block, PERMUTATIONS - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(take, values.size))
        permuted = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
        done += take
    return float((extreme + 1) / (PERMUTATIONS + 1))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, float)
    out = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if valid.any():
        obs = values[valid]
        order = np.argsort(obs)
        ranked = obs[order] * len(obs) / np.arange(1, len(obs) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        restored = np.empty(len(obs))
        restored[order] = np.minimum(ranked, 1.0)
        out[valid] = restored
    return out


def per_assay_metrics(frame: pd.DataFrame, predictors: list[str], label: str) -> pd.DataFrame:
    rows = []
    for assay, grp in frame.groupby("assay", sort=True):
        target = grp["score_double"].to_numpy(float)
        rec = {"subset": label, "assay": assay, "n": int(len(grp))}
        for pred in predictors:
            if pred not in grp.columns:
                rec[pred] = np.nan
                rec[f"n_{pred}"] = 0
                continue
            x = grp[pred].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(target)
            rec[f"n_{pred}"] = int(ok.sum())
            rec[pred] = abs_spearman(x[ok], target[ok]) if ok.sum() >= 3 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def assay_average(metrics: pd.DataFrame, predictors: list[str], min_n: int) -> pd.DataFrame:
    keep = metrics[metrics["n"] >= min_n]
    rows = []
    for pred in predictors:
        vals = keep[pred].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "predictor": pred,
                "min_n_per_assay": min_n,
                "n_assays": int(vals.size),
                "n_candidates_in_eligible_assays": int(keep["n"].sum()),
                "mean_abs_spearman": float(vals.mean()) if vals.size else np.nan,
                "median_abs_spearman": float(np.median(vals)) if vals.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def paired_tests(metrics: pd.DataFrame, min_n: int, family: str, rng: np.random.Generator) -> pd.DataFrame:
    keep = metrics[metrics["n"] >= min_n]
    rows = []
    for method in METHODS:
        pair = keep[[ADDITIVE, method]].to_numpy(float)
        pair = pair[np.isfinite(pair).all(axis=1)]
        diffs = pair[:, 1] - pair[:, 0]
        mean_diff, lo, hi = bootstrap_mean_ci(diffs, rng)
        rows.append(
            {
                "family": family,
                "min_n_per_assay": min_n,
                "method": method,
                "n_paired_assays": int(diffs.size),
                "mean_abs_spearman_method": float(pair[:, 1].mean()) if diffs.size else np.nan,
                "mean_abs_spearman_additive": float(pair[:, 0].mean()) if diffs.size else np.nan,
                "mean_diff_method_minus_additive": mean_diff,
                "bootstrap_95ci_low": lo,
                "bootstrap_95ci_high": hi,
                "assays_method_better": int(np.count_nonzero(diffs > 0)),
                "assays_additive_better": int(np.count_nonzero(diffs < 0)),
                "two_sided_sign_permutation_p": sign_flip_p(diffs, rng),
            }
        )
    out = pd.DataFrame(rows)
    out["benjamini_hochberg_q_eight_methods"] = benjamini_hochberg(
        out["two_sided_sign_permutation_p"].to_numpy(float)
    )
    return out


def candidate_regime(cand: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    published = pd.read_csv(SUMMARY_PATH).set_index("method")["abs_spearman_double_score"]

    target = cand["score_double"].to_numpy(float)
    pooled_rows = []
    for pred in PREDICTORS:
        x = cand[pred].to_numpy(float)
        pooled_rows.append(
            {
                "predictor": pred,
                "published_pooled_abs_spearman": float(published.get(pred, np.nan)),
                "recomputed_pooled_abs_spearman_ordinal_ranks": abs_spearman_ordinal(x, target),
                "recomputed_pooled_abs_spearman_average_ranks": abs_spearman(x, target),
                "n_pooled": int(len(cand)),
            }
        )
    pooled = pd.DataFrame(pooled_rows)

    metrics = per_assay_metrics(cand, PREDICTORS, "published_reciprocal_sign_candidates")
    metrics.to_csv(OUT_DIR / f"{PREFIX}candidate_regime_per_assay_metrics.csv", index=False)

    counts = (
        metrics[["assay", "n"]]
        .sort_values("n", ascending=False)
        .assign(share_of_total=lambda d: d["n"] / d["n"].sum())
    )
    counts["cumulative_share"] = counts["share_of_total"].cumsum()
    counts.to_csv(OUT_DIR / f"{PREFIX}candidate_regime_assay_candidate_counts.csv", index=False)

    averages = []
    for min_n in (3, MIN_N_PRIMARY, MIN_N_STRICT):
        averages.append(assay_average(metrics, PREDICTORS, min_n))
    averages = pd.concat(averages, ignore_index=True)

    comparison = averages.merge(
        pooled[["predictor", "published_pooled_abs_spearman", "recomputed_pooled_abs_spearman_average_ranks"]],
        on="predictor",
        how="left",
    )
    comparison["assay_avg_minus_pooled"] = (
        comparison["mean_abs_spearman"] - comparison["published_pooled_abs_spearman"]
    )
    add_mean = comparison.loc[comparison["predictor"] == ADDITIVE].set_index("min_n_per_assay")["mean_abs_spearman"]
    comparison["gap_over_additive_assay_avg"] = comparison["mean_abs_spearman"] - comparison[
        "min_n_per_assay"
    ].map(add_mean)
    comparison["gap_over_additive_pooled"] = comparison["published_pooled_abs_spearman"] - float(
        published.get(ADDITIVE, np.nan)
    )
    comparison.to_csv(OUT_DIR / f"{PREFIX}candidate_regime_assay_avg_vs_pooled.csv", index=False)

    drop_rows = []
    order = counts["assay"].tolist()
    for k in (1, 2, 5):
        dropped = set(order[:k])
        sub = cand[~cand["assay"].isin(dropped)]
        tgt = sub["score_double"].to_numpy(float)
        rec = {"dropped_top_assays": k, "n_remaining": int(len(sub))}
        for pred in PREDICTORS:
            rec[pred] = abs_spearman(sub[pred].to_numpy(float), tgt)
        drop_rows.append(rec)
    for assay in order[:5]:
        sub = cand[cand["assay"] == assay]
        tgt = sub["score_double"].to_numpy(float)
        rec = {"dropped_top_assays": f"only:{assay}", "n_remaining": int(len(sub))}
        for pred in PREDICTORS:
            rec[pred] = abs_spearman(sub[pred].to_numpy(float), tgt)
        drop_rows.append(rec)
    pooled_drop = pd.DataFrame(drop_rows)
    pooled_drop.to_csv(OUT_DIR / f"{PREFIX}candidate_regime_pooled_leave_out_largest.csv", index=False)

    boot_rows = []
    for min_n in (MIN_N_PRIMARY, MIN_N_STRICT):
        keep = metrics[metrics["n"] >= min_n]
        for pred in PREDICTORS:
            vals = keep[pred].to_numpy(float)
            mean, lo, hi = bootstrap_mean_ci(vals, rng)
            boot_rows.append(
                {
                    "min_n_per_assay": min_n,
                    "predictor": pred,
                    "n_assays": int(np.isfinite(vals).sum()),
                    "mean_abs_spearman": mean,
                    "bootstrap_95ci_low": lo,
                    "bootstrap_95ci_high": hi,
                }
            )
    pd.DataFrame(boot_rows).to_csv(OUT_DIR / f"{PREFIX}candidate_regime_bootstrap_ci.csv", index=False)

    tests = pd.concat(
        [
            paired_tests(metrics, MIN_N_PRIMARY, "reciprocal_sign_assay_level", rng),
            paired_tests(metrics, MIN_N_STRICT, "reciprocal_sign_assay_level", rng),
        ],
        ignore_index=True,
    )
    tests.to_csv(OUT_DIR / f"{PREFIX}candidate_regime_paired_tests.csv", index=False)

    info = {
        "n_candidates": int(len(cand)),
        "n_assays": int(cand["assay"].nunique()),
        "n_assays_min20": int((metrics["n"] >= MIN_N_PRIMARY).sum()),
        "n_assays_min50": int((metrics["n"] >= MIN_N_STRICT).sum()),
        "largest_assay": counts.iloc[0]["assay"],
        "largest_assay_n": int(counts.iloc[0]["n"]),
        "largest_assay_share": float(counts.iloc[0]["share_of_total"]),
        "top2_share": float(counts["share_of_total"].iloc[:2].sum()),
        "top5_share": float(counts["share_of_total"].iloc[:5].sum()),
        "median_assay_n": float(counts["n"].median()),
        "assays_under_50": int((counts["n"] < 50).sum()),
    }
    return metrics, {
        "pooled": pooled,
        "comparison": comparison,
        "tests": tests,
        "info": info,
        "pooled_drop": pooled_drop,
    }


def proxy_shift(cand: pd.DataFrame, rng: np.random.Generator) -> dict:
    usecols = [
        "assay",
        "mutant",
        "score_double",
        "score_single_a",
        "score_single_b",
        "wt_proxy",
        "additive_prediction",
    ]
    resid = pd.read_csv(RESID_PATH, usecols=usecols)
    n_raw = len(resid)
    resid = resid.dropna(subset=["score_double", "score_single_a", "score_single_b", "wt_proxy"])
    resid = resid[np.isfinite(resid[["score_double", "score_single_a", "score_single_b", "wt_proxy"]]).all(axis=1)]

    da = resid["score_single_a"].to_numpy(float) - resid["wt_proxy"].to_numpy(float)
    db = resid["score_single_b"].to_numpy(float) - resid["wt_proxy"].to_numpy(float)
    dab = resid["score_double"].to_numpy(float) - resid["wt_proxy"].to_numpy(float)
    recomputed = (da * dab < 0) & (db * dab < 0)
    resid["recomputed_flag"] = recomputed.astype(int)

    published_keys = set(zip(cand["assay"], cand["mutant"]))
    resid_keys = list(zip(resid["assay"], resid["mutant"]))
    resid["published_flag"] = [1 if k in published_keys else 0 for k in resid_keys]

    tp = int(((resid["recomputed_flag"] == 1) & (resid["published_flag"] == 1)).sum())
    fp = int(((resid["recomputed_flag"] == 1) & (resid["published_flag"] == 0)).sum())
    fn = int(((resid["recomputed_flag"] == 0) & (resid["published_flag"] == 1)).sum())
    tn = int(((resid["recomputed_flag"] == 0) & (resid["published_flag"] == 0)).sum())
    agreement = (tp + tn) / len(resid)
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    check = pd.DataFrame(
        [
            {
                "n_strict_rows_raw": n_raw,
                "n_strict_rows_used": int(len(resid)),
                "n_published_candidates": int(len(cand)),
                "n_recomputed_candidates": int(resid["recomputed_flag"].sum()),
                "both": tp,
                "recomputed_only": fp,
                "published_only": fn,
                "neither": tn,
                "agreement_rate": agreement,
                "jaccard": jaccard,
            }
        ]
    )
    check.to_csv(OUT_DIR / f"{PREFIX}proxy_shift_membership_reproduction.csv", index=False)
    if agreement < 0.999:
        print("WARNING: membership reproduction below 99.9%; downstream shift analysis is unreliable")

    iqr = resid.groupby("assay")["score_double"].agg(lambda s: float(np.percentile(s, 75) - np.percentile(s, 25)))
    iqr.name = "iqr_double"
    resid = resid.join(iqr, on="assay")

    scored = resid.merge(
        cand[["assay", "mutant"] + METHODS],
        on=["assay", "mutant"],
        how="left",
        validate="one_to_one",
    )
    has_scores = scored[METHODS].notna().all(axis=1)
    scored["has_method_scores"] = has_scores.astype(int)

    fixed = scored[scored["published_flag"] == 1].copy()
    inv_rows = []
    for shift in SHIFTS:
        wt_s = fixed["wt_proxy"].to_numpy(float) + shift * fixed["iqr_double"].to_numpy(float)
        fixed[ADDITIVE] = (
            fixed["score_single_a"].to_numpy(float) + fixed["score_single_b"].to_numpy(float) - wt_s
        )
        m = per_assay_metrics(fixed, [ADDITIVE], "fixed_set")
        a = assay_average(m, [ADDITIVE], MIN_N_PRIMARY)
        inv_rows.append(
            {
                "shift_in_iqr_units": shift,
                "assay_avg_additive_fixed_published_set": float(a["mean_abs_spearman"].iloc[0]),
                "pooled_additive_fixed_published_set": abs_spearman(
                    fixed[ADDITIVE].to_numpy(float), fixed["score_double"].to_numpy(float)
                ),
            }
        )
    invariance = pd.DataFrame(inv_rows)
    invariance.to_csv(OUT_DIR / f"{PREFIX}proxy_shift_proxy_shift_invariance_check.csv", index=False)

    shift_summary_rows = []
    per_assay_all = []
    test_frames = []
    for shift in SHIFTS:
        wt_shift = scored["wt_proxy"].to_numpy(float) + shift * scored["iqr_double"].to_numpy(float)
        sa = scored["score_single_a"].to_numpy(float) - wt_shift
        sb = scored["score_single_b"].to_numpy(float) - wt_shift
        sab = scored["score_double"].to_numpy(float) - wt_shift
        member = (sa * sab < 0) & (sb * sab < 0)
        add_pred = (
            scored["score_single_a"].to_numpy(float)
            + scored["score_single_b"].to_numpy(float)
            - wt_shift
        )

        sub = scored.loc[member].copy()
        sub[ADDITIVE] = add_pred[member]
        published_member = scored["published_flag"].to_numpy(bool)
        n_full = int(member.sum())
        n_new = int((member & ~published_member).sum())
        n_lost = int((~member & published_member).sum())

        full_metrics = per_assay_metrics(sub, [ADDITIVE], f"shift_{shift:+.2f}_full")
        add_full = assay_average(full_metrics, [ADDITIVE], MIN_N_PRIMARY)

        evaluable = sub[sub["has_method_scores"] == 1].copy()
        metrics = per_assay_metrics(evaluable, PREDICTORS, f"shift_{shift:+.2f}")
        metrics["shift"] = shift
        per_assay_all.append(metrics)
        avg = assay_average(metrics, PREDICTORS, MIN_N_PRIMARY).set_index("predictor")

        tests = paired_tests(metrics, MIN_N_PRIMARY, f"shift_{shift:+.2f}", rng)
        tests["shift"] = shift
        test_frames.append(tests)

        add_mean = float(avg.loc[ADDITIVE, "mean_abs_spearman"])
        row = {
            "shift_in_iqr_units": shift,
            "n_candidates_full": n_full,
            "n_candidates_evaluable": int(len(evaluable)),
            "coverage_of_perturbed_set": len(evaluable) / n_full if n_full else np.nan,
            "n_new_vs_published": n_new,
            "n_lost_vs_published": n_lost,
            "n_assays_min20": int((metrics["n"] >= MIN_N_PRIMARY).sum()),
            "additive_assay_avg_full_set": float(add_full.loc[add_full["predictor"] == ADDITIVE, "mean_abs_spearman"].iloc[0]),
            "additive_assay_avg_evaluable": add_mean,
            "pooled_additive_evaluable": abs_spearman(
                evaluable[ADDITIVE].to_numpy(float), evaluable["score_double"].to_numpy(float)
            ),
        }
        for pred in PREDICTORS:
            row[f"assay_avg_{pred}"] = float(avg.loc[pred, "mean_abs_spearman"])
            row[f"n_assays_{pred}"] = int(avg.loc[pred, "n_assays"])
            if pred != ADDITIVE:
                row[f"gap_over_additive_{pred}"] = float(avg.loc[pred, "mean_abs_spearman"]) - add_mean
        for pred in METHODS:
            row[f"pooled_{pred}"] = abs_spearman(
                evaluable[pred].to_numpy(float), evaluable["score_double"].to_numpy(float)
            )
        shift_summary_rows.append(row)

    summary = pd.DataFrame(shift_summary_rows)
    summary.to_csv(OUT_DIR / f"{PREFIX}proxy_shift_shift_summary.csv", index=False)
    pd.concat(per_assay_all, ignore_index=True).to_csv(
        OUT_DIR / f"{PREFIX}proxy_shift_shift_per_assay_metrics.csv", index=False
    )
    tests_all = pd.concat(test_frames, ignore_index=True)
    tests_all.to_csv(OUT_DIR / f"{PREFIX}proxy_shift_shift_paired_tests.csv", index=False)
    return {"check": check, "summary": summary, "tests": tests_all, "invariance": invariance}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cand = pd.read_csv(CAND_PATH)

    metrics, t1 = candidate_regime(cand, rng)
    t2 = proxy_shift(cand, rng)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)
    print("\n=== Candidate regime pooled reproduction ===")
    print(t1["pooled"].to_string(index=False))
    print("\n=== Candidate regime assay-averaged vs pooled (min_n=20) ===")
    cols = [
        "predictor",
        "n_assays",
        "mean_abs_spearman",
        "median_abs_spearman",
        "published_pooled_abs_spearman",
        "assay_avg_minus_pooled",
        "gap_over_additive_assay_avg",
        "gap_over_additive_pooled",
    ]
    print(t1["comparison"].query("min_n_per_assay == 20")[cols].to_string(index=False))
    print("\n=== Candidate regime assay-averaged (all 64 assays, min_n=3) ===")
    print(t1["comparison"].query("min_n_per_assay == 3")[cols].to_string(index=False))
    print("\n=== Candidate regime paired tests vs additive ===")
    print(t1["tests"].to_string(index=False))
    print("\n=== Candidate regime candidate concentration ===")
    print(json.dumps(t1["info"], indent=2, default=str))
    print("\n=== Candidate regime pooled value after dropping the largest assays / single-assay pooled ===")
    print(t1["pooled_drop"].to_string(index=False))

    print("\n=== Proxy shift membership reproduction ===")
    print(t2["check"].to_string(index=False))
    print("\n=== Proxy shift proxy-shift invariance check (fixed published candidate set) ===")
    print(t2["invariance"].to_string(index=False))
    print("\n=== Proxy shift shift summary ===")
    show = [
        "shift_in_iqr_units",
        "n_candidates_full",
        "n_candidates_evaluable",
        "coverage_of_perturbed_set",
        "n_new_vs_published",
        "n_lost_vs_published",
        "n_assays_min20",
        "additive_assay_avg_full_set",
        "assay_avg_additive_prediction",
        "assay_avg_VenusREM",
        "assay_avg_ProSST-2048",
        "gap_over_additive_VenusREM",
        "gap_over_additive_ProSST-2048",
        "pooled_VenusREM",
        "pooled_ProSST-2048",
        "pooled_additive_evaluable",
    ]
    print(t2["summary"][show].to_string(index=False))
    print("\n=== Proxy shift paired tests (VenusREM, ProSST-2048) ===")
    print(
        t2["tests"]
        .query("method in ['VenusREM', 'ProSST-2048']")[
            [
                "shift",
                "method",
                "n_paired_assays",
                "mean_diff_method_minus_additive",
                "bootstrap_95ci_low",
                "bootstrap_95ci_high",
                "assays_method_better",
                "assays_additive_better",
                "two_sided_sign_permutation_p",
                "benjamini_hochberg_q_eight_methods",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
