from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(os.environ.get(
    "PROTEIN_DECISION_PACKAGE", ROOT / "analysis_package"
)).resolve()
DATA = Path(
    PACKAGE_ROOT / "active_learning/common_pool_active_learning"
)
OUT = Path(__file__).resolve().parents[1] / "results" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
P = "active_learning_"

BOOT = 5000
PERM = 100000
STAT_SEED = 20260822

UCB = "ensemble_ucb:top_methods"
MEAN = "ensemble_mean:top_methods"
RANDOM = "random"
GUIDED = [
    "EVOLVEpro random-forest top-n",
    "ensemble_mean:top_methods",
    "ensemble_ucb:top_methods",
    "diverse_ensemble:top_methods",
    "ensemble_rank_mean_std:top_methods",
    "ensemble_rank_mean_disagreement:top_methods",
]


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, repetitions: int):
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> float:
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(repetitions, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (repetitions + 1))


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    observed = values.to_numpy(float)
    order = np.argsort(observed)
    ranked = observed[order] * len(observed) / np.arange(1, len(observed) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty(len(observed))
    restored[order] = np.minimum(ranked, 1.0)
    return pd.Series(restored, index=values.index)


ENDPOINTS = {
    "top1_final": "PUBLISHED binary: top-1% found at 70 measurements",
    "bestpct_final": "PUBLISHED continuous: best observed percentile at round 5",
    "auc_found": "2a discovery-curve AUC, mean of top-1%-found over rounds 1-5",
    "auc_found_all": "2a discovery-curve AUC incl. baseline, rounds 0-5",
    "auc_bestpct": "2b trajectory-integrated best percentile, mean over rounds 1-5",
    "neg_regret_pct": "2c negative cumulative percentile regret, -sum_{r=1..5}(1-pct_r)",
    "nines_final": "2c/2b log-scale gap to pool top, -log10(max(1-pct_5, 0.5/N))",
    "neglograk_final": "log-scale rank gap, -log10(1 + #pool variants better than best)",
    "auc_neglograk": "trajectory-integrated log rank gap, mean over rounds 1-5",
    "neg_rmtt70": "2d negative restricted mean measurements-to-first-hit, truncated at 70",
}


def build_endpoints() -> pd.DataFrame:
    metrics = pd.read_csv(DATA / f"{P}external_common_pool_metrics.csv")
    metrics = metrics.sort_values(["assay", "seed", "policy", "round"])
    rows = []
    for (assay, seed, policy), block in metrics.groupby(["assay", "seed", "policy"], sort=False):
        block = block.sort_values("round")
        pct = block["best_observed_percentile"].to_numpy(float)
        found = block["top1pct_found"].to_numpy(float)
        pool = float(block["pool_size"].iloc[0])
        n_better = np.rint(pool * (1.0 - pct))
        floor_gap = 0.5 / pool
        not_found_04 = float(np.count_nonzero(found[:5] == 0))
        rows.append(
            {
                "assay": assay,
                "seed": int(seed),
                "policy": policy,
                "source": block["source"].iloc[0],
                "pool_size": pool,
                "found_round0": float(found[0]),
                "top1_final": float(found[5]),
                "bestpct_final": float(pct[5]),
                "auc_found": float(found[1:].mean()),
                "auc_found_all": float(found.mean()),
                "auc_bestpct": float(pct[1:].mean()),
                "neg_regret_pct": float(-np.sum(1.0 - pct[1:])),
                "nines_final": float(-math.log10(max(1.0 - pct[5], floor_gap))),
                "neglograk_final": float(-math.log10(1.0 + n_better[5])),
                "auc_neglograk": float(np.mean(-np.log10(1.0 + n_better[1:]))),
                "neg_rmtt70": float(-(20.0 + 10.0 * not_found_04)),
            }
        )
    return pd.DataFrame(rows)


def assay_deltas(runs: pd.DataFrame, ref: str, comp: str, metric: str) -> np.ndarray:
    pivot = runs[runs["policy"].isin([ref, comp])].pivot_table(
        index=["assay", "seed"], columns="policy", values=metric, aggfunc="first"
    )
    paired = pivot[[ref, comp]].dropna()
    return (paired[ref] - paired[comp]).groupby(level="assay").mean().to_numpy(float)


def contrast_row(runs, ref, comp, metric, rng, extra=None) -> dict:
    d = assay_deltas(runs, ref, comp, metric)
    low, high = bootstrap_mean(d, rng, BOOT)
    row = {
        "reference": ref,
        "comparison": comp,
        "metric": metric,
        "n_assays": len(d),
        "mean_paired_difference": float(d.mean()),
        "sd_paired_difference": float(d.std(ddof=1)),
        "standardized_effect_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan,
        "bootstrap_95ci_low": low,
        "bootstrap_95ci_high": high,
        "two_sided_sign_flip_p": sign_flip(d, rng, PERM),
    }
    if extra:
        row.update(extra)
    return row


def mde_ttest(sd: float, n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    se = sd / math.sqrt(n)

    def pw(delta):
        nc = delta / se
        return (1 - stats.nct.cdf(tcrit, df, nc)) + stats.nct.cdf(-tcrit, df, nc)

    lo, hi = 0.0, 50.0 * se + 1e-12
    while pw(hi) < power:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if pw(mid) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def power_ttest(sd: float, n: int, delta: float, alpha: float = 0.05) -> float:
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    nc = abs(delta) / (sd / math.sqrt(n))
    return float((1 - stats.nct.cdf(tcrit, df, nc)) + stats.nct.cdf(-tcrit, df, nc))


def signflip_power(d: np.ndarray, delta: float, rng, sims: int = 20000, alpha: float = 0.05) -> float:
    n = len(d)
    d0 = d - d.mean()
    idx = rng.integers(0, n, size=(sims, n))
    x = d0[idx] + delta
    denom = np.sqrt((x ** 2).sum(axis=1))
    denom[denom == 0] = np.inf
    z = np.abs(x.sum(axis=1)) / denom
    return float(np.mean(z > stats.norm.ppf(1 - alpha / 2)))


def mde_signflip(d: np.ndarray, rng, target: float = 0.80) -> float:
    lo, hi = 0.0, max(1e-9, 10.0 * d.std(ddof=1))
    while signflip_power(d, hi, np.random.default_rng(1), sims=4000) < target:
        hi *= 2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if signflip_power(d, mid, np.random.default_rng(7), sims=20000) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def n_required(sd: float, delta: float, alpha: float = 0.05, power: float = 0.80) -> float:
    if delta == 0:
        return np.inf
    for n in range(4, 100001):
        if power_ttest(sd, n, delta, alpha) >= power:
            return n
    return np.inf


def variance_components(runs: pd.DataFrame, metric: str, policies: list[str]) -> dict:
    sub = runs[runs["policy"].isin(policies)]
    a = sub["assay"].nunique()
    s = sub["seed"].nunique()
    p = sub["policy"].nunique()
    assert len(sub) == a * s * p, (len(sub), a, s, p)
    y = sub.pivot_table(index=["assay", "seed"], columns="policy", values=metric).sort_index()
    arr = y.to_numpy(float).reshape(a, s, p)
    gm = arr.mean()
    m_a = arr.mean(axis=(1, 2))
    m_as = arr.mean(axis=2)
    m_p = arr.mean(axis=(0, 1))
    m_ap = arr.mean(axis=1)

    ss_a = s * p * ((m_a - gm) ** 2).sum()
    ss_sa = p * ((m_as - m_a[:, None]) ** 2).sum()
    ss_p = a * s * ((m_p - gm) ** 2).sum()
    ss_ap = s * ((m_ap - m_a[:, None] - m_p[None, :] + gm) ** 2).sum()
    resid = arr - m_as[:, :, None] - m_ap[:, None, :] + m_a[:, None, None]
    ss_e = (resid ** 2).sum()

    df_a, df_sa, df_p, df_ap = a - 1, a * (s - 1), p - 1, (a - 1) * (p - 1)
    df_e = a * (s - 1) * (p - 1)
    ms = {
        "assay": ss_a / df_a,
        "seed_in_assay": ss_sa / df_sa,
        "policy": ss_p / df_p,
        "assay_x_policy": ss_ap / df_ap,
        "residual": ss_e / df_e,
    }
    v_e = ms["residual"]
    v_ap = (ms["assay_x_policy"] - v_e) / s
    v_sa = (ms["seed_in_assay"] - v_e) / p
    v_p = (ms["policy"] - ms["assay_x_policy"]) / (a * s)
    v_a = (ms["assay"] - ms["seed_in_assay"] - ms["assay_x_policy"] + v_e) / (s * p)
    comps = {
        "var_assay": max(v_a, 0.0),
        "var_seed_in_assay": max(v_sa, 0.0),
        "var_policy": max(v_p, 0.0),
        "var_assay_x_policy": max(v_ap, 0.0),
        "var_residual_seed_x_policy": max(v_e, 0.0),
    }
    total = sum(comps.values())
    out = {"metric": metric, "n_policies": p, "total_variance": total}
    out.update(comps)
    for k, v in comps.items():
        out["pct_" + k] = 100.0 * v / total if total > 0 else np.nan
    for k, v in comps.items():
        out["sd_" + k.replace("var_", "")] = math.sqrt(v)
    out["policy_mean_range"] = float(m_p.max() - m_p.min())
    out["policy_mean_sd"] = float(m_p.std(ddof=1))
    return out


def main() -> None:
    runs = build_endpoints()
    runs.to_csv(OUT / f"{P}endpoints_per_run.csv", index=False)
    log = {}

    log["n_runs"] = int(len(runs))
    log["n_assays"] = int(runs["assay"].nunique())
    log["n_seeds"] = int(runs["seed"].nunique())
    log["n_policies"] = int(runs["policy"].nunique())
    cells = runs.groupby(["assay", "seed"])["found_round0"].first()
    log["cells_total"] = int(len(cells))
    log["cells_hit_at_baseline"] = int(cells.sum())
    ht = pd.read_csv(DATA / f"{P}external_common_pool_hit_time.csv")
    log["hit_time_rows"] = int(len(ht))
    log["hit_time_censored"] = int(ht["first_top1pct_measurement"].isna().sum())
    chk = runs.merge(ht, on=["assay", "seed", "policy"], how="left")
    chk_obs = chk["first_top1pct_measurement"].fillna(70.0)
    log["rmtt70_matches_hit_time"] = bool(np.allclose(-chk["neg_rmtt70"], chk_obs))

    fin = runs.pivot_table(index=["assay", "seed"], columns="policy", values="top1_final")
    log["cells_all10_identical_binary"] = int((fin.nunique(axis=1) == 1).sum())
    log["cells_guided6_identical_binary"] = int((fin[GUIDED].nunique(axis=1) == 1).sum())

    rng = np.random.default_rng(STAT_SEED)
    repro = pd.DataFrame(
        [contrast_row(runs, UCB, MEAN, m, rng) for m in ["top1_final", "bestpct_final"]]
    )
    repro.to_csv(OUT / f"{P}ucb_mean_comparison.csv", index=False)
    log["ucb_mean_comparison"] = repro[
        ["metric", "mean_paired_difference", "bootstrap_95ci_low", "bootstrap_95ci_high", "two_sided_sign_flip_p"]
    ].to_dict("records")

    summ = (
        runs.groupby(["policy", "assay"])[list(ENDPOINTS)].mean()
        .groupby("policy").mean().reset_index()
    )
    summ.to_csv(OUT / f"{P}endpoint_policy_summary.csv", index=False)

    eff_rows = []
    for m in ENDPOINTS:
        dz = []
        for pol in [p_ for p_ in runs["policy"].unique() if p_ != RANDOM]:
            d = assay_deltas(runs, pol, RANDOM, m)
            sd = d.std(ddof=1)
            dz.append(abs(d.mean()) / sd if sd > 0 else np.nan)
        eff_rows.append({"metric": m, "mean_abs_dz_vs_random": float(np.nanmean(dz)),
                         "description": ENDPOINTS[m]})
    eff = pd.DataFrame(eff_rows).sort_values("mean_abs_dz_vs_random", ascending=False)
    eff.to_csv(OUT / f"{P}endpoint_efficiency.csv", index=False)
    best_metric = str(eff.iloc[0]["metric"])
    best_cont = str(eff[eff["metric"] != "top1_final"].iloc[0]["metric"])
    log["best_endpoint_overall"] = best_metric
    log["best_continuous_endpoint"] = best_cont

    rng = np.random.default_rng(STAT_SEED)
    rows = [contrast_row(runs, UCB, MEAN, m, rng, {"family": "ucb_vs_mean_all_endpoints"})
            for m in ENDPOINTS]
    fam1 = pd.DataFrame(rows)
    fam1["benjamini_hochberg_q"] = benjamini_hochberg(fam1["two_sided_sign_flip_p"])
    fam1.to_csv(OUT / f"{P}uncertainty_contrasts_by_endpoint.csv", index=False)

    rng = np.random.default_rng(STAT_SEED)
    unc_pols = [UCB, "ensemble_rank_mean_std:top_methods",
                "ensemble_rank_mean_disagreement:top_methods", "diverse_ensemble:top_methods"]
    rows = []
    for pol in unc_pols:
        for m in ["top1_final", best_cont, "auc_found", "neg_rmtt70"]:
            rows.append(contrast_row(runs, pol, MEAN, m, rng, {"family": "uncertainty_vs_mean"}))
    fam2 = pd.DataFrame(rows)
    fam2["benjamini_hochberg_q"] = benjamini_hochberg(fam2["two_sided_sign_flip_p"])
    fam2.to_csv(OUT / f"{P}uncertainty_vs_mean_family.csv", index=False)

    rng = np.random.default_rng(STAT_SEED)
    others = [p_ for p_ in summ["policy"] if p_ != RANDOM]
    vr = []
    for m in ["top1_final", best_cont, "auc_found", "neg_rmtt70"]:
        block = [contrast_row(runs, pol, RANDOM, m, rng, {"family": f"vs_random::{m}"}) for pol in others]
        blk = pd.DataFrame(block)
        blk["benjamini_hochberg_q"] = benjamini_hochberg(blk["two_sided_sign_flip_p"])
        vr.append(blk)
    vs_random = pd.concat(vr, ignore_index=True)
    vs_random.to_csv(OUT / f"{P}vs_random_contrasts.csv", index=False)

    prow = []
    rng = np.random.default_rng(STAT_SEED + 1)
    for m in ENDPOINTS:
        d = assay_deltas(runs, UCB, MEAN, m)
        sd = d.std(ddof=1)
        obs = d.mean()
        mde_t = mde_ttest(sd, len(d))
        mde_sf = mde_signflip(d, rng)
        pm = summ.set_index("policy")[m]
        prow.append(
            {
                "metric": m,
                "description": ENDPOINTS[m],
                "n_assays": len(d),
                "observed_effect_ucb_minus_mean": float(obs),
                "sd_paired_difference": float(sd),
                "se_paired_difference": float(sd / math.sqrt(len(d))),
                "mde80_paired_ttest": float(mde_t),
                "mde80_sign_flip": float(mde_sf),
                "mde80_over_observed_effect": float(mde_t / abs(obs)) if obs != 0 else np.nan,
                "power_at_observed_effect": float(power_ttest(sd, len(d), obs)),
                "n_assays_needed_for_observed_effect": float(n_required(sd, obs)),
                "policy_mean_spread_guided6": float(pm[GUIDED].max() - pm[GUIDED].min()),
                "mde80_over_guided6_spread": float(mde_t / (pm[GUIDED].max() - pm[GUIDED].min()))
                if (pm[GUIDED].max() - pm[GUIDED].min()) != 0 else np.nan,
                "random_to_best_gap": float(pm[GUIDED].max() - pm[RANDOM]),
            }
        )
    power = pd.DataFrame(prow)
    power.to_csv(OUT / f"{P}power_table.csv", index=False)

    rngv = np.random.default_rng(99)
    valid = []
    for m in ["top1_final", best_cont]:
        d = assay_deltas(runs, UCB, MEAN, m)
        exact = sign_flip(d, rngv, PERM)
        z = abs(d.sum()) / math.sqrt((d ** 2).sum())
        approx = 2 * (1 - stats.norm.cdf(z))
        valid.append({"metric": m, "exact_sign_flip_p": exact, "normal_approx_p": approx})
    log["signflip_normal_approx_check"] = valid

    vrows = []
    for m in ["top1_final", "bestpct_final", best_cont, "auc_found", "neg_rmtt70"]:
        vrows.append({**variance_components(runs, m, sorted(runs["policy"].unique())), "policy_set": "all10"})
        vrows.append({**variance_components(runs, m, sorted(GUIDED)), "policy_set": "guided6"})
    vc = pd.DataFrame(vrows)
    vc.to_csv(OUT / f"{P}variance_components.csv", index=False)

    rng = np.random.default_rng(STAT_SEED + 2)
    rank_rows = []
    for m in ["top1_final", best_cont]:
        arr = runs.pivot_table(index=["assay", "seed"], columns="policy", values=m).sort_index()
        pols = list(arr.columns)
        a = runs["assay"].nunique()
        s = runs["seed"].nunique()
        cube = arr.to_numpy(float).reshape(a, s, len(pols))
        wins = np.zeros(len(pols))
        ucb_gt_mean = 0
        B = 5000
        iu, im = pols.index(UCB), pols.index(MEAN)
        for _ in range(B):
            pick = rng.integers(0, s, size=(a, s))
            resampled = np.take_along_axis(cube, pick[:, :, None], axis=1).mean(axis=1).mean(axis=0)
            wins[int(np.argmax(resampled))] += 1
            ucb_gt_mean += int(resampled[iu] > resampled[im])
        for i, pol in enumerate(pols):
            rank_rows.append({"metric": m, "policy": pol, "seed_bootstrap_top1_frequency": wins[i] / B})
        log.setdefault("seed_bootstrap_ucb_beats_mean", {})[m] = ucb_gt_mean / B
    pd.DataFrame(rank_rows).to_csv(OUT / f"{P}seed_bootstrap_ranks.csv", index=False)

    seednoise = []
    for m in ["top1_final", best_cont, "auc_found", "neg_rmtt70"]:
        g = runs.groupby(["assay", "policy"])[m].std(ddof=1)
        within_seed_sd = float(np.sqrt((g ** 2).mean()))
        pm = summ.set_index("policy")[m]
        seed_se = within_seed_sd / math.sqrt(5) / math.sqrt(27)
        seednoise.append(
            {
                "metric": m,
                "within_assay_policy_seed_sd": within_seed_sd,
                "seed_only_se_of_policy_mean": seed_se,
                "ucb_minus_mean_gap": float(pm[UCB] - pm[MEAN]),
                "gap_over_seed_only_se": float((pm[UCB] - pm[MEAN]) / seed_se) if seed_se > 0 else np.nan,
                "guided6_spread": float(pm[GUIDED].max() - pm[GUIDED].min()),
            }
        )
    pd.DataFrame(seednoise).to_csv(OUT / f"{P}seed_noise.csv", index=False)

    inf_cells = cells[cells == 0].index
    runs_inf = runs.set_index(["assay", "seed"]).loc[inf_cells].reset_index()
    rng = np.random.default_rng(STAT_SEED + 3)
    crows = []
    for m in ["top1_final", best_cont, "auc_found", "neg_rmtt70"]:
        crows.append(contrast_row(runs_inf, UCB, MEAN, m, rng, {"family": "informative_cells_only"}))
    cond = pd.DataFrame(crows)
    cond["benjamini_hochberg_q"] = benjamini_hochberg(cond["two_sided_sign_flip_p"])
    cond.to_csv(OUT / f"{P}informative_cells_contrasts.csv", index=False)
    log["informative_cells_n"] = int(len(inf_cells))
    log["informative_cells_n_assays"] = int(runs_inf["assay"].nunique())

    iloss = []
    for m in ENDPOINTS:
        d = assay_deltas(runs, UCB, MEAN, m)
        piv = runs[runs["policy"].isin([UCB, MEAN])].pivot_table(
            index=["assay", "seed"], columns="policy", values=m)
        cell_d = (piv[UCB] - piv[MEAN]).to_numpy(float)
        iloss.append(
            {
                "metric": m,
                "assays_with_zero_difference": int(np.count_nonzero(np.isclose(d, 0))),
                "assays_positive": int(np.count_nonzero(d > 1e-12)),
                "assays_negative": int(np.count_nonzero(d < -1e-12)),
                "cells_with_zero_difference": int(np.count_nonzero(np.isclose(cell_d, 0))),
                "cells_total": int(len(cell_d)),
            }
        )
    pd.DataFrame(iloss).to_csv(OUT / f"{P}information_loss.csv", index=False)

    cens = (
        ht.assign(censored=ht["first_top1pct_measurement"].isna())
        .groupby("policy")["censored"].mean().rename("fraction_never_hit_by_70").reset_index()
    )
    cens.to_csv(OUT / f"{P}censoring_by_policy.csv", index=False)

    with open(OUT / f"{P}analysis_metadata.json", "w") as fh:
        json.dump(log, fh, indent=2, default=float)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    print("== INVENTORY ==")
    print(json.dumps({k: v for k, v in log.items() if k not in ("ucb_mean_comparison",)}, indent=2, default=float))
    print("\n== UCB VERSUS ENSEMBLE MEAN ==")
    print(repro[["metric", "mean_paired_difference", "bootstrap_95ci_low", "bootstrap_95ci_high", "two_sided_sign_flip_p"]])
    print("\n== ENDPOINT EFFICIENCY (positive control: any policy vs random) ==")
    print(eff)
    print("\n== POLICY SUMMARY (assay-averaged) ==")
    print(summ.set_index("policy").round(4))
    print("\n== UCB vs MEAN ACROSS ENDPOINTS ==")
    print(fam1[["metric", "mean_paired_difference", "sd_paired_difference", "standardized_effect_dz",
                "bootstrap_95ci_low", "bootstrap_95ci_high", "two_sided_sign_flip_p", "benjamini_hochberg_q"]].round(5))
    print("\n== UNCERTAINTY FAMILY ==")
    print(fam2[["reference", "metric", "mean_paired_difference", "bootstrap_95ci_low", "bootstrap_95ci_high",
                "two_sided_sign_flip_p", "benjamini_hochberg_q"]].round(5))
    print("\n== POWER ==")
    print(power[["metric", "observed_effect_ucb_minus_mean", "sd_paired_difference", "mde80_paired_ttest",
                 "mde80_sign_flip", "mde80_over_observed_effect", "power_at_observed_effect",
                 "n_assays_needed_for_observed_effect", "policy_mean_spread_guided6", "random_to_best_gap"]].round(4))
    print("\n== VARIANCE COMPONENTS ==")
    print(vc[["metric", "policy_set", "pct_var_assay", "pct_var_seed_in_assay", "pct_var_policy",
              "pct_var_assay_x_policy", "pct_var_residual_seed_x_policy", "policy_mean_range",
              "sd_seed_in_assay", "sd_policy"]].round(4))
    print("\n== SEED NOISE ==")
    print(pd.DataFrame(seednoise).round(5))
    print("\n== SEED BOOTSTRAP: P(policy ranks 1st) ==")
    print(pd.DataFrame(rank_rows).pivot(index="policy", columns="metric", values="seed_bootstrap_top1_frequency").round(3))
    print("\n== VS RANDOM ==")
    print(vs_random[["metric", "reference", "mean_paired_difference", "bootstrap_95ci_low", "bootstrap_95ci_high",
                     "two_sided_sign_flip_p", "benjamini_hochberg_q"]].round(5).to_string())
    print("\n== INFORMATION LOSS (ucb vs mean) ==")
    print(pd.DataFrame(iloss).to_string())
    print("\n== CENSORING BY POLICY ==")
    print(cens.round(4).to_string())
    print("\n== INFORMATIVE CELLS ONLY ==")
    print(cond[["metric", "n_assays", "mean_paired_difference", "bootstrap_95ci_low", "bootstrap_95ci_high",
                "two_sided_sign_flip_p", "benjamini_hochberg_q"]].round(5))


if __name__ == "__main__":
    main()
