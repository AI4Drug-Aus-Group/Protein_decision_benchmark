from __future__ import annotations

from pathlib import Path
import itertools
import os

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(os.environ.get(
    "PROTEIN_DECISION_PACKAGE", ROOT / "analysis_package"
)).resolve()
DATA_ROOT = (
    PACKAGE_ROOT / "multi_mutant"
)
REP_METRICS = os.path.join(
    DATA_ROOT, "single_to_multi", "single_to_multi_representative_metrics.csv"
)
SUMMARY = os.path.join(DATA_ROOT, "single_to_multi", "single_to_multi_summary.csv")
PUBLISHED_RANKING = os.path.join(
    DATA_ROOT, "figure_regime_map", "source_higher_order_ranking.csv"
)

OUT_DIR = str(Path(__file__).resolve().parents[1] / "results" / "experiments")
PREFIX = "higher_order_"

CONTROL = "additive_all_singles"
N_BOOT = 5000
SEED = 20240917
BUCKET = "higher"

LABELS = {
    "MSA_Transformer_ensemble": "MSA Transformer",
    "GEMME": "GEMME",
    "TranceptEVE_L": "TranceptEVE-L",
    "Tranception_L": "Tranception-L",
    "EVmutation": "EVmutation",
    "EVE_ensemble": "EVE",
    "ESM3": "ESM3",
    "ProSST-2048": "ProSST-2048",
    "ESM-IF1": "ESM-IF1",
    "SaProt_650M_AF2": "SaProt",
    "Site_Independent": "Site-Independent",
    "additive_all_singles": "Additive control",
}


def lab(m: str) -> str:
    return LABELS.get(m, m)


def bootstrap_ci(diffs: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED,
                 alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = diffs.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float((boot_means > 0).mean())


def exact_sign_permutation_p(diffs: np.ndarray) -> float:
    n = diffs.size
    if n > 22:
        raise ValueError("too many pairs for exact enumeration")
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    null = signs @ diffs / n
    obs = float(diffs.mean())
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    return p


def exact_sign_test_p(diffs: np.ndarray) -> float:
    from scipy import stats

    pos = int((diffs > 0).sum())
    neg = int((diffs < 0).sum())
    n = pos + neg
    if n == 0:
        return 1.0
    return float(stats.binomtest(pos, n, 0.5, alternative="two-sided").pvalue)


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def paired_row(name_a, name_b, a, b, assays):
    d = a - b
    lo, hi, frac_pos = bootstrap_ci(d)
    return {
        "method_a": name_a,
        "method_b": name_b,
        "label_a": lab(name_a),
        "label_b": lab(name_b),
        "n_assays": int(d.size),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "sd_diff": float(d.std(ddof=1)),
        "boot_ci_lo": lo,
        "boot_ci_hi": hi,
        "boot_frac_positive": frac_pos,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "n_assays_a_wins": int((d > 0).sum()),
        "p_sign_permutation_exact": exact_sign_permutation_p(d),
        "p_sign_test_exact": exact_sign_test_p(d),
        "assays": ";".join(assays),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng_note = []

    rep = pd.read_csv(REP_METRICS)
    summary = pd.read_csv(SUMMARY)
    published = pd.read_csv(PUBLISHED_RANKING)

    hi = rep[(rep["order_bucket"] == BUCKET) & (rep["n"] > 0)].copy()
    assays = sorted(hi["assay"].unique())
    n_assays = len(assays)

    repro = (
        hi.groupby("method")
        .agg(
            n_assays=("assay", "nunique"),
            mean_abs_spearman=("abs_spearman", "mean"),
            mean_top1pct=("top1pct_recall_at_100", "mean"),
            mean_n_variants=("n", "mean"),
        )
        .reset_index()
        .sort_values("mean_abs_spearman", ascending=False)
    )
    pub = published.set_index("method")["mean_abs_spearman"]
    sums = summary[summary["order_bucket"] == BUCKET].set_index("method")
    repro["published_summary_mean"] = repro["method"].map(sums["mean_abs_spearman"])
    repro["published_figure_mean"] = repro["method"].map(pub)
    repro["delta_vs_summary"] = (
        repro["mean_abs_spearman"] - repro["published_summary_mean"]
    )
    repro["delta_vs_figure"] = repro["mean_abs_spearman"] - repro["published_figure_mean"]
    repro["label"] = repro["method"].map(lab)
    repro.to_csv(os.path.join(OUT_DIR, PREFIX + "reproduction.csv"), index=False)

    counts = hi.groupby("method")["assay"].nunique()
    incomplete = counts[counts != n_assays]
    missing_metric = hi[hi["abs_spearman"].isna()]

    ranked = repro[repro["method"] != CONTROL]["method"].tolist()
    top5 = ranked[:5]

    def wide(metric):
        w = hi.pivot(index="assay", columns="method", values=metric)
        return w.loc[assays]

    wide_rho = wide("abs_spearman")
    wide_rec = wide("top1pct_recall_at_100")

    per_assay = wide_rho[top5 + [CONTROL]].copy()
    per_assay.columns = [lab(c) for c in per_assay.columns]
    per_assay.insert(0, "n_higher_order_variants",
                     hi.groupby("assay")["n"].first().loc[assays].astype(int))
    per_assay["share_of_all_higher_variants"] = (
        per_assay["n_higher_order_variants"] / per_assay["n_higher_order_variants"].sum()
    )
    per_assay["best_method_this_assay"] = [
        lab(wide_rho.loc[a, ranked].idxmax()) for a in assays
    ]
    per_assay_rec = wide_rec[top5 + [CONTROL]].copy()
    per_assay_rec.columns = [lab(c) + " (recall)" for c in per_assay_rec.columns]
    per_assay = per_assay.join(per_assay_rec)
    per_assay.to_csv(os.path.join(OUT_DIR, PREFIX + "per_assay.csv"))

    def run_family(wide_df, metric_name):
        pair_rows = []
        for a_name, b_name in itertools.combinations(top5, 2):
            pair_rows.append(
                paired_row(a_name, b_name,
                           wide_df[a_name].to_numpy(float),
                           wide_df[b_name].to_numpy(float), assays)
            )
        pair = pd.DataFrame(pair_rows)
        pair["q_sign_permutation_BH"] = benjamini_hochberg(pair["p_sign_permutation_exact"])
        pair["q_sign_test_BH"] = benjamini_hochberg(pair["p_sign_test_exact"])
        pair["resolved_BH05"] = pair["q_sign_permutation_BH"] < 0.05
        pair["metric"] = metric_name
        pair["family"] = "top5_pairwise"

        ctl_rows = []
        for a_name in top5:
            ctl_rows.append(
                paired_row(a_name, CONTROL,
                           wide_df[a_name].to_numpy(float),
                           wide_df[CONTROL].to_numpy(float), assays)
            )
        ctl = pd.DataFrame(ctl_rows)
        ctl["q_sign_permutation_BH"] = benjamini_hochberg(ctl["p_sign_permutation_exact"])
        ctl["q_sign_test_BH"] = benjamini_hochberg(ctl["p_sign_test_exact"])
        ctl["resolved_BH05"] = ctl["q_sign_permutation_BH"] < 0.05
        ctl["metric"] = metric_name
        ctl["family"] = "vs_additive_control"
        return pair, ctl

    pair_rho, ctl_rho = run_family(wide_rho, "abs_spearman")
    have_recall = wide_rec[top5 + [CONTROL]].notna().all().all()
    if have_recall:
        pair_rec, ctl_rec = run_family(wide_rec, "top1pct_recall_at_100")
        pairs_all = pd.concat([pair_rho, pair_rec], ignore_index=True)
        ctl_all = pd.concat([ctl_rho, ctl_rec], ignore_index=True)
    else:
        pairs_all, ctl_all = pair_rho, ctl_rho

    pairs_all.to_csv(os.path.join(OUT_DIR, PREFIX + "pairwise_tests.csv"), index=False)
    ctl_all.to_csv(os.path.join(OUT_DIR, PREFIX + "vs_additive_tests.csv"), index=False)

    loo_rows = []
    all_methods = ranked + [CONTROL]
    for drop in assays:
        keep = [a for a in assays if a != drop]
        sub = wide_rho.loc[keep, all_methods]
        means = sub.mean()
        top_method = means.drop(CONTROL).idxmax()
        order = means.drop(CONTROL).sort_values(ascending=False)
        loo_rows.append(
            {
                "dropped_assay": drop,
                "dropped_n_variants": int(hi[hi["assay"] == drop]["n"].iloc[0]),
                "top_method": top_method,
                "top_label": lab(top_method),
                "top_mean": float(means[top_method]),
                "second_method": lab(order.index[1]),
                "second_mean": float(order.iloc[1]),
                "gap_top_to_second": float(order.iloc[0] - order.iloc[1]),
                "msa_mean": float(means["MSA_Transformer_ensemble"]),
                "gemme_mean": float(means["GEMME"]),
                "msa_minus_gemme": float(
                    means["MSA_Transformer_ensemble"] - means["GEMME"]
                ),
                "additive_mean": float(means[CONTROL]),
                "top_minus_additive": float(means[top_method] - means[CONTROL]),
                "top_still_beats_additive": bool(means[top_method] > means[CONTROL]),
                "msa_rank": int(order.index.get_loc("MSA_Transformer_ensemble") + 1),
                "gemme_rank": int(order.index.get_loc("GEMME") + 1),
            }
        )
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(os.path.join(OUT_DIR, PREFIX + "leave_one_out.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print("=" * 78)
    print("Reproduction of the reported higher-order values")
    print("=" * 78)
    print(f"assays with >=1 higher-order variant: {n_assays}")
    print(f"methods present in all {n_assays} assays: "
          f"{'YES' if incomplete.empty else 'NO -> ' + str(incomplete.to_dict())}")
    print(f"rows with n>0 but missing abs_spearman: {len(missing_metric)}")
    print(f"total higher-order variants across assays: "
          f"{int(hi.groupby('assay')['n'].first().sum())}")
    print()
    cols = ["label", "n_assays", "mean_abs_spearman", "published_summary_mean",
            "delta_vs_summary", "published_figure_mean", "delta_vs_figure",
            "mean_top1pct"]
    print(repro[cols].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print()
    print(f"max |delta vs published summary| = "
          f"{repro['delta_vs_summary'].abs().max():.2e}")

    print()
    print("=" * 78)
    print("Per-assay values and variant counts")
    print("=" * 78)
    print(per_assay.drop(columns=[c for c in per_assay.columns if "(recall)" in c])
          .to_string(float_format=lambda v: f"{v:.4f}"))
    shares = per_assay["share_of_all_higher_variants"].sort_values(ascending=False)
    print()
    print("variant-count concentration (top 3 assays):")
    print(shares.head(3).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"top 1 assay share = {shares.iloc[0]:.1%}; "
          f"top 3 share = {shares.head(3).sum():.1%}")

    for metric_name, pairs, ctl in [
        ("abs_spearman", pair_rho, ctl_rho),
    ] + ([("top1pct_recall_at_100", pair_rec, ctl_rec)] if have_recall else []):
        print()
        print("=" * 78)
        print(f"Pairwise comparisons among the top five methods  [{metric_name}]")
        print("=" * 78)
        show = ["label_a", "label_b", "mean_diff", "boot_ci_lo", "boot_ci_hi",
                "n_assays_a_wins", "p_sign_permutation_exact",
                "q_sign_permutation_BH", "p_sign_test_exact", "resolved_BH05"]
        print(pairs[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        nres = int(pairs["resolved_BH05"].sum())
        print(f"-> {nres} / {len(pairs)} pairwise orderings resolved at BH q<0.05")
        print(f"-> smallest attainable two-sided exact p with n={n_assays} is "
              f"{2/2**n_assays:.5f}")

        print()
        print("=" * 78)
        print(f"Leading methods against the additive control  [{metric_name}]")
        print("=" * 78)
        print(ctl[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"-> {int(ctl['resolved_BH05'].sum())} / {len(ctl)} methods "
              f"significantly different from the additive control at BH q<0.05")

    print()
    print("=" * 78)
    print("Leave-one-assay-out")
    print("=" * 78)
    loo_show = ["dropped_assay", "dropped_n_variants", "top_label", "top_mean",
                "second_method", "gap_top_to_second", "msa_mean", "gemme_mean",
                "msa_minus_gemme", "additive_mean", "top_minus_additive",
                "top_still_beats_additive", "msa_rank", "gemme_rank"]
    print(loo[loo_show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("top method identity counts across 11 LOO replicates:")
    print(loo["top_label"].value_counts().to_string())
    print(f"MSA Transformer top in {int((loo['top_label']=='MSA Transformer').sum())}"
          f"/{n_assays} replicates; "
          f"GEMME top in {int((loo['top_label']=='GEMME').sum())}/{n_assays}")
    print(f"top_mean range: {loo['top_mean'].min():.4f} - {loo['top_mean'].max():.4f}")
    print(f"MSA mean range: {loo['msa_mean'].min():.4f} - {loo['msa_mean'].max():.4f}")
    print(f"GEMME mean range: {loo['gemme_mean'].min():.4f} - "
          f"{loo['gemme_mean'].max():.4f}")
    print(f"additive mean range: {loo['additive_mean'].min():.4f} - "
          f"{loo['additive_mean'].max():.4f}")
    print(f"top always beats additive: {bool(loo['top_still_beats_additive'].all())}")
    print(f"min margin over additive across LOO: "
          f"{loo['top_minus_additive'].min():.4f}")

    print()
    print("=" * 78)
    print("Bootstrap confidence interval on each reported mean, assay resampling")
    print("=" * 78)
    rows = []
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_assays, size=(N_BOOT, n_assays))
    for m in top5 + [CONTROL]:
        v = wide_rho[m].to_numpy(float)
        bm = v[idx].mean(axis=1)
        lo, higv = np.percentile(bm, [2.5, 97.5])
        rows.append({"method": m, "label": lab(m), "mean": float(v.mean()),
                     "boot_ci_lo": float(lo), "boot_ci_hi": float(higv)})
    stack = np.stack([wide_rho[m].to_numpy(float)[idx].mean(axis=1) for m in top5])
    win = np.bincount(stack.argmax(axis=0), minlength=len(top5)) / N_BOOT
    for r, w in zip(rows, list(win) + [np.nan]):
        r["p_best_of_top5_bootstrap"] = float(w) if w == w else np.nan
    head = pd.DataFrame(rows)
    head.to_csv(os.path.join(OUT_DIR, PREFIX + "headline_bootstrap.csv"), index=False)
    print(head.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print()
    print("=" * 78)
    print("Ranking under the discovery metric, top1pct_recall_at_100")
    print("=" * 78)
    rec_rank = (
        wide_rec.mean().sort_values(ascending=False).rename("mean_recall").to_frame()
    )
    rec_rank["label"] = [lab(m) for m in rec_rank.index]
    rec_rank["rank"] = np.arange(1, len(rec_rank) + 1)
    rec_rank["rank_on_abs_spearman"] = [
        int(repro.reset_index(drop=True).index[repro["method"].tolist().index(m)]) + 1
        for m in rec_rank.index
    ]
    print(rec_rank[["label", "mean_recall", "rank", "rank_on_abs_spearman"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    rec_rank.to_csv(os.path.join(OUT_DIR, PREFIX + "recall_ranking.csv"))
    ctl_rank = int(rec_rank.loc[CONTROL, "rank"])
    print(f"-> additive control ranks {ctl_rank} / {len(rec_rank)} on the "
          f"discovery metric (it ranks 13 / 19 on abs_spearman)")

    print()
    print("=" * 78)
    print("Weighting sensitivity, assay means are unweighted")
    print("=" * 78)
    w = hi.groupby("assay")["n"].first().loc[assays].to_numpy(float)
    wrows = []
    for m in top5 + [CONTROL]:
        v = wide_rho[m].to_numpy(float)
        wrows.append({"label": lab(m), "unweighted_mean": float(v.mean()),
                      "variant_weighted_mean": float((v * w).sum() / w.sum())})
    wdf = pd.DataFrame(wrows).sort_values("variant_weighted_mean", ascending=False)
    wdf.to_csv(os.path.join(OUT_DIR, PREFIX + "weighting_sensitivity.csv"),
               index=False)
    print(wdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("additive control per-assay abs_spearman spread: "
          f"min={wide_rho[CONTROL].min():.4f} max={wide_rho[CONTROL].max():.4f} "
          f"sd={wide_rho[CONTROL].std(ddof=1):.4f}  (top-5 methods sd ~"
          f"{wide_rho[top5].std(ddof=1).mean():.4f})")

    print()
    print("wrote:", ", ".join(sorted(
        f for f in os.listdir(OUT_DIR) if f.startswith(PREFIX))))


if __name__ == "__main__":
    main()
