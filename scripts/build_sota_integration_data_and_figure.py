from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sota_integrated_manuscript"
DATA = OUT / "data"
FIG = OUT / "figures"
SEED = 20260902
RNG = np.random.default_rng(SEED)

LABELS = {
    "ProteinNPT": "ProteinNPT",
    "linear calibration of ProSST-2048": "Calibrated ProSST-2048",
    "ridge on multiple zero-shot scores": "Multi-score ridge",
    "ridge on mutation order, single-mutant sum and zero-shot scores": "Combined low-data ridge",
    "additive_versus_calibrated_prosst_disagreement": "Additive-calibrated disagreement",
    "extreme_additive_prediction": "Extreme additive prediction",
    "mutation_coverage": "Mutation-site coverage",
    "oracle_absolute_epistatic_residual": "Largest residual (retrospective)",
    "random_doubles": "Random selection",
}
COLORS = {
    "ProteinNPT": "#D98C96",
    "Kermut": "#77A88D",
    "Calibrated ProSST-2048": "#7298B8",
    "Multi-score ridge": "#D6A65C",
    "Combined low-data ridge": "#9A86B8",
    "Random selection": "#8796A3",
    "Mutation-site coverage": "#75A98B",
    "Extreme additive prediction": "#D9AF69",
    "Additive-calibrated disagreement": "#D78C96",
    "Largest residual (retrospective)": "#9A88B5",
}

def bootstrap_ci(values, n=5000):
    x = np.asarray(values, float)
    draw = RNG.integers(0, len(x), size=(n, len(x)))
    means = x[draw].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])

def sign_flip_p(values, n=100000):
    x = np.asarray(values, float)
    observed = abs(x.mean())
    if len(x) <= 20:
        signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(x))))
        null = np.abs((signs * x).mean(axis=1))
        return (np.sum(null >= observed - 1e-15) + 1) / (len(null) + 1)
    exceed = 0
    done = 0
    while done < n:
        batch = min(10000, n - done)
        signs = RNG.choice([-1.0, 1.0], size=(batch, len(x)))
        exceed += int(np.sum(np.abs((signs * x).mean(axis=1)) >= observed - 1e-15))
        done += batch
    return (exceed + 1) / (n + 1)

def bh(values):
    p = np.asarray(values, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(len(p))
    q[order] = np.minimum(ranked, 1.0)
    return q

def common_pool_tables():
    path = ROOT / "results/sota_extension/strict_pnpt_common_pool/strict_pnpt_common_pool_detail.csv"
    detail = pd.read_csv(path)
    assay = detail.groupby(["assay", "budget", "method"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        mean_spearman=("spearman_all", "mean"),
        mean_top1pct_recall=("top1pct_recall_at_100", "mean"),
        mean_best_top100_percentile=("best_top100_percentile", "mean"),
    )
    summary = []
    for (budget, method), frame in assay.groupby(["budget", "method"]):
        values = frame["mean_spearman"].dropna().to_numpy()
        lo, hi = bootstrap_ci(values)
        summary.append({
            "analysis_type": "Strict common-candidate comparison",
            "method": LABELS.get(method, method),
            "budget": int(budget),
            "n_assays": int(frame["assay"].nunique()),
            "n_assays_spearman": int(len(values)),
            "n_assays_with_five_seeds": int((frame["n_seeds"] == 5).sum()),
            "mean_spearman": float(np.mean(values)),
            "bootstrap_95ci_low": float(lo),
            "bootstrap_95ci_high": float(hi),
            "mean_top1pct_recall_at_100": float(frame["mean_top1pct_recall"].mean()),
            "mean_best_top100_percentile": float(frame["mean_best_top100_percentile"].mean()),
        })
    tests = []
    controls = [
        "linear calibration of ProSST-2048",
        "ridge on multiple zero-shot scores",
        "ridge on mutation order, single-mutant sum and zero-shot scores",
    ]
    for budget in [20, 50, 100]:
        wide = assay[assay["budget"].eq(budget)].pivot(index="assay", columns="method", values="mean_spearman")
        for control in controls:
            paired = wide[["ProteinNPT", control]].dropna()
            delta = paired["ProteinNPT"] - paired[control]
            lo, hi = bootstrap_ci(delta)
            tests.append({
                "analysis_type": "Strict common-candidate paired comparison",
                "method_a": "ProteinNPT",
                "method_b": LABELS[control],
                "budget": budget,
                "n_assays": int(len(delta)),
                "mean_a": float(paired["ProteinNPT"].mean()),
                "mean_b": float(paired[control].mean()),
                "mean_difference_a_minus_b": float(delta.mean()),
                "bootstrap_95ci_low": float(lo),
                "bootstrap_95ci_high": float(hi),
                "two_sided_sign_permutation_p": float(sign_flip_p(delta)),
            })
    test_frame = pd.DataFrame(tests)
    test_frame["benjamini_hochberg_q"] = bh(test_frame["two_sided_sign_permutation_p"])
    return assay, pd.DataFrame(summary), test_frame

def single_to_multi_table():
    frame = pd.read_csv(ROOT / "results/derived_summaries/external_method_summary.csv")
    frame = frame[frame["evaluable"]].copy()
    frame["analysis_type"] = "Single-mutant training evaluated on multi-mutants"
    frame["method"] = frame["method"].map(LABELS).fillna(frame["method"])
    return frame[["analysis_type", "method", "budget", "order_bucket", "n_assays", "n_assays_with_five_seeds", "mean_n_variants", "mean_spearman", "mean_abs_spearman", "mean_top1pct_recall_at_100", "mean_best_top100_percentile", "analysis_scope", "direct_cross_method_head_to_head", "scope_note"]]

def selected_double_tables():
    sources = {
        "Kermut": ROOT / "results/sota_extension/selected_double20_kermut_resource_compatible_all51_with_baseline/selected_double_external_detail.csv",
        "ProteinNPT": ROOT / "results/sota_extension/selected_double20_expanded4_seed5_proteinnpt_with_baseline/selected_double_external_detail.csv",
    }
    assay_blocks = []
    summaries = []
    tests = []
    for method, path in sources.items():
        detail = pd.read_csv(path)
        assay = detail.groupby(["assay", "selection_policy", "selection_policy_label"], as_index=False).agg(
            n_seeds=("seed", "nunique"),
            mean_spearman=("spearman", "mean"),
            mean_gain_vs_single20_spearman=("gain_vs_single20_spearman", "mean"),
            mean_holdout_coverage=("holdout_coverage", "mean"),
        )
        assay["method"] = method
        assay_blocks.append(assay)
        for policy, frame in assay.groupby("selection_policy"):
            values = frame["mean_gain_vs_single20_spearman"].to_numpy()
            lo, hi = bootstrap_ci(values)
            summaries.append({
                "analysis_type": "External model with 20 single-mutant and 20 double-mutant measurements",
                "method": method,
                "selection_policy": LABELS[policy],
                "selected_double_mutant_budget": 20,
                "n_assays": int(len(frame)),
                "n_assays_with_five_seeds": int((frame["n_seeds"] == 5).sum()),
                "mean_holdout_coverage": float(frame["mean_holdout_coverage"].mean()),
                "mean_spearman": float(frame["mean_spearman"].mean()),
                "mean_gain_vs_single20_spearman": float(values.mean()),
                "bootstrap_95ci_low": float(lo),
                "bootstrap_95ci_high": float(hi),
            })
        wide = assay.pivot(index="assay", columns="selection_policy", values="mean_gain_vs_single20_spearman")
        for policy in [x for x in wide.columns if x != "random_doubles"]:
            paired = wide[[policy, "random_doubles"]].dropna()
            delta = paired[policy] - paired["random_doubles"]
            lo, hi = bootstrap_ci(delta)
            tests.append({
                "analysis_type": "External double-mutant selection compared with random selection",
                "method": method,
                "selection_policy": LABELS[policy],
                "reference": "Random selection",
                "n_assays": int(len(delta)),
                "mean_paired_difference": float(delta.mean()),
                "bootstrap_95ci_low": float(lo),
                "bootstrap_95ci_high": float(hi),
                "two_sided_sign_permutation_p": float(sign_flip_p(delta)),
            })
    test_frame = pd.DataFrame(tests)
    test_frame["benjamini_hochberg_q"] = bh(test_frame["two_sided_sign_permutation_p"])
    return pd.concat(assay_blocks, ignore_index=True), pd.DataFrame(summaries), test_frame

def save_tables():
    common_pool_assay, common_pool_summary, common_pool_tests = common_pool_tables()
    single_to_multi = single_to_multi_table()
    selected_double_assay, selected_double_summary, selected_double_tests = selected_double_tables()
    common_pool_assay.to_csv(DATA / "strict_common_pool_assay_means.csv", index=False)
    common_pool_summary.to_csv(DATA / "strict_common_pool_summary.csv", index=False)
    common_pool_tests.to_csv(DATA / "strict_common_pool_paired_tests.csv", index=False)
    single_to_multi.to_csv(DATA / "external_single_to_multi_summary.csv", index=False)
    selected_double_assay.to_csv(DATA / "external_selected_double_assay_means.csv", index=False)
    selected_double_summary.to_csv(DATA / "external_selected_double_summary.csv", index=False)
    selected_double_tests.to_csv(DATA / "external_selected_double_paired_tests.csv", index=False)
    return common_pool_assay, common_pool_summary, common_pool_tests, single_to_multi, selected_double_assay, selected_double_summary, selected_double_tests

def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA4AA")
    ax.spines["bottom"].set_color("#9AA4AA")
    ax.tick_params(colors="#3C454B", width=0.7, length=3)
    ax.grid(axis="y", color="#E8ECEE", linewidth=0.7, zorder=0)

def build_figure(common_pool_summary, single_to_multi, selected_double_assay, selected_double_summary, selected_double_tests):
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.2, 7.6), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.85, 1.35], hspace=0.72, wspace=0.62)
    axa = fig.add_subplot(gs[0, :])
    methods = ["ProteinNPT", "Calibrated ProSST-2048", "Multi-score ridge", "Combined low-data ridge"]
    for method in methods:
        f = common_pool_summary[common_pool_summary["method"].eq(method)].sort_values("budget")
        axa.plot(f["budget"], f["mean_spearman"], marker="o", ms=5.2, lw=1.7, color=COLORS[method], label=method, zorder=3)
        axa.fill_between(f["budget"], f["bootstrap_95ci_low"], f["bootstrap_95ci_high"], color=COLORS[method], alpha=0.14, linewidth=0)
    axa.set_xticks([20, 50, 100])
    axa.set_xlabel("Measured variants")
    axa.set_ylabel("Mean held-out Spearman correlation")
    axa.set_ylim(0.25, 0.63)
    axa.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.02), handlelength=2.2, columnspacing=1.7)
    style_axis(axa)
    axa.text(-0.075, 1.06, "a", transform=axa.transAxes, fontsize=11, fontweight="bold")

    axb = fig.add_subplot(gs[1, :])
    show = single_to_multi[single_to_multi["budget"].eq(20)].copy()
    orders = ["double", "higher"]
    x = np.arange(len(orders))
    offsets = {"Kermut": -0.14, "ProteinNPT": 0.14}
    for method in ["Kermut", "ProteinNPT"]:
        f = show[show["method"].eq(method)].set_index("order_bucket")
        vals = [f.loc[o, "mean_spearman"] if o in f.index else np.nan for o in orders]
        ns = [int(f.loc[o, "n_assays"]) if o in f.index else 0 for o in orders]
        axb.scatter(x + offsets[method], vals, s=78, color=COLORS[method], edgecolor="white", linewidth=0.8, label=method, zorder=4)
        for xi, yi, n in zip(x + offsets[method], vals, ns):
            if np.isfinite(yi):
                axb.vlines(xi, 0, yi, color=COLORS[method], alpha=0.5, lw=2.0, zorder=2)
                axb.text(xi, yi + 0.025, f"n={n}", ha="center", va="bottom", fontsize=7, color="#4A5358")
    axb.set_xticks(x, ["Double mutants", "Variants with ≥3 substitutions"])
    axb.set_ylabel("Mean Spearman correlation")
    axb.set_ylim(0, 0.43)
    axb.legend(frameon=False, loc="upper right", ncol=2)
    style_axis(axb)
    axb.text(-0.075, 1.06, "b", transform=axb.transAxes, fontsize=11, fontweight="bold")

    policy_order = ["Random selection", "Mutation-site coverage", "Extreme additive prediction", "Additive-calibrated disagreement", "Largest residual (retrospective)"]
    for ax, method, letter in [(fig.add_subplot(gs[2, 0]), "Kermut", "c"), (fig.add_subplot(gs[2, 1]), "ProteinNPT", "d")]:
        f = selected_double_summary[selected_double_summary["method"].eq(method)].set_index("selection_policy").loc[policy_order].reset_index()
        y = np.arange(len(f))[::-1]
        ax.axvline(0, color="#7C878D", lw=0.9, ls="--", zorder=1)
        for yi, row in zip(y, f.itertuples()):
            color = COLORS[row.selection_policy]
            ax.hlines(yi, row.bootstrap_95ci_low, row.bootstrap_95ci_high, color=color, lw=2.2, zorder=2)
            ax.scatter(row.mean_gain_vs_single20_spearman, yi, s=60, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        if method == "ProteinNPT":
            aa = selected_double_assay[selected_double_assay["method"].eq(method)].copy()
            aa["label"] = aa["selection_policy"].map(LABELS)
            for yi, policy in zip(y, policy_order):
                vals = aa.loc[aa["label"].eq(policy), "mean_gain_vs_single20_spearman"].to_numpy()
                jitter = np.linspace(-0.10, 0.10, len(vals))
                ax.scatter(vals, yi + jitter, s=13, facecolor="white", edgecolor=COLORS[policy], linewidth=0.7, zorder=4)
        ax.set_yticks(y, policy_order)
        ax.set_xlabel("Gain over 20-single-mutant baseline\nin held-out Spearman correlation")
        ax.set_title(f"{method}  (n={int(f['n_assays'].iloc[0])} assays)", loc="left", fontsize=9, pad=8)
        ax.set_xlim(-0.28, 0.46)
        style_axis(ax)
        ax.grid(axis="x", color="#E8ECEE", linewidth=0.7, zorder=0)
        ax.grid(axis="y", visible=False)
        ax.text(-0.16, 1.08, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
    cax, dax = fig.axes[-2], fig.axes[-1]
    cax.text(0.44, 0.05, "Coverage vs random: q=0.78\nOther targeted rules: q<0.001", transform=cax.transAxes, ha="right", va="bottom", fontsize=6.8, color="#5D666B")
    dax.text(0.96, 0.05, "All comparisons with random: q≥0.35", transform=dax.transAxes, ha="right", va="bottom", fontsize=6.8, color="#5D666B")
    fig.subplots_adjust(left=0.19, right=0.98, top=0.98, bottom=0.08)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"Supplementary_Figure_S10.{ext}", dpi=450 if ext == "png" else None, facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    tables = save_tables()
    build_figure(tables[1], tables[3], tables[4], tables[5], tables[6])
    print(OUT)
