from __future__ import annotations
import csv
import math
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / 'results' / 'decision_map' / 'assay_decision_features.csv'
BASE = ROOT / 'results' / 'decision_map' / 'assay_strategy_recommendations.csv'
OUT = ROOT / 'results' / 'decision_map'

def f(x: str) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v

def q(vals: list[float], p: float) -> float:
    vals = sorted(vals)
    return vals[round((len(vals) - 1) * p)] if vals else float('nan')

def rec(row: dict[str, str], plm_thr: float, struct_thr: float, epi_high: float, neff_high: float) -> str:
    plm = f(row.get('low_n20_best_plm_gain_over_additive', ''))
    epi = f(row.get('epistasis_mean_abs_epistasis_residual', ''))
    neff = f(row.get('official_MSA_Neff_L', ''))
    prosst = f(row.get('zero_abs_spearman_ProSST-2048', ''))
    site = f(row.get('zero_abs_spearman_Site_Independent', ''))
    n_doubles = f(row.get('n_doubles', '')) or 0
    if plm is not None and plm >= plm_thr:
        return 'calibrate_strong_zero_shot_plm'
    if epi is not None and epi >= epi_high and (n_doubles > 0):
        return 'measure_or_select_double_mutants'
    if prosst is not None and site is not None and (prosst - site >= struct_thr):
        return 'use_structure_aware_scores'
    if neff is not None and neff >= neff_high:
        return 'use_msa_or_evolutionary_models'
    return 'additive_or_simple_ridge_first'

def write(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({k for row in rows for k in row}) if rows else []
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')

def main() -> None:
    rows = list(csv.DictReader(IN.open(newline='', encoding='utf-8')))
    base = {r['assay']: r['recommended_strategy'] for r in csv.DictReader(BASE.open(newline='', encoding='utf-8'))}
    epi_vals = [x for r in rows if (x := f(r.get('epistasis_mean_abs_epistasis_residual', ''))) is not None]
    neff_vals = [x for r in rows if (x := f(r.get('official_MSA_Neff_L', ''))) is not None]
    epi_base = q(epi_vals, 2 / 3)
    neff_base = q(neff_vals, 2 / 3)
    scenarios = []
    for plm_thr in [0.015, 0.02, 0.025]:
        for struct_thr in [0.075, 0.1, 0.125]:
            for epi_mult in [0.8, 1.0, 1.2]:
                for neff_mult in [0.8, 1.0, 1.2]:
                    scenarios.append((plm_thr, struct_thr, epi_base * epi_mult, neff_base * neff_mult))
    detail = []
    stable_counts = Counter()
    for row in rows:
        counts = Counter((rec(row, *s) for s in scenarios))
        modal, n_modal = counts.most_common(1)[0]
        stable_counts[row['assay']] = n_modal / len(scenarios)
        detail.append({'assay': row['assay'], 'base_strategy': base.get(row['assay'], ''), 'modal_strategy': modal, 'stability_fraction': n_modal / len(scenarios), **{f'count_{k}': v for k, v in counts.items()}})
    summary_counts = Counter()
    for r in detail:
        summary_counts[r['base_strategy']] += 1
    summary = []
    for strategy in sorted(summary_counts):
        vals = [float(r['stability_fraction']) for r in detail if r['base_strategy'] == strategy]
        summary.append({'base_strategy': strategy, 'n_assays': len(vals), 'mean_stability_fraction': sum(vals) / len(vals), 'min_stability_fraction': min(vals)})
    write(OUT / 'decision_map_threshold_stability_detail.csv', detail)
    write(OUT / 'decision_map_threshold_stability_summary.csv', summary)
if __name__ == '__main__':
    main()
