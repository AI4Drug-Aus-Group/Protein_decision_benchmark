from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / 'results' / 'decision_map' / 'assay_decision_features.csv'
OUT_DIR = ROOT / 'results' / 'decision_map'

def parse_float(value: str) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float('nan')

def quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float('nan')
    idx = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[idx]

def bin_value(x: float | None, low: float, high: float, labels: tuple[str, str, str]) -> str:
    if x is None:
        return 'missing'
    if x <= low:
        return labels[0]
    if x >= high:
        return labels[2]
    return labels[1]

def strategy(row: dict[str, str], epi_low: float, epi_high: float, neff_low: float, neff_high: float) -> str:
    plm_gain = parse_float(row.get('low_n20_best_plm_gain_over_additive', ''))
    epi = parse_float(row.get('epistasis_mean_abs_epistasis_residual', ''))
    neff_l = parse_float(row.get('official_MSA_Neff_L', ''))
    n_doubles = parse_float(row.get('n_doubles', ''))
    n_higher = parse_float(row.get('n_higher_order', ''))
    struct_gain = None
    prosst = parse_float(row.get('zero_abs_spearman_ProSST-2048', ''))
    site = parse_float(row.get('zero_abs_spearman_Site_Independent', ''))
    if prosst is not None and site is not None:
        struct_gain = prosst - site
    if plm_gain is not None and plm_gain >= 0.02:
        return 'calibrate_strong_zero_shot_plm'
    if epi is not None and epi >= epi_high and ((n_doubles or 0) > 0):
        return 'measure_or_select_double_mutants'
    if struct_gain is not None and struct_gain >= 0.1:
        return 'use_structure_aware_scores'
    if neff_l is not None and neff_l >= neff_high:
        return 'use_msa_or_evolutionary_models'
    if (n_higher or 0) > 0 and epi is not None and (epi >= epi_low):
        return 'validate_higher_order_with_epistasis_controls'
    return 'additive_or_simple_ridge_first'

def summarize(rows: list[dict[str, str]], group_field: str, out_name: str) -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(group_field, '') or 'missing'].append(row)
    out = []
    for group, rs in sorted(groups.items()):
        plm = [x for r in rs if (x := parse_float(r.get('low_n20_best_plm_gain_over_additive', ''))) is not None]
        active = [x for r in rs if (x := parse_float(r.get('active_gain_diverse_ensemble_vs_random', ''))) is not None]
        epi = [x for r in rs if (x := parse_float(r.get('epistasis_mean_abs_epistasis_residual', ''))) is not None]
        struct = []
        for r in rs:
            a = parse_float(r.get('zero_abs_spearman_ProSST-2048', ''))
            b = parse_float(r.get('zero_abs_spearman_Site_Independent', ''))
            if a is not None and b is not None:
                struct.append(a - b)
        out.append({'axis': group_field, 'bin': group, 'n_assays': len(rs), 'mean_low_n_plm_gain_over_additive': mean(plm), 'mean_active_gain_diverse_ensemble_vs_random': mean(active), 'mean_epistasis_abs_residual': mean(epi), 'mean_prosst_gain_over_site_independent': mean(struct), 'recommended_default': max(defaultdict(int, {s: sum((1 for r in rs if r['recommended_strategy'] == s)) for s in {r['recommended_strategy'] for r in rs}}), key=lambda s: sum((1 for r in rs if r['recommended_strategy'] == s)))})
    write_csv(OUT_DIR / out_name, out)

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) if rows else []
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')

def main() -> None:
    with IN.open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    epi_vals = [x for r in rows if (x := parse_float(r.get('epistasis_mean_abs_epistasis_residual', ''))) is not None]
    neff_vals = [x for r in rows if (x := parse_float(r.get('official_MSA_Neff_L', ''))) is not None]
    epi_low, epi_high = (quantile(epi_vals, 1 / 3), quantile(epi_vals, 2 / 3))
    neff_low, neff_high = (quantile(neff_vals, 1 / 3), quantile(neff_vals, 2 / 3))
    recs = []
    for row in rows:
        rec = {'assay': row['assay'], 'protein_key': row.get('protein_key', ''), 'assay_type': row.get('assay_type', ''), 'best_zero_shot_method': row.get('best_zero_shot_method', ''), 'recommended_strategy': strategy(row, epi_low, epi_high, neff_low, neff_high), 'epistasis_bin': bin_value(parse_float(row.get('epistasis_mean_abs_epistasis_residual', '')), epi_low, epi_high, ('low', 'mid', 'high')), 'official_msa_neff_l_bin': bin_value(parse_float(row.get('official_MSA_Neff_L', '')), neff_low, neff_high, ('low', 'mid', 'high')), 'low_n20_best_plm_gain_over_additive': row.get('low_n20_best_plm_gain_over_additive', ''), 'active_gain_diverse_ensemble_vs_random': row.get('active_gain_diverse_ensemble_vs_random', ''), 'zero_abs_spearman_ProSST-2048': row.get('zero_abs_spearman_ProSST-2048', ''), 'zero_abs_spearman_Site_Independent': row.get('zero_abs_spearman_Site_Independent', '')}
        recs.append(rec)
    write_csv(OUT_DIR / 'assay_strategy_recommendations.csv', recs)
    summarize(recs, 'recommended_strategy', 'decision_map_by_recommended_strategy.csv')
    summarize(recs, 'epistasis_bin', 'decision_map_rules_by_epistasis_bin.csv')
    summarize(recs, 'official_msa_neff_l_bin', 'decision_map_rules_by_msa_bin.csv')
    summarize(recs, 'assay_type', 'decision_map_rules_by_assay_type.csv')
if __name__ == '__main__':
    main()
