from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List
ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / 'results' / 'decision_map' / 'assay_decision_features.csv'
OUT_DIR = ROOT / 'results' / 'decision_map'

def parse_float(value: str) -> float | None:
    if value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None

def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float('nan')

def quantile_bins(rows: List[Dict[str, str]], field: str, labels: List[str]) -> Dict[str, str]:
    vals = sorted(((parse_float(r.get(field, '')), r['assay']) for r in rows if parse_float(r.get(field, '')) is not None))
    if not vals:
        return {}
    out = {}
    n = len(vals)
    for i, (_, assay) in enumerate(vals):
        out[assay] = labels[min(len(labels) - 1, int(i * len(labels) / n))]
    return out

def summarize(rows: List[Dict[str, str]], group_field: str, value_fields: List[str], out_name: str) -> None:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(group_field, '')].append(row)
    out = []
    for group, recs in groups.items():
        item = {'group_field': group_field, 'group': group, 'n_assays': len(recs)}
        for field in value_fields:
            vals = [parse_float(r.get(field, '')) for r in recs]
            vals = [v for v in vals if v is not None]
            item[f'mean_{field}'] = mean(vals)
        out.append(item)
    out.sort(key=lambda r: r['group'])
    path = OUT_DIR / out_name
    fields = ['group_field', 'group', 'n_assays'] + [f'mean_{f}' for f in value_fields]
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    print(f'wrote {path}')

def main() -> None:
    with DECISION.open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    msa_bins = quantile_bins(rows, 'msa_msa_n_sequences', ['msa_q1_low', 'msa_q2', 'msa_q3', 'msa_q4_high'])
    official_neff_bins = quantile_bins(rows, 'official_MSA_N_eff', ['neff_q1_low', 'neff_q2', 'neff_q3', 'neff_q4_high'])
    official_neff_l_bins = quantile_bins(rows, 'official_MSA_Neff_L', ['neff_l_q1_low', 'neff_l_q2', 'neff_l_q3', 'neff_l_q4_high'])
    epi_bins = quantile_bins(rows, 'epistasis_mean_abs_epistasis_residual', ['epi_q1_low', 'epi_q2', 'epi_q3', 'epi_q4_high'])
    for row in rows:
        row['msa_depth_bin'] = msa_bins.get(row['assay'], '')
        row['official_msa_neff_bin'] = official_neff_bins.get(row['assay'], '')
        row['official_msa_neff_l_bin'] = official_neff_l_bins.get(row['assay'], '')
        row['epistasis_strength_bin'] = epi_bins.get(row['assay'], '')
        s3f = parse_float(row.get('zero_abs_spearman_S3F_MSA', ''))
        esm3 = parse_float(row.get('zero_abs_spearman_ESM3', ''))
        if s3f is not None and esm3 is not None:
            row['s3f_msa_gain_over_esm3'] = str(s3f - esm3)
        prosst = parse_float(row.get('zero_abs_spearman_ProSST-2048', ''))
        site = parse_float(row.get('zero_abs_spearman_Site_Independent', ''))
        if prosst is not None and site is not None:
            row['prosst_gain_over_site_independent'] = str(prosst - site)
    enriched = OUT_DIR / 'assay_decision_features_with_bins.csv'
    fields = list(rows[0].keys())
    with enriched.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {enriched}')
    values = ['low_n20_best_plm_gain_over_additive', 'active_gain_diverse_ensemble_vs_random', 's3f_msa_gain_over_esm3', 'prosst_gain_over_site_independent', 'zero_abs_spearman_ESM3', 'zero_abs_spearman_ProSST-2048', 'zero_abs_spearman_S3F_MSA']
    summarize(rows, 'assay_type', values, 'decision_map_by_assay_type.csv')
    summarize(rows, 'msa_depth_bin', values, 'decision_map_by_msa_depth.csv')
    summarize(rows, 'official_msa_neff_bin', values, 'decision_map_by_official_msa_neff.csv')
    summarize(rows, 'official_msa_neff_l_bin', values, 'decision_map_by_official_msa_neff_l.csv')
    summarize(rows, 'epistasis_strength_bin', values, 'decision_map_by_epistasis_strength.csv')
if __name__ == '__main__':
    main()
