from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'decision_map' / 'assay_decision_features.csv'

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

def read_manifest() -> Dict[str, Dict[str, str]]:
    with (ROOT / 'results' / 'batch0' / 'assay_manifest.csv').open(newline='', encoding='utf-8') as fh:
        return {r['assay']: r for r in csv.DictReader(fh)}

def read_assay_types() -> Dict[str, str]:
    official = ROOT / 'results' / 'batch0' / 'assay_type_curation_official.csv'
    refined = ROOT / 'results' / 'batch0' / 'assay_type_curation_refined.csv'
    path = official if official.exists() else refined if refined.exists() else ROOT / 'results' / 'batch0' / 'assay_type_curation.csv'
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as fh:
        return {r['assay']: r.get('assay_type_official', r.get('assay_type_refined', r.get('assay_type', r.get('assay_type_curated', 'uncurated')))) for r in csv.DictReader(fh)}

def read_msa_depth() -> Dict[str, Dict[str, str]]:
    path = ROOT / 'results' / 'batch0' / 'msa_depth_summary.csv'
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as fh:
        return {r['assay']: r for r in csv.DictReader(fh)}

def read_official_metadata() -> Dict[str, Dict[str, str]]:
    path = ROOT / 'results' / 'batch0' / 'official_assay_metadata.csv'
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as fh:
        return {r['assay']: r for r in csv.DictReader(fh)}

def zero_shot_features() -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    methods = {'ESM3', 'ProSST-2048', 'S3F_MSA', 'S2F_MSA', 'GEMME', 'MSA_Transformer_ensemble', 'VenusREM', 'Site_Independent'}
    per_assay: Dict[str, Dict[str, float]] = defaultdict(dict)
    best_method: Dict[str, Tuple[str, float]] = {}
    with (ROOT / 'results' / 'zero_shot' / 'zero_shot_all_methods_metrics.csv').open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rho = parse_float(row.get('abs_spearman', ''))
            if rho is None:
                continue
            assay = row['assay']
            method = row['method']
            if method in methods:
                per_assay[assay][f'zero_abs_spearman_{method}'] = rho
            current = best_method.get(assay)
            if current is None or rho > current[1]:
                best_method[assay] = (method, rho)
    return (per_assay, {a: m for a, (m, _) in best_method.items()})

def epistasis_strength() -> Dict[str, Dict[str, float]]:
    path = ROOT / 'results' / 'epistasis' / 'epistasis_assay_summary.csv'
    out: Dict[str, Dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rec = {}
            for key, value in row.items():
                parsed = parse_float(value)
                if parsed is not None:
                    rec[f'epistasis_{key}'] = parsed
            out[row['assay']] = rec
    return out

def active_learning_gain() -> Dict[str, Dict[str, float]]:
    path = ROOT / 'results' / 'active_learning' / 'active_learning_representative_metrics.csv'
    values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row['round'] != '5':
                continue
            val = parse_float(row.get('top1pct_found', ''))
            if val is not None:
                values[row['assay'], row['policy']].append(val)
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (assay, policy), vals in values.items():
        out[assay][f'active_round5_top1_{policy}'] = mean(vals)
    for assay, rec in out.items():
        if 'active_round5_top1_diverse_ensemble:top_methods' in rec and 'active_round5_top1_random' in rec:
            rec['active_gain_diverse_ensemble_vs_random'] = rec['active_round5_top1_diverse_ensemble:top_methods'] - rec['active_round5_top1_random']
    return out

def low_n_gain() -> Dict[str, Dict[str, object]]:
    sources = [(ROOT / 'results' / 'low_n' / 'low_n_calibration_top20_nongiant_metrics.percentile.csv', 'full candidate pool'), (ROOT / 'results' / 'low_n' / 'low_n_calibration_top20_giant_fast_metrics.percentile.csv', 'deterministic 20,000-candidate sample')]
    values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    evaluation_mode: Dict[str, str] = {}
    for path, mode in sources:
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                if row['budget'] != '20' or row['regime'] != 'mixed_random':
                    continue
                if row['model'] not in {'additive_lookup', 'zscore_linear'}:
                    continue
                assay = row['assay']
                previous_mode = evaluation_mode.setdefault(assay, mode)
                if previous_mode != mode:
                    raise ValueError(f'{assay} occurs in low-N sources with incompatible evaluation modes')
                method = 'additive_lookup' if row['model'] == 'additive_lookup' else row['zero_shot_method']
                val = parse_float(row.get('best_true_in_pred_top_100_percentile', ''))
                if val is not None:
                    values[assay, method].append(val)
    out: Dict[str, Dict[str, object]] = defaultdict(dict)
    for (assay, method), vals in values.items():
        out[assay][f'low_n20_mixed_best_pct_{method}'] = mean(vals)
    for assay, rec in out.items():
        rec['low_n20_evaluation_mode'] = evaluation_mode[assay]
        additive = rec.get('low_n20_mixed_best_pct_additive_lookup')
        best_plm = max((v for k, v in rec.items() if k.startswith('low_n20_mixed_best_pct_') and k != 'low_n20_mixed_best_pct_additive_lookup'), default=None)
        if additive is not None and best_plm is not None:
            rec['low_n20_best_plm_gain_over_additive'] = best_plm - additive
    return out

def main() -> None:
    manifest = read_manifest()
    assay_types = read_assay_types()
    msa = read_msa_depth()
    official = read_official_metadata()
    zero, best_zero = zero_shot_features()
    epi = epistasis_strength()
    active = active_learning_gain()
    low_n = low_n_gain()
    rows: List[Dict[str, object]] = []
    for assay, row in manifest.items():
        rec: Dict[str, object] = {'assay': assay, 'protein_key': row['protein_key'], 'assay_type': assay_types.get(assay, 'uncurated'), 'n_variants': row['n_variants'], 'n_singles': row['n_singles'], 'n_doubles': row['n_doubles'], 'n_higher_order': row['n_higher_order'], 'strict_double_evaluable': row['strict_double_evaluable'], 'strict_higher_evaluable': row['strict_higher_evaluable'], 'best_zero_shot_method': best_zero.get(assay, '')}
        if assay in msa:
            rec.update({f'msa_{k}': v for k, v in msa[assay].items() if k not in {'assay', 'protein_key'}})
        if assay in official:
            for key in ['MSA_N_eff', 'MSA_Neff_L', 'MSA_Neff_L_category', 'MSA_num_seqs', 'MSA_perc_cov', 'MSA_num_significant', 'selection_assay', 'selection_type', 'coarse_selection_type']:
                if key in official[assay]:
                    rec[f'official_{key}'] = official[assay][key]
        rec.update(zero.get(assay, {}))
        rec.update(epi.get(assay, {}))
        rec.update(active.get(assay, {}))
        rec.update(low_n.get(assay, {}))
        rows.append(rec)
    fields = sorted({k for row in rows for k in row})
    leading = ['assay', 'protein_key', 'assay_type', 'n_variants', 'n_singles', 'n_doubles', 'n_higher_order', 'strict_double_evaluable', 'strict_higher_evaluable', 'best_zero_shot_method']
    fields = leading + [f for f in fields if f not in leading]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
