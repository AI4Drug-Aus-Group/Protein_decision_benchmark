from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'results' / 'double_data_value'

def parse_float(value: str) -> float | None:
    if value == '':
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None

def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float('nan')

def load_rows(paths: List[Path]) -> Dict[Tuple[str, str, str, str, str, str], Dict[str, float]]:
    out = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                method = row.get('zero_shot_method') or row.get('features') or 'none'
                key = (row['assay'], row['seed'], row['budget'], row['regime'], row['model'], method)
                out[key] = {'spearman_doubles': parse_float(row.get('spearman_doubles', '')), 'spearman_higher': parse_float(row.get('spearman_higher', '')), 'spearman_all': parse_float(row.get('spearman_all', '')), 'top1pct_recall_at_100': parse_float(row.get('top1pct_recall_at_100', ''))}
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default='results/low_n/low_n_calibration_top20_nongiant_metrics.csv,results/low_n/low_n_supervised_baselines_nongiant_metrics.csv,results/low_n/low_n_multi_zscore_ridge_nongiant_metrics.csv')
    parser.add_argument('--out', default=str(OUT_DIR / 'single_plus_double_delta_summary.csv'))
    args = parser.parse_args()
    paths = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in args.inputs.split(',') if p]
    data = load_rows(paths)
    deltas: Dict[Tuple[str, str, str], List[Dict[str, float]]] = defaultdict(list)
    detail_rows = []
    for key, base in data.items():
        assay, seed, budget, regime, model, method = key
        if regime != 'single_only':
            continue
        paired = data.get((assay, seed, budget, 'single_plus_double', model, method))
        if paired is None:
            continue
        rec = {}
        for metric in ['spearman_doubles', 'spearman_higher', 'spearman_all', 'top1pct_recall_at_100']:
            a = paired.get(metric)
            b = base.get(metric)
            if a is not None and b is not None:
                rec[f'delta_{metric}'] = a - b
        if not rec:
            continue
        deltas[budget, model, method].append(rec)
        detail = {'assay': assay, 'seed': seed, 'budget': budget, 'model': model, 'method': method}
        detail.update(rec)
        detail_rows.append(detail)
    summary_rows = []
    for (budget, model, method), records in deltas.items():
        metric_counts = {metric: sum((f'delta_{metric}' in record for record in records)) for metric in ['spearman_doubles', 'spearman_higher', 'spearman_all', 'top1pct_recall_at_100']}
        summary_rows.append({'budget': budget, 'model': model, 'method': method, 'n_pairs': len(records), 'n_pairs_spearman_doubles': metric_counts['spearman_doubles'], 'n_pairs_spearman_higher': metric_counts['spearman_higher'], 'n_pairs_spearman_all': metric_counts['spearman_all'], 'n_pairs_top1pct_recall_at_100': metric_counts['top1pct_recall_at_100'], 'mean_delta_spearman_doubles': mean((r['delta_spearman_doubles'] for r in records if 'delta_spearman_doubles' in r)), 'mean_delta_spearman_higher': mean((r['delta_spearman_higher'] for r in records if 'delta_spearman_higher' in r)), 'mean_delta_spearman_all': mean((r['delta_spearman_all'] for r in records if 'delta_spearman_all' in r)), 'mean_delta_top1pct_recall_at_100': mean((r['delta_top1pct_recall_at_100'] for r in records if 'delta_top1pct_recall_at_100' in r))})
    summary_rows.sort(key=lambda r: (int(r['budget']), -float(r['mean_delta_spearman_doubles']) if str(r['mean_delta_spearman_doubles']) != 'nan' else 0.0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ['budget', 'model', 'method', 'n_pairs', 'n_pairs_spearman_doubles', 'n_pairs_spearman_higher', 'n_pairs_spearman_all', 'n_pairs_top1pct_recall_at_100', 'mean_delta_spearman_doubles', 'mean_delta_spearman_higher', 'mean_delta_spearman_all', 'mean_delta_top1pct_recall_at_100']
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    detail_out = out.parent / 'single_plus_double_delta_detail.csv'
    detail_fields = sorted({k for row in detail_rows for k in row})
    with detail_out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f'wrote {out}')
    print(f'wrote {detail_out}')
if __name__ == '__main__':
    main()
