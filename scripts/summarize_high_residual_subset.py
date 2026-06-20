from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List
ROOT = Path(__file__).resolve().parents[1]

def to_float(value: str) -> float | None:
    if value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None

def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float('nan')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', default='results/epistasis/high_residual_subset_metrics.csv')
    parser.add_argument('--out', default='results/epistasis/high_residual_subset_summary.csv')
    args = parser.parse_args()
    metrics = ROOT / args.metrics if not Path(args.metrics).is_absolute() else Path(args.metrics)
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    by_method: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    with metrics.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rho = to_float(row['abs_spearman_score'])
            recall = to_float(row['top1pct_recall_at_100'])
            rho_res = to_float(row['spearman_residual'])
            if rho is None:
                continue
            by_method[row['method']].append({'abs_rho': rho, 'recall': recall if recall is not None else float('nan'), 'rho_res': rho_res if rho_res is not None else float('nan')})
    rows = []
    for method, records in by_method.items():
        rows.append({'method': method, 'n_assays': len(records), 'mean_abs_spearman_score': mean((r['abs_rho'] for r in records)), 'mean_top1pct_recall_at_100': mean((r['recall'] for r in records if r['recall'] == r['recall'])), 'mean_spearman_residual': mean((r['rho_res'] for r in records if r['rho_res'] == r['rho_res']))})
    rows.sort(key=lambda r: -float(r['mean_abs_spearman_score']))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {out}')
if __name__ == '__main__':
    main()
