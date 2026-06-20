from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
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
    parser.add_argument('--metrics', default='results/multi_evolve_style/pairwise_residual_higher_order_metrics.csv')
    parser.add_argument('--out', default='results/multi_evolve_style/pairwise_residual_higher_order_summary.csv')
    args = parser.parse_args()
    metrics = ROOT / args.metrics if not Path(args.metrics).is_absolute() else Path(args.metrics)
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    groups: Dict[Tuple[str, str, str], List[Dict[str, float]]] = defaultdict(list)
    with metrics.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rho = to_float(row['abs_spearman_higher'])
            recall = to_float(row['top1pct_recall_at_100'])
            coverage = to_float(row['pair_coverage'])
            if rho is None:
                continue
            groups[row['model'], row.get('selection_policy', 'none'), row['double_budget']].append({'rho': rho, 'recall': recall if recall is not None else float('nan'), 'coverage': coverage if coverage is not None else float('nan')})
    rows = []
    for (model, selection_policy, budget), records in groups.items():
        rows.append({'model': model, 'selection_policy': selection_policy, 'double_budget': budget, 'n_rows': len(records), 'mean_abs_spearman_higher': mean((r['rho'] for r in records)), 'mean_top1pct_recall_at_100': mean((r['recall'] for r in records if r['recall'] == r['recall'])), 'mean_pair_coverage': mean((r['coverage'] for r in records if r['coverage'] == r['coverage']))})
    rows.sort(key=lambda r: (r['model'], r['selection_policy'], int(r['double_budget']) if r['double_budget'] != '0' else 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {out}')
if __name__ == '__main__':
    main()
