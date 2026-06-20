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
    parser.add_argument('--metrics', default='results/binary_metrics/zero_shot_binary_metrics.csv')
    parser.add_argument('--out', default='results/binary_metrics/zero_shot_binary_summary.csv')
    args = parser.parse_args()
    metrics = ROOT / args.metrics if not Path(args.metrics).is_absolute() else Path(args.metrics)
    groups: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    with metrics.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            auc = to_float(row['auc'])
            if auc is None:
                continue
            groups[row['method']].append({'auc': auc, 'abs_auc_minus_half': abs(auc - 0.5), 'mcc': to_float(row['mcc_top10pct']) or 0.0, 'rec10': to_float(row['recall_pos_at_top10pct']) or 0.0, 'rec01': to_float(row['recall_pos_at_top1pct']) or 0.0, 'positive_rate': to_float(row['positive_rate']) or 0.0})
    rows = []
    for method, recs in groups.items():
        rows.append({'method': method, 'n_assays': len(recs), 'mean_auc': mean((r['auc'] for r in recs)), 'mean_abs_auc_minus_half': mean((r['abs_auc_minus_half'] for r in recs)), 'mean_mcc_top10pct': mean((r['mcc'] for r in recs)), 'mean_recall_pos_at_top10pct': mean((r['rec10'] for r in recs)), 'mean_recall_pos_at_top1pct': mean((r['rec01'] for r in recs)), 'mean_positive_rate': mean((r['positive_rate'] for r in recs))})
    rows.sort(key=lambda r: (float(r['mean_auc']), float(r['mean_mcc_top10pct'])), reverse=True)
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {out}')
if __name__ == '__main__':
    main()
