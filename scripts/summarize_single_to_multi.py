from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'results' / 'single_to_multi'

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
    parser.add_argument('--metrics', default=str(OUT_DIR / 'single_to_multi_representative_metrics.csv'))
    parser.add_argument('--out', default=str(OUT_DIR / 'single_to_multi_summary.csv'))
    args = parser.parse_args()
    by_key: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    with Path(args.metrics).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rho = to_float(row['abs_spearman'])
            recall = to_float(row['top1pct_recall_at_100'])
            if rho is None:
                continue
            by_key[row['method'], row['order_bucket']].append({'abs_spearman': rho, 'top1pct_recall_at_100': recall if recall is not None else float('nan'), 'n': float(row['n'])})
    rows = []
    for (method, bucket), records in by_key.items():
        rows.append({'method': method, 'order_bucket': bucket, 'n_assays': len(records), 'mean_abs_spearman': mean((r['abs_spearman'] for r in records)), 'mean_top1pct_recall_at_100': mean((r['top1pct_recall_at_100'] for r in records if r['top1pct_recall_at_100'] == r['top1pct_recall_at_100'])), 'mean_n_variants': mean((r['n'] for r in records))})
    rows.sort(key=lambda row: (row['order_bucket'], -float(row['mean_abs_spearman'])))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {out}')
if __name__ == '__main__':
    main()
