from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'results' / 'active_learning'

def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float('nan')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', default=str(OUT_DIR / 'active_learning_representative_metrics.csv'))
    parser.add_argument('--out', default=str(OUT_DIR / 'active_learning_summary.csv'))
    parser.add_argument('--wins-out', default='')
    args = parser.parse_args()
    groups: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    final_by_assay_seed: Dict[Tuple[str, str], Tuple[str, float]] = {}
    with Path(args.metrics).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            key = (row['policy'], row['round'])
            rec = {'best': float(row['best_observed_fitness']), 'best_percentile': float(row.get('best_observed_percentile', 'nan')), 'top1': float(row['top1pct_found']), 'n_observed': float(row['n_observed'])}
            groups[key].append(rec)
            if row['round'] == '5':
                split = (row['assay'], row['seed'])
                current = final_by_assay_seed.get(split)
                if current is None or rec['best'] > current[1]:
                    final_by_assay_seed[split] = (row['policy'], rec['best'])
    rows = []
    for (policy, round_idx), records in groups.items():
        rows.append({'policy': policy, 'round': round_idx, 'n_runs': len(records), 'mean_n_observed': mean((r['n_observed'] for r in records)), 'mean_best_observed_fitness': mean((r['best'] for r in records)), 'mean_best_observed_percentile': mean((r['best_percentile'] for r in records)), 'top1pct_found_rate': mean((r['top1'] for r in records))})
    rows.sort(key=lambda r: (int(r['round']), r['policy']))
    out = Path(args.out)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    wins = defaultdict(int)
    for policy, _ in final_by_assay_seed.values():
        wins[policy] += 1
    wins_out = Path(args.wins_out) if args.wins_out else out.parent / 'active_learning_final_wins.csv'
    with wins_out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['policy', 'final_best_wins'])
        writer.writeheader()
        for policy, count in sorted(wins.items(), key=lambda item: item[1], reverse=True):
            writer.writerow({'policy': policy, 'final_best_wins': count})
    print(f'wrote {out}')
    print(f'wrote {wins_out}')
if __name__ == '__main__':
    main()
