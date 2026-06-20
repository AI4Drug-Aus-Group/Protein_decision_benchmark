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
    parser.add_argument('--out', default=str(OUT_DIR / 'active_learning_hit_time_summary.csv'))
    args = parser.parse_args()
    first_hit: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    final_seen: Dict[Tuple[str, str, str], int] = {}
    with Path(args.metrics).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            key = (row['assay'], row['seed'], row['policy'])
            n_observed = int(row['n_observed'])
            final_seen[key] = max(final_seen.get(key, 0), n_observed)
            if int(row['top1pct_found']) and key not in first_hit:
                first_hit[key] = (int(row['round']), n_observed)
    groups: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for key, final_n in final_seen.items():
        _, _, policy = key
        hit = first_hit.get(key)
        groups[policy].append({'hit': 1.0 if hit else 0.0, 'first_hit_round': float(hit[0]) if hit else float('nan'), 'labels_to_first_hit': float(hit[1]) if hit else float('nan'), 'censored_labels': float(final_n)})
    rows = []
    for policy, records in groups.items():
        rows.append({'policy': policy, 'n_runs': len(records), 'hit_rate': mean((r['hit'] for r in records)), 'mean_first_hit_round_among_hits': mean((r['first_hit_round'] for r in records if r['first_hit_round'] == r['first_hit_round'])), 'mean_labels_to_first_hit_among_hits': mean((r['labels_to_first_hit'] for r in records if r['labels_to_first_hit'] == r['labels_to_first_hit'])), 'mean_censored_labels_all_runs': mean((r['labels_to_first_hit'] if r['labels_to_first_hit'] == r['labels_to_first_hit'] else r['censored_labels'] for r in records))})
    rows.sort(key=lambda r: (-float(r['hit_rate']), float(r['mean_censored_labels_all_runs'])))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ['policy', 'n_runs', 'hit_rate', 'mean_first_hit_round_among_hits', 'mean_labels_to_first_hit_among_hits', 'mean_censored_labels_all_runs']
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {out}')
if __name__ == '__main__':
    main()
