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

def summarize(inputs: List[Path], out: Path) -> None:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, float]]] = defaultdict(list)
    for path in inputs:
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                pct = to_float(row.get('best_true_in_pred_top_100_percentile', ''))
                top1 = to_float(row.get('top1pct_recall_at_100', ''))
                rho = to_float(row.get('spearman_all', ''))
                if pct is None and top1 is None and (rho is None):
                    continue
                method_id = row.get('zero_shot_method') or row.get('features') or 'none'
                groups[row['budget'], row['regime'], row['model'], method_id].append({'best_percentile': pct if pct is not None else float('nan'), 'top1': top1 if top1 is not None else float('nan'), 'abs_rho': abs(rho) if rho is not None else float('nan')})
    rows = []
    for (budget, regime, model, method), records in groups.items():
        rows.append({'budget': budget, 'regime': regime, 'model': model, 'zero_shot_method': method, 'n_rows': len(records), 'mean_best_top100_percentile': mean((r['best_percentile'] for r in records if r['best_percentile'] == r['best_percentile'])), 'mean_top1pct_recall_at_100': mean((r['top1'] for r in records if r['top1'] == r['top1'])), 'mean_abs_spearman_all': mean((r['abs_rho'] for r in records if r['abs_rho'] == r['abs_rho']))})
    rows.sort(key=lambda r: (int(r['budget']), r['regime'], -float(r['mean_best_top100_percentile']), -float(r['mean_top1pct_recall_at_100'])))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {out}')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    inputs = [ROOT / item if not Path(item).is_absolute() else Path(item) for item in args.inputs.split(',') if item]
    summarize(inputs, ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out))
if __name__ == '__main__':
    main()
