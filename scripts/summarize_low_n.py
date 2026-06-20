from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'results' / 'low_n'

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

def summarize(paths: List[Path], out: Path) -> None:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, float]]] = defaultdict(list)
    wins: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_split: Dict[Tuple[str, str, str, str], Tuple[str, float]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                rho = to_float(row.get('spearman_all', ''))
                top1 = to_float(row.get('top1pct_recall_at_100', ''))
                best = to_float(row.get('best_true_in_pred_top_100', ''))
                if rho is None and top1 is None:
                    continue
                model_id = row['model'] if row['zero_shot_method'] == 'none' else f'{row['model']}:{row['zero_shot_method']}'
                key = (row['budget'], row['regime'], row['model'], row['zero_shot_method'])
                groups[key].append({'spearman_all': rho if rho is not None else float('nan'), 'abs_spearman_all': abs(rho) if rho is not None else float('nan'), 'top1pct_recall_at_100': top1 if top1 is not None else float('nan'), 'best_true_in_pred_top_100': best if best is not None else float('nan')})
                split_key = (row['assay'], row['seed'], row['budget'], row['regime'])
                score = top1 if top1 is not None else abs(rho) if rho is not None else float('-inf')
                if split_key not in per_split or score > per_split[split_key][1]:
                    per_split[split_key] = (model_id, score)
    for (assay, seed, budget, regime), (model_id, _) in per_split.items():
        wins[budget, regime, 'top1_or_abs_spearman'][model_id] += 1
    rows = []
    for (budget, regime, model, method), records in groups.items():
        rows.append({'budget': budget, 'regime': regime, 'model': model, 'zero_shot_method': method, 'n_rows': len(records), 'mean_abs_spearman_all': mean((r['abs_spearman_all'] for r in records if r['abs_spearman_all'] == r['abs_spearman_all'])), 'mean_top1pct_recall_at_100': mean((r['top1pct_recall_at_100'] for r in records if r['top1pct_recall_at_100'] == r['top1pct_recall_at_100'])), 'mean_best_true_in_pred_top_100': mean((r['best_true_in_pred_top_100'] for r in records if r['best_true_in_pred_top_100'] == r['best_true_in_pred_top_100']))})
    rows.sort(key=lambda row: (int(row['budget']), row['regime'], -float(row['mean_top1pct_recall_at_100']), -float(row['mean_abs_spearman_all'])))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    wins_out = out.with_name(f'{out.stem}_model_wins.csv')
    win_rows = []
    for (budget, regime, metric), model_counts in wins.items():
        for model_id, count in sorted(model_counts.items(), key=lambda item: item[1], reverse=True):
            win_rows.append({'budget': budget, 'regime': regime, 'metric': metric, 'model_id': model_id, 'wins': count})
    with wins_out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['budget', 'regime', 'metric', 'model_id', 'wins'])
        writer.writeheader()
        writer.writerows(win_rows)
    print(f'wrote {out}')
    print(f'wrote {wins_out}')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default='results/low_n/low_n_calibration_representative_metrics.csv,results/low_n/low_n_calibration_giant_fast_metrics.csv')
    parser.add_argument('--out', default=str(OUT_DIR / 'low_n_summary.csv'))
    args = parser.parse_args()
    paths = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in args.inputs.split(',') if p]
    summarize(paths, Path(args.out))
if __name__ == '__main__':
    main()
