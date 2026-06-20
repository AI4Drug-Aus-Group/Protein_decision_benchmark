from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / 'results' / 'zero_shot' / 'zero_shot_all_methods_metrics.csv'
METHOD_INVENTORY = ROOT / 'results' / 'batch0' / 'method_inventory.csv'
OUT_DIR = ROOT / 'results' / 'zero_shot'

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

def median(values: Iterable[float]) -> float:
    values = sorted(values)
    if not values:
        return float('nan')
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0

def load_categories() -> Dict[str, str]:
    categories: Dict[str, str] = {}
    with METHOD_INVENTORY.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            categories[row['method']] = row['category']
    return categories

def summarize(metrics_path: Path, out_dir: Path) -> None:
    categories = load_categories()
    by_method: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    by_assay_best: Dict[str, Tuple[str, float]] = {}
    with metrics_path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            spearman = to_float(row['spearman'])
            if spearman is None:
                continue
            record = {'spearman': spearman, 'abs_spearman': abs(spearman), 'ndcg_at_k': to_float(row.get('ndcg_at_k', '')) or 0.0, 'top1pct_recall_at_k': to_float(row.get('top1pct_recall_at_k', '')) or 0.0, 'top5pct_recall_at_k': to_float(row.get('top5pct_recall_at_k', '')) or 0.0}
            by_method[row['method']].append(record)
            current = by_assay_best.get(row['assay'])
            if current is None or abs(spearman) > current[1]:
                by_assay_best[row['assay']] = (row['method'], abs(spearman))
    out_dir.mkdir(parents=True, exist_ok=True)
    method_rows = []
    for method, records in by_method.items():
        method_rows.append({'method': method, 'category': categories.get(method, 'unknown'), 'n_assays': len(records), 'mean_spearman': mean((r['spearman'] for r in records)), 'median_spearman': median((r['spearman'] for r in records)), 'mean_abs_spearman': mean((r['abs_spearman'] for r in records)), 'median_abs_spearman': median((r['abs_spearman'] for r in records)), 'mean_ndcg_at_100': mean((r['ndcg_at_k'] for r in records)), 'mean_top1pct_recall_at_100': mean((r['top1pct_recall_at_k'] for r in records)), 'mean_top5pct_recall_at_100': mean((r['top5pct_recall_at_k'] for r in records)), 'best_abs_spearman_assay_wins': sum((1 for best_method, _ in by_assay_best.values() if best_method == method))})
    method_rows.sort(key=lambda row: row['mean_abs_spearman'], reverse=True)
    method_out = out_dir / 'zero_shot_method_summary.csv'
    with method_out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(method_rows[0].keys()))
        writer.writeheader()
        writer.writerows(method_rows)
    by_category: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in method_rows:
        by_category[str(row['category'])].append(row)
    category_rows = []
    for category, rows in by_category.items():
        category_rows.append({'category': category, 'n_methods': len(rows), 'mean_of_method_mean_abs_spearman': mean((float(row['mean_abs_spearman']) for row in rows)), 'best_method': rows[0]['method'], 'best_method_mean_abs_spearman': rows[0]['mean_abs_spearman'], 'assay_wins': sum((int(row['best_abs_spearman_assay_wins']) for row in rows))})
    category_rows.sort(key=lambda row: row['mean_of_method_mean_abs_spearman'], reverse=True)
    category_out = out_dir / 'zero_shot_category_summary.csv'
    with category_out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(category_rows[0].keys()))
        writer.writeheader()
        writer.writerows(category_rows)
    print(f'wrote {method_out}')
    print(f'wrote {category_out}')
    print('top_methods_by_mean_abs_spearman')
    for row in method_rows[:15]:
        print(row['method'], row['category'], f'{float(row['mean_abs_spearman']):.4f}', f'wins={row['best_abs_spearman_assay_wins']}')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', default=str(DEFAULT_METRICS))
    parser.add_argument('--out-dir', default=str(OUT_DIR))
    args = parser.parse_args()
    summarize(Path(args.metrics), Path(args.out_dir))
if __name__ == '__main__':
    main()
