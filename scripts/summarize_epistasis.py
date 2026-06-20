from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List
ROOT = Path(__file__).resolve().parents[1]
EP_DIR = ROOT / 'results' / 'epistasis'

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

def summarize_residuals(residuals: Path, out: Path) -> None:
    by_assay: Dict[str, List[float]] = defaultdict(list)
    sign_counts: Dict[str, int] = defaultdict(int)
    with residuals.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            assay = row['assay']
            by_assay[assay].append(float(row['abs_epistasis_residual']))
            sign_counts[assay] += int(row['sign_epistasis_candidate'])
    rows = []
    for assay, values in by_assay.items():
        rows.append({'assay': assay, 'n_strict_doubles': len(values), 'mean_abs_epistasis_residual': mean(values), 'median_abs_epistasis_residual': median(values), 'sign_epistasis_candidate_count': sign_counts[assay], 'sign_epistasis_candidate_fraction': sign_counts[assay] / len(values)})
    rows.sort(key=lambda row: float(row['mean_abs_epistasis_residual']), reverse=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def summarize_method_metrics(metrics: Path, out: Path) -> None:
    by_method: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    with metrics.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            gain = to_float(row['gain_vs_additive_abs_spearman'])
            rho = to_float(row['abs_spearman_double_score'])
            rho_abs_res = to_float(row['spearman_abs_epistasis_residual'])
            high_recall = to_float(row['top100_high_abs_residual_recall'])
            if rho is None:
                continue
            by_method[row['method']].append({'gain': gain if gain is not None else 0.0, 'abs_rho': rho, 'rho_abs_res': rho_abs_res if rho_abs_res is not None else float('nan'), 'high_recall': high_recall if high_recall is not None else float('nan')})
    rows = []
    for method, records in by_method.items():
        rows.append({'method': method, 'n_assays': len(records), 'mean_abs_spearman_double_score': mean((r['abs_rho'] for r in records)), 'mean_gain_vs_additive_abs_spearman': mean((r['gain'] for r in records)), 'assays_positive_gain': sum((1 for r in records if r['gain'] > 0)), 'mean_spearman_abs_epistasis_residual': mean((r['rho_abs_res'] for r in records if r['rho_abs_res'] == r['rho_abs_res'])), 'mean_top100_high_abs_residual_recall': mean((r['high_recall'] for r in records if r['high_recall'] == r['high_recall']))})
    rows.sort(key=lambda row: float(row['mean_gain_vs_additive_abs_spearman']), reverse=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--residuals', default=str(EP_DIR / 'strict_double_epistasis_residuals.csv'))
    parser.add_argument('--metrics', default=str(EP_DIR / 'zero_shot_epistasis_double_metrics.csv'))
    args = parser.parse_args()
    summarize_residuals(Path(args.residuals), EP_DIR / 'epistasis_assay_summary.csv')
    summarize_method_metrics(Path(args.metrics), EP_DIR / 'zero_shot_epistasis_method_summary.csv')
    print(f'wrote {EP_DIR / 'epistasis_assay_summary.csv'}')
    print(f'wrote {EP_DIR / 'zero_shot_epistasis_method_summary.csv'}')
if __name__ == '__main__':
    main()
