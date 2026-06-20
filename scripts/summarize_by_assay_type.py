from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
ROOT = Path(__file__).resolve().parents[1]
CURATION = ROOT / 'results' / 'batch0' / 'assay_type_curation.csv'
OUT_DIR = ROOT / 'results' / 'assay_type'

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

def load_types(path: Path) -> Dict[str, str]:
    with path.open(newline='', encoding='utf-8') as fh:
        out = {}
        for row in csv.DictReader(fh):
            out[row['assay']] = row.get('assay_type_refined', row.get('assay_type_curated', row.get('assay_type', 'uncurated')))
        return out

def zero_shot(types: Dict[str, str]) -> None:
    methods = {'VenusREM', 'ProSST-2048', 'S3F_MSA', 'ESM3', 'PoET', 'RSALOR', 'GEMME', 'Site_Independent'}
    groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    with (ROOT / 'results' / 'zero_shot' / 'zero_shot_all_methods_metrics.csv').open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row['method'] not in methods:
                continue
            assay_type = types.get(row['assay'], 'missing')
            rho = to_float(row['abs_spearman'])
            if rho is not None:
                groups[assay_type, row['method']].append(rho)
    rows = [{'assay_type': k[0], 'method': k[1], 'n_assays': len(v), 'mean_abs_spearman': mean(v)} for k, v in groups.items()]
    rows.sort(key=lambda r: (r['assay_type'], -float(r['mean_abs_spearman'])))
    write(OUT_DIR / 'zero_shot_by_assay_type.csv', rows)

def low_n(types: Dict[str, str]) -> None:
    methods = {'ESM3', 'S2F_MSA', 'S3F_MSA', 'PoET', 'ProSST-1024', 'VenusREM'}
    groups: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    path = ROOT / 'results' / 'low_n' / 'low_n_calibration_top20_nongiant_metrics.percentile.csv'
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row['budget'] != '20' or row['regime'] != 'mixed_random':
                continue
            if row.get('zero_shot_method') not in methods or row['model'] != 'zscore_linear':
                continue
            pct = to_float(row.get('best_true_in_pred_top_100_percentile', ''))
            if pct is not None:
                groups[types.get(row['assay'], 'missing'), row['model'], row['zero_shot_method'], row['budget']].append(pct)
    rows = [{'assay_type': k[0], 'model': k[1], 'zero_shot_method': k[2], 'budget': k[3], 'n_rows': len(v), 'mean_best_top100_percentile': mean(v)} for k, v in groups.items()]
    rows.sort(key=lambda r: (r['assay_type'], -float(r['mean_best_top100_percentile'])))
    write(OUT_DIR / 'low_n_mixed20_by_assay_type.csv', rows)

def single_to_multi(types: Dict[str, str]) -> None:
    methods = {'additive_all_singles', 'ProSST-2048', 'ESM3', 'MSA_Transformer_ensemble', 'GEMME', 'TranceptEVE_L'}
    groups: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    path = ROOT / 'results' / 'single_to_multi' / 'single_to_multi_representative_metrics.csv'
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row['method'] not in methods:
                continue
            rho = to_float(row['abs_spearman'])
            if rho is not None:
                groups[types.get(row['assay'], 'missing'), row['order_bucket'], row['method']].append(rho)
    rows = [{'assay_type': k[0], 'order_bucket': k[1], 'method': k[2], 'n_assays': len(v), 'mean_abs_spearman': mean(v)} for k, v in groups.items()]
    rows.sort(key=lambda r: (r['assay_type'], r['order_bucket'], -float(r['mean_abs_spearman'])))
    write(OUT_DIR / 'single_to_multi_by_assay_type.csv', rows)

def write(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')

def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--curation', default=str(CURATION))
    parser.add_argument('--out-dir', default=str(OUT_DIR))
    args = parser.parse_args()
    OUT_DIR = Path(args.out_dir)
    types = load_types(Path(args.curation))
    zero_shot(types)
    low_n(types)
    single_to_multi(types)
if __name__ == '__main__':
    main()
