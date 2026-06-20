from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
ROOT = Path(__file__).resolve().parents[1]
STRATA = ROOT / 'results' / 'batch0' / 'assay_strata.csv'

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

def load_strata() -> Dict[str, Dict[str, str]]:
    with STRATA.open(newline='', encoding='utf-8') as fh:
        return {row['assay']: row for row in csv.DictReader(fh)}

def summarize_zero_shot(strata: Dict[str, Dict[str, str]], out: Path) -> None:
    path = ROOT / 'results' / 'zero_shot' / 'zero_shot_all_methods_metrics.csv'
    groups: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    methods = {'ProSST-2048', 'ESM3', 'VenusREM', 'PoET', 'RSALOR', 'GEMME', 'Site_Independent'}
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row['method'] not in methods or row['assay'] not in strata:
                continue
            rho = to_float(row['abs_spearman'])
            if rho is None:
                continue
            for stratum_col in ['mutation_stratum', 'epistasis_stratum', 'assay_type_keyword']:
                groups[stratum_col, strata[row['assay']][stratum_col], row['method']].append(rho)
    rows = [{'source': 'zero_shot', 'stratum_column': key[0], 'stratum': key[1], 'method': key[2], 'n': len(values), 'mean_abs_spearman': mean(values)} for key, values in groups.items()]
    rows.sort(key=lambda r: (r['stratum_column'], r['stratum'], -float(r['mean_abs_spearman'])))
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def summarize_single_to_multi(strata: Dict[str, Dict[str, str]], out: Path) -> None:
    path = ROOT / 'results' / 'single_to_multi' / 'single_to_multi_representative_metrics.csv'
    groups: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    methods = {'additive_all_singles', 'ProSST-2048', 'ESM3', 'MSA_Transformer_ensemble', 'GEMME', 'TranceptEVE_L'}
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row['method'] not in methods or row['assay'] not in strata:
                continue
            rho = to_float(row['abs_spearman'])
            n = int(row['n'])
            if rho is None or n < 3:
                continue
            for stratum_col in ['mutation_stratum', 'epistasis_stratum']:
                groups[stratum_col, strata[row['assay']][stratum_col], row['order_bucket'], row['method']].append(rho)
    rows = [{'source': 'single_to_multi', 'stratum_column': key[0], 'stratum': key[1], 'order_bucket': key[2], 'method': key[3], 'n': len(values), 'mean_abs_spearman': mean(values)} for key, values in groups.items()]
    rows.sort(key=lambda r: (r['stratum_column'], r['stratum'], r['order_bucket'], -float(r['mean_abs_spearman'])))
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-prefix', default='results/batch0/stratified')
    args = parser.parse_args()
    strata = load_strata()
    prefix = ROOT / args.out_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summarize_zero_shot(strata, prefix.with_name(prefix.name + '_zero_shot_summary.csv'))
    summarize_single_to_multi(strata, prefix.with_name(prefix.name + '_single_to_multi_summary.csv'))
    print(f'wrote {prefix.with_name(prefix.name + '_zero_shot_summary.csv')}')
    print(f'wrote {prefix.with_name(prefix.name + '_single_to_multi_summary.csv')}')
if __name__ == '__main__':
    main()
