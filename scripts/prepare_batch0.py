from __future__ import annotations
import argparse
import csv
import json
import os
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
ROOT = Path(__file__).resolve().parents[1]
DMS_DIR = ROOT / 'proteingym' / 'extracted' / 'DMS_ProteinGym_substitutions'
TOP_EXTRACTED = ROOT / 'proteingym' / 'extracted'
MSA_DIR = ROOT / 'proteingym' / 'extracted' / 'DMS_msa_files'
ZERO_ZIP = ROOT / 'proteingym' / 'zips' / 'zero_shot_substitutions_scores.zip'
ZERO_DIR = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
OUT_DIR = ROOT / 'results' / 'batch0'
SPLIT_DIR = ROOT / 'results' / 'splits'
BASE_COLUMNS = {'mutant', 'mutated_sequence', 'DMS_score', 'DMS_score_bin', 'DMS_bin_score'}

def mutation_order(mutant: str) -> int:
    if not mutant or mutant in {'WT', 'wildtype', '_wt'}:
        return 0
    return mutant.count(':') + 1

def mutation_parts(mutant: str) -> List[str]:
    if mutation_order(mutant) == 0:
        return []
    return mutant.split(':')

def protein_key(assay_stem: str) -> str:
    parts = assay_stem.split('_')
    if len(parts) >= 2 and (parts[1].isupper() or parts[1][0].isdigit()):
        return f'{parts[0]}_{parts[1]}'
    return parts[0]

def assay_files() -> List[Path]:
    return sorted(DMS_DIR.glob('*.csv'))

def zero_names() -> set[str]:
    if not ZERO_ZIP.exists():
        return set()
    with zipfile.ZipFile(ZERO_ZIP) as zf:
        return {Path(name).name for name in zf.namelist() if name.endswith('.csv')}

def method_category(method: str) -> str:
    if method == 'Site_Independent':
        return 'additive_or_site_independent'
    if method.startswith(('EVmutation', 'DeepSequence', 'EVE', 'MSA_Transformer', 'GEMME', 'RSALOR')):
        return 'evolution_or_msa'
    if method.startswith(('ESM1', 'ESM2', 'ESM3', 'ESMC', 'Unirep', 'CARP')):
        return 'sequence_plm'
    if method.startswith(('RITA', 'Progen', 'ProtGPT2', 'Wavenet', 'xTrimoPGLM')):
        return 'generative_plm'
    if method.startswith(('Tranception', 'TranceptEVE', 'PoET', 'MULAN')):
        return 'retrieval_or_hybrid'
    if method.startswith(('MIF', 'ESM-IF1', 'ProteinMPNN', 'SaProt', 'ProSST', 'ProtSSN', 'S2F', 'S3F')):
        return 'structure_aware'
    if method in {'VESPA', 'VESPAl', 'VespaG', 'ESCOTT', 'VenusREM', 'SiteRM'}:
        return 'other_variant_effect'
    return 'other'

def extract_zero_shot(force: bool=False) -> None:
    ZERO_DIR.mkdir(parents=True, exist_ok=True)
    marker = ZERO_DIR / '.extract_complete'
    existing = list(ZERO_DIR.glob('*.csv'))
    if marker.exists() and existing and (not force):
        print(f'zero-shot scores already extracted: {ZERO_DIR}')
        return
    if not ZERO_ZIP.exists():
        raise FileNotFoundError(ZERO_ZIP)
    with zipfile.ZipFile(ZERO_ZIP) as zf:
        zf.extractall(ZERO_DIR)
    marker.write_text('ok\n', encoding='utf-8')
    print(f'extracted zero-shot scores to {ZERO_DIR}')

def write_method_inventory() -> List[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ZERO_ZIP.exists():
        raise FileNotFoundError(ZERO_ZIP)
    with zipfile.ZipFile(ZERO_ZIP) as zf:
        csv_name = next((name for name in zf.namelist() if name.endswith('.csv')))
        with zf.open(csv_name) as fh:
            header = fh.readline().decode('utf-8').strip().split(',')
    methods = [col for col in header if col not in BASE_COLUMNS]
    out = OUT_DIR / 'method_inventory.csv'
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['method', 'category'])
        writer.writeheader()
        for method in methods:
            writer.writerow({'method': method, 'category': method_category(method)})
    return methods

def has_msa_for_assay(assay_stem: str, msa_files: Sequence[Path]) -> bool:
    key = protein_key(assay_stem)
    return any((path.name.startswith(key) for path in msa_files))

def build_manifest() -> Dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    msa_files = sorted(MSA_DIR.glob('*.a2m'))
    zero_csvs = zero_names()
    rows = []
    total_rows = 0
    total_order_counts: Counter[int] = Counter()
    top_level_duplicates = 0
    for path in assay_files():
        singles = set()
        variants: List[Tuple[str, int, List[str]]] = []
        order_counts: Counter[int] = Counter()
        fieldnames: Sequence[str] = []
        with path.open(newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            for row in reader:
                mutant = row['mutant']
                order = mutation_order(mutant)
                parts = mutation_parts(mutant)
                variants.append((mutant, order, parts))
                order_counts[order] += 1
                if order == 1:
                    singles.add(mutant)
        strict_double = sum((1 for _, order, parts in variants if order == 2 and all((part in singles for part in parts))))
        strict_higher = sum((1 for _, order, parts in variants if order >= 3 and all((part in singles for part in parts))))
        stem = path.stem
        top_level_duplicates += int((TOP_EXTRACTED / path.name).exists())
        total_rows += len(variants)
        total_order_counts.update(order_counts)
        row = {'assay': path.name, 'protein_key': protein_key(stem), 'n_variants': len(variants), 'n_singles': order_counts.get(1, 0), 'n_doubles': order_counts.get(2, 0), 'n_higher_order': sum((count for order, count in order_counts.items() if order >= 3)), 'max_order': max(order_counts) if order_counts else 0, 'strict_double_evaluable': strict_double, 'strict_higher_evaluable': strict_higher, 'has_msa': int(has_msa_for_assay(stem, msa_files)), 'has_zero_shot': int(path.name in zero_csvs), 'has_duplicate_top_level_csv': int((TOP_EXTRACTED / path.name).exists()), 'columns': '|'.join(fieldnames)}
        for order in range(1, 11):
            row[f'n_order_{order}'] = order_counts.get(order, 0)
        row['n_order_gt10'] = sum((count for order, count in order_counts.items() if order > 10))
        rows.append(row)
    fieldnames = ['assay', 'protein_key', 'n_variants', 'n_singles', 'n_doubles', 'n_higher_order', 'max_order', 'strict_double_evaluable', 'strict_higher_evaluable', 'has_msa', 'has_zero_shot', 'has_duplicate_top_level_csv', 'columns'] + [f'n_order_{order}' for order in range(1, 11)] + ['n_order_gt10']
    manifest_path = OUT_DIR / 'assay_manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {'assays': len(rows), 'total_variants': total_rows, 'dms_dir': str(DMS_DIR.relative_to(ROOT)), 'duplicate_top_level_csv_count': top_level_duplicates, 'msa_files': len(msa_files), 'zero_shot_zip_exists': ZERO_ZIP.exists(), 'zero_shot_csv_count': len(zero_csvs), 'assays_with_msa': sum((row['has_msa'] for row in rows)), 'assays_with_zero_shot': sum((row['has_zero_shot'] for row in rows)), 'assays_with_doubles': sum((1 for row in rows if row['n_doubles'] > 0)), 'assays_with_higher_order': sum((1 for row in rows if row['n_higher_order'] > 0)), 'assays_with_100plus_strict_doubles': sum((1 for row in rows if row['strict_double_evaluable'] >= 100)), 'assays_with_100plus_strict_higher': sum((1 for row in rows if row['strict_higher_evaluable'] >= 100)), 'mutation_order_counts': {str(k): v for k, v in sorted(total_order_counts.items())}}
    with (OUT_DIR / 'batch0_summary.json').open('w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write('\n')
    return summary

def sample_without_replacement(items: Sequence[str], n: int, rng: random.Random) -> List[str]:
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        result = list(items)
        rng.shuffle(result)
        return result
    return rng.sample(list(items), n)

def split_for_assay(path: Path, budgets: Sequence[int], seeds: Sequence[int]) -> List[Dict[str, object]]:
    by_order: Dict[int, List[str]] = {}
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            mutant = row['mutant']
            by_order.setdefault(mutation_order(mutant), []).append(mutant)
    singles = by_order.get(1, [])
    doubles = by_order.get(2, [])
    all_variants = [mutant for order in sorted(by_order) for mutant in by_order[order] if order > 0]
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        for budget in budgets:
            regimes: Dict[str, List[str]] = {}
            regimes['single_only'] = sample_without_replacement(singles, budget, random.Random(f'{path.name}|{seed}|{budget}|single_only'))
            regimes['mixed_random'] = sample_without_replacement(all_variants, budget, random.Random(f'{path.name}|{seed}|{budget}|mixed_random'))
            rng = random.Random(f'{path.name}|{seed}|{budget}|single_plus_double')
            n_double = min(len(doubles), budget // 2)
            n_single = min(len(singles), budget - n_double)
            selected = sample_without_replacement(singles, n_single, rng)
            selected.extend(sample_without_replacement(doubles, budget - len(selected), rng))
            if len(selected) < budget:
                selected.extend((mutant for mutant in sample_without_replacement(all_variants, budget, rng) if mutant not in set(selected)))
            regimes['single_plus_double'] = selected[:budget]
            for regime, mutants in regimes.items():
                for rank, mutant in enumerate(mutants, start=1):
                    rows.append({'assay': path.name, 'seed': seed, 'budget': budget, 'regime': regime, 'rank': rank, 'mutant': mutant})
    return rows

def write_splits(budgets: Sequence[int], seeds: Sequence[int]) -> int:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    out = SPLIT_DIR / 'low_n_splits.csv'
    fieldnames = ['assay', 'seed', 'budget', 'regime', 'rank', 'mutant']
    count = 0
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for path in assay_files():
            rows = split_for_assay(path, budgets, seeds)
            writer.writerows(rows)
            count += len(rows)
    return count

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract-zero-shot', action='store_true')
    parser.add_argument('--force-extract', action='store_true')
    parser.add_argument('--make-splits', action='store_true')
    parser.add_argument('--budgets', default='20,50,100')
    parser.add_argument('--seeds', default='0,1,2,3,4')
    args = parser.parse_args()
    if args.extract_zero_shot:
        extract_zero_shot(force=args.force_extract)
    methods = write_method_inventory()
    summary = build_manifest()
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f'method_count: {len(methods)}')
    if args.make_splits:
        budgets = [int(value) for value in args.budgets.split(',') if value]
        seeds = [int(value) for value in args.seeds.split(',') if value]
        split_count = write_splits(budgets, seeds)
        print(f'split_rows: {split_count}')
if __name__ == '__main__':
    main()
