from __future__ import annotations
import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
ROOT = Path(__file__).resolve().parents[1]
DMS_DIR = ROOT / 'proteingym' / 'extracted' / 'DMS_ProteinGym_substitutions'
OUT_DIR = ROOT / 'results' / 'epistasis'

def mutation_order(mutant: str) -> int:
    if not mutant or mutant in {'WT', 'wildtype', '_wt'}:
        return 0
    return mutant.count(':') + 1

def process_assay(path: Path, min_strict_doubles: int) -> List[Dict[str, object]]:
    scores: Dict[str, float] = {}
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            try:
                scores[row['mutant']] = float(row['DMS_score'])
            except ValueError:
                continue
    single_scores = {m: y for m, y in scores.items() if mutation_order(m) == 1}
    if not single_scores:
        return []
    wt_proxy = sum(single_scores.values()) / len(single_scores)
    rows = []
    for mutant, y in scores.items():
        if mutation_order(mutant) != 2:
            continue
        parts = mutant.split(':')
        if not all((part in single_scores for part in parts)):
            continue
        additive = sum((single_scores[part] for part in parts)) - wt_proxy
        residual = y - additive
        rows.append({'assay': path.name, 'mutant': mutant, 'single_a': parts[0], 'single_b': parts[1], 'score_double': y, 'score_single_a': single_scores[parts[0]], 'score_single_b': single_scores[parts[1]], 'wt_proxy': wt_proxy, 'additive_prediction': additive, 'epistasis_residual': residual, 'abs_epistasis_residual': abs(residual), 'sign_epistasis_candidate': int((single_scores[parts[0]] - wt_proxy) * (y - single_scores[parts[1]]) < 0 or (single_scores[parts[1]] - wt_proxy) * (y - single_scores[parts[0]]) < 0)})
    if len(rows) < min_strict_doubles:
        return []
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--min-strict-doubles', type=int, default=100)
    parser.add_argument('--out', default=str(OUT_DIR / 'strict_double_epistasis_residuals.csv'))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(DMS_DIR.glob('*.csv'))
    fieldnames = ['assay', 'mutant', 'single_a', 'single_b', 'score_double', 'score_single_a', 'score_single_b', 'wt_proxy', 'additive_prediction', 'epistasis_residual', 'abs_epistasis_residual', 'sign_epistasis_candidate']
    out_path = Path(args.out)
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_assay, path, args.min_strict_doubles) for path in paths]
            for future in as_completed(futures):
                writer.writerows(future.result())
    print(f'wrote {out_path}')
if __name__ == '__main__':
    main()
