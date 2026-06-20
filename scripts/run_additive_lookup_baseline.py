from __future__ import annotations
import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
ROOT = Path(__file__).resolve().parents[1]
DMS_DIR = ROOT / 'proteingym' / 'extracted' / 'DMS_ProteinGym_substitutions'
SPLITS = ROOT / 'results' / 'splits' / 'low_n_splits.csv'
OUT_DIR = ROOT / 'results' / 'additive'

def mutation_order(mutant: str) -> int:
    if not mutant or mutant in {'WT', 'wildtype', '_wt'}:
        return 0
    return mutant.count(':') + 1

def ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            result[indexed[k][0]] = avg_rank
        i = j
    return result

def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 3:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum(((a - mean_x) * (b - mean_y) for a, b in zip(x, y)))
    den_x = math.sqrt(sum(((a - mean_x) ** 2 for a in x)))
    den_y = math.sqrt(sum(((b - mean_y) ** 2 for b in y)))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)

def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    return pearson(ranks(x), ranks(y))

def load_scores(assay: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    with (DMS_DIR / assay).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            try:
                scores[row['mutant']] = float(row['DMS_score'])
            except ValueError:
                continue
    return scores

def load_split_groups() -> Dict[Tuple[str, int, int, str], List[str]]:
    groups: Dict[Tuple[str, int, int, str], List[str]] = {}
    with SPLITS.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            key = (row['assay'], int(row['seed']), int(row['budget']), row['regime'])
            groups.setdefault(key, []).append(row['mutant'])
    return groups

def predict(mutant: str, mean_single: float, effects: Dict[str, float]) -> float:
    if mutation_order(mutant) == 0:
        return mean_single
    return mean_single + sum((effects.get(part, 0.0) for part in mutant.split(':')))

def evaluate_group_with_scores(key: Tuple[str, int, int, str], train_mutants: Sequence[str], scores: Dict[str, float]) -> Dict[str, object]:
    assay, seed, budget, regime = key
    train_set = set(train_mutants)
    train_singles = [mutant for mutant in train_mutants if mutation_order(mutant) == 1 and mutant in scores]
    if train_singles:
        mean_single = sum((scores[mutant] for mutant in train_singles)) / len(train_singles)
        effects = {mutant: scores[mutant] - mean_single for mutant in train_singles}
    else:
        observed = [scores[mutant] for mutant in train_mutants if mutant in scores]
        mean_single = sum(observed) / len(observed) if observed else 0.0
        effects = {}
    y_all: List[float] = []
    p_all: List[float] = []
    y_double: List[float] = []
    p_double: List[float] = []
    y_higher: List[float] = []
    p_higher: List[float] = []
    for mutant, score in scores.items():
        if mutant in train_set:
            continue
        pred = predict(mutant, mean_single, effects)
        order = mutation_order(mutant)
        y_all.append(score)
        p_all.append(pred)
        if order == 2:
            y_double.append(score)
            p_double.append(pred)
        elif order >= 3:
            y_higher.append(score)
            p_higher.append(pred)
    rho_all = spearman(p_all, y_all)
    rho_double = spearman(p_double, y_double)
    rho_higher = spearman(p_higher, y_higher)
    return {'assay': assay, 'seed': seed, 'budget': budget, 'regime': regime, 'n_train': len(train_mutants), 'n_train_singles': len(train_singles), 'n_test': len(y_all), 'spearman_all': '' if rho_all is None else rho_all, 'n_test_doubles': len(y_double), 'spearman_doubles': '' if rho_double is None else rho_double, 'n_test_higher': len(y_higher), 'spearman_higher': '' if rho_higher is None else rho_higher}

def evaluate_assay_groups(assay: str, assay_groups: Sequence[Tuple[Tuple[str, int, int, str], Sequence[str]]]) -> List[Dict[str, object]]:
    scores = load_scores(assay)
    return [evaluate_group_with_scores(key, mutants, scores) for key, mutants in assay_groups]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', default=str(OUT_DIR / 'additive_lookup_metrics.csv'))
    parser.add_argument('--limit-groups', type=int, default=0)
    args = parser.parse_args()
    groups = load_split_groups()
    by_assay: Dict[str, List[Tuple[Tuple[str, int, int, str], Sequence[str]]]] = {}
    for key, mutants in groups.items():
        by_assay.setdefault(key[0], []).append((key, mutants))
    items = list(by_assay.items())
    if args.limit_groups:
        items = items[:args.limit_groups]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ['assay', 'seed', 'budget', 'regime', 'n_train', 'n_train_singles', 'n_test', 'spearman_all', 'n_test_doubles', 'spearman_doubles', 'n_test_higher', 'spearman_higher']
    out = Path(args.out)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(evaluate_assay_groups, assay, assay_groups) for assay, assay_groups in items]
            for future in as_completed(futures):
                writer.writerows(future.result())
    print(f'wrote {out}')
if __name__ == '__main__':
    main()
