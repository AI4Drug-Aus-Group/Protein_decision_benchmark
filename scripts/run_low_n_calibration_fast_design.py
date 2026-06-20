from __future__ import annotations
import argparse
import csv
import heapq
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
SPLITS = ROOT / 'results' / 'splits' / 'low_n_splits.csv'
OUT_DIR = ROOT / 'results' / 'low_n'
DEFAULT_METHODS = ['Site_Independent', 'EVmutation', 'EVE_ensemble', 'MSA_Transformer_ensemble', 'GEMME', 'ESM1v_ensemble', 'ESM2_650M', 'ESM2_15B', 'Tranception_L', 'TranceptEVE_L', 'ProteinMPNN', 'ESM-IF1', 'SaProt_650M_AF2', 'ProSST-2048', 'ESM3', 'ESMC-600M', 'xTrimoPGLM-100B-int4', 'Progen3_3b']

def parse_float(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed

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
    if len(x) < 3:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum(((a - mean_x) * (b - mean_y) for a, b in zip(x, y)))
    den_x = math.sqrt(sum(((a - mean_x) ** 2 for a in x)))
    den_y = math.sqrt(sum(((b - mean_y) ** 2 for b in y)))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)

def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    return pearson(ranks(x), ranks(y))

def solve_linear(features: Sequence[Sequence[float]], targets: Sequence[float], ridge: float=1e-08) -> Optional[List[float]]:
    if not features:
        return None
    p = len(features[0])
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]
    for row, y in zip(features, targets):
        for i in range(p):
            xty[i] += row[i] * y
            for j in range(p):
                xtx[i][j] += row[i] * row[j]
    for i in range(1, p):
        xtx[i][i] += ridge
    a = [xtx[i] + [xty[i]] for i in range(p)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            a[col], a[pivot] = (a[pivot], a[col])
        scale = a[col][col]
        for j in range(col, p + 1):
            a[col][j] /= scale
        for r in range(p):
            if r == col:
                continue
            factor = a[r][col]
            for j in range(col, p + 1):
                a[r][j] -= factor * a[col][j]
    return [a[i][p] for i in range(p)]

def dot(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum((w * x for w, x in zip(weights, features)))

def load_split_groups(assays: set[str]) -> Dict[str, List[Tuple[Tuple[str, int, int, str], List[str]]]]:
    groups: Dict[str, Dict[Tuple[str, int, int, str], List[str]]] = {}
    with SPLITS.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            assay = row['assay']
            if assay not in assays:
                continue
            key = (assay, int(row['seed']), int(row['budget']), row['regime'])
            groups.setdefault(assay, {}).setdefault(key, []).append(row['mutant'])
    return {assay: list(assay_groups.items()) for assay, assay_groups in groups.items()}

def load_table(assay: str, methods: Sequence[str]) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    with (ZERO_DIR / assay).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            y = parse_float(row.get('DMS_score'))
            if y is None:
                continue
            record = {'DMS_score': y}
            for method in methods:
                z = parse_float(row.get(method))
                if z is not None:
                    record[method] = z
            table[row['mutant']] = record
    return table

def additive_predictions(table: Dict[str, Dict[str, float]], train_set: set[str], mutants: Sequence[str]) -> Dict[str, float]:
    train_singles = [m for m in train_set if mutation_order(m) == 1 and m in table]
    if train_singles:
        mean_single = sum((table[m]['DMS_score'] for m in train_singles)) / len(train_singles)
        effects = {m: table[m]['DMS_score'] - mean_single for m in train_singles}
    else:
        observed = [table[m]['DMS_score'] for m in train_set if m in table]
        mean_single = sum(observed) / len(observed) if observed else 0.0
        effects = {}
    preds = {}
    for mutant in mutants:
        preds[mutant] = mean_single + sum((effects.get(part, 0.0) for part in mutant.split(':')))
    return preds

def sample_mutants(mutants: Sequence[str], assay: str, key: Tuple[str, int, int, str], max_sample: int) -> List[str]:
    if len(mutants) <= max_sample:
        return list(mutants)
    rng = random.Random(f'{assay}|{key}|sample')
    return rng.sample(list(mutants), max_sample)

def evaluate(assay: str, key: Tuple[str, int, int, str], model: str, method: str, table: Dict[str, Dict[str, float]], train_set: set[str], pred_by_mutant: Dict[str, float], sample_size: int, k: int) -> Dict[str, object]:
    sample = [m for m in pred_by_mutant if m not in train_set and m in table]
    sample_y = [table[m]['DMS_score'] for m in sample]
    sample_p = [pred_by_mutant[m] for m in sample]
    rho = spearman(sample_p, sample_y)
    top_pred = heapq.nlargest(min(k, len(sample)), sample, key=lambda m: pred_by_mutant[m])
    top1_n = max(1, math.ceil(len(sample) * 0.01))
    top_true = set(heapq.nlargest(top1_n, sample, key=lambda m: table[m]['DMS_score']))
    pred_set = set(top_pred)
    _, seed, budget, regime = key
    return {'assay': assay, 'seed': seed, 'budget': budget, 'regime': regime, 'model': model, 'zero_shot_method': method, 'n_train': len(train_set), 'n_test': len(sample), 'spearman_all': '' if rho is None else rho, 'spearman_sampled': 1, 'spearman_sample_size': len(sample), 'design_sampled': 1, 'design_sample_size': len(sample), 'n_test_doubles': '', 'spearman_doubles': '', 'n_test_higher': '', 'spearman_higher': '', 'ndcg_at_100': '', 'top1pct_recall_at_100': len(pred_set & top_true) / len(top_true), 'best_true_in_pred_top_100': max((table[m]['DMS_score'] for m in top_pred)) if top_pred else ''}

def process_assay(assay: str, assay_groups: Sequence[Tuple[Tuple[str, int, int, str], Sequence[str]]], methods: Sequence[str], sample_size: int, k: int) -> List[Dict[str, object]]:
    table = load_table(assay, methods)
    rows: List[Dict[str, object]] = []
    for key, train_mutants in assay_groups:
        train_set = {m for m in train_mutants if m in table}
        if len(train_set) < 3:
            continue
        candidate_sample = sample_mutants([m for m in table if m not in train_set], assay, key, sample_size)
        eval_mutants = sorted(set(candidate_sample) | train_set)
        add_pred = additive_predictions(table, train_set, eval_mutants)
        rows.append(evaluate(assay, key, 'additive_lookup', 'none', table, train_set, add_pred, sample_size, k))
        for method in methods:
            train_y = []
            f1 = []
            f2 = []
            for mutant in train_set:
                record = table.get(mutant)
                if record is None or method not in record:
                    continue
                train_y.append(record['DMS_score'])
                f1.append([1.0, record[method]])
                f2.append([1.0, add_pred[mutant], record[method]])
            if len(train_y) < 3:
                continue
            w1 = solve_linear(f1, train_y)
            if w1 is not None:
                pred = {m: dot(w1, [1.0, table[m][method]]) for m in candidate_sample if m in table and method in table[m]}
                rows.append(evaluate(assay, key, 'zscore_linear', method, table, train_set, pred, sample_size, k))
            w2 = solve_linear(f2, train_y)
            if w2 is not None:
                pred = {m: dot(w2, [1.0, add_pred[m], table[m][method]]) for m in candidate_sample if m in add_pred and m in table and (method in table[m])}
                rows.append(evaluate(assay, key, 'additive_plus_zscore', method, table, train_set, pred, sample_size, k))
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--assays', required=True, help='Comma-separated assay CSV names.')
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--sample-size', type=int, default=20000)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--out', default=str(OUT_DIR / 'low_n_calibration_giant_fast_metrics.csv'))
    args = parser.parse_args()
    assays = {a for a in args.assays.split(',') if a}
    methods = [m for m in args.methods.split(',') if m]
    groups = load_split_groups(assays)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    fieldnames = ['assay', 'seed', 'budget', 'regime', 'model', 'zero_shot_method', 'n_train', 'n_test', 'spearman_all', 'spearman_sampled', 'spearman_sample_size', 'design_sampled', 'design_sample_size', 'n_test_doubles', 'spearman_doubles', 'n_test_higher', 'spearman_higher', 'ndcg_at_100', 'top1pct_recall_at_100', 'best_true_in_pred_top_100']
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_assay, assay, assay_groups, methods, args.sample_size, args.k) for assay, assay_groups in groups.items()]
            for future in as_completed(futures):
                writer.writerows(future.result())
    print(f'wrote {out_path}')
if __name__ == '__main__':
    main()
