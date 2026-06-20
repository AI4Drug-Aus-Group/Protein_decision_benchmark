from __future__ import annotations
import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
ROOT = Path(__file__).resolve().parents[1]
DMS_DIR = ROOT / 'proteingym' / 'extracted' / 'DMS_ProteinGym_substitutions'
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

def dcg(relevances: Sequence[float]) -> float:
    return sum(((2.0 ** rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(relevances)))

def top_metrics(y_true: Sequence[float], y_pred: Sequence[float], k: int) -> Dict[str, float]:
    if len(y_true) < 3:
        return {'ndcg_at_k': float('nan'), 'top1pct_recall_at_k': float('nan'), 'best_true_in_pred_top_k': float('nan')}
    n = len(y_true)
    k = min(k, n)
    pred_idx = sorted(range(n), key=lambda i: y_pred[i], reverse=True)[:k]
    true_idx = sorted(range(n), key=lambda i: y_true[i], reverse=True)
    lo = min(y_true)
    hi = max(y_true)
    rel = [0.0 if hi == lo else (value - lo) / (hi - lo) for value in y_true]
    denom = dcg([rel[i] for i in true_idx[:k]])
    top1 = set(true_idx[:max(1, math.ceil(n * 0.01))])
    pred_set = set(pred_idx)
    return {'ndcg_at_k': dcg([rel[i] for i in pred_idx]) / denom if denom else 0.0, 'top1pct_recall_at_k': len(pred_set & top1) / len(top1), 'best_true_in_pred_top_k': max((y_true[i] for i in pred_idx))}

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

def load_split_groups() -> Dict[str, List[Tuple[Tuple[str, int, int, str], List[str]]]]:
    groups: Dict[str, Dict[Tuple[str, int, int, str], List[str]]] = {}
    with SPLITS.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            assay = row['assay']
            key = (assay, int(row['seed']), int(row['budget']), row['regime'])
            groups.setdefault(assay, {}).setdefault(key, []).append(row['mutant'])
    return {assay: list(assay_groups.items()) for assay, assay_groups in groups.items()}

def load_assay_table(assay: str, methods: Sequence[str]) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    with (ZERO_DIR / assay).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            record: Dict[str, float] = {}
            y = parse_float(row.get('DMS_score'))
            if y is None:
                continue
            record['DMS_score'] = y
            for method in methods:
                value = parse_float(row.get(method))
                if value is not None:
                    record[method] = value
            table[row['mutant']] = record
    return table

def additive_predictions(table: Dict[str, Dict[str, float]], train_mutants: Sequence[str]) -> Dict[str, float]:
    train_singles = [m for m in train_mutants if mutation_order(m) == 1 and m in table]
    if train_singles:
        mean_single = sum((table[m]['DMS_score'] for m in train_singles)) / len(train_singles)
        effects = {m: table[m]['DMS_score'] - mean_single for m in train_singles}
    else:
        observed = [table[m]['DMS_score'] for m in train_mutants if m in table]
        mean_single = sum(observed) / len(observed) if observed else 0.0
        effects = {}
    preds = {}
    for mutant in table:
        if mutation_order(mutant) == 0:
            preds[mutant] = mean_single
        else:
            preds[mutant] = mean_single + sum((effects.get(part, 0.0) for part in mutant.split(':')))
    return preds

def eval_prediction(assay: str, key: Tuple[str, int, int, str], model: str, method: str, table: Dict[str, Dict[str, float]], train_set: set[str], pred_by_mutant: Dict[str, float], k: int) -> Dict[str, object]:
    y_all: List[float] = []
    p_all: List[float] = []
    y_double: List[float] = []
    p_double: List[float] = []
    y_higher: List[float] = []
    p_higher: List[float] = []
    for mutant, record in table.items():
        if mutant in train_set or mutant not in pred_by_mutant:
            continue
        y = record['DMS_score']
        pred = pred_by_mutant[mutant]
        y_all.append(y)
        p_all.append(pred)
        order = mutation_order(mutant)
        if order == 2:
            y_double.append(y)
            p_double.append(pred)
        elif order >= 3:
            y_higher.append(y)
            p_higher.append(pred)
    rho_all = spearman(p_all, y_all)
    rho_double = spearman(p_double, y_double)
    rho_higher = spearman(p_higher, y_higher)
    metrics = top_metrics(y_all, p_all, k)
    _, seed, budget, regime = key
    return {'assay': assay, 'seed': seed, 'budget': budget, 'regime': regime, 'model': model, 'zero_shot_method': method, 'n_train': len(train_set), 'n_test': len(y_all), 'spearman_all': '' if rho_all is None else rho_all, 'n_test_doubles': len(y_double), 'spearman_doubles': '' if rho_double is None else rho_double, 'n_test_higher': len(y_higher), 'spearman_higher': '' if rho_higher is None else rho_higher, 'ndcg_at_100': metrics['ndcg_at_k'], 'top1pct_recall_at_100': metrics['top1pct_recall_at_k'], 'best_true_in_pred_top_100': metrics['best_true_in_pred_top_k']}

def process_assay(assay: str, assay_groups: Sequence[Tuple[Tuple[str, int, int, str], Sequence[str]]], methods: Sequence[str], k: int) -> List[Dict[str, object]]:
    table = load_assay_table(assay, methods)
    rows: List[Dict[str, object]] = []
    for key, train_mutants in assay_groups:
        train_set = {m for m in train_mutants if m in table}
        if len(train_set) < 3:
            continue
        add_pred = additive_predictions(table, list(train_set))
        rows.append(eval_prediction(assay, key, 'additive_lookup', 'none', table, train_set, add_pred, k))
        for method in methods:
            train_features_1 = []
            train_features_2 = []
            train_y = []
            for mutant in train_set:
                record = table.get(mutant)
                if record is None or method not in record:
                    continue
                z = record[method]
                train_features_1.append([1.0, z])
                train_features_2.append([1.0, add_pred[mutant], z])
                train_y.append(record['DMS_score'])
            if len(train_y) < 3:
                continue
            w1 = solve_linear(train_features_1, train_y)
            if w1 is not None:
                pred = {mutant: dot(w1, [1.0, record[method]]) for mutant, record in table.items() if method in record}
                rows.append(eval_prediction(assay, key, 'zscore_linear', method, table, train_set, pred, k))
            w2 = solve_linear(train_features_2, train_y)
            if w2 is not None:
                pred = {mutant: dot(w2, [1.0, add_pred[mutant], record[method]]) for mutant, record in table.items() if method in record}
                rows.append(eval_prediction(assay, key, 'additive_plus_zscore', method, table, train_set, pred, k))
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--out', default=str(OUT_DIR / 'low_n_calibration_representative_metrics.csv'))
    parser.add_argument('--limit-assays', type=int, default=0)
    parser.add_argument('--assays', default='', help='Optional comma-separated assay CSV names.')
    parser.add_argument('--exclude-assays', default='', help='Optional comma-separated assay CSV names to skip.')
    args = parser.parse_args()
    methods = [m for m in args.methods.split(',') if m]
    groups = load_split_groups()
    include = {a for a in args.assays.split(',') if a}
    exclude = {a for a in args.exclude_assays.split(',') if a}
    if include:
        groups = {assay: assay_groups for assay, assay_groups in groups.items() if assay in include}
    if exclude:
        groups = {assay: assay_groups for assay, assay_groups in groups.items() if assay not in exclude}
    items = list(groups.items())
    if args.limit_assays:
        items = items[:args.limit_assays]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    fieldnames = ['assay', 'seed', 'budget', 'regime', 'model', 'zero_shot_method', 'n_train', 'n_test', 'spearman_all', 'n_test_doubles', 'spearman_doubles', 'n_test_higher', 'spearman_higher', 'ndcg_at_100', 'top1pct_recall_at_100', 'best_true_in_pred_top_100']
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_assay, assay, assay_groups, methods, args.k) for assay, assay_groups in items]
            for future in as_completed(futures):
                writer.writerows(future.result())
    print(f'wrote {out_path}')
if __name__ == '__main__':
    main()
