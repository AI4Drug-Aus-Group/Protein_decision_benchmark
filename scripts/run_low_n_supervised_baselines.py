from __future__ import annotations
import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
SPLITS = ROOT / 'results' / 'splits' / 'low_n_splits.csv'
OUT_DIR = ROOT / 'results' / 'low_n'
DEFAULT_METHODS = ['VenusREM', 'ProSST-2048', 'S3F_MSA', 'S2F_MSA', 'PoET', 'ESM3', 'GEMME', 'TranceptEVE_L']

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

def parts(mutant: str) -> List[str]:
    return [] if mutation_order(mutant) == 0 else mutant.split(':')

def ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            result[indexed[k][0]] = avg
        i = j
    return result

def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum(((a - mx) * (b - my) for a, b in zip(x, y)))
    dx = math.sqrt(sum(((a - mx) ** 2 for a in x)))
    dy = math.sqrt(sum(((b - my) ** 2 for b in y)))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)

def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    return pearson(ranks(x), ranks(y))

def dcg(relevances: Sequence[float]) -> float:
    return sum(((2.0 ** rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(relevances)))

def top_metrics(y_true: Sequence[float], y_pred: Sequence[float], k: int) -> Dict[str, float]:
    if len(y_true) < 3:
        return {'ndcg_at_100': float('nan'), 'top1pct_recall_at_100': float('nan'), 'best_true_in_pred_top_100': float('nan')}
    n = len(y_true)
    k = min(k, n)
    pred_idx = sorted(range(n), key=lambda i: y_pred[i], reverse=True)[:k]
    true_idx = sorted(range(n), key=lambda i: y_true[i], reverse=True)
    lo, hi = (min(y_true), max(y_true))
    rel = [0.0 if hi == lo else (v - lo) / (hi - lo) for v in y_true]
    denom = dcg([rel[i] for i in true_idx[:k]])
    true_top1 = set(true_idx[:max(1, math.ceil(n * 0.01))])
    return {'ndcg_at_100': dcg([rel[i] for i in pred_idx]) / denom if denom else 0.0, 'top1pct_recall_at_100': len(set(pred_idx) & true_top1) / len(true_top1), 'best_true_in_pred_top_100': max((y_true[i] for i in pred_idx))}

def solve_linear(features: Sequence[Sequence[float]], targets: Sequence[float], ridge: float=1.0) -> Optional[List[float]]:
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
            y = parse_float(row.get('DMS_score'))
            if y is None:
                continue
            rec = {'DMS_score': y}
            for method in methods:
                val = parse_float(row.get(method))
                if val is not None:
                    rec[method] = val
            table[row['mutant']] = rec
    return table

def additive_predictions(table: Dict[str, Dict[str, float]], train: Sequence[str]) -> Dict[str, float]:
    train_singles = [m for m in train if mutation_order(m) == 1 and m in table]
    if train_singles:
        baseline = sum((table[m]['DMS_score'] for m in train_singles)) / len(train_singles)
        effects = {m: table[m]['DMS_score'] - baseline for m in train_singles}
    else:
        observed = [table[m]['DMS_score'] for m in train if m in table]
        baseline = sum(observed) / len(observed) if observed else 0.0
        effects = {}
    return {m: baseline + sum((effects.get(p, 0.0) for p in parts(m))) for m in table}

def mutation_vocabulary(train: Sequence[str], max_features: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for mutant in train:
        for part in parts(mutant):
            counts[part] = counts.get(part, 0) + 1
    ranked = sorted(counts, key=lambda p: (-counts[p], p))[:max_features]
    return {part: i for i, part in enumerate(ranked)}

def site_features(mutant: str, vocab: Dict[str, int], add_pred: Optional[float]=None) -> List[float]:
    vec = [1.0]
    if add_pred is not None:
        vec.append(add_pred)
    vec.append(float(mutation_order(mutant)))
    values = [0.0] * len(vocab)
    for part in parts(mutant):
        idx = vocab.get(part)
        if idx is not None:
            values[idx] = 1.0
    vec.extend(values)
    return vec

def score_features(mutant: str, table: Dict[str, Dict[str, float]], methods: Sequence[str], add_pred: Optional[float]=None) -> Optional[List[float]]:
    rec = table.get(mutant)
    if rec is None:
        return None
    vec = [1.0]
    if add_pred is not None:
        vec.append(add_pred)
    vec.append(float(mutation_order(mutant)))
    for method in methods:
        if method not in rec:
            return None
        vec.append(rec[method])
    return vec

def fit_predict(table: Dict[str, Dict[str, float]], train: Sequence[str], model: str, methods: Sequence[str], ridge: float, max_features: int, add_pred: Dict[str, float]) -> Dict[str, float]:
    train = [m for m in train if m in table]
    if model == 'global_mean':
        mu = sum((table[m]['DMS_score'] for m in train)) / len(train)
        return {m: mu for m in table}
    if model == 'mutation_order_mean':
        by_order: Dict[int, List[float]] = {}
        for m in train:
            by_order.setdefault(mutation_order(m), []).append(table[m]['DMS_score'])
        global_mu = sum((table[m]['DMS_score'] for m in train)) / len(train)
        order_mu = {k: sum(v) / len(v) for k, v in by_order.items()}
        return {m: order_mu.get(mutation_order(m), global_mu) for m in table}
    features = []
    targets = []
    pred_features: Dict[str, List[float]] = {}
    if model in {'onehot_site_ridge', 'additive_plus_site_ridge'}:
        vocab = mutation_vocabulary(train, max_features)
        with_add = model == 'additive_plus_site_ridge'
        for m in train:
            features.append(site_features(m, vocab, add_pred[m] if with_add else None))
            targets.append(table[m]['DMS_score'])
        for m in table:
            pred_features[m] = site_features(m, vocab, add_pred[m] if with_add else None)
    elif model in {'top_score_ridge', 'additive_plus_top_score_ridge'}:
        with_add = model == 'additive_plus_top_score_ridge'
        for m in train:
            vec = score_features(m, table, methods, add_pred[m] if with_add else None)
            if vec is None:
                continue
            features.append(vec)
            targets.append(table[m]['DMS_score'])
        for m in table:
            vec = score_features(m, table, methods, add_pred[m] if with_add else None)
            if vec is not None:
                pred_features[m] = vec
    else:
        return {}
    weights = solve_linear(features, targets, ridge=ridge)
    if weights is None:
        return {}
    return {m: dot(weights, vec) for m, vec in pred_features.items()}

def eval_prediction(assay: str, key: Tuple[str, int, int, str], model: str, features: str, table: Dict[str, Dict[str, float]], train_set: set[str], pred: Dict[str, float], k: int) -> Dict[str, object]:
    y_all: List[float] = []
    p_all: List[float] = []
    y_double: List[float] = []
    p_double: List[float] = []
    y_higher: List[float] = []
    p_higher: List[float] = []
    for mutant, rec in table.items():
        if mutant in train_set or mutant not in pred:
            continue
        y = rec['DMS_score']
        p = pred[mutant]
        y_all.append(y)
        p_all.append(p)
        order = mutation_order(mutant)
        if order == 2:
            y_double.append(y)
            p_double.append(p)
        elif order >= 3:
            y_higher.append(y)
            p_higher.append(p)
    rho_all = spearman(p_all, y_all)
    rho_double = spearman(p_double, y_double)
    rho_higher = spearman(p_higher, y_higher)
    metrics = top_metrics(y_all, p_all, k)
    _, seed, budget, regime = key
    return {'assay': assay, 'seed': seed, 'budget': budget, 'regime': regime, 'model': model, 'zero_shot_method': features, 'features': features, 'n_train': len(train_set), 'n_test': len(y_all), 'spearman_all': '' if rho_all is None else rho_all, 'n_test_doubles': len(y_double), 'spearman_doubles': '' if rho_double is None else rho_double, 'n_test_higher': len(y_higher), 'spearman_higher': '' if rho_higher is None else rho_higher, 'ndcg_at_100': metrics['ndcg_at_100'], 'top1pct_recall_at_100': metrics['top1pct_recall_at_100'], 'best_true_in_pred_top_100': metrics['best_true_in_pred_top_100']}

def process_assay(assay: str, assay_groups: Sequence[Tuple[Tuple[str, int, int, str], Sequence[str]]], methods: Sequence[str], ridge: float, max_features: int, k: int) -> List[Dict[str, object]]:
    table = load_assay_table(assay, methods)
    rows: List[Dict[str, object]] = []
    models = ['global_mean', 'mutation_order_mean', 'onehot_site_ridge', 'additive_plus_site_ridge', 'top_score_ridge', 'additive_plus_top_score_ridge']
    feature_name = '+'.join(methods)
    for key, train_mutants in assay_groups:
        train_set = {m for m in train_mutants if m in table}
        if len(train_set) < 3:
            continue
        add_pred = additive_predictions(table, list(train_set))
        for model in models:
            pred = fit_predict(table, list(train_set), model, methods, ridge, max_features, add_pred)
            if not pred:
                continue
            rows.append(eval_prediction(assay, key, model, feature_name if 'score' in model else 'mutation_identity', table, train_set, pred, k))
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--ridge', type=float, default=1.0)
    parser.add_argument('--max-features', type=int, default=300)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--out', default=str(OUT_DIR / 'low_n_supervised_baselines_metrics.csv'))
    parser.add_argument('--limit-assays', type=int, default=0)
    parser.add_argument('--exclude-assays', default='')
    args = parser.parse_args()
    methods = [m for m in args.methods.split(',') if m]
    groups = load_split_groups()
    exclude = {a for a in args.exclude_assays.split(',') if a}
    if exclude:
        groups = {assay: assay_groups for assay, assay_groups in groups.items() if assay not in exclude}
    items = list(groups.items())
    if args.limit_assays:
        items = items[:args.limit_assays]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['assay', 'seed', 'budget', 'regime', 'model', 'zero_shot_method', 'features', 'n_train', 'n_test', 'spearman_all', 'n_test_doubles', 'spearman_doubles', 'n_test_higher', 'spearman_higher', 'ndcg_at_100', 'top1pct_recall_at_100', 'best_true_in_pred_top_100']
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_assay, assay, assay_groups, methods, args.ridge, args.max_features, args.k) for assay, assay_groups in items]
            for future in as_completed(futures):
                writer.writerows(future.result())
    print(f'wrote {out}')
if __name__ == '__main__':
    main()
