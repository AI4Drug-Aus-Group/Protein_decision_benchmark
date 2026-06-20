from __future__ import annotations
import argparse
import csv
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
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
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[indexed[k][0]] = avg
        i = j
    return out

def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    mx, my = (sum(x) / len(x), sum(y) / len(y))
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

def load_split_groups() -> Dict[str, List[Tuple[Tuple[str, int, int, str], List[str]]]]:
    groups: Dict[str, Dict[Tuple[str, int, int, str], List[str]]] = {}
    with SPLITS.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            key = (row['assay'], int(row['seed']), int(row['budget']), row['regime'])
            groups.setdefault(row['assay'], {}).setdefault(key, []).append(row['mutant'])
    return {assay: list(v.items()) for assay, v in groups.items()}

def load_assay_table(assay: str, methods: Sequence[str]) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    with (ZERO_DIR / assay).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            y = parse_float(row.get('DMS_score'))
            if y is None:
                continue
            rec = {'DMS_score': y}
            for method in methods:
                z = parse_float(row.get(method))
                if z is not None:
                    rec[method] = z
            table[row['mutant']] = rec
    return table

def additive_predictions(table: Dict[str, Dict[str, float]], train: Sequence[str]) -> Dict[str, float]:
    singles = [m for m in train if mutation_order(m) == 1 and m in table]
    if singles:
        baseline = sum((table[m]['DMS_score'] for m in singles)) / len(singles)
        effects = {m: table[m]['DMS_score'] - baseline for m in singles}
    else:
        ys = [table[m]['DMS_score'] for m in train if m in table]
        baseline = sum(ys) / len(ys) if ys else 0.0
        effects = {}
    return {m: baseline + sum((effects.get(p, 0.0) for p in parts(m))) for m in table}

def feature_vector(mutant: str, table: Dict[str, Dict[str, float]], methods: Sequence[str], add_pred: Dict[str, float]) -> Optional[List[float]]:
    rec = table.get(mutant)
    if rec is None:
        return None
    vec = [float(mutation_order(mutant)), add_pred[mutant]]
    for method in methods:
        if method not in rec:
            return None
        vec.append(rec[method])
    return vec

def model_factory(name: str, seed: int):
    if name == 'ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == 'lasso':
        return make_pipeline(StandardScaler(), Lasso(alpha=0.001, max_iter=5000, random_state=seed))
    if name == 'elastic_net':
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=seed))
    if name == 'random_forest':
        return RandomForestRegressor(n_estimators=24, max_depth=5, min_samples_leaf=2, random_state=seed, n_jobs=1)
    if name == 'hist_gradient_boosting':
        return HistGradientBoostingRegressor(max_iter=30, max_leaf_nodes=15, learning_rate=0.05, random_state=seed)
    if name == 'xgboost':
        return XGBRegressor(n_estimators=40, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9, objective='reg:squarederror', random_state=seed, n_jobs=1, verbosity=0)
    if name == 'gaussian_process':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, alpha=1e-06, normalize_y=True, random_state=seed, optimizer=None))
    raise ValueError(name)

def eval_prediction(assay: str, key: Tuple[str, int, int, str], model: str, features_name: str, table: Dict[str, Dict[str, float]], train_set: set[str], pred: Dict[str, float], k: int) -> Dict[str, object]:
    y_all: List[float] = []
    p_all: List[float] = []
    y_double: List[float] = []
    p_double: List[float] = []
    y_higher: List[float] = []
    p_higher: List[float] = []
    for mutant, rec in table.items():
        if mutant in train_set or mutant not in pred:
            continue
        y, p = (rec['DMS_score'], pred[mutant])
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
    return {'assay': assay, 'seed': seed, 'budget': budget, 'regime': regime, 'model': model, 'zero_shot_method': features_name, 'features': features_name, 'n_train': len(train_set), 'n_test': len(y_all), 'spearman_all': '' if rho_all is None else rho_all, 'n_test_doubles': len(y_double), 'spearman_doubles': '' if rho_double is None else rho_double, 'n_test_higher': len(y_higher), 'spearman_higher': '' if rho_higher is None else rho_higher, 'ndcg_at_100': metrics['ndcg_at_100'], 'top1pct_recall_at_100': metrics['top1pct_recall_at_100'], 'best_true_in_pred_top_100': metrics['best_true_in_pred_top_100']}

def process_assay(assay: str, groups: Sequence[Tuple[Tuple[str, int, int, str], Sequence[str]]], methods: Sequence[str], models: Sequence[str], k: int, max_predict: int) -> List[Dict[str, object]]:
    table = load_assay_table(assay, methods)
    mutants = list(table)
    rows: List[Dict[str, object]] = []
    features_name = '+'.join(methods)
    for key, train_mutants in groups:
        _, seed, budget, _ = key
        train_set = {m for m in train_mutants if m in table}
        if len(train_set) < 5:
            continue
        add_pred = additive_predictions(table, list(train_set))
        train_x, train_y = ([], [])
        for m in train_set:
            vec = feature_vector(m, table, methods, add_pred)
            if vec is not None:
                train_x.append(vec)
                train_y.append(table[m]['DMS_score'])
        if len(train_y) < 5:
            continue
        pred_candidates = [m for m in mutants if m not in train_set]
        if max_predict and len(pred_candidates) > max_predict:
            pred_candidates = pred_candidates[:max_predict]
        pred_x, pred_mutants = ([], [])
        for m in pred_candidates:
            vec = feature_vector(m, table, methods, add_pred)
            if vec is not None:
                pred_x.append(vec)
                pred_mutants.append(m)
        if not pred_x:
            continue
        for model_name in models:
            if model_name == 'gaussian_process' and budget > 50:
                continue
            try:
                model = model_factory(model_name, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model.fit(np.asarray(train_x, dtype=float), np.asarray(train_y, dtype=float))
                y_pred = model.predict(np.asarray(pred_x, dtype=float))
            except Exception:
                continue
            pred = {m: float(p) for m, p in zip(pred_mutants, y_pred)}
            rows.append(eval_prediction(assay, key, model_name, features_name, table, train_set, pred, k))
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--models', default='ridge,lasso,elastic_net,random_forest,hist_gradient_boosting,xgboost')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--max-predict', type=int, default=0, help='Optional deterministic cap on per-split prediction candidates.')
    parser.add_argument('--out', default=str(OUT_DIR / 'low_n_sklearn_baselines_nongiant_metrics.csv'))
    parser.add_argument('--limit-assays', type=int, default=0)
    parser.add_argument('--exclude-assays', default='')
    args = parser.parse_args()
    methods = [m for m in args.methods.split(',') if m]
    models = [m for m in args.models.split(',') if m]
    groups = load_split_groups()
    exclude = {a for a in args.exclude_assays.split(',') if a}
    if exclude:
        groups = {a: g for a, g in groups.items() if a not in exclude}
    items = list(groups.items())
    if args.limit_assays:
        items = items[:args.limit_assays]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ['assay', 'seed', 'budget', 'regime', 'model', 'zero_shot_method', 'features', 'n_train', 'n_test', 'spearman_all', 'n_test_doubles', 'spearman_doubles', 'n_test_higher', 'spearman_higher', 'ndcg_at_100', 'top1pct_recall_at_100', 'best_true_in_pred_top_100']
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_assay, assay, assay_groups, methods, models, args.k, args.max_predict) for assay, assay_groups in items]
            for future in as_completed(futures):
                writer.writerows(future.result())
    print(f'wrote {out}')
if __name__ == '__main__':
    main()
