from __future__ import annotations
import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
SPLITS = ROOT / 'results' / 'splits' / 'low_n_splits.csv'
OUT_DIR = ROOT / 'results' / 'low_n'
DEFAULT_FEATURES = ['VenusREM', 'ProSST-2048', 'S3F_MSA', 'S2F_MSA', 'PoET', 'RSALOR', 'ESM3', 'GEMME', 'SaProt_650M_AF2']

def parse_float(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed

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

def top_recall(y: Sequence[float], pred: Sequence[float], k: int=100) -> Optional[float]:
    if len(y) < 3:
        return None
    k = min(k, len(y))
    pred_top = set(sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:k])
    true_top = set(sorted(range(len(y)), key=lambda i: y[i], reverse=True)[:max(1, math.ceil(len(y) * 0.01))])
    return len(pred_top & true_top) / len(true_top)

def solve_ridge(features: Sequence[Sequence[float]], targets: Sequence[float], ridge: float) -> Optional[List[float]]:
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

def dot(w: Sequence[float], x: Sequence[float]) -> float:
    return sum((a * b for a, b in zip(w, x)))

def load_groups(exclude: set[str]) -> Dict[str, List[Tuple[Tuple[str, int, int, str], List[str]]]]:
    groups: Dict[str, Dict[Tuple[str, int, int, str], List[str]]] = {}
    with SPLITS.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            assay = row['assay']
            if assay in exclude:
                continue
            key = (assay, int(row['seed']), int(row['budget']), row['regime'])
            groups.setdefault(assay, {}).setdefault(key, []).append(row['mutant'])
    return {assay: list(v.items()) for assay, v in groups.items()}

def load_table(assay: str, feature_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    with (ZERO_DIR / assay).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            y = parse_float(row.get('DMS_score'))
            if y is None:
                continue
            rec = {'DMS_score': y}
            ok = True
            for name in feature_names:
                value = parse_float(row.get(name))
                if value is None:
                    ok = False
                    break
                rec[name] = value
            if ok:
                table[row['mutant']] = rec
    return table

def process_assay(assay: str, groups: Sequence[Tuple[Tuple[str, int, int, str], Sequence[str]]], feature_names: Sequence[str], ridge: float) -> List[Dict[str, object]]:
    table = load_table(assay, feature_names)
    rows: List[Dict[str, object]] = []
    for key, train_mutants in groups:
        train = [m for m in train_mutants if m in table]
        train_set = set(train)
        x_train = [[1.0] + [table[m][name] for name in feature_names] for m in train]
        y_train = [table[m]['DMS_score'] for m in train]
        if len(y_train) < len(feature_names) + 2:
            continue
        weights = solve_ridge(x_train, y_train, ridge)
        if weights is None:
            continue
        y = []
        pred = []
        for mutant, rec in table.items():
            if mutant in train_set:
                continue
            y.append(rec['DMS_score'])
            pred.append(dot(weights, [1.0] + [rec[name] for name in feature_names]))
        rho = spearman(pred, y)
        recall = top_recall(y, pred)
        _, seed, budget, regime = key
        rows.append({'assay': assay, 'seed': seed, 'budget': budget, 'regime': regime, 'model': 'multi_zscore_ridge', 'features': '|'.join(feature_names), 'n_features': len(feature_names), 'ridge': ridge, 'n_train': len(train), 'n_test': len(y), 'spearman_all': '' if rho is None else rho, 'top1pct_recall_at_100': '' if recall is None else recall, 'best_true_in_pred_top_100': max((y[i] for i in sorted(range(len(pred)), key=lambda j: pred[j], reverse=True)[:min(100, len(pred))])) if y else ''})
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default=','.join(DEFAULT_FEATURES))
    parser.add_argument('--ridge', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--exclude-assays', default='')
    parser.add_argument('--out', default=str(OUT_DIR / 'low_n_multi_zscore_ridge_nongiant_metrics.csv'))
    args = parser.parse_args()
    feature_names = [x for x in args.features.split(',') if x]
    exclude = {x for x in args.exclude_assays.split(',') if x}
    groups = load_groups(exclude)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    fieldnames = ['assay', 'seed', 'budget', 'regime', 'model', 'features', 'n_features', 'ridge', 'n_train', 'n_test', 'spearman_all', 'top1pct_recall_at_100', 'best_true_in_pred_top_100']
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_assay, assay, assay_groups, feature_names, args.ridge) for assay, assay_groups in groups.items()]
            for future in as_completed(futures):
                writer.writerows(future.result())
    print(f'wrote {out}')
if __name__ == '__main__':
    main()
