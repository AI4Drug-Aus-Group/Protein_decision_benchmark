from __future__ import annotations
import argparse
import csv
import itertools
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
MANIFEST = ROOT / 'results' / 'batch0' / 'assay_manifest.csv'
OUT_DIR = ROOT / 'results' / 'multi_evolve_style'
PLM_METHODS = ['ProSST-2048', 'ESM3', 'S2F_MSA', 'VenusREM']

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

def pair_key(a: str, b: str) -> str:
    return ':'.join(sorted([a, b]))

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

def solve_linear(features: Sequence[Sequence[float]], targets: Sequence[float], ridge: float=1e-06) -> Optional[List[float]]:
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

def load_higher_assays() -> List[str]:
    assays = []
    with MANIFEST.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if int(row['n_higher_order']) > 0:
                assays.append(row['assay'])
    return assays

def load_table(assay: str) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    with (ZERO_DIR / assay).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            y = parse_float(row.get('DMS_score'))
            if y is None:
                continue
            rec = {'DMS_score': y}
            for method in PLM_METHODS:
                z = parse_float(row.get(method))
                if z is not None:
                    rec[method] = z
            table[row['mutant']] = rec
    return table

def additive_model(table: Dict[str, Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
    singles = {m: rec['DMS_score'] for m, rec in table.items() if mutation_order(m) == 1}
    wt_proxy = sum(singles.values()) / len(singles)
    effects = {m: y - wt_proxy for m, y in singles.items()}
    return (wt_proxy, effects)

def additive_predict(mutant: str, wt_proxy: float, effects: Dict[str, float]) -> float:
    return wt_proxy + sum((effects.get(part, 0.0) for part in parts(mutant)))

def pairwise_predict(mutant: str, wt_proxy: float, effects: Dict[str, float], residuals: Dict[str, float]) -> float:
    base = additive_predict(mutant, wt_proxy, effects)
    vals = [residuals[pair_key(a, b)] for a, b in itertools.combinations(parts(mutant), 2) if pair_key(a, b) in residuals]
    return base if not vals else base + sum(vals) / len(vals)

def informative_doubles(doubles: Sequence[str], higher: Sequence[str]) -> List[str]:
    pair_freq: Dict[str, int] = {}
    for mutant in higher:
        for a, b in itertools.combinations(parts(mutant), 2):
            key = pair_key(a, b)
            pair_freq[key] = pair_freq.get(key, 0) + 1
    return sorted(doubles, key=lambda m: pair_freq.get(pair_key(*parts(m)), 0), reverse=True)

def feature_vector(mutant: str, table: Dict[str, Dict[str, float]], base_pred: float, method_set: Sequence[str]) -> Optional[List[float]]:
    rec = table.get(mutant)
    if rec is None:
        return None
    values = [1.0, base_pred]
    for method in method_set:
        if method not in rec:
            return None
        values.append(rec[method])
    return values

def calibrated_predictions(train: Sequence[str], test: Sequence[str], table: Dict[str, Dict[str, float]], base_by_mutant: Dict[str, float], method_set: Sequence[str]) -> Dict[str, float]:
    features = []
    targets = []
    for mutant in train:
        vec = feature_vector(mutant, table, base_by_mutant[mutant], method_set)
        if vec is None:
            continue
        features.append(vec)
        targets.append(table[mutant]['DMS_score'])
    weights = solve_linear(features, targets)
    if weights is None:
        return {}
    preds = {}
    for mutant in test:
        vec = feature_vector(mutant, table, base_by_mutant[mutant], method_set)
        if vec is not None:
            preds[mutant] = dot(weights, vec)
    return preds

def metric_row(assay: str, seed: object, budget: int, model: str, selection_policy: str, n_train_doubles: int, y_by_mutant: Dict[str, float], pred_by_mutant: Dict[str, float], pair_coverage: float) -> Dict[str, object]:
    common = [m for m in y_by_mutant if m in pred_by_mutant]
    y = [y_by_mutant[m] for m in common]
    pred = [pred_by_mutant[m] for m in common]
    rho = spearman(pred, y)
    recall = top_recall(y, pred)
    return {'assay': assay, 'seed': seed, 'double_budget': budget, 'model': model, 'selection_policy': selection_policy, 'n_train_doubles': n_train_doubles, 'n_test_higher': len(common), 'spearman_higher': '' if rho is None else rho, 'abs_spearman_higher': '' if rho is None else abs(rho), 'top1pct_recall_at_100': '' if recall is None else recall, 'pair_coverage': pair_coverage}

def process_assay(assay: str, seeds: Sequence[int], double_budgets: Sequence[int]) -> List[Dict[str, object]]:
    table = load_table(assay)
    wt_proxy, effects = additive_model(table)
    doubles = [m for m in table if mutation_order(m) == 2 and all((p in effects for p in parts(m)))]
    higher = [m for m in table if mutation_order(m) >= 3]
    singles = [m for m in table if mutation_order(m) == 1]
    y_higher = {m: table[m]['DMS_score'] for m in higher}
    additive_base = {m: additive_predict(m, wt_proxy, effects) for m in singles + doubles + higher}
    rows: List[Dict[str, object]] = [metric_row(assay, 'all', 0, 'additive_all_singles', 'none', 0, y_higher, {m: additive_base[m] for m in higher}, 0.0)]
    for method in PLM_METHODS:
        preds = calibrated_predictions(singles, higher, table, additive_base, [method])
        rows.append(metric_row(assay, 'all', 0, f'additive_plus_zscore:{method}', 'singles_only', 0, y_higher, preds, 0.0))
    ensemble_preds = calibrated_predictions(singles, higher, table, additive_base, PLM_METHODS)
    rows.append(metric_row(assay, 'all', 0, 'additive_plus_plm_ensemble', 'singles_only', 0, y_higher, ensemble_preds, 0.0))
    informative_order = informative_doubles(doubles, higher)
    for seed in seeds:
        for budget in double_budgets:
            rng = random.Random(f'{assay}|{seed}|{budget}|pairwise_plm')
            sampled_sets = [('random_doubles', rng.sample(doubles, min(budget, len(doubles)))), ('informative_pair_coverage', informative_order[:min(budget, len(informative_order))])]
            for selection_policy, sampled in sampled_sets:
                residuals = {}
                for mutant in sampled:
                    ps = parts(mutant)
                    residuals[pair_key(ps[0], ps[1])] = table[mutant]['DMS_score'] - additive_base[mutant]
                pair_base = {m: pairwise_predict(m, wt_proxy, effects, residuals) for m in singles + doubles + higher}
                total_pairs = 0
                covered_pairs = 0
                for mutant in higher:
                    for a, b in itertools.combinations(parts(mutant), 2):
                        total_pairs += 1
                        covered_pairs += int(pair_key(a, b) in residuals)
                coverage = covered_pairs / total_pairs if total_pairs else 0.0
                train = singles + sampled
                pair_preds = {m: pair_base[m] for m in higher}
                rows.append(metric_row(assay, seed, budget, 'pairwise_residual_from_doubles', selection_policy, len(sampled), y_higher, pair_preds, coverage))
                for method in PLM_METHODS:
                    preds = calibrated_predictions(train, higher, table, pair_base, [method])
                    rows.append(metric_row(assay, seed, budget, f'pairwise_plus_zscore:{method}', selection_policy, len(sampled), y_higher, preds, coverage))
                preds = calibrated_predictions(train, higher, table, pair_base, PLM_METHODS)
                rows.append(metric_row(assay, seed, budget, 'pairwise_plus_plm_ensemble', selection_policy, len(sampled), y_higher, preds, coverage))
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--double-budgets', default='20,50,100')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', default=str(OUT_DIR / 'pairwise_plm_residual_higher_order_metrics.csv'))
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(',') if x]
    budgets = [int(x) for x in args.double_budgets.split(',') if x]
    assays = load_higher_assays()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    fieldnames = ['assay', 'seed', 'double_budget', 'model', 'selection_policy', 'n_train_doubles', 'n_test_higher', 'spearman_higher', 'abs_spearman_higher', 'top1pct_recall_at_100', 'pair_coverage']
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_assay, assay, seeds, budgets) for assay in assays]
            for future in as_completed(futures):
                writer.writerows(future.result())
    print(f'wrote {out}')
if __name__ == '__main__':
    main()
