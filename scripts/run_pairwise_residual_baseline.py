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
DMS_DIR = ROOT / 'proteingym' / 'extracted' / 'DMS_ProteinGym_substitutions'
MANIFEST = ROOT / 'results' / 'batch0' / 'assay_manifest.csv'
OUT_DIR = ROOT / 'results' / 'multi_evolve_style'

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

def load_higher_assays() -> List[str]:
    assays = []
    with MANIFEST.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if int(row['n_higher_order']) > 0:
                assays.append(row['assay'])
    return assays

def load_scores(assay: str) -> Dict[str, float]:
    scores = {}
    with (DMS_DIR / assay).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            try:
                scores[row['mutant']] = float(row['DMS_score'])
            except ValueError:
                continue
    return scores

def additive_model(scores: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    singles = {m: y for m, y in scores.items() if mutation_order(m) == 1}
    wt_proxy = sum(singles.values()) / len(singles)
    effects = {m: y - wt_proxy for m, y in singles.items()}
    return (wt_proxy, effects)

def additive_predict(mutant: str, wt_proxy: float, effects: Dict[str, float]) -> float:
    return wt_proxy + sum((effects.get(part, 0.0) for part in parts(mutant)))

def pairwise_predict(mutant: str, wt_proxy: float, effects: Dict[str, float], pair_residuals: Dict[str, float]) -> float:
    base = additive_predict(mutant, wt_proxy, effects)
    ps = parts(mutant)
    residuals = [pair_residuals[pair_key(a, b)] for a, b in itertools.combinations(ps, 2) if pair_key(a, b) in pair_residuals]
    if not residuals:
        return base
    return base + sum(residuals) / len(residuals)

def informative_doubles(doubles: Sequence[str], higher: Sequence[str]) -> List[str]:
    pair_freq: Dict[str, int] = {}
    for mutant in higher:
        for a, b in itertools.combinations(parts(mutant), 2):
            key = pair_key(a, b)
            pair_freq[key] = pair_freq.get(key, 0) + 1
    return sorted(doubles, key=lambda m: pair_freq.get(pair_key(*parts(m)), 0), reverse=True)

def process_assay(assay: str, seeds: Sequence[int], double_budgets: Sequence[int]) -> List[Dict[str, object]]:
    scores = load_scores(assay)
    wt_proxy, effects = additive_model(scores)
    doubles = [m for m in scores if mutation_order(m) == 2 and all((p in effects for p in parts(m)))]
    higher = [m for m in scores if mutation_order(m) >= 3]
    rows: List[Dict[str, object]] = []
    y_higher = [scores[m] for m in higher]
    add_pred_higher = [additive_predict(m, wt_proxy, effects) for m in higher]
    add_rho = spearman(add_pred_higher, y_higher)
    add_recall = top_recall(y_higher, add_pred_higher)
    rows.append({'assay': assay, 'seed': 'all', 'double_budget': 0, 'model': 'additive_all_singles', 'selection_policy': 'none', 'n_train_doubles': 0, 'n_test_higher': len(higher), 'spearman_higher': '' if add_rho is None else add_rho, 'abs_spearman_higher': '' if add_rho is None else abs(add_rho), 'top1pct_recall_at_100': '' if add_recall is None else add_recall, 'pair_coverage': 0.0})
    informative_order = informative_doubles(doubles, higher)
    for seed in seeds:
        for budget in double_budgets:
            rng = random.Random(f'{assay}|{seed}|{budget}|pairwise')
            random_sampled = rng.sample(doubles, min(budget, len(doubles)))
            informed_sampled = informative_order[:min(budget, len(informative_order))]
            for selection_policy, sampled in [('random_doubles', random_sampled), ('informative_pair_coverage', informed_sampled)]:
                pair_residuals = {}
                for m in sampled:
                    ps = parts(m)
                    pred = additive_predict(m, wt_proxy, effects)
                    pair_residuals[pair_key(ps[0], ps[1])] = scores[m] - pred
                pred_higher = [pairwise_predict(m, wt_proxy, effects, pair_residuals) for m in higher]
                rho = spearman(pred_higher, y_higher)
                recall = top_recall(y_higher, pred_higher)
                total_pairs = 0
                covered_pairs = 0
                for m in higher:
                    ps = parts(m)
                    for a, b in itertools.combinations(ps, 2):
                        total_pairs += 1
                        covered_pairs += int(pair_key(a, b) in pair_residuals)
                rows.append({'assay': assay, 'seed': seed, 'double_budget': budget, 'model': 'pairwise_residual_from_doubles', 'selection_policy': selection_policy, 'n_train_doubles': len(sampled), 'n_test_higher': len(higher), 'spearman_higher': '' if rho is None else rho, 'abs_spearman_higher': '' if rho is None else abs(rho), 'top1pct_recall_at_100': '' if recall is None else recall, 'pair_coverage': covered_pairs / total_pairs if total_pairs else 0.0})
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--double-budgets', default='20,50,100')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', default=str(OUT_DIR / 'pairwise_residual_higher_order_metrics.csv'))
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
