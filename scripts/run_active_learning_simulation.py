from __future__ import annotations
import argparse
import bisect
import csv
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
ASSAYS = ROOT / 'results' / 'active_learning' / 'representative_assays.csv'
OUT_DIR = ROOT / 'results' / 'active_learning'
METHODS = ['ProSST-2048', 'ESM3', 'S2F_MSA', 'S3F_MSA', 'PoET', 'VenusREM']
ENSEMBLE_METHODS = ['ProSST-2048', 'ESM3', 'S2F_MSA', 'S3F_MSA', 'VenusREM']
POLICIES = ['random', 'additive_lookup', 'zscore_linear:ProSST-2048', 'zscore_linear:ESM3', 'additive_plus_zscore:ProSST-2048', 'disagreement:additive_vs_ProSST-2048', 'ensemble_mean:top_methods', 'ensemble_ucb:top_methods', 'diverse_ensemble:top_methods', 'diverse_zscore:ESM3', 'ensemble_rank_mean_std:top_methods', 'ensemble_rank_mean_disagreement:top_methods']

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

def mutation_sites(mutant: str) -> set[str]:
    if not mutant or mutant in {'WT', 'wildtype', '_wt'}:
        return set()
    return {part[1:-1] for part in mutant.split(':') if len(part) >= 3}

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

def load_table(assay: str) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    with (ZERO_DIR / assay).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            y = parse_float(row.get('DMS_score'))
            if y is None:
                continue
            rec = {'DMS_score': y}
            for method in METHODS:
                z = parse_float(row.get(method))
                if z is not None:
                    rec[method] = z
            table[row['mutant']] = rec
    return table

def candidate_pool(table: Dict[str, Dict[str, float]], assay: str, max_candidates: int, retain_top: bool=True) -> List[str]:
    mutants = list(table)
    if len(mutants) <= max_candidates:
        return mutants
    rng = random.Random(f'{assay}|candidate_pool|retain_top={retain_top}')
    if not retain_top:
        return rng.sample(mutants, max_candidates)
    top_true = sorted(mutants, key=lambda m: table[m]['DMS_score'], reverse=True)[:max(500, max_candidates // 100)]
    remaining = [m for m in mutants if m not in set(top_true)]
    sampled = rng.sample(remaining, max_candidates - len(top_true))
    return top_true + sampled

def additive_pred(table: Dict[str, Dict[str, float]], observed: set[str], pool: Sequence[str]) -> Dict[str, float]:
    singles = [m for m in observed if mutation_order(m) == 1 and m in table]
    if singles:
        mean_single = sum((table[m]['DMS_score'] for m in singles)) / len(singles)
        effects = {m: table[m]['DMS_score'] - mean_single for m in singles}
    else:
        ys = [table[m]['DMS_score'] for m in observed if m in table]
        mean_single = sum(ys) / len(ys) if ys else 0.0
        effects = {}
    return {m: mean_single + sum((effects.get(part, 0.0) for part in m.split(':'))) for m in pool}

def calibrated_pred(table: Dict[str, Dict[str, float]], observed: set[str], pool: Sequence[str], method: str, add: Optional[Dict[str, float]]=None) -> Dict[str, float]:
    features = []
    targets = []
    for mutant in observed:
        if mutant not in table or method not in table[mutant]:
            continue
        if add is None:
            features.append([1.0, table[mutant][method]])
        else:
            features.append([1.0, add[mutant], table[mutant][method]])
        targets.append(table[mutant]['DMS_score'])
    weights = solve_linear(features, targets)
    if weights is None:
        return {m: 0.0 for m in pool}
    if add is None:
        return {m: dot(weights, [1.0, table[m][method]]) for m in pool if method in table[m]}
    return {m: dot(weights, [1.0, add[m], table[m][method]]) for m in pool if method in table[m] and m in add}

def ensemble_pred(table: Dict[str, Dict[str, float]], observed: set[str], pool: Sequence[str], methods: Sequence[str], ucb_weight: float=0.0) -> Dict[str, float]:
    per_method = [calibrated_pred(table, observed, pool, method) for method in methods]
    score: Dict[str, float] = {}
    for mutant in pool:
        vals = [pred[mutant] for pred in per_method if mutant in pred]
        if not vals:
            continue
        mean_val = sum(vals) / len(vals)
        if len(vals) > 1:
            variance = sum(((v - mean_val) ** 2 for v in vals)) / (len(vals) - 1)
            std_val = math.sqrt(max(variance, 0.0))
        else:
            std_val = 0.0
        score[mutant] = mean_val + ucb_weight * std_val
    return score

def rank_scale(score: Dict[str, float], available: Sequence[str]) -> Dict[str, float]:
    vals = [(m, score[m]) for m in available if m in score and math.isfinite(score[m])]
    if not vals:
        return {m: 0.0 for m in available}
    vals.sort(key=lambda x: x[1])
    denom = max(1, len(vals) - 1)
    out = {m: i / denom for i, (m, _) in enumerate(vals)}
    return {m: out.get(m, 0.0) for m in available}

def ensemble_components(table: Dict[str, Dict[str, float]], observed: set[str], pool: Sequence[str], methods: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    per_method = [calibrated_pred(table, observed, pool, method) for method in methods]
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for mutant in pool:
        vals = [pred[mutant] for pred in per_method if mutant in pred]
        if not vals:
            continue
        mean_val = sum(vals) / len(vals)
        means[mutant] = mean_val
        if len(vals) > 1:
            variance = sum(((v - mean_val) ** 2 for v in vals)) / (len(vals) - 1)
            stds[mutant] = math.sqrt(max(variance, 0.0))
        else:
            stds[mutant] = 0.0
    return (means, stds)

def diverse_top_batch(available: Sequence[str], score: Dict[str, float], batch: int, penalty: float=0.02) -> List[str]:
    ranked = sorted(available, key=lambda m: score.get(m, float('-inf')), reverse=True)
    if not ranked:
        return []
    scale_vals = [score.get(m, 0.0) for m in ranked if math.isfinite(score.get(m, float('nan')))]
    score_scale = max(scale_vals) - min(scale_vals) if len(scale_vals) > 1 else 1.0
    chosen: List[str] = []
    chosen_sites: set[str] = set()
    while ranked and len(chosen) < batch:
        best = max(ranked, key=lambda m: score.get(m, float('-inf')) - penalty * score_scale * len(mutation_sites(m) & chosen_sites))
        chosen.append(best)
        chosen_sites.update(mutation_sites(best))
        ranked.remove(best)
    return chosen

def select_batch(policy: str, table: Dict[str, Dict[str, float]], observed: set[str], pool: Sequence[str], batch: int, rng: random.Random) -> List[str]:
    available = [m for m in pool if m not in observed]
    if not available:
        return []
    if policy == 'random':
        return rng.sample(available, min(batch, len(available)))
    add = additive_pred(table, observed, list(set(pool) | observed))
    if policy == 'additive_lookup':
        score = add
    elif policy.startswith('zscore_linear:'):
        method = policy.split(':', 1)[1]
        score = calibrated_pred(table, observed, list(set(pool) | observed), method)
    elif policy.startswith('additive_plus_zscore:'):
        method = policy.split(':', 1)[1]
        score = calibrated_pred(table, observed, list(set(pool) | observed), method, add)
    elif policy.startswith('disagreement:'):
        method = 'ProSST-2048'
        zpred = calibrated_pred(table, observed, list(set(pool) | observed), method)
        score = {m: abs(add.get(m, 0.0) - zpred.get(m, 0.0)) for m in available}
    elif policy.startswith('ensemble_mean:'):
        score = ensemble_pred(table, observed, list(set(pool) | observed), ENSEMBLE_METHODS, ucb_weight=0.0)
    elif policy.startswith('ensemble_ucb:'):
        score = ensemble_pred(table, observed, list(set(pool) | observed), ENSEMBLE_METHODS, ucb_weight=0.5)
    elif policy.startswith('diverse_ensemble:'):
        score = ensemble_pred(table, observed, list(set(pool) | observed), ENSEMBLE_METHODS, ucb_weight=0.0)
        return diverse_top_batch(available, score, min(batch, len(available)))
    elif policy.startswith('diverse_zscore:'):
        method = policy.split(':', 1)[1]
        score = calibrated_pred(table, observed, list(set(pool) | observed), method)
        return diverse_top_batch(available, score, min(batch, len(available)))
    elif policy.startswith('ensemble_rank_mean_std:'):
        pred_pool = list(set(pool) | observed)
        means, stds = ensemble_components(table, observed, pred_pool, ENSEMBLE_METHODS)
        mean_rank = rank_scale(means, available)
        std_rank = rank_scale(stds, available)
        score = {m: mean_rank.get(m, 0.0) + 0.5 * std_rank.get(m, 0.0) for m in available}
    elif policy.startswith('ensemble_rank_mean_disagreement:'):
        pred_pool = list(set(pool) | observed)
        means, _ = ensemble_components(table, observed, pred_pool, ENSEMBLE_METHODS)
        disagreement = {m: abs(add.get(m, 0.0) - means.get(m, 0.0)) for m in available}
        mean_rank = rank_scale(means, available)
        dis_rank = rank_scale(disagreement, available)
        score = {m: mean_rank.get(m, 0.0) + 0.5 * dis_rank.get(m, 0.0) for m in available}
    else:
        score = {m: 0.0 for m in available}
    ranked = sorted(available, key=lambda m: score.get(m, float('-inf')), reverse=True)
    return ranked[:batch]

def initial_observed(pool: Sequence[str], assay: str, seed: int, n: int) -> set[str]:
    rng = random.Random(f'{assay}|{seed}|initial')
    return set(rng.sample(list(pool), min(n, len(pool))))

def simulate_assay(assay: str, seeds: Sequence[int], rounds: int, batch: int, max_candidates: int, retain_top: bool=True) -> List[Dict[str, object]]:
    return simulate_assay_with_policies(assay, seeds, rounds, batch, max_candidates, retain_top, POLICIES)

def simulate_assay_with_policies(assay: str, seeds: Sequence[int], rounds: int, batch: int, max_candidates: int, retain_top: bool, policies: Sequence[str]) -> List[Dict[str, object]]:
    table = load_table(assay)
    pool = candidate_pool(table, assay, max_candidates, retain_top=retain_top)
    true_sorted = sorted(pool, key=lambda m: table[m]['DMS_score'], reverse=True)
    true_scores_ascending = sorted((table[m]['DMS_score'] for m in pool))
    true_top1 = set(true_sorted[:max(1, math.ceil(len(pool) * 0.01))])
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        for policy in policies:
            rng = random.Random(f'{assay}|{seed}|{policy}')
            observed = initial_observed(pool, assay, seed, 20)
            for round_idx in range(rounds + 1):
                best = max((table[m]['DMS_score'] for m in observed))
                best_percentile = bisect.bisect_right(true_scores_ascending, best) / len(true_scores_ascending)
                top1_found = int(bool(observed & true_top1))
                rows.append({'assay': assay, 'seed': seed, 'policy': policy, 'round': round_idx, 'n_observed': len(observed), 'best_observed_fitness': best, 'best_observed_percentile': best_percentile, 'top1pct_found': top1_found, 'pool_size': len(pool), 'retain_top_candidates': int(retain_top)})
                if round_idx == rounds:
                    break
                observed.update(select_batch(policy, table, observed, pool, batch, rng))
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--max-candidates', type=int, default=50000)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--policies', default='')
    parser.add_argument('--no-retain-top-candidates', action='store_true')
    parser.add_argument('--out', default=str(OUT_DIR / 'active_learning_representative_metrics.csv'))
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(',') if x]
    policies = [x for x in args.policies.split(',') if x] if args.policies else POLICIES
    with ASSAYS.open(newline='', encoding='utf-8') as fh:
        assays = [row['assay'] for row in csv.DictReader(fh)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ['assay', 'seed', 'policy', 'round', 'n_observed', 'best_observed_fitness', 'best_observed_percentile', 'top1pct_found', 'pool_size', 'retain_top_candidates']
    out = Path(args.out)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(simulate_assay_with_policies, assay, seeds, args.rounds, args.batch, args.max_candidates, not args.no_retain_top_candidates, policies) for assay in assays]
            for future in as_completed(futures):
                writer.writerows(future.result())
    print(f'wrote {out}')
if __name__ == '__main__':
    main()
