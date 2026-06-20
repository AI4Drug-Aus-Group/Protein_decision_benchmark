from __future__ import annotations
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESID = ROOT / 'results' / 'epistasis' / 'strict_double_epistasis_residuals.csv'
OUT_DIR = ROOT / 'results' / 'double_data_value'
BUDGETS = [20, 50, 100]
SEEDS = [0, 1, 2, 3, 4]

def parse_float(x: str) -> float:
    return float(x)

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float('nan')

def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float('nan')
    mx, my = (mean(xs), mean(ys))
    vx = sum(((x - mx) ** 2 for x in xs))
    vy = sum(((y - my) ** 2 for y in ys))
    if vx <= 0 or vy <= 0:
        return float('nan')
    return sum(((x - mx) * (y - my) for x, y in zip(xs, ys))) / math.sqrt(vx * vy)

def rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks

def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rankdata(xs), rankdata(ys))

def top1_recall(pred: list[float], true: list[float], k: int=100) -> float:
    if not pred:
        return float('nan')
    top_true = set(sorted(range(len(true)), key=lambda i: true[i], reverse=True)[:max(1, math.ceil(len(true) * 0.01))])
    top_pred = set(sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:min(k, len(pred))])
    return len(top_true & top_pred) / len(top_true)

def residual_bin(rows: list[dict[str, float]]) -> str:
    vals = [r['abs_epistasis_residual'] for r in rows]
    m = mean(vals)
    return 'high_epistasis' if m >= 1.0 else 'low_mid_epistasis'

def site_pair(mutant: str) -> str:
    sites = sorted((part[1:-1] for part in mutant.split(':') if len(part) >= 3))
    return ':'.join(sites)

def fit_pair_residual(train: list[dict[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in train:
        grouped[site_pair(str(r['mutant']))].append(r['epistasis_residual'])
    return {k: mean(v) for k, v in grouped.items()}

def predict(rows: list[dict[str, float]], pair_resid: dict[str, float], fallback: float) -> list[float]:
    return [r['additive_prediction'] + pair_resid.get(site_pair(str(r['mutant'])), fallback) for r in rows]

def evaluate(assay: str, rows: list[dict[str, float]], budget: int, seed: int, policy: str) -> dict[str, object] | None:
    if len(rows) < budget + 50:
        return None
    rng = random.Random(f'{assay}|{budget}|{seed}|{policy}')
    if policy == 'random_doubles':
        train = rng.sample(rows, budget)
    elif policy == 'high_additive_disagreement_proxy':
        additive_mean = mean([x['additive_prediction'] for x in rows])
        train = sorted(rows, key=lambda r: abs(r['additive_prediction'] - additive_mean), reverse=True)[:budget]
    elif policy == 'coverage_proxy':
        shuffled = rows[:]
        rng.shuffle(shuffled)
        seen_sites: set[str] = set()
        train = []
        for r in sorted(shuffled, key=lambda x: len(set(x['mutant'].split(':')) - seen_sites), reverse=True):
            train.append(r)
            for part in r['mutant'].split(':'):
                seen_sites.add(part[1:-1])
            if len(train) >= budget:
                break
    elif policy == 'oracle_high_residual':
        train = sorted(rows, key=lambda r: r['abs_epistasis_residual'], reverse=True)[:budget]
    else:
        return None
    train_set = {r['mutant'] for r in train}
    test = [r for r in rows if r['mutant'] not in train_set]
    fallback = mean([r['epistasis_residual'] for r in train])
    pred_add = [r['additive_prediction'] for r in test]
    pred_pair = predict(test, fit_pair_residual(train), fallback)
    true = [r['score_double'] for r in test]
    return {'assay': assay, 'seed': seed, 'budget': budget, 'selection_policy': policy, 'epistasis_bin': residual_bin(rows), 'n_train': len(train), 'n_test': len(test), 'spearman_additive': spearman(pred_add, true), 'spearman_pairwise': spearman(pred_pair, true), 'delta_spearman': spearman(pred_pair, true) - spearman(pred_add, true), 'top1pct_recall_additive': top1_recall(pred_add, true), 'top1pct_recall_pairwise': top1_recall(pred_pair, true), 'delta_top1pct_recall': top1_recall(pred_pair, true) - top1_recall(pred_add, true)}

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) if rows else []
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')

def main() -> None:
    by_assay: dict[str, list[dict[str, float]]] = defaultdict(list)
    with RESID.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            by_assay[row['assay']].append({'mutant': row['mutant'], 'score_double': parse_float(row['score_double']), 'additive_prediction': parse_float(row['additive_prediction']), 'epistasis_residual': parse_float(row['epistasis_residual']), 'abs_epistasis_residual': parse_float(row['abs_epistasis_residual'])})
    detail = []
    for assay, rows in by_assay.items():
        for budget in BUDGETS:
            for seed in SEEDS:
                for policy in ['random_doubles', 'coverage_proxy', 'high_additive_disagreement_proxy', 'oracle_high_residual']:
                    rec = evaluate(assay, rows, budget, seed, policy)
                    if rec:
                        detail.append(rec)
    write_csv(OUT_DIR / 'informative_double_selection_heldout_detail.csv', detail)
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for r in detail:
        grouped[str(r['budget']), str(r['selection_policy']), str(r['epistasis_bin'])].append(r)
    summary = []
    for (budget, policy, epi_bin), rows in grouped.items():
        summary.append({'budget': budget, 'selection_policy': policy, 'epistasis_bin': epi_bin, 'n_runs': len(rows), 'mean_delta_spearman': mean([float(r['delta_spearman']) for r in rows if not math.isnan(float(r['delta_spearman']))]), 'mean_delta_top1pct_recall': mean([float(r['delta_top1pct_recall']) for r in rows if not math.isnan(float(r['delta_top1pct_recall']))]), 'mean_pairwise_spearman': mean([float(r['spearman_pairwise']) for r in rows if not math.isnan(float(r['spearman_pairwise']))])})
    summary.sort(key=lambda r: (int(r['budget']), r['epistasis_bin'], r['selection_policy']))
    write_csv(OUT_DIR / 'informative_double_selection_heldout_summary.csv', summary)
if __name__ == '__main__':
    main()
