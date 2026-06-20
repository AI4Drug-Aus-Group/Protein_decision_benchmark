from __future__ import annotations
import csv
import itertools
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'statistics'
BOOTSTRAPS = 5000
PERMUTATIONS = 50000

def parse_float(value: str) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x

def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float('nan')

def percentile(xs: Sequence[float], q: float) -> float:
    xs = sorted((x for x in xs if x == x))
    if not xs:
        return float('nan')
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

def bootstrap(values: Sequence[float], seed: str) -> tuple[float, float, float]:
    values = [v for v in values if v == v]
    if not values:
        return (float('nan'), float('nan'), float('nan'))
    rng = random.Random(seed)
    sims = [mean((values[rng.randrange(len(values))] for _ in values)) for _ in range(BOOTSTRAPS)]
    return (mean(values), percentile(sims, 0.025), percentile(sims, 0.975))

def paired_pvalue(diffs: Sequence[float], seed: str) -> float:
    diffs = [d for d in diffs if d and d == d]
    if not diffs:
        return 1.0
    obs = abs(mean(diffs))
    if len(diffs) <= 20:
        total = extreme = 0
        for signs in itertools.product((-1, 1), repeat=len(diffs)):
            total += 1
            extreme += int(abs(mean((d * s for d, s in zip(diffs, signs)))) >= obs - 1e-15)
        return extreme / total
    rng = random.Random(seed)
    extreme = 0
    for _ in range(PERMUTATIONS):
        extreme += int(abs(mean((d if rng.randrange(2) else -d for d in diffs))) >= obs - 1e-15)
    return (extreme + 1) / (PERMUTATIONS + 1)

def write(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({k for row in rows for k in row}) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')

def ci_added() -> None:
    rows = []
    configs = [(ROOT / 'results' / 'epistasis' / 'epistatic_uncertainty_enrichment_detail.csv', ['score', 'k'], ['high_abs_residual_enrichment', 'positive_residual_enrichment', 'sign_epistasis_enrichment']), (ROOT / 'results' / 'double_data_value' / 'informative_double_selection_heldout_detail.csv', ['selection_policy', 'budget', 'epistasis_bin'], ['delta_spearman', 'delta_top1pct_recall']), (ROOT / 'results' / 'zero_shot' / 'zero_shot_extended_metrics.csv', ['method'], ['pearson', 'top20_enrichment_top1pct', 'top10_enrichment_top1pct'])]
    for path, group_fields, metric_fields in configs:
        groups: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        with path.open(newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                key = tuple((row[g] for g in group_fields))
                for metric in metric_fields:
                    val = parse_float(row.get(metric, ''))
                    if val is not None:
                        groups[key][metric].append(val)
        for key, metrics in groups.items():
            for metric, vals in metrics.items():
                est, lo, hi = bootstrap(vals, '|'.join(key) + metric)
                rec = {'source': path.name, 'metric': metric, 'n': len(vals), 'mean': est, 'ci95_low': lo, 'ci95_high': hi}
                rec.update({group_fields[i]: key[i] for i in range(len(group_fields))})
                rows.append(rec)
    write(OUT / 'final_added_bootstrap_ci.csv', rows)

def reciprocal_tests() -> None:
    path = ROOT / 'results' / 'epistasis' / 'reciprocal_sign_epistasis_candidates.csv'
    methods = ['VenusREM', 'ProSST-2048', 'additive_prediction', 'ESM3']
    by_method: dict[str, dict[tuple[str, str], float]] = {m: {} for m in methods}
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            key = (row['assay'], row['mutant'])
            target = parse_float(row.get('score_double', ''))
            if target is None:
                continue
            for method in methods:
                pred = parse_float(row.get(method, ''))
                if pred is not None:
                    by_method[method][key] = pred
    assay_rows = []
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            grouped[row['assay']].append(row)
    for assay, rows in grouped.items():
        if len(rows) < 20:
            continue
        y = [parse_float(r['score_double']) for r in rows]
        if any((v is None for v in y)):
            continue
        yr = rank([float(v) for v in y])
        rec = {'assay': assay}
        for method in methods:
            vals = [parse_float(r.get(method, '')) for r in rows]
            if any((v is None for v in vals)):
                continue
            rec[method] = corr(rank([float(v) for v in vals]), yr)
        assay_rows.append(rec)
    tests = []
    for a, b in [('VenusREM', 'additive_prediction'), ('ProSST-2048', 'additive_prediction'), ('VenusREM', 'ProSST-2048')]:
        diffs = [abs(r[a]) - abs(r[b]) for r in assay_rows if a in r and b in r]
        tests.append({'family': 'reciprocal_sign_epistasis', 'method_a': a, 'method_b': b, 'n_pairs': len(diffs), 'mean_diff_abs_spearman': mean(diffs), 'paired_permutation_pvalue': paired_pvalue(diffs, a + b)})
    write(OUT / 'final_added_paired_tests.csv', tests)

def rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    for i, idx in enumerate(order):
        out[idx] = i
    return out

def corr(xs: list[float], ys: list[float]) -> float:
    mx, my = (mean(xs), mean(ys))
    vx = sum(((x - mx) ** 2 for x in xs))
    vy = sum(((y - my) ** 2 for y in ys))
    if vx <= 0 or vy <= 0:
        return float('nan')
    return sum(((x - mx) * (y - my) for x, y in zip(xs, ys))) / math.sqrt(vx * vy)

def main() -> None:
    ci_added()
    reciprocal_tests()
if __name__ == '__main__':
    main()
