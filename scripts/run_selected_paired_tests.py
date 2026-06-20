from __future__ import annotations
import csv
import itertools
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'statistics' / 'selected_paired_tests.csv'
RANDOM_PERMUTATIONS = 50000

def parse_float(value: str) -> float | None:
    if value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None

def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float('nan')

def paired_permutation_pvalue(diffs: Sequence[float], seed: str) -> float:
    diffs = [d for d in diffs if d != 0]
    if not diffs:
        return 1.0
    observed = abs(mean(diffs))
    if len(diffs) <= 20:
        total = 0
        extreme = 0
        for signs in itertools.product((-1, 1), repeat=len(diffs)):
            stat = abs(mean((d * s for d, s in zip(diffs, signs))))
            total += 1
            extreme += int(stat >= observed - 1e-15)
        return extreme / total
    rng = random.Random(seed)
    extreme = 0
    for _ in range(RANDOM_PERMUTATIONS):
        stat = abs(mean((d if rng.randrange(2) else -d for d in diffs)))
        extreme += int(stat >= observed - 1e-15)
    return (extreme + 1) / (RANDOM_PERMUTATIONS + 1)

def load_metric_map(path: Path, key_fields: Sequence[str], group_field: str, metric_field: str, filters: Dict[str, str] | None=None) -> Dict[str, Dict[Tuple[str, ...], float]]:
    out: Dict[str, Dict[Tuple[str, ...], float]] = {}
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if filters and any((row.get(k) != v for k, v in filters.items())):
                continue
            value = parse_float(row.get(metric_field, ''))
            if value is None:
                continue
            out.setdefault(row[group_field], {})[tuple((row[k] for k in key_fields))] = value
    return out

def compare(rows: List[Dict[str, object]], family: str, metric: str, data: Dict[str, Dict[Tuple[str, ...], float]], method_a: str, method_b: str) -> None:
    a = data.get(method_a, {})
    b = data.get(method_b, {})
    keys = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in keys]
    pvalue = paired_permutation_pvalue(diffs, f'{family}|{metric}|{method_a}|{method_b}')
    rows.append({'family': family, 'metric': metric, 'method_a': method_a, 'method_b': method_b, 'n_pairs': len(diffs), 'mean_a': mean((a[k] for k in keys)), 'mean_b': mean((b[k] for k in keys)), 'mean_diff_a_minus_b': mean(diffs), 'paired_permutation_pvalue': pvalue})

def main() -> None:
    rows: List[Dict[str, object]] = []
    zero = load_metric_map(ROOT / 'results' / 'zero_shot' / 'zero_shot_all_methods_metrics.csv', ['assay'], 'method', 'abs_spearman')
    for a, b in [('VenusREM', 'ProSST-2048'), ('ProSST-2048', 'ESM3'), ('S3F_MSA', 'ESM3'), ('ProSST-2048', 'Site_Independent')]:
        compare(rows, 'zero_shot', 'abs_spearman', zero, a, b)
    higher = load_metric_map(ROOT / 'results' / 'single_to_multi' / 'single_to_multi_representative_metrics.csv', ['assay'], 'method', 'abs_spearman', {'order_bucket': 'higher'})
    for a, b in [('MSA_Transformer_ensemble', 'additive_all_singles'), ('GEMME', 'additive_all_singles'), ('ESM3', 'additive_all_singles'), ('ProSST-2048', 'additive_all_singles')]:
        compare(rows, 'single_to_higher', 'abs_spearman', higher, a, b)
    active = load_metric_map(ROOT / 'results' / 'active_learning' / 'active_learning_representative_metrics.csv', ['assay', 'seed'], 'policy', 'top1pct_found', {'round': '5'})
    for a, b in [('diverse_ensemble:top_methods', 'random'), ('diverse_ensemble:top_methods', 'additive_lookup'), ('ensemble_ucb:top_methods', 'zscore_linear:ESM3'), ('ensemble_mean:top_methods', 'zscore_linear:ESM3')]:
        compare(rows, 'active_learning_round5', 'top1pct_found', active, a, b)
    pairwise = load_metric_map(ROOT / 'results' / 'multi_evolve_style' / 'pairwise_plm_residual_higher_order_metrics.csv', ['assay', 'seed'], 'model', 'abs_spearman_higher', {'selection_policy': 'random_doubles', 'double_budget': '100'})
    for a, b in [('pairwise_plus_zscore:S2F_MSA', 'pairwise_residual_from_doubles'), ('pairwise_plus_plm_ensemble', 'pairwise_residual_from_doubles'), ('pairwise_plus_zscore:ProSST-2048', 'pairwise_residual_from_doubles')]:
        compare(rows, 'pairwise_plm_random100', 'abs_spearman_higher', pairwise, a, b)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ['family', 'metric', 'method_a', 'method_b', 'n_pairs', 'mean_a', 'mean_b', 'mean_diff_a_minus_b', 'paired_permutation_pvalue']
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
