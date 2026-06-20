from __future__ import annotations
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'statistics'
BOOTSTRAPS = 1000

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

def percentile(values: Sequence[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float('nan')
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac

def bootstrap_mean_ci(values: Sequence[float], seed: str, n_boot: int=BOOTSTRAPS) -> Tuple[float, float, float]:
    values = [v for v in values if v == v]
    if not values:
        return (float('nan'), float('nan'), float('nan'))
    if len(values) == 1:
        return (values[0], values[0], values[0])
    rng = random.Random(seed)
    sims = []
    for _ in range(n_boot):
        sims.append(mean((values[rng.randrange(len(values))] for _ in values)))
    return (mean(values), percentile(sims, 0.025), percentile(sims, 0.975))

def write_rows(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')

def grouped_ci(path: Path, group_fields: Sequence[str], value_field: str, out_path: Path, filters: Dict[str, set[str]] | None=None) -> None:
    groups: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if filters and any((row.get(k) not in allowed for k, allowed in filters.items())):
                continue
            value = parse_float(row.get(value_field, ''))
            if value is None:
                continue
            groups[tuple((row[k] for k in group_fields))].append(value)
    rows = []
    for key, values in groups.items():
        estimate, lo, hi = bootstrap_mean_ci(values, '|'.join(key) + '|' + value_field)
        row = {field: key[i] for i, field in enumerate(group_fields)}
        row.update({'metric': value_field, 'n': len(values), 'mean': estimate, 'ci95_low': lo, 'ci95_high': hi})
        rows.append(row)
    rows.sort(key=lambda r: tuple((r[f] for f in group_fields)))
    write_rows(out_path, rows, list(group_fields) + ['metric', 'n', 'mean', 'ci95_low', 'ci95_high'])

def main() -> None:
    top_zero_methods = {'VenusREM', 'ProSST-2048', 'S3F_MSA', 'ProSST-4096', 'S2F_MSA', 'ProSST-1024', 'PoET', 'ESM3', 'Site_Independent'}
    grouped_ci(ROOT / 'results' / 'zero_shot' / 'zero_shot_all_methods_metrics.csv', ['method'], 'abs_spearman', OUT / 'zero_shot_abs_spearman_ci.csv', {'method': top_zero_methods})
    grouped_ci(ROOT / 'results' / 'single_to_multi' / 'single_to_multi_representative_metrics.csv', ['order_bucket', 'method'], 'abs_spearman', OUT / 'single_to_multi_abs_spearman_ci.csv', {'method': {'additive_all_singles', 'ProSST-2048', 'ESM3', 'MSA_Transformer_ensemble', 'GEMME', 'TranceptEVE_L', 'VenusREM'}})
    grouped_ci(ROOT / 'results' / 'active_learning' / 'active_learning_representative_metrics.csv', ['round', 'policy'], 'top1pct_found', OUT / 'active_learning_top1pct_found_ci.csv', {'round': {'5'}})
    grouped_ci(ROOT / 'results' / 'active_learning' / 'active_learning_representative_metrics.csv', ['round', 'policy'], 'best_observed_percentile', OUT / 'active_learning_best_percentile_ci.csv', {'round': {'5'}})
    grouped_ci(ROOT / 'results' / 'multi_evolve_style' / 'pairwise_plm_residual_higher_order_metrics.csv', ['model', 'selection_policy', 'double_budget'], 'abs_spearman_higher', OUT / 'pairwise_plm_abs_spearman_ci.csv')
if __name__ == '__main__':
    main()
