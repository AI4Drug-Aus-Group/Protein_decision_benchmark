from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List
ROOT = Path(__file__).resolve().parents[1]
ASA = ROOT / 'results' / 'structure_proxy' / 'assay_structure_asa_features.csv'
GAIN = ROOT / 'results' / 'structure_proxy' / 'structure_aware_gain_detail.csv'
OUT = ROOT / 'results' / 'structure_proxy' / 'structure_asa_gain_summary.csv'
DETAIL = ROOT / 'results' / 'structure_proxy' / 'structure_asa_gain_detail.csv'

def parse_float(value: str) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(val) or math.isinf(val) else val

def mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else float('nan')

def exposure_bin(exposed: float | None, buried: float | None) -> str:
    if exposed is None or buried is None:
        return 'missing'
    if buried - exposed >= 0.15:
        return 'buried_enriched'
    if exposed - buried >= 0.15:
        return 'exposed_enriched'
    return 'mixed'

def main() -> None:
    with ASA.open(newline='', encoding='utf-8') as fh:
        asa = {r['assay']: r for r in csv.DictReader(fh)}
    rows: List[Dict[str, object]] = []
    with GAIN.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            a = asa.get(row['assay'], {})
            exposed = parse_float(a.get('mean_exposed_fraction', ''))
            buried = parse_float(a.get('mean_buried_fraction', ''))
            rows.append({**row, 'mean_neighbor_count_10A': a.get('mean_neighbor_count_10A', ''), 'mean_exposed_fraction': a.get('mean_exposed_fraction', ''), 'mean_buried_fraction': a.get('mean_buried_fraction', ''), 'multi_exposed_fraction': a.get('multi_exposed_fraction', ''), 'multi_buried_fraction': a.get('multi_buried_fraction', ''), 'exposure_bin': exposure_bin(exposed, buried)})
    with DETAIL.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    groups: Dict[tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        gain = parse_float(row.get('structure_aware_gain', ''))
        if gain is not None:
            groups[row['assay_type'], row['exposure_bin']].append(gain)
    out = [{'assay_type': assay_type, 'exposure_bin': bin_name, 'n_assays': len(vals), 'mean_structure_aware_gain': mean(vals)} for (assay_type, bin_name), vals in groups.items()]
    out.sort(key=lambda r: (r['assay_type'], r['exposure_bin']))
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print(f'wrote {DETAIL}')
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
