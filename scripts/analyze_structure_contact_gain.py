from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List
ROOT = Path(__file__).resolve().parents[1]
CONTACT = ROOT / 'results' / 'structure_proxy' / 'assay_structure_contact_features.csv'
GAIN = ROOT / 'results' / 'structure_proxy' / 'structure_aware_gain_detail.csv'
OUT = ROOT / 'results' / 'structure_proxy' / 'structure_contact_gain_summary.csv'
DETAIL = ROOT / 'results' / 'structure_proxy' / 'structure_contact_gain_detail.csv'

def parse_float(value: str) -> float | None:
    if value == '':
        return None
    try:
        val = float(value)
    except ValueError:
        return None
    return None if math.isnan(val) else val

def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float('nan')

def bin_close_fraction(value: float | None) -> str:
    if value is None:
        return 'no_multi_or_missing'
    if value < 0.05:
        return 'close_lt8_low'
    if value < 0.2:
        return 'close_lt8_mid'
    return 'close_lt8_high'

def main() -> None:
    with CONTACT.open(newline='', encoding='utf-8') as fh:
        contact = {r['assay']: r for r in csv.DictReader(fh)}
    rows: List[Dict[str, object]] = []
    with GAIN.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            c = contact.get(row['assay'], {})
            frac = parse_float(c.get('double_fraction_min_ca_lt8', ''))
            rows.append({**row, 'n_double_variants': c.get('n_double_variants', ''), 'n_coord_complete_double': c.get('n_coord_complete_double', ''), 'coord_complete_double_fraction': c.get('coord_complete_double_fraction', ''), 'double_mean_min_ca_distance': c.get('double_mean_min_ca_distance', ''), 'double_fraction_min_ca_lt8': c.get('double_fraction_min_ca_lt8', ''), 'double_fraction_min_ca_lt12': c.get('double_fraction_min_ca_lt12', ''), 'double_close_bin': bin_close_fraction(frac)})
    DETAIL.parent.mkdir(parents=True, exist_ok=True)
    with DETAIL.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {DETAIL}')
    groups: Dict[tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        gain = parse_float(row.get('structure_aware_gain', ''))
        if gain is None:
            continue
        groups[row['assay_type'], row['double_close_bin']].append(gain)
    out = [{'assay_type': assay_type, 'double_close_bin': close_bin, 'n_assays': len(vals), 'mean_structure_aware_gain': mean(vals)} for (assay_type, close_bin), vals in groups.items()]
    out.sort(key=lambda r: (r['assay_type'], r['double_close_bin']))
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
