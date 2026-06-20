from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'results' / 'batch0' / 'assay_manifest.csv'
ASSAY_TYPES = ROOT / 'results' / 'batch0' / 'assay_type_curation.csv'
OFFICIAL = ROOT / 'results' / 'batch0' / 'official_assay_metadata.csv'
ZERO = ROOT / 'results' / 'zero_shot' / 'zero_shot_all_methods_metrics.csv'
OUT_DIR = ROOT / 'results' / 'structure_proxy'
STRUCTURE_DIRS = [ROOT / 'proteingym' / 'structures' / 'AF2', ROOT / 'proteingym' / 'structures']
STRUCTURE_METHODS = {'ProSST-2048', 'ProSST-4096', 'S3F_MSA', 'S2F_MSA', 'SaProt_650M_AF2', 'ProteinMPNN', 'ESM-IF1'}
REFERENCE_METHODS = {'ESM3', 'GEMME', 'MSA_Transformer_ensemble', 'Site_Independent', 'VenusREM'}

def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float('nan')

def parse_float(value: str) -> float | None:
    if value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None

def assay_type_map() -> Dict[str, str]:
    official = ROOT / 'results' / 'batch0' / 'assay_type_curation_official.csv'
    path = official if official.exists() else ROOT / 'results' / 'batch0' / 'assay_type_curation_refined.csv'
    if not path.exists():
        path = ASSAY_TYPES
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as fh:
        return {r['assay']: r.get('assay_type_official', r.get('assay_type_refined', r.get('assay_type_curated', r.get('assay_type', 'uncurated')))) for r in csv.DictReader(fh)}

def official_metadata() -> Dict[str, Dict[str, str]]:
    if not OFFICIAL.exists():
        return {}
    with OFFICIAL.open(newline='', encoding='utf-8') as fh:
        return {r['assay']: r for r in csv.DictReader(fh)}

def local_structure_path(pdb_file: str) -> str:
    if not pdb_file:
        return ''
    for root in STRUCTURE_DIRS:
        candidate = root / pdb_file
        if candidate.exists():
            return str(candidate.relative_to(ROOT))
        nested = list(root.rglob(pdb_file)) if root.exists() else []
        if nested:
            return str(nested[0].relative_to(ROOT))
    return ''

def main() -> None:
    types = assay_type_map()
    official = official_metadata()
    proxy_rows = []
    with MANIFEST.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            meta = official.get(row['assay'], {})
            pdb_file = meta.get('pdb_file', '')
            pdb_path = local_structure_path(pdb_file)
            proxy_rows.append({'assay': row['assay'], 'protein_key': row['protein_key'], 'assay_type': types.get(row['assay'], 'uncurated'), 'has_pdb_in_official_reference': int(bool(pdb_file)), 'pdb_file': pdb_file, 'pdb_range': meta.get('pdb_range', ''), 'has_local_structure_file': int(bool(pdb_path)), 'local_structure_path': pdb_path})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proxy_out = OUT_DIR / 'assay_structure_proxy.csv'
    with proxy_out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(proxy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(proxy_rows)
    print(f'wrote {proxy_out}')
    score: Dict[str, Dict[str, float]] = defaultdict(dict)
    with ZERO.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            val = parse_float(row['abs_spearman'])
            if val is not None:
                score[row['assay']][row['method']] = val
    proxy_by_assay = {r['assay']: r for r in proxy_rows}
    groups: Dict[tuple[str, str], List[float]] = defaultdict(list)
    detail = []
    for assay, vals in score.items():
        struct_best = max((vals[m] for m in STRUCTURE_METHODS if m in vals), default=None)
        ref_best = max((vals[m] for m in REFERENCE_METHODS if m in vals), default=None)
        if struct_best is None or ref_best is None:
            continue
        gain = struct_best - ref_best
        proxy = proxy_by_assay[assay]
        key = (proxy['assay_type'], str(proxy['has_pdb_in_official_reference']), str(proxy['has_local_structure_file']))
        groups[key].append(gain)
        detail.append({**proxy, 'best_structure_aware_abs_spearman': struct_best, 'best_reference_abs_spearman': ref_best, 'structure_aware_gain': gain})
    detail_out = OUT_DIR / 'structure_aware_gain_detail.csv'
    with detail_out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(detail[0].keys()))
        writer.writeheader()
        writer.writerows(detail)
    print(f'wrote {detail_out}')
    summary = []
    for (atype, has_pdb, has_local), gains in groups.items():
        summary.append({'assay_type': atype, 'has_pdb_in_official_reference': has_pdb, 'has_local_structure_file': has_local, 'n_assays': len(gains), 'mean_structure_aware_gain': mean(gains)})
    summary.sort(key=lambda r: (r['assay_type'], r['has_pdb_in_official_reference'], r['has_local_structure_file']))
    summary_out = OUT_DIR / 'structure_aware_gain_summary.csv'
    with summary_out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f'wrote {summary_out}')
if __name__ == '__main__':
    main()
