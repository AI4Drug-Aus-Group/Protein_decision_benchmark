from __future__ import annotations
import csv
import math
import re
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Tuple
ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'results' / 'batch0' / 'official_assay_metadata.csv'
DMS_DIR = ROOT / 'proteingym' / 'extracted' / 'DMS_ProteinGym_substitutions'
OUT = ROOT / 'results' / 'structure_proxy' / 'assay_structure_contact_features.csv'
MUT_RE = re.compile('([A-Z*])(\\d+)([A-Z*])')

def parse_ca_coords(path: Path) -> Dict[int, Tuple[float, float, float]]:
    coords: Dict[int, Tuple[float, float, float]] = {}
    with path.open(encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            if not line.startswith(('ATOM  ', 'HETATM')):
                continue
            if line[12:16].strip() != 'CA':
                continue
            try:
                resi = int(line[22:26])
                coords[resi] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
    return coords

def align_coords(coords: Dict[int, Tuple[float, float, float]], target_positions: set[int], pdb_range: str) -> Dict[int, Tuple[float, float, float]]:
    direct_overlap = len(target_positions.intersection(coords))
    match = re.fullmatch('\\s*(\\d+)\\s*-\\s*(\\d+)\\s*', pdb_range or '')
    if not match or not coords:
        return coords
    start = int(match.group(1))
    shifted = {start + residue - 1: xyz for residue, xyz in coords.items()}
    shifted_overlap = len(target_positions.intersection(shifted))
    return shifted if shifted_overlap > direct_overlap else coords

def parse_mutant(mutant: str) -> List[int]:
    return [int(m.group(2)) for m in MUT_RE.finditer(mutant)]

def dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum(((x - y) ** 2 for x, y in zip(a, b))))

def min_pair_distance(positions: List[int], coords: Dict[int, Tuple[float, float, float]]) -> float | None:
    if len(positions) < 2:
        return None
    vals = []
    for i, left in enumerate(positions):
        ca_left = coords.get(left)
        if ca_left is None:
            return None
        for right in positions[i + 1:]:
            ca_right = coords.get(right)
            if ca_right is None:
                return None
            vals.append(dist(ca_left, ca_right))
    return min(vals) if vals else None

def mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else float('nan')

def local_structure_path(pdb_file: str) -> Path | None:
    for base in [ROOT / 'proteingym' / 'structures' / 'AF2', ROOT / 'proteingym' / 'structures']:
        candidate = base / pdb_file
        if candidate.exists():
            return candidate
        if base.exists():
            matches = list(base.rglob(pdb_file))
            if matches:
                return matches[0]
    return None

def summarize_distances(distances: List[float]) -> Dict[str, object]:
    if not distances:
        return {'n_coord_complete_multi': 0, 'mean_min_ca_distance': '', 'median_min_ca_distance': '', 'fraction_min_ca_lt8': '', 'fraction_min_ca_lt12': '', 'fraction_min_ca_lt20': ''}
    return {'n_coord_complete_multi': len(distances), 'mean_min_ca_distance': mean(distances), 'median_min_ca_distance': median(distances), 'fraction_min_ca_lt8': mean((1.0 if d < 8 else 0.0 for d in distances)), 'fraction_min_ca_lt12': mean((1.0 if d < 12 else 0.0 for d in distances)), 'fraction_min_ca_lt20': mean((1.0 if d < 20 else 0.0 for d in distances))}

def main() -> None:
    with META.open(newline='', encoding='utf-8') as fh:
        meta_rows = list(csv.DictReader(fh))
    rows: List[Dict[str, object]] = []
    for meta in meta_rows:
        assay = meta['assay']
        pdb_file = meta.get('pdb_file', '')
        dms_path = DMS_DIR / assay
        pdb_path = local_structure_path(pdb_file)
        base = {'assay': assay, 'protein_key': meta['protein_key'], 'assay_type': meta.get('coarse_selection_type', ''), 'pdb_file': pdb_file, 'pdb_range': meta.get('pdb_range', ''), 'has_local_structure_file': int(pdb_path is not None), 'n_multi_variants': 0, 'n_coord_complete_multi': 0, 'coord_complete_multi_fraction': '', 'n_double_variants': 0, 'n_coord_complete_double': 0, 'coord_complete_double_fraction': ''}
        if pdb_path is None or not dms_path.exists():
            rows.append(base)
            continue
        variants: List[List[int]] = []
        target_positions = set()
        with dms_path.open(newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                positions = parse_mutant(row.get('mutant', ''))
                if len(positions) < 2:
                    continue
                variants.append(positions)
                target_positions.update(positions)
        coords = align_coords(parse_ca_coords(pdb_path), target_positions, meta.get('pdb_range', ''))
        multi_distances: List[float] = []
        double_distances: List[float] = []
        n_multi = 0
        n_double = 0
        for positions in variants:
            n_multi += 1
            if len(positions) == 2:
                n_double += 1
            d = min_pair_distance(positions, coords)
            if d is None:
                continue
            multi_distances.append(d)
            if len(positions) == 2:
                double_distances.append(d)
        rec = dict(base)
        rec['n_multi_variants'] = n_multi
        rec.update(summarize_distances(multi_distances))
        rec['coord_complete_multi_fraction'] = len(multi_distances) / n_multi if n_multi else ''
        double_summary = summarize_distances(double_distances)
        rec['n_double_variants'] = n_double
        rec['n_coord_complete_double'] = double_summary['n_coord_complete_multi']
        rec['coord_complete_double_fraction'] = len(double_distances) / n_double if n_double else ''
        for key, value in double_summary.items():
            if key == 'n_coord_complete_multi':
                continue
            rec[f'double_{key}'] = value
        rows.append(rec)
    fields = list(rows[0].keys())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
