from __future__ import annotations
import csv
import math
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple
ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'results' / 'batch0' / 'official_assay_metadata.csv'
DMS_DIR = ROOT / 'proteingym' / 'extracted' / 'DMS_ProteinGym_substitutions'
OUT = ROOT / 'results' / 'structure_proxy' / 'assay_structure_asa_features.csv'
MUT_RE = re.compile('([A-Z*])(\\d+)([A-Z*])')

def parse_positions(mutant: str) -> List[int]:
    return [int(m.group(2)) for m in MUT_RE.finditer(mutant or '')]

def local_structure_path(pdb_file: str) -> Path | None:
    for base in [ROOT / 'proteingym' / 'structures' / 'AF2', ROOT / 'proteingym' / 'structures']:
        if not base.exists():
            continue
        direct = base / pdb_file
        if direct.exists():
            return direct
        matches = list(base.rglob(pdb_file))
        if matches:
            return matches[0]
    return None

def parse_ca(path: Path) -> Dict[int, Tuple[float, float, float]]:
    coords: Dict[int, Tuple[float, float, float]] = {}
    with path.open(encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            if not line.startswith(('ATOM  ', 'HETATM')) or line[12:16].strip() != 'CA':
                continue
            try:
                coords[int(line[22:26])] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
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

def distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum(((x - y) ** 2 for x, y in zip(a, b))))

def neighbor_counts(coords: Dict[int, Tuple[float, float, float]], radius: float=10.0) -> Dict[int, int]:
    items = list(coords.items())
    counts = {resi: 0 for resi in coords}
    for i, (resi_i, ca_i) in enumerate(items):
        for resi_j, ca_j in items[i + 1:]:
            if distance(ca_i, ca_j) <= radius:
                counts[resi_i] += 1
                counts[resi_j] += 1
    return counts

def tertile_label(value: int, low: float, high: float) -> str:
    if value <= low:
        return 'exposed'
    if value >= high:
        return 'buried'
    return 'intermediate'

def safe_mean(vals: List[float]) -> str | float:
    return mean(vals) if vals else ''

def main() -> None:
    with META.open(newline='', encoding='utf-8') as fh:
        meta_rows = list(csv.DictReader(fh))
    rows = []
    for meta in meta_rows:
        assay = meta['assay']
        pdb_path = local_structure_path(meta.get('pdb_file', ''))
        base = {'assay': assay, 'protein_key': meta['protein_key'], 'assay_type': meta.get('coarse_selection_type', ''), 'has_local_structure_file': int(pdb_path is not None), 'n_variants_with_coords': 0, 'n_mutated_positions_with_coords': 0, 'mean_neighbor_count_10A': '', 'mean_exposed_fraction': '', 'mean_buried_fraction': '', 'single_mean_neighbor_count_10A': '', 'single_exposed_fraction': '', 'single_buried_fraction': '', 'multi_mean_neighbor_count_10A': '', 'multi_exposed_fraction': '', 'multi_buried_fraction': ''}
        dms_path = DMS_DIR / assay
        if pdb_path is None or not dms_path.exists():
            rows.append(base)
            continue
        variant_positions = []
        target_positions = set()
        with dms_path.open(newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                positions = parse_positions(row.get('mutant', ''))
                if positions:
                    variant_positions.append(positions)
                    target_positions.update(positions)
        coords = align_coords(parse_ca(pdb_path), target_positions, meta.get('pdb_range', ''))
        counts = neighbor_counts(coords)
        if not counts:
            rows.append(base)
            continue
        sorted_counts = sorted(counts.values())
        low = sorted_counts[len(sorted_counts) // 3]
        high = sorted_counts[2 * len(sorted_counts) // 3]
        all_counts: List[float] = []
        all_exposed: List[float] = []
        all_buried: List[float] = []
        single_counts: List[float] = []
        single_exposed: List[float] = []
        single_buried: List[float] = []
        multi_counts: List[float] = []
        multi_exposed: List[float] = []
        multi_buried: List[float] = []
        pos_seen = set()
        for raw_positions in variant_positions:
            positions = [position for position in raw_positions if position in counts]
            if not positions:
                continue
            pos_seen.update(positions)
            vals = [counts[position] for position in positions]
            labels = [tertile_label(counts[position], low, high) for position in positions]
            exposed = sum((1 for label in labels if label == 'exposed')) / len(labels)
            buried = sum((1 for label in labels if label == 'buried')) / len(labels)
            avg = mean(vals)
            all_counts.append(avg)
            all_exposed.append(exposed)
            all_buried.append(buried)
            if len(raw_positions) == 1:
                single_counts.append(avg)
                single_exposed.append(exposed)
                single_buried.append(buried)
            else:
                multi_counts.append(avg)
                multi_exposed.append(exposed)
                multi_buried.append(buried)
        rec = dict(base)
        rec.update({'n_variants_with_coords': len(all_counts), 'n_mutated_positions_with_coords': len(pos_seen), 'mean_neighbor_count_10A': safe_mean(all_counts), 'mean_exposed_fraction': safe_mean(all_exposed), 'mean_buried_fraction': safe_mean(all_buried), 'single_mean_neighbor_count_10A': safe_mean(single_counts), 'single_exposed_fraction': safe_mean(single_exposed), 'single_buried_fraction': safe_mean(single_buried), 'multi_mean_neighbor_count_10A': safe_mean(multi_counts), 'multi_exposed_fraction': safe_mean(multi_exposed), 'multi_buried_fraction': safe_mean(multi_buried)})
        rows.append(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
