from __future__ import annotations
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'results' / 'batch0' / 'official_assay_metadata.csv'
DMS_DIR = ROOT / 'proteingym' / 'extracted' / 'DMS_ProteinGym_substitutions'
OUT = ROOT / 'results' / 'structure_proxy' / 'assay_structure_sasa_features.csv'
MUT_RE = re.compile('([A-Z*])(\\d+)([A-Z*])')
VDW = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'P': 1.8, 'H': 1.2}
PROBE = 1.4

def sphere_points(n: int=32) -> list[tuple[float, float, float]]:
    pts = []
    inc = math.pi * (3 - math.sqrt(5))
    off = 2 / n
    for k in range(n):
        y = k * off - 1 + off / 2
        r = math.sqrt(max(0.0, 1 - y * y))
        phi = k * inc
        pts.append((math.cos(phi) * r, y, math.sin(phi) * r))
    return pts
SPHERE = sphere_points()

def parse_positions(mutant: str) -> list[int]:
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

def atom_element(line: str) -> str:
    elem = line[76:78].strip()
    if elem:
        return elem[0].upper()
    name = line[12:16].strip()
    return name[0].upper() if name else 'C'

def parse_atoms(path: Path) -> list[dict[str, object]]:
    atoms = []
    with path.open(encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            if not line.startswith(('ATOM  ', 'HETATM')):
                continue
            if line[12:16].strip() != 'CA':
                continue
            try:
                resi = int(line[22:26])
                x, y, z = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            atoms.append({'resi': resi, 'xyz': (x, y, z), 'r': 1.9})
    return atoms

def align_atoms(atoms: list[dict[str, object]], target_positions: set[int], pdb_range: str) -> list[dict[str, object]]:
    direct_overlap = sum((int(atom['resi']) in target_positions for atom in atoms))
    match = re.fullmatch('\\s*(\\d+)\\s*-\\s*(\\d+)\\s*', pdb_range or '')
    if not match or not atoms:
        return atoms
    start = int(match.group(1))
    shifted_overlap = sum((start + int(atom['resi']) - 1 in target_positions for atom in atoms))
    if shifted_overlap <= direct_overlap:
        return atoms
    return [{**atom, 'resi': start + int(atom['resi']) - 1} for atom in atoms]

def dist2(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2

def residue_sasa(atoms: list[dict[str, object]], target_residues: set[int]) -> dict[int, float]:
    by_residue = defaultdict(list)
    for atom in atoms:
        by_residue[int(atom['resi'])].append(atom)
    out = {}
    for resi in target_residues:
        total = 0.0
        for atom in by_residue.get(resi, []):
            center = atom['xyz']
            radius = float(atom['r']) + PROBE
            accessible = 0
            neighbors = [a for a in atoms if int(a['resi']) != resi and dist2(center, a['xyz']) <= (radius + float(a['r']) + PROBE + 0.5) ** 2]
            for px, py, pz in SPHERE:
                point = (center[0] + radius * px, center[1] + radius * py, center[2] + radius * pz)
                blocked = False
                for other in neighbors:
                    cutoff = float(other['r']) + PROBE
                    if dist2(point, other['xyz']) < cutoff * cutoff:
                        blocked = True
                        break
                accessible += int(not blocked)
            total += 4.0 * math.pi * radius * radius * accessible / len(SPHERE)
        if total > 0:
            out[resi] = total
    return out

def safe_mean(xs: list[float]) -> str | float:
    return mean(xs) if xs else ''

def main() -> None:
    with META.open(newline='', encoding='utf-8') as fh:
        meta_rows = list(csv.DictReader(fh))
    rows = []
    for meta in meta_rows:
        assay = meta['assay']
        dms_path = DMS_DIR / assay
        pdb_path = local_structure_path(meta.get('pdb_file', ''))
        base = {'assay': assay, 'protein_key': meta.get('protein_key', ''), 'assay_type': meta.get('coarse_selection_type', ''), 'has_local_structure_file': int(pdb_path is not None), 'n_mutated_positions_with_sasa': 0, 'n_variants_with_sasa': 0, 'mean_mutation_site_sasa': '', 'median_position_sasa': '', 'sasa_exposed_fraction': '', 'sasa_buried_fraction': '', 'multi_mean_mutation_site_sasa': '', 'multi_sasa_exposed_fraction': '', 'multi_sasa_buried_fraction': ''}
        if pdb_path is None or not dms_path.exists():
            rows.append(base)
            continue
        variants = []
        target_positions = set()
        with dms_path.open(newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                pos = parse_positions(row.get('mutant', ''))
                if pos:
                    variants.append(pos)
                    target_positions.update(pos)
        atoms = align_atoms(parse_atoms(pdb_path), target_positions, meta.get('pdb_range', ''))
        sasa = residue_sasa(atoms, target_positions)
        if not sasa:
            rows.append(base)
            continue
        vals = sorted(sasa.values())
        low = vals[len(vals) // 3]
        high = vals[2 * len(vals) // 3]
        all_avg, all_exp, all_bur = ([], [], [])
        multi_avg, multi_exp, multi_bur = ([], [], [])
        for pos in variants:
            sv = [sasa[p] for p in pos if p in sasa]
            if not sv:
                continue
            all_avg.append(mean(sv))
            all_exp.append(sum((1 for x in sv if x >= high)) / len(sv))
            all_bur.append(sum((1 for x in sv if x <= low)) / len(sv))
            if len(pos) > 1:
                multi_avg.append(mean(sv))
                multi_exp.append(sum((1 for x in sv if x >= high)) / len(sv))
                multi_bur.append(sum((1 for x in sv if x <= low)) / len(sv))
        rec = dict(base)
        rec.update({'n_mutated_positions_with_sasa': len(sasa), 'n_variants_with_sasa': len(all_avg), 'mean_mutation_site_sasa': safe_mean(all_avg), 'median_position_sasa': vals[len(vals) // 2], 'sasa_exposed_fraction': safe_mean(all_exp), 'sasa_buried_fraction': safe_mean(all_bur), 'multi_mean_mutation_site_sasa': safe_mean(multi_avg), 'multi_sasa_exposed_fraction': safe_mean(multi_exp), 'multi_sasa_buried_fraction': safe_mean(multi_bur)})
        rows.append(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
