from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List
ROOT = Path(__file__).resolve().parents[1]
MSA_DIR = ROOT / 'proteingym' / 'extracted' / 'DMS_msa_files'
MANIFEST = ROOT / 'results' / 'batch0' / 'assay_manifest.csv'
OUT = ROOT / 'results' / 'batch0' / 'msa_depth_summary.csv'

def msa_stats(path: Path) -> Dict[str, object]:
    n_seq = 0
    lengths: List[int] = []
    seq = []
    with path.open(encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq:
                    s = ''.join(seq)
                    lengths.append(len(s.replace('-', '')))
                    n_seq += 1
                    seq = []
            else:
                seq.append(line)
        if seq:
            s = ''.join(seq)
            lengths.append(len(s.replace('-', '')))
            n_seq += 1
    return {'msa_file': path.name, 'msa_n_sequences': n_seq, 'msa_median_ungapped_length': sorted(lengths)[len(lengths) // 2] if lengths else 0, 'msa_min_ungapped_length': min(lengths) if lengths else 0, 'msa_max_ungapped_length': max(lengths) if lengths else 0}

def main() -> None:
    msa_by_stem = {p.stem.split('_full')[0].split('_theta')[0].split('_2023')[0]: p for p in MSA_DIR.glob('*.a2m')}
    cache: Dict[Path, Dict[str, object]] = {}
    rows = []
    with MANIFEST.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            protein = row['protein_key']
            candidates = [p for key, p in msa_by_stem.items() if key.startswith(protein) or protein.startswith(key)]
            if not candidates:
                rows.append({'assay': row['assay'], 'protein_key': protein, 'has_msa_file': 0, 'msa_file': '', 'msa_n_sequences': 0, 'msa_median_ungapped_length': 0, 'msa_min_ungapped_length': 0, 'msa_max_ungapped_length': 0})
                continue
            path = sorted(candidates, key=lambda p: len(p.name))[0]
            if path not in cache:
                cache[path] = msa_stats(path)
            stats = cache[path]
            rows.append({'assay': row['assay'], 'protein_key': protein, 'has_msa_file': 1, **stats})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ['assay', 'protein_key', 'has_msa_file', 'msa_file', 'msa_n_sequences', 'msa_median_ungapped_length', 'msa_min_ungapped_length', 'msa_max_ungapped_length']
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
