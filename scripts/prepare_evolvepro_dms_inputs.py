from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import List
ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
OUT_DIR = ROOT / 'results' / 'external_wrappers' / 'evolvepro_inputs'
DEFAULT_FEATURES = ['VenusREM', 'ProSST-2048', 'S3F_MSA', 'S2F_MSA', 'PoET', 'ESM3', 'GEMME', 'TranceptEVE_L']

def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--assay', required=True)
    parser.add_argument('--dataset-name', default='')
    parser.add_argument('--features', default=','.join(DEFAULT_FEATURES))
    parser.add_argument('--max-rows', type=int, default=0)
    parser.add_argument('--out-dir', default=str(OUT_DIR))
    args = parser.parse_args()
    assay = args.assay
    dataset = args.dataset_name or Path(assay).stem
    features = [x for x in args.features.split(',') if x]
    out_dir = Path(args.out_dir)
    labels_dir = out_dir / 'labels'
    embeddings_dir = out_dir / 'embeddings'
    labels_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with (ZERO_DIR / assay).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            y = parse_float(row.get('DMS_score'))
            vals = [parse_float(row.get(feature, '')) for feature in features]
            if y is None or any((v is None for v in vals)):
                continue
            rows.append((row['mutant'], y, vals))
            if args.max_rows and len(rows) >= args.max_rows:
                break
    ys = [row[1] for row in rows]
    lo, hi = (min(ys), max(ys))
    labels_path = labels_dir / f'{dataset}_labels.csv'
    embeddings_path = embeddings_dir / f'{dataset}_score_embedding.csv'
    with labels_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['variant', 'activity', 'activity_scaled', 'activity_binary'])
        writer.writeheader()
        for mutant, y, _ in rows:
            scaled = 0.0 if hi == lo else (y - lo) / (hi - lo)
            writer.writerow({'variant': mutant, 'activity': y, 'activity_scaled': scaled, 'activity_binary': int(scaled >= 0.5)})
    with embeddings_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['variant'] + features)
        for mutant, _, vals in rows:
            writer.writerow([mutant] + vals)
    print(f'wrote {labels_path}')
    print(f'wrote {embeddings_path}')
if __name__ == '__main__':
    main()
