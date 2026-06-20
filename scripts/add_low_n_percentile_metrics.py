from __future__ import annotations
import argparse
import bisect
import csv
from pathlib import Path
from typing import Dict, List
ROOT = Path(__file__).resolve().parents[1]
DMS_DIR = ROOT / 'proteingym' / 'extracted' / 'DMS_ProteinGym_substitutions'

def load_score_distributions() -> Dict[str, List[float]]:
    distributions: Dict[str, List[float]] = {}
    for path in sorted(DMS_DIR.glob('*.csv')):
        scores: List[float] = []
        with path.open(newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                try:
                    scores.append(float(row['DMS_score']))
                except ValueError:
                    continue
        scores.sort()
        distributions[path.name] = scores
    return distributions

def percentile(scores: List[float], value: str) -> str:
    if not value or not scores:
        return ''
    try:
        parsed = float(value)
    except ValueError:
        return ''
    return str(bisect.bisect_right(scores, parsed) / len(scores))

def process_file(input_path: Path, output_path: Path, distributions: Dict[str, List[float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(newline='', encoding='utf-8') as in_fh:
        reader = csv.DictReader(in_fh)
        if reader.fieldnames is None:
            raise ValueError(f'missing header: {input_path}')
        fieldnames = list(reader.fieldnames)
        metric_col = 'best_true_in_pred_top_100'
        if metric_col not in fieldnames:
            raise ValueError(f'{input_path} lacks {metric_col}')
        new_col = 'best_true_in_pred_top_100_percentile'
        if new_col not in fieldnames:
            fieldnames.append(new_col)
        with output_path.open('w', newline='', encoding='utf-8') as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                row[new_col] = percentile(distributions.get(row['assay'], []), row.get(metric_col, ''))
                writer.writerow(row)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True, help='Comma-separated low-N metric CSVs.')
    parser.add_argument('--suffix', default='.percentile.csv')
    args = parser.parse_args()
    distributions = load_score_distributions()
    for item in args.inputs.split(','):
        if not item:
            continue
        input_path = ROOT / item if not Path(item).is_absolute() else Path(item)
        output_path = input_path.with_name(input_path.stem + args.suffix)
        process_file(input_path, output_path, distributions)
        print(f'wrote {output_path}')
if __name__ == '__main__':
    main()
