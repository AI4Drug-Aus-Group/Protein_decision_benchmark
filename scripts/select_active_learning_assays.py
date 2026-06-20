from __future__ import annotations
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'results' / 'batch0' / 'assay_manifest.csv'
EPI = ROOT / 'results' / 'epistasis' / 'epistasis_assay_summary.csv'
OUT = ROOT / 'results' / 'active_learning' / 'representative_assays.csv'

def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))

def main() -> None:
    manifest = {row['assay']: row for row in read_csv(MANIFEST)}
    epi_rows = read_csv(EPI) if EPI.exists() else []
    selected = {}

    def add(assay: str, reason: str) -> None:
        if assay in manifest:
            selected.setdefault(assay, set()).add(reason)
    high_epi = sorted(epi_rows, key=lambda r: float(r['mean_abs_epistasis_residual']), reverse=True)[:8]
    low_epi = sorted(epi_rows, key=lambda r: float(r['mean_abs_epistasis_residual']))[:8]
    for row in high_epi:
        add(row['assay'], 'high_epistasis')
    for row in low_epi:
        add(row['assay'], 'low_epistasis')
    doubles = sorted(manifest.values(), key=lambda r: int(r['n_doubles']), reverse=True)[:8]
    higher = sorted(manifest.values(), key=lambda r: int(r['n_higher_order']), reverse=True)[:8]
    for row in doubles:
        add(row['assay'], 'many_doubles')
    for row in higher:
        add(row['assay'], 'many_higher_order')
    single_only = [row for row in manifest.values() if int(row['n_doubles']) == 0 and 500 <= int(row['n_variants']) <= 20000][:8]
    for row in single_only:
        add(row['assay'], 'single_only_control')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for assay, reasons in sorted(selected.items()):
        row = manifest[assay]
        rows.append({'assay': assay, 'reasons': '|'.join(sorted(reasons)), 'n_variants': row['n_variants'], 'n_singles': row['n_singles'], 'n_doubles': row['n_doubles'], 'n_higher_order': row['n_higher_order'], 'strict_double_evaluable': row['strict_double_evaluable']})
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['assay', 'reasons', 'n_variants', 'n_singles', 'n_doubles', 'n_higher_order', 'strict_double_evaluable'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {OUT}')
    print(f'selected_assays {len(rows)}')
if __name__ == '__main__':
    main()
