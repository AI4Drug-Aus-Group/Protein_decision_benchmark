from __future__ import annotations
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'results' / 'batch0' / 'assay_manifest.csv'
EPI = ROOT / 'results' / 'epistasis' / 'epistasis_assay_summary.csv'
OUT = ROOT / 'results' / 'batch0' / 'assay_strata.csv'

def assay_type_from_name(name: str) -> str:
    lower = name.lower()
    if 'binding' in lower:
        return 'binding'
    if 'activity' in lower:
        return 'activity'
    if 'expression' in lower or 'abundance' in lower:
        return 'expression_or_abundance'
    if 'surface' in lower:
        return 'surface'
    if 'function' in lower:
        return 'function'
    if 'stability' in lower:
        return 'stability'
    return 'uncurated'

def main() -> None:
    epi_values = {}
    if EPI.exists():
        with EPI.open(newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                epi_values[row['assay']] = float(row['mean_abs_epistasis_residual'])
    epi_sorted = sorted(epi_values.values())
    low_cut = epi_sorted[len(epi_sorted) // 3] if epi_sorted else None
    high_cut = epi_sorted[2 * len(epi_sorted) // 3] if epi_sorted else None
    rows = []
    with MANIFEST.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            n_doubles = int(row['n_doubles'])
            n_higher = int(row['n_higher_order'])
            strict_double = int(row['strict_double_evaluable'])
            if n_higher > 0:
                mutation_stratum = 'higher_order'
            elif n_doubles > 0:
                mutation_stratum = 'double_only'
            else:
                mutation_stratum = 'single_only'
            if strict_double >= 10000:
                double_density = 'very_double_rich'
            elif strict_double >= 100:
                double_density = 'epistasis_evaluable'
            elif n_doubles > 0:
                double_density = 'few_doubles'
            else:
                double_density = 'no_doubles'
            epi = epi_values.get(row['assay'])
            if epi is None:
                epi_stratum = 'not_evaluable'
            elif epi <= low_cut:
                epi_stratum = 'low_epistasis_residual'
            elif epi >= high_cut:
                epi_stratum = 'high_epistasis_residual'
            else:
                epi_stratum = 'mid_epistasis_residual'
            rows.append({'assay': row['assay'], 'protein_key': row['protein_key'], 'assay_type_keyword': assay_type_from_name(row['assay']), 'mutation_stratum': mutation_stratum, 'double_density': double_density, 'epistasis_stratum': epi_stratum, 'mean_abs_epistasis_residual': '' if epi is None else epi, 'n_variants': row['n_variants'], 'n_singles': row['n_singles'], 'n_doubles': row['n_doubles'], 'n_higher_order': row['n_higher_order'], 'strict_double_evaluable': row['strict_double_evaluable'], 'has_msa': row['has_msa']})
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
