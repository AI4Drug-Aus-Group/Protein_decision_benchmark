from __future__ import annotations
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'results' / 'batch0' / 'assay_manifest.csv'
OUT = ROOT / 'results' / 'batch0' / 'assay_type_curation.csv'
RULES = [('binding', ['binding', 'bind', 'darPin'.lower(), 'tapbpr', 'mcl-1', 'fmc']), ('activity', ['activity', 'enzyme', 'catalytic']), ('expression_or_abundance', ['expression', 'abundance', 'surface']), ('function', ['function', 'lof', 'gof', 'positive']), ('viral_fitness_or_escape', ['hiv', 'sars', 'zikv', 'influenza', '9infa', 'i34a1', 'i33a0', 'aav']), ('drug_or_selection', ['nutlin', 'etoposide', 'das_25um', 'h2o2'])]

def curate(name: str) -> tuple[str, str]:
    lower = name.lower()
    matched = []
    for assay_type, keywords in RULES:
        for keyword in keywords:
            if keyword in lower:
                matched.append((assay_type, keyword))
                break
    if not matched:
        return ('uncurated', '')
    priority = ['binding', 'activity', 'expression_or_abundance', 'function', 'drug_or_selection', 'viral_fitness_or_escape']
    matched.sort(key=lambda item: priority.index(item[0]) if item[0] in priority else 999)
    return (matched[0][0], matched[0][1])

def main() -> None:
    rows = []
    with MANIFEST.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            assay_type, keyword = curate(row['assay'])
            rows.append({'assay': row['assay'], 'protein_key': row['protein_key'], 'assay_type_curated': assay_type, 'curation_basis': 'filename_keyword' if keyword else 'uncurated', 'matched_keyword': keyword, 'n_variants': row['n_variants'], 'n_singles': row['n_singles'], 'n_doubles': row['n_doubles'], 'n_higher_order': row['n_higher_order']})
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
