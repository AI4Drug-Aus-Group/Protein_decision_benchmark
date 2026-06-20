from __future__ import annotations
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / 'results' / 'batch0' / 'assay_type_curation.csv'
OUT = ROOT / 'results' / 'batch0' / 'assay_type_curation_refined.csv'
RULES = [('binding', ['binding', 'bind', 'tapbpr', 'mcl-1', 'darPin'.lower(), 'fmc']), ('expression_or_abundance', ['expression', 'abundance', 'surface', 'high-expression', 'low-expression', 'hek293t', 'rpe1']), ('activity', ['activity', 'casp', 'dyr_', 'blat_', 'amie_', 'esta_', 'lgk_', 'hmdh_', 'hxk4', 'cp2c9', 'vkor1']), ('viral_fitness_or_escape', ['hiv', 'hv1', 'sars', 'spike', 'zikv', '9infa', 'i34a1', 'i33a0', 'aav', 'polg_hcv', 'polg_den', 'rdrp', 'ncap']), ('drug_or_selection', ['nutlin', 'etoposide', 'das_25um', 'h2o2', 'antibiotic']), ('stability_or_folding', ['tsuboyama_2023']), ('function', ['function', 'lof', 'gof', 'positive', 'growth'])]

def infer(assay: str, current: str) -> tuple[str, str, str, str]:
    low = assay.lower()
    if current != 'uncurated':
        return (current, 'high', 'original_keyword', '')
    for label, needles in RULES:
        for needle in needles:
            if needle in low:
                confidence = 'high' if label in {'binding', 'expression_or_abundance', 'viral_fitness_or_escape', 'drug_or_selection', 'stability_or_folding'} else 'medium'
                return (label, confidence, 'refined_filename_rule', needle)
    return ('uncurated', 'low', 'needs_manual_review', '')

def main() -> None:
    rows = []
    with IN.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            label, confidence, basis, keyword = infer(row['assay'], row['assay_type_curated'])
            rows.append({**row, 'assay_type_refined': label, 'refined_confidence': confidence, 'refined_basis': basis, 'refined_keyword': keyword, 'manual_review_needed': int(confidence != 'high')})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
