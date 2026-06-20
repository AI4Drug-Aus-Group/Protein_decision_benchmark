from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'proteingym' / 'reference_DMS_substitutions.csv'
MANIFEST = ROOT / 'results' / 'batch0' / 'assay_manifest.csv'
OUT_META = ROOT / 'results' / 'batch0' / 'official_assay_metadata.csv'
OUT_CURATION = ROOT / 'results' / 'batch0' / 'assay_type_curation_official.csv'
KEEP_FIELDS = ['DMS_id', 'DMS_filename', 'UniProt_ID', 'taxon', 'source_organism', 'seq_len', 'first_author', 'title', 'year', 'jo', 'region_mutated', 'molecule_name', 'selection_assay', 'selection_type', 'coarse_selection_type', 'raw_DMS_phenotype_name', 'raw_DMS_directionality', 'DMS_binarization_cutoff', 'DMS_binarization_method', 'MSA_filename', 'MSA_start', 'MSA_end', 'MSA_len', 'MSA_N_eff', 'MSA_Neff_L', 'MSA_Neff_L_category', 'pdb_file', 'pdb_range', 'ProteinGym_version']

def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))

def write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')

def main() -> None:
    reference = {row['DMS_filename']: row for row in read_csv(REFERENCE)}
    manifest = read_csv(MANIFEST)
    meta_rows: List[Dict[str, object]] = []
    curation_rows: List[Dict[str, object]] = []
    missing = []
    for assay_row in manifest:
        assay = assay_row['assay']
        ref = reference.get(assay)
        if ref is None:
            missing.append(assay)
            continue
        meta = {'assay': assay, 'protein_key': assay_row['protein_key']}
        meta.update({field: ref.get(field, '') for field in KEEP_FIELDS})
        meta_rows.append(meta)
        official_type = ref.get('coarse_selection_type', '') or 'Unknown'
        curation_rows.append({'assay': assay, 'protein_key': assay_row['protein_key'], 'assay_type': official_type, 'assay_type_official': official_type, 'selection_assay': ref.get('selection_assay', ''), 'selection_type': ref.get('selection_type', ''), 'coarse_selection_type': ref.get('coarse_selection_type', ''), 'molecule_name': ref.get('molecule_name', ''), 'raw_DMS_phenotype_name': ref.get('raw_DMS_phenotype_name', ''), 'raw_DMS_directionality': ref.get('raw_DMS_directionality', ''), 'title': ref.get('title', ''), 'first_author': ref.get('first_author', ''), 'year': ref.get('year', ''), 'jo': ref.get('jo', ''), 'taxon': ref.get('taxon', ''), 'source_organism': ref.get('source_organism', ''), 'MSA_N_eff': ref.get('MSA_N_eff', ''), 'MSA_Neff_L': ref.get('MSA_Neff_L', ''), 'MSA_Neff_L_category': ref.get('MSA_Neff_L_category', ''), 'pdb_file': ref.get('pdb_file', ''), 'pdb_range': ref.get('pdb_range', ''), 'needs_manual_review': '0'})
    if missing:
        print(f'warning: {len(missing)} manifest assays missing from official reference')
    meta_fields = ['assay', 'protein_key'] + KEEP_FIELDS
    curation_fields = list(curation_rows[0].keys())
    write_csv(OUT_META, meta_rows, meta_fields)
    write_csv(OUT_CURATION, curation_rows, curation_fields)
if __name__ == '__main__':
    main()
