from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'qc' / 'qc_report.json'
REQUIRED_FILES = ['results/batch0/assay_manifest.csv', 'results/batch0/assay_strata.csv', 'results/zero_shot/zero_shot_all_methods_metrics.csv', 'results/zero_shot/zero_shot_method_summary.csv', 'results/low_n/low_n_top20_all_percentile_summary.csv', 'results/low_n/low_n_multi_zscore_ridge_nongiant_percentile_summary.csv', 'results/epistasis/zero_shot_epistasis_method_summary.csv', 'results/epistasis/high_residual_subset_summary.csv', 'results/epistasis/sign_epistasis_subset_summary.csv', 'results/single_to_multi/single_to_multi_summary.csv', 'results/multi_evolve_style/pairwise_residual_higher_order_summary.csv', 'results/active_learning/active_learning_summary.csv', 'results/active_learning/active_learning_summary_seeds0_4.csv', 'results/active_learning/active_learning_hit_time_summary_seeds0_4.csv', 'results/audit/zero_shot_score_orientation_summary.csv', 'results/audit/zero_shot_protein_level_robustness.csv', 'results/binary_metrics/zero_shot_binary_summary.csv', 'results/batch0/msa_depth_summary.csv', 'results/batch0/assay_type_curation_refined.csv', 'results/decision_map/assay_decision_features.csv', 'results/decision_map/decision_map_by_assay_type.csv', 'results/decision_map/decision_map_by_msa_depth.csv', 'results/decision_map/decision_map_by_epistasis_strength.csv', 'results/double_data_value/single_plus_double_delta_summary.csv', 'results/structure_proxy/structure_aware_gain_summary.csv', 'results/low_n/low_n_supervised_baselines_nongiant_percentile_summary.csv', 'results/low_n/low_n_combined_nongiant_percentile_summary.csv', 'results/figure_tables/fig2_zero_shot_top_methods.csv', 'results/figure_tables/fig3_single_to_multi_key_methods.csv', 'results/figure_tables/fig4_epistasis_subset_key_methods.csv', 'results/figure_tables/fig5_low_n_top20_mixed20.csv', 'results/figure_tables/fig6_active_learning_curves.csv', 'results/figure_tables/fig6b_active_learning_hit_time.csv', 'results/figure_tables/fig7_pairwise_residual_higher_order.csv', 'results/figure_tables/fig10_low_n_supervised_combined_mixed20.csv', 'results/figure_tables/fig11_double_data_value_budget20.csv', 'results/figure_tables/fig12_zero_shot_binary_top_methods.csv', 'results/figure_tables/fig13a_decision_map_by_assay_type.csv', 'results/figure_tables/fig13b_decision_map_by_msa_depth.csv', 'results/figure_tables/fig13c_decision_map_by_epistasis_strength.csv', 'results/figure_tables/fig14_structure_aware_gain_summary.csv']

def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))

def as_float(value: str):
    if value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None

def check_range(path: Path, col: str, lo: float, hi: float) -> List[str]:
    issues = []
    for i, row in enumerate(read_rows(path), start=2):
        if col not in row:
            continue
        value = as_float(row[col])
        if value is None:
            continue
        if value < lo or value > hi:
            issues.append(f'{path}:{i} {col}={value} outside [{lo},{hi}]')
    return issues

def check_required_files() -> Dict[str, object]:
    missing = [p for p in REQUIRED_FILES if not (ROOT / p).exists()]
    sizes = {p: (ROOT / p).stat().st_size for p in REQUIRED_FILES if (ROOT / p).exists()}
    return {'missing': missing, 'sizes': sizes}

def check_giant_sampled() -> Dict[str, object]:
    path = ROOT / 'results' / 'low_n' / 'low_n_calibration_top20_giant_fast_metrics.percentile.csv'
    if not path.exists():
        return {'exists': False}
    rows = read_rows(path)
    sampled_cols = {'spearman_sampled': sum((1 for r in rows if r.get('spearman_sampled') == '1')), 'design_sampled': sum((1 for r in rows if r.get('design_sampled') == '1'))}
    assays = sorted({r['assay'] for r in rows})
    return {'exists': True, 'rows': len(rows), 'assays': assays, **sampled_cols}

def main() -> None:
    issues: List[str] = []
    for file_name in REQUIRED_FILES:
        path = ROOT / file_name
        if path.exists() and path.suffix == '.csv':
            for col in ['mean_abs_spearman', 'mean_abs_spearman_all', 'mean_abs_spearman_score', 'mean_abs_spearman_higher', 'top1pct_found_rate', 'mean_top1pct_recall_at_100', 'mean_best_observed_percentile', 'mean_best_top100_percentile']:
                issues.extend(check_range(path, col, 0.0, 1.0))
    report = {'required_files': check_required_files(), 'giant_sampled': check_giant_sampled(), 'range_issues': issues, 'status': 'pass' if not check_required_files()['missing'] and (not issues) else 'review'}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write('\n')
    print(f'wrote {OUT}')
    print(json.dumps(report, indent=2, sort_keys=True)[:4000])
if __name__ == '__main__':
    main()
