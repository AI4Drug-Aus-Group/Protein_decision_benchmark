from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
ROOT = Path(__file__).resolve().parents[1]
RESIDUALS = ROOT / 'results' / 'epistasis' / 'strict_double_epistasis_residuals.csv'
ZERO_DIR = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
OUT_DETAIL = ROOT / 'results' / 'epistasis' / 'epistatic_hit_discovery_detail.csv'
OUT_SUMMARY = ROOT / 'results' / 'epistasis' / 'epistatic_hit_discovery_summary.csv'
METHODS = ['additive_prediction', 'VenusREM', 'ProSST-2048', 'S3F_MSA', 'S2F_MSA', 'PoET', 'ESM3', 'GEMME', 'TranceptEVE_L']
KS = [10, 20, 50, 100]

def parse_float(value: str) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(val) or math.isinf(val) else val

def mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else float('nan')

def load_zero_scores(assay: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    path = ZERO_DIR / assay
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rec = {}
            for method in METHODS:
                if method == 'additive_prediction':
                    continue
                val = parse_float(row.get(method, ''))
                if val is not None:
                    rec[method] = val
            out[row['mutant']] = rec
    return out

def main() -> None:
    by_assay: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with RESIDUALS.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            by_assay[row['assay']].append(row)
    detail = []
    for assay, rows in by_assay.items():
        if len(rows) < 50:
            continue
        zero = load_zero_scores(assay)
        abs_res = [(abs(parse_float(r.get('epistasis_residual', '')) or 0.0), r['mutant']) for r in rows]
        abs_res.sort(reverse=True)
        high_abs = {m for _, m in abs_res[:max(1, math.ceil(len(abs_res) * 0.25))]}
        positive_res = [(parse_float(r.get('epistasis_residual', '')) or 0.0, r['mutant']) for r in rows]
        positive_res.sort(reverse=True)
        positive_hits = {m for v, m in positive_res[:max(1, math.ceil(len(positive_res) * 0.1))] if v > 0}
        sign_hits = {r['mutant'] for r in rows if (r.get('sign_epistasis_candidate') or r.get('sign_epistasis') or r.get('is_sign_epistasis') or '') in {'1', 'True', 'true'}}
        if not sign_hits:
            sign_hits = high_abs
        score_by_method: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        for r in rows:
            mutant = r['mutant']
            add = parse_float(r.get('additive_prediction', ''))
            if add is not None:
                score_by_method['additive_prediction'].append((add, mutant))
            for method, val in zero.get(mutant, {}).items():
                score_by_method[method].append((val, mutant))
        for method, scored in score_by_method.items():
            if len(scored) < 50:
                continue
            scored.sort(reverse=True)
            for k in KS:
                top = {m for _, m in scored[:min(k, len(scored))]}
                expected_high = len(high_abs) / len(rows)
                expected_positive = len(positive_hits) / len(rows) if positive_hits else 0.0
                expected_sign = len(sign_hits) / len(rows) if sign_hits else 0.0
                high_rate = len(top & high_abs) / len(top)
                positive_rate = len(top & positive_hits) / len(top) if positive_hits else 0.0
                sign_rate = len(top & sign_hits) / len(top) if sign_hits else 0.0
                detail.append({'assay': assay, 'method': method, 'k': k, 'n_doubles': len(rows), 'high_abs_residual_hit_rate': high_rate, 'high_abs_residual_enrichment': high_rate / expected_high if expected_high else '', 'positive_residual_hit_rate': positive_rate, 'positive_residual_enrichment': positive_rate / expected_positive if expected_positive else '', 'sign_or_high_residual_hit_rate': sign_rate, 'sign_or_high_residual_enrichment': sign_rate / expected_sign if expected_sign else ''})
    OUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DETAIL.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(detail[0].keys()))
        writer.writeheader()
        writer.writerows(detail)
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in detail:
        groups[row['method'], str(row['k'])].append(row)
    summary = []
    for (method, k), rows in groups.items():
        summary.append({'method': method, 'k': k, 'n_assays': len(rows), 'mean_high_abs_residual_hit_rate': mean((float(r['high_abs_residual_hit_rate']) for r in rows)), 'mean_high_abs_residual_enrichment': mean((float(r['high_abs_residual_enrichment']) for r in rows if r['high_abs_residual_enrichment'] != '')), 'mean_positive_residual_hit_rate': mean((float(r['positive_residual_hit_rate']) for r in rows)), 'mean_positive_residual_enrichment': mean((float(r['positive_residual_enrichment']) for r in rows if r['positive_residual_enrichment'] != '')), 'mean_sign_or_high_residual_hit_rate': mean((float(r['sign_or_high_residual_hit_rate']) for r in rows)), 'mean_sign_or_high_residual_enrichment': mean((float(r['sign_or_high_residual_enrichment']) for r in rows if r['sign_or_high_residual_enrichment'] != ''))})
    summary.sort(key=lambda r: (int(r['k']), -float(r['mean_high_abs_residual_enrichment'])))
    with OUT_SUMMARY.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f'wrote {OUT_DETAIL}')
    print(f'wrote {OUT_SUMMARY}')
if __name__ == '__main__':
    main()
