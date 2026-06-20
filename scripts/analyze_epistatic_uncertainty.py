from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ZERO = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
RESID = ROOT / 'results' / 'epistasis' / 'strict_double_epistasis_residuals.csv'
OUT_DIR = ROOT / 'results' / 'epistasis'
METHODS = ['ProSST-2048', 'ESM3', 'S2F_MSA', 'S3F_MSA', 'PoET', 'VenusREM']

def parse_float(value: str) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float('nan')

def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum(((x - m) ** 2 for x in xs)) / (len(xs) - 1))

def ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    out = {}
    n = len(ordered)
    for i, key in enumerate(ordered):
        out[key] = i / max(1, n - 1)
    return out

def load_scores(assays: set[str]) -> dict[tuple[str, str], dict[str, float]]:
    out = {}
    for assay in assays:
        path = ZERO / assay
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                rec = {}
                for method in METHODS:
                    x = parse_float(row.get(method, ''))
                    if x is not None:
                        rec[method] = x
                if rec:
                    out[assay, row['mutant']] = rec
    return out

def enrich_for_assay(rows: list[dict[str, object]], k_values: list[int]) -> list[dict[str, object]]:
    abs_vals = [float(r['abs_epistasis_residual']) for r in rows]
    pos_vals = [float(r['epistasis_residual']) for r in rows if float(r['epistasis_residual']) > 0]
    abs_cut = sorted(abs_vals)[max(0, int(0.75 * (len(abs_vals) - 1)))]
    pos_cut = sorted(pos_vals)[max(0, int(0.75 * (len(pos_vals) - 1)))] if pos_vals else float('inf')
    base_abs = mean([1.0 if float(r['abs_epistasis_residual']) >= abs_cut else 0.0 for r in rows])
    base_pos = mean([1.0 if float(r['epistasis_residual']) >= pos_cut else 0.0 for r in rows]) if pos_vals else float('nan')
    base_sign = mean([float(r['sign_epistasis_candidate']) for r in rows])
    out = []
    for score_name in ['ensemble_std', 'additive_ensemble_disagreement', 'sequence_structure_disagreement', 'ensemble_mean']:
        ranked = sorted(rows, key=lambda r: float(r[score_name]), reverse=True)
        for k in k_values:
            top = ranked[:min(k, len(ranked))]
            if not top:
                continue
            abs_rate = mean([1.0 if float(r['abs_epistasis_residual']) >= abs_cut else 0.0 for r in top])
            pos_rate = mean([1.0 if float(r['epistasis_residual']) >= pos_cut else 0.0 for r in top]) if pos_vals else float('nan')
            sign_rate = mean([float(r['sign_epistasis_candidate']) for r in top])
            out.append({'assay': rows[0]['assay'], 'score': score_name, 'k': k, 'n_variants': len(rows), 'high_abs_residual_rate': abs_rate, 'high_abs_residual_enrichment': abs_rate / base_abs if base_abs else float('nan'), 'positive_residual_rate': pos_rate, 'positive_residual_enrichment': pos_rate / base_pos if base_pos and (not math.isnan(base_pos)) else float('nan'), 'sign_epistasis_rate': sign_rate, 'sign_epistasis_enrichment': sign_rate / base_sign if base_sign else float('nan')})
    return out

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({k for row in rows for k in row}) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')

def main() -> None:
    with RESID.open(newline='', encoding='utf-8') as fh:
        resid_rows = list(csv.DictReader(fh))
    assays = {r['assay'] for r in resid_rows}
    scores = load_scores(assays)
    by_assay: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in resid_rows:
        rec = scores.get((r['assay'], r['mutant']))
        if not rec or len(rec) < 2:
            continue
        vals = list(rec.values())
        method_ranks = ranks(rec)
        ensemble_mean = mean(vals)
        struct_vals = [method_ranks[m] for m in ['ProSST-2048', 'S2F_MSA', 'S3F_MSA'] if m in method_ranks]
        seq_vals = [method_ranks[m] for m in ['ESM3', 'PoET', 'VenusREM'] if m in method_ranks]
        additive = parse_float(r['additive_prediction'])
        score_rec = {'assay': r['assay'], 'mutant': r['mutant'], 'epistasis_residual': float(r['epistasis_residual']), 'abs_epistasis_residual': float(r['abs_epistasis_residual']), 'sign_epistasis_candidate': float(r['sign_epistasis_candidate']), 'ensemble_mean': ensemble_mean, 'ensemble_std': stdev(vals), 'additive_ensemble_disagreement': abs((additive or 0.0) - ensemble_mean), 'sequence_structure_disagreement': abs(mean(seq_vals) - mean(struct_vals)) if seq_vals and struct_vals else 0.0}
        by_assay[r['assay']].append(score_rec)
    detail = []
    for rows in by_assay.values():
        if len(rows) >= 100:
            detail.extend(enrich_for_assay(rows, [10, 50, 100]))
    write_csv(OUT_DIR / 'epistatic_uncertainty_enrichment_detail.csv', detail)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for r in detail:
        grouped[str(r['score']), str(r['k'])].append(r)
    summary = []
    for (score, k), rows in grouped.items():
        summary.append({'score': score, 'k': k, 'n_assays': len(rows), 'mean_high_abs_residual_enrichment': mean([float(r['high_abs_residual_enrichment']) for r in rows if not math.isnan(float(r['high_abs_residual_enrichment']))]), 'mean_positive_residual_enrichment': mean([float(r['positive_residual_enrichment']) for r in rows if not math.isnan(float(r['positive_residual_enrichment']))]), 'mean_sign_epistasis_enrichment': mean([float(r['sign_epistasis_enrichment']) for r in rows if not math.isnan(float(r['sign_epistasis_enrichment']))])})
    summary.sort(key=lambda r: (r['score'], int(r['k'])))
    write_csv(OUT_DIR / 'epistatic_uncertainty_enrichment_summary.csv', summary)
if __name__ == '__main__':
    main()
