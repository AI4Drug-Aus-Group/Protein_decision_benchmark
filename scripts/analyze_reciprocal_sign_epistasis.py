from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESID = ROOT / 'results' / 'epistasis' / 'strict_double_epistasis_residuals.csv'
ZERO = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
OUT = ROOT / 'results' / 'epistasis'
METHODS = ['additive_prediction', 'VenusREM', 'ProSST-2048', 'S3F_MSA', 'S2F_MSA', 'ESM3', 'PoET', 'GEMME', 'TranceptEVE_L']

def parse_float(x: str) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float('nan')

def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float('nan')
    mx, my = (mean(xs), mean(ys))
    vx = sum(((x - mx) ** 2 for x in xs))
    vy = sum(((y - my) ** 2 for y in ys))
    if vx <= 0 or vy <= 0:
        return float('nan')
    return sum(((x - mx) * (y - my) for x, y in zip(xs, ys))) / math.sqrt(vx * vy)

def rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks

def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rankdata(xs), rankdata(ys))

def load_scores(assay: str) -> dict[str, dict[str, float]]:
    path = ZERO / assay
    out = {}
    if not path.exists():
        return out
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rec = {}
            for method in METHODS:
                if method == 'additive_prediction':
                    continue
                x = parse_float(row.get(method, ''))
                if x is not None:
                    rec[method] = x
            out[row['mutant']] = rec
    return out

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) if rows else []
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')

def main() -> None:
    by_assay = defaultdict(list)
    with RESID.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            wt = parse_float(row['wt_proxy'])
            a = parse_float(row['score_single_a'])
            b = parse_float(row['score_single_b'])
            ab = parse_float(row['score_double'])
            if None in {wt, a, b, ab}:
                continue
            da, db, dab = (a - wt, b - wt, ab - wt)
            reciprocal = int(da * dab < 0 and db * dab < 0)
            if reciprocal:
                row = dict(row)
                row['reciprocal_sign_epistasis_candidate'] = '1'
                by_assay[row['assay']].append(row)
    detail = []
    for assay, rows in by_assay.items():
        scores = load_scores(assay)
        for row in rows:
            rec = {'assay': assay, 'mutant': row['mutant'], 'score_double': row['score_double'], 'additive_prediction': row['additive_prediction'], 'epistasis_residual': row['epistasis_residual'], 'abs_epistasis_residual': row['abs_epistasis_residual'], 'reciprocal_sign_epistasis_candidate': '1'}
            rec.update(scores.get(row['mutant'], {}))
            detail.append(rec)
    write_csv(OUT / 'reciprocal_sign_epistasis_candidates.csv', detail)
    by_method = defaultdict(list)
    for r in detail:
        y = parse_float(str(r['score_double']))
        if y is None:
            continue
        for method in METHODS:
            pred = parse_float(str(r.get(method, '')))
            if pred is not None:
                by_method[method].append((pred, y))
    summary = []
    for method, pairs in by_method.items():
        if len(pairs) < 20:
            continue
        x, y = zip(*pairs)
        summary.append({'method': method, 'n_variants': len(pairs), 'abs_spearman_double_score': abs(spearman(list(x), list(y)))})
    summary.sort(key=lambda r: float(r['abs_spearman_double_score']), reverse=True)
    write_csv(OUT / 'reciprocal_sign_epistasis_summary.csv', summary)
if __name__ == '__main__':
    main()
