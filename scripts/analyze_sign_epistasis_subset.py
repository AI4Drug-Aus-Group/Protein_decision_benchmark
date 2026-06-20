from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence
ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
RESIDUALS = ROOT / 'results' / 'epistasis' / 'strict_double_epistasis_residuals.csv'
OUT_DIR = ROOT / 'results' / 'epistasis'
METHODS = ['additive_prediction', 'VenusREM', 'ProSST-2048', 'S3F_MSA', 'PoET', 'RSALOR', 'ESM3', 'TranceptEVE_L', 'GEMME', 'SaProt_650M_AF2']

def parse_float(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed

def ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[indexed[k][0]] = avg
        i = j
    return out

def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum(((a - mx) * (b - my) for a, b in zip(x, y)))
    dx = math.sqrt(sum(((a - mx) ** 2 for a in x)))
    dy = math.sqrt(sum(((b - my) ** 2 for b in y)))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)

def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    return pearson(ranks(x), ranks(y))

def top_recall(y: Sequence[float], pred: Sequence[float], top_frac: float=0.01, k: int=100) -> Optional[float]:
    if len(y) < 3:
        return None
    k = min(k, len(y))
    top_true_n = max(1, math.ceil(len(y) * top_frac))
    pred_top = set(sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:k])
    true_top = set(sorted(range(len(y)), key=lambda i: y[i], reverse=True)[:top_true_n])
    return len(pred_top & true_top) / len(true_top)

def load_residuals() -> Dict[str, List[Dict[str, str]]]:
    by_assay: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with RESIDUALS.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row['sign_epistasis_candidate'] == '1':
                by_assay[row['assay']].append(row)
    return by_assay

def load_zero(assay: str, methods: Sequence[str]) -> Dict[str, Dict[str, float]]:
    zero_methods = [m for m in methods if m != 'additive_prediction']
    scores: Dict[str, Dict[str, float]] = {}
    with (ZERO_DIR / assay).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rec = {}
            for method in zero_methods:
                value = parse_float(row.get(method))
                if value is not None:
                    rec[method] = value
            scores[row['mutant']] = rec
    return scores

def process_assay(assay: str, rows: Sequence[Dict[str, str]], min_n: int) -> List[Dict[str, object]]:
    if len(rows) < min_n:
        return []
    zero = load_zero(assay, METHODS)
    out = []
    for method in METHODS:
        y = []
        pred = []
        residual = []
        for row in rows:
            mutant = row['mutant']
            if method == 'additive_prediction':
                score = float(row['additive_prediction'])
            else:
                score = zero.get(mutant, {}).get(method)
                if score is None:
                    continue
            y.append(float(row['score_double']))
            residual.append(float(row['epistasis_residual']))
            pred.append(score)
        if len(y) < min_n:
            continue
        rho = spearman(pred, y)
        rho_res = spearman(pred, residual)
        recall = top_recall(y, pred)
        out.append({'assay': assay, 'method': method, 'subset': 'sign_epistasis_candidate', 'n': len(y), 'spearman_score': '' if rho is None else rho, 'abs_spearman_score': '' if rho is None else abs(rho), 'spearman_residual': '' if rho_res is None else rho_res, 'top1pct_recall_at_100': '' if recall is None else recall})
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-n', type=int, default=50)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', default=str(OUT_DIR / 'sign_epistasis_subset_metrics.csv'))
    args = parser.parse_args()
    by_assay = load_residuals()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    fieldnames = ['assay', 'method', 'subset', 'n', 'spearman_score', 'abs_spearman_score', 'spearman_residual', 'top1pct_recall_at_100']
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_assay, assay, rows, args.min_n) for assay, rows in by_assay.items()]
            for future in as_completed(futures):
                writer.writerows(future.result())
    print(f'wrote {out_path}')
if __name__ == '__main__':
    main()
