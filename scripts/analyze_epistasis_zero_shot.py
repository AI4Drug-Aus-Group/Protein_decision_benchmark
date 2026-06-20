from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
RESIDUALS = ROOT / 'results' / 'epistasis' / 'strict_double_epistasis_residuals.csv'
OUT_DIR = ROOT / 'results' / 'epistasis'
DEFAULT_METHODS = ['Site_Independent', 'EVmutation', 'EVE_ensemble', 'MSA_Transformer_ensemble', 'GEMME', 'ESM1v_ensemble', 'ESM2_650M', 'ESM2_15B', 'Tranception_L', 'TranceptEVE_L', 'ProteinMPNN', 'ESM-IF1', 'SaProt_650M_AF2', 'ProSST-2048', 'ESM3', 'ESMC-600M', 'xTrimoPGLM-100B-int4', 'Progen3_3b']

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
    result = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            result[indexed[k][0]] = avg_rank
        i = j
    return result

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

def load_residuals() -> Dict[str, List[Dict[str, str]]]:
    by_assay: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with RESIDUALS.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            by_assay[row['assay']].append(row)
    return by_assay

def load_zero_scores(assay: str, methods: Sequence[str]) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    with (ZERO_DIR / assay).open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            record = {}
            for method in methods:
                value = parse_float(row.get(method))
                if value is not None:
                    record[method] = value
            scores[row['mutant']] = record
    return scores

def percentile_threshold(values: Sequence[float], pct: float) -> float:
    if not values:
        return float('nan')
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(math.ceil(len(values) * pct)) - 1))
    return values[idx]

def top_recall(labels: Sequence[int], scores: Sequence[float], k: int) -> Optional[float]:
    positives = sum(labels)
    if positives == 0 or len(scores) < 3:
        return None
    k = min(k, len(scores))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return sum((labels[i] for i in top_idx)) / positives

def process_assay(assay: str, rows: Sequence[Dict[str, str]], methods: Sequence[str], min_n: int) -> List[Dict[str, object]]:
    if len(rows) < min_n:
        return []
    zero = load_zero_scores(assay, methods)
    mutants = []
    score_double = []
    additive = []
    residual = []
    abs_residual = []
    sign_epi = []
    for row in rows:
        mutant = row['mutant']
        if mutant not in zero:
            continue
        mutants.append(mutant)
        score_double.append(float(row['score_double']))
        additive.append(float(row['additive_prediction']))
        residual.append(float(row['epistasis_residual']))
        abs_residual.append(float(row['abs_epistasis_residual']))
        sign_epi.append(int(row['sign_epistasis_candidate']))
    if len(mutants) < min_n:
        return []
    add_rho = spearman(additive, score_double)
    abs_threshold = percentile_threshold(abs_residual, 0.75)
    high_abs = [int(value >= abs_threshold) for value in abs_residual]
    out = [{'assay': assay, 'method': 'additive_prediction', 'n': len(mutants), 'spearman_double_score': '' if add_rho is None else add_rho, 'abs_spearman_double_score': '' if add_rho is None else abs(add_rho), 'gain_vs_additive_abs_spearman': 0.0, 'spearman_epistasis_residual': '', 'spearman_abs_epistasis_residual': '', 'top100_high_abs_residual_recall': top_recall(high_abs, additive, 100), 'top100_sign_epistasis_recall': top_recall(sign_epi, additive, 100)}]
    for method in methods:
        y = []
        add_subset = []
        res = []
        abs_res = []
        high = []
        sign = []
        pred = []
        for i, mutant in enumerate(mutants):
            value = zero[mutant].get(method)
            if value is None:
                continue
            pred.append(value)
            y.append(score_double[i])
            add_subset.append(additive[i])
            res.append(residual[i])
            abs_res.append(abs_residual[i])
            high.append(high_abs[i])
            sign.append(sign_epi[i])
        if len(y) < min_n:
            continue
        rho_score = spearman(pred, y)
        rho_res = spearman(pred, res)
        rho_abs_res = spearman(pred, abs_res)
        local_add = spearman(add_subset, y)
        gain = ''
        if rho_score is not None and local_add is not None:
            gain = abs(rho_score) - abs(local_add)
        out.append({'assay': assay, 'method': method, 'n': len(y), 'spearman_double_score': '' if rho_score is None else rho_score, 'abs_spearman_double_score': '' if rho_score is None else abs(rho_score), 'gain_vs_additive_abs_spearman': gain, 'spearman_epistasis_residual': '' if rho_res is None else rho_res, 'spearman_abs_epistasis_residual': '' if rho_abs_res is None else rho_abs_res, 'top100_high_abs_residual_recall': top_recall(high, pred, 100), 'top100_sign_epistasis_recall': top_recall(sign, pred, 100)})
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--min-n', type=int, default=100)
    parser.add_argument('--out', default=str(OUT_DIR / 'zero_shot_epistasis_double_metrics.csv'))
    args = parser.parse_args()
    methods = [m for m in args.methods.split(',') if m]
    by_assay = load_residuals()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    fieldnames = ['assay', 'method', 'n', 'spearman_double_score', 'abs_spearman_double_score', 'gain_vs_additive_abs_spearman', 'spearman_epistasis_residual', 'spearman_abs_epistasis_residual', 'top100_high_abs_residual_recall', 'top100_sign_epistasis_recall']
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_assay, assay, rows, methods, args.min_n) for assay, rows in by_assay.items()]
            for future in as_completed(futures):
                writer.writerows(future.result())
    print(f'wrote {out_path}')
if __name__ == '__main__':
    main()
