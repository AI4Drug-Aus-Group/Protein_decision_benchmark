from __future__ import annotations
import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence
ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
INVENTORY = ROOT / 'results' / 'batch0' / 'method_inventory.csv'
OUT_DIR = ROOT / 'results' / 'binary_metrics'

def parse_float(value: str) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v

def parse_bin(value: str) -> Optional[int]:
    v = parse_float(value)
    if v is None:
        return None
    return 1 if v > 0 else 0

def auc_score(y: Sequence[int], s: Sequence[float]) -> Optional[float]:
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    indexed = sorted(enumerate(s), key=lambda item: item[1])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    sum_pos_ranks = sum((r for r, yy in zip(ranks, y) if yy == 1))
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def mcc_at_top_fraction(y: Sequence[int], s: Sequence[float], frac: float=0.1) -> float:
    n = len(y)
    k = max(1, math.ceil(n * frac))
    pred_pos = set(sorted(range(n), key=lambda i: s[i], reverse=True)[:k])
    tp = sum((1 for i in pred_pos if y[i] == 1))
    fp = k - tp
    fn = sum(y) - tp
    tn = n - tp - fp - fn
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom else 0.0

def recall_at_top_fraction(y: Sequence[int], s: Sequence[float], frac: float=0.1) -> Optional[float]:
    positives = sum(y)
    if positives == 0:
        return None
    k = max(1, math.ceil(len(y) * frac))
    pred_idx = sorted(range(len(y)), key=lambda i: s[i], reverse=True)[:k]
    return sum((y[i] for i in pred_idx)) / positives

def load_methods(limit: int=0) -> List[str]:
    with INVENTORY.open(newline='', encoding='utf-8') as fh:
        methods = [row['method'] for row in csv.DictReader(fh)]
    return methods[:limit] if limit else methods

def process_assay(path: Path, methods: Sequence[str]) -> List[Dict[str, object]]:
    labels: List[int] = []
    scores: Dict[str, List[float]] = {m: [] for m in methods}
    present: Dict[str, List[int]] = {m: [] for m in methods}
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            y = parse_bin(row.get('DMS_score_bin', row.get('DMS_bin_score', '')))
            if y is None:
                continue
            row_idx = len(labels)
            labels.append(y)
            for method in methods:
                z = parse_float(row.get(method, ''))
                if z is not None:
                    scores[method].append(z)
                    present[method].append(row_idx)
    rows = []
    for method in methods:
        idx = present[method]
        if len(idx) < 10:
            continue
        y = [labels[i] for i in idx]
        if sum(y) == 0 or sum(y) == len(y):
            continue
        s = scores[method]
        auc = auc_score(y, s)
        rec10 = recall_at_top_fraction(y, s, 0.1)
        rec01 = recall_at_top_fraction(y, s, 0.01)
        rows.append({'assay': path.name, 'method': method, 'n': len(y), 'positive_rate': sum(y) / len(y), 'auc': '' if auc is None else auc, 'abs_auc_minus_half': '' if auc is None else abs(auc - 0.5), 'mcc_top10pct': mcc_at_top_fraction(y, s, 0.1), 'recall_pos_at_top10pct': '' if rec10 is None else rec10, 'recall_pos_at_top1pct': '' if rec01 is None else rec01})
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--method-limit', type=int, default=0)
    parser.add_argument('--out', default=str(OUT_DIR / 'zero_shot_binary_metrics.csv'))
    args = parser.parse_args()
    methods = load_methods(args.method_limit)
    paths = sorted(ZERO_DIR.glob('*.csv'))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ['assay', 'method', 'n', 'positive_rate', 'auc', 'abs_auc_minus_half', 'mcc_top10pct', 'recall_pos_at_top10pct', 'recall_pos_at_top1pct']
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_assay, p, methods) for p in paths]
            for future in as_completed(futures):
                writer.writerows(future.result())
    print(f'wrote {out}')
if __name__ == '__main__':
    main()
