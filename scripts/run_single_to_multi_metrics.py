from __future__ import annotations
import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence
ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
OUT_DIR = ROOT / 'results' / 'single_to_multi'
DEFAULT_METHODS = ['Site_Independent', 'EVmutation', 'EVE_ensemble', 'MSA_Transformer_ensemble', 'GEMME', 'ESM1v_ensemble', 'ESM2_650M', 'ESM2_15B', 'Tranception_L', 'TranceptEVE_L', 'ProteinMPNN', 'ESM-IF1', 'SaProt_650M_AF2', 'ProSST-2048', 'ESM3', 'ESMC-600M', 'xTrimoPGLM-100B-int4', 'Progen3_3b']

def parse_float(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed

def mutation_order(mutant: str) -> int:
    if not mutant or mutant in {'WT', 'wildtype', '_wt'}:
        return 0
    return mutant.count(':') + 1

def order_bucket(order: int) -> str:
    if order == 1:
        return 'single'
    if order == 2:
        return 'double'
    if order >= 3:
        return 'higher'
    return 'wt_or_unknown'

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

def top1_recall(y: Sequence[float], pred: Sequence[float], k: int=100) -> Optional[float]:
    if len(y) < 3:
        return None
    k = min(k, len(y))
    pred_top = set(sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:k])
    true_top = set(sorted(range(len(y)), key=lambda i: y[i], reverse=True)[:max(1, math.ceil(len(y) * 0.01))])
    return len(pred_top & true_top) / len(true_top)

def additive_all_singles(records: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    singles = {m: r['DMS_score'] for m, r in records.items() if mutation_order(m) == 1}
    if not singles:
        return {}
    mean_single = sum(singles.values()) / len(singles)
    effects = {m: score - mean_single for m, score in singles.items()}
    pred = {}
    for mutant in records:
        pred[mutant] = mean_single + sum((effects.get(part, 0.0) for part in mutant.split(':')))
    return pred

def process_assay(path: Path, methods: Sequence[str]) -> List[Dict[str, object]]:
    records: Dict[str, Dict[str, float]] = {}
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            y = parse_float(row.get('DMS_score'))
            if y is None:
                continue
            rec = {'DMS_score': y}
            for method in methods:
                z = parse_float(row.get(method))
                if z is not None:
                    rec[method] = z
            records[row['mutant']] = rec
    pred_by_method: Dict[str, Dict[str, float]] = {'additive_all_singles': additive_all_singles(records)}
    for method in methods:
        pred_by_method[method] = {m: r[method] for m, r in records.items() if method in r}
    rows: List[Dict[str, object]] = []
    for method, preds in pred_by_method.items():
        for bucket in ['single', 'double', 'higher']:
            y = []
            p = []
            for mutant, rec in records.items():
                if mutant not in preds:
                    continue
                if order_bucket(mutation_order(mutant)) != bucket:
                    continue
                y.append(rec['DMS_score'])
                p.append(preds[mutant])
            rho = spearman(p, y)
            recall = top1_recall(y, p)
            rows.append({'assay': path.name, 'method': method, 'order_bucket': bucket, 'n': len(y), 'spearman': '' if rho is None else rho, 'abs_spearman': '' if rho is None else abs(rho), 'top1pct_recall_at_100': '' if recall is None else recall})
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', default=str(OUT_DIR / 'single_to_multi_representative_metrics.csv'))
    args = parser.parse_args()
    methods = [m for m in args.methods.split(',') if m]
    paths = sorted(ZERO_DIR.glob('*.csv'))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    fieldnames = ['assay', 'method', 'order_bucket', 'n', 'spearman', 'abs_spearman', 'top1pct_recall_at_100']
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_assay, path, methods) for path in paths]
            for future in as_completed(futures):
                writer.writerows(future.result())
    print(f'wrote {out_path}')
if __name__ == '__main__':
    main()
