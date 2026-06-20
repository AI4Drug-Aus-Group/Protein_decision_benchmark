from __future__ import annotations
import argparse
import csv
import math
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
ROOT = Path(__file__).resolve().parents[1]
DMS_DIR = ROOT / 'proteingym' / 'extracted' / 'DMS_ProteinGym_substitutions'
ZERO_ZIP = ROOT / 'proteingym' / 'zips' / 'zero_shot_substitutions_scores.zip'
ZERO_DIR = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
OUT_DIR = ROOT / 'results' / 'zero_shot'
BASE_COLUMNS = {'mutant', 'mutated_sequence', 'DMS_score', 'DMS_score_bin', 'DMS_bin_score'}
DEFAULT_METHODS = ['Site_Independent', 'EVmutation', 'EVE_ensemble', 'MSA_Transformer_ensemble', 'GEMME', 'ESM1v_ensemble', 'ESM2_650M', 'ESM2_15B', 'Tranception_L', 'TranceptEVE_L', 'ProteinMPNN', 'ESM-IF1', 'SaProt_650M_AF2', 'ProSST-2048', 'ESM3', 'ESMC-600M', 'xTrimoPGLM-100B-int4', 'Progen3_3b']

def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'na'}:
        return None
    try:
        value_float = float(text)
    except ValueError:
        return None
    if math.isnan(value_float) or math.isinf(value_float):
        return None
    return value_float

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
    n = len(x)
    if n < 3:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum(((a - mean_x) * (b - mean_y) for a, b in zip(x, y)))
    den_x = math.sqrt(sum(((a - mean_x) ** 2 for a in x)))
    den_y = math.sqrt(sum(((b - mean_y) ** 2 for b in y)))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)

def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    return pearson(ranks(x), ranks(y))

def dcg(relevances: Sequence[float]) -> float:
    return sum(((2.0 ** rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(relevances)))

def minmax_relevance(values: Sequence[float]) -> List[float]:
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]

def top_metrics(y_true: Sequence[float], y_pred: Sequence[float], k: int) -> Dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {}
    k = min(k, n)
    top_pred_idx = sorted(range(n), key=lambda i: y_pred[i], reverse=True)[:k]
    rel = minmax_relevance(y_true)
    ndcg_num = dcg([rel[i] for i in top_pred_idx])
    ideal_idx = sorted(range(n), key=lambda i: y_true[i], reverse=True)[:k]
    ndcg_den = dcg([rel[i] for i in ideal_idx])
    actual_top_1 = set(sorted(range(n), key=lambda i: y_true[i], reverse=True)[:max(1, math.ceil(n * 0.01))])
    actual_top_5 = set(sorted(range(n), key=lambda i: y_true[i], reverse=True)[:max(1, math.ceil(n * 0.05))])
    pred_set = set(top_pred_idx)
    return {'k': float(k), 'ndcg_at_k': ndcg_num / ndcg_den if ndcg_den else 0.0, 'top1pct_recall_at_k': len(pred_set & actual_top_1) / len(actual_top_1), 'top5pct_recall_at_k': len(pred_set & actual_top_5) / len(actual_top_5), 'best_true_in_pred_top_k': max((y_true[i] for i in top_pred_idx)), 'mean_true_in_pred_top_k': sum((y_true[i] for i in top_pred_idx)) / k}

def score_rows_from_extracted(assay: str) -> Iterable[Dict[str, str]]:
    path = ZERO_DIR / assay
    with path.open(newline='', encoding='utf-8') as fh:
        yield from csv.DictReader(fh)

def score_rows_from_zip(assay: str) -> Iterable[Dict[str, str]]:
    with zipfile.ZipFile(ZERO_ZIP) as zf:
        with zf.open(assay) as fh:
            text = (line.decode('utf-8') for line in fh)
            yield from csv.DictReader(text)

def available_methods() -> List[str]:
    if ZERO_DIR.exists() and any(ZERO_DIR.glob('*.csv')):
        path = next(ZERO_DIR.glob('*.csv'))
        with path.open(newline='', encoding='utf-8') as fh:
            header = next(csv.reader(fh))
    else:
        with zipfile.ZipFile(ZERO_ZIP) as zf:
            name = next((name for name in zf.namelist() if name.endswith('.csv')))
            with zf.open(name) as fh:
                header = fh.readline().decode('utf-8').strip().split(',')
    return [col for col in header if col not in BASE_COLUMNS]

def process_assay(assay: str, methods: Sequence[str], k: int, from_zip: bool) -> List[Dict[str, object]]:
    reader = score_rows_from_zip(assay) if from_zip else score_rows_from_extracted(assay)
    y_by_method: Dict[str, Tuple[List[float], List[float]]] = {method: ([], []) for method in methods}
    order_counts: Dict[int, int] = {}
    for row in reader:
        y = parse_float(row.get('DMS_score', ''))
        if y is None:
            continue
        mutant = row.get('mutant', '')
        order = mutant.count(':') + 1 if mutant and mutant not in {'WT', 'wildtype', '_wt'} else 0
        order_counts[order] = order_counts.get(order, 0) + 1
        for method in methods:
            pred = parse_float(row.get(method, ''))
            if pred is not None:
                y_by_method[method][0].append(y)
                y_by_method[method][1].append(pred)
    rows: List[Dict[str, object]] = []
    for method, (ys, preds) in y_by_method.items():
        rho = spearman(preds, ys)
        metrics = top_metrics(ys, preds, k) if len(ys) >= 3 else {}
        row = {'assay': assay, 'method': method, 'n': len(ys), 'spearman': '' if rho is None else rho, 'abs_spearman': '' if rho is None else abs(rho), 'n_singles': order_counts.get(1, 0), 'n_doubles': order_counts.get(2, 0), 'n_higher_order': sum((count for order, count in order_counts.items() if order >= 3))}
        row.update(metrics)
        rows.append(row)
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--all-methods', action='store_true')
    parser.add_argument('--method-chunk-size', type=int, default=12, help='When --all-methods is used, evaluate this many methods at a time to bound memory.')
    parser.add_argument('--assays', default='')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--out', default=str(OUT_DIR / 'zero_shot_metrics.csv'))
    args = parser.parse_args()
    methods = available_methods() if args.all_methods else [m for m in args.methods.split(',') if m]
    available = set(available_methods())
    missing = [method for method in methods if method not in available]
    if missing:
        raise SystemExit(f'missing methods in zero-shot scores: {missing}')
    if args.assays:
        assays = [name for name in args.assays.split(',') if name]
    else:
        assays = sorted((path.name for path in DMS_DIR.glob('*.csv')))
    from_zip = not (ZERO_DIR.exists() and all(((ZERO_DIR / assay).exists() for assay in assays)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['assay', 'method', 'n', 'spearman', 'abs_spearman', 'n_singles', 'n_doubles', 'n_higher_order', 'k', 'ndcg_at_k', 'top1pct_recall_at_k', 'top5pct_recall_at_k', 'best_true_in_pred_top_k', 'mean_true_in_pred_top_k']
    method_chunks = [methods]
    if args.all_methods and args.method_chunk_size > 0:
        method_chunks = [methods[i:i + args.method_chunk_size] for i in range(0, len(methods), args.method_chunk_size)]
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for chunk_index, method_chunk in enumerate(method_chunks, start=1):
            print(f'method_chunk {chunk_index}/{len(method_chunks)} size={len(method_chunk)}')
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(process_assay, assay, method_chunk, args.k, from_zip) for assay in assays]
                for future in as_completed(futures):
                    for row in future.result():
                        writer.writerow(row)
    print(f'wrote {out_path}')
if __name__ == '__main__':
    main()
