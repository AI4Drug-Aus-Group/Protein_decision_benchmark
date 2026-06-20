from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
ZERO = ROOT / 'proteingym' / 'extracted' / 'zero_shot_substitutions_scores'
INV = ROOT / 'results' / 'batch0' / 'method_inventory.csv'
OUT_DIR = ROOT / 'results' / 'zero_shot'

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float('nan')

def metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(y) & np.isfinite(s)
    y = y[ok]
    s = s[ok]
    n = len(y)
    if n < 2:
        return {}
    y_std = float(np.std(y)) or 1.0
    s_std = float(np.std(s)) or 1.0
    ys = (y - float(np.mean(y))) / y_std
    sscores = (s - float(np.mean(s))) / s_std
    top_true_1 = set(np.argsort(-y)[:max(1, math.ceil(0.01 * n))])
    top_true_10 = set(np.argsort(-y)[:max(1, math.ceil(0.1 * n))])
    top_pred_10 = set(np.argsort(-s)[:min(10, n)])
    top_pred_20 = set(np.argsort(-s)[:min(20, n)])
    random_rate = len(top_true_1) / n
    hit10 = len(top_pred_10 & top_true_1) / len(top_pred_10)
    hit20 = len(top_pred_20 & top_true_1) / len(top_pred_20)
    return {'pearson': float(np.corrcoef(y, s)[0, 1]) if y_std > 0 and s_std > 0 else float('nan'), 'rmse_z': float(np.sqrt(np.mean((ys - sscores) ** 2))), 'mae_z': float(np.mean(np.abs(ys - sscores))), 'top10_hit_rate_top1pct': hit10, 'top20_hit_rate_top1pct': hit20, 'top10_enrichment_top1pct': hit10 / random_rate if random_rate else float('nan'), 'top20_enrichment_top1pct': hit20 / random_rate if random_rate else float('nan'), 'top10_recall_top10pct': len(top_pred_10 & top_true_10) / len(top_true_10), 'top20_recall_top10pct': len(top_pred_20 & top_true_10) / len(top_true_10)}

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) if rows else []
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')

def main() -> None:
    with INV.open(newline='', encoding='utf-8') as fh:
        methods = [r['method'] for r in csv.DictReader(fh)]
    detail = []
    for path in sorted(ZERO.glob('*.csv')):
        header = pd.read_csv(path, nrows=0).columns
        usecols = ['DMS_score'] + [m for m in methods if m in header]
        df = pd.read_csv(path, usecols=usecols)
        y = pd.to_numeric(df['DMS_score'], errors='coerce').to_numpy(dtype=float)
        for method in usecols[1:]:
            s = pd.to_numeric(df[method], errors='coerce').to_numpy(dtype=float)
            ok_n = int(np.sum(np.isfinite(y) & np.isfinite(s)))
            if ok_n < 20:
                continue
            rec = {'assay': path.name, 'method': method, 'n': ok_n}
            rec.update(metrics(y, s))
            detail.append(rec)
    write_csv(OUT_DIR / 'zero_shot_extended_metrics.csv', detail)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in detail:
        grouped[str(r['method'])].append(r)
    summary = []
    for method, rows in grouped.items():
        summary.append({'method': method, 'n_assays': len(rows), 'mean_pearson': mean([float(r['pearson']) for r in rows if not math.isnan(float(r['pearson']))]), 'mean_rmse_z': mean([float(r['rmse_z']) for r in rows if not math.isnan(float(r['rmse_z']))]), 'mean_mae_z': mean([float(r['mae_z']) for r in rows if not math.isnan(float(r['mae_z']))]), 'mean_top10_enrichment_top1pct': mean([float(r['top10_enrichment_top1pct']) for r in rows if not math.isnan(float(r['top10_enrichment_top1pct']))]), 'mean_top20_enrichment_top1pct': mean([float(r['top20_enrichment_top1pct']) for r in rows if not math.isnan(float(r['top20_enrichment_top1pct']))]), 'mean_top10_hit_rate_top1pct': mean([float(r['top10_hit_rate_top1pct']) for r in rows]), 'mean_top20_hit_rate_top1pct': mean([float(r['top20_hit_rate_top1pct']) for r in rows])})
    summary.sort(key=lambda r: float(r['mean_top20_enrichment_top1pct']), reverse=True)
    write_csv(OUT_DIR / 'zero_shot_extended_metrics_summary.csv', summary)
if __name__ == '__main__':
    main()
