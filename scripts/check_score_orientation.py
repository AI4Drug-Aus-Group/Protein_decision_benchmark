from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "zero_shot" / "zero_shot_all_methods_metrics.csv"
MANIFEST = ROOT / "results" / "batch0" / "assay_manifest.csv"
OUT_DIR = ROOT / "results" / "checks"


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def load_protein_keys() -> Dict[str, str]:
    out = {}
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["assay"]] = row["protein_key"]
    return out


def write_rows(path: Path, rows: List[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(METRICS))
    args = parser.parse_args()

    protein_key = load_protein_keys()
    by_method: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    with Path(args.metrics).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rho = parse_float(row.get("spearman", ""))
            if rho is None:
                continue
            rec = dict(row)
            rec["signed_spearman"] = rho
            rec["protein_key"] = protein_key.get(row["assay"], row["assay"])
            by_method[row["method"]].append(rec)

    summary = []
    protein_summary = []
    for method, rows in by_method.items():
        signed = [float(r["signed_spearman"]) for r in rows]
        abs_vals = [abs(v) for v in signed]
        pos = sum(1 for v in signed if v > 0)
        neg = sum(1 for v in signed if v < 0)
        orientation = "higher_is_better" if mean(signed) >= 0 else "lower_is_better"
        summary.append(
            {
                "method": method,
                "n_assays": len(rows),
                "mean_signed_spearman": mean(signed),
                "mean_abs_spearman": mean(abs_vals),
                "fraction_positive": pos / len(rows),
                "fraction_negative": neg / len(rows),
                "suggested_orientation": orientation,
            }
        )

        per_protein: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            per_protein[str(r["protein_key"])].append(float(r["signed_spearman"]))
        protein_signed = [mean(vs) for vs in per_protein.values()]
        protein_summary.append(
            {
                "method": method,
                "n_proteins": len(per_protein),
                "protein_level_mean_signed_spearman": mean(protein_signed),
                "protein_level_mean_abs_spearman": mean(abs(v) for v in protein_signed),
                "assay_level_mean_signed_spearman": mean(signed),
                "assay_level_mean_abs_spearman": mean(abs_vals),
                "protein_minus_assay_abs_spearman": mean(abs(v) for v in protein_signed) - mean(abs_vals),
            }
        )

    summary.sort(key=lambda r: float(r["mean_abs_spearman"]), reverse=True)
    protein_summary.sort(key=lambda r: float(r["protein_level_mean_abs_spearman"]), reverse=True)
    write_rows(
        OUT_DIR / "zero_shot_score_orientation_summary.csv",
        summary,
        [
            "method",
            "n_assays",
            "mean_signed_spearman",
            "mean_abs_spearman",
            "fraction_positive",
            "fraction_negative",
            "suggested_orientation",
        ],
    )
    write_rows(
        OUT_DIR / "zero_shot_protein_level_robustness.csv",
        protein_summary,
        [
            "method",
            "n_proteins",
            "protein_level_mean_signed_spearman",
            "protein_level_mean_abs_spearman",
            "assay_level_mean_signed_spearman",
            "assay_level_mean_abs_spearman",
            "protein_minus_assay_abs_spearman",
        ],
    )


if __name__ == "__main__":
    main()
