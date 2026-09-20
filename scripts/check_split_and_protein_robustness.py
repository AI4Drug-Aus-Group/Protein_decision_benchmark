from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "checks"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def deterministic_pick(protein_key: str, assays: list[str]) -> str:
    return min(assays, key=lambda a: hashlib.md5(f"{protein_key}|{a}".encode()).hexdigest())


def one_assay_per_protein() -> None:
    manifest = {}
    by_protein: dict[str, list[str]] = defaultdict(list)
    with (ROOT / "results" / "batch0" / "assay_manifest.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            manifest[row["assay"]] = row["protein_key"]
            by_protein[row["protein_key"]].append(row["assay"])
    selected = {deterministic_pick(pk, assays) for pk, assays in by_protein.items()}
    rows_by_method: dict[str, list[float]] = defaultdict(list)
    with (ROOT / "results" / "zero_shot" / "zero_shot_all_methods_metrics.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["assay"] not in selected:
                continue
            try:
                rows_by_method[row["method"]].append(abs(float(row["spearman"])))
            except (ValueError, KeyError):
                pass
    out = []
    for method, vals in rows_by_method.items():
        out.append(
            {
                "method": method,
                "n_selected_assays": len(vals),
                "mean_abs_spearman_one_assay_per_protein": sum(vals) / len(vals) if vals else "",
            }
        )
    out.sort(key=lambda r: float(r["mean_abs_spearman_one_assay_per_protein"]), reverse=True)
    write_csv(OUT / "zero_shot_one_assay_per_protein_sensitivity.csv", out)


def split_leakage() -> None:
    split_rows: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    duplicates = 0
    with (ROOT / "results" / "splits" / "low_n_splits.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["assay"], row["seed"], row["budget"], row["regime"])
            before = len(split_rows[key])
            split_rows[key].add(row["mutant"])
            duplicates += int(len(split_rows[key]) == before)
    manifest_n = {}
    with (ROOT / "results" / "batch0" / "assay_manifest.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            manifest_n[row["assay"]] = int(row["n_variants"])
    rows = []
    for key, mutants in split_rows.items():
        assay, seed, budget, regime = key
        b = int(budget)
        rows.append(
            {
                "assay": assay,
                "seed": seed,
                "budget": budget,
                "regime": regime,
                "n_train_unique": len(mutants),
                "expected_budget": b,
                "budget_overrun": int(len(mutants) > b),
                "budget_underrun": int(len(mutants) < min(b, manifest_n.get(assay, b))),
                "train_fraction": len(mutants) / manifest_n.get(assay, len(mutants)),
            }
        )
    summary = [
        {
            "n_split_groups": len(rows),
            "n_duplicate_split_entries": duplicates,
            "n_budget_overrun_groups": sum(int(r["budget_overrun"]) for r in rows),
            "n_budget_underrun_groups": sum(int(r["budget_underrun"]) for r in rows),
            "max_train_fraction": max(float(r["train_fraction"]) for r in rows) if rows else "",
        }
    ]
    write_csv(OUT / "low_n_split_leakage_check_detail.csv", rows)
    write_csv(OUT / "low_n_split_leakage_check_summary.csv", summary)


def active_candidate_pool_check() -> None:
    rows = []
    path = ROOT / "results" / "active_learning" / "active_learning_representative_metrics_seeds0_4.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["round"] == "0":
                rows.append(row)
    by_assay = defaultdict(list)
    for r in rows:
        by_assay[r["assay"]].append(float(r["pool_size"]))
    out = [
        {
            "assay": assay,
            "n_round0_rows": len(vals),
            "min_pool_size": min(vals),
            "max_pool_size": max(vals),
            "constant_pool_size": int(min(vals) == max(vals)),
        }
        for assay, vals in by_assay.items()
    ]
    write_csv(OUT / "active_learning_candidate_pool_check.csv", out)


def main() -> None:
    one_assay_per_protein()
    split_leakage()
    active_candidate_pool_check()


if __name__ == "__main__":
    main()
