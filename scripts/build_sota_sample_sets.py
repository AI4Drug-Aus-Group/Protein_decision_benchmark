from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sota_extension" / "sample_sets"
HOLDOUT = ROOT / "results" / "sota_extension" / "selected_double_splits" / "selected_double_common_holdout.csv"


MULTIMUTANT_ASSAYS = [
    "GFP_AEQVI_Sarkisyan_2016.csv",
    "GRB2_HUMAN_Faure_2021.csv",
    "HIS7_YEAST_Pokusaeva_2019.csv",
]

SELECTED_DOUBLE_ASSAYS = [
    "F7YBW8_MESOW_Aakre_2015.csv",
    "GFP_AEQVI_Sarkisyan_2016.csv",
    "GRB2_HUMAN_Faure_2021.csv",
    "HIS7_YEAST_Pokusaeva_2019.csv",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    single_to_multi = pd.DataFrame({"assay": MULTIMUTANT_ASSAYS})
    single_to_multi.to_csv(OUT / "proteinnpt_multimutant_assays.csv", index=False)

    holdout = pd.read_csv(HOLDOUT)
    holdout = holdout[holdout["assay"].isin(SELECTED_DOUBLE_ASSAYS)].copy()
    rows = []
    for assay, block in holdout.groupby("assay", sort=True):
        mutants = sorted(block["mutant"].astype(str).unique())
        chosen = mutants[: min(100, len(mutants))]
        rows.extend({"assay": assay, "mutant": mutant} for mutant in chosen)
    selected_double = pd.DataFrame(rows)
    selected_double.to_csv(OUT / "selected_double_evaluation_candidates.csv", index=False)
    selected_double_by_seed = holdout.merge(selected_double, on=["assay", "mutant"], how="inner")
    selected_double_by_seed.to_csv(OUT / "selected_double_evaluation_holdout_by_seed.csv", index=False)

    metadata = {
        "multimutant_assays": MULTIMUTANT_ASSAYS,
        "selected_double_assays": SELECTED_DOUBLE_ASSAYS,
        "selected_double_max_heldout_double_mutants_per_assay": 100,
        "selected_double_selection_rule": "Lexicographic mutant-order sampling within each assay from the precomputed held-out double-mutant set. Fitness values are not used.",
        "selected_double_n_required_evaluation_candidates": int(len(selected_double)),
        "selected_double_n_seeded_evaluation_rows": int(len(selected_double_by_seed)),
    }
    (OUT / "sample_set_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
