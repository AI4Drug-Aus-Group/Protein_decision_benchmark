from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD_OUT = ROOT / "results" / "local_matched_reproduction" / "proteinnpt_sampled_2000_len400"
NEW_OUT = ROOT / "results" / "sota_extension" / "proteinnpt_single_only_budget_len400"


def main() -> None:
    old_manifest = pd.read_csv(OLD_OUT / "proteinnpt_assay_resource_manifest.csv")
    new_manifest = pd.read_csv(NEW_OUT / "proteinnpt_assay_resource_manifest.csv")
    old_lookup = {
        row["assay_id"]: row
        for _, row in old_manifest[old_manifest["eligible"].astype(str).str.lower().isin(["true", "1"])].iterrows()
    }
    rows = []
    for _, row in new_manifest[new_manifest["eligible"].astype(str).str.lower().isin(["true", "1"])].iterrows():
        assay_id = row["assay_id"]
        old = old_lookup.get(assay_id)
        new_h5 = NEW_OUT / "runtime" / "embeddings" / "MSA_Transformer" / f"{assay_id}.h5"
        old_h5 = OLD_OUT / "runtime" / "embeddings" / "MSA_Transformer" / f"{assay_id}.h5"
        new_meta = new_h5.with_suffix(new_h5.suffix + ".metadata.json")
        old_meta = old_h5.with_suffix(old_h5.suffix + ".metadata.json")
        copied = False
        reason = "no_matching_old_manifest"
        if old is not None:
            same_input = str(old.get("embedding_input_sha256")) == str(row.get("embedding_input_sha256"))
            same_model = str(old.get("embedding_model_sha256")) == str(row.get("embedding_model_sha256"))
            if same_input and same_model and old_h5.exists() and old_meta.exists():
                new_h5.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_h5, new_h5)
                shutil.copy2(old_meta, new_meta)
                copied = True
                reason = "copied_exact_matching_embedding"
            elif not same_input:
                reason = "candidate_pool_differs"
            elif not same_model:
                reason = "embedding_model_hash_differs"
            else:
                reason = "old_embedding_file_missing"
        rows.append({
            "assay": row["assay"],
            "assay_id": assay_id,
            "copied": copied,
            "reason": reason,
            "old_embedding": str(old_h5),
            "new_embedding": str(new_h5),
        })
    out = NEW_OUT / "embedding_reuse_status.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(json.dumps({
        "status_file": str(out),
        "n_assays": len(rows),
        "n_copied": int(sum(item["copied"] for item in rows)),
    }, indent=2))


if __name__ == "__main__":
    main()
