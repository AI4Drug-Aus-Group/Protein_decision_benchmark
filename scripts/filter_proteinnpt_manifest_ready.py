import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assays", default="")
    parser.add_argument("--require-embedding", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.assays:
        assays = {item.strip() for item in args.assays.split(",") if item.strip()}
        df = df[df["assay_id"].astype(str).isin(assays)].copy()
    if args.require_embedding:
        df = df[df["embedding_file"].map(lambda path: Path(str(path)).exists())].copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print({"out": str(args.out), "n_runs": int(len(df)), "n_assays": int(df["assay_id"].nunique())})


if __name__ == "__main__":
    main()
