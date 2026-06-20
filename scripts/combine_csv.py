from __future__ import annotations
import argparse
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    inputs = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in args.inputs.split(',') if p]
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    fieldnames = None
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as out_fh:
        writer = None
        for path in inputs:
            with path.open(newline='', encoding='utf-8') as in_fh:
                reader = csv.DictReader(in_fh)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
                    writer.writeheader()
                if reader.fieldnames != fieldnames:
                    raise ValueError(f'header mismatch in {path}')
                assert writer is not None
                writer.writerows(reader)
    print(f'wrote {out}')
if __name__ == '__main__':
    main()
