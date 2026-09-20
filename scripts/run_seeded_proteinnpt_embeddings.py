from __future__ import annotations

import argparse
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def install_fast_msa_patch() -> None:
    import proteinnpt.utils.embedding_utils as embedding_utils

    def patched_fast_msa_mode_setup(
        model,
        seqs_to_score,
        MSA_sequences,
        MSA_weights,
        num_MSA_sequences,
        path_to_clustalomega,
    ):
        model.MSA_sample_sequences = [
            (description, sequence.upper())
            for description, sequence in MSA_sequences[:num_MSA_sequences]
        ]
        fast_MSA_short_names_mapping = {}
        fast_MSA_short_names = []
        for seq_index, sequence in enumerate(list(seqs_to_score)):
            short_name = f"mutant_to_score_{seq_index}"
            fast_MSA_short_names_mapping[sequence] = short_name
            fast_MSA_short_names.append(short_name)
        if model.MSA_sample_sequences:
            msa_length = len(model.MSA_sample_sequences[0][1])
            if all(len(sequence) == msa_length for sequence in seqs_to_score):
                fast_MSA_aligned_sequences = [
                    (name, sequence.upper())
                    for name, sequence in zip(fast_MSA_short_names, seqs_to_score)
                ] + model.MSA_sample_sequences
                return model, fast_MSA_aligned_sequences, fast_MSA_short_names_mapping
        fast_MSA_aligned_sequences = embedding_utils.align_new_sequences_to_msa(
            model.MSA_sample_sequences,
            list(seqs_to_score),
            fast_MSA_short_names,
            clustalomega_path=path_to_clustalomega,
        )
        excluded = set(fast_MSA_short_names)
        model.MSA_sample_sequences = [
            item for item in fast_MSA_aligned_sequences if item[0] not in excluded
        ]
        return model, fast_MSA_aligned_sequences, fast_MSA_short_names_mapping

    embedding_utils.fast_MSA_mode_setup = patched_fast_msa_mode_setup


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wrapper-seed", type=int, required=True)
    parser.add_argument("--official-script", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    random.seed(args.wrapper_seed)
    np.random.seed(args.wrapper_seed)
    torch.manual_seed(args.wrapper_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.wrapper_seed)
    if "--fast_MSA_mode" in remaining:
        install_fast_msa_patch()
    sys.argv = [str(args.official_script), *remaining]
    runpy.run_path(str(args.official_script), run_name="__main__")


if __name__ == "__main__":
    main()
