"""Check that a released checkpoint's puzzle-embedding table lines up with the
local dataset build.

The embedding table lives at `inner.puzzle_emb.weights` (see
models/sparse_embedding.py, where it is a persistent buffer named `weights`).
Checkpoints saved from a torch.compile'd model carry an `_orig_mod.` prefix,
which is stripped here the same way evaluate.py:53 does it.

Usage:
    python scripts/arc/check_puzzle_emb_alignment.py \
        --data data/arc-2-aug-1000 \
        --ckpt /path/to/HRM-checkpoint-ARC-2/<step-file>
"""

import argparse
import json
import os
from collections import Counter

import torch


PUZZLE_EMB_SUFFIX = "inner.puzzle_emb.weights"


def load_emb_rows(ckpt_path: str) -> int:
    """Row count of the puzzle-embedding table in a checkpoint."""
    try:
        state = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    except Exception:
        state = torch.load(ckpt_path, map_location="cpu")

    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    matches = [k for k in state if k.endswith(PUZZLE_EMB_SUFFIX)]
    if not matches:
        raise KeyError(
            f"no key ending in {PUZZLE_EMB_SUFFIX!r} in {ckpt_path}; "
            f"first keys present: {list(state)[:5]}"
        )
    if len(matches) > 1:
        raise KeyError(f"ambiguous puzzle-embedding keys: {matches}")

    return state[matches[0]].shape[0]


def base_puzzle(identifier: str) -> str:
    """Strip the `_t<transform>_<permutation>` augmentation suffix.

    Identifiers look like `df9fd884` (the original) or `df9fd884_t3_0768314952`
    (an augmentation), per dataset/build_arc_dataset.py.
    """
    return identifier.split("_", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="dataset dir, e.g. data/arc-2-aug-1000")
    ap.add_argument("--ckpt", required=True, help="checkpoint file to inspect")
    args = ap.parse_args()

    with open(os.path.join(args.data, "train", "dataset.json")) as f:
        num_puzzle_identifiers = json.load(f)["num_puzzle_identifiers"]

    emb_rows = load_emb_rows(args.ckpt)

    print(f"local build  num_puzzle_identifiers : {num_puzzle_identifiers}")
    print(f"released     {PUZZLE_EMB_SUFFIX} rows : {emb_rows}")
    aligned = emb_rows == num_puzzle_identifiers
    print(f"row counts   {'MATCH' if aligned else 'MISMATCH'}")

    # Sanity read on identifiers.json: distinct base puzzles and mean variants
    # per puzzle, which should land near 1001 (original + 1000 augmentations).
    with open(os.path.join(args.data, "identifiers.json")) as f:
        identifiers = json.load(f)

    print(f"\nidentifiers.json entries          : {len(identifiers)}")
    if len(identifiers) != num_puzzle_identifiers:
        print("  WARNING: disagrees with dataset.json num_puzzle_identifiers")

    # index 0 is the reserved <blank> identifier, not a real puzzle
    real = [i for i in identifiers if i != "<blank>"]
    counts = Counter(base_puzzle(i) for i in real)
    mean_variants = len(real) / len(counts)

    print(f"distinct base puzzles             : {len(counts)}")
    print(f"mean variants per puzzle          : {mean_variants:.2f}")

    short = {name: n for name, n in counts.items() if n < 1001}
    if short:
        print(f"\npuzzles with < 1001 variants      : {len(short)}")
        for name, n in sorted(short.items(), key=lambda kv: kv[1])[:10]:
            print(f"  {name}: {n}")
        print(
            "  -> these hit the 'augmentation not full' path during the build;\n"
            "     every identifier after the first one is shifted."
        )

    if not aligned:
        print("\nMismatch: alignment failed, the fine-tune was compensating.")
    else:
        print(
            "\nRow counts agree, but that only means both builds enumerated the same\n"
            "NUMBER of variants -- the RNG could still have drawn a different set in a\n"
            "different order. The decisive test is behavioral; run evaluate.py."
        )


if __name__ == "__main__":
    main()
