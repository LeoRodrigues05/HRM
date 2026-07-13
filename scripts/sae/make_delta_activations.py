#!/usr/bin/env python3
"""SAE Exp-1: Build Δz_H activation dumps for the update-space SAE.

Hypothesis (F3 follow-up): the *state* z_H is a dense accumulated solution
(state-SAEs never sparsified it, L0 stuck ~700), but the per-step *writes*
Δz_H = z_H^{t+1} − z_H^t are sparse and interpretable. This script turns an
existing z_H trajectory dump into a Δz_H dump so the unchanged SAE pipeline
(sae_train.py / sae_sweep.py) can train on it — those read only ``data['z_H']``
and expect layout ``[N, steps, 81, D]``, so the deltas are stored under the same
``z_H`` key.

Input
-----
  results/sae_study/bptt_study/<ckpt>/activations_zH.pt   # [N,16,81,512]

Output
------
  results/sae_study/bptt_study/<ckpt>/activations_dzH.pt  # z_H := [N,15,81,512]

Usage
-----
    python scripts/sae/make_delta_activations.py \
        --input  results/sae_study/bptt_study/hrm_stock/activations_zH.pt \
        --output results/sae_study/bptt_study/hrm_stock/activations_dzH.pt
"""

import os
import sys
import argparse
import logging

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


def make_delta_dump(input_path: str, output_path: str, dtype: str = "float32") -> dict:
    """Load a z_H trajectory dump and write a Δz_H dump.

    Δz_H[t] = z_H[t+1] − z_H[t], for t = 0 .. steps-2  (so steps → steps-1).
    Stored under key ``z_H`` (layout preserved) so the SAE pipeline is untouched;
    also duplicated under ``dz_H`` for clarity, and the source metadata is kept
    for the Stage-2 causal work.
    """
    logger.info(f"Loading z_H trajectory dump from {input_path} ...")
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    z_H = data["z_H"]  # [N, steps, 81, D]
    logger.info(f"  z_H shape: {tuple(z_H.shape)}  dtype={z_H.dtype}")
    n, steps, cells, dim = z_H.shape

    # Compute the per-step write. Do it in float32 for numerical headroom,
    # then optionally downcast for on-disk size (the SAE loader upcasts anyway).
    dz = (z_H[:, 1:].float() - z_H[:, :-1].float())  # [N, steps-1, 81, D]
    out_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
    dz = dz.to(out_dtype)
    logger.info(f"  Δz_H shape: {tuple(dz.shape)}  dtype={dz.dtype}")

    # Quick sanity stats: per-step delta norm (mean L2 over cells/hidden).
    with torch.no_grad():
        step_norms = dz.float().norm(dim=-1).mean(dim=(0, 2))  # [steps-1]
    logger.info("  mean ‖Δz_H‖ per transition: "
                + ", ".join(f"{v:.3f}" for v in step_norms.tolist()))

    result = {
        "z_H": dz,      # canonical key consumed by sae_train.load_activations
        "dz_H": dz,     # explicit alias
        "inputs": data.get("inputs"),
        "labels": data.get("labels"),
        "puzzle_ids": data.get("puzzle_ids"),
        # Predictions/accuracy are per-state (steps entries); the deltas index the
        # transitions between them. Keep the originals verbatim for Stage-2 use.
        "per_step_accuracy": data.get("per_step_accuracy"),
        "per_step_preds": data.get("per_step_preds"),
        "n_puzzles": data.get("n_puzzles", n),
        "max_steps": steps - 1,
        "source_max_steps": steps,
        "hidden_dim": dim,
        "is_delta": True,
        "source_path": os.path.abspath(input_path),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(result, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Saved Δz_H dump to {output_path} ({size_mb:.1f} MB)")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build Δz_H dump from a z_H trajectory dump")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to activations_zH.pt ([N,steps,81,D])")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: sibling activations_dzH.pt)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"],
                        help="On-disk dtype for the deltas (loader upcasts float16)")
    args = parser.parse_args()

    output = args.output or os.path.join(os.path.dirname(args.input), "activations_dzH.pt")
    make_delta_dump(args.input, output, dtype=args.dtype)


if __name__ == "__main__":
    main()
