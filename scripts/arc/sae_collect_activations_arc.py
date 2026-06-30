#!/usr/bin/env python3
"""Collect ARC z_H (and optionally z_L) activations for SAE training.

ARC sibling of ``scripts/sae/sae_collect_activations_maze.py``. The only ARC-specific
part is model loading: the Path-A adapted checkpoint is a *file*
(``checkpoints/arc2-adapted-evalonly/step_7391``) with ``all_config.yaml`` alongside,
and its puzzle_emb table needs the size-reconciliation that
``controlled_common.load_model_and_dataloader`` already implements. The activation
collection itself is task-agnostic, so we reuse the maze ``collect_all_activations``
(it slices the trailing ``seq_len`` grid-cell positions of z_H, dropping the
puzzle-emb prefix — identical for ARC's 30x30 = 900-cell encoding).

Output (under --output_dir):
  activations_zH.pt   {'z_H': [N, steps, 900, 512] fp16, 'inputs','labels',
                       'puzzle_ids','per_step_*','n_puzzles','max_steps',
                       'seq_len','hidden_dim'}

Usage (GPU, via srun on ws-l3-019):
  python scripts/arc/sae_collect_activations_arc.py --n_puzzles 300 \
      --output_dir results/arc/sae_study
"""
from __future__ import annotations
import os, sys, time, argparse, logging

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import numpy as np

from scripts.controlled.controlled_common import load_model_and_dataloader
from scripts.sae.sae_collect_activations_maze import collect_all_activations
from utils.arc_targets import SEQ_LEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

ARC_ADAPTED_CKPT = os.path.join(
    REPO_ROOT, "checkpoints", "arc2-adapted-evalonly", "step_7391")


def main():
    ap = argparse.ArgumentParser(description="Collect ARC z_H/z_L activations for SAE")
    ap.add_argument("--checkpoint", default=ARC_ADAPTED_CKPT)
    ap.add_argument("--n_puzzles", type=int, default=300)
    ap.add_argument("--max_steps", type=int, default=16)
    ap.add_argument("--also_collect_zL", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output_dir", default="results/arc/sae_study")
    ap.add_argument("--store_dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    logger.info(f"Device: {device}  checkpoint: {args.checkpoint}")

    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    hidden_dim = int(model.config.hidden_size)
    logger.info(f"Model loaded. seq_len={SEQ_LEN} hidden={hidden_dim}")

    store_dtype = torch.float16 if args.store_dtype == "float16" else torch.float32
    t0 = time.time()
    result = collect_all_activations(
        model, test_loader, device,
        n_puzzles=args.n_puzzles, seq_len=SEQ_LEN, hidden_dim=hidden_dim,
        max_steps=args.max_steps, also_collect_zL=args.also_collect_zL,
        store_dtype=store_dtype)
    logger.info(f"Collection took {time.time()-t0:.1f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    z_H_path = os.path.join(args.output_dir, "activations_zH.pt")
    if args.also_collect_zL:
        z_H_only = {k: v for k, v in result.items() if k != "z_L"}
        torch.save(z_H_only, z_H_path)
        z_L_only = {k: v for k, v in z_H_only.items() if k != "z_H"}
        z_L_only["z_L"] = result["z_L"]
        torch.save(z_L_only, os.path.join(args.output_dir, "activations_zL.pt"))
    else:
        torch.save(result, z_H_path)
    logger.info(f"Saved {z_H_path} ({os.path.getsize(z_H_path)/2**20:.1f} MB) "
                f"z_H shape {tuple(result['z_H'].shape)}")


if __name__ == "__main__":
    main()
