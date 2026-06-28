#!/usr/bin/env python3
"""Experiment H1 (Track B): Localizability scorecard for the Universal Transformer pair.

Tests H1 on the available 1-step-gradient vs BPTT UT pair:
  - ut_1step: checkpoints/baselines/universal_transformer  (one_step_grad=True)
  - ut_bptt:  checkpoints/baselines/universal_transformer_standalone (one_step_grad=False)
              NOTE: no checkpoint files yet — skipped gracefully until trained.

Implements UTActivationHarness for single-state (z) rollout.

Usage:
    python scripts/analysis/baseline_localizability.py --tag ut_1step --device cuda
    python scripts/analysis/baseline_localizability.py --tag ut_bptt  --device cuda  # skipped if no ckpt

Outputs: results/localizability/<tag>/scorecard.json
"""
from __future__ import annotations
import os, sys, re, json, argparse, logging, time
from typing import Dict, List, Optional, Tuple, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from scripts.controlled.controlled_common import bootstrap_ci
from scripts.probes.e8_constraint_probes import (
    derive_per_cell_labels, train_binary, puzzle_disjoint_split,
)
from scripts.analysis.policy_improvement import task_value
from scripts.core.activation_ablation import _patch_attention_for_cpu

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

PROBE_TARGETS = ["violated_in_row", "violated_in_col", "violated_in_box", "per_cell_correct"]
PROBE_STEPS_1STEP = [0, 4, 8, 12, 15]   # for 16-step UT
PROBE_STEPS_BPTT  = [0]                  # for 1-step UT (halt_max_steps=1)

UT_CHECKPOINT_DIRS = {
    "ut_1step": os.path.join(REPO_ROOT, "checkpoints", "baselines", "universal_transformer"),
    "ut_bptt":  os.path.join(REPO_ROOT, "checkpoints", "baselines", "universal_transformer_standalone"),
}


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_ut_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return path to the latest step_* checkpoint file (not _all_preds.*), or None."""
    if not os.path.isdir(ckpt_dir):
        return None
    step_files = [
        f for f in os.listdir(ckpt_dir)
        if re.match(r'^step_\d+$', f)
    ]
    if not step_files:
        return None
    latest = max(step_files, key=lambda f: int(f.split("_")[1]))
    return os.path.join(ckpt_dir, latest)


# ---------------------------------------------------------------------------
# UT model loader
# ---------------------------------------------------------------------------

def load_ut_model(ckpt_dir: str, ckpt_file: str, device: torch.device):
    """Load a UniversalTransformerModel from a checkpoint directory.

    Returns (ut_model, test_loader, config).
    """
    import yaml
    from pretrain import PretrainConfig
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    from torch.utils.data import DataLoader
    from utils.functions import load_model_class
    from models.baselines.universal_transformer import UniversalTransformerModel

    config_path = os.path.join(ckpt_dir, "all_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(config_path) as f:
        config = PretrainConfig(**yaml.safe_load(f))

    # Build dataset / loader
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=1,
        rank=0,
        num_replicas=1,
    ), split="test")
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    meta = dataset.metadata

    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=1,
        vocab_size=meta.vocab_size,
        seq_len=meta.seq_len,
        num_puzzle_identifiers=meta.num_puzzle_identifiers,
        causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model_raw = model_cls(model_cfg)
        model_full = loss_cls(model_raw, **config.arch.loss.__pydantic_extra__)

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    mk = set(model_full.state_dict().keys())
    ck = set(ckpt.keys())
    if any(k.startswith("_orig_mod.") for k in mk) and not any(k.startswith("_orig_mod.") for k in ck):
        ckpt = {f"_orig_mod.{k}": v for k, v in ckpt.items()}
    elif any(k.startswith("_orig_mod.") for k in ck) and not any(k.startswith("_orig_mod.") for k in mk):
        ckpt = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}
    model_full.load_state_dict(ckpt, assign=True)
    model_full.to(device).eval()

    if device.type == "cpu":
        _patch_attention_for_cpu(model_full)

    # Unwrap
    m: Any = model_full
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if not isinstance(m, UniversalTransformerModel) and hasattr(m, "model"):
        m = m.model
    if not isinstance(m, UniversalTransformerModel):
        raise TypeError(f"Expected UniversalTransformerModel, got {type(m)}")

    return m, loader, config


# ---------------------------------------------------------------------------
# UTActivationHarness — single-state rollout with subspace ablation
# ---------------------------------------------------------------------------

class UTActivationHarness:
    """Rolls out a UniversalTransformerModel step by step, caching z.

    Analogous to ActivationAblator but for the single-state UT architecture.
    """

    def __init__(self, model, device: torch.device):
        from models.baselines.universal_transformer import (
            UniversalTransformerModel, UniversalTransformerInnerCarry,
        )
        self.model = model
        self.device = device
        self._InnerCarry = UniversalTransformerInnerCarry

    def _extract_batch(self, item) -> Dict[str, torch.Tensor]:
        if isinstance(item, (tuple, list)):
            return item[1] if len(item) >= 2 and isinstance(item[1], dict) else item[0]
        return item

    def collect_puzzles(self, loader, num_puzzles: int, seed: int = 42):
        """Collect (puzzle_idx, batch) pairs."""
        rng = np.random.RandomState(seed)
        items = []
        for idx, item in enumerate(loader):
            if len(items) >= num_puzzles:
                break
            batch = self._extract_batch(item)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            items.append((idx, batch))
        return items

    def _init_carry(self, batch: Dict[str, torch.Tensor]):
        carry = self.model.initial_carry(batch)
        carry.inner_carry = self._InnerCarry(
            z=carry.inner_carry.z.to(self.device)
        )
        carry.steps = carry.steps.to(self.device)
        carry.halted = carry.halted.to(self.device)
        carry.current_data = {k: v.to(self.device) for k, v in carry.current_data.items()}
        return carry

    def _step_carry(self, carry, batch: Dict[str, torch.Tensor],
                    subspace_Q: Optional[torch.Tensor] = None):
        """Run one ACT outer step, optionally projecting z before model.inner call.

        subspace_Q: [r, D] if provided → z' = z - (z @ Q.T) @ Q
        """
        # Compute new inner carry (apply halting reset)
        new_inner = self.model.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v,
            )
            for k, v in carry.current_data.items()
        }

        z_in = new_inner.z.clone()

        # Apply subspace projection
        if subspace_Q is not None:
            z = new_inner.z.float()
            Q = subspace_Q.float().to(self.device)
            coeffs = z @ Q.T   # [B, T, r]
            z_proj = coeffs @ Q  # [B, T, D]
            z_new = (z - z_proj).to(new_inner.z.dtype)
            new_inner = self._InnerCarry(z=z_new)

        # Forward inner
        with torch.no_grad():
            new_inner_out, logits, (q_halt, q_continue) = self.model.inner(
                new_inner, new_current_data
            )

        z_out = new_inner_out.z.clone()

        # Update carry halting
        new_steps_out = new_steps + 1
        is_last = new_steps_out >= self.model.config.halt_max_steps
        halted = is_last

        from models.baselines.universal_transformer import UniversalTransformerCarry
        new_carry = UniversalTransformerCarry(
            inner_carry=new_inner_out,
            steps=new_steps_out,
            halted=halted,
            current_data=new_current_data,
        )

        return new_carry, logits, z_in, z_out

    def run_and_cache(
        self, batch: Dict[str, torch.Tensor], max_steps: int,
    ) -> Dict[int, Dict[str, Any]]:
        """Roll out, caching z_in, z_out, logits, preds per step."""
        carry = self._init_carry(batch)
        cache = {}
        pel = int(self.model.inner.puzzle_emb_len)

        with torch.no_grad():
            for step in range(max_steps):
                carry, logits, z_in, z_out = self._step_carry(carry, batch, subspace_Q=None)
                preds = logits.argmax(-1)  # already stripped of pel
                cache[step] = {
                    "z_in": z_in.detach().clone(),
                    "z_out": z_out.detach().clone(),
                    "logits": logits.detach().clone(),
                    "preds": preds.detach().clone(),
                }
                if torch.all(carry.halted):
                    break

        return cache

    def run_with_subspace_ablation(
        self, batch: Dict[str, torch.Tensor],
        Q: torch.Tensor,  # [r, D]
        max_steps: int,
    ) -> Dict[int, Dict[str, Any]]:
        """Roll out with subspace ablation (z' = z - QQ^T z) at each step."""
        carry = self._init_carry(batch)
        cache = {}

        with torch.no_grad():
            for step in range(max_steps):
                carry, logits, z_in, z_out = self._step_carry(carry, batch, subspace_Q=Q)
                preds = logits.argmax(-1)
                cache[step] = {
                    "z_in": z_in.detach().clone(),
                    "z_out": z_out.detach().clone(),
                    "logits": logits.detach().clone(),
                    "preds": preds.detach().clone(),
                }
                if torch.all(carry.halted):
                    break

        return cache

    def run_with_full_ablation(
        self, batch: Dict[str, torch.Tensor], max_steps: int,
    ) -> Dict[int, Dict[str, Any]]:
        """Zeroing ablation on z at each step."""
        carry = self._init_carry(batch)
        cache = {}

        with torch.no_grad():
            for step in range(max_steps):
                # Zero out z
                new_inner = self.model.inner.reset_carry(carry.halted, carry.inner_carry)
                new_inner = self._InnerCarry(z=torch.zeros_like(new_inner.z))

                new_steps = torch.where(carry.halted, 0, carry.steps)
                new_current_data = {
                    k: torch.where(
                        carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                        batch[k], v,
                    )
                    for k, v in carry.current_data.items()
                }
                new_inner_out, logits, (_, _) = self.model.inner(new_inner, new_current_data)
                preds = logits.argmax(-1)

                new_steps_out = new_steps + 1
                is_last = new_steps_out >= self.model.config.halt_max_steps
                from models.baselines.universal_transformer import UniversalTransformerCarry
                carry = UniversalTransformerCarry(
                    inner_carry=new_inner_out,
                    steps=new_steps_out,
                    halted=is_last,
                    current_data=new_current_data,
                )
                cache[step] = {
                    "z_in": torch.zeros_like(new_inner.z),
                    "z_out": new_inner_out.z.detach().clone(),
                    "logits": logits.detach().clone(),
                    "preds": preds.detach().clone(),
                }
                if torch.all(carry.halted):
                    break

        return cache


# ---------------------------------------------------------------------------
# Scorecard computation for UT
# ---------------------------------------------------------------------------

def _get_ut_baselines(harness, batches, task, max_steps):
    baseline_values = {}
    for puzzle_idx, batch in batches:
        cache = harness.run_and_cache(batch, max_steps)
        last_s = max(cache.keys())
        preds_row = cache[last_s]["preds"][0].cpu().numpy()
        labels_row = batch["labels"][0].cpu().numpy()
        inputs_row = batch["inputs"][0].cpu().numpy()
        tv = task_value(preds_row, labels_row, inputs_row, task)
        baseline_values[puzzle_idx] = tv["value"]
    return baseline_values


def collect_ut_activations(
    harness: UTActivationHarness, batches: list,
    steps_to_record: List[int], max_steps: int,
):
    """Collect z_out per step for probe training."""
    pel = int(harness.model.inner.puzzle_emb_len)
    step_features: Dict[int, List[torch.Tensor]] = {s: [] for s in steps_to_record}
    step_labels: Dict[int, Dict[str, List[torch.Tensor]]] = {s: {} for s in steps_to_record}

    for puzzle_idx, batch in batches:
        cache = harness.run_and_cache(batch, max_steps)

        preds_last = cache[max(cache.keys())]["preds"][0]
        labels_1d = batch["labels"][0]
        inputs_1d = batch["inputs"][0]

        cell_labels = derive_per_cell_labels(
            preds_last.unsqueeze(0).cpu(),
            labels_1d.unsqueeze(0).cpu(),
            inputs_1d.unsqueeze(0).cpu(),
        )

        for s in steps_to_record:
            if s >= max_steps or s not in cache:
                continue
            z_out = cache[s]["z_out"][0, pel:].float().cpu()  # [seq, D]
            step_features[s].append(z_out)
            for tgt in PROBE_TARGETS:
                if tgt in cell_labels:
                    lbl = cell_labels[tgt][0].float().cpu()
                    if tgt not in step_labels[s]:
                        step_labels[s][tgt] = []
                    step_labels[s][tgt].append(lbl)

    result = {}
    for s in steps_to_record:
        if not step_features[s]:
            continue
        X = torch.cat(step_features[s], dim=0)
        lbls = {}
        for tgt in PROBE_TARGETS:
            if step_labels[s].get(tgt):
                lbls[tgt] = torch.cat(step_labels[s][tgt], dim=0)
        result[s] = {"X": X, "labels": lbls}
    return result


def run_ut_scorecard(
    harness: UTActivationHarness, batches: list, task: str,
    probe_steps: List[int], seeds: List[int],
    ranks: List[int], max_steps: int,
    device: torch.device, seed: int = 42,
) -> dict:
    """Compute all four scorecard metrics for the UT."""
    from scripts.analysis.localizability_scorecard import compute_probe_decodability
    D = int(harness.model.inner.config.hidden_size)
    pel = int(harness.model.inner.puzzle_emb_len)

    logger.info("[H1-UT] Collecting activations...")
    act_data = collect_ut_activations(harness, batches, probe_steps, max_steps)

    logger.info("[H1-UT] Metric 1: probe_decodability...")
    probe_dec = compute_probe_decodability(act_data, seeds, device)
    logger.info(f"  probe_decodability = {probe_dec['mean_val_acc']:.4f}")

    logger.info("[H1-UT] Metric 2: probe_causal_gap...")
    baseline_values = _get_ut_baselines(harness, batches, task, max_steps)
    baseline_mean = float(np.mean(list(baseline_values.values())))

    # Full ablation reference
    full_deltas = []
    for puzzle_idx, batch in batches:
        cache_abl = harness.run_with_full_ablation(batch, max_steps)
        last_s = max(cache_abl.keys())
        preds_row = cache_abl[last_s]["preds"][0].cpu().numpy()
        labels_row = batch["labels"][0].cpu().numpy()
        inputs_row = batch["inputs"][0].cpu().numpy()
        tv = task_value(preds_row, labels_row, inputs_row, task)
        full_deltas.append(tv["value"] - baseline_values[puzzle_idx])
    delta_full = float(np.mean(full_deltas))
    logger.info(f"  Δfull = {delta_full:.4f}")

    # Probe-direction ablation for causal gap
    best_step = max(act_data.keys())
    X_best = act_data[best_step]["X"].to(device)
    rng = torch.Generator()
    rng.manual_seed(seed)

    probe_deltas, random_deltas = [], []

    def _run_subspace_delta(Q_1d: torch.Tensor) -> List[float]:
        """Single-direction subspace ablation via UTActivationHarness."""
        Q = Q_1d.unsqueeze(0).to(device)  # [1, D]
        deltas = []
        for puzzle_idx, batch in batches:
            cache_abl = harness.run_with_subspace_ablation(batch, Q, max_steps)
            last_s = max(cache_abl.keys())
            preds_row = cache_abl[last_s]["preds"][0].cpu().numpy()
            labels_row = batch["labels"][0].cpu().numpy()
            inputs_row = batch["inputs"][0].cpu().numpy()
            tv = task_value(preds_row, labels_row, inputs_row, task)
            deltas.append(tv["value"] - baseline_values[puzzle_idx])
        return deltas

    for tgt in PROBE_TARGETS:
        if tgt not in act_data[best_step]["labels"]:
            continue
        y = act_data[best_step]["labels"][tgt].to(device)
        n_rows = X_best.shape[0]
        tr_idx, va_idx = puzzle_disjoint_split(n_rows, val_frac=0.2, seed=seeds[0], device=device)
        probe, _, _ = train_binary(X_best[tr_idx], y[tr_idx], X_best[va_idx], y[va_idx],
                                   epochs=50, lr=1e-2)
        w = probe.linear.weight[0].detach().float()
        w = w / w.norm().clamp(min=1e-8)
        probe_deltas.extend(_run_subspace_delta(w))

    for _ in range(5):
        w_rand = F.normalize(torch.randn(D, generator=rng), dim=0)
        random_deltas.extend(_run_subspace_delta(w_rand))

    probe_mean = float(np.mean(probe_deltas)) if probe_deltas else float("nan")
    random_mean = float(np.mean(random_deltas)) if random_deltas else float("nan")
    gap = (probe_mean - random_mean) if not (np.isnan(probe_mean) or np.isnan(random_mean)) else float("nan")

    causal_gap = {
        "probe_mean_delta": probe_mean,
        "random_mean_delta": random_mean,
        "probe_causal_gap": float(gap),
        "probe_ci": bootstrap_ci(probe_deltas),
        "random_ci": bootstrap_ci(random_deltas),
    }
    logger.info(f"  probe_causal_gap = {gap:+.4f}")

    logger.info("[H1-UT] Metrics 3+4: min_causal_rank + subspace_linearity...")
    # Build PCA pool from collected activations
    all_X = [data["X"] for data in act_data.values()]
    M = torch.cat(all_X, dim=0)
    if M.shape[0] > 200_000:
        idx = torch.randperm(M.shape[0])[:200_000]
        M = M[idx]
    M_c = M.float() - M.mean(0, keepdim=True).float()
    _, _, pca_V = torch.linalg.svd(M_c, full_matrices=False)
    pca_V = pca_V.float()

    # Damage curves
    from scripts.analysis.causal_subspace import _find_r_star, _subspace_linearity

    pca_curve = {}
    for r in ranks:
        r_c = min(r, pca_V.shape[0])
        Q = pca_V[:r_c].to(device)
        deltas_r = []
        for puzzle_idx, batch in batches:
            cache_abl = harness.run_with_subspace_ablation(batch, Q, max_steps)
            last_s = max(cache_abl.keys())
            preds_row = cache_abl[last_s]["preds"][0].cpu().numpy()
            labels_row = batch["labels"][0].cpu().numpy()
            inputs_row = batch["inputs"][0].cpu().numpy()
            tv = task_value(preds_row, labels_row, inputs_row, task)
            deltas_r.append(tv["value"] - baseline_values[puzzle_idx])
        ci = bootstrap_ci(deltas_r)
        pca_curve[r_c] = ci
        logger.info(f"  r={r_c}: Δ={ci['mean']:+.4f}")

    min_causal_rank = _find_r_star(pca_curve, delta_full, frac=0.9)
    subspace_linearity = _subspace_linearity(pca_curve)

    return {
        "probe_decodability": probe_dec,
        "probe_causal_gap": causal_gap,
        "min_causal_rank": int(min_causal_rank),
        "subspace_linearity": float(subspace_linearity),
        "delta_full": float(delta_full),
        "baseline_mean": float(baseline_mean),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H1 Localizability Scorecard (UT pair)")
    parser.add_argument("--tag", required=True, choices=["ut_1step", "ut_bptt"],
                        help="Which UT checkpoint to evaluate")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override checkpoint directory")
    parser.add_argument("--task", choices=["sudoku"], default="sudoku",
                        help="Only Sudoku supported for UT (maze not available)")
    parser.add_argument("--num_puzzles", type=int, default=300)
    parser.add_argument("--probe_steps", default=None,
                        help="Comma-separated steps (auto-set by tag if omitted)")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--ranks", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Auto-set from config if omitted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ckpt_dir = args.checkpoint_dir or UT_CHECKPOINT_DIRS.get(args.tag)
    if ckpt_dir is None:
        print(f"Unknown tag: {args.tag}")
        sys.exit(1)

    ckpt_file = find_ut_checkpoint(ckpt_dir)
    if ckpt_file is None:
        print(f"No checkpoint found in {ckpt_dir} — skipping {args.tag}.")
        print("(For ut_bptt: checkpoint will be available once training completes.)")
        sys.exit(0)

    logger.info(f"[H1-UT] tag={args.tag}  ckpt={ckpt_file}  device={device}")

    output_dir = args.output_dir or os.path.join(
        REPO_ROOT, "results", "localizability", args.tag
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, loader, config = load_ut_model(ckpt_dir, ckpt_file, device)
    model.eval()

    max_steps = args.max_steps or int(model.config.halt_max_steps)

    # Set probe steps
    if args.probe_steps:
        probe_steps = [int(s) for s in args.probe_steps.split(",")]
    elif args.tag == "ut_bptt":
        probe_steps = PROBE_STEPS_BPTT
    else:
        probe_steps = [s for s in PROBE_STEPS_1STEP if s < max_steps]

    seeds = [int(s) for s in args.seeds.split(",")]
    ranks = [int(r) for r in args.ranks.split(",")]

    # Create harness and collect puzzles
    harness = UTActivationHarness(model, device)
    batches = harness.collect_puzzles(loader, args.num_puzzles)
    logger.info(f"[H1-UT] Collected {len(batches)} puzzles  max_steps={max_steps}")

    t0 = time.time()
    scorecard = run_ut_scorecard(
        harness=harness,
        batches=batches,
        task=args.task,
        probe_steps=probe_steps,
        seeds=seeds,
        ranks=ranks,
        max_steps=max_steps,
        device=device,
    )
    elapsed = time.time() - t0

    # CAVEAT note (§1.6 of plan)
    caveat = (
        "Track B UT pair differs in both gradient regime AND recurrence layout "
        "(ut_1step: 16 outer steps × 4 inner iters; ut_bptt: 1 outer step × 16 iters). "
        "This is a strong but not airtight control. The airtight comparison is "
        "HRM-1step vs HRM-BPTT (identical arch), available via localizability_scorecard.py "
        "once the HRM-BPTT checkpoint lands."
    )

    meta = {
        "tag": args.tag,
        "checkpoint": ckpt_file,
        "task": args.task,
        "num_puzzles": len(batches),
        "probe_steps": probe_steps,
        "seeds": seeds,
        "ranks": ranks,
        "max_steps": max_steps,
        "one_step_grad": getattr(model.config, "one_step_grad", "unknown"),
        "num_iterations": getattr(model.config, "num_iterations", "unknown"),
        "hidden_size": model.config.hidden_size,
        "elapsed_s": round(elapsed, 1),
        "track_b_caveat": caveat,
    }

    output = {"meta": meta, "scorecard": scorecard}

    out_path = os.path.join(output_dir, "scorecard.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved {out_path}")

    try:
        from scripts.core.provenance import write_meta
        write_meta(output_dir, f"H1_ut_scorecard_{args.tag}", meta, repo_root=REPO_ROOT)
    except Exception as e:
        logger.warning(f"provenance write failed: {e}")

    print("\n" + "=" * 60)
    print(f"H1 UT LOCALIZABILITY SCORECARD — {args.tag}")
    print("=" * 60)
    sc = scorecard
    print(f"  probe_decodability  : {sc['probe_decodability']['mean_val_acc']:.4f}")
    pcg = sc['probe_causal_gap']
    print(f"  probe_causal_gap    : {pcg['probe_causal_gap']:+.4f}  "
          f"(probe={pcg['probe_mean_delta']:+.4f}, random={pcg['random_mean_delta']:+.4f})")
    print(f"  min_causal_rank r*  : {sc['min_causal_rank']}")
    print(f"  subspace_linearity  : {sc['subspace_linearity']:.4f}")
    print(f"  Δfull               : {sc['delta_full']:+.4f}")
    print(f"\nNOTE: {caveat[:120]}...")


if __name__ == "__main__":
    main()
