#!/usr/bin/env python3
"""SAE E10: Causal validation — ablate SAE features vs. probe directions.

The KEY experiment: compares causal effect of SAE feature ablation against
E9's probe direction ablation. If SAE features have significantly larger
causal effects → evidence that the model uses distributed sparse computation,
and SAE is better at finding functional features than linear probes.

Methodology
-----------
For each top-K SAE feature:
    1. Encode z_H through SAE: h = sae.encode(z_H)
    2. Zero out the single feature: h[:, feat_idx] = 0
    3. Reconstruct: z_H_ablated = sae.decode(h)
    4. Replace z_H with z_H_ablated
    5. Record Δaccuracy, violation changes

Controls:
    - Random SAE features (baseline for feature ablation)
    - E9 probe directions projected out (direct comparison)
    - Random directions (baseline for direction ablation)

Output
------
  results/sae_study/causal_ablation/
    per_puzzle.jsonl
    aggregate.json

Usage
-----
    python scripts/sae/sae_causal_ablation.py --sae_path results/sae_study/sae_d2048_l10.01.pt
    python scripts/sae/sae_causal_ablation.py --sae_path ... --n_puzzles 200 --top_k 50
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml
from pretrain import PretrainConfig, create_dataloader
from utils.functions import load_model_class
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1InnerCarry,
)
from scripts.core.activation_ablation import (
    ActivationAblator, ActivationCache, _patch_attention_for_cpu,
    _make_inner_carry, _make_carry, ACTCarry,
)
from models.sae import SparseAutoencoder, TopKSparseAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

SUDOKU_SIZE = 9
SUDOKU_CELLS = 81
DIGIT_OFFSET = 1


# ═══════════════════════════════════════════════════════════════════════════
# Constraint violation counting
# ═══════════════════════════════════════════════════════════════════════════

def count_violations(preds_tok: torch.Tensor) -> Dict[str, int]:
    """Count row/col/box violations from predicted tokens."""
    if preds_tok.ndim == 1:
        preds_tok = preds_tok.unsqueeze(0)
    B = preds_tok.shape[0]
    digits = (preds_tok.long() - DIGIT_OFFSET).clamp(0, SUDOKU_SIZE)
    grid = digits.view(B, SUDOKU_SIZE, SUDOKU_SIZE)

    def _unit_violations(unit):
        u = unit[0]
        nonzero = u[u > 0]
        return (len(nonzero) - len(nonzero.unique())) if len(nonzero) > 0 else 0

    row_viol = sum(_unit_violations(grid[:, r, :]) for r in range(SUDOKU_SIZE))
    col_viol = sum(_unit_violations(grid[:, :, c]) for c in range(SUDOKU_SIZE))
    box_viol = 0
    for br in range(3):
        for bc in range(3):
            box = grid[:, br*3:(br+1)*3, bc*3:(bc+1)*3].reshape(B, 9)
            box_viol += _unit_violations(box)

    return {
        "violated_rows": row_viol,
        "violated_cols": col_viol,
        "violated_boxes": box_viol,
        "violated_total": row_viol + col_viol + box_viol,
    }


def cell_accuracy(preds_tok, targets_tok):
    return float((preds_tok.view(-1) == targets_tok.view(-1)).float().mean().item())


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_data(device: torch.device):
    ckpt_dir = os.path.join(REPO_ROOT, "checkpoints", "sapientinc-sudoku-extreme")
    config_path = os.path.join(ckpt_dir, "config.yaml")
    if os.path.exists(os.path.join(ckpt_dir, "all_config.yaml")):
        config_path = os.path.join(ckpt_dir, "all_config.yaml")
    with open(config_path) as f:
        config = PretrainConfig(**yaml.safe_load(f))

    test_loader, test_meta = create_dataloader(
        config, "test", test_set_mode=True,
        epochs_per_iter=1, global_batch_size=1, rank=0, world_size=1,
    )

    model_cfg = dict(
        **config.arch.__pydantic_extra__, batch_size=1,
        vocab_size=test_meta.vocab_size, seq_len=test_meta.seq_len,
        num_puzzle_identifiers=test_meta.num_puzzle_identifiers, causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model_raw = model_cls(model_cfg)
        model_full = loss_cls(model_raw, **config.arch.loss.__pydantic_extra__)

    ckpt = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"),
                      map_location=device, weights_only=False)
    mk = set(model_full.state_dict().keys())
    ck = set(ckpt.keys())
    if any(k.startswith("_orig_mod.") for k in mk) and not any(
        k.startswith("_orig_mod.") for k in ck
    ):
        ckpt = {f"_orig_mod.{k}": v for k, v in ckpt.items()}
    elif any(k.startswith("_orig_mod.") for k in ck) and not any(
        k.startswith("_orig_mod.") for k in mk
    ):
        ckpt = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}
    model_full.load_state_dict(ckpt, assign=True)
    model_full.to(device).eval()

    if device.type == "cpu":
        _patch_attention_for_cpu(model_full)

    m: Any = model_full
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if not isinstance(m, HierarchicalReasoningModel_ACTV1) and hasattr(m, "model"):
        m = m.model

    return m, test_loader, test_meta


# ═══════════════════════════════════════════════════════════════════════════
# SAE Feature Ablator
# ═══════════════════════════════════════════════════════════════════════════

class SAEFeatureAblator(ActivationAblator):
    """Ablate individual SAE features during HRM forward pass."""

    def __init__(self, model, sae: SparseAutoencoder,
                 device: torch.device = torch.device("cpu")):
        super().__init__(model, device=device)
        self.sae = sae.to(device).eval()

    def run_with_sae_feature_ablation(
        self,
        batch: Dict[str, torch.Tensor],
        feature_indices: List[int],
        max_steps: int = 16,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, ActivationCache]]:
        """Forward pass with specific SAE features zeroed out at every step.

        At each step:
            h = sae.encode(z_H)
            h[:, :, feature_indices] = 0
            z_H_ablated = sae.decode(h)
        """
        cache: Dict[int, ActivationCache] = {}
        carry = self._init_carry(batch)

        original_max = self.model.config.halt_max_steps
        self.model.config.halt_max_steps = max_steps

        all_outputs = []
        step = 0

        try:
            with torch.no_grad():
                while (step == 0) or (not self._bool_all(carry.halted)):
                    inner_in, steps_in, current_data = self._prepare_step_inputs(carry, batch)

                    # SAE feature ablation on z_H
                    z_H = inner_in.z_H.float()  # [B, T, D]
                    B, T, D = z_H.shape

                    # Encode through SAE
                    z_flat = z_H.reshape(B * T, D)
                    h = self.sae.encode(z_flat)  # [B*T, dict_size]

                    # Zero out specified features
                    for fi in feature_indices:
                        h[:, fi] = 0

                    # Decode back
                    z_recon = self.sae.decode(h)  # [B*T, D]
                    z_H_ablated = z_recon.reshape(B, T, D).to(inner_in.z_H.dtype)

                    inner_in = _make_inner_carry(
                        self.model, z_H=z_H_ablated, z_L=inner_in.z_L,
                    )

                    new_carry, outputs, inner_used = self._forward_one_step(
                        carry, batch, patched_inner_carry=inner_in,
                    )

                    cache[step] = ActivationCache(
                        z_H=inner_used.z_H.detach().clone(),
                        z_L=inner_used.z_L.detach().clone(),
                        step=step,
                        z_H_out=new_carry.inner_carry.z_H.detach().clone(),
                        z_L_out=new_carry.inner_carry.z_L.detach().clone(),
                        logits=outputs["logits"].detach().clone(),
                        preds=outputs["intermediate_preds_step"].detach().clone(),
                        q_halt_logits=outputs["q_halt_logits"].detach().clone(),
                        q_continue_logits=outputs["q_continue_logits"].detach().clone(),
                    )

                    all_outputs.append(outputs)
                    carry = new_carry
                    step += 1
                    if step >= max_steps:
                        break
        finally:
            self.model.config.halt_max_steps = original_max

        return all_outputs[-1] if all_outputs else {}, cache

    def run_with_direction_ablation(
        self,
        batch: Dict[str, torch.Tensor],
        direction: torch.Tensor,  # [D]
        max_steps: int = 16,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, ActivationCache]]:
        """Forward pass with a direction projected out of z_H at every step."""
        cache: Dict[int, ActivationCache] = {}
        carry = self._init_carry(batch)

        original_max = self.model.config.halt_max_steps
        self.model.config.halt_max_steps = max_steps

        d_hat = direction.float().to(self.device)
        d_hat = d_hat / d_hat.norm().clamp(min=1e-8)

        all_outputs = []
        step = 0

        try:
            with torch.no_grad():
                while (step == 0) or (not self._bool_all(carry.halted)):
                    inner_in, steps_in, current_data = self._prepare_step_inputs(carry, batch)

                    z_H = inner_in.z_H.float()  # [B, T, D]
                    # Project out direction: z' = z - (z · d) d
                    proj = (z_H @ d_hat).unsqueeze(-1) * d_hat  # [B, T, D]
                    z_H_ablated = (z_H - proj).to(inner_in.z_H.dtype)

                    inner_in = _make_inner_carry(
                        self.model, z_H=z_H_ablated, z_L=inner_in.z_L,
                    )

                    new_carry, outputs, inner_used = self._forward_one_step(
                        carry, batch, patched_inner_carry=inner_in,
                    )

                    cache[step] = ActivationCache(
                        z_H=inner_used.z_H.detach().clone(),
                        z_L=inner_used.z_L.detach().clone(),
                        step=step,
                        z_H_out=new_carry.inner_carry.z_H.detach().clone(),
                        z_L_out=new_carry.inner_carry.z_L.detach().clone(),
                        logits=outputs["logits"].detach().clone(),
                        preds=outputs["intermediate_preds_step"].detach().clone(),
                        q_halt_logits=outputs["q_halt_logits"].detach().clone(),
                        q_continue_logits=outputs["q_continue_logits"].detach().clone(),
                    )

                    all_outputs.append(outputs)
                    carry = new_carry
                    step += 1
                    if step >= max_steps:
                        break
        finally:
            self.model.config.halt_max_steps = original_max

        return all_outputs[-1] if all_outputs else {}, cache


# ═══════════════════════════════════════════════════════════════════════════
# Select top SAE features to ablate
# ═══════════════════════════════════════════════════════════════════════════

def select_top_features(
    sae: SparseAutoencoder,
    activations_path: str,
    top_k: int = 50,
    device: torch.device = torch.device("cpu"),
) -> List[int]:
    """Select top-K SAE features by activation frequency."""
    data = torch.load(activations_path, map_location="cpu", weights_only=False)
    z_H = data['z_H']  # [N, steps, 81, D]
    flat = z_H.reshape(-1, z_H.shape[-1]).to(device)

    batch_size = 8192
    fire_count = torch.zeros(sae.dict_size)
    for start in range(0, flat.shape[0], batch_size):
        batch = flat[start:start + batch_size]
        with torch.no_grad():
            h = sae.encode(batch).cpu()
        fire_count += (h > 0).float().sum(dim=0)

    top_features = fire_count.argsort(descending=True)[:top_k].tolist()
    logger.info(f"Top {top_k} features by activation frequency: {top_features[:10]}...")
    return top_features


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
    model,
    sae: SparseAutoencoder,
    test_loader,
    device: torch.device,
    top_features: List[int],
    n_random_features: int = 10,
    n_puzzles: int = 200,
    max_steps: int = 16,
    probe_weights_path: Optional[str] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run the full causal ablation experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ablator = SAEFeatureAblator(model, sae, device=device)

    # Collect batches
    batches = []
    for i, data in enumerate(test_loader):
        if i >= n_puzzles:
            break
        if isinstance(data, (tuple, list)):
            batch = data[1] if len(data) >= 2 and isinstance(data[1], dict) else data[0]
        else:
            batch = data
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batches.append(batch)
    logger.info(f"Loaded {len(batches)} puzzles")

    # Load E8 probe directions if available — select best step per target
    # (matches E9's select_best_directions pattern)
    probe_directions = {}
    if probe_weights_path and os.path.exists(probe_weights_path):
        probe_data = torch.load(probe_weights_path, map_location="cpu", weights_only=False)
        # Group by target, pick highest val_score across steps
        best_per_target = {}  # target → (val_score, W_vec, step)
        for key, info in probe_data.items():
            if 'W' not in info:
                continue
            target = info.get('target', key)
            z_level = info.get('z_level', 'H')
            if z_level != 'H':
                continue
            score = info.get('val_score', 0)
            W = info['W']
            if W.shape[0] == 1:
                w_vec = W[0].float()
            else:
                # Multiclass: use first principal component
                U, S, _ = torch.svd(W.float())
                w_vec = (W.float().T @ U[:, 0])
                w_vec = w_vec / w_vec.norm().clamp(min=1e-8)
            if target not in best_per_target or score > best_per_target[target][0]:
                best_per_target[target] = (score, w_vec, info.get('step', 0))
        for target, (score, w_vec, step) in best_per_target.items():
            probe_directions[target] = w_vec
            logger.info(f"  Probe direction '{target}': step={step}, val={score:.4f}")
        logger.info(f"Loaded {len(probe_directions)} probe directions from E8")

    # Random features for control
    all_features = list(range(sae.dict_size))
    random_features = np.random.choice(
        [f for f in all_features if f not in top_features],
        size=min(n_random_features, len(all_features) - len(top_features)),
        replace=False,
    ).tolist()

    # Random directions for control
    random_directions = {}
    for i in range(min(3, len(probe_directions))):
        rd = torch.randn(sae.input_dim)
        rd = rd / rd.norm()
        random_directions[f"random_dir_{i}"] = rd

    per_puzzle_results = []

    for pi, batch in enumerate(batches):
        targets_tok = batch["labels"][:, -SUDOKU_CELLS:].cpu()
        inputs_tok = batch["inputs"][:, -SUDOKU_CELLS:].cpu()

        # 1. Baseline (no ablation)
        baseline_cache: Dict[int, ActivationCache] = {}
        ablator.run_and_cache_activations(batch, baseline_cache, max_steps=max_steps)
        last_step = max(baseline_cache.keys())
        baseline_preds = baseline_cache[last_step].preds[:, -SUDOKU_CELLS:].cpu()
        baseline_acc = cell_accuracy(baseline_preds, targets_tok)
        baseline_viols = count_violations(baseline_preds)

        puzzle_result = {
            'puzzle_idx': pi,
            'baseline_acc': baseline_acc,
            'baseline_violations': baseline_viols,
            'sae_feature_ablations': {},
            'random_feature_ablations': {},
            'probe_direction_ablations': {},
            'random_direction_ablations': {},
        }

        # 2. Ablate top SAE features individually
        for feat_idx in top_features:
            _, abl_cache = ablator.run_with_sae_feature_ablation(
                batch, [feat_idx], max_steps=max_steps,
            )
            last_step_abl = max(abl_cache.keys())
            abl_preds = abl_cache[last_step_abl].preds[:, -SUDOKU_CELLS:].cpu()
            abl_acc = cell_accuracy(abl_preds, targets_tok)
            abl_viols = count_violations(abl_preds)

            puzzle_result['sae_feature_ablations'][str(feat_idx)] = {
                'acc': abl_acc,
                'delta_acc': abl_acc - baseline_acc,
                'violations': abl_viols,
                'delta_violations': {
                    k: abl_viols[k] - baseline_viols[k]
                    for k in baseline_viols
                },
            }

        # 3. Ablate random SAE features (control)
        for feat_idx in random_features:
            _, abl_cache = ablator.run_with_sae_feature_ablation(
                batch, [feat_idx], max_steps=max_steps,
            )
            last_step_abl = max(abl_cache.keys())
            abl_preds = abl_cache[last_step_abl].preds[:, -SUDOKU_CELLS:].cpu()
            abl_acc = cell_accuracy(abl_preds, targets_tok)
            abl_viols = count_violations(abl_preds)

            puzzle_result['random_feature_ablations'][str(feat_idx)] = {
                'acc': abl_acc,
                'delta_acc': abl_acc - baseline_acc,
                'violations': abl_viols,
                'delta_violations': {
                    k: abl_viols[k] - baseline_viols[k]
                    for k in baseline_viols
                },
            }

        # 4. Ablate E9 probe directions
        for target_name, direction in probe_directions.items():
            _, abl_cache = ablator.run_with_direction_ablation(
                batch, direction, max_steps=max_steps,
            )
            last_step_abl = max(abl_cache.keys())
            abl_preds = abl_cache[last_step_abl].preds[:, -SUDOKU_CELLS:].cpu()
            abl_acc = cell_accuracy(abl_preds, targets_tok)
            abl_viols = count_violations(abl_preds)

            puzzle_result['probe_direction_ablations'][target_name] = {
                'acc': abl_acc,
                'delta_acc': abl_acc - baseline_acc,
                'violations': abl_viols,
                'delta_violations': {
                    k: abl_viols[k] - baseline_viols[k]
                    for k in baseline_viols
                },
            }

        # 5. Ablate random directions (control)
        for dir_name, direction in random_directions.items():
            _, abl_cache = ablator.run_with_direction_ablation(
                batch, direction, max_steps=max_steps,
            )
            last_step_abl = max(abl_cache.keys())
            abl_preds = abl_cache[last_step_abl].preds[:, -SUDOKU_CELLS:].cpu()
            abl_acc = cell_accuracy(abl_preds, targets_tok)
            abl_viols = count_violations(abl_preds)

            puzzle_result['random_direction_ablations'][dir_name] = {
                'acc': abl_acc,
                'delta_acc': abl_acc - baseline_acc,
                'violations': abl_viols,
                'delta_violations': {
                    k: abl_viols[k] - baseline_viols[k]
                    for k in baseline_viols
                },
            }

        per_puzzle_results.append(puzzle_result)

        if (pi + 1) % 10 == 0:
            # Quick aggregate
            sae_deltas = []
            for pr in per_puzzle_results:
                for v in pr['sae_feature_ablations'].values():
                    sae_deltas.append(v['delta_acc'])
            mean_sae = np.mean(sae_deltas) if sae_deltas else 0

            rand_deltas = []
            for pr in per_puzzle_results:
                for v in pr['random_feature_ablations'].values():
                    rand_deltas.append(v['delta_acc'])
            mean_rand = np.mean(rand_deltas) if rand_deltas else 0

            logger.info(f"  {pi+1}/{len(batches)} | "
                       f"SAE Δacc={mean_sae:.4f} | Random Δacc={mean_rand:.4f}")

    return per_puzzle_results


def compute_aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics with t-tests."""
    from scipy import stats as scipy_stats

    # Collect all delta_acc values per condition
    sae_deltas = []
    random_feature_deltas = []
    probe_deltas = []
    random_dir_deltas = []

    # Also per-feature deltas for the bar chart
    per_feature_deltas = {}

    for pr in results:
        for feat_str, v in pr['sae_feature_ablations'].items():
            sae_deltas.append(v['delta_acc'])
            per_feature_deltas.setdefault(f"sae_{feat_str}", []).append(v['delta_acc'])
        for feat_str, v in pr['random_feature_ablations'].items():
            random_feature_deltas.append(v['delta_acc'])
        for target, v in pr['probe_direction_ablations'].items():
            probe_deltas.append(v['delta_acc'])
            per_feature_deltas.setdefault(f"probe_{target}", []).append(v['delta_acc'])
        for dir_name, v in pr['random_direction_ablations'].items():
            random_dir_deltas.append(v['delta_acc'])

    sae_arr = np.array(sae_deltas) if sae_deltas else np.array([0])
    rand_arr = np.array(random_feature_deltas) if random_feature_deltas else np.array([0])
    probe_arr = np.array(probe_deltas) if probe_deltas else np.array([0])
    rand_dir_arr = np.array(random_dir_deltas) if random_dir_deltas else np.array([0])

    aggregate = {
        'n_puzzles': len(results),
        'conditions': {
            'sae_top_features': {
                'mean_delta_acc': float(sae_arr.mean()),
                'std_delta_acc': float(sae_arr.std()),
                'n_samples': len(sae_deltas),
            },
            'random_sae_features': {
                'mean_delta_acc': float(rand_arr.mean()),
                'std_delta_acc': float(rand_arr.std()),
                'n_samples': len(random_feature_deltas),
            },
            'probe_directions': {
                'mean_delta_acc': float(probe_arr.mean()),
                'std_delta_acc': float(probe_arr.std()),
                'n_samples': len(probe_deltas),
            },
            'random_directions': {
                'mean_delta_acc': float(rand_dir_arr.mean()),
                'std_delta_acc': float(rand_dir_arr.std()),
                'n_samples': len(random_dir_deltas),
            },
        },
    }

    # T-tests
    tests = {}
    if len(sae_deltas) > 1 and len(rand_arr) > 1:
        t_stat, p_val = scipy_stats.ttest_ind(sae_arr, rand_arr)
        tests['sae_vs_random_features'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'significant_0.05': bool(p_val < 0.05),
        }
    if len(sae_deltas) > 1 and len(probe_deltas) > 1:
        t_stat, p_val = scipy_stats.ttest_ind(sae_arr, probe_arr)
        tests['sae_vs_probe_directions'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'significant_0.05': bool(p_val < 0.05),
        }
    if len(probe_deltas) > 1 and len(random_dir_deltas) > 1:
        t_stat, p_val = scipy_stats.ttest_ind(probe_arr, rand_dir_arr)
        tests['probe_vs_random_directions'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'significant_0.05': bool(p_val < 0.05),
        }
    aggregate['statistical_tests'] = tests

    # Per-feature/direction mean delta (for bar chart)
    per_item_means = {}
    for key, deltas in per_feature_deltas.items():
        per_item_means[key] = {
            'mean_delta_acc': float(np.mean(deltas)),
            'std_delta_acc': float(np.std(deltas)),
            'n': len(deltas),
        }
    aggregate['per_item_means'] = per_item_means

    return aggregate


def main():
    parser = argparse.ArgumentParser(description="SAE E10: Causal validation")
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--activations_path", type=str,
                        default="results/sae_study/activations_zH.pt")
    parser.add_argument("--probe_weights_path", type=str,
                        default="results/probes/e8_constraint_probes/probe_weights.pt")
    parser.add_argument("--n_puzzles", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--n_random_features", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/sae_study/causal_ablation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load SAE
    logger.info(f"Loading SAE from {args.sae_path}")
    ckpt = torch.load(args.sae_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    if cfg.get('activation', 'relu') == 'topk':
        sae = TopKSparseAutoencoder(
            input_dim=cfg['input_dim'],
            dict_size=cfg['dict_size'],
            k=cfg['k'],
        )
    else:
        sae = SparseAutoencoder(
            input_dim=cfg['input_dim'],
            dict_size=cfg['dict_size'],
            l1_coeff=cfg['l1_coeff'],
        )
    sae.load_state_dict(ckpt['model_state_dict'])
    sae.to(device).eval()
    logger.info(f"SAE: dict_size={cfg['dict_size']}, activation={cfg.get('activation', 'relu')}")

    # Select top features
    top_features = select_top_features(
        sae, args.activations_path, top_k=args.top_k, device=device,
    )

    # Load model
    model, test_loader, test_meta = load_model_and_data(device)
    logger.info("Model loaded.")

    # Run experiment
    t0 = time.time()
    results = run_experiment(
        model, sae, test_loader, device,
        top_features=top_features,
        n_random_features=args.n_random_features,
        n_puzzles=args.n_puzzles,
        max_steps=args.max_steps,
        probe_weights_path=args.probe_weights_path,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    logger.info(f"Experiment took {elapsed:.1f}s")

    # Save per-puzzle results
    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "per_puzzle.jsonl")
    with open(jsonl_path, 'w') as f:
        for pr in results:
            f.write(json.dumps(pr) + "\n")
    logger.info(f"Saved per-puzzle results to {jsonl_path}")

    # Compute and save aggregate
    aggregate = compute_aggregate(results)
    agg_path = os.path.join(args.output_dir, "aggregate.json")
    with open(agg_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    logger.info(f"Saved aggregate results to {agg_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("CAUSAL ABLATION SUMMARY")
    logger.info("="*60)
    for cond, stats in aggregate['conditions'].items():
        logger.info(f"  {cond:30s}: Δacc = {stats['mean_delta_acc']:+.4f} ± {stats['std_delta_acc']:.4f}")
    if 'statistical_tests' in aggregate:
        for test_name, test_info in aggregate['statistical_tests'].items():
            sig = "***" if test_info.get('significant_0.05') else "n.s."
            logger.info(f"  {test_name}: t={test_info['t_statistic']:.3f}, "
                       f"p={test_info['p_value']:.4f} {sig}")


if __name__ == "__main__":
    main()
