"""scripts/batch_ablation_1k.py

Full test-set ablation experiment runner.

Loads the model ONCE and iterates over ALL puzzles in the test set, running
baseline + ablated forward passes for each.  For every puzzle it runs ablation
at EACH specified single step (independent runs) AND an all-steps ablation run.

This produces comprehensive data suitable for plotting:
  - per-puzzle accuracy curves (baseline vs ablated at each step)
  - distribution of accuracy deltas across the test set
  - activation norm statistics before/after ablation
  - per-step causal impact (how much does ablating step N hurt final accuracy?)
  - cell-level transitions (fixed / broken / unchanged)
  - logit entropy changes (confidence shift from ablation)
  - step-to-step accuracy trajectories (how accuracy evolves across reasoning steps)

Output files:
  - results_per_puzzle.jsonl    (one JSON object per line — puzzle-level detail)
  - aggregate_stats.json        (summary statistics for plotting)
  - step_accuracy_matrix.json   (NxS matrices for baseline & ablated step accuracy)

Usage:
    python scripts/batch_ablation_1k.py \\
        --checkpoint Checkpoint_HRM_Sudoku/.../checkpoint.pt \\
        --ablate_level H \\
        --ablate_steps 4,6,8,10 \\
        --max_steps 16 \\
        --output_dir results/batch_ablation_zH \\
        --device cpu
"""

import os
import sys
import json
import time
import argparse
import yaml
import math
from typing import Any, Dict, Optional, List, Tuple, cast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from pretrain import PretrainConfig, create_dataloader
from utils.functions import load_model_class
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.hrm_v2.hrm_v2 import HierarchicalReasoningModel_V2
from scripts.core.activation_ablation import (
    ActivationAblator,
    ACTModel,
    _patch_attention_for_cpu,
)
from scripts.core.activation_patching import (
    ActivationCache,
    compute_metrics,
    compute_diff_metrics,
    _normalize_patch_level,
    _parse_int_list_arg,
)


# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def _logit_entropy(logits: torch.Tensor) -> float:
    """Mean per-position entropy of the softmax distribution (nats)."""
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    ent = -(probs * log_probs).sum(-1)  # (B, seq_len)
    return float(ent.mean().item())


def _logit_confidence(logits: torch.Tensor) -> float:
    """Mean max-probability across positions (higher = more confident)."""
    probs = F.softmax(logits.float(), dim=-1)
    return float(probs.max(-1).values.mean().item())


def _activation_norms(cache: Dict[int, ActivationCache]) -> Dict[int, Dict[str, float]]:
    """Per-step L2 norms of z_H and z_L (input to inner model)."""
    out: Dict[int, Dict[str, float]] = {}
    for s, ac in cache.items():
        out[s] = {
            "z_H_norm": float(ac.z_H.norm().item()),
            "z_L_norm": float(ac.z_L.norm().item()),
            "z_H_out_norm": float(ac.z_H_out.norm().item()),
            "z_L_out_norm": float(ac.z_L_out.norm().item()),
        }
    return out


def _cell_transitions(
    baseline_preds: torch.Tensor,
    ablated_preds: torch.Tensor,
    labels: torch.Tensor,
    ignore_label_id: int = -100,
) -> Dict[str, int]:
    """Categorize every valid cell into one of four transition categories."""
    valid = labels != ignore_label_id
    base_ok = (baseline_preds == labels) & valid
    abl_ok = (ablated_preds == labels) & valid
    return {
        "stayed_correct": int((base_ok & abl_ok).sum().item()),
        "stayed_wrong": int((~base_ok & ~abl_ok & valid).sum().item()),
        "fixed": int((~base_ok & abl_ok).sum().item()),          # wrong→right
        "broken": int((base_ok & ~abl_ok).sum().item()),         # right→wrong
        "total_changed": int(((baseline_preds != ablated_preds) & valid).sum().item()),
    }


def _step_accuracy_delta(
    baseline_cache: Dict[int, ActivationCache],
    ablated_cache: Dict[int, ActivationCache],
    labels: torch.Tensor,
) -> Dict[int, Dict[str, float]]:
    """Per-step accuracy for both baseline and ablated, with delta."""
    common = sorted(set(baseline_cache.keys()) & set(ablated_cache.keys()))
    out: Dict[int, Dict[str, float]] = {}
    for s in common:
        bm = compute_metrics(baseline_cache[s].preds, labels)
        am = compute_metrics(ablated_cache[s].preds, labels)
        out[s] = {
            "baseline_acc": bm["accuracy"],
            "ablated_acc": am["accuracy"],
            "delta_acc": am["accuracy"] - bm["accuracy"],
            "baseline_correct": bm["correct"],
            "ablated_correct": am["correct"],
        }
    return out


def _inter_step_accuracy_change(
    cache: Dict[int, ActivationCache],
    labels: torch.Tensor,
) -> List[float]:
    """Accuracy change between consecutive steps: acc[s+1] - acc[s]."""
    steps = sorted(cache.keys())
    accs = [compute_metrics(cache[s].preds, labels)["accuracy"] for s in steps]
    return [round(accs[i+1] - accs[i], 6) for i in range(len(accs) - 1)]


# ─────────────────────────────────────────────────────────────
# Model loading (shared with run_ablation_experiments.py)
# ─────────────────────────────────────────────────────────────

def _load_model_and_dataloader(checkpoint_path: str, device: torch.device):
    config_path = os.path.join(os.path.dirname(checkpoint_path), "all_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

    test_loader, test_metadata = create_dataloader(
        config, "test", test_set_mode=True,
        epochs_per_iter=1, global_batch_size=1,
        rank=0, world_size=1,
    )

    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=1,
        vocab_size=test_metadata.vocab_size,
        seq_len=test_metadata.seq_len,
        num_puzzle_identifiers=test_metadata.num_puzzle_identifiers,
        causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model_raw: nn.Module = model_cls(model_cfg)
        model_full = loss_head_cls(model_raw, **config.arch.loss.__pydantic_extra__)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_keys = set(model_full.state_dict().keys())
    ckpt_keys = set(ckpt.keys())
    m_pfx = any(k.startswith("_orig_mod.") for k in model_keys)
    c_pfx = any(k.startswith("_orig_mod.") for k in ckpt_keys)
    if m_pfx and not c_pfx:
        ckpt = {f"_orig_mod.{k}": v for k, v in ckpt.items()}
    elif c_pfx and not m_pfx:
        ckpt = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}

    model_full.load_state_dict(ckpt, assign=True)
    model_full.to(device)
    model_full.eval()

    if device.type == "cpu":
        _patch_attention_for_cpu(model_full)

    # Unwrap to ACT model
    model_obj: Any = model_full
    if hasattr(model_obj, "_orig_mod"):
        model_obj = model_obj._orig_mod
    if not isinstance(model_obj, (HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_V2)) \
            and hasattr(model_obj, "model"):
        model_obj = getattr(model_obj, "model")
    if not isinstance(model_obj, (HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_V2)):
        raise TypeError(f"Expected ACTV1 or V2, got {type(model_obj)}")

    ablator = ActivationAblator(cast(ACTModel, model_obj), device=device)
    return ablator, test_loader


def _extract_batch(item):
    if isinstance(item, (tuple, list)):
        if len(item) == 3 and isinstance(item[1], dict):
            return item[1]
        if len(item) == 2 and isinstance(item[1], dict):
            return item[1]
        if len(item) == 1 and isinstance(item[0], dict):
            return item[0]
    if isinstance(item, dict):
        return item
    raise TypeError(f"Unsupported dataloader item: {type(item)}")


# ─────────────────────────────────────────────────────────────
# Core experiment loop
# ─────────────────────────────────────────────────────────────

def process_one_puzzle(
    ablator: ActivationAblator,
    batch: Dict[str, torch.Tensor],
    puzzle_idx: int,
    ablate_level: str,
    single_ablate_steps: Optional[List[int]],  # individual steps for single-step ablation
    max_steps: int,
) -> Dict[str, Any]:
    """Run baseline + all ablation variants for one puzzle, return rich metrics.

    Ablation variants run:
      1. "all_steps" — ablate at every step
      2. One run per entry in single_ablate_steps — ablate only at that step
    """
    labels = batch["labels"]

    # ── 1. Baseline (shared across all ablation variants) ─────
    base_cache: Dict[int, ActivationCache] = {}
    base_out = ablator.run_and_cache_activations(batch, base_cache, max_steps=max_steps)
    base_preds = base_out["logits"].argmax(-1)
    base_metrics = compute_metrics(base_preds, labels)
    base_entropy = _logit_entropy(base_out["logits"])
    base_confidence = _logit_confidence(base_out["logits"])
    base_norms = _activation_norms(base_cache)
    base_inter_step_deltas = _inter_step_accuracy_change(base_cache, labels)

    # Per-step baseline accuracy (for the accuracy trajectory)
    steps_sorted = sorted(base_cache.keys())
    baseline_step_accs = {
        s: compute_metrics(base_cache[s].preds, labels)["accuracy"] for s in steps_sorted
    }

    # ── 2. All-steps ablation ─────────────────────────────────
    abl_all_out, abl_all_cache, abl_all_info = ablator.run_with_ablation(
        batch, ablate_level=ablate_level, ablate_steps=None, max_steps=max_steps,
    )
    abl_all_preds = abl_all_out["logits"].argmax(-1)
    abl_all_metrics = compute_metrics(abl_all_preds, labels)
    abl_all_cell = _cell_transitions(base_preds, abl_all_preds, labels)
    abl_all_step_detail = _step_accuracy_delta(base_cache, abl_all_cache, labels)
    abl_all_entropy = _logit_entropy(abl_all_out["logits"])
    abl_all_confidence = _logit_confidence(abl_all_out["logits"])
    abl_all_norms = _activation_norms(abl_all_cache)
    abl_all_inter_step_deltas = _inter_step_accuracy_change(abl_all_cache, labels)

    all_steps_result = {
        "final_accuracy": abl_all_metrics["accuracy"],
        "accuracy_delta": abl_all_metrics["accuracy"] - base_metrics["accuracy"],
        "correct": abl_all_metrics["correct"],
        "cell_transitions": abl_all_cell,
        "entropy": abl_all_entropy,
        "confidence": abl_all_confidence,
        "entropy_delta": abl_all_entropy - base_entropy,
        "confidence_delta": abl_all_confidence - base_confidence,
        "activation_norms": {str(k): v for k, v in abl_all_norms.items()},
        "step_accuracies": {
            str(s): d for s, d in abl_all_step_detail.items()
        },
        "inter_step_accuracy_deltas": abl_all_inter_step_deltas,
    }

    # ── 3. Single-step ablations ──────────────────────────────
    single_step_results: Dict[str, Any] = {}
    if single_ablate_steps:
        for step in single_ablate_steps:
            abl_out, abl_cache, abl_info = ablator.run_with_ablation(
                batch, ablate_level=ablate_level,
                ablate_steps=[step], max_steps=max_steps,
            )
            abl_preds = abl_out["logits"].argmax(-1)
            abl_m = compute_metrics(abl_preds, labels)
            cell = _cell_transitions(base_preds, abl_preds, labels)
            step_detail = _step_accuracy_delta(base_cache, abl_cache, labels)
            abl_ent = _logit_entropy(abl_out["logits"])
            abl_conf = _logit_confidence(abl_out["logits"])
            abl_norms_s = _activation_norms(abl_cache)
            inter_step_deltas = _inter_step_accuracy_change(abl_cache, labels)

            # Get pre-ablation norm from the info dict for the ablated step
            pre_norm_info = {}
            if step in abl_info:
                pre_norm_info = abl_info[step]

            single_step_results[str(step)] = {
                "final_accuracy": abl_m["accuracy"],
                "accuracy_delta": abl_m["accuracy"] - base_metrics["accuracy"],
                "correct": abl_m["correct"],
                "cell_transitions": cell,
                "entropy": abl_ent,
                "confidence": abl_conf,
                "entropy_delta": abl_ent - base_entropy,
                "confidence_delta": abl_conf - base_confidence,
                "pre_ablation_norm": pre_norm_info,
                "activation_norms": {str(k): v for k, v in abl_norms_s.items()},
                "step_accuracies": {
                    str(s): d for s, d in step_detail.items()
                },
                "inter_step_accuracy_deltas": inter_step_deltas,
            }

    # ── Assemble ──────────────────────────────────────────────
    return {
        "puzzle_idx": puzzle_idx,
        "total_positions": base_metrics["total_positions"],
        "baseline": {
            "final_accuracy": base_metrics["accuracy"],
            "correct": base_metrics["correct"],
            "entropy": base_entropy,
            "confidence": base_confidence,
            "step_accuracies": {str(s): a for s, a in baseline_step_accs.items()},
            "activation_norms": {str(k): v for k, v in base_norms.items()},
            "inter_step_accuracy_deltas": base_inter_step_deltas,
        },
        "ablation_all_steps": all_steps_result,
        "ablation_single_step": single_step_results,
    }


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────

def _stats(vals: List[float]) -> Dict[str, float]:
    a = np.array(vals, dtype=np.float64)
    if len(a) == 0:
        return {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0, "q25": 0, "q75": 0}
    return {
        "mean": round(float(np.mean(a)), 6),
        "std": round(float(np.std(a)), 6),
        "median": round(float(np.median(a)), 6),
        "min": round(float(np.min(a)), 6),
        "max": round(float(np.max(a)), 6),
        "q25": round(float(np.percentile(a, 25)), 6),
        "q75": round(float(np.percentile(a, 75)), 6),
    }


def _pct_bucket(vals: List[float]) -> Dict[str, float]:
    n = len(vals)
    if n == 0:
        return {"pct_decreased": 0, "pct_unchanged": 0, "pct_increased": 0}
    return {
        "pct_decreased": round(sum(1 for v in vals if v < -1e-9) / n * 100, 2),
        "pct_unchanged": round(sum(1 for v in vals if abs(v) < 1e-9) / n * 100, 2),
        "pct_increased": round(sum(1 for v in vals if v > 1e-9) / n * 100, 2),
    }


def compute_aggregates(
    all_results: List[Dict[str, Any]],
    single_ablate_steps: Optional[List[int]],
) -> Dict[str, Any]:
    """Compute aggregate statistics across all puzzles for plotting."""
    n = len(all_results)
    if n == 0:
        return {}

    # ── Baseline aggregates ───────────────────────────────────
    base_accs = [r["baseline"]["final_accuracy"] for r in all_results]
    base_entropies = [r["baseline"]["entropy"] for r in all_results]
    base_confidences = [r["baseline"]["confidence"] for r in all_results]

    agg: Dict[str, Any] = {
        "num_puzzles": n,
        "baseline": {
            "accuracy": _stats(base_accs),
            "entropy": _stats(base_entropies),
            "confidence": _stats(base_confidences),
        },
    }

    # ── All-steps ablation aggregates ─────────────────────────
    all_step_accs = [r["ablation_all_steps"]["final_accuracy"] for r in all_results]
    all_step_deltas = [r["ablation_all_steps"]["accuracy_delta"] for r in all_results]
    all_step_ent = [r["ablation_all_steps"]["entropy"] for r in all_results]
    all_step_conf = [r["ablation_all_steps"]["confidence"] for r in all_results]
    all_step_ent_delta = [r["ablation_all_steps"]["entropy_delta"] for r in all_results]
    all_step_conf_delta = [r["ablation_all_steps"]["confidence_delta"] for r in all_results]
    all_step_fixed = [r["ablation_all_steps"]["cell_transitions"]["fixed"] for r in all_results]
    all_step_broken = [r["ablation_all_steps"]["cell_transitions"]["broken"] for r in all_results]
    all_step_changed = [r["ablation_all_steps"]["cell_transitions"]["total_changed"] for r in all_results]

    agg["ablation_all_steps"] = {
        "accuracy": _stats(all_step_accs),
        "accuracy_delta": _stats(all_step_deltas),
        "accuracy_delta_buckets": _pct_bucket(all_step_deltas),
        "entropy": _stats(all_step_ent),
        "confidence": _stats(all_step_conf),
        "entropy_delta": _stats(all_step_ent_delta),
        "confidence_delta": _stats(all_step_conf_delta),
        "cells_fixed": _stats(all_step_fixed),
        "cells_broken": _stats(all_step_broken),
        "cells_changed": _stats(all_step_changed),
    }

    # Per-step accuracy trajectory for all-steps ablation
    all_step_keys: set = set()
    for r in all_results:
        all_step_keys.update(r["ablation_all_steps"]["step_accuracies"].keys())
    per_step_all: Dict[str, Dict[str, Any]] = {}
    for sk in sorted(all_step_keys, key=lambda x: int(x)):
        b_vals = [r["baseline"]["step_accuracies"].get(sk, 0) for r in all_results
                  if sk in r["baseline"]["step_accuracies"]]
        a_vals = [r["ablation_all_steps"]["step_accuracies"][sk]["ablated_acc"]
                  for r in all_results if sk in r["ablation_all_steps"]["step_accuracies"]]
        d_vals = [r["ablation_all_steps"]["step_accuracies"][sk]["delta_acc"]
                  for r in all_results if sk in r["ablation_all_steps"]["step_accuracies"]]
        per_step_all[sk] = {
            "baseline_acc": _stats(b_vals) if b_vals else {},
            "ablated_acc": _stats(a_vals) if a_vals else {},
            "delta_acc": _stats(d_vals) if d_vals else {},
            "n": len(b_vals),
        }
    agg["ablation_all_steps"]["per_step_trajectory"] = per_step_all

    # ── Single-step ablation aggregates ───────────────────────
    if single_ablate_steps:
        single_agg: Dict[str, Any] = {}
        for step in single_ablate_steps:
            sk = str(step)
            accs = [r["ablation_single_step"][sk]["final_accuracy"]
                    for r in all_results if sk in r.get("ablation_single_step", {})]
            deltas = [r["ablation_single_step"][sk]["accuracy_delta"]
                      for r in all_results if sk in r.get("ablation_single_step", {})]
            ents = [r["ablation_single_step"][sk]["entropy"]
                    for r in all_results if sk in r.get("ablation_single_step", {})]
            confs = [r["ablation_single_step"][sk]["confidence"]
                     for r in all_results if sk in r.get("ablation_single_step", {})]
            ent_d = [r["ablation_single_step"][sk]["entropy_delta"]
                     for r in all_results if sk in r.get("ablation_single_step", {})]
            conf_d = [r["ablation_single_step"][sk]["confidence_delta"]
                      for r in all_results if sk in r.get("ablation_single_step", {})]
            fixed = [r["ablation_single_step"][sk]["cell_transitions"]["fixed"]
                     for r in all_results if sk in r.get("ablation_single_step", {})]
            broken = [r["ablation_single_step"][sk]["cell_transitions"]["broken"]
                      for r in all_results if sk in r.get("ablation_single_step", {})]
            changed = [r["ablation_single_step"][sk]["cell_transitions"]["total_changed"]
                       for r in all_results if sk in r.get("ablation_single_step", {})]

            if accs:
                single_agg[sk] = {
                    "accuracy": _stats(accs),
                    "accuracy_delta": _stats(deltas),
                    "accuracy_delta_buckets": _pct_bucket(deltas),
                    "entropy": _stats(ents),
                    "confidence": _stats(confs),
                    "entropy_delta": _stats(ent_d),
                    "confidence_delta": _stats(conf_d),
                    "cells_fixed": _stats(fixed),
                    "cells_broken": _stats(broken),
                    "cells_changed": _stats(changed),
                    "n": len(accs),
                }

                # Per-step trajectory for this single-step ablation
                traj_keys: set = set()
                for r in all_results:
                    if sk in r.get("ablation_single_step", {}):
                        traj_keys.update(r["ablation_single_step"][sk]["step_accuracies"].keys())
                traj: Dict[str, Any] = {}
                for tk in sorted(traj_keys, key=lambda x: int(x)):
                    b2 = [r["baseline"]["step_accuracies"].get(tk, 0)
                          for r in all_results
                          if sk in r.get("ablation_single_step", {}) and
                          tk in r["baseline"]["step_accuracies"]]
                    a2 = [r["ablation_single_step"][sk]["step_accuracies"][tk]["ablated_acc"]
                          for r in all_results
                          if sk in r.get("ablation_single_step", {}) and
                          tk in r["ablation_single_step"][sk]["step_accuracies"]]
                    d2 = [r["ablation_single_step"][sk]["step_accuracies"][tk]["delta_acc"]
                          for r in all_results
                          if sk in r.get("ablation_single_step", {}) and
                          tk in r["ablation_single_step"][sk]["step_accuracies"]]
                    traj[tk] = {
                        "baseline_acc": _stats(b2) if b2 else {},
                        "ablated_acc": _stats(a2) if a2 else {},
                        "delta_acc": _stats(d2) if d2 else {},
                        "n": len(b2),
                    }
                single_agg[sk]["per_step_trajectory"] = traj

        agg["ablation_single_step"] = single_agg

    # ── Comparison: single-step impact ranking ────────────────
    # Which step, when ablated alone, causes the biggest accuracy drop?
    if single_ablate_steps:
        impact_ranking = []
        for step in single_ablate_steps:
            sk = str(step)
            if sk in agg.get("ablation_single_step", {}):
                impact_ranking.append({
                    "step": step,
                    "mean_accuracy_delta": agg["ablation_single_step"][sk]["accuracy_delta"]["mean"],
                    "mean_cells_broken": agg["ablation_single_step"][sk]["cells_broken"]["mean"],
                })
        impact_ranking.sort(key=lambda x: x["mean_accuracy_delta"])
        agg["single_step_impact_ranking"] = impact_ranking

    return agg


def build_step_accuracy_matrix(
    all_results: List[Dict[str, Any]],
    single_ablate_steps: Optional[List[int]],
) -> Dict[str, Any]:
    """Build NxS matrices (puzzles x steps) for easy plotting.

    Returns dict with:
      - puzzle_indices: [int]
      - steps: [int]
      - baseline_matrix: [[float]]  (N puzzles x S steps)
      - allstep_ablated_matrix: [[float]]
      - single_step_matrices: {step: [[float]]}
    """
    if not all_results:
        return {}

    # Determine step range from first puzzle
    steps = sorted(int(k) for k in all_results[0]["baseline"]["step_accuracies"].keys())
    puzzle_indices = [r["puzzle_idx"] for r in all_results]

    baseline_matrix = []
    allstep_matrix = []
    for r in all_results:
        baseline_matrix.append([
            r["baseline"]["step_accuracies"].get(str(s), 0) for s in steps
        ])
        allstep_matrix.append([
            r["ablation_all_steps"]["step_accuracies"].get(str(s), {}).get("ablated_acc", 0)
            for s in steps
        ])

    out: Dict[str, Any] = {
        "puzzle_indices": puzzle_indices,
        "steps": steps,
        "baseline_matrix": baseline_matrix,
        "allstep_ablated_matrix": allstep_matrix,
    }

    if single_ablate_steps:
        single_matrices: Dict[str, list] = {}
        for ablate_step in single_ablate_steps:
            sk = str(ablate_step)
            mat = []
            for r in all_results:
                if sk in r.get("ablation_single_step", {}):
                    row = [
                        r["ablation_single_step"][sk]["step_accuracies"].get(str(s), {}).get("ablated_acc", 0)
                        for s in steps
                    ]
                else:
                    row = [0.0] * len(steps)
                mat.append(row)
            single_matrices[sk] = mat
        out["single_step_ablated_matrices"] = single_matrices

    return out


# ─────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────

def save_results_jsonl(results: List[Dict[str, Any]], path: str):
    """Append-safe JSONL (one JSON per line)."""
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def save_checkpoint(results: List[Dict[str, Any]], path: str):
    """Quick checkpoint that can be resumed from."""
    with open(path, "w") as f:
        json.dump(results, f)


def load_checkpoint(path: str) -> List[Dict[str, Any]]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full test-set batch ablation experiment")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_puzzles", type=int, default=0,
                        help="Number of puzzles to process (0 = all)")
    parser.add_argument("--ablate_level", type=str, default="H",
                        help="Which stream to ablate: H (z_H) or L (z_L)")
    parser.add_argument("--ablate_steps", type=str, default=None,
                        help="Comma-separated steps for single-step ablation "
                             "(e.g. 4,6,8,10). The all-steps ablation always runs.")
    parser.add_argument("--max_steps", type=int, default=16,
                        help="Max reasoning steps (default 16 = v1 native)")
    parser.add_argument("--output_dir", type=str,
                        default="results/batch_ablation")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_every", type=int, default=200,
                        help="Checkpoint every N puzzles")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    parser.add_argument(
        "--target_hours",
        type=float,
        default=0.0,
        help="Wall-clock runtime budget in hours (0 disables; run stops when budget is reached)",
    )
    args = parser.parse_args()

    args.ablate_level = _normalize_patch_level(args.ablate_level)
    if args.ablate_level == "both":
        raise ValueError("Ablating both z_H and z_L simultaneously is not supported. "
                         "Use --ablate_level H or --ablate_level L.")

    single_ablate_steps = _parse_int_list_arg(args.ablate_steps)
    if single_ablate_steps:
        single_ablate_steps = sorted(set(single_ablate_steps))

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Number of ablation variants per puzzle:
    # 1 (baseline) + 1 (all-steps) + len(single_ablate_steps) single-step runs
    n_variants = 2 + (len(single_ablate_steps) if single_ablate_steps else 0)

    print("=" * 70)
    print("BATCH ABLATION EXPERIMENT")
    print("=" * 70)
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Ablate target: z_{args.ablate_level}")
    print(f"  Single steps:  {single_ablate_steps if single_ablate_steps else 'none'}")
    print(f"  All-steps:     always")
    print(f"  Max steps:     {args.max_steps}")
    print(f"  Puzzles:       {'all' if args.num_puzzles == 0 else args.num_puzzles}")
    print(f"  Variants/puz:  {n_variants} (baseline + all-steps + {len(single_ablate_steps) if single_ablate_steps else 0} single-step)")
    print(f"  Device:        {device}")
    print(f"  Output:        {args.output_dir}")
    if args.target_hours > 0:
        print(f"  Runtime cap:   {args.target_hours:.2f} hours")
    print("=" * 70)

    # Load model
    ablator, test_loader = _load_model_and_dataloader(args.checkpoint, device)
    print("Model loaded.\n")

    # Resume support
    ckpt_path = os.path.join(args.output_dir, ".checkpoint.json")
    all_results: List[Dict[str, Any]] = []
    start_idx = 0
    if args.resume:
        all_results = load_checkpoint(ckpt_path)
        if all_results:
            start_idx = max(r["puzzle_idx"] for r in all_results) + 1
            print(f"Resuming from puzzle {start_idx} ({len(all_results)} already done)")

    # Determine total
    total = args.num_puzzles if args.num_puzzles > 0 else 999_999_999
    budget_seconds = args.target_hours * 3600.0 if args.target_hours > 0 else 0.0
    t_start = time.time()
    processed_this_session = 0

    for puzzle_idx, data in enumerate(test_loader):
        if puzzle_idx < start_idx:
            continue
        if puzzle_idx >= total:
            break

        batch = _extract_batch(data)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        result = process_one_puzzle(
            ablator, batch, puzzle_idx,
            ablate_level=args.ablate_level,
            single_ablate_steps=single_ablate_steps,
            max_steps=args.max_steps,
        )
        all_results.append(result)
        processed_this_session += 1

        # Progress logging
        elapsed = time.time() - t_start
        rate = processed_this_session / elapsed if elapsed > 0 else 0
        remaining = (total - puzzle_idx - 1) if total < 999_999_999 else 0
        eta_str = f"{remaining / rate / 60:.1f}m" if rate > 0 and remaining > 0 else "?"

        base_acc = result["baseline"]["final_accuracy"]
        all_delta = result["ablation_all_steps"]["accuracy_delta"]

        if processed_this_session <= 3 or processed_this_session % 50 == 0:
            # Condensed single-step summary
            ss_str = ""
            if single_ablate_steps:
                ss_parts = []
                for s in single_ablate_steps:
                    sk = str(s)
                    if sk in result["ablation_single_step"]:
                        d = result["ablation_single_step"][sk]["accuracy_delta"]
                        ss_parts.append(f"s{s}:{d:+.3f}")
                ss_str = "  " + " ".join(ss_parts)
            print(
                f"[{puzzle_idx:>6}] "
                f"base={base_acc:.3f}  all_Δ={all_delta:+.3f}"
                f"{ss_str}  "
                f"| {rate:.2f} puz/s  ETA {eta_str}"
            )

        # Optional runtime cap (keeps wall-clock near the requested budget)
        if budget_seconds > 0 and elapsed >= budget_seconds:
            print(
                f"\n[time-budget] Reached target runtime of {args.target_hours:.2f} hours "
                f"after {processed_this_session} puzzles in this session. Stopping cleanly."
            )
            break

        # Periodic checkpoint
        if args.save_every > 0 and processed_this_session % args.save_every == 0:
            save_checkpoint(all_results, ckpt_path)
            print(f"  [checkpoint] {len(all_results)} puzzles saved")

    # ── Final outputs ─────────────────────────────────────────
    total_elapsed = time.time() - t_start
    print(f"\nProcessed {processed_this_session} puzzles in {total_elapsed / 60:.1f} minutes "
          f"({processed_this_session / total_elapsed:.2f} puz/s)")

    # 1. Per-puzzle JSONL
    jsonl_path = os.path.join(args.output_dir, "results_per_puzzle.jsonl")
    save_results_jsonl(all_results, jsonl_path)
    print(f"Per-puzzle results → {jsonl_path}")

    # 2. Aggregate stats
    agg = compute_aggregates(all_results, single_ablate_steps)
    agg["config"] = {
        "checkpoint": args.checkpoint,
        "ablate_level": args.ablate_level,
        "single_ablate_steps": single_ablate_steps,
        "max_steps": args.max_steps,
        "num_puzzles_processed": len(all_results),
        "total_time_seconds": round(total_elapsed, 1),
    }
    agg_path = os.path.join(args.output_dir, "aggregate_stats.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate stats  → {agg_path}")

    # 3. Step accuracy matrix (for heatmaps / line charts)
    matrix = build_step_accuracy_matrix(all_results, single_ablate_steps)
    matrix_path = os.path.join(args.output_dir, "step_accuracy_matrix.json")
    with open(matrix_path, "w") as f:
        json.dump(matrix, f)
    print(f"Step acc matrix  → {matrix_path}")

    # 4. Summary to console
    _print_summary(agg, single_ablate_steps)

    # Clean up checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def _print_summary(agg: Dict[str, Any], single_ablate_steps: Optional[List[int]]):
    n = agg.get("num_puzzles", 0)
    if n == 0:
        return

    print(f"\n{'='*70}")
    print(f"SUMMARY — {n} puzzles")
    print(f"{'='*70}")

    b = agg["baseline"]["accuracy"]
    print(f"Baseline accuracy:  {b['mean']:.4f} ± {b['std']:.4f}  "
          f"[{b['min']:.4f}, {b['max']:.4f}]  median={b['median']:.4f}")

    a = agg["ablation_all_steps"]
    aa = a["accuracy"]
    ad = a["accuracy_delta"]
    print(f"\nAll-steps ablation:")
    print(f"  Accuracy:         {aa['mean']:.4f} ± {aa['std']:.4f}  "
          f"[{aa['min']:.4f}, {aa['max']:.4f}]")
    print(f"  Accuracy Δ:       {ad['mean']:+.4f} ± {ad['std']:.4f}")
    bk = a["accuracy_delta_buckets"]
    print(f"  Decreased/Same/Increased: "
          f"{bk['pct_decreased']:.1f}% / {bk['pct_unchanged']:.1f}% / {bk['pct_increased']:.1f}%")
    print(f"  Entropy Δ:        {a['entropy_delta']['mean']:+.4f}")
    print(f"  Confidence Δ:     {a['confidence_delta']['mean']:+.4f}")
    print(f"  Cells broken/puz: {a['cells_broken']['mean']:.1f}   "
          f"fixed: {a['cells_fixed']['mean']:.1f}")

    if single_ablate_steps:
        print(f"\nSingle-step ablation impact:")
        print(f"  {'Step':>6}  {'Acc Δ':>10}  {'Broken':>8}  {'Fixed':>8}  {'Changed':>8}  {'Ent Δ':>8}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
        for step in single_ablate_steps:
            sk = str(step)
            if sk in agg.get("ablation_single_step", {}):
                s = agg["ablation_single_step"][sk]
                print(f"  {step:>6}  {s['accuracy_delta']['mean']:>+10.4f}  "
                      f"{s['cells_broken']['mean']:>8.1f}  "
                      f"{s['cells_fixed']['mean']:>8.1f}  "
                      f"{s['cells_changed']['mean']:>8.1f}  "
                      f"{s['entropy_delta']['mean']:>+8.4f}")

        if "single_step_impact_ranking" in agg:
            print(f"\n  Impact ranking (most harmful first):")
            for r in agg["single_step_impact_ranking"]:
                print(f"    Step {r['step']}: mean Δacc = {r['mean_accuracy_delta']:+.4f}, "
                      f"mean broken = {r['mean_cells_broken']:.1f}")

    # Per-step trajectory preview
    traj = agg.get("ablation_all_steps", {}).get("per_step_trajectory", {})
    if traj:
        print(f"\nPer-step accuracy trajectory (all-steps ablation, mean across puzzles):")
        print(f"  {'Step':>6}  {'Baseline':>10}  {'Ablated':>10}  {'Δ':>10}")
        for sk in sorted(traj.keys(), key=lambda x: int(x)):
            t = traj[sk]
            if t.get("baseline_acc") and t.get("ablated_acc"):
                print(f"  {sk:>6}  {t['baseline_acc']['mean']:>10.4f}  "
                      f"{t['ablated_acc']['mean']:>10.4f}  "
                      f"{t['delta_acc']['mean']:>+10.4f}")


if __name__ == "__main__":
    main()
