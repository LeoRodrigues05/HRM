#!/usr/bin/env python3
"""SAE E10: Feature analysis — correlations, specialization, geometry, step profiles.

Analyzes trained SAE features to understand what they represent and how
they relate to constraint-tracking probe directions from E8.

Analyses:
  A. Feature activation patterns (which puzzles/cells/steps activate each feature)
  B. Feature specialization (correlation with constraint targets)
  C. Feature geometry (decoder column similarity, comparison with E8 probes)
  D. Feature activation across steps (early vs late features)
  E. Dead feature analysis

Output
------
  results/sae_study/feature_analysis/
    activation_patterns.json
    specialization_matrix.pt
    geometry_analysis.json
    step_profiles.json
    dead_feature_analysis.json

Usage
-----
    python scripts/sae/sae_analyze_features.py --sae_path results/sae_study/sae_d2048_l10.01.pt
"""

import os
import sys
import json
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.sae import SparseAutoencoder, TopKSparseAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

SUDOKU_SIZE = 9
SUDOKU_CELLS = 81
DIGIT_OFFSET = 1

# Probe targets matching E8
BINARY_TARGETS = [
    "per_cell_correct", "is_given", "is_empty",
    "violated_in_row", "violated_in_col", "violated_in_box",
    "is_naked_single",
    "is_hidden_single_row", "is_hidden_single_col", "is_hidden_single_box",
]


# ═══════════════════════════════════════════════════════════════════════════
# Constraint label derivation (self-contained)
# ═══════════════════════════════════════════════════════════════════════════

def derive_per_cell_labels(preds, targets, inputs):
    """Derive constraint labels for every cell. Returns dict of [B,81] tensors."""
    B = preds.shape[0]
    labels = {}
    digits = (preds.long() - DIGIT_OFFSET).clamp(0, SUDOKU_SIZE)
    target_digits = (targets.long() - DIGIT_OFFSET).clamp(0, SUDOKU_SIZE)
    input_digits = (inputs.long() - DIGIT_OFFSET).clamp(0, SUDOKU_SIZE)
    grid = digits.view(B, SUDOKU_SIZE, SUDOKU_SIZE)

    labels["per_cell_correct"] = (digits == target_digits).int()
    labels["is_given"] = (input_digits != 0).int()
    labels["is_empty"] = (digits == 0).int()

    violated_row = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    violated_col = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    violated_box = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    for r in range(SUDOKU_SIZE):
        for c in range(SUDOKU_SIZE):
            idx = r * SUDOKU_SIZE + c
            d = grid[:, r, c]
            nonzero = d > 0
            row_count = (grid[:, r, :] == d.unsqueeze(1)).sum(dim=1)
            violated_row[:, idx] = (nonzero & (row_count > 1)).int()
            col_count = (grid[:, :, c] == d.unsqueeze(1)).sum(dim=1)
            violated_col[:, idx] = (nonzero & (col_count > 1)).int()
            br, bc = (r // 3) * 3, (c // 3) * 3
            box = grid[:, br:br+3, bc:bc+3].reshape(B, 9)
            box_count = (box == d.unsqueeze(1)).sum(dim=1)
            violated_box[:, idx] = (nonzero & (box_count > 1)).int()
    labels["violated_in_row"] = violated_row
    labels["violated_in_col"] = violated_col
    labels["violated_in_box"] = violated_box

    # Candidate-based features
    used_row = torch.zeros(B, SUDOKU_SIZE, SUDOKU_SIZE + 1, dtype=torch.bool)
    used_col = torch.zeros(B, SUDOKU_SIZE, SUDOKU_SIZE + 1, dtype=torch.bool)
    used_box = torch.zeros(B, SUDOKU_SIZE, SUDOKU_SIZE + 1, dtype=torch.bool)
    for r in range(SUDOKU_SIZE):
        for c in range(SUDOKU_SIZE):
            d = grid[:, r, c]
            b = (r // 3) * 3 + (c // 3)
            for dd in range(1, SUDOKU_SIZE + 1):
                mask = (d == dd)
                used_row[:, r, dd] |= mask
                used_col[:, c, dd] |= mask
                used_box[:, b, dd] |= mask

    candidate_set = torch.zeros(B, SUDOKU_CELLS, SUDOKU_SIZE, dtype=torch.bool)
    is_naked_single = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    for r in range(SUDOKU_SIZE):
        for c in range(SUDOKU_SIZE):
            idx = r * SUDOKU_SIZE + c
            blank = (grid[:, r, c] == 0)
            b = (r // 3) * 3 + (c // 3)
            allowed = ~(used_row[:, r, 1:SUDOKU_SIZE+1] |
                        used_col[:, c, 1:SUDOKU_SIZE+1] |
                        used_box[:, b, 1:SUDOKU_SIZE+1])
            cands = allowed & blank.unsqueeze(1)
            candidate_set[:, idx, :] = cands
            cc = cands.sum(dim=1)
            is_naked_single[:, idx] = (blank & (cc == 1)).int()
    labels["is_naked_single"] = is_naked_single

    # Hidden singles
    hs_row = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    hs_col = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    hs_box = torch.zeros(B, SUDOKU_CELLS, dtype=torch.int32)
    for r in range(SUDOKU_SIZE):
        for c in range(SUDOKU_SIZE):
            idx = r * SUDOKU_SIZE + c
            blank = (grid[:, r, c] == 0)
            if not blank.any():
                continue
            cell_cands = candidate_set[:, idx, :]
            for d in range(SUDOKU_SIZE):
                has = cell_cands[:, d]
                if not has.any():
                    continue
                others_row = torch.zeros(B, dtype=torch.bool)
                for cc in range(SUDOKU_SIZE):
                    if cc != c:
                        others_row |= candidate_set[:, r * SUDOKU_SIZE + cc, d]
                hs_row[:, idx] |= (has & ~others_row).int()
                others_col = torch.zeros(B, dtype=torch.bool)
                for rr in range(SUDOKU_SIZE):
                    if rr != r:
                        others_col |= candidate_set[:, rr * SUDOKU_SIZE + c, d]
                hs_col[:, idx] |= (has & ~others_col).int()
                br2, bc2 = (r // 3) * 3, (c // 3) * 3
                others_box = torch.zeros(B, dtype=torch.bool)
                for dr in range(3):
                    for dc in range(3):
                        rr2, cc2 = br2 + dr, bc2 + dc
                        if rr2 != r or cc2 != c:
                            others_box |= candidate_set[:, rr2 * SUDOKU_SIZE + cc2, d]
                hs_box[:, idx] |= (has & ~others_box).int()
    labels["is_hidden_single_row"] = hs_row
    labels["is_hidden_single_col"] = hs_col
    labels["is_hidden_single_box"] = hs_box

    return labels


# ═══════════════════════════════════════════════════════════════════════════
# Load SAE and activations
# ═══════════════════════════════════════════════════════════════════════════

def load_sae(sae_path: str, device: torch.device) -> SparseAutoencoder:
    """Load trained SAE from checkpoint (supports both relu and topk)."""
    ckpt = torch.load(sae_path, map_location=device, weights_only=False)
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
    return sae


def load_activations_and_labels(
    activations_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Load activations and derive constraint labels."""
    data = torch.load(activations_path, map_location="cpu", weights_only=False)
    z_H = data['z_H']          # [N, steps, 81, D]
    inputs = data['inputs']     # [N, 81]
    labels = data['labels']     # [N, 81]
    preds = data['per_step_preds']  # [N, steps, 81]
    per_step_acc = data['per_step_accuracy']  # [N, steps]

    return {
        'z_H': z_H,
        'inputs': inputs,
        'labels': labels,
        'preds': preds,
        'per_step_accuracy': per_step_acc,
        'n_puzzles': z_H.shape[0],
        'n_steps': z_H.shape[1],
    }


# ═══════════════════════════════════════════════════════════════════════════
# A. Feature activation patterns
# ═══════════════════════════════════════════════════════════════════════════

def analyze_activation_patterns(
    sae: SparseAutoencoder,
    z_H: torch.Tensor,          # [N, steps, 81, D]
    device: torch.device,
    top_k: int = 50,
) -> Dict[str, Any]:
    """For each SAE feature, compute which puzzles/cells/steps activate it most.

    Computes statistics incrementally per puzzle to avoid materializing the
    full [N*S*C, dict_size] encoded tensor (which can exceed 10 GB).
    """
    N, S, C, D = z_H.shape
    dict_size = sae.dict_size
    logger.info("Analyzing feature activation patterns...")

    # Accumulators — kept on CPU, computed per-puzzle
    total_activation = torch.zeros(dict_size)            # [dict_size]
    fire_count = torch.zeros(dict_size)                  # [dict_size]
    puzzle_activation = torch.zeros(N, dict_size)        # [N, dict_size]
    step_activation = torch.zeros(S, dict_size)          # [S, dict_size]
    total_samples = 0

    for pi in range(N):
        # z_H[pi]: [S, C, D] — encode all steps×cells for one puzzle
        z_puzzle = z_H[pi].reshape(S * C, D).to(device)  # [S*C, D]
        with torch.no_grad():
            h_puzzle = sae.encode(z_puzzle).cpu()  # [S*C, dict_size]

        total_activation += h_puzzle.sum(dim=0)
        fire_count += (h_puzzle > 0).float().sum(dim=0)
        total_samples += h_puzzle.shape[0]

        # Per-puzzle total
        puzzle_activation[pi] = h_puzzle.sum(dim=0)

        # Per-step total: reshape to [S, C, dict_size], sum over cells
        h_by_step = h_puzzle.reshape(S, C, dict_size)
        step_activation += h_by_step.sum(dim=1)  # [S, dict_size]

        if (pi + 1) % 200 == 0:
            logger.info(f"  Activation patterns: {pi+1}/{N} puzzles")

    # Select top features by total activation
    top_features = total_activation.argsort(descending=True)[:top_k]

    feature_patterns = {}
    for rank, feat_idx in enumerate(top_features.tolist()):
        # Top puzzles for this feature
        pa = puzzle_activation[:, feat_idx]         # [N]
        top_puzzles = pa.argsort(descending=True)[:10].tolist()

        # Top steps for this feature
        sa = step_activation[:, feat_idx]           # [S]
        top_steps = sa.argsort(descending=True)[:5].tolist()

        # Mean activation per step = step_total / (N * C)
        mean_per_step = (sa / (N * C)).tolist()

        feature_patterns[feat_idx] = {
            'rank': rank,
            'total_activation': float(total_activation[feat_idx]),
            'fire_fraction': float(fire_count[feat_idx]) / total_samples,
            'top_puzzles': top_puzzles,
            'top_steps': top_steps,
            'mean_activation_per_step': mean_per_step,
        }

    return {
        'top_features': top_features.tolist(),
        'feature_patterns': feature_patterns,
        'total_alive': int((fire_count > 0).sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# B. Feature specialization (correlation with constraint targets)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_specialization(
    sae: SparseAutoencoder,
    z_H: torch.Tensor,          # [N, steps, 81, D]
    preds: torch.Tensor,        # [N, steps, 81]
    targets: torch.Tensor,      # [N, 81]
    inputs: torch.Tensor,       # [N, 81]
    device: torch.device,
    steps_to_analyze: Optional[List[int]] = None,
    top_k: int = 100,
) -> Dict[str, Any]:
    """Compute correlation between SAE features and constraint targets."""
    N, S, C, D = z_H.shape
    if steps_to_analyze is None:
        steps_to_analyze = list(range(S))

    logger.info(f"Analyzing feature specialization at steps {steps_to_analyze}...")

    # Accumulate statistics incrementally to avoid materializing full h_cat
    # We need: sum_h, sum_h_sq, sum_y, sum_y_sq, sum_hy, total_count
    dict_size = sae.dict_size
    target_names = list(BINARY_TARGETS)
    n_targets = len(target_names)

    sum_h = torch.zeros(dict_size)                    # [dict_size]
    sum_h_sq = torch.zeros(dict_size)                 # [dict_size]
    sum_y = torch.zeros(n_targets)                    # [n_targets]
    sum_y_sq = torch.zeros(n_targets)                 # [n_targets]
    sum_hy = torch.zeros(dict_size, n_targets)        # [dict_size, n_targets]
    total_count = 0

    for step in steps_to_analyze:
        z_step = z_H[:, step, :, :].reshape(-1, D).to(device)  # [N*81, D]
        with torch.no_grad():
            h_step = sae.encode(z_step).cpu()  # [N*81, dict_size]

        # Derive labels at this step
        step_preds = preds[:, step, :]  # [N, 81]
        cell_labels = derive_per_cell_labels(step_preds, targets, inputs)

        # Build label matrix for this step [N*81, n_targets]
        y_step = torch.stack(
            [cell_labels[t].reshape(-1).float() for t in target_names], dim=1
        )

        # Accumulate sufficient statistics
        n = h_step.shape[0]
        sum_h += h_step.sum(dim=0)
        sum_h_sq += (h_step ** 2).sum(dim=0)
        sum_y += y_step.sum(dim=0)
        sum_y_sq += (y_step ** 2).sum(dim=0)
        sum_hy += h_step.T @ y_step  # [dict_size, n_targets]
        total_count += n

    # Compute correlation from sufficient statistics
    # corr = (E[HY] - E[H]E[Y]) / (std_H * std_Y)
    N_total = total_count
    mean_h = sum_h / N_total                        # [dict_size]
    mean_y = sum_y / N_total                        # [n_targets]
    var_h = (sum_h_sq / N_total - mean_h ** 2).clamp(min=0)
    var_y = (sum_y_sq / N_total - mean_y ** 2).clamp(min=0)
    h_std = var_h.sqrt()                            # [dict_size]
    y_std = var_y.sqrt()                            # [n_targets]

    cov = sum_hy / N_total - mean_h.unsqueeze(1) * mean_y.unsqueeze(0)

    denom = (h_std.unsqueeze(1) * y_std.unsqueeze(0)).clamp(min=1e-8)
    correlation_matrix = cov / denom

    # Find specialized features: high correlation with one target, low with others
    specialized_features = []
    for feat in range(dict_size):
        corrs = correlation_matrix[feat]
        max_corr, max_idx = corrs.abs().max(dim=0)
        if max_corr > 0.3:
            # Check other correlations are low
            others = torch.cat([corrs[:max_idx], corrs[max_idx+1:]])
            max_other = others.abs().max()
            specialized_features.append({
                'feature_idx': feat,
                'best_target': target_names[max_idx.item()],
                'best_corr': float(corrs[max_idx]),
                'max_other_corr': float(max_other),
                'is_highly_specialized': bool(max_corr > 0.5 and max_other < 0.1),
            })

    specialized_features.sort(key=lambda x: abs(x['best_corr']), reverse=True)

    return {
        'correlation_matrix': correlation_matrix,
        'target_names': target_names,
        'specialized_features': specialized_features[:top_k],
        'n_highly_specialized': sum(1 for f in specialized_features if f['is_highly_specialized']),
    }


# ═══════════════════════════════════════════════════════════════════════════
# C. Feature geometry
# ═══════════════════════════════════════════════════════════════════════════

def analyze_geometry(
    sae: SparseAutoencoder,
    probe_weights_path: Optional[str] = None,
    top_k: int = 100,
) -> Dict[str, Any]:
    """Analyze SAE decoder column geometry and comparison with E8 probes.

    Computing pairwise cosine similarity of decoder columns shows how
    diverse or redundant the learned dictionary is. A well-trained SAE
    should have mostly dissimilar columns (diverse feature set).

    Comparing decoder columns to E8 probe directions (MMCS — Mean Max
    Cosine Similarity) shows how well the SAE features align with the
    linear probe directions. High MMCS means SAE features span the same
    subspace; low MMCS means SAE found features that probes missed.
    This is significant because E9 showed probe directions have minimal
    causal effect — if SAE features are different from probe directions
    AND have causal effects, it would support the hypothesis that SAE
    finds the model's actual computational features vs. mere readout
    directions.
    """
    logger.info("Analyzing feature geometry...")

    # Decoder columns: [input_dim, dict_size]
    decoder_W = sae.decoder.weight.data.float()  # [input_dim, dict_size]
    dict_size = decoder_W.shape[1]

    # Normalize columns
    decoder_norm = F.normalize(decoder_W, dim=0)  # [input_dim, dict_size]

    # Pairwise cosine similarity of top-k features
    # (full matrix is too large for large dict_size)
    cos_sim = (decoder_norm.T @ decoder_norm).cpu()  # [dict_size, dict_size]

    # Statistics (exclude diagonal)
    mask = ~torch.eye(dict_size, dtype=torch.bool)
    off_diag = cos_sim[mask]
    geometry_stats = {
        'mean_pairwise_cosine': float(off_diag.mean()),
        'max_pairwise_cosine': float(off_diag.max()),
        'std_pairwise_cosine': float(off_diag.std()),
        'median_pairwise_cosine': float(off_diag.median()),
        'frac_above_0.5': float((off_diag.abs() > 0.5).float().mean()),
        'frac_above_0.9': float((off_diag.abs() > 0.9).float().mean()),
    }

    # Compare with E8 probe directions if available
    probe_comparison = {}
    if probe_weights_path and os.path.exists(probe_weights_path):
        logger.info(f"Loading E8 probe weights from {probe_weights_path}")
        probe_data = torch.load(probe_weights_path, map_location="cpu", weights_only=False)

        # Select best probe direction per target (best step), matching E9 pattern
        best_per_target = {}  # target → (val_score, W_vec)
        probe_directions = []
        probe_names = []
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
                U, S, _ = torch.svd(W.float())
                w_vec = (W.float().T @ U[:, 0])
                w_vec = w_vec / w_vec.norm().clamp(min=1e-8)
            if target not in best_per_target or score > best_per_target[target][0]:
                best_per_target[target] = (score, w_vec)
        for target, (score, w_vec) in best_per_target.items():
            probe_directions.append(w_vec)
            probe_names.append(target)

        if probe_directions:
            probe_matrix = torch.stack(probe_directions)  # [n_probes, input_dim]
            probe_matrix = F.normalize(probe_matrix, dim=1).to(decoder_norm.device)

            # MMCS: for each SAE decoder column, find max cosine with any probe
            cos_with_probes = probe_matrix @ decoder_norm  # [n_probes, dict_size]
            max_cos_per_feature = cos_with_probes.abs().max(dim=0).values  # [dict_size]
            mmcs = float(max_cos_per_feature.mean())

            # Also: for each probe, find max cosine with any SAE feature
            max_cos_per_probe = cos_with_probes.abs().max(dim=1).values  # [n_probes]

            probe_comparison = {
                'mmcs_sae_to_probes': mmcs,
                'mmcs_probes_to_sae': float(max_cos_per_probe.mean()),
                'per_probe_max_cos': {
                    name: float(val) for name, val in zip(probe_names, max_cos_per_probe)
                },
                'n_probes': len(probe_names),
                'probe_names': probe_names,
            }
            logger.info(f"  MMCS (SAE→probes): {mmcs:.4f}")
            logger.info(f"  MMCS (probes→SAE): {probe_comparison['mmcs_probes_to_sae']:.4f}")

    return {
        'geometry_stats': geometry_stats,
        'probe_comparison': probe_comparison,
        'pairwise_cosine_sim': cos_sim,
    }


# ═══════════════════════════════════════════════════════════════════════════
# D. Feature activation across steps
# ═══════════════════════════════════════════════════════════════════════════

def analyze_step_profiles(
    sae: SparseAutoencoder,
    z_H: torch.Tensor,     # [N, steps, 81, D]
    per_step_accuracy: torch.Tensor,  # [N, steps]
    device: torch.device,
    top_k: int = 50,
) -> Dict[str, Any]:
    """Analyze how each feature's activation varies across ACT steps."""
    N, S, C, D = z_H.shape
    logger.info("Analyzing step profiles...")

    # Encode at each step
    mean_activation_per_step = torch.zeros(sae.dict_size, S)
    fire_rate_per_step = torch.zeros(sae.dict_size, S)

    for step in range(S):
        z_step = z_H[:, step, :, :].reshape(-1, D).to(device)
        with torch.no_grad():
            h = sae.encode(z_step).cpu()  # [N*81, dict_size]
        mean_activation_per_step[:, step] = h.mean(dim=0)
        fire_rate_per_step[:, step] = (h > 0).float().mean(dim=0)

    # Classify features as early vs late
    # Early: peak activation in steps 0-4
    # Late: peak activation in steps 12-15
    peak_step = mean_activation_per_step.argmax(dim=1)  # [dict_size]
    early_mask = peak_step <= 4
    late_mask = peak_step >= 12
    middle_mask = ~early_mask & ~late_mask

    # Correlation of feature activation profiles with accuracy profiles
    mean_acc_profile = per_step_accuracy.mean(dim=0)  # [S]

    step_corr_with_acc = torch.zeros(sae.dict_size)
    for feat in range(sae.dict_size):
        profile = mean_activation_per_step[feat]
        if profile.std() < 1e-8:
            continue
        corr = torch.corrcoef(torch.stack([profile, mean_acc_profile]))[0, 1]
        step_corr_with_acc[feat] = corr if not torch.isnan(corr) else 0.0

    # Top features by total activation
    total_act = mean_activation_per_step.sum(dim=1)
    top_features = total_act.argsort(descending=True)[:top_k]

    feature_profiles = {}
    for feat_idx in top_features.tolist():
        feature_profiles[feat_idx] = {
            'mean_activation_per_step': mean_activation_per_step[feat_idx].tolist(),
            'fire_rate_per_step': fire_rate_per_step[feat_idx].tolist(),
            'peak_step': int(peak_step[feat_idx]),
            'category': 'early' if early_mask[feat_idx] else ('late' if late_mask[feat_idx] else 'middle'),
            'corr_with_accuracy': float(step_corr_with_acc[feat_idx]),
        }

    return {
        'n_early': int(early_mask.sum()),
        'n_late': int(late_mask.sum()),
        'n_middle': int(middle_mask.sum()),
        'mean_accuracy_profile': mean_acc_profile.tolist(),
        'feature_profiles': feature_profiles,
        'top_features': top_features.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# E. Dead feature analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_dead_features(
    sae: SparseAutoencoder,
    z_H: torch.Tensor,     # [N, steps, 81, D]
    device: torch.device,
) -> Dict[str, Any]:
    """Count dead and rarely-active features."""
    N, S, C, D = z_H.shape

    total_fire_count = torch.zeros(sae.dict_size)
    batch_size = 8192
    flat = z_H.reshape(-1, D)
    total_inputs = flat.shape[0]

    for start in range(0, total_inputs, batch_size):
        batch = flat[start:start + batch_size].to(device)
        with torch.no_grad():
            h = sae.encode(batch).cpu()
        total_fire_count += (h > 0).float().sum(dim=0)
    fire_rate = total_fire_count / total_inputs

    return {
        'total_inputs': total_inputs,
        'n_dead': int((total_fire_count == 0).sum()),
        'n_rare_0.1pct': int((fire_rate < 0.001).sum()),
        'n_rare_1pct': int((fire_rate < 0.01).sum()),
        'alive_frac': float((total_fire_count > 0).float().mean()),
        'mean_fire_rate': float(fire_rate.mean()),
        'median_fire_rate': float(fire_rate.median()),
        'fire_rate_distribution': {
            '0%': int((fire_rate == 0).sum()),
            '0-0.1%': int(((fire_rate > 0) & (fire_rate < 0.001)).sum()),
            '0.1-1%': int(((fire_rate >= 0.001) & (fire_rate < 0.01)).sum()),
            '1-10%': int(((fire_rate >= 0.01) & (fire_rate < 0.1)).sum()),
            '10-50%': int(((fire_rate >= 0.1) & (fire_rate < 0.5)).sum()),
            '50-100%': int((fire_rate >= 0.5).sum()),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SAE E10: Feature analysis")
    parser.add_argument("--sae_path", type=str, required=True,
                        help="Path to trained SAE checkpoint")
    parser.add_argument("--activations_path", type=str,
                        default="results/sae_study/activations_zH.pt")
    parser.add_argument("--probe_weights_path", type=str,
                        default="results/probes/e8_constraint_probes/probe_weights.pt",
                        help="Path to E8 probe weights (for geometry comparison)")
    parser.add_argument("--steps_to_analyze", type=str, default=None,
                        help="Comma-separated steps for specialization analysis (default: all)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of top features to analyze in detail")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/sae_study/feature_analysis")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load SAE and activations
    sae = load_sae(args.sae_path, device)
    data = load_activations_and_labels(args.activations_path, device)
    z_H = data['z_H']
    logger.info(f"SAE: input_dim={sae.input_dim}, dict_size={sae.dict_size}, l1={sae.l1_coeff}")
    logger.info(f"Activations: {z_H.shape}")

    # A. Activation patterns
    logger.info("\n=== A. Feature Activation Patterns ===")
    patterns = analyze_activation_patterns(sae, z_H, device, top_k=args.top_k)
    with open(os.path.join(args.output_dir, "activation_patterns.json"), 'w') as f:
        json.dump(patterns, f, indent=2)
    logger.info(f"  Alive features: {patterns['total_alive']}/{sae.dict_size}")

    # B. Feature specialization
    logger.info("\n=== B. Feature Specialization ===")
    steps_to_analyze = None
    if args.steps_to_analyze:
        steps_to_analyze = [int(s) for s in args.steps_to_analyze.split(",")]
    spec = analyze_specialization(
        sae, z_H, data['preds'], data['labels'], data['inputs'],
        device, steps_to_analyze=steps_to_analyze, top_k=args.top_k,
    )
    # Save correlation matrix separately (large tensor)
    torch.save({
        'correlation_matrix': spec['correlation_matrix'],
        'target_names': spec['target_names'],
    }, os.path.join(args.output_dir, "specialization_matrix.pt"))
    # Save summary as JSON
    with open(os.path.join(args.output_dir, "specialization_summary.json"), 'w') as f:
        json.dump({
            'target_names': spec['target_names'],
            'specialized_features': spec['specialized_features'],
            'n_highly_specialized': spec['n_highly_specialized'],
        }, f, indent=2)
    logger.info(f"  Highly specialized features (|corr|>0.5, others<0.1): {spec['n_highly_specialized']}")
    if spec['specialized_features']:
        for sf in spec['specialized_features'][:5]:
            logger.info(f"    Feature {sf['feature_idx']}: {sf['best_target']} "
                       f"corr={sf['best_corr']:.3f}")

    # C. Geometry
    logger.info("\n=== C. Feature Geometry ===")
    geo = analyze_geometry(sae, args.probe_weights_path, top_k=args.top_k)
    # Save pairwise cosine separately
    torch.save(geo['pairwise_cosine_sim'],
               os.path.join(args.output_dir, "pairwise_cosine_sim.pt"))
    with open(os.path.join(args.output_dir, "geometry_analysis.json"), 'w') as f:
        json.dump({
            'geometry_stats': geo['geometry_stats'],
            'probe_comparison': geo['probe_comparison'],
        }, f, indent=2)
    logger.info(f"  Mean pairwise cosine: {geo['geometry_stats']['mean_pairwise_cosine']:.4f}")

    # D. Step profiles
    logger.info("\n=== D. Step Profiles ===")
    step_prof = analyze_step_profiles(
        sae, z_H, data['per_step_accuracy'], device, top_k=args.top_k,
    )
    with open(os.path.join(args.output_dir, "step_profiles.json"), 'w') as f:
        json.dump(step_prof, f, indent=2)
    logger.info(f"  Early features: {step_prof['n_early']}, "
               f"Late: {step_prof['n_late']}, "
               f"Middle: {step_prof['n_middle']}")

    # E. Dead feature analysis
    logger.info("\n=== E. Dead Feature Analysis ===")
    dead = analyze_dead_features(sae, z_H, device)
    with open(os.path.join(args.output_dir, "dead_feature_analysis.json"), 'w') as f:
        json.dump(dead, f, indent=2)
    logger.info(f"  Dead: {dead['n_dead']}, Alive: {dead['alive_frac']:.2%}")
    logger.info(f"  Fire rate distribution: {dead['fire_rate_distribution']}")

    logger.info(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
