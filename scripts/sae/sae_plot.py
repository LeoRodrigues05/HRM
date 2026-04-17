#!/usr/bin/env python3
"""SAE E10: Publication-quality plots for the SAE study.

Generates plots for:
1. Reconstruction quality (cosine similarity scatter)
2. Hyperparameter sweep heatmaps
3. Feature specialization matrix
4. Feature activation profiles across steps
5. Causal comparison (SAE vs probes vs random)
6. Geometry comparison (decoder columns vs probe directions)

Output
------
  results/sae_study/plots/  (PNG + PDF)

Usage
-----
    python scripts/sae/sae_plot.py --sae_path results/sae_study/sae_d2048_l10.01.pt
"""

import os
import sys
import json
import argparse
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from models.sae import SparseAutoencoder, TopKSparseAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def save_fig(fig, output_dir, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(os.path.join(output_dir, f"{name}.png"))
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"))
    plt.close(fig)
    logger.info(f"  Saved {name}.png/pdf")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Reconstruction quality
# ═══════════════════════════════════════════════════════════════════════════

def plot_reconstruction_quality(sae, z_H, output_dir, device, n_samples=5000):
    """Scatter plot of original vs. SAE-reconstructed z_H, with cosine similarity."""
    logger.info("Plotting reconstruction quality...")
    flat = z_H.reshape(-1, z_H.shape[-1])
    indices = torch.randperm(flat.shape[0])[:n_samples]
    sample = flat[indices].to(device)

    with torch.no_grad():
        x_hat, h, loss_dict = sae(sample)
        cos_sim = F.cosine_similarity(sample, x_hat, dim=-1).cpu()
        recon_error = (sample - x_hat).pow(2).sum(dim=-1).sqrt().cpu()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Cosine similarity histogram
    ax = axes[0]
    ax.hist(cos_sim.numpy(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(cos_sim.mean(), color='red', linestyle='--', label=f'Mean={cos_sim.mean():.4f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Reconstruction Cosine Similarity')
    ax.legend()

    # Reconstruction error histogram
    ax = axes[1]
    ax.hist(recon_error.numpy(), bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(recon_error.mean(), color='red', linestyle='--', label=f'Mean={recon_error.mean():.2f}')
    ax.set_xlabel('L2 Error')
    ax.set_ylabel('Count')
    ax.set_title('Reconstruction Error')
    ax.legend()

    # Scatter: original norm vs reconstruction norm
    ax = axes[2]
    orig_norms = sample.cpu().norm(dim=-1).numpy()
    recon_norms = x_hat.cpu().norm(dim=-1).numpy()
    ax.scatter(orig_norms, recon_norms, alpha=0.1, s=2, color='steelblue')
    lim = max(orig_norms.max(), recon_norms.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.4, label='y=x')
    ax.set_xlabel('Original ‖z_H‖')
    ax.set_ylabel('Reconstructed ‖z_H‖')
    ax.set_title('Norm Comparison')
    ax.legend()
    ax.set_aspect('equal')

    fig.suptitle(f'SAE Reconstruction Quality (d={sae.dict_size}, λ₁={sae.l1_coeff})', fontsize=14)
    fig.tight_layout()
    save_fig(fig, output_dir, 'reconstruction_quality')


# ═══════════════════════════════════════════════════════════════════════════
# 2. Hyperparameter sweep heatmaps
# ═══════════════════════════════════════════════════════════════════════════

def plot_sweep_heatmaps(output_dir, sweep_csv_path):
    """Heatmap of (dict_size × l1_coeff) → reconstruction_loss, alive_features."""
    logger.info("Plotting sweep heatmaps...")
    import csv

    rows = []
    with open(sweep_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        logger.warning("No sweep results found")
        return

    dict_sizes = sorted(set(int(r['dict_size']) for r in rows))
    l1_coeffs = sorted(set(float(r['l1_coeff']) for r in rows))

    metrics = [
        ('final_recon_loss', 'Reconstruction Loss', 'YlOrRd_r'),
        ('alive_count', 'Alive Features', 'YlGn'),
        ('L0', 'L0 (Mean Active Features)', 'YlOrBr'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, title, cmap) in zip(axes, metrics):
        matrix = np.full((len(dict_sizes), len(l1_coeffs)), np.nan)
        for row in rows:
            di = dict_sizes.index(int(row['dict_size']))
            li = l1_coeffs.index(float(row['l1_coeff']))
            matrix[di, li] = float(row[metric])

        im = ax.imshow(matrix, aspect='auto', cmap=cmap)
        ax.set_xticks(range(len(l1_coeffs)))
        ax.set_xticklabels([f'{l:.3f}' for l in l1_coeffs], rotation=45, ha='right')
        ax.set_yticks(range(len(dict_sizes)))
        ax.set_yticklabels([str(d) for d in dict_sizes])
        ax.set_xlabel('L1 Coefficient')
        ax.set_ylabel('Dictionary Size')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Annotate cells
        for i in range(len(dict_sizes)):
            for j in range(len(l1_coeffs)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text = f'{val:.3f}' if val < 10 else f'{int(val)}'
                    ax.text(j, i, text, ha='center', va='center', fontsize=8)

    fig.suptitle('SAE Hyperparameter Sweep', fontsize=14)
    fig.tight_layout()
    save_fig(fig, output_dir, 'sweep_heatmaps')


# ═══════════════════════════════════════════════════════════════════════════
# 3. Feature specialization matrix
# ═══════════════════════════════════════════════════════════════════════════

def plot_specialization_matrix(output_dir, analysis_dir, top_k=20):
    """Heatmap of [top features × constraint targets], colored by correlation."""
    logger.info("Plotting specialization matrix...")

    spec_path = os.path.join(analysis_dir, "specialization_matrix.pt")
    if not os.path.exists(spec_path):
        logger.warning(f"Specialization matrix not found: {spec_path}")
        return

    data = torch.load(spec_path, map_location="cpu", weights_only=False)
    corr_matrix = data['correlation_matrix']  # [dict_size, n_targets]
    target_names = data['target_names']

    # Select top features by max absolute correlation
    max_corr_per_feat = corr_matrix.abs().max(dim=1).values
    top_indices = max_corr_per_feat.argsort(descending=True)[:top_k]
    subset = corr_matrix[top_indices].numpy()

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.35)))
    im = ax.imshow(subset, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f'Feature {i}' for i in top_indices.tolist()], fontsize=8)
    ax.set_xlabel('Constraint Target')
    ax.set_ylabel('SAE Feature')
    ax.set_title(f'Top-{top_k} SAE Features × Constraint Target Correlations')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')

    fig.tight_layout()
    save_fig(fig, output_dir, 'specialization_matrix')


# ═══════════════════════════════════════════════════════════════════════════
# 4. Feature activation profiles across steps
# ═══════════════════════════════════════════════════════════════════════════

def plot_step_profiles(output_dir, analysis_dir, top_k=20):
    """Line plot of mean activation per step for top features."""
    logger.info("Plotting step profiles...")

    profiles_path = os.path.join(analysis_dir, "step_profiles.json")
    if not os.path.exists(profiles_path):
        logger.warning(f"Step profiles not found: {profiles_path}")
        return

    with open(profiles_path) as f:
        data = json.load(f)

    feature_profiles = data['feature_profiles']
    mean_acc_profile = data['mean_accuracy_profile']
    top_features = data['top_features'][:top_k]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    cmap = plt.cm.get_cmap('tab20', min(top_k, 20))
    steps = list(range(len(mean_acc_profile)))

    for i, feat_idx in enumerate(top_features):
        fp = feature_profiles.get(str(feat_idx), {})
        profile = fp.get('mean_activation_per_step', [])
        category = fp.get('category', 'middle')
        if profile:
            linestyle = {'early': ':', 'middle': '--', 'late': '-'}.get(category, '-')
            ax1.plot(steps[:len(profile)], profile, label=f'F{feat_idx} ({category})',
                    color=cmap(i % 20), linewidth=1.5, linestyle=linestyle)

    ax1.set_xlabel('ACT Step')
    ax1.set_ylabel('Mean Activation')
    ax1.set_title(f'Top-{top_k} SAE Feature Activation Across Steps')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    ax1.grid(alpha=0.3)

    # Accuracy profile
    ax2.plot(steps, mean_acc_profile, 'k-o', linewidth=2, markersize=4, label='Cell Accuracy')
    ax2.set_xlabel('ACT Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Mean Cell Accuracy per Step')
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    save_fig(fig, output_dir, 'step_profiles')


# ═══════════════════════════════════════════════════════════════════════════
# 5. Causal comparison bar chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_causal_comparison(output_dir, causal_dir):
    """Bar chart: SAE features vs probe directions vs random."""
    logger.info("Plotting causal comparison...")

    agg_path = os.path.join(causal_dir, "aggregate.json")
    if not os.path.exists(agg_path):
        logger.warning(f"Aggregate results not found: {agg_path}")
        return

    with open(agg_path) as f:
        agg = json.load(f)

    conditions = agg.get('conditions', {})
    if not conditions:
        return

    names = []
    means = []
    stds = []
    colors = []
    color_map = {
        'sae_top_features': '#2196F3',
        'random_sae_features': '#90CAF9',
        'probe_directions': '#FF9800',
        'random_directions': '#FFE0B2',
    }

    for cond_name, stats in conditions.items():
        names.append(cond_name.replace('_', '\n'))
        means.append(stats['mean_delta_acc'])
        stds.append(stats['std_delta_acc'] / max(1, stats['n_samples']**0.5))  # SEM
        colors.append(color_map.get(cond_name, '#999999'))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Δ Cell Accuracy')
    ax.set_title('Causal Effect of Ablation: SAE Features vs. Probe Directions')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Add significance annotations from t-tests
    tests = agg.get('statistical_tests', {})
    if tests:
        y_max = max(abs(m) + s for m, s in zip(means, stds)) * 1.2
        for test_name, test_info in tests.items():
            if test_info.get('significant_0.05'):
                ax.text(len(names) / 2, -y_max * 0.85,
                       f"{test_name}: p={test_info['p_value']:.3e} ***",
                       ha='center', fontsize=8, style='italic')

    fig.tight_layout()
    save_fig(fig, output_dir, 'causal_comparison')

    # Also: per-item bar chart for top features
    per_item = agg.get('per_item_means', {})
    if per_item:
        sae_items = [(k, v) for k, v in per_item.items() if k.startswith('sae_')]
        probe_items = [(k, v) for k, v in per_item.items() if k.startswith('probe_')]

        # Sort by effect size
        sae_items.sort(key=lambda x: x[1]['mean_delta_acc'])
        probe_items.sort(key=lambda x: x[1]['mean_delta_acc'])

        all_items = sae_items[:20] + probe_items  # top 20 SAE + all probes

        if all_items:
            fig2, ax2 = plt.subplots(figsize=(12, max(5, len(all_items) * 0.3)))
            y_pos = range(len(all_items))
            item_means = [v['mean_delta_acc'] for _, v in all_items]
            item_sems = [v['std_delta_acc'] / max(1, v['n']**0.5) for _, v in all_items]
            item_colors = ['#2196F3' if k.startswith('sae_') else '#FF9800'
                          for k, _ in all_items]
            item_labels = [k.replace('sae_', 'SAE ').replace('probe_', 'Probe: ')
                          for k, _ in all_items]

            ax2.barh(y_pos, item_means, xerr=item_sems, capsize=3,
                    color=item_colors, edgecolor='black', linewidth=0.3, height=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(item_labels, fontsize=7)
            ax2.set_xlabel('Δ Cell Accuracy')
            ax2.set_title('Per-Feature/Direction Causal Effect')
            ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax2.grid(axis='x', alpha=0.3)

            fig2.tight_layout()
            save_fig(fig2, output_dir, 'causal_per_feature')


# ═══════════════════════════════════════════════════════════════════════════
# 6. Geometry comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_geometry(output_dir, analysis_dir):
    """Cosine similarity matrix between SAE decoder columns and E8 probe directions."""
    logger.info("Plotting geometry comparison...")

    geo_path = os.path.join(analysis_dir, "geometry_analysis.json")
    if not os.path.exists(geo_path):
        logger.warning(f"Geometry analysis not found: {geo_path}")
        return

    with open(geo_path) as f:
        geo = json.load(f)

    # Pairwise cosine similarity heatmap (top-100 features)
    cos_path = os.path.join(analysis_dir, "pairwise_cosine_sim.pt")
    if os.path.exists(cos_path):
        cos_sim = torch.load(cos_path, map_location="cpu", weights_only=False)
        top_k = min(50, cos_sim.shape[0])
        subset = cos_sim[:top_k, :top_k].numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(subset, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'Pairwise Cosine Similarity of Top-{top_k} Decoder Columns')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        plt.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        save_fig(fig, output_dir, 'decoder_cosine_similarity')

    # Probe comparison stats
    probe_comp = geo.get('probe_comparison', {})
    if probe_comp:
        per_probe = probe_comp.get('per_probe_max_cos', {})
        if per_probe:
            fig, ax = plt.subplots(figsize=(8, max(4, len(per_probe) * 0.4)))
            names = list(per_probe.keys())
            vals = [per_probe[n] for n in names]
            y_pos = range(len(names))
            ax.barh(y_pos, vals, color='steelblue', edgecolor='black', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel('Max Cosine Similarity with any SAE Feature')
            ax.set_title(f'E8 Probe Directions vs SAE Decoder Columns\n'
                        f'MMCS(SAE→probes)={probe_comp.get("mmcs_sae_to_probes", 0):.3f}, '
                        f'MMCS(probes→SAE)={probe_comp.get("mmcs_probes_to_sae", 0):.3f}')
            ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='0.5 threshold')
            ax.legend()
            ax.grid(axis='x', alpha=0.3)
            fig.tight_layout()
            save_fig(fig, output_dir, 'probe_sae_alignment')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SAE E10: Plotting")
    parser.add_argument("--sae_path", type=str, default=None,
                        help="Path to trained SAE checkpoint (for reconstruction plot)")
    parser.add_argument("--activations_path", type=str,
                        default="results/sae_study/activations_zH.pt")
    parser.add_argument("--analysis_dir", type=str,
                        default="results/sae_study/feature_analysis")
    parser.add_argument("--causal_dir", type=str,
                        default="results/sae_study/causal_ablation")
    parser.add_argument("--sweep_csv", type=str,
                        default="results/sae_study/sweep_results.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/sae_study/plots")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Reconstruction quality
    if args.sae_path and os.path.exists(args.sae_path):
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

        if os.path.exists(args.activations_path):
            data = torch.load(args.activations_path, map_location="cpu", weights_only=False)
            z_H = data['z_H']
            plot_reconstruction_quality(sae, z_H, args.output_dir, device)
        else:
            logger.warning(f"Activations not found: {args.activations_path}")
    else:
        logger.info("Skipping reconstruction plot (no SAE path provided)")

    # 2. Sweep heatmaps
    if os.path.exists(args.sweep_csv):
        plot_sweep_heatmaps(args.output_dir, args.sweep_csv)
    else:
        logger.info("Skipping sweep heatmaps (no sweep CSV found)")

    # 3. Specialization matrix
    if os.path.exists(args.analysis_dir):
        plot_specialization_matrix(args.output_dir, args.analysis_dir)
    else:
        logger.info("Skipping specialization matrix (no analysis dir)")

    # 4. Step profiles
    if os.path.exists(args.analysis_dir):
        plot_step_profiles(args.output_dir, args.analysis_dir)
    else:
        logger.info("Skipping step profiles (no analysis dir)")

    # 5. Causal comparison
    if os.path.exists(args.causal_dir):
        plot_causal_comparison(args.output_dir, args.causal_dir)
    else:
        logger.info("Skipping causal comparison (no causal dir)")

    # 6. Geometry
    if os.path.exists(args.analysis_dir):
        plot_geometry(args.output_dir, args.analysis_dir)
    else:
        logger.info("Skipping geometry plot (no analysis dir)")

    logger.info(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
