#!/usr/bin/env python3
"""Generate publication-quality comparison figures: HRM vs baselines.

Reads *_eval.json files from the output directory (produced by evaluate_baselines.py)
and generates 4 figures:

  Figure 1: Accuracy vs. reasoning step
  Figure 2: Hamming distance convergence
  Figure 3: Activation patching effect (bar chart)
  Figure 4: Constraint satisfaction (violations vs. step)

Usage:
    python scripts/plotting/plot_baseline_comparison.py \\
        --results_dir results/Sudoku/baseline_comparison \\
        --output_dir results/Sudoku/baseline_comparison/figures
"""

import os
import sys
import json
import glob
import argparse

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__))))
from paper_style import apply_paper_style, COL_W, TEXT_W

# Single source of truth for all paper figures; keep the grid these figures use.
apply_paper_style(**{"axes.grid": True, "grid.alpha": 0.3})

MODEL_STYLES = {
    "HRM": {"color": "#2196F3", "marker": "o", "linestyle": "-"},
    "UT_best": {"color": "#4CAF50", "marker": "^", "linestyle": "-."},
    "RNN_best": {"color": "#F44336", "marker": "s", "linestyle": "--"},
    "PT_best": {"color": "#FF9800", "marker": "D", "linestyle": ":"},
    "SRNN_best": {"color": "#9C27B0", "marker": "v", "linestyle": ":"},
}

# Naming scheme carried over from the earlier version of this work. Note these
# are not just shorter: `RNN_best` is VanillaRNNModel, which applies a
# Transformer block recurrently (hence "Recurrent Trans", not "Vanilla RNN"),
# and `SRNN_best` is StandardRNNModel, which is an nn.GRU (hence "GRU").
MODEL_LABELS = {
    "HRM": "HRM",
    "UT_best": "Univ. Trans",
    "RNN_best": "Recurrent Trans",
    "PT_best": "Plain Trans",
    "SRNN_best": "GRU",
}

# For trajectory plots (all checkpoints of each family)
FAMILY_STYLES = {
    "UT": {"color": "#4CAF50", "marker": "^"},
    "RNN": {"color": "#F44336", "marker": "s"},
    "PT": {"color": "#FF9800", "marker": "D"},
    "SRNN": {"color": "#9C27B0", "marker": "v"},
}


def load_results(results_dir):
    """Load all *_eval.json files from the results directory."""
    models = {}
    for path in glob.glob(os.path.join(results_dir, "*_eval.json")):
        with open(path) as f:
            data = json.load(f)
        name = data.get("model_name", os.path.splitext(os.path.basename(path))[0])
        models[name] = data
    return models


# Legend/plot order. Matplotlib fills a multi-column legend COLUMN-major, so with
# ncol=3 this order lays the five models out as
#     row 1:  HRM             GRU            Plain Trans
#     row 2:  Recurrent Trans Univ. Trans
# which is the arrangement used across all three baseline figures.
MODEL_ORDER = ["HRM", "RNN_best", "SRNN_best", "UT_best", "PT_best"]
LEGEND_NCOL = 3


def get_primary_models(models):
    """Return HRM + best checkpoints for all baselines, in MODEL_ORDER."""
    return {name: models[name] for name in MODEL_ORDER if name in models}


def get_family_checkpoints(models, family_prefix):
    """Return all checkpoints for a model family sorted by training step."""
    family = {}
    for name, data in models.items():
        if name.startswith(family_prefix):
            # Extract step number from name like UT_step15624 or UT_best
            if "best" in name:
                # Get step from checkpoint path
                ckpt = data.get("checkpoint", "")
                step = int(ckpt.split("step_")[-1]) if "step_" in ckpt else 0
            elif "step" in name:
                step = int(name.split("step")[-1])
            else:
                step = 0
            family[step] = data
    return dict(sorted(family.items()))


def get_style(model_name):
    if model_name in MODEL_STYLES:
        return MODEL_STYLES[model_name]
    for key, style in MODEL_STYLES.items():
        if key.lower() in model_name.lower():
            return style
    return {"color": "#9E9E9E", "marker": "D", "linestyle": ":"}


def get_label(model_name):
    if model_name in MODEL_LABELS:
        return MODEL_LABELS[model_name]
    return model_name


def plot_accuracy_vs_step(models, output_dir):
    """Figure 1: Cell accuracy vs. reasoning step (primary models only)."""
    primary = get_primary_models(models)
    if not primary:
        print("  Skipping fig1 (no primary models)")
        return
    # Included at \columnwidth in 09appendix.tex and authored at COL_W, so
    # LaTeX applies a 1.0x scale and PAPER_RC sizes land as-is.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_W, 2.3))

    for name, data in primary.items():
        steps_data = data["per_step_metrics"]
        steps = sorted([int(s) for s in steps_data.keys()])
        style = get_style(name)
        label = get_label(name)

        # Cell accuracy
        cell_accs = [steps_data[str(s)]["cell_accuracy"]["mean"] for s in steps]
        ax1.plot(steps, cell_accs, label=label, markersize=3, markevery=3, **style)

        # Unknown-cell accuracy
        unk_accs = [steps_data[str(s)]["unknown_cell_accuracy"]["mean"] for s in steps]
        ax2.plot(steps, unk_accs, label=label, markersize=3, markevery=3, **style)

    ax1.set_xlabel("Reasoning step")
    ax1.set_ylabel("Cell accuracy")
    ax1.set_title("(a) Cell accuracy")
    ax1.set_ylim(0, 1.05)

    ax2.set_xlabel("Reasoning step")
    ax2.set_ylabel("Unknown-cell accuracy")
    ax2.set_title("(b) Unknown-cell accuracy")
    ax2.set_ylim(0, 1.05)

    # One shared legend below both panels (fits the narrow column better than
    # two in-axes legends); see MODEL_ORDER for the row layout ncol=3 produces.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=LEGEND_NCOL,
               frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=1.2,
               handletextpad=0.3, columnspacing=0.7)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"fig1_accuracy_vs_step.{ext}"))
    plt.close(fig)
    print("  Saved fig1_accuracy_vs_step")


def plot_hamming_convergence(models, output_dir):
    """Figure 2: Hamming distance to solution vs. step (primary models only)."""
    primary = get_primary_models(models)
    if not primary:
        print("  Skipping fig2 (no primary models)")
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.75))

    for name, data in primary.items():
        steps_data = data["per_step_metrics"]
        steps = sorted([int(s) for s in steps_data.keys()])
        style = get_style(name)
        label = get_label(name)

        hamming = [steps_data[str(s)]["hamming_distance"]["mean"] for s in steps]
        ax.plot(steps, hamming, label=label, markersize=3, markevery=3, **style)

    ax.set_xlabel("Reasoning step")
    ax.set_ylabel("Hamming distance to solution")
    ax.set_title("Hamming distance convergence")

    # Match the reference legend used by every baseline figure; see MODEL_ORDER.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=LEGEND_NCOL,
               frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=1.2,
               handletextpad=0.3, columnspacing=0.7)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"fig2_hamming_convergence.{ext}"))
    plt.close(fig)
    print("  Saved fig2_hamming_convergence")


def plot_activation_patching(models, output_dir):
    """Figure 3: Activation patching effect (bar chart)."""
    # Collect patching data
    model_names = []
    mean_deltas = []
    colors = []

    for name, data in sorted(models.items()):
        if "activation_patching" not in data:
            continue
        patching = data["activation_patching"]
        # Average across all patched steps
        all_deltas = [v["mean_delta_accuracy"] for v in patching.values()]
        model_names.append(get_label(name))
        mean_deltas.append(np.mean(all_deltas))
        colors.append(get_style(name)["color"])

    if not model_names:
        print("  Skipping fig3 (no patching data)")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(model_names))
    bars = ax.bar(x, mean_deltas, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("Mean Δ Accuracy (Patched − Baseline)")
    ax.set_title("Activation Patching: Causal Impact of Hidden State")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Add value labels on bars
    for bar, val in zip(bars, mean_deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:+.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"fig3_activation_patching.{ext}"))
    plt.close(fig)
    print("  Saved fig3_activation_patching")


def plot_constraint_violations(models, output_dir):
    """Figure 4: Constraint violations vs. step (primary models only)."""
    primary = get_primary_models(models)
    if not primary:
        print("  Skipping fig4 (no primary models)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    violation_keys = ["row_violations", "col_violations", "box_violations"]
    titles = ["Row Violations", "Column Violations", "Box Violations"]

    for ax, vkey, title in zip(axes, violation_keys, titles):
        for name, data in primary.items():
            steps_data = data["per_step_metrics"]
            steps = sorted([int(s) for s in steps_data.keys()])
            style = get_style(name)
            label = get_label(name)

            violations = [steps_data[str(s)][vkey]["mean"] for s in steps]
            ax.plot(steps, violations, label=label, markersize=4, **style)

        ax.set_xlabel("Reasoning Step")
        ax.set_title(title)
        ax.legend()

    axes[0].set_ylabel("Mean Violations per Puzzle")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"fig4_constraint_violations.{ext}"))
    plt.close(fig)
    print("  Saved fig4_constraint_violations")


def plot_training_trajectory(models, output_dir):
    """Figure 5: How baseline accuracy changes across training checkpoints.

    Shows that baselines diverge after ~20k steps while HRM converges.
    Uses the final-step cell accuracy from each checkpoint as summary.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for family, ax, title in [
        ("UT", ax1, "Univ. Trans"),
        ("RNN", ax2, "Recurrent Trans"),
        # PT and SRNN families can be added here if needed
    ]:
        ckpts = get_family_checkpoints(models, family)
        if not ckpts:
            ax.set_title(f"{title} (no data)")
            continue

        train_steps = []
        final_cell_accs = []
        final_puzzle_accs = []
        final_hamming = []

        for train_step, data in ckpts.items():
            steps_data = data["per_step_metrics"]
            last_step = str(max(int(s) for s in steps_data.keys()))
            train_steps.append(train_step / 1000)  # in k
            final_cell_accs.append(steps_data[last_step]["cell_accuracy"]["mean"])
            final_puzzle_accs.append(steps_data[last_step]["puzzle_accuracy"]["mean"])
            final_hamming.append(steps_data[last_step]["hamming_distance"]["mean"])

        style = FAMILY_STYLES.get(family, {"color": "gray", "marker": "o"})

        ax.plot(train_steps, final_cell_accs, marker=style["marker"],
                color=style["color"], label="Cell Accuracy", linestyle="-")
        ax.plot(train_steps, final_puzzle_accs, marker=style["marker"],
                color=style["color"], label="Puzzle Accuracy", linestyle="--", alpha=0.7)

        # Mark the best checkpoint
        best_idx = np.argmax(final_cell_accs)
        ax.axvline(train_steps[best_idx], color="gold", linestyle=":",
                   linewidth=2, label=f"Best @ {train_steps[best_idx]:.0f}k")

        # HRM reference line (if available)
        if "HRM" in models:
            hrm_data = models["HRM"]["per_step_metrics"]
            hrm_last = str(max(int(s) for s in hrm_data.keys()))
            hrm_acc = hrm_data[hrm_last]["cell_accuracy"]["mean"]
            ax.axhline(hrm_acc, color="#2196F3", linestyle="-",
                       linewidth=1.5, alpha=0.6, label=f"HRM ref ({hrm_acc:.2f})")

        ax.set_xlabel("Training Step (k)")
        ax.set_ylabel("Accuracy (at final reasoning step)")
        ax.set_title(f"{title}: Training Trajectory")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Baseline Training Divergence: Accuracy Degrades After ~20k Steps",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"fig5_training_trajectory.{ext}"))
    plt.close(fig)
    print("  Saved fig5_training_trajectory")


def main():
    parser = argparse.ArgumentParser(description="Plot baseline comparison figures")
    parser.add_argument("--results_dir", type=str, default="results/Sudoku/baseline_comparison")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for figures (default: results_dir/figures)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)

    models = load_results(args.results_dir)
    if not models:
        print(f"No *_eval.json files found in {args.results_dir}")
        return

    print(f"Found {len(models)} model(s): {list(models.keys())}")
    print("Generating figures...")

    # Main comparison: HRM vs best baselines (fig 1-4)
    plot_accuracy_vs_step(models, args.output_dir)
    plot_hamming_convergence(models, args.output_dir)
    plot_activation_patching(models, args.output_dir)
    plot_constraint_violations(models, args.output_dir)

    # Training trajectory: shows divergence across checkpoints (fig 5)
    plot_training_trajectory(models, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
