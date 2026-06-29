#!/usr/bin/env python3
"""H1 figures: localizability scorecard comparison across checkpoints.

Reads results/localizability/<tag>/scorecard.json for each tag and produces
a bar-chart comparison of the four scorecard metrics.

Usage:
    python scripts/analysis/plot_localizability.py
    python scripts/analysis/plot_localizability.py --tags hrm_1step,hrm_bptt,ut_1step,ut_bptt
"""
from __future__ import annotations
import os, sys, json, argparse
from typing import Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_TAGS = ["hrm_1step", "hrm_bptt", "ut_1step", "ut_bptt"]

TAG_STYLES = {
    "hrm_1step": {"color": "#E63946", "label": "HRM 1-step (current)"},
    "hrm_bptt":  {"color": "#A8DADC", "label": "HRM BPTT (airtight ctrl)"},
    "ut_1step":  {"color": "#457B9D", "label": "UT 1-step"},
    "ut_bptt":   {"color": "#2A9D8F", "label": "UT BPTT"},
}


def _load_scorecard(tag: str, results_root: str) -> Optional[dict]:
    path = os.path.join(results_root, "localizability", tag, "scorecard.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fig_scorecard_bars(data: Dict[str, dict], output_path: str):
    """4-panel bar chart: one panel per metric."""
    metrics = [
        ("probe_decodability", "Probe decodability\n(mean val acc)", 0.5, 1.0, True),
        ("probe_causal_gap", "Probe causal gap\n(probe Δ − random Δ)", None, None, False),
        ("min_causal_rank", "Min causal rank r*\n(smaller = more localized)", 0, None, False),
        ("subspace_linearity", "Subspace linearity R²\n(lower = more knee-shaped)", 0.0, 1.0, False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    for ax, (metric_key, ylabel, ymin, ymax, higher_is_better) in zip(axes, metrics):
        tags_present = [t for t in data if data[t] is not None]
        values, errs, colors, labels = [], [], [], []

        for tag in tags_present:
            sc = data[tag]["scorecard"]
            style = TAG_STYLES.get(tag, {"color": "#888888", "label": tag})

            if metric_key == "probe_decodability":
                v = sc["probe_decodability"]["mean_val_acc"]
                ci = sc["probe_decodability"].get("ci", {})
                e = (ci.get("ci_upper", v) - ci.get("ci_lower", v)) / 2
            elif metric_key == "probe_causal_gap":
                v = sc["probe_causal_gap"]["probe_causal_gap"]
                e = 0.0
            elif metric_key == "min_causal_rank":
                v = sc["min_causal_rank"]
                e = 0.0
            elif metric_key == "subspace_linearity":
                v = sc["subspace_linearity"]
                e = 0.0
            else:
                continue

            values.append(float(v) if not np.isnan(float(v)) else 0.0)
            errs.append(float(e))
            colors.append(style["color"])
            labels.append(style["label"])

        if not values:
            ax.set_title(ylabel, fontsize=9)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, yerr=errs, capsize=4, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(ylabel, fontsize=9)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        ax.grid(True, axis="y", alpha=0.3)

        # H1 hypothesis direction arrow
        if metric_key == "probe_decodability":
            ax.annotate("H1: similar\nacross regimes", xy=(0.5, 0.02),
                        xycoords="axes fraction", ha="center", fontsize=7, color="gray")
        elif metric_key == "probe_causal_gap":
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.annotate("H1: BPTT → more negative", xy=(0.5, 0.98),
                        xycoords="axes fraction", ha="center", va="top", fontsize=7, color="#2A9D8F")
        elif metric_key == "min_causal_rank":
            ax.annotate("H1: BPTT → smaller r*", xy=(0.5, 0.98),
                        xycoords="axes fraction", ha="center", va="top", fontsize=7, color="#2A9D8F")
        elif metric_key == "subspace_linearity":
            ax.annotate("H1: BPTT → lower R²", xy=(0.5, 0.98),
                        xycoords="axes fraction", ha="center", va="top", fontsize=7, color="#2A9D8F")

    fig.suptitle("H1 Localizability Scorecard — Training Regime Comparison", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def fig_delta_table(data: Dict[str, dict], output_path: str):
    """Print BPTT−1step deltas for each metric as a text summary."""
    lines = ["H1 Scorecard Deltas (BPTT − 1step)\n", "=" * 40]
    for prefix in [("hrm", "HRM"), ("ut", "UT")]:
        tag_1step = f"{prefix[0]}_1step"
        tag_bptt = f"{prefix[0]}_bptt"
        if tag_1step not in data or data[tag_1step] is None or tag_bptt not in data or data[tag_bptt] is None:
            lines.append(f"{prefix[1]}: missing one or both scorecards")
            continue
        sc1 = data[tag_1step]["scorecard"]
        sc2 = data[tag_bptt]["scorecard"]
        lines.append(f"\n{prefix[1]} (BPTT − 1step):")
        lines.append(f"  probe_decodability  : {sc2['probe_decodability']['mean_val_acc'] - sc1['probe_decodability']['mean_val_acc']:+.4f}")
        lines.append(f"  probe_causal_gap    : {sc2['probe_causal_gap']['probe_causal_gap'] - sc1['probe_causal_gap']['probe_causal_gap']:+.4f}")
        lines.append(f"  min_causal_rank     : {sc2['min_causal_rank'] - sc1['min_causal_rank']:+d}")
        lines.append(f"  subspace_linearity  : {sc2['subspace_linearity'] - sc1['subspace_linearity']:+.4f}")

    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    print(text)
    print(f"\n  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="H1 localizability figures")
    parser.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    parser.add_argument("--results_root", default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(args.results_root, "reports", "localizability_figures")
    os.makedirs(outdir, exist_ok=True)

    tags = [t.strip() for t in args.tags.split(",")]
    data = {}
    for tag in tags:
        sc = _load_scorecard(tag, args.results_root)
        if sc is None:
            print(f"  {tag}: no scorecard found (skipping)")
        else:
            print(f"  {tag}: loaded")
        data[tag] = sc

    if not any(v is not None for v in data.values()):
        print("No scorecards found. Run localizability_scorecard.py / baseline_localizability.py first.")
        return

    fig_scorecard_bars(data, os.path.join(outdir, "fig_localizability_scorecard.pdf"))
    fig_scorecard_bars(data, os.path.join(outdir, "fig_localizability_scorecard.png"))
    fig_delta_table(data, os.path.join(outdir, "localizability_delta_summary.txt"))

    print(f"\nAll H1 figures written to {outdir}")


if __name__ == "__main__":
    main()
