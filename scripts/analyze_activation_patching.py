"""
Visualization and Analysis Tools for Activation Patching Results

This script provides utilities to analyze and visualize activation patching results,
including comparing accuracy changes across different configurations and identifying
which steps/layers are most important.

Usage:
    python scripts/analyze_activation_patching.py \
        --results_dir results/activation_patching_batch \
        --output_dir results/activation_patching_analysis
"""

import os
import argparse
import json
import yaml
from typing import Dict, List, Tuple
import numpy as np


def _normalize_steps_key(steps) -> str:
    """Normalize patch_steps from result YAML into a stable string key.

    Expected encodings from different runners:
    - None / null -> "all"
    - [] -> "all" (defensive)
    - int -> "[<int>]"
    - list[int] -> "[0, 1, 2]"
    - str like "all" or "0,1" -> returns "all" or "[0, 1]" when possible
    """
    if steps is None:
        return "all"
    if isinstance(steps, str):
        s = steps.strip()
        if s == "" or s.lower() == "all":
            return "all"
        # Try comma-separated ints
        try:
            vals = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
            return str(sorted(set(vals))) if vals else "all"
        except Exception:
            return s
    if isinstance(steps, (int, np.integer)):
        return str([int(steps)])
    if isinstance(steps, list):
        if len(steps) == 0:
            return "all"
        try:
            vals = [int(x) for x in steps]
            return str(sorted(set(vals)))
        except Exception:
            return str(steps)
    return str(steps)


def load_all_results(results_dir: str) -> List[Dict]:
    """Load all YAML result files from a directory"""
    results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.yaml') and filename.startswith('patch_'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    result = yaml.safe_load(f)
                    results.append(result)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
    
    return results


def aggregate_by_patch_level(results: List[Dict]) -> Dict[str, List[float]]:
    """Group accuracy changes by patch level (H, L, both)"""
    by_level = {"H": [], "L": [], "both": []}
    
    for result in results:
        level = result.get("config", {}).get("patch_level")
        accuracy_change = result.get("metrics", {}).get("accuracy_change")
        
        if level in by_level and accuracy_change is not None:
            by_level[level].append(accuracy_change)
    
    return by_level


def aggregate_by_step(results: List[Dict]) -> Dict[str, List[float]]:
    """Group accuracy changes by patching step configuration"""
    by_step = {}
    
    for result in results:
        steps = result.get("config", {}).get("patch_steps")
        accuracy_change = result.get("metrics", {}).get("accuracy_change")
        
        if accuracy_change is None:
            continue

        step_key = _normalize_steps_key(steps)
        by_step.setdefault(step_key, []).append(accuracy_change)
    
    return by_step


def aggregate_by_puzzle_pair(results: List[Dict]) -> Dict[Tuple[int, int], Dict[str, Dict[str, List[float]]]]:
    """Group results by puzzle pair.

    Returns:
        by_pair[(source, target)][level][step_key] -> list[accuracy_change]

    This avoids silently overwriting results when multiple step configurations exist.
    """
    by_pair: Dict[Tuple[int, int], Dict[str, Dict[str, List[float]]]] = {}

    for result in results:
        cfg = result.get("config", {})
        source = cfg.get("source_puzzle_idx")
        target = cfg.get("target_puzzle_idx")
        level = cfg.get("patch_level")
        step_key = _normalize_steps_key(cfg.get("patch_steps"))
        accuracy_change = result.get("metrics", {}).get("accuracy_change")

        if source is None or target is None or level is None or accuracy_change is None:
            continue
        pair_key = (int(source), int(target))
        by_pair.setdefault(pair_key, {}).setdefault(str(level), {}).setdefault(step_key, []).append(float(accuracy_change))

    return by_pair


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics"""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "count": 0}
    
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "count": len(values)
    }


def create_analysis_report(results: List[Dict], output_path: str):
    """Generate a comprehensive analysis report"""
    
    report_lines = [
        "=" * 80,
        "ACTIVATION PATCHING ANALYSIS REPORT",
        "=" * 80,
        "",
        f"Total experiments: {len(results)}",
        ""
    ]
    
    # Analysis by patch level
    report_lines.extend([
        "-" * 80,
        "ANALYSIS BY PATCH LEVEL",
        "-" * 80,
        ""
    ])
    
    by_level = aggregate_by_patch_level(results)
    for level, changes in sorted(by_level.items()):
        stats = compute_statistics(changes)
        report_lines.extend([
            f"Level: {level}",
            f"  Count:      {stats['count']}",
            f"  Mean:       {stats['mean']:+.6f}",
            f"  Std:        {stats['std']:.6f}",
            f"  Median:     {stats['median']:+.6f}",
            f"  Min:        {stats['min']:+.6f}",
            f"  Max:        {stats['max']:+.6f}",
            ""
        ])
    
    # Analysis by step configuration
    report_lines.extend([
        "-" * 80,
        "ANALYSIS BY STEP CONFIGURATION",
        "-" * 80,
        ""
    ])
    
    by_step = aggregate_by_step(results)
    for step_config, changes in sorted(by_step.items()):
        stats = compute_statistics(changes)
        report_lines.extend([
            f"Steps: {step_config}",
            f"  Count:      {stats['count']}",
            f"  Mean:       {stats['mean']:+.6f}",
            f"  Std:        {stats['std']:.6f}",
            f"  Median:     {stats['median']:+.6f}",
            ""
        ])

    # Optional: stepwise prediction-change summaries (only present in newer runner outputs)
    stepwise_available = any(isinstance(r.get("stepwise_metrics"), dict) for r in results)
    if stepwise_available:
        report_lines.extend([
            "-" * 80,
            "STEPWISE PREDICTION CHANGE (if available)",
            "-" * 80,
            "",
            "Each experiment may include per-step counts of positions where patched preds differ from baseline preds.",
            "",
        ])

        # Aggregate mean changed_count over patched steps, grouped by patch_level
        by_level_changed = {"H": [], "L": [], "both": []}
        by_level_deltaacc = {"H": [], "L": [], "both": []}

        for r in results:
            level = r.get("config", {}).get("patch_level")
            stepwise = r.get("stepwise_metrics")
            if level not in by_level_changed or not isinstance(stepwise, dict) or not stepwise:
                continue

            # Determine which steps to treat as "patched" for summary
            cfg_steps = r.get("config", {}).get("patch_steps")
            step_key = _normalize_steps_key(cfg_steps)
            if step_key == "all":
                steps_to_use = list(stepwise.keys())
            else:
                try:
                    # step_key is like "[0, 1]"
                    steps_list = json.loads(step_key.replace("'", "\"") ) if step_key.startswith("[") else []
                    steps_to_use = [str(int(s)) for s in steps_list if str(int(s)) in stepwise]
                except Exception:
                    steps_to_use = list(stepwise.keys())

            changed_vals = []
            deltaacc_vals = []
            for sk in steps_to_use:
                m = stepwise.get(sk, {})
                if isinstance(m, dict):
                    if "changed_count" in m:
                        changed_vals.append(float(m["changed_count"]))
                    if "delta_accuracy" in m:
                        deltaacc_vals.append(float(m["delta_accuracy"]))

            if changed_vals:
                by_level_changed[level].append(float(np.mean(changed_vals)))
            if deltaacc_vals:
                by_level_deltaacc[level].append(float(np.mean(deltaacc_vals)))

        for level in ["H", "L", "both"]:
            cstats = compute_statistics(by_level_changed[level])
            dstats = compute_statistics(by_level_deltaacc[level])
            report_lines.extend([
                f"Level: {level}",
                f"  Mean(changed_count over patched steps): {cstats['mean']:.3f} (n={cstats['count']})",
                f"  Mean(delta_accuracy over patched steps): {dstats['mean']:+.6f} (n={dstats['count']})",
                "",
            ])
    
    # Find most impactful configurations
    report_lines.extend([
        "-" * 80,
        "TOP 10 MOST DISRUPTIVE PATCHES (Largest negative impact)",
        "-" * 80,
        ""
    ])
    
    sorted_results = sorted(
        results, 
        key=lambda r: r.get("metrics", {}).get("accuracy_change", 0)
    )
    
    for i, result in enumerate(sorted_results[:10], 1):
        config = result.get("config", {})
        metrics = result.get("metrics", {})
        report_lines.append(
            f"{i:2d}. Source {config.get('source_puzzle_idx', '?'):2d} → "
            f"Target {config.get('target_puzzle_idx', '?'):2d} | "
            f"Level: {config.get('patch_level', '?'):4s} | "
            f"Steps: {str(config.get('patch_steps', '?')):10s} | "
            f"Δ Acc: {metrics.get('accuracy_change', 0):+.4f}"
        )
    
    report_lines.extend(["", ""])
    
    # Find least impactful configurations
    report_lines.extend([
        "-" * 80,
        "TOP 10 LEAST DISRUPTIVE PATCHES (Smallest impact)",
        "-" * 80,
        ""
    ])
    
    sorted_by_abs = sorted(
        results,
        key=lambda r: abs(r.get("metrics", {}).get("accuracy_change", 0))
    )
    
    for i, result in enumerate(sorted_by_abs[:10], 1):
        config = result.get("config", {})
        metrics = result.get("metrics", {})
        report_lines.append(
            f"{i:2d}. Source {config.get('source_puzzle_idx', '?'):2d} → "
            f"Target {config.get('target_puzzle_idx', '?'):2d} | "
            f"Level: {config.get('patch_level', '?'):4s} | "
            f"Steps: {str(config.get('patch_steps', '?')):10s} | "
            f"Δ Acc: {metrics.get('accuracy_change', 0):+.4f}"
        )
    
    report_lines.extend(["", "", "=" * 80])
    
    # Write report
    report_text = "\n".join(report_lines)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    return report_text


def create_matrix_visualization(results: List[Dict], output_path: str):
    """Create a text-based matrix visualization of accuracy changes"""
    
    by_pair = aggregate_by_puzzle_pair(results)
    
    if not by_pair:
        return
    
    # Get all unique puzzle indices
    all_indices = set()
    for source, target in by_pair.keys():
        all_indices.add(source)
        all_indices.add(target)
    
    indices = sorted(all_indices)
    
    # Collect all step configs present
    step_keys = set()
    for _pair, level_data in by_pair.items():
        for _level, step_data in level_data.items():
            step_keys.update(step_data.keys())
    step_keys = sorted(step_keys)

    def mean_or_none(vals: List[float]):
        return float(np.mean(vals)) if vals else None
    
    # Write visualization
    lines = [
        "=" * 80,
        "ACCURACY CHANGE MATRIX (Source → Target)",
        "=" * 80,
        "",
        "Legend:",
        "  Each cell shows accuracy change when patching from source (row) to target (col)",
        "  Negative values (red): Patching disrupted performance",
        "  Near-zero values: Minimal impact",
        "  Positive values: Patching improved performance (rare)",
        ""
    ]
    
    # Matrices for each step_key (plus an aggregate over all step_keys)
    for step_key in (step_keys + ["__ALL__"]):
        step_label = "all_step_configs" if step_key == "__ALL__" else f"patch_steps={step_key}"
        lines.extend([
            "-" * 80,
            f"STEP CONFIG: {step_label}",
            "-" * 80,
            ""
        ])

        for level in ["H", "L", "both"]:
            # Build a matrix of (source,target)->mean(acc_change)
            matrix_data: Dict[Tuple[int, int], float] = {}
            for (source, target), level_data in by_pair.items():
                if level not in level_data:
                    continue
                step_data = level_data[level]
                if step_key == "__ALL__":
                    vals = [v for vv in step_data.values() for v in vv]
                else:
                    vals = step_data.get(step_key, [])
                m = mean_or_none(vals)
                if m is not None:
                    matrix_data[(source, target)] = m

            if not matrix_data:
                continue

            lines.extend([
                f"Level: {level}",
                ""
            ])

            # Header
            header = "      " + "".join(f"{idx:6d}" for idx in indices)
            lines.append(header)
            lines.append("      " + "-" * (6 * len(indices)))

            # Rows
            for source in indices:
                row = f"{source:4d} |"
                for target in indices:
                    if source == target:
                        row += "  --  "
                    elif (source, target) in matrix_data:
                        val = matrix_data[(source, target)]
                        row += f"{val:+6.3f}"
                    else:
                        row += "   .  "
                lines.append(row)

            lines.append("")
    
    # Write to file
    matrix_text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(matrix_text)
    
    return matrix_text


def main():
    parser = argparse.ArgumentParser(description="Analyze activation patching results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing result YAML files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save analysis (defaults to results_dir/analysis)")
    parser.add_argument("--print_report", action="store_true",
                        help="Print report to console")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "analysis")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading results...")
    results = load_all_results(args.results_dir)
    print(f"Loaded {len(results)} experiments")
    
    if not results:
        print("No results found!")
        return
    
    # Generate analysis report
    print("\nGenerating analysis report...")
    report_path = os.path.join(args.output_dir, "analysis_report.txt")
    report_text = create_analysis_report(results, report_path)
    print(f"Saved to: {report_path}")
    
    if args.print_report:
        print("\n" + report_text)
    
    # Generate matrix visualization
    print("\nGenerating matrix visualization...")
    matrix_path = os.path.join(args.output_dir, "accuracy_matrix.txt")
    matrix_text = create_matrix_visualization(results, matrix_path)
    print(f"Saved to: {matrix_path}")
    
    # Save summary JSON
    print("\nGenerating summary statistics...")
    summary = {
        "total_experiments": len(results),
        "by_level": {
            level: compute_statistics(changes)
            for level, changes in aggregate_by_patch_level(results).items()
        },
        "by_step": {
            step: compute_statistics(changes)
            for step, changes in aggregate_by_step(results).items()
        }
    }
    
    summary_path = os.path.join(args.output_dir, "summary_statistics.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
