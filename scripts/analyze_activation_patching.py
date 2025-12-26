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
        
        if steps is not None and accuracy_change is not None:
            step_key = str(steps) if steps else "all"
            if step_key not in by_step:
                by_step[step_key] = []
            by_step[step_key].append(accuracy_change)
    
    return by_step


def aggregate_by_puzzle_pair(results: List[Dict]) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Group results by puzzle pair"""
    by_pair = {}
    
    for result in results:
        source = result.get("config", {}).get("source_puzzle_idx")
        target = result.get("config", {}).get("target_puzzle_idx")
        level = result.get("config", {}).get("patch_level")
        accuracy_change = result.get("metrics", {}).get("accuracy_change")
        
        if source is not None and target is not None and accuracy_change is not None:
            pair_key = (source, target)
            if pair_key not in by_pair:
                by_pair[pair_key] = {}
            by_pair[pair_key][level] = accuracy_change
    
    return by_pair


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics"""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    
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
    
    # Create matrices for each level
    matrices = {level: {} for level in ["H", "L", "both"]}
    
    for (source, target), level_data in by_pair.items():
        for level, acc_change in level_data.items():
            if level in matrices:
                matrices[level][(source, target)] = acc_change
    
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
    
    for level, matrix_data in matrices.items():
        if not matrix_data:
            continue
        
        lines.extend([
            "-" * 80,
            f"Level: {level}",
            "-" * 80,
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
