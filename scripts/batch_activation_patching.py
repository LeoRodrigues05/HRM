"""
Batch Activation Patching Experiments

This script runs multiple activation patching experiments across different puzzle pairs,
patch levels, and steps to systematically analyze the causal role of activations.

Usage:
    python scripts/batch_activation_patching.py \
        --checkpoint <path_to_checkpoint> \
        --num_puzzles 10 \
        --output_dir results/activation_patching_batch
"""

import os
import sys
import argparse
import subprocess
import json
import yaml
from typing import List, Dict, Tuple
from itertools import combinations

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_single_experiment(
    checkpoint: str,
    source_idx: int,
    target_idx: int,
    patch_level: str,
    patch_steps: str,
    patch_positions: str,
    max_steps: int,
    output_dir: str,
    device: str
) -> Dict:
    """Run a single activation patching experiment"""
    
    cmd = [
        "python", "scripts/activation_patching.py",
        "--checkpoint", checkpoint,
        "--source_puzzle_idx", str(source_idx),
        "--target_puzzle_idx", str(target_idx),
        "--patch_level", patch_level,
        "--max_steps", str(max_steps),
        "--output_dir", output_dir,
        "--device", device
    ]
    
    if patch_steps != "all":
        cmd.extend(["--patch_steps", patch_steps])
    
    if patch_positions != "all":
        cmd.extend(["--patch_positions", patch_positions])
    
    print(f"\n{'='*60}")
    print(f"Running: source={source_idx}, target={target_idx}, level={patch_level}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running experiment:")
        print(result.stderr)
        return {"success": False, "error": result.stderr}
    
    # Parse results
    results_file = os.path.join(
        output_dir, 
        f"patch_s{source_idx}_t{target_idx}_{patch_level}.yaml"
    )
    
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = yaml.safe_load(f)
        return {"success": True, "results": results}
    else:
        return {"success": False, "error": "Results file not found"}


def main():
    parser = argparse.ArgumentParser(description="Batch Activation Patching")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--num_puzzles", type=int, default=10,
                        help="Number of puzzles to test")
    parser.add_argument("--patch_levels", type=str, default="H,L,both",
                        help="Comma-separated patch levels to test")
    parser.add_argument("--patch_steps_configs", type=str, default="all,0,1,2",
                        help="Comma-separated step configurations (use 'all' or specific steps)")
    parser.add_argument("--max_steps", type=int, default=8,
                        help="Maximum reasoning steps")
    parser.add_argument("--output_dir", type=str, default="results/activation_patching_batch",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--mode", type=str, default="pairwise", 
                        choices=["pairwise", "one_to_many"],
                        help="Experiment mode: pairwise tests all pairs, one_to_many tests first puzzle to all others")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse configurations
    patch_levels = args.patch_levels.split(",")
    patch_steps_configs = args.patch_steps_configs.split(",")
    
    # Generate puzzle pairs
    puzzle_indices = list(range(args.num_puzzles))
    
    if args.mode == "pairwise":
        # Test all pairs
        puzzle_pairs = list(combinations(puzzle_indices, 2))
    else:  # one_to_many
        # Test first puzzle against all others
        puzzle_pairs = [(0, i) for i in range(1, args.num_puzzles)]
    
    print(f"Testing {len(puzzle_pairs)} puzzle pairs")
    print(f"Patch levels: {patch_levels}")
    print(f"Step configs: {patch_steps_configs}")
    
    # Run experiments
    all_results = []
    total_experiments = len(puzzle_pairs) * len(patch_levels) * len(patch_steps_configs)
    completed = 0
    
    for source_idx, target_idx in puzzle_pairs:
        for patch_level in patch_levels:
            for patch_steps in patch_steps_configs:
                result = run_single_experiment(
                    checkpoint=args.checkpoint,
                    source_idx=source_idx,
                    target_idx=target_idx,
                    patch_level=patch_level,
                    patch_steps=patch_steps,
                    patch_positions="all",
                    max_steps=args.max_steps,
                    output_dir=args.output_dir,
                    device=args.device
                )
                
                result["config"] = {
                    "source_idx": source_idx,
                    "target_idx": target_idx,
                    "patch_level": patch_level,
                    "patch_steps": patch_steps
                }
                
                all_results.append(result)
                completed += 1
                
                print(f"Progress: {completed}/{total_experiments}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Batch experiments completed!")
    print(f"Summary saved to: {summary_path}")
    print('='*60)
    
    # Compute and display statistics
    successful = [r for r in all_results if r.get("success", False)]
    print(f"\nSuccessful experiments: {len(successful)}/{len(all_results)}")
    
    if successful:
        accuracy_changes = [
            r["results"]["metrics"]["accuracy_change"] 
            for r in successful
        ]
        
        print(f"\nAccuracy change statistics:")
        print(f"  Mean: {sum(accuracy_changes)/len(accuracy_changes):.4f}")
        print(f"  Min:  {min(accuracy_changes):.4f}")
        print(f"  Max:  {max(accuracy_changes):.4f}")
        
        # Group by patch level
        by_level = {}
        for r in successful:
            level = r["config"]["patch_level"]
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(r["results"]["metrics"]["accuracy_change"])
        
        print(f"\nBy patch level:")
        for level, changes in by_level.items():
            mean_change = sum(changes) / len(changes)
            print(f"  {level:6s}: {mean_change:+.4f} (n={len(changes)})")


if __name__ == "__main__":
    main()
