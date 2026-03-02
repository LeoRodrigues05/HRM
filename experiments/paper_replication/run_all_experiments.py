#!/usr/bin/env python3
"""Master script to run all paper replication experiments.

This script runs all experiments from the HRM paper:
1. Easy vs Hard puzzle analysis
2. Grokking dynamics
3. Step-wise dynamics
4. Hierarchical specialization probes
5. Activation patching ablations

Usage:
    python run_all_experiments.py --all
    python run_all_experiments.py --exp 1 3 5
    python run_all_experiments.py --exp easy_hard step_dynamics
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS = {
    "1": ("easy_hard", "exp1_easy_hard_analysis.py", "Easy vs Hard Puzzle Analysis"),
    "2": ("grokking", "exp2_grokking_analysis.py", "Grokking Dynamics Analysis"),
    "3": ("step_dynamics", "exp3_step_dynamics.py", "Step-wise Dynamics Analysis"),
    "4": ("specialization", "exp4_specialization_probes.py", "Hierarchical Specialization Probes"),
    "5": ("activation_patching", "exp5_activation_patching.py", "Activation Patching Ablations"),
}

# Aliases
ALIASES = {
    "easy_hard": "1",
    "grokking": "2", 
    "step_dynamics": "3",
    "specialization": "4",
    "activation_patching": "5",
}


def run_experiment(exp_id: str, args: argparse.Namespace) -> dict:
    """Run a single experiment and return results."""
    
    if exp_id in ALIASES:
        exp_id = ALIASES[exp_id]
    
    if exp_id not in EXPERIMENTS:
        print(f"Unknown experiment: {exp_id}")
        return {"status": "error", "message": f"Unknown experiment: {exp_id}"}
    
    short_name, script_name, description = EXPERIMENTS[exp_id]
    script_path = Path(__file__).parent / script_name
    
    print(f"\n{'='*60}")
    print(f"Running Experiment {exp_id}: {description}")
    print(f"{'='*60}")
    
    cmd = [
        "python", str(script_path),
        "--checkpoint", args.checkpoint,
        "--data_path", args.data_path,
        "--device", args.device,
    ]
    
    # Add experiment-specific args
    if exp_id == "2" and args.use_synthetic:
        cmd.append("--use_synthetic")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            print(result.stdout)
            return {
                "status": "success",
                "experiment": description,
                "duration_seconds": duration,
                "output": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
            }
        else:
            print(f"STDERR: {result.stderr}")
            return {
                "status": "error",
                "experiment": description,
                "duration_seconds": duration,
                "error": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
            }
            
    except Exception as e:
        return {
            "status": "error",
            "experiment": description,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run HRM Paper Replication Experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--exp", nargs="+", default=[], 
                       help="Specific experiments to run (1-5 or names)")
    parser.add_argument("--checkpoint", type=str,
                       default="Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt")
    parser.add_argument("--data_path", type=str, default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_synthetic", action="store_true",
                       help="Use synthetic data for grokking experiment")
    parser.add_argument("--output_dir", type=str,
                       default="experiments/paper_replication/results")
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    if args.all:
        experiments_to_run = list(EXPERIMENTS.keys())
    elif args.exp:
        experiments_to_run = args.exp
    else:
        print("Please specify --all or --exp <experiment_ids>")
        print("\nAvailable experiments:")
        for exp_id, (short_name, _, desc) in EXPERIMENTS.items():
            print(f"  {exp_id} ({short_name}): {desc}")
        return
    
    print("="*60)
    print("HRM PAPER REPLICATION EXPERIMENTS")
    print("="*60)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Data path: {args.data_path}")
    print(f"Device: {args.device}")
    print(f"Experiments to run: {experiments_to_run}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    results = {}
    start_time = datetime.now()
    
    for exp_id in experiments_to_run:
        result = run_experiment(exp_id, args)
        results[exp_id] = result
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    successful = 0
    failed = 0
    
    for exp_id, result in results.items():
        status_icon = "✓" if result["status"] == "success" else "✗"
        exp_name = EXPERIMENTS.get(exp_id, (exp_id, "", exp_id))[2]
        duration = result.get("duration_seconds", 0)
        print(f"  {status_icon} {exp_name}: {result['status']} ({duration:.1f}s)")
        
        if result["status"] == "success":
            successful += 1
        else:
            failed += 1
    
    print(f"\nTotal: {successful} successful, {failed} failed")
    print(f"Total duration: {total_duration:.1f}s")
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_duration_seconds": total_duration,
        "successful": successful,
        "failed": failed,
        "results": results,
        "config": {
            "checkpoint": args.checkpoint,
            "data_path": args.data_path,
            "device": args.device,
        }
    }
    
    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {summary_path}")
    
    # Print result locations
    print("\nResult locations:")
    for exp_id in experiments_to_run:
        if exp_id in ALIASES:
            exp_id = ALIASES[exp_id]
        if exp_id in EXPERIMENTS:
            short_name = EXPERIMENTS[exp_id][0]
            print(f"  {EXPERIMENTS[exp_id][2]}: {output_dir}/{short_name}/")


if __name__ == "__main__":
    main()
