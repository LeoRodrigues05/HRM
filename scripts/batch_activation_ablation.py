"""scripts/batch_activation_ablation.py

Batch runner for the activation ablation experiment.

Iterates over multiple puzzles × ablation levels × step configurations
and collects summary statistics, analogous to batch_activation_patching.py.

Usage:
    python scripts/batch_activation_ablation.py \\
        --checkpoint Checkpoint_HRM_v2_Sudoku/best.pt \\
        --num_puzzles 10 \\
        --output_dir results/activation_ablation_batch
"""

import os
import sys
import json
import argparse
import subprocess
from typing import Dict, List, Any
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Batch activation ablation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_puzzles", type=int, default=10,
                        help="Number of test-set puzzles to ablate")
    parser.add_argument("--ablate_levels", type=str, default="H,L,both",
                        help="Comma-separated ablation levels")
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--output_dir", type=str,
                        default="results/activation_ablation_batch")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    levels = [l.strip() for l in args.ablate_levels.split(",")]
    max_steps = args.max_steps

    # Step configurations to sweep
    step_configs = {
        "all": None,                          # None means all steps
        "early": "0,1,2",
        "late": ",".join(str(s) for s in range(3, max_steps)),
        "first_only": "0",
        "last_only": str(max_steps - 1),
    }

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "activation_ablation.py")
    results: List[Dict[str, Any]] = []

    total = args.num_puzzles * len(levels) * len(step_configs)
    count = 0

    for puzzle_idx in range(args.num_puzzles):
        for level in levels:
            for config_name, steps_arg in step_configs.items():
                count += 1
                print(f"\n[{count}/{total}] puzzle={puzzle_idx} level={level} steps={config_name}")

                run_dir = os.path.join(
                    args.output_dir,
                    f"p{puzzle_idx}_{level}_{config_name}",
                )
                os.makedirs(run_dir, exist_ok=True)

                cmd = [
                    sys.executable, script,
                    "--checkpoint", args.checkpoint,
                    "--puzzle_idx", str(puzzle_idx),
                    "--ablate_level", level,
                    "--max_steps", str(max_steps),
                    "--num_runs", str(args.num_runs),
                    "--output_dir", run_dir,
                    "--device", args.device,
                    "--report_html", "report.html",
                ]
                if steps_arg is not None:
                    cmd.extend(["--ablate_steps", steps_arg])

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    print(f"  FAILED (rc={proc.returncode})")
                    print(proc.stderr[-500:] if proc.stderr else "(no stderr)")
                    continue

                # Read back YAML results
                yaml_glob = list(Path(run_dir).glob("ablation_*.yaml"))
                if not yaml_glob:
                    print("  WARNING: no YAML output found")
                    continue

                with open(yaml_glob[0], "r") as f:
                    run_result = yaml.safe_load(f)

                metrics = run_result.get("metrics", {})
                results.append({
                    "puzzle_idx": puzzle_idx,
                    "ablate_level": level,
                    "step_config": config_name,
                    "baseline_accuracy": metrics.get("baseline", {}).get("accuracy", 0),
                    "ablated_accuracy": metrics.get("ablated", {}).get("accuracy", 0),
                    "accuracy_change": metrics.get("accuracy_change", 0),
                })
                print(
                    f"  baseline={results[-1]['baseline_accuracy']:.4f} "
                    f"ablated={results[-1]['ablated_accuracy']:.4f} "
                    f"Δ={results[-1]['accuracy_change']:+.4f}"
                )

    # Summary statistics
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Summary saved to: {summary_path}")

    # Print grouped stats
    import numpy as np
    print(f"\n{'='*60}")
    print("Accuracy change by ablation level:")
    for level in levels:
        changes = [r["accuracy_change"] for r in results if r["ablate_level"] == level]
        if changes:
            print(f"  {level:5s}: mean={np.mean(changes):+.4f} std={np.std(changes):.4f} n={len(changes)}")

    print("\nAccuracy change by step config:")
    for cfg_name in step_configs:
        changes = [r["accuracy_change"] for r in results if r["step_config"] == cfg_name]
        if changes:
            print(f"  {cfg_name:12s}: mean={np.mean(changes):+.4f} std={np.std(changes):.4f} n={len(changes)}")


if __name__ == "__main__":
    main()
