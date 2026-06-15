"""scripts/run_all_controlled_experiments.py

Master runner for all controlled experiments.

Runs each experiment in sequence:
  1. Controlled Ablation (E1 revised)
  2. Controlled Freeze (E2b revised)
  3. Controlled Time-Shift (E5 revised)
  4. Controlled Directed Ablation (E9 revised) — skipped if no probe weights

Use --quick for a fast smoke test (N=20 for each experiment).

Usage:
    python scripts/run_all_controlled_experiments.py --quick
    python scripts/run_all_controlled_experiments.py --output_root results/controlled
"""

import os
import sys
import time
import argparse
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

from scripts.core.sudoku_sample import (
    default_puzzle_indices,
    load_puzzle_indices,
    save_puzzle_indices,
)


def run_script(script_name: str, extra_args: list, label: str) -> bool:
    """Run a Python script as a subprocess. Returns True on success."""
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, script_name)] + extra_args
    print(f"\n{'='*70}")
    print(f"  RUNNING: {label}")
    print(f"  CMD:     {' '.join(cmd)}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - t0
    status = "SUCCESS" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{status}] {label} — {elapsed:.1f}s")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all controlled experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test (N=20)")
    parser.add_argument("--output_root", type=str, default="results/controlled",
                        help="Root directory for all experiment outputs")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument(
        "--num_puzzles", type=int, default=500,
        help="Uniform Sudoku sample size forwarded to every controlled intervention",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--puzzle_indices", type=str, default=None,
        help="Existing shared Sudoku puzzle-index manifest to use",
    )
    parser.add_argument(
        "--save_puzzle_indices", type=str, default=None,
        help="Where to write the generated shared Sudoku puzzle-index manifest",
    )
    parser.add_argument(
        "--skip", type=str, default="",
        help="Comma-separated experiments to skip: ablation,freeze,timeshift,directed",
    )
    args = parser.parse_args()

    skip = set(args.skip.lower().split(",")) if args.skip else set()
    sample_n = 20 if args.quick else args.num_puzzles

    os.makedirs(args.output_root, exist_ok=True)
    if args.puzzle_indices:
        sample_path = args.puzzle_indices
        sample_indices = load_puzzle_indices(sample_path, limit=sample_n)
    else:
        sample_path = args.save_puzzle_indices or os.path.join(
            args.output_root,
            f"sudoku_puzzle_indices_seed{args.seed}_n{sample_n}.json",
        )
        sample_indices = default_puzzle_indices(sample_n)
        save_puzzle_indices(
            sample_path,
            sample_indices,
            metadata={
                "experiment": "controlled_suite",
                "seed": args.seed,
                "selection": "first_n_test_loader_indices",
                "num_puzzles": sample_n,
            },
        )
    print(f"Shared Sudoku sample: {sample_path} (N={len(sample_indices)})")

    # Common args forwarded to each script
    common = []
    if args.quick:
        common.append("--quick")
    if args.checkpoint:
        common.extend(["--checkpoint", args.checkpoint])
    if args.device != "auto":
        common.extend(["--device", args.device])
    common.extend([
        "--max_steps", str(args.max_steps),
        "--num_puzzles", str(sample_n),
        "--seed", str(args.seed),
        "--puzzle_indices", sample_path,
    ])

    results = {}
    t_total = time.time()

    # 1. Controlled Ablation
    if "ablation" not in skip:
        ok = run_script(
            "controlled_ablation.py",
            common + ["--output_dir", os.path.join(args.output_root, "ablation")],
            "E1-Revised: Controlled z_H/z_L Ablation",
        )
        results["ablation"] = ok
    else:
        print("  SKIPPING: ablation")

    # 2. Controlled Freeze
    if "freeze" not in skip:
        ok = run_script(
            "controlled_freeze.py",
            common + ["--output_dir", os.path.join(args.output_root, "freeze")],
            "E2b-Revised: Controlled Freeze z_H/z_L",
        )
        results["freeze"] = ok
    else:
        print("  SKIPPING: freeze")

    # 3. Controlled Time-Shift (fixed mode)
    if "timeshift" not in skip:
        ok = run_script(
            "controlled_time_shift.py",
            common + [
                "--mode", "fixed",
                "--output_dir", os.path.join(args.output_root, "time_shift"),
            ],
            "E5-Revised: Controlled Time-Shift (fixed mode)",
        )
        results["timeshift"] = ok
    else:
        print("  SKIPPING: timeshift")

    # 4. Controlled Directed Ablation
    if "directed" not in skip:
        probe_path = os.path.join(
            REPO_ROOT, "results", "probes", "e8_constraint_probes", "probe_weights.pt"
        )
        if os.path.exists(probe_path):
            ok = run_script(
                "controlled_directed_ablation.py",
                common + [
                    "--probe_weights", probe_path,
                    "--output_dir", os.path.join(args.output_root, "directed_ablation"),
                ],
                "E9-Revised: Controlled Directed Ablation",
            )
            results["directed"] = ok
        else:
            print(f"\n  SKIPPING directed ablation: probe weights not found at {probe_path}")
            print("  Run E8 first: python scripts/e8_constraint_probes.py")
            results["directed"] = None
    else:
        print("  SKIPPING: directed")

    # Summary
    total_elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE — {total_elapsed:.1f}s total")
    print(f"{'='*70}")
    for name, ok in results.items():
        if ok is None:
            status = "SKIPPED (missing deps)"
        elif ok:
            status = "SUCCESS"
        else:
            status = "FAILED"
        print(f"  {name:20s} {status}")
    print()


if __name__ == "__main__":
    main()
