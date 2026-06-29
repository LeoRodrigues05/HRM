---
name: hrm-maze-data-integrity
description: Data-integrity caveats in the HRM maze results to fix before submission
metadata:
  type: project
---

Issues found auditing the maze extension of the HRM interp project (see [[hrm-interp-project]]), verified against on-disk artifacts on 2026-06-19:

1. **Metric artifact (the headline)**: on the 900-cell maze, token accuracy is blind to path validity. z_H all-steps ablation = token −0.8% but valid_sg_path **−32.5%** [−39.5,−26.0] ([results/maze/hardened/ablation_controlled/zH/aggregate.json](results/maze/hardened/ablation_controlled/zH/aggregate.json), key `all_steps_maze_deltas`). Always report path-validity, not token acc, for maze.

2. **Patching is missing the readout step (step 15)**: the −45% z_L / +0% z_H numbers in [docs/MAZE_VS_SUDOKU_STORY.md](docs/MAZE_VS_SUDOKU_STORY.md) §5 ARE sourced — they are the `"all"` group (full-grid cross-puzzle patch) in [results/maze/hardened/patching_spatial/aggregate.json](results/maze/hardened/patching_spatial/aggregate.json) (groups also include on_path/off_path/near_S/near_G which are localized). The real gap: that run only patched steps **4,8,12** — NOT step 15. The mechanistic claim ("mid-step foreign z_H is rebuildable but the final/readout z_H is load-bearing") is therefore untested by patching; the story doc itself says "a direct test would patch at step 15." Re-run the full patch including steps 0/14/15 to show the z_H asymmetry directly. (Output: results/maze/hardened/patching_full_steps.)

3. **Broken maze regression probes**: in [results/maze/hardened/linear_probes/probe_summary.json](results/maze/hardened/linear_probes/probe_summary.json), regression targets have garbage R² (path_f1 ≈ −7368, path_length_ratio ≈ −300231). Drop them; keep classification probes (is_wall, on_optimal_path, etc.).

4. **Maze SAE recon inverts with size** (undertrained): best recon is d=1024 (3.1e-4); it *worsens* to 6.2e-3 at d=8192 ([results/maze/sae_study/](results/maze/sae_study/) *_log.json, 50 epochs). Sudoku's plateaus correctly. Retrain larger maze SAEs longer or cap the claim.

5. **Finding 3 NOW replicated on maze (2026-06-20)**: ran SAE causal ablation + directed-probe ablation (z_H,z_L). Results:
   - SAE causal ([results/maze/sae_study/causal_ablation/aggregate.json](results/maze/sae_study/causal_ablation/aggregate.json)): top-50 −3.0% ≈ random-50 −3.4% (p=0.18 n.s.); probe dirs −0.2% ≈ random +0.2% (p=0.33). Mirrors Sudoku → F3 holds on both tasks. Ordering full-z_H(−32.5%) ≫ SAE(−3%) ≫ probe(≈0).
   - Directed ablation ([results/maze/hardened/directed_ablation/](results/maze/hardened/directed_ablation/), N=500): z_L NO direction significant. z_H 2/9 significant — `on_optimal_path` (−2.8%, p=0.010, d=−0.16) and `is_free` (−2.6%, p=0.015). NUANCE vs Sudoku (where NO probe dir was causal): in maze the output-aligned direction (on-path) is weakly causal because it coincides with the task output; but is_wall (100% decodable) is NOT causal (p=0.23) → readability still ≠ causality (F2 holds via the is_wall counterexample).
6. **Issue #2 RESOLVED (2026-06-20)**: full cross-puzzle patch incl. step 15 ([results/maze/hardened/patching_full_steps/aggregate.json](results/maze/hardened/patching_full_steps/aggregate.json), N=100, all-group): z_H replaceable mid-traj (s4/8/12/14 ≈0) but **readout z_H load-bearing (s15 −26% valid_path)**; z_L catastrophic at ALL steps (−45% to −56%). Confirms z_H=output-buffer / z_L=substitution-fragile-working-buffer. The SAE used: d2048/λ0.01, recon 4.6e-4, 141 dead (93% alive).

6. **Paper vs disk drift**: workshop text says Sudoku z_H all-steps −20.5%, z_L −19.3%; controlled re-run says z_H −18.0%, z_L −20.8% ([results/controlled/ablation/](results/controlled/ablation/)). Sync the paper to the hardened numbers.

New Sudoku result that nuances F3: counterfactual constraint localization ([results/constraint_localization/](results/constraint_localization/)) — patching a cell's H-units flips that cell 69% vs 48% random (+21pp, Wilcoxon p=6e-5); H+L 76% vs 48%. So info is distributed across *channels* but localized across *tokens/cells*.
