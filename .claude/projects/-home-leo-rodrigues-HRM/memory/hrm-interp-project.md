---
name: hrm-interp-project
description: HRM mechanistic-interpretability project — scope, paper status, task contrast
metadata:
  type: project
---

Mechanistic-interpretability study of the Hierarchical Reasoning Model (HRM; H/L recurrent states, 27M params, halt_max_steps=16, hidden=512). Workshop paper (Sudoku-only, 3 findings) accepted at ICML Mech Interp Workshop 2026 ([docs/paper/](docs/paper/), PDF in [docs/](docs/)). Now extending to a **conference/AAAI-track** submission (roadmap: [docs/plan_aaai.md](docs/plan_aaai.md); target ~2026-07-06) by adding the **Maze 30×30-hard** task on the *same checkpoint* and hardening every number with seeds + bootstrap CIs.

The 3 workshop findings: (F1) HRM iteratively refines a solution in z_H, z_L is a working buffer; (F2) probe readout ≠ causal use; (F3) computation is deeply distributed (SAE>probe>random but top-50≈random-50).

Canonical maze results live under [results/maze/hardened/](results/maze/hardened/); Sudoku hardened under [results/controlled/](results/controlled/). The author's own narrative contrast is [docs/MAZE_VS_SUDOKU_STORY.md](docs/MAZE_VS_SUDOKU_STORY.md). Key cross-task result: same architecture, **task-dependent depth** — maze solves in ~1 ACT step (z_H is an output buffer, only final step causal), Sudoku iterates all 16 (z_H accumulates, every step causal). See [[hrm-maze-data-integrity]] for verification caveats.
