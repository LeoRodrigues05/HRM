# Query 04: Publication Polish — Figures, Statistics, and Final Presentation

## Objective
Generate all publication-quality figures with consistent styling, proper error bars, effect sizes, and statistical tests. Consolidate results from all experiments (E1–E10 + baselines) into a unified result set suitable for a paper submission. This is the final step before writing.

---

## Prompt to Execute

Copy and paste the following into the AI assistant:

---

### QUERY START

I need to generate publication-quality figures and consolidate all results from my HRM mechanistic interpretability study in `/home/leo.rodrigues/HRM/`.

The experiments span multiple result directories. I need a single unified plotting and analysis pipeline that produces paper-ready output. Please create the following:

#### 1. Unified Results Loader (`scripts/paper_utils.py`)

Create a utility module that:
- Loads results from ALL experiment directories:
  - `results/controlled_ablation_zH/` — single-step z_H ablation (E1 revised)
  - `results/controlled_ablation_zL/` — z_L ablation complement
  - `results/controlled_freeze/` — freeze z_H vs z_L (E2b revised)
  - `results/controlled_time_shift/` — full transfer matrix (E5 revised)
  - `results/controlled_directed_ablation/` — directed ablation with stats (E9 revised)
  - `results/e8_constraint_probes/` — linear probe results (E8)
  - `results/sae_study/` — SAE feature analysis and causal ablation (E10)
  - `results/baseline_comparison/` — RNN + UT baseline results
  - `results/batch_ablation_zH/` — original E1 (for comparison/fallback)
  - `results/freeze_h/` — original E2b (for comparison/fallback)
  - `results/time_shift/` — original E5 (for comparison/fallback)
- Provides a clean API for accessing any result set
- Handles missing directories gracefully (fall back to original results if controlled versions not yet available)

#### 2. Publication Figure Style (`scripts/paper_style.py`)

Create a matplotlib style configuration:
- Font: Serif (Times New Roman or Computer Modern)
- Font sizes: title=14, axes=12, tick=10, legend=10
- Figure sizes: single column (3.5 × 2.5 in), double column (7 × 2.5 in), full page (7 × 5 in)
- Color palette: colorblind-friendly (use seaborn's "colorblind" or custom)
- Grid: light gray grid, white background
- Error bars: 95% CI using bootstrap (1000 resamples)
- DPI: 300 for PNG, vector for PDF
- Line widths: 1.5 for data, 0.75 for grid
- Marker sizes: 4pt for data points
- All text should use LaTeX rendering if available (`plt.rcParams['text.usetex'] = True`)

#### 3. Master Figure Script (`scripts/generate_paper_figures.py`)

Create a comprehensive script that generates ALL figures for the paper:

**Figure 1: Model Overview & Baseline Comparison** (double column)
- Panel A: HRM architecture diagram (schematic — can be placeholder box diagram)
- Panel B: Accuracy comparison — HRM vs Vanilla RNN vs Universal Transformer
  - Bar chart with 95% CI
  - Metrics: cell accuracy, puzzle accuracy
- Panel C: Hamming convergence across steps — all three models on same axes
  - Line plot with shaded CI regions

**Figure 2: z_H is the Planmaker** (double column, 3 panels)
- Panel A: Controlled z_H ablation — Δaccuracy vs ablated step (single curve with CI)
  - Overlay: z_L ablation for comparison (should be much smaller)
- Panel B: Freeze z_H vs freeze z_L — final accuracy vs freeze step
  - Two lines with CI, showing asymmetry
- Panel C: Time-shift transfer matrix — 16×16 heatmap
  - Clear upper/lower triangle pattern, with colorbar

**Figure 3: z_H Encodes Constraints** (double column, 3 panels)
- Panel A: Probe accuracy evolution across steps
  - Lines for each probe target (row, col, box violations, per_cell_correct)
  - From `results/e8_constraint_probes/sweep_results.csv`
- Panel B: Cosine similarity heatmap of probe directions
  - From `results/e8_constraint_probes/geometric_analysis.json`
- Panel C: PCA of constraint subspace
  - Show that constraint info lives in ~3-d subspace of 512-d z_H

**Figure 4: Readout vs Computation** (double column, 2–3 panels)
- Panel A: Directed ablation — Δaccuracy per probe direction vs random baseline
  - Bar chart with significance stars, error bars
  - From `results/controlled_directed_ablation/`
- Panel B: Specificity matrix — [ablated direction × violation type]
  - Heatmap with significance annotations
- Panel C (if SAE results available): Causal comparison
  - Four bars: SAE feature ablation, probe direction ablation, random SAE feature, random direction
  - With CI and significance tests

**Figure 5: SAE Feature Analysis** (double column, 3 panels) — if SAE results available
- Panel A: Feature specialization matrix (top-20 features × constraint targets)
- Panel B: Feature activation profiles across steps
- Panel C: SAE reconstruction quality (scatter: original vs reconstructed norm)

**Figure 6: Summary / Combined Impact** (single column)
- Consolidated bar chart showing key effect sizes from all experiments:
  - z_H ablation effect, z_L ablation effect, freeze asymmetry, patching effect, probe accuracy, SAE causal effect
  - All with CI, all on the same scale for visual comparison

**Supplementary Figures** (save separately):
- S1: Full per-step accuracy trajectories (baseline vs each ablation condition)
- S2: SAE hyperparameter sweep heatmap
- S3: Per-difficulty analysis (accuracy stratified by puzzle difficulty)
- S4: Norm and rotation analysis of z_H across steps (from diagnostic data)
- S5: Easy vs hard puzzle analysis (E1 contradiction with paper)

#### 4. Statistical Summary Table (`scripts/generate_stats_table.py`)

Create a script that:
- Computes and formats key statistics from all experiments into a LaTeX-ready table
- Columns: Experiment, N, Effect Size (Cohen's d), 95% CI, p-value, Key Finding
- Rows: one per experiment (E1-ext, E2b, E5-ext, E8, E9, E10, Baselines)
- Output: LaTeX table code in `results/paper_figures/stats_table.tex`
- Also output as CSV in `results/paper_figures/stats_table.csv`

#### 5. Results Narrative Script (`scripts/generate_results_summary.py`)

Create a script that:
- Loads all results and generates a structured text summary
- For each experiment: one-paragraph finding with exact numbers and significance levels
- Example output:
  ```
  E1 (z_H Ablation): Zeroing z_H at step 10 alone reduced cell accuracy by 
  X.X% (95% CI: [X.X, X.X], p < 0.001, Cohen's d = X.XX), compared to only 
  Y.Y% for z_L ablation at the same step (p = X.XX). This confirms z_H is 
  the primary carrier of reasoning state.
  ```
- Save to `results/paper_figures/results_narrative.txt`

#### 6. File Organization

All outputs should go to:
```
results/paper_figures/
├── fig1_baseline_comparison.png
├── fig1_baseline_comparison.pdf
├── fig2_zh_planmaker.png
├── fig2_zh_planmaker.pdf
├── fig3_constraint_encoding.png
├── fig3_constraint_encoding.pdf
├── fig4_readout_vs_computation.png
├── fig4_readout_vs_computation.pdf
├── fig5_sae_features.png          (if available)
├── fig5_sae_features.pdf          (if available)
├── fig6_summary.png
├── fig6_summary.pdf
├── supplementary/
│   ├── figS1_trajectories.png/pdf
│   ├── figS2_sae_sweep.png/pdf
│   ├── figS3_difficulty_analysis.png/pdf
│   ├── figS4_zh_dynamics.png/pdf
│   └── figS5_easy_vs_hard.png/pdf
├── stats_table.tex
├── stats_table.csv
└── results_narrative.txt
```

#### Important Implementation Notes

- **Graceful degradation**: If some result directories don't exist yet (e.g., SAE results, baseline results), the script should skip those figures and generate what it can. Print warnings for missing data.
- **Fallback**: If `results/controlled_*/` don't exist, fall back to `results/batch_ablation_zH/`, `results/freeze_h/`, etc. (original smaller-N results)
- **Consistent puzzle set**: Where possible, report which puzzles were used and verify consistency across experiments
- **Bootstrap CI**: Use `scipy.stats.bootstrap` or manual resampling (1000 iterations) for all confidence intervals
- **Statistical tests**: Use `scipy.stats.ttest_rel` for paired tests, `scipy.stats.mannwhitneyu` for non-paired. Report both p-values and effect sizes.
- **Color consistency**: Same color for z_H across all plots, same color for z_L, same for baselines
- **Base existing plots on**: `scripts/plot_e8_e9.py` and `scripts/plot_presentation.py` for current styling patterns

### QUERY END

---

## Expected Outcomes

1. Style configuration module
2. Results loader utility
3. 6 main figures + 5 supplementary figures, each as PNG + PDF
4. LaTeX stats table
5. Results narrative text

## Success Criteria

- All figures are publication-quality (300 DPI, LaTeX fonts, proper axes labels)
- All data points have 95% CI error bars
- Statistical significance reported where applicable
- Figures degrade gracefully when some results are unavailable
- Color palette is colorblind-friendly
- Figures match typical NeurIPS/ICLR formatting (single/double column widths)
