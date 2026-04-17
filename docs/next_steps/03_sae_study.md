# Query 03: Sparse Autoencoder (SAE) Study on HRM

## Objective
Train a sparse autoencoder on z_H activations to discover interpretable, sparse features that the model actually uses for computation (vs. the readout-only features found by linear probes in E9). Then causally validate the SAE features by ablation and compare their specificity to E9's probe directions. This is the novel methodological contribution — applying SAE-based mechanistic interpretability to a recurrent reasoning model (HRM) for the first time.

---

## Prompt to Execute

Copy and paste the following into the AI assistant:

---

### QUERY START

I need to implement a Sparse Autoencoder (SAE) study on the HRM model's z_H activations in `/home/leo.rodrigues/HRM/`. This is experiment E10 — the follow-up to E9's finding that linear probe directions are "readout features, not computational features." The hypothesis is that SAE can find overcomplete sparse features that ARE causally relevant.

Background:
- The HRM model has a high-level hidden state z_H ∈ ℝ^(81×512) that is updated at each of 16 deep supervision steps
- E8 showed that linear probes can decode constraint violations from z_H at ~90% accuracy  
- E9 showed that projecting out these probe directions has <1% causal effect — they're readout, not computation features
- The SAE should find sparse, overcomplete features in z_H that may correspond to the model's actual computational units

Please implement the full SAE pipeline:

#### 1. Activation Collection Script (`scripts/sae_collect_activations.py`)

Create a script that:
- Loads the trained HRM checkpoint (auto-detect from `checkpoints/sapientinc-sudoku-extreme`)
- Runs forward passes on N puzzles (default N=1000, configurable via `--n_puzzles`)
- At each of the 16 deep supervision steps, records z_H after the H-level update
  - Use the existing `probe_recorder` mechanism from `models/hrm/hrm_act_v1.py` — the forward pass already supports `probe_recorder` argument
- Saves activations to disk:
  - File: `results/sae_study/activations_zH.pt`
  - Shape: `[N_puzzles, 16_steps, 81_cells, 512_dim]` — or store as a flat tensor `[N_puzzles * 16 * 81, 512]` for SAE training
  - Also save metadata: puzzle IDs, labels, givens, per-step accuracy
- Support `--also_collect_zL` flag to collect z_L activations (for later comparison)
- Memory management: collect in batches, save incrementally if needed (81 cells × 16 steps × 1000 puzzles × 512 dims × 4 bytes ≈ 2.6 GB — manageable)

Base the model loading on the pattern in `scripts/e8_constraint_probes.py` or `scripts/batch_ablation_1k.py`.(NOTE: THIS IS NOT NECESSARY BUT CHECK IF IT FITS THIS USE CASE)

#### 2. SAE Module (`models/sae.py`)

Create a clean, minimal sparse autoencoder implementation:

```python
class SparseAutoencoder(nn.Module):
    """Single-layer SAE with L1 regularization for mechanistic interpretability.
    
    Architecture:
        encoder: Linear(input_dim, dict_size) + ReLU
        decoder: Linear(dict_size, input_dim)  (unit-norm columns)
    
    Loss: MSE_reconstruction + l1_coeff * L1(hidden_activations)
    """
```

Requirements:
- `input_dim = 512` (z_H hidden size)
- `dict_size` configurable (default 2048 = 4× expansion)
- Encoder: `Linear(512, dict_size)` → `ReLU` → hidden activations `h ∈ ℝ^dict_size`
- Decoder: `Linear(dict_size, 512)` with **unit-norm columns** (normalize decoder weight columns on each forward pass)
- Loss: `reconstruction_loss + l1_coeff * mean(|h|)`
  - `reconstruction_loss = MSE(decoder(h), input)`
  - `l1_coeff` is a hyperparameter
- Track metrics: reconstruction loss, L1 loss, number of dead features (features that haven't activated in last 1000 batches), mean activations per feature, feature sparsity (fraction of inputs where feature > 0)
- Weight initialization: Kaiming for encoder, orthogonal for decoder (then normalize columns) -- check if this is accurate, also explain what this is in your output
- Support `encode(x)` returning hidden activations only (for analysis)
- Support `decode(h)` returning reconstruction only
- Support `forward(x)` returning `(reconstruction, hidden_activations, loss_dict)`

#### 3. SAE Training Script (`scripts/sae_train.py`)

Create a training script that:
- Loads pre-collected activations from `results/sae_study/activations_zH.pt`
- Trains the SAE with configurable hyperparameters:
  - `--dict_size`: dictionary size (default 2048; sweep: 1024, 2048, 4096)
  - `--l1_coeff`: L1 penalty coefficient (default 0.01; sweep: 0.001, 0.003, 0.01, 0.03, 0.1)
  - `--lr`: learning rate (default 3e-4)
  - `--batch_size`: default 4096
  - `--epochs`: default 50
  - `--steps_filter`: if set, only train on activations from specific steps (e.g., "8,12,15" for late steps only)
- Training loop:
  - Shuffle activations, train in mini-batches
  - Adam optimizer (not AdamAtan2 — standard SAE training)
  - Log every 100 steps: reconstruction loss, L1 loss, alive features count, mean sparsity
  - Save checkpoint every 5 epochs
  - Re-initialize dead features periodically (every 5 epochs): if a feature hasn't activated in 1000+ batches, re-init its encoder weights near the input with highest reconstruction error
- Save:
  - Trained SAE: `results/sae_study/sae_d{dict_size}_l1{l1_coeff}.pt`
  - Training log: `results/sae_study/sae_d{dict_size}_l1{l1_coeff}_log.json`
  - Feature statistics: `results/sae_study/sae_d{dict_size}_l1{l1_coeff}_features.json`

#### 4. SAE Hyperparameter Sweep Script (`scripts/sae_sweep.py`)

Create a sweep script that:
- Trains SAEs across a grid of (dict_size, l1_coeff) values
- For each configuration, records: final reconstruction loss, alive features count, mean sparsity, L0 (mean number of active features per input)
- Finds the "Goldilocks zone" (see Sharkey et al. 2022) — the l1_coeff range where reconstruction is good AND features are sparse
- Outputs a summary table comparing all configurations
- Save: `results/sae_study/sweep_results.csv`

#### 5. SAE Feature Analysis Script (`scripts/sae_analyze_features.py`)

Create an analysis script that:
- Loads a trained SAE and the original activations
- Performs feature-level analysis:

**A. Feature activation patterns**:
- For each SAE feature, compute which puzzles/cells/steps activate it most strongly
- Cluster features by activation pattern
- Identify "constraint features" — features that activate primarily when row/col/box violations are present (correlate feature activations with the constraint labels from E8)

**B. Feature specialization**:
- For each SAE feature, compute its correlation with each constraint target (row_violation, col_violation, box_violation, per_cell_correct, is_given) -- there can also be other contraints (in sudoku), can test with multiple other options.
- Create a [num_features × num_targets] correlation matrix
- Identify highly specialized features (corr > 0.5 with one target, < 0.1 with others)

**C. Feature geometry**:
- Compute pairwise cosine similarity of SAE decoder weight vectors
- Compare to E8's probe weight geometry (not necessary, if performing this explain significance)
- Compute MMCS (Mean Max Cosine Similarity) between SAE decoder columns and E8 probe directions — how well do SAE features align with linear probe directions?

**D. Feature activation across steps**:
- For each feature, plot its mean activation at each of the 16 steps
- Identify "early features" (active mainly at steps 0–4) vs "late features" (active mainly at steps 12–15)
- Correlate step-activation profiles with step-accuracy profiles

**E. Dead feature analysis**:
- Count features that never activate (dead features) or rarely activate
- Report alive feature percentage as function of dict_size and l1_coeff

Save all to: `results/sae_study/feature_analysis/`

#### 6. SAE Causal Validation Script (`scripts/sae_causal_ablation.py`)

This is the **key experiment** — the direct comparison with E9:

Create a script that:
- Loads the trained SAE and the HRM checkpoint
- For each of the top-K SAE features (K=50, ranked by activation frequency or constraint correlation):
  - Run a modified forward pass where, at each step:
    1. Encode z_H through the SAE encoder: `h = sae.encode(z_H)`
    2. Zero out the single feature: `h[:, feature_idx] = 0`
    3. Reconstruct: `z_H_ablated = sae.decode(h)`
    4. Replace z_H with z_H_ablated
  - Record Δaccuracy, Δrow_violations, Δcol_violations, Δbox_violations
- Also ablate RANDOM features as control (10 random features per puzzle)
- Also ablate the E9 PROBE DIRECTIONS using the SAE encoding (for direct comparison)
- Statistical tests: t-test SAE feature ablation effect vs. random feature ablation vs. E9 probe direction ablation
- N ≥ 200 puzzles (same consistent set as other experiments, `--seed 42`)

Save: `results/sae_study/causal_ablation/per_puzzle.jsonl`, `results/sae_study/causal_ablation/aggregate.json`

**Key output**: Bar chart comparing causal effect of:
  1. Top SAE features (individually)
  2. Random SAE features
  3. E9 probe directions
  4. Random directions (from E9)

If SAE features have significantly larger causal effects than probe directions → strong evidence that the model uses distributed, sparse computation, and SAE is better at finding functional features.

#### 7. SAE Visualization Script (`scripts/sae_plot.py`)

Create publication-quality plots:
1. **Reconstruction quality**: Original z_H vs. SAE reconstruction, scatter plot of cosine similarity
2. **Hyperparameter sweep**: Heatmap of (dict_size × l1_coeff) → reconstruction_loss, alive_features
3. **Feature specialization matrix**: [top features × constraint targets], color-coded by correlation
4. **Feature activation profiles**: Mean activation per step for top-20 features
5. **Causal comparison**: SAE features vs. probe directions vs. random (bar chart with CI)
6. **Geometry comparison**: SAE decoder columns vs. E8 probe directions (cosine similarity matrix)

Save PNG + PDF to `results/sae_study/plots/`

#### Important Implementation Notes

- Base model loading on `scripts/e8_constraint_probes.py` — it already handles checkpoint loading, device management, and probe recording
- The `probe_recorder` in `models/hrm/hrm_act_v1.py` is already called during forward passes — extend or reuse it
- For the causal ablation (step 6), you need to hook into the model's forward pass to modify z_H in-place. Look at how `scripts/activation_ablation.py` implements `ActivationAblator` for the pattern
- The SAE should be trained on DETACHED z_H (no gradients to the main model) — this is standard for SAE interpretability
- When filtering by step (e.g., training SAE only on step 12-15 activations), this tests whether late-step representations have different feature structure than early-step ones
- Use `--device cpu` support for all scripts (the codebase already handles this in existing scripts)

#### 8. Optional: SAE on z_L (`scripts/sae_collect_activations.py --also_collect_zL`)

If time permits, repeat the full pipeline on z_L activations:
- Expectation: z_L features should be more specialized (lower dimensionality per E4 analysis)
- z_L SAE may find features corresponding to individual constraint checks (more interpretable)
- Compare z_H SAE features vs. z_L SAE features

### QUERY END

---

## Expected Outcomes

1. Activation collection pipeline
2. Clean SAE implementation in `models/sae.py`
3. Training + sweep infrastructure
4. Feature analysis with constraint correlations
5. Causal validation comparing SAE features to E9 probe directions
6. Publication-quality visualizations

## Success Criteria

- SAE achieves <5% reconstruction error (cosine similarity > 0.95 between original and reconstructed z_H)
- ≥80% of features are alive (not dead) in the best configuration
- Some SAE features show clear constraint specialization (correlation > 0.5 with specific violation types)
- SAE feature ablation produces LARGER causal effects than probe direction ablation (the key prediction)
- If SAE feature ablation effects are still <1% → important null result suggesting even more distributed computation than expected
- All plots generated with error bars and proper statistics
