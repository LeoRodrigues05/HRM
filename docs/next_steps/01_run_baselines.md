# Query 01: Baseline Implementation & Comparative Analysis

## Objective
Implement, train, and evaluate two baseline models (Vanilla RNN and Universal Transformer) on the same Sudoku-Extreme dataset used by HRM, then run the same MI experiments to enable direct comparison. This is the most critical gap for publication — without baselines, reviewers cannot assess whether HRM's internal representations are special.

---

## Prompt to Execute

Copy and paste the following into the AI assistant:

---

### QUERY START

I need you to implement baseline models for comparing against the HRM (Hierarchical Reasoning Model) in my mechanistic interpretability study. The codebase is at `/home/leo.rodrigues/HRM/`. Please do the following:

#### 1. Implement a Vanilla RNN Baseline (`models/baselines/vanilla_rnn.py`)

Create a recurrent model that:
- **Matches HRM's parameter budget** (~27M params): use hidden_size=512, vocab_size=11, seq_len=81
- **Uses the same Transformer block backbone** (reuse `models/layers.py`: `Attention`, `SwiGLU`, `rms_norm`, `RotaryEmbedding`) — this ensures the only difference is architecture, not component quality
- **Has a FLAT recurrent loop** (no H/L hierarchy): single hidden state `z` updated identically at each of 16 steps (matching HRM's total steps = 16 segments × 2 H-cycles × 2 L-cycles, but since HRM paper uses 16 deep supervision segments as the comparable unit, use 16 iterations)
- **Architecture**: At each step: `z = TransformerBlock(z + input_embeddings)` repeated through N layers, where N is chosen to match ~27M total params (likely 8 layers of the same block HRM uses, since HRM has 4 H-layers + 4 L-layers = 8 total)
- **Deep supervision**: Apply the same loss at each step (output_head on z, detach between steps), exactly as HRM does
- **ACT halting**: Skip for simplicity — use fixed 16 steps
- **Input/output format**: Same as HRM — `embed_tokens` → sequence → `lm_head` producing logits over vocab
- **Puzzle embeddings**: Include the same puzzle embedding mechanism from HRM for fair comparison
- **Must follow the same interface** as HRM: `initial_carry(batch)`, forward returns `(carry, outputs, loss_info)` per the `ACTLossHead` expectations. Look at how `HierarchicalReasoningModel_ACTV1` wraps its inner model and implements the ACT loop in `models/hrm/hrm_act_v1.py`.

Key difference from HRM: **NO hierarchy** — single hidden state updated in a flat loop, NOT two interdependent modules at different timescales.

#### 2. Implement a Universal Transformer Baseline (`models/baselines/universal_transformer.py`)

Create a recurrent Transformer that:
- **Matches HRM's parameter budget** (~27M params)
- **Uses weight-sharing across iterations** (classic Universal Transformer design): same set of transformer layers applied repeatedly at each step
- **Has a flat structure**: single hidden state, single set of shared layers
- **Same deep supervision** as HRM (loss at each of 16 steps, detach between steps)
- **Same input format** and puzzle embeddings
- **Same interface** as HRM for compatibility with the training loop

Key difference from HRM: **Recurrent but flat** — same compute budget with recurrence, but no hierarchical separation of H and L modules.

#### 3. Create Architecture Configs

Create `config/arch/vanilla_rnn.yaml` and `config/arch/universal_transformer.yaml` with appropriate configs that reference the new model classes. Follow the pattern in `config/arch/hrm_v1.yaml` — the `name` field should be `baselines.vanilla_rnn@VanillaRNNModel` (or similar, matching your model class name and module path).

#### 4. Create a Baseline Training Config

Create `config/cfg_pretrain_baselines.yaml` (or modify the existing one with arch override) that:
- Points to the same Sudoku-Extreme dataset (`data_path`)
- Uses the same training hyperparameters (lr=1e-4, weight_decay=0.1, warmup, cosine schedule)
- Uses the same batch size and epoch count
- Look at `config/cfg_pretrain.yaml` for the exact values

#### 5. Create a Baseline Evaluation Script (`scripts/evaluate_baselines.py`)

A script that:
- Loads a trained baseline checkpoint
- Runs it on the test set (same puzzles HRM was evaluated on)
- Computes: cell accuracy, puzzle accuracy, unknown-cell accuracy, constraint violations per step, Hamming distance per step
- Outputs results in the same format as HRM evaluation for direct comparison
- Also runs activation patching (patch hidden state between two different puzzles at each step, measure accuracy change) to compare the causal importance of the hidden state to HRM's z_H

#### 6. Create a Comparison Plotting Script (`scripts/plot_baseline_comparison.py`)

Generate publication-quality comparison figures:
- **Figure 1**: Accuracy vs. reasoning step — HRM, Vanilla RNN, Universal Transformer on the same plot
- **Figure 2**: Hamming distance convergence — all three models
- **Figure 3**: Activation patching effect — bar chart comparing causal impact of hidden state patching across models
- **Figure 4**: Constraint satisfaction — row/col/box violations across steps for all three models

Save as PNG + PDF in `results/baseline_comparison/`.

#### Important Implementation Notes

- Study the existing code carefully before implementing:
  - `models/hrm/hrm_act_v1.py` — the full HRM model with ACT. Your baselines need a compatible interface.
  - `models/losses.py` — the `ACTLossHead` that wraps models. Your baselines should work with this.
  - `pretrain.py` — the training loop. Your baselines should be trainable with this same loop (or a minimal adaptation).
  - `models/layers.py` — reuse these components directly.
  - `models/common.py` — initialization utilities.
- The Vanilla RNN should demonstrate **premature convergence** (the key failure mode HRM solves via hierarchical convergence). If after training it achieves comparable accuracy to HRM, that would also be interesting — report accurately either way.
- Count parameters after implementation: `sum(p.numel() for p in model.parameters())` — should be within 10% of HRM's 27M.
- Use `models/baselines/__init__.py` to make the module importable.

### QUERY END

---

## Expected Outcomes

1. Two new model files in `models/baselines/`
2. Two architecture configs in `config/arch/`
3. Training configs for baselines
4. Evaluation and comparison scripts
5. After training + evaluation: publication-ready comparison figures in `results/baseline_comparison/`

## Success Criteria

- RNN baseline trains without errors on Sudoku-Extreme
- RNN likely achieves <50% accuracy (demonstrating premature convergence)
- Universal Transformer likely achieves 50–70% accuracy (recurrence helps, but no hierarchy limits it)
- HRM significantly outperforms both (>80% accuracy)
- Activation patching of RNN hidden state has lower causal impact than HRM's z_H patching
- All comparison plots generated with consistent styling
