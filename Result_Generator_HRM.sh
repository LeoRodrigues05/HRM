#Async evaluation to get predictions
nohup bash -c 'OMP_NUM_THREADS=8 uv run python evaluate.py checkpoint=Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt \
save_outputs="[\"inputs\",\"labels\",\"puzzle_identifiers\",\"logits\",\"q_halt_logits\",\"q_continue_logits\",\"intermediate_preds\"]"'   > eval.log 2>&1 &

# Get .npz
uv run python - <<'PY'
import torch, numpy as np, os

in_path = "Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/step_0_all_preds.0"
out_path = os.path.splitext(in_path)[0] + ".npz"
data = torch.load(in_path, map_location="cpu")

def to_numpy(x):
    import torch as T, numpy as np
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):  # torch tensor
        t = x.detach().cpu()
        if t.dtype in (T.bfloat16, T.float16):
            t = t.to(T.float32)
        return t.numpy()
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        arrs = [to_numpy(v) for v in x]
        # Try to stack along axis 0 if shapes match; else keep as object array
        try:
            return np.stack(arrs, axis=0)
        except Exception:
            return np.array(arrs, dtype=object)
    return np.array(x, dtype=object)

np_data = to_numpy(data)
# Ensure dict for savez; if not, wrap
if not isinstance(np_data, dict):
    np_data = {"data": np_data}
np.savez_compressed(out_path, **np_data)
print(f"✅ Saved {out_path}")
PY

#get report for sudoku (coloured)
python sudoku_report_colored_ai.py

#To get metrics
# Hamming (ignore givens), also plots
python result_metrics_sudoku.py \
  --npz Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/step_0_all_preds.npz \
  --ignore_givens \
  --outdir results/metrics


# python result_metrics_sudoku_multiple.py  \
#  --npz Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/step_0_all_preds.npz  \
#  --ignore_givens   --outdir results/hamming_multi --grid_rows 4 --grid_cols 6 --overlay_max 150