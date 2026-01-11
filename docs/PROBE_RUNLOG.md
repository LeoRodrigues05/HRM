# Probe Workflow Run Log

This document records the scripts and commands used to set up the environment, build datasets, collect probes, inspect outputs, and train linear probes, including troubleshooting steps.

## Environment setup

- Activate virtualenv and add repo to Python path:
```
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
```

- (Optional) Set CUDA toolkit path (if using GPU builds):
```
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

## Dataset build (Sudoku 1k)

- Build Sudoku dataset:
```
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
```

- Verify dataset presence:
```
ls -lh data/sudoku-extreme-1k-aug-1000/test/dataset.json
```

## Probe collection

- Default collection (prediction-based labels; conservative resource caps):
```
export PROBE_BATCH_SIZE=8
export HRM_HALT_MAX_STEPS=6
export MAX_PROBE_BATCHES=20
python scripts/run_probes_driver.py
```

- More coverage (to increase chance of solved states):
```
export PROBE_BATCH_SIZE=8
export HRM_HALT_MAX_STEPS=16
export MAX_PROBE_BATCHES=80
python scripts/run_probes_driver.py
```

- Optional CPU-only run to avoid GPU OOM:
```
export CPU_ONLY=1
python scripts/run_probes_driver.py
```

- Inline driver (train split) for more solved states (no code changes):
```
python - <<'PY'
import os, torch, json, numpy as np
from torch.utils.data import DataLoader
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.probes import ProbeRecorder
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
root="data/sudoku-extreme-1k-aug-1000"
meta=json.load(open(os.path.join(root,"test","dataset.json")))
seq_len=int(np.load(os.path.join(root,"test","all__inputs.npy"), mmap_mode="r").shape[1])
config={"batch_size":8,"seq_len":seq_len,"puzzle_emb_ndim":0,"num_puzzle_identifiers":1024,
        "vocab_size":meta.get("vocab_size",512),"H_cycles":2,"L_cycles":2,"H_layers":2,"L_layers":2,
        "hidden_size":512,"expansion":2.0,"num_heads":8,"pos_encodings":"rope","rms_norm_eps":1e-5,
        "rope_theta":10000.0,"halt_max_steps":16,"halt_exploration_prob":0.0,"forward_dtype":"bfloat16"}
model=HierarchicalReasoningModel_ACTV1(config).to(device).eval()
ds_cfg=PuzzleDatasetConfig(seed=42,dataset_path=root,global_batch_size=config["batch_size"],
                           test_set_mode=True,epochs_per_iter=1,rank=0,num_replicas=1)
dl=DataLoader(PuzzleDataset(ds_cfg, split="train"), batch_size=None, shuffle=False)
rec=ProbeRecorder(output_dir=os.path.join("results","probes"))
with torch.no_grad():
    processed=0
    for _, batch, _ in dl:
        batch={k:v.to(device) for k,v in batch.items()}
        carry=model.initial_carry(batch)
        carry.inner_carry=type(carry.inner_carry)(z_H=carry.inner_carry.z_H.to(device),z_L=carry.inner_carry.z_L.to(device))
        carry.steps=carry.steps.to(device); carry.halted=carry.halted.to(device)
        carry.current_data={k:v.to(device) for k,v in carry.current_data.items()}
        for _ in range(model.config.halt_max_steps):
            carry, outputs = model(carry, batch)
            preds = outputs.get("intermediate_preds_step")
            rec.record_hidden(step_index=int(carry.steps.max().item()), phase="grad",
                              z_H=carry.inner_carry.z_H, z_L=carry.inner_carry.z_L,
                              batch=batch, preds=preds)
            if torch.all(carry.halted): break
        processed+=1
        if processed>=80: break
rec.finalize_and_save(); print("Saved probe datasets to results/probes")
PY
```

## Probe inspection

- List saved probes:
```
ls -lh results/probes/
```

- Quick summary:
```
python - <<'PY'
import torch
g=torch.load("results/probes/probe_global.pt"); l=torch.load("results/probes/probe_local.pt")
print("global entries:",len(g),"local entries:",len(l))
if len(g)>0:
    z=g[0].get("z_H") or g[0].get("z_L")
    if z is not None:
        print("example pooled shape:", z.shape)
PY
```

- Global label balance (is_solved):
```
python - <<'PY'
import torch
g = torch.load("results/probes/probe_global.pt")
ys=[]
for row in g:
    y = row["labels"].get("is_solved")
    ys.append(int((y.float().mean()>0.5).item()) if torch.is_tensor(y) else int(y))
print("Global is_solved ratio:", sum(ys)/len(ys), "n:", len(ys))
PY
```

## Train linear probes

- Default (z_H global, z_L local):
```
python scripts/train_linear_probes.py --probes_dir results/probes
```

- Swap features (z_L global, z_H local):
```
python scripts/train_linear_probes.py --probes_dir results/probes --use_global_z z_L --use_local_z z_H
```

## Hold-out evaluation (sklearn)

- Install scikit-learn if needed:
```
pip install scikit-learn
```

- Hold-out check (global z_H):
```
python - <<'PY'
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
g = torch.load("results/probes/probe_global.pt")
X=[]; y=[]
for row in g:
    z=row["z_H"]; lab=row["labels"]["is_solved"]
    lab=int((lab.float().mean()>0.5).item()) if torch.is_tensor(lab) else int(lab)
    x = z.mean(0).cpu().numpy() if z.ndim==2 else z.cpu().numpy()
    X.append(x); y.append(lab)
X=np.stack(X); y=np.array(y)
idx=np.random.permutation(len(y)); split=int(0.8*len(y))
train, test = idx[:split], idx[split:]
clf=LogisticRegression(max_iter=1000).fit(X[train], y[train])
print("Train acc:", clf.score(X[train], y[train]))
print("Test acc:", accuracy_score(y[test], clf.predict(X[test])))
PY
```

## Troubleshooting notes

- Single-class global labels (`is_solved` all zeros): Increase `HRM_HALT_MAX_STEPS` and `MAX_PROBE_BATCHES`, use training split, or a stronger checkpoint.
- OOM/Killed (exit 137): reduce `PROBE_BATCH_SIZE`, cap `MAX_PROBE_BATCHES`, set `CPU_ONLY=1`.
- Import errors for `utils`/`models`: ensure `export PYTHONPATH="$PWD:$PYTHONPATH"` before running scripts.
