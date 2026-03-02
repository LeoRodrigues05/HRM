"""Quick diagnostic: cosine similarity of z_H across ACT steps."""
import os, sys, yaml, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretrain import PretrainConfig, create_dataloader
from utils.functions import load_model_class
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from scripts.activation_ablation import ActivationAblator, ActivationCache, _patch_attention_for_cpu
from torch import nn
from typing import Any, cast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

ckpt_dir = 'Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku'
config_path = os.path.join(ckpt_dir, 'config.yaml')
if os.path.exists(os.path.join(ckpt_dir, 'all_config.yaml')):
    config_path = os.path.join(ckpt_dir, 'all_config.yaml')
with open(config_path) as f:
    config = PretrainConfig(**yaml.safe_load(f))

test_loader, test_meta = create_dataloader(config, 'test', test_set_mode=True,
    epochs_per_iter=1, global_batch_size=1, rank=0, world_size=1)

model_cfg = dict(**config.arch.__pydantic_extra__, batch_size=1,
    vocab_size=test_meta.vocab_size, seq_len=test_meta.seq_len,
    num_puzzle_identifiers=test_meta.num_puzzle_identifiers, causal=False)
model_cls = load_model_class(config.arch.name)
loss_cls = load_model_class(config.arch.loss.name)

with torch.device(device):
    model_raw = model_cls(model_cfg)
    model_full = loss_cls(model_raw, **config.arch.loss.__pydantic_extra__)

ckpt = torch.load(os.path.join(ckpt_dir, 'checkpoint.pt'), map_location=device, weights_only=False)
mk = set(model_full.state_dict().keys())
ck = set(ckpt.keys())
if any(k.startswith('_orig_mod.') for k in mk) and not any(k.startswith('_orig_mod.') for k in ck):
    ckpt = {f'_orig_mod.{k}': v for k, v in ckpt.items()}
elif any(k.startswith('_orig_mod.') for k in ck) and not any(k.startswith('_orig_mod.') for k in mk):
    ckpt = {k.removeprefix('_orig_mod.'): v for k, v in ckpt.items()}
model_full.load_state_dict(ckpt, assign=True)
model_full.to(device).eval()

# Patch flash_attn for CPU
if device.type == 'cpu':
    _patch_attention_for_cpu(model_full)

m: Any = model_full
if hasattr(m, '_orig_mod'): m = m._orig_mod
if not isinstance(m, HierarchicalReasoningModel_ACTV1) and hasattr(m, 'model'): m = m.model

ablator = ActivationAblator(m, device=device)

def extract_batch(item):
    if isinstance(item, (tuple, list)):
        if len(item) >= 2 and isinstance(item[1], dict): return item[1]
        if isinstance(item[0], dict): return item[0]
    if isinstance(item, dict): return item
    raise TypeError

# Collect puzzles 0..9
puzzles = []
for i, data in enumerate(test_loader):
    if i >= 10:
        break
    batch = extract_batch(data)
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    puzzles.append(batch)

print(f"\nCollected {len(puzzles)} puzzles")
print("=" * 80)

for pi in [0, 2, 5, 8]:
    batch = puzzles[pi]
    cache = {}
    ablator.run_and_cache_activations(batch, cache, max_steps=16)
    steps = sorted(cache.keys())
    
    # Consecutive cosine similarities of z_H_out
    consec_sims = []
    for j in range(len(steps)-1):
        z1 = cache[steps[j]].z_H_out.flatten().float()
        z2 = cache[steps[j+1]].z_H_out.flatten().float()
        cos = torch.nn.functional.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item()
        consec_sims.append(cos)
    
    # First vs last
    z0 = cache[steps[0]].z_H_out.flatten().float()
    zlast = cache[steps[-1]].z_H_out.flatten().float()
    first_last = torch.nn.functional.cosine_similarity(z0.unsqueeze(0), zlast.unsqueeze(0)).item()
    
    # L2 distance
    l2_consec = []
    for j in range(len(steps)-1):
        z1 = cache[steps[j]].z_H_out.flatten().float()
        z2 = cache[steps[j+1]].z_H_out.flatten().float()
        l2_consec.append((z1-z2).norm().item())
    
    print(f"Puzzle {pi}:")
    print(f"  Consecutive cosine sims: {' '.join(f'{s:.4f}' for s in consec_sims)}")
    print(f"  First vs Last cosine:    {first_last:.6f}")
    print(f"  Mean consecutive L2:     {sum(l2_consec)/len(l2_consec):.4f}")
    print(f"  z_H_out norm step0:      {cache[steps[0]].z_H_out.flatten().float().norm().item():.2f}")
    print(f"  z_H_out norm step15:     {cache[steps[-1]].z_H_out.flatten().float().norm().item():.2f}")

print("\nDone.")
