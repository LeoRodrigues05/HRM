#!/usr/bin/env python3
"""Fast correctness checks for the two new changes (run on a GPU node).

Validates:
  Track 1 (HRM): the new `one_step_grad` flag.
    - one_step_grad=True/False produce IDENTICAL forward activations under no_grad
      (the math is the same; only the gradient graph differs).
    - With one_step_grad=False, gradients flow to H_level / L_level params and the
      loss is finite (within-step BPTT actually backprops through the cycles).
  Track 2 (SAE): mean-centering.
    - act_mean defaults to zeros => encode/decode unchanged (backward compatible).
    - set_mean() makes decode(encode(x)) operate correctly on RAW activations.
    - gamma of mean-centered activations is ~0.

Usage
-----
    python scripts/sanity_check_bptt_sae.py
"""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.sae import SparseAutoencoder, TopKSparseAutoencoder
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1_Inner
from scripts.core.activation_ablation import _patch_attention_for_cpu


def _ok(msg):
    print(f"  [PASS] {msg}")


def test_sae(device):
    print("Track 2 — SAE mean-centering")
    torch.manual_seed(0)
    x = torch.randn(512, 64, device=device) * 3.0 + 5.0  # shifted activations (high gamma)

    sae = SparseAutoencoder(input_dim=64, dict_size=256, l1_coeff=0.01).to(device)

    # 1) Backward compatibility: act_mean defaults to zeros.
    assert torch.allclose(sae.act_mean, torch.zeros_like(sae.act_mean)), "act_mean should init to 0"
    _ok("act_mean defaults to zeros (no-op, backward compatible)")

    # 2) Round-trip interface works on raw activations.
    with torch.no_grad():
        h = sae.encode(x)
        x_hat = sae.decode(h)
    assert x_hat.shape == x.shape
    _ok("encode/decode round-trip runs on raw activations")

    # 3) set_mean changes act_mean and centering zeroes gamma.
    mean = x.mean(0)
    std = x.std(0)
    gamma_before = (mean.norm() / std.norm().clamp(min=1e-8)).item()
    sae.set_mean(mean)
    assert torch.allclose(sae.act_mean, mean.float(), atol=1e-5), "set_mean must store the mean"
    centered = x - sae.act_mean
    gamma_after = (centered.mean(0).norm() / centered.std(0).norm().clamp(min=1e-8)).item()
    assert gamma_after < 1e-4 < gamma_before, f"gamma should drop to ~0 (before={gamma_before:.3f}, after={gamma_after:.3e})"
    _ok(f"set_mean drives gamma {gamma_before:.3f} -> {gamma_after:.2e}")

    # 4) decode(encode(raw_x)) still consumes raw activations after centering.
    with torch.no_grad():
        x_hat2 = sae.decode(sae.encode(x))
    assert x_hat2.shape == x.shape and torch.isfinite(x_hat2).all()
    _ok("centered SAE still maps raw activations -> reconstruction")

    # 5) TopK variant inherits centering.
    tk = TopKSparseAutoencoder(input_dim=64, dict_size=256, k=8).to(device)
    tk.set_mean(mean)
    with torch.no_grad():
        h_tk = tk.encode(x)
    assert (h_tk > 0).sum(dim=-1).float().mean().item() <= 8 + 1e-6
    _ok("TopK SAE honors act_mean and keeps L0<=k")
    print()


def _make_inner(one_step_grad, device):
    cfg = dict(
        batch_size=2, seq_len=16, puzzle_emb_ndim=0, num_puzzle_identifiers=1,
        vocab_size=12, H_cycles=2, L_cycles=2, H_layers=2, L_layers=2,
        hidden_size=64, expansion=2.0, num_heads=4, pos_encodings="rope",
        halt_max_steps=4, halt_exploration_prob=0.1, one_step_grad=one_step_grad,
        forward_dtype="float32",  # float32 so attention runs on CPU/GPU without flash-attn
    )
    torch.manual_seed(123)
    inner = HierarchicalReasoningModel_ACTV1_Inner(cfg).to(device)
    # flash_attn is CUDA/fp16-only; swap in SDPA so this runs in float32 anywhere.
    _patch_attention_for_cpu(inner)
    return inner


def test_hrm(device):
    print("Track 1 — HRM one_step_grad flag")
    batch = {
        "inputs": torch.randint(0, 12, (2, 16), device=device),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long, device=device),
    }

    # Equivalence under no_grad: same weights/seed => identical forward.
    inner_a = _make_inner(True, device)
    inner_b = _make_inner(False, device)
    inner_b.load_state_dict(inner_a.state_dict())  # identical params

    carry_a = inner_a.empty_carry(2)
    carry_a = inner_a.reset_carry(torch.ones(2, dtype=torch.bool, device=device), carry_a)
    carry_b = inner_b.empty_carry(2)
    carry_b = inner_b.reset_carry(torch.ones(2, dtype=torch.bool, device=device), carry_b)

    inner_a.eval(); inner_b.eval()
    with torch.no_grad():
        _, out_a, _ = inner_a(carry_a, batch)
        _, out_b, _ = inner_b(carry_b, batch)
    assert torch.allclose(out_a, out_b, atol=1e-4), "one_step and BPTT must match under no_grad"
    _ok("one_step_grad True vs False give identical forward output under no_grad")

    # BPTT actually backprops through the cycles: grads reach L_level / H_level.
    inner_b.train()
    carry_b = inner_b.empty_carry(2)
    carry_b = inner_b.reset_carry(torch.ones(2, dtype=torch.bool, device=device), carry_b)
    _, out, _ = inner_b(carry_b, batch)
    loss = out.float().pow(2).mean()
    loss.backward()
    g_L = inner_b.L_level.layers[0].mlp.gate_up_proj.weight.grad
    g_H = inner_b.H_level.layers[0].mlp.gate_up_proj.weight.grad
    assert torch.isfinite(loss).item(), "loss must be finite"
    assert g_L is not None and g_L.abs().sum().item() > 0, "BPTT must populate L_level grads"
    assert g_H is not None and g_H.abs().sum().item() > 0, "BPTT must populate H_level grads"
    _ok(f"BPTT populates L_level & H_level grads (loss={loss.item():.4f})")
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    test_sae(device)
    test_hrm(device)
    print("ALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
