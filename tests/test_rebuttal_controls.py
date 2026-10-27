"""CPU-only checks for the rebuttal controls (E0, E1 and E7 helpers).

Run from the repo root before the GPU jobs:  python -m pytest tests/test_rebuttal_controls.py
"""
import math

import numpy as np
import torch

from models.sae import SparseAutoencoder
from scripts.analysis.bptt_arm_eval import noise_transform
from scripts.probes.probe_metrics import binary_metrics, multiclass_metrics, mean_metrics
from scripts.sae.sae_error_controls import (
    SAEIntervention, draw_random_features, joint_comparison, paired_comparison,
)


def _sae(seed: int = 0) -> SparseAutoencoder:
    torch.manual_seed(seed)
    sae = SparseAutoencoder(input_dim=8, dict_size=32, l1_coeff=0.01).eval()
    with torch.no_grad():
        sae.pre_bias.normal_(0.0, 0.1)   # non-zero bias, so its cancellation is tested
    return sae


def test_error_preserving_matches_reconstruction_difference() -> None:
    sae = _sae()
    z = torch.randn(2, 5, 8)
    feats = [1, 4, 7]
    out, _ = SAEIntervention(sae, "all", n_cells=5).transform("error_preserving", feats)(z, z, 0)
    with torch.no_grad():
        x = z.reshape(-1, 8)
        h = sae.encode(x)
        h_abl = h.clone()
        h_abl[:, feats] = 0
        expected = (x + sae.decode(h_abl) - sae.decode(h)).view_as(z)
    assert torch.allclose(out, expected, atol=1e-5)


def test_error_preserving_without_features_is_identity() -> None:
    z = torch.randn(1, 5, 8)
    out, _ = SAEIntervention(_sae(), "all", n_cells=5).transform("error_preserving", [])(z, z, 0)
    assert torch.equal(out, z)


def test_replace_without_features_equals_recon_only() -> None:
    inter = SAEIntervention(_sae(), "all", n_cells=5)
    z = torch.randn(1, 5, 8)
    a, _ = inter.transform("replace", [])(z, z, 0)
    b, _ = inter.transform("recon_only")(z, z, 0)
    assert torch.allclose(a, b)


def test_noise_matched_has_reconstruction_error_norm() -> None:
    sae = _sae()
    z = torch.randn(1, 5, 8)
    gen = torch.Generator().manual_seed(0)
    out, _ = SAEIntervention(sae, "all", n_cells=5).transform("noise_matched", generator=gen)(z, z, 0)
    with torch.no_grad():
        x = z.reshape(-1, 8)
        err = (x - sae.decode(sae.encode(x))).norm(dim=-1)
    assert torch.allclose((out - z).reshape(-1, 8).norm(dim=-1), err, atol=1e-4)


def test_answer_positions_leave_prefix_untouched() -> None:
    z = torch.randn(1, 5, 8)
    out, _ = SAEIntervention(_sae(), "answer", n_cells=4).transform("recon_only")(z, z, 0)
    assert torch.equal(out[:, 0], z[:, 0])
    assert not torch.equal(out[:, 1:], z[:, 1:])


def test_random_draws_exclude_top_and_match_rates() -> None:
    rng = np.random.default_rng(0)
    fire = np.concatenate([np.full(10, 0.9), np.full(10, 0.5), np.full(10, 0.01), np.zeros(10)])
    top = list(range(5))
    matched = draw_random_features(rng, fire, top, [0.9] * 5, "freq_matched", window=1)
    assert set(matched) == {5, 6, 7, 8, 9}
    alive = draw_random_features(rng, fire, top, [0.0] * 8, "alive", window=5)
    assert len(set(alive)) == 8 and all(fire[f] > 0 for f in alive) and not set(alive) & set(top)
    anyf = draw_random_features(rng, fire, top, [0.0] * 8, "all", window=5)
    assert len(set(anyf)) == 8 and not set(anyf) & set(top)


def test_paired_comparison_declares_equivalence_inside_bound() -> None:
    diffs = np.random.default_rng(0).normal(0.0, 0.01, size=400)   # SE about 0.0005
    out = paired_comparison(diffs, bound=0.01)
    assert out["equivalent"]
    assert out["smallest_rejectable_bound"] < 0.01
    assert out["mde_80power"] < 0.01


def test_joint_comparison_rank_test() -> None:
    top = np.full(50, -0.2)
    rand = np.full((50, 19), -0.01)
    out = joint_comparison(top, rand, bound=0.01)
    assert math.isclose(out["monte_carlo_rank"]["p_top_more_damaging"], 1 / 20)


def test_binary_metrics_majority_and_perfect() -> None:
    y = torch.tensor([0] * 90 + [1] * 10)
    always_neg = binary_metrics(torch.full((100,), -5.0), y, y)
    assert math.isclose(always_neg["majority_acc"], 0.9)
    assert math.isclose(always_neg["balanced_acc"], 0.5)
    assert math.isclose(always_neg["prevalence"], 0.1)
    perfect = binary_metrics(y.float() * 10 - 5, y, y)
    assert perfect["auroc"] == 1.0 and math.isclose(perfect["mcc"], 1.0)


def test_multiclass_balanced_accuracy() -> None:
    y = torch.tensor([0] * 8 + [1] * 2)
    logits = torch.zeros(10, 3)
    logits[:, 0] = 1.0                                              # always predicts class 0
    m = multiclass_metrics(logits, y, y, n_classes=3)
    assert math.isclose(m["majority_acc"], 0.8) and math.isclose(m["balanced_acc"], 0.5)
    assert math.isnan(mean_metrics([m])["auroc"])


def test_noise_transform_relative_norm() -> None:
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(1, 6, 16)
    zh, zl = noise_transform(0.1, "H", gen)(z, z, 0)
    rel = (zh - z).norm(dim=-1) / z.norm(dim=-1)
    assert torch.allclose(rel, torch.full_like(rel, 0.1), atol=1e-5)
    assert torch.equal(zl, z)
    untouched, _ = noise_transform(0.1, "H", gen, only_step=3)(z, z, 0)
    assert torch.equal(untouched, z)
