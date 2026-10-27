#!/usr/bin/env python3
"""E1: reconstruction controls for the SAE feature-ablation experiment.

``sae_causal_ablation.py`` (E10) replaces z_H with the SAE reconstruction at
every ACT step, so each feature ablation also pays the SAE's reconstruction
error. This script measures that cost on its own and repeats the ablations with
the error term kept, as standard SAE intervention work does.

Conditions (all applied to the z_H carry entering every ACT step):
  recon_only        z_H <- decode(encode(z_H))                    nothing zeroed
  noise_matched     z_H <- z_H + eps, |eps| = |z_H - decode(encode(z_H))| per position
  replace           z_H <- decode(h with features S zeroed)       the E10 protocol
  error_preserving  z_H <- z_H - sum_{f in S} h_f d_f             error term kept
  ep_norm_matched   z_H <- z_H - |sum_{f in S} h_f d_f| u         random unit u (control)

Feature sets S:
  single  each of the top-k features alone, and each of n random features alone
          (the E10 design, now under both replace and error_preserving)
  joint   the top-k features together for each k in --joint_ks, against
          --n_random_draws random sets of the same size

Features are ranked by firing frequency (as in E10) or mean activation. Random
features come from --random_pool: ``all`` (E10: any non-top feature, dead ones
included), ``alive`` (fired at least once), or ``freq_matched`` (for each top
feature, a random non-top feature among the --match_window with the closest
firing rate).

Statistics (per-puzzle paired differences; puzzles are the sampling unit):
  bootstrap 95% CI, paired t-test, TOST against +/- --tost_bound_pp, the
  smallest bound the 90% CI rules out, and the 80%-power minimum detectable
  effect. Joint sets add a Monte Carlo rank test over the random draws,
  p = (b + 1) / (R + 1), and a bootstrap that resamples puzzles and draws.

SAE statistics (L0, dead fraction, fraction of variance explained, relative
reconstruction error) are computed over the whole activation bank.

Usage
-----
  # Sudoku, released checkpoint (the paper's main SAE)
  python scripts/sae/sae_error_controls.py --task sudoku \\
      --sae_path results/Sudoku/sae_study/sae_d2048_l10.01.pt \\
      --activations_path results/Sudoku/sae_study/activations_zH.pt \\
      --n_puzzles 300 --output_dir results/Sudoku/sae_study/error_controls

  # Maze
  python scripts/sae/sae_error_controls.py --task maze \\
      --sae_path results/Maze/sae_study/sae_d2048_l10.01.pt \\
      --activations_path results/Maze/sae_study/activations_zH.pt \\
      --n_puzzles 300 --output_dir results/Maze/sae_study/error_controls
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.sae import SparseAutoencoder, TopKSparseAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

SUDOKU_CELLS = 81
MAZE_CELLS = 900
ALL_MODES = ("replace", "error_preserving")


# ═══════════════════════════════════════════════════════════════════════════
# SAE loading, statistics and ranking
# ═══════════════════════════════════════════════════════════════════════════

def load_sae(path: str, device: torch.device) -> Tuple[SparseAutoencoder, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if cfg.get("activation", "relu") == "topk":
        sae = TopKSparseAutoencoder(input_dim=cfg["input_dim"], dict_size=cfg["dict_size"], k=cfg["k"])
    else:
        sae = SparseAutoencoder(input_dim=cfg["input_dim"], dict_size=cfg["dict_size"],
                                l1_coeff=cfg["l1_coeff"])
    # strict=False: SAEs trained before the act_mean buffer existed lack that key.
    sae.load_state_dict(ckpt["model_state_dict"], strict=False)
    return sae.to(device).eval(), cfg


@torch.no_grad()
def sae_statistics(sae: SparseAutoencoder, activations_path: str, device: torch.device,
                   batch_size: int = 8192) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Firing rates, mean activations and reconstruction quality over the bank.

    Uses every row, like ``select_top_features`` in E10, so the frequency
    ranking matches the features E10 ablated.
    """
    data = torch.load(activations_path, map_location="cpu", weights_only=False)
    z = data["z_H"] if isinstance(data, dict) else data
    flat = z.reshape(-1, z.shape[-1])
    n_rows, d_in = flat.shape
    dict_size = sae.dict_size

    fire = torch.zeros(dict_size, dtype=torch.float64)
    act_sum = torch.zeros(dict_size, dtype=torch.float64)
    sum_x = torch.zeros(d_in, dtype=torch.float64)
    sum_sq = sq_err = rel_err = l0 = 0.0
    for start in range(0, n_rows, batch_size):
        x = flat[start:start + batch_size].to(device).float()
        h = sae.encode(x)
        err = x - sae.decode(h)
        active = h > 0
        fire += active.sum(0).double().cpu()
        act_sum += h.sum(0).double().cpu()
        l0 += float(active.sum().item())
        sq_err += float(err.pow(2).sum().item())
        sum_x += x.sum(0).double().cpu()
        sum_sq += float(x.pow(2).sum().item())
        rel_err += float((err.norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-8)).sum().item())

    total_var = sum_sq - float(sum_x.pow(2).sum().item()) / n_rows
    fire_rate = (fire / n_rows).numpy()
    stats = {
        "n_rows": int(n_rows),
        "input_dim": int(d_in),
        "dict_size": int(dict_size),
        "expansion": dict_size / d_in,
        "L0_mean_active_per_row": l0 / n_rows,
        "dead_count": int((fire == 0).sum().item()),
        "dead_frac": float((fire == 0).double().mean().item()),
        "rare_below_0.1pct_count": int(((fire_rate > 0) & (fire_rate < 1e-3)).sum()),
        "above_50pct_count": int((fire_rate > 0.5).sum()),
        "mse_per_element": sq_err / (n_rows * d_in),
        "fraction_variance_explained": 1.0 - sq_err / max(total_var, 1e-12),
        "mean_relative_error_norm": rel_err / n_rows,
    }
    return stats, fire_rate, (act_sum / n_rows).numpy()


def rank_features(fire_rate: np.ndarray, mean_act: np.ndarray, by: str) -> List[int]:
    key = fire_rate if by == "freq" else mean_act
    return [int(i) for i in np.argsort(-key, kind="stable")]


def draw_random_features(rng: np.random.Generator, fire_rate: np.ndarray,
                         exclude: Sequence[int], reference_rates: Sequence[float],
                         pool: str, window: int) -> List[int]:
    """One random feature set, the same size as ``reference_rates``."""
    excluded = set(int(f) for f in exclude)
    candidates = np.array([f for f in range(len(fire_rate)) if f not in excluded])
    n = len(reference_rates)
    if pool == "all":
        return rng.choice(candidates, size=min(n, len(candidates)), replace=False).tolist()
    alive = candidates[fire_rate[candidates] > 0]
    if pool == "alive":
        return rng.choice(alive, size=min(n, len(alive)), replace=False).tolist()
    if pool != "freq_matched":
        raise ValueError(f"unknown random pool {pool!r}")
    log_rate = np.log10(fire_rate + 1e-12)
    chosen: List[int] = []
    for rate in reference_rates:
        available = np.array([f for f in alive if f not in chosen])
        if available.size == 0:
            break
        dist = np.abs(log_rate[available] - math.log10(rate + 1e-12))
        nearest = available[np.argsort(dist, kind="stable")[:window]]
        chosen.append(int(rng.choice(nearest)))
    return chosen


# ═══════════════════════════════════════════════════════════════════════════
# Interventions on z_H
# ═══════════════════════════════════════════════════════════════════════════

class SAEIntervention:
    """Builds per-step transforms of the z_H carry for one SAE."""

    def __init__(self, sae: SparseAutoencoder, positions: str, n_cells: int):
        self.sae = sae
        self.W = sae.decoder.weight.detach().float()   # [input_dim, dict_size]
        self.positions = positions
        self.n_cells = n_cells

    def transform(self, mode: str, features: Sequence[int] = (),
                  generator: Optional[torch.Generator] = None) -> Callable:
        sae, W = self.sae, self.W
        idx = torch.as_tensor(list(features), dtype=torch.long, device=W.device)

        def fn(z_H: torch.Tensor, z_L: torch.Tensor, step: int):
            z = z_H.float()
            B, T, D = z.shape
            start = 0 if self.positions == "all" else T - self.n_cells
            x = z[:, start:, :].reshape(-1, D)
            h = sae.encode(x)
            if mode == "recon_only":
                x_new = sae.decode(h)
            elif mode == "noise_matched":
                err_norm = (x - sae.decode(h)).norm(dim=-1, keepdim=True)
                eps = torch.randn(x.shape, generator=generator, device=x.device)
                x_new = x + eps * (err_norm / eps.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            elif mode == "replace":
                h_abl = h.clone()
                if idx.numel():
                    h_abl[:, idx] = 0
                x_new = sae.decode(h_abl)
            elif mode == "error_preserving":
                # decode(h) - decode(h_abl) = W[:, S] h_S, so keeping the error
                # term means subtracting only the ablated features' contribution.
                x_new = x - h[:, idx] @ W[:, idx].T if idx.numel() else x
            elif mode == "ep_norm_matched":
                removed = (h[:, idx] @ W[:, idx].T).norm(dim=-1, keepdim=True)
                u = torch.randn(x.shape, generator=generator, device=x.device)
                x_new = x - removed * (u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            else:
                raise ValueError(f"unknown mode {mode!r}")
            z_new = z.clone()
            z_new[:, start:, :] = x_new.view(B, T - start, D)
            return z_new.to(z_H.dtype), z_L

        return fn


# ═══════════════════════════════════════════════════════════════════════════
# Tasks
# ═══════════════════════════════════════════════════════════════════════════

def _extract_batch(data):
    if isinstance(data, (tuple, list)):
        return data[1] if len(data) >= 2 and isinstance(data[1], dict) else data[0]
    return data


class SudokuTask:
    metric = "cell_acc"
    n_cells = SUDOKU_CELLS

    def __init__(self, args, device: torch.device):
        from scripts.sae.sae_causal_ablation import load_model_and_data
        from scripts.core.sudoku_sample import collect_indexed_batches, load_puzzle_indices
        self.model, loader, _ = load_model_and_data(
            device, ckpt_dir=args.checkpoint_dir, ckpt_file=args.checkpoint_file)
        indices = (load_puzzle_indices(args.puzzle_indices, limit=args.n_puzzles)
                   if args.puzzle_indices else None)
        self.puzzles = collect_indexed_batches(
            loader, device, num_puzzles=args.n_puzzles, puzzle_indices=indices,
            extract_batch=_extract_batch)

    def score(self, preds: torch.Tensor, batch: Dict[str, torch.Tensor]) -> float:
        p = preds[:, -SUDOKU_CELLS:].reshape(-1).cpu()
        t = batch["labels"][:, -SUDOKU_CELLS:].reshape(-1).cpu()
        return float((p == t).float().mean().item())


class MazeTask:
    metric = "valid_sg_path"
    n_cells = MAZE_CELLS

    def __init__(self, args, device: torch.device):
        from scripts.controlled.controlled_common import load_model_and_dataloader, collect_puzzles
        from scripts.maze.maze_common import MAZE_CHECKPOINT
        self.model, loader, _ = load_model_and_dataloader(args.checkpoint or MAZE_CHECKPOINT, device)
        self.puzzles = collect_puzzles(loader, device, args.n_puzzles, seed=args.seed)

    def score(self, preds: torch.Tensor, batch: Dict[str, torch.Tensor]) -> float:
        from scripts.maze.maze_common import maze_prediction_metrics
        from scripts.maze.linear_probes_maze import _flat, _slice_preds
        inp, label = _flat(batch["inputs"]), _flat(batch["labels"])
        p = _slice_preds(_flat(preds), label.size)
        return float(maze_prediction_metrics(p, label, inp)["valid_sg_path"])


# ═══════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════

def summarize(values: Sequence[float], seed: int = 0, n_boot: int = 2000) -> Dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    n = int(a.size)
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    boot = a[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if n > 1 else 0.0,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n": n,
    }


def paired_comparison(diffs: Sequence[float], bound: float) -> Dict[str, Any]:
    """Paired test, TOST equivalence test and detectable-effect size for diffs."""
    from scipy import stats
    d = np.asarray(diffs, dtype=np.float64)
    out = summarize(d)
    n = int(d.size)
    if n < 2:
        return out
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    if se < 1e-12:
        equivalent = abs(mean) < bound
        out.update(t=float("nan"), p_two_sided=1.0 if mean == 0 else 0.0,
                   ci90=[mean, mean], tost_bound=bound,
                   tost_p=0.0 if equivalent else 1.0, equivalent=bool(equivalent),
                   smallest_rejectable_bound=abs(mean), mde_80power=0.0)
        return out
    t_stat = mean / se
    p_two = float(2 * stats.t.sf(abs(t_stat), n - 1))
    half90 = float(stats.t.ppf(0.95, n - 1)) * se
    p_lower = float(stats.t.sf((mean + bound) / se, n - 1))     # H0: diff <= -bound
    p_upper = float(stats.t.cdf((mean - bound) / se, n - 1))    # H0: diff >= +bound
    out.update(
        t=float(t_stat), p_two_sided=p_two,
        ci90=[mean - half90, mean + half90],
        tost_bound=bound, tost_p=max(p_lower, p_upper),
        equivalent=bool(max(p_lower, p_upper) < 0.05),
        smallest_rejectable_bound=max(abs(mean - half90), abs(mean + half90)),
        mde_80power=float((stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * se),
    )
    return out


def joint_comparison(top: Sequence[float], random_draws: Sequence[Sequence[float]],
                     bound: float, seed: int = 0, n_boot: int = 2000) -> Dict[str, Any]:
    """top: [n_puzzles]; random_draws: [n_puzzles][R]. Lower delta = more damage."""
    t = np.asarray(top, dtype=np.float64)
    r = np.asarray(random_draws, dtype=np.float64)
    out: Dict[str, Any] = {"top": summarize(t), "random_pooled": summarize(r.mean(axis=1))}
    if r.ndim != 2 or r.shape[1] == 0:
        return out
    n, R = r.shape
    draw_means = r.mean(axis=0)
    b = int((draw_means <= t.mean()).sum())
    out["random_draw_means"] = draw_means.tolist()
    out["monte_carlo_rank"] = {
        "R": int(R),
        "random_at_least_as_damaging": b,
        "p_top_more_damaging": (b + 1) / (R + 1),
        "top_percentile_among_random": float((draw_means < t.mean()).mean()),
    }
    out["paired_top_minus_random"] = paired_comparison(t - r.mean(axis=1), bound)
    rng = np.random.default_rng(seed)
    crossed = np.empty(n_boot)
    for i in range(n_boot):
        pi = rng.integers(0, n, size=n)
        ri = rng.integers(0, R, size=R)
        crossed[i] = t[pi].mean() - r[np.ix_(pi, ri)].mean()
    out["crossed_bootstrap_ci95"] = [float(np.percentile(crossed, 2.5)),
                                     float(np.percentile(crossed, 97.5))]
    return out


def aggregate(records: List[Dict[str, Any]], bound: float) -> Dict[str, Any]:
    agg: Dict[str, Any] = {
        "n_puzzles": len(records),
        "units": "deltas are fractions of the task metric; multiply by 100 for pp",
        "baseline": summarize([r["baseline"] for r in records]),
        "recon_only": summarize([r["recon_only"] for r in records]),
        "noise_matched": summarize([r["noise_matched"] for r in records]),
        "single": {}, "joint": {},
    }
    for mode in records[0].get("single", {}):
        top = [float(np.mean(r["single"][mode]["top"])) for r in records]
        rand = [float(np.mean(r["single"][mode]["random"])) for r in records]
        agg["single"][mode] = {
            "top_mean_over_features": summarize(top),
            "random_mean_over_features": summarize(rand),
            "paired_top_minus_random": paired_comparison(np.subtract(top, rand), bound),
        }
        if mode == "replace":
            agg["single"][mode]["paired_top_minus_recon_only"] = paired_comparison(
                np.subtract(top, [r["recon_only"] for r in records]), bound)
    for mode in records[0].get("joint", {}):
        agg["joint"][mode] = {}
        for k in records[0]["joint"][mode]:
            agg["joint"][mode][k] = joint_comparison(
                [r["joint"][mode][k]["top"] for r in records],
                [r["joint"][mode][k]["random"] for r in records], bound)
    if "ep_norm_matched" in records[0]:
        agg["ep_norm_matched"] = {
            k: summarize([r["ep_norm_matched"][k] for r in records])
            for k in records[0]["ep_norm_matched"]}
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _csv(raw: str, cast=str) -> List[Any]:
    return [cast(x) for x in raw.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description="E1: SAE reconstruction controls")
    p.add_argument("--task", choices=["sudoku", "maze"], default="sudoku")
    p.add_argument("--sae_path", required=True)
    p.add_argument("--activations_path", required=True,
                   help="z_H activation bank the SAE was trained on (for ranking and stats)")
    p.add_argument("--checkpoint_dir", default=None,
                   help="Sudoku HRM checkpoint dir (default: bundled sapientinc model)")
    p.add_argument("--checkpoint_file", default=None)
    p.add_argument("--checkpoint", default=None, help="Maze checkpoint file (default: MAZE_CHECKPOINT)")
    p.add_argument("--n_puzzles", type=int, default=200)
    p.add_argument("--puzzle_indices", default=None, help="Sudoku: JSON list/manifest of puzzle indices")
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--rank_by", choices=["freq", "mean_act"], default="freq",
                   help="freq reproduces E10's ranking")
    p.add_argument("--top_k", type=int, default=50, help="number of top features ablated one at a time")
    p.add_argument("--n_random_features", type=int, default=50)
    p.add_argument("--random_pool", choices=["all", "alive", "freq_matched"], default="freq_matched")
    p.add_argument("--match_window", type=int, default=5)
    p.add_argument("--single_modes", default="replace,error_preserving",
                   help="modes for single-feature ablations; empty string skips them")
    p.add_argument("--joint_ks", default="10,50", help="sizes for joint ablation; empty skips")
    p.add_argument("--joint_modes", default="replace,error_preserving")
    p.add_argument("--n_random_draws", type=int, default=20)
    p.add_argument("--norm_control", action="store_true",
                   help="also remove a random vector with the norm of each top-k contribution")
    p.add_argument("--positions", choices=["all", "answer"], default="all",
                   help="all = every sequence position (E10); answer = grid cells only")
    p.add_argument("--tost_bound_pp", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    single_modes = _csv(args.single_modes)
    joint_modes = _csv(args.joint_modes)
    joint_ks = _csv(args.joint_ks, int)
    for m in single_modes + joint_modes:
        if m not in ALL_MODES:
            raise ValueError(f"mode {m!r} not in {ALL_MODES}")

    sae, cfg = load_sae(args.sae_path, device)
    logger.info(f"SAE {args.sae_path}: dict_size={cfg['dict_size']} activation={cfg.get('activation', 'relu')}")
    stats, fire_rate, mean_act = sae_statistics(sae, args.activations_path, device)
    logger.info("SAE stats: " + json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                           for k, v in stats.items()}))

    order = rank_features(fire_rate, mean_act, args.rank_by)
    top_single = order[:args.top_k]
    # Reference rates only matter for freq_matched; cycle the top features' rates
    # so n_random_features may exceed top_k.
    top_rates = [float(fire_rate[f]) for f in top_single]
    ref_rates = [top_rates[i % len(top_rates)] for i in range(args.n_random_features)]
    rand_single = draw_random_features(rng, fire_rate, top_single, ref_rates,
                                       args.random_pool, args.match_window)
    joint_draws = {
        k: [draw_random_features(rng, fire_rate, order[:k], [fire_rate[f] for f in order[:k]],
                                 args.random_pool, args.match_window)
            for _ in range(args.n_random_draws)]
        for k in joint_ks
    }
    feature_info = {
        "top_single": top_single, "random_single": rand_single,
        "top_single_fire_rate": [float(fire_rate[f]) for f in top_single],
        "random_single_fire_rate": [float(fire_rate[f]) for f in rand_single],
        "joint_random_draws": {str(k): v for k, v in joint_draws.items()},
    }
    active_random = sum(fire_rate[f] for f in rand_single)
    logger.info(f"Top {len(top_single)} by {args.rank_by}; {len(rand_single)} random ({args.random_pool}); "
                f"expected active random features per position = {active_random:.2f}")

    # Imported here so the helpers above can be imported (and tested) without
    # the full training stack that activation_ablation pulls in.
    from scripts.core.intervention_runner import InterventionRunner

    task = SudokuTask(args, device) if args.task == "sudoku" else MazeTask(args, device)
    runner = InterventionRunner(task.model, device=device)
    inter = SAEIntervention(sae, args.positions, task.n_cells)

    per_puzzle_runs = (3 + len(single_modes) * (len(top_single) + len(rand_single))
                       + len(joint_modes) * len(joint_ks) * (1 + args.n_random_draws)
                       + (len(joint_ks) if args.norm_control else 0))
    logger.info(f"{len(task.puzzles)} puzzles x {per_puzzle_runs} rollouts each")

    records: List[Dict[str, Any]] = []
    t0 = time.time()
    for pi, (pidx, batch) in enumerate(task.puzzles):
        gen = torch.Generator(device=device).manual_seed(args.seed * 100_003 + pi)
        base = task.score(runner.rollout(batch, None, args.max_steps)["preds"], batch)

        def delta(fn) -> float:
            return task.score(runner.rollout(batch, fn, args.max_steps)["preds"], batch) - base

        rec: Dict[str, Any] = {
            "puzzle_idx": int(pidx), "baseline": base,
            "recon_only": delta(inter.transform("recon_only")),
            "noise_matched": delta(inter.transform("noise_matched", generator=gen)),
            "single": {}, "joint": {},
        }
        for mode in single_modes:
            rec["single"][mode] = {
                "top": [delta(inter.transform(mode, [f])) for f in top_single],
                "random": [delta(inter.transform(mode, [f])) for f in rand_single],
            }
        for mode in joint_modes:
            rec["joint"][mode] = {
                str(k): {"top": delta(inter.transform(mode, order[:k])),
                         "random": [delta(inter.transform(mode, draw)) for draw in joint_draws[k]]}
                for k in joint_ks
            }
        if args.norm_control:
            rec["ep_norm_matched"] = {
                str(k): delta(inter.transform("ep_norm_matched", order[:k], generator=gen))
                for k in joint_ks}
        records.append(rec)
        if (pi + 1) % 10 == 0:
            el = time.time() - t0
            logger.info(f"  {pi + 1}/{len(task.puzzles)} puzzles, {el:.0f}s elapsed, "
                        f"~{el / (pi + 1) * (len(task.puzzles) - pi - 1):.0f}s left | "
                        f"recon_only so far {100 * np.mean([r['recon_only'] for r in records]):+.2f} pp")

    agg = aggregate(records, bound=args.tost_bound_pp / 100.0)
    agg["task"], agg["metric"] = args.task, task.metric
    agg["sae_stats"] = stats
    agg["features"] = feature_info
    agg["settings"] = {k: v for k, v in vars(args).items()}

    with open(os.path.join(args.output_dir, "per_puzzle.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "sae_error_controls", vars(args), repo_root=REPO_ROOT)
    except Exception as e:  # provenance is best-effort, as in the other scripts
        logger.warning(f"Could not write _meta.json: {e}")

    pp = lambda s: f"{100 * s['mean']:+.2f} pp [{100 * s['ci95'][0]:+.2f}, {100 * s['ci95'][1]:+.2f}]"
    print("\n" + "=" * 72)
    print(f"E1 SAE ERROR CONTROLS  task={args.task}  metric={task.metric}  n={len(records)}")
    print("=" * 72)
    print(f"  SAE: L0={stats['L0_mean_active_per_row']:.1f}  dead={100 * stats['dead_frac']:.1f}%  "
          f"FVE={stats['fraction_variance_explained']:.4f}  rel.err={stats['mean_relative_error_norm']:.4f}")
    print(f"  baseline metric        {100 * agg['baseline']['mean']:.2f}%")
    print(f"  recon_only             {pp(agg['recon_only'])}")
    print(f"  noise_matched          {pp(agg['noise_matched'])}")
    for mode, s in agg["single"].items():
        d = s["paired_top_minus_random"]
        print(f"  single/{mode:17s} top {pp(s['top_mean_over_features'])} | "
              f"random {pp(s['random_mean_over_features'])} | top-random "
              f"{100 * d['mean']:+.2f} pp, p={d.get('p_two_sided', float('nan')):.3f}, "
              f"TOST(+/-{args.tost_bound_pp} pp) p={d.get('tost_p', float('nan')):.3f}, "
              f"rejects gaps > {100 * d.get('smallest_rejectable_bound', float('nan')):.2f} pp")
    for mode, by_k in agg["joint"].items():
        for k, s in by_k.items():
            mc = s.get("monte_carlo_rank", {})
            print(f"  joint/{mode:16s} k={k:>3s} top {pp(s['top'])} | random {pp(s['random_pooled'])} | "
                  f"MC rank p={mc.get('p_top_more_damaging', float('nan')):.3f} (R={mc.get('R', 0)})")
    if "ep_norm_matched" in agg:
        for k, s in agg["ep_norm_matched"].items():
            print(f"  ep_norm_matched k={k:>3s}  {pp(s)}")
    print(f"  wrote {args.output_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
