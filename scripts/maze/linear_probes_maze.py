"""Hardened Maze probes on HRM z_H/z_L activations (linear + MLP).

Maze replication of the Sudoku E8 probe protocol
(``scripts/probes/e8_constraint_probes.py``). For *every* (stream, ACT step,
target) we fit an independent probe under a **puzzle-disjoint** train/val split,
repeat over a seed ensemble, and report mean ± 95% t-CI. This mirrors the
Sudoku hardening so the two tasks are directly comparable.

Why this differs from the earlier maze probe run
-------------------------------------------------
The first maze probe script (a) pooled *all* ACT steps into one dataset (no
per-(step, target) breakdown), (b) split rows randomly — for the per-cell
"local" probes that leaks cells of the same puzzle across train/val and inflates
val accuracy, and (c) read ``z_H`` (the *input* carry to a step) rather than
``z_H_out`` (the *output* state the lm_head reads, which is what E8 probes).
This version fixes all three.

Targets (path-validity, not token accuracy)
-------------------------------------------
Global (per-puzzle, decoded from the mean-pooled answer state):
    exact_solved, connects_start_goal, valid_sg_path, valid_optimal_path,
    path_f1, path_jaccard, wall_path_rate, path_length_ratio
Local (per-cell):
    on_optimal_path, is_wall, is_dead_end, is_junction,
    near_start_5, near_goal_5, distance_to_start_norm, distance_to_goal_norm
These cover the requested feature families: **on-optimal-path**, **wall
structure** (is_wall / is_dead_end / is_junction), and **connecting start & goal**
(connects_start_goal / valid_sg_path / near_start / near_goal / distances).

Outputs
-------
  <output_dir>/probe_results.json   full per-(stream,step,target) table
  <output_dir>/probe_summary.json   flat {key: mean ± CI} summary (E8 parity)
  <output_dir>/probe_report.html    human-readable report
  <output_dir>/_meta.json           provenance (git SHA, seeds, N, split, ...)

Example quick run (CPU smoke):
  python scripts/maze/linear_probes_maze.py \
      --num_puzzles 6 --steps 0,1 --epochs 5 --positions_per_sample 64 \
      --seeds 0,1 --device cpu --output_dir /tmp/maze_probe_smoke
"""
from __future__ import annotations

import os
import sys
import json
import math
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from torch import nn

from scripts.controlled.controlled_common import (
    load_model_and_dataloader, collect_puzzles,
)
from scripts.core.activation_ablation import ActivationAblator, ActivationCache
from scripts.maze.maze_common import (
    MAZE_CHECKPOINT, get_puzzle_emb_len, maze_prediction_metrics,
)
from scripts.maze.maze_render_common import html_doc, metrics_table
from utils.maze_targets import (
    SEQ_LEN, UNREACHABLE,
    distance_to_goal, distance_to_start, on_optimal_path,
    is_wall, is_dead_end, is_junction,
    is_free, passable_neighbor_count, on_path_corner, off_path_passable,
)


GLOBAL_TARGETS: Dict[str, str] = {
    "exact_solved": "binary",
    "connects_start_goal": "binary",
    "valid_sg_path": "binary",
    "valid_optimal_path": "binary",
    "path_f1": "regression",
    "path_jaccard": "regression",
    "wall_path_rate": "regression",
    "path_length_ratio": "regression",
}

LOCAL_TARGETS: Dict[str, str] = {
    "on_optimal_path": "binary",
    "is_wall": "binary",
    "is_free": "binary",
    "is_dead_end": "binary",
    "is_junction": "binary",
    "is_corner_on_path": "binary",
    "off_path_passable": "binary",
    "near_start_5": "binary",
    "near_goal_5": "binary",
    "num_passable_neighbors": "regression",
    "distance_to_start_norm": "regression",
    "distance_to_goal_norm": "regression",
}


def _flat(t: torch.Tensor) -> np.ndarray:
    if t.ndim == 2:
        t = t[0]
    return t.detach().to("cpu").to(torch.int64).numpy().reshape(-1)


def _slice_preds(preds: np.ndarray, label_len: int) -> np.ndarray:
    preds = preds.reshape(-1)
    if preds.size > label_len:
        return preds[-label_len:]
    return preds


def _answer_slice(z: torch.Tensor, puzzle_emb_len: int) -> torch.Tensor:
    if z.shape[1] >= puzzle_emb_len + SEQ_LEN:
        return z[:, puzzle_emb_len:puzzle_emb_len + SEQ_LEN, :]
    return z[:, -SEQ_LEN:, :]


def _norm_dist(dist: np.ndarray) -> np.ndarray:
    finite = dist < UNREACHABLE
    out = np.ones(dist.shape, dtype=np.float32)
    if finite.any():
        denom = max(float(dist[finite].max()), 1.0)
        out[finite] = dist[finite].astype(np.float32) / denom
    return out.reshape(-1)


def _local_label_dict(inp: np.ndarray, label: np.ndarray) -> Dict[str, np.ndarray]:
    d_start = distance_to_start(inp)
    d_goal = distance_to_goal(inp)
    return {
        "on_optimal_path": on_optimal_path(label).reshape(-1).astype(np.int64),
        "is_wall": is_wall(inp).reshape(-1).astype(np.int64),
        "is_free": is_free(inp).reshape(-1).astype(np.int64),
        "is_dead_end": is_dead_end(inp).reshape(-1).astype(np.int64),
        "is_junction": is_junction(inp).reshape(-1).astype(np.int64),
        "is_corner_on_path": on_path_corner(label).reshape(-1).astype(np.int64),
        "off_path_passable": off_path_passable(inp, label).reshape(-1).astype(np.int64),
        "near_start_5": ((d_start <= 5) & (d_start >= 0) & (d_start < UNREACHABLE)).reshape(-1).astype(np.int64),
        "near_goal_5": ((d_goal <= 5) & (d_goal >= 0) & (d_goal < UNREACHABLE)).reshape(-1).astype(np.int64),
        "num_passable_neighbors": passable_neighbor_count(inp).reshape(-1).astype(np.float32),
        "distance_to_start_norm": _norm_dist(d_start),
        "distance_to_goal_norm": _norm_dist(d_goal),
    }


def _sample_positions(labels: Dict[str, np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0 or n >= SEQ_LEN:
        return np.arange(SEQ_LEN, dtype=np.int64)

    # Keep local probes from missing the rare path/start/goal cells entirely.
    path_pos = np.nonzero(labels["on_optimal_path"] > 0)[0]
    wall_pos = np.nonzero(labels["is_wall"] > 0)[0]
    keep: List[int] = []
    for pool, budget in ((path_pos, n // 3), (wall_pos, n // 3)):
        if pool.size:
            take = min(pool.size, budget)
            keep.extend(rng.choice(pool, size=take, replace=False).astype(np.int64).tolist())

    remaining = n - len(keep)
    if remaining > 0:
        exclude = np.zeros(SEQ_LEN, dtype=bool)
        if keep:
            exclude[np.asarray(keep, dtype=np.int64)] = True
        pool = np.nonzero(~exclude)[0]
        keep.extend(rng.choice(pool, size=remaining, replace=False).astype(np.int64).tolist())
    return np.asarray(sorted(set(keep)), dtype=np.int64)


def _standardize(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (x_train - mean) / std, (x_test - mean) / std


def _stable_seed(base: int, *parts: str) -> int:
    val = int(base)
    for part in parts:
        for ch in part:
            val = (val * 131 + ord(ch)) % 1000003
    return val


def puzzle_disjoint_split(
    groups: np.ndarray, val_frac: float, seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train/val row indices with a PUZZLE-DISJOINT split.

    ``groups`` is a per-row puzzle id. Unique puzzles are partitioned into
    train/val so no puzzle contributes rows to both sides (prevents per-cell
    leakage that inflates val scores). For the global probes each puzzle is a
    single row, so this reduces to an ordinary split.
    """
    groups = np.asarray(groups).reshape(-1)
    uniq = np.unique(groups)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(uniq)
    n_val_p = max(1, int(round(uniq.size * val_frac)))
    if uniq.size >= 2:
        n_val_p = min(n_val_p, uniq.size - 1)  # keep >=1 puzzle in train
    val_ids = set(perm[:n_val_p].tolist())
    val_mask = np.fromiter((g in val_ids for g in groups), dtype=bool, count=groups.size)
    val_idx = np.nonzero(val_mask)[0]
    train_idx = np.nonzero(~val_mask)[0]
    return train_idx, val_idx


# ───────────────────────────── probe modules ──────────────────────────────

class _LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.head(x)


class _MLPHead(nn.Module):
    """2-layer MLP probe: in_dim -> hidden -> out_dim (ReLU), matches E9b."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.head(x)


def _make_head(probe_type: str, in_dim: int, out_dim: int, hidden_dim: int) -> nn.Module:
    if probe_type == "linear":
        return _LinearHead(in_dim, out_dim)
    if probe_type == "mlp":
        return _MLPHead(in_dim, out_dim, hidden_dim)
    raise ValueError(probe_type)


def _iter_minibatches(n: int, batch_size: int, device: torch.device, shuffle: bool = True):
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def _fit_classifier(
    x: torch.Tensor, y: torch.Tensor, tr_idx: np.ndarray, te_idx: np.ndarray,
    *, probe_type: str, hidden_dim: int, epochs: int, lr: float,
    batch_size: int, fit_seed: int, device: torch.device,
) -> Dict[str, float]:
    y = y.long().view(-1)
    if torch.unique(y).numel() < 2:
        return {"status": "skipped_one_class", "n": int(x.shape[0])}

    tr = torch.from_numpy(np.asarray(tr_idx, dtype=np.int64))
    te = torch.from_numpy(np.asarray(te_idx, dtype=np.int64))
    x_train, x_test = x[tr].float(), x[te].float()
    y_train, y_test = y[tr], y[te]

    train_classes = torch.unique(y_train)
    if train_classes.numel() < 2:
        return {"status": "skipped_train_one_class", "n": int(x.shape[0])}

    x_train, x_test = _standardize(x_train, x_test)
    remap = {int(c.item()): i for i, c in enumerate(train_classes.sort().values)}
    y_train_m = torch.tensor([remap[int(v)] for v in y_train], dtype=torch.long)
    y_test_m = torch.tensor([remap.get(int(v), -1) for v in y_test], dtype=torch.long)
    known = y_test_m >= 0
    if not bool(known.any()):
        return {"status": "skipped_test_unseen_classes", "n": int(x.shape[0])}

    x_train, x_test = x_train.to(device), x_test.to(device)
    y_train_m, y_test_m, known = y_train_m.to(device), y_test_m.to(device), known.to(device)

    torch.manual_seed(fit_seed)
    model = _make_head(probe_type, x_train.shape[1], len(remap), hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for idx in _iter_minibatches(x_train.shape[0], batch_size, device):
            loss = loss_fn(model(x_train[idx]), y_train_m[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(x_train).argmax(1) == y_train_m).float().mean().item()
        test_pred = model(x_test).argmax(1)
        test_acc = (test_pred[known] == y_test_m[known]).float().mean().item()
        majority = torch.mode(y_train_m).values.item()
        baseline = (y_test_m[known] == majority).float().mean().item()

    return {
        "status": "ok", "n": int(x.shape[0]),
        "n_train": int(x_train.shape[0]), "n_test": int(known.sum().item()),
        "train_score": float(train_acc), "test_score": float(test_acc),
        "baseline": float(baseline),
        "positive_rate": float(y.float().mean().item()) if len(remap) == 2 else float("nan"),
    }


def _fit_regressor(
    x: torch.Tensor, y: torch.Tensor, tr_idx: np.ndarray, te_idx: np.ndarray,
    *, probe_type: str, hidden_dim: int, epochs: int, lr: float,
    batch_size: int, fit_seed: int, device: torch.device,
) -> Dict[str, float]:
    y = y.float().view(-1, 1)
    tr = torch.from_numpy(np.asarray(tr_idx, dtype=np.int64))
    te = torch.from_numpy(np.asarray(te_idx, dtype=np.int64))
    x_train, x_test = x[tr].float(), x[te].float()
    y_train, y_test = y[tr], y[te]
    x_train, x_test = _standardize(x_train, x_test)

    x_train, x_test = x_train.to(device), x_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    torch.manual_seed(fit_seed)
    model = _make_head(probe_type, x_train.shape[1], 1, hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for idx in _iter_minibatches(x_train.shape[0], batch_size, device):
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        test_mse = loss_fn(model(x_test), y_test).item()
        var = torch.var(y_test, unbiased=False).item()
        r2 = 0.0 if var <= 1e-12 else 1.0 - (test_mse / var)
        train_mse = loss_fn(model(x_train), y_train).item()
    return {
        "status": "ok", "n": int(x.shape[0]),
        "n_train": int(x_train.shape[0]), "n_test": int(x_test.shape[0]),
        "train_score": float(0.0), "test_score": float(r2),
        "test_mse": float(test_mse), "train_mse": float(train_mse),
        "baseline": float(0.0),
        "target_mean": float(y.mean().item()), "target_std": float(y.std(unbiased=False).item()),
    }


def _fit_probe(x, y, task, tr_idx, te_idx, *, probe_type, **kw) -> Dict[str, float]:
    if task == "binary":
        return _fit_classifier(x, y, tr_idx, te_idx, probe_type=probe_type, **kw)
    if task == "regression":
        return _fit_regressor(x, y, tr_idx, te_idx, probe_type=probe_type, **kw)
    raise ValueError(task)


def _mean_ci(values: List[float], confidence: float = 0.95):
    """Mean + t-based CI over the seed ensemble (mirrors E8 ``mean_ci``)."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0.0
    mean = float(arr.mean())
    if n < 2:
        return mean, mean, mean, 0.0
    std = float(arr.std(ddof=1))
    sem = std / math.sqrt(n)
    try:
        from scipy import stats as _sps
        tcrit = float(_sps.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    except Exception:
        tcrit = 1.96
    half = tcrit * sem
    return mean, mean - half, mean + half, std


def _ensemble_probe(
    x: torch.Tensor, y: torch.Tensor, task: str, groups: np.ndarray,
    *, kind: str, stream: str, step: int, target: str, probe_type: str,
    seeds: List[int], base_seed: int, val_frac: float,
    hidden_dim: int, epochs: int, lr: float, batch_size: int, device: torch.device,
) -> Dict[str, object]:
    """Fit one probe per seed on an independent puzzle-disjoint split; aggregate.

    When ``probe_type == 'mlp'`` an identically-split linear probe is also fit per
    seed so the row carries the linear baseline and the MLP-minus-linear delta
    (the Sudoku non-linear-probe comparison)."""
    scores: List[float] = []
    baselines: List[float] = []
    lin_scores: List[float] = []
    rep: Optional[Dict[str, float]] = None
    for s in seeds:
        split_seed = _stable_seed(base_seed + s, kind, stream, str(step), target)
        tr_idx, te_idx = puzzle_disjoint_split(groups, val_frac, split_seed)
        fit_seed = _stable_seed(base_seed + s, "fit", stream, str(step), target)
        r = _fit_probe(
            x, y, task, tr_idx, te_idx, probe_type=probe_type,
            hidden_dim=hidden_dim, epochs=epochs, lr=lr,
            batch_size=batch_size, fit_seed=fit_seed, device=device,
        )
        if rep is None:
            rep = dict(r)
        if r.get("status") == "ok":
            scores.append(float(r["test_score"]))
            baselines.append(float(r.get("baseline", 0.0)))
            if probe_type == "mlp":
                lr_r = _fit_probe(
                    x, y, task, tr_idx, te_idx, probe_type="linear",
                    hidden_dim=hidden_dim, epochs=epochs, lr=lr,
                    batch_size=batch_size, fit_seed=fit_seed, device=device,
                )
                if lr_r.get("status") == "ok":
                    lin_scores.append(float(lr_r["test_score"]))

    out: Dict[str, object] = dict(rep or {"status": "empty"})
    out.update({"stream": stream, "step": int(step), "target": target,
                "task": task, "probe_type": probe_type})
    if scores:
        m, lo, hi, sd = _mean_ci(scores)
        out.update({
            "score_mean": m, "score_ci_lower": lo, "score_ci_upper": hi,
            "score_std": sd, "score_per_seed": scores,
            "baseline_mean": float(np.mean(baselines)) if baselines else float("nan"),
        })
    out["n_seeds"] = len(scores)
    if probe_type == "mlp" and lin_scores:
        lm, llo, lhi, lsd = _mean_ci(lin_scores)
        out.update({
            "linear_score_mean": lm, "linear_score_ci_lower": llo,
            "linear_score_ci_upper": lhi, "linear_score_per_seed": lin_scores,
        })
        if scores:
            out["delta_mlp_minus_linear"] = float(np.mean(scores) - lm)
    return out


def collect_probe_data(
    *, model, test_loader, device: torch.device, num_puzzles: int,
    steps: List[int], max_steps: int, positions_per_sample: int,
    puzzle_emb_len: int, seed: int,
):
    """Run inference; cache per-(puzzle, step) mean-pooled and per-cell states.

    Uses ``z_*_out`` (the post-step state the readout consumes), aligned with the
    step's prediction — matching the Sudoku E8 protocol.
    """
    ablator = ActivationAblator(model, device=device)
    puzzles = collect_puzzles(test_loader, device, num_puzzles, seed=seed)
    rng = np.random.default_rng(seed)
    want_steps = set(int(s) for s in steps)

    global_rows: List[Dict[str, object]] = []
    local_blocks: List[Dict[str, object]] = []
    solved_by_step: Dict[int, List[float]] = {}

    for n, (puzzle_idx, batch) in enumerate(puzzles):
        cache: Dict[int, ActivationCache] = {}
        ablator.run_and_cache_activations(batch, cache, max_steps=max_steps)
        inp = _flat(batch["inputs"])
        label = _flat(batch["labels"])
        local_labels = _local_label_dict(inp, label)

        for step in sorted(s for s in cache.keys() if s in want_steps):
            act = cache[step]
            preds = _slice_preds(_flat(act.preds), label.size)
            metrics = maze_prediction_metrics(preds, label, inp)
            solved_by_step.setdefault(step, []).append(metrics["exact_solved"])

            z_h = _answer_slice(act.z_H_out, puzzle_emb_len)[0].detach().to("cpu").float()
            z_l = _answer_slice(act.z_L_out, puzzle_emb_len)[0].detach().to("cpu").float()
            global_rows.append({
                "puzzle_idx": puzzle_idx, "step": step,
                "z_H": z_h.mean(dim=0), "z_L": z_l.mean(dim=0),
                "metrics": metrics,
            })

            pos_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + puzzle_idx * 1009 + step)
            pos = _sample_positions(local_labels, positions_per_sample, pos_rng)
            pos_t = torch.from_numpy(pos.astype(np.int64))
            local_blocks.append({
                "puzzle_idx": puzzle_idx, "step": step, "positions": pos,
                "z_H": z_h[pos_t], "z_L": z_l[pos_t],
                "labels": {k: torch.from_numpy(v[pos]) for k, v in local_labels.items()},
            })

        if (n + 1) <= 3 or (n + 1) % 25 == 0:
            print(f"[maze_probes] collected {n + 1}/{len(puzzles)} puzzles")

    return global_rows, local_blocks, solved_by_step


def _build_global_dataset(rows, stream, target):
    x = torch.stack([row[stream] for row in rows], dim=0).float()
    y = torch.tensor([float(row["metrics"][target]) for row in rows], dtype=torch.float32)
    groups = np.asarray([int(row["puzzle_idx"]) for row in rows], dtype=np.int64)
    return x, y, groups


def _build_local_dataset(blocks, stream, target):
    x = torch.cat([b[stream].float() for b in blocks], dim=0)
    y = torch.cat([b["labels"][target] for b in blocks], dim=0)
    groups = np.concatenate([
        np.full(b[stream].shape[0], int(b["puzzle_idx"]), dtype=np.int64) for b in blocks
    ])
    return x, y, groups


def _parse_targets(raw: str, valid: Dict[str, str]) -> List[str]:
    if raw.strip().lower() == "all":
        return list(valid.keys())
    out = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = sorted(set(out) - set(valid))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}. Valid: {sorted(valid)}")
    return out


def _render_report(result: Dict[str, object]) -> str:
    cfg = result["config"]
    body: List[str] = []
    body.append(f"<h1>Maze - {cfg['probe_type'].upper()} Probe Report (hardened)</h1>")
    body.append(
        f"<div class='meta'>num_puzzles={cfg['num_puzzles']} | steps={cfg['steps']} | "
        f"split=puzzle_disjoint | seeds={result['dataset_summary'].get('probe_seeds')} | "
        f"epochs={cfg['epochs']} | generated {datetime.now():%Y-%m-%d %H:%M:%S}</div>"
    )
    body.append("<h2>Dataset summary</h2>")
    summary = result["dataset_summary"]
    body.append(metrics_table([{"field": k, "value": v} for k, v in summary.items()],
                              ["field", "value"]))
    cols = ["stream", "step", "target", "task", "status", "n_seeds",
            "score_mean", "score_ci_lower", "score_ci_upper", "score_std",
            "baseline_mean", "linear_score_mean", "delta_mlp_minus_linear",
            "n_train", "n_test"]
    body.append("<h2>Global probes (per-puzzle)</h2>")
    body.append(metrics_table(result["global_probes"], cols))
    body.append("<h2>Local probes (per-cell)</h2>")
    body.append(metrics_table(result["local_probes"], cols))
    return html_doc("Maze - Probe Report", "".join(body))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=MAZE_CHECKPOINT)
    p.add_argument("--num_puzzles", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--steps", type=str, default="0,1,2,4,8,15",
                   help="Comma-separated ACT steps to probe (per-step probes).")
    p.add_argument("--probe_type", type=str, default="linear", choices=["linear", "mlp"],
                   help="linear = E8-style; mlp = 2-layer MLP + linear baseline + delta.")
    p.add_argument("--mlp_hidden", type=int, default=256)
    p.add_argument("--positions_per_sample", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--streams", type=str, default="z_H,z_L")
    p.add_argument("--global_targets", type=str, default="all")
    p.add_argument("--local_targets", type=str, default="all")
    p.add_argument("--output_dir", type=str, default="results/maze/hardened/linear_probes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4",
                   help="Comma-separated probe-ensemble seeds (each = independent split).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--train_device", type=str, default="auto")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if args.train_device == "auto":
        train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        train_device = torch.device(args.train_device)

    streams = [s.strip() for s in args.streams.split(",") if s.strip()]
    for s in streams:
        if s not in {"z_H", "z_L"}:
            raise ValueError("--streams entries must be z_H or z_L")
    steps = [int(s) for s in args.steps.split(",") if s.strip() != ""]
    global_targets = _parse_targets(args.global_targets, GLOBAL_TARGETS)
    local_targets = _parse_targets(args.local_targets, LOCAL_TARGETS)
    seed_list = [int(s) for s in args.seeds.split(",") if s.strip()] or [0]
    lr_mlp_default = 1e-3
    fit_lr = args.lr if args.probe_type == "linear" else lr_mlp_default

    print(f"[maze_probes] probe_type={args.probe_type} streams={streams} steps={steps} "
          f"seeds={seed_list} split=puzzle_disjoint model_device={device} train_device={train_device}")
    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    pel = get_puzzle_emb_len(model)
    print(f"[maze_probes] puzzle_emb_len={pel}")

    global_rows, local_blocks, solved_by_step = collect_probe_data(
        model=model, test_loader=test_loader, device=device,
        num_puzzles=args.num_puzzles, steps=steps, max_steps=args.max_steps,
        positions_per_sample=args.positions_per_sample, puzzle_emb_len=pel, seed=args.seed,
    )

    n_puzzles_collected = len({r["puzzle_idx"] for r in global_rows})
    result: Dict[str, object] = {
        "config": vars(args),
        "dataset_summary": {
            "n_puzzles_collected": n_puzzles_collected,
            "steps_probed": steps,
            "global_rows": len(global_rows),
            "local_blocks": len(local_blocks),
            "positions_per_sample": args.positions_per_sample,
            "puzzle_emb_len": pel,
            "split": "puzzle_disjoint",
            "probe_seeds": seed_list,
            "exact_solved_by_step": {str(k): float(np.mean(v)) for k, v in sorted(solved_by_step.items())},
        },
        "global_probes": [],
        "local_probes": [],
    }

    common = dict(probe_type=args.probe_type, seeds=seed_list, base_seed=args.seed,
                  val_frac=args.val_frac, hidden_dim=args.mlp_hidden, epochs=args.epochs,
                  lr=fit_lr, batch_size=args.batch_size, device=train_device)

    rows_by_step = {step: [r for r in global_rows if r["step"] == step] for step in steps}
    blocks_by_step = {step: [b for b in local_blocks if b["step"] == step] for step in steps}

    for stream in streams:
        for step in steps:
            rows_s = rows_by_step.get(step, [])
            if not rows_s:
                continue
            for target in global_targets:
                x, y, groups = _build_global_dataset(rows_s, stream, target)
                row = _ensemble_probe(x, y, GLOBAL_TARGETS[target], groups,
                                      kind="global", stream=stream, step=step,
                                      target=target, **common)
                result["global_probes"].append(row)
                sc = row.get("score_mean", float("nan"))
                print(f"[maze_probes] global {stream} s{step} {target}: "
                      f"{row.get('status')} score={sc if isinstance(sc,float) else float('nan'):.4f} "
                      f"(n_seeds={row.get('n_seeds')})")

    for stream in streams:
        for step in steps:
            blocks_s = blocks_by_step.get(step, [])
            if not blocks_s:
                continue
            for target in local_targets:
                x, y, groups = _build_local_dataset(blocks_s, stream, target)
                row = _ensemble_probe(x, y, LOCAL_TARGETS[target], groups,
                                      kind="local", stream=stream, step=step,
                                      target=target, **common)
                result["local_probes"].append(row)
                sc = row.get("score_mean", float("nan"))
                print(f"[maze_probes] local {stream} s{step} {target}: "
                      f"{row.get('status')} score={sc if isinstance(sc,float) else float('nan'):.4f} "
                      f"(n_seeds={row.get('n_seeds')})")

    json_path = os.path.join(args.output_dir, "probe_results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # E8-parity flat summary keyed by stream/step/target.
    summary = {}
    for row in result["global_probes"] + result["local_probes"]:
        key = f"{row['stream']}_step{row['step']}_{row['target']}"
        summary[key] = {k: row.get(k) for k in (
            "stream", "step", "target", "task", "probe_type", "status",
            "score_mean", "score_ci_lower", "score_ci_upper", "score_std",
            "baseline_mean", "linear_score_mean", "delta_mlp_minus_linear",
            "score_per_seed", "n_seeds", "n_train", "n_test")}
    with open(os.path.join(args.output_dir, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    html_path = os.path.join(args.output_dir, "probe_report.html")
    with open(html_path, "w") as f:
        f.write(_render_report(result))

    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, f"maze_{args.probe_type}_probes", {
            "num_puzzles": args.num_puzzles, "n_puzzles_collected": n_puzzles_collected,
            "steps": steps, "max_steps": args.max_steps, "probe_type": args.probe_type,
            "mlp_hidden": args.mlp_hidden, "epochs": args.epochs, "lr": fit_lr,
            "val_frac": args.val_frac, "seeds": seed_list, "streams": streams,
            "positions_per_sample": args.positions_per_sample,
            "split": "puzzle_disjoint", "checkpoint": args.checkpoint,
        }, repo_root=REPO_ROOT)
    except Exception as e:
        print(f"[maze_probes] WARN could not write _meta.json: {e}")

    print(f"[maze_probes] wrote {json_path}")
    print(f"[maze_probes] wrote {html_path}")


if __name__ == "__main__":
    main()
