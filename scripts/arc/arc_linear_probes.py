"""ARC-AGI probes on HRM z_H/z_L activations (linear + MLP).

ARC replication of the Sudoku E8 / Maze probe protocol
(``scripts/probes/e8_constraint_probes.py``,
``scripts/maze/linear_probes_maze.py``). For *every* (stream, ACT step, target)
we fit an independent probe under a **puzzle-disjoint** train/val split, repeat
over a seed ensemble, and report mean ± 95% t-CI. This mirrors the Sudoku and
Maze hardening so all three tasks are directly comparable.

ARC-native targets (see ``utils/arc_targets.py``)
-------------------------------------------------
Global (per-puzzle, decoded from the mean-pooled answer state):
    exact_solved, shape_correct, size_preserved          (binary)
    token_acc, colour_cell_acc, colour_iou,
    input_height, input_width, output_height, output_width,
    num_input_colours, num_output_colours                (regression)
Local (per-cell):
    per_cell_correct, input_is_background, input_inside_grid,
    output_inside_grid, is_eos, colour_changed, same_as_input,
    is_object_boundary                                   (binary)
    input_component_size, num_same_colour_neighbours     (regression)
    input_colour, output_colour                          (multiclass, 10-way)

These cover ARC-native feature families: **grid geometry** (heights/widths,
shape/size preservation), **colour structure** (colour identity, distinct-colour
counts, colour IoU), **object structure** (connected-component size, object
boundaries, same-colour neighbours), and **the transformation itself**
(colour_changed / same_as_input / per_cell_correct).

Outputs
-------
  <output_dir>/probe_results.json   full per-(stream,step,target) table
  <output_dir>/probe_summary.json   flat {key: mean ± CI} summary (E8 parity)
  <output_dir>/probe_report.html    human-readable report
  <output_dir>/probe_weights.pt     (if --save_probe_weights) directions for E9
  <output_dir>/_meta.json           provenance

Example quick run (CPU smoke):
  python scripts/arc/arc_linear_probes.py \
      --num_puzzles 6 --steps 0,1 --epochs 5 --positions_per_sample 64 \
      --seeds 0,1 --device cpu --output_dir /tmp/arc_probe_smoke
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from torch import nn

from scripts.controlled.controlled_common import (
    load_model_and_dataloader, collect_puzzles,
)
from scripts.core.activation_ablation import ActivationAblator, ActivationCache
from scripts.arc.arc_common import (
    ARC_CHECKPOINT, get_puzzle_emb_len, arc_global_features,
)
# Reuse the task-agnostic fitting primitives (puzzle-disjoint split, classifier /
# regressor fits, seed-ensemble CI) from the maze driver.
from scripts.maze.linear_probes_maze import (
    _flat, _slice_preds, _standardize, _stable_seed, _mean_ci,
    puzzle_disjoint_split, _fit_classifier, _fit_regressor,
    _build_global_dataset, _build_local_dataset, _parse_targets,
)
from utils.arc_targets import (
    SEQ_LEN, ALL_TARGETS, PER_CELL_BINARY, PER_CELL_REGRESSION,
    PER_CELL_MULTICLASS, inside_grid_mask, object_boundary_mask,
    derive_per_cell_labels,
)
from scripts.maze.maze_render_common import html_doc, metrics_table


GLOBAL_TARGETS: Dict[str, str] = {
    "exact_solved": "binary",
    "shape_correct": "binary",
    "size_preserved": "binary",
    "token_acc": "regression",
    "colour_cell_acc": "regression",
    "colour_iou": "regression",
    "input_height": "regression",
    "input_width": "regression",
    "output_height": "regression",
    "output_width": "regression",
    "num_input_colours": "regression",
    "num_output_colours": "regression",
}

LOCAL_TARGETS: Dict[str, str] = {
    **{t: "binary" for t in PER_CELL_BINARY},
    **{t: "regression" for t in PER_CELL_REGRESSION},
    **{t: "multiclass" for t in PER_CELL_MULTICLASS},
}


def _answer_slice(z: torch.Tensor, puzzle_emb_len: int) -> torch.Tensor:
    if z.shape[1] >= puzzle_emb_len + SEQ_LEN:
        return z[:, puzzle_emb_len:puzzle_emb_len + SEQ_LEN, :]
    return z[:, -SEQ_LEN:, :]


def _sample_positions(inp_flat: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample cell positions, oversampling inside-grid and object-boundary cells.

    The 30x30 canvas is mostly PAD; uniform sampling would starve the local probes
    of the colour cells that carry the ARC-native structure. We therefore keep a
    budget of boundary + inside-grid cells and fill the rest uniformly.
    """
    if n <= 0 or n >= SEQ_LEN:
        return np.arange(SEQ_LEN, dtype=np.int64)
    inside = np.nonzero(inside_grid_mask(inp_flat))[0]
    boundary = np.nonzero(object_boundary_mask(inp_flat))[0]
    keep: List[int] = []
    for pool, budget in ((boundary, n // 4), (inside, n // 2)):
        if pool.size:
            take = min(pool.size, budget)
            keep.extend(rng.choice(pool, size=take, replace=False).astype(np.int64).tolist())
    remaining = n - len(keep)
    if remaining > 0:
        exclude = np.zeros(SEQ_LEN, dtype=bool)
        if keep:
            exclude[np.asarray(keep, dtype=np.int64)] = True
        pool = np.nonzero(~exclude)[0]
        take = min(pool.size, remaining)
        keep.extend(rng.choice(pool, size=take, replace=False).astype(np.int64).tolist())
    return np.asarray(sorted(set(keep)), dtype=np.int64)


def _fit_probe(x, y, task, tr_idx, te_idx, *, probe_type, **kw) -> Dict[str, float]:
    """Route binary/multiclass -> classifier, regression -> regressor."""
    if task in ("binary", "multiclass"):
        return _fit_classifier(x, y, tr_idx, te_idx, probe_type=probe_type, **kw)
    if task == "regression":
        return _fit_regressor(x, y, tr_idx, te_idx, probe_type=probe_type, **kw)
    raise ValueError(task)


def _ensemble_probe(
    x: torch.Tensor, y: torch.Tensor, task: str, groups: np.ndarray,
    *, kind: str, stream: str, step: int, target: str, probe_type: str,
    seeds: List[int], base_seed: int, val_frac: float,
    hidden_dim: int, epochs: int, lr: float, batch_size: int, device: torch.device,
) -> Dict[str, object]:
    """Fit one probe per seed on an independent puzzle-disjoint split; aggregate.

    For ``probe_type == 'mlp'`` an identically-split linear probe is also fit per
    seed so the row carries the linear baseline and the MLP-minus-linear delta
    (the non-linear-probe comparison from Sudoku E9b / Maze)."""
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


def _fit_direction(
    x: torch.Tensor, y: torch.Tensor, *, base_seed: int, val_frac: float,
    epochs: int, lr: float, batch_size: int, device: torch.device,
) -> Optional[Dict[str, object]]:
    """Train a single linear binary probe on a fixed split; return its unit
    weight direction (positive-class normal) and held-out accuracy.

    Used to export probe directions for the E9 directed-ablation script. Returns
    None if the target is single-class.
    """
    y = y.long().view(-1)
    if torch.unique(y).numel() < 2:
        return None
    tr_idx, te_idx = puzzle_disjoint_split(
        np.arange(x.shape[0]), val_frac, base_seed)
    x_tr, x_te = x[tr_idx].float(), x[te_idx].float()
    y_tr, y_te = y[tr_idx], y[te_idx]
    if torch.unique(y_tr).numel() < 2:
        return None
    x_tr, x_te = _standardize(x_tr, x_te)
    torch.manual_seed(base_seed)
    head = nn.Linear(x_tr.shape[1], 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    head.train()
    n = x_tr.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for st in range(0, n, batch_size):
            idx = order[st:st + batch_size]
            loss = loss_fn(head(x_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        acc = (head(x_te.to(device)).argmax(1) == y_te.to(device)).float().mean().item()
    # Positive-vs-negative class direction.
    W = head.weight.detach().to("cpu")           # [2, hidden]
    direction = (W[1] - W[0]).float()
    return {"W_mean": direction, "val_acc_mean": float(acc)}


def collect_probe_data(
    *, model, test_loader, device: torch.device, num_puzzles: int,
    steps: List[int], max_steps: int, positions_per_sample: int,
    puzzle_emb_len: int, seed: int,
):
    """Run inference; cache per-(puzzle, step) mean-pooled and per-cell states.

    Uses ``z_*_out`` (the post-step state the readout consumes) aligned with the
    step's prediction, matching the Sudoku E8 / Maze protocol.
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

        for step in sorted(s for s in cache.keys() if s in want_steps):
            act = cache[step]
            preds = _slice_preds(_flat(act.preds), label.size)
            gfeat = arc_global_features(preds, label, inp)
            solved_by_step.setdefault(step, []).append(gfeat["exact_solved"])

            z_h = _answer_slice(act.z_H_out, puzzle_emb_len)[0].detach().to("cpu").float()
            z_l = _answer_slice(act.z_L_out, puzzle_emb_len)[0].detach().to("cpu").float()
            global_rows.append({
                "puzzle_idx": puzzle_idx, "step": step,
                "z_H": z_h.mean(dim=0), "z_L": z_l.mean(dim=0),
                "metrics": gfeat,
            })

            # Per-cell labels from the ARC-native derivation.
            local_labels = {k: v[0] for k, v in derive_per_cell_labels(
                preds[None, :], label[None, :], inp[None, :]).items()}

            pos_rng = np.random.default_rng(
                int(rng.integers(0, 2**31 - 1)) + puzzle_idx * 1009 + step)
            pos = _sample_positions(inp, positions_per_sample, pos_rng)
            pos_t = torch.from_numpy(pos.astype(np.int64))
            local_blocks.append({
                "puzzle_idx": puzzle_idx, "step": step, "positions": pos,
                "z_H": z_h[pos_t], "z_L": z_l[pos_t],
                "labels": {k: torch.from_numpy(np.asarray(v)[pos].astype(np.int64))
                           for k, v in local_labels.items()},
            })

        if (n + 1) <= 3 or (n + 1) % 25 == 0:
            print(f"[arc_probes] collected {n + 1}/{len(puzzles)} puzzles")

    return global_rows, local_blocks, solved_by_step


def _render_report(result: Dict[str, object]) -> str:
    cfg = result["config"]
    body: List[str] = []
    body.append(f"<h1>ARC-AGI - {cfg['probe_type'].upper()} Probe Report (hardened)</h1>")
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
    return html_doc("ARC-AGI - Probe Report", "".join(body))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=ARC_CHECKPOINT)
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
    p.add_argument("--output_dir", type=str, default="results/arc/hardened/linear_probes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4",
                   help="Comma-separated probe-ensemble seeds (each = independent split).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--train_device", type=str, default="auto")
    p.add_argument("--save_probe_weights", action="store_true",
                   help="Export per-binary-feature linear directions for E9 directed ablation.")
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
    fit_lr = args.lr if args.probe_type == "linear" else 1e-3

    print(f"[arc_probes] probe_type={args.probe_type} streams={streams} steps={steps} "
          f"seeds={seed_list} split=puzzle_disjoint model_device={device} train_device={train_device}")
    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    pel = get_puzzle_emb_len(model)
    print(f"[arc_probes] puzzle_emb_len={pel}")

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
                print(f"[arc_probes] global {stream} s{step} {target}: "
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
                print(f"[arc_probes] local {stream} s{step} {target}: "
                      f"{row.get('status')} score={sc if isinstance(sc,float) else float('nan'):.4f} "
                      f"(n_seeds={row.get('n_seeds')})")

    json_path = os.path.join(args.output_dir, "probe_results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # E8-parity flat summary. Regression R^2 can diverge on near-constant targets;
    # tag those non-reportable. For binary/multiclass, record headroom over the
    # majority baseline so probes pinned at baseline are not mistaken for signal.
    summary = {}
    for row in result["global_probes"] + result["local_probes"]:
        key = f"{row['stream']}_step{row['step']}_{row['target']}"
        entry = {k: row.get(k) for k in (
            "stream", "step", "target", "task", "probe_type", "status",
            "score_mean", "score_ci_lower", "score_ci_upper", "score_std",
            "baseline_mean", "linear_score_mean", "delta_mlp_minus_linear",
            "score_per_seed", "n_seeds", "n_train", "n_test")}
        sc, bl = entry.get("score_mean"), entry.get("baseline_mean")
        if entry.get("task") == "regression":
            entry["reportable"] = False
            entry["note"] = "R^2 unreliable on near-constant target; not a decodability measure"
        else:
            entry["reportable"] = True
            entry["headroom"] = (None if (sc is None or bl is None) else round(float(sc) - float(bl), 4))
            entry["informative"] = (entry["headroom"] is not None and entry["headroom"] >= 0.03)
        summary[key] = entry
    with open(os.path.join(args.output_dir, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    html_path = os.path.join(args.output_dir, "probe_report.html")
    with open(html_path, "w") as f:
        f.write(_render_report(result))

    # Export per-binary-feature linear directions for E9 directed ablation.
    if args.save_probe_weights:
        weights: Dict[str, object] = {}
        binary_local = [t for t in local_targets if LOCAL_TARGETS[t] == "binary"]
        for stream in streams:
            for step in steps:
                blocks_s = blocks_by_step.get(step, [])
                if not blocks_s:
                    continue
                for target in binary_local:
                    x, y, _ = _build_local_dataset(blocks_s, stream, target)
                    d = _fit_direction(
                        x, y, base_seed=args.seed, val_frac=args.val_frac,
                        epochs=args.epochs, lr=fit_lr, batch_size=args.batch_size,
                        device=train_device)
                    if d is None:
                        continue
                    weights[f"{stream}_step{step}_{target}"] = {
                        "stream": stream, "step": int(step), "target": target,
                        **d,
                    }
        torch.save(weights, os.path.join(args.output_dir, "probe_weights.pt"))
        print(f"[arc_probes] wrote {len(weights)} probe directions -> probe_weights.pt")

    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, f"arc_{args.probe_type}_probes", {
            "num_puzzles": args.num_puzzles, "n_puzzles_collected": n_puzzles_collected,
            "steps": steps, "max_steps": args.max_steps, "probe_type": args.probe_type,
            "mlp_hidden": args.mlp_hidden, "epochs": args.epochs, "lr": fit_lr,
            "val_frac": args.val_frac, "seeds": seed_list, "streams": streams,
            "positions_per_sample": args.positions_per_sample,
            "split": "puzzle_disjoint", "checkpoint": args.checkpoint,
        }, repo_root=REPO_ROOT)
    except Exception as e:
        print(f"[arc_probes] WARN could not write _meta.json: {e}")

    print(f"[arc_probes] wrote {json_path}")
    print(f"[arc_probes] wrote {html_path}")


if __name__ == "__main__":
    main()
