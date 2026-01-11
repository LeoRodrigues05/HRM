import os
import sys
import argparse
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure repo root is on sys.path when executed as `python scripts/...`.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.train_linear_probes import (  # noqa: E402
    load_probes,
    build_global_dataset,
    build_local_dataset,
    build_global_feature_vector,
    train_binary_probe,
    train_multiclass_probe,
    train_regression_probe,
)


@dataclass
class SweepRow:
    scope: str
    target: str
    task: str
    feature_set: str
    use_z: str
    metric: str
    score: float
    train_n: int
    val_n: int
    in_dim: int


def _split_indices(n: int, val_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(round(n * val_frac))
    n_val = max(1, min(n - 1, n_val))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _eval_binary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(X.to(device))
        preds = (torch.sigmoid(logits) > 0.5).long().view(-1).cpu()
    return float((preds == y.view(-1).long().cpu()).float().mean().item())


def _eval_multiclass(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    device = next(model.parameters()).device
    with torch.no_grad():
        preds = model(X.to(device)).argmax(dim=1).view(-1).cpu()
    return float((preds == y.view(-1).long().cpu()).float().mean().item())


def _eval_r2(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    device = next(model.parameters()).device
    y = y.float().view(-1, 1)
    with torch.no_grad():
        pred = model(X.to(device)).cpu()
    y_cpu = y.cpu()
    ss_res = ((y_cpu - pred) ** 2).sum()
    ss_tot = ((y_cpu - y_cpu.mean()) ** 2).sum().clamp(min=1e-8)
    return float((1.0 - ss_res / ss_tot).item())


def _infer_global_task(target: str) -> str:
    if target in {"pct_filled", "violated_units_total", "violated_rows_count", "violated_cols_count", "violated_boxes_count"}:
        return "regression"
    return "binary"


def _infer_local_task(target: str) -> str:
    if target in {"row_idx", "col_idx"}:
        return "multiclass"
    return "binary"


def _subsample(X: torch.Tensor, y: torch.Tensor, max_n: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if X.shape[0] <= max_n:
        return X, y
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)[:max_n]
    return X[idx], y[idx]


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep linear probes over targets/features for z_H and z_L")
    p.add_argument("--probes_dir", default=os.path.join("results", "probes"))
    p.add_argument("--out_csv", default=os.path.join("results", "probes", "linear_probe_sweep.csv"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_local_samples", type=int, default=200_000)
    p.add_argument("--max_global_samples", type=int, default=50_000)

    p.add_argument(
        "--global_targets",
        default="is_solved,pct_filled,violated_units_total",
        help="Comma-separated global targets to test",
    )
    p.add_argument(
        "--local_targets",
        default="per_cell_correct,is_forced_cell,cell_changed_from_input,row_idx,col_idx",
        help="Comma-separated local targets to test",
    )
    p.add_argument(
        "--global_feature_sets",
        default="z_H,z_L,concat,diff,prod,concat_norms",
        help="Comma-separated global feature sets",
    )
    p.add_argument(
        "--local_feature_sets",
        default="z_only,concat,diff,prod,z_and_norms",
        help="Comma-separated local feature sets",
    )
    p.add_argument("--no_step", action="store_true", help="Do not add step scalar as feature")
    p.add_argument("--no_phase", action="store_true", help="Do not add phase scalar as feature")
    p.add_argument("--no_position", action="store_true", help="Do not add row/col/box one-hots as local features")

    args = p.parse_args()

    global_samples, local_samples, _index = load_probes(args.probes_dir)

    global_targets = [s.strip() for s in args.global_targets.split(",") if s.strip()]
    local_targets = [s.strip() for s in args.local_targets.split(",") if s.strip()]
    global_feature_sets = [s.strip() for s in args.global_feature_sets.split(",") if s.strip()]
    local_feature_sets = [s.strip() for s in args.local_feature_sets.split(",") if s.strip()]

    add_step_feature = not args.no_step
    add_phase_feature = not args.no_phase
    add_position_features = not args.no_position

    results: List[SweepRow] = []

    # --- Global sweep ---
    for target in global_targets:
        task = _infer_global_task(target)
        for feature_set in global_feature_sets:
            # Build dataset
            try:
                X, y = build_global_dataset(
                    global_samples,
                    target_key=target,
                    feature_set=feature_set,
                    add_step_feature=add_step_feature,
                    add_phase_feature=add_phase_feature,
                )
            except Exception:
                continue

            X, y = _subsample(X, y, max_n=args.max_global_samples, seed=args.seed)
            train_idx, val_idx = _split_indices(X.shape[0], args.val_frac, args.seed)
            Xtr, ytr = X[train_idx], y[train_idx]
            Xva, yva = X[val_idx], y[val_idx]

            if task == "regression":
                model, _ = train_regression_probe(Xtr, ytr.float(), epochs=max(args.epochs, 50), lr=args.lr)
                score = _eval_r2(model, Xva, yva.float())
                metric = "r2"
            else:
                # interpret as binary
                model, _ = train_binary_probe(Xtr, ytr, epochs=args.epochs, lr=args.lr)
                score = _eval_binary(model, Xva, (yva > 0.5).long())
                metric = "acc"

            results.append(
                SweepRow(
                    scope="global",
                    target=target,
                    task=task,
                    feature_set=feature_set,
                    use_z="-",
                    metric=metric,
                    score=float(score),
                    train_n=int(Xtr.shape[0]),
                    val_n=int(Xva.shape[0]),
                    in_dim=int(X.shape[1]),
                )
            )

    # --- Local sweep ---
    if not local_samples:
        print("NOTE: probe_local.pt not found; skipping local sweep.")
    else:
        for target in local_targets:
            task = _infer_local_task(target)
            for feature_set in local_feature_sets:
                use_z_values = ["z_H", "z_L"] if feature_set in {"z_only", "z_and_norms"} else ["z_L"]
                for use_z in use_z_values:
                    try:
                        X, y = build_local_dataset(
                            local_samples,
                            use_z=use_z,
                            feature_set=feature_set,
                            target_key=target,
                            add_position_features=add_position_features,
                            add_step_feature=add_step_feature,
                            add_phase_feature=add_phase_feature,
                        )
                    except Exception:
                        continue

                    X, y = _subsample(X, y, max_n=args.max_local_samples, seed=args.seed)
                    train_idx, val_idx = _split_indices(X.shape[0], args.val_frac, args.seed)
                    Xtr, ytr = X[train_idx], y[train_idx]
                    Xva, yva = X[val_idx], y[val_idx]

                    if task == "multiclass":
                        model, _ = train_multiclass_probe(Xtr, ytr, num_classes=9, epochs=args.epochs, lr=args.lr)
                        score = _eval_multiclass(model, Xva, yva)
                        metric = "acc"
                    else:
                        model, _ = train_binary_probe(Xtr, ytr, epochs=args.epochs, lr=args.lr)
                        score = _eval_binary(model, Xva, yva)
                        metric = "acc"

                    results.append(
                        SweepRow(
                            scope="local",
                            target=target,
                            task=task,
                            feature_set=feature_set,
                            use_z=use_z,
                            metric=metric,
                            score=float(score),
                            train_n=int(Xtr.shape[0]),
                            val_n=int(Xva.shape[0]),
                            in_dim=int(X.shape[1]),
                        )
                    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "target", "task", "feature_set", "use_z", "metric", "score", "train_n", "val_n", "in_dim"])
        for r in results:
            w.writerow([r.scope, r.target, r.task, r.feature_set, r.use_z, r.metric, f"{r.score:.6f}", r.train_n, r.val_n, r.in_dim])

    # Print a compact summary to stdout
    results_sorted = sorted(results, key=lambda r: (r.scope, r.target, -r.score, r.feature_set, r.use_z))
    print(f"Wrote {len(results_sorted)} rows to {args.out_csv}")
    for r in results_sorted[:25]:
        print(f"{r.scope:6s} {r.target:28s} {r.feature_set:12s} {r.use_z:3s} {r.metric}:{r.score:.4f} (val_n={r.val_n})")


if __name__ == "__main__":
    main()
