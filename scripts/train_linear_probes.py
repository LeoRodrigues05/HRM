import os
import argparse
import json
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim


def load_probes(probes_dir: str):
    global_path = os.path.join(probes_dir, "probe_global.pt")
    local_path = os.path.join(probes_dir, "probe_local.pt")
    index_path = os.path.join(probes_dir, "probe_index.json")

    if not os.path.exists(global_path):
        raise FileNotFoundError(f"Missing {global_path}")

    global_samples = torch.load(global_path)
    local_samples = []
    if os.path.exists(local_path):
        local_samples = torch.load(local_path)
    index = {}
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
    return global_samples, local_samples, index


def _as_tensor(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x
    try:
        return torch.as_tensor(x)
    except Exception:
        return None


def list_available_label_keys(samples: List[Dict]) -> List[str]:
    keys = set()
    for row in samples:
        labels = row.get("labels", {}) or {}
        if isinstance(labels, dict):
            keys.update(labels.keys())
    return sorted(keys)


def _phase_to_scalar(phase: Any) -> float:
    # Common phases: "grad" / "nograd". Default to 0.
    if phase is None:
        return 0.0
    s = str(phase).lower()
    return 1.0 if s == "grad" else 0.0


def build_global_feature_vector(row: Dict, feature_set: str = "z_H") -> Optional[torch.Tensor]:
    """Return a 1D feature vector for a global probe row.

    Supported feature_set:
      - z_H | z_L
      - concat (z_H || z_L)
      - diff (z_H - z_L)
      - prod (z_H * z_L)
      - concat_norms (z_H || z_L || ||z_H|| || ||z_L||)
    """
    z_H = _as_tensor(row.get("z_H"))
    z_L = _as_tensor(row.get("z_L"))
    if z_H is None or z_L is None:
        return None

    # Unify to [D]
    if z_H.ndim == 2:
        z_H = z_H.mean(dim=0)
    if z_L.ndim == 2:
        z_L = z_L.mean(dim=0)
    if z_H.ndim != 1 or z_L.ndim != 1:
        return None

    if feature_set == "z_H":
        return z_H.float()
    if feature_set == "z_L":
        return z_L.float()
    if feature_set == "concat":
        return torch.cat([z_H, z_L], dim=0).float()
    if feature_set == "diff":
        return (z_H - z_L).float()
    if feature_set == "prod":
        return (z_H * z_L).float()
    if feature_set == "concat_norms":
        nH = torch.linalg.vector_norm(z_H.float()).view(1)
        nL = torch.linalg.vector_norm(z_L.float()).view(1)
        return torch.cat([z_H.float(), z_L.float(), nH, nL], dim=0)

    raise ValueError(f"Unknown global feature_set: {feature_set}")


def build_local_feature_matrix(
    row: Dict,
    *,
    use_z: str = "z_L",
    feature_set: str = "z_only",
    add_position_features: bool = True,
    add_step_feature: bool = True,
    add_phase_feature: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (X, y) for a single local sample row.

    - X: [N, F]
    - y: [N]

    Aligns z token length to y length by taking the last `len(y)` tokens.

    Supported feature_set:
      - z_only: selected z only
      - concat: concat z_H and z_L per token
      - diff: z_H - z_L per token
      - prod: z_H * z_L per token
      - z_and_norms: selected z plus per-token norm(s)
    """
    labels = row.get("labels", {}) or {}
    per_cell = _as_tensor(labels.get("per_cell_correct"))
    if per_cell is None:
        return None, None

    z_H = _as_tensor(row.get("z_H"))
    z_L = _as_tensor(row.get("z_L"))
    if z_H is None or z_L is None:
        return None, None

    # z tensors may be [B,T,D] or [T,D]. per_cell is usually [B,81] or [81].
    # Flatten batch for y.
    y = per_cell
    if y.ndim > 1:
        y = y.reshape(-1)
    y = y.long()

    def _flatten_z(z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 3:
            return z.reshape(-1, z.shape[-1])
        if z.ndim == 2:
            return z
        raise ValueError(f"Unexpected z ndim: {z.ndim}")

    zH2 = _flatten_z(z_H).float()
    zL2 = _flatten_z(z_L).float()

    # Align token count: use last N tokens to match y length.
    N = int(y.numel())
    if zH2.shape[0] < N or zL2.shape[0] < N:
        return None, None
    zH2 = zH2[-N:, :]
    zL2 = zL2[-N:, :]

    # Base per-token representation
    if feature_set == "z_only":
        zbase = zL2 if use_z == "z_L" else zH2
    elif feature_set == "concat":
        zbase = torch.cat([zH2, zL2], dim=1)
    elif feature_set == "diff":
        zbase = zH2 - zL2
    elif feature_set == "prod":
        zbase = zH2 * zL2
    elif feature_set == "z_and_norms":
        zsel = zL2 if use_z == "z_L" else zH2
        n = torch.linalg.vector_norm(zsel, dim=1, keepdim=True)
        zbase = torch.cat([zsel, n], dim=1)
    else:
        raise ValueError(f"Unknown local feature_set: {feature_set}")

    feats: List[torch.Tensor] = [zbase]

    # Sudoku position features (row/col/box one-hot) for indices 0..80
    if add_position_features:
        idx = torch.arange(N, dtype=torch.long)
        row_idx = (idx // 9).clamp(0, 8)
        col_idx = (idx % 9).clamp(0, 8)
        box_idx = ((row_idx // 3) * 3 + (col_idx // 3)).clamp(0, 8)

        row_oh = torch.nn.functional.one_hot(row_idx, num_classes=9).float()
        col_oh = torch.nn.functional.one_hot(col_idx, num_classes=9).float()
        box_oh = torch.nn.functional.one_hot(box_idx, num_classes=9).float()
        feats.extend([row_oh, col_oh, box_oh])

    if add_step_feature:
        step_val = float(row.get("step", 0))
        feats.append(torch.full((N, 1), step_val, dtype=torch.float32))

    if add_phase_feature:
        phase_val = _phase_to_scalar(row.get("phase"))
        feats.append(torch.full((N, 1), phase_val, dtype=torch.float32))

    X = torch.cat(feats, dim=1)
    return X, y


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)


def train_binary_probe(X: torch.Tensor, y: torch.Tensor, epochs: int = 50, lr: float = 1e-2) -> Tuple[LinearProbe, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearProbe(X.shape[1], 1).to(device)
    X = X.to(device)
    y = y.to(device).float().view(-1, 1)

    criterion = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        logits = model(X)
        loss = criterion(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate on training set
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X)) > 0.5).long().view(-1)
        acc = (preds == y.view(-1).long()).float().mean().item()
    return model, acc


def train_regression_probe(X: torch.Tensor, y: torch.Tensor, epochs: int = 100, lr: float = 1e-2) -> Tuple[LinearProbe, float]:
    """Train a linear regression probe; returns (model, R^2 on training set)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearProbe(X.shape[1], 1).to(device)
    X = X.to(device)
    y = y.to(device).float().view(-1, 1)

    criterion = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        pred = model(X)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(X)
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum().clamp(min=1e-8)
        r2 = (1.0 - ss_res / ss_tot).item()
    return model, float(r2)


def train_multiclass_probe(X: torch.Tensor, y: torch.Tensor, num_classes: int, epochs: int = 50, lr: float = 1e-2) -> Tuple[LinearProbe, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearProbe(X.shape[1], num_classes).to(device)
    X = X.to(device)
    y = y.to(device).long().view(-1)

    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        logits = model(X)
        loss = criterion(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return model, acc


def build_global_dataset(
    global_samples: List[Dict],
    *,
    target_key: str = "is_solved",
    feature_set: str = "z_H",
    add_step_feature: bool = True,
    add_phase_feature: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a global probe dataset.

    Features are derived from z_H/z_L entries.
    Labels are pulled from row['labels'][target_key].
    """
    X_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []

    for row in global_samples:
        labels = row.get("labels", {}) or {}
        y_raw = labels.get(target_key)
        y_t = _as_tensor(y_raw)
        if y_t is None:
            continue

        x = build_global_feature_vector(row, feature_set=feature_set)
        if x is None:
            continue

        # Optional scalar features
        extra: List[torch.Tensor] = []
        if add_step_feature:
            extra.append(torch.tensor([float(row.get("step", 0))], dtype=torch.float32))
        if add_phase_feature:
            extra.append(torch.tensor([_phase_to_scalar(row.get("phase"))], dtype=torch.float32))
        if extra:
            x = torch.cat([x, *extra], dim=0)

        # Reduce y to scalar (keep raw numeric value; caller chooses task type)
        if y_t.ndim == 0:
            y_val = float(y_t.item())
        else:
            y_val = float(y_t.float().mean().item())

        X_list.append(x.float())
        y_list.append(torch.tensor(y_val, dtype=torch.float32))

    if not X_list:
        raise ValueError(f"No global samples produced for target_key={target_key} feature_set={feature_set}")

    X = torch.stack(X_list, dim=0)
    y = torch.stack(y_list, dim=0)
    return X, y


def build_local_dataset(
    local_samples: List[Dict],
    *,
    use_z: str = "z_L",
    feature_set: str = "z_only",
    target_key: str = "per_cell_correct",
    add_position_features: bool = True,
    add_step_feature: bool = True,
    add_phase_feature: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []

    for row in local_samples:
        X, y_align = build_local_feature_matrix(
            row,
            use_z=use_z,
            feature_set=feature_set,
            add_position_features=add_position_features,
            add_step_feature=add_step_feature,
            add_phase_feature=add_phase_feature,
        )
        if X is None or y_align is None:
            continue

        labels = row.get("labels", {}) or {}
        y_t = _as_tensor(labels.get(target_key))
        if y_t is None:
            continue
        if y_t.ndim > 1:
            y_t = y_t.reshape(-1)
        y_t = y_t.long()

        # Align labels to the last N tokens (same convention as feature alignment).
        N = int(X.shape[0])
        if y_t.numel() < N:
            continue
        y_t = y_t[-N:]

        X_list.append(X)
        y_list.append(y_t)

    if not X_list:
        raise ValueError("No local samples produced; check probe collection outputs")

    Xall = torch.cat(X_list, dim=0)
    yall = torch.cat(y_list, dim=0)
    return Xall, yall


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on HRM hidden states")
    parser.add_argument("--probes_dir", default=os.path.join("results", "probes"), help="Directory containing probe_global.pt and probe_local.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    # Backward compatible (kept): which single z tensor to use for the default feature_set
    parser.add_argument("--use_global_z", choices=["z_H", "z_L"], default="z_H")
    parser.add_argument("--use_local_z", choices=["z_H", "z_L"], default="z_L")

    # New: richer feature sets
    parser.add_argument(
        "--global_feature_set",
        choices=["z_H", "z_L", "concat", "diff", "prod", "concat_norms"],
        default=None,
        help="Global feature set; default uses --use_global_z.",
    )
    parser.add_argument(
        "--local_feature_set",
        choices=["z_only", "concat", "diff", "prod", "z_and_norms"],
        default="z_only",
        help="Local feature set; default uses --use_local_z for z_only/z_and_norms.",
    )
    parser.add_argument("--global_target", default="is_solved", help="Global label key to predict (e.g. is_solved)")
    parser.add_argument(
        "--global_task",
        choices=["binary", "regression"],
        default=None,
        help="Global probe task. If omitted, inferred from target name.",
    )
    parser.add_argument(
        "--local_target",
        default="per_cell_correct",
        help="Local label key to predict (e.g. per_cell_correct, is_forced_cell, cell_changed_from_input, row_idx, col_idx)",
    )
    parser.add_argument(
        "--local_task",
        choices=["binary", "multiclass"],
        default=None,
        help="Local probe task. If omitted, inferred from target name.",
    )
    parser.add_argument("--add_step_feature", action="store_true", help="Add step index as an extra feature")
    parser.add_argument("--add_phase_feature", action="store_true", help="Add phase (grad=1/nograd=0) as an extra feature")
    parser.add_argument("--add_position_features", action="store_true", help="Add Sudoku row/col/box one-hot position features for local probes")
    parser.add_argument("--list_label_keys", action="store_true", help="Print available label keys in probe files and exit")
    args = parser.parse_args()

    global_samples, local_samples, _index = load_probes(args.probes_dir)

    if args.list_label_keys:
        gk = list_available_label_keys(global_samples)
        lk = list_available_label_keys(local_samples)
        print("Available global label keys:", gk)
        if local_samples:
            print("Available local  label keys:", lk)
        else:
            print("Available local  label keys: (no probe_local.pt found)")
        return

    global_feature_set = args.global_feature_set or args.use_global_z

    # Global: configurable target
    Xg, yg = build_global_dataset(
        global_samples,
        target_key=args.global_target,
        feature_set=global_feature_set,
        add_step_feature=args.add_step_feature,
        add_phase_feature=args.add_phase_feature,
    )

    global_task = args.global_task
    if global_task is None:
        global_task = "regression" if args.global_target in {"pct_filled", "violated_units_total", "violated_rows_count", "violated_cols_count", "violated_boxes_count"} else "binary"

    if global_task == "regression":
        global_model, global_score = train_regression_probe(Xg, yg.float(), epochs=max(args.epochs, 50), lr=args.lr)
        print(f"Global probe ({global_feature_set}) {args.global_target} R^2(train): {global_score:.4f}")
    else:
        # yg is float, typically in {0,1}
        global_model, global_score = train_binary_probe(Xg, yg, epochs=args.epochs, lr=args.lr)
        print(f"Global probe ({global_feature_set}) {args.global_target} accuracy: {global_score:.4f}")

    local_model = None
    local_acc = None
    if not local_samples:
        print("NOTE: probe_local.pt not found; skipping local probe training.")
    else:
        local_task = args.local_task
        if local_task is None:
            local_task = "multiclass" if args.local_target in {"row_idx", "col_idx"} else "binary"

        # Local: configurable target
        Xl, yl = build_local_dataset(
            local_samples,
            use_z=args.use_local_z,
            feature_set=args.local_feature_set,
            target_key=args.local_target,
            add_position_features=args.add_position_features,
            add_step_feature=args.add_step_feature,
            add_phase_feature=args.add_phase_feature,
        )
        # To keep runtime/memory manageable, subsample if too large
        max_local = 200_000
        if Xl.shape[0] > max_local:
            idx = torch.randperm(Xl.shape[0])[:max_local]
            Xl = Xl[idx]
            yl = yl[idx]
        if local_task == "multiclass":
            local_model, local_score = train_multiclass_probe(Xl, yl, num_classes=9, epochs=args.epochs, lr=args.lr)
            print(f"Local probe ({args.local_feature_set}, base={args.use_local_z}) {args.local_target} accuracy: {local_score:.4f}")
        else:
            local_model, local_score = train_binary_probe(Xl, yl, epochs=args.epochs, lr=args.lr)
            print(f"Local probe ({args.local_feature_set}, base={args.use_local_z}) {args.local_target} accuracy: {local_score:.4f}")

    # Save models
    g_name = f"global_probe_{args.global_target}_{global_feature_set}"
    if args.add_step_feature:
        g_name += "_step"
    if args.add_phase_feature:
        g_name += "_phase"
    torch.save(global_model.state_dict(), os.path.join(args.probes_dir, f"{g_name}.pt"))

    if local_model is not None:
        l_name = f"local_probe_{args.local_target}_{args.local_feature_set}_{args.use_local_z}"
        if args.add_position_features:
            l_name += "_pos"
        if args.add_step_feature:
            l_name += "_step"
        if args.add_phase_feature:
            l_name += "_phase"
        torch.save(local_model.state_dict(), os.path.join(args.probes_dir, f"{l_name}.pt"))
    print(f"Saved trained probes to {args.probes_dir}")


if __name__ == "__main__":
    main()
