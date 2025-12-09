import os
import argparse
import json
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def load_probes(probes_dir: str):
    global_path = os.path.join(probes_dir, "probe_global.pt")
    local_path = os.path.join(probes_dir, "probe_local.pt")
    index_path = os.path.join(probes_dir, "probe_index.json")

    if not (os.path.exists(global_path) and os.path.exists(local_path)):
        raise FileNotFoundError(f"Expected probe files in {probes_dir}")

    global_samples = torch.load(global_path)
    local_samples = torch.load(local_path)
    index = {}
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
    return global_samples, local_samples, index


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


def build_global_is_solved_dataset(global_samples: List[Dict], use_z: str = "z_H") -> Tuple[torch.Tensor, torch.Tensor]:
    X_list = []
    y_list = []
    for row in global_samples:
        z = row.get(use_z)
        labels = row.get("labels", {})
        y = labels.get("is_solved")
        if z is None or y is None:
            continue
        # z shape [B, D] or [D]; unify to [D]
        z_tensor = z
        if z_tensor.ndim == 2:
            # Mean over batch axis for a stable single vector per entry
            z_tensor = z_tensor.mean(dim=0)
        X_list.append(z_tensor)
        # Handle label types: tensor or int/bool
        if torch.is_tensor(y):
            if y.ndim == 0:
                y_val = int(y.item())
            else:
                # Reduce per-batch labels to a single scalar (majority via mean > 0.5)
                y_val = int((y.float().mean() > 0.5).item())
        else:
            y_val = int(y)
        y_list.append(torch.as_tensor(y_val))
    X = torch.stack(X_list).float()
    y = torch.stack(y_list).long()
    return X, y


def build_local_per_cell_dataset(local_samples: List[Dict], use_z: str = "z_L") -> Tuple[torch.Tensor, torch.Tensor]:
    X_list = []
    y_list = []
    for row in local_samples:
        z = row.get(use_z)
        labels = row.get("labels", {})
        per_cell = labels.get("per_cell_correct")
        if z is None or per_cell is None:
            continue
        z_tensor = z  # [T, D] or [B, T, D]
        y_tensor = per_cell

        # Flatten batch if present
        if z_tensor.ndim == 3:
            z_tensor = z_tensor.reshape(-1, z_tensor.shape[-1])
        if y_tensor.ndim > 1:
            y_tensor = y_tensor.reshape(-1)

        # Add all tokens
        X_list.append(z_tensor.float())
        y_list.append(y_tensor.long())

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on HRM hidden states")
    parser.add_argument("--probes_dir", default=os.path.join("results", "probes"), help="Directory containing probe_global.pt and probe_local.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--use_global_z", choices=["z_H", "z_L"], default="z_H")
    parser.add_argument("--use_local_z", choices=["z_H", "z_L"], default="z_L")
    args = parser.parse_args()

    global_samples, local_samples, _index = load_probes(args.probes_dir)

    # Global: is_solved
    Xg, yg = build_global_is_solved_dataset(global_samples, use_z=args.use_global_z)
    global_model, global_acc = train_binary_probe(Xg, yg, epochs=args.epochs, lr=args.lr)
    print(f"Global probe ({args.use_global_z}) is_solved accuracy: {global_acc:.4f}")

    # Local: per_cell_correct
    Xl, yl = build_local_per_cell_dataset(local_samples, use_z=args.use_local_z)
    # To keep runtime/memory manageable, subsample if too large
    max_local = 200_000
    if Xl.shape[0] > max_local:
        idx = torch.randperm(Xl.shape[0])[:max_local]
        Xl = Xl[idx]
        yl = yl[idx]
    local_model, local_acc = train_binary_probe(Xl, yl, epochs=args.epochs, lr=args.lr)
    print(f"Local probe ({args.use_local_z}) per_cell_correct accuracy: {local_acc:.4f}")

    # Save models
    torch.save(global_model.state_dict(), os.path.join(args.probes_dir, f"global_probe_{args.use_global_z}.pt"))
    torch.save(local_model.state_dict(), os.path.join(args.probes_dir, f"local_probe_{args.use_local_z}.pt"))
    print(f"Saved trained probes to {args.probes_dir}")


if __name__ == "__main__":
    main()
