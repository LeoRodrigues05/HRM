"""Linear Probe Training for HRM Hidden States.

This module provides functionality to train linear probes on the hidden states
(z_H and z_L) captured from the Hierarchical Reasoning Model (HRM) during inference.

Linear probes are used to analyze what information is encoded in the model's
internal representations at different levels:
- Global probes: Analyze puzzle-level properties (is_solved, violation counts, etc.)
- Local probes: Analyze per-cell properties (correctness, position, etc.)

Usage:
    python scripts/train_linear_probes.py --probes_dir results/probes
    python scripts/train_linear_probes.py --list_label_keys  # Show available targets
"""
import os
import argparse
import json
import logging
from typing import List, Dict, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
import torch.optim as optim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants for probe targets and feature sets
# ============================================================================

# Global targets that use regression (continuous values)
REGRESSION_TARGETS = frozenset({
    "pct_filled",
    "violated_units_total",
    "violated_rows_count",
    "violated_cols_count",
    "violated_boxes_count",

    "constraint_pressure",  # Can also be used globally (mean/max)
})

# Local targets that use multiclass classification (9 classes for Sudoku)
MULTICLASS_TARGETS = frozenset({
    "row_idx", 
    "col_idx", 
    "box_idx",
    "position_in_box",
    "cell_digit",  # 0-9, though 0 means empty
    "candidate_count",  # 0-9 candidates
})

# Local targets that use regression (continuous per-cell values)
LOCAL_REGRESSION_TARGETS = frozenset({
    "filled_in_row",
    "filled_in_col", 
    "filled_in_box",
    "constraint_pressure",
    "candidate_count",  # Can also be treated as regression
})

# Binary local targets
BINARY_LOCAL_TARGETS = frozenset({
    "per_cell_correct",
    "is_forced_cell",
    "is_given",
    "is_empty",
    "cell_changed_from_input",
    "cells_changed_since_prev_step",
    # Naked/Hidden singles (basic techniques)
    "is_naked_single",
    "is_hidden_single",
    "is_hidden_single_row",
    "is_hidden_single_col",
    "is_hidden_single_box",
    # Constraint-based
    "is_min_cand_in_row",
    "is_min_cand_in_col",
    "is_min_cand_in_box",
    "is_most_constrained",
    # Locked candidates (intermediate technique)
    "is_pointing_cell",
    "is_claiming_cell",
    "is_locked_candidate",
})

# Supported global feature sets
GLOBAL_FEATURE_SETS = frozenset({"z_H", "z_L", "concat", "diff", "prod", "concat_norms"})

# Supported local feature sets
LOCAL_FEATURE_SETS = frozenset({"z_only", "concat", "diff", "prod", "z_and_norms"})

# Default hyperparameters
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-2
DEFAULT_MAX_LOCAL_SAMPLES = 200_000

# Sudoku grid constants
SUDOKU_GRID_SIZE = 9
SUDOKU_NUM_CELLS = 81


# ============================================================================
# Data Loading
# ============================================================================

def load_probes(probes_dir: str) -> Tuple[List[Dict], List[Dict], Dict]:
    """Load probe data from disk.
    
    Args:
        probes_dir: Directory containing probe_global.pt, probe_local.pt, and probe_index.json
        
    Returns:
        Tuple of (global_samples, local_samples, index_metadata)
        
    Raises:
        FileNotFoundError: If probe_global.pt is missing
    """
    global_path = os.path.join(probes_dir, "probe_global.pt")
    local_path = os.path.join(probes_dir, "probe_local.pt")
    index_path = os.path.join(probes_dir, "probe_index.json")

    if not os.path.exists(global_path):
        raise FileNotFoundError(
            f"Missing {global_path}. Run probe collection first with run_probes_driver.py"
        )

    logger.info(f"Loading global probes from {global_path}")
    global_samples = torch.load(global_path, weights_only=False)
    
    local_samples: List[Dict] = []
    if os.path.exists(local_path):
        logger.info(f"Loading local probes from {local_path}")
        local_samples = torch.load(local_path, weights_only=False)
    else:
        logger.warning(f"Local probes not found at {local_path}")
        
    index: Dict = {}
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
            
    logger.info(f"Loaded {len(global_samples)} global samples, {len(local_samples)} local samples")
    return global_samples, local_samples, index


# ============================================================================
# Utility Functions
# ============================================================================

def _as_tensor(x: Any) -> Optional[torch.Tensor]:
    """Safely convert input to a PyTorch tensor.
    
    Args:
        x: Input value (tensor, array, list, or scalar)
        
    Returns:
        Tensor if conversion succeeds, None otherwise
    """
    if x is None:
        return None
    if torch.is_tensor(x):
        return x
    try:
        return torch.as_tensor(x)
    except (ValueError, TypeError, RuntimeError):
        return None


def list_available_label_keys(samples: List[Dict]) -> List[str]:
    """Extract all unique label keys from probe samples.
    
    Args:
        samples: List of probe sample dictionaries
        
    Returns:
        Sorted list of available label key names
    """
    keys: set = set()
    for row in samples:
        labels = row.get("labels", {}) or {}
        if isinstance(labels, dict):
            keys.update(labels.keys())
    return sorted(keys)


def _phase_to_scalar(phase: Any) -> float:
    """Convert phase string to numeric scalar.
    
    The HRM model has two phases per step:
    - "grad": The final step where gradients are computed (encoded as 1.0)
    - "nograd": The rollout steps without gradients (encoded as 0.0)
    
    Args:
        phase: Phase identifier ("grad", "nograd", or None)
        
    Returns:
        1.0 for "grad" phase, 0.0 otherwise
    """
    if phase is None:
        return 0.0
    s = str(phase).lower()
    return 1.0 if s == "grad" else 0.0


# ============================================================================
# Feature Building Functions
# ============================================================================

def build_global_feature_vector(row: Dict, feature_set: str = "z_H") -> Optional[torch.Tensor]:
    """Build a 1D feature vector from global (pooled) hidden states.
    
    Global features are derived by pooling z_H and z_L across the sequence dimension.
    These are used to predict puzzle-level properties.

    Args:
        row: Dictionary containing "z_H" and "z_L" tensors
        feature_set: One of:
            - "z_H": Use only the high-level hidden state
            - "z_L": Use only the low-level hidden state  
            - "concat": Concatenate z_H and z_L [z_H || z_L]
            - "diff": Element-wise difference (z_H - z_L)
            - "prod": Element-wise product (z_H * z_L)
            - "concat_norms": Concatenate with L2 norms [z_H || z_L || ||z_H|| || ||z_L||]
            
    Returns:
        1D feature tensor, or None if inputs are invalid
        
    Raises:
        ValueError: If feature_set is not recognized
    """
    if feature_set not in GLOBAL_FEATURE_SETS:
        raise ValueError(
            f"Unknown global feature_set: {feature_set}. "
            f"Valid options: {sorted(GLOBAL_FEATURE_SETS)}"
        )
        
    z_H = _as_tensor(row.get("z_H"))
    z_L = _as_tensor(row.get("z_L"))
    if z_H is None or z_L is None:
        return None

    # Pool to [D] if needed (mean over sequence dimension)
    if z_H.ndim == 2:
        z_H = z_H.mean(dim=0)
    if z_L.ndim == 2:
        z_L = z_L.mean(dim=0)
    if z_H.ndim != 1 or z_L.ndim != 1:
        return None

    # Build feature vector based on selected feature set
    if feature_set == "z_H":
        return z_H.float()
    elif feature_set == "z_L":
        return z_L.float()
    elif feature_set == "concat":
        return torch.cat([z_H, z_L], dim=0).float()
    elif feature_set == "diff":
        return (z_H - z_L).float()
    elif feature_set == "prod":
        return (z_H * z_L).float()
    elif feature_set == "concat_norms":
        nH = torch.linalg.vector_norm(z_H.float()).view(1)
        nL = torch.linalg.vector_norm(z_L.float()).view(1)
        return torch.cat([z_H.float(), z_L.float(), nH, nL], dim=0)
    
    # This should never be reached due to the check above
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
    """Build per-token feature matrix for local (cell-level) probes.
    
    Local features preserve the per-token structure to predict cell-level
    properties like correctness, position indices, etc.

    Args:
        row: Dictionary containing "z_H", "z_L" tensors and "labels"
        use_z: Which hidden state to use as base ("z_H" or "z_L")
        feature_set: One of:
            - "z_only": Use only the selected z tensor
            - "concat": Concatenate z_H and z_L per token
            - "diff": Per-token difference (z_H - z_L)
            - "prod": Per-token product (z_H * z_L)
            - "z_and_norms": Selected z plus per-token L2 norm
        add_position_features: Add Sudoku row/col/box one-hot encodings (27 dims)
        add_step_feature: Add ACT step index as a scalar feature
        add_phase_feature: Add phase indicator (grad=1/nograd=0) as scalar
            
    Returns:
        Tuple of (X, y) where:
            - X: Feature matrix [N, F] where N is number of cells
            - y: Label vector [N]
        Returns (None, None) if inputs are invalid
        
    Raises:
        ValueError: If feature_set is not recognized
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


# ============================================================================
# Model Definition
# ============================================================================

class LinearProbe(nn.Module):
    """Simple linear probe for classification or regression.
    
    A linear probe is a single linear layer used to predict labels from
    frozen hidden representations. The probe's performance indicates how
    linearly decodable the target information is from the representations.
    
    Args:
        in_dim: Input feature dimension
        out_dim: Output dimension (1 for binary/regression, num_classes for multiclass)
        bias: Whether to include bias term
    """
    
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear layer."""
        return self.linear(x)
    
    def __repr__(self) -> str:
        return f"LinearProbe(in_dim={self.in_dim}, out_dim={self.out_dim})"


# ============================================================================
# Probe Training Functions
# ============================================================================

def train_binary_probe(
    X: torch.Tensor, 
    y: torch.Tensor, 
    epochs: int = DEFAULT_EPOCHS, 
    lr: float = DEFAULT_LR
) -> Tuple[LinearProbe, float]:
    """Train a binary classification probe.
    
    Uses BCEWithLogitsLoss for numerical stability.
    
    Args:
        X: Feature matrix [N, D]
        y: Binary labels [N] (values in {0, 1})
        epochs: Number of training epochs
        lr: Learning rate for AdamW optimizer
        
    Returns:
        Tuple of (trained_model, training_accuracy)
    """
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


def train_regression_probe(
    X: torch.Tensor, 
    y: torch.Tensor, 
    epochs: int = 100, 
    lr: float = DEFAULT_LR
) -> Tuple[LinearProbe, float]:
    """Train a linear regression probe.
    
    Uses MSELoss and evaluates using R² (coefficient of determination).
    
    Args:
        X: Feature matrix [N, D]
        y: Continuous labels [N]
        epochs: Number of training epochs (default higher for regression)
        lr: Learning rate for AdamW optimizer
        
    Returns:
        Tuple of (trained_model, training_R²_score)
    """
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


def train_multiclass_probe(
    X: torch.Tensor, 
    y: torch.Tensor, 
    num_classes: int, 
    epochs: int = DEFAULT_EPOCHS, 
    lr: float = DEFAULT_LR
) -> Tuple[LinearProbe, float]:
    """Train a multiclass classification probe.
    
    Uses CrossEntropyLoss for multi-class classification.
    
    Args:
        X: Feature matrix [N, D]
        y: Class labels [N] (values in {0, 1, ..., num_classes-1})
        num_classes: Number of output classes (e.g., 9 for Sudoku row/col indices)
        epochs: Number of training epochs
        lr: Learning rate for AdamW optimizer
        
    Returns:
        Tuple of (trained_model, training_accuracy)
    """
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


# ============================================================================
# Dataset Building Functions
# ============================================================================


def build_global_dataset(
    global_samples: List[Dict],
    *,
    target_key: str = "is_solved",
    feature_set: str = "z_H",
    add_step_feature: bool = True,
    add_phase_feature: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a dataset for global (puzzle-level) probes.
    
    Assembles feature vectors and labels from all global probe samples.
    
    Args:
        global_samples: List of dictionaries from probe_global.pt
        target_key: Label key to predict (e.g., "is_solved", "pct_filled")
        feature_set: Feature construction method (see build_global_feature_vector)
        add_step_feature: Include ACT step index as a feature
        add_phase_feature: Include phase indicator as a feature
        
    Returns:
        Tuple of (X, y) where X is [N, D] features and y is [N] labels
        
    Raises:
        ValueError: If no valid samples could be processed
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
    """Build a dataset for local (cell-level) probes.
    
    Assembles per-cell feature vectors and labels from all local probe samples.
    
    Args:
        local_samples: List of dictionaries from probe_local.pt
        use_z: Base hidden state to use ("z_H" or "z_L")
        feature_set: Feature construction method (see build_local_feature_matrix)
        target_key: Label key to predict (e.g., "per_cell_correct", "row_idx")
        add_position_features: Include Sudoku position one-hot encodings
        add_step_feature: Include ACT step index as a feature
        add_phase_feature: Include phase indicator as a feature
        
    Returns:
        Tuple of (X, y) where X is [N_cells, D] features and y is [N_cells] labels
        
    Raises:
        ValueError: If no valid samples could be processed
    """
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
