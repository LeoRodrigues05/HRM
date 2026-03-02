"""Configuration for paper replication experiments."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

# Base paths
REPO_ROOT = Path(__file__).parent.parent.parent
CHECKPOINT_PATH = REPO_ROOT / "Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/Checkpoint_HRM_Sudoku/checkpoint.pt"
DATA_PATH = REPO_ROOT / "data/sudoku-extreme-1k-aug-1000"
OUTPUT_BASE = REPO_ROOT / "experiments/paper_replication/results"


@dataclass
class ExperimentConfig:
    """Base configuration for experiments."""
    name: str
    output_dir: Path = field(default_factory=lambda: OUTPUT_BASE)
    checkpoint_path: Path = field(default_factory=lambda: CHECKPOINT_PATH)
    data_path: Path = field(default_factory=lambda: DATA_PATH)
    device: str = "cuda"
    seed: int = 42
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir) / self.name
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass  
class EasyHardConfig(ExperimentConfig):
    """Config for easy vs hard puzzle analysis."""
    name: str = "easy_hard_analysis"
    difficulty_metric: str = "num_givens"  # or "solution_path_length"
    num_bins: int = 5
    

@dataclass
class GrokkingConfig(ExperimentConfig):
    """Config for grokking analysis."""
    name: str = "grokking_analysis"
    checkpoint_dir: Optional[Path] = None  # Directory with multiple checkpoints
    eval_interval: int = 100
    metrics: List[str] = field(default_factory=lambda: ["train_loss", "train_acc", "val_acc", "test_acc"])


@dataclass
class StepDynamicsConfig(ExperimentConfig):
    """Config for step-wise dynamics analysis."""
    name: str = "step_dynamics"
    max_steps: int = 8
    metrics: List[str] = field(default_factory=lambda: ["hamming", "kl", "accuracy", "violations"])


@dataclass
class SpecializationConfig(ExperimentConfig):
    """Config for hierarchical specialization probes."""
    name: str = "specialization_probes"
    probe_targets: List[str] = field(default_factory=lambda: [
        "is_solved", "num_violations", "pct_filled", "row_complete", "col_complete", "box_complete"
    ])
    feature_sets: List[str] = field(default_factory=lambda: ["z_H", "z_L", "concat", "diff"])


@dataclass
class ActivationPatchingConfig(ExperimentConfig):
    """Config for activation patching experiments."""
    name: str = "activation_patching"


@dataclass
class ActivationAblationConfig(ExperimentConfig):
    """Config for activation ablation experiments (zeroing H/L activations)."""
    name: str = "activation_ablation"
    ablate_levels: List[str] = field(default_factory=lambda: ["H", "L", "both"])
    ablation_value: float = 0.0
    max_steps: int = 8
    num_puzzles: int = 10
    num_runs: int = 3


@dataclass
class CrossStepTransferConfig(ExperimentConfig):
    """Config for cross-step activation transfer experiments."""
    name: str = "cross_step_transfer"
    transfer_levels: List[str] = field(default_factory=lambda: ["H", "L"])
    donor_steps: List[int] = field(default_factory=lambda: [6, 7])
    recipient_steps: List[int] = field(default_factory=lambda: [2, 3, 4])
    max_steps: int = 8
    num_puzzles: int = 10
    num_runs: int = 3
    num_puzzle_pairs: int = 50
    patch_levels: List[str] = field(default_factory=lambda: ["H", "L", "both"])
    patch_step_configs: List[str] = field(default_factory=lambda: ["all", "0", "1,2", "3,4,5,6,7"])
