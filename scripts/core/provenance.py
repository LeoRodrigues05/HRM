"""Lightweight run-provenance helper.

Writes a ``_meta.json`` next to a results directory recording the git SHA,
timestamp, environment, GPU, command line and experiment parameters so every
re-run is reproducible and auditable.

Usage
-----
    from scripts.core.provenance import write_meta
    write_meta(output_dir, "E8_constraint_probes", {"n_puzzles": 500, ...})
"""

import os
import sys
import json
import time
import platform
import subprocess
from typing import Any, Dict, Optional


def git_sha(repo_root: Optional[str] = None) -> str:
    """Return the current git commit SHA, or 'unknown' if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def git_dirty(repo_root: Optional[str] = None) -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode()
        return len(out.strip()) > 0
    except Exception:
        return False


def gpu_name() -> str:
    """Return the active CUDA device name, or 'cpu' if no GPU is available."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def write_meta(
    output_dir: str,
    experiment: str,
    params: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> str:
    """Write ``output_dir/_meta.json`` with run provenance. Returns the path."""
    meta = {
        "experiment": experiment,
        "git_sha": git_sha(repo_root),
        "git_dirty": git_dirty(repo_root),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu_name(),
        "argv": sys.argv,
        "params": params,
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path
