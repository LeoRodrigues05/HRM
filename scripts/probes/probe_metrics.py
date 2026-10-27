"""Prevalence-aware metrics for probe evaluation (rebuttal E0).

Plain accuracy hides weak probes on imbalanced targets: a probe that always
predicts the majority class scores the class prevalence. Every function here
reports the majority-class rate (taken from the training split) next to
balanced accuracy, and AUROC and MCC for binary targets, so a probe can be
judged against what predicting the majority would give.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch

METRIC_KEYS = ("prevalence", "majority_acc", "balanced_acc", "auroc", "mcc", "macro_f1")
NAN = float("nan")


def _auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    from scipy.stats import rankdata
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return NAN
    ranks = rankdata(scores)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(logits: torch.Tensor, y: torch.Tensor, y_train: torch.Tensor) -> Dict[str, float]:
    """Metrics for a binary probe; ``logits`` are pre-sigmoid scores."""
    s = logits.detach().float().view(-1).cpu().numpy()
    yt = y.detach().long().view(-1).cpu().numpy()
    ytr = y_train.detach().long().view(-1).cpu().numpy()
    pred = (s > 0).astype(np.int64)
    tp = int(((pred == 1) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    tpr = tp / (tp + fn) if tp + fn else NAN
    tnr = tn / (tn + fp) if tn + fp else NAN
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    majority_label = int(ytr.mean() >= 0.5)
    return {
        "prevalence": float(yt.mean()),
        "majority_acc": float((yt == majority_label).mean()),
        "balanced_acc": 0.5 * (tpr + tnr) if not (math.isnan(tpr) or math.isnan(tnr)) else NAN,
        "auroc": _auroc(s, yt),
        "mcc": (tp * tn - fp * fn) / denom if denom > 0 else 0.0,
        "macro_f1": NAN,
    }


def multiclass_metrics(logits: torch.Tensor, y: torch.Tensor, y_train: torch.Tensor,
                       n_classes: int) -> Dict[str, float]:
    """Metrics for a multiclass probe; balanced accuracy averages recall over
    the classes present in ``y``."""
    pred = logits.detach().argmax(1).view(-1).cpu().numpy()
    yt = y.detach().long().view(-1).cpu().numpy()
    ytr = y_train.detach().long().view(-1).cpu().numpy()
    majority_label = int(np.bincount(ytr, minlength=n_classes).argmax())
    recalls, f1s = [], []
    for c in range(n_classes):
        support = int((yt == c).sum())
        predicted = int((pred == c).sum())
        if support == 0 and predicted == 0:
            continue
        tp = int(((pred == c) & (yt == c)).sum())
        if support:
            recalls.append(tp / support)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "prevalence": NAN,
        "majority_acc": float((yt == majority_label).mean()),
        "balanced_acc": float(np.mean(recalls)) if recalls else NAN,
        "auroc": NAN,
        "mcc": NAN,
        "macro_f1": float(np.mean(f1s)) if f1s else NAN,
    }


def nan_to_none(x):
    """JSON-safe value: NaN becomes None (null)."""
    return None if isinstance(x, float) and math.isnan(x) else x


def mean_metrics(per_seed: List[Dict[str, float]]) -> Dict[str, float]:
    """Average each metric over seeds, ignoring NaNs; NaN when no seed has it."""
    out: Dict[str, float] = {}
    for key in METRIC_KEYS:
        vals = [m[key] for m in per_seed if key in m and not math.isnan(m[key])]
        out[key] = float(np.mean(vals)) if vals else NAN
    return out
