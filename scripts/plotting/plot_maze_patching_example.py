#!/usr/bin/env python3
"""Qualitative Maze cross-puzzle patching example (donor-z_L into target).

Companion to the Sudoku activation-patching example figure. Searches a pool of
donor/target pairs for a case that reproduces the aggregate effect reported in
``results/Final/maze/hardened/patching_full_steps`` -- donor-$z_L$ patching at a
mid-trajectory step destroys the target's valid start--goal path while
donor-$z_H$ patching at the same step leaves it intact -- and renders it as a
four-panel figure.

Usage (needs the maze checkpoint and a GPU):
    python scripts/plotting/plot_maze_patching_example.py \
        --patch_step 8 --num_pool 40 \
        --output_dir results/Final/figures
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle, Patch, Rectangle

from scripts.plotting.paper_style import apply_paper_style, COL_W, scaled_rc
from scripts.controlled.controlled_common import load_model_and_dataloader, collect_puzzles
from scripts.core.activation_patching import ActivationPatcher, ActivationCache
from scripts.maze.maze_common import (
    MAZE_CHECKPOINT, get_puzzle_emb_len, cell_positions, maze_prediction_metrics,
)
from utils.maze_targets import (
    GRID_SIZE, SEQ_LEN, PAD_ID, WALL_ID, FREE_ID, START_ID, GOAL_ID, PATH_ID,
)

# Cell colours follow the HTML maze reports (scripts/maze/maze_render_common.py)
# so the paper figure and the browsable reports read the same, with the path
# hue darkened to #c47f00 so it clears the 3:1 contrast floor against the white
# free cells. Start/goal additionally carry S/G glyphs, so those two never rest
# on colour alone. CHANGE_COLOR marks cells that the patch moved relative to the
# baseline run; it is drawn as an outline (texture), not a fill.
CELL_ORDER = [PAD_ID, WALL_ID, FREE_ID, START_ID, GOAL_ID, PATH_ID]
CELL_COLORS = {
    PAD_ID: "#e8e8e8",
    WALL_ID: "#222222",
    FREE_ID: "#ffffff",
    START_ID: "#1a7f37",
    GOAL_ID: "#cf222e",
    PATH_ID: "#c47f00",
}
CHANGE_COLOR = "#0969da"


def _flat(t: torch.Tensor) -> np.ndarray:
    if t.ndim == 2:
        t = t[0]
    return t.detach().to("cpu").to(torch.int64).numpy().reshape(-1)


def _slice_preds(preds: np.ndarray, n: int = SEQ_LEN) -> np.ndarray:
    return preds[-n:] if preds.size > n else preds


def _grid(flat: np.ndarray) -> np.ndarray:
    return flat.reshape(GRID_SIZE, GRID_SIZE)


def _to_index_grid(flat: np.ndarray) -> np.ndarray:
    """Map token ids to contiguous colour indices, unknown ids -> FREE."""
    lookup = {tok: i for i, tok in enumerate(CELL_ORDER)}
    free_i = lookup[FREE_ID]
    return np.vectorize(lambda v: lookup.get(int(v), free_i))(_grid(flat))


MARK_BOX = 6.0  # side length, in cells, of the donor-endpoint marker box


def _draw(ax, flat: np.ndarray, title: str, *, label_fs: float,
          marked: Optional[np.ndarray] = None, mark_lw: float = 0.7) -> None:
    idx = _to_index_grid(flat)
    cmap = ListedColormap([CELL_COLORS[t] for t in CELL_ORDER])
    norm = BoundaryNorm(np.arange(len(CELL_ORDER) + 1) - 0.5, len(CELL_ORDER))
    ax.imshow(idx, cmap=cmap, norm=norm, interpolation="nearest")

    # S and G occupy a single cell each, which is ~3 px at column width. Draw
    # them as an oversized labelled disc so the endpoints stay findable and the
    # letter -- not the hue -- carries the identity.
    g = _grid(flat)
    for tok, ch in ((START_ID, "S"), (GOAL_ID, "G")):
        rs, cs = np.nonzero(g == tok)
        for r, c in zip(rs, cs):
            ax.add_patch(Circle((c, r), 1.6, facecolor=CELL_COLORS[tok],
                                edgecolor="white", linewidth=0.4 * label_fs / 2.6,
                                zorder=4))
            ax.text(c, r, ch, ha="center", va="center", color="white",
                    fontsize=label_fs, fontweight="bold", zorder=5)

    # Marker box sits ABOVE the endpoint disc, and is drawn several cells wide
    # so it reads at column width -- a 1-cell outline would be hidden under the
    # disc it is pointing at.
    if marked is not None:
        h = MARK_BOX / 2.0
        for r, c in zip(*np.nonzero(_grid(marked))):
            ax.add_patch(Rectangle((c - h, r - h), MARK_BOX, MARK_BOX,
                                   facecolor="none", edgecolor=CHANGE_COLOR,
                                   linewidth=mark_lw, zorder=6))

    ax.set_title(title, pad=2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#666666")


# --------------------------------------------------------------------------
# model runs
# --------------------------------------------------------------------------

def _run_pair(patcher: ActivationPatcher, src_batch, tgt_batch, *,
              positions: List[int], patch_step: int, max_steps: int) -> Dict[str, np.ndarray]:
    src_cache: Dict[int, ActivationCache] = {}
    patcher.run_and_cache_activations(src_batch, src_cache, max_steps=max_steps)

    base_cache: Dict[int, ActivationCache] = {}
    patcher.run_and_cache_activations(tgt_batch, base_cache, max_steps=max_steps)

    out = {}
    for level in ("L", "H"):
        _, cache, _ = patcher.run_with_patching(
            tgt_batch, src_cache,
            patch_level=level, patch_steps=[patch_step],
            patch_positions=positions, max_steps=max_steps,
        )
        out[f"patched_{level}"] = _slice_preds(_flat(cache[sorted(cache)[-1]].preds))

    out["src_final"] = _slice_preds(_flat(src_cache[sorted(src_cache)[-1]].preds))
    out["base_final"] = _slice_preds(_flat(base_cache[sorted(base_cache)[-1]].preds))
    return out


def _donor_unique_endpoints(src_input: np.ndarray, tgt_input: np.ndarray) -> np.ndarray:
    """Cells where the donor has S or G and the target does not agree.

    This is the denominator of ``source_endpoint_imprint_rate`` in
    ``scripts/maze/maze_common.py``; a patched prediction that emits the donor's
    token at one of these cells has imported the donor's *puzzle*, not just
    corrupted its own answer.
    """
    return (((src_input == START_ID) | (src_input == GOAL_ID)) & (src_input != tgt_input))


def _score(res: Dict[str, np.ndarray], tgt_label: np.ndarray, tgt_input: np.ndarray,
           src_input: np.ndarray) -> Optional[Dict]:
    """Keep pairs that show donor puzzle content appearing in the output.

    The headline aggregate for Maze cross-puzzle patching is the endpoint
    imprint: donor-z_L patching stamps the donor's S/G into the prediction at
    46.5pp of donor-unique endpoint cells, while donor-z_H patching stamps none
    (results/Final/maze/hardened/patching_full_steps). We select an example that
    reproduces that asymmetry rather than one that merely breaks the path.
    """
    base, pL, pH = res["base_final"], res["patched_L"], res["patched_H"]
    m_base = maze_prediction_metrics(base, tgt_label, tgt_input)
    m_L = maze_prediction_metrics(pL, tgt_label, tgt_input)
    m_H = maze_prediction_metrics(pH, tgt_label, tgt_input)

    donor_eps = _donor_unique_endpoints(src_input, tgt_input)
    n_eps = int(donor_eps.sum())
    if n_eps == 0:
        return None
    imprinted_L = donor_eps & (pL == src_input)
    imprinted_H = donor_eps & (pH == src_input)

    # Baseline must be a clean solve, and the imprint must be z_L-specific.
    if m_base["valid_sg_path"] != 1.0 or m_base["token_acc"] != 1.0:
        return None
    if imprinted_L.sum() == 0 or imprinted_H.sum() != 0:
        return None

    imprint_rate = float(imprinted_L.sum()) / n_eps
    changed = int((pL != base).sum())
    return {
        "metrics": {"baseline": m_base, "patched_L": m_L, "patched_H": m_H},
        "n_cells_changed": changed,
        "n_donor_endpoints": n_eps,
        "n_imprinted_zL": int(imprinted_L.sum()),
        "imprint_rate_zL": imprint_rate,
        "imprinted_mask": imprinted_L,
        # Prefer both donor endpoints imprinted, then a visibly damaged path.
        "rank": (imprint_rate, m_base["path_cell_acc"] - m_L["path_cell_acc"], changed),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=MAZE_CHECKPOINT)
    ap.add_argument("--patch_step", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=16)
    ap.add_argument("--num_pool", type=int, default=40)
    ap.add_argument("--output_dir", default="results/Final/figures")
    ap.add_argument("--basename", default="maze_patching_example")
    ap.add_argument("--display_w", type=float, default=0.95 * COL_W,
                    help="Width (in) at which the figure is included in the paper.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model, test_loader, _ = load_model_and_dataloader(args.checkpoint, device)
    pel = get_puzzle_emb_len(model)
    patcher = ActivationPatcher(model, device=device)
    positions = cell_positions(pel).tolist()

    puzzles = collect_puzzles(test_loader, device, args.num_pool)
    print(f"[maze_patch_fig] pool={len(puzzles)} patch_step={args.patch_step} device={device}")

    best = None
    for i in range(0, len(puzzles) - 1, 2):
        src_idx, src_batch = puzzles[i]
        tgt_idx, tgt_batch = puzzles[i + 1]
        res = _run_pair(patcher, src_batch, tgt_batch,
                        positions=positions, patch_step=args.patch_step,
                        max_steps=args.max_steps)
        tgt_label = _slice_preds(_flat(tgt_batch["labels"]))
        tgt_input = _slice_preds(_flat(tgt_batch["inputs"]))
        src_input = _slice_preds(_flat(src_batch["inputs"]))
        scored = _score(res, tgt_label, tgt_input, src_input)
        flag = "" if scored is None else (
            f"  imprint {scored['n_imprinted_zL']}/{scored['n_donor_endpoints']}"
            f"  rank={tuple(round(x, 3) for x in scored['rank'])}"
        )
        print(f"  pair src={src_idx} tgt={tgt_idx}{flag or '  (rejected)'}")
        if scored is None:
            continue
        if best is None or scored["rank"] > best["scored"]["rank"]:
            best = {
                "scored": scored, "res": res,
                "src_idx": src_idx, "tgt_idx": tgt_idx,
                "src_input": src_input,
                "src_label": _slice_preds(_flat(src_batch["labels"])),
                "tgt_input": tgt_input, "tgt_label": tgt_label,
            }

    if best is None:
        raise SystemExit(
            "No pair satisfied the selection criteria; raise --num_pool or "
            "try a different --patch_step."
        )

    print(f"[maze_patch_fig] selected src={best['src_idx']} tgt={best['tgt_idx']} "
          f"changed={best['scored']['n_cells_changed']} cells")

    os.makedirs(args.output_dir, exist_ok=True)
    _render(best, args)


def _render(best: Dict, args) -> None:
    res, tgt_input, tgt_label = best["res"], best["tgt_input"], best["tgt_label"]
    src_input, src_label = best["src_input"], best["src_label"]

    base, pL, pH = res["base_final"], res["patched_L"], res["patched_H"]
    changed_L = (pL != base)
    changed_H = (pH != base)
    imprinted = best["scored"]["imprinted_mask"]

    # The prediction panels are shown RAW -- the donor's endpoints only appear
    # because the patched run emitted them, so redrawing the target layout on
    # top (as _overlay_layout does) would erase exactly the effect being shown.
    panels = [
        (src_input, "(a) Donor maze", imprinted),
        (tgt_input, "(b) Target maze", None),
        (base, "(c) Baseline prediction", None),
        (pL, "(d) Donor-$z_L$ patch", imprinted),
    ]

    authored_w = COL_W
    rc = scaled_rc(authored_w, args.display_w)
    with plt.rc_context(rc):
        k = authored_w / args.display_w
        fig, axes = plt.subplots(2, 2, figsize=(authored_w, authored_w * 1.08))
        for ax, (grid_flat, title, marked) in zip(axes.ravel(), panels):
            _draw(ax, grid_flat, title, label_fs=3.4 * k,
                  marked=marked, mark_lw=0.7 * k)

        def _swatch(tok, label):
            return Patch(facecolor=CELL_COLORS[tok], edgecolor="#666666",
                         lw=0.4 * k, label=label)

        handles = [
            _swatch(WALL_ID, "wall"), _swatch(FREE_ID, "free"),
            _swatch(PATH_ID, "path"), _swatch(START_ID, "S"), _swatch(GOAL_ID, "G"),
            Patch(facecolor="none", edgecolor=CHANGE_COLOR, lw=0.9 * k,
                  label="donor endpoint"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
                   bbox_to_anchor=(0.5, -0.008), handlelength=1.0,
                   columnspacing=0.7, handletextpad=0.3)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.06,
                            wspace=0.06, hspace=0.14)

        stem = os.path.join(args.output_dir, args.basename)
        for ext in ("pdf", "png"):
            fig.savefig(f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

    meta = {
        "patch_step": args.patch_step,
        "max_steps": args.max_steps,
        "src_puzzle_idx": best["src_idx"],
        "tgt_puzzle_idx": best["tgt_idx"],
        "patch_positions": "all 900 maze cells",
        "n_cells_changed_vs_baseline": {
            "zL": int(changed_L.sum()), "zH": int(changed_H.sum()),
        },
        "baseline_equals_ground_truth": bool((base == tgt_label).all()),
        "donor_endpoint_imprint": {
            "n_donor_unique_endpoints": best["scored"]["n_donor_endpoints"],
            "n_imprinted_zL": best["scored"]["n_imprinted_zL"],
            "n_imprinted_zH": 0,
            "rate_zL": best["scored"]["imprint_rate_zL"],
            "imprinted_tokens": [
                {START_ID: "START", GOAL_ID: "GOAL"}[int(src_input[i])]
                for i in np.nonzero(imprinted)[0]
            ],
            "imprinted_cells_rc": [
                [int(i) // GRID_SIZE, int(i) % GRID_SIZE]
                for i in np.nonzero(imprinted)[0]
            ],
            # The target keeps its own S/G: the donor's endpoint is added to the
            # output, it does not replace the recipient's.
            "target_endpoints_retained": bool(
                all(pL[i] == tgt_input[i]
                    for i in np.nonzero((tgt_input == START_ID) | (tgt_input == GOAL_ID))[0])
            ),
        },
        "metrics": best["scored"]["metrics"],
    }
    with open(f"{stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[maze_patch_fig] wrote {stem}.pdf/.png and {stem}_meta.json")


if __name__ == "__main__":
    main()
