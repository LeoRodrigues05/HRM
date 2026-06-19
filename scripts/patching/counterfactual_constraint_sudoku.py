#!/usr/bin/env python3
"""Counterfactual constraint localization (Sudoku) via unit-interchange patching.

Question: is a *single* Sudoku constraint — the value of one solved/forced cell C,
which is determined by C's row/col/box — causally carried by a localized subset of
z_H/z_L, namely the representations at C's 20 unit-peers?

Test (interchange / resampling intervention):
  For a target cell C (input-blank, model-correct; optionally a naked single), and a
  random DONOR puzzle, patch the donor's z_H at C's **unit-peer positions only**
  (NOT C itself) into the target run, then read C's new prediction.
    * UNIT patch  : patch the 20 peers of C (row ∪ col ∪ box − C)
    * RANDOM patch: patch 20 random non-peer, non-C cells (position-matched control)
  Metrics per (cell, donor):
    flip_C            : did C's prediction change?
    flip_to_donor     : did C change *to the donor's value at C*? (constraint transfer)
    selectivity       : of all cells that changed, fraction inside C's unit (∪C)
  Localization ⇔ UNIT flip_C ≫ RANDOM flip_C (paired, CI excludes 0), flip_to_donor>0,
  and selectivity high. If UNIT ≈ RANDOM, the constraint is distributed.

This is a *sufficiency + selectivity* test (transplant the donor's unit, see if C
follows) — strictly stronger than the rank-1 necessity ablation in E9/E10.

Modes: --dry_run (build + validate, no patching), normal (one config), --sweep
(grid over patch_level × patch_steps). Writes JSON (+ sweep CSV) + _meta.json.

Usage
  python scripts/patching/counterfactual_constraint_sudoku.py --dry_run
  python scripts/patching/counterfactual_constraint_sudoku.py --num_samples 1 --num_donors 2 --device cpu
  python scripts/patching/counterfactual_constraint_sudoku.py --num_samples 60 --num_donors 5 --device cuda
  python scripts/patching/counterfactual_constraint_sudoku.py --sweep --num_samples 60 --num_donors 5 --device cuda
"""
from __future__ import annotations
import os, sys, json, csv, time, argparse
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from scipy import stats as scipy_stats

from scripts.controlled.controlled_common import collect_puzzles, bootstrap_ci
from scripts.core.activation_patching import ActivationPatcher, ActivationCache
from scripts.probes.e8_constraint_probes import (
    load_model_and_data, SUDOKU_CELLS, SUDOKU_SIZE,
)

N = SUDOKU_SIZE  # 9


# ── geometry: unit peers of a cell ─────────────────────────────────────────

def unit_peers(i: int) -> List[int]:
    """The 20 cells sharing a row, column, or box with cell i (excluding i)."""
    r, c = divmod(i, N)
    peers = set()
    peers.update(r * N + cc for cc in range(N))           # row
    peers.update(rr * N + c for rr in range(N))           # col
    br, bc = (r // 3) * 3, (c // 3) * 3
    peers.update((br + dr) * N + (bc + dc) for dr in range(3) for dc in range(3))  # box
    peers.discard(i)
    return sorted(peers)


def random_control(i: int, peers: List[int], k: int, rng: np.random.Generator) -> List[int]:
    """k random cells that are neither i nor a unit-peer of i (position-matched control)."""
    excluded = set(peers) | {i}
    pool = [p for p in range(SUDOKU_CELLS) if p not in excluded]
    k = min(k, len(pool))
    return sorted(rng.choice(pool, size=k, replace=False).tolist())


# ── helpers ────────────────────────────────────────────────────────────────

def cache_preds(cache: Dict[int, ActivationCache]) -> np.ndarray:
    """Final-step predictions over the 81 answer cells (token ids)."""
    c = cache[max(cache.keys())]
    return c.preds.detach().cpu().numpy().reshape(-1)[-SUDOKU_CELLS:].astype(np.int64)


def puzzle_emb_len_from_cache(cache: Dict[int, ActivationCache]) -> int:
    seq = cache[max(cache.keys())].z_H.shape[1]
    return seq - SUDOKU_CELLS


def forced_given_peers(base_preds: np.ndarray) -> set:
    """Cells whose predicted digit is the ONLY remaining candidate given solved peers.

    Uses the model's own final prediction grid: digit = token - 1 (blank token 1 → 0).
    A cell is 'forced' if its 20 peers collectively use every digit except its own, so
    the single Sudoku constraint at that cell has a unique answer."""
    digits = base_preds.astype(np.int64) - 1   # token -> digit; blank -> 0, filled 1..9
    forced = set()
    for i in range(SUDOKU_CELLS):
        d = int(digits[i])
        if d < 1:
            continue
        peer_d = {int(digits[p]) for p in unit_peers(i) if int(digits[p]) >= 1}
        if set(range(1, N + 1)) - peer_d == {d}:
            forced.add(i)
    return forced


def select_target_cells(base_preds: np.ndarray, inputs_tok, labels_tok, forced_only: bool) -> List[int]:
    """Cells the model had to solve (input-blank) and got right; optionally 'forced' ones."""
    inp = inputs_tok.detach().cpu().numpy().reshape(-1)[-SUDOKU_CELLS:]
    lab = labels_tok.detach().cpu().numpy().reshape(-1)[-SUDOKU_CELLS:]
    blank = inp == 1                       # token 1 == digit 0 (blank), per DIGIT_OFFSET
    correct = base_preds == lab
    cand = [i for i in range(SUDOKU_CELLS) if bool(blank[i]) and bool(correct[i])]
    if forced_only:
        fset = forced_given_peers(base_preds)
        cand = [i for i in cand if i in fset]
    return cand


# ── core run ────────────────────────────────────────────────────────────────

def run_config(patcher, base_caches, base_preds, batches, samples, donor_pool, num_donors, pel,
               patch_level, patch_steps, max_steps, k_ctrl, rng) -> Dict:
    """Run the interchange test for one (patch_level, patch_steps) config.

    samples: list of (pool_pos, cell). donor_pool: pool positions usable as donors;
    each sample uses the first ``num_donors`` of them that are != its own puzzle.
    Returns aggregate metrics over all samples (each averaged across its donors).
    """
    per_sample = []
    for (sp, cell) in samples:
        peers = unit_peers(cell)
        unit_pos = [pel + p for p in peers]
        unit_set = set(peers) | {cell}
        nonunit = [j for j in range(SUDOKU_CELLS) if j not in unit_set]  # 60 NON-patched cells
        base = base_preds[sp]
        donors = [d for d in donor_pool if d != sp][:num_donors]
        u_flips, r_flips, to_donor, selec, outside = [], [], [], [], []
        for dp in donors:
            donor_cache = base_caches[dp]
            steps_eff = [s for s in patch_steps if s in donor_cache]
            if not steps_eff:
                continue
            rand_pos = [pel + p for p in random_control(cell, peers, k_ctrl, rng)]
            # UNIT interchange: transplant donor's z_* at C's 20 peers (NOT C itself)
            _, uc, _ = patcher.run_with_patching(
                batches[sp], donor_cache, patch_level=patch_level,
                patch_steps=steps_eff, patch_positions=unit_pos, max_steps=max_steps)
            up = cache_preds(uc)
            # RANDOM control: transplant the same donor at 20 non-peer, non-C cells
            _, rc, _ = patcher.run_with_patching(
                batches[sp], donor_cache, patch_level=patch_level,
                patch_steps=steps_eff, patch_positions=rand_pos, max_steps=max_steps)
            rp = cache_preds(rc)

            u_flips.append(int(up[cell] != base[cell]))
            r_flips.append(int(rp[cell] != base[cell]))
            to_donor.append(int(up[cell] != base[cell] and up[cell] == base_preds[dp][cell]))
            # selectivity: of the 60 NON-patched non-unit cells, how many changed (lower=cleaner)
            ch_out = int(np.sum((up != base)[nonunit]))
            outside.append(ch_out)
            selec.append(1.0 - ch_out / max(1, len(nonunit)))
        if u_flips:
            per_sample.append({
                "cell": cell,
                "unit_flip_rate": float(np.mean(u_flips)),
                "random_flip_rate": float(np.mean(r_flips)),
                "flip_to_donor_rate": float(np.mean(to_donor)),
                "selectivity": float(np.mean(selec)),
                "changed_outside_unit": float(np.mean(outside)),
                "n_donors": len(u_flips),
            })

    if not per_sample:
        return {"n_samples": 0, "note": "no usable samples"}
    u = np.array([s["unit_flip_rate"] for s in per_sample])
    r = np.array([s["random_flip_rate"] for s in per_sample])
    diff = u - r
    try:
        w_stat, w_p = scipy_stats.wilcoxon(u, r) if np.any(diff != 0) else (float("nan"), 1.0)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    return {
        "patch_level": patch_level, "patch_steps": patch_steps, "n_samples": len(per_sample),
        "unit_flip_rate": bootstrap_ci(u),
        "random_flip_rate": bootstrap_ci(r),
        "unit_minus_random": bootstrap_ci(diff),
        "flip_to_donor_rate": bootstrap_ci(np.array([s["flip_to_donor_rate"] for s in per_sample])),
        "selectivity": bootstrap_ci(np.array([s["selectivity"] for s in per_sample])),
        "changed_outside_unit": bootstrap_ci(np.array([s["changed_outside_unit"] for s in per_sample])),
        "wilcoxon_p": float(w_p),
        "per_sample": per_sample,
    }


def parse_steps(spec: str, max_steps: int) -> List[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(max_steps))
    if spec == "early":
        return list(range(max_steps // 2))
    if spec == "late":
        return list(range(max_steps // 2, max_steps))
    if spec == "final":
        return [max_steps - 1]
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def main():
    p = argparse.ArgumentParser(description="Counterfactual constraint localization (Sudoku)")
    p.add_argument("--n_puzzles", type=int, default=60, help="Pool size (targets + donors).")
    p.add_argument("--num_samples", type=int, default=60, help="Target (puzzle,cell) samples.")
    p.add_argument("--num_donors", type=int, default=5)
    p.add_argument("--cells_per_puzzle", type=int, default=4)
    p.add_argument("--patch_level", default="H", choices=["H", "L", "both"])
    p.add_argument("--patch_steps", default="all",
                   help="all|early|late|final or comma list, e.g. 8,12,15")
    p.add_argument("--forced_only", action="store_true", help="restrict targets to naked singles")
    p.add_argument("--max_steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="results/constraint_localization")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--sweep", action="store_true",
                   help="grid over patch_level × patch_steps (ignores --patch_level/--patch_steps)")
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    print(f"[ccl] device={device} n_puzzles={args.n_puzzles} num_samples={args.num_samples} "
          f"num_donors={args.num_donors} forced_only={args.forced_only} dry_run={args.dry_run}")

    model, test_loader, _ = load_model_and_data(device)
    patcher = ActivationPatcher(model, device=device)
    pool = collect_puzzles(test_loader, device, args.n_puzzles, seed=args.seed)
    batches = [b for _, b in pool]
    print(f"[ccl] pool={len(batches)} puzzles; caching base runs (max_steps={args.max_steps})...")

    base_caches, base_preds = [], []
    for i, b in enumerate(batches):
        cache: Dict[int, ActivationCache] = {}
        patcher.run_and_cache_activations(b, cache, max_steps=args.max_steps)
        base_caches.append(cache)
        base_preds.append(cache_preds(cache))
        if (i + 1) % 20 == 0:
            print(f"  cached {i+1}/{len(batches)}")
    pel = puzzle_emb_len_from_cache(base_caches[0])
    print(f"[ccl] puzzle_emb_len={pel} (seq positions for cell i = {pel}+i)")

    # build target samples (pool_pos, cell)
    samples: List[Tuple[int, int]] = []
    for sp in range(len(batches)):
        if len(samples) >= args.num_samples:
            break
        cells = select_target_cells(base_preds[sp], batches[sp]["inputs"], batches[sp]["labels"],
                                    args.forced_only)
        for cell in cells[:args.cells_per_puzzle]:
            samples.append((sp, cell))
            if len(samples) >= args.num_samples:
                break
    donor_pool = list(range(len(batches)))  # any pool puzzle can be a donor; per-sample skips self
    print(f"[ccl] built {len(samples)} target samples; donor_pool={len(donor_pool)} (num_donors={args.num_donors}/sample)")

    meta = {"n_puzzles": args.n_puzzles, "num_samples": len(samples), "num_donors": args.num_donors,
            "cells_per_puzzle": args.cells_per_puzzle, "forced_only": args.forced_only,
            "max_steps": args.max_steps, "seed": args.seed, "puzzle_emb_len": pel}

    if args.dry_run:
        ex = samples[0] if samples else None
        if ex:
            sp, cell = ex
            peers = unit_peers(cell)
            ctrl = random_control(cell, peers, len(peers), rng)
            print(f"[dry_run] example sample: puzzle_pos={sp} cell={cell} (r{cell//9},c{cell%9})")
            print(f"[dry_run]   #peers={len(peers)} (expect 20) -> abs positions {pel+peers[0]}..{pel+peers[-1]}")
            print(f"[dry_run]   #control={len(ctrl)} disjoint-from-peers={set(ctrl).isdisjoint(peers)} "
                  f"excludes-cell={cell not in ctrl}")
            print(f"[dry_run]   max abs pos = {pel+max(peers+ctrl)} < seq_len = {base_caches[sp][0].z_H.shape[1]}")
            print(f"[dry_run]   donor cache steps = {sorted(base_caches[0].keys())} (expect 0..{args.max_steps-1})")
        with open(os.path.join(args.output_dir, "dry_run.json"), "w") as f:
            json.dump({**meta, "n_target_samples": len(samples),
                       "samples_head": samples[:10]}, f, indent=2)
        print(f"[dry_run] OK — wrote {args.output_dir}/dry_run.json (no patching performed)")
        return

    t0 = time.time()
    if args.sweep:
        grid_levels = ["H", "L", "both"]
        grid_steps = ["all", "early", "late", "final"]
        rows = []
        for lv in grid_levels:
            for st in grid_steps:
                res = run_config(patcher, base_caches, base_preds, batches, samples, donor_pool,
                                 args.num_donors, pel, lv, parse_steps(st, args.max_steps),
                                 args.max_steps, k_ctrl=20, rng=np.random.default_rng(args.seed))
                if res.get("n_samples", 0):
                    rows.append({
                        "patch_level": lv, "patch_steps": st, "n_samples": res["n_samples"],
                        "unit_flip": res["unit_flip_rate"]["mean"],
                        "random_flip": res["random_flip_rate"]["mean"],
                        "unit_minus_random": res["unit_minus_random"]["mean"],
                        "umr_ci_lo": res["unit_minus_random"]["ci_lower"],
                        "umr_ci_hi": res["unit_minus_random"]["ci_upper"],
                        "flip_to_donor": res["flip_to_donor_rate"]["mean"],
                        "selectivity": res["selectivity"].get("mean"),
                        "wilcoxon_p": res["wilcoxon_p"],
                    })
                    print(f"  [{lv:4s} {st:5s}] unit={rows[-1]['unit_flip']:.3f} "
                          f"rand={rows[-1]['random_flip']:.3f} Δ={rows[-1]['unit_minus_random']:+.3f} "
                          f"to_donor={rows[-1]['flip_to_donor']:.3f} sel={rows[-1]['selectivity']}")
        with open(os.path.join(args.output_dir, "sweep_results.csv"), "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"[ccl] sweep done in {time.time()-t0:.1f}s -> {args.output_dir}/sweep_results.csv")
    else:
        res = run_config(patcher, base_caches, base_preds, batches, samples, donor_pool, args.num_donors, pel,
                         args.patch_level, parse_steps(args.patch_steps, args.max_steps),
                         args.max_steps, k_ctrl=20, rng=rng)
        with open(os.path.join(args.output_dir, "result.json"), "w") as f:
            json.dump({"meta": meta, "config": {"patch_level": args.patch_level,
                       "patch_steps": args.patch_steps}, "result": res}, f, indent=2)
        if res.get("n_samples"):
            print(f"[ccl] n={res['n_samples']} | unit_flip={res['unit_flip_rate']['mean']:.3f} "
                  f"random_flip={res['random_flip_rate']['mean']:.3f} "
                  f"Δ={res['unit_minus_random']['mean']:+.3f} "
                  f"[{res['unit_minus_random']['ci_lower']:.3f},{res['unit_minus_random']['ci_upper']:.3f}] "
                  f"flip_to_donor={res['flip_to_donor_rate']['mean']:.3f} "
                  f"selectivity={res['selectivity'].get('mean')} wilcoxon_p={res['wilcoxon_p']:.4g}")
        print(f"[ccl] done in {time.time()-t0:.1f}s -> {args.output_dir}/result.json")

    try:
        from scripts.core.provenance import write_meta
        write_meta(args.output_dir, "counterfactual_constraint_localization",
                   {**meta, "sweep": args.sweep, "patch_level": args.patch_level,
                    "patch_steps": args.patch_steps}, repo_root=REPO_ROOT)
    except Exception as e:
        print(f"[ccl] WARN _meta: {e}")


if __name__ == "__main__":
    main()
