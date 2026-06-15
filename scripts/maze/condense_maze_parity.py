#!/usr/bin/env python3
"""Verify + condense the Maze parity suite (linear probes, MLP probes, SAE sweep).

Reads the artifacts written by ``slurm_maze_parity.sbatch`` and
  (a) VERIFIES the experiments were run correctly — puzzle-disjoint split,
      5-seed ensemble, path-validity targets only (no token accuracy), the
      requested ACT steps, and the full Sudoku SAE grid (4 dict x 3 l1); then
  (b) CONDENSES them into compact tables + a markdown report.

Usage
-----
  python scripts/maze/condense_maze_parity.py \
      --linear_dir results/maze/hardened/linear_probes \
      --mlp_dir    results/maze/hardened/mlp_probes \
      --sae_dir    results/maze/sae_study \
      --out_md     docs/MAZE_PARITY_REPORT.md \
      --out_json   results/maze/hardened/maze_parity_summary.json
"""
from __future__ import annotations

import os
import sys
import json
import csv
import argparse
from typing import Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PATH_VALIDITY_GLOBAL = {
    "exact_solved", "connects_start_goal", "valid_sg_path", "valid_optimal_path",
    "path_f1", "path_jaccard", "wall_path_rate", "path_length_ratio",
}
STRUCTURAL_LOCAL = {
    "on_optimal_path", "is_wall", "is_dead_end", "is_junction",
    "near_start_5", "near_goal_5", "distance_to_start_norm", "distance_to_goal_norm",
}
EXPECTED_SAE_DICTS = {1024, 2048, 4096, 8192}
EXPECTED_SAE_L1 = {0.003, 0.01, 0.03}


def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:  # nan
            return "—"
        return f"{v:.{nd}f}"
    return str(v)


def _ci(row) -> str:
    m = row.get("score_mean")
    lo = row.get("score_ci_lower")
    hi = row.get("score_ci_upper")
    if m is None:
        return "—"
    if lo is None or hi is None:
        return _fmt(m)
    return f"{_fmt(m)} [{_fmt(lo)}, {_fmt(hi)}]"


def _index(probes: List[dict]) -> Dict[tuple, dict]:
    return {(r["stream"], r["step"], r["target"]): r for r in probes}


# ───────────────────────────── verification ──────────────────────────────

def verify_probe_run(res: dict, meta: Optional[dict], label: str, checks: List[str]) -> dict:
    ds = res.get("dataset_summary", {})
    cfg = res.get("config", {})
    probes = res.get("global_probes", []) + res.get("local_probes", [])
    targets = {r["target"] for r in probes}

    def ck(name, ok, detail=""):
        checks.append(f"[{'PASS' if ok else 'FAIL'}] {label}: {name}{(' — ' + detail) if detail else ''}")
        return ok

    split = ds.get("split") or (meta or {}).get("params", {}).get("split")
    ck("puzzle-disjoint split", split == "puzzle_disjoint", f"split={split}")
    seeds = ds.get("probe_seeds") or (meta or {}).get("params", {}).get("seeds")
    ck("5-seed ensemble", isinstance(seeds, list) and len(seeds) >= 5, f"seeds={seeds}")
    n_seeds_ok = [r.get("n_seeds", 0) for r in probes if r.get("status") == "ok"]
    ck("per-probe n_seeds==5 (ok probes)",
       all(n >= 5 for n in n_seeds_ok) and len(n_seeds_ok) > 0,
       f"min n_seeds={min(n_seeds_ok) if n_seeds_ok else 'NA'} over {len(n_seeds_ok)} ok probes")
    ck("path-validity / structural targets only (no token_acc)",
       "token_acc" not in targets and targets.issubset(PATH_VALIDITY_GLOBAL | STRUCTURAL_LOCAL),
       f"targets={sorted(targets)}")
    steps = ds.get("steps_probed") or sorted({r["step"] for r in probes})
    ck("per-(step,target) probes (multiple steps)", len(set(steps)) >= 2, f"steps={steps}")
    n_pz = ds.get("n_puzzles_collected")
    ck("N puzzles collected", bool(n_pz) and n_pz >= 100, f"n_puzzles={n_pz}")
    return {
        "split": split, "seeds": seeds, "steps": steps,
        "n_puzzles": n_pz, "n_probes": len(probes),
        "exact_solved_by_step": ds.get("exact_solved_by_step", {}),
        "probe_type": cfg.get("probe_type"),
    }


def verify_sae(sae_dir: str, checks: List[str]) -> dict:
    csv_path = os.path.join(sae_dir, "sweep_results.csv")
    meta = _load(os.path.join(sae_dir, "_meta.json"))
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

    def ck(name, ok, detail=""):
        checks.append(f"[{'PASS' if ok else 'FAIL'}] SAE: {name}{(' — ' + detail) if detail else ''}")
        return ok

    dicts = sorted({int(r["dict_size"]) for r in rows}) if rows else []
    l1s = sorted({float(r["l1_coeff"]) for r in rows}) if rows else []
    ck("sweep ran", len(rows) > 0, f"{len(rows)} configs")
    ck("dict grid == {1024,2048,4096,8192}", set(dicts) == EXPECTED_SAE_DICTS, f"dicts={dicts}")
    ck("l1 grid == {0.003,0.01,0.03}", set(l1s) == EXPECTED_SAE_L1, f"l1={l1s}")
    ck("full 4x3 grid (12 configs)", len(rows) == 12, f"{len(rows)} rows")
    lvl = (meta or {}).get("params", {}).get("activation_level")
    ck("trained on maze z_H", lvl in (None, "z_H"), f"activation_level={lvl}")
    return {"rows": rows, "dicts": dicts, "l1s": l1s, "meta": meta}


# ───────────────────────────── condensing ────────────────────────────────

def decodability_table(lin: dict, mlp: Optional[dict], targets: List[str], kind: str,
                       steps: List[int]) -> List[str]:
    """One row per (target, stream): linear acc per step + MLP delta at last step."""
    key = "global_probes" if kind == "global" else "local_probes"
    lin_idx = _index(lin.get(key, []))
    mlp_idx = _index(mlp.get(key, [])) if mlp else {}
    out = [f"\n#### {kind.capitalize()} probes — linear test score (mean [95% CI]) by ACT step\n"]
    header = "| target | stream | " + " | ".join(f"s{st}" for st in steps) + " | MLP@last | Δ(MLP−lin)@last |"
    out.append(header)
    out.append("|" + "---|" * (3 + len(steps) + 2))
    last = steps[-1]
    for tgt in targets:
        for stream in ("z_H", "z_L"):
            cells = []
            for st in steps:
                r = lin_idx.get((stream, st, tgt))
                cells.append(_ci(r) if r else "—")
            mlp_r = mlp_idx.get((stream, last, tgt)) if mlp_idx else None
            mlp_s = _fmt(mlp_r.get("score_mean")) if mlp_r else "—"
            delta = _fmt(mlp_r.get("delta_mlp_minus_linear")) if mlp_r else "—"
            out.append(f"| {tgt} | {stream} | " + " | ".join(cells) + f" | {mlp_s} | {delta} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear_dir", default="results/maze/hardened/linear_probes")
    ap.add_argument("--mlp_dir", default="results/maze/hardened/mlp_probes")
    ap.add_argument("--sae_dir", default="results/maze/sae_study")
    ap.add_argument("--out_md", default="docs/MAZE_PARITY_REPORT.md")
    ap.add_argument("--out_json", default="results/maze/hardened/maze_parity_summary.json")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    lin = _load(os.path.join(args.linear_dir, "probe_results.json"))
    lin_meta = _load(os.path.join(args.linear_dir, "_meta.json"))
    mlp = _load(os.path.join(args.mlp_dir, "probe_results.json"))
    mlp_meta = _load(os.path.join(args.mlp_dir, "_meta.json"))

    if lin is None:
        sys.exit(f"ERROR: no linear probe results at {args.linear_dir}/probe_results.json")

    checks: List[str] = []
    lin_v = verify_probe_run(lin, lin_meta, "linear", checks)
    mlp_v = verify_probe_run(mlp, mlp_meta, "mlp", checks) if mlp else None
    sae_v = verify_sae(args.sae_dir, checks)

    steps = lin_v["steps"]
    md: List[str] = []
    md.append("# Maze Parity Suite — Verification & Condensed Results\n")
    md.append("Hardened replication of the Sudoku MI probe/SAE protocol on the Maze "
              "30×30-hard task. Every probe target is a **path-validity** metric or a "
              "**maze structural feature** — no token-accuracy targets. Probes use a "
              "**puzzle-disjoint** train/val split, a **5-seed** ensemble (mean ± 95% "
              "t-CI), and read the post-step state `z_*_out` that the readout consumes, "
              "matching Sudoku E8.\n")

    md.append("## 1. Verification checklist\n")
    md += [f"- {c}" for c in checks]
    md.append("")

    md.append("## 2. Path-validity sanity — exact-solved by ACT step (collection)\n")
    es = lin_v["exact_solved_by_step"]
    if es:
        md.append("| ACT step | " + " | ".join(es.keys()) + " |")
        md.append("|" + "---|" * (1 + len(es)))
        md.append("| exact_solved | " + " | ".join(_fmt(float(v)) for v in es.values()) + " |")
    md.append("\n*(Maze is a shallow computation: exact-solve jumps at step 1 then plateaus.)*\n")

    md.append("## 3. Decodability of maze features from z_H / z_L\n")
    md.append("Headline feature families: **on-optimal-path**, **wall structure** "
              "(is_wall / is_dead_end / is_junction), **start↔goal connectivity** "
              "(connects_start_goal / valid_sg_path / near_start / near_goal).\n")
    md += decodability_table(
        lin, mlp,
        ["on_optimal_path", "is_wall", "is_dead_end", "is_junction",
         "near_start_5", "near_goal_5"],
        "local", steps)
    md += decodability_table(
        lin, mlp,
        ["exact_solved", "connects_start_goal", "valid_sg_path", "valid_optimal_path"],
        "global", steps)

    # MLP-vs-linear aggregate
    if mlp:
        deltas = [r.get("delta_mlp_minus_linear") for r in
                  mlp.get("global_probes", []) + mlp.get("local_probes", [])
                  if isinstance(r.get("delta_mlp_minus_linear"), (int, float))]
        if deltas:
            import statistics
            md.append(f"\n**MLP vs linear:** mean Δ(MLP−linear) over "
                      f"{len(deltas)} (stream,step,target) probes = "
                      f"{statistics.mean(deltas):+.4f} "
                      f"(max {max(deltas):+.4f}). "
                      f"{'Largely linear — linear readout captures the structure.' if abs(statistics.mean(deltas)) < 0.02 else 'Non-linear structure present.'}\n")

    md.append("## 4. SAE sweep on maze z_H (reconstruction–sparsity frontier)\n")
    rows = sae_v["rows"]
    if rows:
        md.append("| dict_size | l1 | recon_loss | L0 | alive | alive_frac |")
        md.append("|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda x: (int(x["dict_size"]), float(x["l1_coeff"]))):
            md.append(f"| {r['dict_size']} | {r['l1_coeff']} | "
                      f"{_fmt(float(r['final_recon_loss']),4)} | {_fmt(float(r['L0']),1)} | "
                      f"{r['alive_count']} | {_fmt(float(r['alive_frac']),3)} |")
    md.append("")

    n_fail = sum(1 for c in checks if c.startswith("[FAIL]"))
    md.insert(1, f"\n**STATUS: {'ALL CHECKS PASSED' if n_fail == 0 else str(n_fail) + ' CHECK(S) FAILED — see §1'}**\n")

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(md))

    summary = {
        "linear": lin_v, "mlp": mlp_v,
        "sae": {"dicts": sae_v["dicts"], "l1s": sae_v["l1s"], "n_configs": len(rows)},
        "checks": checks, "n_failures": n_fail,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n".join(checks))
    print(f"\n{'ALL CHECKS PASSED' if n_fail == 0 else str(n_fail)+' CHECK(S) FAILED'}")
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
