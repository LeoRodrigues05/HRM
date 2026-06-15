#!/usr/bin/env python3
"""Compact progress + results reader for the hardened maze suite."""
import json, os, sys, glob

BASE = "results/maze/hardened"

def ci(d):
    if not isinstance(d, dict):
        return str(d)
    return f"{d['mean']:+.4f} [{d.get('ci_lower',0):+.4f},{d.get('ci_upper',0):+.4f}]"

def count(p):
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0

def status():
    print("=== HARDENED MAZE SUITE PROGRESS ===")
    print("step_dynamics:", "DONE" if os.path.exists(f"{BASE}/step_dynamics/aggregate.json") else "...")
    for lv in ("zH", "zL"):
        f = f"{BASE}/ablation_controlled/{lv}/per_puzzle.jsonl"
        done = os.path.exists(f"{BASE}/ablation_controlled/{lv}/aggregate.json")
        print(f"ablation {lv}: {count(f)}/200", "(agg DONE)" if done else "")
    pf = f"{BASE}/patching_spatial/per_pair.jsonl"
    print(f"patching: {count(pf)}/100", "(agg DONE)" if os.path.exists(f'{BASE}/patching_spatial/aggregate.json') else "")
    tf = f"{BASE}/time_shift_controlled/per_puzzle.jsonl"
    print(f"time_shift: {count(tf)}/80", "(agg DONE)" if os.path.exists(f'{BASE}/time_shift_controlled/aggregate.json') else "")
    for lv in ("",):
        pass
    fagg = f"{BASE}/freeze_controlled/aggregate.json"
    ff = glob.glob(f"{BASE}/freeze_controlled/*per_puzzle*.jsonl") + glob.glob(f"{BASE}/freeze_controlled/per_puzzle.jsonl")
    print("freeze:", "agg DONE" if os.path.exists(fagg) else (f"{count(ff[0])}/100" if ff else "..."))

def patching():
    a = json.load(open(f"{BASE}/patching_spatial/aggregate.json"))
    md = a["maze_metric_deltas_by_group_level_step"]
    print(f"=== PATCHING cross-puzzle, group=all, N={a['num_pairs']} ===")
    for level in ("H", "L"):
        print(f"-- level {level} --")
        for metric in ("valid_sg_path", "exact_solved", "connects_start_goal", "token_acc"):
            row = md.get(metric, {}).get("all", {}).get(level, {})
            print(f"  {metric:>20}:", {s: ci(row[s]) for s in sorted(row)})

def freeze():
    a = json.load(open(f"{BASE}/freeze_controlled/aggregate.json"))
    print(f"=== FREEZE N={a.get('num_puzzles')} baseline_maze={('baseline_maze_metrics' in a)} ===")
    for stream in ("freeze_H", "freeze_L"):
        if stream not in a:
            continue
        print(f"-- {stream} (valid_sg_path / token) --")
        fh = a[stream]
        for k in sorted(fh, key=lambda x: int(x))[:8]:
            node = fh[k]
            mm = node.get("maze_metric_deltas", {})
            vs = mm.get("valid_sg_path")
            tok = node.get("delta_accuracy")
            print(f"  k={k}: valid_sg={ci(vs) if vs else 'NA'}  token={ci(tok) if tok else 'NA'}")

def timeshift():
    a = json.load(open(f"{BASE}/time_shift_controlled/aggregate.json"))
    print(f"=== TIME_SHIFT N={a['num_puzzles']} has_maze={('baseline_maze_metrics' in a)} ===")
    pp = a["per_pair"]
    # summarize by recipient step: best/worst valid_sg_path delta
    rows = []
    for key, v in pp.items():
        mm = v.get("maze_metric_deltas", {}).get("valid_sg_path")
        if mm:
            rows.append((key, v["donor_step"], v["recipient_step"], mm["mean"], mm["ci_lower"], mm["ci_upper"]))
    rows.sort(key=lambda r: r[3])
    print("  most damaging transfers (valid_sg_path delta):")
    for r in rows[:6]:
        print(f"    {r[0]:>10} donor={r[1]} recip={r[2]}: {r[3]:+.3f} [{r[4]:+.3f},{r[5]:+.3f}]")
    print("  least damaging:")
    for r in rows[-4:]:
        print(f"    {r[0]:>10} donor={r[1]} recip={r[2]}: {r[3]:+.3f} [{r[4]:+.3f},{r[5]:+.3f}]")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"status": status, "patching": patching, "freeze": freeze, "timeshift": timeshift}[cmd]()
