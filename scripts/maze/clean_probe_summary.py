#!/usr/bin/env python3
"""Produce a corrected, reportable maze probe summary (data-integrity fix #3).

The raw probe_summary.json mixes two kinds of probes:
  * binary classification probes  -> accuracy in [0,1]  (trustworthy)
  * regression probes             -> R^2, which DIVERGES to large negative
    values on the maze's near-constant targets (e.g. path_f1 R^2 ~= -73 because
    path_f1 ~= 0.97 for almost every solved puzzle -> near-zero variance).

Those negative-R^2 numbers are an optimization/variance artifact, not a
decodability measurement, and must never be cited as "decoded at X%". This
script splits the summary into:
  * classification  -> the reportable decodability table, each row tagged with
        `informative` = (score_mean - baseline_mean) >= --headroom (default 0.03)
        so probes pinned at their majority baseline (is_dead_end, near_*_5) are
        flagged as carrying no signal.
  * regression_unreliable -> kept verbatim for transparency, marked unreliable.

CPU-only; reads/writes JSON. No model, no GPU.

Usage:
  python scripts/maze/clean_probe_summary.py \
      --in_dir results/maze/hardened/linear_probes \
      --headroom 0.03
"""
from __future__ import annotations
import os, sys, json, argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/maze/hardened/linear_probes")
    ap.add_argument("--summary_name", default="probe_summary.json")
    ap.add_argument("--headroom", type=float, default=0.03,
                    help="min (score-baseline) for a binary probe to count as informative")
    args = ap.parse_args()

    src = os.path.join(args.in_dir, args.summary_name)
    with open(src) as f:
        summary = json.load(f)

    classification, regression = {}, {}
    for key, row in summary.items():
        if not isinstance(row, dict) or "task" not in row:
            continue
        if row.get("task") == "binary":
            sc, bl = row.get("score_mean"), row.get("baseline_mean")
            row = dict(row)
            row["headroom"] = (None if (sc is None or bl is None) else round(float(sc) - float(bl), 4))
            row["informative"] = (row["headroom"] is not None and row["headroom"] >= args.headroom)
            classification[key] = row
        elif row.get("task") == "regression":
            row = dict(row)
            row["reportable"] = False
            row["note"] = ("R^2 unreliable: near-constant target -> tiny variance; "
                           "negative values are an artifact, not decodability")
            regression[key] = row

    # console: reportable classification decodability, grouped by (stream,target) across steps
    from collections import defaultdict
    grid = defaultdict(dict)
    for key, r in classification.items():
        grid[(r["stream"], r["target"])][r["step"]] = r
    lines = []
    lines.append(f"# Maze probe decodability (classification only, headroom>={args.headroom})\n")
    lines.append("| stream | target | informative? | " +
                 " | ".join(f"s{st}" for st in [0, 1, 2, 4, 8, 15]) + " |")
    lines.append("|---|---|---|" + "---|" * 6)
    for (stream, target), steps in sorted(grid.items()):
        any_inf = any(steps[st].get("informative") for st in steps)
        cells = []
        for st in [0, 1, 2, 4, 8, 15]:
            r = steps.get(st)
            if r is None:
                cells.append("—")
            else:
                sc = r.get("score_mean"); bl = r.get("baseline_mean")
                cells.append(f"{sc*100:.0f} (b{bl*100:.0f})" if sc is not None else "—")
        lines.append(f"| {stream} | {target} | {'YES' if any_inf else 'no'} | " + " | ".join(cells) + " |")
    table_md = "\n".join(lines) + "\n"

    out = {
        "_meta": {
            "source": src,
            "headroom_threshold": args.headroom,
            "n_classification": len(classification),
            "n_regression_unreliable": len(regression),
            "n_informative_classification": sum(1 for r in classification.values() if r.get("informative")),
            "fix": "data-integrity #3: regression probes flagged unreliable; binary probes tagged with headroom/informative",
        },
        "classification": classification,
        "regression_unreliable": regression,
    }
    clean_path = os.path.join(args.in_dir, "probe_summary_clean.json")
    with open(clean_path, "w") as f:
        json.dump(out, f, indent=2)
    md_path = os.path.join(args.in_dir, "probe_decodability_table.md")
    with open(md_path, "w") as f:
        f.write(table_md)

    print(table_md)
    print(f"[clean_probe_summary] classification={len(classification)} "
          f"(informative={out['_meta']['n_informative_classification']}) "
          f"regression_unreliable={len(regression)}")
    print(f"[clean_probe_summary] wrote {clean_path}")
    print(f"[clean_probe_summary] wrote {md_path}")


if __name__ == "__main__":
    main()
