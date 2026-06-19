#!/usr/bin/env python3
"""Generate three master HTML result reports from the on-disk experiment outputs.

  results/reports/sudoku_report.html       all Sudoku scores + CIs
  results/reports/maze_report.html         all Maze scores + CIs (path-validity)
  results/reports/comparison_report.html   side-by-side Sudoku vs Maze

Reads whatever exists and skips what doesn't, so it can be re-run as more
experiments land (e.g. the N=1000 maze freeze/time-shift). Pure CPU/JSON — no
model, no GPU.

Usage:  python scripts/reports/generate_master_reports.py
"""
from __future__ import annotations
import os, sys, json, csv, html, time
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO_ROOT)
OUT = "results/reports"
os.makedirs(OUT, exist_ok=True)

CSS = """<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a;background:#fafafa}
h1{border-bottom:3px solid #333;padding-bottom:6px} h2{margin-top:34px;color:#0b3d91}
h3{margin-top:22px;color:#444} table{border-collapse:collapse;margin:10px 0;font-size:13px;background:#fff}
th,td{border:1px solid #ddd;padding:5px 9px;text-align:right} th{background:#0b3d91;color:#fff;position:sticky;top:0}
td:first-child,th:first-child{text-align:left} tr:nth-child(even){background:#f4f6fb}
.meta{color:#666;font-size:12px;margin:4px 0 12px} .neg{color:#b00020;font-weight:600} .pos{color:#0a7d00}
.sig{background:#fff3cd} .miss{color:#999;font-style:italic} .wrap{max-height:520px;overflow:auto;border:1px solid #eee}
code{background:#eee;padding:1px 4px;border-radius:3px}
</style>"""


def load(p) -> Optional[dict]:
    try:
        return json.load(open(p))
    except Exception:
        return None


def load_csv(p):
    try:
        return list(csv.DictReader(open(p)))
    except Exception:
        return None


def num(x):
    try:
        return float(x)
    except Exception:
        return None


def ci(node, pct=False, nd=3):
    """Format a {mean,ci_lower,ci_upper} dict or a bare number as 'm [lo, hi]'."""
    if node is None:
        return '<span class="miss">—</span>'
    mul = 100.0 if pct else 1.0
    suf = "%" if pct else ""
    if isinstance(node, dict) and "mean" in node:
        m, lo, hi = node.get("mean"), node.get("ci_lower"), node.get("ci_upper")
        if m is None:
            return '<span class="miss">—</span>'
        cls = "neg" if m < 0 else ("pos" if m > 0 else "")
        s = f'<span class="{cls}">{m*mul:.{nd}f}{suf}</span>'
        if lo is not None and hi is not None:
            s += f" [{lo*mul:.{nd}f}, {hi*mul:.{nd}f}]"
        return s
    v = num(node)
    if v is None:
        return html.escape(str(node))
    cls = "neg" if v < 0 else ("pos" if v > 0 else "")
    return f'<span class="{cls}">{v*mul:.{nd}f}{suf}</span>'


def f(x, nd=3, pct=False):
    v = num(x)
    if v is None:
        return '<span class="miss">—</span>'
    return f"{v*(100 if pct else 1):.{nd}f}{'%' if pct else ''}"


def table(headers, rows, cls=""):
    h = [f'<table class="{cls}"><tr>'] + [f"<th>{html.escape(str(x))}</th>" for x in headers] + ["</tr>"]
    for r in rows:
        h.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    h.append("</table>")
    return "".join(h)


def meta_line(path):
    m = load(os.path.join(os.path.dirname(path), "_meta.json"))
    if not m:
        return ""
    p = m.get("params", {})
    return (f"<div class='meta'>source: <code>{path}</code> · git {str(m.get('git_sha',''))[:8]} · "
            f"{m.get('timestamp','')} · N={p.get('num_puzzles') or p.get('n_puzzles','?')} "
            f"seeds={p.get('seeds','')} {('split='+p['split']) if 'split' in p else ''}</div>")


def missing(label):
    return f"<p class='miss'>[{label}] not found on disk — skipped.</p>"


def doc(title, body):
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title>{CSS}</head><body>{body}</body></html>"


# ════════════════════════ shared experiment renderers ════════════════════════

def render_ablation(path, maze=False):
    d = load(path)
    if not d:
        return None
    out = [meta_line(path)]
    out.append("<p><b>All-steps ablation</b> (zero the stream at every step):</p>")
    rows = [["token / exact accuracy Δ", ci(d.get("all_steps_delta"), pct=True)]]
    for k, v in (d.get("all_steps_maze_deltas") or {}).items():
        rows.append([f"{k} Δ (path-validity)" if maze else k, ci(v, pct=True)])
    out.append(table(["metric", "Δ mean [95% CI]"], rows))
    ps = d.get("per_step_ablation")
    if isinstance(ps, list) and ps:
        hdr = ["ACT step", "token Δ", "ablated acc"]
        prow = []
        for i, s in enumerate(ps):
            step = s.get("step", i)
            r = [step, f(s.get("delta_accuracy"), pct=True), f(s.get("ablated_accuracy"), pct=True)]
            md = s.get("maze_metric_deltas") or {}
            if md:
                r += [f(md.get("valid_sg_path"), pct=True), f(md.get("exact_solved"), pct=True)]
            prow.append(r)
        if prow and len(prow[0]) > 3:
            hdr += ["valid_sg_path Δ", "exact_solved Δ"]
        out.append("<p><b>Per-step single ablation</b>:</p>")
        out.append('<div class="wrap">' + table(hdr, prow) + "</div>")
    return "".join(out)


def render_freeze(path, maze=False):
    d = load(path)
    if not d:
        return None
    fh, fl = d.get("freeze_H", {}), d.get("freeze_L", {})

    def itr(node):
        return node.items() if isinstance(node, dict) else enumerate(node)
    rows = []
    keys = sorted({str(k) for k, _ in itr(fh)}, key=lambda x: int(x))
    fhd = dict(itr(fh)); fld = dict(itr(fl))
    for k in keys:
        h = fhd.get(k, fhd.get(int(k), {})); l = fld.get(k, fld.get(int(k), {}))
        row = [f"freeze from k={k}", ci(h.get("delta_accuracy"), pct=True), ci(l.get("delta_accuracy"), pct=True)]
        if maze:
            row += [ci((h.get("maze_metric_deltas") or {}).get("valid_sg_path"), pct=True)]
        rows.append(row)
    hdr = ["condition", "z_H token Δ", "z_L token Δ"] + (["z_H valid_sg_path Δ"] if maze else [])
    return meta_line(path) + table(hdr, rows)


def render_timeshift(path, maze=False):
    d = load(path)
    if not d:
        return None
    pp = d.get("per_pair")
    items = list(pp.items()) if isinstance(pp, dict) else list(enumerate(pp or []))
    rows = []
    for _, p in items[:40]:
        if not isinstance(p, dict):
            continue
        row = [f"donor {p.get('donor_step')}→recip {p.get('recipient_step')}",
               ci(p.get("delta_accuracy"), pct=True)]
        if maze:
            row.append(ci((p.get("maze_metric_deltas") or {}).get("valid_sg_path"), pct=True))
        rows.append(row)
    hdr = ["transfer", "token Δ"] + (["valid_sg_path Δ"] if maze else [])
    return meta_line(path) + f"<p>baseline acc = {f(d.get('baseline_accuracy'))}</p>" + table(hdr, rows)


def render_sae_sweep(path):
    rows = load_csv(path)
    if not rows:
        return None
    trows = []
    for r in sorted(rows, key=lambda x: (int(x["dict_size"]), float(x["l1_coeff"]))):
        trows.append([r["dict_size"], r["l1_coeff"], f(r.get("final_recon_loss"), 5),
                      f(r.get("L0"), 1), r.get("alive_count"), f(r.get("alive_frac")),
                      f(r.get("mean_sparsity"))])
    return meta_line(path) + table(
        ["dict_size", "λ (l1)", "recon_loss", "L0", "alive", "alive_frac", "mean_sparsity"], trows)


# ════════════════════════ Sudoku-specific ════════════════════════

def sudoku_linear_probes():
    d = load("results/probes/e8_constraint_probes/probe_summary.json")
    if not d:
        return missing("E8 linear probes")
    rows = []
    for k, v in d.items():
        rows.append([v.get("z_level"), v.get("step"), v.get("target"), v.get("metric"),
                     ci({"mean": v.get("val_mean"), "ci_lower": v.get("val_ci_lower"),
                         "ci_upper": v.get("val_ci_upper")}), v.get("n_seeds")])
    rows.sort(key=lambda r: (str(r[0]), r[1] if isinstance(r[1], int) else 0, str(r[2])))
    return meta_line("results/probes/e8_constraint_probes/_meta.json").replace(os.path.dirname(
        "results/probes/e8_constraint_probes/_meta.json"), "") + '<div class="wrap">' + table(
        ["z", "step", "target", "metric", "val mean [95% CI]", "seeds"], rows) + "</div>"


def sudoku_mlp_probes():
    rows = load_csv("results/probes/nonlinear_probes/sweep_results.csv")
    if not rows:
        return missing("nonlinear (MLP) probes")
    trows, deltas = [], []
    for r in rows:
        trows.append([r["z_level"], r["step"], r["target"], r["metric"],
                      ci({"mean": num(r["linear_val"]), "ci_lower": num(r.get("linear_ci_lower")),
                          "ci_upper": num(r.get("linear_ci_upper"))}),
                      ci({"mean": num(r["mlp_val"]), "ci_lower": num(r.get("mlp_ci_lower")),
                          "ci_upper": num(r.get("mlp_ci_upper"))}),
                      ci(num(r["delta_val"]))])
        deltas.append(num(r["delta_val"]))
    md = sum(deltas) / len(deltas) if deltas else 0
    head = (f"<p><b>Mean Δ(MLP−linear)</b> over {len(deltas)} probes = "
            f"<b>{md:+.4f}</b> → {'linearly encoded' if abs(md)<0.02 else 'non-linear structure'}.</p>")
    return meta_line("results/probes/nonlinear_probes/_meta.json") + head + '<div class="wrap">' + table(
        ["z", "step", "target", "metric", "linear [CI]", "MLP [CI]", "Δ"], trows) + "</div>"


def sudoku_directed():
    d = load("results/controlled/directed_ablation/analysis.json")
    if not d:
        return missing("directed ablation")
    rows = []
    for tgt, v in d.items():
        if not isinstance(v, dict):
            continue
        sig = v.get("significant_at_005")
        rows.append([tgt, ci(v.get("probe_delta_accuracy"), pct=True),
                     ci(v.get("random_control_delta_accuracy"), pct=True),
                     f(v.get("paired_p_value"), 4), f(v.get("wilcoxon_p_value"), 4),
                     f(v.get("cohens_d"), 3), "✔" if sig else ""])
    return meta_line("results/controlled/directed_ablation/_meta.json") + table(
        ["direction", "probe Δ [CI]", "random Δ [CI]", "paired p", "wilcoxon p", "Cohen's d", "sig@.05"], rows)


def sudoku_sae_causal():
    d = load("results/sae_study/causal_ablation/aggregate.json")
    if not d:
        return missing("SAE causal ablation")
    conds = d.get("conditions", {})
    rows = [[k, f(v.get("mean_delta_acc"), pct=True), f(v.get("std_delta_acc"), pct=True), v.get("n_samples")]
            for k, v in conds.items()]
    out = [f"<p>N puzzles = {d.get('n_puzzles')}</p>",
           table(["condition", "mean Δacc", "std", "n"], rows)]
    st = d.get("statistical_tests", {})
    strows = [[k, f(v.get("t_statistic"), 3), f(v.get("p_value"), 4),
               "✔" if v.get("significant_0.05") else ""] for k, v in st.items()]
    out.append("<p><b>Statistical tests</b>:</p>" + table(["comparison", "t", "p", "sig@.05"], strows))
    return "".join(out)


# ════════════════════════ Maze-specific ════════════════════════

def maze_step_dynamics():
    d = load("results/maze/hardened/step_dynamics/aggregate.json")
    if not d:
        return missing("maze step dynamics")
    ks = ["step", "exact_solved", "valid_sg_path", "connects_start_goal", "path_f1", "token_acc"]

    def cell(s, k):  # step_dynamics metrics are {mean,std,n} dicts
        if k == "step":
            return s.get("step")
        v = s.get(k)
        return ci(v, nd=3) if isinstance(v, dict) else f(v, 3)
    rows = [[cell(s, k) for k in ks] for s in d.get("per_step", [])]
    return f"<p>N={d.get('num_puzzles')}</p>" + '<div class="wrap">' + table(ks, rows) + "</div>"


def maze_probes(path, is_mlp=False):
    d = load(path)
    if not d:
        return None
    out = [meta_line(path)]
    for grp, title in [("local_probes", "Local (per-cell) features"), ("global_probes", "Global (per-puzzle) path-validity")]:
        rows = []
        for r in d.get(grp, []):
            row = [r.get("stream"), r.get("step"), r.get("target"), r.get("task"),
                   ci({"mean": r.get("score_mean"), "ci_lower": r.get("score_ci_lower"),
                       "ci_upper": r.get("score_ci_upper")}), f(r.get("baseline_mean"))]
            if is_mlp:
                row += [f(r.get("linear_score_mean")), ci(r.get("delta_mlp_minus_linear"))]
            rows.append(row)
        rows.sort(key=lambda r: (str(r[0]), str(r[2]), r[1] if isinstance(r[1], int) else 0))
        hdr = ["stream", "step", "feature", "task", "score mean [95% CI]", "baseline"]
        if is_mlp:
            hdr += ["linear", "Δ(MLP−lin)"]
        out.append(f"<h3>{title}</h3>" + '<div class="wrap">' + table(hdr, rows) + "</div>")
    return "".join(out)


def maze_trajectory():
    d = load("results/maze/hardened/trajectory_pca/trajectory_pca.json")
    if not d:
        return missing("maze trajectory PCA")
    g = d.get("trajectory_geometry", {})
    out = [meta_line("results/maze/hardened/trajectory_pca/trajectory_pca.json"),
           table(["quantity", "value [95% CI]"], [
               ["PC1 explained var", ci({"mean": d.get("pc1_explained"),
                "ci_lower": d.get("pc1_explained_bootstrap_ci", [None, None, None])[1],
                "ci_upper": d.get("pc1_explained_bootstrap_ci", [None, None, None])[2]})],
               ["2-PC cumulative var", ci({"mean": d.get("cumulative_2pc_explained"),
                "ci_lower": d.get("cumulative_2pc_bootstrap_ci", [None, None, None])[1],
                "ci_upper": d.get("cumulative_2pc_bootstrap_ci", [None, None, None])[2]})],
               ["mean consecutive-step cos", ci(g.get("mean_consecutive_cos"))],
               ["cos(step0, final)", ci(g.get("cos_step0_to_final"))],
               ["2-D path length", ci(g.get("path_length_2d"))],
           ])]
    pv = d.get("per_step_path_validity", {})
    if pv:
        steps = len(next(iter(pv.values())))
        rows = [[k] + [f(v[s], 3) for s in range(steps)] for k, v in pv.items()]
        out.append("<p>Per-step path-validity along the trajectory:</p>"
                   + table(["metric"] + [f"s{i}" for i in range(steps)], rows))
    return "".join(out)


def maze_geometry():
    d = load("results/maze/hardened/probe_geometry/geometric_analysis.json")
    if not d:
        return missing("maze probe geometry")
    pca = d.get("pca_pc1_ensemble", [])
    prows = [[p.get("group"), p.get("n_directions"),
              ci({"mean": p.get("pc1_explained_mean"), "ci_lower": p.get("pc1_explained_ci_lower"),
                  "ci_upper": p.get("pc1_explained_ci_upper")}), p.get("n_seeds")] for p in pca]
    out = ["<p>PCA of the binary feature directions (low PC1 ⇒ distributed directions; "
           "high PC1 ⇒ they share a subspace):</p>",
           table(["group (stream_step)", "#dirs", "PC1 explained [CI]", "seeds"], prows)]
    cos = sorted(d.get("constraint_cosines_ensemble", []),
                 key=lambda c: -abs(c.get("cosine_mean", 0)))[:25]
    crows = [[c.get("group"), c.get("probe_a"), c.get("probe_b"),
              ci({"mean": c.get("cosine_mean"), "ci_lower": c.get("cosine_ci_lower"),
                  "ci_upper": c.get("cosine_ci_upper")})] for c in cos]
    out.append("<p>Top-25 |cosine| feature-direction pairs:</p>"
               + '<div class="wrap">' + table(["group", "feature A", "feature B", "cosine [CI]"], crows) + "</div>")
    return "".join(out)


def maze_patching():
    d = load("results/maze/hardened/patching_spatial/aggregate.json")
    if not d:
        return missing("maze patching")
    md = d.get("maze_metric_deltas_by_group_level_step", {})
    rows = []
    for key, m in (md.items() if isinstance(md, dict) else []):
        v = m.get("valid_sg_path") if isinstance(m, dict) else None
        rows.append([key, ci(v, pct=True) if v is not None else f(m)])
    return (f"<p>N pairs={d.get('num_pairs')}, patch_steps={d.get('patch_steps')}</p>"
            + '<div class="wrap">' + table(["group / level / step", "valid_sg_path Δ"], rows[:60]) + "</div>")


# ════════════════════════ build the three reports ════════════════════════

def section(title, content):
    return f"<h2>{title}</h2>" + (content or missing(title))


def build_sudoku():
    b = [f"<h1>Sudoku — All Results</h1><div class='meta'>generated {time.strftime('%Y-%m-%d %H:%M')}</div>"]
    b.append(section("Ablation — z_H (E1)", render_ablation("results/controlled/ablation/zH/aggregate.json")))
    b.append(section("Ablation — z_L", render_ablation("results/controlled/ablation/zL/aggregate.json")))
    b.append(section("Freeze (E2)", render_freeze("results/controlled/freeze/aggregate.json")))
    b.append(section("Time-shift (E5)", render_timeshift("results/controlled/time_shift/aggregate.json")))
    b.append(section("Linear probes (E8)", sudoku_linear_probes()))
    b.append(section("Non-linear / MLP probes (E9b, 5-seed)", sudoku_mlp_probes()))
    b.append(section("Directed ablation (E9)", sudoku_directed()))
    b.append(section("SAE sweep (E10)", render_sae_sweep("results/sae_study/sweep_results.csv")))
    b.append(section("SAE causal ablation (E10)", sudoku_sae_causal()))
    open(f"{OUT}/sudoku_report.html", "w").write(doc("Sudoku — All Results", "".join(b)))


def build_maze():
    b = [f"<h1>Maze 30×30 — All Results (path-validity)</h1><div class='meta'>generated {time.strftime('%Y-%m-%d %H:%M')}</div>"]
    b.append(section("Step dynamics", maze_step_dynamics()))
    b.append(section("Ablation — z_H", render_ablation("results/maze/hardened/ablation_controlled/zH/aggregate.json", maze=True)))
    b.append(section("Ablation — z_L", render_ablation("results/maze/hardened/ablation_controlled/zL/aggregate.json", maze=True)))
    b.append(section("Freeze", render_freeze("results/maze/hardened/freeze_controlled/aggregate.json", maze=True)))
    b.append(section("Time-shift", render_timeshift("results/maze/hardened/time_shift_controlled/aggregate.json", maze=True)))
    b.append(section("Linear probes", maze_probes("results/maze/hardened/linear_probes/probe_results.json")))
    b.append(section("Non-linear / MLP probes", maze_probes("results/maze/hardened/mlp_probes/probe_results.json", is_mlp=True)))
    b.append(section("SAE sweep (z_H)", render_sae_sweep("results/maze/sae_study/sweep_results.csv")))
    b.append(section("z_H trajectory PCA", maze_trajectory()))
    b.append(section("Probe-weight geometry", maze_geometry()))
    b.append(section("Activation patching (spatial)", maze_patching()))
    open(f"{OUT}/maze_report.html", "w").write(doc("Maze — All Results", "".join(b)))


def _allsteps(path, key):
    d = load(path)
    if not d:
        return None
    if key == "token":
        return d.get("all_steps_delta")
    return (d.get("all_steps_maze_deltas") or {}).get(key)


def build_comparison():
    b = [f"<h1>Sudoku vs Maze — Side by Side</h1><div class='meta'>generated {time.strftime('%Y-%m-%d %H:%M')} · "
         "Δ shown as mean [95% CI]; Maze scored on path-validity, Sudoku on token/exact.</div>"]

    # 1. Headline causal effect of ablating each stream (all steps)
    rows = [
        ["z_H all-steps ablation — token/exact Δ",
         ci(_allsteps("results/controlled/ablation/zH/aggregate.json", "token"), pct=True),
         ci(_allsteps("results/maze/hardened/ablation_controlled/zH/aggregate.json", "token"), pct=True)],
        ["z_H all-steps ablation — task metric Δ",
         ci(_allsteps("results/controlled/ablation/zH/aggregate.json", "token"), pct=True),
         ci(_allsteps("results/maze/hardened/ablation_controlled/zH/aggregate.json", "valid_sg_path"), pct=True)],
        ["z_L all-steps ablation — task metric Δ",
         ci(_allsteps("results/controlled/ablation/zL/aggregate.json", "token"), pct=True),
         ci(_allsteps("results/maze/hardened/ablation_controlled/zL/aggregate.json", "valid_sg_path"), pct=True)],
    ]
    b.append("<h2>1. Is the stream causally necessary? (all-steps ablation)</h2>")
    b.append("<p class='meta'>Maze rows use <b>valid_sg_path</b>; the z_H token row exposes the metric trap "
             "(token accuracy barely moves while the path collapses).</p>")
    b.append(table(["quantity", "Sudoku", "Maze"], rows))

    # 2. Freeze decay (k=0)
    def freeze0(path):
        d = load(path)
        if not d:
            return None
        fh = d.get("freeze_H", {})
        e = fh.get("0", fh.get(0)) if isinstance(fh, dict) else (fh[0] if fh else None)
        return (e or {}).get("delta_accuracy")
    b.append("<h2>2. Freeze z_H from step 0 (token Δ)</h2>")
    b.append(table(["quantity", "Sudoku", "Maze"],
                   [["freeze-from-0 token Δ", ci(freeze0("results/controlled/freeze/aggregate.json"), pct=True),
                     ci(freeze0("results/maze/hardened/freeze_controlled/aggregate.json"), pct=True)]]))

    # 3. Representation linearity (mean MLP-linear delta)
    def mean_delta(path):  # binary accuracy probes only (bounded, comparable)
        rows = load_csv(path)
        if not rows:
            return None
        ds = [num(r["delta_val"]) for r in rows
              if r.get("task") == "binary" and num(r.get("delta_val")) is not None]
        return sum(ds) / len(ds) if ds else None

    def maze_mean_delta(path):  # binary accuracy probes only
        d = load(path)
        if not d:
            return None
        ds = [r.get("delta_mlp_minus_linear") for r in d.get("local_probes", []) + d.get("global_probes", [])
              if r.get("task") == "binary" and isinstance(r.get("delta_mlp_minus_linear"), (int, float))]
        return sum(ds) / len(ds) if ds else None
    sd = mean_delta("results/probes/nonlinear_probes/sweep_results.csv")
    mz = maze_mean_delta("results/maze/hardened/mlp_probes/probe_results.json")
    b.append("<h2>3. Are the representations linear? (mean Δ MLP − linear over all probes)</h2>")
    b.append("<p class='meta'>≈0 ⇒ a linear readout already captures the feature (no hidden non-linear structure).</p>")
    b.append(table(["quantity", "Sudoku", "Maze"],
                   [["mean Δ(MLP − linear)", ci(sd), ci(mz)]]))

    # 4. SAE frontier at a matched config (d=2048, λ=0.01)
    def sae_pick(path, dsz="2048", l1="0.01"):
        rows = load_csv(path)
        if not rows:
            return None
        for r in rows:
            if r["dict_size"] == dsz and abs(num(r["l1_coeff"]) - float(l1)) < 1e-6:
                return r
        return None
    ss = sae_pick("results/sae_study/sweep_results.csv")
    mzs = sae_pick("results/maze/sae_study/sweep_results.csv")
    b.append("<h2>4. SAE on z_H at matched config (d=2048, λ=0.01)</h2>")
    b.append(table(["quantity", "Sudoku", "Maze"], [
        ["alive_frac", f(ss and ss.get("alive_frac")), f(mzs and mzs.get("alive_frac"))],
        ["L0 (avg active features)", f(ss and ss.get("L0"), 1), f(mzs and mzs.get("L0"), 1)],
        ["recon_loss", f(ss and ss.get("final_recon_loss"), 5), f(mzs and mzs.get("final_recon_loss"), 5)],
    ]))

    # 5. Trajectory low-dimensionality (maze available; sudoku if present)
    def traj(path):
        d = load(path)
        return d
    mt = traj("results/maze/hardened/trajectory_pca/trajectory_pca.json")
    st = traj("results/sudoku/trajectory_pca/trajectory_pca.json") or traj("results/trajectory_pca/trajectory_pca.json")
    b.append("<h2>5. z_H trajectory dimensionality (PCA)</h2>")
    b.append(table(["quantity", "Sudoku", "Maze"], [
        ["PC1 explained var", ci(st and st.get("pc1_explained")), ci(mt and mt.get("pc1_explained"))],
        ["2-PC cumulative var", ci(st and st.get("cumulative_2pc_explained")), ci(mt and mt.get("cumulative_2pc_explained"))],
    ]))
    if st is None:
        b.append("<p class='meta'>Sudoku trajectory PCA not yet generated in this format "
                 "(run scripts/maze/trajectory_pca_maze.py on a Sudoku z_H dump).</p>")

    open(f"{OUT}/comparison_report.html", "w").write(doc("Sudoku vs Maze", "".join(b)))


def main():
    build_sudoku(); build_maze(); build_comparison()
    for name in ("sudoku_report", "maze_report", "comparison_report"):
        print(f"wrote {OUT}/{name}.html")


if __name__ == "__main__":
    main()
