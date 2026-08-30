#!/usr/bin/env python3
"""Generate complete endpoint linear-probe tables for the paper appendix.

The temporal figure shows representative targets. These tables cover every
target defined in Appendix ``app:probes`` at steps 0 and 15 for both recurrent
streams. Categorical targets report validation accuracy as mean +/- standard
deviation across seeds. Targets explicitly marked unreliable by the hardened
aggregates are retained as ``n/r`` rather than assigned misleading R^2 values.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "Final"
OUTPUT = RESULTS / "tables" / "full_linear_probe_results.tex"


def load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def target_label(name: str) -> str:
    return rf"\texttt{{{name.replace('_', r'\_')}}}"


def score_cell(row: dict[str, Any] | None, mean_key: str, std_key: str) -> str:
    if row is None or row.get("status") not in (None, "ok"):
        return "---"
    mean = row.get(mean_key)
    std = row.get(std_key)
    if mean is None or std is None:
        return "---"
    return rf"${100 * float(mean):.1f} \pm {100 * float(std):.1f}$"


def endpoint_row(
    target: str,
    getter: Callable[[str, int, str], dict[str, Any] | None],
    mean_key: str,
    std_key: str,
) -> str:
    cells = [
        score_cell(getter(stream, step, target), mean_key, std_key)
        for stream, step in (("H", 0), ("H", 15), ("L", 0), ("L", 15))
    ]
    return " & ".join([target_label(target), *cells]) + r" \\"


def unavailable_row(target: str) -> str:
    return " & ".join([target_label(target), "n/r", "n/r", "n/r", "n/r"]) + r" \\"


def table_header() -> str:
    return "\n".join([
        r"\toprule",
        r"\textbf{Target} & $\zH^{(0)}$ & $\zH^{(15)}$ & "
        r"$\zL^{(0)}$ & $\zL^{(15)}$ \\",
        r"\midrule",
    ])


def sudoku_table() -> str:
    summary = load(RESULTS / "sudoku/probes/e8_constraint_probes/probe_summary.json")
    targets = [
        "cell_digit",
        "is_given",
        "per_cell_correct",
        "violated_in_row",
        "violated_in_col",
        "violated_in_box",
    ]

    def get(stream: str, step: int, target: str) -> dict[str, Any] | None:
        return summary.get(f"z_{stream}_step{step}_{target}")

    rows = [endpoint_row(t, get, "val_mean", "val_std") for t in targets]
    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete Sudoku linear-probe endpoint results. Values are "
        r"validation accuracy (\%), reported as mean $\pm$ standard deviation over "
        r"five seeds. Each probe uses $32{,}400$ training cells and $8{,}100$ "
        r"validation cells.}",
        r"\label{tab:sudoku_probe_full}",
        r"\vskip 0.1in",
        r"\small",
        r"\begin{tabular}{@{}lcccc@{}}",
        table_header(),
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])


def maze_table() -> str:
    summary = load(RESULTS / "maze/hardened/linear_probes/probe_summary_clean.json")
    classification = summary["classification"]
    global_classification = [
        "exact_solved",
        "connects_start_goal",
        "valid_sg_path",
        "valid_optimal_path",
    ]
    global_unreliable = [
        "path_f1",
        "path_jaccard",
        "wall_path_rate",
        "path_length_ratio",
    ]
    local_classification = [
        "on_optimal_path",
        "is_wall",
        "is_free",
        "is_dead_end",
        "is_junction",
        "is_corner_on_path",
        "off_path_passable",
        "near_start_5",
        "near_goal_5",
    ]
    local_unreliable = [
        "num_passable_neighbors",
        "distance_to_start_norm",
        "distance_to_goal_norm",
    ]

    def get(stream: str, step: int, target: str) -> dict[str, Any] | None:
        return classification.get(f"z_{stream}_step{step}_{target}")

    rows = [r"\multicolumn{5}{@{}l}{\textit{Global path targets}} \\"]
    rows.extend(endpoint_row(t, get, "score_mean", "score_std")
                for t in global_classification)
    rows.extend(unavailable_row(t) for t in global_unreliable)
    rows.extend([r"\midrule", r"\multicolumn{5}{@{}l}{\textit{Local cell targets}} \\"])
    rows.extend(endpoint_row(t, get, "score_mean", "score_std")
                for t in local_classification)
    rows.extend(unavailable_row(t) for t in local_unreliable)
    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete Maze linear-probe endpoint results. Categorical targets "
        r"report validation accuracy (\%) as mean $\pm$ standard deviation over five "
        r"puzzle-disjoint seeds. Global probes use $800/200$ train/test mazes; local "
        r"probes use $102{,}400/25{,}600$ train/test cells. \textnormal{n/r} denotes "
        r"continuous targets whose $R^2$ is explicitly marked unreliable because the "
        r"evaluation target is nearly constant.}",
        r"\label{tab:maze_probe_full}",
        r"\vskip 0.1in",
        r"\scriptsize",
        r"\begin{tabular}{@{}lcccc@{}}",
        table_header(),
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])


def arc_table() -> str:
    summary = load(RESULTS / "arc/hardened/linear_probes/probe_summary.json")
    per_cell = [
        "per_cell_correct",
        "input_inside_grid",
        "output_inside_grid",
        "colour_changed",
        "same_as_input",
        "is_object_boundary",
        "input_colour",
        "output_colour",
    ]
    per_grid_classification = ["exact_solved", "shape_correct"]
    per_grid_unreliable = ["num_input_colours", "num_output_colours"]

    def get(stream: str, step: int, target: str) -> dict[str, Any] | None:
        return summary.get(f"z_{stream}_step{step}_{target}")

    rows = [r"\multicolumn{5}{@{}l}{\textit{Per-cell targets}} \\"]
    rows.extend(endpoint_row(t, get, "score_mean", "score_std") for t in per_cell)
    rows.extend([r"\midrule", r"\multicolumn{5}{@{}l}{\textit{Per-grid targets}} \\"])
    rows.extend(endpoint_row(t, get, "score_mean", "score_std")
                for t in per_grid_classification)
    rows.extend(unavailable_row(t) for t in per_grid_unreliable)
    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete ARC-AGI-2 linear-probe endpoint results. Reportable "
        r"categorical targets give validation accuracy (\%) as mean $\pm$ standard "
        r"deviation over five seeds, except \texttt{shape\_correct} at step~15 "
        r"($n=4$ valid seeds). Dashes denote one-class step-0 targets for which a "
        r"classifier cannot be fit. \textnormal{n/r} denotes near-constant continuous "
        r"targets with unreliable $R^2$.}",
        r"\label{tab:arc_probe_full}",
        r"\vskip 0.1in",
        r"\scriptsize",
        r"\begin{tabular}{@{}lcccc@{}}",
        table_header(),
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n\n".join([sudoku_table(), maze_table(), arc_table()]) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
