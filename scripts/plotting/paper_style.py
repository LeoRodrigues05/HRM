"""Single source of truth for the AAAI paper figure style.

Every script that produces a figure used in the paper imports and calls
``apply_paper_style()`` so that all figures share ONE font family, math font,
and size scale. The paper body is Times (TeX Gyre Termes); we match it with a
metric-compatible serif (Liberation Serif / Nimbus Roman) and STIX math, and
embed TrueType (Type 42) fonts because AAAI forbids Type 3.

Sizing convention
-----------------
Figures are authored at (approximately) the width at which they appear in the
paper, so LaTeX applies a scale of ~1x and the point sizes below are the point
sizes the reader actually sees. Use ``COL_W`` for ``\\columnwidth`` figures and
``TEXT_W`` for full-width ``figure*`` figures when choosing ``figsize``.
"""

from __future__ import annotations

import matplotlib as mpl

# AAAI two-column geometry (from aaai2027.sty: textwidth 7.0in, columnsep 0.375in)
TEXT_W = 7.0                      # full text width (figure*)
COL_W = (TEXT_W - 0.375) / 2      # single column width  ~= 3.31in

# Font point sizes. With figures authored at their display width these are the
# on-page sizes; kept close to the ~9-10pt body text so figures read cleanly at
# column width.
PAPER_RC = {
    # ── font family: match the Times-like paper body, embed TrueType ──
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                   "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # ── sizes ──
    # These are only the on-page sizes if the figure is authored at the width it
    # is displayed at. A figure authored at TEXT_W but included at \columnwidth
    # gets scaled 0.47x by LaTeX, which is what made several figures illegible —
    # the fix is the figsize, not the point size. Keep authored width == display
    # width and these land just under body text.
    "font.size": 9.5,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    # legend: a tasteful middle-ground size with compact spacing so the box
    # stays small relative to the plot.
    "legend.fontsize": 8.0,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
    "legend.labelspacing": 0.3,
    "legend.columnspacing": 1.0,
    "legend.borderpad": 0.35,
    "legend.borderaxespad": 0.3,
    "figure.titlesize": 10.0,
    # ── consistent linework / output ──
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.0,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def apply_paper_style(**overrides):
    """Apply the canonical paper style to the global rcParams.

    Pass rcParam ``overrides`` for the rare figure that needs a tweak (e.g. a
    denser grid); prefer NOT to override the font family or sizes so figures
    stay consistent.
    """
    mpl.rcParams.update(PAPER_RC)
    if overrides:
        mpl.rcParams.update(overrides)


# Every rcParam above that is a font size, so `scaled_rc` knows what to scale.
_SIZE_KEYS = (
    "font.size", "axes.titlesize", "axes.labelsize",
    "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
    "figure.titlesize",
)


def scaled_rc(authored_w: float, display_w: float, **overrides) -> dict:
    """rcParams whose font sizes survive LaTeX's scaling of the figure.

    A figure authored ``authored_w`` inches wide but included at ``display_w``
    inches is scaled by ``display_w / authored_w`` — and so is every glyph in
    it. A figure authored at TEXT_W (7.0in) and included at ``\\columnwidth``
    (3.31in) is shrunk to 0.47x, which turns nominal 9.5pt text into 4.5pt on
    the page.

    This pre-multiplies the sizes by ``authored_w / display_w`` so that what
    lands on the page is the point size named in ``PAPER_RC``. The figure's own
    dimensions are untouched, so the paper's layout does not move.

    Use with ``plt.rc_context(scaled_rc(...))`` around the figure body, and pass
    ``display_w`` straight from the \\includegraphics width in the .tex.
    """
    k = authored_w / display_w
    rc = {key: PAPER_RC[key] * k for key in _SIZE_KEYS}
    # Linework reads as hairline-thin once shrunk, so scale it alongside the text.
    rc["axes.linewidth"] = PAPER_RC["axes.linewidth"] * k
    rc["lines.linewidth"] = PAPER_RC["lines.linewidth"] * k
    rc["lines.markersize"] = PAPER_RC["lines.markersize"] * k
    rc.update(overrides)
    return rc
