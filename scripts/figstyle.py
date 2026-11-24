"""Shared figure style for the colorimetry manuscript.

Every figure script imports this module instead of carrying its own rcParams
block. The rules behind each value are in ``figure_style.md`` at the project
root; change a value here and in that file together.

    import figstyle as fs

    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.SINGLE_COL, fs.CAL_H))
    ...
    fs.panel_label(ax, "A")
    fs.save(fig, out_dir / "figure_x", data=df)

``save`` refuses a figure whose smallest lettering, at the width it will be
placed, falls below ``MIN_FONT``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text

SINGLE_COL = 3.33
DOUBLE_COL = 7.00
MIN_DOUBLE_COL = 4.17
MAX_DEPTH = 9.167
MAX_BODY_DEPTH = 8.5

CAL_H = 3.00
SPEC_H = 2.60
ROW_H = 2.00

FONT_BASE = 9.0
FONT_SMALL = 8.0
FONT_PANEL = 10.0
MIN_FONT = 7.0
MIN_LINE = 0.5

DPI = 600
DPI_LINEART = 1200

# Paper-wide colour roles (jdc, 2026-10-06); figure_style.md "Colors" gives the rules.
# Routes own the hues: UV-Vis grey/black, RAW blues, JPEG oranges, dark to light within a
# family so variants stay apart in grayscale; 450/40 filter = dashed line or ROUTE_HATCH.
# CIELAB L* of the five Figure 5e lines: 21, 33, 45, 58, 70, so each prints a different grey.
ROUTE_COLORS = {
    "uv": ("#252525", "#6B6B6B", "#A0A0A0"),
    "raw": ("#08306B", "#2171B5", "#4292C6"),
    "jpeg": ("#8C2D04", "#D94801", "#FD8D3C"),
}
ROUTE_HATCH = "////"
# Model, fit and reference lines, annotations.
FIT_COLOR = "#2C3E50"
MODEL_COLOR = FIT_COLOR
GRAY = "#626567"
# Bayer channels: only where the channel itself is plotted against wavelength.
CHANNEL_COLORS = {"R": "#C0392B", "G": "#27AE60", "B": "#2471A3"}
# Analytes: only for molar-absorptivity spectra (Figure 3d-f). Ferricyanide is ochre so
# it does not read as the JPEG route.
FERIC_COLOR = "#9C7A00"
MB_COLOR = "#7D3C98"
DIMER_COLOR = "#1A5276"
FERROIN_COLOR = "#A62A6E"
# Replicate sets / samples (e.g. Wine 1, 2): hues no route or channel uses; the marker
# carries the distinction in grayscale.
SET_COLORS = ("#5E3C99", "#1B7F7A", "#9C7A00")
SET_MARKERS = ("o", "s", "^")
# Concentration series: one neutral ramp, darker = more concentrated (conc_colors).
CONC_CMAP = "Greys"
CONC_RANGE = (0.35, 0.95)


def conc_colors(n):
    """n greys, light (lowest concentration) to dark (highest)."""
    import matplotlib.pyplot as _plt
    import numpy as _np

    return [_plt.get_cmap(CONC_CMAP)(v) for v in _np.linspace(*CONC_RANGE, n)]


RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
    "font.size": FONT_BASE,
    "axes.labelsize": FONT_BASE,
    "axes.titlesize": FONT_BASE,
    "xtick.labelsize": FONT_BASE,
    "ytick.labelsize": FONT_BASE,
    "legend.fontsize": FONT_SMALL,
    "legend.title_fontsize": FONT_SMALL,
    "figure.titlesize": FONT_BASE,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "none",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "lines.linewidth": 0.8,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    "savefig.dpi": DPI,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}


def apply() -> None:
    plt.rcParams.update(RC)


def panel_label(ax, letter: str, x: float = -0.15, y: float = 1.05) -> Text:
    return ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=FONT_PANEL,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _saved_size(fig, tight: bool) -> tuple[float, float]:
    if not tight:
        return tuple(fig.get_size_inches())
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    pad = plt.rcParams["savefig.pad_inches"]
    return bbox.width + 2 * pad, bbox.height + 2 * pad


def _visible_texts(fig):
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            yield t


def audit(fig, placed_width: float | None = None, tight: bool = False) -> dict:
    """Printed sizes of the figure's lettering and lines at `placed_width`.

    The figure is drawn first so that tick labels exist. `scale` is how much
    the figure is shrunk (or enlarged) when placed at `placed_width` inches.
    """
    fig.canvas.draw()
    width, height = _saved_size(fig, tight)
    placed = width if placed_width is None else placed_width
    scale = placed / width

    small_text = []
    for t in _visible_texts(fig):
        printed = float(t.get_fontsize() * scale)
        if printed < MIN_FONT - 1e-6:
            small_text.append((round(printed, 2), t.get_text()[:40]))

    thin_lines = []
    for ln in fig.findobj(Line2D):
        lw = float(ln.get_linewidth())
        if ln.get_visible() and ln.get_linestyle() != "None" and lw > 0:
            if lw * scale < MIN_LINE - 1e-6:
                thin_lines.append(round(float(lw) * scale, 2))
    for coll in fig.findobj(LineCollection):
        for lw in coll.get_linewidths():
            if coll.get_visible() and 0 < float(lw) * scale < MIN_LINE - 1e-6:
                thin_lines.append(round(float(lw) * scale, 2))

    problems = []
    if placed > DOUBLE_COL + 0.01:
        problems.append(f"placed width {placed:.2f} in exceeds {DOUBLE_COL} in")
    elif SINGLE_COL + 0.01 < placed < MIN_DOUBLE_COL - 0.01:
        problems.append(
            f"placed width {placed:.2f} in falls between the "
            f"single ({SINGLE_COL}) and double "
            f"({MIN_DOUBLE_COL}-{DOUBLE_COL}) column widths"
        )
    if height * scale > MAX_BODY_DEPTH + 0.01:
        problems.append(f"printed depth {height * scale:.2f} in exceeds {MAX_BODY_DEPTH} in")
    if small_text:
        problems.append(
            f"{len(small_text)} text item(s) print below {MIN_FONT} pt, smallest {min(small_text)}"
        )
    if thin_lines:
        problems.append(
            f"{len(thin_lines)} line(s) print below {MIN_LINE} pt, thinnest {min(thin_lines)}"
        )

    return {
        "drawn_width": width,
        "drawn_height": height,
        "placed_width": placed,
        "scale": scale,
        "small_text": sorted(small_text),
        "thin_lines": thin_lines,
        "problems": problems,
    }


def save(
    fig,
    stem,
    data=None,
    placed_width: float | None = None,
    formats=("png", "svg"),
    dpi: int = DPI,
    tight: bool = False,
    strict: bool = True,
) -> dict:
    """Check the figure at its placed size, then write it and its CSV.

    `stem` is the output path without extension. `placed_width` is the width
    in inches the figure will occupy on the page; it defaults to the drawn
    width, which is the only case the style guide allows. With `strict`,
    any problem raises before anything is written.
    """
    stem = Path(stem)
    report = audit(fig, placed_width, tight)
    if report["problems"]:
        msg = f"{stem.name}: " + "; ".join(report["problems"])
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)

    stem.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"} if tight else {}
    for ext in formats:
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=dpi, format=ext, **kwargs)
    if data is not None:
        data.to_csv(stem.with_suffix(".csv"), index=False)
    else:
        warnings.warn(f"{stem.name}: no companion CSV written", stacklevel=2)
    return report
