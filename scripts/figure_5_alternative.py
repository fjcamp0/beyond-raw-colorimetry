"""Alternative rendering of main-text Figure 6 (disk folder ``figure_5``).

Purpose
-------
Figure 6 shows the Beer-Lambert-Bayer integrand collapse for ferricyanide
under white LED illumination (panel A: B channel, panel B: G channel), but
never plots the absorbance curve itself -- a reader has to already know
Eq. 22 to understand *why* the region above 480 nm is transparent. This
script builds an alternative panel A that adds two things Supplementary
Figure S9a already does well: a printed f_tail percentage, and a light
absorbance-shape overlay showing where the analyte actually absorbs. Panel B
(G-channel inertness) is reproduced unchanged for context.

See the editorial discussion of Figure 6 vs S9a/b (2026-07-09) for the
reasoning behind this compromise: keep Figure 6's multi-concentration
collapse and G-channel companion (which S9a/b lacks), while borrowing S9a's
mechanism-clarity annotation (which Figure 6 lacks).

This is a comparison variant, not a replacement. It does not touch
``figure_5/`` or any file the manuscript currently cites.

2026-07-29 update (Q5, correction_plan_javier-steph_2026-07-29.md, jdc's
ruling): added methylene blue panels (c) R channel and (d) G channel,
reusing figure_5.py's new figure_5c_LED.csv/figure_5d_LED.csv as-is. Panel
letters changed from bare "A"/"B" to "(a)"/"(b)"/"(c)"/"(d)" to match the
lowercase-in-brackets convention used everywhere else in the manuscript
(this file was the source of the capital-letter inconsistency Steph flagged).
Panel (c)/(d) have no absorbance overlay or f_tail annotation, unlike (a) --
MB has no single sharp lambda_max and no vetted MB f_tail number exists at
the 426 nm boundary definition, so adding one here risked the
same numeric-drift bug class already fixed twice elsewhere in this round
(F1, and the 2026-07-28 MB epsilon reversal). Layout changed from a 1x2 row
to a 2x2 grid.

Inputs (read-only; see figure_5_alternative_config.yaml to change any of
these without editing this file)
------
- Figure 6's own companion CSVs (``figure_5a_LED.csv``, ``figure_5b_LED.csv``)
  -- reused as-is, so the collapse curves are guaranteed identical to the
  current main-text figure.
- Supplementary Figure S9's companion CSV (``figure_S7_ftail_data.csv``) for
  the absorbance-shape overlay -- reused, not recomputed, to avoid a second
  numerically-drifted estimate of a quantity already vetted elsewhere.
- f_tail is cited from the YAML config, not recomputed here. It is 62% at the
  473 nm boundary (2026-09-21); it read 56.1% at the superseded 480 nm one.

Outputs
-------
``<output_dir>/<output_basename>_assembled_LED.{png,svg,tiff}`` (the figure)
``<output_dir>/<output_basename>_assembled_LED.csv`` (companion data, per
figure_style.md's "every figure must have a companion CSV" rule)

Run from the project root::

    python scripts/figure_5_alternative.py
    python scripts/figure_5_alternative.py --config path/to/other_config.yaml
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import sys  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
from forward_model import split_at_cutoff  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "figure_5_alternative_config.yaml"


@dataclass(frozen=True)
class PanelConfig:
    """Typed view over the parts of the YAML config each function needs.

    Keeping this as a thin dataclass (rather than passing the raw dict
    everywhere) means a typo in the YAML key name fails at load time, in one
    place, instead of silently producing a KeyError deep inside a plotting
    function.
    """

    integrand_b_csv: Path
    integrand_g_csv: Path
    integrand_mb_r_csv: Path
    integrand_mb_g_csv: Path
    absorbance_csv: Path
    absorbance_column: str
    output_dir: Path
    output_basename: str
    conc_columns: list[str]
    blank_column: str
    mb_conc_columns: list[str]
    mb_blank_column: str
    blues_cmap_range: tuple[float, float]
    blank_color: str
    lambda_max_nm: float
    lambda_max_color: str
    tail_boundary_nm: float
    tail_boundary_color: str
    tail_shade_color: str
    tail_shade_alpha: float
    mb_tail_boundary_nm: float
    mb_tail_boundary_color: str
    mb_tail_shade_color: str
    mb_tail_shade_alpha: float
    f_tail_percent: float
    f_tail_source: str
    f_tail_position_nm: float
    f_tail_position_frac_y: float
    overlay_color: str
    overlay_linestyle: str
    overlay_linewidth: float
    overlay_alpha: float
    overlay_ylabel: str
    wavelength_range_nm: tuple[float, float]
    acs_panel_width_in: float
    acs_panel_height_in: float
    assembled_width_in: float
    dpi: int
    font_family: list[str] = field(default_factory=list)


def load_config(config_path: Path) -> PanelConfig:
    """Read the YAML config and resolve every relative path against PROJ_ROOT."""
    with open(config_path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    paths = raw["paths"]
    concs = raw["concentrations"]
    mb_concs = raw["mb_concentrations"]
    refs = raw["reference_lines"]
    mb_refs = raw["mb_reference_lines"]
    ann = raw["annotation"]
    overlay = raw["absorbance_overlay"]
    style = raw["style"]

    return PanelConfig(
        integrand_b_csv=PROJ_ROOT / paths["integrand_b_channel_csv"],
        integrand_g_csv=PROJ_ROOT / paths["integrand_g_channel_csv"],
        integrand_mb_r_csv=PROJ_ROOT / paths["integrand_mb_r_channel_csv"],
        integrand_mb_g_csv=PROJ_ROOT / paths["integrand_mb_g_channel_csv"],
        absorbance_csv=PROJ_ROOT / paths["absorbance_reference_csv"],
        absorbance_column=paths["absorbance_reference_column"],
        output_dir=PROJ_ROOT / paths["output_dir"],
        output_basename=paths["output_basename"],
        conc_columns=list(concs["columns"]),
        blank_column=concs["blank_column"],
        mb_conc_columns=list(mb_concs["columns"]),
        mb_blank_column=mb_concs["blank_column"],
        blues_cmap_range=tuple(concs["blues_cmap_range"]),
        blank_color=concs["blank_color"],
        lambda_max_nm=float(refs["lambda_max_nm"]),
        lambda_max_color=refs["lambda_max_color"],
        tail_boundary_nm=float(refs["tail_boundary_nm"]),
        tail_boundary_color=refs["tail_boundary_color"],
        tail_shade_color=refs["tail_shade_color"],
        tail_shade_alpha=float(refs["tail_shade_alpha"]),
        mb_tail_boundary_nm=float(mb_refs["tail_boundary_nm"]),
        mb_tail_boundary_color=mb_refs["tail_boundary_color"],
        mb_tail_shade_color=mb_refs["tail_shade_color"],
        mb_tail_shade_alpha=float(mb_refs["tail_shade_alpha"]),
        f_tail_percent=float(ann["f_tail_percent"]),
        f_tail_source=ann["f_tail_source"],
        f_tail_position_nm=float(ann["f_tail_position_nm"]),
        f_tail_position_frac_y=float(ann["f_tail_position_frac_y"]),
        overlay_color=overlay["color"],
        overlay_linestyle=overlay["linestyle"],
        overlay_linewidth=float(overlay["linewidth"]),
        overlay_alpha=float(overlay["alpha"]),
        overlay_ylabel=overlay["ylabel"],
        wavelength_range_nm=tuple(raw["wavelength_range_nm"]),
        acs_panel_width_in=float(style["acs_panel_width_in"]),
        acs_panel_height_in=float(style["acs_panel_height_in"]),
        assembled_width_in=float(style["assembled_width_in"]),
        dpi=int(style["dpi"]),
        font_family=list(style["font_family"]),
    )


def apply_acs_rcparams(cfg: PanelConfig) -> None:
    """Apply the mandatory rcParams block from figure_style.md."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": cfg.font_family,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "patch.linewidth": 0.5,
            "figure.dpi": 150,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def load_integrand_csv(path: Path, wl_range: tuple[float, float]) -> pd.DataFrame:
    """Load a figure_5-style integrand CSV, trimmed to the plotting range."""
    df = pd.read_csv(path)
    lo, hi = wl_range
    mask = (df["wavelength_nm"] >= lo) & (df["wavelength_nm"] <= hi)
    return df.loc[mask].reset_index(drop=True)


def load_absorbance_shape(path: Path, column: str, wl_range: tuple[float, float]) -> pd.DataFrame:
    """Load the S9 absorbance column and renormalise its shape to peak = 1.

    A(lambda) = eps(lambda) * c * l is proportional to c at fixed path
    length l, so renormalising removes the dependence on which
    concentration the source file happened to compute it at (8000 uM here,
    vs. up to 2000 uM in the collapse panel) -- only the spectral *shape*
    is meaningful for this overlay.
    """
    df = pd.read_csv(path, usecols=["wavelength_nm", column])
    lo, hi = wl_range
    mask = (df["wavelength_nm"] >= lo) & (df["wavelength_nm"] <= hi)
    df = df.loc[mask].reset_index(drop=True)
    peak = df[column].max()
    df["shape"] = df[column] / peak
    return df[["wavelength_nm", "shape"]]


def conc_colors(n_nonzero: int, cmap_range: tuple[float, float]) -> list[str]:
    """Return n_nonzero greys, light (lowest c) to dark (highest c)."""
    return [matplotlib.colors.to_hex(c) for c in fs.conc_colors(n_nonzero)]


def conc_label(column_name: str) -> str:
    """Format a concentration column name ('62.5uM', 'blank', ...) for the legend."""
    if column_name.lower() == "blank":
        return "blank (c = 0)"
    match = re.match(r"([\d.]+)\s*uM", column_name)
    if not match:
        return column_name
    c_um = float(match.group(1))
    if c_um >= 1000:
        return f"{c_um / 1000:g} mM"
    return f"{c_um:g} µM"


def plot_collapse_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    cfg: PanelConfig,
    channel_name: str,
    panel_letter: str,
    conc_columns: list[str],
    blank_column: str,
    tail_boundary_nm: float,
    tail_boundary_color: str,
    tail_shade_color: str,
    tail_shade_alpha: float,
    lambda_max_nm: float | None,
    lambda_max_color: str,
    tail_above: bool,
    absorbance_overlay: pd.DataFrame | None,
    legend_loc: str | None = "upper right",
    ftail_xy: tuple[float, float] = (0.68, 0.15),
) -> None:
    """Draw one integrand-collapse panel; optionally overlay the absorbance shape.

    Reproduces the shading/reference-line/colour conventions of
    scripts/figure_5.py's ``plot_panel`` exactly, so the two figures are
    visually comparable side by side. The absorbance overlay and f_tail
    annotation are the only additions.

    lambda_max_nm=None omits the lambda_max marker (methylene blue has no
    single sharp peak); tail_above=False shades the tail below
    tail_boundary_nm instead of above it (MB's R-channel tail geometry is the
    opposite of ferricyanide's B-channel geometry).

    Every panel gets an f_tail % annotation. Panel (a) keeps its existing,
    separately-cited value (drawn by ``_add_absorbance_overlay``, sourced
    from Eq. 22 in the main text, not recomputed here). The other panels
    compute f_tail geometrically from the same ``blank`` curve already drawn
    on the axes (fraction of its area inside ``tail_mask``) -- this cannot
    drift from the shaded tail region shown on the same plot, since it is
    integrated from that exact curve, not a second independent estimate.
    """
    wl = data["wavelength_nm"].to_numpy()
    nonzero_cols = [c for c in conc_columns if c != blank_column]
    colors = conc_colors(len(nonzero_cols), cfg.blues_cmap_range)

    blank = data[blank_column].to_numpy()
    tail_mask = (wl > tail_boundary_nm) if tail_above else (wl < tail_boundary_nm)
    tail_dir = ">" if tail_above else "<"
    ax.fill_between(
        wl[tail_mask],
        0,
        blank[tail_mask],
        color=fs.CHANNEL_COLORS[channel_name],  # tint of the plotted channel
        alpha=0.18,
        zorder=0,
        label=f"Transparent tail ({tail_dir} {tail_boundary_nm:.0f} nm)",
    )

    ax.plot(
        wl,
        blank,
        color=fs.GRAY,
        lw=1.2,
        ls="--",
        zorder=3,
        label=conc_label(blank_column),
    )
    for column, color in zip(nonzero_cols, colors):
        ax.plot(
            wl,
            data[column].to_numpy(),
            color=color,
            lw=1.0,
            ls="-",
            zorder=2,
            label=conc_label(column),
        )

    if lambda_max_nm is not None:
        ax.axvline(
            lambda_max_nm,
            color=fs.FIT_COLOR,
            lw=0.9,
            ls="--",
            zorder=4,
            alpha=0.85,
            label=f"{lambda_max_nm:.0f} nm (λ$_{{max}}$)",
        )
    ax.axvline(
        tail_boundary_nm,
        color=fs.FIT_COLOR,
        lw=0.9,
        ls=":",
        zorder=4,
        alpha=0.85,
        label=f"{tail_boundary_nm:.0f} nm (tail boundary)",
    )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"φ·S$_{{{channel_name}}}$·10$^{{−ε c l}}$  [normalised to blank]")
    ax.set_xlim(*cfg.wavelength_range_nm)
    ax.set_ylim(bottom=0)

    if absorbance_overlay is not None:
        _add_absorbance_overlay(ax, absorbance_overlay, cfg)
    else:
        f_tail = compute_f_tail(wl, blank, tail_mask)
        _add_ftail_text(ax, f_tail, ftail_xy)

    ax.text(
        -0.15,
        1.05,
        panel_letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
    )
    if legend_loc is None:
        return
    ax.legend(
        fontsize=7,
        framealpha=0.0,
        edgecolor="none",
        loc=legend_loc,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.3,
    )


def compute_f_tail(wl: np.ndarray, blank: np.ndarray, tail_mask: np.ndarray) -> float:
    """Fraction of the blank integrand's area lying in the shaded tail region.

    ``blank`` is the panel's own blank-normalised phi*T curve (identical to
    the CSV column figure_5.py wrote out) -- integrating over ``tail_mask``
    reproduces exactly the light-blue "Transparent tail" area already filled
    on the same axes, so this is a re-integration of the plotted curve, not
    an independent phi(lambda)/T_ch(lambda) recomputation that could drift
    from the source-of-truth number for panel (a) (which figure_5.py's own
    tail-shading logic and this function both derive from the same
    phi*T-vs-boundary definition).
    """
    tail_area, _ = split_at_cutoff(blank, tail_mask, wl)
    total_area = float(np.trapezoid(blank, wl))
    return tail_area / total_area


def _add_ftail_text(ax: plt.Axes, f_tail: float, xy: tuple[float, float]) -> None:
    """Axes-fraction f_tail % label, matching panel (a)'s bold-text style."""
    ax.text(
        xy[0],
        xy[1],
        f"f$_{{tail}}$ = {f_tail * 100:.2g}%",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        color="#1a1a1a",
        ha="left",
        va="center",
    )


def _add_absorbance_overlay(ax: plt.Axes, overlay: pd.DataFrame, cfg: PanelConfig) -> None:
    """Add the twin-axis absorbance-shape curve and the f_tail % annotation."""
    twin = ax.twinx()
    twin.plot(
        overlay["wavelength_nm"],
        overlay["shape"],
        color=fs.GRAY,
        ls="-",
        lw=cfg.overlay_linewidth,
        alpha=cfg.overlay_alpha,
        zorder=5,
        label="Absorbance shape (ε(λ), normalised)",
    )
    twin.set_ylabel(cfg.overlay_ylabel, color=fs.GRAY)
    twin.tick_params(axis="y", colors=fs.GRAY)
    twin.set_ylim(0, 1.15)
    twin.spines["right"].set_color(fs.GRAY)

    ax.text(
        cfg.f_tail_position_nm,
        cfg.f_tail_position_frac_y * ax.get_ylim()[1],
        f"f$_{{tail}}$ = {cfg.f_tail_percent:.2g}%",
        fontsize=8,
        fontweight="bold",
        color="#1a1a1a",
        ha="left",
        va="center",
    )


def build_figure(cfg: PanelConfig) -> tuple[plt.Figure, pd.DataFrame]:
    """Assemble the 4-panel figure (ferricyanide a/b, methylene blue c/d) and
    the companion data table."""
    apply_acs_rcparams(cfg)

    b_data = load_integrand_csv(cfg.integrand_b_csv, cfg.wavelength_range_nm)
    g_data = load_integrand_csv(cfg.integrand_g_csv, cfg.wavelength_range_nm)
    mb_r_data = load_integrand_csv(cfg.integrand_mb_r_csv, cfg.wavelength_range_nm)
    mb_g_data = load_integrand_csv(cfg.integrand_mb_g_csv, cfg.wavelength_range_nm)
    absorbance = load_absorbance_shape(
        cfg.absorbance_csv, cfg.absorbance_column, cfg.wavelength_range_nm
    )

    fig, ((ax_b, ax_g), (ax_mb_r, ax_mb_g)) = plt.subplots(
        2, 2, figsize=(cfg.assembled_width_in, 2 * cfg.acs_panel_height_in)
    )
    plot_collapse_panel(
        ax_b,
        b_data,
        cfg,
        "B",
        "(a)",
        conc_columns=cfg.conc_columns,
        blank_column=cfg.blank_column,
        tail_boundary_nm=cfg.tail_boundary_nm,
        tail_boundary_color=cfg.tail_boundary_color,
        tail_shade_color=cfg.tail_shade_color,
        tail_shade_alpha=cfg.tail_shade_alpha,
        lambda_max_nm=cfg.lambda_max_nm,
        lambda_max_color=cfg.lambda_max_color,
        tail_above=True,
        absorbance_overlay=absorbance,
    )
    plot_collapse_panel(
        ax_g,
        g_data,
        cfg,
        "G",
        "(b)",
        conc_columns=cfg.conc_columns,
        blank_column=cfg.blank_column,
        tail_boundary_nm=cfg.tail_boundary_nm,
        tail_boundary_color=cfg.tail_boundary_color,
        tail_shade_color=cfg.tail_shade_color,
        tail_shade_alpha=cfg.tail_shade_alpha,
        lambda_max_nm=cfg.lambda_max_nm,
        lambda_max_color=cfg.lambda_max_color,
        tail_above=True,
        absorbance_overlay=None,
        legend_loc=None,  # same key as (a); dropped at the larger font (2026-10-05)
        ftail_xy=(0.68, 0.15),
    )
    plot_collapse_panel(
        ax_mb_r,
        mb_r_data,
        cfg,
        "R",
        "(c)",
        conc_columns=cfg.mb_conc_columns,
        blank_column=cfg.mb_blank_column,
        tail_boundary_nm=cfg.mb_tail_boundary_nm,
        tail_boundary_color=cfg.mb_tail_boundary_color,
        tail_shade_color=cfg.mb_tail_shade_color,
        tail_shade_alpha=cfg.mb_tail_shade_alpha,
        lambda_max_nm=None,
        lambda_max_color=cfg.lambda_max_color,
        tail_above=False,
        absorbance_overlay=None,
        legend_loc="center left",
        ftail_xy=(0.06, 0.92),
    )
    plot_collapse_panel(
        ax_mb_g,
        mb_g_data,
        cfg,
        "G",
        "(d)",
        conc_columns=cfg.mb_conc_columns,
        blank_column=cfg.mb_blank_column,
        tail_boundary_nm=cfg.mb_tail_boundary_nm,
        tail_boundary_color=cfg.mb_tail_boundary_color,
        tail_shade_color=cfg.mb_tail_shade_color,
        tail_shade_alpha=cfg.mb_tail_shade_alpha,
        lambda_max_nm=None,
        lambda_max_color=cfg.lambda_max_color,
        tail_above=False,
        absorbance_overlay=None,
        legend_loc=None,  # same key as (c)
        ftail_xy=(0.06, 0.15),
    )
    fig.tight_layout(pad=0.5)

    mb_r_data = mb_r_data.rename(
        columns={c: f"{c}_MB_R_channel" for c in mb_r_data.columns if c != "wavelength_nm"}
    )
    mb_g_data = mb_g_data.rename(
        columns={c: f"{c}_MB_G_channel" for c in mb_g_data.columns if c != "wavelength_nm"}
    )
    companion = (
        b_data.merge(g_data, on="wavelength_nm", suffixes=("_B_channel", "_G_channel"))
        .merge(mb_r_data, on="wavelength_nm")
        .merge(mb_g_data, on="wavelength_nm")
        .merge(absorbance, on="wavelength_nm", how="left")
    )
    companion = companion.rename(columns={"shape": "absorbance_shape_normalised"})

    return fig, companion


def save_outputs(fig: plt.Figure, companion: pd.DataFrame, cfg: PanelConfig) -> None:
    """Save PNG/SVG/TIFF at 600 dpi plus the companion CSV, per figure_style.md."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.output_basename}_assembled_LED"
    for ext in ("png", "svg", "tiff"):
        fig.savefig(
            cfg.output_dir / f"{stem}.{ext}",
            dpi=cfg.dpi,
            format=ext,
        )
    companion.to_csv(cfg.output_dir / f"{stem}.csv", index=False)


def main() -> None:
    """CLI entry point: load config, build the figure, save all outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config (default: figure_5_alternative_config.yaml "
        "next to this script).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    fig, companion = build_figure(cfg)
    save_outputs(fig, companion, cfg)
    plt.close(fig)
    print(f"Wrote {cfg.output_dir / (cfg.output_basename + '_assembled_LED.png')}")


if __name__ == "__main__":
    main()
