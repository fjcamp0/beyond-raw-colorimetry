"""Figure 6 — Forward model vs empirical calibration (LED and daylight).

Compares Beer-Lambert-Bayer (BLB) forward-model predictions against
empirical RAW calibration data for:
  Panel A – Ferricyanide B channel (forward model undershoots -> A_common signature)
  Panel B – Ferricyanide B/G (channel-common component cancels -> close agreement)
  Panel C – Methylene blue R channel (contrast case, broad epsilon)

Produces individual panel files (PNG+TIFF+SVG+CSV) and assembled 3-panel
figures (PNG+TIFF+SVG) for both LED and diffuse daylight illuminants.

ACS format: 3.25 x 3.0 in, Arial 8 pt, 600 dpi, log x-axis, inside ticks
on all four sides, no title on panels.

Run from project root: python scripts/figure_6.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import figstyle as fs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # colorimetry_Project root

DATA = PROJECT_ROOT / "Data" / "analysis"
_FC_DAY = DATA / "ferricyanide" / "daylight_diffuser" / "image_analysis"
_MB_DAY = DATA / "methylene_blue" / "daylight_diffuser" / "image_analysis"

PATHS = {
    "LED": {
        "ferricyanide": {
            "calib": DATA / "ferricyanide" / "beer_lambert" / "calibration_table.csv",
            "model": DATA / "ferricyanide" / "forward_model" / "forward_model_table.csv",
        },
        "methylene_blue": {
            "calib": DATA / "methylene_blue" / "beer_lambert" / "calibration_table.csv",
            "model": DATA / "methylene_blue" / "forward_model" / "forward_model_table.csv",
        },
    },
    "daylight": {
        "ferricyanide": {
            "calib": _FC_DAY / "s02_calibrate" / "calibration_table.csv",
            "model": _FC_DAY / "forward_model_daylight_diffuser" / "forward_model_table.csv",
        },
        "methylene_blue": {
            "calib": _MB_DAY / "s02_calibrate" / "calibration_table.csv",
            "model": _MB_DAY / "forward_model_daylight_diffuser" / "forward_model_table.csv",
        },
    },
}

OUT_ROOT = PROJECT_ROOT / "Data" / "paper_figures" / "figure_6"
# assemble_manuscript.py reads figures only from Manuscript/_drafts/figures/,
# never from Data/paper_figures/ — mirror every render there.
DRAFTS_OUT = PROJECT_ROOT / "Manuscript" / "_drafts" / "figures" / "figure_6"

# ---------------------------------------------------------------------------
# ACS style constants
# ---------------------------------------------------------------------------

PANEL_W = 3.25  # inches
PANEL_H = 3.0  # inches
DPI = 600
FONTSIZE = 9
FONT_FAMILY = "Arial"
LW_MODEL = 1.2  # model line width
MS = 4  # marker size for empirical points
CAPSIZE = 2  # error bar cap size

# Colour palette (colour-blind-friendly)
COLOR_MODEL = fs.MODEL_COLOR  # forward model line
COLOR_EMPIRICAL = fs.ROUTE_COLORS["raw"][0]  # RAW camera data points

# Low-c zoom range (µM) for the bottom row / second panel: BL review flagged the
# near-zero/flat low-concentration region as "not very easy to see" against the
# full log-x range in all three panels. FC cap matches the BL-slope fit range
# used elsewhere for these same channels (figure_7.py); MB cap is tighter since
# the MB R curve visibly departs from flat well before 50 µM.
ZOOM_MAX_FC = 1000.0  # µM
ZOOM_MAX_MB = 25.0  # µM


def _apply_acs_style(ax: plt.Axes) -> None:
    """Apply ACS journal formatting to a single Axes."""
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=FONTSIZE,
        width=0.8,
        length=3,
    )
    ax.tick_params(which="minor", length=1.5)
    ax.set_xscale("log")


def _make_fig_ax() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Create a two-row ACS figure: full-range panel on top, low-c zoom below
    (BL review: the near-zero/flat low-c region is "not very easy to see"
    against the full log-x range), with room left at the bottom for a
    shared legend.
    """
    fig, (ax_top, ax_zoom) = plt.subplots(2, 1, figsize=(PANEL_W, PANEL_H * 2))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.10, top=0.97, hspace=0.32)
    _apply_acs_style(ax_top)
    _apply_acs_style(ax_zoom)
    return fig, ax_top, ax_zoom


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_calib(path: Path, channel: str) -> pd.DataFrame:
    """Return rows for *channel* from a calibration_table.csv.

    Columns: concentration_um, absorbance_mean, absorbance_std
    """
    df = pd.read_csv(path)
    sub = df[df["channel"] == channel].copy()
    sub = sub.sort_values("concentration_um").reset_index(drop=True)
    return sub[["concentration_um", "absorbance_mean", "absorbance_std"]]


def load_model(path: Path) -> pd.DataFrame:
    """Return forward_model_table.csv with relevant columns."""
    df = pd.read_csv(path)
    df = df.sort_values("concentration_um").reset_index(drop=True)
    return df


def _interpolate_model_at(model_df: pd.DataFrame, col: str, concs: np.ndarray) -> np.ndarray:
    """Linearly interpolate model values at given concentrations (log-space)."""
    return np.interp(
        np.log10(concs),
        np.log10(model_df["concentration_um"].values),
        model_df[col].values,
    )


# ---------------------------------------------------------------------------
# Panel-drawing function
# ---------------------------------------------------------------------------


def draw_panel(
    ax: plt.Axes,
    model_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    model_col: str,
    ylabel: str,
    xlabel: str = "Concentration (µM)",
    annotation: str | None = None,
    c_max: float | None = None,
    panel_label: str | None = None,
    show_legend: bool = False,
) -> None:
    """
    Draw model line + empirical data with error bars on *ax*, restricted to
    concentration <= c_max (None = full range). Used for both the full-range
    panel and the low-c zoom panel below it — same drawing, different cutoff.
    """
    mask_model = (model_df["concentration_um"] > 0) & model_df[model_col].notna()
    if c_max is not None:
        mask_model &= model_df["concentration_um"] <= c_max
    mx = model_df.loc[mask_model, "concentration_um"].values
    my = model_df.loc[mask_model, model_col].values
    ax.plot(mx, my, color=COLOR_MODEL, lw=LW_MODEL, zorder=3, label="BLB model")

    # --- Empirical points + error bars ---
    mask_calib = calib_df["concentration_um"] > 0
    if c_max is not None:
        mask_calib &= calib_df["concentration_um"] <= c_max
    ex = calib_df.loc[mask_calib, "concentration_um"].values
    ey = calib_df.loc[mask_calib, "absorbance_mean"].values
    esd = calib_df.loc[mask_calib, "absorbance_std"].values

    ax.errorbar(
        ex,
        ey,
        yerr=esd,
        fmt="o",
        color=COLOR_EMPIRICAL,
        markersize=MS,
        linewidth=0.8,
        capsize=CAPSIZE,
        zorder=4,
        label="Empirical (mean ± SD)",
    )

    # --- Axes labels ---
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)

    if panel_label:
        ax.text(
            0.03,
            0.97,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONTSIZE,
            fontweight="bold",
        )

    # --- Optional annotation --- (mid-left; the data curve is flat near the
    # low-c end of the log x-axis, so bottom-left would sit on top of it)
    if annotation:
        ax.text(
            0.04,
            0.55,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONTSIZE - 0.5,
            color="#444444",
            wrap=True,
        )

    if show_legend:
        ax.legend(
            fontsize=FONTSIZE + 1, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.90)
        )


# ---------------------------------------------------------------------------
# CSV export helper
# ---------------------------------------------------------------------------


def export_csv(
    out_path: Path,
    model_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    model_col: str,
    col_model_name: str,
    col_mean_name: str,
    col_std_name: str,
) -> None:
    """Export plotted data (model interpolated at empirical concentrations) to CSV."""
    mask = calib_df["concentration_um"] > 0
    sub = calib_df.loc[mask].copy()
    concs = sub["concentration_um"].values

    model_interp = _interpolate_model_at(model_df, model_col, concs)

    out = pd.DataFrame(
        {
            "concentration_um": concs,
            col_model_name: model_interp,
            col_mean_name: sub["absorbance_mean"].values,
            col_std_name: sub["absorbance_std"].values,
        }
    )
    out.to_csv(out_path, index=False, float_format="%.8g")
    print(f"  [CSV] {out_path}")


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_panel(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save as PNG, TIFF and SVG at 600 dpi."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "tiff", "svg"):
        fp = out_dir / f"{stem}.{ext}"
        fig.savefig(fp, dpi=DPI, bbox_inches="tight")
        print(f"  [saved] {fp}")


# ---------------------------------------------------------------------------
# Per-illuminant figure generation
# ---------------------------------------------------------------------------


def make_illuminant_figures(illuminant: str) -> None:
    """Generate all panels and assembled figure for one illuminant."""

    print(f"\n=== {illuminant.upper()} ===")
    out_dir = OUT_ROOT / illuminant

    ill_paths = PATHS[illuminant]

    fc_calib = load_calib(ill_paths["ferricyanide"]["calib"], "B")
    fc_bdivg_c = load_calib(ill_paths["ferricyanide"]["calib"], "BdivG")
    mb_calib = load_calib(ill_paths["methylene_blue"]["calib"], "R")

    fc_model = load_model(ill_paths["ferricyanide"]["model"])
    mb_model = load_model(ill_paths["methylene_blue"]["model"])

    il_label = illuminant.upper().replace("_", " ")
    # human-readable label for axis
    if illuminant == "LED":
        il_str = "LED"
    else:
        il_str = "Daylight"

    # ----------------------------------------------------------------
    # Panel A — Ferricyanide B channel
    # ----------------------------------------------------------------
    print(f"\nPanel A - Ferricyanide B ({il_str})")
    fig_a, ax_a_top, ax_a_zoom = _make_fig_ax()
    draw_panel(
        ax_a_top,
        fc_model,
        fc_calib,
        model_col="A_B",
        ylabel=r"$A_B$",
        annotation=("ferricyanide B > model:\nchannel-common\ncomponent"),
    )
    draw_panel(
        ax_a_zoom,
        fc_model,
        fc_calib,
        model_col="A_B",
        ylabel=r"$A_B$",
        c_max=ZOOM_MAX_FC,
    )
    fig_a.legend(
        *ax_a_top.get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=FONTSIZE + 1,
    )
    export_csv(
        out_dir / f"figure_6a_{illuminant}.csv",
        fc_model,
        fc_calib,
        model_col="A_B",
        col_model_name="A_B_model",
        col_mean_name="A_B_mean",
        col_std_name="A_B_std",
    )
    save_panel(fig_a, out_dir, f"figure_6a_{illuminant}")
    plt.close(fig_a)

    # ----------------------------------------------------------------
    # Panel B — Ferricyanide B/G channel
    # ----------------------------------------------------------------
    print(f"\nPanel B - Ferricyanide B/G ({il_str})")
    fig_b, ax_b_top, ax_b_zoom = _make_fig_ax()
    draw_panel(
        ax_b_top,
        fc_model,
        fc_bdivg_c,
        model_col="A_BdivG",
        ylabel=r"$A_{B/G}$",
        annotation=("ferricyanide B/G cancels\nchannel-common\ncomponent"),
    )
    draw_panel(
        ax_b_zoom,
        fc_model,
        fc_bdivg_c,
        model_col="A_BdivG",
        ylabel=r"$A_{B/G}$",
        c_max=ZOOM_MAX_FC,
    )
    fig_b.legend(
        *ax_b_top.get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=FONTSIZE + 1,
    )
    export_csv(
        out_dir / f"figure_6b_{illuminant}.csv",
        fc_model,
        fc_bdivg_c,
        model_col="A_BdivG",
        col_model_name="A_BdivG_model",
        col_mean_name="A_BdivG_mean",
        col_std_name="A_BdivG_std",
    )
    save_panel(fig_b, out_dir, f"figure_6b_{illuminant}")
    plt.close(fig_b)

    # ----------------------------------------------------------------
    # Panel C — Methylene blue R channel
    # ----------------------------------------------------------------
    print(f"\nPanel C - Methylene blue R ({il_str})")
    fig_c, ax_c_top, ax_c_zoom = _make_fig_ax()
    draw_panel(
        ax_c_top,
        mb_model,
        mb_calib,
        model_col="A_R",
        ylabel=r"$A_R$",
        annotation=("MB R channel;\nOhlab 20-phone\navg (proxy)"),
    )
    draw_panel(
        ax_c_zoom,
        mb_model,
        mb_calib,
        model_col="A_R",
        ylabel=r"$A_R$",
        c_max=ZOOM_MAX_MB,
    )
    fig_c.legend(
        *ax_c_top.get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=FONTSIZE + 1,
    )
    export_csv(
        out_dir / f"figure_6c_{illuminant}.csv",
        mb_model,
        mb_calib,
        model_col="A_R",
        col_model_name="A_R_model",
        col_mean_name="A_R_mean",
        col_std_name="A_R_std",
    )
    save_panel(fig_c, out_dir, f"figure_6c_{illuminant}")
    plt.close(fig_c)

    # ----------------------------------------------------------------
    # Assembled 6-panel grid: full-range row (a-c) over low-c zoom row (d-f)
    # ----------------------------------------------------------------
    print(f"\nAssembled figure ({il_str})")
    fig_all, axes = plt.subplots(
        2,
        3,
        figsize=(7.0, PANEL_H * 2 + 0.3),
    )
    fig_all.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.97, wspace=0.38, hspace=0.30)

    for ax in axes.flat:
        _apply_acs_style(ax)

    cols = [
        (
            fc_model,
            fc_calib,
            "A_B",
            r"$A_B$",
            "ferricyanide B > model:\nchannel-common\ncomponent",
            None,
        ),
        (
            fc_model,
            fc_bdivg_c,
            "A_BdivG",
            r"$A_{B/G}$",
            "ferricyanide B/G cancels\nchannel-common\ncomponent",
            None,
        ),
        (mb_model, mb_calib, "A_R", r"$A_R$", "MB R channel;\nOhlab 20-phone\navg (proxy)", None),
    ]
    zoom_caps = [ZOOM_MAX_FC, ZOOM_MAX_FC, ZOOM_MAX_MB]
    top_labels = ["(a)", "(b)", "(c)"]
    bot_labels = ["(d)", "(e)", "(f)"]

    for col_idx, (model_df, calib_df, model_col, ylabel, annotation, _) in enumerate(cols):
        _draw_assembled_panel(
            axes[0, col_idx],
            model_df,
            calib_df,
            model_col=model_col,
            ylabel=ylabel,
            annotation=annotation,
            panel_label=top_labels[col_idx],
            show_legend=False,
        )
        _draw_assembled_panel(
            axes[1, col_idx],
            model_df,
            calib_df,
            model_col=model_col,
            ylabel=ylabel,
            annotation=None,
            panel_label=bot_labels[col_idx],
            show_legend=False,
            c_max=zoom_caps[col_idx],
        )

    for ax in axes.flat:
        ax.set_xlabel("Concentration (µM)", fontsize=FONTSIZE)

    fig_all.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=FONTSIZE + 1,
        bbox_to_anchor=(0.5, -0.02),
    )

    save_panel(fig_all, out_dir, f"figure_6_assembled_{illuminant}")
    plt.close(fig_all)

    # ----------------------------------------------------------------
    # Copy everything to Manuscript/_drafts/figures/figure_6/<illuminant>/
    # ----------------------------------------------------------------
    dst_dir = DRAFTS_OUT / illuminant
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, dst_dir / f.name)


def _draw_assembled_panel(
    ax: plt.Axes,
    model_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    model_col: str,
    ylabel: str,
    annotation: str | None,
    panel_label: str,
    show_legend: bool = True,
    c_max: float | None = None,
) -> None:
    """
    Draw one panel into the assembled figure axes, restricted to
    concentration <= c_max (None = full range). Used for both the full-range
    row and the low-c zoom row below it.
    """
    mask_model = (model_df["concentration_um"] > 0) & model_df[model_col].notna()
    if c_max is not None:
        mask_model &= model_df["concentration_um"] <= c_max
    mx = model_df.loc[mask_model, "concentration_um"].values
    my = model_df.loc[mask_model, model_col].values
    ax.plot(mx, my, color=COLOR_MODEL, lw=LW_MODEL, zorder=3, label="BLB model")

    mask_calib = calib_df["concentration_um"] > 0
    if c_max is not None:
        mask_calib &= calib_df["concentration_um"] <= c_max
    ex = calib_df.loc[mask_calib, "concentration_um"].values
    ey = calib_df.loc[mask_calib, "absorbance_mean"].values
    esd = calib_df.loc[mask_calib, "absorbance_std"].values

    ax.errorbar(
        ex,
        ey,
        yerr=esd,
        fmt="o",
        color=COLOR_EMPIRICAL,
        markersize=MS,
        linewidth=0.8,
        capsize=CAPSIZE,
        zorder=4,
        label="Empirical (mean ± SD)",
    )

    ax.set_ylabel(ylabel, fontsize=FONTSIZE)

    # Panel label (a)-(f) — top-left, matching the other main-text figures.
    ax.text(
        0.03,
        0.97,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONTSIZE,
        fontweight="bold",
    )

    if show_legend:
        # Anchored just below the panel label, same corner, so it reads label-then-legend.
        ax.legend(
            fontsize=FONTSIZE + 1, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.90)
        )

    if annotation:
        # Mid-left, below the tag + legend above; the data curve is flat near the
        # low-c end of this log x-axis, so bottom-left would sit on top of it.
        ax.text(
            0.03,
            0.55,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONTSIZE - 1.5,
            color="#444444",
        )


# ---------------------------------------------------------------------------
# matplotlib global font settings
# ---------------------------------------------------------------------------


def _configure_matplotlib() -> None:
    """Set global matplotlib rcParams for ACS style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "axes.titlesize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "legend.fontsize": FONTSIZE - 1,
            "axes.linewidth": 0.8,
            "lines.linewidth": LW_MODEL,
            "pdf.fonttype": 42,  # TrueType in PDF
            "ps.fonttype": 42,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _configure_matplotlib()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output root:  {OUT_ROOT}")

    for illuminant in ("LED", "daylight"):
        make_illuminant_figures(illuminant)

    print("\nDone.")
