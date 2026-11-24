"""
Linearisation by forward-model inversion for Figure 7 (LED and daylight).

Three independent panel files (6a, 6b, 6c) + one assembled figure per illuminant.
Panels:
  6a — Ferricyanide B channel:    original vs model-corrected absorbance
  6b — Ferricyanide B/G channel:  original vs model-corrected absorbance
  6c — Methylene Blue R channel:  original vs model-corrected absorbance

Scientific context: forward-model inversion corrects Layer 1 (finite-bandpass
curvature) but does NOT remove the channel-common component (Layer 2), so
corrected B still shows scatter/offset vs the true Beer-Lambert line while
corrected BdivG tracks it closely because A_common cancelled before inversion.

Correction algorithm (from forward_model.py :: plot_linearisation):
  1. Fit Beer-Lambert slope from low-c empirical data (force-through-origin on µM scale).
  2. For each measured A_obs: interpolate on the monotone model curve A_model(c) to get
     c_corrected. Then A_corrected = slope_BL × c_corrected (in M).
  3. Expected Beer-Lambert line: A_BL = slope_BL × c (in M) over [0, c_max].

ACS formatting: 3.25 x 3.0 in per panel, Arial 8 pt, 600 dpi, inside ticks, no
title. Grey filled circles = original/measured; coloured filled squares =
model-corrected; dashed grey = expected Beer-Lambert line. Circle vs. square
(not just color) keeps the two series distinguishable in grayscale/B&W print
(review_notes_fjcampo.md, page 31, jdc 2026-07-11).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import figstyle as fs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "Data"

# LED paths
LED_FC_CAL = DATA / "analysis/ferricyanide/beer_lambert/calibration_table.csv"
LED_FC_MOD = DATA / "analysis/ferricyanide/forward_model/forward_model_table.csv"
LED_MB_CAL = DATA / "analysis/methylene_blue/beer_lambert/calibration_table.csv"
LED_MB_MOD = DATA / "analysis/methylene_blue/forward_model/forward_model_table.csv"

# Daylight (diffuser) paths
DAY_FC_CAL = (
    DATA / "analysis/ferricyanide/daylight_diffuser/image_analysis/"
    "s02_calibrate/calibration_table.csv"
)
DAY_FC_MOD = (
    DATA / "analysis/ferricyanide/daylight_diffuser/image_analysis/"
    "forward_model_daylight_diffuser/forward_model_table.csv"
)
DAY_MB_CAL = (
    DATA / "analysis/methylene_blue/daylight_diffuser/image_analysis/"
    "s02_calibrate/calibration_table.csv"
)
DAY_MB_MOD = (
    DATA / "analysis/methylene_blue/daylight_diffuser/image_analysis/"
    "forward_model_daylight_diffuser/forward_model_table.csv"
)

OUT_LED = PROJECT / "Data/paper_figures/figure_7/LED"
OUT_DAY = PROJECT / "Data/paper_figures/figure_7/daylight"
OUT_LED.mkdir(parents=True, exist_ok=True)
OUT_DAY.mkdir(parents=True, exist_ok=True)

# assemble_manuscript.py reads figures only from Manuscript/_drafts/figures/,
# never from Data/paper_figures/ — mirror every render there.
DRAFTS_OUT = PROJECT / "Manuscript" / "_drafts" / "figures" / "figure_7"

# ── ACS style constants ─────────────────────────────────────────────────────────

PANEL_W_IN = 3.25
PANEL_H_IN = 3.0
DPI = 600
FONTSIZE = 13  # drawn at ~10 in, placed at 7 in (x0.70): prints at 9 pt (jdc, 2026-10-05)
# the text in general a bit bigger", p.26)
PANEL_LABEL_FONTSIZE = 14  # prints at ~10 pt
# larger than axis text to match other figures' sizing
FONT_FAMILY = "Arial"

# Colours (match forward_model.py CHANNEL_COLORS)
COL_ORIG = fs.ROUTE_COLORS["raw"][2]  # RAW blue, open markers: measured
COL_GREY_LINE = fs.GRAY  # dashed Beer-Lambert reference
COL_FC_B = fs.ROUTE_COLORS["raw"][0]  # RAW blue, filled markers: corrected
COL_FC_BDIVG = COL_FC_B
COL_MB_R = COL_FC_B

# Low-c linear range for BL slope fit (µM): ferricyanide B is linear to 4000 µM
# and BdivG to 1000 µM, but we use the conservative forward-model fit range
# (62-1000 µM) to match the model's fitted l_cam. MB R and BdivG are linear to
# 50 µM (forward model fit range 0.78-100 µM), so we use ≤ 50 µM for MB.
LIN_MAX_FC_B = 1000.0  # µM; fits BL slope for ferricyanide B
LIN_MAX_FC_BDIVG = 1000.0  # µM; fits BL slope for ferricyanide B/G
LIN_MAX_MB_R = 50.0  # µM; fits BL slope for MB R

# Zoom-row display cap (µM): independent of the BL-slope fit range above.
# MB is capped tighter than its 50 µM fit range because the R-channel curve
# visibly departs from flat well before that.
ZOOM_MAX_FC = LIN_MAX_FC_B  # µM
ZOOM_MAX_MB = 25.0  # µM


# ── Helper functions ────────────────────────────────────────────────────────────


def apply_acs_style() -> None:
    """Set global rcParams for ACS style: Arial 8 pt, inside ticks."""
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "axes.titlesize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "legend.fontsize": FONTSIZE,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.0,
            "axes.linewidth": 0.8,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def load_cal_channel(cal_path: Path, channel: str) -> pd.DataFrame:
    """Return rows for a single channel, sorted by concentration."""
    cal = pd.read_csv(cal_path)
    sub = cal[cal["channel"] == channel].copy()
    if sub.empty:
        # try G1 proxy if G requested
        if channel.endswith("G"):
            sub = cal[cal["channel"] == channel + "1"].copy()
    sub = sub.sort_values("concentration_um").reset_index(drop=True)
    return sub


def bl_slope_origin(c_um: np.ndarray, A: np.ndarray, lin_max_um: float) -> float:
    """
    Force-through-origin Beer-Lambert slope (µM⁻¹) fitted to data up to lin_max_um.
    Returns slope in µM⁻¹.
    """
    mask = c_um <= lin_max_um
    c_fit = c_um[mask] * 1e-6  # convert to M for consistent units
    A_fit = A[mask]
    slope_M = float(np.dot(c_fit, A_fit) / np.dot(c_fit, c_fit))  # M⁻¹
    return slope_M  # A = slope_M × c_M = (slope_M * 1e-6) × c_µM


def model_correct(
    A_meas: np.ndarray,
    c_model_um: np.ndarray,
    A_model: np.ndarray,
    slope_M: float,
) -> np.ndarray:
    """
    Invert A_meas onto A_model(c) by interpolation → c_corrected_um.
    Then rescale to Beer-Lambert absorbance: A_corr = slope_M × c_corr_M.

    A_model must be monotonically increasing; we enforce this with cumulative max.
    """
    A_mono = np.maximum.accumulate(A_model)
    c_corr_um = np.interp(A_meas, A_mono, c_model_um)
    A_corr = slope_M * c_corr_um * 1e-6  # slope_M [M⁻¹] × c [M] → A [dimensionless]
    return A_corr


def save_csv(
    path: Path,
    c_um: np.ndarray,
    A_orig_mean: np.ndarray,
    A_orig_std: np.ndarray,
    A_corr_mean: np.ndarray,
    A_corr_std: np.ndarray,
    A_bl_expected: np.ndarray,
) -> None:
    """Export plotted data as CSV alongside each panel."""
    df = pd.DataFrame(
        {
            "concentration_um": c_um,
            "A_original_mean": A_orig_mean,
            "A_original_std": A_orig_std,
            "A_corrected_mean": A_corr_mean,
            "A_corrected_std": A_corr_std,
            "A_BL_expected": A_bl_expected,
        }
    )
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"  CSV: {path.name}")


def save_fig(fig: plt.Figure, stem: Path) -> None:
    """Save figure as PNG + TIFF + SVG at 600 dpi."""
    for ext in (".png", ".tiff", ".svg"):
        out = stem.with_suffix(ext)
        fig.savefig(str(out))
        print(f"  Figure: {out.name}")
    plt.close(fig)


# ── Panel rendering ─────────────────────────────────────────────────────────────


def _draw_axes(
    ax: plt.Axes,
    c_um: np.ndarray,
    A_orig: np.ndarray,
    A_std: np.ndarray,
    A_corr: np.ndarray,
    slope_M: float,
    corr_color: str,
    x_label: str,
    y_label: str,
    c_max: float,
    panel_label: str | None = None,
    show_legend: bool = False,
) -> None:
    """
    Draw one Beer-Lambert-vs-corrected axes over concentration range [0, c_max].
    Shared by the full-range row and the low-c zoom row (main panel and zoom
    are the same drawing at a different x-limit, not a separate inset), and by
    both the single-analyte figures and the assembled grid.
    """
    c_ref = np.array([0.0, c_max * 1.08])
    ax.plot(
        c_ref,
        slope_M * c_ref * 1e-6,
        color=COL_GREY_LINE,
        lw=1.0,
        ls="--",
        zorder=1,
        label="Beer-Lambert (low-c)",
    )
    mask = c_um <= c_max
    ax.errorbar(
        c_um[mask],
        A_orig[mask],
        yerr=A_std[mask],
        fmt="o",
        color=COL_ORIG,
        mfc="white",
        markersize=4,
        capsize=2,
        lw=0.8,
        zorder=2,
        label="Measured",
    )
    # same error bars for corrected: model inversion is a deterministic
    # rescaling, not a new measurement, so noise propagates unchanged
    # square (not circle) marker keeps this series distinguishable from
    # "Measured" in grayscale/B&W print, not just by color (jdc, page 31)
    ax.errorbar(
        c_um[mask],
        A_corr[mask],
        yerr=A_std[mask],
        fmt="s",
        color=corr_color,
        markersize=4,
        capsize=2,
        lw=0.8,
        zorder=3,
        label="Model-corrected",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(left=0, right=c_max * 1.05)
    y_sub_orig, y_sub_corr = A_orig[mask], A_corr[mask]
    y_max = max(y_sub_orig.max(), y_sub_corr.max()) * 1.08
    # anchor bottom at 0 (concentration/absorbance both physically >= 0) so the
    # x/y axes meet at the corner even when noisy near-zero points dip slightly
    # negative — those error-bar tails clip rather than pushing y=0 up the axis
    ax.set_ylim(0, y_max)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))

    if panel_label:
        # Panel label (a)-(f) — top-left inside
        ax.text(
            0.04,
            0.97,
            panel_label,
            transform=ax.transAxes,
            fontsize=PANEL_LABEL_FONTSIZE,
            fontweight="bold",
            va="top",
            ha="left",
        )

    if show_legend:
        ax.legend(
            frameon=False,
            loc="upper left",
            fontsize=FONTSIZE + 1,
            handlelength=1.4,
            handletextpad=0.4,
        )


def make_panel(
    cal_path: Path,
    mod_path: Path,
    channel: str,  # e.g. "B", "BdivG", "R"
    corr_color: str,
    lin_max_um: float,
    x_label: str,  # e.g. "[Fe(CN)₆³⁻] (µM)"
    y_label: str,  # e.g. "A_B"
    x_max_override: float | None = None,
    zoom_max_um: float | None = None,
) -> tuple[plt.Figure, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a two-row ACS panel (3.25 in wide, 2x3.0 in tall): full-range plot
    on top, low-concentration zoom below (BL review: "not very easy to see"
    the near-flat low-c region against the full-range axis), with one shared
    legend underneath both.

    Returns (fig, c_um, A_orig_mean, A_orig_std, A_corr, A_corr_std, A_bl_at_c).
    A_corr_std is taken equal to A_orig_std (same measurement noise;
    the model inversion is a deterministic rescaling, not a new measurement).
    """
    emp = load_cal_channel(cal_path, channel)
    c_um = emp["concentration_um"].to_numpy(dtype=float)
    A_orig = emp["absorbance_mean"].to_numpy(dtype=float)
    A_std = emp["absorbance_std"].to_numpy(dtype=float)

    mod = pd.read_csv(mod_path)
    c_model_um = mod["concentration_um"].to_numpy(dtype=float)

    col_map = {
        "B": "A_B",
        "G": "A_G",
        "R": "A_R",
        "BdivG": "A_BdivG",
        "RdivG": "A_RdivG",
        "RdivB": "A_RdivB",
        "GdivB": "A_GdivB",
    }
    model_col = col_map.get(channel, f"A_{channel}")
    if model_col not in mod.columns:
        sys.exit(f"Column '{model_col}' not in {mod_path.name}. Available: {list(mod.columns)}")
    A_model = mod[model_col].to_numpy(dtype=float)

    slope_M = bl_slope_origin(c_um, A_orig, lin_max_um)
    A_corr = model_correct(A_orig, c_model_um, A_model, slope_M)
    A_bl_at_c = slope_M * c_um * 1e-6

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(PANEL_W_IN, PANEL_H_IN * 2))

    c_max_full = x_max_override or c_um.max()
    _draw_axes(
        ax_full, c_um, A_orig, A_std, A_corr, slope_M, corr_color, x_label, y_label, c_max_full
    )
    _draw_axes(
        ax_zoom,
        c_um,
        A_orig,
        A_std,
        A_corr,
        slope_M,
        corr_color,
        x_label,
        y_label,
        zoom_max_um or lin_max_um,
    )

    fig.tight_layout(pad=0.5)
    fig.legend(
        *ax_full.get_legend_handles_labels(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=FONTSIZE + 1,
        handlelength=1.4,
        handletextpad=0.4,
        bbox_to_anchor=(0.5, -0.09),
    )

    return fig, c_um, A_orig, A_std, A_corr, A_std, A_bl_at_c


# ── Per-illuminant runner ───────────────────────────────────────────────────────


def run_illuminant(
    out_dir: Path,
    suffix: str,  # "LED" or "daylight"
    fc_cal: Path,
    fc_mod: Path,
    mb_cal: Path,
    mb_mod: Path,
) -> None:
    """
    Produce 6a, 6b, 6c panels + assembled figure for one illuminant.
    """
    print(f"\n=== {suffix} ===")

    # ── Panel A: Ferricyanide B ──
    fig_a, c_um_a, A_orig_a, A_std_a, A_corr_a, A_corr_std_a, A_bl_a = make_panel(
        cal_path=fc_cal,
        mod_path=fc_mod,
        channel="B",
        corr_color=COL_FC_B,
        lin_max_um=LIN_MAX_FC_B,
        x_label=r"$[\mathrm{Fe(CN)_6^{3-}}]$ ($\mu$M)",
        y_label=r"$A_\mathrm{B}$",
    )
    stem_a = out_dir / f"figure_7a_{suffix}"
    save_fig(fig_a, stem_a)
    save_csv(
        stem_a.with_suffix(".csv"),
        c_um_a,
        A_orig_a,
        A_std_a,
        A_corr_a,
        A_corr_std_a,
        A_bl_a,
    )

    # ── Panel B: Ferricyanide B/G ──
    fig_b, c_um_b, A_orig_b, A_std_b, A_corr_b, A_corr_std_b, A_bl_b = make_panel(
        cal_path=fc_cal,
        mod_path=fc_mod,
        channel="BdivG",
        corr_color=COL_FC_BDIVG,
        lin_max_um=LIN_MAX_FC_BDIVG,
        x_label=r"$[\mathrm{Fe(CN)_6^{3-}}]$ ($\mu$M)",
        y_label=r"$A_\mathrm{B/G}$",
    )
    stem_b = out_dir / f"figure_7b_{suffix}"
    save_fig(fig_b, stem_b)
    save_csv(
        stem_b.with_suffix(".csv"),
        c_um_b,
        A_orig_b,
        A_std_b,
        A_corr_b,
        A_corr_std_b,
        A_bl_b,
    )

    # ── Panel C: Methylene Blue R ──
    fig_c, c_um_c, A_orig_c, A_std_c, A_corr_c, A_corr_std_c, A_bl_c = make_panel(
        cal_path=mb_cal,
        mod_path=mb_mod,
        channel="R",
        corr_color=COL_MB_R,
        lin_max_um=LIN_MAX_MB_R,
        x_label=r"[MB] ($\mu$M)",
        y_label=r"$A_\mathrm{R}$",
        zoom_max_um=ZOOM_MAX_MB,
    )
    stem_c = out_dir / f"figure_7c_{suffix}"
    save_fig(fig_c, stem_c)
    save_csv(
        stem_c.with_suffix(".csv"),
        c_um_c,
        A_orig_c,
        A_std_c,
        A_corr_c,
        A_corr_std_c,
        A_bl_c,
    )

    # ── Assembled 6-panel grid: full-range row (a-c) over low-c zoom row (d-f) ──
    fig_asm, axes = plt.subplots(2, 3, figsize=(PANEL_W_IN * 3 + 0.3, PANEL_H_IN * 2 + 0.3))

    # Recompute slopes for assembled panels (same logic as make_panel)
    slope_M_a = bl_slope_origin(c_um_a, A_orig_a, LIN_MAX_FC_B)
    slope_M_b = bl_slope_origin(c_um_b, A_orig_b, LIN_MAX_FC_BDIVG)
    slope_M_c = bl_slope_origin(c_um_c, A_orig_c, LIN_MAX_MB_R)

    cols = [
        (
            c_um_a,
            A_orig_a,
            A_std_a,
            A_corr_a,
            slope_M_a,
            LIN_MAX_FC_B,
            COL_FC_B,
            r"$[\mathrm{Fe(CN)_6^{3-}}]$ ($\mu$M)",
            r"$A_\mathrm{B}$",
        ),
        (
            c_um_b,
            A_orig_b,
            A_std_b,
            A_corr_b,
            slope_M_b,
            LIN_MAX_FC_BDIVG,
            COL_FC_BDIVG,
            r"$[\mathrm{Fe(CN)_6^{3-}}]$ ($\mu$M)",
            r"$A_\mathrm{B/G}$",
        ),
        (
            c_um_c,
            A_orig_c,
            A_std_c,
            A_corr_c,
            slope_M_c,
            ZOOM_MAX_MB,
            COL_MB_R,
            "[MB] (µM)",
            r"$A_\mathrm{R}$",
        ),
    ]
    top_labels = ["(a)", "(b)", "(c)"]
    bot_labels = ["(d)", "(e)", "(f)"]

    for col_idx, (
        c_um,
        A_orig,
        A_std,
        A_corr,
        slope_M,
        zoom_max_um,
        corr_color,
        x_label,
        y_label,
    ) in enumerate(cols):
        _draw_axes(
            axes[0, col_idx],
            c_um,
            A_orig,
            A_std,
            A_corr,
            slope_M,
            corr_color,
            x_label,
            y_label,
            c_um.max(),
            panel_label=top_labels[col_idx],
        )
        _draw_axes(
            axes[1, col_idx],
            c_um,
            A_orig,
            A_std,
            A_corr,
            slope_M,
            corr_color,
            x_label,
            y_label,
            zoom_max_um,
            panel_label=bot_labels[col_idx],
        )

    fig_asm.tight_layout(pad=0.5)
    fig_asm.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=FONTSIZE + 1,
        handlelength=1.4,
        handletextpad=0.4,
        bbox_to_anchor=(0.5, -0.07),
    )
    stem_asm = out_dir / f"figure_7_assembled_{suffix}"
    save_fig(fig_asm, stem_asm)

    # ── Copy everything to Manuscript/_drafts/figures/figure_7/<suffix>/ ──
    dst_dir = DRAFTS_OUT / suffix
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, dst_dir / f.name)


# ── Main ────────────────────────────────────────────────────────────────────────


def main() -> None:
    apply_acs_style()

    run_illuminant(
        out_dir=OUT_LED,
        suffix="LED",
        fc_cal=LED_FC_CAL,
        fc_mod=LED_FC_MOD,
        mb_cal=LED_MB_CAL,
        mb_mod=LED_MB_MOD,
    )

    run_illuminant(
        out_dir=OUT_DAY,
        suffix="daylight",
        fc_cal=DAY_FC_CAL,
        fc_mod=DAY_FC_MOD,
        mb_cal=DAY_MB_CAL,
        mb_mod=DAY_MB_MOD,
    )

    print("\nAll figure_7 outputs written.")


if __name__ == "__main__":
    main()
