"""
figure_S3_ftail.py

SI Figure S7 companion — f_tail concept illustration, pedagogical
complement to the Beer-Lambert-Bayer integral / transparent-tail-fraction
derivation in S7's text. Ferricyanide is the clean transparent-tail case;
methylene blue is shown as the contrast case (broad, non-flat epsilon(lambda),
small transparent tail below the 426 nm cutoff).

Generates two separate 2x3 composite panels, one per analyte:

  figure_S3_ftail_ferricyanide.{png,svg}  — B-channel analysis
  figure_S3_ftail_mb.{png,svg}            — R-channel analysis

Layout (per composite):
  Row 1 (a-c)  Approach A — phi(lambda) * T_ch(lambda) effective illumination
               with transparent-tail shading, overlaid with A(lambda) at
               c_min and c_max [twin y-axes]
  Row 2 (d-f)  Approach B — integrand phi*T*10^(-A) at blank, c_min, c_max:
               shows that only the transparent tail survives at high concentration

Columns (left to right): Measured daylight | CIE D65 | Spelec LED (this work)

f_tail (transparent-tail fraction) definition:
  f_tail = integral_tail[ phi(lambda) * T_ch(lambda) d(lambda) ]
           / integral_all[ phi(lambda) * T_ch(lambda) d(lambda) ]
  where "tail" = wavelengths where the analyte has epsilon approx 0

  Ferricyanide B-channel: tail at lambda > 473 nm
  Methylene blue R-channel: tail at lambda < 426 nm
  (both MB peaks — 665 nm monomer + 610 nm dimer — and the valley at 530 nm
   all absorb within the R-ch sensitive range; eps only falls below 1% of peak
   between 411 and 426 nm, where the R-ch barely responds, so f_tail is
   nearly zero for MB)

    python Data\\paper_figures\\figure_S3_ftail.py

Outputs -> Data/paper_figures/figure_S3/
    figure_S3_ftail_ferricyanide.{png,svg}
    figure_S3_ftail_ferricyanide_abc.{png,svg}   trimmed 1x3 (round 2e, G-F/W6):
        panels (d-f) duplicate main-text Figure 5 panel A and are dropped from
        the SI render; this is the file that ships as the SI figure. The full
        2x6 composite above is kept on disk for provenance, not for the SI.
    figure_S3_ftail_mb.{png,svg}                 ships whole, all six panels
    figure_S3_ftail_feric_{a-f}.{png,svg}   individual panels
    figure_S3_ftail_mb_{a-f}.{png,svg}      individual panels
    figure_S3_ftail_data.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJ_ROOT / "Data"
ILLUM_DIR = DATA_ROOT / "spectra" / "illuminant"
FERIC_DIR = DATA_ROOT / "spectra" / "ferricyanide"
MB_DIR = DATA_ROOT / "spectra" / "methylene_blue"
OUT_DIR = DATA_ROOT / "paper_figures" / "figure_S3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Vendored raw-image-analysis modules (vendor/, see README); override with COLORIMETRY_VENDOR.
SENS_DIR = (
    Path(
        os.environ.get("COLORIMETRY_VENDOR", str(Path(__file__).resolve().parent.parent / "vendor"))
    )
    / "_resources"
    / "spectral_sensitivity"
)
sys.path.insert(0, str(SCRIPT_DIR))
from forward_model import split_at_cutoff  # noqa: E402
from uvvis_calibration_figures import (  # noqa: E402
    _mean_spectrum,
    load_ferricyanide_set,
    load_mb_set,
)

# ── CIE D65 standard illuminant (ASTM E308, 350-800 nm) ──────────────────────
_D65_WL = np.arange(350, 810, 10, dtype=float)  # 46 points
_D65_SPD = np.array(
    [
        # 350-370 nm (UV tail, linearly extrapolated)
        15.0,
        35.0,
        46.64,
        # 380-780 nm (ASTM E308)
        49.98,
        54.65,
        82.75,
        91.49,
        93.43,
        86.68,
        104.87,
        117.01,
        117.81,
        114.86,
        115.92,
        108.81,
        109.35,
        107.80,
        104.79,
        107.69,
        104.41,
        104.04,
        100.00,
        96.33,
        95.79,
        88.69,
        90.01,
        89.60,
        87.70,
        83.29,
        83.70,
        80.03,
        80.21,
        82.28,
        78.28,
        73.47,
        81.55,
        83.07,
        78.47,
        75.29,
        80.30,
        74.66,
        71.29,
        77.14,
        69.11,
        # 790-800 nm (linearly extrapolated)
        68.79,
        68.09,
    ],
    dtype=float,
)

# ── Constants ─────────────────────────────────────────────────────────────────
X_MIN, X_MAX = 350, 800
X_TICKS = [400, 500, 600, 700, 800]
WL_GRID = np.arange(X_MIN, X_MAX + 1, 1, dtype=float)  # 451 points
SENS_WL = np.arange(400, 710, 10, dtype=float)  # Ohlab native grid

# Analyte / channel settings
FERIC_CH = 2  # B = column index 2 in Ohlab files (R=0, G=1, B=2)
FERIC_TAIL_WL = 473.0  # eps < 1% of its 420 nm peak above 473 nm; tail is lambda > 473
FERIC_TAIL_ABOVE = True  # True: tail at lambda > tail_wl
FERIC_TAIL_CLR = fs.CHANNEL_COLORS["B"]  # B channel is plotted (tail shading + integrand)
FERIC_SPEC_CLR = fs.FIT_COLOR  # absorbance lines

MB_CH = 0  # R = column index 0
MB_TAIL_WL = 426.0  # eps < 1% of its 665 nm peak below 426 nm; tail is lambda < 426
MB_TAIL_ABOVE = False  # False: tail at lambda < tail_wl
MB_TAIL_CLR = fs.CHANNEL_COLORS["R"]  # R channel is plotted (tail shading + integrand)
MB_SPEC_CLR = fs.FIT_COLOR  # absorbance lines

# Figure geometry
PANEL_W, PANEL_H = 7.0, 5.2  # double-column composite (in)
INDIV_W, INDIV_H = 3.33, 2.60  # single-column individual panel (in)

ILLUM_KEYS = ["d65", "led", "daylight"]
ILLUM_TITLES = ["CIE D65", "White LED", "Daylight"]
LETTERS = list("abcdef")


def _conc_label(c_um: float) -> str:
    """Format concentration for figure labels: nM / µM / mM as appropriate."""
    if c_um >= 1000:
        return f"{c_um / 1000:.4g} mM"
    elif c_um >= 1:
        return f"{c_um:.4g} μM"
    else:
        return f"{c_um * 1000:.4g} nM"


# ── ACS rcParams ──────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "legend.fontsize": 5.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.9,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "figure.dpi": 150,
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def _load_and_normalise_csv(csv_path: Path) -> np.ndarray:
    """Load 2-column illuminant CSV; interpolate onto WL_GRID, normalise peak=1."""
    df = pd.read_csv(csv_path)
    wl = df.iloc[:, 0].values.astype(float)
    spd = df.iloc[:, 1].values.astype(float)
    out = np.interp(WL_GRID, wl, spd, left=0.0, right=0.0)
    peak = out.max()
    return out / peak if peak > 0 else out


def load_d65() -> np.ndarray:
    """CIE D65 on WL_GRID: cubic spline + Gaussian smooth (sigma=4 nm), peak=1."""
    cs = CubicSpline(_D65_WL, _D65_SPD)
    out = np.clip(cs(WL_GRID), 0, None)
    out = gaussian_filter1d(out, sigma=4)
    peak = out.max()
    return out / peak if peak > 0 else out


def load_channel(channel_idx: int) -> np.ndarray:
    """
    Load average_all.txt from Ohlab Bayer database.
    Returns the requested channel (R=0, G=1, B=2) normalised to peak=1 on WL_GRID.
    """
    fp = SENS_DIR / "average_all.txt"
    if not fp.exists():
        raise FileNotFoundError(
            f"average_all.txt not found in:\n  {SENS_DIR}\n"
            "Set COLORIMETRY_VENDOR to the vendor/ folder."
        )
    raw = np.loadtxt(fp)  # (31, 3)
    ch = np.interp(WL_GRID, SENS_WL, raw[:, channel_idx], left=0.0, right=0.0)
    peak = ch.max() or 1.0
    return ch / peak


def load_all_spectra(csv_dir: Path, loader_fn):
    """Load all CSVs in csv_dir; return (sets_data, sorted_concs) — no concentration cap."""
    csvs = sorted(csv_dir.glob("*.csv"))
    sets = [loader_fn(p) for p in csvs]
    concs = sorted(set().union(*[set(d.keys()) for d in sets]))
    return sets, concs


def mean_A_on_grid(sets_data: list, c: float) -> np.ndarray:
    """Mean absorbance at concentration c interpolated onto WL_GRID (0 outside range)."""
    wl, mean_A, _ = _mean_spectrum(sets_data, c)
    if not len(wl):
        return np.zeros_like(WL_GRID)
    mean_A = np.clip(mean_A, 0, None)  # suppress negative noise
    return np.interp(WL_GRID, wl, mean_A, left=0.0, right=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SPECTRAL COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def compute_phi_T(
    phi: np.ndarray,
    T_ch: np.ndarray,
    tail_wl: float,
    tail_above: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the effective channel illumination phi(lambda) * T_ch(lambda).

    Returns:
      phi_T_norm : array normalised to peak = 1
      tail_mask  : boolean array — True in the transparent-tail region
      f_tail     : fraction of total integral in the tail region (0 to 1)
    """
    phi_T = phi * T_ch
    peak = phi_T.max()
    phi_T_norm = phi_T / peak if peak > 0 else phi_T

    tail_mask = (WL_GRID > tail_wl) if tail_above else (WL_GRID < tail_wl)

    total = float(np.trapezoid(phi_T, WL_GRID))
    tail, _ = split_at_cutoff(phi_T, tail_mask, WL_GRID)
    f_tail = tail / total if total > 0 else 0.0

    return phi_T_norm, tail_mask, f_tail


def compute_integrand(phi_T_norm: np.ndarray, A_on_grid: np.ndarray) -> np.ndarray:
    """
    Attenuated integrand: phi * T * 10^(-A(lambda)).
    phi_T_norm already normalised to peak=1; transmittance clips negative A to 0.
    """
    trans = np.power(10.0, -np.clip(A_on_grid, 0, None))
    return phi_T_norm * trans


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _setup_ax(ax: plt.Axes, ylabel: str = "", xlabel: bool = True) -> None:
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks(X_TICKS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:
        ax.set_xlabel("Wavelength (nm)")
    if ylabel:
        ax.set_ylabel(ylabel)


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.03,
        0.97,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _save_fig(fig: plt.Figure, stem: str, dpi: int = 600) -> None:
    """Save fig as PNG, SVG and TIFF (TIFF via PIL, avoids a large-canvas bug
    in matplotlib's native TIFF writer with bbox_inches='tight')."""
    png_path = OUT_DIR / f"{stem}.png"
    for ext, path in (("png", png_path), ("svg", OUT_DIR / f"{stem}.svg")):
        fig.savefig(path, dpi=dpi, bbox_inches="tight", format=ext)
    from PIL import Image

    img = Image.open(png_path)
    img.save(OUT_DIR / f"{stem}.tiff", dpi=(dpi, dpi), compression="tiff_lzw")
    print(f"    -> {stem}.png / .svg / .tiff")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL DRAWING
# ═══════════════════════════════════════════════════════════════════════════════


def draw_row1(
    ax: plt.Axes,
    phi_T_norm: np.ndarray,
    tail_mask: np.ndarray,
    A_cmin: np.ndarray,
    A_cmax: np.ndarray,
    tail_color: str,
    spec_color: str,
    f_tail: float,
    tail_wl: float,
    channel_label: str,
    col_title: str = "",
    show_right_axis: bool = True,
    ftail_xy: tuple[float, float] = (0.04, 0.18),
    c_min: float = 0.0,
    c_max: float = 0.0,
    legend_loc: str = "lower left",
    legend_bbox: tuple[float, float] | None = None,
) -> plt.Axes:
    """
    Approach A panel.

    Left y-axis: phi*T_ch curve (grey) with transparent-tail region (coloured).
    Right y-axis: absorbance A(lambda) at c_min (thin dashed) and c_max (thick solid).
    Returns the twinx Axes object.
    """
    # -- Left y-axis: effective channel illumination ---------------------------
    tail_dir = ">" if bool(tail_mask[-1]) else "<"
    ax.fill_between(WL_GRID, 0, phi_T_norm, color="#D0D0D0", alpha=0.85, zorder=1, label="Blank")
    ax.fill_between(
        WL_GRID,
        0,
        np.where(tail_mask, phi_T_norm, 0.0),
        color=tail_color,
        alpha=0.48,
        zorder=2,
        label=f"Tail ({tail_dir} {tail_wl:.0f} nm)",
    )
    ax.plot(WL_GRID, phi_T_norm, color="#555555", lw=0.7, zorder=3)
    ax.axvline(
        tail_wl, color="#777777", lw=0.6, ls="--", alpha=0.75, zorder=4, label="Tail boundary"
    )

    ax.set_ylim(-0.05, 1.20)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    _setup_ax(ax, ylabel=f"Eff. illum. ({channel_label}, norm.)")

    # f_tail annotation
    ax.text(
        ftail_xy[0],
        ftail_xy[1],
        f"f_tail = {f_tail * 100:.2g}%",
        transform=ax.transAxes,
        fontsize=6.5,
        ha="left",
        va="bottom",
        color=tail_color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
    )

    if col_title:
        ax.set_title(col_title, fontsize=7, pad=3)

    # -- Right y-axis: absorbance spectra -------------------------------------
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    A_max_val = float(np.nanmax(np.where(A_cmax > 0, A_cmax, np.nan)))
    y_top = A_max_val * 1.12 if np.isfinite(A_max_val) else 0.5
    y_top = max(y_top, 0.01)

    ax2.plot(
        WL_GRID, A_cmin, color=spec_color, lw=1.0, ls=":", alpha=0.90, zorder=5, label="A (c_min)"
    )
    ax2.plot(WL_GRID, A_cmax, color=spec_color, lw=1.0, alpha=0.90, zorder=6, label="A (c_max)")
    ax2.set_ylim(-0.04 * y_top, y_top)

    if show_right_axis:
        ax2.set_ylabel("Absorbance (AU)", fontsize=7, color=spec_color)
        ax2.tick_params(axis="y", colors=spec_color, labelsize=6.5, width=0.6, length=3.0)
        ax2.spines["right"].set_edgecolor(spec_color)
        ax2.spines["right"].set_linewidth(0.6)
    else:
        ax2.set_yticklabels([])
        ax2.tick_params(axis="y", length=0)
        ax2.spines["right"].set_visible(False)

    # -- Combined legend (left + right axis handles) --------------------------
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend_kwargs = dict(
        fontsize=5.5,
        framealpha=0.0,
        edgecolor="none",
        loc=legend_loc,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.3,
    )
    if legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox
    ax.legend(h1 + h2, l1 + l2, **legend_kwargs)

    return ax2


def draw_row2(
    ax: plt.Axes,
    phi_T_norm: np.ndarray,
    intg_cmin: np.ndarray,
    intg_cmax: np.ndarray,
    tail_mask: np.ndarray,
    tail_color: str,
    tail_wl: float,
    c_min: float,
    c_max: float,
    legend_loc: str = "center left",
) -> None:
    """
    Approach B panel: integrand collapse.

    Shows phi*T*10^(-A) at blank, c_min and c_max.  At c_max only the
    transparent tail survives; the absorbing region collapses toward zero.
    All curves share a single normalised y-axis (blank peak = 1).
    """
    # Blank — light grey (all channel photons)
    ax.fill_between(WL_GRID, 0, phi_T_norm, color="#D0D0D0", alpha=0.80, label="Blank", zorder=1)
    ax.plot(WL_GRID, phi_T_norm, color="#888888", lw=0.6, zorder=2)

    # c_min — dotted line only (barely different from blank)
    ax.plot(
        WL_GRID,
        intg_cmin,
        color="#666666",
        lw=0.7,
        ls=":",
        alpha=0.90,
        label=f"c = {_conc_label(c_min)}",
        zorder=3,
    )

    # c_max — coloured fill: only transparent tail survives
    ax.fill_between(
        WL_GRID,
        0,
        intg_cmax,
        color=tail_color,
        alpha=0.55,
        label=f"c = {_conc_label(c_max)}",
        zorder=4,
    )
    ax.plot(WL_GRID, intg_cmax, color=tail_color, lw=0.8, zorder=5)

    ax.axvline(tail_wl, color="#777777", lw=0.6, ls="--", alpha=0.75, zorder=6)

    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    _setup_ax(ax, ylabel="Integrand (norm.)")

    ax.legend(
        framealpha=0.8,
        facecolor="white",
        edgecolor="none",
        fontsize=5.5,
        handlelength=1.2,
        loc=legend_loc,
        labelspacing=0.2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════


def build_analyte_figure(
    analyte_label: str,
    stem: str,
    panel_prefix: str,
    illuminants: dict[str, np.ndarray],
    T_ch: np.ndarray,
    channel_label: str,
    sets_data: list,
    c_min: float,
    c_max: float,
    tail_wl: float,
    tail_above: bool,
    tail_color: str,
    spec_color: str,
    csv_rows: dict,
    ftail_xy: tuple[float, float] = (0.04, 0.18),
    legend_loc: str = "center left",
    row1_legend_loc: str = "lower left",
    row1_legend_bbox: tuple[float, float] | None = None,
) -> None:
    """Build the 2x3 composite and 6 individual panels for one analyte."""
    print(f"\n  Building {analyte_label} figure ...")

    # Pre-compute absorbance spectra on WL_GRID (same for all illuminant columns)
    A_cmin = mean_A_on_grid(sets_data, c_min)
    A_cmax = mean_A_on_grid(sets_data, c_max)

    csv_rows[f"{stem}_A_cmin_{c_min:.4g}uM"] = A_cmin
    csv_rows[f"{stem}_A_cmax_{c_max:.4g}uM"] = A_cmax

    # Pre-compute per-illuminant spectral quantities
    col_data: list[tuple] = []
    for key in ILLUM_KEYS:
        phi = illuminants[key]
        phi_T_norm, tail_mask, f_tail = compute_phi_T(phi, T_ch, tail_wl, tail_above)
        intg_cmin = compute_integrand(phi_T_norm, A_cmin)
        intg_cmax = compute_integrand(phi_T_norm, A_cmax)
        col_data.append((phi_T_norm, tail_mask, f_tail, intg_cmin, intg_cmax))
        print(f"    {key:8s}: f_tail = {f_tail:.1%}")

        pfx = f"{stem}_{key}"
        csv_rows[f"{pfx}_phi_T_norm"] = phi_T_norm
        csv_rows[f"{pfx}_tail_mask"] = tail_mask.astype(float)
        csv_rows[f"{pfx}_intg_cmin"] = intg_cmin
        csv_rows[f"{pfx}_intg_cmax"] = intg_cmax

    # ── 2x3 composite ─────────────────────────────────────────────────────────
    print(f"  Saving {analyte_label} composite ...")
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(PANEL_W, PANEL_H),
        gridspec_kw={"hspace": 0.55, "wspace": 0.48},
    )

    for col, title in enumerate(ILLUM_TITLES):
        phi_T_norm, tail_mask, f_tail, intg_cmin, intg_cmax = col_data[col]

        # Row 1: only rightmost column shows right y-axis decorations
        draw_row1(
            axes[0, col],
            phi_T_norm,
            tail_mask,
            A_cmin,
            A_cmax,
            tail_color,
            spec_color,
            f_tail,
            tail_wl,
            channel_label,
            col_title=title,
            show_right_axis=(col == 2),
            ftail_xy=ftail_xy,
            c_min=c_min,
            c_max=c_max,
            legend_loc=row1_legend_loc,
            legend_bbox=row1_legend_bbox,
        )
        _panel_letter(axes[0, col], LETTERS[col])

        # Row 2: no twin axis needed
        draw_row2(
            axes[1, col],
            phi_T_norm,
            intg_cmin,
            intg_cmax,
            tail_mask,
            tail_color,
            tail_wl,
            c_min,
            c_max,
            legend_loc=legend_loc,
        )
        _panel_letter(axes[1, col], LETTERS[col + 3])

    _save_fig(fig, stem)
    plt.close(fig)

    # ── Individual panels ─────────────────────────────────────────────────────
    print(f"  Saving {analyte_label} individual panels ...")
    for col, title in enumerate(ILLUM_TITLES):
        phi_T_norm, tail_mask, f_tail, intg_cmin, intg_cmax = col_data[col]

        # Row 1 individual panel (always show right axis)
        letter_r1 = LETTERS[col]
        fig1, ax1 = plt.subplots(figsize=(INDIV_W, INDIV_H))
        draw_row1(
            ax1,
            phi_T_norm,
            tail_mask,
            A_cmin,
            A_cmax,
            tail_color,
            spec_color,
            f_tail,
            tail_wl,
            channel_label,
            col_title=title,
            show_right_axis=True,
            ftail_xy=ftail_xy,
            c_min=c_min,
            c_max=c_max,
            legend_loc=row1_legend_loc,
            legend_bbox=row1_legend_bbox,
        )
        _panel_letter(ax1, letter_r1)
        fig1.tight_layout(pad=0.5)
        _save_fig(fig1, f"{panel_prefix}_{letter_r1}")
        plt.close(fig1)

        # Row 2 individual panel
        letter_r2 = LETTERS[col + 3]
        fig2, ax2 = plt.subplots(figsize=(INDIV_W, INDIV_H))
        draw_row2(
            ax2,
            phi_T_norm,
            intg_cmin,
            intg_cmax,
            tail_mask,
            tail_color,
            tail_wl,
            c_min,
            c_max,
            legend_loc=legend_loc,
        )
        _panel_letter(ax2, letter_r2)
        fig2.tight_layout(pad=0.5)
        _save_fig(fig2, f"{panel_prefix}_{letter_r2}")
        plt.close(fig2)


def build_trimmed_row1_composite(
    stem: str,
    illuminants: dict[str, np.ndarray],
    T_ch: np.ndarray,
    channel_label: str,
    sets_data: list,
    c_min: float,
    c_max: float,
    tail_wl: float,
    tail_above: bool,
    tail_color: str,
    spec_color: str,
    ftail_xy: tuple[float, float],
    legend_loc: str = "lower left",
    legend_bbox: tuple[float, float] | None = None,
) -> None:
    """Round 2e, gate G-F / W6: rebuild S9a (now S8a) with only row 1 (a-c) --
    row 2 (d-f) is a strict subset of main-text Figure 5 panel A (the same
    integrand for the same analyte and channel) and is dropped from the SI
    render. Recomputes rather than crops the existing composite, so panel
    sizing and font scale match every other individual-row figure in this
    script exactly."""
    print(f"\n  Building trimmed (a-c) composite for {stem} ...")
    A_cmin = mean_A_on_grid(sets_data, c_min)
    A_cmax = mean_A_on_grid(sets_data, c_max)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(PANEL_W, PANEL_H / 2),
        gridspec_kw={"wspace": 0.48},
    )
    for col, title in enumerate(ILLUM_TITLES):
        phi = illuminants[ILLUM_KEYS[col]]
        phi_T_norm, tail_mask, f_tail = compute_phi_T(phi, T_ch, tail_wl, tail_above)
        draw_row1(
            axes[col],
            phi_T_norm,
            tail_mask,
            A_cmin,
            A_cmax,
            tail_color,
            spec_color,
            f_tail,
            tail_wl,
            channel_label,
            col_title=title,
            show_right_axis=(col == 2),
            ftail_xy=ftail_xy,
            c_min=c_min,
            c_max=c_max,
            legend_loc=legend_loc,
            legend_bbox=legend_bbox,
        )
        _panel_letter(axes[col], LETTERS[col])

    _save_fig(fig, f"{stem}_abc")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    print(f"[figure_S3_ftail] Output -> {OUT_DIR}\n")

    # ── Illuminants ───────────────────────────────────────────────────────────
    print("Loading illuminants ...")
    illuminants = {
        "daylight": _load_and_normalise_csv(ILLUM_DIR / "illuminant_daylight_clean.csv"),
        "d65": load_d65(),
        "led": _load_and_normalise_csv(ILLUM_DIR / "illuminant_spelec_led.csv"),
    }
    for key, spd in illuminants.items():
        print(f"  {key}: peak at {WL_GRID[spd.argmax()]:.0f} nm")

    # ── Bayer sensitivity (average_all) ───────────────────────────────────────
    print("Loading Bayer sensitivity (average_all) ...")
    T_B = load_channel(FERIC_CH)  # B channel for ferricyanide
    T_R = load_channel(MB_CH)  # R channel for MB
    print(f"  B channel peak at {WL_GRID[T_B.argmax()]:.0f} nm")
    print(f"  R channel peak at {WL_GRID[T_R.argmax()]:.0f} nm")

    # ── Spectra (no concentration cap — use full dataset range) ───────────────
    print("Loading ferricyanide spectra ...")
    feric_sets, feric_concs = load_all_spectra(FERIC_DIR, load_ferricyanide_set)
    feric_cmin, feric_cmax = min(feric_concs), max(feric_concs)
    print(f"  {len(feric_concs)} concs; c_min = {feric_cmin:.4g} uM, c_max = {feric_cmax:.4g} uM")

    print("Loading methylene blue spectra ...")
    mb_sets, mb_concs = load_all_spectra(MB_DIR, load_mb_set)
    mb_cmin, mb_cmax = min(mb_concs), max(mb_concs)
    print(f"  {len(mb_concs)} concs; c_min = {mb_cmin:.4g} uM, c_max = {mb_cmax:.4g} uM")

    # Shared CSV accumulator
    csv_rows: dict[str, np.ndarray] = {"wavelength_nm": WL_GRID}
    for key, spd in illuminants.items():
        csv_rows[f"illuminant_{key}_norm"] = spd
    csv_rows["sensitivity_B_norm"] = T_B
    csv_rows["sensitivity_R_norm"] = T_R

    # ── Ferricyanide figure ────────────────────────────────────────────────────
    # Fill peaks left (B-ch at 460-480 nm); f_tail and legend go upper-left
    # where λ < 420 nm has near-zero fill, or upper-right beyond the orange tail.
    build_analyte_figure(
        analyte_label="Ferricyanide",
        stem="figure_S3_ftail_ferricyanide",
        panel_prefix="figure_S3_ftail_feric",
        illuminants=illuminants,
        T_ch=T_B,
        channel_label="B",
        sets_data=feric_sets,
        c_min=feric_cmin,
        c_max=feric_cmax,
        tail_wl=FERIC_TAIL_WL,
        tail_above=FERIC_TAIL_ABOVE,
        tail_color=FERIC_TAIL_CLR,
        spec_color=FERIC_SPEC_CLR,
        csv_rows=csv_rows,
        ftail_xy=(0.58, 0.55),
        legend_loc="upper right",
        row1_legend_loc="upper right",
    )
    build_trimmed_row1_composite(
        stem="figure_S3_ftail_ferricyanide",
        illuminants=illuminants,
        T_ch=T_B,
        channel_label="B",
        sets_data=feric_sets,
        c_min=feric_cmin,
        c_max=feric_cmax,
        tail_wl=FERIC_TAIL_WL,
        tail_above=FERIC_TAIL_ABOVE,
        tail_color=FERIC_TAIL_CLR,
        spec_color=FERIC_SPEC_CLR,
        ftail_xy=(0.58, 0.55),
        legend_loc="upper right",
    )

    # ── Methylene blue figure ─────────────────────────────────────────────────
    # Fill peaks right (R-ch at 590-600 nm); f_tail goes mid-left (empty space)
    build_analyte_figure(
        analyte_label="Methylene blue",
        stem="figure_S3_ftail_mb",
        panel_prefix="figure_S3_ftail_mb",
        illuminants=illuminants,
        T_ch=T_R,
        channel_label="R",
        sets_data=mb_sets,
        c_min=mb_cmin,
        c_max=mb_cmax,
        tail_wl=MB_TAIL_WL,
        tail_above=MB_TAIL_ABOVE,
        tail_color=MB_TAIL_CLR,
        spec_color=MB_SPEC_CLR,
        csv_rows=csv_rows,
        ftail_xy=(0.04, 0.22),
        legend_loc="center left",
        row1_legend_loc="upper left",
        row1_legend_bbox=(0.02, 0.90),
    )

    # ── Write CSV ─────────────────────────────────────────────────────────────
    print("\nWriting data CSV ...")
    df = pd.DataFrame(csv_rows)
    csv_path = OUT_DIR / "figure_S3_ftail_data.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"    -> figure_S3_ftail_data.csv  ({len(df.columns)} columns, {len(df)} rows)")

    print(f"\n[figure_S3_ftail] Done.\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
