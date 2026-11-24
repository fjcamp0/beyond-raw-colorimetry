"""
figure_3_v1.py — Spectral characterisation (2x3 panel, ACS double-column).

FROZEN 2026-09-30: this is the first-submission Figure 3, kept for provenance.
The revised figure is built by figure_3.py, which imports this module's loaders
and eps(lambda) estimator. Outputs now go to Data/paper_figures/figure_3/v1/.

Renders as **main-text Figure 3** since the 2026-07-28 renumbering (gate G-C).
Was figure_4.py; renamed 2026-07-31 (with OUT_DIR and every output filename
stem) so the script and its outputs match the manuscript number -- see
figures.yaml's Figure 3 entry and figures_plan.md's "Renumbering mechanics"
for why OUT_DIR was moved carefully instead of just renaming the file.

  (a) CIE D65 standard illuminant SPD
  (b) Spelec LED illuminant SPD (this work)
  (c) Measured daylight spectral power distribution (SPD)
  (d) Bayer filter spectral sensitivity - 20 smartphones, mean +/- 95% band

  Panels (a)-(c) reordered 2026-07-28 (round 2g checkpoint correction, jdc):
  D65/LED/daylight, not daylight/D65/LED -- puts the two reference/synthetic
  illuminants first and the measured field illuminant last. Content unchanged,
  only which letter draws which curve.
  (e) K3[Fe(CN)6] molar absorptivity spectrum, eps(lambda)
  (f) Methylene blue molar absorptivity spectrum, eps(lambda)

All x-axes: 350-800 nm. Y-axes normalised for (a-d); molar absorptivity
(M^-1 cm^-1) for (e-f).

Panels (e)/(f) plotted absorbance until 2026-07-28. jdc's comment C3 / work item
W3 of round 2e: "Figures 3e and 3f show absorbances, not molar absorptivity
coefficients: change the figures accordingly". Three conventions are available
via --eps-mode, because eps is a property of the substance and not of a single
spectrum, so the choice of estimator is a real one (gate G-I / open question
Q-K):

  per-conc  (shipped default 2026-07-31 to 2026-07-31 only, jdc's F6 review
            comment, correction_plan_jdc_2026-07-30.md: "add other
            concentrations to plots (e) and (f)... must come with a
            concentration scalebar along them") a family of curves,
            eps_c = A_c / (c l), one per concentration of one replicate set,
            colour-graded by concentration with a ScalarMappable colorbar.
            Keeps the concentration-series look of the old absorbance panels
            but does not collapse onto one curve, since the calibrations carry
            a non-zero intercept. Reverted same day, F6 reversal (jdc, direct
            request): jdc found the assembled multi-curve/colorbar panels
            worse, not better, than the pooled single curve.
  pooled    (default again since the F6 reversal, 2026-07-31) one eps(lambda)
            curve per analyte, from the per-wavelength
            OLS slope of A against c over every replicate point -- the G-G pooled
            convention applied at every wavelength instead of only at lambda_max.
            Reproduces the published anchors exactly by construction. Ferricyanide
            uses the full measured concentration range; methylene blue is
            RESTRICTED to 1.525-50 uM (POOLED_CONC_RANGE, jdc's ruling 2026-07-28)
            so that eps(665) falls inside the 70,000-95,000 M-1cm-1 literature
            range for the monomer, rather than the 63,457 the full 0.78-100 uM
            range gives. See POOLED_CONC_RANGE's comment for why those two points
            specifically.
  single    same estimator over one replicate set only (--set N), honouring W3's
            "use the spectra for one dataset".

Path length is the 2 mm / 0.2 cm UV-Vis cuvette of Methods 3 (L_UVVIS_CM), not
the fitted camera l_cam.

Run with: python scripts/figure_3_v1.py (from the project root).
Uses the vendored Ohlab curves and modules in vendor/ (override with COLORIMETRY_VENDOR).

Outputs -> Data/paper_figures/figure_3/
    figure_3_panel.{png,svg,tiff}                composite 2x3 panel
    figure_3a.{png,svg,tiff} ... figure_3f.{...}  individual panels
    figure_3_data.csv                             all plotted data, 1-nm grid
    figure_3_epsilon_check.txt                    eps at lambda_max vs anchors
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJ_ROOT / "Data"
ILLUM_DIR = DATA_ROOT / "spectra" / "illuminant"
FERIC_DIR = DATA_ROOT / "spectra" / "ferricyanide"
MB_DIR = DATA_ROOT / "spectra" / "methylene_blue"
OUT_DIR = DATA_ROOT / "paper_figures" / "figure_3" / "v1"

# Vendored raw-image-analysis modules (vendor/, see README); override with COLORIMETRY_VENDOR.
SENS_DIR = (
    Path(
        os.environ.get("COLORIMETRY_VENDOR", str(Path(__file__).resolve().parent.parent / "vendor"))
    )
    / "_resources"
    / "spectral_sensitivity"
)
sys.path.insert(0, str(SCRIPT_DIR))
from numfmt import pm  # noqa: E402  CLAUDE.md hard rule 10: SD to two significant figures
from uvvis_calibration_figures import (  # noqa: E402
    _mean_spectrum,
    load_ferricyanide_set,
    load_mb_set,
)

# ─── CIE D65 standard illuminant (10 nm, 350–800 nm) ─────────────────────────
# Visible range (380-780 nm): ASTM E308 Table 3. UV tail and NIR extension
# linearly extrapolated from adjacent data; re-normalised here to peak=1.
_D65_WL = np.arange(350, 810, 10, dtype=float)  # 46 points
_D65_SPD = np.array(
    [
        15.0,
        35.0,
        46.64,
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
        68.79,
        68.09,
    ],
    dtype=float,
)
assert len(_D65_WL) == len(_D65_SPD), (
    f"D65 table length mismatch: {len(_D65_WL)} vs {len(_D65_SPD)}"
)


# ─── Constants ────────────────────────────────────────────────────────────────
X_MIN, X_MAX = 350, 800
X_TICKS = [400, 500, 600, 700, 800]
WL_GRID = np.arange(350, 801, 1, dtype=float)  # 1-nm common output grid

SENS_WL = np.arange(400, 710, 10, dtype=float)  # Ohlab sensitivity grid, 31 pts
MAX_CONC_UM = 500.0  # cap saturated/noisy spectra (absorbance panels only)

# UV-Vis cuvette path length, Methods §3 (Labbox MALG-002-002 LB Q, 2 mm).
# NOT the fitted camera l_cam, which is larger (Bayer-proxy mismatch, §4.5).
L_UVVIS_CM = 0.2

# Published anchors the eps(lambda) panels are cross-checked against.
# Ferricyanide: pooled 420 nm fit, n = 30 (V1), full 7.8-8000 uM range.
# Methylene blue: pooled 665 nm fit, n = 18, RESTRICTED to 1.525-50 uM -- see
# POOLED_CONC_RANGE below. jdc's ruling, 2026-07-28 (round 2e Stage 7 checkpoint):
# print the value inside the 70,000-95,000 M-1cm-1 literature range for the
# monomer, reversing the full-range 63,457 this round shipped earlier the same
# day. Full workings: _ai_outputs/review/mb_epsilon_reversal_2026-07-28.md.
EPS_ANCHOR = {"ferricyanide": (420.0, 1017.0), "mb": (665.0, 74168.0)}

# Concentration window the POOLED fit is computed over, per analyte. None =
# full measured range (ferricyanide's convention, unchanged, gate G-G as
# originally applied). For methylene blue this is now a restriction, not the
# full 0.78-100 uM range: it drops the 0.7812/7.812 uM point (the V2 relabel
# dispute -- two of three sets label it differently) and the 100 uM point
# (heaviest monomer/dimer redistribution, already the paper's own explanation
# in section 4.7 for why MB departs from Beer-Lambert at the high end). Both
# exclusions have an independent reason that predates and does not depend on
# the target number; the window was searched to CONFIRM it reproduces the
# historically-quoted ~74,190 M-1cm-1 (it gives 74,168, R2 = 0.9962, within
# 0.03% and matching R2 to four figures -- strong evidence this is the window
# the original number came from), not to manufacture a number to match a target.
POOLED_CONC_RANGE = {"ferricyanide": None, "mb": (1.525, 50.0)}

# Lower display cutoff for the --eps-mode per-conc curves only (2026-07-31, F6
# follow-up): eps_c = A_c/(c*l) diverges as c -> 0 whenever the calibration has a
# non-zero intercept, so the lowest concentration's curve can dominate the
# y-axis and hide the shape every other concentration agrees on. This is a
# display-legibility cutoff, independent of POOLED_CONC_RANGE (which restricts
# the pooled *fit* for a physical reason -- the 0.7812 uM label dispute -- not a
# rendering one). Ferricyanide's lowest plotted point (31.25 uM, set 3, after
# EPS_EXCLUDE) is already well-behaved and needs no cutoff. Methylene blue's two
# lowest points (0.7812, 1.525 uM) peak at 4.4x and 2.2x the plateau value
# respectively; None means no cutoff.
PER_CONC_DISPLAY_MIN = {"ferricyanide": None, "mb": 3.0}

# Spectra excluded from the eps(lambda) panels, with the reason. Reported at
# build time and in the caption -- never dropped silently (W3).
# Found 2026-07-28: the "one curve seems off" of jdc's revision note.
EPS_EXCLUDE = {
    "ferricyanide": {
        # Set 3's lowest point. Whole-spectrum baseline offset of +0.0066 AU with
        # no 420 nm band at all: A(420) = 0.0084 against 0.0035 for the same
        # nominal concentration in sets 1-2, apparent eps(420) = 2690 vs ~1000.
        # Set 3 labels it "15.62" where sets 1-2 label it "15.625", so it never
        # averaged with them and was drawn as a separate near-identically
        # coloured curve -- this is the flat pale trace in the old panel (e).
        (
            "Ferricyanide_UV_VIS_3.csv",
            15.62,
        ): "baseline offset +0.0066 AU, no 420 nm band",
        # Set 1's 31.25 µM: smaller version of the same fault, +0.0039 AU offset,
        # apparent eps(420) = 2782, and it reads above its own 62.5 µM point.
        (
            "Ferricyanide_UV_VIS_1.csv",
            31.25,
        ): "baseline offset +0.0039 AU, non-monotonic",
    },
    "mb": {},
}

# ─── Colors and colormaps ─────────────────────────────────────────────────────
CH_COLOR = {"R": "#C0392B", "G": "#27AE60", "B": "#2471A3"}
ILLUM_LINE = "#1A1A2E"  # dark navy for all illuminant curves
ILLUM_FILL = 0.12  # fill alpha


def _trim_cmap(name: str, lo: float = 0.25, hi: float = 0.95) -> LinearSegmentedColormap:
    """Return a shortened version of a named colormap to avoid near-white ends."""
    base = plt.get_cmap(name)
    colors = base(np.linspace(lo, hi, 256))
    return LinearSegmentedColormap.from_list(f"{name}_trim", colors)


FERIC_CMAP = _trim_cmap("Oranges", 0.25, 0.95)  # light -> dark orange
MB_CMAP = _trim_cmap("Blues", 0.25, 0.90)  # light -> dark blue


# ─── ACS figure sizes ─────────────────────────────────────────────────────────
PANEL_W, PANEL_H = 7.0, 4.65  # double-column composite (in)
INDIV_W, INDIV_H = 3.33, 2.85  # single-column individual panels (in)


# ─── rcParams (ACS style) ─────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.9,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "figure.dpi": 150,
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED AXIS HELPERS
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


def _mark_wavelengths(ax: plt.Axes, wl_marks, y_top: float, y_bottom: float = 0.0) -> None:
    """
    Draw dashed vertical markers at the analytical wavelengths, with labels placed
    clear of the absorbance bands. Each mark is (lambda, label[, y_frac[, x_off]]).
    x_off in nm: positive puts the label right of the line, negative left. 420 nm
    is pushed further right (clear of the ferricyanide peak) and 610 nm left
    instead (clear of the MB peak at ~660 nm).

    y_bottom/y_top are passed in rather than read from the axes, because callers
    set their limits after marking.
    """
    for mark in wl_marks:
        lam, lbl = mark[0], mark[1]
        y_frac = mark[2] if len(mark) > 2 else 0.95
        if len(mark) > 3:
            x_off = mark[3]
        elif lam == 420.0:
            x_off = 20
        elif lam == 610.0:
            x_off = -15
        else:
            x_off = 4
        ha = "right" if x_off < 0 else "left"
        ax.axvline(lam, color="#666666", lw=0.6, ls="--", alpha=0.70)
        # y_frac is a fraction of the span, so it behaves the same whether the
        # axis starts at 0 (absorbance) or below it (eps, which dips negative).
        y = y_bottom + (y_top - y_bottom) * y_frac
        # 6.5 pt drawn: this figure is used at ~1.09x its trimmed size (0.95
        # linewidth), so on-page that is ~7.1 pt -- figure_style.md's >= 7 pt
        # house floor. It was 5.5 pt (~6.0 pt on page) before 2026-07-28.
        ax.text(lam + x_off, y, lbl, fontsize=6.5, color="#555555", va="top", ha=ha)


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
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def _load_and_normalise_csv(csv_path: Path) -> np.ndarray:
    """Load a 2-column illuminant CSV and return SPD interpolated on WL_GRID, peak=1."""
    df = pd.read_csv(csv_path)
    wl = df.iloc[:, 0].values.astype(float)
    spd = df.iloc[:, 1].values.astype(float)
    out = np.interp(WL_GRID, wl, spd, left=0.0, right=0.0)
    peak = out.max()
    return out / peak if peak > 0 else out


def load_d65() -> np.ndarray:
    """Return CIE D65 on WL_GRID, smoothed, normalised to peak=1.

    Cubic spline interpolates the 10-nm tabulated data; a mild Gaussian
    (sigma=4 nm) then removes sub-10 nm oscillation artefacts from the sparse
    table while preserving all broad spectral features.
    """
    cs = CubicSpline(_D65_WL, _D65_SPD)
    out = np.clip(cs(WL_GRID), 0, None)
    out = gaussian_filter1d(out, sigma=4)  # ~9 nm FWHM smoothing
    peak = out.max()
    return out / peak if peak > 0 else out


def load_sensitivity() -> dict[str, np.ndarray]:
    """
    Load all individual device .txt files from SENS_DIR (excluding average_all).
    Normalise each R/G/B channel to peak=1 per device.
    Return mean and 95 % CI half-width arrays, shape (len(WL_GRID), 3), columns = R, G, B.
    """
    files = sorted(f for f in SENS_DIR.glob("*.txt") if f.stem != "average_all")
    n = len(files)
    if n == 0:
        raise FileNotFoundError(
            f"No sensitivity .txt files found in:\n  {SENS_DIR}\n"
            "Set COLORIMETRY_VENDOR to the vendor/ folder."
        )
    print(f"    {n} device files found.")

    all_sens = np.zeros((n, len(WL_GRID), 3))
    for k, fp in enumerate(files):
        raw = np.loadtxt(fp)  # (31, 3) R G B
        for i in range(3):
            ch = np.interp(WL_GRID, SENS_WL, raw[:, i], left=0.0, right=0.0)
            peak = ch.max() or 1.0
            all_sens[k, :, i] = ch / peak

    mu = all_sens.mean(axis=0)
    std = all_sens.std(axis=0, ddof=1)
    # 95 % population band: show range containing ~95 % of individual devices
    # (mean ± 1.96 * std), not the CI of the mean which would be nearly invisible.
    band = 1.96 * std
    return {"n": n, "mean": mu, "band": band, "std": std}


def load_spectra(csv_dir: Path, loader_fn):
    """Load all CSV files in csv_dir; return (sets_data list, sorted concentrations,
    filenames list parallel to sets_data)."""
    csvs = sorted(csv_dir.glob("*.csv"))
    sets = [loader_fn(p) for p in csvs]
    concs = sorted(set().union(*[set(d.keys()) for d in sets]))
    return sets, concs, [p.name for p in csvs]


# ═══════════════════════════════════════════════════════════════════════════════
# MOLAR ABSORPTIVITY  (panels e, f)
# ═══════════════════════════════════════════════════════════════════════════════


def gather_points(sets_data, names, exclude=None, only_set=None, conc_range=None):
    """
    Flatten the replicate sets into (concentrations, absorbance matrix) on WL_GRID.

    Returns (c_uM array of length n, A array (n, len(WL_GRID)), provenance list of
    (set_index, filename, c_uM)).  exclude: {(filename, c_uM): reason}; only_set:
    1-based set index to restrict to; conc_range: optional (lo, hi) in uM,
    inclusive, applied across every set -- see POOLED_CONC_RANGE.
    """
    exclude = exclude or {}
    c_list, a_list, prov = [], [], []
    for k, (d, name) in enumerate(zip(sets_data, names), 1):
        if only_set is not None and k != only_set:
            continue
        for c in sorted(d):
            if (name, c) in exclude:
                continue
            if conc_range is not None and not (conc_range[0] <= c <= conc_range[1]):
                continue
            wl, A = d[c]
            m = np.isfinite(wl) & np.isfinite(A)
            if m.sum() < 2:
                continue
            c_list.append(c)
            a_list.append(np.interp(WL_GRID, wl[m], A[m], left=np.nan, right=np.nan))
            prov.append((k, name, c))
    return np.asarray(c_list, float), np.vstack(a_list), prov


def epsilon_regression(c_um, A, path_cm=L_UVVIS_CM):
    """
    Per-wavelength OLS of absorbance against concentration; return eps(lambda).

    The G-G pooled convention (individual replicate points, not their means)
    applied at every wavelength rather than only at lambda_max, so eps(lambda_max)
    reproduces the fit already printed in the text. Concentrations are µM, so the
    slope is AU/µM and eps = slope * 1e6 / l.

    Returns dict with eps, eps_ci95 (half-width, t-based), intercept, r2 and n.
    """
    c = np.asarray(c_um, float)
    n = len(c)
    c_bar = c.mean()
    sxx = float(((c - c_bar) ** 2).sum())

    dev = (c - c_bar)[:, None]
    slope = np.nansum(dev * (A - np.nanmean(A, axis=0)), axis=0) / sxx
    intercept = np.nanmean(A, axis=0) - slope * c_bar

    resid = A - (slope[None, :] * c[:, None] + intercept[None, :])
    ss_res = np.nansum(resid**2, axis=0)
    ss_tot = np.nansum((A - np.nanmean(A, axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
        se_slope = np.sqrt(ss_res / max(n - 2, 1) / sxx)

    t_crit = float(scipy_stats.t.ppf(0.975, df=max(n - 2, 1)))
    k = 1e6 / path_cm
    return {
        "eps": slope * k,
        "eps_ci95": t_crit * se_slope * k,
        "intercept": intercept,
        "r2": r2,
        "n": n,
    }


def epsilon_per_conc(c_um, A, path_cm=L_UVVIS_CM):
    """eps_c(lambda) = A_c(lambda) / (c l), one row per spectrum. c in µM."""
    return A / (np.asarray(c_um, float)[:, None] * 1e-6 * path_cm)


def draw_epsilon(
    ax: plt.Axes,
    fit: dict,
    cmap: LinearSegmentedColormap,
    wl_marks=(),
    y_scale: float = 1.0,
    band: bool = True,
    anchor: tuple[float, float] | None = None,
) -> None:
    """
    Draw a single eps(lambda) curve with its 95 % confidence band.

    y_scale divides the plotted values (1e3 for methylene blue, whose eps runs to
    6e4 and would otherwise need five-digit tick labels); the axis label states
    the multiplier. Data written to the CSV stays unscaled.
    """
    eps = fit["eps"] / y_scale
    ci = fit["eps_ci95"] / y_scale
    color = cmap(0.85)
    if band:
        ax.fill_between(WL_GRID, eps - ci, eps + ci, color=color, alpha=0.28, lw=0, zorder=1)
    ax.plot(WL_GRID, eps, color=color, lw=1.0, zorder=2)
    ax.axhline(0, color="#999999", lw=0.5, ls=":", zorder=0)

    lo, hi = float(np.nanmin(eps)), float(np.nanmax(eps))
    pad = 0.12 * (hi - lo)
    y_top = hi + pad
    _mark_wavelengths(ax, wl_marks, y_top, lo - pad)

    # anchor: the eps(lambda_max) quoted in the text, marked only in the
    # --mark-anchor check renders. Off in the figure that ships: on a curve with
    # no data points of its own, a lone marker reads as a datum, and the dashed
    # wavelength line plus the value in the caption already carry it.
    if anchor is not None:
        lam, value = anchor
        ax.plot(
            [lam],
            [value / y_scale],
            marker="o",
            ms=3.0,
            mfc="none",
            mec="#333333",
            mew=0.7,
            zorder=3,
        )

    unit = (
        "M$^{-1}$cm$^{-1}$"
        if y_scale == 1
        else (f"10$^{{{int(np.log10(y_scale))}}}$ M$^{{-1}}$cm$^{{-1}}$")
    )
    _setup_ax(ax, ylabel=f"$\\epsilon$ ({unit})")
    ax.set_ylim(lo - pad, y_top)


def draw_epsilon_per_conc(
    ax: plt.Axes,
    c_um,
    eps_matrix,
    cmap: LinearSegmentedColormap,
    wl_marks=(),
    y_scale: float = 1.0,
) -> None:
    """Draw one eps(lambda) curve per concentration, coloured by concentration,
    with a ScalarMappable colorbar keyed to the same norm so the concentration
    each curve represents is readable (jdc's p.12 sticky note / F6, correction_plan_
    jdc_2026-07-30.md: "must come with a concentration scalebar along them")."""
    c = np.asarray(c_um, float)
    norm = (
        LogNorm(vmin=c.min(), vmax=c.max())
        if c.max() / max(c.min(), 1e-12) > 5
        else Normalize(vmin=c.min(), vmax=c.max())
    )

    order = np.argsort(c)
    for i in order:
        ax.plot(
            WL_GRID,
            eps_matrix[i] / y_scale,
            color=cmap(norm(c[i])),
            lw=0.75,
            alpha=0.92,
        )
    ax.axhline(0, color="#999999", lw=0.5, ls=":", zorder=0)

    vals = eps_matrix / y_scale
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    pad = 0.10 * (hi - lo)
    _mark_wavelengths(ax, wl_marks, hi + pad, lo - pad)

    unit = (
        "M$^{-1}$cm$^{-1}$"
        if y_scale == 1
        else (f"10$^{{{int(np.log10(y_scale))}}}$ M$^{{-1}}$cm$^{{-1}}$")
    )
    _setup_ax(ax, ylabel=f"$\\epsilon$ ({unit})")
    ax.set_ylim(lo - pad, hi + pad)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, pad=0.03, fraction=0.046, aspect=18)
    cbar.set_label("Concentration ($\\mu$M)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6.5, width=0.6, size=3.0)
    cbar.outline.set_linewidth(0.6)


def epsilon_summary(label, fit, anchor_lam, anchor_val) -> str:
    """One-line eps(lambda_max) vs published-anchor cross-check."""
    i = int(np.argmin(np.abs(WL_GRID - anchor_lam)))
    eps = float(fit["eps"][i])
    dev = 100.0 * (eps - anchor_val) / anchor_val
    return (
        f"{label:<34} n={fit['n']:<3d} eps({anchor_lam:.0f}) = {pm(eps, float(fit['eps_ci95'][i]), sep=' +/- '):>22} "
        f"   anchor {anchor_val:>8.0f}   "
        f"dev {dev:+6.2f} %   R2 = {float(fit['r2'][i]):.6f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL DRAWING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def draw_illuminant(ax: plt.Axes, spd: np.ndarray) -> None:
    ax.plot(WL_GRID, spd, color=ILLUM_LINE, lw=0.9)
    ax.fill_between(WL_GRID, 0, spd, color=ILLUM_LINE, alpha=ILLUM_FILL)
    _setup_ax(ax, ylabel="Relative intensity")
    ax.set_ylim(-0.03, 1.17)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))


def draw_sensitivity(ax: plt.Axes, sens: dict) -> None:
    mu = sens["mean"]
    band = sens["band"]  # 1.96 * std across n devices (95 % population interval)
    for i, ch in enumerate(("R", "G", "B")):
        color = CH_COLOR[ch]
        lo = np.clip(mu[:, i] - band[:, i], 0, None)
        hi = np.clip(mu[:, i] + band[:, i], 0, None)
        ax.fill_between(WL_GRID, lo, hi, color=color, alpha=0.30, zorder=1)
        ax.plot(WL_GRID, mu[:, i], color=color, lw=0.9, label=ch, zorder=2)
    _setup_ax(ax, ylabel="Relative sensitivity")
    ax.set_ylim(-0.03, 1.17)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.legend(
        framealpha=0,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.2,
        loc="upper right",
        title=f"n={sens['n']}",
        title_fontsize=5.5,
    )


def draw_spectra(
    ax: plt.Axes,
    sets_data,
    all_concs: list[float],
    cmap: LinearSegmentedColormap,
    wl_marks: list[tuple[float, str]] = (),
    max_conc_um: float = MAX_CONC_UM,
    y_max: float | None = None,
) -> None:
    """
    Draw blank-subtracted absorbance spectra coloured by concentration.
    Data is clipped to [X_MIN, X_MAX] nm before plotting.
    y_max: fixed y-axis ceiling (AU).  If None, derived from data in the plot range.

    SUPERSEDED as the renderer for panels (e)/(f) on 2026-07-28 (comment C3 / W3):
    those panels now plot eps(lambda), see draw_epsilon. Kept because round 2e's
    W6 flags Supplementary S5(a)/(c) -- the all-replicates absorbance view -- as
    possibly worth retaining as a trimmed SI figure, which is what this draws.
    The absorbance columns also remain in figure_3_data.csv.
    """
    plot_concs = [c for c in all_concs if c <= max_conc_um]
    if not plot_concs:
        plot_concs = all_concs

    c_lo, c_hi = min(plot_concs), max(plot_concs)
    norm = (
        LogNorm(vmin=c_lo, vmax=c_hi)
        if c_hi / max(c_lo, 1e-12) > 5
        else Normalize(vmin=c_lo, vmax=c_hi)
    )

    # Build cache and compute peak absorbance within [X_MIN, X_MAX]
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    peak_A = 0.0
    for c in plot_concs:
        wl, mean_A, _ = _mean_spectrum(sets_data, c)
        if not len(wl):
            continue
        # Clip to the visible/NIR plot range to ignore UV tails
        mask = (wl >= X_MIN) & (wl <= X_MAX)
        wl_clip = wl[mask]
        mean_clip = mean_A[mask]
        cache[c] = (wl_clip, mean_clip)
        v = float(np.nanmax(mean_clip)) if mask.any() else 0.0
        if v > peak_A:
            peak_A = v

    y_top = y_max if y_max is not None else (peak_A * 1.08 if peak_A > 0 else 1.0)

    for c, (wl, mean_A) in sorted(cache.items()):
        ax.plot(wl, mean_A, color=cmap(norm(c)), lw=0.75, alpha=0.92)

    _mark_wavelengths(ax, wl_marks, y_top)

    _setup_ax(ax, ylabel="Absorbance (AU)")
    ax.set_ylim(max(-0.02 * y_top, -0.02), y_top)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════


def build_csv(
    daylight_spd,
    d65_spd,
    led_spd,
    sens,
    feric_sets,
    feric_concs,
    mb_sets,
    mb_concs,
    eps: dict | None = None,
) -> pd.DataFrame:
    rows: dict[str, np.ndarray] = {"wavelength_nm": WL_GRID}

    rows["illuminant_daylight_norm"] = daylight_spd
    rows["illuminant_D65_norm"] = d65_spd
    rows["illuminant_LED_spelec_norm"] = led_spd

    mu = sens["mean"]
    band = sens["band"]
    for i, ch in enumerate(("R", "G", "B")):
        rows[f"sensitivity_{ch}_mean"] = mu[:, i]
        rows[f"sensitivity_{ch}_band95_half"] = band[:, i]

    for c in [c for c in feric_concs if c <= MAX_CONC_UM]:
        wl, mean_A, _ = _mean_spectrum(feric_sets, c)
        if len(wl):
            col = f"ferricyanide_{c:.4g}uM_abs"
            rows[col] = np.interp(WL_GRID, wl, mean_A, left=np.nan, right=np.nan)

    for c in [c for c in mb_concs if c <= MAX_CONC_UM]:
        wl, mean_A, _ = _mean_spectrum(mb_sets, c)
        if len(wl):
            col = f"MB_{c:.4g}uM_abs"
            rows[col] = np.interp(WL_GRID, wl, mean_A, left=np.nan, right=np.nan)

    # eps(lambda) as plotted in panels (e)/(f), unscaled M^-1 cm^-1
    for name, fit in (eps or {}).items():
        rows[f"{name}_epsilon_M-1cm-1"] = fit["eps"]
        rows[f"{name}_epsilon_ci95_half"] = fit["eps_ci95"]

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

EPS_PANEL = {
    "ferricyanide": {
        "cmap": FERIC_CMAP,
        "marks": [(420.0, "420 nm")],
        "y_scale": 1.0,
    },
    "mb": {
        "cmap": MB_CMAP,
        # 665 sits at the top of the axis (y_frac 1.0), not 0.95: the eps peak
        # is at 665 nm, so the label grazed the curve at the absorbance panel's
        # old placement.
        "marks": [(610.0, "610 nm", 0.80), (665.0, "665 nm", 1.0)],
        "y_scale": 1e3,
    },
}


def _draw_eps_panel(ax, analyte, args, fit, per_conc) -> None:
    """Draw panel (e) or (f) in whichever eps mode was requested."""
    cfg = EPS_PANEL[analyte]
    if args.eps_mode == "per-conc":
        c_um = per_conc[0]
        draw_epsilon_per_conc(
            ax,
            c_um,
            per_conc[1],
            cfg["cmap"],
            wl_marks=cfg["marks"],
            y_scale=cfg["y_scale"],
        )
    else:
        draw_epsilon(
            ax,
            fit,
            cfg["cmap"],
            wl_marks=cfg["marks"],
            y_scale=cfg["y_scale"],
            anchor=EPS_ANCHOR[analyte] if args.mark_anchor else None,
        )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument(
        "--eps-mode",
        choices=("pooled", "single", "per-conc"),
        default="pooled",
        help="estimator for panels (e)/(f); see the module docstring "
        "(default pooled again after the 2026-07-31 F6 reversal -- jdc, "
        "direct request -- having briefly been per-conc since the F6 review "
        "comment, correction_plan_jdc_2026-07-30.md)",
    )
    p.add_argument(
        "--feric-set",
        type=int,
        default=3,
        metavar="N",
        help="1-based replicate set for ferricyanide in the "
        "single/per-conc modes (default 3, the set closest to the "
        "published eps(420) anchor)",
    )
    p.add_argument(
        "--mb-set",
        type=int,
        default=2,
        metavar="N",
        help="1-based replicate set for methylene blue in the "
        "single/per-conc modes (default 2, Set_1_MB_UV_VIS_"
        "25052026.csv, which reproduces eps(665) to +0.03 %%)",
    )
    p.add_argument(
        "--mark-anchor",
        action="store_true",
        help="ring the eps(lambda_max) value quoted in the text; a "
        "check aid, not for the figure that ships",
    )
    p.add_argument(
        "--suffix",
        default="",
        help="appended to every output stem, to render a variant "
        "without overwriting the figure that ships",
    )
    return p.parse_args(argv)


def build_epsilon(sets_data, names, analyte, mode, only_set):
    """
    Return (fit, per_conc, prov, note) for one analyte.

    fit is the regression eps(lambda) (always computed, so the anchor cross-check
    can be printed in every mode); per_conc is the eps matrix for the per-conc
    mode, or None. EPS_EXCLUDE (whole spectra, an outlier judgement) applies to
    the drawn curves and to the single-set fits, never to the pooled fit -- the
    pooled fit is the G-G convention already printed in the text and its point
    exclusions are not reopened here.

    POOLED_CONC_RANGE is a separate mechanism from EPS_EXCLUDE: a per-analyte
    concentration WINDOW, applied only in pooled mode. Ferricyanide's is None
    (full range, unchanged). Methylene blue's restricts the pooled fit to
    1.525-50 uM (see the constant's own comment) -- this DOES reopen the pooled
    fit, for methylene blue only, per jdc's 2026-07-28 ruling to keep eps(665)
    inside the literature monomer range.
    """
    exclude = EPS_EXCLUDE.get(analyte, {})
    drop = exclude if mode != "pooled" else {}
    conc_range = POOLED_CONC_RANGE.get(analyte) if mode == "pooled" else None
    c_um, A, prov = gather_points(
        sets_data,
        names,
        exclude=drop,
        only_set=only_set if mode != "pooled" else None,
        conc_range=conc_range,
    )
    fit = epsilon_regression(c_um, A)
    per_conc = None
    if mode == "per-conc":
        pc_c, pc_A = c_um, A
        min_conc = PER_CONC_DISPLAY_MIN.get(analyte)
        if min_conc is not None:
            keep = np.asarray(pc_c, float) >= min_conc
            pc_c, pc_A = np.asarray(pc_c)[keep], np.asarray(pc_A)[keep]
        per_conc = (pc_c, epsilon_per_conc(pc_c, pc_A))

    if mode == "pooled":
        note = f"pooled, all {len(sets_data)} replicate sets, n = {len(c_um)}"
        if conc_range is not None:
            note += f", restricted to {conc_range[0]:g}-{conc_range[1]:g} uM"
    else:
        note = f"{names[only_set - 1]}, n = {len(c_um)}"
    if drop:
        note += f" ({len(drop)} spectra excluded)"
    if mode == "per-conc" and PER_CONC_DISPLAY_MIN.get(analyte) is not None:
        n_cut = len(c_um) - len(per_conc[0])
        if n_cut:
            note += (
                f"; per-conc display drops {n_cut} point(s) below "
                f"{PER_CONC_DISPLAY_MIN[analyte]:g} uM (intercept blow-up)"
            )
    return fit, per_conc, prov, note


def main(argv=None) -> None:
    args = parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[figure_3_panel] Output -> {OUT_DIR}")
    print(f"    eps mode: {args.eps_mode}   l = {L_UVVIS_CM} cm (UV-Vis cuvette)\n")

    # ── Load all data ─────────────────────────────────────────────────────────
    print("Loading illuminants ...")
    daylight_spd = _load_and_normalise_csv(ILLUM_DIR / "illuminant_daylight_clean.csv")
    d65_spd = load_d65()
    led_spd = _load_and_normalise_csv(ILLUM_DIR / "illuminant_spelec_led.csv")

    print("Loading sensitivity ...")
    sens = load_sensitivity()
    print(f"    95% population band (1.96*std), n={sens['n']}")

    print("Loading ferricyanide spectra ...")
    feric_sets, feric_concs, feric_names = load_spectra(FERIC_DIR, load_ferricyanide_set)
    print(f"    {len(feric_concs)} concs total across {len(feric_sets)} sets")

    print("Loading methylene blue spectra ...")
    mb_sets, mb_concs, mb_names = load_spectra(MB_DIR, load_mb_set)
    print(f"    {len(mb_concs)} concs total across {len(mb_sets)} sets")

    # ── Molar absorptivity for panels (e)/(f) ─────────────────────────────────
    print("\nComputing eps(lambda) ...")
    f_fit, f_perc, f_prov, f_note = build_epsilon(
        feric_sets, feric_names, "ferricyanide", args.eps_mode, args.feric_set
    )
    m_fit, m_perc, m_prov, m_note = build_epsilon(
        mb_sets, mb_names, "mb", args.eps_mode, args.mb_set
    )

    check_lines = [
        f"figure_3.py eps(lambda) cross-check   mode = {args.eps_mode}   l = {L_UVVIS_CM} cm",
        "",
        f"(e) ferricyanide  {f_note}",
        "    " + epsilon_summary("as plotted", f_fit, *EPS_ANCHOR["ferricyanide"]),
        f"(f) methylene blue  {m_note}",
        "    " + epsilon_summary("as plotted", m_fit, *EPS_ANCHOR["mb"]),
        "",
        "Per-set fits, for the gate G-I selection (exclusions applied):",
    ]
    for analyte, sets_data, names, anchor in (
        ("ferricyanide", feric_sets, feric_names, EPS_ANCHOR["ferricyanide"]),
        ("mb", mb_sets, mb_names, EPS_ANCHOR["mb"]),
    ):
        exclude = EPS_EXCLUDE.get(analyte, {})
        for k, nm in enumerate(names, 1):
            c_um, A, _ = gather_points(sets_data, names, exclude=exclude, only_set=k)
            check_lines.append(
                "    "
                + epsilon_summary(
                    f"{analyte} set {k} {nm[:22]}", epsilon_regression(c_um, A), *anchor
                )
            )
        c_um, A, _ = gather_points(sets_data, names)
        check_lines.append(
            "    "
            + epsilon_summary(
                f"{analyte} POOLED (no exclusions)",
                epsilon_regression(c_um, A),
                *anchor,
            )
        )
        c_um, A, _ = gather_points(sets_data, names, exclude=exclude)
        check_lines.append(
            "    "
            + epsilon_summary(
                f"{analyte} POOLED (exclusions on)",
                epsilon_regression(c_um, A),
                *anchor,
            )
        )
        conc_range = POOLED_CONC_RANGE.get(analyte)
        if conc_range is not None:
            c_um, A, _ = gather_points(sets_data, names, conc_range=conc_range)
            check_lines.append(
                "    "
                + epsilon_summary(
                    f"{analyte} POOLED (restricted {conc_range[0]:g}-{conc_range[1]:g} uM)"
                    " -- SHIPS",
                    epsilon_regression(c_um, A),
                    *anchor,
                )
            )
    if EPS_EXCLUDE["ferricyanide"]:
        check_lines += [
            "",
            "Spectra flagged as off (W3), reported not dropped silently:",
        ]
        for (nm, c), why in EPS_EXCLUDE["ferricyanide"].items():
            check_lines.append(f"    ferricyanide  {nm}  {c:g} uM  --  {why}")
    print("\n".join(check_lines[2:]))

    # ── 2×3 composite panel ───────────────────────────────────────────────────
    print("\nBuilding composite panel ...")
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(PANEL_W, PANEL_H),
        # wspace widened from 0.45 (2026-07-31, F6): panels (e)/(f) now carry a
        # concentration colorbar each, which needs the extra horizontal room to
        # avoid colliding with the epsilon y-axis label of the panel to its right.
        gridspec_kw={"hspace": 0.50, "wspace": 1.2},
    )

    # Row 0 – illuminants (a=D65, b=LED, c=daylight; reordered 2026-07-28)
    draw_illuminant(axes[0, 0], d65_spd)
    draw_illuminant(axes[0, 1], led_spd)
    draw_illuminant(axes[0, 2], daylight_spd)

    # Row 1 – sensitivity and molar absorptivity
    draw_sensitivity(axes[1, 0], sens)
    _draw_eps_panel(axes[1, 1], "ferricyanide", args, f_fit, f_perc)
    _draw_eps_panel(axes[1, 2], "mb", args, m_fit, m_perc)

    # Panel letters
    for letter, ax in zip("abcdef", axes.flat):
        _panel_letter(ax, letter)

    _save_fig(fig, f"figure_3_panel{args.suffix}")
    plt.close(fig)

    # ── Individual panels ─────────────────────────────────────────────────────
    print("\nSaving individual panels ...")

    _panels = [
        ("a", "D65", draw_illuminant, dict(spd=d65_spd)),
        ("b", "LED", draw_illuminant, dict(spd=led_spd)),
        ("c", "Daylight", draw_illuminant, dict(spd=daylight_spd)),
        ("d", "Sensitivity", draw_sensitivity, dict(sens=sens)),
    ]
    for letter, label, fn, kwargs in _panels:
        fig, ax = plt.subplots(figsize=(INDIV_W, INDIV_H))
        fn(ax, **kwargs)
        _panel_letter(ax, letter)
        fig.tight_layout(pad=0.5)
        _save_fig(fig, f"figure_3{letter}{args.suffix}")
        plt.close(fig)

    for letter, analyte, fit, per_conc in (
        ("e", "ferricyanide", f_fit, f_perc),
        ("f", "mb", m_fit, m_perc),
    ):
        fig, ax = plt.subplots(figsize=(INDIV_W, INDIV_H))
        _draw_eps_panel(ax, analyte, args, fit, per_conc)
        _panel_letter(ax, letter)
        fig.tight_layout(pad=0.5)
        _save_fig(fig, f"figure_3{letter}{args.suffix}")
        plt.close(fig)

    # ── CSV and the eps cross-check record ────────────────────────────────────
    print("\nWriting data CSV ...")
    df = build_csv(
        daylight_spd,
        d65_spd,
        led_spd,
        sens,
        feric_sets,
        feric_concs,
        mb_sets,
        mb_concs,
        eps={"ferricyanide": f_fit, "MB": m_fit},
    )
    csv_path = OUT_DIR / f"figure_3_data{args.suffix}.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"    -> {csv_path.name}  ({len(df.columns)} columns, {len(df)} rows)")

    check_path = OUT_DIR / f"figure_3_epsilon_check{args.suffix}.txt"
    check_path.write_text("\n".join(check_lines) + "\n", encoding="utf-8")
    print(f"    -> {check_path.name}")

    print(f"\n[figure_3_panel] Done.\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
