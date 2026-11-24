"""
daylight_diffuser_figures.py

Re-render each panel from the daylight_diffuser analysis as a separate
publication-quality figure.  Follows figure_style.md exactly.

Run from the project root:
    python scripts/daylight_diffuser_figures.py

Outputs: Data/analysis/ferricyanide/daylight_diffuser/image_analysis/figures_separate/
"""

import os as _os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Vendored raw-image-analysis modules (vendor/, see README); override with COLORIMETRY_VENDOR.
_RA = pathlib.Path(
    _os.environ.get(
        "COLORIMETRY_VENDOR", str(pathlib.Path(__file__).resolve().parent.parent / "vendor")
    )
)
sys.path.insert(0, str(_RA))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from core.spectral import load_sensitivity  # Ohlab Bayer sensitivity
from forward_model import load_illuminant, load_uvvis  # reuse exact loaders

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT = pathlib.Path(__file__).resolve().parent.parent
ANALYSIS = PROJECT / "Data/analysis/ferricyanide/daylight_diffuser/image_analysis"
CAL_DIR = ANALYSIS / "s02_calibrate"
FM_DIR = ANALYSIS / "forward_model_daylight_diffuser_l0p2"  # corrected (0.2 cm path length)
ILLUM_DIR = PROJECT / "Data/spectra/illuminant"
UVVIS_DIR = PROJECT / "Data/spectra/ferricyanide"
OUT_DIR = ANALYSIS / "figures_separate"
OUT_DIR.mkdir(exist_ok=True)

# ── Style (figure_style.md) ──────────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.8,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.5,
        "figure.dpi": 150,
    }
)

W1 = 3.33  # single-column width (in) — ACS max 240 pt
H_CAL = 3.33  # roughly square
H_SPEC = 3.33  # roughly square

# Colors (figure_style.md)
C_B = "#2471A3"
C_G = "#27AE60"
C_R = "#C0392B"
C_BDIVG = "#e67e22"
C_FIT = "#2C3E50"
C_GRAY = "#626567"
C_FERIC = "#D4820A"

SET_COLORS = ["#2E86AB", "#A23B72", "#F18F01"]
SET_MARKERS = ["o", "s", "^"]
SET_LABELS = ["Set 2", "Set 3", "Set 4"]


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, stem, csv_df=None):
    for ext in ("png", "svg"):
        # No bbox_inches="tight": keep the exact figsize so width/height stay ACS-compliant.
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=600, format=ext)
    plt.close(fig)
    if csv_df is not None:
        csv_df.to_csv(OUT_DIR / f"{stem}.csv", index=False)
    print(f"  Saved: {stem}")


# ── Load data ────────────────────────────────────────────────────────────────

cal = pd.read_csv(CAL_DIR / "calibration_table.csv")
abs_t = pd.read_csv(CAL_DIR / "absorbance_table.csv")
fits = pd.read_csv(CAL_DIR / "fit_results.csv", index_col="channel")
fm = pd.read_csv(FM_DIR / "forward_model_table.csv")


# ── 1. Beer-Lambert — B channel ──────────────────────────────────────────────


def bl_channel(channel, color, label):
    ch_cal = cal[cal["channel"] == channel].sort_values("concentration_um")
    ch_abs = abs_t[abs_t["channel"] == channel]
    row = fits.loc[channel]

    c_fit = np.linspace(0, ch_cal["concentration_um"].max() * 1.05, 300)
    A_fit = row["slope"] * c_fit + row["intercept"]

    fig, ax = plt.subplots(figsize=(W1, H_CAL))

    # individual replicates
    for (set_name, grp), col_s, mrk, lbl_s in zip(
        ch_abs.groupby("set"), SET_COLORS, SET_MARKERS, SET_LABELS
    ):
        ax.scatter(
            grp["concentration_um"],
            grp["absorbance"],
            s=12,
            color=col_s,
            marker=mrk,
            alpha=0.60,
            zorder=2,
            label=lbl_s,
        )

    # mean ± SD
    ax.errorbar(
        ch_cal["concentration_um"],
        ch_cal["absorbance_mean"],
        yerr=ch_cal["absorbance_std"],
        fmt="o",
        color=color,
        markersize=4.5,
        capsize=2.5,
        elinewidth=0.8,
        zorder=3,
        label="Mean ± SD  (n = 3)",
    )

    # fit line
    ax.plot(
        c_fit,
        A_fit,
        color=C_FIT,
        lw=0.9,
        ls="--",
        zorder=4,
        label=rf"OLS fit  (R² = {row['r2']:.4f})",
    )

    ax.set_xlim(left=0)
    ax.set_xlabel(r"[FC] (µM)")
    ax.set_ylabel(f"Absorbance  {label}")
    ax.legend(framealpha=0.85, edgecolor="none")
    fig.tight_layout(pad=0.5)
    _style(ax)

    csv_df = ch_cal[["concentration_um", "absorbance_mean", "absorbance_std"]].copy()
    csv_df.columns = ["concentration_um", "absorbance_mean", "absorbance_sd"]
    _save(fig, f"BL_{channel}", csv_df)


bl_channel("B", C_B, "(A$_B$)")
bl_channel("BdivG", C_BDIVG, "(A$_{B/G}$)")


# ── 2. Beer-Lambert — log x-axis variants ───────────────────────────────────


def bl_channel_log(channel, color, label):
    ch_cal = cal[cal["concentration_um"] > 0][cal["channel"] == channel].sort_values(
        "concentration_um"
    )
    ch_abs = abs_t[abs_t["channel"] == channel]
    row = fits.loc[channel]

    c_fit = np.logspace(
        np.log10(ch_cal["concentration_um"].min() * 0.5),
        np.log10(ch_cal["concentration_um"].max() * 1.5),
        300,
    )
    A_fit = row["slope"] * c_fit + row["intercept"]

    fig, ax = plt.subplots(figsize=(W1, H_CAL))

    for (set_name, grp), col_s, mrk, lbl_s in zip(
        ch_abs[ch_abs["concentration_um"] > 0].groupby("set"),
        SET_COLORS,
        SET_MARKERS,
        SET_LABELS,
    ):
        ax.scatter(
            grp["concentration_um"],
            grp["absorbance"],
            s=12,
            color=col_s,
            marker=mrk,
            alpha=0.60,
            zorder=2,
            label=lbl_s,
        )

    ax.errorbar(
        ch_cal["concentration_um"],
        ch_cal["absorbance_mean"],
        yerr=ch_cal["absorbance_std"],
        fmt="o",
        color=color,
        markersize=4.5,
        capsize=2.5,
        elinewidth=0.8,
        zorder=3,
        label="Mean ± SD  (n = 3)",
    )
    ax.plot(
        c_fit,
        A_fit,
        color=C_FIT,
        lw=0.9,
        ls="--",
        zorder=4,
        label=rf"OLS fit  (R² = {row['r2']:.4f})",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"[FC] (µM)")
    ax.set_ylabel(f"Absorbance  {label}")
    ax.legend(framealpha=0.85, edgecolor="none")
    fig.tight_layout(pad=0.5)
    _style(ax)

    csv_df = ch_cal[["concentration_um", "absorbance_mean", "absorbance_std"]].copy()
    csv_df.columns = ["concentration_um", "absorbance_mean", "absorbance_sd"]
    _save(fig, f"BL_{channel}_log", csv_df)


bl_channel_log("B", C_B, "(A$_B$)")
bl_channel_log("BdivG", C_BDIVG, "(A$_{B/G}$)")


# ── 3. Forward model calibration — model curve vs empirical ─────────────────


def fm_calibration(channel, fm_col, color, label):
    ch_col = channel if channel in cal["channel"].unique() else channel + "1"
    ch_cal = cal[cal["channel"] == ch_col].sort_values("concentration_um")

    fig, ax = plt.subplots(figsize=(W1, H_CAL))

    ax.errorbar(
        ch_cal["concentration_um"],
        ch_cal["absorbance_mean"],
        yerr=ch_cal["absorbance_std"],
        fmt="o",
        color=color,
        markersize=4.5,
        capsize=2.5,
        elinewidth=0.8,
        zorder=3,
        label="Empirical (mean ± SD, n = 3)",
    )

    fm_um = fm["concentration_um"].values
    fm_A = fm[fm_col].values
    valid = np.isfinite(fm_A)
    ax.plot(
        fm_um[valid],
        fm_A[valid],
        color=color,
        lw=0.9,
        ls="-",
        alpha=0.85,
        zorder=4,
        label="Forward model  (l$_{cam}$ = 5.03 mm)",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"[FC] (µM)")
    ax.set_ylabel(f"Absorbance  {label}")
    ax.legend(framealpha=0.85, edgecolor="none")
    fig.tight_layout(pad=0.5)
    _style(ax)

    csv_df = pd.DataFrame(
        {
            "concentration_um": ch_cal["concentration_um"],
            "absorbance_mean": ch_cal["absorbance_mean"],
            "absorbance_sd": ch_cal["absorbance_std"],
        }
    )
    _save(fig, f"FM_calibration_{channel}", csv_df)


fm_calibration("B", "A_B", C_B, "(A$_B$)")
fm_calibration("BdivG", "A_BdivG", C_BDIVG, "(A$_{B/G}$)")


# ── 4. Forward model linearisation ──────────────────────────────────────────

LINEAR_MAX_UM = 500.0


def fm_linearisation(channel, fm_col, color, label):
    ch_col = channel if channel in cal["channel"].unique() else channel + "1"
    ch_cal = cal[cal["channel"] == ch_col].sort_values("concentration_um")
    c_true = ch_cal["concentration_um"].to_numpy(dtype=float)
    A_meas = ch_cal["absorbance_mean"].to_numpy(dtype=float)
    A_std = ch_cal["absorbance_std"].to_numpy(dtype=float)

    lin_mask = c_true <= LINEAR_MAX_UM
    c_lin_M = c_true[lin_mask] * 1e-6
    A_lin = A_meas[lin_mask]
    slope_lin = float(np.dot(c_lin_M, A_lin) / np.dot(c_lin_M, c_lin_M))

    fm_um = fm["concentration_um"].values
    fm_A = np.maximum.accumulate(fm[fm_col].values.copy())
    c_corr_um = np.interp(A_meas, fm_A, fm_um)
    A_corr = slope_lin * c_corr_um * 1e-6

    c_ref = np.array([0.0, c_true.max() * 1.08])

    fig, ax = plt.subplots(figsize=(W1, H_CAL))

    ax.plot(
        c_ref,
        slope_lin * c_ref * 1e-6,
        color="#aaa",
        lw=1.2,
        ls="--",
        zorder=0,
        label="Beer-Lambert  (low-c slope)",
    )
    ax.errorbar(
        c_true,
        A_meas,
        yerr=A_std,
        fmt="o",
        color=C_GRAY,
        markersize=4.5,
        capsize=2.5,
        elinewidth=0.8,
        zorder=2,
        label="Original  (measured A)",
    )
    ax.errorbar(
        c_true,
        A_corr,
        yerr=A_std,
        fmt="o",
        color=color,
        markersize=4.5,
        capsize=2.5,
        elinewidth=0.8,
        zorder=3,
        label="Corrected  (model inversion)",
    )

    ax.set_xlabel(r"[FC] (µM)")
    ax.set_ylabel(f"Absorbance  {label}")
    ax.legend(framealpha=0.85, edgecolor="none")
    fig.tight_layout(pad=0.5)
    _style(ax)

    csv_df = pd.DataFrame(
        {
            "concentration_um": c_true,
            "A_original": A_meas,
            "A_corrected": A_corr,
            "A_sd": A_std,
        }
    )
    _save(fig, f"FM_linearisation_{channel}", csv_df)


fm_linearisation("B", "A_B", C_B, "(A$_B$)")
fm_linearisation("BdivG", "A_BdivG", C_BDIVG, "(A$_{B/G}$)")


# ── 5. Spectral figures ──────────────────────────────────────────────────────

LAMBDA_MAX = 420.0

# Load spectral data
phi_wl, phi = load_illuminant(ILLUM_DIR / "illuminant_daylight_diffuser.csv")
phi = phi / phi.max()  # normalise to 1

uvvis_paths = sorted(UVVIS_DIR.glob("Ferricyanide_UV_VIS_*.csv"))
# 2.0 mm cuvette. No concentration window: ferricyanide's canonical eps(420) = 1017
# is a pooled fit over the full measured 7.8 uM - 8 mM range (n = 30). load_uvvis
# used to default to a 1000 uM ceiling, which returned 981.
eps_wl, eps_mean, eps_std = load_uvvis(uvvis_paths, pathlength_cm=0.2)

S_raw = load_sensitivity("average_all")  # (31, 3): 400–700 nm, 10 nm steps
wl_S_full = np.arange(400, 701, 10, dtype=float)
# interpolate to 1-nm grid 400–700
_GRID = np.arange(400, 701, 1, dtype=float)
S = np.column_stack([np.interp(_GRID, wl_S_full, S_raw[:, i]) for i in range(3)])

X_MIN, X_MAX = 350, 800  # unified spectral x-range (matches illuminant)


def fig_illuminant():
    vis = (phi_wl >= X_MIN) & (phi_wl <= X_MAX)
    fig, ax = plt.subplots(figsize=(W1, H_SPEC))
    ax.fill_between(phi_wl[vis], phi[vis], alpha=0.18, color=C_GRAY)
    ax.plot(phi_wl[vis], phi[vis], color=C_GRAY, lw=0.9)
    ax.axvline(
        LAMBDA_MAX, color=C_B, lw=0.8, ls="--", label=rf"{LAMBDA_MAX:.0f} nm (FC $\lambda_{{max}}$)"
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$\varphi(\lambda)$  [normalised]")
    ax.legend(framealpha=0.85, edgecolor="none")
    ax.set_xlim(X_MIN, X_MAX)
    fig.tight_layout(pad=0.5)
    _style(ax)
    csv_df = pd.DataFrame({"wavelength_nm": phi_wl[vis], "phi_normalised": phi[vis]})
    _save(fig, "FM_spectra_illuminant", csv_df)


def fig_bayer():
    fig, ax = plt.subplots(figsize=(W1, H_SPEC))
    for i, (ch, col) in enumerate(zip(["R", "G", "B"], [C_R, C_G, C_B])):
        ax.plot(_GRID, S[:, i], color=col, lw=0.9, label=ch)
    ax.axvline(LAMBDA_MAX, color=C_B, lw=0.8, ls="--", label=rf"{LAMBDA_MAX:.0f} nm")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$T_{ch}(\lambda)$  [normalised]")
    ax.legend(framealpha=0.85, edgecolor="none")
    ax.set_xlim(X_MIN, X_MAX)
    fig.tight_layout(pad=0.5)
    _style(ax)
    csv_df = pd.DataFrame({"wavelength_nm": _GRID, "R": S[:, 0], "G": S[:, 1], "B": S[:, 2]})
    _save(fig, "FM_spectra_bayer", csv_df)


def fig_uvvis():
    vis = (eps_wl >= X_MIN) & (eps_wl <= X_MAX)
    fig, ax = plt.subplots(figsize=(W1, H_SPEC))
    if np.any(eps_std[vis] > 0):
        ax.fill_between(
            eps_wl[vis],
            (eps_mean - eps_std)[vis],
            (eps_mean + eps_std)[vis],
            alpha=0.25,
            color=C_GRAY,
        )
    ax.plot(eps_wl[vis], eps_mean[vis], color=C_GRAY, lw=0.9)
    ax.axvline(
        LAMBDA_MAX, color=C_B, lw=0.8, ls="--", label=rf"$\lambda_{{max}}$ = {LAMBDA_MAX:.0f} nm"
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$\varepsilon(\lambda)$  (M$^{-1}$ cm$^{-1}$)")
    ax.legend(framealpha=0.85, edgecolor="none")
    ax.set_xlim(X_MIN, X_MAX)
    fig.tight_layout(pad=0.5)
    _style(ax)
    csv_df = pd.DataFrame(
        {
            "wavelength_nm": eps_wl[vis],
            "eps_mean": eps_mean[vis],
            "eps_se": eps_std[vis],
        }
    )
    _save(fig, "FM_spectra_uvvis", csv_df)


fig_illuminant()
fig_bayer()
fig_uvvis()

print(f"\nDone. {len(list(OUT_DIR.glob('*.png')))} PNGs in {OUT_DIR}")
