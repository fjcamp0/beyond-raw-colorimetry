"""
figure_4_spectral_inputs_narrow.py — Spectral inputs for the Beer-Lambert-Bayer integral.
Produces three independent single-column panels (A: illuminant, B: Bayer
sensitivity, C: molar absorptivity) plus assembled 3-panel versions for both
LED and daylight illuminants.

ACS format: 3.25 in wide, Arial 8 pt, 600 dpi, inside ticks, all four sides.
Output: LED/ and daylight/ subfolders inside the figure_4_spectral_inputs_narrow/ folder.

Run with: python scripts/figure_4_spectral_inputs_narrow.py (from the project root).
Uses the vendored Ohlab curves and modules in vendor/ (override with COLORIMETRY_VENDOR).
"""

import os
import pathlib
import shutil
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import rcParams  # noqa: E402

# ---------------------------------------------------------------------------
# Paths — resolved relative to this script's location
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE.parent

DATA_DIR = PROJECT / "Data"
SPECTRA_DIR = DATA_DIR / "spectra"
ANALYSIS_DIR = DATA_DIR / "analysis"
# Vendored raw-image-analysis modules (vendor/, see README); override with COLORIMETRY_VENDOR.
SENSITIVITY_DIR = (
    pathlib.Path(
        os.environ.get(
            "COLORIMETRY_VENDOR", str(pathlib.Path(__file__).resolve().parent.parent / "vendor")
        )
    )
    / "_resources"
    / "spectral_sensitivity"
)

LED_OUT = DATA_DIR / "paper_figures" / "figure_4_spectral_inputs_narrow" / "LED"
DAY_OUT = DATA_DIR / "paper_figures" / "figure_4_spectral_inputs_narrow" / "daylight"
LED_OUT.mkdir(parents=True, exist_ok=True)
DAY_OUT.mkdir(parents=True, exist_ok=True)

# assemble_manuscript.py reads figures only from Manuscript/_drafts/figures/,
# never from Data/paper_figures/ — mirror every render there.
DRAFTS_DIR = PROJECT / "Manuscript" / "_drafts" / "figures" / "figure_4_spectral_inputs_narrow"

# ---------------------------------------------------------------------------
# ACS style
# ---------------------------------------------------------------------------
PANEL_W = 3.25  # inches, single column
PANEL_H = 3.00  # inches
ASSM_H = 8.50  # inches, assembled 3-panel
DPI = 600
FONTSIZE = 8

COL_B = "#1a6faf"
COL_G = "#009e73"
COL_R = "#d55e00"
COL_FERIC = "#2c7bb6"
COL_MB = "#d7191c"


def set_acs_style():
    """Apply ACS-compatible rcParams."""
    rcParams.update(
        {
            "font.family": "Arial",
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
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "figure.dpi": 100,
        }
    )


def save_panel(fig, stem, outdir):
    """Save fig as PNG, SVG and TIFF in outdir.

    TIFF is produced via PNG->PIL conversion to avoid a PIL struct.error that
    can trigger when matplotlib passes very large pixel dimensions to its TIFF
    backend with bbox_inches='tight'.
    """
    png_path = outdir / f"{stem}.png"
    svg_path = outdir / f"{stem}.svg"
    tiff_path = outdir / f"{stem}.tiff"

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    from PIL import Image

    img = Image.open(png_path)
    img.save(tiff_path, dpi=(DPI, DPI), compression="tiff_lzw")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_led_illuminant():
    """Load the SpElec white-LED spectrum. Returns (wl, phi) arrays, normalised."""
    path = SPECTRA_DIR / "illuminant" / "illuminant_spelec_led.csv"
    df = pd.read_csv(path)
    wl = df["wavelength_nm"].to_numpy(dtype=float)
    phi = df["normalised_spectral_power"].to_numpy(dtype=float)
    # Filter to visible range 350–800 nm
    mask = (wl >= 350) & (wl <= 800)
    wl, phi = wl[mask], phi[mask]
    phi = phi / phi.max()
    return wl, phi


def load_daylight_illuminant():
    """Load daylight-with-diffuser spectrum. Returns (wl, phi) arrays, normalised."""
    path = SPECTRA_DIR / "illuminant" / "Daylight_spectra_with diffuser.csv"
    df = pd.read_csv(path)
    wl = pd.to_numeric(df["Wavelenght (nm)"], errors="coerce")
    dark = pd.to_numeric(df["Dark"], errors="coerce")
    sig = pd.to_numeric(df["Daylight_spectra_with diffuser"], errors="coerce")
    valid = wl.notna() & dark.notna() & sig.notna()
    wl = wl[valid].to_numpy(dtype=float)
    phi = (sig[valid] - dark[valid]).to_numpy(dtype=float)
    idx = np.argsort(wl)
    wl, phi = wl[idx], phi[idx]
    phi = np.clip(phi, 0, None)
    # Filter to visible range 350–800 nm
    mask = (wl >= 350) & (wl <= 800)
    wl, phi = wl[mask], phi[mask]
    phi = phi / phi.max()
    return wl, phi


def load_bayer_sensitivity():
    """Load the Ohlab 21-phone average Bayer sensitivity.

    File: average_all.txt — 31 rows (400–700 nm, 10 nm steps), 3 tab-separated
    columns: R, G, B. No header.
    Returns wl (31,), T (31, 3) where columns are R, G, B.
    """
    path = SENSITIVITY_DIR / "average_all.txt"
    T = np.loadtxt(path)  # shape (31, 3)
    wl = np.arange(400, 710, 10, dtype=float)  # 400, 410, ..., 700
    # Normalise each channel to its own peak
    T = T / T.max(axis=0, keepdims=True)
    return wl, T


def load_ferricyanide_epsilon():
    """Compute ferricyanide molar absorptivity from all three UV-VIS datasets.

    Uses ε = A / (c [mol/L] × l [cm]) with l = 0.2 cm.
    Returns (wl, eps_median) on a regular 2 nm grid, 350–800 nm.
    """
    path_len = 0.2  # cm

    def _parse_conc(col_name):
        """Parse concentration string to mol/L."""
        col_name = col_name.strip()
        if "uM" in col_name:
            return float(col_name.replace("uM", "").strip()) * 1e-6
        if "mM" in col_name:
            return float(col_name.replace("mM", "").strip()) * 1e-3
        return None

    all_wl = []
    all_eps = []

    # Dataset 1: "Wavelenght (nm)" header (typo)
    df1 = pd.read_csv(SPECTRA_DIR / "ferricyanide" / "Ferricyanide_UV_VIS_1.csv")
    wl1 = pd.to_numeric(df1["Wavelenght (nm)"], errors="coerce")
    mask1 = wl1.notna()
    wl1_v = wl1[mask1].to_numpy(float)
    blank1 = pd.to_numeric(df1.loc[mask1, "Blank"], errors="coerce").to_numpy(float)
    for col in df1.columns:
        if col == "Blank":
            continue
        c = _parse_conc(col)
        if c is None:
            continue
        A = pd.to_numeric(df1.loc[mask1, col], errors="coerce").to_numpy(float) - blank1
        eps = A / (c * path_len)
        all_wl.append(wl1_v)
        all_eps.append(eps)

    # Dataset 2: "Wavelength (nm)" header
    df2 = pd.read_csv(SPECTRA_DIR / "ferricyanide" / "Ferricyanide_UV_VIS_2.csv")
    wl2 = pd.to_numeric(df2["Wavelength (nm)"], errors="coerce")
    mask2 = wl2.notna()
    wl2_v = wl2[mask2].to_numpy(float)
    blank2 = pd.to_numeric(df2.loc[mask2, "Blank"], errors="coerce").to_numpy(float)
    for col in df2.columns:
        if col == "Blank":
            continue
        c = _parse_conc(col)
        if c is None:
            continue
        A = pd.to_numeric(df2.loc[mask2, col], errors="coerce").to_numpy(float) - blank2
        eps = A / (c * path_len)
        all_wl.append(wl2_v)
        all_eps.append(eps)

    # Dataset 3: same as dataset 2
    df3 = pd.read_csv(SPECTRA_DIR / "ferricyanide" / "Ferricyanide_UV_VIS_3.csv")
    wl3 = pd.to_numeric(df3["Wavelength (nm)"], errors="coerce")
    mask3 = wl3.notna()
    wl3_v = wl3[mask3].to_numpy(float)
    blank3 = pd.to_numeric(df3.loc[mask3, "Blank"], errors="coerce").to_numpy(float)
    for col in df3.columns:
        if col == "Blank":
            continue
        c = _parse_conc(col)
        if c is None:
            continue
        A = pd.to_numeric(df3.loc[mask3, col], errors="coerce").to_numpy(float) - blank3
        eps = A / (c * path_len)
        all_wl.append(wl3_v)
        all_eps.append(eps)

    # Build common grid 350–800 nm, 2 nm step
    wl_grid = np.arange(350, 801, 2, dtype=float)
    eps_interp = []
    for wl_raw, eps_raw in zip(all_wl, all_eps):
        good = np.isfinite(wl_raw) & np.isfinite(eps_raw)
        if good.sum() < 5:
            continue
        # Sort
        idx = np.argsort(wl_raw[good])
        wl_s = wl_raw[good][idx]
        eps_s = eps_raw[good][idx]
        # Interpolate onto grid (within bounds; NaN outside)
        eps_g = np.interp(wl_grid, wl_s, eps_s, left=np.nan, right=np.nan)
        eps_interp.append(eps_g)

    eps_all = np.array(eps_interp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        eps_median = np.nanmedian(eps_all, axis=0)

    # Clip to visible range for plotting; NaN outside data
    return wl_grid, eps_median


def load_mb_epsilon():
    """Load pre-computed MB molar absorptivity (L mol⁻¹ cm⁻¹).

    File has columns: wavelength_nm, mean, std, n.
    Returns (wl, eps) arrays.
    """
    df = pd.read_csv(ANALYSIS_DIR / "methylene_blue" / "uvvis_epsilon.csv")
    wl = df["wavelength_nm"].to_numpy(float)
    eps = df["mean"].to_numpy(float)
    return wl, eps


# ---------------------------------------------------------------------------
# Panel A — Illuminant φ(λ)
# ---------------------------------------------------------------------------


def plot_panel_a(wl_phi, phi, illuminant_label, outdir, stem):
    """Plot illuminant spectrum with vertical markers at 420 and 665 nm."""
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

    ax.plot(wl_phi, phi, color="#555555", lw=1.2, label=illuminant_label)
    ax.axvline(420, color=COL_FERIC, lw=0.9, ls="--", label="420 nm (ferricyanide)")
    ax.axvline(665, color=COL_MB, lw=0.9, ls="--", label="665 nm (MB)")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised spectral power")
    ax.set_xlim(350, 800)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax.text(
        420 + 15,
        ax.get_ylim()[1] * 0.95,
        "420 nm",
        color=COL_FERIC,
        fontsize=6,
        ha="left",
        va="top",
        rotation=90,
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        665,
        ax.get_ylim()[1] * 0.95,
        "665 nm",
        color=COL_MB,
        fontsize=6,
        ha="left",
        va="top",
        rotation=90,
        transform=ax.get_xaxis_transform(),
    )
    ax.legend(frameon=False, fontsize=6, loc="upper center")

    # Panel label
    ax.text(
        0.03,
        0.97,
        "(a)",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )

    fig.tight_layout()
    save_panel(fig, stem, outdir)
    plt.close(fig)

    # Save CSV
    csv_path = outdir / f"{stem}.csv"
    df_out = pd.DataFrame({"wavelength_nm": wl_phi, "phi_lambda": phi})
    df_out.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  Saved {stem} -> {outdir.name}/")


# ---------------------------------------------------------------------------
# Panel B — Bayer sensitivity T_ch(λ)
# ---------------------------------------------------------------------------


def plot_panel_b(wl_T, T, outdir):
    """Plot Ohlab average Bayer spectral sensitivity (R, G, B channels)."""
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

    ax.plot(wl_T, T[:, 0], color=COL_R, lw=1.2, label="R")
    ax.plot(wl_T, T[:, 1], color=COL_G, lw=1.2, label="G")
    ax.plot(wl_T, T[:, 2], color=COL_B, lw=1.2, label="B")

    # Annotate B-channel tail cutoff ~480 nm
    ax.axvline(480, color=COL_B, lw=0.8, ls=":", alpha=0.7)
    ax.text(480, 0.12, "~480 nm\nB tail", color=COL_B, fontsize=6, ha="left", va="bottom")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised sensitivity")
    ax.set_xlim(400, 700)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.legend(frameon=False, fontsize=6, loc="upper right")

    ax.text(
        0.03,
        0.97,
        "(b)",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )

    fig.tight_layout()
    save_panel(fig, "figure_4_spectral_inputs_narrowb", outdir)
    plt.close(fig)

    csv_path = outdir / "figure_4_spectral_inputs_narrowb.csv"
    df_out = pd.DataFrame(
        {
            "wavelength_nm": wl_T,
            "T_R": T[:, 0],
            "T_G": T[:, 1],
            "T_B": T[:, 2],
        }
    )
    df_out.to_csv(csv_path, index=False, float_format="%.6f")
    print("  Saved figure_4_spectral_inputs_narrowb -> LED/")


# ---------------------------------------------------------------------------
# Panel C — Molar absorptivity ε(λ)
# ---------------------------------------------------------------------------


def plot_panel_c(wl_fc, eps_fc, wl_mb, eps_mb, outdir):
    """Plot ε(λ) for ferricyanide (left axis) and MB (right axis)."""
    fig, ax1 = plt.subplots(figsize=(PANEL_W, PANEL_H))
    ax2 = ax1.twinx()

    # Clip negative epsilon values (noise at long wavelengths)
    eps_fc_plot = np.where(np.isfinite(eps_fc), eps_fc, np.nan)
    eps_mb_plot = np.where(eps_mb > 0, eps_mb, np.nan)

    (line_fc,) = ax1.plot(
        wl_fc,
        eps_fc_plot,
        color=COL_FERIC,
        lw=1.2,
        label=r"Ferricyanide $\varepsilon$",
    )
    (line_mb,) = ax2.plot(
        wl_mb,
        eps_mb_plot,
        color=COL_MB,
        lw=1.2,
        label=r"MB $\varepsilon$",
    )

    # Vertical markers
    ax1.axvline(420, color=COL_FERIC, lw=0.8, ls="--", alpha=0.7)
    ax1.axvline(665, color=COL_MB, lw=0.8, ls="--", alpha=0.7)

    # Axis labels
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel(r"$\varepsilon$ ferricyanide (L mol$^{-1}$ cm$^{-1}$)", color=COL_FERIC)
    ax2.set_ylabel(r"$\varepsilon$ MB (L mol$^{-1}$ cm$^{-1}$)", color=COL_MB)
    ax1.tick_params(axis="y", labelcolor=COL_FERIC)
    ax2.tick_params(axis="y", labelcolor=COL_MB)

    ax1.set_xlim(350, 800)

    # Set reasonable y-limits: ferricyanide max ~1600 in visible
    vis_fc = wl_fc[(wl_fc >= 350) & (wl_fc <= 800)]
    eps_vis = eps_fc_plot[(wl_fc >= 350) & (wl_fc <= 800)]
    fc_max = np.nanmax(eps_vis)
    ax1.set_ylim(0, fc_max * 1.25)
    ax2.set_ylim(0, np.nanmax(eps_mb_plot) * 1.25)

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(5))

    y1_max = ax1.get_ylim()[1]
    ax1.text(
        420 + 15,
        y1_max * 0.85,
        "420 nm",
        color=COL_FERIC,
        fontsize=6,
        ha="left",
        va="top",
        rotation=90,
    )
    ax1.text(
        665, y1_max * 0.85, "665 nm", color=COL_MB, fontsize=6, ha="left", va="top", rotation=90
    )

    handles = [line_fc, line_mb]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, frameon=False, fontsize=6, loc="upper right")

    ax1.text(
        0.03,
        0.97,
        "(c)",
        transform=ax1.transAxes,
        fontsize=FONTSIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Right-side ticks need to be "inside" too
    ax2.tick_params(direction="in")

    fig.tight_layout()
    save_panel(fig, "figure_4_spectral_inputs_narrowc", outdir)
    plt.close(fig)

    # Save CSV on shared wavelength grid (ferricyanide grid, MB interpolated)
    eps_mb_interp = np.interp(wl_fc, wl_mb, eps_mb, left=np.nan, right=np.nan)
    df_out = pd.DataFrame(
        {
            "wavelength_nm": wl_fc,
            "epsilon_ferricyanide": eps_fc_plot,
            "epsilon_MB": eps_mb_interp,
        }
    )
    df_out.to_csv(outdir / "figure_4_spectral_inputs_narrowc.csv", index=False, float_format="%.4f")
    print("  Saved figure_4_spectral_inputs_narrowc -> LED/")


# ---------------------------------------------------------------------------
# Assembled 3-panel
# ---------------------------------------------------------------------------


def plot_assembled(
    wl_phi, phi, wl_T, T, wl_fc, eps_fc, wl_mb, eps_mb, illuminant_label, outdir, stem
):
    """Assemble panels A, B, C vertically into one figure."""
    fig, axes = plt.subplots(3, 1, figsize=(PANEL_W, ASSM_H), constrained_layout=True)
    ax_a, ax_b, ax_c = axes

    # --- Panel A ---
    ax_a.plot(wl_phi, phi, color="#555555", lw=1.2, label=illuminant_label)
    ax_a.axvline(420, color=COL_FERIC, lw=0.9, ls="--")
    ax_a.axvline(665, color=COL_MB, lw=0.9, ls="--")
    ax_a.text(
        420 + 15,
        0.98,
        "420 nm",
        color=COL_FERIC,
        fontsize=6,
        ha="left",
        va="top",
        rotation=90,
        transform=ax_a.get_xaxis_transform(),
    )
    ax_a.text(
        665,
        0.98,
        "665 nm",
        color=COL_MB,
        fontsize=6,
        ha="left",
        va="top",
        rotation=90,
        transform=ax_a.get_xaxis_transform(),
    )
    ax_a.set_xlabel("Wavelength (nm)")
    ax_a.set_ylabel("Normalised spectral power")
    ax_a.set_xlim(350, 800)
    ax_a.set_ylim(bottom=0)
    ax_a.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_a.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax_a.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax_a.xaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax_a.legend(frameon=False, fontsize=6, loc="upper center")
    ax_a.text(
        0.03,
        0.97,
        "(a)",
        transform=ax_a.transAxes,
        fontsize=FONTSIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # --- Panel B ---
    ax_b.plot(wl_T, T[:, 0], color=COL_R, lw=1.2, label="R")
    ax_b.plot(wl_T, T[:, 1], color=COL_G, lw=1.2, label="G")
    ax_b.plot(wl_T, T[:, 2], color=COL_B, lw=1.2, label="B")
    ax_b.axvline(480, color=COL_B, lw=0.8, ls=":", alpha=0.7)
    ax_b.text(480, 0.12, "~480 nm\nB tail", color=COL_B, fontsize=6, ha="left")
    ax_b.set_xlabel("Wavelength (nm)")
    ax_b.set_ylabel("Normalised sensitivity")
    ax_b.set_xlim(400, 700)
    ax_b.set_ylim(0, 1.15)
    ax_b.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_b.yaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    ax_b.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax_b.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax_b.legend(frameon=False, fontsize=6, loc="upper right")
    ax_b.text(
        0.03,
        0.97,
        "(b)",
        transform=ax_b.transAxes,
        fontsize=FONTSIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # --- Panel C (twin y-axis) ---
    ax_c2 = ax_c.twinx()
    eps_fc_plot = np.where(np.isfinite(eps_fc), eps_fc, np.nan)
    eps_mb_plot = np.where(eps_mb > 0, eps_mb, np.nan)

    (line_fc,) = ax_c.plot(
        wl_fc, eps_fc_plot, color=COL_FERIC, lw=1.2, label=r"Ferricyanide $\varepsilon$"
    )
    (line_mb,) = ax_c2.plot(wl_mb, eps_mb_plot, color=COL_MB, lw=1.2, label=r"MB $\varepsilon$")
    ax_c.axvline(420, color=COL_FERIC, lw=0.8, ls="--", alpha=0.7)
    ax_c.axvline(665, color=COL_MB, lw=0.8, ls="--", alpha=0.7)

    ax_c.set_xlabel("Wavelength (nm)")
    ax_c.set_ylabel(r"$\varepsilon$ ferricyanide (L mol$^{-1}$ cm$^{-1}$)", color=COL_FERIC)
    ax_c2.set_ylabel(r"$\varepsilon$ MB (L mol$^{-1}$ cm$^{-1}$)", color=COL_MB)
    ax_c.tick_params(axis="y", labelcolor=COL_FERIC)
    ax_c2.tick_params(axis="y", labelcolor=COL_MB, direction="in")
    ax_c.set_xlim(350, 800)
    vis_fc = eps_fc_plot[(wl_fc >= 350) & (wl_fc <= 800)]
    fc_max = np.nanmax(vis_fc)
    ax_c.set_ylim(0, fc_max * 1.25)
    ax_c2.set_ylim(0, np.nanmax(eps_mb_plot) * 1.25)
    ax_c.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax_c.xaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax_c.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax_c2.yaxis.set_major_locator(ticker.MaxNLocator(5))

    y1_max = ax_c.get_ylim()[1]
    ax_c.text(
        420, y1_max * 0.85, "420 nm", color=COL_FERIC, fontsize=6, ha="left", va="top", rotation=90
    )
    ax_c.text(
        665, y1_max * 0.85, "665 nm", color=COL_MB, fontsize=6, ha="left", va="top", rotation=90
    )

    handles = [line_fc, line_mb]
    labels = [h.get_label() for h in handles]
    ax_c.legend(handles, labels, frameon=False, fontsize=6, loc="upper right")
    ax_c.text(
        0.03,
        0.97,
        "(c)",
        transform=ax_c.transAxes,
        fontsize=FONTSIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # constrained_layout already set at figure creation; no tight_layout call
    save_panel(fig, stem, outdir)
    plt.close(fig)
    print(f"  Saved {stem} -> {outdir.name}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    set_acs_style()

    print("Loading data...")
    wl_led, phi_led = load_led_illuminant()
    wl_day, phi_day = load_daylight_illuminant()
    wl_T, T = load_bayer_sensitivity()
    wl_fc, eps_fc = load_ferricyanide_epsilon()
    wl_mb, eps_mb = load_mb_epsilon()

    print("\nGenerating LED figures...")
    # Panel A — LED illuminant
    plot_panel_a(
        wl_led,
        phi_led,
        illuminant_label="White LED (SpElec)",
        outdir=LED_OUT,
        stem="figure_4_spectral_inputs_narrowa_LED",
    )

    # Panel B — Bayer sensitivity (illuminant-independent, save in LED/ only)
    plot_panel_b(wl_T, T, outdir=LED_OUT)

    # Panel C — epsilon (illuminant-independent, save in LED/ only)
    plot_panel_c(wl_fc, eps_fc, wl_mb, eps_mb, outdir=LED_OUT)

    # Assembled LED
    plot_assembled(
        wl_led,
        phi_led,
        wl_T,
        T,
        wl_fc,
        eps_fc,
        wl_mb,
        eps_mb,
        illuminant_label="White LED (SpElec)",
        outdir=LED_OUT,
        stem="figure_4_spectral_inputs_narrow_assembled_LED",
    )

    print("\nGenerating daylight figures...")
    # Panel A — daylight illuminant
    plot_panel_a(
        wl_day,
        phi_day,
        illuminant_label="Daylight + diffuser",
        outdir=DAY_OUT,
        stem="figure_4_spectral_inputs_narrowa_daylight",
    )

    # Assembled daylight (B and C panels are the same)
    plot_assembled(
        wl_day,
        phi_day,
        wl_T,
        T,
        wl_fc,
        eps_fc,
        wl_mb,
        eps_mb,
        illuminant_label="Daylight + diffuser",
        outdir=DAY_OUT,
        stem="figure_4_spectral_inputs_narrow_assembled_daylight",
    )

    print("\nAll outputs written.")
    print("LED/ :", sorted([p.name for p in LED_OUT.iterdir()]))
    print("daylight/ :", sorted([p.name for p in DAY_OUT.iterdir()]))

    print("\nCopying outputs to Manuscript/_drafts/figures/ ...")
    for src_dir, subname in ((LED_OUT, "LED"), (DAY_OUT, "daylight")):
        dst_dir = DRAFTS_DIR / subname
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, dst_dir / f.name)


if __name__ == "__main__":
    main()
