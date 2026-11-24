"""
figure_S2.py — Ohlab sensitivity (21 devices) + narrow spectral overlay.

Two independent panels, satisfying both halves of the S8 SI slot:
  (a) All 21 individual smartphone Bayer sensitivity curves (R/G/B, thin
      lines) with the 21-device mean overlaid (bold), showing device-to-
      device variability directly rather than as a summary band.
  (b) White-LED illuminant phi(lambda), mean Bayer sensitivity T_ch(lambda),
      and both analytes' molar absorptivity epsilon(lambda) on shared axes
      (twin y-axes for epsilon) — the narrow overlay merged in from the
      parked figure_4_spectral_inputs_narrow.py, now showing why the
      420/480/665 nm wavelengths matter together in one view rather than
      as separate panels (cf. Figure 4).

Run with: python scripts/figure_S2.py (from the project root).
Uses the vendored Ohlab curves and modules in vendor/ (override with COLORIMETRY_VENDOR).

Outputs -> Data/paper_figures/figure_S2/
    figure_S2_panel.{png,svg,tiff}    assembled 1x2 panel
    figure_S2a.{png,svg,tiff}         21-device sensitivity spaghetti plot
    figure_S2b.{png,svg,tiff}         narrow phi/T_ch/epsilon overlay
    figure_S2_data.csv                all plotted data
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
from PIL import Image  # noqa: E402

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJ_ROOT / "Data"
SPECTRA_DIR = DATA_ROOT / "spectra"
ANALYSIS_DIR = DATA_ROOT / "analysis"
OUT_DIR = DATA_ROOT / "paper_figures" / "figure_S2"
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
import figstyle as fs  # noqa: E402
from figure_4_spectral_inputs_narrow import (  # noqa: E402
    load_ferricyanide_epsilon,
    load_led_illuminant,
    load_mb_epsilon,
)

# ─── Constants ────────────────────────────────────────────────────────────────
SENS_WL = np.arange(400, 710, 10, dtype=float)  # Ohlab native grid, 31 pts
CH_COLOR = fs.CHANNEL_COLORS
COL_FERIC = fs.FERIC_COLOR
COL_MB = fs.MB_COLOR

PANEL_W, PANEL_H = 7.0, 3.1  # double-column composite (in)
INDIV_W, INDIV_H = 3.33, 2.85  # single-column individual panels (in)
DPI = 600

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "legend.fontsize": 6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.8,
        "figure.dpi": 150,
    }
)


def _save_fig(fig: plt.Figure, stem: str) -> None:
    """Save fig as PNG, SVG and TIFF (TIFF via PIL to avoid a large-canvas bug
    in matplotlib's native TIFF writer with bbox_inches='tight')."""
    png_path = OUT_DIR / f"{stem}.png"
    for ext, path in (("png", png_path), ("svg", OUT_DIR / f"{stem}.svg")):
        fig.savefig(path, dpi=DPI, bbox_inches="tight", format=ext)
    img = Image.open(png_path)
    img.save(OUT_DIR / f"{stem}.tiff", dpi=(DPI, DPI), compression="tiff_lzw")
    print(f"    -> {stem}.png / .svg / .tiff")


def load_all_devices() -> dict:
    """Load all individual device .txt files from SENS_DIR (excluding average_all).

    Returns dict with 'names' (list[str]), 'curves' array (n, 31, 3) columns
    R/G/B, each device normalised to its own peak per channel, and 'mean'.
    """
    files = sorted(f for f in SENS_DIR.glob("*.txt") if f.stem != "average_all")
    n = len(files)
    if n == 0:
        raise FileNotFoundError(
            f"No sensitivity .txt files found in:\n  {SENS_DIR}\n"
            "Set COLORIMETRY_VENDOR to the vendor/ folder."
        )
    curves = np.zeros((n, len(SENS_WL), 3))
    names = []
    for k, fp in enumerate(files):
        raw = np.loadtxt(fp)  # (31, 3) R G B
        for i in range(3):
            peak = raw[:, i].max() or 1.0
            curves[k, :, i] = raw[:, i] / peak
        names.append(fp.stem)
    mean = curves.mean(axis=0)
    return {"n": n, "names": names, "curves": curves, "mean": mean}


def draw_device_spaghetti(ax: plt.Axes, sens: dict) -> None:
    for i, ch in enumerate(("R", "G", "B")):
        color = CH_COLOR[ch]
        for k in range(sens["n"]):
            ax.plot(SENS_WL, sens["curves"][k, :, i], color=color, lw=0.4, alpha=0.35, zorder=1)
        ax.plot(SENS_WL, sens["mean"][:, i], color=color, lw=1.4, label=ch, zorder=2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised sensitivity")
    ax.set_xlim(400, 700)
    ax.set_ylim(-0.03, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.legend(
        framealpha=0,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.2,
        loc="upper right",
        title=f"n={sens['n']} devices\n(thin) + mean (bold)",
        title_fontsize=5.5,
    )
    ax.text(
        0.03,
        0.97,
        "(a)",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_narrow_overlay(
    ax: plt.Axes, wl_phi, phi, wl_T, T, wl_fc, eps_fc, wl_mb, eps_mb
) -> plt.Axes:
    ax.plot(wl_phi, phi, color=fs.GRAY, lw=1.1, label=r"$\phi_{LED}$", zorder=2)
    for i, ch in enumerate(("R", "G", "B")):
        ax.plot(wl_T, T[:, i], color=CH_COLOR[ch], lw=1.0, ls="--", label=f"$T_{{{ch}}}$", zorder=2)
    ax.axvline(420, color=COL_FERIC, lw=0.7, ls=":", alpha=0.75)
    ax.axvline(480, color=CH_COLOR["B"], lw=0.7, ls=":", alpha=0.6)
    ax.axvline(665, color=COL_MB, lw=0.7, ls=":", alpha=0.75)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised phi, T (norm.)")
    ax.set_xlim(350, 800)
    ax.set_ylim(-0.03, 1.15)
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    eps_fc_plot = np.where(np.isfinite(eps_fc), eps_fc, np.nan)
    eps_mb_plot = np.where(eps_mb > 0, eps_mb, np.nan)
    (l_fc,) = ax2.plot(
        wl_fc, eps_fc_plot, color=COL_FERIC, lw=1.1, label=r"$\varepsilon$ ferricyanide", zorder=3
    )

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.28))
    (l_mb,) = ax3.plot(
        wl_mb, eps_mb_plot, color=COL_MB, lw=1.1, label=r"$\varepsilon$ MB", zorder=3
    )

    fc_max = np.nanmax(eps_fc_plot[(wl_fc >= 350) & (wl_fc <= 800)])
    ax2.set_ylim(0, fc_max * 1.25)
    ax3.set_ylim(0, np.nanmax(eps_mb_plot) * 1.25)
    ax2.set_ylabel(r"$\varepsilon$ ferricyanide (M$^{-1}$cm$^{-1}$)", color=COL_FERIC, fontsize=6.5)
    ax3.set_ylabel(r"$\varepsilon$ MB (M$^{-1}$cm$^{-1}$)", color=COL_MB, fontsize=6.5)
    ax2.tick_params(axis="y", colors=COL_FERIC, labelsize=6)
    ax3.tick_params(axis="y", colors=COL_MB, labelsize=6)

    lines = [ax.lines[0]] + ax.lines[1:4] + [l_fc, l_mb]
    labels = [ln.get_label() for ln in lines]
    ax.legend(
        lines,
        labels,
        framealpha=0.85,
        facecolor="white",
        edgecolor="none",
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=0.2,
        loc="upper center",
        fontsize=5.5,
        ncol=3,
        bbox_to_anchor=(0.5, 1.18),
    )
    ax.text(
        0.97,
        0.60,
        "(b)",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="right",
    )
    return ax2, ax3


def main() -> None:
    print(f"[figure_S2] Output -> {OUT_DIR}\n")

    print("Loading 21-device Bayer sensitivity ...")
    sens = load_all_devices()
    print(f"    {sens['n']} devices found")

    print("Loading LED illuminant, mean sensitivity, epsilon(lambda) ...")
    wl_phi, phi = load_led_illuminant()
    wl_T = SENS_WL
    T = sens["mean"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wl_fc, eps_fc = load_ferricyanide_epsilon()
    wl_mb, eps_mb = load_mb_epsilon()

    print("\nBuilding composite panel ...")
    fig, axes = plt.subplots(1, 2, figsize=(PANEL_W, PANEL_H), gridspec_kw={"wspace": 0.9})
    draw_device_spaghetti(axes[0], sens)
    draw_narrow_overlay(axes[1], wl_phi, phi, wl_T, T, wl_fc, eps_fc, wl_mb, eps_mb)
    _save_fig(fig, "figure_S2_panel")
    plt.close(fig)

    print("\nSaving individual panels ...")
    fig_a, ax_a = plt.subplots(figsize=(INDIV_W, INDIV_H))
    draw_device_spaghetti(ax_a, sens)
    fig_a.tight_layout(pad=0.5)
    _save_fig(fig_a, "figure_S2a")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(INDIV_W + 0.9, INDIV_H))
    draw_narrow_overlay(ax_b, wl_phi, phi, wl_T, T, wl_fc, eps_fc, wl_mb, eps_mb)
    fig_b.tight_layout(pad=0.5)
    _save_fig(fig_b, "figure_S2b")
    plt.close(fig_b)

    print("\nWriting data CSV ...")
    rows = {"wavelength_nm_sens": SENS_WL}
    for i, ch in enumerate(("R", "G", "B")):
        rows[f"sensitivity_{ch}_mean"] = sens["mean"][:, i]
        for k, name in enumerate(sens["names"]):
            rows[f"sensitivity_{ch}_{name}"] = sens["curves"][k, :, i]
    df_sens = pd.DataFrame(rows)

    eps_mb_interp = np.interp(wl_fc, wl_mb, eps_mb, left=np.nan, right=np.nan)
    df_overlay = pd.DataFrame(
        {
            "wavelength_nm_overlay": wl_fc,
            "epsilon_ferricyanide": eps_fc,
            "epsilon_MB": eps_mb_interp,
        }
    )
    phi_interp = np.interp(wl_fc, wl_phi, phi, left=np.nan, right=np.nan)
    df_overlay["phi_LED_norm"] = phi_interp
    for i, ch in enumerate(("R", "G", "B")):
        df_overlay[f"T_{ch}_mean"] = np.interp(wl_fc, wl_T, T[:, i], left=np.nan, right=np.nan)

    df_sens.to_csv(OUT_DIR / "figure_S2_data_sensitivity.csv", index=False, float_format="%.6f")
    df_overlay.to_csv(OUT_DIR / "figure_S2_data_overlay.csv", index=False, float_format="%.6f")
    print(
        f"    -> figure_S2_data_sensitivity.csv  ({len(df_sens.columns)} columns, {len(df_sens)} rows)"
    )
    print(
        f"    -> figure_S2_data_overlay.csv  ({len(df_overlay.columns)} columns, {len(df_overlay)} rows)"
    )

    print(f"\n[figure_S2] Done.\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
