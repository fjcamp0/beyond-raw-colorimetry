"""
figure_S8.py — Daylight calibration, methylene blue (R + RdivB), forward-
model fit and l_cam comparison, 1x3 panel, ACS double-column.

  (a) R channel: forward-model curve (l_cam fitted on R) overlaid on
      empirical daylight+diffuser calibration (mean +/- SD, n = 2 acquisition
      sets: Set 2, Set 3; Set 1 excluded — a 25 uM image taken 20 min after
      the blank).
  (b) RdivB channel: A_RdivB = A_R - A_B per image (algebraically identical
      to -log10((R/B)_sample / (R/B)_blank), matching figure_S4.py's
      definition), aggregated to mean +/- SD.
  (c) l_cam comparison: daylight-illuminant model vs LED-illuminant model,
      same daylight image data, against the physical 2.0 mm cuvette.

Model curves and l_cam values are read directly from the already-computed
pipeline outputs (forward_model_daylight_diffuser/, unaffected by the
UV-Vis path-length bug that affected ferricyanide — see
Manuscript/_ai_outputs/daylight_lcam_correction_2026-06-30.md).

Run with: python scripts/figure_S8.py (from the project root).

Outputs -> Data/paper_figures/figure_S8/
    figure_S8_panel.{png,svg,tiff}                composite 1x3 panel
    figure_S8a.{png,svg,tiff} ... figure_S8c.{...} individual panels
    figure_S8_data.csv                             calibration + model data
    figure_S8_lcam.csv                              l_cam comparison table
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJ_ROOT / "Data"
ANALYSIS_DIR = DATA_ROOT / "analysis" / "methylene_blue" / "daylight_diffuser" / "image_analysis"
CAL_CSV = ANALYSIS_DIR / "s02_calibrate" / "calibration_table.csv"
ABS_CSV = ANALYSIS_DIR / "s02_calibrate" / "absorbance_table.csv"
FM_CSV = ANALYSIS_DIR / "forward_model_daylight_diffuser" / "forward_model_table.csv"
OUT_DIR = DATA_ROOT / "paper_figures" / "figure_S8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

L_CAM_MM = 1.55  # fitted on R, daylight+diffuser image data, daylight phi
CUVETTE_MM = 2.0

# l_cam cross-check: same daylight image data, model illuminant swapped
# (from daylight_lcam_correction_2026-06-30.md).
LCAM_COMPARISON = [
    ("average_all, daylight φ", 1.55),
    ("average_all, LED φ", 1.59),
]

PANEL_W, PANEL_H = 9.0, 3.0
INDIV_W, INDIV_H = 3.33, 2.85
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
        "figure.dpi": 150,
    }
)


def build_rdivb_calibration(abs_df: pd.DataFrame) -> pd.DataFrame:
    """A_RdivB = A_R - A_B per image (matches daylight_diffuser_figures_MB.py),
    aggregated to mean +/- SD per concentration."""
    r = abs_df[abs_df.channel == "R"][["filename", "concentration_um", "absorbance"]]
    b = abs_df[abs_df.channel == "B"][["filename", "absorbance"]]
    m = r.merge(b, on="filename", suffixes=("_R", "_B"))
    m["absorbance"] = m["absorbance_R"] - m["absorbance_B"]

    grouped = m.groupby("concentration_um")["absorbance"]
    return pd.DataFrame(
        {
            "concentration_um": grouped.mean().index,
            "channel": "RdivB",
            "absorbance_mean": grouped.mean().to_numpy(),
            "absorbance_std": grouped.std(ddof=1).fillna(0).to_numpy(),
        }
    )


def draw_calibration(
    ax: plt.Axes,
    ch_cal: pd.DataFrame,
    fm: pd.DataFrame,
    fm_col: str,
    color: str,
    label: str,
    title: str,
    letter: str,
) -> None:
    ch_cal = ch_cal.sort_values("concentration_um")
    ax.errorbar(
        ch_cal["concentration_um"],
        ch_cal["absorbance_mean"],
        yerr=ch_cal["absorbance_std"],
        fmt="o",
        color=color,
        markersize=4,
        capsize=2.2,
        elinewidth=0.8,
        zorder=3,
        label="Empirical (mean ± SD, n = 2)",
    )

    x_min = ch_cal["concentration_um"].min() / 3
    x_max = ch_cal["concentration_um"].max() * 3
    fm_um = fm["concentration_um"].to_numpy()
    fm_a = fm[fm_col].to_numpy()
    valid = np.isfinite(fm_a) & (fm_um >= x_min) & (fm_um <= x_max)
    ax.plot(
        fm_um[valid],
        fm_a[valid],
        color=fs.MODEL_COLOR,
        lw=1.0,
        alpha=0.85,
        zorder=2,
        label=f"Forward model (l$_{{cam}}$ = {L_CAM_MM:g} mm)",
    )

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("[MB] (µM)")
    ax.set_ylabel(f"Absorbance ({label})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(framealpha=0, fontsize=5.5, loc="upper left", handlelength=1.4)
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.text(
        0.97,
        0.03,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def draw_lcam_comparison(ax: plt.Axes) -> None:
    labels = [lab for lab, _ in LCAM_COMPARISON]
    values = [val for _, val in LCAM_COMPARISON]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=fs.GRAY, width=0.5, zorder=2)
    bars[0].set_color(fs.ROUTE_COLORS["raw"][0])
    ax.axhline(
        CUVETTE_MM,
        color=fs.FIT_COLOR,
        lw=1.0,
        ls="--",
        zorder=1,
        label=f"Physical cuvette ({CUVETTE_MM:g} mm)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5, rotation=25, ha="right")
    ax.set_ylabel("l$_{cam}$ (mm)")
    ax.set_ylim(0, 2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(framealpha=0, fontsize=5.5, loc="upper right")
    ax.set_title("l$_{cam}$ cross-check", fontsize=7.5, pad=3)
    ax.text(
        0.97,
        0.03,
        "(c)",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="right",
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


def main() -> None:
    print(f"[figure_S8] Output -> {OUT_DIR}\n")

    cal = pd.read_csv(CAL_CSV)
    abs_df = pd.read_csv(ABS_CSV)
    fm = pd.read_csv(FM_CSV)

    r_cal = cal[cal.channel == "R"]
    rdivb_cal = build_rdivb_calibration(abs_df)

    print("Building composite panel ...")
    fig, axes = plt.subplots(1, 3, figsize=(PANEL_W, PANEL_H), gridspec_kw={"wspace": 0.5})
    draw_calibration(
        axes[0],
        r_cal,
        fm,
        "A_R",
        fs.ROUTE_COLORS["raw"][1],
        "A$_R$",
        "Methylene blue — R channel",
        "a",
    )
    draw_calibration(
        axes[1],
        rdivb_cal,
        fm,
        "A_RdivB",
        fs.ROUTE_COLORS["raw"][0],
        "A$_{R/B}$",
        "Methylene blue — R/B channel",
        "b",
    )
    draw_lcam_comparison(axes[2])
    _save_fig(fig, "figure_S8_panel")
    plt.close(fig)

    print("\nSaving individual panels ...")
    fig_a, ax_a = plt.subplots(figsize=(INDIV_W, INDIV_H))
    draw_calibration(
        ax_a,
        r_cal,
        fm,
        "A_R",
        fs.ROUTE_COLORS["raw"][1],
        "A$_R$",
        "Methylene blue — R channel",
        "a",
    )
    fig_a.tight_layout(pad=0.5)
    _save_fig(fig_a, "figure_S8a")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(INDIV_W, INDIV_H))
    draw_calibration(
        ax_b,
        rdivb_cal,
        fm,
        "A_RdivB",
        fs.ROUTE_COLORS["raw"][0],
        "A$_{R/B}$",
        "Methylene blue — R/B channel",
        "b",
    )
    fig_b.tight_layout(pad=0.5)
    _save_fig(fig_b, "figure_S8b")
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(INDIV_W, INDIV_H))
    draw_lcam_comparison(ax_c)
    fig_c.tight_layout(pad=0.5)
    _save_fig(fig_c, "figure_S8c")
    plt.close(fig_c)

    print("\nWriting data CSVs ...")
    r_cal_out = r_cal.copy()
    r_cal_out["panel"] = "a"
    rdivb_cal_out = rdivb_cal.copy()
    rdivb_cal_out["panel"] = "b"
    cal_all = pd.concat([r_cal_out, rdivb_cal_out], ignore_index=True)
    cal_all.to_csv(OUT_DIR / "figure_S8_data.csv", index=False, float_format="%.6f")
    print(f"    -> figure_S8_data.csv  ({len(cal_all)} rows)")

    lcam_df = pd.DataFrame(LCAM_COMPARISON, columns=["assumption", "l_cam_mm"])
    lcam_df.to_csv(OUT_DIR / "figure_S8_lcam.csv", index=False)
    print(f"    -> figure_S8_lcam.csv  ({len(lcam_df)} rows)")

    print(f"\n[figure_S8] Done.\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
