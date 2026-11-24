"""
figure_S7.py — Daylight calibration, ferricyanide (B + BdivG), forward-model
fit and l_cam comparison, 1x3 panel, ACS double-column.

  (a) B channel: forward-model curve (l_cam fitted on BdivG, 62.5-1000 uM)
      overlaid on empirical daylight+diffuser calibration (mean +/- SD, n = 3
      acquisition sets, sets 2-4; set 1 excluded for a blank-timing mismatch).
  (b) BdivG channel: same, primary fit channel.
  (c) l_cam comparison across illuminant-model and camera-proxy assumptions
      (from the 2026-06-30 UV-Vis path-length correction, all re-derived with
      the corrected 0.2 cm UV-Vis path length), against the physical 2.0 mm
      cuvette.

Model curves and l_cam values are read directly from the already-computed,
corrected pipeline outputs (forward_model_daylight_diffuser_l0p2/), not
re-fitted here — see Manuscript/_ai_outputs/daylight_lcam_correction_2026-06-30.md
for the fit methodology and the l_cam cross-check table reproduced in panel (c).

Run with: python scripts/figure_S7.py (from the project root).

Outputs -> Data/paper_figures/figure_S7/
    figure_S7_panel.{png,svg,tiff}                composite 1x3 panel
    figure_S7a.{png,svg,tiff} ... figure_S7c.{...} individual panels
    figure_S7_data.csv                             calibration + model data
    figure_S7_lcam.csv                              l_cam comparison table
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
ANALYSIS_DIR = DATA_ROOT / "analysis" / "ferricyanide" / "daylight_diffuser" / "image_analysis"
CAL_CSV = ANALYSIS_DIR / "s02_calibrate" / "calibration_table.csv"
FM_CSV = ANALYSIS_DIR / "forward_model_daylight_diffuser_l0p2" / "forward_model_table.csv"
OUT_DIR = DATA_ROOT / "paper_figures" / "figure_S7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

L_CAM_MM = 5.03  # fitted on BdivG, 62.5-1000 uM, corrected UV-Vis path length
CUVETTE_MM = 2.0

# l_cam cross-check across illuminant-model / camera-proxy assumptions,
# reproduced from daylight_lcam_correction_2026-06-30.md (all fit on the same
# daylight+diffuser BdivG image data unless noted).
LCAM_COMPARISON = [
    ("average_all, daylight φ", 5.03),
    ("average_all, LED φ", 4.38),
    ("huawei_nova_lite2, daylight φ", 6.02),
    ("huawei_p10_lite, daylight φ", 3.66),
]

PANELS = [
    {
        "letter": "a",
        "channel": "B",
        "fm_col": "A_B",
        "color": fs.ROUTE_COLORS["raw"][1],
        "label": "A$_B$",
    },
    {
        "letter": "b",
        "channel": "BdivG",
        "fm_col": "A_BdivG",
        "color": fs.ROUTE_COLORS["raw"][0],
        "label": "A$_{B/G}$",
    },
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


def draw_calibration(ax: plt.Axes, cal: pd.DataFrame, fm: pd.DataFrame, cfg: dict) -> None:
    ch_cal = cal[cal["channel"] == cfg["channel"]].sort_values("concentration_um")
    ax.errorbar(
        ch_cal["concentration_um"],
        ch_cal["absorbance_mean"],
        yerr=ch_cal["absorbance_std"],
        fmt="o",
        color=cfg["color"],
        markersize=4,
        capsize=2.2,
        elinewidth=0.8,
        zorder=3,
        label="Empirical (mean ± SD, n = 3)",
    )

    fm_um = fm["concentration_um"].to_numpy()
    fm_a = fm[cfg["fm_col"]].to_numpy()
    valid = np.isfinite(fm_a)
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
    ax.set_xlabel("[FC] (µM)")
    ax.set_ylabel(f"Absorbance ({cfg['label']})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(framealpha=0, fontsize=5.5, loc="upper left", handlelength=1.4)
    ax.set_title(f"Ferricyanide — {cfg['channel']} channel", fontsize=7.5, pad=3)
    ax.text(
        0.97,
        0.03,
        f"({cfg['letter']})",
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
    bars = ax.bar(x, values, color=fs.GRAY, width=0.6, zorder=2)
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
    print(f"[figure_S7] Output -> {OUT_DIR}\n")

    cal = pd.read_csv(CAL_CSV)
    fm = pd.read_csv(FM_CSV)

    print("Building composite panel ...")
    fig, axes = plt.subplots(1, 3, figsize=(PANEL_W, PANEL_H), gridspec_kw={"wspace": 0.5})
    for ax, cfg in zip(axes[:2], PANELS):
        draw_calibration(ax, cal, fm, cfg)
    draw_lcam_comparison(axes[2])
    _save_fig(fig, "figure_S7_panel")
    plt.close(fig)

    print("\nSaving individual panels ...")
    for cfg in PANELS:
        fig_i, ax_i = plt.subplots(figsize=(INDIV_W, INDIV_H))
        draw_calibration(ax_i, cal, fm, cfg)
        fig_i.tight_layout(pad=0.5)
        _save_fig(fig_i, f"figure_S7{cfg['letter']}")
        plt.close(fig_i)

    fig_c, ax_c = plt.subplots(figsize=(INDIV_W, INDIV_H))
    draw_lcam_comparison(ax_c)
    fig_c.tight_layout(pad=0.5)
    _save_fig(fig_c, "figure_S7c")
    plt.close(fig_c)

    print("\nWriting data CSVs ...")
    cal_rows = []
    for cfg in PANELS:
        sub = cal[cal["channel"] == cfg["channel"]].copy()
        sub["panel"] = cfg["letter"]
        cal_rows.append(sub)
    cal_all = pd.concat(cal_rows, ignore_index=True)
    cal_all.to_csv(OUT_DIR / "figure_S7_data.csv", index=False, float_format="%.6f")
    print(f"    -> figure_S7_data.csv  ({len(cal_all)} rows)")

    lcam_df = pd.DataFrame(LCAM_COMPARISON, columns=["assumption", "l_cam_mm"])
    lcam_df.to_csv(OUT_DIR / "figure_S7_lcam.csv", index=False)
    print(f"    -> figure_S7_lcam.csv  ({len(lcam_df)} rows)")

    print(f"\n[figure_S7] Done.\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
