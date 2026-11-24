"""
figure_S1.py — ROI selection (CV maps), 2x4 panel, ACS double-column.

One representative LED set per analyte (ferricyanide led_set1, methylene
blue led_set1):
  (a/e) Recommended ROI overlaid on a mid-series viewer image
  (b/f) Blank channel CV per 100x100 px block
  (c/g) Blank-corrected signal CV across samples, same blocks
  (d/h) Combined homogeneity score with recommended ROI outlined

Re-runs the raw-image-analysis roi_selector.py block-CV computation directly
on the DNG sets (channel B for ferricyanide, R for methylene blue, matching
the primary channel used throughout the main text) so the panel data is
backed by a CSV rather than just the pre-rendered preview PNGs.

Run with: python scripts/figure_S1.py (from the project root).
Uses the vendored Ohlab curves and modules in vendor/ (override with COLORIMETRY_VENDOR).

Outputs -> Data/paper_figures/figure_S1/
    figure_S1_panel.{png,svg,tiff}                composite 2x4 panel
    figure_S1a.{png,svg,tiff} ... figure_S1h.{...} individual panels
    figure_S1_data.csv                             block CV grids, both analytes
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJ_ROOT / "Data"
IMG_ROOT = DATA_ROOT / "images"
OUT_DIR = DATA_ROOT / "paper_figures" / "figure_S1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Vendored raw-image-analysis modules (vendor/, see README); override with COLORIMETRY_VENDOR.
_RIA_DIR = Path(
    os.environ.get("COLORIMETRY_VENDOR", str(Path(__file__).resolve().parent.parent / "vendor"))
)
sys.path.insert(0, str(_RIA_DIR))
import figstyle as fs  # noqa: E402
import roi_selector  # noqa: E402

# ─── ACS figure sizes ─────────────────────────────────────────────────────────
PANEL_W, PANEL_H = 7.0, 3.7  # double-column composite (in)
INDIV_W, INDIV_H = 3.33, 2.85  # single-column individual panels (in)
DPI = 600

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "figure.dpi": 150,
    }
)

SETS = [
    {
        "letters": "abcd",
        "analyte": "ferricyanide",
        "label": "Ferricyanide",
        "img_dir": IMG_ROOT / "ferricyanide" / "led_set1",
        "blank": "WaterIMG_260505_123740.dng",
        "channel": "B",
        "cmap_cv": "plasma",
        "block": 100,
        "win_h": 10,
        "win_w": 7,
    },
    {
        "letters": "efgh",
        "analyte": "methylene_blue",
        "label": "Methylene blue",
        "img_dir": IMG_ROOT / "methylene_blue" / "led_set1",
        "blank": "Blank_IMG_260526_123449.dng",
        "channel": "R",
        "cmap_cv": "plasma",
        "block": 100,
        "win_h": 10,
        "win_w": 7,
    },
]


def _save_fig(fig: plt.Figure, stem: str) -> None:
    """Save fig as PNG, SVG and TIFF (TIFF via PIL to avoid a large-canvas bug
    in matplotlib's native TIFF writer with bbox_inches='tight')."""
    png_path = OUT_DIR / f"{stem}.png"
    for ext, path in (("png", png_path), ("svg", OUT_DIR / f"{stem}.svg")):
        fig.savefig(path, dpi=DPI, bbox_inches="tight", format=ext)
    img = Image.open(png_path)
    img.save(OUT_DIR / f"{stem}.tiff", dpi=(DPI, DPI), compression="tiff_lzw")
    print(f"    -> {stem}.png / .svg / .tiff")


def run_roi_selection(cfg: dict) -> dict:
    """Re-run the roi_selector block-CV analysis for one acquisition set."""
    img_dir = cfg["img_dir"]
    channel = cfg["channel"]
    block = cfg["block"]

    blank_path = img_dir / cfg["blank"]
    plane_blank, info = roi_selector.load_channel_plane_raw(blank_path, channel)
    pitch = info["pitch"]
    BLOCK = max(1, block // pitch)
    H, W = plane_blank.shape
    extent = [0, W * pitch, H * pitch, 0]

    blank_mean_grid, blank_cv_grid = roi_selector.block_stats(plane_blank, BLOCK)

    sample_files = [
        f
        for f in roi_selector.images.list_images(img_dir)
        if f.name != blank_path.name and f.suffix.lower() == ".dng"
    ]
    gH, gW = H // BLOCK, W // BLOCK
    sample_block_means = np.zeros((len(sample_files), gH, gW))
    for i, f in enumerate(sample_files):
        plane, _ = roi_selector.load_channel_plane_raw(f, channel)
        for gi in range(gH):
            for gj in range(gW):
                s = plane[gi * BLOCK : (gi + 1) * BLOCK, gj * BLOCK : (gj + 1) * BLOCK].mean()
                bl = blank_mean_grid[gi, gj]
                sample_block_means[i, gi, gj] = s / max(bl, 1e-9)

    signal_cv_grid = sample_block_means.std(axis=0) / (sample_block_means.mean(axis=0) + 1e-9) * 100

    combined = blank_cv_grid / blank_cv_grid.max() + signal_cv_grid / signal_cv_grid.max()
    area = cfg["win_h"] * cfg["win_w"]
    (win_h, win_w), (r0, c0), best_val = roi_selector.best_window_optimal_aspect(combined, area)

    sx, sy = c0 * BLOCK * pitch, r0 * BLOCK * pitch
    sw, sh = win_w * BLOCK * pitch, win_h * BLOCK * pitch
    roi_sensor = (sx, sy, sw, sh)
    roi_viewer = roi_selector.images.bayer_roi_to_viewer(
        roi_sensor, info["flip"], info["bayer_shape"]
    )

    mid_file = sample_files[len(sample_files) // 2]
    rgb = roi_selector.render_viewer_image(mid_file)

    print(
        f"    [{cfg['label']}] channel {channel}: blank CV "
        f"{blank_cv_grid.min():.2f}-{blank_cv_grid.max():.2f}%, "
        f"signal CV {signal_cv_grid.min():.2f}-{signal_cv_grid.max():.2f}%, "
        f"window {win_h}x{win_w} (area={area}, score={best_val:.4f}), "
        f"ROI (viewer) {roi_viewer}"
    )

    return dict(
        blank_cv=blank_cv_grid,
        signal_cv=signal_cv_grid,
        combined=combined,
        extent=extent,
        roi_sensor=roi_sensor,
        roi_viewer=roi_viewer,
        rgb=rgb,
        block_px=BLOCK * pitch,
        best_val=best_val,
        gH=gH,
        gW=gW,
    )


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.04,
        0.96,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(boxstyle="square,pad=0.15", fc="black", alpha=0.55, ec="none"),
    )


def draw_real_image(ax: plt.Axes, res: dict, label: str, letter: str) -> None:
    vx, vy, vw, vh = res["roi_viewer"]
    img = Image.fromarray(res["rgb"])
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [vx, vy, vx + vw, vy + vh],
        outline=tuple(int(255 * v) for v in to_rgb(fs.FIT_COLOR)),
        width=8,
    )
    ax.imshow(np.asarray(img))
    ax.set_xlim(max(0, vx - vw * 0.4), min(img.width, vx + vw * 1.4))
    ax.set_ylim(min(img.height, vy + vh * 1.4), max(0, vy - vh * 0.4))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{label} ROI", fontsize=7.5, pad=3)
    _panel_letter(ax, letter)


def draw_cv_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    extent: list,
    cmap: str,
    label: str,
    letter: str,
    roi_sensor: tuple | None = None,
    diverging: bool = False,
    ylabel: bool = True,
) -> None:
    im = ax.imshow(data, cmap=cmap, aspect="auto", extent=extent)
    cb = plt.colorbar(im, ax=ax, fraction=0.055, pad=0.10, aspect=14)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_xlabel("x (px)", fontsize=6.5)
    if ylabel:
        ax.set_ylabel("y (px)", fontsize=6.5)
    ax.set_title(label, fontsize=7.5, pad=3)
    ax.tick_params(labelsize=5.5)
    if roi_sensor is not None:
        rx, ry, rw, rh = roi_sensor
        rect = plt.Rectangle(
            (rx, ry), rw, rh, linewidth=1.4, edgecolor=fs.FIT_COLOR, facecolor="none"
        )
        ax.add_patch(rect)
    _panel_letter(ax, letter)


def build_csv(all_results: dict) -> pd.DataFrame:
    rows = []
    for key, res in all_results.items():
        gH, gW = res["gH"], res["gW"]
        for gi in range(gH):
            for gj in range(gW):
                rows.append(
                    {
                        "analyte": key,
                        "block_row": gi,
                        "block_col": gj,
                        "x_px": gj * res["block_px"],
                        "y_px": gi * res["block_px"],
                        "blank_cv_pct": res["blank_cv"][gi, gj],
                        "signal_cv_pct": res["signal_cv"][gi, gj],
                        "combined_score_norm": res["combined"][gi, gj],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    print(f"[figure_S1] Output -> {OUT_DIR}\n")

    all_results = {}
    for cfg in SETS:
        print(f"Running ROI selection for {cfg['label']} ({cfg['img_dir'].name}) ...")
        all_results[cfg["analyte"]] = run_roi_selection(cfg)

    print("\nBuilding composite panel ...")
    fig, axes = plt.subplots(
        2, 4, figsize=(PANEL_W, PANEL_H), gridspec_kw={"hspace": 0.65, "wspace": 0.75}
    )

    for row, cfg in enumerate(SETS):
        res = all_results[cfg["analyte"]]
        letters = cfg["letters"]
        draw_real_image(axes[row, 0], res, cfg["label"], letters[0])
        draw_cv_heatmap(
            axes[row, 1],
            res["blank_cv"],
            res["extent"],
            cfg["cmap_cv"],
            f"Blank CV ({cfg['channel']})",
            letters[1],
        )
        draw_cv_heatmap(
            axes[row, 2],
            res["signal_cv"],
            res["extent"],
            cfg["cmap_cv"],
            f"Signal CV ({cfg['channel']})",
            letters[2],
        )
        draw_cv_heatmap(
            axes[row, 3],
            res["combined"],
            res["extent"],
            "RdYlGn_r",
            "Combined score",
            letters[3],
            roi_sensor=res["roi_sensor"],
            diverging=True,
        )

    _save_fig(fig, "figure_S1_panel")
    plt.close(fig)

    print("\nSaving individual panels ...")
    for cfg in SETS:
        res = all_results[cfg["analyte"]]
        letters = cfg["letters"]

        fig_a, ax_a = plt.subplots(figsize=(INDIV_W, INDIV_H))
        draw_real_image(ax_a, res, cfg["label"], letters[0])
        fig_a.tight_layout(pad=0.5)
        _save_fig(fig_a, f"figure_S1{letters[0]}")
        plt.close(fig_a)

        fig_b, ax_b = plt.subplots(figsize=(INDIV_W, INDIV_H))
        draw_cv_heatmap(
            ax_b,
            res["blank_cv"],
            res["extent"],
            cfg["cmap_cv"],
            f"{cfg['label']} — blank CV ({cfg['channel']})",
            letters[1],
        )
        fig_b.tight_layout(pad=0.5)
        _save_fig(fig_b, f"figure_S1{letters[1]}")
        plt.close(fig_b)

        fig_c, ax_c = plt.subplots(figsize=(INDIV_W, INDIV_H))
        draw_cv_heatmap(
            ax_c,
            res["signal_cv"],
            res["extent"],
            cfg["cmap_cv"],
            f"{cfg['label']} — signal CV ({cfg['channel']})",
            letters[2],
        )
        fig_c.tight_layout(pad=0.5)
        _save_fig(fig_c, f"figure_S1{letters[2]}")
        plt.close(fig_c)

        fig_d, ax_d = plt.subplots(figsize=(INDIV_W, INDIV_H))
        draw_cv_heatmap(
            ax_d,
            res["combined"],
            res["extent"],
            "RdYlGn_r",
            f"{cfg['label']} — combined score",
            letters[3],
            roi_sensor=res["roi_sensor"],
            diverging=True,
        )
        fig_d.tight_layout(pad=0.5)
        _save_fig(fig_d, f"figure_S1{letters[3]}")
        plt.close(fig_d)

    print("\nWriting data CSV ...")
    df = build_csv(all_results)
    csv_path = OUT_DIR / "figure_S1_data.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"    -> figure_S1_data.csv  ({len(df.columns)} columns, {len(df)} rows)")

    print(f"\n[figure_S1] Done.\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
