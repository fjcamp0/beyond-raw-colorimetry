"""Table of Contents graphic, 3.25 x 1.75 in (ACS Measurement Science Au).

Drawn entirely in matplotlib from the paper's own data: ACS does not allow
AI-generated images in TOC graphics. Left: the transmission light path
(light bulb, cuvette, RGB camera with Bayer mosaic). Right: methylene blue R-channel
calibration from RAW and from native camera-app JPEG images (Figure 4e
data, 0.78-50 uM): RAW is linear, JPEG is not.

Run from the project root: python scripts/toc_graphic.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import figstyle as fs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
FIG4 = ROOT / "Data" / "paper_figures" / "figure_dominant_routes"  # from figure 4 script
OUT = ROOT / "Data" / "paper_figures" / "toc" / "toc_graphic"
DRAFTS = ROOT / "Manuscript" / "_drafts" / "figures" / "toc"

TOC_W, TOC_H = 3.25, 1.75
C_MAX = 50.0
LIQUID = "#4A7FC1"
RAW_COLOR = fs.ROUTE_COLORS["raw"][1]
JPEG_COLOR = fs.ROUTE_COLORS["jpeg"][1]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    pts = pd.read_csv(FIG4 / "figure_4_dominant_routes.csv")
    fits = pd.read_csv(FIG4 / "figure_4_dominant_routes_fits.csv")
    keep = (
        (pts["analyte"] == "methylene_blue")
        & pts["series"].isin(["RAW R", "JPEG R"])
        & (pts["concentration_um"] <= C_MAX)
    )
    fit = fits[(fits["analyte"] == "methylene_blue") & fits["series"].isin(["RAW R", "JPEG R"])]
    return pts[keep], fit


def draw_bulb(ax, x: float, y: float, edge: dict) -> None:
    """Light bulb centred on the glass at (x, y), base below, short rays."""
    r = 0.12
    ax.add_patch(Circle((x, y), r, facecolor="#FFE680", **edge))
    ax.add_patch(Rectangle((x - 0.055, y - r - 0.12), 0.11, 0.13, facecolor="#BFBFBF", **edge))
    for k in (1, 2):
        yk = y - r - 0.12 + k * 0.043
        ax.plot([x - 0.055, x + 0.055], [yk, yk], color=fs.FIT_COLOR, linewidth=0.6)
    for ang in np.radians([90, 135, 180, 45]):
        r0, r1 = r + 0.04, r + 0.10
        ax.plot(
            [x + r0 * np.cos(ang), x + r1 * np.cos(ang)],
            [y + r0 * np.sin(ang), y + r1 * np.sin(ang)],
            color="#E0A800",
            linewidth=0.9,
            solid_capstyle="round",
        )


def draw_light_path(fig) -> None:
    ax = fig.add_axes([0.0, 0.0, 0.40, 0.80])
    ax.set_xlim(-0.14, 1.30)
    ax.set_ylim(0, 1.40)
    ax.set_axis_off()
    edge = dict(edgecolor=fs.FIT_COLOR, linewidth=0.8)

    draw_bulb(ax, 0.15, 0.80, edge)
    ax.add_patch(Rectangle((0.50, 0.50), 0.17, 0.55, facecolor="white", **edge))
    ax.add_patch(Rectangle((0.50, 0.50), 0.17, 0.38, facecolor=LIQUID, edgecolor="none"))
    ax.add_patch(
        FancyBboxPatch(
            (0.92, 0.35),
            0.30,
            0.85,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor="#E8E8E8",
            **edge,
        )
    )
    size = 0.09
    mosaic = (("R", 0, 1), ("G", 1, 1), ("G", 0, 0), ("B", 1, 0))
    for ch, i, j in mosaic:
        ax.add_patch(
            Rectangle(
                (0.98 + i * size, 0.68 + j * size),
                size,
                size,
                facecolor=fs.CHANNEL_COLORS[ch],
                edgecolor="white",
                linewidth=0.6,
            )
        )

    for x0, x1 in ((0.26, 0.48), (0.69, 0.90)):
        ax.add_patch(
            FancyArrowPatch(
                (x0, 0.77),
                (x1, 0.77),
                arrowstyle="-|>",
                mutation_scale=7,
                color=fs.GRAY,
                linewidth=0.8,
            )
        )

    labels = ((0.15, "light"), (0.585, "sample"), (1.07, "RGB\ncamera"))
    for x, label in labels:
        ax.text(x, 0.28, label, ha="center", va="top", fontsize=fs.FONT_SMALL, linespacing=1.0)


def draw_calibration(fig, pts: pd.DataFrame, fit: pd.DataFrame) -> None:
    ax = fig.add_axes([0.58, 0.25, 0.39, 0.53])
    styles = {
        "RAW R": dict(marker="o", color=RAW_COLOR, mfc=RAW_COLOR, ls="--", label="RAW"),
        "JPEG R": dict(marker="s", color=JPEG_COLOR, mfc="white", ls=":", label="JPEG"),
    }
    for series, st in styles.items():
        d = pts[pts["series"] == series]
        f = fit[fit["series"] == series].iloc[0]
        x = np.linspace(f["range_lo"], f["range_hi"], 50)
        ax.plot(
            x,
            f["slope"] * x + f["intercept"],
            color=st["color"],
            linestyle=st["ls"],
            linewidth=0.9,
            zorder=1,
        )
        ax.plot(
            d["concentration_um"],
            d["absorbance_mean"],
            st["marker"],
            markersize=3.4,
            color=st["color"],
            markerfacecolor=st["mfc"],
            markeredgewidth=0.8,
            zorder=2,
            label=st["label"],
        )
    ax.set_xlim(-2, 55)
    ax.set_ylim(-0.02, 0.45)
    ax.set_xticks([0, 25, 50])
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_xlabel(r"methylene blue ($\mu$M)", fontsize=fs.FONT_SMALL, labelpad=1)
    ax.set_ylabel("A (R channel)", fontsize=fs.FONT_SMALL, labelpad=1)
    ax.tick_params(labelsize=fs.FONT_SMALL, pad=1.5)
    ax.legend(
        loc="upper left",
        fontsize=fs.FONT_SMALL,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        borderaxespad=0.1,
        labelspacing=0.2,
    )


def main() -> None:
    fs.apply()
    pts, fit = load()
    fig = plt.figure(figsize=(TOC_W, TOC_H))
    fig.text(
        0.5,
        0.97,
        "Image-based colorimetry: unprocessed RAW vs processed JPEG",
        ha="center",
        va="top",
        fontsize=fs.MIN_FONT,
        fontweight="bold",
        color=fs.FIT_COLOR,
    )
    draw_light_path(fig)
    draw_calibration(fig, pts, fit)
    fs.save(fig, OUT, data=pts, formats=("png", "tiff", "svg", "pdf"))
    DRAFTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "tiff"):
        shutil.copy2(OUT.with_suffix(f".{ext}"), DRAFTS / f"toc_graphic.{ext}")


if __name__ == "__main__":
    main()
