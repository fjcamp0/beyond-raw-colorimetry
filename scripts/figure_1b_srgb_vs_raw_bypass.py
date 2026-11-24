"""Schematic (box/flow diagram): why sRGB pixel values cannot be corrected
back to linear photometric data, motivating RAW DNG capture (manuscript §1).

Two diverging pipelines from a shared sensor-plane starting point:
  - sRGB/JPEG path: white balance -> gamma encoding -> 8-bit quantization,
    ending in a blocked state (systematic, concentration-dependent distortion).
  - RAW DNG path: no white balance, no gamma, native Bayer-plane values,
    ending in a valid state (linear in detected irradiance).

Run with any interpreter that has the requirements installed:
    python scripts/figure_1b_srgb_vs_raw_bypass.py
"""

import csv
from pathlib import Path

import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.7,
        "figure.dpi": 150,
    }
)

DOUBLE_COL = 7.00  # in, ACS double-column max
OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "Data"
    / "paper_figures"
    / "figure_1b_srgb_vs_raw_bypass"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRAY = fs.GRAY
FIT_COLOR = fs.FIT_COLOR
FAIL_COLOR = fs.ROUTE_COLORS["jpeg"][1]  # JPEG orange
OK_COLOR = fs.ROUTE_COLORS["raw"][1]  # RAW blue
FAIL_FILL = "#FDEBDD"  # light tint of the JPEG orange
OK_FILL = "#E3EEF8"  # light tint of the RAW blue
FAIL_FILL_DARK = "#FBD2B3"  # stronger tint, end box
OK_FILL_DARK = "#BCD6EC"  # stronger tint, end box
NEUTRAL_FILL = "#F2F2F2"


def box(ax, x, y, w, h, text, edgecolor=GRAY, facecolor=NEUTRAL_FILL, fontsize=7, weight="normal"):
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.06,rounding_size=0.10",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(p)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1B2631",
        weight=weight,
        zorder=4,
        linespacing=1.35,
    )
    return p


def arrow(ax, xy1, xy2, color=FIT_COLOR, lw=0.9, style="-|>"):
    a = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle=style,
        mutation_scale=9,
        linewidth=lw,
        color=color,
        zorder=2,
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.05))
    ax.set_xlim(0, 10)
    ax.set_ylim(3.55, 13.9)
    ax.axis("off")

    boxes = []  # (id, column, order, x, y, w, h, text)

    # Shared starting point
    start = dict(
        id="start",
        col="shared",
        order=0,
        x=5.0,
        y=13.2,
        w=6.4,
        h=1.15,
        text="Sensor / Bayer plane\n(linear pixel counts $\\propto$ photon flux, minus black level)",
    )
    boxes.append(start)

    # Left column = sRGB / JPEG path
    L1 = dict(
        id="L1",
        col="sRGB",
        order=1,
        x=2.35,
        y=11.35,
        w=4.3,
        h=1.05,
        text="White balance applied\n$p_{\\mathrm{processed}} = WB_{ch}\\times p_{\\mathrm{raw}}$\n(session- and scene-dependent)",
    )
    L2 = dict(
        id="L2",
        col="sRGB",
        order=2,
        x=2.35,
        y=9.55,
        w=4.3,
        h=1.05,
        text="Gamma encoding\n$p_{\\mathrm{sRGB}} \\approx p_{\\mathrm{linear}}^{1/2.2}$ (IEC 61966-2-1)\nlinear toe below $p_{\\mathrm{linear}} \\approx 0.003$",
    )
    L3 = dict(
        id="L3",
        col="sRGB",
        order=3,
        x=2.35,
        y=7.85,
        w=4.3,
        h=0.95,
        text="8-bit quantization, demosaic,\nJPEG / PNG compression",
    )
    Fa = dict(
        id="Fa",
        col="sRGB",
        order=4,
        x=1.15,
        y=6.15,
        w=2.05,
        h=1.15,
        text="$\\mathbf{\\times}$  Gamma distortion is\nconcentration-dependent\n(no single rescaling factor corrects it)",
    )
    Fb = dict(
        id="Fb",
        col="sRGB",
        order=4,
        x=3.55,
        y=6.15,
        w=2.05,
        h=1.15,
        text="$\\mathbf{\\times}$  WB multiplier is baked in\n(drifts session-to-session,\nunrecoverable after 8-bit clipping)",
    )
    Lend = dict(
        id="Lend",
        col="sRGB",
        order=5,
        x=2.35,
        y=4.35,
        w=4.3,
        h=1.1,
        text="Beer-Lambert calibration\nsystematically distorted—\nnot correctable by rescaling",
    )

    # Right column = RAW DNG path
    R1 = dict(
        id="R1", col="RAW", order=1, x=7.65, y=11.35, w=4.3, h=1.05, text="No white balance applied"
    )
    R2 = dict(
        id="R2", col="RAW", order=2, x=7.65, y=9.55, w=4.3, h=1.05, text="No gamma encoding applied"
    )
    R3 = dict(
        id="R3",
        col="RAW",
        order=3,
        x=7.65,
        y=7.85,
        w=4.3,
        h=0.95,
        text="Native Bayer-plane pixel values\n(e.g. rawpy / LibRaw), 10-bit,\n$\\propto$ photon counts",
    )
    Rchk = dict(
        id="Rchk",
        col="RAW",
        order=4,
        x=7.65,
        y=6.15,
        w=4.3,
        h=1.15,
        text="$\\checkmark$  Pixel values remain proportional\nto detected irradiance at every stage",
    )
    Rend = dict(
        id="Rend",
        col="RAW",
        order=5,
        x=7.65,
        y=4.35,
        w=4.3,
        h=1.1,
        text="Pixel values remain linear in\ndetected irradiance—usable\ndirectly for Beer-Lambert absorbance",
    )

    boxes += [L1, L2, L3, Fa, Fb, Lend, R1, R2, R3, Rchk, Rend]

    # Draw shared start box
    box(
        ax,
        start["x"],
        start["y"],
        start["w"],
        start["h"],
        start["text"],
        edgecolor=FIT_COLOR,
        facecolor="#FFFFFF",
        fontsize=7.5,
        weight="bold",
    )

    # Draw sRGB (left) path
    box(ax, L1["x"], L1["y"], L1["w"], L1["h"], L1["text"])
    box(ax, L2["x"], L2["y"], L2["w"], L2["h"], L2["text"])
    box(ax, L3["x"], L3["y"], L3["w"], L3["h"], L3["text"])
    box(
        ax,
        Fa["x"],
        Fa["y"],
        Fa["w"],
        Fa["h"],
        Fa["text"],
        edgecolor=FAIL_COLOR,
        facecolor=FAIL_FILL,
        fontsize=6.4,
    )
    box(
        ax,
        Fb["x"],
        Fb["y"],
        Fb["w"],
        Fb["h"],
        Fb["text"],
        edgecolor=FAIL_COLOR,
        facecolor=FAIL_FILL,
        fontsize=6.4,
    )
    box(
        ax,
        Lend["x"],
        Lend["y"],
        Lend["w"],
        Lend["h"],
        Lend["text"],
        edgecolor=FAIL_COLOR,
        facecolor=FAIL_FILL_DARK,
        fontsize=7.3,
        weight="bold",
    )

    # Draw RAW (right) path
    box(ax, R1["x"], R1["y"], R1["w"], R1["h"], R1["text"])
    box(ax, R2["x"], R2["y"], R2["w"], R2["h"], R2["text"])
    box(ax, R3["x"], R3["y"], R3["w"], R3["h"], R3["text"])
    box(
        ax,
        Rchk["x"],
        Rchk["y"],
        Rchk["w"],
        Rchk["h"],
        Rchk["text"],
        edgecolor=OK_COLOR,
        facecolor=OK_FILL,
        fontsize=6.8,
    )
    box(
        ax,
        Rend["x"],
        Rend["y"],
        Rend["w"],
        Rend["h"],
        Rend["text"],
        edgecolor=OK_COLOR,
        facecolor=OK_FILL_DARK,
        fontsize=7.3,
        weight="bold",
    )

    # Arrows: start -> L1, start -> R1
    arrow(ax, (4.1, 12.65), (2.6, 11.9))
    arrow(ax, (5.9, 12.65), (7.4, 11.9))

    # Left column vertical chain
    arrow(ax, (2.35, 10.82), (2.35, 10.08))
    arrow(ax, (2.35, 9.02), (2.35, 8.33))
    arrow(ax, (1.9, 7.37), (1.35, 6.73))
    arrow(ax, (2.8, 7.37), (3.35, 6.73))
    arrow(ax, (1.35, 5.57), (2.0, 4.90))
    arrow(ax, (3.35, 5.57), (2.7, 4.90))

    # Right column vertical chain
    arrow(ax, (7.65, 10.82), (7.65, 10.08))
    arrow(ax, (7.65, 9.02), (7.65, 8.33))
    arrow(ax, (7.65, 7.37), (7.65, 6.73))
    arrow(ax, (7.65, 5.57), (7.65, 4.90))

    # Column headers
    ax.text(
        0.2,
        12.05,
        "sRGB / JPEG pipeline",
        ha="left",
        va="bottom",
        fontsize=7.5,
        weight="bold",
        color=FAIL_COLOR,
    )
    ax.text(
        9.8,
        12.05,
        "RAW DNG pipeline",
        ha="right",
        va="bottom",
        fontsize=7.5,
        weight="bold",
        color=OK_COLOR,
    )

    fig.tight_layout(pad=0.5)
    for ext in ("png", "svg"):
        fig.savefig(
            OUT_DIR / f"figure_1b_srgb_vs_raw_bypass.{ext}",
            dpi=600,
            bbox_inches="tight",
            format=ext,
        )
    plt.close(fig)

    # Companion "data" CSV: box contents (schematic, not a numeric plot)
    with open(OUT_DIR / "figure_1b_srgb_vs_raw_bypass.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["box_id", "column", "order", "text"])
        for b in boxes:
            w.writerow([b["id"], b["col"], b["order"], b["text"].replace("\n", " / ")])

    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
