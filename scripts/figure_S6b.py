# Copyright 2024 F. Javier del Campo
# Licensed under the Apache License, Version 2.0

"""
figure_S6b.py — Supplementary Figure S7b: channel-common component origin check.

Companion to Figure S7 (channel-common absorbance component diagnostics): where S7
establishes that the component exists and is shared across channels, S7b tests its two
candidate physical origins (section 4.3) against the existing ferricyanide LED data —
temporal drift and lens vignetting coupled to positional shifts.

Reads `Data/analysis/ferricyanide/channel_common_origin/channel_common_origin_data.csv`
(from `scripts/channel_common_origin_check.py`; run that script first).

Panel (a): channel-common residual vs. blank-to-sample capture time gap, per set.
Panel (b): center-ROI vs. edge-ROI residual, paired scatter + SD comparison.

Outputs (canonical + drafting copy, kept byte-identical):
    Data/paper_figures/figure_S6b/figure_S6b_panel.{png,svg,tiff}
    Manuscript/_drafts/figures/figure_S6b/figure_S6b_panel.{png,svg,tiff}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DATA_CSV = (
    BASE
    / "Data"
    / "analysis"
    / "ferricyanide"
    / "channel_common_origin"
    / "channel_common_origin_data.csv"
)
OUT_DIRS = [
    BASE / "Data" / "paper_figures" / "figure_S6b",
    BASE / "Manuscript" / "_drafts" / "figures" / "figure_S6b",
]

MARKERS = {"led_set1": "o", "led_set2": "s", "led_set3": "^"}
SET_LABELS = {"led_set1": "Set 1", "led_set2": "Set 2", "led_set3": "Set 3"}


def main() -> None:
    """Build the two-panel origin-check figure and write it to both figure folders."""
    df = pd.read_csv(DATA_CSV)

    valid1 = df.dropna(subset=["gap_seconds", "residual_full"])

    valid2 = df.dropna(subset=["residual_center", "residual_edge"])
    sd_center = float(valid2["residual_center"].std(ddof=1))
    sd_edge = float(valid2["residual_edge"].std(ddof=1))
    r_ce, _ = stats.pearsonr(valid2["residual_center"], valid2["residual_edge"])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))

    ax = axes[0]
    for s, grp in valid1.groupby("set"):
        set_color = fs.SET_COLORS[list(MARKERS).index(s)] if s in MARKERS else fs.GRAY
        ax.scatter(
            grp["gap_seconds"] / 60.0,
            grp["residual_full"],
            marker=MARKERS.get(s, "x"),
            label=SET_LABELS.get(s, s),
            s=28,
            alpha=0.85,
            color=set_color,
        )
    ax.axhline(0, color=fs.GRAY, lw=0.7, ls="--")
    ax.set_xlabel("Blank→sample time gap (min)", fontsize=8)
    ax.set_ylabel("Channel-common residual (AU)", fontsize=8)
    ax.set_title("(a) Temporal drift", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="best")

    ax = axes[1]
    ax.scatter(
        valid2["residual_center"],
        valid2["residual_edge"],
        s=24,
        alpha=0.8,
        color=fs.ROUTE_COLORS["raw"][0],
    )
    lim = max(valid2["residual_center"].abs().max(), valid2["residual_edge"].abs().max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color=fs.GRAY, lw=0.7, ls="--")
    ax.set_xlabel("Center-ROI residual (AU)", fontsize=8)
    ax.set_ylabel("Edge-ROI residual (AU)", fontsize=8)
    ax.set_title(
        f"(b) Radial position  r = {r_ce:.3f}\nSD: center {sd_center:.4f}, edge {sd_edge:.4f}",
        fontsize=8,
    )
    ax.tick_params(labelsize=7)

    fig.tight_layout()

    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "svg", "tiff"):
            path = out_dir / f"figure_S6b_panel.{ext}"
            fig.savefig(path, dpi=300 if ext != "svg" else None)
            print(f"[figure_S6b] Written: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
