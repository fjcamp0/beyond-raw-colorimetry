"""
figure_S4.py - RAW vs sRGB comparison, ferricyanide, four Bayer channels.

Produces the 2x2 ACS panel: R, B, G1-vs-G, G2-vs-G, each overlaying RAW (dark,
dashed fit) and sRGB (light, dotted fit) absorbance vs. concentration. Source
content: RAW_vs_sRGB.md / comparison_all_channels.png (exploratory precursor).

Was main-text Figure 2 at an earlier numbering; moved to the Supporting
Information at round 2e Stage 5 (2026-07-28, W5) and currently renders as
Supplementary Figure S4. Renamed from figure_2.py 2026-07-31 (collided with
the unrelated main-text Figure 2 folder) then from figure_raw_vs_srgb.py
2026-07-31 as part of the SI rename pass, to match the rendered number.

Run with: python scripts/figure_S4.py (from the project root).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import linregress  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import figstyle as fs  # noqa: E402

PROJ_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJ_ROOT / "Data"

RAW_CSV = DATA_DIR / "analysis" / "ferricyanide" / "beer_lambert" / "calibration_table.csv"
# 2026-10-01 (Stephania): the sRGB arm is now the phone's native camera-app JPEG,
# pixel values as saved (scripts/native_jpeg_arm.py); the rawpy-rendered PNGs are no
# longer used anywhere in the paper. Previous outputs: superseded/2026-10-01_rawpy_srgb/.
SRGB_CSV = (
    DATA_DIR
    / "analysis"
    / "ferricyanide"
    / "beer_lambert"
    / "native_jpeg"
    / "calibration_table.csv"
)
OUT_DIR = DATA_DIR / "paper_figures" / "figure_S4"
# assemble_manuscript.py reads figures only from here, never from Data/paper_figures/.
DRAFTS_OUT_DIR = PROJ_ROOT / "Manuscript" / "_drafts" / "figures" / "figure_S4"

ACS_W = 7.0
ACS_H = 6.2
DPI = 600

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "figure.dpi": 150,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
    }
)

COL_RAW = fs.ROUTE_COLORS["raw"][0]
COL_SRGB = fs.ROUTE_COLORS["jpeg"][1]

# (panel label, RAW channel, sRGB channel, title)
PANELS = [
    ("a", "R", "R", "R channel"),
    ("b", "B", "B", "B channel"),
    ("c", "G1", "G", "G$_1$ (RAW) vs G (JPEG)"),
    ("d", "G2", "G", "G$_2$ (RAW) vs G (JPEG)"),
]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "concentration_um",
        "channel",
        "absorbance_mean",
        "absorbance_std",
        "n_replicates",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def series(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    return df[df["channel"] == channel].sort_values("concentration_um")


def plot_series(ax, sub, color, marker, ls, label):
    x = sub["concentration_um"].to_numpy(float)
    y = sub["absorbance_mean"].to_numpy(float)
    yerr = sub["absorbance_std"].to_numpy(float)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt=marker,
        color=color,
        label=label,
        elinewidth=0.7,
        capsize=2,
        capthick=0.7,
        zorder=4,
    )
    if len(x) >= 3:
        slope, intercept, r, _, _ = linregress(x, y)
        xfit = np.linspace(0, x.max(), 400)
        ax.plot(xfit, slope * xfit + intercept, color=color, lw=0.8, ls=ls, zorder=3)
    return x, y, yerr


def build_figure(df_raw: pd.DataFrame, df_srgb: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(ACS_W, ACS_H))
    csv_rows = []

    for (label, ch_raw, ch_srgb, title), ax in zip(PANELS, axes.flat):
        raw_sub = series(df_raw, ch_raw)
        srgb_sub = series(df_srgb, ch_srgb)

        x_raw, y_raw, e_raw = plot_series(ax, raw_sub, COL_RAW, "o", "--", "RAW DNG")
        x_srgb, y_srgb, e_srgb = plot_series(ax, srgb_sub, COL_SRGB, "s", ":", "JPEG (camera app)")

        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Concentration (µM)")
        ax.set_ylabel("Absorbance (AU)")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.90),
            handlelength=1.5,
            handletextpad=0.4,
            borderpad=0.3,
        )
        ax.text(
            0.03,
            0.97,
            f"({label})",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
        )

        for src, x, y, e, n_col in (
            ("RAW", x_raw, y_raw, e_raw, ch_raw),
            ("JPEG", x_srgb, y_srgb, e_srgb, ch_srgb),
        ):
            csv_rows.append(
                pd.DataFrame(
                    {
                        "panel": label,
                        "source": src,
                        "channel": n_col,
                        "concentration_um": x,
                        "absorbance_mean": y,
                        "absorbance_std": e,
                    }
                )
            )

    fig.tight_layout(pad=0.6)
    return fig, pd.concat(csv_rows, ignore_index=True)


def save(fig: plt.Figure, csv_df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "tiff", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight", format=ext)
    csv_df.to_csv(out_dir / f"{stem}.csv", index=False, float_format="%.6g")

    if out_dir != DRAFTS_OUT_DIR:
        DRAFTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "tiff", "svg", "csv"):
            shutil.copy2(out_dir / f"{stem}.{ext}", DRAFTS_OUT_DIR / f"{stem}.{ext}")


def main() -> None:
    df_raw = load(RAW_CSV)
    df_srgb = load(SRGB_CSV)
    fig, csv_df = build_figure(df_raw, df_srgb)
    save(fig, csv_df, OUT_DIR, "figure_S4_raw_vs_jpeg")
    plt.close(fig)
    print(f"[figure_S4] Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
