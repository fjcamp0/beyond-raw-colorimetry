"""
table_S4.py — Full calibration table, ferricyanide sRGB channels, ACS
single-column table figure.

All three sRGB channels (R, G, B), mean +/- SD across n = 3 acquisition
sets, at each of the 11 concentrations in the LED series.
Source: Data/analysis/ferricyanide/beer_lambert/sRGB/calibration_table.csv

Run with: python scripts/table_S4.py (from the project root).

Outputs -> Data/paper_figures/table_S4/
    table_S4.{png,svg,tiff}   table image
    table_S4_data.csv         wide-format table (one row per concentration)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from numfmt import pm  # CLAUDE.md hard rule 10: SD to two significant figures
from PIL import Image  # noqa: E402

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJ_ROOT / "Data"
# 2026-10-01 (Stephania): the sRGB arm is now the phone's native camera-app JPEG,
# pixel values as saved (scripts/native_jpeg_arm.py); the rawpy-rendered PNGs are no
# longer used anywhere in the paper. Previous outputs: superseded/2026-10-01_rawpy_srgb/.
SRC_CSV = (
    DATA_ROOT
    / "analysis"
    / "ferricyanide"
    / "beer_lambert"
    / "native_jpeg"
    / "calibration_table.csv"
)
OUT_DIR = DATA_ROOT / "paper_figures" / "table_S4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL_ORDER = ["R", "G", "B", "BdivG"]
CHANNEL_LABELS = {"R": "A_R", "G": "A_G", "B": "A_B", "BdivG": "A_B/G"}

DPI = 600
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "figure.dpi": 150,
    }
)


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format calibration table into one 'mean +/- SD' cell
    per (concentration, channel)."""
    df = df.copy()
    df["cell"] = df.apply(lambda r: pm(r.absorbance_mean, r.absorbance_std), axis=1)
    wide = df.pivot(index="concentration_um", columns="channel", values="cell")
    wide = wide[CHANNEL_ORDER].sort_index()
    n_rep = df.groupby("concentration_um")["n_replicates"].first().reindex(wide.index)
    wide.insert(0, "n", n_rep.astype(int))
    wide.index.name = "[c] (uM)"
    return wide


def draw_table(wide: pd.DataFrame) -> plt.Figure:
    n_rows = len(wide) + 1
    fig_h = 0.28 * n_rows + 0.4
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    ax.axis("off")

    col_labels = ["[c] (µM)", "n"] + [CHANNEL_LABELS[c] for c in CHANNEL_ORDER]
    cell_text = []
    for c_um, row in wide.iterrows():
        cell_text.append([f"{c_um:g}", str(row["n"])] + [row[c] for c in CHANNEL_ORDER])

    tbl = ax.table(
        cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center", colLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    tbl.scale(1.0, 1.35)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#D6E4F0")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F2F2")

    fig.suptitle(
        "Table S4. Ferricyanide native camera-app JPEG calibration "
        "(mean ± SD, n = 3 replicate images)",
        fontsize=8,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


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
    print(f"[table_S4] Output -> {OUT_DIR}\n")
    df = pd.read_csv(SRC_CSV)
    wide = build_table(df)

    print("Building table figure ...")
    fig = draw_table(wide)
    _save_fig(fig, "table_S4")
    plt.close(fig)

    csv_path = OUT_DIR / "table_S4_data.csv"
    wide.to_csv(csv_path)
    print(f"    -> table_S4_data.csv  ({len(wide.columns)} columns, {len(wide)} rows)")

    print(f"\n[table_S4] Done.\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
