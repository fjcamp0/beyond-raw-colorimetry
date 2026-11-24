"""
table_S5.py — Full calibration table, methylene blue RAW channels, ACS
single-column table figure.

All eight RAW-derived channels (R, G1, G2, B, BdivG, GmB, RdivG, RdivB),
mean +/- SD across n = 3 acquisition sets, at each of the 8 concentrations
in the LED series. R, G1, G2, B, BdivG, GmB come directly from
Data/analysis/methylene_blue/beer_lambert/calibration_table.csv; RdivG and
RdivB (the recommended MB quantification channels — see
methylene_blue_calibration_report.md sec 6) are computed here from the
per-image absorbance_table.csv, following the same
-log10((ch/ref)_sample / (ch/ref)_blank) definition used for BdivG.

Run with: python scripts/table_S5.py (from the project root).

Outputs -> Data/paper_figures/table_S5/
    table_S5.{png,svg,tiff}   table image
    table_S5_data.csv         wide-format table (one row per concentration)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from numfmt import pm  # CLAUDE.md hard rule 10: SD to two significant figures
from PIL import Image  # noqa: E402

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJ_ROOT / "Data"
BL_DIR = DATA_ROOT / "analysis" / "methylene_blue" / "beer_lambert"
CALIB_CSV = BL_DIR / "calibration_table.csv"
ABS_CSV = BL_DIR / "absorbance_table.csv"
OUT_DIR = DATA_ROOT / "paper_figures" / "table_S5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL_ORDER = ["R", "G1", "G2", "B", "BdivG", "GmB", "RdivG", "RdivB"]
CHANNEL_LABELS = {
    "R": "A_R",
    "G1": "A_G1",
    "G2": "A_G2",
    "B": "A_B",
    "BdivG": "A_B/G",
    "GmB": "A_G-B",
    "RdivG": "A_R/G",
    "RdivB": "A_R/B",
}

DPI = 600
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "figure.dpi": 150,
    }
)


def compute_ratio_channels(abs_df: pd.DataFrame) -> pd.DataFrame:
    """Derive A_RdivG and A_RdivB per image from the per-channel raw/blank
    means, matching the -log10(ratio_sample / ratio_blank) definition used
    for A_BdivG elsewhere in the pipeline."""
    raw = abs_df.pivot_table(
        index=["filename", "set", "concentration_um"], columns="channel", values="raw_mean"
    )
    blank = abs_df.pivot_table(
        index=["filename", "set", "concentration_um"], columns="channel", values="blank_mean"
    )

    g_sample = (raw["G1"] + raw["G2"]) / 2
    g_blank = (blank["G1"] + blank["G2"]) / 2

    a_rdivg = -np.log10((raw["R"] / g_sample) / (blank["R"] / g_blank))
    a_rdivb = -np.log10((raw["R"] / raw["B"]) / (blank["R"] / blank["B"]))

    out = pd.DataFrame({"RdivG": a_rdivg, "RdivB": a_rdivb}).reset_index()
    return out


def build_table(calib_df: pd.DataFrame, abs_df: pd.DataFrame) -> pd.DataFrame:
    """Combine the calibration_table.csv channels with the newly computed
    RdivG/RdivB channels into a single wide 'mean +/- SD' table."""
    calib_df = calib_df.copy()
    calib_df["cell"] = calib_df.apply(lambda r: pm(r.absorbance_mean, r.absorbance_std), axis=1)
    wide = calib_df.pivot(index="concentration_um", columns="channel", values="cell")

    ratios = compute_ratio_channels(abs_df)
    stats = ratios.groupby("concentration_um")[["RdivG", "RdivB"]].agg(["mean", "std"])
    for ch in ["RdivG", "RdivB"]:
        wide[ch] = stats[ch].apply(lambda r: pm(r["mean"], r["std"]), axis=1)

    wide = wide[CHANNEL_ORDER].sort_index()
    n_rep = calib_df.groupby("concentration_um")["n_replicates"].first().reindex(wide.index)
    wide.insert(0, "n", n_rep.astype(int))
    wide.index.name = "[c] (uM)"
    return wide


def draw_table(wide: pd.DataFrame) -> plt.Figure:
    n_rows = len(wide) + 1
    fig_h = 0.28 * n_rows + 0.4
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
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
        "Table S5. Methylene blue RAW-channel calibration (mean ± SD, n = 3 acquisition sets)",
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
    print(f"[table_S5] Output -> {OUT_DIR}\n")
    calib_df = pd.read_csv(CALIB_CSV)
    abs_df = pd.read_csv(ABS_CSV)
    wide = build_table(calib_df, abs_df)

    print("Building table figure ...")
    fig = draw_table(wide)
    _save_fig(fig, "table_S5")
    plt.close(fig)

    csv_path = OUT_DIR / "table_S5_data.csv"
    wide.to_csv(csv_path)
    print(f"    -> table_S5_data.csv  ({len(wide.columns)} columns, {len(wide)} rows)")

    print(f"\n[table_S5] Done.\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
