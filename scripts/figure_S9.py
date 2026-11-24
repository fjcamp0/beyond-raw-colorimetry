"""Figure S9: forward model for the methylene blue R/G ratio (LED), 2026-10-07.

Stephania asked why Figures 6 and 7 show a ratio channel for ferricyanide but not
for methylene blue. This SI figure adds the R/G counterpart of Figure 6c,f and 7c:
(a) model against measured R/G over the full range, (b) the same up to 25 uM,
(c) measured and model-corrected R/G against the low-concentration Beer-Lambert line.
The LED forward model table was computed with l_cam fitted on R (2.41 mm). As for
ferricyanide B/G, l_cam is refitted here on the ratio itself, over 0.78-100 uM; since the
model depends only on c * l, this rescales the concentration axis by k = l'/l. The
inversion and Beer-Lambert slope follow figure_7.py (fit through the origin, <= 50 uM).
"""

from pathlib import Path

import figstyle as fs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_7 import bl_slope_origin, model_correct

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "Data/analysis/methylene_blue/forward_model/forward_model_table.csv"
MEASURED = ROOT / "Data/paper_figures/table_S5/table_S5_data.csv"
OUT = ROOT / "Data/paper_figures/figure_S9/figure_S9"
DRAFTS = ROOT / "Manuscript/_drafts/figures/figure_S9"
LIN_MAX = 50.0
ZOOM_MAX = 25.0
LCAM_R_MM = 2.41
FIT_RANGE = (0.78, 100.0)


def measured():
    t = pd.read_csv(MEASURED)
    mean_sd = t["RdivG"].str.split("±", expand=True).astype(float)
    return pd.DataFrame({"c_uM": t["[c] (uM)"].astype(float), "A": mean_sd[0], "sd": mean_sd[1]})


def refit_lcam(model, data):
    """Scale factor k on c (k = l_cam'/l_cam) minimising the squared error on R/G means."""
    d = data[(data.c_uM >= FIT_RANGE[0]) & (data.c_uM <= FIT_RANGE[1])]
    ks = np.linspace(0.5, 1.5, 20001)
    sse = [
        np.sum((np.interp(d.c_uM * k, model.concentration_um, model.A_RdivG) - d.A) ** 2)
        for k in ks
    ]
    return ks[int(np.argmin(sse))]


def model_vs_data(ax, model, data, c_max):
    m = model[(model.concentration_um > 0) & (model.concentration_um <= c_max * 1.05)]
    ax.plot(m.concentration_um, m.A_RdivG, color=fs.MODEL_COLOR, label="BLB model")
    d = data[data.c_uM <= c_max]
    ax.errorbar(
        d.c_uM,
        d.A,
        yerr=d.sd,
        fmt="o",
        color=fs.ROUTE_COLORS["raw"][0],
        capsize=2,
        label="Measured (mean ± SD)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Concentration (µM)")
    ax.set_ylabel(r"$A_{R/G}$")


def main():
    fs.apply()
    model = pd.read_csv(MODEL).sort_values("concentration_um")
    data = measured()
    k = refit_lcam(model, data)
    print(f"l_cam refitted on R/G: {LCAM_R_MM * k:.2f} mm (k = {k:.4f})")
    model = model.assign(concentration_um=model.concentration_um / k)
    slope = bl_slope_origin(data.c_uM.values, data.A.values, LIN_MAX)
    corr = model_correct(data.A.values, model.concentration_um.values, model.A_RdivG.values, slope)

    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE_COL, 2.6), layout="constrained")
    model_vs_data(axes[0], model, data, data.c_uM.max())
    model_vs_data(axes[1], model, data, ZOOM_MAX)
    axes[0].legend(loc="upper left")

    ax = axes[2]
    c_ref = np.array([0, data.c_uM.max() * 1.05])
    ax.plot(c_ref, slope * c_ref * 1e-6, color=fs.GRAY, ls="--", label="Beer-Lambert (low-c)")
    ax.errorbar(
        data.c_uM,
        data.A,
        yerr=data.sd,
        fmt="o",
        mfc="white",
        color=fs.ROUTE_COLORS["raw"][2],
        capsize=2,
        label="Measured",
    )
    ax.errorbar(
        data.c_uM,
        corr,
        yerr=data.sd,
        fmt="s",
        color=fs.ROUTE_COLORS["raw"][0],
        capsize=2,
        label="Model-corrected",
    )
    ax.set_xlim(0, data.c_uM.max() * 1.05)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("[MB] (µM)")
    ax.set_ylabel(r"$A_{R/G}$")
    ax.legend(loc="upper left")
    for a, letter in zip(axes, "abc"):
        fs.panel_label(a, f"({letter})")

    out = pd.DataFrame(
        {
            "concentration_um": data.c_uM,
            "A_RdivG_mean": data.A,
            "A_RdivG_sd": data.sd,
            "A_RdivG_model": np.interp(data.c_uM, model.concentration_um, model.A_RdivG),
            "A_RdivG_corrected": corr,
            "A_BL_expected": slope * data.c_uM * 1e-6,
        }
    )
    fs.save(fig, OUT, data=out)
    DRAFTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "csv"):
        (DRAFTS / f"figure_S9.{ext}").write_bytes(OUT.with_suffix(f".{ext}").read_bytes())
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
