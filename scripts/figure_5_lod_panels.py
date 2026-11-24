"""Figure 5e,f: ferricyanide B-channel calibrations and detection limits by route.

The LOD error bars are 1 SD of ln(LOD), propagated from the standard error of the
slope and from sigma's own scatter (relative SD 1/sqrt(2 dof), dof pooled over the
three lowest levels); they are drawn as LOD x/÷ exp(SD).
Every arm follows the Table S1 convention (bandpass_filter_metrics.fit, 7.8-500 uM;
B/G 7.8-1000 uM as in Table S1). UV-Vis rows are the measured replicate spectra,
at 420 nm, the absorption maximum, and camera-weighted over the B channel through the
450/40 filter, both with a per-spectrum baseline (table1_uvvis_band).
JPEG arms are native camera-app images, pixel values as saved (native_jpeg_arm).
"""

from pathlib import Path

import bandpass_filter_metrics as bfm
import figstyle as fs
import matplotlib.pyplot as plt
import native_jpeg_arm as nja
import numpy as np
import pandas as pd
import table1_uvvis_band as band

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Manuscript/_drafts/figures/figure_5/LED/figure_5ef_LED"
OUT_E, OUT_F = OUT.with_name("figure_5e_LED"), OUT.with_name("figure_5f_LED")
RNG, RNG_BG = (7.8, 500), (7.8, 1000)
JPEG_450 = ROOT / "Data/images/ferricyanide/Ferricyanide_with Filter_450nm_40nm_nativeapp"


def uvvis_points(phi=None, wavelength=None):
    s = pd.read_csv(band.REC / "uvvis_spectra_all.csv")
    s = s[s.analyte == "ferricyanide"]
    w = band.weights("B", phi)
    rows = []
    for (_, c), g in s.groupby(["source", "concentration_uM"]):
        g = g.sort_values("wavelength_nm")
        a = band.corrected("ferricyanide", g.wavelength_nm, g.absorbance)
        val = np.interp(wavelength, band.GRID, a) if wavelength else -np.log10(np.sum(w * 10**-a))
        rows.append((c, val))
    return pd.DataFrame(rows, columns=["c_uM", "A"])


def raw_points(channel):
    t = pd.read_csv(ROOT / "Data/analysis/ferricyanide/beer_lambert/absorbance_table.csv")
    t = t[t.channel == channel]
    return pd.DataFrame({"c_uM": t.concentration_um, "A": t.absorbance})


def jpeg_points(folder=None, channel="B"):
    if folder is None:
        t = pd.read_csv(
            ROOT / "Data/analysis/ferricyanide/beer_lambert/native_jpeg/absorbance_table.csv"
        )
        t = t[t.channel == channel]
        return pd.DataFrame({"c_uM": t.concentration_um, "A": t.absorbance})
    cfg = dict(folder=folder, ladder=4000.0 / 2.0 ** np.arange(0, 11))
    f = nja.read_frames("ferricyanide", cfg)
    f = f[f.channel == "B"]
    water = f[f.kind == "water"].mean_DN.mean()
    s = f[f.kind == "sample"]
    labels = dict(zip(sorted(cfg["ladder"]), sorted(nja.RAW_LABELS["ferricyanide"])))
    return pd.DataFrame(
        {"c_uM": s.concentration_exact.map(labels), "A": -np.log10(s.mean_DN / water)}
    )


def arms():
    phi = band.filtered_led("Experiment_14_Filter_450nm_40nm")
    filt = pd.read_csv(bfm.OUT / "ferricyanide_filtered_raw_points.csv")
    return [
        ("UV-Vis, 420 nm", "uv", False, uvvis_points(wavelength=420.0), RNG),
        ("UV-Vis, B band", "uv", False, uvvis_points(), RNG),
        ("UV-Vis, B band + filter", "uv", True, uvvis_points(phi), RNG),
        ("RAW B", "raw", False, raw_points("B"), RNG),
        ("RAW B/G", "bg", False, raw_points("BdivG"), RNG_BG),
        ("RAW B + filter", "raw", True, pd.DataFrame({"c_uM": filt.c_uM, "A": filt.A}), RNG),
        ("JPEG B", "jpeg", False, jpeg_points(), RNG),
        ("JPEG B/G", "jpeg", False, jpeg_points(channel="BdivG"), RNG_BG),
        ("JPEG B + filter", "jpeg", True, jpeg_points(JPEG_450), RNG),
    ]


UV, RAW, JPEG = fs.ROUTE_COLORS["uv"], fs.ROUTE_COLORS["raw"], fs.ROUTE_COLORS["jpeg"]
# UV-Vis at 420 nm (slope 20.1e-5 AU/uM) is left out of panel e and quoted in the caption.
STYLE = {  # colour, marker, line style; hue = route, dashed = 450/40 filter
    "UV-Vis, B band + filter": (UV[1], "d", "-."),
    "RAW B + filter": (RAW[0], "^", "--"),
    "RAW B": (RAW[2], "v", ":"),
    "JPEG B + filter": (JPEG[0], "s", "--"),
    "JPEG B": (JPEG[2], "o", ":"),
}
GROUPS = [  # panel f: route, then (arm, tick label, colour, filtered)
    (
        "UV-Vis",
        [("UV-Vis, 420 nm", "420 nm", UV[0], False), ("UV-Vis, B band + filter", "B", UV[1], True)],
    ),
    (
        "RAW",
        [
            ("RAW B + filter", "B", RAW[0], True),
            ("RAW B/G", "B/G", RAW[1], False),
            ("RAW B", "B", RAW[2], False),
        ],
    ),
    (
        "JPEG",
        [
            ("JPEG B + filter", "B", JPEG[0], True),
            ("JPEG B/G", "B/G", JPEG[1], False),
            ("JPEG B", "B", JPEG[2], False),
        ],
    ),
]
HIGH = (125, 4000)
JPEG_HIGH = (125, 1000)  # 2000 and 4000 uM read below the blank (Table 1)
# LOD outside the fitted 7.8-500 uM range: printed as not determinable (R2, R3-3).
ND_ARMS = {"JPEG B"}


def ln_lod_sd(x, y, rng):
    x, y = np.round(np.asarray(x, float), 4), np.asarray(y, float)
    m = (x >= rng[0] - 1e-6) & (x <= rng[1] + 1e-6)
    x, y = x[m], y[m]
    slope, icpt = np.polyfit(x, y, 1)
    resid = y - (slope * x + icpt)
    se = np.sqrt(np.sum(resid**2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2))
    dof = sum(np.sum(x == c) - 1 for c in sorted(set(x))[:3])
    return np.sqrt((se / slope) ** 2 + 1 / (2 * dof)), se, dof


def draw_e(ax, data, rows):
    handles = []
    for name, (col, mk, ls) in STYLE.items():
        p = data[name]
        g = p.assign(c_uM=p.c_uM.round(3)).groupby("c_uM").A.agg(["mean", "std"])
        g = g[(g.index > 0) & (g.index <= 4000)]
        rng = JPEG_HIGH if name == "JPEG B" else HIGH
        fit = bfm.fit(p.c_uM, p.A, rng)
        used = (g.index >= rng[0]) & (g.index <= rng[1])
        low = g.index < rng[0]
        excl = g.index > rng[1]
        for m, face in ((used | low, col), (excl, "white")):
            ax.errorbar(
                g.index[m] / 1000,
                g["mean"][m],
                yerr=g["std"][m],
                marker=mk,
                ls="none",
                color=col,
                mfc=face,
                markersize=4,
                capsize=1.5,
                elinewidth=0.6,
            )
        x = np.linspace(rng[0], rng[1], 20)
        ax.plot(x / 1000, fit["slope"] * x + fit["intercept"], color=col, lw=1.1, ls=ls)
        label = name.replace(", B band + filter", ", 450/40").replace(" B + filter", ", 450/40")
        label = label.replace(" B", ", no filter")
        if name == "JPEG B":
            ax.text(
                2.12,
                -0.019,
                "below blank, excluded",
                color=fs.GRAY,
                fontsize=fs.FONT_SMALL,
                va="center",
            )
        handles.append(
            plt.Line2D(
                [],
                [],
                color=col,
                marker=mk,
                ls=ls,
                lw=1.1,
                markersize=4,
                label=f"{label}: {fit['slope'] * 1e5:.2f}",
            )
        )
        rows += [
            dict(
                panel="e",
                arm=name,
                c_uM=c,
                A_mean=m,
                A_sd=sd,
                fit_range=fit["range"],
                slope=fit["slope"],
            )
            for c, m, sd in zip(g.index, g["mean"], g["std"])
        ]
    ax.axhline(0, color=fs.GRAY, lw=0.6, ls=":")
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=fs.FONT_SMALL,
        frameon=False,
        handlelength=2.2,
        borderaxespad=0.2,
    )
    ax.set_xlim(0, 4.3)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel(r"[K$_3$Fe(CN)$_6$] (mM)")
    ax.set_ylabel("Absorbance, blue channel (AU)")
    ax.set_title(r"Slopes above 125 µM (10$^{-5}$ AU/µM)", pad=3)


def draw_f(ax, lod, rows):
    lod = lod.set_index("arm")
    x, ticks, labels, centres, edges = 0.0, [], [], [], [-0.6]
    for route, bars in GROUPS:
        first = x
        for name, lab, col, filt in bars:
            v, k = lod.LOD_uM[name], np.exp(lod.ln_LOD_sd[name])
            if name in ND_ARMS:
                ax.text(
                    x,
                    12,
                    "n.d.",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=fs.FONT_SMALL,
                    color=fs.FIT_COLOR,
                )
                ticks.append(x)
                labels.append(lab)
                rows.append(
                    dict(panel="f", arm=name, LOD_uM=np.nan, LOD_low_uM=np.nan, LOD_high_uM=np.nan)
                )
                x += 1
                continue
            ax.bar(
                x,
                v,
                color=col,
                hatch=fs.ROUTE_HATCH if filt else None,
                edgecolor="white",
                lw=0,
                width=0.7,
            )
            ax.bar(x, v, fill=False, edgecolor=fs.FIT_COLOR, lw=0.6, width=0.7)
            ax.errorbar(
                x,
                v,
                yerr=[[v - v / k], [v * k - v]],
                color=fs.FIT_COLOR,
                lw=0.8,
                capsize=2.5,
                capthick=0.8,
            )
            ax.text(
                x,
                v * k * 1.08,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=fs.FONT_SMALL,
                color=fs.FIT_COLOR,
            )
            ticks.append(x)
            labels.append(lab)
            rows.append(dict(panel="f", arm=name, LOD_uM=v, LOD_low_uM=v / k, LOD_high_uM=v * k))
            x += 1
        centres.append((first + x - 1) / 2)
        edges.append(x - 0.4)
        x += 0.6
    ax.set_yscale("log")
    ax.set_ylim(10, 3e3)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=fs.FONT_SMALL)
    ax.tick_params(axis="x", length=0)
    grp = ax.secondary_xaxis("bottom")
    grp.set_xticks(centres)
    grp.set_xticklabels([g for g, _ in GROUPS])
    grp.tick_params(length=0, pad=14)
    grp.spines["bottom"].set_visible(False)
    sep = ax.secondary_xaxis("bottom")
    sep.set_xticks(edges[1:-1])
    sep.set_xticklabels([])
    sep.tick_params(length=24, width=0.6, direction="out", color=fs.GRAY)
    sep.spines["bottom"].set_visible(False)
    ax.legend(
        handles=[
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                hatch=fs.ROUTE_HATCH,
                edgecolor=fs.FIT_COLOR,
                lw=0.6,
                label="450/40 nm filter",
            )
        ],
        loc="upper left",
        fontsize=fs.FONT_SMALL,
        frameon=False,
        borderaxespad=0.2,
    )
    ax.set_ylabel("Detection limit (µM)")
    ax.set_title("Detection limit", pad=3)


def main():
    fs.apply()
    data = {n: p for n, _, _, p, _ in arms()}
    ranges = {n: r for n, _, _, _, r in arms()}
    lod = pd.DataFrame(
        [
            dict(
                arm=n,
                **bfm.fit(p.c_uM, p.A, ranges[n]),
                **dict(
                    zip(("ln_LOD_sd", "slope_se", "sigma_dof"), ln_lod_sd(p.c_uM, p.A, ranges[n]))
                ),
            )
            for n, p in data.items()
        ]
    )
    print(
        lod[["arm", "range", "slope", "slope_se", "sigma_dof", "LOD_uM", "ln_LOD_sd"]].to_string()
    )

    fig, (ax_e, ax_f) = plt.subplots(
        1,
        2,
        figsize=(fs.DOUBLE_COL, 2.6),
        gridspec_kw={"width_ratios": [1.15, 1]},
        layout="constrained",
    )
    rows = []
    draw_e(ax_e, data, rows)
    draw_f(ax_f, lod, rows)
    fs.panel_label(ax_e, "(e)")
    fs.panel_label(ax_f, "(f)")
    fs.save(fig, OUT, data=pd.DataFrame(rows))

    for draw, arg, out, letter in ((draw_e, data, OUT_E, "(e)"), (draw_f, lod, OUT_F, "(f)")):
        fig, ax = plt.subplots(figsize=(fs.SINGLE_COL, 2.8), layout="constrained")
        rows = []
        draw(ax, arg, rows)
        fs.panel_label(ax, letter)
        fs.save(fig, out, data=pd.DataFrame(rows))
    lod.to_csv(OUT.with_name("figure_5f_LOD_fits.csv"), index=False)


if __name__ == "__main__":
    main()
