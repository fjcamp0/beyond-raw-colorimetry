"""wine_ferroin_uvvis.py — UV-Vis analysis of the Fe(II)/1,10-phenanthroline standards.

Reviewer 3's requested application (R3-EXP1): iron by the ferroin method, the third
analyte of the paper. This script handles the reference-method half only - the Cary 60
spectra, the calibration curve, the transparent-tail fractions of all three Bayer
channels, and the camera-side prediction those fractions license before any DNG is
opened.

Inputs  Data/Phenanothroline_FeII estimation_wine appliation/UV-Vis/{set1,set2,Set3}.csv
Outputs Data/paper_figures/wine_ferroin/  (figures + companion CSVs)
    ferroin_spectra.{png,svg,csv}        blank-corrected spectra, all three sets
    ferroin_calibration.{png,svg,csv}    pooled calibration at lambda_max + residuals
    ferroin_channel_match.{png,svg,csv}  epsilon(lambda) vs the G and B channels,
                                         each with its transparent tail shaded
    ferroin_prediction.{png,svg,csv}     A_ch and slope decay vs camera-path A_510
        and Data/analysis/wine_ferroin/
    ferroin_summary.md                   every number this script derives

Run from the project root:
    python scripts/wine_ferroin_uvvis.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "scripts"))
from forward_model import load_sensitivity, split_at_cutoff  # noqa: E402

UVVIS_DIR = _PROJECT / "Data" / "Phenanothroline_FeII estimation_wine appliation" / "UV-Vis"
ILLUM_DIR = _PROJECT / "Data" / "spectra" / "illuminant"
# Figures and their companion CSVs go to the project's figures area; the prose
# summary stays with the analysis outputs (CLAUDE.md source/output rule).
FIG_DIR = _PROJECT / "Data" / "paper_figures" / "wine_ferroin"
OUT_DIR = _PROJECT / "Data" / "analysis" / "wine_ferroin"

SET_FILES = {
    "Set1": "set1_calibration_phenanthroline.csv",
    "Set2": "set2_calibration_phenanthroline.csv",
    "Set3": "Set3_calibration_phenanthroline.csv",
}
CONC_UM = np.array([0, 4, 8, 16, 24, 32, 40, 48, 56], dtype=float)
N_DATA_ROWS = 121

GRID = np.arange(400.0, 701.0, 1.0)
TAIL_THRESHOLD = 0.01
LOW_LEVELS = 3
EPS_LIT_510 = 11100.0
M_FE = 55.845

# figure_style.md
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.8,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.5,
        "figure.dpi": 150,
    }
)

SINGLE_COL = 3.33
DOUBLE_COL = 7.00

FIT_COLOR = "#2C3E50"
GRAY = "#626567"
CH_COLORS = {"R": "#C0392B", "G": "#27AE60", "B": "#2471A3"}
SET_STYLE = {"Set1": ("#2E86AB", "o"), "Set2": ("#A23B72", "s"), "Set3": ("#F18F01", "^")}


def read_cary_csv(path):
    """Return (wavelength, absorbance[n_wl, n_conc]) from a Cary 60 export."""
    rows = list(csv.reader(path.open(encoding="latin-1")))
    offset = 1 if rows[1][0].strip() == "" else 0
    block = np.array(
        [[float(v) for v in r[offset : offset + 18]] for r in rows[2 : 2 + N_DATA_ROWS]]
    )
    order = np.argsort(block[:, 0])
    return block[order, 0], block[order, 1::2]


def load_illuminant(name):
    """Load a 2-column illuminant CSV onto GRID, peak-normalised."""
    rows = list(csv.reader((ILLUM_DIR / name).open(encoding="utf-8")))
    lam = np.array([float(r[0]) for r in rows[1:]])
    spd = np.array([float(r[1]) for r in rows[1:]])
    out = np.interp(GRID, lam, spd, left=0.0, right=0.0)
    return out / out.max()


def ols(x, y):
    """OLS fit with standard errors and R-squared."""
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sse = float(np.sum(resid**2))
    sxx = float(np.sum((x - x.mean()) ** 2))
    syx = np.sqrt(sse / (n - 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_se": float(syx / np.sqrt(sxx)),
        "intercept_se": float(syx * np.sqrt(1.0 / n + x.mean() ** 2 / sxx)),
        "r2": float(1.0 - sse / np.sum((y - y.mean()) ** 2)),
        "syx": float(syx),
        "n": n,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = {k: read_cary_csv(UVVIS_DIR / f) for k, f in SET_FILES.items()}
    wl = raw["Set1"][0]
    for key, (w, _) in raw.items():
        assert np.allclose(w, wl), key + " is on a different wavelength grid"

    # Every set carries its own 0 uM reagent blank.
    corrected = {k: a - a[:, [0]] for k, (_, a) in raw.items()}
    blanks = np.column_stack([a[:, 0] for _, a in raw.values()])

    stack = np.stack([corrected[k] for k in SET_FILES], axis=0)
    mean_spec = stack.mean(axis=0)
    sd_spec = stack.std(axis=0, ddof=1)

    vis = (wl >= 400) & (wl <= 700)
    lam_max = float(wl[vis][np.argmax(mean_spec[vis, -1])])

    # Per-wavelength calibration slope IS epsilon(lambda) * l, pooled over all
    # 27 individual replicate points (gate G-G: never fit the set means).
    conc_pooled = np.tile(CONC_UM, len(SET_FILES))
    abs_pooled = np.hstack([corrected[k] for k in SET_FILES])
    slope_lam = (
        np.array([np.polyfit(conc_pooled, abs_pooled[i], 1)[0] for i in range(len(wl))]) * 1e6
    )

    j_max = int(np.argmin(np.abs(wl - lam_max)))
    fit = ols(conc_pooled, abs_pooled[j_max])
    per_set = {k: ols(CONC_UM, corrected[k][j_max]) for k in SET_FILES}
    quad = np.polyfit(conc_pooled, abs_pooled[j_max], 2)

    # Rajendra's workbook tabulated 505 nm, one grid node off lambda_max.
    j_505 = int(np.argmin(np.abs(wl - 505)))
    fit_505 = ols(conc_pooled, abs_pooled[j_505])

    # LOD/LOQ with sigma pooled over the lowest levels (Q-I convention).
    level_sd = stack[:, j_max, :].std(axis=0, ddof=1)
    sigma = float(np.sqrt(np.mean(level_sd[:LOW_LEVELS] ** 2)))
    slope_um = fit["slope"]
    lod = 3.3 * sigma / slope_um
    loq = 10.0 * sigma / slope_um

    # Leave-one-set-out prediction: the residual structure is between-set, not
    # curvature, so a within-set R2 overstates what a held-out session gets.
    loso = {}
    for held in SET_FILES:
        train = [k for k in SET_FILES if k != held]
        f = ols(np.tile(CONC_UM, len(train)), np.hstack([corrected[k][j_max] for k in train]))
        pred = (corrected[held][j_max] - f["intercept"]) / f["slope"]
        err = pred[1:] - CONC_UM[1:]
        loso[held] = {
            "bias_um": float(err.mean()),
            "rmse_um": float(np.sqrt(np.mean(err**2))),
            "max_abs_pct": float(np.max(np.abs(100.0 * err / CONC_UM[1:]))),
            "mean_abs_pct": float(np.mean(np.abs(100.0 * err / CONC_UM[1:]))),
        }
    loso_rmse = float(np.sqrt(np.mean([v["rmse_um"] ** 2 for v in loso.values()])))

    # Transparent tail, G channel.
    eps_l = np.interp(GRID, wl, slope_lam)
    i_peak = int(np.argmin(np.abs(GRID - lam_max)))
    eps_peak = float(eps_l[i_peak])
    above = eps_l >= TAIL_THRESHOLD * eps_peak
    red_idx = np.where(~above[i_peak:])[0]
    lam_tail = float(GRID[i_peak + red_idx[0]]) if len(red_idx) else float("nan")

    sens_wl, sens = load_sensitivity("average_all")
    sens_r = np.interp(GRID, sens_wl, sens[:, 0], left=0.0, right=0.0)
    sens_g = np.interp(GRID, sens_wl, sens[:, 1], left=0.0, right=0.0)
    sens_b = np.interp(GRID, sens_wl, sens[:, 2], left=0.0, right=0.0)

    illums = {
        "led": load_illuminant("illuminant_spelec_led.csv"),
        "daylight": load_illuminant("illuminant_daylight_clean.csv"),
        "daylight_diffuser": load_illuminant("illuminant_daylight_diffuser.csv"),
    }
    tail_mask = GRID > lam_tail
    ftail = {}
    for name, phi in illums.items():
        for ch, s in (("R", sens_r), ("G", sens_g), ("B", sens_b)):
            inside, _ = split_at_cutoff(phi * s, tail_mask, GRID)
            ftail[(name, ch)] = 100.0 * inside / float(np.trapezoid(phi * s, GRID))

    # Figure 1 — spectra.
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.60))
    norm = colors.Normalize(vmin=CONC_UM[1], vmax=CONC_UM[-1])
    cmap = plt.get_cmap("plasma")
    ax = axes[0]
    for j, c in enumerate(CONC_UM):
        col = GRAY if c == 0 else cmap(norm(c))
        ax.plot(wl, mean_spec[:, j], color=col, lw=0.7, alpha=0.9)
        ax.fill_between(
            wl,
            mean_spec[:, j] - sd_spec[:, j],
            mean_spec[:, j] + sd_spec[:, j],
            color=col,
            alpha=0.18,
            lw=0,
        )
    ax.axvline(lam_max, color=FIT_COLOR, ls="--", lw=0.8)
    ax.annotate(
        r"$\lambda_{\max}$ = %.0f nm" % lam_max,
        xy=(lam_max + 25, mean_spec[j_max, -1] * 0.90),
        fontsize=7,
        color=FIT_COLOR,
    )
    # The scans run to 200 nm, where absorbance saturates near 4 AU; autoscaling
    # on the whole array flattens the visible band out of sight.
    shown = (wl >= 350) & (wl <= 805)
    ax.set_xlim(350, 805)
    ax.set_ylim(-0.02, mean_spec[shown].max() * 1.15)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (AU)")
    ax.text(-0.15, 1.05, "A", transform=ax.transAxes, fontsize=9, fontweight="bold")
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label(r"[Fe(II)] ($\mu$M)", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    ax = axes[1]
    for k in SET_FILES:
        ax.plot(
            wl,
            raw[k][1][:, 0],
            color=SET_STYLE[k][0],
            lw=0.7,
            alpha=0.9,
            label=k + " reagent blank",
        )
    ax.axvline(lam_max, color=FIT_COLOR, ls="--", lw=0.8)
    ax.set_xlim(350, 805)
    ax.set_ylim(blanks[shown].min() - 0.005, blanks[shown].max() * 1.35)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reagent-blank absorbance (AU)")
    ax.legend(framealpha=0.85, edgecolor="none")
    ax.text(-0.15, 1.05, "B", transform=ax.transAxes, fontsize=9, fontweight="bold")
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5)
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / ("ferroin_spectra." + ext), dpi=600, bbox_inches="tight", format=ext)
    plt.close(fig)

    with (FIG_DIR / "ferroin_spectra.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        head = ["wavelength_nm"]
        head += ["A_mean_%.0fuM" % c for c in CONC_UM]
        head += ["A_sd_%.0fuM" % c for c in CONC_UM]
        head += ["A_blank_" + k for k in SET_FILES]
        head += ["slope_pooled_per_M"]
        w.writerow(head)
        for i in range(len(wl)):
            w.writerow(
                ["%.4f" % wl[i]]
                + ["%.6f" % v for v in mean_spec[i]]
                + ["%.6f" % v for v in sd_spec[i]]
                + ["%.6f" % v for v in blanks[i]]
                + ["%.3f" % slope_lam[i]]
            )

    # Figure 2 — calibration.
    fig, axes = plt.subplots(
        2, 1, figsize=(SINGLE_COL, 3.60), height_ratios=[2.6, 1.0], sharex=True
    )
    ax = axes[0]
    for k in SET_FILES:
        col, mk = SET_STYLE[k]
        ax.scatter(
            CONC_UM, corrected[k][j_max], s=12, alpha=0.60, marker=mk, color=col, label=k, zorder=3
        )
    ax.errorbar(
        CONC_UM,
        stack[:, j_max, :].mean(axis=0),
        yerr=level_sd,
        fmt="o",
        markersize=4.5,
        color=FIT_COLOR,
        capsize=2.5,
        elinewidth=0.8,
        lw=0,
        zorder=4,
        label=r"mean $\pm$ 1 SD",
    )
    xx = np.linspace(0, CONC_UM[-1] * 1.03, 50)
    ax.plot(
        xx,
        fit["slope"] * xx + fit["intercept"],
        color=FIT_COLOR,
        lw=0.9,
        ls="--",
        zorder=2,
        label=(r"OLS, $R^2$ = %.5f" % fit["r2"] + "\nn = %d pooled points" % fit["n"]),
    )
    ax.set_ylabel("Absorbance at %.0f nm (AU)" % lam_max)
    ax.legend(framealpha=0.85, edgecolor="none", loc="upper left")
    ax.text(-0.19, 1.04, "A", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1]
    for k in SET_FILES:
        col, mk = SET_STYLE[k]
        res = corrected[k][j_max] - (fit["slope"] * CONC_UM + fit["intercept"])
        ax.scatter(CONC_UM, res * 1e3, s=12, alpha=0.60, marker=mk, color=col)
    ax.axhline(0.0, color=FIT_COLOR, lw=0.6, ls="--")
    ax.set_xlabel(r"[Fe(II)] ($\mu$M)")
    ax.set_ylabel(r"Residual ($\times 10^{-3}$ AU)")
    ax.text(-0.19, 1.10, "B", transform=ax.transAxes, fontsize=9, fontweight="bold")
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5)
    for ext in ("png", "svg"):
        fig.savefig(
            FIG_DIR / ("ferroin_calibration." + ext), dpi=600, bbox_inches="tight", format=ext
        )
    plt.close(fig)

    with (FIG_DIR / "ferroin_calibration.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["conc_uM", "set", "A_blank_corrected", "residual_pooled_fit"])
        for k in SET_FILES:
            for j, c in enumerate(CONC_UM):
                a = corrected[k][j_max, j]
                w.writerow(
                    [
                        "%.0f" % c,
                        k,
                        "%.6f" % a,
                        "%.6f" % (a - (fit["slope"] * c + fit["intercept"])),
                    ]
                )

    # Figure 3 — how well ferroin's band covers the G and B channels, and how
    # little of each channel is left transparent.
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.80), sharey=True)
    for ax, ch, sens_ch in ((axes[0], "G", sens_g), (axes[1], "B", sens_b)):
        eff = illums["led"] * sens_ch
        eff = eff / eff.max()
        ax.fill_between(
            GRID,
            0,
            eff,
            where=GRID <= lam_tail,
            color=CH_COLORS[ch],
            alpha=0.30,
            lw=0,
            label="absorbed by ferroin",
        )
        ax.fill_between(
            GRID,
            0,
            eff,
            where=GRID >= lam_tail,
            color=GRAY,
            alpha=0.45,
            lw=0,
            label=r"transparent tail, $f_\mathrm{tail}$ = %.1f%%" % ftail[("led", ch)],
        )
        ax.plot(GRID, eff, color=CH_COLORS[ch], lw=0.9)
        ax.plot(
            GRID, eps_l / eps_peak, color=FIT_COLOR, lw=0.9, label=r"ferroin $\varepsilon(\lambda)$"
        )
        ax.axvline(lam_tail, color=FIT_COLOR, ls="--", lw=0.8)
        ax.axhline(TAIL_THRESHOLD, color=FIT_COLOR, lw=0.6, ls=":")
        ax.annotate("%.0f nm" % lam_tail, xy=(lam_tail + 5, 1.15), fontsize=7, color=FIT_COLOR)
        ax.annotate(
            r"1% of $\varepsilon(\lambda_{\max})$", xy=(405, 0.05), fontsize=6, color=FIT_COLOR
        )
        ax.set_xlim(400, 700)
        ax.set_ylim(0, 1.32)
        ax.set_xlabel("Wavelength (nm)")
        ax.legend(framealpha=0.85, edgecolor="none", loc="upper left", handlelength=1.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(
        r"Normalised $\varphi(\lambda)\,S_\mathrm{ch}(\lambda)$"
        "\n"
        r"and $\varepsilon(\lambda)/\varepsilon(\lambda_{\max})$"
    )
    for ax, letter, ch in ((axes[0], "A", "G"), (axes[1], "B", "B")):
        ax.text(-0.13, 1.05, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.text(
            0.97,
            1.05,
            "%s channel" % ch,
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            color=CH_COLORS[ch],
            fontweight="bold",
        )
    fig.tight_layout(pad=0.5)
    for ext in ("png", "svg"):
        fig.savefig(
            FIG_DIR / ("ferroin_channel_match." + ext), dpi=600, bbox_inches="tight", format=ext
        )
    plt.close(fig)

    with (FIG_DIR / "ferroin_channel_match.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "wavelength_nm",
                "eps_l_per_M",
                "eps_rel_peak",
                "sens_R",
                "sens_G",
                "sens_B",
                "phi_led",
                "phi_daylight",
                "phi_daylight_diffuser",
            ]
        )
        for i, lam in enumerate(GRID):
            w.writerow(
                [
                    "%.0f" % lam,
                    "%.3f" % eps_l[i],
                    "%.6f" % (eps_l[i] / eps_peak),
                    "%.6f" % sens_r[i],
                    "%.6f" % sens_g[i],
                    "%.6f" % sens_b[i],
                    "%.6f" % illums["led"][i],
                    "%.6f" % illums["daylight"][i],
                    "%.6f" % illums["daylight_diffuser"][i],
                ]
            )

    # Figure 4 — what f_tail predicts for the camera, before any image is read.
    # Parameterised by A_510 on the camera's own path length, so it needs no
    # value of epsilon and no value of l_cam.
    eps_rel = eps_l / eps_peak
    a510 = np.linspace(0.0, 1.5, 301)
    a_ch = {}
    for ch, s in (("R", sens_r), ("G", sens_g), ("B", sens_b)):
        w_ch = illums["led"] * s
        denom = float(np.trapezoid(w_ch, GRID))
        num = np.array([float(np.trapezoid(w_ch * 10.0 ** (-a * eps_rel), GRID)) for a in a510])
        a_ch[ch] = -np.log10(num / denom)
    k0 = {ch: a_ch[ch][1] / a510[1] for ch in a_ch}
    dev = {
        ch: 100.0
        * (np.divide(a_ch[ch], k0[ch] * a510, out=np.ones_like(a510), where=a510 > 0) - 1.0)
        for ch in a_ch
    }
    recovery = 1.0 + dev["G"] / 100.0
    ceiling = {ch: -np.log10(ftail[("led", ch)] / 100.0) for ch in a_ch}

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.60))
    ax = axes[0]
    for ch in ("B", "G", "R"):
        ax.plot(a510, a_ch[ch], color=CH_COLORS[ch], lw=0.9, label=ch)
        ax.axhline(ceiling[ch], color=CH_COLORS[ch], lw=0.6, ls=":")
        ax.annotate(
            "%s ceiling %.2f" % (ch, ceiling[ch]),
            xy=(1.50, ceiling[ch] + 0.03),
            fontsize=7,
            color=CH_COLORS[ch],
            ha="right",
        )
    ax.plot(a510, k0["G"] * a510, color=FIT_COLOR, lw=0.9, ls="--", label="G, initial slope")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.95)
    ax.set_xlabel(r"$A_{510}$ on the camera path (AU)")
    ax.set_ylabel(r"$A_\mathrm{ch}$ (AU)")
    ax.legend(framealpha=0.85, edgecolor="none", loc="upper left")
    ax.text(-0.15, 1.05, "A", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1]
    for ch in ("B", "G", "R"):
        ax.plot(a510, dev[ch], color=CH_COLORS[ch], lw=0.9, label=ch)
    ax.axhline(0.0, color=FIT_COLOR, lw=0.6, ls="--")
    ax.set_xlim(0, 1.5)
    ax.set_xlabel(r"$A_{510}$ on the camera path (AU)")
    ax.set_ylabel("Deviation from the initial slope (%)")
    ax.legend(framealpha=0.85, edgecolor="none", loc="lower left")
    ax.text(-0.15, 1.05, "B", transform=ax.transAxes, fontsize=9, fontweight="bold")
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5)
    for ext in ("png", "svg"):
        fig.savefig(
            FIG_DIR / ("ferroin_prediction." + ext), dpi=600, bbox_inches="tight", format=ext
        )
    plt.close(fig)

    with (FIG_DIR / "ferroin_prediction.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["A_510_camera_path", "A_R", "A_G", "A_B", "dev_R_pct", "dev_G_pct", "dev_B_pct"]
        )
        for i, a in enumerate(a510):
            w.writerow(
                [
                    "%.4f" % a,
                    "%.6f" % a_ch["R"][i],
                    "%.6f" % a_ch["G"][i],
                    "%.6f" % a_ch["B"][i],
                    "%.4f" % dev["R"][i],
                    "%.4f" % dev["G"][i],
                    "%.4f" % dev["B"][i],
                ]
            )

    slope_m = fit["slope"] * 1e6
    spread = 100.0 * (
        max(per_set[k]["slope"] for k in SET_FILES) / min(per_set[k]["slope"] for k in SET_FILES)
        - 1.0
    )
    lines = [
        "# Fe(II)/1,10-phenanthroline UV-Vis - reference-method analysis",
        "",
        "Generated by `scripts/wine_ferroin_uvvis.py`.",
        "Source: `Data/Phenanothroline_FeII estimation_wine appliation/UV-Vis/`",
        "",
        "## Acquisition",
        "",
        "| Property | Value |",
        "|---|---|",
        "| Instrument | Agilent Cary 60, Scan 5.1.0.1016 |",
        "| Range | 800 -> 200 nm, 5.00 nm data interval, 121 points |",
        "| Scan rate | 24 000 nm/min, 0.0125 s average time |",
        "| Beam / baseline | dual beam, baseline correction OFF |",
        "| Collected | 22/09/2026, 14:00-14:12, the three sets back to back |",
        "| Standards | 0, 4, 8, 16, 24, 32, 40, 48, 56 uM Fe(II); 3 sets |",
        "| Path length | NOT RECORDED - see open questions |",
        "",
        "## Spectrum",
        "",
        "- lambda_max = **%.0f nm** in all three sets; literature ferroin is 510 nm." % lam_max,
        "- `Calibration_curve.xlsx` tabulates **505 nm** instead (reproduced "
        "here: pooled slope %.1f AU/M, R2 %.6f). That is one 5 nm grid node "
        "below the peak and gives away %.2f%% of the slope for nothing. Use "
        "510 nm."
        % (fit_505["slope"] * 1e6, fit_505["r2"], 100.0 * (1.0 - fit_505["slope"] / fit["slope"])),
        "- Reagent blank at %.0f nm: %.4f +/- %.4f AU across the three sets. "
        "Small, but not zero, so the blank correction earns its keep."
        % (lam_max, blanks[j_max].mean(), blanks[j_max].std(ddof=1)),
        "- The band is strongly asymmetric: down to 1%% of peak by %.0f nm on "
        "the red side, but still %.0f%% of peak at 400 nm on the blue side."
        % (lam_tail, 100.0 * eps_l[0] / eps_peak),
        "- **The reagent blank carries an instrument baseline, not just "
        "chemistry.** Over 350-800 nm it swings %.4f AU peak to peak and goes "
        "as negative as %.4f AU at %.0f nm, in a slow ripple of roughly 100 nm "
        "period. Negative absorbance is not a solution property; with "
        "`Baseline Correction` off and a 24 000 nm/min scan this is the "
        "instrument's own baseline. It is harmless here because it is stable - "
        "the three sets agree to %.4f AU RMS - so blank subtraction removes "
        "it. But it means absolute absorbances from these files are only good "
        "to about +/-0.015 AU unless blank-corrected, and it is the floor "
        "under the %.4f AU sigma below."
        % (
            blanks[shown].max() - blanks[shown].min(),
            blanks[shown].mean(axis=1).min(),
            wl[shown][blanks[shown].mean(axis=1).argmin()],
            float(np.sqrt((blanks[shown].std(axis=1, ddof=1) ** 2).mean())),
            sigma,
        ),
        "",
        "## Linearity (blank-corrected, pooled individual replicates, gate G-G)",
        "",
        "| Fit | n | slope (AU/M) | intercept (AU) | R2 | s_y.x (AU) |",
        "|---|---|---|---|---|---|",
    ]
    for k in SET_FILES:
        f = per_set[k]
        lines.append(
            "| %s | %d | %.1f | %+.5f | %.6f | %.5f |"
            % (k, f["n"], f["slope"] * 1e6, f["intercept"], f["r2"], f["syx"])
        )
    lines += [
        "| **Pooled** | **%d** | **%.1f +/- %.1f** | **%+.6f +/- %.6f** | "
        "**%.6f** | **%.5f** |"
        % (
            fit["n"],
            slope_m,
            fit["slope_se"] * 1e6,
            fit["intercept"],
            fit["intercept_se"],
            fit["r2"],
            fit["syx"],
        ),
        "",
        "- The intercept sits %.2f standard errors from zero - "
        "indistinguishable from zero, unlike the methylene blue "
        "A_R = +0.0198 problem." % (abs(fit["intercept"]) / fit["intercept_se"]),
        "- A quadratic term gives a2 = %.2e AU/uM^2, worth %.4f AU over the "
        "whole range (%.1f%% of the top signal). No detectable curvature."
        % (
            quad[0],
            quad[0] * CONC_UM[-1] ** 2,
            100.0 * quad[0] * CONC_UM[-1] ** 2 / abs_pooled[j_max].max(),
        ),
        "- Between-set slope spread: %.2f%%, monotone with run order "
        "(Set1 > Set2 > Set3)." % spread,
        "",
        "## Detection limits",
        "",
        "Q-I convention: sigma is the RMS of the per-level SDs over the "
        "lowest %d levels." % LOW_LEVELS,
        "",
        "| Quantity | Value |",
        "|---|---|",
        "| sigma | %.5f AU |" % sigma,
        "| slope | %.3e AU/uM |" % slope_um,
        "| LOD = 3.3 sigma / slope | **%.2f uM** = %.3f mg/L Fe |" % (lod, lod * M_FE / 1000.0),
        "| LOQ = 10 sigma / slope | **%.2f uM** = %.3f mg/L Fe |" % (loq, loq * M_FE / 1000.0),
        "",
        "Iron in wine runs roughly 1-10 mg/L (18-180 uM), so the reference "
        "method clears the whole range with room to spare.",
        "",
        "## Leave-one-set-out prediction",
        "",
        "R2 within a set is 0.9999, but the residuals stratify by set rather "
        "than by concentration (see `ferroin_calibration.png` panel B), so the "
        "within-set figure overstates what a held-out session achieves. "
        "Calibrating on two sets and predicting the third, blank excluded:",
        "",
        "| Held out | bias (uM) | RMSE (uM) | mean abs. error | worst |",
        "|---|---|---|---|---|",
    ]
    for k in SET_FILES:
        v = loso[k]
        lines.append(
            "| %s | %+.3f | %.3f | %.2f%% | %.2f%% |"
            % (k, v["bias_um"], v["rmse_um"], v["mean_abs_pct"], v["max_abs_pct"])
        )
    lines += [
        "| **Pooled** | | **%.3f** | | |" % loso_rmse,
        "",
        "%.2f uM RMSE on held-out sessions, against a %.2f uM LOQ. The error "
        "is dominated by the %.2f%% run-to-run slope drift, not by noise or "
        "nonlinearity, which is worth knowing before the same test is run on "
        "the camera: any camera-side error below this is not resolvable "
        "against the reference method." % (loso_rmse, loq, spread),
        "",
        "## Transparent-tail fraction",
        "",
        "1%-of-visible-peak rule, 1 nm grid, half-boundary split - the "
        "convention of `tail_boundary_1pc_rule_2026-09-21.md`.",
        "",
        "- Visible-band peak eps*l = %.0f AU/M at %.0f nm; 1%% threshold = "
        "%.1f AU/M." % (eps_peak, lam_max, TAIL_THRESHOLD * eps_peak),
        "- Red-side boundary **%.0f nm**; the tail is lambda > %.0f nm." % (lam_tail, lam_tail),
        "- Blue-side boundary: none inside 400-700 nm. Ferroin never falls "
        "below 1% of peak on the blue side within the camera's range, so the "
        "tail is one-sided - unlike ferricyanide and methylene blue.",
        "",
        "| Illuminant | f_tail(G) | f_tail(R) | f_tail(B) |",
        "|---|---|---|---|",
    ]
    for name in illums:
        lines.append(
            "| %s | **%.1f%%** | %.1f%% | %.1f%% |"
            % (name, ftail[(name, "G")], ftail[(name, "R")], ftail[(name, "B")])
        )
    lines += [
        "",
        "Ohlab 20-phone mean sensitivities. For comparison, ferricyanide/B is "
        "62.4% and methylene blue/R is 0.04%.",
        "",
        "## The prediction this licenses, written down before the DNGs are read",
        "",
        "f_tail(G) = %.1f%% is NOT what the workplan assumed. "
        '`06_workplan_2026-09-21.md` predicted that "a moderately narrow band '
        "inside a broad G passband should give a substantial transparent "
        'tail", hence a kink that the channel ratio then repairs. The '
        "measured spectrum says otherwise: the ferroin band is broad and "
        "asymmetric, still %.0f%% of peak at 400 nm, so it covers essentially "
        "the whole G passband (5%%-support 462-632 nm under the LED). Almost "
        "no G light is unabsorbable, and the ceiling -log10(f_tail) = "
        "**%.2f AU** is far out of reach. **There is no kink to find, and no "
        "c* for this analyte.**" % (ftail[("led", "G")], 100.0 * eps_l[0] / eps_peak, ceiling["G"]),
        "",
        "That is not the same as predicting a straight line, and this is the "
        "part worth getting right. f_tail fixes where the response *ceases* "
        "to respond; it does not say the approach to that ceiling is "
        "straight. Across the G passband eps still runs from 74% of peak at "
        "462 nm down to 0.3% at 632 nm, and integrating over that spread bends "
        "A_G well below the ceiling:",
        "",
        "| Prediction | Value |",
        "|---|---|",
        "| A_G ceiling = -log10(f_tail) | %.2f AU |" % ceiling["G"],
        "| initial dA_G/dA_510 | %.3f |" % k0["G"],
    ]
    for probe in (0.1, 0.2, 0.3, 0.5, 1.0):
        i = int(np.argmin(np.abs(a510 - probe)))
        lines.append("| slope deviation at A_510 = %.1f AU | %+.1f%% |" % (probe, dev["G"][i]))

    a_top = float(abs_pooled[j_max].max())
    lines += [
        "",
        "So the prediction is **smooth monotone compression with no kink** - "
        "qualitatively methylene blue's behaviour, quantitatively rather more "
        "of it. How much more depends entirely on the A_510 the camera "
        "actually sees, which is l_cam / l_cuvette times the cuvette value "
        "(%.3f AU at %.0f uM). Taking l_cam = 2.85 mm, as for ferricyanide "
        "under the same LED:" % (a_top, CONC_UM[-1]),
        "",
        "| If the cuvette was | camera A_510 at %.0f uM | predicted slope "
        "deviation |" % CONC_UM[-1],
        "|---|---|---|",
    ]
    for l_cuv, label in ((1.0, "1 cm"), (0.2, "0.2 cm")):
        a_cam = a_top * 0.285 / l_cuv
        i = int(np.argmin(np.abs(a510 - a_cam)))
        lines.append("| %s | %.3f AU | %+.1f%% |" % (label, a_cam, 100.0 * (recovery[i] - 1.0)))
    lines += [
        "",
        "Open question 1 below is therefore not bookkeeping: it moves the "
        "predicted compression by a factor of about five, and it has to be "
        "settled before the prediction can be called pre-registered.",
        "",
        "Two further predictions, both falsifiable from the existing DNGs:",
        "",
        "- **R is the channel with the tail here**, f_tail(R) = %.1f%% "
        "against %.1f%% for G, because the R channel peaks at 590 nm where "
        "ferroin has nearly stopped absorbing. So **G/R is the natural "
        "reference pair**, and the correction should work through R's tail "
        "rather than G's." % (ftail[("led", "R")], ftail[("led", "G")]),
        "- **A_R should saturate where A_G does not.** f_tail(R) = %.1f%% "
        "puts the R-channel ceiling at only %.2f AU, which is inside reach. "
        "That is the ferricyanide signature appearing in a different channel "
        "for a different analyte - the sharpest generalisation test in the "
        "dataset." % (ftail[("led", "R")], ceiling["R"]),
        "- **B, not G, is the strongest measurement channel for ferroin.** "
        "This contradicts the workplan's framing that ferroin \"lands in the "
        'GREEN channel". The initial responses are dA/dA_510 = %.3f (B), '
        "%.3f (G), %.3f (R): the blue channel peaks at 458 nm, where ferroin "
        "still absorbs at %.0f%% of peak, and it is more tightly confined to "
        "the strongly absorbing side of the band than G is. Its ceiling is "
        "also the highest of the three, %.2f AU. The model says B; whether B "
        "wins in practice depends on shot noise, and a Bayer mosaic has half "
        "as many blue pixels as green, so this should be settled on the "
        "images rather than asserted here."
        % (
            k0["B"],
            k0["G"],
            k0["R"],
            100.0 * float(eps_l[int(np.argmin(np.abs(GRID - 458)))]) / eps_peak,
            ceiling["B"],
        ),
        "",
        "These come from the UV-Vis spectra and the public sensitivity curves "
        "alone, so the DNG sets in `Smartphone/` can falsify them. A "
        "prediction that lands answers R3-M1 better than one tuned after the "
        "fact - and a third analyte that generalises the framework in a "
        "direction the workplan did not expect is a stronger result than a "
        "third confirmation of the ferricyanide story.",
        "",
        "## Open questions",
        "",
        "1. **Path length, and whether the labelled concentrations are the "
        "in-cuvette ones.** The measured eps*l is %.0f AU/M. Literature "
        "eps(510) for ferroin is about %.0f M-1 cm-1, which implies an "
        "effective path of %.2f cm - neither the project's 0.2 cm cuvette "
        "(eps would be %.0f, %.1fx literature) nor a 1 cm cuvette "
        "(eps = %.0f, %.2fx literature). A 1 cm cuvette plus a reagent "
        "dilution of about %.2f reconciles both; so would incomplete "
        "complexation. Nothing else in this analysis moves, but eps cannot be "
        "quoted until this is settled."
        % (
            slope_m,
            EPS_LIT_510,
            slope_m / EPS_LIT_510,
            slope_m / 0.2,
            slope_m / 0.2 / EPS_LIT_510,
            slope_m,
            slope_m / EPS_LIT_510,
            slope_m / EPS_LIT_510,
        ),
        "2. **Which illuminant lit the DNG sets?** The f_tail table above "
        "spans several percentage points across the three candidates.",
        "3. **The wine itself.** These nine standards are the calibration "
        "set. Reviewer 3 asked for samples OUTSIDE it - spike and recovery on "
        "real wine. No wine spectra are in this folder yet.",
        "4. **White or red?** Red-wine anthocyanins absorb right on top of "
        "the %.0f nm band." % lam_max,
    ]
    (OUT_DIR / "ferroin_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    captions = [
        "# Captions - wine_ferroin",
        "",
        "Generated by `scripts/wine_ferroin_uvvis.py`. Not yet assigned "
        "manuscript figure numbers; see `figures_plan.md` before promoting "
        "any of these to `Manuscript/_drafts/figures/`.",
        "",
        "## ferroin_spectra",
        "",
        "**Fe(II)/1,10-phenanthroline UV-Vis spectra.** (A) Blank-corrected "
        "absorbance of the nine Fe(II) standards (0-%.0f uM), mean of three "
        "independently prepared sets with +/-1 SD bands; the dashed line "
        "marks lambda_max = %.0f nm. (B) The three reagent blanks on their own "
        "scale, showing the instrument baseline (%.4f AU peak to peak, "
        "negative near 720 nm) that blank subtraction removes."
        % (CONC_UM[-1], lam_max, blanks[shown].max() - blanks[shown].min()),
        "",
        "## ferroin_calibration",
        "",
        "**Calibration at %.0f nm.** (A) All %d individual replicate points "
        "(one marker per set) with the mean +/- 1 SD and the pooled OLS fit "
        "(R2 = %.5f, slope %.0f AU/M). (B) Residuals about the pooled fit: "
        "they stratify by set rather than by concentration, identifying a "
        "%.2f%% run-to-run slope drift rather than curvature."
        % (lam_max, fit["n"], fit["r2"], slope_m, spread),
        "",
        "## ferroin_channel_match",
        "",
        "**Spectral overlap of the ferroin band with the green and blue Bayer "
        "channels.** Filled curves are the effective channel illumination "
        "phi(lambda) S_ch(lambda) under the white LED, normalised to unit "
        "peak; the dark line is the measured ferroin epsilon(lambda), "
        "normalised to its %.0f nm maximum. Grey shading marks the "
        "analyte-transparent region, where epsilon falls below 1%% of peak "
        "(dotted line), which for ferroin begins at %.0f nm (dashed line). "
        "(A) Green channel, f_tail = %.1f%%. (B) Blue channel, "
        "f_tail = %.1f%%. Ferroin's band is broad and asymmetric enough to "
        "cover both passbands almost completely, so neither channel retains "
        "the large transparent tail that drives ferricyanide's nonlinearity "
        "in blue (62.4%%)." % (lam_max, lam_tail, ftail[("led", "G")], ftail[("led", "B")]),
        "",
        "## ferroin_prediction",
        "",
        "**Camera-side response predicted from the spectra alone.** "
        "(A) Channel absorbance against the 510 nm absorbance on the camera's "
        "own path length, from the Beer-Lambert-Bayer integral with the "
        "measured epsilon(lambda), the Ohlab 20-phone mean sensitivities and "
        "the white LED; dotted lines are the ceilings -log10(f_tail). "
        "(B) Departure of each channel from its own initial slope. No value "
        "of epsilon or of l_cam enters, so these curves are a prediction the "
        "DNG sets can falsify.",
    ]
    (FIG_DIR / "CAPTIONS.md").write_text("\n".join(captions) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
