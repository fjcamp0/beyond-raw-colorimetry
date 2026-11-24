"""figure_3.py — Figure 3 of the revised manuscript: spectral inputs (3 x 3 panel).

    Row 1  A  CIE D65          B  LED (SPELEC)          C  measured daylight
    Row 2  D  eps ferricyanide  E  eps methylene blue    F  eps Fe(II)-phenanthroline
    Row 3  G  Bayer sensitivity H  LED x 450/40 filter   I  LED x 650/10 filter

Rows 1 and 2 (A-E) are the first-submission panels (a)-(c), (e), (f), with the
same data and estimators; the first-submission figure is kept, with its script
figure_3_v1.py, in figure_3/v1/. Every panel spans 350-800 nm.

eps(lambda) is the per-wavelength OLS slope of absorbance against concentration
over every replicate point (gate G-G), l = 0.2 cm. Ferroin: the three 22 Sep
Cary 60 sets, each blanked against its own 0 uM reagent blank, n = 27; eps(510)
reproduces ferroin_summary.md's slope. At 0.2 cm that eps is 1.77x the
literature value (QUESTIONS_FOR_RAJENDRA.md, Block 1 Q2, still open).

H and I are the 23 Sep SPELEC spectra of the LED through each filter, without
diffuser, peak-normalised: the band the camera receives. The B / R channel
sensitivity of G is drawn dotted behind each. Dividing by the bare
LED to get filter transmission fails below ~440 nm, where the LED is weak.

Outputs -> Data/paper_figures/figure_3/, synced to Manuscript/_drafts/figures/figure_3/
    figure_3_panel.{png,svg,csv}     composite, 7.00 in
    figure_3<a-i>.{png,svg,csv}      each panel alone, 3.33 in

Run from the project root: python scripts/figure_3.py
"""

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import figstyle as fs  # noqa: E402
import figure_3_v1 as v1  # noqa: E402
import figure_optical_filters as fo  # noqa: E402
from ferroin_scale import FE_SCALE, true_uM  # noqa: E402
from wine_ferroin_uvvis import CONC_UM, UVVIS_DIR, read_cary_csv  # noqa: E402

OUT_DIR = PROJ / "Data" / "paper_figures" / "figure_3"
MS_DIR = PROJ / "Manuscript" / "_drafts" / "figures" / "figure_3"

WL = v1.WL_GRID
X_TICKS = [400, 500, 600, 700, 800]
L_CM = v1.L_UVVIS_CM

FERROIN_FILES = (
    "set1_calibration_phenanthroline.csv",
    "set2_calibration_phenanthroline.csv",
    "Set3_calibration_phenanthroline.csv",
)
FERROIN_SLOPE_510 = 3935.5  # AU/M, pooled, ferroin_summary.md
FILTERS = {"450/40": "B", "650/10": "R"}

EPS = {
    "ferricyanide": {"color": fs.FERIC_COLOR, "scale": 1.0, "marks": [(420, "420 nm", 1.0, 12)]},
    "MB": {
        "color": fs.MB_COLOR,
        "scale": 1e3,
        "marks": [(610, "610 nm", 0.80, -8), (665, "665 nm", 1.0, 6)],
    },
    "ferroin": {"color": fs.FERROIN_COLOR, "scale": 1e3, "marks": [(510, "510 nm", 1.0, 10)]},
}


def ferroin_epsilon():
    """Pooled eps(lambda) of ferroin on WL, blank-corrected per set."""
    c_list, a_list = [], []
    for name in FERROIN_FILES:
        wl, a = read_cary_csv(UVVIS_DIR / name)
        a = a - a[:, [0]]
        for j, c in enumerate(CONC_UM):
            c_list.append(true_uM(c))
            a_list.append(np.interp(WL, wl, a[:, j]))
    fit = v1.epsilon_regression(np.array(c_list), np.vstack(a_list), L_CM)
    eps_510 = fit["eps"][WL == 510][0]
    expected = FERROIN_SLOPE_510 / L_CM / FE_SCALE
    assert abs(eps_510 / expected - 1) < 0.005, (eps_510, expected)
    return fit


def filter_band(label):
    """LED x filter on WL, smoothed and normalised to its peak."""
    w, c = fo.read_spelec(fo.FILTER_FILES[label])
    y = np.interp(WL, w, fo.smooth(c))
    return y / y.max()


def load():
    sens = v1.load_sensitivity()
    feric_sets, _, feric_names = v1.load_spectra(v1.FERIC_DIR, v1.load_ferricyanide_set)
    mb_sets, _, mb_names = v1.load_spectra(v1.MB_DIR, v1.load_mb_set)
    feric, *_ = v1.build_epsilon(feric_sets, feric_names, "ferricyanide", "pooled", None)
    mb, *_ = v1.build_epsilon(mb_sets, mb_names, "mb", "pooled", None)
    return {
        "illum": {
            "D65": v1.load_d65(),
            "LED": v1._load_and_normalise_csv(v1.ILLUM_DIR / "illuminant_spelec_led.csv"),
            "daylight": v1._load_and_normalise_csv(v1.ILLUM_DIR / "illuminant_daylight_clean.csv"),
        },
        "eps": {"ferricyanide": feric, "MB": mb, "ferroin": ferroin_epsilon()},
        "sens": sens,
        "filters": {lab: filter_band(lab) for lab in FILTERS},
    }


def setup(ax, ylabel, xlabel=True):
    ax.set_xlim(350, 800)
    ax.set_xticks(X_TICKS)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("Wavelength (nm)")


def draw_relative(
    ax, y, color, xlabel, label=None, loc=(0.97, "right"), channel=None, outline=None
):
    ax.plot(WL, y, color=color, label=label)
    ax.fill_between(WL, 0, y, color=color, alpha=0.12, lw=0)
    ax.set_ylim(-0.03, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    setup(ax, "Relative intensity", xlabel)
    out = {"wavelength_nm": WL, "relative_intensity": y}
    if channel is not None and outline:
        ch, sens = channel
        ax.plot(WL, sens, color=fs.GRAY, ls=outline, label=f"{ch} channel")
        ax.legend(loc=f"upper {loc[1]}", handlelength=1.6, borderaxespad=0.2)
        out[f"{ch}_sensitivity_mean"] = sens
    elif label:
        ax.text(
            loc[0],
            0.95,
            label,
            transform=ax.transAxes,
            ha=loc[1],
            va="top",
            fontsize=fs.FONT_SMALL,
            color=fs.GRAY,
        )
    return pd.DataFrame(out)


def draw_eps(ax, fit, name, xlabel):
    cfg = EPS[name]
    eps, ci = fit["eps"] / cfg["scale"], fit["eps_ci95"] / cfg["scale"]
    ax.fill_between(WL, eps - ci, eps + ci, color=cfg["color"], alpha=0.25, lw=0)
    ax.plot(WL, eps, color=cfg["color"])
    ax.axhline(0, color=fs.GRAY, lw=0.5, ls=":")
    lo, hi = np.nanmin(eps), np.nanmax(eps)
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    for lam, text, frac, dx in cfg["marks"]:
        ax.axvline(lam, color=fs.GRAY, lw=0.6, ls="--")
        ax.text(
            lam + dx,
            lo - pad + frac * (hi - lo + 2 * pad),
            text,
            fontsize=fs.FONT_SMALL,
            color=fs.GRAY,
            va="top",
            ha="left" if dx > 0 else "right",
        )
    unit = r"M$^{-1}$ cm$^{-1}$" if cfg["scale"] == 1 else r"10$^3$ M$^{-1}$ cm$^{-1}$"
    setup(ax, rf"$\varepsilon$ ({unit})", xlabel)
    return pd.DataFrame(
        {"wavelength_nm": WL, "epsilon_M-1cm-1": fit["eps"], "epsilon_ci95_half": fit["eps_ci95"]}
    )


def draw_sens(ax, sens, xlabel):
    out = {"wavelength_nm": WL}
    for i, ch in enumerate("RGB"):
        mu, band = sens["mean"][:, i], sens["band"][:, i]
        col = fs.CHANNEL_COLORS[ch]
        ax.fill_between(
            WL,
            np.clip(mu - band, 0, None),
            np.clip(mu + band, 0, None),
            color=col,
            alpha=0.25,
            lw=0,
        )
        ax.plot(WL, mu, color=col, label=ch)
        out[f"{ch}_mean"], out[f"{ch}_band95_half"] = mu, band
    ax.set_ylim(-0.03, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(
        loc="upper right",
        title=f"n = {sens['n']}",
        handlelength=1.0,
        labelspacing=0.2,
        borderaxespad=0.2,
    )
    setup(ax, "Relative sensitivity", xlabel)
    return pd.DataFrame(out)


def panels(d, outline=None):
    """(letter, drawer) in reading order; drawer(ax, xlabel) -> DataFrame."""
    il, eps, flt = d["illum"], d["eps"], d["filters"]
    mean = d["sens"]["mean"]
    blue, red = ("B", mean[:, 2]), ("R", mean[:, 0])
    return [
        ("a", lambda ax, x: draw_relative(ax, il["D65"], fs.FIT_COLOR, x)),
        ("b", lambda ax, x: draw_relative(ax, il["LED"], fs.FIT_COLOR, x)),
        ("c", lambda ax, x: draw_relative(ax, il["daylight"], fs.FIT_COLOR, x)),
        ("d", lambda ax, x: draw_eps(ax, eps["ferricyanide"], "ferricyanide", x)),
        ("e", lambda ax, x: draw_eps(ax, eps["MB"], "MB", x)),
        ("f", lambda ax, x: draw_eps(ax, eps["ferroin"], "ferroin", x)),
        ("g", lambda ax, x: draw_sens(ax, d["sens"], x)),
        (
            "h",
            lambda ax, x: draw_relative(
                ax,
                flt["450/40"],
                fs.CHANNEL_COLORS["B"],
                x,
                "450/40 filter",
                channel=blue,
                outline=outline,
            ),
        ),
        (
            "i",
            lambda ax, x: draw_relative(
                ax,
                flt["650/10"],
                fs.CHANNEL_COLORS["R"],
                x,
                "650/10 filter",
                (0.03, "left"),
                channel=red,
                outline=outline,
            ),
        ),
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description="Figure 3, revised version")
    p.add_argument(
        "--outline",
        choices=("dotted", "dashed", "none"),
        default="dotted",
        help="line style of the B / R channel sensitivity drawn in "
        "H / I (default dotted, jdc 2026-09-30); any other "
        "choice writes _outline_<style> variants of the "
        "panel, H and I, not copied to Manuscript/",
    )
    args = p.parse_args(argv)
    suffix = "" if args.outline == "dotted" else f"_outline_{args.outline}"
    ls = {"dotted": ":", "dashed": "--"}.get(args.outline)

    fs.apply()
    d = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stems = []

    fig, axes = plt.subplots(3, 3, figsize=(fs.DOUBLE_COL, 3 * fs.ROW_H))
    frames = []
    for k, (ax, (letter, draw)) in enumerate(zip(axes.flat, panels(d, ls))):
        df = draw(ax, k >= 6)
        fs.panel_label(ax, f"({letter})")
        frames.append(df.set_index("wavelength_nm").add_prefix(f"{letter}_"))
    fs.save(fig, OUT_DIR / f"figure_3_panel{suffix}", data=pd.concat(frames, axis=1).reset_index())
    plt.close(fig)
    stems.append(f"figure_3_panel{suffix}")

    for letter, draw in panels(d, ls):
        if suffix and letter not in "hi":
            continue
        fig, ax = plt.subplots(figsize=(fs.SINGLE_COL, fs.SPEC_H))
        df = draw(ax, True)
        fs.panel_label(ax, f"({letter})")
        fs.save(fig, OUT_DIR / f"figure_3{letter}{suffix}", data=df)
        plt.close(fig)
        stems.append(f"figure_3{letter}{suffix}")

    MS_DIR.mkdir(parents=True, exist_ok=True)
    for stem in stems if not suffix else ():
        for ext in ("png", "svg", "csv"):
            shutil.copy2(OUT_DIR / f"{stem}.{ext}", MS_DIR / f"{stem}.{ext}")

    i510 = int(np.argmin(np.abs(WL - 510)))
    print(
        f"ferroin eps(510) = {d['eps']['ferroin']['eps'][i510]:.0f} "
        f"M-1 cm-1 at l = {L_CM} cm, n = {d['eps']['ferroin']['n']}"
    )
    for name, (lam, anchor) in (
        ("ferricyanide", v1.EPS_ANCHOR["ferricyanide"]),
        ("MB", v1.EPS_ANCHOR["mb"]),
    ):
        i = int(np.argmin(np.abs(WL - lam)))
        print(f"{name} eps({lam:.0f}) = {d['eps'][name]['eps'][i]:.0f} (anchor {anchor:.0f})")
    print(f"{len(stems)} figures -> {OUT_DIR}" + ("" if suffix else f", copied to {MS_DIR}"))


if __name__ == "__main__":
    main()
