"""figure_optical_filters.py — publication figure of the 23 Sep LED / bandpass-filter spectra.

Four panels, ACS double column, style per `figure_style.md`:
    A  LED emission with and without the diffuser, each normalised to its maximum
    B  spectral transmission of the diffuser: (with / without), normalised at 550 nm
    C  LED spectrum through each bandpass filter (without diffuser), normalised to its own
       peak: the band the camera actually receives, LED x filter. Dividing by the bare lamp
       spectrum to get filter transmission is unstable where the LED is weak (blue edge,
       480 nm dip), so it is reported in filter_metrics.csv only, not plotted
    D  lamp output over the session: peak signal of repeated configurations relative to
       the first measurement of each, against clock time

Only spectral shapes are compared across files (A-C): the lamp faded during the session
(panel D), so absolute intensities from different times are not comparable. Counts >= 1000
are written "1,030.7" in the SPELEC export; the comma is stripped before conversion.

Inputs  Data/20260923_OpticalFilters_spelec/*.csv (SPELEC exports, 23 Sep 2026)
Outputs Data/paper_figures/optical_filters/
            figure_optical_filters.{png,svg,csv}          four-panel figure (600 dpi) + data
            figure_optical_filters_<A-D>_<slug>.{png,svg,csv}   each panel alone, single column
            filter_metrics.csv                 measured centre and FWHM per filter

Run from the project root:
    python scripts/figure_optical_filters.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "Data" / "20260923_OpticalFilters_spelec"
OUT_DIR = PROJ / "Data" / "paper_figures" / "optical_filters"
NAME = "figure_optical_filters"

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        8,     # ACS: ≥ 4.5 pt floor; 8 pt nominal for readability
    "axes.labelsize":   8,
    "axes.titlesize":   8,
    "legend.fontsize":  7,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "axes.linewidth":   0.6,   # ACS min line thickness 0.5 pt
    "lines.linewidth":  0.8,
    "lines.markersize": 3.5,
    "patch.linewidth":  0.5,
    "figure.dpi":       150,   # screen preview only; output always 600 dpi
})  # fmt: skip

SINGLE_COL = 3.33  # in — ACS single-column max (240 pt)
DOUBLE_COL = 7.00  # in — ACS double-column max (504 pt)
MAX_DEPTH = 9.167  # in — ACS max figure depth incl. caption (660 pt)

FIT_COLOR = "#2C3E50"
GRAY = "#626567"
SET_COLORS = ("#2E86AB", "#A23B72", "#F18F01")
SET_MARKERS = ("o", "s", "^")

BIAS_BAND_NM = 850  # above this the LED emits nothing: median counts there = detector bias
LED_FLOOR = 0.03  # divide by the lamp spectrum only where it exceeds 3 % of its maximum
SMOOTH_PTS = 5  # ~1.8 nm boxcar at 0.36 nm/pixel, for peak and FWHM only

LED_FILES = {"with diffuser": "Experiment_1_LED_with Diffuser",
             "without diffuser": "Experiment_1_LED_without Diffuser"}  # fmt: skip
# One spectrum per filter, without diffuser (highest signal-to-noise); label -> file
FILTER_FILES = {
    "450/10": "Experiment_13_Filter_450nm_10nm_LED_withoutdiffuser_100ms",
    "450/40": "Experiment_14_Filter_450nm_40nm_LED_withoutdiffuser_10ms",
    "500/10": "Experiment_17_Filter_500nm_10nm_LED_withoutdiffuser_100ms",
    "550/40": "Experiment_19_Filter_550nm_40nm_LED_withoutdiffuser_10ms",
    "650/10": "Experiment_9_Filter_650nm_10nm_LED_withoutdiffuser_10ms",
    "650/40": "Experiment_23_Filter_650nm_40nm_LED_withoutdiffuser_10ms",
}
# Configurations measured twice: label -> [(file, integration time ms, clock time)]
REPEATS = {
    "650/10, no diffuser": [
        ("Experiment_9_Filter_650nm_10nm_LED_withoutdiffuser_10ms", 10, "14:23:48"),
        ("Experiment_24_Filter_650nm_10nm_LED_withoutdiffuser_10ms", 10, "16:21:16")],
    "650/10, diffuser": [
        ("Experiment_10_Filter_650nm_10nm_LED_withdiffuser_10ms", 10, "14:26:06"),
        ("Experiment_25_Filter_650nm_10nm_LED_withdiffuser_100ms", 100, "16:22:46")],
    "500/10, no diffuser": [
        ("Experiment_17_Filter_500nm_10nm_LED_withoutdiffuser_100ms", 100, "15:13:24"),
        ("Experiment_18_Filter_500nm_10nm_LED_withoutdiffuser_10ms", 10, "15:18:14")],
}  # fmt: skip


def read_spelec(stem):
    """(wavelength nm, bias-subtracted counts) from one SPELEC CSV export."""
    d = pd.read_csv(DATA / f"{stem}.csv", sep=";", skiprows=2, header=None, usecols=[0, 1],
                    dtype=str)  # fmt: skip
    d = d.apply(lambda s: pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce"))
    d = d.dropna().sort_values(0)
    w, c = d[0].to_numpy(), d[1].to_numpy()
    return w, c - np.median(c[w > BIAS_BAND_NM])


def smooth(y):
    """Boxcar smoothing used only for locating peaks and half-maximum crossings."""
    return np.convolve(y, np.ones(SMOOTH_PTS) / SMOOTH_PTS, mode="same")


def centre_fwhm(w, y):
    """Peak wavelength and FWHM (nm) of the contiguous band around the maximum."""
    s = smooth(y)
    k = int(np.argmax(s))
    half = s[k] / 2
    lo, hi = k, k
    while lo > 0 and s[lo] > half:
        lo -= 1
    while hi < len(s) - 1 and s[hi] > half:
        hi += 1
    # linear interpolation of the two half-maximum crossings
    x_lo = np.interp(half, [s[lo], s[lo + 1]], [w[lo], w[lo + 1]])
    x_hi = np.interp(half, [s[hi], s[hi - 1]], [w[hi], w[hi - 1]])
    return w[k], x_hi - x_lo


def panel_letter(ax, letter):
    """Bold panel letter, upper left, per figure_style.md."""
    ax.text(-0.15, 1.05, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")


def tidy(ax):
    """Remove top and right spines, per figure_style.md."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, stem, rows):
    """PNG + SVG at 600 dpi and the companion CSV of the plotted data."""
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=600, bbox_inches="tight", format=ext)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"{stem}.csv", index=False)
    plt.close(fig)


def render(panels, name, nrows, ncols, height, single_height=2.6):
    """Multi-panel figure, plus each panel alone at single-column width (no panel letter).

    panels: list of (letter, slug, draw) where draw(ax) plots and returns its data rows.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL, height))
    rows = []
    for ax, (letter, _, draw) in zip(np.ravel(axes), panels):
        tidy(ax)
        rows += [{"panel": letter, **r} for r in draw(ax)]
        panel_letter(ax, letter)
    fig.tight_layout(pad=0.5)
    save(fig, name, rows)
    for letter, slug, draw in panels:
        fig, ax = plt.subplots(figsize=(SINGLE_COL, single_height))
        tidy(ax)
        panel_rows = draw(ax)
        fig.tight_layout(pad=0.5)
        save(fig, f"{name}_{letter}_{slug}", panel_rows)


def draw_led(ax):
    """A: LED emission with and without diffuser, each normalised to its maximum."""
    rows = []
    for (lab, stem), col, ls in zip(LED_FILES.items(), SET_COLORS, ("-", "--")):
        w, c = read_spelec(stem)
        y = c / smooth(c).max()
        ax.plot(w, y, color=col, ls=ls, label=lab)
        rows += [{"series": lab, "x": xi, "y": yi} for xi, yi in zip(w, y)]
    ax.annotate("blue pump", xy=(455, 0.76), xytext=(372, 0.9), fontsize=7, color=FIT_COLOR,
                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": FIT_COLOR})  # fmt: skip
    ax.text(585, 1.03, "phosphor", fontsize=7, color=FIT_COLOR, ha="center")
    ax.set(xlim=(350, 800), ylim=(-0.02, 1.12), xlabel="Wavelength (nm)",
           ylabel="Normalised emission")  # fmt: skip
    ax.legend(frameon=True, framealpha=0.85, edgecolor="none", loc="upper right")
    return rows


def draw_diffuser(ax):
    """B: spectral transmission of the diffuser, (with / without), 1 at 550 nm."""
    (w_d, c_d), (w_n, c_n) = (read_spelec(s) for s in LED_FILES.values())
    c_n_on_d = np.interp(w_d, w_n, c_n)
    ok = (c_d > LED_FLOOR * c_d.max()) & (c_n_on_d > LED_FLOOR * c_n_on_d.max())
    ratio = smooth(c_d) / smooth(c_n_on_d)
    ratio = ratio / np.interp(550, w_d, ratio)
    ax.axhline(1, color=GRAY, lw=0.6, ls=":")
    ax.plot(w_d[ok], ratio[ok], color=FIT_COLOR)
    ax.set(xlim=(350, 800), ylim=(0.7, 1.6), xlabel="Wavelength (nm)",
           ylabel="Diffuser transmission\n(relative, 1 at 550 nm)")  # fmt: skip
    return [{"series": "diffuser with/without, norm. 550 nm", "x": xi, "y": yi}
            for xi, yi in zip(w_d[ok], ratio[ok])]  # fmt: skip


def draw_filters(ax):
    """C: LED spectrum through each bandpass filter, normalised to its own peak."""
    rows = []
    for lab, stem in FILTER_FILES.items():
        w, c = read_spelec(stem)
        narrow = lab.endswith("/10")
        y = c / smooth(c).max()
        band = (w > 420) & (w < 700)
        ax.plot(w[band], y[band], color=FIT_COLOR if narrow else GRAY,
                ls="-" if narrow else "--", lw=0.7)  # fmt: skip
        if narrow or lab == "550/40":  # one centre label per band; line style gives the width
            ax.text(int(lab.split("/")[0]), 1.05, f"{lab.split('/')[0]} nm", fontsize=7,
                    ha="center", color=FIT_COLOR)  # fmt: skip
        rows += [{"series": lab, "x": xi, "y": yi} for xi, yi in zip(w[band], y[band])]
    ax.plot([], [], color=FIT_COLOR, label="10 nm nominal")
    ax.plot([], [], color=GRAY, ls="--", label="40 nm nominal")
    ax.legend(frameon=True, framealpha=0.85, edgecolor="none", loc="center",
              bbox_to_anchor=(0.72, 0.55))  # fmt: skip
    ax.set(xlim=(420, 700), ylim=(-0.02, 1.15), xlabel="Wavelength (nm)",
           ylabel="Normalised signal (LED × filter)")  # fmt: skip
    return rows


def draw_lamp(ax):
    """D: peak signal per ms of repeated configurations, relative to the first, vs time."""
    rows = []
    for (lab, reps), col, mk in zip(REPEATS.items(), SET_COLORS, SET_MARKERS):
        vals, hours = [], []
        for stem, t_ms, clock in reps:
            _, c = read_spelec(stem)
            vals.append(smooth(c).max() / t_ms)  # counts per ms: integration times differ
            t = datetime.strptime(clock, "%H:%M:%S")
            hours.append(t.hour + t.minute / 60 + t.second / 3600)
        rel = np.array(vals) / vals[0]
        ax.plot(hours, rel, color=col, marker=mk, ms=4.5)
        pct = f"{100 * (rel[-1] - 1):+.0f} %".replace("-", "\u2212")
        ax.text(hours[-1] + 0.1, rel[-1], f"{lab}\n{pct}", fontsize=7, va="center",
                color=FIT_COLOR)  # fmt: skip
        rows += [{"series": lab, "x": h, "y": r, "file": s}
                 for h, r, (s, _, _) in zip(hours, rel, reps)]  # fmt: skip
    ax.set_yscale("log")
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1])
    ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
    ax.minorticks_off()
    ax.set(xlim=(14.0, 17.6), ylim=(0.04, 1.4), xlabel="Clock time, 23 Sep 2026",
           ylabel="Peak signal per ms,\nrelative to first measurement")  # fmt: skip
    ax.set_xticks([14, 15, 16])
    ax.set_xticklabels(["14:00", "15:00", "16:00"])
    return rows


def filter_metrics():
    """Measured centre and FWHM per filter, with and without dividing by the bare LED."""
    w_led, c_led = read_spelec(LED_FILES["without diffuser"])
    out = []
    for lab, stem in FILTER_FILES.items():
        w, c = read_spelec(stem)
        lamp = np.interp(w, w_led, smooth(c_led))
        ok = lamp > LED_FLOOR * lamp.max()
        t = np.where(ok, c / np.where(ok, lamp, 1), np.nan)
        t = t / np.nanmax(smooth(np.nan_to_num(t)))
        c_nm, fwhm = centre_fwhm(w[ok], np.nan_to_num(t[ok]))
        c_raw, fwhm_raw = centre_fwhm(w, c)
        out.append({"filter": lab, "nominal_centre_nm": int(lab.split("/")[0]),
                    "nominal_fwhm_nm": int(lab.split("/")[1]),
                    "centre_nm": round(c_raw, 1), "fwhm_nm": round(fwhm_raw, 1),
                    "centre_nm_lamp_divided": round(c_nm, 1),
                    "fwhm_nm_lamp_divided": round(fwhm, 1), "file": stem})  # fmt: skip
    return pd.DataFrame(out)


def main():
    """Four-panel figure, the four single-panel figures, and the filter metrics table."""
    render([("A", "led_emission", draw_led), ("B", "diffuser", draw_diffuser),
            ("C", "filter_bands", draw_filters), ("D", "lamp_fading", draw_lamp)],
           NAME, 2, 2, 5.0)  # fmt: skip
    m = filter_metrics()
    m.to_csv(OUT_DIR / "filter_metrics.csv", index=False)
    print(m.drop(columns="file").to_string(index=False))


if __name__ == "__main__":
    main()
