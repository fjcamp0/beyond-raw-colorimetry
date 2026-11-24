"""
figure_dominant_channel_routes.py — main-text Figure 4.

Calibration of both analytes by three routes, one analyte per row, 3x2
(jdc, 2026-07-28; round 2e W1, gates G-A/G-B/G-G):

  (a) Ferricyanide,   UV-Vis at 420 nm            was Supplementary S5(b)
  (b) Ferricyanide,   camera B      (sRGB + RAW)  was Supplementary S4a
  (c) Ferricyanide,   camera B/G correction (sRGB + RAW)
  (d) Methylene blue, UV-Vis at 665 nm            was Supplementary S5(d)
  (e) Methylene blue, camera R      (sRGB + RAW)  was Supplementary S4b
  (f) Methylene blue, camera R/G correction (sRGB + RAW)

Six panels, relettered (a)-(f) (G-A). The ratio panel plots methylene blue
R/G, not B/G and not R/B (G-B, D8); RdivB stays in Table 2 and the SI.

Panels (c)/(f) gained the sRGB ratio series (round 2g checkpoint correction,
2026-07-28, jdc): both ratio panels now pair sRGB and RAW exactly as (b)/(e)
already do, sRGB ranges from Table S3. Previously RAW-only.

The camera and UV-Vis routes stay in separate panels rather than sharing one
through a secondary axis (jdc, 2026-07-27). They are not on a common
absorbance scale -- UV-Vis is a single-wavelength measurement at a 0.2 cm
cuvette path length -- and a twin axis invited exactly the cross-series
comparison the paper does not make. sRGB and RAW DO share a scale and stay
together in one panel, because their divergence (Eq. 15) is a claim the paper
makes.

UV-Vis fits use pooled individual replicate points, not the per-concentration
means (G-G): ferricyanide n = 30 (the three sets carry 11, 10 and 9 points,
not three copies of 11), methylene blue n = 24 with the corrected 0.7812 uM
relabel. The means-fit numbers (R2 = 0.99989 / 0.99004) are retired.

Non-dominant channels live in the SI.

Named for its content, not for a figure number, and its own filename and
output folder keep that name for the same reason -- figure numbers have been
remapped three times and scripts/figure_3.py, figure_4.py and figure_5.py
already build figures whose numbers do not match their names. The disk-folder
to rendered-number map lives in Manuscript/_drafts/figures/SI_FOLDER_MAP.md.

2026-07-31: output *files* (not the script or its folder) were prefixed
figure_dominant_routes* -> figure_4_dominant_routes* at jdc's explicit
request, purely for findability while this content renders as Figure 4 --
if the number moves again, only these filenames need to move, same as the
existing figure_3.py/figure_4.py situation above.

2026-10-04 (jdc, Reviewer 3 minor 6): panels (b) and (c) -- the ferricyanide
camera panels, whose eleven geometric levels collapsed at the origin on a
linear axis -- now use a log concentration axis and mark the RAW series' LOD
and LOQ (Table S1 v4). Every other panel stays linear. Moved onto figstyle at
the same edit (figure_style.md rule: migrate when next edited).

Run with: python scripts/figure_dominant_channel_routes.py [--sync] [--dry-run]

Outputs -> Data/paper_figures/figure_dominant_routes/
    figure_4_dominant_routes.{png,svg}   six-panel composite
    figure_4_dominant_routes.csv         all plotted points
    figure_4_dominant_routes_fits.csv    fitted slope/intercept/R2/range per series
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import linregress  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJ_ROOT / "Data"

FERIC_RAW = DATA_DIR / "analysis" / "ferricyanide" / "beer_lambert"
MB_RAW = DATA_DIR / "analysis" / "methylene_blue" / "beer_lambert"
UVVIS_DIR = DATA_DIR / "spectra" / "dataanalysis"

OUT_DIR = DATA_DIR / "paper_figures" / "figure_dominant_routes"
# Manuscript-facing folder is "figure_4" (post-final rename pass, 2026-07-29), not
# "figure_dominant_routes" -- this constant pointed at the latter, a folder that no
# longer exists on disk, until fixed 2026-07-31.
DRAFTS_OUT = PROJ_ROOT / "Manuscript" / "_drafts" / "figures" / "figure_4"

# ── Style (figure_style.md, via figstyle) ───────────────────────────────────

fs.apply()

DOUBLE_COL = fs.DOUBLE_COL
PANEL_H = 2.45

GRAY = fs.GRAY
# Route owns the colour (figure_style.md); the channel is named in the label.
# Dark to light separates the ratio from the plain dominant channel.
UV_DARK, UV_MID = fs.ROUTE_COLORS["uv"][0], fs.ROUTE_COLORS["uv"][1]
RAW_DOM, RAW_RATIO = fs.ROUTE_COLORS["raw"][1], fs.ROUTE_COLORS["raw"][0]
JPEG_DOM, JPEG_RATIO = fs.ROUTE_COLORS["jpeg"][2], fs.ROUTE_COLORS["jpeg"][1]

# Linear x by default (jdc, 2026-07-27/28): Beer-Lambert is linear in
# concentration, and a log axis draws a straight calibration as a curve.
# Exception, 2026-10-04 (jdc, Reviewer 3 minor 6): the two ferricyanide camera
# panels set "log_x_cam", because their levels are geometrically spaced and
# eight of eleven fell into the first quarter of a linear axis. The caption says
# the fits there are straight lines in concentration.

MS_MAIN = 3.0
MS_OPEN = 3.4
CAPSIZE = 1.8
ELW = 0.7

# Linear ranges are Table_2_v6's, not refitted here. Points are plotted over
# the full measured range so departure from the fit stays visible -- that
# departure is the ferricyanide kink, which is the argument of panel (b).
# 2026-10-01 (Stephania): the sRGB arm is now the phone's native camera-app JPEG,
# pixel values as saved (scripts/native_jpeg_arm.py); the rawpy-rendered PNGs are no
# longer used anywhere in the paper. Previous outputs: superseded/2026-10-01_rawpy_srgb/.
PANELS = {
    "ferricyanide": {
        "analyte": "Ferricyanide",
        "letters": ("(a)", "(b)", "(c)"),
        "raw_dir": FERIC_RAW,
        "dominant": "B",
        "ratio": "BdivG",
        "ratio_label": "B/G",
        "uvvis_csv": UVVIS_DIR / "beer_lambert_ferricyanide_8mM.csv",
        "uvvis_nm": 420,
        "uvvis_fit_range": None,
        # ratio_srgb: Table S3's "B/G (sRGB)" row -- same 7.8-1000 uM range as
        # the RAW ratio row (added 2026-07-28, round 2g checkpoint correction).
        "ranges": {
            "srgb": (7.8, 500),
            "raw": (7.8, 500),
            "ratio": (7.8, 1000),
            "ratio_srgb": (7.8, 1000),
        },
        "excluded": (),
        # Camera panels stop one level past the widest declared linear range
        # (B/G, to 1000 uM), with a little slack so the 1000 uM points and
        # their SD bars do not sit on the spine. The UV-Vis panel keeps the
        # full span: it is straight across all three decades and needs no
        # truncation.
        #
        # Left bound is -50, not 0 (round 2g checkpoint correction, jdc): an
        # explicit xlim starting exactly at the lowest data value leaves no
        # margin, so the 7.8 uM point sits on the y-axis spine instead of
        # clear of it -- unlike the methylene blue panels below, whose
        # xlim_cam=None lets matplotlib's default ~5% autoscale margin apply.
        # Log axis for (b)/(c) (Reviewer 3 minor 6). Upper bound clears the
        # 1401 uM LOQ of RAW B; the 2000-8000 uM points stay in the SI.
        "xlim_cam": (5, 1600),
        "xlim_uv": None,
        "log_x_cam": True,
        # RAW LOD/LOQ (uM), Table S1 v4 -- not recomputed here.
        "lod_loq": {"raw": (462, 1401), "ratio": (136, 413)},
    },
    "methylene_blue": {
        "analyte": "Methylene blue",
        "letters": ("(d)", "(e)", "(f)"),
        "raw_dir": MB_RAW,
        "dominant": "R",
        "ratio": "RdivG",
        "ratio_label": "R/G",
        "uvvis_csv": UVVIS_DIR / "beer_lambert_mb_665nm.csv",
        "uvvis_nm": 665,
        # eps(665) restricted to 1.525-50 uM, jdc's ruling 2026-07-28 (round 2e
        # Stage 7 checkpoint): keeps eps(665) inside the 70,000-95,000 M-1cm-1
        # literature range for the monomer (74,168 vs the full-range 63,457).
        # Points outside stay plotted, per this script's own convention for the
        # camera panels -- see the module comment above PANELS. Mirrors
        # POOLED_CONC_RANGE in figure_4.py exactly; see that constant's comment
        # for why these two points specifically. Full workings:
        # _ai_outputs/review/mb_epsilon_reversal_2026-07-28.md.
        "uvvis_fit_range": (1.525, 50.0),
        # ratio_srgb: Table S3's "R/G (sRGB)" row, 0.78-25 uM -- narrower than
        # the RAW ratio row, matching the R (sRGB) dominant-channel range
        # (added 2026-07-28, round 2g checkpoint correction).
        "ranges": {
            "srgb": (0.78, 25),
            "raw": (0.78, 50),
            "ratio": (0.78, 50),
            "ratio_srgb": (0.78, 25),
        },
        # Native JPEG: no point is excluded; every level is plotted.
        "excluded": (),
        # Two decades only. Nothing to truncate.
        "xlim_cam": None,
        "xlim_uv": None,
        "log_x_cam": False,
        # RAW LOD/LOQ (uM), Table S2 (renumbered from S1 on 2026-10-06), marked as
        # in (b)/(c) so the two analytes match (Stephania, 2026-10-06).
        "lod_loq": {"raw": (3.09, 9.35), "ratio": (2.12, 6.41)},
    },
}

# ── Data loading ─────────────────────────────────────────────────────────────


def load_channel(path, channel):
    """Per-concentration mean +/- SD for one channel of a calibration_table."""
    df = pd.read_csv(path)
    out = df[df["channel"] == channel].copy()
    if out.empty:
        raise ValueError(f"channel {channel!r} not in {path}")
    return (
        out[["concentration_um", "absorbance_mean", "absorbance_std"]]
        .sort_values("concentration_um")
        .reset_index(drop=True)
    )


def derive_ratio_from_images(path, numerator):
    """
    Per-concentration mean +/- SD of numerator-over-G, computed per image.

    Ratio of transmittances in log space is a difference of absorbances, so
    A_(X/G) = A_X - mean(A_G1, A_G2) image by image -- the identity used for
    every ratio row in Table 2 (note C9). Derived per image rather than from
    the per-concentration means because the two channels are correlated and
    the SD would not otherwise propagate correctly.
    """
    df = pd.read_csv(path)
    wide = df.pivot_table(
        index=["filename", "concentration_um"],
        columns="channel",
        values="absorbance",
    )
    g = wide[["G1", "G2"]].mean(axis=1)
    ratio = (wide[numerator] - g).rename("A").reset_index()
    grp = ratio.groupby("concentration_um")["A"]
    return (
        pd.DataFrame(
            {
                "concentration_um": grp.mean().index,
                "absorbance_mean": grp.mean().values,
                "absorbance_std": grp.std(ddof=1).values,
            }
        )
        .sort_values("concentration_um")
        .reset_index(drop=True)
    )


def load_uvvis(path):
    """
    UV-Vis calibration points, both as pooled individual replicates and as
    per-concentration mean +/- SD.

    G-G rules that every UV-Vis fit printed in the paper uses the pooled
    individual points (ferricyanide n = 30, methylene blue n = 24), so the
    pooled frame is what gets fitted and plotted; the means frame is kept only
    to draw the SD bars.
    """
    df = pd.read_csv(path)
    sets = [c for c in df.columns if c.startswith("set")]
    long = pd.concat(
        [df[["c_uM", s]].dropna().rename(columns={s: "A"}).assign(set=s) for s in sets]
    ).rename(columns={"c_uM": "concentration_um"})
    grp = long.groupby("concentration_um")["A"]
    means = (
        pd.DataFrame(
            {
                "concentration_um": grp.mean().index,
                "absorbance_mean": grp.mean().values,
                "absorbance_std": grp.std(ddof=1).fillna(0.0).values,
            }
        )
        .sort_values("concentration_um")
        .reset_index(drop=True)
    )
    return long.reset_index(drop=True), means


def fit(x, y, lo=None, hi=None):
    r = linregress(x, y)
    return {
        "slope": r.slope,
        "intercept": r.intercept,
        "r2": r.rvalue**2,
        "n": len(x),
        "range_lo": lo,
        "range_hi": hi,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────


def fit_curve(p, lo, hi, log_x=False, n=200):
    """The fitted line sampled across [lo, hi]. Dense because on a log x-axis
    a two-point line is a chord, not the function."""
    xs = np.logspace(np.log10(lo), np.log10(hi), n) if log_x else np.linspace(lo, hi, n)
    return xs, p["slope"] * xs + p["intercept"]


def range_bar(ax, lo, hi, color, slot):
    """Declared linear range, drawn as a capped bar at the foot of the panel.

    The fits are drawn only across their own range, but on a log axis a
    truncated line is not self-evidently truncated, so the range is stated
    graphically as well (round 2e, W1 defect 2). Slots stack upward so two
    series with different ranges -- methylene blue sRGB R to 25 uM, RAW R to
    50 uM -- stay distinguishable.
    """
    y = 0.045 + 0.055 * slot
    ax.plot(
        [lo, hi],
        [y, y],
        transform=ax.get_xaxis_transform(),
        color=color,
        lw=1.2,
        solid_capstyle="butt",
        zorder=1,
    )
    for x in (lo, hi):
        ax.plot(
            [x, x],
            [y - 0.018, y + 0.018],
            transform=ax.get_xaxis_transform(),
            color=color,
            lw=1.2,
            zorder=1,
        )


def draw_series(ax, data, lo, hi, color, marker, label, filled, slot, excluded=(), log_x=False):
    """Points with +/- 1 SD bars over the full measured range; fit and range
    bar over the declared linear range only.

    R2 rides in the legend label rather than in a corner text box, so each
    figure of merit stays attached to its own series and nothing has to be
    positioned around the range bars.
    """
    d = data[~data["concentration_um"].isin(excluded)]
    sel = d[(d["concentration_um"] >= lo) & (d["concentration_um"] <= hi)]
    p = fit(sel["concentration_um"].values, sel["absorbance_mean"].values, lo, hi)

    face = color if filled else "none"
    ax.errorbar(
        d["concentration_um"],
        d["absorbance_mean"],
        yerr=d["absorbance_std"],
        fmt=marker,
        color=color,
        markerfacecolor=face,
        markeredgecolor=color,
        markeredgewidth=0.7,
        markersize=MS_MAIN if filled else MS_OPEN,
        capsize=CAPSIZE,
        elinewidth=ELW,
        linestyle="none",
        zorder=3,
        label=f"{label} ($R^2$ = {p['r2']:.4f})",
    )
    ax.plot(
        *fit_curve(p, lo, hi, log_x),
        ls="--" if filled else ":",
        color=color,
        lw=0.9,
        zorder=2,
    )
    range_bar(ax, lo, hi, color, slot)
    return p


def mark_lod_loq(ax, lod, loq, color):
    """LOD (dotted) and LOQ (dash-dot) as vertical lines, named in the legend
    rather than on the panel, where a label would sit on the data."""
    for x, ls, name in ((lod, ":", "LOD"), (loq, "-.", "LOQ")):
        ax.axvline(x, color=color, ls=ls, lw=0.8, zorder=1, label=f"RAW {name}")


def _yfit(ax, frames, xlim=None, pad_frac=0.16, bottom_frac=0.20):
    """Scale y to the points actually visible, not to data beyond xlim, and
    leave headroom at the foot for the range bars and at the top for the
    legend."""
    lows, highs = [], []
    for d in frames:
        v = d if xlim is None else d[d["concentration_um"] <= xlim[1]]
        sd = v.get("absorbance_std", pd.Series(0.0, index=v.index)).fillna(0.0)
        lows.append((v["absorbance_mean"] - sd).values)
        highs.append((v["absorbance_mean"] + sd).values)
    lo = np.concatenate(lows).min()
    hi = np.concatenate(highs).max()
    span = hi - lo
    ax.set_ylim(lo - bottom_frac * span, hi + pad_frac * span)


def _finish(ax, letter, title, ylabel, xlim=None, legend=True, log_x=False):
    if log_x:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(r"Concentration ($\mu$M)")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        # Data series first, then the LOD/LOQ lines: matplotlib would otherwise
        # list plain lines ahead of the errorbar containers.
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda k: labels[k].startswith("RAW LO"))
        ax.legend(
            [handles[k] for k in order],
            [labels[k] for k in order],
            loc="upper left",
            handletextpad=0.5,
            borderpad=0.3,
            labelspacing=0.25,
        )
    fs.panel_label(ax, letter, x=-0.30, y=1.02)
    ax.set_title(title, fontsize=fs.FONT_BASE)


def _record(rows, key, name, frame):
    for _, r in frame.iterrows():
        rows.append(
            {
                "analyte": key,
                "series": name,
                "concentration_um": r["concentration_um"],
                "absorbance_mean": r["absorbance_mean"],
                "absorbance_std": r["absorbance_std"],
            }
        )


def build_uvvis_panel(ax, key, cfg, rows, fits):
    """UV-Vis alone, on its own absorbance axis, fitted on pooled replicates.

    Points are plotted over the full measured range, per this module's own
    convention (see the comment above PANELS); the fit -- and hence the eps(lambda_max)
    printed in the legend and caption -- is restricted to cfg["uvvis_fit_range"]
    when set (methylene blue only; see that config entry's comment)."""
    pooled, means = load_uvvis(cfg["uvvis_csv"])
    nm = cfg["uvvis_nm"]

    # The individual replicates are the points the pooled fit actually uses
    # (G-G), so they are drawn as the wider open marker with the mean sitting
    # inside them -- the reverse would hide them, there being only 2-3 per
    # level.
    ax.plot(
        pooled["concentration_um"],
        pooled["A"],
        marker="o",
        ls="none",
        mfc="none",
        mec=UV_MID,
        mew=0.6,
        ms=4.4,
        zorder=2,
        label="individual replicates",
    )
    ax.errorbar(
        means["concentration_um"],
        means["absorbance_mean"],
        yerr=means["absorbance_std"],
        fmt="^",
        color=UV_DARK,
        markerfacecolor=UV_DARK,
        markersize=2.6,
        capsize=CAPSIZE,
        elinewidth=ELW,
        linestyle="none",
        zorder=3,
        label="mean $\\pm$ SD",
    )

    fit_range = cfg.get("uvvis_fit_range")
    if fit_range is None:
        lo = float(pooled["concentration_um"].min())
        hi = float(pooled["concentration_um"].max())
    else:
        lo, hi = fit_range
    sel = pooled[(pooled["concentration_um"] >= lo) & (pooled["concentration_um"] <= hi)]
    p = fit(sel["concentration_um"].values, sel["A"].values, lo, hi)
    ax.plot(
        *fit_curve(p, lo, hi),
        ls="--",
        color=UV_DARK,
        lw=0.9,
        zorder=2,
        label=f"OLS ($R^2$ = {p['r2']:.4f}, n = {p['n']})",
    )
    range_bar(ax, lo, hi, UV_DARK, 0)
    fits[(key, f"UV-Vis {nm} nm")] = p

    # Extra headroom: these two panels carry a three-line legend over data that
    # climbs diagonally into it.
    _yfit(
        ax,
        (means, pooled.rename(columns={"A": "absorbance_mean"})),
        cfg["xlim_uv"],
        pad_frac=0.32,
    )
    _record(rows, key, f"UV-Vis {nm} nm", means)

    _finish(
        ax,
        cfg["letters"][0],
        f"{cfg['analyte']} — UV-Vis {nm} nm",
        f"Absorbance at {nm} nm (AU)",
        cfg["xlim_uv"],
    )


def build_camera_panel(ax, key, cfg, rows, fits):
    """sRGB and RAW dominant channel together. No twin axis -- both are camera
    absorbances on one scale, and their divergence (Eq. 15) is the claim."""
    srgb = load_channel(cfg["raw_dir"] / "native_jpeg" / "calibration_table.csv", cfg["dominant"])
    raw = load_channel(cfg["raw_dir"] / "calibration_table.csv", cfg["dominant"])
    excl = tuple(c for arm, c in cfg["excluded"] if arm == "srgb")

    # sRGB first so it sits above RAW in the legend (jdc, 2026-07-28).
    ps = draw_series(
        ax,
        srgb,
        *cfg["ranges"]["srgb"],
        JPEG_DOM,
        "s",
        f"JPEG {cfg['dominant']}",
        filled=False,
        slot=1,
        excluded=excl,
        log_x=cfg["log_x_cam"],
    )
    pr = draw_series(
        ax,
        raw,
        *cfg["ranges"]["raw"],
        RAW_DOM,
        "o",
        f"RAW {cfg['dominant']}",
        filled=True,
        slot=0,
        log_x=cfg["log_x_cam"],
    )
    if cfg["lod_loq"]:
        mark_lod_loq(ax, *cfg["lod_loq"]["raw"], RAW_DOM)
    fits[(key, "JPEG " + cfg["dominant"])] = ps
    fits[(key, "RAW " + cfg["dominant"])] = pr

    _record(rows, key, "JPEG " + cfg["dominant"], srgb)
    _record(rows, key, "RAW " + cfg["dominant"], raw)

    # A gamut-clipped point records A = 4.73 AU, which is not a measurement:
    # the sRGB pixel is driven to zero in the colour matrix, so -log10(0/I0)
    # diverges. Letting it set the y-scale crushes every real series into the
    # bottom tenth of the panel, so it is drawn off-scale at the top instead.
    _yfit(ax, (srgb[~srgb["concentration_um"].isin(excl)], raw), cfg["xlim_cam"])
    for _, c in cfg["excluded"]:
        y = srgb.loc[srgb["concentration_um"] == c, "absorbance_mean"]
        if y.empty:
            continue
        top = ax.get_ylim()[1]
        ax.plot(
            [c],
            [top],
            marker="^",
            color=JPEG_DOM,
            markerfacecolor="none",
            markeredgewidth=0.7,
            markersize=MS_OPEN,
            clip_on=False,
            zorder=4,
        )
        # Parked in the lower half of the panel, clear of the legend, with a
        # leader back to the off-scale marker.
        ax.annotate(
            f"{y.iloc[0]:.2f} AU off scale:\ngamut clipping",
            xy=(c, top),
            xycoords="data",
            xytext=(0.98, 0.42),
            textcoords="axes fraction",
            fontsize=fs.FONT_SMALL,
            color=GRAY,
            ha="right",
            va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=GRAY, shrinkA=2, shrinkB=3),
        )

    _finish(
        ax,
        cfg["letters"][1],
        f"{cfg['analyte']} — camera, {cfg['dominant']}",
        "Absorbance (AU)",
        cfg["xlim_cam"],
        log_x=cfg["log_x_cam"],
    )


def _load_ratio(raw_dir, ratio_key, dominant):
    """Load a ratio channel from calibration_table.csv if it's precomputed there,
    otherwise derive it image by image from absorbance_table.csv in the same folder."""
    try:
        return load_channel(raw_dir / "calibration_table.csv", ratio_key)
    except ValueError:
        return derive_ratio_from_images(raw_dir / "absorbance_table.csv", dominant)


def build_ratio_panel(ax, key, cfg, rows, fits):
    """The ratio channel that corrects the dominant channel: the channel-common
    component divides out, and the declared linear range extends.

    sRGB and RAW together since round 2g checkpoint correction, 2026-07-28 (jdc) --
    mirrors build_camera_panel's pairing in (b)/(e); see this figure's caption for
    why. sRGB ranges and the (f) 100 uM exclusion both come from the same Table S3 /
    cfg["excluded"] sources build_camera_panel already uses."""
    srgb = _load_ratio(cfg["raw_dir"] / "native_jpeg", cfg["ratio"], cfg["dominant"])
    raw = _load_ratio(cfg["raw_dir"], cfg["ratio"], cfg["dominant"])
    excl = tuple(c for arm, c in cfg["excluded"] if arm == "srgb")

    name_srgb = f"JPEG {cfg['ratio_label']}"
    name_raw = f"RAW {cfg['ratio_label']}"

    # sRGB first so it sits above RAW in the legend, matching build_camera_panel.
    ps = draw_series(
        ax,
        srgb,
        *cfg["ranges"]["ratio_srgb"],
        JPEG_RATIO,
        "D",
        name_srgb,
        filled=False,
        slot=1,
        excluded=excl,
        log_x=cfg["log_x_cam"],
    )
    pr = draw_series(
        ax,
        raw,
        *cfg["ranges"]["ratio"],
        RAW_RATIO,
        "D",
        name_raw,
        filled=True,
        slot=0,
        log_x=cfg["log_x_cam"],
    )
    if cfg["lod_loq"]:
        mark_lod_loq(ax, *cfg["lod_loq"]["ratio"], RAW_RATIO)
    fits[(key, name_srgb)] = ps
    fits[(key, name_raw)] = pr
    _record(rows, key, name_srgb, srgb)
    _record(rows, key, name_raw, raw)

    _yfit(ax, (srgb[~srgb["concentration_um"].isin(excl)], raw), cfg["xlim_cam"])

    # Same off-scale treatment as build_camera_panel's dominant-channel gamut-clip
    # point: the ratio inherits the divergence from its sRGB numerator channel.
    for _, c in cfg["excluded"]:
        y = srgb.loc[srgb["concentration_um"] == c, "absorbance_mean"]
        if y.empty:
            continue
        top = ax.get_ylim()[1]
        ax.plot(
            [c],
            [top],
            marker="^",
            color=JPEG_RATIO,
            markerfacecolor="none",
            markeredgewidth=0.7,
            markersize=MS_OPEN,
            clip_on=False,
            zorder=4,
        )
        ax.annotate(
            f"{y.iloc[0]:.2f} AU off scale:\ngamut clipping",
            xy=(c, top),
            xycoords="data",
            xytext=(0.98, 0.42),
            textcoords="axes fraction",
            fontsize=fs.FONT_SMALL,
            color=GRAY,
            ha="right",
            va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=GRAY, shrinkA=2, shrinkB=3),
        )

    _finish(
        ax,
        cfg["letters"][2],
        f"{cfg['analyte']} — camera, {cfg['ratio_label']}",
        "Absorbance ratio (AU)",
        cfg["xlim_cam"],
        log_x=cfg["log_x_cam"],
    )


def main():
    """Build Figure 4, save it through figstyle, and optionally sync it to the drafts."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sync",
        action="store_true",
        help="copy outputs into Manuscript/_drafts/figures/",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="with --sync, report what would be copied only",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 2 * PANEL_H))

    rows = []
    fits = {}
    for row, (key, cfg) in zip(axes, PANELS.items()):
        build_uvvis_panel(row[0], key, cfg, rows, fits)
        build_camera_panel(row[1], key, cfg, rows, fits)
        build_ratio_panel(row[2], key, cfg, rows, fits)

    stem = "figure_4_dominant_routes"
    fs.save(fig, OUT_DIR / stem, data=pd.DataFrame(rows))
    plt.close(fig)

    fit_df = pd.DataFrame([{"analyte": a, "series": s, **p} for (a, s), p in fits.items()])
    fit_df.to_csv(OUT_DIR / f"{stem}_fits.csv", index=False, float_format="%.6g")

    w, h = fig.get_size_inches()
    print(f"[figure_4_dominant_routes] -> {OUT_DIR} ({w:.2f} x {h:.2f} in)")
    print(fit_df.to_string(index=False))

    if args.sync:
        DRAFTS_OUT.mkdir(parents=True, exist_ok=True)
        for p in sorted(OUT_DIR.glob(f"{stem}.*")) + sorted(OUT_DIR.glob(f"{stem}_fits.*")):
            dst = DRAFTS_OUT / p.name
            verb = "would copy" if args.dry_run else "copied"
            print(f"  {verb}: {p.name} -> {dst}")
            if not args.dry_run:
                shutil.copy2(p, dst)


if __name__ == "__main__":
    main()
