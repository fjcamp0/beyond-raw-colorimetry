"""Wine case study: iron in two wines by UV-Vis, RAW and native-app JPEG.

Builds on the verified per-image and UV-Vis tables of
Data/analysis/current_recalculation_2026-09-30. The two active JPEG series
are re-read as stored 8-bit RGB, without inverse linearisation. No archived
input is used. Scope: Manuscript/WINE_DATA_SCOPE.md -- 2000 uL wine
in 9000 uL, wine-specific no-phenanthroline controls, dilution 4.5, 0/8/32 uM
added Fe.

Decisions (Stephania, 2026-09-30):
- Every wine signal is referenced to that wine's no-phenanthroline control,
  never to water.
- Main camera signal G/R (green absorbs, red is the near-blind reference);
  B/G, G and B are reported alongside.
- Two routes, both reported: standard addition inside each wine, and an
  external calibration (UV-Vis sets 1-3; RAW sets 1-3 pooled, ratios only;
  JPEG 30 Sep ladder fitted 0-24 uM).
- JPEG standard addition is fitted on 0 and +8 uM only; +32 uM is shown, not
  fitted, to expose the native app's automatic processing.
- Per-spike recoveries from a wine's own standard-addition line are partly
  circular and are labelled so.

Iron scale (jdc, 2026-10-02): every labelled standard and spike concentration is
multiplied by FE_SCALE = 1.636 on reading (ferroin_scale.py), since the FeSO4.7H2O
is assumed to have effloresced to the monohydrate. Spikes 8/32 uM become
13.1/52.4 uM; standards 0-56 uM become 0-91.6 uM. Recoveries are unchanged.

Run from the project root: python scripts/wine_case_study.py
"""

import hashlib
import json
import os
from pathlib import Path

import matplotlib
from PIL import Image

matplotlib.use("Agg")
import figstyle as fs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ferroin_scale import FE_SCALE, true_uM
from forward_model import split_at_cutoff
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numfmt import pm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Data/analysis/current_recalculation_2026-09-30"
OUT = ROOT / "Data/analysis/wine_case_study_2026-09-30"
FIG = ROOT / "Data/paper_figures/wine_case_study"

DILUTION = 4.5
FE_MOLAR_MASS = 55.845
MAIN = "G/R"
SIGNALS = ["G/R", "B/G", "G", "B"]
RAW_WINES = "Wine_realsample_anaysis_Motioncam_nofilter_2"
JPEG_WINES = "Wine_realsample_spiked_native app_fixed ISO100_30fps_nophencontrol"
RAW_STANDARDS = ["Set1_Motioncam_nofilter", "Set2_Motioncam_nofilter", "Set3_Motioncam_nofilter"]
JPEG_STANDARDS = "Phenanthroline_Native-aap_Controlled conditions_ISO100_30FPS"
JPEG_CAL_UPPER = true_uM(24.0)
CAL_TOP = true_uM(56.0)
SPIKES = {8: true_uM(8.0), 32: true_uM(32.0)}  # labelled -> true uM
CEILING_PCT = 1.0  # a photo counts as saturated above 1 % of ROI pixels at the ceiling


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    res = y - (slope * x + intercept)
    ss = np.sum((y - y.mean()) ** 2)
    return dict(
        n=len(x),
        slope=slope,
        intercept=intercept,
        r2=1 - np.sum(res**2) / ss,
        rmse=float(np.sqrt(np.mean(res**2))),
    )


def mg_per_L(c_cuvette_uM):
    return c_cuvette_uM * DILUTION * FE_MOLAR_MASS / 1000


def add_signals(frame, a_col):
    """Wide per-image table with channel absorbances and ratio absorbances.

    A ratio absorbance is -log10 of the ratio of transmittances, i.e. the
    difference of the two channel absorbances.
    """
    keys = ["dataset", "file", "kind", "wine", "concentration_uM", "date"]
    frame = frame.assign(wine=frame.wine.fillna(""))
    wide = frame.pivot_table(index=keys, columns="channel", values=a_col).reset_index()
    sat = frame.pivot_table(index="file", columns="channel", values="near_ceiling_pct").add_prefix(
        "sat_"
    )
    wide = wide.merge(sat, left_on="file", right_index=True)
    wide["G/R"] = wide.G - wide.R
    wide["B/G"] = wide.B - wide.G
    return wide


def camera_tables():
    ch = pd.read_csv(SRC / "image_channels.csv")
    ch = ch[ch.roi_fraction == 0.25]
    ch = stored_jpeg_channels(ch)
    ch = ch.assign(concentration_uM=true_uM(ch.concentration_uM))

    # Wines: absorbance against the same wine's no-phenanthroline control.
    wines = ch[ch.dataset.isin([RAW_WINES, JPEG_WINES]) & ch.wine.notna()].copy()
    ref = (
        wines[wines.kind == "control"]
        .groupby(["dataset", "wine", "channel"])
        .intensity.mean()
        .rename("I_control")
    )
    wines = wines.join(ref, on=["dataset", "wine", "channel"])
    wines["A_vs_control"] = -np.log10(wines.intensity / wines.I_control)
    wines = add_signals(wines, "A_vs_control")
    wines["arm"] = np.where(wines.dataset == RAW_WINES, "RAW", "JPEG")

    # Standards: absorbance against the water blank of the same folder.
    std = ch[ch.dataset.isin(RAW_STANDARDS + [JPEG_STANDARDS]) & (ch.kind == "standard")]
    std = add_signals(std, "A_pooled")
    std["arm"] = np.where(std.dataset == JPEG_STANDARDS, "JPEG", "RAW")
    return wines, std


def stored_jpeg_channels(ch):
    """Replace the two JPEG series with means of stored 8-bit RGB values.

    Retain the existing central ROI and sample mapping. Do not undo gamma,
    apply white balance, rescale brightness or modify the source JPEGs.
    """
    ch = ch.copy()
    mask = ch.dataset.isin([JPEG_WINES, JPEG_STANDARDS])
    manifest = []
    for source, group in ch[mask].groupby("source"):
        path = ROOT / source
        assert not any(
            part.lower() in ("old measurements", "old_results", "previous") for part in path.parts
        ), f"Archived JPEG forbidden: {path}"
        # On Windows, paths in this OneDrive folder can exceed 260 characters.
        long = Path("\\\\?\\" + str(path.resolve())) if os.name == "nt" else path
        with Image.open(long) as im:
            rgb = np.asarray(im.convert("RGB"))
        h, w = rgb.shape[:2]
        frac = 0.25
        roi = rgb[
            h // 2 - int(h * frac / 2) : h // 2 + int(h * frac / 2),
            w // 2 - int(w * frac / 2) : w // 2 + int(w * frac / 2),
        ]
        for i, channel in enumerate("RGB"):
            index = group.index[group.channel == channel]
            values = roi[..., i].astype(float)
            ch.loc[index, "intensity"] = values.mean()
            ch.loc[index, "pixel_sd"] = values.std()
            ch.loc[index, "near_ceiling_pct"] = 100 * (values >= 254).mean()
            ch.loc[index, "floor_pct"] = 100 * (values <= 1).mean()
            ch.loc[index, "arm"] = "JPEG_stored_RGB"
        with long.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest.append(dict(source=source, bytes=long.stat().st_size, sha256=digest))
    ch.loc[mask, ["A_pooled", "A_interpolated"]] = np.nan
    keys = ["dataset", "date", "iso", "exposure", "roi_fraction", "channel"]
    for _, group in ch[mask].groupby(keys, dropna=False):
        blank = group[group.kind == "water"].intensity.mean()
        if np.isfinite(blank) and blank > 0:
            ch.loc[group.index, "A_pooled"] = -np.log10(group.intensity / blank)
    OUT.mkdir(parents=True, exist_ok=True)
    ch[mask].to_csv(OUT / "jpeg_stored_rgb_channels.csv", index=False)
    pd.DataFrame(manifest).to_csv(OUT / "jpeg_stored_rgb_manifest.csv", index=False)
    assert len(manifest) == 57, f"Unexpected JPEG source count: {len(manifest)}"
    return ch


def uvvis_tables():
    cal = pd.read_csv(SRC / "ferroin_uvvis_calibration.csv")
    labelled = json.loads((SRC / "ferroin_uvvis_fit.json").read_text())
    cal["concentration_uM"] = true_uM(cal.concentration_uM)
    calfit = fit(cal.concentration_uM, cal.A510_corrected)
    assert abs(calfit["slope"] * FE_SCALE / labelled["slope"] - 1) < 1e-9
    w = pd.read_csv(SRC / "wine_uvvis_measurements.csv")
    w["added_uM"] = true_uM(w.added_uM)
    ctrl = w[w.kind == "control"].groupby("wine").A510.mean().rename("A_control")
    w = w.join(ctrl, on="wine")
    w["A_vs_control"] = w.A510 - w.A_control
    return cal, calfit, w


def summarise(arm, signal, wine, samples, ext_fit, ext_note, sa_levels, sat_col=None):
    """One tidy row per wine x arm x signal.

    samples: frame with columns added_uM and A (the wine samples, not controls).
    """
    g = samples.groupby("added_uM").A
    mean, sd = g.mean(), g.std()
    row = dict(wine=wine, arm=arm, signal=signal, main_signal=signal in (MAIN, "A510"))

    sa_pts = samples[samples.added_uM.isin(sa_levels)]
    sa = fit(sa_pts.added_uM, sa_pts.A)
    c_sa = sa["intercept"] / sa["slope"]
    row.update(
        sa_levels_uM="/".join(f"{v:g}" for v in sa_levels),
        sa_n=sa["n"],
        sa_slope_per_uM=sa["slope"],
        sa_intercept=sa["intercept"],
        sa_r2=sa["r2"],
        sa_rmse=sa["rmse"],
        sa_native_uM=c_sa,
        sa_wine_Fe_mg_L=mg_per_L(c_sa),
    )
    for nom, lvl in SPIKES.items():
        rec = 100 * (mean[lvl] - mean[0.0]) / sa["slope"] / lvl
        row[f"sa_recovery_{nom}uM_pct"] = rec
        row[f"sa_recovery_{nom}uM_fitted"] = lvl in sa_levels

    if ext_fit is not None:
        c_ext = (mean[0.0] - ext_fit["intercept"]) / ext_fit["slope"]
        row.update(
            ext_calibration=ext_note,
            ext_n=ext_fit["n"],
            ext_slope_per_uM=ext_fit["slope"],
            ext_intercept=ext_fit["intercept"],
            ext_r2=ext_fit["r2"],
            ext_rmse=ext_fit["rmse"],
            ext_native_uM=c_ext,
            ext_wine_Fe_mg_L=mg_per_L(c_ext),
        )
        recs = []
        for nom, lvl in SPIKES.items():
            rec = 100 * (mean[lvl] - mean[0.0]) / ext_fit["slope"] / lvl
            row[f"ext_recovery_{nom}uM_pct"] = rec
            recs.append(rec)
        row["ext_recovery_min_pct"], row["ext_recovery_max_pct"] = min(recs), max(recs)
        row["slope_ratio_wine_over_standards"] = sa["slope"] / ext_fit["slope"]
    else:
        row["ext_calibration"] = ext_note
    if sat_col is not None:
        for nom, lvl in {0: 0.0, **SPIKES}.items():
            row[f"saturated_pct_max_{nom}uM"] = samples.loc[samples.added_uM == lvl, sat_col].max()
    row["A_sd_max"] = sd.max()
    return row


def slope_se(x, y, f):
    """Standard error of an OLS slope."""
    x = np.asarray(x, float)
    s = f["rmse"] * np.sqrt(f["n"] / (f["n"] - 2))
    return s / np.sqrt(np.sum((x - x.mean()) ** 2))


def intercept_conc_se(x, y, f):
    """SD of the standard-addition result, -intercept/slope (Miller & Miller, eq. 5.13)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = f["rmse"] * np.sqrt(f["n"] / (f["n"] - 2))
    b = f["slope"]
    return s / abs(b) * np.sqrt(1 / len(x) + y.mean() ** 2 / (b**2 * np.sum((x - x.mean()) ** 2)))


def table2_with_sd(pts, calfit, cal, std):
    """Table 2 rows (UV-Vis A510, RAW G/R, JPEG G/R) with standard deviations.

    Slopes: regression standard error. Fe by calibration and recoveries: mean +/- SD of
    the three replicate determinations. Fe by standard addition: SD of the extrapolated
    x-intercept. Stephania, 2026-10-01.
    """
    cals = {"UV-Vis": (cal.concentration_uM, cal.A510_corrected)}
    for arm, upper in (("RAW", CAL_TOP), ("JPEG", JPEG_CAL_UPPER)):
        s = std[(std.arm == arm) & (std.concentration_uM <= upper + 1e-9)]
        cals[arm] = (s.concentration_uM, s[MAIN])
    rows = []
    for wine in ("Wine1", "Wine2"):
        for arm, sig in (("UV-Vis", "A510"), ("RAW", MAIN), ("JPEG", MAIN)):
            cx, cy = cals[arm]
            fc = fit(cx, cy)
            p = pts[(pts.arm == arm) & (pts.signal == sig) & (pts.wine == wine)]
            levels = [0.0, SPIKES[8]] if arm == "JPEG" else [0.0, SPIKES[8], SPIKES[32]]
            sp = p[p.added_uM.isin(levels)]
            fs_ = fit(sp.added_uM, sp.A)
            a0 = p[p.added_uM == 0].A
            c_cal = mg_per_L((a0 - fc["intercept"]) / fc["slope"])
            c_sa = fs_["intercept"] / fs_["slope"]
            row = dict(
                wine=wine,
                arm=arm,
                signal=sig,
                cal_slope=fc["slope"],
                cal_slope_se=slope_se(cx, cy, fc),
                cal_r2=fc["r2"],
                sa_slope=fs_["slope"],
                sa_slope_se=slope_se(sp.added_uM, sp.A, fs_),
                sa_r2=fs_["r2"],
                Fe_cal_mg_L=c_cal.mean(),
                Fe_cal_sd=c_cal.std(ddof=1),
                Fe_sa_mg_L=mg_per_L(c_sa),
                Fe_sa_sd=mg_per_L(intercept_conc_se(sp.added_uM, sp.A, fs_)),
            )
            for nom, lvl in SPIKES.items():
                rec = 100 * (p[p.added_uM == lvl].A - a0.mean()) / fc["slope"] / lvl
                row[f"rec_{nom}"] = rec.mean()
                row[f"rec_{nom}_sd"] = rec.std(ddof=1)
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table2_with_sd.csv", index=False)
    # The same table as printed in the manuscript (hard rule 10: SD to two significant figures).
    f = pd.DataFrame(
        {
            "Wine": t.wine.str[-1],
            "Method": t.arm,
            "Signal": t.signal.replace({"A510": "A510"}),
            "Sensitivity, standards (1e-3 AU/uM)": [
                pm(a * 1e3, b * 1e3) for a, b in zip(t.cal_slope, t.cal_slope_se)
            ],
            "R2, standards": t.cal_r2.round(4),
            "Slope in wine (1e-3 AU/uM)": [
                pm(a * 1e3, b * 1e3) for a, b in zip(t.sa_slope, t.sa_slope_se)
            ],
            "R2, in wine": t.sa_r2.round(4),
            "Fe, calibration (mg/L)": [pm(a, b) for a, b in zip(t.Fe_cal_mg_L, t.Fe_cal_sd)],
            "Fe, st. addition (mg/L)": [pm(a, b) for a, b in zip(t.Fe_sa_mg_L, t.Fe_sa_sd)],
            f"Recovery +{SPIKES[8]:.1f} uM (%)": [pm(a, b) for a, b in zip(t.rec_8, t.rec_8_sd)],
            f"Recovery +{SPIKES[32]:.1f} uM (%)": [pm(a, b) for a, b in zip(t.rec_32, t.rec_32_sd)],
        }
    )
    f.to_csv(OUT / "table2_formatted.csv", index=False, encoding="utf-8-sig")
    return t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cal, calfit, uv = uvvis_tables()
    wines, std = camera_tables()

    rows, points = [], []

    # UV-Vis, the reference.
    for wine, g in uv[uv.kind == "wine"].groupby("wine"):
        s = g.rename(columns={"added_uM": "added_uM", "A_vs_control": "A"})
        rows.append(
            summarise(
                "UV-Vis",
                "A510",
                wine,
                s,
                calfit,
                "UV-Vis sets 1-3 pooled, 0-91.6 uM",
                [0.0, SPIKES[8], SPIKES[32]],
            )
        )
        points.append(
            s.assign(arm="UV-Vis", signal="A510", saturated=False)[
                ["arm", "signal", "wine", "added_uM", "A", "saturated"]
            ]
        )

    # External camera calibrations.
    ext = {}
    raw_std = std[std.arm == "RAW"]
    jpg_std = std[(std.arm == "JPEG") & (std.concentration_uM <= JPEG_CAL_UPPER + 1e-9)]
    for sig in SIGNALS:
        if "/" in sig:  # RAW single channels carry the dim 0 uM frames of sets 1-2 (Q3)
            ext[("RAW", sig)] = (
                fit(raw_std.concentration_uM, raw_std[sig]),
                "RAW sets 1-3 pooled, 0-91.6 uM",
            )
        else:
            ext[("RAW", sig)] = (
                None,
                "not used: single channels carry the dim 0 uM frames of sets 1-2",
            )
        ext[("JPEG", sig)] = (
            fit(jpg_std.concentration_uM, jpg_std[sig]),
            "JPEG 30 Sep ladder, 0-39.3 uM",
        )

    for (arm, wine), g in wines[wines.kind == "wine"].groupby(["arm", "wine"]):
        levels = [0.0, SPIKES[8], SPIKES[32]] if arm == "RAW" else [0.0, SPIKES[8]]
        for sig in SIGNALS:
            s = g.rename(columns={"concentration_uM": "added_uM", sig: "A"})
            chans = sig.split("/")
            s["sat"] = s[[f"sat_{c}" for c in chans]].max(axis=1)
            f_ext, note = ext[(arm, sig)]
            rows.append(summarise(arm, sig, wine, s, f_ext, note, levels, "sat"))
            points.append(
                s.assign(arm=arm, signal=sig, saturated=s.sat > CEILING_PCT)[
                    ["arm", "signal", "wine", "added_uM", "A", "saturated"]
                ]
            )

    table = pd.DataFrame(rows)
    uvref = table[table.arm == "UV-Vis"].set_index("wine").ext_native_uM
    table["ext_diff_vs_uvvis_pct"] = 100 * (table.ext_native_uM / table.wine.map(uvref) - 1)
    table["sa_diff_vs_uvvis_pct"] = 100 * (table.sa_native_uM / table.wine.map(uvref) - 1)
    table.to_csv(OUT / "wine_case_study_results.csv", index=False)
    pts = pd.concat(points, ignore_index=True)
    pts.to_csv(OUT / "wine_case_study_points.csv", index=False)

    calrows = [
        dict(
            arm="UV-Vis",
            signal="A510",
            calibration="UV-Vis sets 1-3 pooled, 0-91.6 uM",
            **{k: calfit[k] for k in ("n", "slope", "intercept", "r2", "rmse")},
        )
    ]
    for (arm, sig), (f_ext, note) in ext.items():
        if f_ext is not None:
            calrows.append(dict(arm=arm, signal=sig, calibration=note, **f_ext))
    pd.DataFrame(calrows).to_csv(OUT / "external_calibrations.csv", index=False)

    make_figure(pts, table, calfit, cal, std)
    print(table2_with_sd(pts, calfit, cal, std).round(5).to_string())
    print(
        table[
            [
                "wine",
                "arm",
                "signal",
                "sa_r2",
                "sa_native_uM",
                "ext_native_uM",
                "ext_recovery_8uM_pct",
                "ext_recovery_32uM_pct",
                "sa_recovery_8uM_pct",
                "sa_recovery_32uM_pct",
                "ext_diff_vs_uvvis_pct",
                "sa_diff_vs_uvvis_pct",
            ]
        ]
        .round(3)
        .to_string()
    )


ARM_COLOR = {
    "UV-Vis": fs.ROUTE_COLORS["uv"][0],
    "RAW": fs.ROUTE_COLORS["raw"][1],
    "JPEG": fs.ROUTE_COLORS["jpeg"][2],
}
# Panel G: the route owns the hue, the wine is the shade (Wine 1 dark, Wine 2 light).
WINE_SHADE = {"Wine1": 0, "Wine2": 2}
WINE_STYLE = {"Wine1": dict(marker="o", label="Wine 1"), "Wine2": dict(marker="s", label="Wine 2")}
ROUTE_KEY = {"UV-Vis": "uv", "RAW": "raw", "JPEG": "jpeg"}
CHANNEL_MATCH = ROOT / "Data/paper_figures/wine_ferroin/ferroin_channel_match.csv"
TAIL_FRACTION = 0.01  # transparent region: eps below 1 % of the visible-band peak


def ferroin_spectra():
    """Blank-corrected mean spectrum per standard, from the current recalculation."""
    s = pd.read_csv(SRC / "ferroin_uvvis_spectra.csv")
    blank = s[s.concentration_uM == 0].set_index(["file", "wavelength_nm"]).absorbance
    s = s.join(blank.rename("blank"), on=["file", "wavelength_nm"])
    s["A"] = s.absorbance - s.blank
    s = s[(s.concentration_uM > 0) & s.wavelength_nm.between(350, 805)]
    s = s.assign(concentration_uM=true_uM(s.concentration_uM))
    return s.groupby(["concentration_uM", "wavelength_nm"]).A.agg(["mean", "std"]).reset_index()


def channel_match_panel(ax, cm, ch, boundary, legend):
    w = cm.wavelength_nm.values
    eff = (cm.phi_led * cm[f"sens_{ch}"]).values
    eff = eff / eff.max()
    tail = w > boundary
    # Same convention as wine_ferroin_uvvis.py: the boundary trapezoid is shared evenly.
    f_tail = split_at_cutoff(eff, tail, w)[0] / np.trapezoid(eff, w)
    color = fs.CHANNEL_COLORS[ch]
    ax.fill_between(w, eff, where=w <= boundary, color=color, alpha=0.3, lw=0, label="absorbed")
    ax.fill_between(w, eff, where=w >= boundary, color=fs.GRAY, alpha=0.45, lw=0, label="tail")
    ax.plot(w, eff, color=color, lw=0.8)
    ax.plot(w, cm.eps_rel_peak, color=fs.FIT_COLOR, lw=0.8, label=r"$\varepsilon(\lambda)$")
    ax.axvline(boundary, color=fs.FIT_COLOR, lw=0.8, ls="--")
    ax.set_xlim(400, 700)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("Wavelength (nm)")
    ax.text(
        0.66,
        0.98,
        f"{ch} channel\n" + r"$f_\mathrm{tail}$" + f" = {100 * f_tail:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fs.FONT_SMALL,
        color=color,
    )
    ax.text(
        boundary + 4,
        1.2,
        f"{boundary:.0f} nm",
        va="top",
        fontsize=fs.FONT_SMALL,
        color=fs.FIT_COLOR,
    )
    if legend:
        # Compact legend on the right, over the transparent region beyond the dashed line.
        ax.legend(
            loc="upper right", bbox_to_anchor=(1.03, 0.86), handlelength=1.0, borderaxespad=0.2
        )
    return f_tail


def make_figure(pts, table, calfit, cal, std):
    fs.apply()
    fig = plt.figure(figsize=(fs.DOUBLE_COL, 8.0))
    gs = fig.add_gridspec(3, 6, height_ratios=[0.95, 1.0, 1.0])
    letters = iter(f"({c})" for c in "abcdefgh")
    plotted = []

    # Row 1: ferroin spectra and the overlap of its band with the G and B channels.
    ax = fig.add_subplot(gs[0, 0:2])
    spec = ferroin_spectra()
    concs = sorted(spec.concentration_uM.unique())
    norm = matplotlib.colors.LogNorm(min(concs), max(concs))
    # Higher concentration darker: the paper's concentration greys.
    cmap = matplotlib.colors.ListedColormap(
        plt.get_cmap(fs.CONC_CMAP)(np.linspace(*fs.CONC_RANGE, 256))
    )
    for c in concs:
        g = spec[spec.concentration_uM == c]
        ax.plot(g.wavelength_nm, g["mean"], color=cmap(norm(c)), lw=0.8, alpha=0.9)
        ax.fill_between(
            g.wavelength_nm,
            g["mean"] - g["std"],
            g["mean"] + g["std"],
            color=cmap(norm(c)),
            alpha=0.18,
            lw=0,
        )
    ax.axvline(510, color=fs.FIT_COLOR, lw=0.8, ls="--")
    ax.set_xlim(350, 700)
    ax.set_ylim(bottom=-0.02)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (AU)")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Colorbar inside the empty region above 600 nm, so it does not take a layout slot.
    cax = ax.inset_axes([0.86, 0.3, 0.04, 0.62])
    cb_ticks = [true_uM(v) for v in (4, 8, 16, 32, 56)]
    cb = fig.colorbar(sm, cax=cax, ticks=cb_ticks)
    cb.dividers.set_visible(False)
    cb.outline.set_linewidth(0.6)
    cb.ax.set_yticklabels([f"{v:.0f}" for v in cb_ticks])
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.set_label(r"Fe(II) ($\mu$M)", fontsize=fs.FONT_SMALL)
    cb.ax.tick_params(labelsize=fs.FONT_SMALL)
    cb.ax.minorticks_off()
    fs.panel_label(ax, next(letters), x=-0.3)
    plotted.append(spec.assign(panel="A"))

    cm = pd.read_csv(CHANNEL_MATCH)
    peak = cm.eps_rel_peak.idxmax()
    after = cm.loc[peak:]
    boundary = float(after[after.eps_rel_peak < TAIL_FRACTION].wavelength_nm.iloc[0])
    for k, ch in enumerate(["G", "B"]):
        ax = fig.add_subplot(gs[0, 2 + 2 * k : 4 + 2 * k])
        f_tail = channel_match_panel(ax, cm, ch, boundary, legend=True)
        if k == 0:
            ax.set_ylabel("Normalised response")
        else:
            ax.set_yticklabels([])
        fs.panel_label(ax, next(letters), x=-0.15)
        plotted.append(
            pd.DataFrame(dict(panel="BC"[k], channel=ch, boundary_nm=[boundary], f_tail=[f_tail]))
        )

    # Row 2: (D) calibration standards of the three methods; (E, F) each wine by the three
    # methods. UV-Vis on the left axis, camera G/R on the right (Stephania, 2026-09-30).
    methods = [("UV-Vis", "A510", "o"), ("RAW", MAIN, "s"), ("JPEG", MAIN, "^")]

    def twin(ax):
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_linewidth(0.6)
        return ax2

    def draw(ax, x, y, fitted, sat, color, marker):
        """Replicates and mean +/- SD; points left out of the fit drawn faded."""
        d = pd.DataFrame(dict(x=x, y=y, fitted=fitted, sat=sat))
        for keep in (True, False):
            sel = d[d.fitted == keep]
            if sel.empty:
                continue
            fade = 1.0 if keep else 0.35
            ok, bad = sel[~sel.sat], sel[sel.sat]
            ax.scatter(ok.x, ok.y, s=12, alpha=0.6 * fade, color=color, marker=marker, linewidths=0)
            if len(bad):
                ax.scatter(
                    bad.x,
                    bad.y,
                    s=16,
                    facecolors="none",
                    marker=marker,
                    edgecolors=fs.CHANNEL_COLORS["R"],
                    linewidths=0.7,
                    alpha=fade,
                )
            m = sel.groupby("x").y.agg(["mean", "std"])
            ax.errorbar(
                m.index,
                m["mean"],
                yerr=m["std"],
                fmt=marker,
                markersize=4.5,
                color=color,
                capsize=2.5,
                elinewidth=0.8,
                alpha=fade,
            )

    def legend(ax, shown, **where):
        # Drawn on the twin (top) axes so the camera points never cover it.
        ax = ax.right_twin
        h = [
            Line2D(
                [],
                [],
                ls="none",
                marker=mk,
                color=ARM_COLOR[a],
                label=a if a == "UV-Vis" else f"{a}, G/R",
            )
            for a, _, mk in methods
            if a in shown
        ]
        where = where or {"loc": "upper left", "bbox_to_anchor": (0.0, 0.84)}
        ax.legend(handles=h, **where)

    # D-F share both y ranges; the left axis is labelled on D only, the right on F only.
    left_top, right_top = 0.24, 0.10

    def axes_pair(slot, x_label, show_left, show_right):
        ax = fig.add_subplot(slot)
        ax2 = twin(ax)
        ax.right_twin = ax2
        # Same zero on both axes: the right range is the left range scaled.
        ax.set_ylim(-0.06 * left_top, left_top)
        ax2.set_ylim(-0.06 * right_top, right_top)
        ax.set_xlabel(x_label)
        if show_left:
            ax.set_ylabel(r"UV-Vis $A_{510}$ (AU)")
        else:
            ax.tick_params(axis="y", labelleft=False)
        if show_right:
            ax2.set_ylabel(r"Camera $A_G - A_R$")
        else:
            ax2.tick_params(axis="y", labelright=False)
        return ax, ax2

    # (D) Calibration standards.
    # D-F sit in their own sub-grid with no gap, since only D and F carry axis labels.
    row2 = gs[1, :].subgridspec(1, 3, wspace=0.0)
    ax, ax2 = axes_pair(row2[0], r"Fe(II) standard ($\mu$M)", True, False)
    xs = np.array([0, CAL_TOP])
    draw(
        ax,
        cal.concentration_uM,
        cal.A510_corrected,
        np.ones(len(cal), bool),
        np.zeros(len(cal), bool),
        ARM_COLOR["UV-Vis"],
        "o",
    )
    ax.plot(
        xs, calfit["slope"] * xs + calfit["intercept"], ls="--", lw=0.9, color=ARM_COLOR["UV-Vis"]
    )
    for arm, upper, mk in (("RAW", CAL_TOP, "s"), ("JPEG", JPEG_CAL_UPPER, "^")):
        s = std[std.arm == arm]
        inside = (s.concentration_uM <= upper + 1e-9).values
        draw(ax2, s.concentration_uM, s[MAIN], inside, np.zeros(len(s), bool), ARM_COLOR[arm], mk)
        f = fit(s.concentration_uM[inside], s[MAIN][inside])
        xf = np.array([0, upper])
        ax2.plot(xf, f["slope"] * xf + f["intercept"], ls="--", lw=0.9, color=ARM_COLOR[arm])
        plotted.append(
            s[["concentration_uM", MAIN]]
            .rename(columns={MAIN: "A"})
            .assign(panel="D", arm=arm, fitted=inside)
        )
    plotted.append(
        cal[["concentration_uM", "A510_corrected"]]
        .rename(columns={"A510_corrected": "A"})
        .assign(panel="D", arm="UV-Vis")
    )
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.text(0.04, 0.97, "Calibration", transform=ax.transAxes, va="top", fontsize=fs.FONT_SMALL)
    # Lower right: clear of the 52 uM points at the larger font (2026-10-05).
    legend(ax, ("UV-Vis", "RAW", "JPEG"), loc="lower right", bbox_to_anchor=(1.0, 0.1))
    fs.panel_label(ax, next(letters), x=-0.3)

    # (E, F) Each wine, unspiked and spiked, by UV-Vis and RAW. JPEG is left out of these
    # panels (Stephania, 2026-10-01); its wine results are in G, H and Table 2.
    for k, (wine, name) in enumerate([("Wine1", "Wine 1"), ("Wine2", "Wine 2")]):
        ax, ax2 = axes_pair(row2[1 + k], r"Added Fe ($\mu$M)", False, k == 1)
        for arm, sig, mk in methods[:2]:
            target = ax if arm == "UV-Vis" else ax2
            p = pts[(pts.arm == arm) & (pts.signal == sig) & (pts.wine == wine)]
            row = table[(table.arm == arm) & (table.signal == sig) & (table.wine == wine)].iloc[0]
            top = SPIKES[32]
            draw(
                target,
                p.added_uM,
                p.A,
                (p.added_uM <= top).values,
                p.saturated.values,
                ARM_COLOR[arm],
                mk,
            )
            xf = np.array([-row.sa_native_uM, top])
            target.plot(
                xf,
                row.sa_slope_per_uM * xf + row.sa_intercept,
                ls="--",
                lw=0.9,
                color=ARM_COLOR[arm],
            )
            plotted.append(p.assign(panel="EF"[k]))
        ax.set_xlim(-13, 57)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.text(0.04, 0.97, name, transform=ax.transAxes, va="top", fontsize=fs.FONT_SMALL)
        legend(ax, ("UV-Vis", "RAW"))
        fs.panel_label(ax, next(letters), x=-0.1)

    # Row 3, left (G): iron in the wine by method and route.
    ax = fig.add_subplot(gs[2, 0:3])
    methods = [
        ("UV-Vis", "A510", "ext", "UV-Vis\ncalib."),
        ("UV-Vis", "A510", "sa", "UV-Vis\nst. add."),
        ("RAW", MAIN, "ext", "RAW\ncalib."),
        ("RAW", MAIN, "sa", "RAW\nst. add."),
        ("JPEG", MAIN, "ext", "JPEG\ncalib."),
        ("JPEG", MAIN, "sa", "JPEG\nst. add."),
    ]
    width = 0.38
    fe_rows = []
    for k, (wine, st) in enumerate(WINE_STYLE.items()):
        for j, (arm, sig, route, _) in enumerate(methods):
            row = table[(table.arm == arm) & (table.signal == sig) & (table.wine == wine)].iloc[0]
            v = row[f"{route}_wine_Fe_mg_L"]
            shade = fs.ROUTE_COLORS[ROUTE_KEY[arm]][WINE_SHADE[wine]]
            ax.bar(j + (k - 0.5) * width, v, width, color=shade, linewidth=0)
            fe_rows.append(dict(panel="G", arm=arm, route=route, wine=wine, Fe_mg_L=v))
        ref = table[(table.arm == "UV-Vis") & (table.wine == wine)].ext_wine_Fe_mg_L.iloc[0]
        ax.axhline(ref, color=fs.ROUTE_COLORS["uv"][WINE_SHADE[wine]], lw=0.8, ls="--")
    ax.set_xticks(range(len(methods)), [m[3] for m in methods])
    ax.tick_params(axis="x", labelsize=fs.FONT_SMALL)
    ax.set_ylabel("Fe in wine (mg/L)")
    # Each wine's key shows its shade in all three route colours, as the bars use them.
    ax.legend(
        handles=[
            tuple(
                Patch(color=fs.ROUTE_COLORS[r][WINE_SHADE[w]], lw=0) for r in ("uv", "raw", "jpeg")
            )
            for w in WINE_STYLE
        ],
        labels=[st["label"] for st in WINE_STYLE.values()],
        loc="upper left",
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        handlelength=3.0,
    )
    fs.panel_label(ax, next(letters), x=-0.15)
    plotted.append(pd.DataFrame(fe_rows))

    # Row 3, right (H): spike recovery against each method's external calibration.
    ax = fig.add_subplot(gs[2, 3:6])
    groups = [("Wine1", 8), ("Wine1", 32), ("Wine2", 8), ("Wine2", 32)]
    width = 0.26
    rec_rows = []
    for j, (arm, sig) in enumerate([("UV-Vis", "A510"), ("RAW", MAIN), ("JPEG", MAIN)]):
        for i, (wine, lvl) in enumerate(groups):
            row = table[(table.arm == arm) & (table.signal == sig) & (table.wine == wine)].iloc[0]
            v = row[f"ext_recovery_{lvl}uM_pct"]
            extrap = arm == "JPEG" and SPIKES[lvl] > JPEG_CAL_UPPER
            ax.bar(
                i + (j - 1) * width,
                v,
                width,
                color=ARM_COLOR[arm],
                hatch="////" if extrap else None,
                edgecolor="white" if extrap else None,
                linewidth=0,
                label=arm if i == 0 else None,
            )
            rec_rows.append(
                dict(panel="H", arm=arm, wine=wine, added_uM=SPIKES[lvl], recovery_pct=v)
            )
    ax.axhline(100, color=fs.FIT_COLOR, lw=0.8, ls="--")
    lo, hi = (f"+{SPIKES[n]:.0f}" for n in (8, 32))
    ax.set_xticks(range(4), [f"Wine 1\n{lo}", f"Wine 1\n{hi}", f"Wine 2\n{lo}", f"Wine 2\n{hi}"])
    ax.set_ylabel("Spike recovery vs calibration (%)")
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, 430)
    fs.panel_label(ax, next(letters), x=-0.15)
    plotted.append(pd.DataFrame(rec_rows))

    fs.save(fig, FIG / "figure_wine_case_study", data=pd.concat(plotted, ignore_index=True))


if __name__ == "__main__":
    main()
