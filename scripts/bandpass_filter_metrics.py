"""Bandpass-filter rows for Supplementary Table S1 (item 21, 2026-10-05).

RAW + 450/40 (ferricyanide, B, 29 Sep) and RAW + 650/10 (methylene blue, R, 28 Sep),
with roi_selector ROIs and water blanks interpolated in capture time. UV-Vis rows are
the measured replicate spectra weighted by the LED (or the LED through the filter,
measured just after it) and the 20-device mean Bayer channel, as in table1_uvvis_band.

Table S1 convention (Methods 3.6): OLS on individual points over the range of the
matching unfiltered RAW row; LOD = 3.3 sigma / slope, LOQ = 10 sigma / slope, sigma =
RMS replicate SD at the three lowest levels; R2 on level means, also over the full ladder.
The unfiltered RAW rows are recomputed first and must reproduce Table S1.

The ROI selector and RAW reader come from vendor/ (override with COLORIMETRY_VENDOR).
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import table1_uvvis_band as band
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Vendored raw-image-analysis modules (vendor/, see README); override with COLORIMETRY_VENDOR.
ENGINE = Path(
    os.environ.get("COLORIMETRY_VENDOR", str(Path(__file__).resolve().parent.parent / "vendor"))
)
sys.path.insert(0, str(ENGINE))
from core.images import read_raw_means  # noqa: E402

IMG = ROOT / "Data/images"
OUT = ROOT / "Data/analysis/bandpass_filter_metrics"
# ROI chosen by roi_selector on the 29 Sep 450/40 set, shipped with the scripts.
FERRI_ROI = Path(__file__).resolve().parent / "rois_ferricyanide_450_40.yaml"
DROP = {"7.815uM_2_Fe"}

ARMS = {
    "ferricyanide": dict(
        channel="B",
        rng=(7.8, 500),
        filt="Experiment_14_Filter_450nm_40nm",
        label="450/40",
        dir="ferricyanide/Ferricyanide_withfilter450nm_40nm_Motioncam",
        ladder=[4000 / 2**k for k in range(11)],
        s1=dict(slope=4.762e-5, r2=0.8878, LOD_uM=462),
    ),
    "methylene_blue": dict(
        channel="R",
        rng=(0.78, 50),
        filt="Experiment_9_Filter_650nm_10nm",
        label="650/10",
        dir="methylene_blue/Methyleneblue_withfilter650nm_10nm_Motioncam",
        ladder=[100 / 2**k for k in range(8)],
        s1=dict(slope=8.248e-3, r2=0.9992, LOD_uM=3.09),
    ),
}


def r2_means(x, y):
    lv = np.array(sorted(set(x)))
    mean = np.array([y[x == c].mean() for c in lv])
    res = mean - np.polyval(np.polyfit(lv, mean, 1), lv)
    return 1 - np.sum(res**2) / np.sum((mean - mean.mean()) ** 2)


def fit(x, y, rng):
    x, y = np.round(np.asarray(x, float), 4), np.asarray(y, float)
    r2_full = r2_means(x, y)
    m = (x >= rng[0] - 1e-6) & (x <= rng[1] + 1e-6)
    x, y = x[m], y[m]
    slope, icpt = np.polyfit(x, y, 1)
    low = sorted(set(x))[:3]
    sigma = np.sqrt(np.mean([np.var(y[x == c], ddof=1) for c in low]))
    return dict(
        range=f"{rng[0]:g}-{rng[1]:g}",
        n=len(x),
        slope=slope,
        intercept=icpt,
        r2=r2_means(x, y),
        r2_full=r2_full,
        sigma=sigma,
        LOD_uM=3.3 * sigma / slope,
        LOQ_uM=10 * sigma / slope,
    )


def unfiltered_raw(analyte, a):
    t = pd.read_csv(ROOT / "Data/analysis" / analyte / "beer_lambert/absorbance_table.csv")
    t = t[t.channel == a["channel"]]
    return fit(t.concentration_um, t.absorbance, a["rng"])


def capture_time(name):
    d, t = re.search(r"IMG_(\d{6})_(\d{6})", name).groups()
    return datetime.strptime(d + t, "%y%m%d%H%M%S").timestamp()


def level(name, ladder):
    if re.search(r"water|blank|control", name, re.I):
        return None
    c = float(re.match(r"([0-9.]+)u?M", name).group(1))
    return min(ladder, key=lambda v: abs(v - c))


def roi_for(analyte, a):
    if analyte == "ferricyanide":
        r = yaml.safe_load(FERRI_ROI.read_text(encoding="utf-8"))["rois"][0]
    else:
        out = OUT / "roi_selection" / analyte
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ENGINE / "roi_selector.py"),
            "--img-dir",
            str(IMG / a["dir"]),
            "--blank",
            "Water_blank*.dng",
            "--out-dir",
            str(out),
            "--channel",
            a["channel"],
            "--block",
            "50",
            "--win-h",
            "4",
            "--win-w",
            "8",
            "--search-aspect",
        ]
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        log = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", env=env, check=True
        ).stdout
        (out / "roi_selector.log").write_text(log, encoding="utf-8")
        r = yaml.safe_load((out / "rois.yaml").read_text(encoding="utf-8"))["rois"][0]
    return r["x"], r["y"], r["w"], r["h"]


def filtered_raw(analyte, a, roi):
    blanks, samples = [], []
    for f in sorted((IMG / a["dir"]).glob("*.dng")):
        if any(k in f.name for k in DROP):
            continue
        v = read_raw_means(f, roi)[a["channel"]]
        c = level(f.name, a["ladder"])
        (blanks if c is None else samples).append((capture_time(f.name), c, v, f.name))
    blanks.sort()
    bt, bv = np.array([b[0] for b in blanks]), np.array([b[2] for b in blanks])
    rows = [dict(file=n, c_uM=c, A=np.log10(np.interp(t, bt, bv) / v)) for t, c, v, n in samples]
    return pd.DataFrame(rows)


def uvvis_band(analyte, a, phi):
    s = pd.read_csv(band.REC / "uvvis_spectra_all.csv")
    s = s[s.analyte == analyte]
    w = band.weights(band.ANALYTES[analyte][1], phi)
    xs, ys = [], []
    for (src, c), g in s.groupby(["source", "concentration_uM"]):
        g = g.sort_values("wavelength_nm")
        spec = band.corrected(analyte, g.wavelength_nm, g.absorbance)
        xs.append(c), ys.append(-np.log10(np.sum(w * 10**-spec)))
    return fit(xs, ys, a["rng"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for analyte, a in ARMS.items():
        ref = unfiltered_raw(analyte, a)
        ok = all(abs(ref[k] / v - 1) < 0.01 for k, v in a["s1"].items())
        print(f"{analyte}: Table S1 RAW row reproduced: {'ok' if ok else 'MISMATCH'}")
        if not ok:
            sys.exit(f"{ref} vs {a['s1']}")
        roi = roi_for(analyte, a)
        pts = filtered_raw(analyte, a, roi)
        pts.to_csv(OUT / f"{analyte}_filtered_raw_points.csv", index=False)
        phi = band.filtered_led(a["filt"])
        arms = [
            ("RAW", "none", ref),
            ("RAW", a["label"], fit(pts.c_uM, pts.A, a["rng"])),
            ("UV-Vis band", "none", uvvis_band(analyte, a, None)),
            ("UV-Vis band", a["label"], uvvis_band(analyte, a, phi)),
        ]
        for arm, filt, r in arms:
            rows.append(
                dict(
                    analyte=analyte,
                    channel=a["channel"],
                    arm=arm,
                    filter=filt,
                    roi=str(roi) if (arm, filt) == ("RAW", a["label"]) else "",
                    **r,
                )
            )
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "bandpass_filter_metrics.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.drop(columns="roi").to_string(float_format=lambda v: f"{v:.5g}"))
