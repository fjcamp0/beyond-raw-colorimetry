"""Calibration metrics for Supplementary Table S1 and Table 1, native camera-app JPEG.

Same conventions as the RAW rows of Table S1 (scripts/recompute_lod_loq_pooled.py):
OLS over the individual replicate points of the stated linear range; LOD = 3.3 sigma /
slope and LOQ = 10 sigma / slope, sigma = pooled replicate SD of the three lowest
levels in the fit range. Linear ranges are those of the rows they replace (Stephania,
2026-10-01). Run after scripts/native_jpeg_arm.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BL = lambda analyte: ROOT / "Data/analysis" / analyte / "beer_lambert"
OUT = ROOT / "Data/analysis/native_jpeg_metrics"

ROWS = [  # analyte, channel, range, arm folder ("" = RAW)
    ("ferricyanide", "B", (7.8, 500), "native_jpeg"),
    ("ferricyanide", "BdivG", (7.8, 1000), "native_jpeg"),
    ("methylene_blue", "R", (0.78, 25), "native_jpeg"),
    ("methylene_blue", "RdivB", (0.78, 25), "native_jpeg"),
    ("methylene_blue", "RdivG", (0.78, 25), "native_jpeg"),
    # RAW rows recomputed only as a check of the method against Table S1 v2.
    ("ferricyanide", "BdivG", (7.8, 1000), ""),
    ("methylene_blue", "R", (0.78, 50), ""),
]
TABLE1 = {
    "ferricyanide": ("B", [15.625, 250.0, 1000.0, 2000.0, 4000.0]),
    "methylene_blue": ("R", [0.78, 6.25, 25.0, 50.0, 100.0]),
}


def points(analyte, arm, channel):
    t = pd.read_csv(BL(analyte) / arm / "absorbance_table.csv")
    t = t[t.channel == channel]
    return t.concentration_um.round(4).values, t.absorbance.values


def metrics(analyte, channel, rng, arm):
    x, y = points(analyte, arm, channel)
    m = (x >= rng[0] - 1e-6) & (x <= rng[1] + 1e-6)
    x, y = x[m], y[m]
    slope, icpt = np.polyfit(x, y, 1)
    res = y - (slope * x + icpt)
    r2 = 1 - np.sum(res**2) / np.sum((y - y.mean()) ** 2)
    low = sorted(set(x))[:3]
    sigma = np.sqrt(np.mean([np.var(y[x == c], ddof=1) for c in low]))
    return dict(
        analyte=analyte,
        arm=arm or "RAW",
        channel=channel,
        range=f"{rng[0]:g}-{rng[1]:g}",
        n=len(x),
        slope=slope,
        intercept=icpt,
        r2=r2,
        sigma=sigma,
        LOD_uM=3.3 * sigma / slope,
        LOQ_uM=10 * sigma / slope,
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame([metrics(*r) for r in ROWS])
    t.to_csv(OUT / "table_S1_native_jpeg_rows.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(6).to_string())
    rows = []
    for analyte, (ch, concs) in TABLE1.items():
        raw = pd.read_csv(BL(analyte) / "calibration_table.csv")
        jpg = pd.read_csv(BL(analyte) / "native_jpeg" / "calibration_table.csv")
        for c in concs:
            a_raw = raw[
                (raw.channel == ch) & np.isclose(raw.concentration_um, c, rtol=1e-3)
            ].absorbance_mean
            a_jpg = jpg[
                (jpg.channel == ch) & np.isclose(jpg.concentration_um, c, rtol=1e-3)
            ].absorbance_mean
            a_raw, a_jpg = float(a_raw.iloc[0]), float(a_jpg.iloc[0])
            rows.append(
                dict(
                    analyte=analyte,
                    channel=ch,
                    c_uM=c,
                    A_raw=a_raw,
                    A_jpeg=a_jpg,
                    ratio=a_jpg / a_raw,
                )
            )
    t1 = pd.DataFrame(rows)
    t1.to_csv(OUT / "table_1_native_jpeg.csv", index=False)
    print(t1.round(4).to_string())
