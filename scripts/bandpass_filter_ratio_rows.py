"""Ratio-channel rows with the bandpass filter, for Supplementary Table S2 (2026-10-07).

Stephania asked for B/G (ferricyanide, 450/40) and R/B, R/G (methylene blue, 650/10)
through the filter, for completeness. Same images, ROIs, time-interpolated water blanks
and fit convention as bandpass_filter_metrics.py; the ratio absorbance is A_X - A_Y,
with G the mean of G1 and G2. The unfiltered ratio rows are refitted first and must
reproduce Table S2.
"""

import ast
import sys

import bandpass_filter_metrics as bfm
import numpy as np
import pandas as pd

RATIOS = {
    "ferricyanide": dict(pairs=[("B", "G", (7.8, 1000), dict(slope=2.082e-5, LOD_uM=136))]),
    "methylene_blue": dict(
        pairs=[
            ("R", "B", (0.78, 50), dict(slope=7.007e-3, LOD_uM=2.20)),
            ("R", "G", (0.78, 50), dict(slope=5.343e-3, LOD_uM=2.12)),
        ]
    ),
}


def channel_means(f, roi):
    m = bfm.read_raw_means(f, roi)
    return dict(R=m["R"], G=(m["G1"] + m["G2"]) / 2, B=m["B"])


def unfiltered_points(table):
    """Per-image A_R, A_G, A_B from the stored channel intensities and blanks."""
    t = table[table.channel.isin(["R", "G1", "G2", "B"])]
    raw = t.pivot_table(
        index=["filename", "concentration_um"], columns="channel", values="raw_mean"
    )
    blank = t.pivot_table(
        index=["filename", "concentration_um"], columns="channel", values="blank_mean"
    )
    out = pd.DataFrame(index=raw.index)
    for ch, cols in (("R", ["R"]), ("G", ["G1", "G2"]), ("B", ["B"])):
        out[f"A_{ch}"] = np.log10(blank[cols].mean(axis=1) / raw[cols].mean(axis=1))
    return out.reset_index()


def filtered_points(analyte, a, roi):
    blanks, samples = [], []
    for f in sorted((bfm.IMG / a["dir"]).glob("*.dng")):
        if any(k in f.name for k in bfm.DROP):
            continue
        c = bfm.level(f.name, a["ladder"])
        rec = (bfm.capture_time(f.name), c, channel_means(f, roi), f.name)
        (blanks if c is None else samples).append(rec)
    blanks.sort(key=lambda b: b[0])
    bt = np.array([b[0] for b in blanks])
    rows = []
    for t, c, v, name in samples:
        row = dict(file=name, c_uM=c)
        for ch in "RGB":
            blank = np.interp(t, bt, [b[2][ch] for b in blanks])
            row[f"A_{ch}"] = np.log10(blank / v[ch])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    rows = []
    metrics = pd.read_csv(bfm.OUT / "bandpass_filter_metrics.csv")
    for analyte, spec in RATIOS.items():
        a = bfm.ARMS[analyte]
        t = pd.read_csv(bfm.ROOT / "Data/analysis" / analyte / "beer_lambert/absorbance_table.csv")
        u = unfiltered_points(t)
        roi_text = metrics[
            (metrics.analyte == analyte)
            & (metrics.arm == "RAW")
            & (metrics["filter"] == a["label"])
        ].roi.iloc[0]
        pts = filtered_points(analyte, a, ast.literal_eval(roi_text))
        pts.to_csv(bfm.OUT / f"{analyte}_filtered_raw_points_all_channels.csv", index=False)
        for x, y, rng, ref in spec["pairs"]:
            unf = bfm.fit(u.concentration_um, u[f"A_{x}"] - u[f"A_{y}"], rng)
            ok = all(abs(unf[k] / v - 1) < 0.01 for k, v in ref.items())
            print(f"{analyte} {x}/{y} unfiltered reproduces Table S2: {'ok' if ok else 'MISMATCH'}")
            if not ok:
                sys.exit(f"{unf} vs {ref}")
            filt = bfm.fit(pts.c_uM, pts[f"A_{x}"] - pts[f"A_{y}"], rng)
            for label, r in (("none", unf), (a["label"], filt)):
                rows.append(dict(analyte=analyte, signal=f"{x}/{y}", filter=label, **r))
    out = pd.DataFrame(rows)
    out.to_csv(bfm.OUT / "bandpass_filter_ratio_rows.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.to_string(float_format=lambda v: f"{v:.5g}"))


if __name__ == "__main__":
    main()
