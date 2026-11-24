"""Native-camera JPEG arm for ferricyanide and methylene blue.

Replaces the earlier rawpy-rendered sRGB arm everywhere in the paper (Stephania,
2026-10-01): the sRGB values now come from the phone's own camera app, not from a
conversion of the DNG files. Decisions recorded with the request:
- absorbance is computed on the JPEG pixel values as saved (8-bit, no inverse-sRGB
  linearisation), i.e. what a non-specialist obtains;
- no mechanistic modelling of the app's processing (it is unknown).

Inputs (unfiltered native app, ISO 100, 1/30 s):
- Data/images/ferricyanide/Ferricyanide_nofilter_nativeapp: 29 Sep ladder, with the
  2000 and 4000 uM levels replaced by the 30 Sep repeats and their own water blanks
  (Rajendra's selection, scripts/select_ferricyanide_native_repeats.py).
- Data/images/methylene_blue/Methylene Blue_nofilter_ISO100_exposure1by30_NativeAapp.

Method: centred ROI of 25 % of each image dimension; per channel, the mean pixel value
of each sample image against the pooled mean of that session's water images,
A = -log10(sample / water). Ratio absorbances are channel differences (B/G = A_B - A_G).

Outputs, in the schema of the old sRGB tables so the figure and table scripts only
change their input path:
    Data/analysis/<analyte>/beer_lambert/native_jpeg/absorbance_table.csv
    Data/analysis/<analyte>/beer_lambert/native_jpeg/calibration_table.csv
    Data/analysis/<analyte>/beer_lambert/native_jpeg/frame_checks.csv
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ROI_FRACTION = 0.25
CHANNELS = ("R", "G", "B")
RATIOS = {"BdivG": ("B", "G"), "RdivB": ("R", "B"), "RdivG": ("R", "G")}

ANALYTES = {
    "ferricyanide": dict(
        folder=ROOT / "Data/images/ferricyanide/Ferricyanide_nofilter_nativeapp",
        ladder=4000.0 / 2.0 ** np.arange(0, 11),  # 4000 ... 3.90625 uM, exact 1:2 dilutions
        # 2000 and 4000 uM come from the 30 Sep repeat; everything else from 29 Sep.
        session_for=lambda c: "2026-09-30" if c >= 2000 else "2026-09-29",
    ),
    "methylene_blue": dict(
        folder=ROOT
        / "Data/images/methylene_blue/Methylene Blue_nofilter_ISO100_exposure1by30_NativeAapp",
        ladder=100.0 / 2.0 ** np.arange(0, 8),  # 100 ... 0.78125 uM
        session_for=lambda c: "2026-09-29",
    ),
}
# Concentrations are written with the labels of the RAW calibration tables, so rows match.
RAW_LABELS = {
    "ferricyanide": [
        3.90625,
        7.8,
        15.625,
        31.25,
        62.5,
        125.0,
        250.0,
        500.0,
        1000.0,
        2000.0,
        4000.0,
    ],
    "methylene_blue": [0.78, 1.56, 3.13, 6.25, 12.5, 25.0, 50.0, 100.0],
}


def capture_date(path, exif):
    if exif.get(36867):
        return datetime.strptime(str(exif[36867]), "%Y:%m:%d %H:%M:%S")
    m = re.search(r"IMG_(\d{8}_\d{6})", path.name)
    return datetime.strptime(m[1], "%Y%m%d_%H%M%S")


def read_frames(analyte, cfg):
    rows = []
    for p in sorted(cfg["folder"].glob("*.jpg")):
        with Image.open(p) as im:
            exif = im.getexif().get_ifd(34665)
            a = np.asarray(im.convert("RGB"))
        t = capture_date(p, exif)
        m = re.match(r"([\d.]+)\s*uM", p.name, re.I)
        if m:
            label = float(m[1])
            exact = cfg["ladder"][np.argmin(np.abs(cfg["ladder"] - label))]
            assert abs(label / exact - 1) < 0.15, p.name
            kind = "sample"
        elif p.name.lower().startswith("water"):
            exact, kind = np.nan, "water"
        else:
            raise ValueError(f"unrecognised file name: {p.name}")
        h, w = a.shape[:2]
        dy, dx = int(h * ROI_FRACTION / 2), int(w * ROI_FRACTION / 2)
        roi = a[h // 2 - dy : h // 2 + dy, w // 2 - dx : w // 2 + dx]
        for i, ch in enumerate(CHANNELS):
            x = roi[..., i].astype(float)
            rows.append(
                dict(
                    analyte=analyte,
                    filename=p.name,
                    kind=kind,
                    concentration_exact=exact,
                    session=t.date().isoformat(),
                    time=t.isoformat(),
                    channel=ch,
                    mean_DN=x.mean(),
                    pct_ge254=100 * np.mean(x >= 254),
                    pct_le1=100 * np.mean(x <= 1),
                    iso=str(exif.get(34855, "")),
                    exposure_s=float(exif.get(33434, np.nan)),
                )
            )
    return pd.DataFrame(rows)


def build(analyte, cfg):
    f = read_frames(analyte, cfg)
    water = f[f.kind == "water"].groupby(["session", "channel"]).mean_DN.mean().rename("water_DN")
    f = f.join(water, on=["session", "channel"])
    f["A"] = -np.log10(f.mean_DN / f.water_DN)
    samples = f[f.kind == "sample"].copy()
    keep = samples.apply(lambda r: r.session == cfg["session_for"](r.concentration_exact), axis=1)
    samples = samples[keep]
    labels = dict(zip(sorted(cfg["ladder"]), sorted(RAW_LABELS[analyte])))
    samples["concentration_um"] = samples.concentration_exact.map(labels)
    wide = samples.pivot_table(
        index=["filename", "session", "concentration_um"], columns="channel", values="A"
    ).reset_index()
    for name, (num, den) in RATIOS.items():
        wide[name] = wide[num] - wide[den]
    long = wide.melt(
        id_vars=["filename", "session", "concentration_um"],
        var_name="channel",
        value_name="absorbance",
    )
    long = long.rename(columns={"session": "set"})
    long["transmittance"] = 10 ** (-long.absorbance)
    cal = (
        long.groupby(["concentration_um", "channel"])
        .absorbance.agg(n_replicates="size", absorbance_mean="mean", absorbance_std="std")
        .reset_index()
    )
    cal["absorbance_sem"] = cal.absorbance_std / np.sqrt(cal.n_replicates)

    out = ROOT / "Data/analysis" / analyte / "beer_lambert" / "native_jpeg"
    out.mkdir(parents=True, exist_ok=True)
    long.sort_values(["concentration_um", "channel", "filename"]).to_csv(
        out / "absorbance_table.csv", index=False
    )
    cal.to_csv(out / "calibration_table.csv", index=False)
    f.to_csv(out / "frame_checks.csv", index=False)
    print(
        f"{analyte}: {samples.filename.nunique()} sample images, "
        f"{f[f.kind == 'water'].filename.nunique()} water images, "
        f"sessions {sorted(samples.session.unique())}; "
        f"max ROI clipping {samples.pct_ge254.max():.3f} %"
    )
    return cal


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for name, cfg in ANALYTES.items():
        cal = build(name, cfg)
        main = "B" if name == "ferricyanide" else "R"
        print(
            cal[cal.channel.isin([main, "BdivG" if main == "B" else "RdivG"])]
            .pivot(index="concentration_um", columns="channel", values="absorbance_mean")
            .round(4)
        )
