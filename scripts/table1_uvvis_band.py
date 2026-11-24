"""Band-matched UV-Vis absorbance for Table 1.

A_band(c) = -log10( sum phi S_ch 10^-A(lambda, c) / sum phi S_ch ), from the measured
UV-Vis spectra (0.2 cm, the camera cuvette), the Spelec LED and the 20-device mean
Bayer sensitivity over 400-700 nm. This is what an ideal camera channel would read.
Ferricyanide spectra each get their own linear baseline, fitted over 550-800 nm where
ferricyanide reads below 0.001 AU even at 4 mM: single spectra carry flat or sloped
offsets of up to 0.007 AU that would otherwise count as noise. Methylene blue absorbs
there, so it keeps the per-wavelength intercept of the paper's epsilon fit
(1.5625-50 uM).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "Data/analysis/all_active_recalculation_20260930T161847Z"
SPEC = ROOT / "Data/spectra/dataanalysis"
OUT = ROOT / "Data/analysis/table1_uvvis_band"
GRID = np.arange(400.0, 701.0, 1.0)
OWN_BASELINE = {"ferricyanide": (550.0, 800.0)}

ANALYTES = {
    "ferricyanide": (
        "spectra_ferricyanide.csv",
        "B",
        "full_range",
        {15.625: "15.62", 250.0: "250", 1000.0: "1000", 2000.0: "2000", 4000.0: "4000"},
    ),
    "methylene_blue": (
        "spectra_mb.csv",
        "R",
        "restricted_1p5625_to_50uM",
        {0.78: "0.7812", 6.25: "6.25", 25.0: "25", 50.0: "50", 100.0: "100"},
    ),
}


def filtered_led(experiment):
    f = pd.read_csv(REC / "filter_spectra.csv")
    f = f[f.source.str.contains(experiment, regex=False)]
    return f.wavelength_nm.values, f.bias_corrected_counts.clip(lower=0).values


def weights(channel, phi=None):
    bayer = pd.read_csv(REC / "bayer_sensitivity_mean_20_devices.csv")
    if phi is None:
        led = pd.read_csv(REC / "illuminant_spectra.csv")
        led = led[led.illuminant == "illuminant_spelec_led"]
        phi = led.wavelength_nm.values, led.relative_power.clip(lower=0).values
    w = np.interp(GRID, *phi) * np.interp(GRID, bayer.wavelength_nm, bayer[channel])
    return w / w.sum()


def baseline(analyte, window):
    e = pd.read_csv(REC / "epsilon_spectra.csv")
    e = e[(e.analyte == analyte) & (e.window == window)]
    return np.interp(GRID, e.wavelength_nm, e.intercept)


def corrected(analyte, wavelength, absorbance):
    """Baseline-corrected absorbance on GRID."""
    wl, a = np.asarray(wavelength, float), np.asarray(absorbance, float)
    ok = np.isfinite(a)
    wl, a = wl[ok], a[ok]
    on_grid = np.interp(GRID, wl, a)
    if analyte in OWN_BASELINE:
        lo, hi = OWN_BASELINE[analyte]
        m = (wl >= lo) & (wl <= hi)
        return on_grid - np.polyval(np.polyfit(wl[m], a[m], 1), GRID)
    return on_grid - baseline(analyte, ANALYTES[analyte][2])


def band_absorbance(analyte, phi=None, concs=None):
    fname, ch, _, cols = ANALYTES[analyte]
    s = pd.read_csv(SPEC / fname)
    w = weights(ch, phi)
    labels = cols if concs is None else concs
    out = {}
    for c, lab in labels.items():
        a = corrected(analyte, s.wavelength_nm, s[f"mean_{lab}uM"])
        out[c] = -np.log10(np.sum(w * 10**-a))
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t1 = pd.read_csv(ROOT / "Data/analysis/native_jpeg_metrics/table_1_native_jpeg.csv")
    uv = {(a, c): v for a in ANALYTES for c, v in band_absorbance(a).items()}
    t1["A_uvvis_band"] = [uv[(a, c)] for a, c in zip(t1.analyte, t1.c_uM)]
    t1["raw_over_uvvis"] = t1.A_raw / t1.A_uvvis_band
    t1.to_csv(OUT / "table_1_with_uvvis_band.csv", index=False)
    print(t1.round(4).to_string())
