"""Lamp drift during the ferricyanide LED sets, read from the R channel.

Ferricyanide barely absorbs in the R channel, so the R-channel absorbance of
each sample image, minus what the forward model predicts for ferricyanide,
measures the change in lamp output since the blank. Input: the per-image
absorbances of channel_common_origin_check.py. Output:
Data/analysis/ferricyanide/lamp_drift_check/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJ = Path(__file__).resolve().parent.parent
RECALC = PROJ / "Data/analysis/all_active_recalculation_20260930T161847Z"
IMAGES = PROJ / ("Data/analysis/ferricyanide/channel_common_origin/channel_common_origin_data.csv")
OUT = PROJ / "Data/analysis/ferricyanide/lamp_drift_check"
L_CAM_CM = 0.285


def predicted_r_absorbance(conc_m):
    eps = pd.read_csv(RECALC / "epsilon_spectra.csv")
    eps = eps[(eps.analyte == "ferricyanide") & (eps.window == "full_range")]
    sens = pd.read_csv(RECALC / "bayer_sensitivity_mean_20_devices.csv")
    lamp = pd.read_csv(RECALC / "illuminant_spectra.csv")
    lamp = lamp[lamp.illuminant == "illuminant_spelec_led"]
    wl = np.arange(400, 701, 1.0)
    e = np.clip(np.interp(wl, eps.wavelength_nm, eps.epsilon_M_inv_cm_inv), 0, None)
    w = np.interp(wl, lamp.wavelength_nm, lamp.relative_power) * np.interp(
        wl, sens.wavelength_nm, sens.R
    )
    return np.array([-np.log10((w * 10 ** (-e * c * L_CAM_CM)).sum() / w.sum()) for c in conc_m])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(IMAGES)
    d["A_R_model"] = predicted_r_absorbance(d.concentration_um * 1e-6)
    d["A_lamp"] = d.A_R_full - d.A_R_model
    d["minutes"] = d.gap_seconds / 60
    rows = []
    for name, g in d.groupby("set"):
        fit = stats.linregress(g.minutes, g.A_lamp)
        low = g[g.concentration_um <= 125]
        rows.append(
            {
                "set": name,
                "n": len(g),
                "run_min": round(g.minutes.max(), 1),
                "r_lamp_vs_time": round(fit.rvalue, 3),
                "p": f"{fit.pvalue:.2g}",
                "slope_AU_per_min": round(fit.slope, 4),
                "output_change_pct_per_min": round((10 ** (-fit.slope) - 1) * 100, 2),
                "A_lamp_last": round(g.sort_values("minutes").A_lamp.iloc[-1], 4),
                "max_A_R_model": round(g.A_R_model.max(), 4),
                "max_A_R_obs_le_125uM": round(low.A_R_full.max(), 4),
            }
        )
    d.to_csv(OUT / "lamp_drift_per_image.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "lamp_drift_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
