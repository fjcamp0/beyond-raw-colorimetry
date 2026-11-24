"""Full width at half maximum of each Bayer channel, from the 20 Ohlab device curves.

Answers Reviewer 3, minor 3: the uncited "typically more than 100 nm, full width" in §2.
Inputs: the Ohlab spectral sensitivity files read by forward_model.load_sensitivity
(400-700 nm, 10 nm steps; vendored in vendor/).
Output: Data/analysis/bayer_channel_fwhm/ (per-device and summary CSVs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_model import _RIA_DIR, load_sensitivity  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "Data" / "analysis" / "bayer_channel_fwhm"
SENS_FILES = _RIA_DIR / "_resources" / "spectral_sensitivity"
CHANNELS = ("R", "G", "B")


def fwhm(wl: np.ndarray, s: np.ndarray) -> tuple[float, bool]:
    """Return (FWHM in nm, truncated) for one curve, interpolating the half-maximum crossings."""
    i_pk = int(np.argmax(s))
    half = s[i_pk] / 2.0
    above = s >= half
    lo = i_pk
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = i_pk
    while hi < len(s) - 1 and above[hi + 1]:
        hi += 1
    # A curve still above half maximum at the grid edge gives only a lower bound.
    truncated = lo == 0 or hi == len(s) - 1
    left = wl[lo] if lo == 0 else np.interp(half, [s[lo - 1], s[lo]], [wl[lo - 1], wl[lo]])
    right = (
        wl[hi] if hi == len(s) - 1 else np.interp(half, [s[hi + 1], s[hi]], [wl[hi + 1], wl[hi]])
    )
    return float(right - left), truncated


def main() -> None:
    """Compute per-device FWHM for R, G, B and write per-device and summary tables."""
    devices = sorted(p.stem for p in SENS_FILES.glob("*.txt") if p.stem != "average_all")
    rows = []
    for dev in [*devices, "average_all"]:
        wl, s = load_sensitivity(dev)
        for k, ch in enumerate(CHANNELS):
            width, trunc = fwhm(wl, s[:, k])
            rows.append({"device": dev, "channel": ch, "fwhm_nm": width, "truncated": trunc})
    per = pd.DataFrame(rows)
    dev_only = per[per.device != "average_all"]
    summ = (
        dev_only.groupby("channel")
        .agg(
            n=("fwhm_nm", "size"),
            mean_nm=("fwhm_nm", "mean"),
            sd_nm=("fwhm_nm", "std"),
            min_nm=("fwhm_nm", "min"),
            max_nm=("fwhm_nm", "max"),
            n_truncated=("truncated", "sum"),
        )
        .reindex(CHANNELS)
    )
    avg = per[per.device == "average_all"].set_index("channel")["fwhm_nm"]
    summ["average_curve_nm"] = avg
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT_DIR / "per_device_fwhm.csv", index=False, float_format="%.1f")
    summ.to_csv(OUT_DIR / "fwhm_summary.csv", float_format="%.1f")
    print(summ.round(1).to_string())


if __name__ == "__main__":
    main()
