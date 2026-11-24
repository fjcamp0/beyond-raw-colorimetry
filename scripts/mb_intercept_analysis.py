"""
MB R-channel non-zero intercept — diagnostic.
Resolves the FATAL flag (risk_map Cat 5): is the positive A_R intercept a real c=0
offset, inter-session drift, or forward-model compression curvature?

Reads the per-image and per-concentration tables produced by beer_lambert.py.
Pure stdlib + numpy; reads CSVs with csv module to avoid pandas/Arrow quirks.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numfmt import pm  # CLAUDE.md hard rule 10: SD to two significant figures

BASE = (
    Path(__file__).resolve().parent.parent / "Data" / "analysis" / "methylene_blue" / "beer_lambert"
)
PER_IMG = BASE / "absorbance_table.csv"
PER_CONC = BASE / "calibration_table.csv"


def ols(c, a):
    """Ordinary least squares with intercept. Returns slope, intercept, R^2."""
    c = np.asarray(c, float)
    a = np.asarray(a, float)
    A = np.vstack([c, np.ones_like(c)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, a, rcond=None)
    pred = slope * c + intercept
    ss_res = np.sum((a - pred) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2


def ols_intercept_ci(c, a):
    """Intercept with its standard error and 95% CI (t not applied; ~2*SE)."""
    c = np.asarray(c, float)
    a = np.asarray(a, float)
    n = len(c)
    slope, intercept, _ = ols(c, a)
    pred = slope * c + intercept
    dof = n - 2
    s2 = np.sum((a - pred) ** 2) / dof
    cbar = c.mean()
    sxx = np.sum((c - cbar) ** 2)
    se_int = np.sqrt(s2 * (1.0 / n + cbar**2 / sxx))
    return intercept, se_int


# ---- load per-concentration means (R channel) ----
rows = list(csv.DictReader(open(PER_CONC, encoding="utf-8")))
R = [
    (float(r["concentration_um"]), float(r["absorbance_mean"]), float(r["absorbance_std"]))
    for r in rows
    if r["channel"] == "R"
]
R.sort()
c_all = [x[0] for x in R]
a_all = [x[1] for x in R]
sd_all = [x[2] for x in R]

print("=== MB R-channel per-concentration means ===")
for cc, aa, ss in R:
    print(f"  {cc:7.2f} µM   A_R = {pm(aa, ss, sign=True)}")

slope, intercept, r2 = ols(c_all, a_all)
ic, se = ols_intercept_ci(c_all, a_all)
print(f"\nFull-range OLS (0.78-100): slope={slope:.5e}  intercept={intercept:+.5f}  R2={r2:.5f}")
print(
    f"  intercept = {pm(ic, se, sign=True)} (1 SE)  -> 95% ~ {ic - 2 * se:+.5f} .. {ic + 2 * se:+.5f}"
)
print(f"  intercept in conc units = {intercept / slope:+.2f} µM-equivalent")

# fit high-c only (>=12.5) and back-extrapolate to c=0
hi = [(cc, aa) for cc, aa in zip(c_all, a_all) if cc >= 12.5]
s_hi, i_hi, r2_hi = ols([x[0] for x in hi], [x[1] for x in hi])
print(f"\nHigh-c OLS (>=12.5):  slope={s_hi:.5e}  intercept={i_hi:+.5f}  R2={r2_hi:.5f}")

# fit low-c only (<=12.5)
lo = [(cc, aa) for cc, aa in zip(c_all, a_all) if cc <= 12.5]
s_lo, i_lo, r2_lo = ols([x[0] for x in lo], [x[1] for x in lo])
print(f"Low-c OLS  (<=12.5):  slope={s_lo:.5e}  intercept={i_lo:+.5f}  R2={r2_lo:.5f}")

# force-through-origin slope and residual at low c
c_arr = np.array(c_all)
a_arr = np.array(a_all)
slope0 = np.dot(c_arr, a_arr) / np.dot(c_arr, c_arr)
print(f"\nForce-through-origin slope = {slope0:.5e}")
print("Residuals (A_obs - slope0*c):")
for cc, aa in zip(c_all, a_all):
    print(f"  {cc:7.2f} µM   resid = {aa - slope0 * cc:+.5f}")

# ---- per-set R-channel from per-image table ----
img = list(csv.DictReader(open(PER_IMG, encoding="utf-8")))
print("\n=== Per-set R-channel (is the low-c bias systematic?) ===")
sets = {}
for r in img:
    if r["channel"] != "R":
        continue
    sets.setdefault(r["set"], {})[round(float(r["concentration_um"]), 2)] = float(r["absorbance"])
concs = sorted({round(float(r["concentration_um"]), 2) for r in img if r["channel"] == "R"})
hdr = "  conc(µM) " + " ".join(f"{s:>10}" for s in sorted(sets))
print(hdr)
for cc in concs:
    line = f"  {cc:7.2f} " + " ".join(
        f"{sets[s].get(cc, float('nan')):+10.5f}" for s in sorted(sets)
    )
    print(line)

# per-set intercepts
print("\nPer-set full-range OLS intercept:")
for s in sorted(sets):
    cs = sorted(sets[s])
    aas = [sets[s][cc] for cc in cs]
    sl, ic_s, r2_s = ols(cs, aas)
    print(f"  {s}: slope={sl:.5e}  intercept={ic_s:+.5f}  R2={r2_s:.5f}")

# ---- ratio channels intercepts for comparison (from report values) ----
print("\n=== Ratio-channel intercepts (from calibration; lower = drift removed) ===")
for ch in ("RdivG", "RdivB", "BdivG"):
    rr = [
        (float(r["concentration_um"]), float(r["absorbance_mean"]))
        for r in rows
        if r["channel"] == ch
    ]
    if not rr:
        print(f"  {ch}: (not in calibration_table; computed post-hoc in report)")
        continue
    rr.sort()
    sl, ic_r, r2_r = ols([x[0] for x in rr], [x[1] for x in rr])
    print(f"  {ch}: slope={sl:.5e}  intercept={ic_r:+.5f}  R2={r2_r:.5f}")
