"""Diagnose the high apparent molar absorptivity of ferroin (Block 1 Q2).

Per-set epsilon(510), intercept and band shape from the three UV-Vis
calibration sets, and the iron-salt hydration that would explain the excess.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
UVVIS = PROJ / "Data" / "Phenanothroline_FeII estimation_wine appliation" / "UV-Vis"
OUT = PROJ / "Data" / "analysis" / "wine_ferroin" / "epsilon_diagnostic"
SETS = {
    "set1": "set1_calibration_phenanthroline.csv",
    "set2": "set2_calibration_phenanthroline.csv",
    "set3": "Set3_calibration_phenanthroline.csv",
}
PATH_CM = 0.2  # Labbox MALG-002-002, CLAUDE.md
EPS_LIT = 11100.0  # Fortune & Mellon 1938, as cited by Smith et al. 2021
EPS_SMITH = 12450.0  # Smith et al. 2021, AAS-standardised FeSO4.7H2O
M_LABEL = 278.01  # FeSO4.7H2O, Rajendra 2026-09-26
M_FE = 55.845
M_FESO4 = 151.90
M_H2O = 18.015


def read_set(path):
    raw = pd.read_csv(path, header=None, encoding="latin-1", dtype=str)
    cols = [i for i, lab in enumerate(raw.iloc[0]) if "M" in str(lab)]
    conc = [float(str(raw.iloc[0, i]).split()[0]) for i in cols]
    data = raw.iloc[2:].apply(pd.to_numeric, errors="coerce")
    wl = data.iloc[:, cols[0]].to_numpy()
    absb = data.iloc[:, [i + 1 for i in cols]].to_numpy()
    keep = ~np.isnan(wl) & (wl > 100)
    return np.array(conc), wl[keep], absb[keep]


def at(wl, absb, target):
    return absb[np.argmin(np.abs(wl - target))]


def fit(x, y):
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return slope, icpt, r2, y - pred


def hydrate_for(factor):
    m_actual = M_LABEL / factor
    return m_actual, (m_actual - M_FESO4) / M_H2O


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, shape_rows, pooled_x, pooled_y = [], [], [], []
    for name, fname in SETS.items():
        conc, wl, absb = read_set(UVVIS / fname)
        a510 = at(wl, absb, 510.0)
        slope, icpt, r2, resid = fit(conc, a510)
        pooled_x.extend(conc)
        pooled_y.extend(a510)
        eps = slope * 1e6 / PATH_CM
        rows.append(
            {
                "set": name,
                "n": len(conc),
                "slope_AU_per_uM": slope,
                "intercept_AU": icpt,
                "r2": r2,
                "eps_510": eps,
                "max_abs_resid_AU": np.max(np.abs(resid)),
                "A510_blank": a510[conc == 0][0] if (conc == 0).any() else np.nan,
            }
        )
        blank = absb[:, conc == 0][:, 0]
        for c, col in zip(conc, absb.T):
            if c == 0:
                continue
            col = col - blank
            peak = col[np.argmin(np.abs(wl - 510.0))]
            shape_rows.append(
                {
                    "set": name,
                    "conc_uM": c,
                    "A510": peak,
                    "lambda_max_nm": wl[(wl > 450) & (wl < 560)][
                        np.argmax(col[(wl > 450) & (wl < 560)])
                    ],
                    "A480_over_A510": at(wl, col, 480.0) / peak,
                    "A560_over_A510": at(wl, col, 560.0) / peak,
                    "A600_over_A510": at(wl, col, 600.0) / peak,
                }
            )
    slope, icpt, r2, _ = fit(np.array(pooled_x), np.array(pooled_y))
    eps = slope * 1e6 / PATH_CM
    rows.append(
        {
            "set": "pooled",
            "n": len(pooled_x),
            "slope_AU_per_uM": slope,
            "intercept_AU": icpt,
            "r2": r2,
            "eps_510": eps,
        }
    )
    per_set = pd.DataFrame(rows)
    shape = pd.DataFrame(shape_rows)
    per_set.to_csv(OUT / "epsilon_per_set.csv", index=False)
    shape.to_csv(OUT / "band_shape.csv", index=False)

    salts = pd.DataFrame(
        [
            {
                "reference": ref,
                "eps_ref": e,
                "factor": eps / e,
                "implied_M_g_mol": hydrate_for(eps / e)[0],
                "implied_water_per_FeSO4": hydrate_for(eps / e)[1],
            }
            for ref, e in [("literature 11,100", EPS_LIT), ("Smith 2021", EPS_SMITH)]
        ]
    )
    salts.to_csv(OUT / "implied_hydration.csv", index=False)
    candidates = pd.DataFrame(
        [
            {
                "salt": f"FeSO4.{n}H2O" if n else "FeSO4",
                "M": M_FESO4 + n * M_H2O,
                "factor_if_weighed_as_heptahydrate": M_LABEL / (M_FESO4 + n * M_H2O),
                "eps_apparent_from_11100": EPS_LIT * M_LABEL / (M_FESO4 + n * M_H2O),
                "eps_apparent_from_12450": EPS_SMITH * M_LABEL / (M_FESO4 + n * M_H2O),
            }
            for n in (7, 4, 1, 0)
        ]
    )
    candidates.to_csv(OUT / "candidate_salts.csv", index=False)

    with pd.option_context("display.width", 160, "display.precision", 4):
        print(per_set.to_string(index=False))
        print()
        print(
            shape.groupby("set")[
                ["lambda_max_nm", "A480_over_A510", "A560_over_A510", "A600_over_A510"]
            ]
            .agg(["mean", "std"])
            .to_string()
        )
        print()
        print(salts.to_string(index=False))
        print()
        print(candidates.to_string(index=False))
    print(f"\nFe in 1.112 mg FeSO4.7H2O: {1.112 * M_FE / M_LABEL * 1000:.1f} ug")


if __name__ == "__main__":
    main()
