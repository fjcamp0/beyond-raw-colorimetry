"""Propagate the spread of the 20 Ohlab device curves through the forward model.

Answers Reviewer 2, Question 3. Each individual device curve replaces the
20-device average in turn. For each one: f_tail, a refitted l_cam (canonical
fit settings), and the concentration error made when a camera with that
device's response is read back through the average-curve model.
Output: Data/analysis/device_sensitivity_propagation/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_5 as f5  # noqa: E402
from forward_model import (  # noqa: E402
    _RIA_DIR,
    _add_ratio_channel,
    fit_l_cam,
    load_sensitivity,
    load_uvvis,
    predict_absorbance,
    split_at_cutoff,
)

OUT_DIR = f5.DATA_DIR / "analysis" / "device_sensitivity_propagation"
SENS_FILES = _RIA_DIR / "_resources" / "spectral_sensitivity"
C_GRID_M = np.logspace(-8, -2, 4000)

CASES = {
    "ferricyanide": dict(
        primary="B",
        ratio="BdivG",
        fit_channel="BdivG",
        fit_min=62.5,
        fit_max=1000.0,
        tail_wl=f5.TAIL_WL,
        tail_above=True,
        ch_idx=2,
    ),
    "methylene_blue": dict(
        primary="R",
        ratio="RdivG",
        fit_channel="R",
        fit_min=0.78,
        fit_max=100.0,
        tail_wl=f5.MB_TAIL_WL,
        tail_above=False,
        ch_idx=0,
    ),
}


def devices() -> list[str]:
    return sorted(p.stem for p in SENS_FILES.glob("*.txt") if p.stem != "average_all")


def f_tail(phi_g, T_g, ch_idx, tail_wl, tail_above) -> float:
    y = phi_g * T_g[:, ch_idx]
    mask = f5.GRID > tail_wl if tail_above else f5.GRID < tail_wl
    tail, active = split_at_cutoff(y, mask, f5.GRID)
    return tail / (tail + active)


def invert(c_true_M, a_true, phi_g, T_ref, eps_g, l_ref, channel):
    a_ref = predict_absorbance(C_GRID_M, phi_g, T_ref, eps_g, l_ref)[channel]
    ok = np.isfinite(a_ref)
    a_ref, c_ref = np.maximum.accumulate(a_ref[ok]), C_GRID_M[ok]
    c_hat = np.interp(a_true, a_ref, c_ref, left=np.nan, right=np.nan)
    return 100.0 * (c_hat - c_true_M) / c_true_M


def load_inputs():
    phi_lam, phi = f5.load_illuminant(f5.ILLUM_DIR / "illuminant_spelec_led.csv")
    phi_g = f5.to_grid(phi_lam, phi)
    feric = [f5.FERIC_DIR / f"Ferricyanide_UV_VIS_{i}.csv" for i in (1, 2, 3)]
    lam, eps, _ = load_uvvis(feric, f5.L_UVVIS)
    eps_f = f5.to_grid(lam, eps)
    mb = [
        f5.MB_DIR / n
        for n in (
            "Set_1_MB_UV_VIS_25052026.csv",
            "Set_2_MB_UV_VIS_27052026.csv",
            "Set-3_MB_UV_VIS_27052026.csv",
        )
    ]
    lam, eps, _ = load_uvvis(mb, f5.L_UVVIS, **f5.MB_EPS_WINDOW)
    eps_m = f5.to_grid(lam, eps)
    cal = {
        a: pd.read_csv(f5.DATA_DIR / "analysis" / a / "beer_lambert" / "calibration_table.csv")
        for a in CASES
    }
    return phi_g, {"ferricyanide": eps_f, "methylene_blue": eps_m}, cal


def add_ratio(cal: pd.DataFrame, name: str) -> pd.DataFrame:
    return cal if name in set(cal.channel) else _add_ratio_channel(cal, name)


def main() -> None:
    phi_g, eps, cal = load_inputs()
    wl, s_avg = load_sensitivity("average_all")
    T_avg = f5.sens_to_grid(wl, s_avg)
    stack = np.stack([load_sensitivity(d)[1] for d in devices()])
    print(
        f"average_all vs mean of {len(stack)} files: max |diff| = "
        f"{np.abs(stack.mean(axis=0) - s_avg).max():.2e}"
    )

    param_rows, err_rows, meas_rows = [], [], []
    for analyte, k in CASES.items():
        c_tab = add_ratio(cal[analyte], k["ratio"])
        ref_l, _ = fit_l_cam(
            c_tab, phi_g, T_avg, eps[analyte], k["fit_channel"], k["fit_max"], k["fit_min"]
        )
        conc = np.sort(c_tab.loc[c_tab.channel == k["primary"], "concentration_um"].unique())
        c_M = conc * 1e-6

        for ch in (k["primary"], k["ratio"]):
            meas = c_tab[c_tab.channel == ch].set_index("concentration_um")
            a_meas = meas.loc[conc, "absorbance_mean"].to_numpy()
            e = invert(c_M, a_meas, phi_g, T_avg, eps[analyte], ref_l, ch)
            meas_rows += [
                dict(analyte=analyte, channel=ch, c_uM=c, error_pct=v) for c, v in zip(conc, e)
            ]

        for dev in ["average_all"] + devices():
            T = f5.sens_to_grid(*load_sensitivity(dev))
            l_dev, _ = fit_l_cam(
                c_tab, phi_g, T, eps[analyte], k["fit_channel"], k["fit_max"], k["fit_min"]
            )
            param_rows.append(
                dict(
                    analyte=analyte,
                    device=dev,
                    l_cam_mm=10 * l_dev,
                    f_tail_pct=100 * f_tail(phi_g, T, k["ch_idx"], k["tail_wl"], k["tail_above"]),
                )
            )
            if dev == "average_all":
                continue
            pred = predict_absorbance(c_M, phi_g, T, eps[analyte], l_dev)
            for ch in (k["primary"], k["ratio"]):
                e = invert(c_M, pred[ch], phi_g, T_avg, eps[analyte], ref_l, ch)
                err_rows += [
                    dict(analyte=analyte, channel=ch, device=dev, c_uM=c, error_pct=v)
                    for c, v in zip(conc, e)
                ]

    params = pd.DataFrame(param_rows)
    errs = pd.DataFrame(err_rows)
    meas = pd.DataFrame(meas_rows)
    summ = (
        errs.assign(abs_err=errs.error_pct.abs())
        .groupby(["analyte", "channel", "c_uM"])
        .agg(
            sd_pct=("error_pct", "std"),
            median_abs_pct=("abs_err", "median"),
            max_abs_pct=("abs_err", "max"),
        )
        .reset_index()
        .merge(
            meas.rename(columns={"error_pct": "avg_model_on_data_pct"}),
            on=["analyte", "channel", "c_uM"],
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params.to_csv(OUT_DIR / "per_device_ftail_lcam.csv", index=False, float_format="%.6g")
    errs.to_csv(OUT_DIR / "per_device_concentration_error.csv", index=False, float_format="%.6g")
    summ.to_csv(OUT_DIR / "concentration_error_summary.csv", index=False, float_format="%.4g")

    dev_only = params[params.device != "average_all"]
    stats = dev_only.groupby("analyte")[["f_tail_pct", "l_cam_mm"]].agg(
        ["mean", "std", "min", "max"]
    )
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(params.to_string(index=False))
        print(stats.to_string())
        print(summ.to_string(index=False))
    print(f"Saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
