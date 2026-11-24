"""Transparent-tail decomposition of the Figure 5 integrands versus concentration.

Answers Reviewer 2, Question 2. Uses the same spectra, eps, l_cam, grid and
tail boundaries as figure_5.py (LED), at every measured concentration.
The *_no_tail columns treat the channel as if its transparent tail were
removed, which separates the tail's share of the compression from that of
the variation of eps inside the absorbing band.
Output: Data/analysis/integrand_tail_decomposition/integrand_tail_decomposition_LED.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_5 as f5  # noqa: E402
from forward_model import load_uvvis, split_at_cutoff  # noqa: E402

OUT_DIR = f5.DATA_DIR / "analysis" / "integrand_tail_decomposition"

FERIC_CONCS = tuple(7.8125 * 2**k for k in range(11))
MB_CONCS = (0.78125, 1.5625, 3.125, 6.25, 12.5, 25.0, 50.0, 100.0)


def decompose(phi_g, T_g, eps_g, l_cam, concs, ch_idx, tail_wl, tail_above):
    concs = (0.0,) + tuple(concs)
    integ, _ = f5.compute_integrands(phi_g, T_g, eps_g, l_cam, concs, ch_idx)
    mask = f5.GRID > tail_wl if tail_above else f5.GRID < tail_wl
    rows = []
    for c, y in zip(concs, integ):
        tail, active = split_at_cutoff(y, mask, f5.GRID)
        rows.append(dict(c_uM=c, tail_part=tail, active_part=active))
    df = pd.DataFrame(rows)
    blank_tail, blank_active = df.loc[0, ["tail_part", "active_part"]]
    df["transmitted"] = df.tail_part + df.active_part
    df["tail_share_of_surviving_light"] = df.tail_part / df.transmitted
    df["active_transmittance"] = df.active_part / blank_active
    df["A_channel"] = -np.log10(df.transmitted)
    df["A_ceiling"] = -np.log10(blank_tail)
    df["A_beer_lambert"] = (
        np.trapezoid(integ[0] * eps_g, x=f5.GRID) * np.array(concs) * 1e-6 * l_cam
    )
    df["A_over_beer_lambert"] = df.A_channel / df.A_beer_lambert
    act = ~mask
    x = f5.GRID[act]
    base = np.trapezoid(integ[0][act], x=x)
    eps_active = np.trapezoid(integ[0][act] * eps_g[act], x=x) / base
    df["A_no_tail"] = [-np.log10(np.trapezoid(y[act], x=x) / base) for y in integ]
    df["A_no_tail_over_beer_lambert"] = df.A_no_tail / (eps_active * df.c_uM * 1e-6 * l_cam)
    return df.iloc[1:].drop(columns=["tail_part", "active_part"])


def main() -> None:
    wl_s, s = f5.load_bayer_sensitivity()
    T_g = f5.sens_to_grid(wl_s, s)
    phi_lam, phi = f5.load_illuminant(f5.ILLUM_DIR / "illuminant_spelec_led.csv")
    phi_g = f5.to_grid(phi_lam, phi)

    feric = [f5.FERIC_DIR / f"Ferricyanide_UV_VIS_{i}.csv" for i in (1, 2, 3)]
    lam, eps, _ = load_uvvis(feric, f5.L_UVVIS)
    feric_eps = f5.to_grid(lam, eps)
    mb = [
        f5.MB_DIR / n
        for n in (
            "Set_1_MB_UV_VIS_25052026.csv",
            "Set_2_MB_UV_VIS_27052026.csv",
            "Set-3_MB_UV_VIS_27052026.csv",
        )
    ]
    lam, eps, _ = load_uvvis(mb, f5.L_UVVIS, **f5.MB_EPS_WINDOW)
    mb_eps = f5.to_grid(lam, eps)

    cases = [
        ("ferricyanide", "B", feric_eps, f5.L_CAM_LED, FERIC_CONCS, 2, f5.TAIL_WL, True),
        ("ferricyanide", "G", feric_eps, f5.L_CAM_LED, FERIC_CONCS, 1, f5.TAIL_WL, True),
        ("methylene_blue", "R", mb_eps, f5.MB_L_CAM_LED, MB_CONCS, 0, f5.MB_TAIL_WL, False),
        ("methylene_blue", "G", mb_eps, f5.MB_L_CAM_LED, MB_CONCS, 1, f5.MB_TAIL_WL, False),
    ]
    frames = []
    for analyte, ch, eps_g, l_cam, concs, idx, wl, above in cases:
        df = decompose(phi_g, T_g, eps_g, l_cam, concs, idx, wl, above)
        df.insert(0, "channel", ch)
        df.insert(0, "analyte", analyte)
        df["tail_boundary_nm"] = wl
        df["l_cam_cm"] = l_cam
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "integrand_tail_decomposition_LED.csv"
    out.to_csv(path, index=False, float_format="%.6g")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(out.to_string(index=False))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
