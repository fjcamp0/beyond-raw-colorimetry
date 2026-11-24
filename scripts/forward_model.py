# Copyright 2024 F. Javier del Campo
# Licensed under the Apache License, Version 2.0

"""
forward_model.py — Beer-Lambert-Bayer forward model for smartphone colorimetry

Numerically integrates:

    I_ch(c) = ∫ φ(λ) T_ch(λ) 10^(−ε(λ) c l) dλ
    A_ch(c) = −log10[ I_ch(c) / I_ch(0) ]

and overlays predictions against empirical calibration data from beer_lambert.py.

Usage — ferricyanide (default behaviour, backward-compatible):
    python raw-image-analysis/forward_model.py \\
        --project  /path/to/Ferricyanide \\
        --illuminant /path/to/Spelec_Specta_LED_Illuminator.csv \\
        --uvvis /path/to/UV-Vis_1.csv \\
        --uvvis /path/to/UV-Vis_2.csv \\
        --uvvis /path/to/UV-Vis_3.csv

Usage — methylene blue:
    python raw-image-analysis/forward_model.py \\
        --project  /path/to/"Methyelene blue" \\
        --illuminant /path/to/Spelec_Specta_LED_Illuminator.csv \\
        --uvvis /path/to/MB_UV-Vis_1.csv \\
        --uvvis /path/to/MB_UV-Vis_2.csv \\
        --uvvis /path/to/MB_UV-Vis_3.csv \\
        --analyte "Methylene blue" \\
        --analyte-formula "MB" \\
        --primary-channel R \\
        --ratio-channel RdivB \\
        --lambda-max 665 \\
        --tail-wavelength 426 \\
        --fit-channel R \\
        --fit-min-conc 0 \\
        --fit-max-conc 100

Outputs (project/image_analysis/forward_model/):
    01_forward_model_spectra.png        — φ(λ), T_ch(λ), ε(λ) diagnostic overlay
    02_forward_model_components.png     — integrand decomposition at several concentrations
    03_forward_model_calibration.png    — predicted vs empirical A_primary and A_ratio
    04_forward_model_tail_fraction.png  — primary/reference channel transparent-tail fraction vs [c]
    05_forward_model_correction.png     — retrieved vs true c: linear calibration vs model inversion
    06_forward_model_linearisation.png  — original vs corrected A on linear axes
    forward_model_table.csv             — model predictions on dense concentration grid
    forward_model_summary.md            — fitted l_cam, channel integrals, summary
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).parent

# Vendored raw-image-analysis modules (vendor/, see README); override with COLORIMETRY_VENDOR.
import os as _os

_RIA_DIR = Path(
    _os.environ.get("COLORIMETRY_VENDOR", str(Path(__file__).resolve().parent.parent / "vendor"))
)

# Integration grid: 400–700 nm at 1 nm steps
_GRID = np.arange(400.0, 701.0, 1.0)  # 301 points

CHANNEL_COLORS = {
    "R": "#e74c3c",
    "G": "#27ae60",
    "G1": "#2ecc71",
    "G2": "#27ae60",
    "B": "#2980b9",
    "BdivG": "#e67e22",
    "RdivG": "#c0392b",
    "RdivB": "#922b21",
    "GdivB": "#1abc9c",
    "GmB": "#16a085",
}

_CH_IDX = {"R": 0, "G": 1, "B": 2}
_RATIO_PAIRS = {
    "BdivG": ("B", "G"),
    "BdivR": ("B", "R"),
    "RdivG": ("R", "G"),
    "RdivB": ("R", "B"),
    "GdivB": ("G", "B"),
}

LN10 = np.log(10.0)


@dataclass
class AnalyteConfig:
    """Analyte optical parameters: signal channels and absorption peak/tail wavelengths."""

    analyte: str = "analyte"
    formula: str = "analyte"
    primary_ch: str = "B"
    ratio_ch: str = "BdivG"
    lambda_max: float = 420.0
    tail_wl: float = 473.0


def _ch_label(ch: str) -> str:
    """Human-readable channel label for axis annotations."""
    label_map = {
        "BdivG": "B/G",
        "RdivG": "R/G",
        "RdivB": "R/B",
        "GdivB": "G/B",
        "G1": "G₁",
        "G2": "G₂",
        "GmB": "G−B",
    }
    return label_map.get(ch, ch)


# ─── CSV helpers ──────────────────────────────────────────────────────────────


def _read_csv_numeric(path: Path) -> pd.DataFrame:
    """
    Read a UV-Vis or spectrometer CSV robustly.
    Pandas 3 defaults to an Arrow string backend; this forces all columns to
    numeric, dropping non-parseable footer/header rows automatically.
    """
    df = pd.read_csv(path, header=0)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[df.columns[0]]).reset_index(drop=True)
    return df


# ─── Data loaders ─────────────────────────────────────────────────────────────


def load_illuminant(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load illuminant CSV.  Two formats are supported:
    - USB spectrometer (≥3 cols): wavelength, dark, LED light → φ = LED − dark, clipped ≥ 0
    - Standard illuminant (2 cols): wavelength, S(λ) → φ = S(λ) directly
    Returns (lam, phi) sorted ascending.
    """
    df = _read_csv_numeric(path)
    lam = df.iloc[:, 0].to_numpy(dtype=float)
    if len(df.columns) >= 3:
        dark = df.iloc[:, 1].to_numpy(dtype=float)
        led = df.iloc[:, 2].to_numpy(dtype=float)
        phi = np.clip(led - dark, 0.0, None)
    else:
        phi = df.iloc[:, 1].to_numpy(dtype=float)
    order = np.argsort(lam)
    return lam[order], phi[order]


def _col_to_um(col: str) -> float | None:
    """
    Parse a UV-Vis column header to µM.  None if not a concentration label.
    Handles optional prefix (e.g. 'MB_7.812uM', 'FC_125µM').
    """
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*([µuU]M|mM|M)\s*$", str(col).strip())
    if not m:
        return None
    val = float(m.group(1).replace(",", "."))
    unit = m.group(2).replace("µ", "u").lower()
    if unit == "mm":
        return val * 1000.0
    if unit == "m":
        return val * 1e6
    return val


def load_uvvis(
    paths: list[Path],
    pathlength_cm: float,
    min_fit_conc_um: float | None = None,
    max_fit_conc_um: float | None = None,
    exclude_columns: tuple[str, ...] = (),
    free_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one or more UV-Vis replicates and compute ε(λ) from a single pooled fit.

    Replicates are POOLED: every individual (concentration, absorbance) point from
    every file enters one least-squares fit. This is gate G-G in CLAUDE.md, "pooled
    individual replicate points, never means". Until 2026-09-21 this function fitted
    each file separately and averaged the resulting ε spectra, which is not the
    project's convention. Pooling reproduces the canonical values exactly:
    ferricyanide ε(420) = 1017, n = 30, R² = 0.9986, intercept −0.0014 AU over the
    full 7.8 µM–8 mM range; methylene blue ε(665) = 74,168, n = 18, R² = 0.9962,
    intercept +0.0200 AU over 1.525–50 µM with MB_7.812uM excluded.

    There is deliberately NO default concentration window. The old
    max_fit_conc_um = 1000.0 default silently truncated ferricyanide at 1 mM and
    returned ε(420) = 981, 3.5% below the canonical value, and every caller relied
    on that default. Averaging versus pooling is worth only 0.4% by comparison
    (1012.7 vs 1016.7); the truncation was the real defect. The window is
    analyte-specific and must be passed explicitly: ferricyanide uses its full
    measured range, methylene blue does not (CLAUDE.md, and
    _ai_outputs/review/mb_epsilon_reversal_2026-07-28.md).

    exclude_columns drops columns by exact header name, which a concentration window
    cannot do: Set 1's ``MB_7.812uM`` is really 0.7812 µM, so it sits inside the
    1.525–50 µM window under a disputed label. Leaving it in returns ε(665) = 75,135
    instead of 74,168. The exclusion is therefore explicit and logged, never a side
    effect of the window.

    The fit carries a free intercept by default. Pass free_intercept=False for the
    old zero-intercept behaviour. Note that the free intercept absorbs two
    physically different things depending on wavelength, and only one is an
    artefact: at 420 nm it removes a genuine additive offset (+0.0335 AU, flat in
    every concentration window), which a zero-intercept slope would otherwise fold
    into ε — the methylene blue files carry no blank column, and that baseline was
    inflating ε(420) fivefold. At 665 nm it is not a baseline at all but real
    monomer/dimer curvature: the intercept collapses from +0.049 to +0.0007 as the
    window narrows. The canonical ε does the same thing, so this is consistent, but
    do not describe the free intercept as simply "removing the baseline".

    Returns
    -------
    lam      : wavelength array (nm), ascending
    eps      : ε(λ) in M⁻¹ cm⁻¹ from the pooled fit, clipped ≥ 0
    eps_se   : standard error of ε(λ) from that same fit.
               NOTE: this is NOT the old inter-replicate standard deviation — pooling
               leaves no per-replicate spread to take an SD of. Callers that shade
               ε ± this value are now drawing a 1-SE confidence band on the fit, so
               any figure legend or CSV column still calling it an inter-replicate SD
               needs relabelling (daylight_diffuser_figures.py,
               daylight_diffuser_figures_MB.py, and plot_inputs below).
    """
    ref_lam: np.ndarray | None = None
    pooled_c: list[float] = []
    pooled_A: list[np.ndarray] = []
    kept: list[str] = []
    dropped: list[str] = []

    for path in paths:
        df = _read_csv_numeric(path)
        lam = df.iloc[:, 0].to_numpy(dtype=float)

        if lam[0] > lam[-1]:
            df = df.iloc[::-1].reset_index(drop=True)
            lam = df.iloc[:, 0].to_numpy(dtype=float)

        concs_um: list[float] = []
        a_cols: list[np.ndarray] = []
        labels: list[str] = []

        # Detect format: if col 0 header is itself a concentration label, the CSV uses
        # interleaved pairs — (wavelength_col, absorbance_col) per concentration.
        # Otherwise standard format: col 0 = wavelength, optional blank, then concentrations.
        if _col_to_um(df.columns[0]) is not None:
            # Interleaved format: even cols = wavelength (labeled with conc), odd = absorbance
            blank = np.zeros(len(df))
            for i in range(0, len(df.columns) - 1, 2):
                c = _col_to_um(df.columns[i])
                if c is not None:
                    concs_um.append(c)
                    a_cols.append(df.iloc[:, i + 1].to_numpy(dtype=float))
                    labels.append(str(df.columns[i]).strip())
        else:
            # Standard format: col 0 = wavelength, col 1 = blank (if no conc label) or conc
            has_blank = _col_to_um(df.columns[1]) is None
            blank = df.iloc[:, 1].to_numpy(dtype=float) if has_blank else np.zeros(len(df))
            col_start = 2 if has_blank else 1
            for i in range(col_start, len(df.columns)):
                c = _col_to_um(df.columns[i])
                if c is not None:
                    concs_um.append(c)
                    a_cols.append(df.iloc[:, i].to_numpy(dtype=float))
                    labels.append(str(df.columns[i]).strip())

        if not concs_um:
            raise ValueError(f"No concentration columns found in {path.name}")

        if ref_lam is None:
            ref_lam = lam

        # Pool onto the first file's wavelength grid. Interpolating the net absorbance
        # before the fit, rather than interpolating ε after three separate fits, is
        # what makes the single pooled fit possible at all.
        for c_um, a_col, label in zip(concs_um, a_cols, labels):
            why = None
            if label in exclude_columns:
                why = "excluded by name"
            elif min_fit_conc_um is not None and c_um < min_fit_conc_um:
                why = f"below {min_fit_conc_um} µM"
            elif max_fit_conc_um is not None and c_um > max_fit_conc_um:
                why = f"above {max_fit_conc_um} µM"
            if why:
                dropped.append(f"{path.name}:{label} ({why})")
                continue
            pooled_c.append(c_um * 1e-6)
            pooled_A.append(np.interp(ref_lam, lam, a_col - blank, left=0.0, right=0.0))
            kept.append(f"{path.name}:{label}")

    if len(pooled_c) < 3:
        raise ValueError(
            f"Only {len(pooled_c)} point(s) survived the window "
            f"{min_fit_conc_um}–{max_fit_conc_um} µM with {list(exclude_columns)} "
            "excluded; a pooled fit needs at least three."
        )

    c = np.asarray(pooled_c, dtype=float)  # M
    A = np.column_stack(pooled_A)  # (n_lambda, n_points)
    n = c.size

    if free_intercept:
        denom = n * float(np.dot(c, c)) - c.sum() ** 2
        slope = (n * (A @ c) - A.sum(axis=1) * c.sum()) / denom
        icpt = (A.sum(axis=1) - slope * c.sum()) / n
        resid = A - (slope[:, None] * c[None, :] + icpt[:, None])
        dof = n - 2
        c_ss = float(((c - c.mean()) ** 2).sum())
    else:
        slope = (A @ c) / float(np.dot(c, c))
        resid = A - slope[:, None] * c[None, :]
        dof = n - 1
        c_ss = float(np.dot(c, c))

    slope_se = np.sqrt((resid**2).sum(axis=1) / max(dof, 1) / c_ss)

    print(
        f"  load_uvvis: pooled fit over n={n} points from {len(paths)} file(s); "
        f"window {min_fit_conc_um}–{max_fit_conc_um} µM; "
        f"{len(dropped)} column(s) dropped"
    )
    for d in dropped:
        print(f"    dropped {d}")

    return (
        ref_lam,
        np.clip(slope / pathlength_cm, 0.0, None),  # M⁻¹ cm⁻¹
        slope_se / pathlength_cm,
    )


def load_sensitivity(device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (WAVELENGTHS 400–700 / 10 nm, S 31×3) from the Ohlab database."""
    sys.path.insert(0, str(_RIA_DIR))
    from core.spectral import WAVELENGTHS
    from core.spectral import load_sensitivity as _load

    return WAVELENGTHS.copy(), _load(device).copy()


def split_at_cutoff(
    y: np.ndarray, mask: np.ndarray, x: np.ndarray | None = None
) -> tuple[float, float]:
    """Split the integral of y into (masked, unmasked) parts.

    Integrating a boolean-sliced array drops the trapezoid spanning the mask
    boundary, so the two parts no longer sum to the total. That boundary
    trapezoid is shared evenly, which conserves the total and matches the
    convention of the 2026-09-01 c* audit.
    """
    kw = {"x": x} if x is not None else {"dx": 1.0}
    total = float(np.trapezoid(y, **kw))
    inside = float(np.trapezoid(y[mask], **({"x": x[mask]} if x is not None else {"dx": 1.0})))
    outside = float(np.trapezoid(y[~mask], **({"x": x[~mask]} if x is not None else {"dx": 1.0})))
    half = (total - inside - outside) / 2.0
    return inside + half, outside + half


# ─── Grid helpers ─────────────────────────────────────────────────────────────


def _to_grid(lam: np.ndarray, vals: np.ndarray) -> np.ndarray:
    return np.interp(_GRID, lam, vals, left=0.0, right=0.0)


def _sens_to_grid(wl: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Interpolate 31×3 Ohlab sensitivity onto _GRID. Returns (n_grid, 3)."""
    out = np.zeros((len(_GRID), 3))
    for i in range(3):
        out[:, i] = np.interp(_GRID, wl, S[:, i], left=0.0, right=0.0)
    return out


# ─── Forward model ────────────────────────────────────────────────────────────


def predict_absorbance(
    c_arr: np.ndarray,  # concentrations in M
    phi_g: np.ndarray,  # φ(λ) on _GRID
    T_g: np.ndarray,  # T_ch(λ) on _GRID, shape (n_grid, 3): R G B
    eps_g: np.ndarray,  # ε(λ) in M⁻¹ cm⁻¹ on _GRID
    l_cam: float,  # path length in cm
) -> dict[str, np.ndarray]:
    """
    Compute A_ch(c) = −log10[I_ch(c)/I_ch(0)] for R, G, B and all ratio channels.

    Spectral transmittance: T(λ,c) = 10^(−ε·c·l) = exp(−ε·c·l·ln10)
    """
    I0 = np.array([np.trapezoid(phi_g * T_g[:, i], dx=1.0) for i in range(3)])

    result: dict[str, np.ndarray] = {
        ch: np.full(len(c_arr), np.nan)
        for ch in ("R", "G", "B", "BdivG", "BdivR", "RdivG", "RdivB", "GdivB")
    }

    for j, c in enumerate(c_arr):
        trans = np.exp(-eps_g * (c * l_cam) * LN10)
        for i, ch in enumerate(("R", "G", "B")):
            Ic = np.trapezoid(phi_g * T_g[:, i] * trans, dx=1.0)
            result[ch][j] = -np.log10(Ic / I0[i]) if Ic > 0 else np.nan
        result["BdivG"][j] = result["B"][j] - result["G"][j]
        result["BdivR"][j] = result["B"][j] - result["R"][j]
        result["RdivG"][j] = result["R"][j] - result["G"][j]
        result["RdivB"][j] = result["R"][j] - result["B"][j]
        result["GdivB"][j] = result["G"][j] - result["B"][j]

    return result


# ─── l_cam fitting ────────────────────────────────────────────────────────────


def fit_l_cam(
    cal: pd.DataFrame,
    phi_g: np.ndarray,
    T_g: np.ndarray,
    eps_g: np.ndarray,
    fit_channel: str = "BdivG",
    fit_max_um: float = 1000.0,
    fit_min_um: float = 62.5,
) -> tuple[float, float]:
    """
    Log-space grid search (0.005–20 cm) for l_cam that matches the empirical
    Beer-Lambert slope in the linear concentration range.

    Returns (l_cam_cm, empirical_slope_M).
    """
    emp = cal[
        (cal["channel"] == fit_channel)
        & (cal["concentration_um"] >= fit_min_um)
        & (cal["concentration_um"] <= fit_max_um)
    ].copy()
    if emp.empty:
        raise ValueError(
            f"No calibration data for channel '{fit_channel}' in "
            f"[{fit_min_um}, {fit_max_um}] µM.  "
            f"Available channels: {sorted(cal['channel'].unique())}"
        )
    c_M = emp["concentration_um"].to_numpy(dtype=float) * 1e-6
    A_emp = emp["absorbance_mean"].to_numpy(dtype=float)

    emp_slope = float(np.dot(c_M, A_emp) / np.dot(c_M, c_M))

    best_l, best_err = 0.1, np.inf
    for l_trial in np.logspace(-2.3, 1.3, 600):
        pred = predict_absorbance(c_M, phi_g, T_g, eps_g, l_trial)
        A_pred = pred[fit_channel]
        ok = np.isfinite(A_pred)
        if not ok.any():
            continue
        pred_slope = float(np.dot(c_M[ok], A_pred[ok]) / np.dot(c_M[ok], c_M[ok]))
        err = abs(pred_slope - emp_slope)
        if err < best_err:
            best_err = err
            best_l = l_trial

    return float(best_l), emp_slope


# ─── Calibration table: derive missing ratio channels ─────────────────────────


def _add_ratio_channel(cal: pd.DataFrame, ratio_ch: str) -> pd.DataFrame:
    """
    Derive a ratio-channel row set if not present in cal.

    ratio_ch must be one of BdivG, RdivG, RdivB, GdivB.
    Numerator and denominator channels are looked up from _RATIO_PAIRS.
    The denominator G is proxied by G1 if plain G is absent.
    std is propagated as sqrt(var_num + var_den) assuming independence.
    """
    if ratio_ch not in _RATIO_PAIRS:
        raise ValueError(f"Unknown ratio channel '{ratio_ch}'")
    if ratio_ch in cal["channel"].unique():
        return cal

    num_ch, den_ch = _RATIO_PAIRS[ratio_ch]

    # resolve G → G1 if G absent
    available = set(cal["channel"].unique())
    if num_ch not in available:
        alt = num_ch + "1"
        if alt in available:
            num_ch = alt
        else:
            print(f"  Warning: cannot derive {ratio_ch} — {num_ch} not in cal table")
            return cal
    if den_ch not in available:
        alt = den_ch + "1"
        if alt in available:
            den_ch = alt
        else:
            print(f"  Warning: cannot derive {ratio_ch} — {den_ch} not in cal table")
            return cal

    num = cal[cal["channel"] == num_ch].set_index("concentration_um")
    den = cal[cal["channel"] == den_ch].set_index("concentration_um")
    common = num.index.intersection(den.index)
    if common.empty:
        print(f"  Warning: no overlapping concentrations for {ratio_ch}")
        return cal

    rows = []
    for c in common:
        A_n = num.loc[c, "absorbance_mean"]
        A_d = den.loc[c, "absorbance_mean"]
        s_n = num.loc[c, "absorbance_std"]
        s_d = den.loc[c, "absorbance_std"]
        row = {
            "concentration_um": c,
            "channel": ratio_ch,
            "absorbance_mean": A_n - A_d,
            "absorbance_std": float(np.sqrt(s_n**2 + s_d**2)),
        }
        for extra_col in ("n_images", "n_replicates", "absorbance_sem"):
            if extra_col in num.columns:
                row[extra_col] = num.loc[c, extra_col]
        rows.append(row)

    new_rows = pd.DataFrame(rows)
    return pd.concat([cal, new_rows], ignore_index=True)


# ─── Figures ──────────────────────────────────────────────────────────────────


def _ax_style(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", top=True, right=True, which="both")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def plot_spectra(
    phi_lam: np.ndarray,
    phi: np.ndarray,
    wl_S: np.ndarray,
    S: np.ndarray,
    eps_lam: np.ndarray,
    eps: np.ndarray,
    eps_std: np.ndarray,
    out_path: Path,
    cfg: AnalyteConfig,
    led_pump_wl: float | None = None,
    illuminant_label: str = "Illuminant spectrum",
) -> None:
    """Figure 1 — three-panel spectral overview: φ(λ), T_ch(λ), ε(λ)."""
    X_MIN, X_MAX = 400, 700
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    pc = CHANNEL_COLORS.get(cfg.primary_ch, "#333")

    ax = axes[0]
    vis = (phi_lam >= X_MIN) & (phi_lam <= X_MAX)
    ax.fill_between(phi_lam[vis], phi[vis], alpha=0.18, color="#555")
    ax.plot(phi_lam[vis], phi[vis], color="#333", lw=1.4)
    ax.axvline(
        cfg.lambda_max,
        color=pc,
        lw=1.0,
        ls="--",
        label=f"{cfg.lambda_max:.0f} nm ({cfg.analyte} λmax)",
    )
    if led_pump_wl is not None:
        ax.axvline(
            led_pump_wl,
            color="#8e44ad",
            lw=0.8,
            ls=":",
            alpha=0.8,
            label=f"~{led_pump_wl:.0f} nm (LED pump peak)",
        )
    ax.set_ylabel("φ(λ)  [a.u.]", fontsize=9)
    ax.set_title(f"A  —  {illuminant_label}", loc="left", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    _ax_style(ax)

    ax = axes[1]
    for i, (ch, col) in enumerate(
        zip(
            ["R", "G", "B"],
            [CHANNEL_COLORS["R"], CHANNEL_COLORS["G"], CHANNEL_COLORS["B"]],
        )
    ):
        ax.plot(wl_S, S[:, i], color=col, lw=2.0, label=ch)
    ax.axvline(cfg.lambda_max, color=pc, lw=1.0, ls="--")
    ax.set_ylabel("T$_{ch}$(λ)  [normalised]", fontsize=9)
    ax.set_title(
        "B  —  Bayer spectral sensitivity (Ohlab database)",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    _ax_style(ax)

    ax = axes[2]
    vis_uv = (eps_lam >= X_MIN) & (eps_lam <= X_MAX)
    if np.any(eps_std[vis_uv] > 0):
        ax.fill_between(
            eps_lam[vis_uv],
            (eps - eps_std)[vis_uv],
            (eps + eps_std)[vis_uv],
            alpha=0.25,
            color="#555",
        )
    ax.plot(eps_lam[vis_uv], eps[vis_uv], color="#333", lw=1.4)
    ax.axvline(cfg.lambda_max, color=pc, lw=1.0, ls="--", label=f"λmax = {cfg.lambda_max:.0f} nm")
    ax.set_ylabel("ε(λ)  [M$^{-1}$ cm$^{-1}$]", fontsize=9)
    ax.set_xlabel("Wavelength (nm)", fontsize=9)
    ax.set_title(
        f"C  —  {cfg.analyte} molar absorptivity (UV-Vis)",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    _ax_style(ax)

    axes[0].set_xlim(X_MIN, X_MAX)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_components(
    phi_g: np.ndarray,
    T_g: np.ndarray,
    eps_g: np.ndarray,
    l_cam: float,
    out_path: Path,
    cfg: AnalyteConfig,
    concs_um: tuple[float, ...] | None = None,
) -> None:
    """
    Figure 2 — integrand φ·T_ch·trans at several concentrations.

    Left panel: primary channel. Right panel: G (reference/secondary).
    Shows how the λmax peak is progressively extinguished while the
    transparent tail of the primary channel remains — mechanistic source
    of the two-regime Beer-Lambert behaviour.
    """
    if concs_um is None:
        concs_um = (0, 125, 500, 2000, 8000)

    # determine which Bayer column index maps to primary_ch
    ch_pairs = [(cfg.primary_ch, _CH_IDX.get(cfg.primary_ch, 2)), ("G", 1)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.cm.Blues_r

    for ax_idx, (ax, (ch, ch_idx)) in enumerate(zip(axes, ch_pairs)):
        T = T_g[:, ch_idx]
        norms = np.linspace(0.15, 0.85, len(concs_um))
        blank_peak = (phi_g * T).max()

        for k, c_um in enumerate(concs_um):
            c_M = c_um * 1e-6
            trans = np.exp(-eps_g * (c_M * l_cam) * LN10)
            integ = phi_g * T * trans
            label = f"c = {c_um:.0f} µM" if c_um > 0 else "c = 0 (blank)"
            ax.plot(
                _GRID,
                integ / blank_peak if blank_peak > 0 else integ,
                color=cmap(norms[k]),
                lw=1.5,
                label=label,
            )

        ax.axvline(
            cfg.lambda_max,
            color="#e74c3c",
            lw=0.9,
            ls="--",
            alpha=0.7,
            label=f"{cfg.lambda_max:.0f} nm",
        )
        ax.axvline(
            cfg.tail_wl, color="#8e44ad", lw=0.9, ls=":", alpha=0.7, label=f"{cfg.tail_wl:.0f} nm"
        )
        ax.set_xlabel("Wavelength (nm)", fontsize=9)
        ax.set_ylabel(f"φ·T$_{{{ch}}}$·trans  [rel. to blank]", fontsize=9)
        panel = "A" if ax_idx == 0 else "B"
        ax.set_title(
            f"{panel}  —  {ch}-channel integrand vs concentration",
            loc="left",
            fontsize=9,
            fontweight="bold",
        )
        ax.legend(fontsize=7.5, ncol=2)
        _ax_style(ax)

    fig.suptitle(
        f"Beer-Lambert-Bayer integrand decomposition  (l$_{{cam}}$ = {l_cam:.3f} cm)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_calibration(
    cal: pd.DataFrame,
    c_model_M: np.ndarray,
    pred: dict[str, np.ndarray],
    l_cam: float,
    out_path: Path,
    cfg: AnalyteConfig,
) -> None:
    """
    Figure 3 — predicted vs empirical calibration curves.
    Left: primary channel.  Right: ratio channel.  Log-scale x-axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    c_model_um = c_model_M * 1e6

    for ax, ch in zip(axes, [cfg.primary_ch, cfg.ratio_ch]):
        col = CHANNEL_COLORS.get(ch, "#555")
        # match G proxy (G1 ≈ G)
        cal_ch = ch
        if cal_ch not in cal["channel"].unique() and cal_ch + "1" in cal["channel"].unique():
            cal_ch = cal_ch + "1"
        emp = cal[cal["channel"] == cal_ch].sort_values("concentration_um")
        model_ch = ch if ch in pred else cfg.primary_ch

        ax.errorbar(
            emp["concentration_um"].to_numpy(dtype=float),
            emp["absorbance_mean"].to_numpy(dtype=float),
            yerr=emp["absorbance_std"].to_numpy(dtype=float),
            fmt="o",
            color=col,
            markersize=5,
            capsize=3,
            lw=1.2,
            zorder=3,
            label="Empirical (mean ± SD, n = 3)",
        )
        valid = np.isfinite(pred[model_ch])
        ax.plot(
            c_model_um[valid],
            pred[model_ch][valid],
            color=col,
            lw=2.0,
            ls="-",
            alpha=0.85,
            label=f"Forward model  (l$_{{cam}}$ = {l_cam:.3f} cm)",
        )
        ax.set_xscale("log")
        ax.set_xlabel(f"[{cfg.formula}] (µM)", fontsize=9)
        ax.set_ylabel(f"A$_{{{_ch_label(ch)}}}$", fontsize=9)
        ax.set_title(_ch_label(ch), fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        _ax_style(ax)

    fig.suptitle(
        f"Beer-Lambert-Bayer forward model vs empirical calibration  —  {cfg.analyte}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_tail_fraction(
    phi_g: np.ndarray,
    T_g: np.ndarray,
    eps_g: np.ndarray,
    l_cam: float,
    out_path: Path,
    cfg: AnalyteConfig,
    break_um: float | None = None,
) -> None:
    """
    Figure 4 — fraction of primary/reference channel signal from the analyte-
    transparent spectral tail vs [analyte].

    For analytes absorbing at λmax < tail_wl (e.g., ferricyanide at 420 nm,
    tail_wl = 473 nm): tail = λ > tail_wl.
    For analytes absorbing at λmax > tail_wl (e.g., MB at 665 + 610 nm,
    tail_wl = 426 nm — eps falls below 1% of its 665 nm peak below this
    wavelength; do NOT use 620 nm, which places the 610 nm dimer shoulder
    outside the absorbing region and spuriously inflates f_tail to ~63-76%,
    see _ai_outputs/analysis_notes/tail_boundary_1pc_rule_2026-09-21.md): tail = λ < tail_wl.
    """
    c_arr = np.logspace(np.log10(1e-8), np.log10(1e-2), 400)
    frac_primary = np.empty(len(c_arr))
    frac_ref = np.empty(len(c_arr))

    # transparent tail direction
    if cfg.tail_wl >= cfg.lambda_max:
        mask_tail = _GRID > cfg.tail_wl
        tail_label = f"λ > {cfg.tail_wl:.0f} nm"
    else:
        mask_tail = _GRID < cfg.tail_wl
        tail_label = f"λ < {cfg.tail_wl:.0f} nm"

    primary_idx = _CH_IDX.get(cfg.primary_ch, 2)
    ref_idx = 1  # G always as reference

    for j, c in enumerate(c_arr):
        trans = np.exp(-eps_g * (c * l_cam) * LN10)
        for arr, ch_idx in [(frac_primary, primary_idx), (frac_ref, ref_idx)]:
            T = T_g[:, ch_idx]
            integ = phi_g * T * trans
            I_tot = float(np.trapezoid(integ, dx=1.0))
            I_tail, _ = split_at_cutoff(integ, mask_tail)
            arr[j] = (I_tail / I_tot * 100) if I_tot > 0 else np.nan

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    c_um = c_arr * 1e6
    pc = CHANNEL_COLORS.get(cfg.primary_ch, "#555")
    ax.semilogx(
        c_um,
        frac_primary,
        color=pc,
        lw=2.0,
        label=f"{cfg.primary_ch} channel: fraction from {tail_label}",
    )
    ax.semilogx(
        c_um,
        frac_ref,
        color=CHANNEL_COLORS["G"],
        lw=1.8,
        ls="--",
        label=f"G channel: fraction from {tail_label}",
    )
    ax.axhline(50, color="#aaa", lw=0.8, ls=":", zorder=0)
    if break_um is not None:
        ax.axvline(
            break_um,
            color="#e74c3c",
            lw=1.0,
            ls="--",
            alpha=0.75,
            label=f"~{break_um:.0f} µM  (observed two-regime break)",
        )
    ax.set_xlabel(f"[{cfg.formula}] (µM)", fontsize=9)
    ax.set_ylabel(f"Channel signal from {tail_label} (%)", fontsize=9)
    ax.set_title(
        f"Spectral origin of the two-regime {cfg.primary_ch}-channel response\n"
        f"Fraction of channel signal from the {cfg.analyte}-transparent tail ({tail_label})",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    _ax_style(ax)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_correction(
    cal: pd.DataFrame,
    c_model_M: np.ndarray,
    pred: dict[str, np.ndarray],
    out_path: Path,
    cfg: AnalyteConfig,
    linear_max_um: float = 500.0,
) -> None:
    """
    Figure 5 — model inversion vs linear calibration: retrieved vs true concentration.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    c_model_um = c_model_M * 1e6

    for ax, ch in zip(axes, [cfg.primary_ch, cfg.ratio_ch]):
        col = CHANNEL_COLORS.get(ch, "#555")
        cal_ch = ch
        if cal_ch not in cal["channel"].unique() and cal_ch + "1" in cal["channel"].unique():
            cal_ch = cal_ch + "1"
        model_ch = ch if ch in pred else cfg.primary_ch
        emp = cal[cal["channel"] == cal_ch].sort_values("concentration_um")
        c_true = emp["concentration_um"].to_numpy(dtype=float)
        A_meas = emp["absorbance_mean"].to_numpy(dtype=float)

        lin_mask = c_true <= linear_max_um
        c_lin_M = c_true[lin_mask] * 1e-6
        A_lin = A_meas[lin_mask]
        slope_lin = float(np.dot(c_lin_M, A_lin) / np.dot(c_lin_M, c_lin_M))
        c_uncorr = A_meas / slope_lin * 1e6

        A_model = pred[model_ch].copy()
        A_model = np.maximum.accumulate(A_model)
        c_corr = np.interp(A_meas, A_model, c_model_um)

        lims = np.array([c_true.min() * 0.4, c_true.max() * 2.5])
        ax.loglog(lims, lims, "k--", lw=1.0, alpha=0.35, zorder=0, label="1:1  (ideal)")

        ax.errorbar(
            c_true,
            c_uncorr,
            fmt="s",
            color="#95a5a6",
            markersize=5,
            capsize=0,
            lw=0,
            zorder=2,
            label=f"Linear calibration  (≤ {linear_max_um:.0f} µM fit)",
        )
        ax.errorbar(
            c_true,
            c_corr,
            fmt="o",
            color=col,
            markersize=6,
            capsize=0,
            lw=0,
            zorder=3,
            label="Forward model inversion",
        )

        ax.set_xlabel("True concentration (µM)", fontsize=9)
        ax.set_ylabel("Retrieved concentration (µM)", fontsize=9)
        ax.set_title(
            f"A$_{{{_ch_label(ch)}}}$ — linear vs model-corrected retrieval",
            loc="left",
            fontsize=9,
            fontweight="bold",
        )
        ax.legend(fontsize=8)
        _ax_style(ax)

    fig.suptitle("Effect of spectral correction on concentration retrieval", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_linearisation(
    cal: pd.DataFrame,
    c_model_M: np.ndarray,
    pred: dict[str, np.ndarray],
    out_path: Path,
    cfg: AnalyteConfig,
    linear_max_um: float = 500.0,
) -> None:
    """Figure 6 — linearised calibration on linear axes."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    c_model_um = c_model_M * 1e6

    for ax, ch in zip(axes, [cfg.primary_ch, cfg.ratio_ch]):
        col = CHANNEL_COLORS.get(ch, "#555")
        cal_ch = ch
        if cal_ch not in cal["channel"].unique() and cal_ch + "1" in cal["channel"].unique():
            cal_ch = cal_ch + "1"
        model_ch = ch if ch in pred else cfg.primary_ch
        emp = cal[cal["channel"] == cal_ch].sort_values("concentration_um")
        c_true = emp["concentration_um"].to_numpy(dtype=float)
        A_meas = emp["absorbance_mean"].to_numpy(dtype=float)
        A_std = emp["absorbance_std"].to_numpy(dtype=float)

        lin_mask = c_true <= linear_max_um
        c_lin_M = c_true[lin_mask] * 1e-6
        A_lin = A_meas[lin_mask]
        slope_lin = float(np.dot(c_lin_M, A_lin) / np.dot(c_lin_M, c_lin_M))

        A_model = np.maximum.accumulate(pred[model_ch].copy())
        c_corr_um = np.interp(A_meas, A_model, c_model_um)
        A_corr = slope_lin * c_corr_um * 1e-6

        c_ref = np.array([0.0, c_true.max() * 1.08])
        ax.plot(
            c_ref,
            slope_lin * c_ref * 1e-6,
            color="#aaa",
            lw=1.2,
            ls="--",
            zorder=0,
            label="Linear Beer-Lambert  (low-c slope)",
        )

        ax.errorbar(
            c_true,
            A_meas,
            yerr=A_std,
            fmt="o",
            color="#95a5a6",
            markersize=5,
            capsize=3,
            lw=1.2,
            zorder=2,
            label="Original (measured A)",
        )
        ax.errorbar(
            c_true,
            A_corr,
            yerr=A_std,
            fmt="o",
            color=col,
            markersize=5,
            capsize=3,
            lw=1.2,
            zorder=3,
            label="Corrected (model inversion)",
        )

        ax.set_xlabel(f"[{cfg.formula}] (µM)", fontsize=9)
        ax.set_ylabel(f"A$_{{{_ch_label(ch)}}}$", fontsize=9)
        ax.set_title(
            f"A$_{{{_ch_label(ch)}}}$ — original vs model-corrected  (linear axes)",
            loc="left",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        _ax_style(ax)

    fig.suptitle(
        "Linearisation of the Beer-Lambert calibration by forward model correction",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── Summary markdown ─────────────────────────────────────────────────────────


def write_summary(
    out_path: Path,
    l_cam: float,
    emp_slope_M: float,
    fit_channel: str,
    fit_min_um: float,
    fit_max_um: float,
    device: str,
    uvvis_paths: list[Path],
    uvvis_pathlength_cm: float,
    illuminant_path: Path,
    phi_g: np.ndarray,
    T_g: np.ndarray,
    eps_g: np.ndarray,
    cfg: AnalyteConfig,
) -> None:
    """Write the forward-model summary markdown (effective wavelengths, slopes, fit range)."""
    # determine tail mask for channel integral reporting
    if cfg.tail_wl >= cfg.lambda_max:
        mask_tail = _GRID > cfg.tail_wl
        tail_label = f"> {cfg.tail_wl:.0f} nm"
    else:
        mask_tail = _GRID < cfg.tail_wl
        tail_label = f"< {cfg.tail_wl:.0f} nm"

    rows = []
    for i, ch in enumerate(("R", "G", "B")):
        T = T_g[:, i]
        I0 = float(np.trapezoid(phi_g * T, dx=1.0))
        lam_eff = float(np.trapezoid(phi_g * T * _GRID, dx=1.0) / I0) if I0 > 0 else np.nan
        I_tail, _ = split_at_cutoff(phi_g * T, mask_tail)
        frac = I_tail / I0 * 100 if I0 > 0 else np.nan
        eps_eff = float(np.trapezoid(phi_g * T * eps_g, dx=1.0) / I0) if I0 > 0 else np.nan
        rows.append((ch, lam_eff, frac, eps_eff))

    emp_slope_um = emp_slope_M * 1e-6
    from datetime import date

    today = date.today().strftime("%Y-%m-%d")

    lines = [
        "# Forward Model Summary",
        "",
        f"*Generated {today}. Camera: {device} (Ohlab database). Analyte: {cfg.analyte}.*",
        "",
        "## Fitted parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| l_cam (fitted) | **{l_cam:.4f} cm**  ({l_cam * 10:.2f} mm) |",
        f"| UV-Vis path length | {uvvis_pathlength_cm:.3g} cm  ({uvvis_pathlength_cm * 10:.2g} mm) |",
        f"| Fit channel | {fit_channel} |",
        f"| Fit range | {fit_min_um:.2g}–{fit_max_um:.4g} µM |",
        f"| Empirical {fit_channel} slope | {emp_slope_um:.3e} µM⁻¹  ({emp_slope_M:.2f} M⁻¹) |",
        "",
        f"> ε is computed as ε_l / {uvvis_pathlength_cm:.3g} cm; only the product ε·l_cam is",
        "> constrained by the fit, so an error in the UV-Vis path length rescales l_cam",
        "> inversely. l_cam is therefore conditional on this path length, not an independent",
        "> measurement of the camera geometry.",
        "",
        "## Channel integrals at c = 0 (blank)",
        "",
        f"| Channel | λ_eff (nm) | Tail fraction λ {tail_label} (%) | ε_eff (M⁻¹ cm⁻¹) |",
        "|---------|-----------|------------------------------|-------------------|",
    ]
    for ch, lam_eff, frac, eps_eff in rows:
        lines.append(f"| {ch} | {lam_eff:.1f} | {frac:.1f} | {eps_eff:.1f} |")

    # mechanistic note: primary channel transparent tail
    pc = cfg.primary_ch
    lines += [
        "",
        f"The {pc}-channel tail fraction (λ {tail_label}) represents signal from wavelengths",
        f"where {cfg.analyte} does not absorb.  This fraction grows to dominate at high",
        "concentration, producing the two-regime calibration curve.",
        "",
        "## Inputs",
        "",
        f"- Illuminant: `{illuminant_path.name}`",
        f"- UV-Vis replicates: {len(uvvis_paths)}",
    ]
    for p in uvvis_paths:
        lines.append(f"  - `{p.name}`")
    lines += [
        "",
        "## Methods note",
        "",
        "ε(λ): force-through-origin fit of blank-subtracted absorbance vs c (≤ 1 mM).",
        "Integration grid: 400–700 nm, 1 nm steps (numpy.trapz).",
        "l_cam: log-space grid search 0.005–20 cm (600 pts), minimises |slope_pred − slope_emp|.",
        "",
        "Cite Bayer sensitivity data:",
        "  Tominaga, Nishi, Ohtera, Sensors 2021, 21(15), 4985. doi:10.3390/s21154985",
        "  (corrected 2026-07-01: previously misattributed to 'Jiang et al.'; verified via CrossRef)",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {out_path.name}")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Beer-Lambert-Bayer forward model for smartphone colorimetry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--project", required=True, type=Path, help="Project directory (contains image_analysis/)"
    )
    p.add_argument(
        "--illuminant",
        required=True,
        type=Path,
        help="USB spectrometer CSV: wavelength, dark, LED light",
    )
    p.add_argument(
        "--uvvis",
        required=True,
        type=Path,
        action="append",
        metavar="CSV",
        help="UV-Vis CSV (repeat for multiple replicates)",
    )
    p.add_argument(
        "--uvvis-pathlength",
        type=float,
        default=0.2,
        help="UV-Vis cuvette path length in cm (default 0.2 cm = 2.0 mm cuvette). "
        "Must match the cuvette actually used; ε ∝ 1/pathlength and l_cam "
        "absorbs any error in this value (only the product ε·l_cam is fitted).",
    )
    p.add_argument(
        "--uvvis-min-conc",
        type=float,
        default=None,
        metavar="UM",
        help="Lower bound (µM) of the ε-fit concentration window. Default: no "
        "bound, i.e. the full measured range, which is what ferricyanide "
        "uses. Methylene blue needs --uvvis-min-conc 1.525 "
        "--uvvis-max-conc 50 to land inside the 70,000–95,000 M⁻¹ cm⁻¹ "
        "literature range for the monomer; a full-range MB fit gives "
        "63,457 (CLAUDE.md, mb_epsilon_reversal_2026-07-28.md).",
    )
    p.add_argument(
        "--uvvis-max-conc",
        type=float,
        default=None,
        metavar="UM",
        help="Upper bound (µM) of the ε-fit concentration window. Default: no "
        "bound. This replaced a hardcoded 1000 µM default that silently "
        "truncated ferricyanide at 1 mM and returned ε(420) = 981 instead "
        "of the canonical 1017.",
    )
    p.add_argument(
        "--uvvis-exclude-column",
        type=str,
        action="append",
        default=[],
        metavar="HEADER",
        help="Drop a UV-Vis column by exact header name (repeatable). Needed "
        "for MB_7.812uM in Set_1_MB_UV_VIS_25052026.csv, which CLAUDE.md "
        "records as really 0.7812 µM: the label puts it inside the "
        "1.525–50 µM window, so no concentration bound can remove it, and "
        "leaving it in returns ε(665) = 75,135 instead of 74,168.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="average_all",
        help="Ohlab camera key (see core/spectral.py list_devices())",
    )

    # analyte configuration
    p.add_argument(
        "--analyte",
        type=str,
        default="analyte",
        help="Analyte name used in plot titles and summary",
    )
    p.add_argument(
        "--analyte-formula",
        type=str,
        default=None,
        help="Chemical formula/abbreviation for axis labels (default: --analyte)",
    )
    p.add_argument(
        "--primary-channel", type=str, default="B", help="Primary Bayer signal channel (R|G|B)"
    )
    p.add_argument(
        "--ratio-channel",
        type=str,
        default="BdivG",
        help="Ratio channel for second panel (BdivG|RdivG|RdivB|GdivB)",
    )
    p.add_argument(
        "--lambda-max",
        type=float,
        default=420.0,
        help="Analyte λmax (nm) for vertical reference lines",
    )
    p.add_argument(
        "--tail-wavelength",
        type=float,
        default=473.0,
        help=(
            "Boundary wavelength (nm) for transparent-tail fraction. "
            "If > lambda_max, tail = λ > tail_wl (ferricyanide default, 473 nm). "
            "If < lambda_max, tail = λ < tail_wl (MB: use 426 nm — eps falls "
            "below 1%% of its 665 nm peak below this point; 620 nm, 575 nm "
            "and 490 nm were tried and rejected, see "
            "_ai_outputs/analysis_notes/tail_boundary_1pc_rule_2026-09-21.md)."
        ),
    )

    # fit configuration
    p.add_argument("--fit-channel", type=str, default="BdivG", help="Channel used to fit l_cam")
    p.add_argument(
        "--fit-min-conc", type=float, default=62.5, help="Lower bound (µM) for l_cam slope fitting"
    )
    p.add_argument(
        "--fit-max-conc",
        type=float,
        default=1000.0,
        help="Upper bound (µM) for l_cam slope fitting",
    )
    p.add_argument(
        "--n-conc",
        type=int,
        default=400,
        help="Points on the model concentration grid (10 nM – 10 mM)",
    )
    p.add_argument(
        "--out-subdir",
        type=str,
        default="forward_model",
        help="Output subfolder inside image_analysis/ (default: forward_model)",
    )
    p.add_argument(
        "--l-cam",
        type=float,
        default=None,
        help="Fix optical path length (cm) and skip l_cam fitting. "
        "Use when you want to study illuminant effects at a known geometry.",
    )
    p.add_argument(
        "--led-pump-wl",
        type=float,
        default=None,
        help="Draw a vertical reference at this wavelength (nm) in Figure 1 "
        "to mark the LED pump peak (e.g. 450). Omit for standard illuminants.",
    )
    p.add_argument(
        "--illuminant-label",
        type=str,
        default=None,
        help="Panel-A title for Figure 1 (default: auto from filename)",
    )
    return p.parse_args()


def main() -> None:
    """Run the spectral forward model: predict channel responses and calibration slopes."""
    args = _parse_args()

    cfg = AnalyteConfig(
        analyte=args.analyte,
        formula=args.analyte_formula if args.analyte_formula else args.analyte,
        primary_ch=args.primary_channel,
        ratio_ch=args.ratio_channel,
        lambda_max=args.lambda_max,
        tail_wl=args.tail_wavelength,
    )

    out_dir = args.project / "image_analysis" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\nAnalyte: {cfg.analyte}  |  Primary channel: {cfg.primary_ch}  |  λmax: {cfg.lambda_max:.0f} nm"
    )
    print(f"Output: {out_dir}\n")

    illum_label = args.illuminant_label or f"Illuminant spectrum ({args.illuminant.stem})"

    print("Loading illuminant …")
    phi_lam, phi = load_illuminant(args.illuminant)

    print(f"Loading UV-Vis ({len(args.uvvis)} replicate(s)) …")
    eps_lam, eps, eps_std = load_uvvis(
        args.uvvis,
        args.uvvis_pathlength,
        min_fit_conc_um=args.uvvis_min_conc,
        max_fit_conc_um=args.uvvis_max_conc,
        exclude_columns=tuple(args.uvvis_exclude_column),
    )

    print(f"Loading Bayer sensitivity: {args.device} …")
    wl_S, S = load_sensitivity(args.device)

    phi_g = _to_grid(phi_lam, phi)
    eps_g = _to_grid(eps_lam, eps)
    T_g = _sens_to_grid(wl_S, S)

    cal_path = args.project / "image_analysis" / "s02_calibrate" / "calibration_table.csv"
    if not cal_path.exists():  # legacy layout (pre-s02_calibrate)
        cal_path = args.project / "image_analysis" / "beer_lambert" / "calibration_table.csv"
    if not cal_path.exists():  # flat layout: beer_lambert/ directly under project
        cal_path = args.project / "beer_lambert" / "calibration_table.csv"
    if not cal_path.exists():
        sys.exit(
            "ERROR: calibration_table.csv not found under image_analysis/s02_calibrate/, image_analysis/beer_lambert/, or beer_lambert/"
        )
    cal = pd.read_csv(cal_path)

    # derive ratio channel if missing
    cal = _add_ratio_channel(cal, cfg.ratio_ch)

    if args.l_cam is not None:
        l_cam = args.l_cam
        emp_slope_M = np.nan
        print(f"  l_cam fixed at {l_cam:.4f} cm  ({l_cam * 10:.2f} mm)  [no fit]\n")
    else:
        print(
            f"Fitting l_cam to {args.fit_channel} slope ({args.fit_min_conc:.3g}–{args.fit_max_conc:.4g} µM) …"
        )
        l_cam, emp_slope_M = fit_l_cam(
            cal,
            phi_g,
            T_g,
            eps_g,
            args.fit_channel,
            args.fit_max_conc,
            args.fit_min_conc,
        )
        print(f"  l_cam = {l_cam:.4f} cm  ({l_cam * 10:.2f} mm)\n")

    c_model_M = np.logspace(np.log10(1e-8), np.log10(1e-2), args.n_conc)
    pred = predict_absorbance(c_model_M, phi_g, T_g, eps_g, l_cam)

    # auto-derive component plot concentrations spanning 0 to 2× fit_max
    _concs = (0.0,) + tuple(
        float(x)
        for x in np.logspace(
            np.log10(max(args.fit_min_conc, 0.1)), np.log10(args.fit_max_conc * 2), 4
        )
    )

    print("Generating figures …")
    plot_spectra(
        phi_lam,
        phi,
        wl_S,
        S,
        eps_lam,
        eps,
        eps_std,
        out_dir / "01_forward_model_spectra.png",
        cfg,
        led_pump_wl=args.led_pump_wl,
        illuminant_label=illum_label,
    )
    plot_components(
        phi_g, T_g, eps_g, l_cam, out_dir / "02_forward_model_components.png", cfg, concs_um=_concs
    )
    plot_calibration(cal, c_model_M, pred, l_cam, out_dir / "03_forward_model_calibration.png", cfg)
    plot_tail_fraction(
        phi_g, T_g, eps_g, l_cam, out_dir / "04_forward_model_tail_fraction.png", cfg
    )
    plot_correction(
        cal,
        c_model_M,
        pred,
        out_dir / "05_forward_model_correction.png",
        cfg,
        linear_max_um=args.fit_max_conc,
    )
    plot_linearisation(
        cal,
        c_model_M,
        pred,
        out_dir / "06_forward_model_linearisation.png",
        cfg,
        linear_max_um=args.fit_max_conc,
    )

    tbl_path = out_dir / "forward_model_table.csv"
    pd.DataFrame(
        {
            "concentration_M": c_model_M,
            "concentration_um": c_model_M * 1e6,
            "A_R": pred["R"],
            "A_G": pred["G"],
            "A_B": pred["B"],
            "A_BdivG": pred["BdivG"],
            "A_BdivR": pred["BdivR"],
            "A_RdivG": pred["RdivG"],
            "A_RdivB": pred["RdivB"],
            "A_GdivB": pred["GdivB"],
        }
    ).to_csv(str(tbl_path), index=False, float_format="%.6g")
    print(f"  Saved: {tbl_path.name}")

    write_summary(
        out_dir / "forward_model_summary.md",
        l_cam,
        emp_slope_M,
        args.fit_channel,
        args.fit_min_conc,
        args.fit_max_conc,
        args.device,
        args.uvvis,
        args.uvvis_pathlength,
        args.illuminant,
        phi_g,
        T_g,
        eps_g,
        cfg,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
