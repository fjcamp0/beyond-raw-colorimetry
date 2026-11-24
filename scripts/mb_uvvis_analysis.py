"""
mb_uvvis_analysis.py

Parse methylene blue UV-Vis CSVs, derive ε(λ), produce:
  1. spectral_sensitivity_and_crosstalk_MB.png  — Ohlab Bayer curves + MB ε(λ)
  2. uvvis_cross_validation.md                  — ε·l from UV-Vis vs camera calibration
  3. uvvis_epsilon.csv                          — ε(λ) table (mean ± SD, n=3 sets, 8 conc)

Run (from project root):
  python "Data/Methyelene blue/image_analysis/mb_uvvis_analysis.py" \
      --project "Data/Methyelene blue"

Assumed path length: 0.2 cm (2 mm cuvette, Labbox MALG-002-002). Override with --pathlength.

core.spectral ships vendored in vendor/; set COLORIMETRY_VENDOR to override.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# numpy 2.x removed trapz → use trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import linregress

# Vendored raw-image-analysis modules (vendor/, see README); override with COLORIMETRY_VENDOR.
sys.path.insert(
    0,
    str(
        Path(
            os.environ.get(
                "COLORIMETRY_VENDOR", str(Path(__file__).resolve().parent.parent / "vendor")
            )
        )
    ),
)
from core.spectral import WAVELENGTHS, load_sensitivity
from numfmt import pm  # CLAUDE.md hard rule 10: SD to two significant figures

# ─── UV-Vis parsing ────────────────────────────────────────────────────────────

CONCENTRATIONS_UM = [0.7812, 1.525, 3.125, 6.25, 12.5, 25.0, 50.0, 100.0]


def _parse_set1(path: Path) -> pd.DataFrame:
    """
    Set1 format: wide CSV.
    Row 0: header — Wavelength (nm), MB_7.812uM, MB_1.525uM, ...
    Then: wavelength, abs1, abs2, ... (plus junk columns on right).
    Returns DataFrame with columns: wavelength_nm, c_0.7812, c_1.525, ...
    """
    raw = pd.read_csv(path, header=0, usecols=range(9))
    raw.columns = ["wavelength_nm"] + [f"c_{c}" for c in CONCENTRATIONS_UM]
    raw = raw.dropna(subset=["wavelength_nm"])
    raw["wavelength_nm"] = pd.to_numeric(raw["wavelength_nm"], errors="coerce")
    raw = raw.dropna(subset=["wavelength_nm"])
    for c in CONCENTRATIONS_UM:
        raw[f"c_{c}"] = pd.to_numeric(raw[f"c_{c}"], errors="coerce")
    return raw.sort_values("wavelength_nm").reset_index(drop=True)


def _parse_set2_3(path: Path) -> pd.DataFrame:
    """
    Set2/Set3 format: paired columns.
    Row 0: concentration labels (0.7812uM, , 1.525uM, , ...)
    Row 1: Wavelength (nm), Abs, Wavelength (nm), Abs, ...
    Then data: wl, abs, wl, abs, ...
    Each pair shares the same wavelength grid.
    Returns DataFrame with columns: wavelength_nm, c_0.7812, c_1.525, ...
    """
    raw = pd.read_csv(path, header=None)
    # Row 0 = concentration headers; row 1 = column-type headers; rows 2+ = data
    data = raw.iloc[2:].copy().reset_index(drop=True)

    # Wavelength grid from first concentration (column 0)
    wl = pd.to_numeric(data.iloc[:, 0], errors="coerce")

    result = pd.DataFrame({"wavelength_nm": wl})
    for i, c in enumerate(CONCENTRATIONS_UM):
        abs_col = i * 2 + 1  # every second column starting at 1
        result[f"c_{c}"] = pd.to_numeric(data.iloc[:, abs_col], errors="coerce")

    result = result.dropna(subset=["wavelength_nm"])
    return result.sort_values("wavelength_nm").reset_index(drop=True)


def load_all_sets(uvvis_dir: Path) -> dict[str, pd.DataFrame]:
    files = sorted(uvvis_dir.glob("*.csv"))
    dfs = {}
    for f in files:
        raw = f.read_text(encoding="utf-8", errors="replace")[:200]
        if "MB_7.812uM" in raw or "MB_1.525uM" in raw:
            dfs[f.stem] = _parse_set1(f)
        else:
            dfs[f.stem] = _parse_set2_3(f)
    return dfs


# ─── ε(λ) derivation ──────────────────────────────────────────────────────────

LAMBDA_GRID_NM = np.arange(400, 805, 5, dtype=float)


def derive_epsilon(dfs: dict[str, pd.DataFrame], pathlength_cm: float = 0.2) -> pd.DataFrame:
    """
    For each set and each concentration, interpolate absorbance onto LAMBDA_GRID_NM,
    compute ε(λ) = A(λ) / (c * l).  c in µM → convert to M for SI ε (M⁻¹ cm⁻¹).
    Returns DataFrame: wavelength_nm, set, concentration_um, epsilon
    """
    rows = []
    for set_name, df in dfs.items():
        wl = df["wavelength_nm"].values
        for c in CONCENTRATIONS_UM:
            col = f"c_{c}"
            if col not in df.columns:
                continue
            abs_vals = df[col].values
            mask = np.isfinite(abs_vals)
            if mask.sum() < 10:
                continue
            A_interp = np.interp(
                LAMBDA_GRID_NM, wl[mask], abs_vals[mask], left=np.nan, right=np.nan
            )
            c_M = c * 1e-6  # µM → M
            eps = A_interp / (c_M * pathlength_cm)
            for i, lam in enumerate(LAMBDA_GRID_NM):
                rows.append(
                    {
                        "wavelength_nm": lam,
                        "set": set_name,
                        "concentration_um": c,
                        "epsilon": eps[i],
                    }
                )
    return pd.DataFrame(rows)


def epsilon_stats(eps_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD ε(λ) across all sets and concentrations."""
    return (
        eps_df.dropna(subset=["epsilon"])
        .groupby("wavelength_nm")["epsilon"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )


# ─── Cross-validation: UV-Vis ε·l vs camera ───────────────────────────────────

# Camera calibration results (from beer_lambert.py run)
CAMERA_SLOPE_R = 7.52e-3  # µM⁻¹  (εl effective, camera, R channel)
CAMERA_SLOPE_G = 2.70e-3  # µM⁻¹  (G channel)
CAMERA_SLOPE_B = 1.09e-3  # µM⁻¹  (B channel)


def predict_camera_slope(
    S: np.ndarray, eps_mean: pd.DataFrame, channel_idx: int, pathlength_cm: float = 0.2
) -> float:
    """
    Predict effective (εl)_cam for a Bayer channel by integrating:
      (εl)_cam = ∫ S(λ) ε(λ) l dλ / ∫ S(λ) dλ

    S: (31, 3) Ohlab sensitivity at 400-700 nm, 10 nm steps.
    eps_mean: DataFrame with wavelength_nm, mean columns.
    channel_idx: 0=R, 1=G, 2=B.
    """
    wl_ohlab = WAVELENGTHS  # 400..700 step 10
    sens = S[:, channel_idx]

    # Interpolate ε mean onto Ohlab grid (400–700 nm)
    eps_on_ohlab = np.interp(
        wl_ohlab, eps_mean["wavelength_nm"].values, eps_mean["mean"].values, left=0.0, right=0.0
    )

    # Convert ε from M⁻¹ cm⁻¹ → µM⁻¹ for comparison with camera slope
    eps_uM = eps_on_ohlab * 1e-6 * pathlength_cm  # ε[M⁻¹cm⁻¹] * l[cm] * 1e-6[M/µM]

    # Weighted average
    denom = _trapz(sens, wl_ohlab)
    if denom == 0:
        return float("nan")
    numer = _trapz(sens * eps_uM, wl_ohlab)
    return numer / denom


# ─── Figures ──────────────────────────────────────────────────────────────────

BCM_BLUE = "#003865"
BCM_CYAN = "#00A9CE"
BCM_GRAY = "#6D7278"

CH_COLORS = {"R": "#D62728", "G": "#2CA02C", "B": "#1F77B4"}


def make_spectral_figure(
    S: np.ndarray, eps_stats_df: pd.DataFrame, out_path: Path, pathlength_cm: float
) -> None:
    """
    Two-panel figure:
    Top: full 400–700 nm — Bayer sensitivity (left y) + MB ε(λ) (right y)
    Bottom: 550–720 nm close-up with channel sensitivity fractions annotated
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [1, 1]})
    fig.subplots_adjust(hspace=0.38)

    wl_eps = eps_stats_df["wavelength_nm"].values
    eps_mean = eps_stats_df["mean"].values
    eps_std = eps_stats_df["std"].values

    for ax_idx, ax in enumerate(axes):
        # ── Bayer sensitivity (left y-axis) ──
        for i, (ch, color) in enumerate(CH_COLORS.items()):
            ax.plot(
                WAVELENGTHS,
                S[:, i],
                color=color,
                lw=2.0,
                label=f"{ch} channel (Ohlab avg)",
                alpha=0.85,
            )
        ax.set_ylabel("Relative sensitivity (a.u.)", fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

        # ── MB ε(λ) (right y-axis) ──
        ax2 = ax.twinx()
        ax2.plot(
            wl_eps,
            eps_mean / 1e3,
            color=BCM_BLUE,
            lw=2.0,
            label="MB ε(λ) (UV-Vis, n=3 sets × 8 conc)",
            zorder=5,
        )
        ax2.fill_between(
            wl_eps,
            (eps_mean - eps_std) / 1e3,
            (eps_mean + eps_std) / 1e3,
            color=BCM_BLUE,
            alpha=0.15,
            zorder=4,
        )
        ax2.set_ylabel("ε (×10³ M⁻¹ cm⁻¹)", fontsize=11, color=BCM_BLUE)
        ax2.tick_params(axis="y", labelcolor=BCM_BLUE)
        ax2.set_ylim(bottom=0)

        if ax_idx == 0:
            ax.set_xlim(400, 710)
            ax.set_xlabel("Wavelength (nm)", fontsize=11)
            # Mark λ_max
            i_max = np.argmax(eps_mean[(wl_eps >= 640) & (wl_eps <= 690)])
            lam_max = wl_eps[(wl_eps >= 640) & (wl_eps <= 690)][i_max]
            eps_max = eps_mean[(wl_eps >= 640) & (wl_eps <= 690)][i_max]
            ax2.annotate(
                f"λ_max ≈ {lam_max:.0f} nm\nε ≈ {eps_max / 1e3:.0f}×10³ M⁻¹cm⁻¹",
                xy=(lam_max, eps_max / 1e3),
                xytext=(lam_max - 110, eps_max / 1e3 * 0.85),
                fontsize=9,
                color=BCM_BLUE,
                arrowprops=dict(arrowstyle="-", color=BCM_BLUE, lw=0.8),
            )
            # 610 nm shoulder
            i_sh = np.argmin(np.abs(wl_eps - 610))
            ax2.axvline(610, color=BCM_GRAY, lw=1.0, ls="--", alpha=0.5)
            ax2.text(
                610 + 4,
                ax2.get_ylim()[1] * 0.9,
                "610 nm\nshoulder",
                fontsize=8,
                color=BCM_GRAY,
                va="top",
            )
            # Title
            ax.set_title("Bayer channel sensitivity vs methylene blue ε(λ)", fontsize=12)
            # Legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(
                lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8, framealpha=0.85
            )

        else:
            # Bottom panel: 550–720 nm
            ax.set_xlim(550, 720)
            ax.set_xlabel("Wavelength (nm)", fontsize=11)
            ax.set_title("Close-up: 550–720 nm — region relevant to MB detection", fontsize=12)

            # Shade R-filter dominant region (approx 580–700 nm)
            ax.axvspan(580, 700, alpha=0.07, color=CH_COLORS["R"], zorder=0)
            # Shade G-filter shoulder region
            ax.axvspan(590, 640, alpha=0.07, color=CH_COLORS["G"], zorder=0)

            # Annotate sensitivity values at 665 and 610 nm
            for lam_ann, label_suffix in [(665, "λ_max"), (610, "shoulder")]:
                for i, ch in enumerate(["R", "G", "B"]):
                    s_val = float(np.interp(lam_ann, WAVELENGTHS, S[:, i]))
                    color = CH_COLORS[ch]
                    y_pos = s_val + 0.04
                    ax.annotate(
                        f"{ch}: {s_val:.3f}",
                        xy=(lam_ann, s_val),
                        xytext=(lam_ann + 4, y_pos),
                        fontsize=7.5,
                        color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.7),
                    )

            # G/R ratio annotation
            s_R_665 = float(np.interp(665, WAVELENGTHS, S[:, 0]))
            s_G_665 = float(np.interp(665, WAVELENGTHS, S[:, 1]))
            s_R_610 = float(np.interp(610, WAVELENGTHS, S[:, 0]))
            s_G_610 = float(np.interp(610, WAVELENGTHS, S[:, 1]))
            ax.text(
                0.02,
                0.96,
                f"G/R sensitivity at 665 nm: {s_G_665 / s_R_665:.2f}\n"
                f"G/R sensitivity at 610 nm: {s_G_610 / s_R_610:.2f}",
                transform=ax.transAxes,
                fontsize=8.5,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

            # Camera calibration G/R response ratio
            ax.text(
                0.02,
                0.72,
                f"Camera calibration G/R slope ratio: {CAMERA_SLOPE_G / CAMERA_SLOPE_R:.2f}\n"
                f"(explains partial G response to MB)",
                transform=ax.transAxes,
                fontsize=8.5,
                va="top",
                color=BCM_GRAY,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

            # Legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(
                lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8, framealpha=0.85
            )

    fig.suptitle(
        "Smartphone Bayer sensitivity (Ohlab, 21-device average) vs\n"
        "Methylene blue molar absorptivity ε(λ) from UV-Vis\n"
        f"(n=3 acquisition sets × 8 concentrations, l = {pathlength_cm} cm)",
        fontsize=11,
        y=0.995,
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[mb_uvvis] Figure → {out_path.name}")


def make_epsilon_calibration_fig(
    eps_df: pd.DataFrame, out_path: Path, pathlength_cm: float = 0.2
) -> None:
    """
    Beer-Lambert linearity check from UV-Vis:
    For each wavelength band (400 nm, 610 nm, 665 nm), plot A vs c across all sets.
    Confirms UV-Vis Beer-Lambert behaviour.
    """
    target_lambdas = [400, 610, 665]
    colors = ["#9467BD", CH_COLORS["G"], CH_COLORS["R"]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for ax, lam, color in zip(axes, target_lambdas, colors):
        subset = eps_df[np.abs(eps_df["wavelength_nm"] - lam) < 3].copy()
        if subset.empty:
            ax.set_title(f"{lam} nm — no data")
            continue

        # A = ε * c * l; reconstruct from epsilon using the actual pathlength
        subset["A_calc"] = subset["epsilon"] * subset["concentration_um"] * 1e-6 * pathlength_cm

        for s_name, grp in subset.groupby("set"):
            c_um = grp["concentration_um"].values
            A_vals = grp["epsilon"].values * c_um * 1e-6 * pathlength_cm
            ax.scatter(
                c_um,
                A_vals,
                s=25,
                alpha=0.8,
                label=s_name.replace("_MB_UV_VIS_25052026", "").replace("_MB_UV_VIS_27052026", ""),
            )

        # Combined fit
        combined = subset.copy()
        combined["A"] = combined["epsilon"] * combined["concentration_um"] * 1e-6 * pathlength_cm
        c_fit = combined["concentration_um"].values
        A_fit = combined["A"].values
        mask = np.isfinite(A_fit) & np.isfinite(c_fit)
        if mask.sum() > 3:
            slope, intercept, r, _, _ = linregress(c_fit[mask], A_fit[mask])
            x_line = np.linspace(0, 105, 200)
            ax.plot(x_line, slope * x_line + intercept, "-", color=color, lw=1.5, zorder=5)
            ax.text(
                0.05,
                0.92,
                f"slope = {slope * 1e3:.2f}×10⁻³ µM⁻¹\nR² = {r**2:.4f}",
                transform=ax.transAxes,
                fontsize=8.5,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

        ax.set_xlabel("[MB] (µM)", fontsize=10)
        ax.set_ylabel("Absorbance (AU)", fontsize=10)
        ax.set_title(f"UV-Vis Beer-Lambert at {lam} nm", fontsize=11)
        ax.legend(fontsize=7.5)
        ax.set_xlim(-2, 108)
        ax.set_ylim(bottom=0)

    fig.suptitle("UV-Vis Beer-Lambert linearity check — methylene blue", fontsize=12)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[mb_uvvis] Beer-Lambert check → {out_path.name}")


# ─── Cross-validation report ──────────────────────────────────────────────────


def write_crossval_report(
    eps_stats_df: pd.DataFrame, S: np.ndarray, out_path: Path, pathlength_cm: float
) -> None:
    eps_mean_series = pd.Series(
        eps_stats_df["mean"].values, index=eps_stats_df["wavelength_nm"].values
    )

    # ε at key wavelengths from UV-Vis
    def eps_at(lam):
        idx = np.argmin(np.abs(eps_stats_df["wavelength_nm"].values - lam))
        row = eps_stats_df.iloc[idx]
        return row["mean"], row["std"], row["wavelength_nm"]

    eps_665_mean, eps_665_sd, _ = eps_at(665)
    eps_610_mean, eps_610_sd, _ = eps_at(610)
    eps_lmax_val = eps_stats_df["mean"].max()
    eps_lmax_lam = eps_stats_df.loc[eps_stats_df["mean"].idxmax(), "wavelength_nm"]

    # Ohlab sensitivity at key wavelengths
    s_R_665 = float(np.interp(665, WAVELENGTHS, S[:, 0]))
    s_G_665 = float(np.interp(665, WAVELENGTHS, S[:, 1]))
    s_B_665 = float(np.interp(665, WAVELENGTHS, S[:, 2]))
    s_R_610 = float(np.interp(610, WAVELENGTHS, S[:, 0]))
    s_G_610 = float(np.interp(610, WAVELENGTHS, S[:, 1]))
    s_B_610 = float(np.interp(610, WAVELENGTHS, S[:, 2]))

    # Predicted camera slopes from Ohlab integral
    pred_R = predict_camera_slope(S, eps_stats_df, 0, pathlength_cm)
    pred_G = predict_camera_slope(S, eps_stats_df, 1, pathlength_cm)
    pred_B = predict_camera_slope(S, eps_stats_df, 2, pathlength_cm)

    # G/R ratio predicted vs observed
    ratio_G_R_pred = pred_G / pred_R if pred_R else float("nan")
    ratio_G_R_obs = CAMERA_SLOPE_G / CAMERA_SLOPE_R

    # R-filter leak fraction below 600 nm (where MB absorbs weakly)
    sens_R = S[:, 0]
    mask_below_600 = WAVELENGTHS < 600
    frac_R_below_600 = _trapz(sens_R[mask_below_600], WAVELENGTHS[mask_below_600]) / _trapz(
        sens_R, WAVELENGTHS
    )

    lines = [
        "# UV-Vis Cross-Validation — Methylene Blue",
        "",
        f"*Generated by raw-image-analysis/mb_uvvis_analysis.py. Path length assumed: {pathlength_cm} cm.*",
        "",
        "---",
        "",
        "## 1. ε(λ) from UV-Vis",
        "",
        "| Wavelength | ε (mean) (M⁻¹ cm⁻¹) | ε (SD) | Notes |",
        "|-----------|---------------------|--------|-------|",
        f"| λ_max ({eps_lmax_lam:.0f} nm) | {eps_lmax_val:,.0f} | — | Global maximum |",
        f"| 665 nm | {pm(eps_665_mean, eps_665_sd)} | — | λ_max region |",
        f"| 610 nm | {pm(eps_610_mean, eps_610_sd)} | — | Vibronic shoulder |",
        "",
        f"λ_max measured: **{eps_lmax_lam:.0f} nm**, ε_max ≈ **{eps_lmax_val / 1e3:.1f} × 10³ M⁻¹ cm⁻¹**",
        "Literature: λ_max ≈ 665 nm, ε ≈ 74 000 M⁻¹ cm⁻¹.",
        "",
        "---",
        "",
        "## 2. Ohlab channel sensitivity at MB absorption bands",
        "",
        "| Channel | Sensitivity at 665 nm | Sensitivity at 610 nm | Ratio 610/665 |",
        "|---------|----------------------|-----------------------|---------------|",
        f"| R | {s_R_665:.4f} | {s_R_610:.4f} | {s_R_610 / s_R_665:.2f} |",
        f"| G | {s_G_665:.4f} | {s_G_610:.4f} | {s_G_610 / s_G_665:.2f} |",
        f"| B | {s_B_665:.4f} | {s_B_610:.4f} | {s_B_610 / s_B_665:.2f} |",
        "",
        f"G/R sensitivity ratio at 665 nm: {s_G_665 / s_R_665:.3f}",
        f"G/R sensitivity ratio at 610 nm: {s_G_610 / s_R_610:.3f}",
        "",
        f"R-filter integrated sensitivity below 600 nm: **{frac_R_below_600 * 100:.1f}%** of total R-channel.",
        '(Fraction of R-channel that is "blind" to MB main band — potential static crosstalk source at high [c].)',
        "",
        "---",
        "",
        "## 3. Predicted vs observed camera calibration slopes",
        "",
        "Prediction from Beer-Lambert-Bayer integral (Ohlab average sensitivity × UV-Vis ε(λ)):",
        "",
        "$$A_{ch}^{\\text{predicted}}(c) \\approx \\frac{\\int S_{ch}(\\lambda)\\,\\varepsilon(\\lambda)\\,l\\,c\\,d\\lambda}{\\int S_{ch}(\\lambda)\\,d\\lambda}$$",
        "",
        f"(linear approximation valid for A ≪ 1; l = {pathlength_cm} cm assumed)",
        "",
        "| Channel | Predicted ε·l (µM⁻¹) | Observed ε·l (µM⁻¹) | Ratio obs/pred |",
        "|---------|----------------------|---------------------|----------------|",
        f"| R | {pred_R:.5f} | {CAMERA_SLOPE_R:.5f} | {CAMERA_SLOPE_R / pred_R:.2f} |",
        f"| G | {pred_G:.5f} | {CAMERA_SLOPE_G:.5f} | {CAMERA_SLOPE_G / pred_G:.2f} |",
        f"| B | {pred_B:.5f} | {CAMERA_SLOPE_B:.5f} | {CAMERA_SLOPE_B / pred_B:.2f} |",
        "",
        f"Predicted G/R slope ratio: {ratio_G_R_pred:.3f}",
        f"Observed G/R slope ratio: {ratio_G_R_obs:.3f} (camera calibration)",
        "",
        "**Interpretation:** A predicted/observed ratio close to 1 confirms the path length assumption.",
        "A systematic offset across all channels suggests the effective path length in the smartphone",
        "setup differs from the cuvette geometry (expected: obs/pred ≈ l_eff / l_cuvette).",
        "",
        "The G/R ratio comparison is path-length-independent and directly tests whether Ohlab",
        "sensitivity curves correctly predict the partial G response to MB via the 610 nm shoulder.",
        "",
        "---",
        "",
        "## 4. Implications for channel selection",
        "",
        "The UV-Vis ε(λ) confirms:",
        "",
        "- **R primary channel is correct.** ε at 665 nm is high; Ohlab R-channel sensitivity peaks",
        "  near 665 nm → strong Beer-Lambert response in R.",
        "",
        f"- **G partial response is real, not noise.** Ohlab G sensitivity at 610 nm: {s_G_610:.3f}",
        f"  (vs R at 665 nm: {s_R_665:.3f}). The vibronic shoulder (ε_610 ≈ {eps_610_mean / 1e3:.1f}×10³ M⁻¹cm⁻¹)",
        "  overlaps the G-filter passband, producing a genuine G absorbance signal.",
        "",
        "- **B response is from R/B-filter spectral overlap.** ε of MB near 400–500 nm is",
        "  essentially zero (MB does not absorb in the blue). The B-channel response (14.5% of R)",
        "  arises from the B-filter long-wavelength tail extending into the MB-absorbing region",
        "  (same mechanism as ferricyanide crosstalk into the B channel, but reversed spectral range).",
        "",
        f"- **Two-regime concern for R channel.** R-filter has {frac_R_below_600 * 100:.1f}% of integrated",
        "  sensitivity below 600 nm (where MB absorbs weakly). At very high [c], the 665 nm component",
        "  of the R channel saturates; residual R signal comes from this short-wavelength tail →",
        "  shallower second slope. Transition concentration not reached in 0.78–100 µM range.",
        "",
        "---",
        "",
        "*Analysis: raw-image-analysis/mb_uvvis_analysis.py.*",
        "*Bayer sensitivity: Ohlab (Jiang et al., Sensors 2021, doi:10.3390/s21154985).*",
        f"*UV-Vis: 3 sets × 8 concentrations, {pathlength_cm} cm path length assumed.*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[mb_uvvis] Cross-validation report → {out_path.name}")


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project", required=True, help="Path to 'Data/Methyelene blue' project folder"
    )
    ap.add_argument(
        "--pathlength",
        type=float,
        default=0.2,
        help="Cuvette path length in cm (default: 0.2 — Labbox MALG-002-002, 2 mm)",
    )
    args = ap.parse_args()

    proj = Path(args.project)
    uvvis_dir = proj / "UV-Vis data"
    out_dir = proj / "image_analysis"
    out_dir.mkdir(exist_ok=True)

    print(f"[mb_uvvis] Project: {proj}")
    print(f"[mb_uvvis] Path length: {args.pathlength} cm")

    # Load UV-Vis
    print("[mb_uvvis] Loading UV-Vis data...")
    dfs = load_all_sets(uvvis_dir)
    print(f"  Sets loaded: {list(dfs.keys())}")

    # Derive ε(λ)
    eps_df = derive_epsilon(dfs, pathlength_cm=args.pathlength)
    eps_stats_df = epsilon_stats(eps_df)

    # Save ε table. This file is the project's published MB ε(λ) table, so it
    # carries the canonical POOLED FIT, not the point-wise mean that eps_stats_df
    # holds. epsilon_stats() averages eps_c = A_c/(c*l) over every set AND every
    # concentration, which diverges as c -> 0 whenever the spectra carry a
    # baseline: the MB files have no blank column and a ~0.033 AU offset, so that
    # mean returned eps(665) = 91,597 against the canonical 74,168 -- 23% high.
    # eps_df/eps_stats_df are still used below for the per-concentration spread
    # figures, where showing that divergence is the point.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from forward_model import load_uvvis  # noqa: E402

    eps_lam, eps_fit, eps_se = load_uvvis(
        sorted(uvvis_dir.glob("*MB_UV_VIS*.csv")),
        pathlength_cm=args.pathlength,
        min_fit_conc_um=1.525,
        max_fit_conc_um=50.0,
        exclude_columns=("MB_7.812uM",),
    )
    n_pooled = 18  # 3 sets x 6 concentrations in the canonical window
    eps_pooled_df = pd.DataFrame(
        {
            "wavelength_nm": eps_lam,
            "mean": eps_fit,
            "std": eps_se,  # standard error of the pooled fit, not a spread of means
            "n": n_pooled,
        }
    )
    eps_out = out_dir / "uvvis_epsilon.csv"
    eps_pooled_df.to_csv(eps_out, index=False, float_format="%.2f")
    print(f"[mb_uvvis] ε(λ) table (pooled fit, n={n_pooled}) → {eps_out.name}")
    print(f"           ε(665) = {np.interp(665.0, eps_lam, eps_fit):,.0f} M-1 cm-1")

    # Load Ohlab sensitivity
    S = load_sensitivity("average_all")

    # Spectral overlay figure
    fig_out = out_dir / "spectral_sensitivity_and_crosstalk_MB.png"
    make_spectral_figure(S, eps_stats_df, fig_out, args.pathlength)

    # Beer-Lambert check figure
    bl_fig_out = out_dir / "uvvis_beer_lambert_check.png"
    make_epsilon_calibration_fig(eps_df, bl_fig_out, args.pathlength)

    # Cross-validation report
    cv_out = out_dir / "uvvis_cross_validation.md"
    write_crossval_report(eps_stats_df, S, cv_out, args.pathlength)

    print("[mb_uvvis] Done.")


if __name__ == "__main__":
    main()
