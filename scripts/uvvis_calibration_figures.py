"""
uvvis_calibration_figures.py

Publication-quality UV-Vis calibration figures for:
  - K3[Fe(CN)6] (ferricyanide): full spectra + Beer-Lambert at 420 nm
  - Methylene blue (MB): full spectra + Beer-Lambert at 665 nm and 610 nm

Run from anywhere inside the project:
    python scripts/uvvis_calibration_figures.py

Outputs -> Data/spectra/dataanalysis/
  spectra_ferricyanide.{png,svg,csv}
  spectra_mb.{png,svg,csv}
  beerlam bert_ferricyanide_{8mM,4mM,1mM}.{png,svg,csv}
  beer_lambert_mb_665nm.{png,svg,csv}
  beer_lambert_mb_610nm.{png,svg,csv}
  calibration_stats.txt
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import stats as scipy_stats

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "Data"
FERIC_DIR = DATA_ROOT / "spectra" / "ferricyanide"
MB_DIR = DATA_ROOT / "spectra" / "methylene_blue"
OUT_DIR = DATA_ROOT / "spectra" / "dataanalysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Physical constants ───────────────────────────────────────────────────────
L_CM = 0.2  # Labbox MALG-002-002, 2 mm path length
FERIC_WL = 420.0  # ferricyanide analytical wavelength (nm)
MB_WL_MAIN = 665.0  # MB monomer peak (nm)
MB_WL_DIMER = 610.0  # MB dimer shoulder (nm)

# ─── Figure style: single column (8.7 cm = 3.43 in) ─────────────────────────
COL_W = 3.43  # figure width (in)
CAL_H = 3.10  # height for calibration plots
SPEC_H = 2.80  # height for spectra plots

FERIC_COLOR = "#D4820A"  # ferricyanide orange
MB_COLOR = "#7D3C98"  # MB monomer violet
DIMER_COLOR = "#1A5276"  # MB dimer blue
FIT_COLOR = "#2C3E50"  # regression line

SET_MARKERS = ["o", "s", "^"]
SET_COLORS = ["#2E86AB", "#A23B72", "#F18F01"]  # colorblind-friendly triad

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.8,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.5,
        "figure.dpi": 150,
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_conc_um(label: str) -> float | None:
    """Convert a concentration label to µM. Returns None if unparseable."""
    s = str(label).strip().replace(" ", "").replace("µ", "u").replace("μ", "u")
    s = re.sub(r"^[A-Za-z_]+", "", s)  # strip leading prefix like "MB_"
    sl = s.lower()
    try:
        if sl.endswith("mm"):
            return float(sl[:-2]) * 1000.0
        if sl.endswith("um"):
            return float(sl[:-2])
    except ValueError:
        pass
    return None


def _set_number(csv_path: Path) -> int:
    """Acquisition set number parsed from a "Set_1"/"Set-3" style filename."""
    m = re.search(r"set[_-](\d+)", csv_path.name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def load_ferricyanide_set(csv_path: Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """
    Load one ferricyanide UV-Vis CSV (wide format: wavelength + Blank + conc columns).
    Returns {c_µM: (wavelengths_nm ascending, net_absorbance)} after blank subtraction.
    """
    df = pd.read_csv(csv_path, header=0)
    df.rename(columns={df.columns[0]: "wl"}, inplace=True)
    df["wl"] = pd.to_numeric(df["wl"], errors="coerce")
    df = df.dropna(subset=["wl"]).sort_values("wl").reset_index(drop=True)
    wl = df["wl"].values

    blank_col = next((c for c in df.columns[1:] if str(c).strip().lower() == "blank"), None)
    if blank_col is None:
        warnings.warn(f"No blank column in {csv_path.name}; skipping blank subtraction")
        A_blank = np.zeros(len(wl))
    else:
        A_blank = pd.to_numeric(df[blank_col], errors="coerce").fillna(0.0).values

    result: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for col in df.columns[1:]:
        if col == blank_col:
            continue
        c_um = _parse_conc_um(str(col))
        if c_um is None:
            continue
        A_raw = pd.to_numeric(df[col], errors="coerce").values
        A_net = np.where(np.isfinite(A_raw), A_raw - A_blank, np.nan)
        result[c_um] = (wl, A_net)

    return result


def load_mb_set1(csv_path: Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """
    Load MB UV-Vis Set 1 (wide format, already blank-corrected).
    Filters out non-spectral junk columns embedded in the CSV by requiring
    that >80% of values in a column are numeric.

    NOTE: Set 1 labels the lowest concentration column "MB_7.812uM" but the
    embedded summary table in the same file uses 0.7812 µM for that entry.
    This is a missing decimal point, confirmed three ways: Sets 2 and 3 both
    label the equivalent column 0.7812uM; the series is an exact 2-fold serial
    dilution from 100 µM, which 7.812 would break; and the absorbance itself
    (0.0222 AU at 665 nm) is a 0.78 µM reading, where 12.5 µM reads 0.220 AU.
    Corrected on load since 2026-07-27 — before that the point was fitted at
    7.812 µM, which cost 0.003 in R² and inflated ε(665) by 270 M⁻¹ cm⁻¹.
    """
    df = pd.read_csv(csv_path, header=0)
    df.rename(columns={df.columns[0]: "wl"}, inplace=True)
    df["wl"] = pd.to_numeric(df["wl"], errors="coerce")
    df = df.dropna(subset=["wl"]).sort_values("wl").reset_index(drop=True)
    wl = df["wl"].values

    result: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for col in df.columns[1:]:
        c_um = _parse_conc_um(str(col))
        if c_um is None:
            continue
        if c_um == 7.812:
            c_um = 0.7812  # missing decimal point in the source header
        A_series = pd.to_numeric(df[col], errors="coerce")
        # Reject columns where fewer than 80% of rows are numeric
        # (catches the embedded summary table that appears as sparse columns)
        if A_series.notna().mean() < 0.8:
            continue
        result[c_um] = (wl, A_series.values)

    return result


def load_mb_set23(csv_path: Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """
    Load MB UV-Vis Sets 2 or 3 (multi-paired format).
    Row 0: concentration labels at even columns ("0.7812uM", "", "1.525uM", ...).
    Row 1: column type labels ("Wavelength (nm)", "Abs", ...).
    Rows 2+: data pairs (wavelength, absorbance) per concentration.
    """
    df_raw = pd.read_csv(csv_path, header=None, dtype=str)
    conc_labels = df_raw.iloc[0].tolist()
    data = df_raw.iloc[2:].reset_index(drop=True)

    result: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    n_cols = len(conc_labels)
    k = 0
    while k < n_cols:
        label = str(conc_labels[k]).strip()
        c_um = _parse_conc_um(label)
        if c_um is not None and k + 1 < n_cols:
            wl_col = data.iloc[:, k].apply(pd.to_numeric, errors="coerce")
            ab_col = data.iloc[:, k + 1].apply(pd.to_numeric, errors="coerce")
            mask = wl_col.notna() & ab_col.notna()
            wl_arr = wl_col[mask].values
            ab_arr = ab_col[mask].values
            order = np.argsort(wl_arr)
            result[c_um] = (wl_arr[order], ab_arr[order])
            k += 2
        else:
            k += 1

    return result


def _detect_mb_format(csv_path: Path) -> str:
    """Return 'set1' (wide) or 'set23' (multi-paired) from first line."""
    with open(csv_path, "r") as fh:
        first = fh.readline().strip().lower()
    return "set1" if first.startswith("wavelength") else "set23"


def load_mb_set(csv_path: Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    fmt = _detect_mb_format(csv_path)
    return load_mb_set1(csv_path) if fmt == "set1" else load_mb_set23(csv_path)


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION TABLE + OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def interp_abs(wl: np.ndarray, A: np.ndarray, target_nm: float) -> float:
    """Linear interpolation to get absorbance at target wavelength."""
    if len(wl) < 2:
        return np.nan
    return float(np.interp(target_nm, wl, A, left=np.nan, right=np.nan))


def _grubbs_crit(n: int, alpha: float = 0.05) -> float:
    """
    Grubbs critical value via t-distribution:
        G_crit = ((n-1)/sqrt(n)) * sqrt(t^2 / (n-2+t^2))
    where t = t(alpha/(2n), df=n-2).
    """
    if n < 3:
        return np.inf
    t_crit = float(scipy_stats.t.ppf(1.0 - alpha / (2.0 * n), df=n - 2))
    return ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))


def grubbs_test(values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, int | None]:
    """
    Single-iteration Grubbs test (one-outlier).
    Returns (valid_mask, outlier_index_or_None).
    """
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    n = int(finite.sum())
    if n < 3:
        return np.ones(len(vals), dtype=bool), None

    mean = vals[finite].mean()
    sd = vals[finite].std(ddof=1)
    if sd == 0:
        return np.ones(len(vals), dtype=bool), None

    G_stats = np.where(finite, np.abs(vals - mean) / sd, -np.inf)
    max_idx = int(np.argmax(G_stats))
    G_max = G_stats[max_idx]
    G_crit = _grubbs_crit(n, alpha)

    mask = np.ones(len(vals), dtype=bool)
    if G_max > G_crit:
        mask[max_idx] = False
        return mask, max_idx
    return mask, None


def build_calibration_table(
    sets_data: list[dict[float, tuple[np.ndarray, np.ndarray]]],
    target_nm: float,
) -> pd.DataFrame:
    """
    Build {concentration: [set1_A, set2_A, set3_A]} DataFrame.
    Rows = all concentrations across any set; missing entries = NaN.
    """
    all_concs: set[float] = set()
    for d in sets_data:
        all_concs.update(d.keys())

    records = []
    for c in sorted(all_concs):
        row: dict = {"c_uM": c}
        for i, d in enumerate(sets_data):
            row[f"set{i + 1}"] = interp_abs(*d[c], target_nm) if c in d else np.nan
        records.append(row)

    return pd.DataFrame(records).set_index("c_uM")


def clean_calibration_table(
    df: pd.DataFrame, alpha: float = 0.05
) -> tuple[pd.DataFrame, list[str]]:
    """
    Grubbs test per concentration row; replace outliers with NaN.
    Returns (cleaned DataFrame, list of log messages).
    """
    df_clean = df.copy().astype(float)
    log: list[str] = []

    for c_um, row in df.iterrows():
        vals = row.values.astype(float)
        n_finite = int(np.sum(np.isfinite(vals)))
        if n_finite < 3:
            continue
        mask, out_idx = grubbs_test(vals, alpha=alpha)
        if out_idx is not None:
            col_name = df.columns[out_idx]
            log.append(
                f"  c = {c_um:.4f} µM | {col_name} flagged "
                f"(A = {vals[out_idx]:.5f} AU; G > G_crit at α={alpha})"
            )
            df_clean.loc[c_um, col_name] = np.nan

    return df_clean, log


def summarize_calibration(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD across sets (NaN = outlier or missing)."""
    df = df_clean.astype(float)

    def _safe_std(row: pd.Series) -> float:
        vals = row.dropna().values
        return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0

    return pd.DataFrame(
        {
            "mean": df.mean(axis=1, skipna=True),
            "sd": df.apply(_safe_std, axis=1),
            "n": df.notna().sum(axis=1),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LINEAR REGRESSION + FIGURES OF MERIT
# ═══════════════════════════════════════════════════════════════════════════════


def linear_calibration(concs: np.ndarray, means: np.ndarray) -> dict:
    """
    OLS regression A = m·c + b on finite (conc, mean) pairs.
    LoD = 3.3·σ/m, LoQ = 10·σ/m  (IUPAC/ICH, σ from regression residuals).
    σ uses ddof=2 (slope + intercept consumed).
    """
    mask = np.isfinite(concs) & np.isfinite(means)
    x, y = concs[mask], means[mask]
    if len(x) < 3:
        return {}

    slope, intercept, r, p_val, se = scipy_stats.linregress(x, y)
    residuals = y - (slope * x + intercept)
    sigma = float(np.std(residuals, ddof=2))

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r**2),
        "p_value": float(p_val),
        "se_slope": float(se),
        "sigma_resid": sigma,
        "lod_uM": 3.3 * sigma / abs(slope) if slope != 0 else np.nan,
        "loq_uM": 10.0 * sigma / abs(slope) if slope != 0 else np.nan,
        "n_points": int(len(x)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: UV-Vis SPECTRA
# ═══════════════════════════════════════════════════════════════════════════════


def _nanstd_safe(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """nanstd ddof=1 that returns 0 where fewer than 2 finite values exist."""
    n_finite = np.sum(np.isfinite(arr), axis=axis)
    mean_ax = np.nanmean(arr, axis=axis, keepdims=True)
    sq_diff = np.where(np.isfinite(arr), (arr - mean_ax) ** 2, np.nan)
    variance = np.nansum(sq_diff, axis=axis) / np.maximum(n_finite - 1, 1)
    return np.where(n_finite >= 2, np.sqrt(variance), 0.0)


def _mean_spectrum(
    sets_data: list[dict[float, tuple[np.ndarray, np.ndarray]]], c_um: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (wl_grid, mean_A, sd_A) averaged across all sets that have c_um.
    Uses wavelength grid of the first available set; others are interpolated.
    """
    available = [d[c_um] for d in sets_data if c_um in d]
    if not available:
        return np.array([]), np.array([]), np.array([])

    wl_ref = available[0][0]
    stacked = np.array([np.interp(wl_ref, wl, A, left=np.nan, right=np.nan) for wl, A in available])
    mean_A = np.nanmean(stacked, axis=0)
    sd_A = _nanstd_safe(stacked, axis=0) if len(available) > 1 else np.zeros_like(mean_A)
    return wl_ref, mean_A, sd_A


def plot_spectra(
    sets_data: list[dict[float, tuple[np.ndarray, np.ndarray]]],
    lambda_marks: list[tuple[float, str]],
    ylabel: str,
    outname: str,
) -> pd.DataFrame:
    """
    Plot mean UV-Vis spectra for all concentrations, colored by log10(c / µM).
    SD shown as shaded band. Returns companion CSV DataFrame.
    """
    all_concs = sorted(set().union(*[set(d.keys()) for d in sets_data]))
    log_concs = np.log10(all_concs)
    norm = Normalize(vmin=log_concs.min(), vmax=log_concs.max())
    cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(COL_W, SPEC_H))

    wl_ref: np.ndarray | None = None
    csv_data: dict[str, np.ndarray] = {}

    for c in all_concs:
        wl, mean_A, sd_A = _mean_spectrum(sets_data, c)
        if len(wl) == 0:
            continue
        if wl_ref is None:
            wl_ref = wl
            csv_data["wavelength_nm"] = wl

        color = cmap(norm(np.log10(c)))
        ax.plot(wl, mean_A, color=color, lw=0.7, alpha=0.9)
        ax.fill_between(wl, mean_A - sd_A, mean_A + sd_A, color=color, alpha=0.18)

        tag = f"{c:.4g}uM"
        csv_data[f"mean_{tag}"] = np.interp(wl_ref, wl, mean_A, left=np.nan, right=np.nan)
        csv_data[f"sd_{tag}"] = np.interp(wl_ref, wl, sd_A, left=np.nan, right=np.nan)

    # Wavelength markers
    y_lims = ax.get_ylim()
    for lam, lbl in lambda_marks:
        ax.axvline(lam, color=FIT_COLOR, lw=0.8, ls="--", alpha=0.75)
        ax.text(lam + 4, y_lims[1] * 0.93, lbl, fontsize=6, color=FIT_COLOR, va="top")

    # Colorbar (log10(c / µM))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label(r"$\log_{10}(c\,/\,\mu\mathrm{M})$", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(350, 805)
    ax.set_ylim(bottom=-0.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{outname}.{ext}", dpi=600, bbox_inches="tight", format=ext)
    plt.close(fig)
    print(f"  -> {outname}.png  /  .svg")

    return pd.DataFrame(csv_data)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: BEER-LAMBERT CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_beer_lambert(
    cal_clean: pd.DataFrame,
    cal_summary: pd.DataFrame,
    fit_params: dict,
    analyte_color: str,
    c_max_um: float,
    x_label: str,
    y_label: str,
    outname: str,
) -> pd.DataFrame:
    """
    Beer-Lambert calibration plot:
      - individual set points (small, semi-transparent)
      - mean ± 1 SD (filled marker with error bar)
      - OLS linear fit line
    Returns companion CSV DataFrame.
    """
    mask_c = cal_clean.index <= c_max_um
    clean = cal_clean[mask_c].copy().astype(float)
    summ = cal_summary[mask_c]

    fig, ax = plt.subplots(figsize=(COL_W, CAL_H))

    # Individual set points
    for i, col in enumerate(clean.columns):
        y_vals = clean[col].values
        x_vals = clean.index.values
        finite = np.isfinite(y_vals)
        ax.scatter(
            x_vals[finite],
            y_vals[finite],
            marker=SET_MARKERS[i],
            color=SET_COLORS[i],
            s=12,
            alpha=0.60,
            linewidths=0.4,
            zorder=3,
            label=f"Set {i + 1}",
        )

    # Mean ± 1 SD
    ax.errorbar(
        summ.index,
        summ["mean"],
        yerr=summ["sd"],
        fmt="o",
        color=analyte_color,
        markersize=4.5,
        capsize=2.5,
        capthick=0.8,
        elinewidth=0.8,
        linewidth=0,
        zorder=4,
        label="Mean ± 1 SD",
    )

    # OLS fit
    if fit_params:
        m = fit_params["slope"]
        b = fit_params["intercept"]
        r2 = fit_params["r2"]
        x_fit = np.array([0.0, c_max_um])
        ax.plot(
            x_fit,
            m * x_fit + b,
            color=FIT_COLOR,
            lw=0.9,
            ls="--",
            zorder=5,
            label=f"OLS (R² = {r2:.4f})",
        )

    # x-axis: µM for ≤1000 µM; mM for larger ranges
    if c_max_um >= 2000:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1000:.1f}"))
        ax.set_xlabel(x_label.replace("µM", "mM"))
    else:
        ax.set_xlabel(x_label)

    ax.set_ylabel(y_label)
    ax.set_xlim(-0.02 * c_max_um, 1.06 * c_max_um)

    y_min = (summ["mean"] - summ["sd"]).min()
    ax.set_ylim(bottom=min(-0.02, y_min * 1.15))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6, framealpha=0.85, edgecolor="none")

    fig.tight_layout(pad=0.5)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{outname}.{ext}", dpi=600, bbox_inches="tight", format=ext)
    plt.close(fig)
    print(f"  -> {outname}.png  /  .svg")

    # Companion CSV: mean + SD + individual sets
    csv_df = clean.copy()
    csv_df.insert(0, "mean_A_AU", summ["mean"])
    csv_df.insert(1, "sd_A_AU", summ["sd"])
    csv_df.index.name = "c_uM"
    return csv_df


# ═══════════════════════════════════════════════════════════════════════════════
# STATS FILE
# ═══════════════════════════════════════════════════════════════════════════════


def write_stats(stats_sections: list[tuple[str, dict, list[str]]]) -> None:
    """
    Write calibration statistics and outlier log to calibration_stats.txt.
    stats_sections: [(section_title, fit_params_dict, outlier_log_lines), ...]
    """
    lines = [
        "UV-Vis Calibration Statistics",
        "=" * 62,
        f"Date              : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Path length       : {L_CM} cm  (Labbox MALG-002-002, 2 mm cuvette)",
        "Outlier detection : Grubbs test, single-outlier, alpha = 0.05",
        "LoD               : 3.3 × sigma_resid / |slope|   (IUPAC/ICH)",
        "LoQ               : 10  × sigma_resid / |slope|   (IUPAC/ICH)",
        "sigma_resid       : SD of OLS residuals, ddof = 2",
        "",
        "CORRECTED ON LOAD (2026-07-27) — MB Set 1 concentration label:",
        "  Column 'MB_7.812uM' in Set_1_MB_UV_VIS_25052026.csv is a",
        "  missing decimal point; the true value is 0.7812 µM, and it",
        "  is relabelled at load. Confirmed by: Sets 2 and 3 both label",
        "  the equivalent column '0.7812uM'; the series 0.7812 ... 100",
        "  is an exact 2-fold serial dilution that 7.812 would break;",
        "  and A = 0.0222 AU at 665 nm is a 0.78 µM reading (12.5 µM",
        "  reads 0.220 AU). Fitting it at 7.812 µM cost 0.003 in R² and",
        "  inflated eps(665) by 270 M-1 cm-1.",
        "",
        "  Replicate sets are ordered by set number, not filename: Set-3",
        "  uses a hyphen where Sets 1 and 2 use an underscore, so before",
        "  2026-07-27 the 'Set 1/2/3' legend labels were permuted 3,1,2.",
        "  Pooled fits are unaffected; per-set labels were not.",
        "",
    ]

    for title, params, outlier_log in stats_sections:
        lines += ["", title, "-" * len(title)]
        if params:
            lod = params.get("lod_uM", np.nan)
            loq = params.get("loq_uM", np.nan)
            # ε = slope [AU µM⁻¹] × 1e6 [µM M⁻¹] / L_CM [cm]
            eps = params["slope"] * 1e6 / L_CM
            lines += [
                f"  slope        : {params['slope']:.6e} AU µM⁻¹",
                f"  ε (from slope): {eps:.1f} M⁻¹ cm⁻¹   [= slope × 10⁶ / {L_CM} cm]",
                f"  intercept    : {params['intercept']:.6e} AU",
                f"  R²           : {params['r2']:.6f}",
                f"  p (slope≠0)  : {params['p_value']:.3e}",
                f"  SE slope     : {params['se_slope']:.6e} AU µM⁻¹",
                f"  sigma_resid  : {params['sigma_resid']:.6e} AU",
                f"  LoD          : {lod:.4f} µM",
                f"  LoQ          : {loq:.4f} µM",
                f"  n points     : {params['n_points']}",
            ]
        else:
            lines.append("  (insufficient data for regression)")

        if outlier_log:
            lines += ["", "  Outliers removed:"] + outlier_log
        else:
            lines.append("  No outliers detected.")

    out_path = OUT_DIR / "calibration_stats.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("  -> calibration_stats.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    print(f"[uvvis_calibration] Output -> {OUT_DIR}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading ferricyanide spectra...")
    feric_csvs = sorted(FERIC_DIR.glob("*.csv"))
    feric_sets = [load_ferricyanide_set(p) for p in feric_csvs]
    for p, d in zip(feric_csvs, feric_sets):
        concs = sorted(d.keys())
        print(f"  {p.name}: {len(d)} concs [{concs[0]:.2f} – {concs[-1]:.0f} µM]")

    print("\nLoading methylene blue spectra...")
    # Sort on the set number, not the filename: Set-3 uses a hyphen where
    # Set_1 and Set_2 use an underscore, so a plain sort orders them 3, 1, 2
    # and every "Set N" legend label is wrong by that permutation.
    mb_csvs = sorted(MB_DIR.glob("*.csv"), key=_set_number)
    mb_sets = [load_mb_set(p) for p in mb_csvs]
    for p, d in zip(mb_csvs, mb_sets):
        concs = sorted(d.keys())
        print(f"  {p.name}: {len(d)} concs [{concs[0]:.4f} – {concs[-1]:.0f} µM]")

    # ── Spectra figures ───────────────────────────────────────────────────────
    print("\nSpectra figures...")

    df_spec_feric = plot_spectra(
        feric_sets,
        lambda_marks=[(FERIC_WL, "420 nm")],
        ylabel="Absorbance (AU)",
        outname="spectra_ferricyanide",
    )
    df_spec_feric.to_csv(OUT_DIR / "spectra_ferricyanide.csv", index=False, float_format="%.6f")

    df_spec_mb = plot_spectra(
        mb_sets,
        lambda_marks=[(MB_WL_DIMER, "610 nm"), (MB_WL_MAIN, "665 nm")],
        ylabel="Absorbance (AU)",
        outname="spectra_mb",
    )
    df_spec_mb.to_csv(OUT_DIR / "spectra_mb.csv", index=False, float_format="%.6f")

    # ── Ferricyanide Beer-Lambert ─────────────────────────────────────────────
    print("\nFerricyanide Beer-Lambert @ 420 nm...")
    cal_feric_raw = build_calibration_table(feric_sets, FERIC_WL)
    cal_feric_clean, feric_outliers = clean_calibration_table(cal_feric_raw)
    cal_feric_summ = summarize_calibration(cal_feric_clean)

    feric_sections: list[tuple[str, dict, list[str]]] = []
    for label, c_max in [("8mM", 8000.0), ("4mM", 4000.0), ("1mM", 1000.0)]:
        mask = cal_feric_summ.index <= c_max
        summ_sub = cal_feric_summ[mask]
        fp = linear_calibration(summ_sub.index.values, summ_sub["mean"].values)
        section_title = f"Ferricyanide  420 nm — up to {label}"
        print(
            f"  Fitting {section_title}: n={fp.get('n_points', 0)}, "
            f"R²={fp.get('r2', float('nan')):.5f}"
        )

        csv_df = plot_beer_lambert(
            cal_clean=cal_feric_clean,
            cal_summary=cal_feric_summ,
            fit_params=fp,
            analyte_color=FERIC_COLOR,
            c_max_um=c_max,
            x_label="Concentration (µM)",
            y_label=f"Absorbance at {FERIC_WL:.0f} nm (AU)",
            outname=f"beer_lambert_ferricyanide_{label}",
        )
        csv_df.to_csv(
            OUT_DIR / f"beer_lambert_ferricyanide_{label}.csv",
            float_format="%.6f",
        )
        feric_sections.append((section_title, fp, feric_outliers))

    # ── Methylene blue Beer-Lambert ───────────────────────────────────────────
    print("\nMethylene blue Beer-Lambert...")
    mb_sections: list[tuple[str, dict, list[str]]] = []

    for wl_target, color, nm_str in [
        (MB_WL_MAIN, MB_COLOR, "665"),
        (MB_WL_DIMER, DIMER_COLOR, "610"),
    ]:
        print(f"  @ {wl_target:.0f} nm")
        cal_mb_raw = build_calibration_table(mb_sets, wl_target)
        cal_mb_clean, mb_outliers = clean_calibration_table(cal_mb_raw)
        cal_mb_summ = summarize_calibration(cal_mb_clean)

        mask = cal_mb_summ.index <= 105.0  # generous upper bound (100 µM max)
        summ_sub = cal_mb_summ[mask]
        fp = linear_calibration(summ_sub.index.values, summ_sub["mean"].values)
        section_title = f"Methylene blue  {wl_target:.0f} nm"
        print(f"    n={fp.get('n_points', 0)}, R²={fp.get('r2', float('nan')):.5f}")

        csv_df = plot_beer_lambert(
            cal_clean=cal_mb_clean,
            cal_summary=cal_mb_summ,
            fit_params=fp,
            analyte_color=color,
            c_max_um=100.0,
            x_label="Concentration (µM)",
            y_label=f"Absorbance at {wl_target:.0f} nm (AU)",
            outname=f"beer_lambert_mb_{nm_str}nm",
        )
        csv_df.to_csv(OUT_DIR / f"beer_lambert_mb_{nm_str}nm.csv", float_format="%.6f")
        mb_sections.append((section_title, fp, mb_outliers))

    # ── Stats file ────────────────────────────────────────────────────────────
    print("\nWriting stats file...")
    write_stats(feric_sections + mb_sections)

    print(f"\n[uvvis_calibration] Done. All outputs in:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
