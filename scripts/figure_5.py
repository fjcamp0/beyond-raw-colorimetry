"""
Beer-Lambert-Bayer integrand collapse panels for Figure 5 (LED and daylight).

Panel A — B-channel integrand phi(lambda)*T_B(lambda)*10^(-eps*c*l) at five
ferricyanide concentrations, normalised to blank. Marks 420 nm (ferricyanide
lambda_max) and 473 nm (tail boundary); shades the transparent tail (>473 nm).

Panel B — G-channel integrand at the same concentrations, showing G is
nearly spectrally inert for ferricyanide.

Panel C — R-channel integrand at five methylene blue concentrations (MB's
active channel, analogous to Panel A). Added 2026-07-29 per Q5 of
correction_plan_javier-steph_2026-07-29.md.

Panel D — G-channel integrand for methylene blue (control channel, analogous
to Panel B).

Outputs (PNG + TIFF + SVG at 600 dpi) and companion CSVs saved to:
  LED/figure_5a_LED.{png,tiff,svg,csv}
  LED/figure_5b_LED.{png,tiff,svg,csv}
  LED/figure_5c_LED.{png,tiff,svg,csv}
  LED/figure_5d_LED.{png,tiff,svg,csv}
  LED/figure_5_assembled_LED.{png,tiff,svg}
  daylight/figure_5a_daylight.{png,tiff,svg,csv}
  daylight/figure_5b_daylight.{png,tiff,svg,csv}
  daylight/figure_5c_daylight.{png,tiff,svg,csv}
  daylight/figure_5d_daylight.{png,tiff,svg,csv}
  daylight/figure_5_assembled_daylight.{png,tiff,svg}

ACS format: 3.25 in x 3.0 in per panel, 7.0 in x 3.0 in assembled,
Arial 8 pt, 600 dpi, inside ticks, no title on image.

Run from project root: python scripts/figure_5.py

Scientific constraint: Do NOT claim B/G ratio removes the transparent-tail
term. The ideal BLB model shows G is spectrally inert for ferricyanide, but
model-only fits show BdivG R2 is not materially better than B alone (R2 for
A_B = 0.998898 vs A_BdivG = 0.998870 over 7.8-1000 uM). The empirical B/G
improvement is explained by cancellation of an observed channel-common
absorbance component not captured by the ideal spectral model.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_model import load_uvvis  # noqa: E402

# ε is fitted by forward_model.load_uvvis, not here. Until 2026-09-21 this script
# carried two fitters of its own, load_uvvis_eps() and load_mb_eps_spectrum(), both
# forcing the fit through the origin and averaging ε over replicates, with a 1000 µM
# ceiling hardcoded at the call site. They returned ε(420) = 992.4 against the
# canonical 1017 and ε(665) = 66,601 against the canonical 74,168. Both are removed:
# one ε fitter for the project, reproducing the canonical values exactly.
#
# Canonical windows (CLAUDE.md, gate G-G; see also
# _ai_outputs/review/mb_epsilon_reversal_2026-07-28.md):
#   ferricyanide   full measured range, no bound       -> ε(420) = 1017, n = 30
#   methylene blue 1.525–50 µM, MB_7.812uM excluded    -> ε(665) = 74,168, n = 18
MB_EPS_WINDOW = dict(min_fit_conc_um=1.525, max_fit_conc_um=50.0, exclude_columns=("MB_7.812uM",))

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJ_ROOT / "Data"
ILLUM_DIR = DATA_DIR / "spectra" / "illuminant"
FERIC_DIR = DATA_DIR / "spectra" / "ferricyanide"
MB_DIR = DATA_DIR / "spectra" / "methylene_blue"

# Vendored raw-image-analysis modules (vendor/, see README); override with COLORIMETRY_VENDOR.
SENS_DIR = (
    Path(
        os.environ.get("COLORIMETRY_VENDOR", str(Path(__file__).resolve().parent.parent / "vendor"))
    )
    / "_resources"
    / "spectral_sensitivity"
)

LED_OUT = DATA_DIR / "paper_figures" / "figure_5" / "LED"
DAYLIGHT_OUT = DATA_DIR / "paper_figures" / "figure_5" / "daylight"
LED_OUT.mkdir(parents=True, exist_ok=True)
DAYLIGHT_OUT.mkdir(parents=True, exist_ok=True)

# ── ACS rcParams ───────────────────────────────────────────────────────────────

ACS_W = 3.25  # single column width, in
ACS_H = 3.0  # panel height, in
ASM_W = 7.0  # assembled 2-panel width, in
DPI = 600

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "patch.linewidth": 0.5,
        "figure.dpi": 150,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    }
)

LN10 = np.log(10.0)

# Integration grid: 400-700 nm, 1 nm steps
GRID = np.arange(400.0, 701.0, 1.0)  # 301 points

# Concentrations to plot (blank + four non-zero), in µM
CONCS_UM: tuple[float, ...] = (0.0, 62.5, 250.0, 1000.0, 2000.0)

# Methylene blue concentrations to plot (blank + four non-zero, spanning the
# measured 1.525-100 uM range), in uM
MB_CONCS_UM: tuple[float, ...] = (0.0, 3.125, 12.5, 50.0, 100.0)

# Reference wavelengths
LAM_MAX = 420.0  # ferricyanide absorption peak
TAIL_WL = 473.0  # transparent-tail boundary: eps < 1% of the 420 nm peak

# Methylene blue R-channel tail boundary. Same rule as ferricyanide: the
# transparent region is where eps falls below 1% of its peak, which for MB is
# 426 nm. The tail is on the LOW-wavelength side here, opposite ferricyanide's
# B-channel geometry. MB has no single sharp lambda_max (665 nm monomer +
# 610 nm dimer both contribute), so no lambda_max marker is drawn for MB panels.
MB_TAIL_WL = 426.0
MB_TAIL_ABOVE = False

# Path lengths (from fitted forward models). Each MUST come from a run whose own
# UV-Vis path length was 0.2 cm, to match L_UVVIS below: the fit constrains only the
# product eps*l_cam, so pairing an eps fitted at 0.2 cm with an l_cam fitted at 1.0 cm
# inflates the modelled absorbance fivefold.
L_CAM_LED = 0.2851  # cm - ferricyanide/forward_model (pl 0.2 cm), regenerated 2026-09-21
# 2026-09-21: was 2.5032, taken from the NON-suffixed daylight_diffuser run, which
# used --uvvis-pathlength 1.0. Its own summary records no path length (the field
# postdates it), but its eps_eff are exactly 1/5 of the _l0p2 twin's (B 17.1 vs 85.3),
# and 2.5032/0.5027 = 4.98, so it is a 1.0 cm run. The _l0p2 value below is what the
# project publishes as 5.03 mm.
# WARNING: this changes a manuscript claim. CONTEXT.md:80-85 and :129-131 build a
# "ferricyanide 8x l_cam anomaly under daylight" argument in S4.6 on the 25.03 mm
# figure. At 0.2 cm the anomaly is 5.03/2.97 = 1.69x, not 8.43x. See
# _ai_outputs/review/NOTE_lcam_daylight_pathlength_2026-09-21.md.
L_CAM_DAYLIGHT = (
    0.4823  # cm - ferricyanide daylight_diffuser _l0p2 (pl 0.2 cm), regenerated 2026-09-21
)
L_UVVIS = 0.2  # cm - UV-Vis cuvette path length

# Methylene blue path lengths (R channel, from MB forward_model_summary.md files)
MB_L_CAM_LED = 0.2414  # cm - methylene_blue/forward_model (pl 0.2 cm), regenerated 2026-09-21
MB_L_CAM_DAYLIGHT = (
    0.1616  # cm - methylene_blue daylight_diffuser _l0p2 (pl 0.2 cm), regenerated 2026-09-21
)

# ── Data loaders ───────────────────────────────────────────────────────────────


def load_illuminant(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load illuminant CSV (wavelength_nm, normalised_spectral_power), sorted ascending."""
    df = pd.read_csv(path)
    lam = df.iloc[:, 0].to_numpy(dtype=float)
    phi = df.iloc[:, 1].to_numpy(dtype=float)
    order = np.argsort(lam)
    return lam[order], phi[order]


def _col_to_um(col: str) -> float | None:
    """Parse a UV-Vis column header to µM. None if not a concentration label."""
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


def load_bayer_sensitivity() -> tuple[np.ndarray, np.ndarray]:
    """
    Load Ohlab average Bayer sensitivity from average_all.txt.
    Returns (wl_31, S_31x3) where wl_31 = 400, 410, ..., 700 nm.
    """
    path = SENS_DIR / "average_all.txt"
    S = np.loadtxt(path)  # 31 x 3  (R, G, B columns)
    wl = np.arange(400.0, 710.0, 10.0)  # 31 points
    return wl, S


# ── Grid interpolation ─────────────────────────────────────────────────────────


def to_grid(lam: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Interpolate any spectrum onto GRID (400-700 nm, 1 nm steps)."""
    return np.interp(GRID, lam, vals, left=0.0, right=0.0)


def sens_to_grid(wl: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Interpolate 31x3 Ohlab sensitivity onto GRID. Returns (301, 3)."""
    out = np.zeros((len(GRID), 3))
    for i in range(3):
        out[:, i] = np.interp(GRID, wl, S[:, i], left=0.0, right=0.0)
    return out


# ── Integrand computation ──────────────────────────────────────────────────────


def compute_integrands(
    phi_g: np.ndarray,  # (301,) illuminant on GRID
    T_g: np.ndarray,  # (301, 3) Bayer sensitivity on GRID
    eps_g: np.ndarray,  # (301,) molar absorptivity on GRID, M-1 cm-1
    l_cam: float,  # effective path length, cm
    concs_um: tuple[float, ...],
    ch_idx: int,  # 0=R, 1=G, 2=B
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the BLB integrand phi(lambda)*T_ch(lambda)*10^(-eps*c*l)
    for each concentration, normalised so blank integrand integrates to 1.

    Returns:
        integrands  : (n_concs, 301) array of normalised integrands
        blank_norm  : scalar normalisation factor (blank integral)
    """
    T = T_g[:, ch_idx]
    blank_norm = float(np.trapezoid(phi_g * T, dx=1.0))
    if blank_norm == 0:
        blank_norm = 1.0

    integrands = np.empty((len(concs_um), len(GRID)))
    for k, c_um in enumerate(concs_um):
        c_M = c_um * 1e-6
        trans = np.exp(-eps_g * (c_M * l_cam) * LN10)
        integ = phi_g * T * trans
        integrands[k] = integ / blank_norm

    return integrands, blank_norm


# ── Plotting ───────────────────────────────────────────────────────────────────

# Blue gradient for increasing concentration: dark→light blue
# Blank uses dashed line (grey-blue); higher concs progressively lighter blue
_BLANK_COLOR = "#7f7f7f"  # grey dashed for blank
_BLUE_DARK = "#08306b"  # darkest blue (lowest non-zero conc)
_BLUE_LIGHT = "#c6dbef"  # lightest blue (highest conc)


def _conc_colors(n_nonzero: int) -> list[str]:
    """Return n_nonzero hex colours from dark to light blue."""
    cmap = plt.cm.Blues
    # sample at 0.85 (dark) down to 0.30 (light)
    vals = np.linspace(0.85, 0.30, n_nonzero)
    return [matplotlib.colors.to_hex(cmap(v)) for v in vals]


def _conc_label(c_um: float) -> str:
    """Format concentration for legend: 0 → 'blank', else with unit."""
    if c_um == 0:
        return "blank (c = 0)"
    if c_um >= 1000:
        return f"{c_um / 1000:.1f} mM"
    return f"{c_um:.0f} µM"


def plot_panel(
    phi_g: np.ndarray,
    T_g: np.ndarray,
    eps_g: np.ndarray,
    l_cam: float,
    concs_um: tuple[float, ...],
    ch_idx: int,  # 0=R, 1=G, 2=B
    ch_name: str,  # "B" or "G"
    out_stem: str,  # e.g. "figure_5a_LED"
    out_dir: Path,
    panel_letter: str,  # "A" or "B"
    lam_max: float | None = LAM_MAX,  # None to omit the lambda_max marker
    tail_wl: float = TAIL_WL,
    tail_above: bool = True,  # True: tail is wavelength > tail_wl
    legend_loc: str = "upper right",
) -> pd.DataFrame:
    """Generate and save one panel (PNG+TIFF+SVG+CSV); return the plotted data.

    lam_max/tail_wl/tail_above default to the ferricyanide B-channel geometry
    (tail above 473 nm). Methylene blue's R-channel tail sits on the opposite
    side of its boundary (tail below 426 nm, per figure_S7_ftail.py's
    MB_TAIL_WL/MB_TAIL_ABOVE) and has no single sharp lambda_max, so callers
    must override all three for MB panels.
    """
    integrands, _ = compute_integrands(phi_g, T_g, eps_g, l_cam, concs_um, ch_idx)

    is_blank = [c == 0 for c in concs_um]
    n_nonzero = sum(not b for b in is_blank)
    nonzero_colors = _conc_colors(n_nonzero)
    color_iter = iter(nonzero_colors)

    fig, ax = plt.subplots(figsize=(ACS_W, ACS_H))

    # Shade the transparent tail using the blank integrand as its envelope
    blank_idx = next(i for i, b in enumerate(is_blank) if b)
    blank_integ = integrands[blank_idx]
    tail_mask = (GRID > tail_wl) if tail_above else (GRID < tail_wl)
    tail_dir = ">" if tail_above else "<"
    ax.fill_between(
        GRID[tail_mask],
        0,
        blank_integ[tail_mask],
        color="#deebf7",
        alpha=0.65,
        zorder=0,
        label=f"Transparent tail ({tail_dir} {tail_wl:.0f} nm)",
    )

    for k, c_um in enumerate(concs_um):
        if is_blank[k]:
            color = _BLANK_COLOR
            lw = 1.2
            ls = "--"
            zorder = 3
        else:
            color = next(color_iter)
            lw = 1.0
            ls = "-"
            zorder = 2
        ax.plot(
            GRID,
            integrands[k],
            color=color,
            lw=lw,
            ls=ls,
            zorder=zorder,
            label=_conc_label(c_um),
        )

    if lam_max is not None:
        ax.axvline(
            lam_max,
            color="#cc0000",
            lw=0.9,
            ls="--",
            zorder=4,
            alpha=0.85,
            label=f"{lam_max:.0f} nm (λ$_{{max}}$)",
        )
    ax.axvline(
        tail_wl,
        color="#2166ac",
        lw=0.9,
        ls=":",
        zorder=4,
        alpha=0.85,
        label=f"{tail_wl:.0f} nm (tail boundary)",
    )

    ax.set_xlabel("Wavelength (nm)", fontsize=8)
    ax.set_ylabel(
        f"φ·S$_{{{ch_name}}}$·10$^{{−ε c l}}$  [normalised to blank]",
        fontsize=8,
    )
    ax.set_xlim(390, 700)
    ax.set_ylim(bottom=0)

    ax.legend(
        fontsize=6,
        framealpha=0.0,
        edgecolor="none",
        loc=legend_loc,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    ax.text(
        0.03,
        0.97,
        panel_letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )

    fig.tight_layout(pad=0.4)

    for ext in ("png", "tiff", "svg"):
        out_path = out_dir / f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.close(fig)

    csv_data = {"wavelength_nm": GRID.astype(int)}
    for k, c_um in enumerate(concs_um):
        col_name = "blank" if c_um == 0 else f"{c_um:.1f}uM"
        csv_data[col_name] = integrands[k]
    df_csv = pd.DataFrame(csv_data)

    csv_path = out_dir / f"{out_stem}.csv"
    df_csv.to_csv(csv_path, index=False, float_format="%.6g")
    print(f"  Saved: {csv_path}")

    return df_csv


def plot_assembled(
    phi_g: np.ndarray,
    T_g: np.ndarray,
    eps_g: np.ndarray,
    l_cam: float,
    concs_um: tuple[float, ...],
    out_stem: str,
    out_dir: Path,
) -> None:
    """Generate and save the assembled 2-panel figure ((a): B-channel, (b): G-channel)."""
    fig, axes = plt.subplots(1, 2, figsize=(ASM_W, ACS_H))

    ch_configs = [
        (2, "B", "(a)"),  # left panel: B channel
        (1, "G", "(b)"),  # right panel: G channel
    ]

    for ax, (ch_idx, ch_name, panel_letter) in zip(axes, ch_configs):
        integrands, _ = compute_integrands(phi_g, T_g, eps_g, l_cam, concs_um, ch_idx)

        is_blank = [c == 0 for c in concs_um]
        n_nonzero = sum(not b for b in is_blank)
        nonzero_colors = _conc_colors(n_nonzero)
        color_iter = iter(nonzero_colors)

        blank_idx = next(i for i, b in enumerate(is_blank) if b)
        blank_integ = integrands[blank_idx]
        tail_mask = GRID > TAIL_WL

        ax.fill_between(
            GRID[tail_mask],
            0,
            blank_integ[tail_mask],
            color="#deebf7",
            alpha=0.65,
            zorder=0,
            label=f"Transparent tail (> {TAIL_WL:.0f} nm)",
        )

        for k, c_um in enumerate(concs_um):
            if is_blank[k]:
                color = _BLANK_COLOR
                lw = 1.2
                ls = "--"
                zorder = 3
            else:
                color = next(color_iter)
                lw = 1.0
                ls = "-"
                zorder = 2
            ax.plot(
                GRID,
                integrands[k],
                color=color,
                lw=lw,
                ls=ls,
                zorder=zorder,
                label=_conc_label(c_um),
            )

        ax.axvline(
            LAM_MAX,
            color="#cc0000",
            lw=0.9,
            ls="--",
            zorder=4,
            alpha=0.85,
            label=f"{LAM_MAX:.0f} nm (λ$_{{max}}$)",
        )
        ax.axvline(
            TAIL_WL,
            color="#2166ac",
            lw=0.9,
            ls=":",
            zorder=4,
            alpha=0.85,
            label=f"{TAIL_WL:.0f} nm (tail boundary)",
        )

        ax.set_xlabel("Wavelength (nm)", fontsize=8)
        ax.set_ylabel(
            f"φ·S$_{{{ch_name}}}$·10$^{{−ε c l}}$  [normalised to blank]",
            fontsize=8,
        )
        ax.set_xlim(390, 700)
        ax.set_ylim(bottom=0)

        ax.legend(
            fontsize=5.5,
            framealpha=0.0,
            edgecolor="none",
            loc="upper right",
            handlelength=1.4,
            handletextpad=0.4,
            labelspacing=0.3,
        )

        ax.text(
            0.03,
            0.97,
            panel_letter,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.tight_layout(pad=0.4, w_pad=1.0)

    for ext in ("png", "tiff", "svg"):
        out_path = out_dir / f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Load spectral data and generate all Figure 5 panels and assembled figures."""
    print("\nLoading spectral data ...")

    # Bayer sensitivity (shared for both illuminants)
    wl_S, S = load_bayer_sensitivity()
    T_g = sens_to_grid(wl_S, S)

    # Ferricyanide ε(λ) (shared for both illuminants)
    uvvis_paths = [
        FERIC_DIR / "Ferricyanide_UV_VIS_1.csv",
        FERIC_DIR / "Ferricyanide_UV_VIS_2.csv",
        FERIC_DIR / "Ferricyanide_UV_VIS_3.csv",
    ]
    for p in uvvis_paths:
        if not p.exists():
            sys.exit(f"ERROR: UV-Vis file not found: {p}")

    eps_lam, eps_mean, _ = load_uvvis(uvvis_paths, L_UVVIS)
    eps_g = to_grid(eps_lam, eps_mean)
    print(
        f"  ε(λ) loaded: {len(eps_lam)} points, "
        f"ε(420 nm) = {np.interp(420.0, eps_lam, eps_mean):.0f} M-1 cm-1"
    )

    # Methylene blue ε(λ) (shared for both illuminants)
    mb_uvvis_paths = [
        MB_DIR / "Set_1_MB_UV_VIS_25052026.csv",
        MB_DIR / "Set_2_MB_UV_VIS_27052026.csv",
        MB_DIR / "Set-3_MB_UV_VIS_27052026.csv",
    ]
    for p in mb_uvvis_paths:
        if not p.exists():
            sys.exit(f"ERROR: MB UV-Vis file not found: {p}")

    mb_eps_lam, mb_eps_mean, _ = load_uvvis(mb_uvvis_paths, L_UVVIS, **MB_EPS_WINDOW)
    mb_eps_g = to_grid(mb_eps_lam, mb_eps_mean)
    mb_eps_665 = float(np.interp(665.0, mb_eps_lam, mb_eps_mean))
    print(f"  MB ε(λ) loaded: {len(mb_eps_lam)} points, ε(665 nm) = {mb_eps_665:.0f} M-1 cm-1")

    # ── LED illuminant ─────────────────────────────────────────────────────────
    print("\n[LED illuminant]")
    led_path = ILLUM_DIR / "illuminant_spelec_led.csv"
    if not led_path.exists():
        sys.exit(f"ERROR: LED illuminant not found: {led_path}")
    phi_lam_led, phi_led = load_illuminant(led_path)
    phi_g_led = to_grid(phi_lam_led, phi_led)
    print(f"  Illuminant: {led_path.name}, l_cam = {L_CAM_LED:.4f} cm")

    print("  Panel A (B-channel) ...")
    plot_panel(
        phi_g_led,
        T_g,
        eps_g,
        L_CAM_LED,
        CONCS_UM,
        ch_idx=2,
        ch_name="B",
        out_stem="figure_5a_LED",
        out_dir=LED_OUT,
        panel_letter="(a)",
    )

    print("  Panel B (G-channel) ...")
    plot_panel(
        phi_g_led,
        T_g,
        eps_g,
        L_CAM_LED,
        CONCS_UM,
        ch_idx=1,
        ch_name="G",
        out_stem="figure_5b_LED",
        out_dir=LED_OUT,
        panel_letter="(b)",
    )

    print("  Assembled 2-panel ...")
    plot_assembled(
        phi_g_led,
        T_g,
        eps_g,
        L_CAM_LED,
        CONCS_UM,
        out_stem="figure_5_assembled_LED",
        out_dir=LED_OUT,
    )

    print("  Panel C (MB R-channel) ...")
    plot_panel(
        phi_g_led,
        T_g,
        mb_eps_g,
        MB_L_CAM_LED,
        MB_CONCS_UM,
        ch_idx=0,
        ch_name="R",
        out_stem="figure_5c_LED",
        out_dir=LED_OUT,
        panel_letter="(c)",
        lam_max=None,
        tail_wl=MB_TAIL_WL,
        tail_above=MB_TAIL_ABOVE,
        legend_loc="center left",
    )

    print("  Panel D (MB G-channel) ...")
    plot_panel(
        phi_g_led,
        T_g,
        mb_eps_g,
        MB_L_CAM_LED,
        MB_CONCS_UM,
        ch_idx=1,
        ch_name="G",
        out_stem="figure_5d_LED",
        out_dir=LED_OUT,
        panel_letter="(d)",
        lam_max=None,
        tail_wl=MB_TAIL_WL,
        tail_above=MB_TAIL_ABOVE,
    )

    # ── Daylight illuminant ────────────────────────────────────────────────────
    print("\n[Daylight (diffuser) illuminant]")
    day_path = ILLUM_DIR / "illuminant_daylight_diffuser.csv"
    if not day_path.exists():
        sys.exit(f"ERROR: Daylight illuminant not found: {day_path}")
    phi_lam_day, phi_day = load_illuminant(day_path)
    phi_g_day = to_grid(phi_lam_day, phi_day)
    print(f"  Illuminant: {day_path.name}, l_cam = {L_CAM_DAYLIGHT:.4f} cm")

    print("  Panel A (B-channel) ...")
    plot_panel(
        phi_g_day,
        T_g,
        eps_g,
        L_CAM_DAYLIGHT,
        CONCS_UM,
        ch_idx=2,
        ch_name="B",
        out_stem="figure_5a_daylight",
        out_dir=DAYLIGHT_OUT,
        panel_letter="(a)",
    )

    print("  Panel B (G-channel) ...")
    plot_panel(
        phi_g_day,
        T_g,
        eps_g,
        L_CAM_DAYLIGHT,
        CONCS_UM,
        ch_idx=1,
        ch_name="G",
        out_stem="figure_5b_daylight",
        out_dir=DAYLIGHT_OUT,
        panel_letter="(b)",
    )

    print("  Assembled 2-panel ...")
    plot_assembled(
        phi_g_day,
        T_g,
        eps_g,
        L_CAM_DAYLIGHT,
        CONCS_UM,
        out_stem="figure_5_assembled_daylight",
        out_dir=DAYLIGHT_OUT,
    )

    print("  Panel C (MB R-channel) ...")
    plot_panel(
        phi_g_day,
        T_g,
        mb_eps_g,
        MB_L_CAM_DAYLIGHT,
        MB_CONCS_UM,
        ch_idx=0,
        ch_name="R",
        out_stem="figure_5c_daylight",
        out_dir=DAYLIGHT_OUT,
        panel_letter="(c)",
        lam_max=None,
        tail_wl=MB_TAIL_WL,
        tail_above=MB_TAIL_ABOVE,
        legend_loc="center left",
    )

    print("  Panel D (MB G-channel) ...")
    plot_panel(
        phi_g_day,
        T_g,
        mb_eps_g,
        MB_L_CAM_DAYLIGHT,
        MB_CONCS_UM,
        ch_idx=1,
        ch_name="G",
        out_stem="figure_5d_daylight",
        out_dir=DAYLIGHT_OUT,
        panel_letter="(d)",
        lam_max=None,
        tail_wl=MB_TAIL_WL,
        tail_above=MB_TAIL_ABOVE,
    )

    print("\nDone. All Figure 5 outputs written.")
    print(f"  LED/       -> {LED_OUT}")
    print(f"  daylight/  -> {DAYLIGHT_OUT}")


if __name__ == "__main__":
    main()
