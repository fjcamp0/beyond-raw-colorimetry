"""
illuminant_spectra_figures.py

Separate publication-quality figures for each illuminant spectrum measured with
the Spelec fibre spectrometer.  All traces are dark-subtracted and peak-normalised
to relative spectral power (0–1).  Visible range 350–800 nm.

Run from anywhere inside the project:
    python scripts/illuminant_spectra_figures.py

Outputs -> Data/spectra/illuminant/
    illuminant_spelec_led.{png,svg,csv}
    illuminant_led_diffuser_10s.{png,svg,csv}
    illuminant_led_diffuser_1s.{png,svg,csv}
    illuminant_daylight.{png,svg,csv}
    illuminant_daylight_clean.{png,svg,csv}
    illuminant_daylight_diffuser.{png,svg,csv}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ILLUM_DIR = PROJECT_ROOT / "Data" / "spectra" / "illuminant"
OUT_DIR = ILLUM_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Style constants (from figure_style.md) ───────────────────────────────────
COL_W = 3.43  # single-column width (in)
SPEC_H = 2.80  # spectra figure height (in)

FIT_COLOR = "#2C3E50"  # dashed markers + annotations
LED_COLOR = "#E67E22"  # warm amber for LED-type illuminants
DAY_COLOR = "#5DADE2"  # sky blue for daylight-type illuminants

WL_MIN, WL_MAX = 350, 800  # visible crop (nm)

# analytical wavelengths to mark on every plot
MARKERS_NM = {420: r"K$_3$[Fe(CN)$_6$]", 665: "MB"}

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


# ─── Illuminant definitions ───────────────────────────────────────────────────
# Each entry: (csv_filename, signal_col_index, line_color, output_stem, y_title)
# signal_col_index: column index for the raw signal (before dark subtraction).
# dark_col_index is always 1; wavelength_col_index is always 0.
ILLUMINANTS: list[dict] = [
    {
        "file": "Spelec_Specta_LED_Illuminator.csv",
        "sig_col": 2,
        "color": LED_COLOR,
        "stem": "illuminant_spelec_led",
        "title": "LED (no diffuser)",
    },
    {
        "file": "LED_with_diffuser_10s_int_time_spectra.csv",
        "sig_col": 3,
        "color": LED_COLOR,
        "stem": "illuminant_led_diffuser_10s",
        "title": "LED + diffuser (10 s integration)",
    },
    {
        "file": "LED_with_diffuser_1s_int_time_spectra.csv",
        "sig_col": 3,
        "color": LED_COLOR,
        "stem": "illuminant_led_diffuser_1s",
        "title": "LED + diffuser (1 s integration)",
    },
    {
        "file": "Daylight_spectra.csv",
        "sig_col": 3,
        "color": DAY_COLOR,
        "stem": "illuminant_daylight",
        "title": "Daylight (no diffuser)",
    },
    {
        "file": "Daylight_spectra_clean.csv",
        "sig_col": 2,
        "color": DAY_COLOR,
        "stem": "illuminant_daylight_clean",
        "title": "Daylight (clean)",
    },
    {
        "file": "Daylight_spectra_with diffuser.csv",
        "sig_col": 3,
        "color": DAY_COLOR,
        "stem": "illuminant_daylight_diffuser",
        "title": "Daylight + diffuser",
    },
]


def load_illuminant(csv_path: Path, sig_col: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (wavelength_nm, normalised_net_signal) cropped to WL_MIN–WL_MAX."""
    df = pd.read_csv(csv_path)
    wl = df.iloc[:, 0].to_numpy(dtype=float)
    drk = df.iloc[:, 1].to_numpy(dtype=float)
    sig = df.iloc[:, sig_col].to_numpy(dtype=float)

    net = sig - drk
    mask = (wl >= WL_MIN) & (wl <= WL_MAX)
    wl = wl[mask]
    net = net[mask]

    peak = net.max()
    if peak > 0:
        net = net / peak
    return wl, net


def plot_illuminant(illum: dict) -> None:
    csv_path = ILLUM_DIR / illum["file"]
    wl, net = load_illuminant(csv_path, illum["sig_col"])

    fig, ax = plt.subplots(figsize=(COL_W, SPEC_H))

    ax.plot(wl, net, color=illum["color"], lw=0.8, alpha=0.9)

    # mark analytical wavelengths
    for wl_mark, label in MARKERS_NM.items():
        if WL_MIN <= wl_mark <= WL_MAX:
            ax.axvline(wl_mark, color=FIT_COLOR, lw=0.8, ls="--", alpha=0.7)
            pct = float(np.interp(wl_mark, wl, net)) * 100
            ax.text(
                wl_mark + 4,
                0.96,
                rf"{label}  {pct:.1f}%",
                color=FIT_COLOR,
                fontsize=6,
                va="top",
                ha="left",
            )

    ax.set_xlabel(r"Wavelength (nm)")
    ax.set_ylabel(r"Relative spectral power (arb. u.)")
    ax.set_title(illum["title"], fontsize=7)
    ax.set_xlim(WL_MIN, WL_MAX)
    ax.set_ylim(-0.02, 1.08)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5)

    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{illum['stem']}.{ext}", dpi=600, bbox_inches="tight", format=ext)
    plt.close(fig)

    # companion CSV
    csv_out = pd.DataFrame({"wavelength_nm": wl, "normalised_spectral_power": net})
    csv_out.to_csv(OUT_DIR / f"{illum['stem']}.csv", index=False)
    print(
        f"  saved {illum['stem']}  (420 nm: {float(np.interp(420, wl, net)) * 100:.1f}%,"
        f"  665 nm: {float(np.interp(665, wl, net)) * 100:.1f}%)"
    )


def main() -> None:
    print(f"Output -> {OUT_DIR}")
    for illum in ILLUMINANTS:
        plot_illuminant(illum)
    print("Done.")


if __name__ == "__main__":
    main()
