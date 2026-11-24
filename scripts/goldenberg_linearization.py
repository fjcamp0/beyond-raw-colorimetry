"""Goldenberg (1954) A/C vs. A linearization test.

Goldenberg (Anal. Chem. 26, 690-693, 1954) showed empirically that for
broadband instruments, where A vs. C is measurably curved, plotting A/C
(y-axis) vs. A (x-axis) rectifies the curve to a line: C = A/(k - mA), i.e.
A/C = k - mA. This project already explains the same broadband-curvature
phenomenon mechanistically (finite-bandpass / f_tail); this script asks
whether Goldenberg's purely empirical 1954 fix reproduces on this project's
own image-derived data and on its physics-based forward model.

Axis convention is A/C vs. A (not A/C vs. C — a different, more common
Beer's-law-deviation check). Scope: ferricyanide B channel, methylene blue
R channel (each analyte's primary absorbing channel). See
Manuscript/_ai_outputs/handoff_goldenberg_linearization.md for full context
and Manuscript/_ai_outputs/goldenberg_linearization_findings.md for the
write-up of these results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "Data" / "analysis" / "goldenberg_linearization"

FERRI_DIR = BASE / "Data" / "analysis" / "ferricyanide"
MB_DIR = BASE / "Data" / "analysis" / "methylene_blue"

ANALYTES = {
    "ferricyanide": {
        "channel": "B",
        "real_csv": FERRI_DIR / "beer_lambert" / "absorbance_table.csv",
        "model_csv": FERRI_DIR / "forward_model" / "forward_model_table.csv",
        "model_col": "A_B",
        "color": "#D4820A",  # FERIC_COLOR, figure_style.md
        "label": r"K$_3$[Fe(CN)$_6$], B channel",
    },
    "methylene_blue": {
        "channel": "R",
        "real_csv": MB_DIR / "beer_lambert" / "absorbance_table.csv",
        "model_csv": MB_DIR / "forward_model" / "forward_model_table.csv",
        "model_col": "A_R",
        "color": "#7D3C98",  # MB_COLOR, figure_style.md
        "label": "Methylene blue, R channel",
    },
}

# Multi-set colors/markers, figure_style.md
SET_STYLE = {
    "Set1": dict(color="#2E86AB", marker="o"),
    "Set2": dict(color="#A23B72", marker="s"),
    "Set3": dict(color="#F18F01", marker="^"),
}
FIT_COLOR = "#2C3E50"  # figure_style.md


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS with intercept. Returns slope, intercept, R^2."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    design = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2


def load_real(csv_path: Path, channel: str) -> pd.DataFrame:
    """Load the per-image absorbance table filtered to one Bayer channel, with A/C added."""
    df = pd.read_csv(csv_path)
    df = df[df["channel"] == channel].copy()
    df["ac_ratio"] = df["absorbance"] / df["concentration_um"]
    return df[["set", "concentration_um", "absorbance", "ac_ratio"]].reset_index(drop=True)


def load_model(csv_path: Path, model_col: str, c_min: float, c_max: float) -> pd.DataFrame:
    """Load the forward-model table, restricted to the real-data concentration range."""
    df = pd.read_csv(csv_path)
    df = df[(df["concentration_um"] >= c_min) & (df["concentration_um"] <= c_max)].copy()
    df["absorbance"] = df[model_col]
    df["ac_ratio"] = df["absorbance"] / df["concentration_um"]
    df = df[["concentration_um", "absorbance", "ac_ratio"]].sort_values("concentration_um")
    return df.reset_index(drop=True)


def main() -> None:
    """Run the Goldenberg A/C vs. A test for both analytes and write figure + CSV + summary."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.8,
            "lines.markersize": 3.5,
            "patch.linewidth": 0.5,
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.00, 3.00))
    all_rows: list[pd.DataFrame] = []
    summary_lines: list[str] = []

    for ax, (name, cfg) in zip(axes, ANALYTES.items()):
        real = load_real(cfg["real_csv"], cfg["channel"])
        c_min, c_max = real["concentration_um"].min(), real["concentration_um"].max()
        model = load_model(cfg["model_csv"], cfg["model_col"], c_min, c_max)

        summary_lines.append(f"\n=== {name} ({cfg['channel']} channel) ===")

        # per-set fits: sensitivity check on whether pooling across sets is justified
        for set_name, grp in real.groupby("set"):
            s, i, r2 = ols(grp["absorbance"].to_numpy(), grp["ac_ratio"].to_numpy())
            summary_lines.append(
                f"  {set_name}: A/C-vs-A slope={s:+.5e}  intercept={i:+.5e}  "
                f"R2={r2:.5f}  n={len(grp)}"
            )

        # pooled fit (all sets concatenated) -- primary result
        s_pool, i_pool, r2_pool = ols(real["absorbance"].to_numpy(), real["ac_ratio"].to_numpy())
        summary_lines.append(
            f"  POOLED: A/C-vs-A slope={s_pool:+.5e}  intercept={i_pool:+.5e}  "
            f"R2={r2_pool:.5f}  n={len(real)}"
        )

        # raw A vs C linear fit, for curvature contrast (known nonlinear)
        _, _, r2_ac = ols(real["concentration_um"].to_numpy(), real["absorbance"].to_numpy())
        summary_lines.append(f"  contrast: raw A-vs-C linear-fit R2={r2_ac:.5f}")

        # forward model, same transform, same concentration window
        s_mod, i_mod, r2_mod = ols(model["absorbance"].to_numpy(), model["ac_ratio"].to_numpy())
        _, _, r2_mod_ac = ols(model["concentration_um"].to_numpy(), model["absorbance"].to_numpy())
        summary_lines.append(
            f"  MODEL: A/C-vs-A slope={s_mod:+.5e}  intercept={i_mod:+.5e}  R2={r2_mod:.5f}"
        )
        summary_lines.append(f"  contrast: model raw A-vs-C linear-fit R2={r2_mod_ac:.5f}")

        # ---- plot ----
        for set_name, grp in real.groupby("set"):
            style = SET_STYLE.get(set_name, dict(color="gray", marker="x"))
            ax.scatter(grp["absorbance"], grp["ac_ratio"], s=12, alpha=0.7, label=set_name, **style)
        xs = np.linspace(real["absorbance"].min(), real["absorbance"].max(), 50)
        ax.plot(
            xs,
            s_pool * xs + i_pool,
            color=FIT_COLOR,
            lw=0.9,
            ls="--",
            label=rf"pooled fit ($R^2$={r2_pool:.3f})",
        )
        ax.plot(
            model["absorbance"],
            model["ac_ratio"],
            color=cfg["color"],
            lw=1.0,
            label="forward model",
        )

        ax.set_xlabel("Absorbance, A (AU)")
        ax.set_ylabel(r"A/C ($\mu$M$^{-1}$)")
        ax.set_title(cfg["label"], fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")

        real_out = real.copy()
        real_out["analyte"] = name
        real_out["source"] = "real"
        model_out = model.copy()
        model_out["analyte"] = name
        model_out["source"] = "model"
        model_out["set"] = "model"
        all_rows.append(real_out)
        all_rows.append(model_out)

    fig.tight_layout(pad=0.5)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"goldenberg_ac_vs_a.{ext}", dpi=600, bbox_inches="tight", format=ext)

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined[["analyte", "source", "set", "concentration_um", "absorbance", "ac_ratio"]]
    combined.to_csv(OUT_DIR / "goldenberg_ac_vs_a_data.csv", index=False, float_format="%.6g")

    summary_text = "\n".join(summary_lines) + "\n"
    (OUT_DIR / "goldenberg_ac_vs_a_summary.txt").write_text(summary_text)
    print(summary_text)
    print(f"Wrote {OUT_DIR / 'goldenberg_ac_vs_a.png'} (+ .svg)")
    print(f"Wrote {OUT_DIR / 'goldenberg_ac_vs_a_data.csv'}")


if __name__ == "__main__":
    main()
