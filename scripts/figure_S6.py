"""
Channel-common absorbance component diagnostics, LED ferricyanide dataset
(n = 3 acquisition sets).

Three independent panels:
  S13a  Residual scatter plots (B vs G, B vs R, G vs R after subtracting conc. mean)
  S13b  Across-set SD collapse: SD(B) vs SD(BdivG) per concentration
  S13c  Nested regression R² comparison

ACS single-column spec: 3.25 in x 3.0 in per panel, Arial 8 pt, 600 dpi.
Ticks inside, all four sides, no panel titles.

Run from project root: python scripts/figure_S6.py
"""

import pathlib
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_BL = PROJECT_ROOT / "Data/analysis/ferricyanide/beer_lambert"
DATA_FM = PROJECT_ROOT / "Data/analysis/ferricyanide/forward_model"
OUT_DIR = PROJECT_ROOT / "Data" / "paper_figures" / "figure_S6" / "LED"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ACS style
# ---------------------------------------------------------------------------
rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    }
)

PANEL_W = 3.25  # inches
PANEL_H = 3.00  # inches
DPI = 600

# RAW blues, dark to light: plain B, B/G, B/R
ORANGE, BLUE, GREEN = fs.ROUTE_COLORS["raw"]


def save_panel(fig, stem):
    """Save PNG, TIFF and SVG for a figure."""
    for ext in ("png", "tiff", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    print(f"  Saved {stem}.png, {stem}.tiff and {stem}.svg")


# ===========================================================================
# Load data
# ===========================================================================
abs_tbl = pd.read_csv(DATA_BL / "absorbance_table.csv")
cal_tbl = pd.read_csv(DATA_BL / "calibration_table.csv")
fwd_tbl = pd.read_csv(DATA_FM / "forward_model_table.csv")


def pivot_per_set(df):
    """Return wide DataFrame indexed by (set, concentration_um) with B/G1/G2/Gavg/R/BdivG."""
    keep_channels = {"B", "G1", "G2", "R", "BdivG"}
    d = df[df["channel"].isin(keep_channels)].copy()
    wide = d.pivot_table(
        index=["set", "concentration_um"],
        columns="channel",
        values="absorbance",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide["Gavg"] = (wide["G1"] + wide["G2"]) / 2.0
    return wide


wide = pivot_per_set(abs_tbl)

# Restrict to the three LED acquisition sets. The set label follows the source folder
# name, which changed from Set1/2/3 to led_set1/2/3 when these tables were regenerated
# by s02_calibrate; match on the trailing digit so either spelling works.
n3_sets = sorted(s for s in wide["set"].unique() if str(s).rstrip()[-1] in "123")
if len(n3_sets) != 3:
    sys.exit(f"figure_S6: expected 3 acquisition sets, found {n3_sets}")
wide3 = wide[wide["set"].isin(n3_sets)].copy()

conc_all = sorted(wide3["concentration_um"].unique())
conc_4k = [c for c in conc_all if c <= 4000.0]  # SD collapse analysis range


# ===========================================================================
# PANEL A — Residual scatter plots
# ===========================================================================
print("\n--- Panel A: Residual scatter plots ---")


def compute_residuals(df_wide, concs):
    """Subtract the per-concentration mean (B, Gavg, R) to yield set-to-set residuals."""
    records = []
    for c in concs:
        sub = df_wide[df_wide["concentration_um"] == c].copy()
        mean_B = sub["B"].mean()
        mean_Gavg = sub["Gavg"].mean()
        mean_R = sub["R"].mean()
        for _, row in sub.iterrows():
            records.append(
                {
                    "set": row["set"],
                    "concentration_um": c,
                    "resid_B": row["B"] - mean_B,
                    "resid_G": row["Gavg"] - mean_Gavg,
                    "resid_R": row["R"] - mean_R,
                }
            )
    return pd.DataFrame(records)


resid_df = compute_residuals(wide3, conc_all)

r_BG, p_BG = stats.pearsonr(resid_df["resid_B"], resid_df["resid_G"])
r_BR, p_BR = stats.pearsonr(resid_df["resid_B"], resid_df["resid_R"])
r_GR, p_GR = stats.pearsonr(resid_df["resid_G"], resid_df["resid_R"])

print(f"  B vs Gavg: r = {r_BG:.4f}")
print(f"  B vs R:    r = {r_BR:.4f}")
print(f"  Gavg vs R: r = {r_GR:.4f}")

# Color points by concentration (log scale for visibility)
log_conc = np.log10(resid_df["concentration_um"])
norm_lc = (log_conc - log_conc.min()) / (log_conc.max() - log_conc.min())
cmap_pts = matplotlib.colors.ListedColormap(fs.conc_colors(256))

fig_a, axes_a = plt.subplots(1, 3, figsize=(PANEL_W * 3, PANEL_H))

panels = [
    ("resid_B", "resid_G", r"Residual $A_B$ (AU)", r"Residual $A_G$ (AU)", r_BG),
    ("resid_B", "resid_R", r"Residual $A_B$ (AU)", r"Residual $A_R$ (AU)", r_BR),
    ("resid_G", "resid_R", r"Residual $A_G$ (AU)", r"Residual $A_R$ (AU)", r_GR),
]

for ax, (xcol, ycol, xlabel, ylabel, r_val) in zip(axes_a, panels):
    sc = ax.scatter(
        resid_df[xcol],
        resid_df[ycol],
        c=norm_lc,
        cmap=cmap_pts,
        s=18,
        alpha=0.85,
        linewidths=0.3,
        edgecolors="k",
        zorder=3,
    )
    # 1:1 identity line
    lims = [
        min(resid_df[xcol].min(), resid_df[ycol].min()) * 1.1,
        max(resid_df[xcol].max(), resid_df[ycol].max()) * 1.1,
    ]
    ax.plot(lims, lims, "k--", lw=0.7, alpha=0.5, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.annotate(
        f"$r$ = {r_val:.4f}",
        xy=(0.05, 0.93),
        xycoords="axes fraction",
        fontsize=7,
        ha="left",
    )
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.set_xlim(lims)
    ax.set_ylim(lims)

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap=cmap_pts, norm=plt.Normalize(log_conc.min(), log_conc.max()))
sm.set_array([])
cb = fig_a.colorbar(sm, ax=axes_a, fraction=0.015, pad=0.04)
cb.set_label(r"log$_{10}$[$c$ / µM]", fontsize=7)
cb.ax.tick_params(labelsize=7)

fig_a.tight_layout()
save_panel(fig_a, "figure_S6a")
plt.close(fig_a)

# Save CSV
resid_csv = resid_df.rename(columns={"resid_G": "resid_Gavg"})[
    ["set", "concentration_um", "resid_B", "resid_Gavg", "resid_R"]
]
resid_csv.to_csv(OUT_DIR / "figure_S6a.csv", index=False)
print(f"  Saved figure_S6a.csv  ({len(resid_csv)} rows)")


# ===========================================================================
# PANEL B — Across-set SD collapse (concentrations up to 4000 µM)
# ===========================================================================
print("\n--- Panel B: SD collapse ---")

# Compute per-concentration SD from the calibration table (n=3 rows only)
cal3 = cal_tbl[cal_tbl["n_replicates"] == 3].copy()

# Build lookup: concentration -> SD for B, BdivG, R (as proxy for BdivR)
sd_rows = []
for c in conc_4k:
    row_B = cal3[(cal3["concentration_um"].round(3) == round(c, 3)) & (cal3["channel"] == "B")]
    row_BdivG = cal3[
        (cal3["concentration_um"].round(3) == round(c, 3)) & (cal3["channel"] == "BdivG")
    ]
    row_R = cal3[(cal3["concentration_um"].round(3) == round(c, 3)) & (cal3["channel"] == "R")]

    if row_B.empty or row_BdivG.empty or row_R.empty:
        continue

    sd_B = float(row_B["absorbance_std"].values[0])
    sd_BdivG = float(row_BdivG["absorbance_std"].values[0])
    sd_R = float(row_R["absorbance_std"].values[0])

    # BdivR absorbance = -log10(T_B/T_R) = A_B - A_R, same logic the pipeline
    # uses for BdivG = A_B - A_Gavg; computed here from per-set wide data.
    sub = wide3[(wide3["concentration_um"] == c) & (wide3["set"].isin(n3_sets))].copy()
    if len(sub) >= 2:
        BdivR_abs = sub["B"] - sub["R"]
        sd_BdivR = float(BdivR_abs.std(ddof=1))
    else:
        sd_BdivR = np.nan

    sd_rows.append(
        {
            "concentration_um": c,
            "SD_B": sd_B,
            "SD_BdivG": sd_BdivG,
            "SD_BdivR": sd_BdivR,
        }
    )

sd_df = pd.DataFrame(sd_rows)

valid_mask = sd_df["SD_BdivG"] > 0
ratio_BG = sd_df.loc[valid_mask, "SD_B"] / sd_df.loc[valid_mask, "SD_BdivG"]
ratio_BR = sd_df.loc[valid_mask, "SD_B"] / sd_df.loc[valid_mask, "SD_BdivR"]
med_BG = float(ratio_BG.median())
med_BR = float(ratio_BR.median())
print(f"  Median SD(B)/SD(BdivG) = {med_BG:.1f}×")
print(f"  Median SD(B)/SD(BdivR) = {med_BR:.1f}×")

fig_b, ax_b = plt.subplots(figsize=(PANEL_W, PANEL_H))

x_vals = sd_df["concentration_um"]
ax_b.semilogy(
    x_vals,
    sd_df["SD_B"],
    color=BLUE,
    marker="o",
    ms=5,
    lw=1.1,
    label=r"$\sigma(A_B)$",
    zorder=3,
)
ax_b.semilogy(
    x_vals,
    sd_df["SD_BdivG"],
    color=ORANGE,
    marker="s",
    ms=5,
    lw=1.1,
    label=r"$\sigma(A_{B/G})$",
    zorder=3,
)
ax_b.semilogy(
    x_vals,
    sd_df["SD_BdivR"],
    color=GREEN,
    marker="^",
    ms=5,
    lw=1.1,
    label=r"$\sigma(A_{B/R})$",
    zorder=3,
)

ax_b.set_xlabel(r"Concentration (µM)")
ax_b.set_ylabel(r"Across-set SD (AU)")
ax_b.legend(loc="lower right", frameon=True)
ax_b.tick_params(which="both", direction="in", top=True, right=True)
ax_b.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

# Annotate median ratios
ax_b.annotate(
    f"Median $\\sigma_B/\\sigma_{{B/G}} \\approx {med_BG:.0f}\\times$\n"
    f"Median $\\sigma_B/\\sigma_{{B/R}} \\approx {med_BR:.0f}\\times$",
    xy=(0.04, 0.97),
    xycoords="axes fraction",
    fontsize=6.5,
    va="top",
    ha="left",
)

fig_b.tight_layout()
save_panel(fig_b, "figure_S6b")
plt.close(fig_b)

sd_df.to_csv(OUT_DIR / "figure_S6b.csv", index=False)
print(f"  Saved figure_S6b.csv  ({len(sd_df)} rows)")


# ===========================================================================
# PANEL C — Nested regression R² comparison
# ===========================================================================
print("\n--- Panel C: Nested R² comparison ---")

# Use mean data from calibration_table.csv, range 7.8–1000 µM, n=3 only
cal3_1k = cal3[cal3["concentration_um"] <= 1000.0].copy()


def get_mean_col(channel):
    """Return sorted (concentration_um, absorbance_mean) rows for one channel."""
    sub = (
        cal3_1k[cal3_1k["channel"] == channel][["concentration_um", "absorbance_mean"]]
        .sort_values("concentration_um")
        .reset_index(drop=True)
    )
    return sub


B_mean = get_mean_col("B")
G1_mean = get_mean_col("G1")
G2_mean = get_mean_col("G2")
BdivG_mean = get_mean_col("BdivG")

merged = B_mean.rename(columns={"absorbance_mean": "B_obs"})
merged["G1_obs"] = G1_mean["absorbance_mean"].values
merged["G2_obs"] = G2_mean["absorbance_mean"].values
merged["Gavg_obs"] = (merged["G1_obs"] + merged["G2_obs"]) / 2.0
merged["BdivG_obs"] = BdivG_mean["absorbance_mean"].values
merged["c_um"] = merged["concentration_um"]


def fm_interp(col, c_um_array):
    """Linearly interpolate forward-model column to given concentrations."""
    return np.interp(c_um_array, fwd_tbl["concentration_um"], fwd_tbl[col])


merged["model_B"] = fm_interp("A_B", merged["c_um"].values)
merged["model_BdivG"] = fm_interp("A_BdivG", merged["c_um"].values)

# ---- (1) B_obs ~ c (linear) ----
x1 = merged["c_um"].values.reshape(-1, 1)
y = merged["B_obs"].values
lr1 = LinearRegression().fit(x1, y)
r2_1 = lr1.score(x1, y)
print(f"  (1) B ~ c:                R² = {r2_1:.6f}")

# ---- (2) B_obs ~ model_B ----
x2 = merged["model_B"].values.reshape(-1, 1)
lr2 = LinearRegression().fit(x2, y)
r2_2 = lr2.score(x2, y)
coef2 = lr2.coef_[0]
int2 = lr2.intercept_
print(f"  (2) B ~ model_B:          R² = {r2_2:.6f}  coef={coef2:.3f}  int={int2:.5f}")

# ---- (3) B_obs ~ model_B + Gavg_obs ----
x3 = np.column_stack([merged["model_B"].values, merged["Gavg_obs"].values])
lr3 = LinearRegression().fit(x3, y)
r2_3 = lr3.score(x3, y)
resid3 = y - lr3.predict(x3)
rmse3 = np.sqrt(np.mean(resid3**2))
print(f"  (3) B ~ model_B + Gavg:   R² = {r2_3:.6f}  RMSE = {rmse3:.6f} AU")

# ---- (4) BdivG_obs ~ model_BdivG ----
y4 = merged["BdivG_obs"].values
x4 = merged["model_BdivG"].values.reshape(-1, 1)
lr4 = LinearRegression().fit(x4, y4)
r2_4 = lr4.score(x4, y4)
print(f"  (4) BdivG ~ model_BdivG:  R² = {r2_4:.6f}")

model_labels = [
    r"$A_B \sim c$",
    r"$A_B \sim A_{B,\mathrm{model}}$",
    r"$A_B \sim A_{B,\mathrm{model}} + A_{G,\mathrm{obs}}$",
    r"$A_{B/G} \sim A_{B/G,\mathrm{model}}$",
]
r2_vals = [r2_1, r2_2, r2_3, r2_4]

fig_c, ax_c = plt.subplots(figsize=(PANEL_W, PANEL_H))

bar_colors = [BLUE, BLUE, ORANGE, GREEN]
bar_alpha = [1.0, 1.0, 1.0, 1.0]
y_pos = np.arange(len(model_labels))

bars = ax_c.barh(
    y_pos,
    r2_vals,
    color=[matplotlib.colors.to_rgba(c, a) for c, a in zip(bar_colors, bar_alpha)],
    height=0.55,
    edgecolor="k",
    linewidth=0.5,
)

# Annotate R² values on bars
for bar, val in zip(bars, r2_vals):
    xpos = val - 0.002
    ax_c.text(
        xpos,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center",
        ha="right",
        fontsize=7,
        color="white",
        fontweight="bold",
    )

ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(model_labels, fontsize=7.5)
ax_c.set_xlabel(r"$R^2$")
ax_c.set_xlim(0.88, 1.004)
ax_c.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
ax_c.tick_params(which="both", direction="in", top=True, right=True)

# Annotate RMSE on bar 3
ax_c.annotate(
    f"RMSE = {rmse3:.5f} AU",
    xy=(r2_3, 2),
    xytext=(r2_3 - 0.045, 2.35),
    fontsize=6.5,
    arrowprops=dict(arrowstyle="-", lw=0.6),
)

fig_c.tight_layout()
save_panel(fig_c, "figure_S6c")
plt.close(fig_c)

# CSV
r2_csv = pd.DataFrame(
    {
        "model_label": [
            "B_obs ~ c",
            "B_obs ~ model_B",
            "B_obs ~ model_B + Gavg_obs",
            "BdivG_obs ~ model_BdivG",
        ],
        "R2": r2_vals,
    }
)
r2_csv.to_csv(OUT_DIR / "figure_S6c.csv", index=False)
print("  Saved figure_S6c.csv")


# ===========================================================================
# ASSEMBLED figure — 3-panel row
# ===========================================================================
print("\n--- Assembled figure ---")

fig_asm, axes_asm = plt.subplots(
    1,
    4,
    figsize=(PANEL_W * 3 + 2.0, PANEL_H),  # leave room for colorbar
    gridspec_kw={"width_ratios": [1, 1, 1, 0.05]},
)

# Reproduce Panel A subplots in axes_asm[0..1] but compressed
panel_a_specs = [
    ("resid_B", "resid_G", r"Residual $A_B$", r"Residual $A_G$", r_BG),
    ("resid_B", "resid_R", r"Residual $A_B$", r"Residual $A_R$", r_BR),
    ("resid_G", "resid_R", r"Residual $A_G$", r"Residual $A_R$", r_GR),
]

# Embed all 3 scatter subpanels in a 3×1 split of axes_asm[0] via a dedicated
# gridspec for finer control; remove the plain subplot axes first.
fig_asm.delaxes(axes_asm[0])
fig_asm.delaxes(axes_asm[1])
fig_asm.delaxes(axes_asm[2])
fig_asm.delaxes(axes_asm[3])

fig_full = plt.figure(figsize=(PANEL_W * 2 + 0.6, PANEL_H * 2 + 0.2))
gs = GridSpec(
    2,
    3,
    figure=fig_full,
    width_ratios=[1, 1, 0.06],
    wspace=0.42,
    hspace=0.38,
)

# 2x2 layout keeps the assembled composite near SI page width, so LaTeX no
# longer has to shrink it (and its 8pt fonts) to fit -- see item #15.
axes_scatter = [
    fig_full.add_subplot(gs[0, 0]),
    fig_full.add_subplot(gs[0, 1]),
    fig_full.add_subplot(gs[1, 0]),
]
ax_sd = fig_full.add_subplot(gs[1, 1])
ax_cbar = fig_full.add_subplot(gs[:, 2])

# --- Scatter subpanels ---
for ax, (xcol, ycol, xlabel, ylabel, r_val) in zip(axes_scatter, panel_a_specs):
    sc = ax.scatter(
        resid_df[xcol],
        resid_df[ycol],
        c=norm_lc,
        cmap=cmap_pts,
        s=14,
        alpha=0.85,
        linewidths=0.25,
        edgecolors="k",
        zorder=3,
    )
    lims = [
        min(resid_df[xcol].min(), resid_df[ycol].min()) * 1.15,
        max(resid_df[xcol].max(), resid_df[ycol].max()) * 1.15,
    ]
    ax.plot(lims, lims, "k--", lw=0.6, alpha=0.45, zorder=2)
    ax.set_xlabel(xlabel + " (AU)", fontsize=9)
    ax.set_ylabel(ylabel + " (AU)", fontsize=9)
    ax.annotate(
        f"$r$ = {r_val:.4f}",
        xy=(0.05, 0.92),
        xycoords="axes fraction",
        fontsize=8.5,
        ha="left",
    )
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

sm_asm = plt.cm.ScalarMappable(cmap=cmap_pts, norm=plt.Normalize(log_conc.min(), log_conc.max()))
sm_asm.set_array([])
cb_asm = fig_full.colorbar(sm_asm, cax=ax_cbar)
cb_asm.set_label(r"log$_{10}$[$c$ / µM]", fontsize=8.5)
cb_asm.ax.tick_params(labelsize=8.5)

# --- SD collapse subplot ---
ax_sd.semilogy(x_vals, sd_df["SD_B"], color=BLUE, marker="o", ms=4, lw=1.0, label=r"$\sigma(A_B)$")
ax_sd.semilogy(
    x_vals,
    sd_df["SD_BdivG"],
    color=ORANGE,
    marker="s",
    ms=4,
    lw=1.0,
    label=r"$\sigma(A_{B/G})$",
)
ax_sd.semilogy(
    x_vals,
    sd_df["SD_BdivR"],
    color=GREEN,
    marker="^",
    ms=4,
    lw=1.0,
    label=r"$\sigma(A_{B/R})$",
)
ax_sd.set_xlabel(r"Concentration (µM)", fontsize=9)
ax_sd.set_ylabel(r"Across-set SD (AU)", fontsize=9)
ax_sd.legend(loc="lower right", fontsize=8, frameon=True)
ax_sd.tick_params(which="both", direction="in", top=True, right=True, labelsize=9)
ax_sd.annotate(
    f"Median\n$\\sigma_B/\\sigma_{{B/G}} \\approx {med_BG:.0f}\\times$\n"
    f"$\\sigma_B/\\sigma_{{B/R}} \\approx {med_BR:.0f}\\times$",
    xy=(0.04, 0.97),
    xycoords="axes fraction",
    fontsize=8.5,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85),
)

plt.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.15)

for ax, letter in zip(axes_scatter + [ax_sd], "abcd"):
    ax.text(
        -0.22,
        1.02,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

for ext in ("png", "tiff", "svg"):
    fig_full.savefig(OUT_DIR / f"figure_S6_assembled.{ext}", dpi=DPI, bbox_inches="tight")
print("  Saved figure_S6_assembled.{png,tiff,svg}")
plt.close(fig_full)
plt.close("all")


# ===========================================================================
# Summary
# ===========================================================================
print("\n=== figure_S6.py complete ===")
print(f"  Output directory: {OUT_DIR}")
print("  Files written:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"    {f.name}  ({f.stat().st_size // 1024} kB)")
