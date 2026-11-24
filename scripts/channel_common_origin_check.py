# Copyright 2024 F. Javier del Campo
# Licensed under the Apache License, Version 2.0

"""
channel_common_origin_check.py

Tests the two candidate origins for the ferricyanide LED channel-common
absorbance component (manuscript section 4.3) that are checkable against
data already on file, without a new acquisition:

  1. Temporal drift — per-image blank-to-sample capture time gap (parsed from
     the DNG filename timestamp, IMG_YYMMDD_HHMMSS) vs. the channel-common
     residual. A drift origin predicts a positive correlation.
  2. Lens vignetting coupled to positional shifts — the channel-common
     residual recomputed separately in a center sub-ROI and an edge (annular)
     sub-ROI of the existing calibration ROI. Vignetting predicts larger
     residual variation at the edge (steeper radial falloff there); pure
     sensor/temporal drift predicts comparable center/edge variation, since
     it acts uniformly across the frame.

Both diagnostics use the channel-common residual as defined in section 4.3:
per concentration, subtract the cross-set mean absorbance (nonparametric
detrend) from each image's absorbance, per raw channel (R, G1, G2, B), then
average the four channel residuals — isolating the shared, concentration-
independent component from the channel-specific Beer-Lambert-Bayer signal.

Uses the ferricyanide calibration ROI, i.e. the fixed-area, minimum-block-CV window
reported in Supplementary Figure S1 and used by every ferricyanide number in the
manuscript since the 2026-07-25 ROI unification: viewer (180, 400, 700, 1000).

Outputs (Data/analysis/ferricyanide/channel_common_origin/):
    channel_common_origin_data.csv   — per-image absorbances and residuals
    channel_common_origin_report.md  — correlation/variance results
    drift_scatter.png                — time gap vs. channel-common residual
    radial_comparison.png            — center vs. edge residual comparison
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Vendored raw-image-analysis modules (vendor/, see README); override with COLORIMETRY_VENDOR.
_RIA_DIR = Path(
    os.environ.get("COLORIMETRY_VENDOR", str(Path(__file__).resolve().parent.parent / "vendor"))
)
sys.path.insert(0, str(_RIA_DIR))

from core import images, naming  # noqa: E402

BASE = Path(__file__).resolve().parent.parent / "Data"
IMG_DIR = BASE / "images" / "ferricyanide"
OUT_DIR = BASE / "analysis" / "ferricyanide" / "channel_common_origin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SETS = ["led_set1", "led_set2", "led_set3"]
ROI_VIEWER = (180, 400, 700, 1000)  # Supplementary Figure S1 selection, ferricyanide
RAW_CHANNELS = ["R", "G1", "G2", "B"]

TS_RE = re.compile(r"(\d{6})_(\d{6})")


def parse_timestamp(stem: str) -> float | None:
    """Epoch seconds from an IMG_YYMMDD_HHMMSS filename fragment, or None."""
    m = TS_RE.search(stem)
    if not m:
        return None
    dt = datetime.strptime("".join(m.groups()), "%y%m%d%H%M%S")
    return dt.timestamp()


def region_means(
    bayer: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    black_levels: list[int],
    white_level: int,
    color_desc: str,
) -> dict[str, float]:
    """Per-Bayer-channel means within an arbitrary boolean mask, black-subtracted."""
    ch_names = images.bayer_channel_names(color_desc)
    means: dict[str, float] = {}
    for ch_idx in range(4):
        ch_mask = (colors == ch_idx) & mask
        if not np.any(ch_mask):
            continue
        vals = bayer[ch_mask].astype(np.float64)
        bl = black_levels[ch_idx] if len(black_levels) > ch_idx else 0
        wl_eff = white_level - bl
        vals = np.clip(vals - bl, 0, wl_eff)
        means[ch_names.get(ch_idx, f"CH{ch_idx}")] = float(np.mean(vals))
    return means


def read_full_center_edge(path: Path, roi_viewer: images.ROI) -> tuple[dict, dict, dict]:
    """One rawpy decode; returns (full, center, edge) channel-mean dicts.

    center = concentric 50%x50% (by side length) sub-rectangle of the ROI;
    edge = the remaining annular frame of the ROI.
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        bayer = raw.raw_image_visible.copy()
        colors = raw.raw_colors_visible
        flip = int(raw.sizes.flip)
        white_level = int(raw.white_level)
        black_levels = [int(v) for v in raw.black_level_per_channel]
        color_desc = raw.color_desc.decode("ascii") if raw.color_desc else "RGGB"

    rx, ry, rw, rh = images.viewer_roi_to_bayer(roi_viewer, flip, bayer.shape)
    crop_b = bayer[ry : ry + rh, rx : rx + rw]
    crop_c = colors[ry : ry + rh, rx : rx + rw]

    full_mask = np.ones(crop_b.shape, dtype=bool)
    cy0, cx0 = rh // 4, rw // 4
    ch_, cw_ = rh // 2, rw // 2
    center_mask = np.zeros_like(full_mask)
    center_mask[cy0 : cy0 + ch_, cx0 : cx0 + cw_] = True
    edge_mask = full_mask & ~center_mask

    full = region_means(crop_b, crop_c, full_mask, black_levels, white_level, color_desc)
    center = region_means(crop_b, crop_c, center_mask, black_levels, white_level, color_desc)
    edge = region_means(crop_b, crop_c, edge_mask, black_levels, white_level, color_desc)
    return full, center, edge


def process_set(set_dir: Path) -> list[dict]:
    """Blank-relative absorbance (full/center/edge, per raw channel) and
    blank-to-sample capture time gap for every sample image in one set."""
    files = images.list_images(set_dir)
    blanks = [f for f in files if naming.is_blank(f.stem)]
    samples = [f for f in files if not naming.is_blank(f.stem)]
    if not blanks:
        raise RuntimeError(f"No blank image in {set_dir}")
    blank = blanks[0]
    blank_t = parse_timestamp(blank.stem)
    blank_full, blank_center, blank_edge = read_full_center_edge(blank, ROI_VIEWER)

    rows: list[dict] = []
    for f in sorted(samples):
        conc = naming.parse_concentration_um(f.stem)
        if conc is None:
            continue
        t = parse_timestamp(f.stem)
        gap_s = (t - blank_t) if (t is not None and blank_t is not None) else None
        full, center, edge = read_full_center_edge(f, ROI_VIEWER)

        row: dict = {
            "set": set_dir.name,
            "filename": f.name,
            "concentration_um": conc,
            "gap_seconds": gap_s,
        }
        for region_name, sample_means, blank_means in (
            ("full", full, blank_full),
            ("center", center, blank_center),
            ("edge", edge, blank_edge),
        ):
            for ch in RAW_CHANNELS:
                s_val = sample_means.get(ch)
                b_val = blank_means.get(ch)
                if s_val is not None and b_val and b_val > 0 and s_val > 0:
                    row[f"A_{ch}_{region_name}"] = -np.log10(s_val / b_val)
                else:
                    row[f"A_{ch}_{region_name}"] = np.nan
        rows.append(row)
    print(f"[channel_common_origin] {set_dir.name}: blank={blank.name}, {len(rows)} samples")
    return rows


def channel_common_residual(df: pd.DataFrame, channel_cols: list[str]) -> pd.Series:
    """Per concentration, subtract the cross-set mean absorbance from each
    image (nonparametric detrend), then average the per-channel residuals —
    the channel-common component isolated from the concentration-dependent
    Beer-Lambert-Bayer trend (same logic as section 4.3's residual-
    correlation diagnostic)."""
    resid = pd.DataFrame(index=df.index)
    for col in channel_cols:
        conc_mean = df.groupby("concentration_um")[col].transform("mean")
        resid[col] = df[col] - conc_mean
    return resid.mean(axis=1)


def main() -> None:
    """Run both origin diagnostics and write the CSV, report, and plots."""
    all_rows: list[dict] = []
    for s in SETS:
        all_rows.extend(process_set(IMG_DIR / s))
    df = pd.DataFrame(all_rows)

    df["residual_full"] = channel_common_residual(df, [f"A_{ch}_full" for ch in RAW_CHANNELS])
    df["residual_center"] = channel_common_residual(df, [f"A_{ch}_center" for ch in RAW_CHANNELS])
    df["residual_edge"] = channel_common_residual(df, [f"A_{ch}_edge" for ch in RAW_CHANNELS])

    csv_path = OUT_DIR / "channel_common_origin_data.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"[channel_common_origin] Written: {csv_path}")

    # ── Test 1: temporal drift ────────────────────────────────────────────
    valid = df.dropna(subset=["gap_seconds", "residual_full"])
    r_signed, p_signed = stats.pearsonr(valid["gap_seconds"], valid["residual_full"])
    r_abs, p_abs = stats.pearsonr(valid["gap_seconds"], valid["residual_full"].abs())

    per_set_corr = {}
    for s, grp in valid.groupby("set"):
        if len(grp) >= 3:
            r_s, p_s = stats.pearsonr(grp["gap_seconds"], grp["residual_full"])
            per_set_corr[s] = (r_s, p_s, len(grp))

    # ── Test 2: radial (center vs. edge) comparison ───────────────────────
    valid2 = df.dropna(subset=["residual_center", "residual_edge"])
    sd_center = float(valid2["residual_center"].std(ddof=1))
    sd_edge = float(valid2["residual_edge"].std(ddof=1))
    var_ratio = (sd_edge**2) / (sd_center**2) if sd_center > 0 else float("nan")
    f_stat, f_p = stats.levene(valid2["residual_center"], valid2["residual_edge"])
    r_ce, p_ce = stats.pearsonr(valid2["residual_center"], valid2["residual_edge"])
    diff = valid2["residual_edge"].abs() - valid2["residual_center"].abs()
    t_stat, t_p = stats.ttest_1samp(diff, 0.0)

    # ── Plots ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5))
    markers = {"led_set1": "o", "led_set2": "s", "led_set3": "^"}
    for s, grp in valid.groupby("set"):
        ax.scatter(
            grp["gap_seconds"] / 60.0,
            grp["residual_full"],
            marker=markers.get(s, "x"),
            label=s,
            alpha=0.8,
        )
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Blank→sample capture time gap (min)")
    ax.set_ylabel("Channel-common residual (AU)")
    ax.set_title(f"Temporal drift test: r={r_signed:.3f} (p={p_signed:.3g}), n={len(valid)}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "drift_scatter.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].scatter(valid2["residual_center"], valid2["residual_edge"], alpha=0.7)
    lim = max(valid2["residual_center"].abs().max(), valid2["residual_edge"].abs().max()) * 1.1
    axes[0].plot([-lim, lim], [-lim, lim], color="grey", lw=0.8, ls="--")
    axes[0].set_xlabel("Center-ROI residual (AU)")
    axes[0].set_ylabel("Edge-ROI residual (AU)")
    axes[0].set_title(f"r={r_ce:.3f}")
    axes[1].bar(
        ["center", "edge"],
        [sd_center, sd_edge],
        color=["#1f77b4", "#d62728"],
    )
    axes[1].set_ylabel("SD of channel-common residual (AU)")
    axes[1].set_title(f"Variance ratio (edge/center) = {var_ratio:.2f}")
    fig.suptitle("Radial (vignetting) test")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "radial_comparison.png", dpi=150)
    plt.close(fig)

    # ── Report ─────────────────────────────────────────────────────────────
    lines = [
        "# Channel-common component origin check",
        "",
        "*Generated by `scripts/channel_common_origin_check.py` — tests the two candidate",
        "origins named in manuscript section 4.3 against existing data (no new acquisition).*",
        "",
        f"Ferricyanide LED sets: {', '.join(SETS)}. ROI (viewer): {ROI_VIEWER}. "
        f"n = {len(df)} sample images.",
        "",
        "## Test 1 — temporal drift (blank→sample capture time gap)",
        "",
        f"- Pearson r (signed residual vs. gap): **{r_signed:.3f}** "
        f"(p = {p_signed:.3g}, n = {len(valid)})",
        f"- Pearson r (|residual| vs. gap): **{r_abs:.3f}** (p = {p_abs:.3g})",
        "",
        "Per-set (within-session) correlation:",
        "",
        "| Set | r | p | n |",
        "|-----|---|---|---|",
    ]
    for s, (r_s, p_s, n_s) in per_set_corr.items():
        lines.append(f"| {s} | {r_s:.3f} | {p_s:.3g} | {n_s} |")

    lines += [
        "",
        "## Test 2 — radial position (center vs. edge sub-ROI)",
        "",
        f"- SD(residual), center sub-ROI: **{sd_center:.5f}** AU",
        f"- SD(residual), edge sub-ROI: **{sd_edge:.5f}** AU",
        f"- Variance ratio (edge/center): **{var_ratio:.2f}**",
        f"- Levene's test for equal variance: stat = {f_stat:.3f}, p = {f_p:.3g}",
        f"- Pearson r, center vs. edge residual (paired, same images): {r_ce:.3f}",
        f"- Paired t-test, |edge residual| − |center residual|: t = {t_stat:.3f}, p = {t_p:.3g}",
        "",
        "## Interpretation notes (for human review, not a finished conclusion)",
        "",
        "- A drift origin predicts a positive, statistically supported r in Test 1.",
        "- A vignetting/positional-shift origin predicts SD(edge) > SD(center) in Test 2",
        "  (variance ratio > 1, ideally with a significant Levene/t-test), since the macro",
        "  lens's radial falloff is steeper away from the optical axis.",
        "- A high center/edge correlation (r_ce near 1) with a similar center/edge SD is",
        "  consistent with a spatially uniform (sensor/temporal) origin rather than a",
        "  position-dependent (vignetting) one.",
        "",
        "*End of report.*",
    ]
    report_path = OUT_DIR / "channel_common_origin_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[channel_common_origin] Written: {report_path}")
    print(f"[channel_common_origin] Plots -> {OUT_DIR}")


if __name__ == "__main__":
    main()
