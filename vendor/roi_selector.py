# Copyright 2024 F. Javier del Campo
# Licensed under the Apache License, Version 2.0

"""
roi_selector.py — automated ROI selection for s02_calibrate.

Works on RAW (DNG/NEF/...) and sRGB (PNG/JPEG/TIFF) sets alike — the format
is auto-detected from the folder contents.

Finds the most homogeneous rectangular region in a set of images using two
complementary criteria:

  1. Blank CV map   : per-block coefficient of variation in the blank image
                      (detects hotspots, reflections, vignetting edges)
  2. Signal CV map  : per-block CV of blank-corrected channel signal across
                      all sample images (detects unstable measurement regions)

The combined score (normalised sum) is minimised over a sliding window.

Outputs (in --out-dir):
  blank_cv_grid.png         Spatial CV map of the blank
  signal_cv_grid.png        Spatial CV map of blank-corrected signal
  combined_score_roi.png    Composite score map + recommended ROI box
  roi_on_real_image.png     Recommended ROI overlaid on a mid-series image
  rois.yaml                 Ready-to-use ROI in VIEWER coordinates —
                            review, then copy to <project>/image_analysis/

Usage:
    python raw-image-analysis/scripts/roi_selector.py \\
        --img-dir /path/to/Set1/ \\
        --blank "Control*.dng" \\
        --out-dir /path/to/project/image_analysis/roi_selection/ \\
        [--channel B] [--block 100] [--win-h 400] [--win-w 800] [--search-aspect]

Block and window sizes are in viewer pixels (the image as you see it).

Choosing window shape: if win_h (or win_w) is close to or equal to the
grid's full extent in that axis, best_window() has no slack to slide in
that direction and the window is forced to span the entire frame there —
including whatever high-CV edge/vignetting rows or columns exist, even
though a same-area window with a different aspect ratio could avoid them.
CV inhomogeneity from vignetting is typically worse near the frame edges
in both axes, and the best aspect ratio is scene-dependent, not a fixed
default. Pass --search-aspect to have roi_selector fix win_h*win_w as an
area budget and search every divisor-pair aspect ratio that fits the grid
(best_window_optimal_aspect()), instead of trusting the literal win_h/win_w
you passed. (Empirically confirmed on the colorimetry_Project ferricyanide/
methylene blue sets, 2026-07-02: a 10x7-block window forced onto that
project's grid had zero vertical slack and was pinned to both high-CV
frame edges; the equal-area 7x10 shape --search-aspect finds instead
lowered the mean combined score and cleared both edges. See that project's
Manuscript/_drafts/figures/figure_S1/CAPTIONS.md for the worked example.)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import typer
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))  # core/ sits next to this file

from core import images
from core.provenance import write_run_config

# ─── Channel plane loading (one 2-D array per image) ─────────────────────────


def load_channel_plane_raw(path: Path, channel: str) -> tuple[np.ndarray, dict]:
    """Sub-sampled sensor plane for one Bayer channel, black-subtracted.

    Returns (plane, info) where info carries flip/bayer_shape for coordinate
    conversion. Plane pixel pitch = 2 sensor px (Bayer subsampling).
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        bayer = raw.raw_image_visible.copy().astype(np.float64)
        colors = raw.raw_colors_visible
        flip = int(raw.sizes.flip)
        black_levels = [int(v) for v in raw.black_level_per_channel]
        color_desc = raw.color_desc.decode("ascii") if raw.color_desc else "RGGB"

    ch_names = images.bayer_channel_names(color_desc)
    targets = [i for i, n in ch_names.items() if n == channel or n.rstrip("12") == channel]
    if not targets:
        raise ValueError(f"channel {channel!r} not in Bayer pattern {color_desc}")
    ch_idx = targets[0]

    # Locate the channel inside the 2×2 Bayer tile and subsample
    tile = colors[:2, :2]
    pos = np.argwhere(tile == ch_idx)
    if len(pos) == 0:
        raise ValueError(f"channel index {ch_idx} not found in Bayer tile")
    r0, c0 = pos[0]
    plane = bayer[r0::2, c0::2] - black_levels[ch_idx]

    info = {"flip": flip, "bayer_shape": bayer.shape, "pitch": 2, "offset": (int(c0), int(r0))}
    return plane, info


def load_channel_plane_srgb(path: Path, channel: str) -> tuple[np.ndarray, dict]:
    """Linearised channel plane from an sRGB image. Pitch = 1 px."""
    img = Image.open(str(path)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = images.srgb_to_linear(arr)
    idx = {"R": 0, "G": 1, "B": 2}.get(channel.rstrip("12"))
    if idx is None:
        raise ValueError(f"channel {channel!r} not valid for sRGB (use R, G or B)")
    info = {"flip": 0, "bayer_shape": arr.shape[:2], "pitch": 1, "offset": (0, 0)}
    return arr[:, :, idx].astype(np.float64), info


def load_channel_plane(path: Path, channel: str) -> tuple[np.ndarray, dict]:
    """Load one channel plane (+ geometry info), dispatching on RAW vs sRGB format."""
    if images.is_raw(path):
        return load_channel_plane_raw(path, channel)
    return load_channel_plane_srgb(path, channel)


def render_viewer_image(path: Path) -> np.ndarray:
    """Image as the user sees it (uint8 RGB), for the ROI overlay."""
    if images.is_raw(path):
        import rawpy

        with rawpy.imread(str(path)) as raw:
            return raw.postprocess(use_camera_wb=True, output_bps=8)
    return np.array(Image.open(str(path)).convert("RGB"), dtype=np.uint8)


# ─── Grid statistics ──────────────────────────────────────────────────────────


def block_stats(arr: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and CV (%) for non-overlapping blocks."""
    H, W = arr.shape
    gH, gW = H // block, W // block
    mean_grid = np.zeros((gH, gW))
    cv_grid = np.zeros((gH, gW))
    for gi in range(gH):
        for gj in range(gW):
            cell = arr[gi * block : (gi + 1) * block, gj * block : (gj + 1) * block]
            m = cell.mean()
            mean_grid[gi, gj] = m
            cv_grid[gi, gj] = cell.std() / m * 100 if m > 0 else np.inf
    return mean_grid, cv_grid


def best_window(score_map: np.ndarray, win_h: int, win_w: int) -> tuple[tuple[int, int], float]:
    """Window position minimising mean score (summed-area table).

    If win_h == score_map.shape[0] (or win_w == shape[1]), there is no slack
    to slide in that axis and the window is forced to span the full grid
    there regardless of score — see best_window_optimal_aspect() below."""
    gH, gW = score_map.shape
    cum = np.cumsum(np.cumsum(score_map, 0), 1)

    def box_mean(r1, c1, r2, c2):
        s = cum[r2, c2]
        if r1 > 0:
            s -= cum[r1 - 1, c2]
        if c1 > 0:
            s -= cum[r2, c1 - 1]
        if r1 > 0 and c1 > 0:
            s += cum[r1 - 1, c1 - 1]
        return s / ((r2 - r1 + 1) * (c2 - c1 + 1))

    best_val = np.inf
    best_pos = (0, 0)
    for r in range(gH - win_h + 1):
        for c in range(gW - win_w + 1):
            val = box_mean(r, c, r + win_h - 1, c + win_w - 1)
            if val < best_val:
                best_val, best_pos = val, (r, c)
    return best_pos, best_val


def best_window_optimal_aspect(
    score_map: np.ndarray, area: int
) -> tuple[tuple[int, int], tuple[int, int], float]:
    """Try every (win_h, win_w) exact-divisor pair of `area` that fits inside
    score_map, run best_window() for each, and return the shape/position/value
    of whichever has the lowest mean score.

    Same total block count (same pixel count, no shot-noise tradeoff) at
    whichever aspect ratio actually clears the CV hotspots for this scene —
    see the module docstring. Falls back to (1, area) / (area, 1) if area has
    no divisor pair that fits the grid.
    """
    gH, gW = score_map.shape
    best_shape = None
    best_pos = (0, 0)
    best_val = np.inf
    for win_h in range(1, gH + 1):
        if area % win_h != 0:
            continue
        win_w = area // win_h
        if win_w < 1 or win_w > gW:
            continue
        pos, val = best_window(score_map, win_h, win_w)
        if val < best_val:
            best_val, best_pos, best_shape = val, pos, (win_h, win_w)
    if best_shape is None:
        raise ValueError(
            f"area={area} has no (win_h, win_w) divisor pair fitting a "
            f"{gH}x{gW} grid; pick an area with more divisors or shrink it"
        )
    return best_shape, best_pos, best_val


# ─── Plotting ─────────────────────────────────────────────────────────────────


def save_heatmap(data, title, colorlabel, cmap, path, extent, roi=None):
    """Save a labelled heatmap PNG, optionally outlining the recommended ROI."""
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data, cmap=cmap, vmin=data.min(), vmax=data.max(), aspect="auto", extent=extent)
    plt.colorbar(im, ax=ax, label=colorlabel)
    ax.set_title(title)
    ax.set_xlabel("x (sensor px)")
    ax.set_ylabel("y (sensor px)")
    if roi is not None:
        rx, ry, rw, rh = roi
        rect = mpatches.Rectangle(
            (rx, ry),
            rw,
            rh,
            linewidth=3,
            edgecolor="blue",
            facecolor="none",
            label="Recommended ROI",
        )
        ax.add_patch(rect)
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(
    img_dir: Annotated[
        Path, typer.Option("--img-dir", help="Folder with one acquisition set (RAW or sRGB)")
    ] = ...,
    blank: Annotated[
        str, typer.Option("--blank", help="Blank/control filename (basename or glob)")
    ] = ...,
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Output directory")] = ...,
    channel: Annotated[str, typer.Option("--channel", help="Channel to analyse (R, G or B)")] = "B",
    block: Annotated[
        Optional[int], typer.Option("--block", help="Block size in viewer px (default 100)")
    ] = None,
    win_h: Annotated[
        int,
        typer.Option("--win-h", help="Window height in blocks (area-only with --search-aspect)"),
    ] = 4,
    win_w: Annotated[
        int,
        typer.Option("--win-w", help="Window width in blocks (area-only with --search-aspect)"),
    ] = 8,
    search_aspect: Annotated[
        bool,
        typer.Option(
            "--search-aspect",
            help="Fix win_h*win_w as an area budget, search all divisor-pair aspect "
            "ratios that fit the grid, use whichever minimises the combined score",
        ),
    ] = False,
):
    """Scan a blank image for the flattest region and recommend ROI coordinates."""
    img_dir = Path(img_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve blank
    blank_cand = sorted(img_dir.glob(blank)) or [img_dir / blank]
    blank_path = blank_cand[0]
    if not blank_path.exists():
        print(f"[roi_selector] ERROR: blank not found: {blank_path}")
        raise typer.Exit(1)
    print(f"[roi_selector] Blank: {blank_path.name}")

    plane_blank, info = load_channel_plane(blank_path, channel)
    pitch = info["pitch"]
    BLOCK = max(1, (block or 100) // pitch)  # plane units
    H, W = plane_blank.shape
    extent = [0, W * pitch, H * pitch, 0]
    print(
        f"[roi_selector] Channel {channel} plane: {W}×{H} (pitch {pitch} px), "
        f"block grid: {W // BLOCK}×{H // BLOCK}"
    )

    # 1. Blank CV map
    blank_mean_grid, blank_cv_grid = block_stats(plane_blank, BLOCK)
    save_heatmap(
        blank_cv_grid,
        f"Blank {channel}-channel CV per {BLOCK * pitch}×{BLOCK * pitch} px block",
        "CV (%)",
        "plasma",
        out_dir / "blank_cv_grid.png",
        extent,
    )
    print(f"[roi_selector] Blank CV range: {blank_cv_grid.min():.2f}–{blank_cv_grid.max():.2f}%")

    # 2. Signal CV across all sample images
    sample_files = [f for f in images.list_images(img_dir) if f.name != blank_path.name]
    print(f"[roi_selector] Sample images: {len(sample_files)}")
    if not sample_files:
        print("[roi_selector] ERROR: no sample images besides the blank.")
        raise typer.Exit(1)

    gH, gW = H // BLOCK, W // BLOCK
    sample_block_means = np.zeros((len(sample_files), gH, gW))
    for i, f in enumerate(sample_files):
        plane, _ = load_channel_plane(f, channel)
        for gi in range(gH):
            for gj in range(gW):
                s = plane[gi * BLOCK : (gi + 1) * BLOCK, gj * BLOCK : (gj + 1) * BLOCK].mean()
                bl = blank_mean_grid[gi, gj]
                sample_block_means[i, gi, gj] = s / max(bl, 1e-9)

    signal_cv_grid = sample_block_means.std(axis=0) / (sample_block_means.mean(axis=0) + 1e-9) * 100
    save_heatmap(
        signal_cv_grid,
        f"Signal CV across samples ({channel}-channel, blank-corrected)\n"
        f"Block = {BLOCK * pitch}×{BLOCK * pitch} px",
        "CV (%)",
        "plasma",
        out_dir / "signal_cv_grid.png",
        extent,
    )
    print(f"[roi_selector] Signal CV range: {signal_cv_grid.min():.2f}–{signal_cv_grid.max():.2f}%")

    # 3. Combined score → best window
    combined = blank_cv_grid / blank_cv_grid.max() + signal_cv_grid / signal_cv_grid.max()
    if search_aspect:
        area = win_h * win_w
        (win_h, win_w), (r0, c0), best_val = best_window_optimal_aspect(combined, area)
        print(
            f"[roi_selector] --search-aspect: area={area} blocks -> "
            f"best shape win_h={win_h} win_w={win_w} (combined score={best_val:.4f})"
        )
    else:
        (r0, c0), best_val = best_window(combined, win_h, win_w)

    # ROI in sensor/native pixel coordinates
    sx = c0 * BLOCK * pitch
    sy = r0 * BLOCK * pitch
    sw = win_w * BLOCK * pitch
    sh = win_h * BLOCK * pitch
    roi_sensor = (sx, sy, sw, sh)

    save_heatmap(
        combined,
        "Combined homogeneity score (lower = better)\nRecommended ROI shown",
        "Combined score (norm.)",
        "RdYlGn_r",
        out_dir / "combined_score_roi.png",
        extent,
        roi=roi_sensor,
    )

    # Convert to viewer coordinates (identity for sRGB)
    roi_viewer = images.bayer_roi_to_viewer(roi_sensor, info["flip"], info["bayer_shape"])
    print(f"\n[roi_selector] Recommended ROI (combined score={best_val:.4f}):")
    print(
        f"  viewer coords:  --roi {roi_viewer[0]} {roi_viewer[1]} {roi_viewer[2]} {roi_viewer[3]}"
    )

    # 4. Overlay on a real mid-series image (viewer space)
    mid_file = sample_files[len(sample_files) // 2]
    rgb = render_viewer_image(mid_file)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    vx, vy, vw, vh = roi_viewer
    draw.rectangle([vx, vy, vx + vw, vy + vh], outline=(0, 200, 0), width=6)
    draw.text((vx + 5, vy + 5), f"--roi {vx} {vy} {vw} {vh}", fill=(0, 200, 0))
    img.save(out_dir / "roi_on_real_image.png")
    print("[roi_selector] Saved roi_on_real_image.png")

    # 5. Write rois.yaml in viewer coordinates
    rois_yaml = {
        "rois": [
            {
                "name": "signal",
                "x": int(vx),
                "y": int(vy),
                "w": int(vw),
                "h": int(vh),
                "space": "viewer",
                "type": "signal",
                "description": (
                    f"Auto-selected by roi_selector.py from {img_dir.name} "
                    f"(channel {channel}). Combined homogeneity score={best_val:.4f} "
                    f"(blank CV + signal CV, block={BLOCK * pitch}px, "
                    f"window={sh}×{sw}px"
                    + (", aspect ratio chosen by --search-aspect" if search_aspect else "")
                    + "). REVIEW before use."
                ),
            }
        ]
    }
    rois_path = out_dir / "rois.yaml"
    with rois_path.open("w", encoding="utf-8") as f:
        f.write(
            "# rois.yaml — auto-generated by roi_selector.py\n"
            "# Coordinates are in VIEWER space (image as displayed, origin top-left).\n"
            "# REVIEW the ROI overlay, then copy to <project>/image_analysis/rois.yaml.\n\n"
        )
        yaml.dump(rois_yaml, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print("[roi_selector] Saved rois.yaml → review, then copy to <project>/image_analysis/")

    write_run_config(
        out_dir,
        "roi_selector",
        {
            "img_dir": img_dir,
            "blank": blank_path.name,
            "channel": channel,
            "block_px": BLOCK * pitch,
            "win_h": win_h,
            "win_w": win_w,
            "roi_viewer": list(roi_viewer),
            "combined_score": float(best_val),
        },
    )

    print(
        f"\nNext step:\n  python raw-image-analysis/s02_calibrate/calibrate.py \\\n"
        f"    --project <project>/ \\\n"
        f"    --sets {img_dir} \\\n"
        f"    --roi {vx} {vy} {vw} {vh}"
    )


if __name__ == "__main__":
    typer.run(main)
