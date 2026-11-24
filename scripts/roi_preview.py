"""
roi_preview.py
Generate ROI preview figures for ferricyanide and methylene blue DNG images.

Layout (3 rows × n_images columns):
  Row 0 — full image with square ROI box
  Row 1 — square ROI crop
  Row 2 — per-channel histogram of the ROI (R / G / B overlay)

ROI: square, centred on each image.
"""

from pathlib import Path

import matplotlib
import numpy as np
import rawpy

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# ── ROI: square side in pixels ────────────────────────────────────────────────
ROI_SIDE = 700  # px; centred on the image

BASE = Path(__file__).resolve().parent.parent / "Data"
CH_COLORS = {"R": "#D62728", "G": "#2CA02C", "B": "#1F77B4"}


def load_rgb(dng_path: Path) -> np.ndarray:
    with rawpy.imread(str(dng_path)) as raw:
        return raw.postprocess(
            use_camera_wb=True,
            output_bps=8,
            no_auto_bright=True,
            bright=1.5,
        )


def centered_square_roi(img_h: int, img_w: int, side: int) -> tuple[int, int, int, int]:
    x = (img_w - side) // 2
    y = (img_h - side) // 2
    return x, y, side, side


def roi_figure(pairs: list[tuple[str, Path]], out_path: Path, suptitle: str):
    n = len(pairs)
    fig, axes = plt.subplots(
        3, n, figsize=(5.5 * n, 14), gridspec_kw={"height_ratios": [2.2, 1, 1]}
    )
    if n == 1:
        axes = axes.reshape(3, 1)

    for col, (label, dng_path) in enumerate(pairs):
        print(f"  Loading {dng_path.name} …")
        rgb = load_rgb(dng_path)
        h, w = rgb.shape[:2]
        x, y, rw, rh = centered_square_roi(h, w, ROI_SIDE)

        # ── row 0: full image + ROI box ──
        ax0 = axes[0, col]
        ax0.imshow(rgb)
        rect = patches.Rectangle(
            (x, y),
            rw,
            rh,
            linewidth=2.5,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
        ax0.add_patch(rect)
        ax0.text(
            x + 8,
            y + 26,
            "ROI",
            color="red",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )
        ax0.set_title(label, fontsize=10)
        ax0.axis("off")

        # ── row 1: ROI crop ──
        crop = rgb[y : y + rh, x : x + rw]
        ax1 = axes[1, col]
        ax1.imshow(crop)
        ax1.set_title(f"ROI  {rw}×{rh} px", fontsize=9)
        ax1.axis("off")

        # ── row 2: per-channel histogram ──
        ax2 = axes[2, col]
        for ch_idx, (ch_name, ch_color) in enumerate(CH_COLORS.items()):
            vals = crop[:, :, ch_idx].ravel()
            ax2.hist(
                vals,
                bins=64,
                range=(0, 255),
                color=ch_color,
                alpha=0.55,
                label=ch_name,
                density=True,
            )
            mean_v = vals.mean()
            std_v = vals.std()
            ax2.axvline(mean_v, color=ch_color, lw=1.2, ls="--")
            ax2.text(
                0.02,
                0.97 - ch_idx * 0.14,
                f"{ch_name}  μ={mean_v:.0f}  σ={std_v:.1f}",
                transform=ax2.transAxes,
                fontsize=7.5,
                color=ch_color,
                va="top",
            )
        ax2.set_xlabel("Pixel value (8-bit)", fontsize=8)
        ax2.set_ylabel("Density", fontsize=8)
        ax2.set_title("ROI histogram", fontsize=9)
        ax2.legend(fontsize=7, loc="upper right")
        ax2.tick_params(labelsize=7)

    fig.suptitle(
        suptitle + f"\nSquare ROI: {ROI_SIDE}×{ROI_SIDE} px, centred",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path.name}")


# ── Ferricyanide ──────────────────────────────────────────────────────────────
fe_dir = BASE / "images" / "ferricyanide"
fe_out = BASE / "analysis" / "ferricyanide" / "roi_preview_ferricyanide.png"

fe_pairs = [
    ("Blank (water)", fe_dir / "led_set2" / "Blank_IMG_260507_122659.dng"),
    ("62.5 µM (low)", fe_dir / "led_set2" / "62.5uM_IMG_260507_122932.dng"),
    ("1000 µM (mid)", fe_dir / "led_set2" / "1000uM_IMG_260507_123212.dng"),
    ("8000 µM (high)", fe_dir / "led_set1" / "8000µM_IMG_260505_124457.dng"),
]
fe_pairs = [(lbl, p) for lbl, p in fe_pairs if p.exists()]

print(f"\n--- Ferricyanide ({len(fe_pairs)} images) ---")
roi_figure(fe_pairs, fe_out, "Ferricyanide — smartphone RAW with analysis ROI")

# ── Methylene blue ────────────────────────────────────────────────────────────
mb_dir = BASE / "images" / "methylene_blue"
mb_out = BASE / "analysis" / "methylene_blue" / "roi_preview_methylene_blue.png"

mb_pairs = [
    ("Blank (water)", mb_dir / "led_set1" / "Blank_IMG_260526_123449.dng"),
    ("6.25 µM (low)", mb_dir / "led_set1" / "6.25µM_IMG_260526_123911.dng"),
    ("25 µM (mid)", mb_dir / "led_set1" / "25µM_IMG_260526_124045.dng"),
    ("100 µM (high)", mb_dir / "led_set1" / "100µM_IMG_260526_124218.dng"),
]
mb_pairs = [(lbl, p) for lbl, p in mb_pairs if p.exists()]

print(f"\n--- Methylene blue ({len(mb_pairs)} images) ---")
roi_figure(mb_pairs, mb_out, "Methylene blue — smartphone RAW with analysis ROI")

print("\nDone.")
