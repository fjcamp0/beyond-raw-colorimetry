"""
raw_image_histograms.py

Per-channel Bayer pixel histograms for the ferricyanide and methylene_blue RAW
image sets, one PNG plot plus a companion CSV of bin counts saved next to each
plot. This is a diagnostic pass to look for clues behind the "channel-common"
absorbance component described in NOTE_TO_CLAUDE_CODE_2026-06-30.md section 4:
R, G, and B track a shared, unmodeled component that B/G and B/R ratios
cancel. Whole-image and (where a reviewed ROI exists) signal-ROI histograms
are both written, since a shared offset or clipping pattern visible in one
but not the other narrows down candidate causes (sensor-wide vs.
sample/ROI-local).

Run from the project root:
    python scripts/raw_image_histograms.py

Outputs: Data/image_analysis/<system>/<set>/histograms/<stem>_hist.{png,csv}
         Data/image_analysis/<system>/<set>/histograms/<stem>_<roi>_hist.{png,csv}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rawpy  # noqa: E402
import yaml  # noqa: E402

# Vendored raw-image-analysis modules (vendor/, see README); override with COLORIMETRY_VENDOR.
_RA = Path(
    os.environ.get("COLORIMETRY_VENDOR", str(Path(__file__).resolve().parent.parent / "vendor"))
)
sys.path.insert(0, str(_RA))
from core.images import bayer_channel_names, viewer_roi_to_bayer  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJ_ROOT / "Data"

N_BINS = 512
CHANNEL_COLORS = {"R": "#e74c3c", "G1": "#27ae60", "G2": "#1abc9c", "B": "#3498db"}


def _roi(analysis_dir: Path, *parts: str) -> Path:
    """Build a path to a roi_selector.py-reviewed rois.yaml under analysis_dir."""
    return analysis_dir.joinpath(*parts, "roi_selection", "rois.yaml")


# Per-system config: image folder names (case-sensitive on disk), the reviewed
# ROI file for sets that have one (viewer-space coords from
# raw-image-analysis/scripts/roi_selector.py), and the acquisition sets to
# process. Sets with no ROI entry get whole-image histograms only.
_FERRI_ANALYSIS = DATA_DIR / "analysis" / "ferricyanide"
_MB_ANALYSIS = DATA_DIR / "analysis" / "methylene_blue"

SYSTEMS = {
    "ferricyanide": {
        "images_dir": DATA_DIR / "images" / "ferricyanide",
        "out_dir": DATA_DIR / "image_analysis" / "ferricyanide",
        "sets": [
            "daylight_set1",
            "daylight_set2",
            "daylight_set3",
            "daylight_set4",
            "daylight_diffuser_set1",
            "daylight_diffuser_set2",
            "daylight_diffuser_set3",
            "daylight_diffuser_set4",
            "led_set1",
            "led_set2",
            "led_set3",
        ],
        "roi_files": {
            "daylight_set1": _roi(_FERRI_ANALYSIS, "daylight", "set1"),
            "daylight_diffuser_set1": _roi(_FERRI_ANALYSIS, "daylight_diffuser", "set1"),
            "daylight_diffuser_set2": _roi(_FERRI_ANALYSIS, "daylight_diffuser", "set2"),
            "daylight_diffuser_set3": _roi(_FERRI_ANALYSIS, "daylight_diffuser", "set3"),
            "daylight_diffuser_set4": _roi(_FERRI_ANALYSIS, "daylight_diffuser", "set4"),
        },
    },
    "methylene_blue": {
        "images_dir": DATA_DIR / "images" / "methylene_blue",
        "out_dir": DATA_DIR / "image_analysis" / "methylene_blue",
        # Image folder names are capitalised inconsistently on disk for this
        # analyte ("Daylight_set*" vs "daylight_diffuser_set*", "led_set*").
        "sets": [
            "Daylight_set1",
            "Daylight_set2",
            "Daylight_set3",
            "daylight_diffuser_set1",
            "daylight_diffuser_set2",
            "daylight_diffuser_set3",
            "led_set1",
            "led_set2",
            "led_set3",
        ],
        "roi_files": {
            "Daylight_set1": _roi(_MB_ANALYSIS, "daylight", "set1"),
            "daylight_diffuser_set1": _roi(_MB_ANALYSIS, "daylight_diffuser", "set1"),
            "daylight_diffuser_set2": _roi(_MB_ANALYSIS, "daylight_diffuser", "set2"),
            "daylight_diffuser_set3": _roi(_MB_ANALYSIS, "daylight_diffuser", "set3"),
        },
    },
}


def channel_histograms(
    bayer: np.ndarray,
    colors: np.ndarray,
    black_levels: list[int],
    white_level: int,
    ch_names: dict[int, str],
    region: tuple[int, int, int, int] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Black-subtracted per-channel histograms for one image or ROI.

    Returns {channel_name: (counts, bin_edges)}; region is (x, y, w, h) in
    sensor (Bayer) coordinates, or None for the whole image.
    """
    if region is not None:
        x, y, w, h = region
        b = bayer[y : y + h, x : x + w]
        c = colors[y : y + h, x : x + w]
    else:
        b, c = bayer, colors

    out = {}
    for ch_idx, ch_name in ch_names.items():
        mask = c == ch_idx
        if not np.any(mask):
            continue
        bl = black_levels[ch_idx] if len(black_levels) > ch_idx else 0
        wl_eff = white_level - bl
        vals = np.clip(b[mask].astype(np.float64) - bl, 0, wl_eff)
        counts, edges = np.histogram(vals, bins=N_BINS, range=(0, wl_eff))
        out[ch_name] = (counts, edges)
    return out


def save_hist_png_csv(
    hist_data: dict[str, tuple[np.ndarray, np.ndarray]],
    stem: str,
    out_dir: Path,
    label: str,
) -> None:
    """Save one histogram plot (PNG) and its companion bin-count CSV, same stem."""
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = list(hist_data.keys())
    fig, axes = plt.subplots(1, len(channels), figsize=(4 * len(channels), 3), squeeze=False)

    rows = []
    for ax, ch_name in zip(axes[0], channels):
        counts, edges = hist_data[ch_name]
        centers = (edges[:-1] + edges[1:]) / 2
        color = CHANNEL_COLORS.get(ch_name, "#95a5a6")
        ax.bar(centers, counts, width=edges[1] - edges[0], color=color, alpha=0.8, linewidth=0)
        ax.set_title(ch_name, fontsize=10)
        ax.set_xlabel("Pixel value (black-subtracted)")
        ax.set_ylabel("Count")
        for bl, br, bc, cnt in zip(edges[:-1], edges[1:], centers, counts):
            rows.append(
                {
                    "channel": ch_name,
                    "bin_left": float(bl),
                    "bin_right": float(br),
                    "bin_center": float(bc),
                    "count": int(cnt),
                }
            )

    fig.suptitle(label, fontsize=9, y=1.02)
    fig.tight_layout()
    png_path = out_dir / f"{stem}_hist.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    csv_path = out_dir / f"{stem}_hist.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  {png_path.name} + {csv_path.name}")


def process_set(system: str, set_name: str, config: dict) -> None:
    """Compute whole-image and (if a reviewed ROI exists) ROI histograms for one set."""
    images_dir = config["images_dir"] / set_name
    dngs = sorted(images_dir.glob("*.dng"))
    if not dngs:
        print(f"[{system}/{set_name}] no DNG files found, skipping")
        return

    roi = None
    roi_path = config["roi_files"].get(set_name)
    if roi_path and roi_path.exists():
        with roi_path.open(encoding="utf-8") as f:
            roi = yaml.safe_load(f)["rois"][0]

    out_dir = config["out_dir"] / set_name / "histograms"
    print(f"[{system}/{set_name}] {len(dngs)} image(s) -> {out_dir}")

    for path in dngs:
        with rawpy.imread(str(path)) as raw:
            bayer = raw.raw_image_visible.copy()
            colors = raw.raw_colors_visible
            white_level = int(raw.white_level)
            black_levels = [int(v) for v in raw.black_level_per_channel]
            color_desc = raw.color_desc.decode("ascii") if raw.color_desc else "RGGB"
            flip = int(raw.sizes.flip)

        ch_names = bayer_channel_names(color_desc)
        stem = path.stem

        whole_hist = channel_histograms(bayer, colors, black_levels, white_level, ch_names)
        save_hist_png_csv(whole_hist, stem, out_dir, label=stem)

        if roi is not None:
            region = viewer_roi_to_bayer(
                (roi["x"], roi["y"], roi["w"], roi["h"]), flip, bayer.shape
            )
            roi_hist = channel_histograms(
                bayer, colors, black_levels, white_level, ch_names, region=region
            )
            save_hist_png_csv(
                roi_hist, f"{stem}_{roi['name']}", out_dir, label=f"{stem} — {roi['name']}"
            )


def main() -> None:
    """Generate per-image RAW channel histograms (PNG + companion CSV) for every system/set."""
    for system, config in SYSTEMS.items():
        for set_name in config["sets"]:
            process_set(system, set_name, config)


if __name__ == "__main__":
    main()
