# Copyright 2024 F. Javier del Campo
# Licensed under the Apache License, Version 2.0

"""
images.py — unified image reading and ROI coordinate handling.

Canonical user-facing coordinate space: VIEWER coordinates — the image as
rendered by any viewer (postprocessed, orientation-corrected), origin
top-left, x right, y down. Users never need to know what Bayer space is.

For RAW files the library converts viewer ROIs to sensor (Bayer) coordinates
internally, using the libraw flip value:
    0 = no rotation, 3 = 180°, 5 = 90° CCW, 6 = 90° CW
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RAW_EXT = frozenset(
    {
        ".arw",
        ".cr2",
        ".cr3",
        ".nef",
        ".orf",
        ".raf",
        ".rw2",
        ".dng",
        ".pef",
        ".srw",
        ".x3f",
        ".arq",
    }
)

STD_EXT = frozenset({".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"})

ROI = tuple[int, int, int, int]  # (x, y, w, h)


def is_raw(path: Path) -> bool:
    """True if the file extension is a camera RAW format (see RAW_EXT)."""
    return path.suffix.lower() in RAW_EXT


def list_images(folder: Path) -> list[Path]:
    """All supported images in a folder, sorted. RAW takes priority: if any
    RAW files are present, standard-format files are ignored (they are
    usually previews of the same shots)."""
    files = sorted(f for f in folder.iterdir() if f.is_file())
    raws = [f for f in files if f.suffix.lower() in RAW_EXT]
    if raws:
        return raws
    return [f for f in files if f.suffix.lower() in STD_EXT]


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 piecewise sRGB transfer function. Input/output in [0, 1]."""
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def bayer_channel_names(color_desc: str) -> dict[int, str]:
    """Map Bayer plane indices 0-3 to channel names from libraw's color_desc.

    The two greens are disambiguated as G1/G2 in scan order so the duplicate
    'G' in patterns like RGGB does not collide.
    """
    names: dict[int, str] = {0: "R", 1: "G1", 2: "B", 3: "G2"}
    if len(color_desc) >= 4:
        seen_g = 0
        for i, ch in enumerate(color_desc[:4]):
            if ch == "G":
                names[i] = f"G{seen_g + 1}"
                seen_g += 1
            else:
                names[i] = ch
    return names


# ─── Coordinate conversion: viewer ↔ sensor (Bayer) ──────────────────────────


def viewer_roi_to_bayer(roi: ROI, flip: int, bayer_shape: tuple[int, int]) -> ROI:
    """Convert an (x, y, w, h) ROI from viewer coordinates to sensor
    (raw_image_visible) coordinates. bayer_shape is (rows, cols)."""
    H_s, W_s = bayer_shape
    x, y, w, h = roi
    if flip == 0:
        return roi
    if flip == 3:  # 180°
        return (W_s - x - w, H_s - y - h, w, h)
    if flip == 5:  # viewer = sensor rotated 90° CCW
        return (W_s - y - h, x, h, w)
    if flip == 6:  # viewer = sensor rotated 90° CW
        return (y, H_s - x - w, h, w)
    print(f"[core.images] WARNING: unsupported flip={flip}; treating ROI as sensor coords.")
    return roi


def bayer_roi_to_viewer(roi: ROI, flip: int, bayer_shape: tuple[int, int]) -> ROI:
    """Inverse of viewer_roi_to_bayer."""
    H_s, W_s = bayer_shape
    x, y, w, h = roi
    if flip == 0:
        return roi
    if flip == 3:
        return (W_s - x - w, H_s - y - h, w, h)
    if flip == 5:
        return (y, W_s - x - w, h, w)
    if flip == 6:
        return (H_s - y - h, x, h, w)
    print(f"[core.images] WARNING: unsupported flip={flip}; returning ROI unchanged.")
    return roi


def resolve_roi_to_bayer(roi: dict | ROI, flip: int, bayer_shape: tuple[int, int]) -> ROI:
    """Resolve a ROI to sensor coordinates.

    Accepts a plain (x, y, w, h) tuple (assumed viewer space) or a rois.yaml
    dict with optional 'space': 'viewer' (default) | 'bayer'.
    """
    if isinstance(roi, dict):
        coords = (int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"]))
        space = roi.get("space", "viewer")
    else:
        coords = tuple(int(v) for v in roi)  # type: ignore[assignment]
        space = "viewer"
    if space == "bayer":
        return coords
    return viewer_roi_to_bayer(coords, flip, bayer_shape)


# ─── RAW reading ──────────────────────────────────────────────────────────────


def probe_raw(path: Path) -> dict:
    """Return {bayer_shape, flip, viewer_size, color_desc} without full decode."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        shape = raw.raw_image_visible.shape
        flip = int(raw.sizes.flip)
        color_desc = raw.color_desc.decode("ascii") if raw.color_desc else "RGGB"
    H_s, W_s = shape
    viewer = (H_s, W_s) if flip in (5, 6) else (W_s, H_s)  # (w, h)
    return {"bayer_shape": shape, "flip": flip, "viewer_size": viewer, "color_desc": color_desc}


def read_raw_means(path: Path, roi_viewer: ROI | None = None) -> dict[str, float]:
    """Per-Bayer-channel means (R, G1, G2, B), black-subtracted, linear ADU.
    roi_viewer is in viewer coordinates; converted internally."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        bayer = raw.raw_image_visible.copy()
        colors = raw.raw_colors_visible
        flip = int(raw.sizes.flip)
        white_level = int(raw.white_level)
        black_levels = [int(v) for v in raw.black_level_per_channel]
        color_desc = raw.color_desc.decode("ascii") if raw.color_desc else "RGGB"

    ch_names = bayer_channel_names(color_desc)

    if roi_viewer is not None:
        rx, ry, rw, rh = viewer_roi_to_bayer(roi_viewer, flip, bayer.shape)
        b = bayer[ry : ry + rh, rx : rx + rw]
        c = colors[ry : ry + rh, rx : rx + rw]
    else:
        b, c = bayer, colors

    means: dict[str, float] = {}
    for ch_idx in range(4):
        mask = c == ch_idx
        if not np.any(mask):
            continue
        vals = b[mask].astype(np.float64)
        bl = black_levels[ch_idx] if len(black_levels) > ch_idx else 0
        wl_eff = white_level - bl
        vals = np.clip(vals - bl, 0, wl_eff)
        means[ch_names.get(ch_idx, f"CH{ch_idx}")] = float(np.mean(vals))
    return means


# ─── Standard (sRGB) reading ──────────────────────────────────────────────────


def read_srgb_means(path: Path, roi_viewer: ROI | None = None) -> dict[str, float]:
    """Per-channel {R, G, B} linearised means from an 8-bit sRGB image.
    Values in [0, 1] linear range. ROI in image pixel coordinates."""
    from PIL import Image

    img = Image.open(str(path)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = srgb_to_linear(arr)

    if roi_viewer is not None:
        x, y, w, h = roi_viewer
        arr = arr[y : y + h, x : x + w]

    return {
        "R": float(arr[:, :, 0].mean()),
        "G": float(arr[:, :, 1].mean()),
        "B": float(arr[:, :, 2].mean()),
    }


# ─── Unified interface ────────────────────────────────────────────────────────


def read_channel_means(path: Path, roi_viewer: ROI | None = None) -> dict[str, float]:
    """Format-dispatching channel means. RAW → black-subtracted Bayer ADU
    (R, G1, G2, B); sRGB → linearised [0, 1] (R, G, B). ROI is always in
    viewer coordinates."""
    if is_raw(path):
        return read_raw_means(path, roi_viewer)
    return read_srgb_means(path, roi_viewer)


def probe_dims(path: Path) -> tuple[int, int]:
    """(width, height) of the image as the user sees it (viewer space)."""
    if is_raw(path):
        return probe_raw(path)["viewer_size"]
    from PIL import Image

    with Image.open(str(path)) as im:
        return im.size
