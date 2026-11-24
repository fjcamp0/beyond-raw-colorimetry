# Copyright 2024 F. Javier del Campo
# Licensed under the Apache License, Version 2.0

"""
spectral.py (formerly _resources/spectral_utils.py)

Unified spectral sensitivity utilities for camera-based analysis.

Two data sources are merged into one interface:

  SOURCE A — Ohlab smartphone database (21 devices, 400–700 nm, 10 nm steps)
    Citation REQUIRED:
      Jiang, J. et al. "What Is the Space of Spectral Sensitivity Functions
      for Digital Color Cameras?" Sensors 2021, 21(15), 4985.
      https://doi.org/10.3390/s21154985

  SOURCE B — colour-science built-in database (2 DSLRs, 380/400–780/680 nm)
    Nikon D5100 (NPL), Sigma SD Merill (NPL)
    colour-science: https://www.colour-science.org/
    No additional citation needed beyond colour-science package attribution.

  SOURCE C — Color-Lab-Eilat estimated curves (1000+ cameras, 400–700 nm)
    ESTIMATED (predicted), not measured — a fallback when no measured curve
    exists for the camera. Fetched on demand by
    scripts/fetch_estimated_sensitivity.py into spectral_sensitivity/estimated/.
    Citation REQUIRED:
      Solomatov, G.; Akkaynak, D. "Spectral Sensitivity Estimation Without a
      Camera." IEEE ICCP 2023. arXiv:2304.11549.
      https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation

Workflow guidance
-----------------
- Smartphones (iPhone, Samsung, Huawei, etc.)  → use SOURCE A (Ohlab, measured)
- Nikon D5100 or Sigma SD Merill               → use SOURCE B (colour-science)
- Other DSLR/mirrorless, no measured curve     → use SOURCE C (estimated)
- Scientific cameras (Thorlabs, PCO, Andor)    → add datasheet CSV to
  spectral_sensitivity/ following the format described in README.md
- Camera model unknown                          → use "average_all" (SOURCE A)

Measured sources (A, B) always win over estimated (C) on a key collision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_SENSITIVITY_DIR = Path(__file__).parent.parent / "_resources" / "spectral_sensitivity"
# Estimated (predicted) curves live in a subfolder so they never shadow the
# measured Ohlab files and so their provenance/citation stays distinct.
_ESTIMATED_DIR = _SENSITIVITY_DIR / "estimated"

# Standard output grid: 400–700 nm at 10 nm steps (matches Ohlab format)
WAVELENGTHS = np.arange(400, 710, 10, dtype=float)  # 31 points


# ─── Source A: Ohlab smartphone database ──────────────────────────────────────


def _ohlab_devices() -> list[str]:
    return sorted(p.stem for p in _SENSITIVITY_DIR.glob("*.txt"))


def _load_ohlab(device: str) -> np.ndarray:
    """Load (31, 3) array from Ohlab txt file. Raises FileNotFoundError if absent."""
    path = _SENSITIVITY_DIR / f"{device}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Ohlab: no file for '{device}'. Run init_project to download.")
    data = np.loadtxt(str(path))
    if data.shape != (31, 3):
        raise ValueError(f"Unexpected shape {data.shape} in {path.name}; expected (31, 3).")
    return data


# ─── Source C: Color-Lab-Eilat estimated curves ───────────────────────────────


def _estimated_devices() -> list[str]:
    """Device keys for estimated curves fetched into the estimated/ subfolder."""
    if not _ESTIMATED_DIR.exists():
        return []
    return sorted(p.stem for p in _ESTIMATED_DIR.glob("*.txt"))


def _load_estimated(device: str) -> np.ndarray:
    """Load (31, 3) estimated curve from estimated/<device>.txt. Same format as Ohlab."""
    path = _ESTIMATED_DIR / f"{device}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Estimated: no file for '{device}'. "
            f"Fetch it with scripts/fetch_estimated_sensitivity.py --camera <name>."
        )
    data = np.loadtxt(str(path))
    if data.shape != (31, 3):
        raise ValueError(f"Unexpected shape {data.shape} in {path.name}; expected (31, 3).")
    return data


# ─── Source B: colour-science built-in ────────────────────────────────────────

_COLOUR_SCIENCE_CAMERAS: dict[str, str] = {
    # canonical key → colour-science lookup name
    "nikon_d5100": "Nikon 5100 (NPL)",
    "sigma_sd_merill": "Sigma SDMerill (NPL)",
}


def _colour_science_devices() -> list[str]:
    return list(_COLOUR_SCIENCE_CAMERAS.keys())


def _load_colour_science(device: str) -> np.ndarray:
    """
    Load sensitivity from colour-science, interpolated onto WAVELENGTHS grid.
    Returns (31, 3) array at 400–700 nm / 10 nm steps.
    """
    import colour

    lookup = _COLOUR_SCIENCE_CAMERAS[device]
    css = colour.characterisation.datasets.MSDS_CAMERA_SENSITIVITIES
    s = css[lookup]
    # Interpolate each channel onto standard 400–700 nm grid
    out = np.zeros((len(WAVELENGTHS), 3))
    for i in range(3):
        out[:, i] = np.interp(WAVELENGTHS, s.domain, s.range[:, i], left=0.0, right=0.0)
    return out


# ─── Unified public API ───────────────────────────────────────────────────────


def list_devices() -> dict[str, list[str]]:
    """
    Return all available devices grouped by source.

    Returns
    -------
    dict with keys:
      "ohlab"          — Ohlab smartphone database (21 devices + average_all)
      "colour_science" — colour-science built-in (2 DSLRs)

    Usage guidance:
      - Smartphones → ohlab
      - Nikon D5100 / Sigma SD Merill → colour_science
      - Camera unknown → ohlab["average_all"]
      - Scientific / industrial cameras → add to spectral_sensitivity/ manually
    """
    return {
        "ohlab": _ohlab_devices(),
        "estimated": _estimated_devices(),
        "colour_science": _colour_science_devices(),
    }


def load_sensitivity(device: str = "average_all") -> np.ndarray:
    """
    Load normalised spectral sensitivity for a camera, interpolated onto
    WAVELENGTHS (400–700 nm, 10 nm steps).

    Searches Ohlab first, then colour-science.

    Parameters
    ----------
    device : str
        Device key. Use list_devices() to see options.
        Use "average_all" when camera model is unknown.

    Returns
    -------
    sensitivity : np.ndarray, shape (31, 3)
        Rows = 400, 410, ..., 700 nm. Columns = R, G, B sensitivity (peak ≈ 1).

    Citation
    --------
    If device is from Ohlab (most cases), cite:
      Jiang et al., Sensors 2021, 21(15), 4985. https://doi.org/10.3390/s21154985
    """
    # Try Ohlab (measured) first — measured always wins over estimated.
    if (_SENSITIVITY_DIR / f"{device}.txt").exists():
        return _load_ohlab(device)

    # Then estimated (Eilat) curves.
    if (_ESTIMATED_DIR / f"{device}.txt").exists():
        return _load_estimated(device)

    # Then colour-science built-ins.
    if device in _COLOUR_SCIENCE_CAMERAS:
        return _load_colour_science(device)

    # Helpful error listing all options
    all_devices = list_devices()
    raise FileNotFoundError(
        f"No spectral sensitivity data for '{device}'.\n"
        f"  Ohlab ({len(all_devices['ohlab'])}): {', '.join(all_devices['ohlab'])}\n"
        f"  estimated ({len(all_devices['estimated'])}): "
        f"{', '.join(all_devices['estimated']) or '— none fetched yet'}\n"
        f"  colour-science ({len(all_devices['colour_science'])}): "
        f"{', '.join(all_devices['colour_science'])}\n"
        f"  Fetch an estimated curve: "
        f"scripts/fetch_estimated_sensitivity.py --camera <name>\n"
        f"  Or add a custom camera: place a 31×3 tab-delimited txt file\n"
        f"  (400–700 nm, 10 nm steps, R G B columns) in:\n"
        f"  {_SENSITIVITY_DIR}/<device_name>.txt"
    )


def source_of(device: str) -> str:
    """Return which database a device key belongs to (measured Ohlab wins on collision)."""
    if (_SENSITIVITY_DIR / f"{device}.txt").exists():
        return "ohlab"
    if (_ESTIMATED_DIR / f"{device}.txt").exists():
        return "estimated_eilat"
    if device in _COLOUR_SCIENCE_CAMERAS:
        return "colour_science"
    return "unknown"


def citation_for(device: str) -> str:
    """Return the citation string required when using this device's data."""
    src = source_of(device)
    if src == "ohlab":
        return (
            "Jiang, J. et al. 'What Is the Space of Spectral Sensitivity Functions "
            "for Digital Color Cameras?' Sensors 2021, 21(15), 4985. "
            "https://doi.org/10.3390/s21154985"
        )
    if src == "estimated_eilat":
        return (
            "ESTIMATED curve. Solomatov, G.; Akkaynak, D. 'Spectral Sensitivity "
            "Estimation Without a Camera.' IEEE ICCP 2023. arXiv:2304.11549. "
            "https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation"
        )
    if src == "colour_science":
        return "colour-science (https://www.colour-science.org/)"
    return "Unknown source — add citation manually."


# ─── Analysis helpers ─────────────────────────────────────────────────────────


def rgb_channel_peaks(device: str = "average_all") -> dict[str, float]:
    """Peak wavelength (nm) for R, G, B channels."""
    s = load_sensitivity(device)
    return {
        "R_peak_nm": float(WAVELENGTHS[np.argmax(s[:, 0])]),
        "G_peak_nm": float(WAVELENGTHS[np.argmax(s[:, 1])]),
        "B_peak_nm": float(WAVELENGTHS[np.argmax(s[:, 2])]),
    }


def effective_wavelength(device: str = "average_all") -> dict[str, float]:
    """Centroid (effective) wavelength for R, G, B: λ_eff = Σ(λ·s) / Σ(s)."""
    s = load_sensitivity(device)
    result = {}
    for i, ch in enumerate(["R", "G", "B"]):
        sens = s[:, i]
        result[f"{ch}_eff_nm"] = (
            float(np.sum(WAVELENGTHS * sens) / np.sum(sens)) if np.sum(sens) > 0 else float("nan")
        )
    return result


def sensitivity_at_wavelength(
    wavelength_nm: float, device: str = "average_all"
) -> dict[str, float]:
    """Interpolated R, G, B sensitivity at a specific wavelength."""
    s = load_sensitivity(device)
    return {
        ch: float(np.interp(wavelength_nm, WAVELENGTHS, s[:, i]))
        for i, ch in enumerate(["R", "G", "B"])
    }


def best_channel_for_wavelength(wavelength_nm: float, device: str = "average_all") -> str:
    """Channel (R/G/B) with highest sensitivity at the given wavelength."""
    sens = sensitivity_at_wavelength(wavelength_nm, device)
    return max(sens, key=sens.get)


def analyte_channel_report(absorption_peak_nm: float, device: str = "average_all") -> str:
    """
    Print report on which channel best detects an analyte absorbing at
    `absorption_peak_nm`. Includes citation for the data source used.
    """
    peaks = rgb_channel_peaks(device)
    eff = effective_wavelength(device)
    sens = sensitivity_at_wavelength(absorption_peak_nm, device)
    best = best_channel_for_wavelength(absorption_peak_nm, device)
    src = source_of(device)

    lines = [
        f"Device: {device}  [source: {src}]",
        f"Analyte absorption peak: {absorption_peak_nm:.0f} nm",
        "",
        "Channel peak wavelengths:",
        f"  R: {peaks['R_peak_nm']:.0f} nm  (effective centroid: {eff['R_eff_nm']:.1f} nm)",
        f"  G: {peaks['G_peak_nm']:.0f} nm  (effective centroid: {eff['G_eff_nm']:.1f} nm)",
        f"  B: {peaks['B_peak_nm']:.0f} nm  (effective centroid: {eff['B_eff_nm']:.1f} nm)",
        "",
        f"Sensitivity at {absorption_peak_nm:.0f} nm:",
        f"  R: {sens['R']:.4f}",
        f"  G: {sens['G']:.4f}",
        f"  B: {sens['B']:.4f}",
        "",
        f"Recommended channel: {best}",
        "  → Use this channel for Beer-Lambert absorbance (highest SNR).",
        "",
        f"Citation: {citation_for(device)}",
    ]
    return "\n".join(lines)


def print_catalog() -> None:
    """Print all available devices with source and wavelength range."""
    devices = list_devices()
    print(f"{'Device':<50} {'Source':<15} {'Range'}")
    print("-" * 80)
    for d in devices["ohlab"]:
        print(f"  {d:<48} ohlab          400–700 nm")
    for d in devices["estimated"]:
        print(f"  {d:<48} estimated      400–700 nm (Eilat, predicted)")
    for d in devices["colour_science"]:
        print(f"  {d:<48} colour-science 400–700 nm (interpolated)")
    total = len(devices["ohlab"]) + len(devices["estimated"]) + len(devices["colour_science"])
    print(
        f"\nTotal: {total} devices  "
        f"({len(devices['ohlab'])} Ohlab, {len(devices['estimated'])} estimated, "
        f"{len(devices['colour_science'])} colour-science)"
    )
    print("\nFor scientific cameras not listed: add <device>.txt to:")
    print(f"  {_SENSITIVITY_DIR}/")


if __name__ == "__main__":
    print_catalog()
    print()
    print(analyte_channel_report(620, device="average_all"))  # Ru(bpy)3²⁺ ECL
    print()
    print(analyte_channel_report(665, device="average_all"))  # Methylene blue
