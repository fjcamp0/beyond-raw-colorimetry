# Beyond RAW: The Beer-Lambert-Bayer Integral and Channel-Ratio Correction in Smartphone Transmission Colorimetry

Analysis and figure-generation code accompanying the paper by Gajjala, Bach, Rodríguez
Muiña, Lakard, Hihn and del Campo, accepted in *ACS Measurement Science Au* (2026). The
article DOI will be added here on publication; see `CITATION.cff`.

The paper derives and validates the Beer-Lambert-Bayer integral, the governing equation for
transmission colorimetry through a Bayer-filtered smartphone camera channel:

```
I_ch(c) = ∫ φ(λ) · S_ch(λ) · 10^[−ε(λ)·c·l] dλ
```

where φ(λ) is the illuminant spectral irradiance, S_ch(λ) is the relative spectral
sensitivity of the Bayer channel, and ε(λ) is the analyte's molar absorptivity. When an
analyte only partially absorbs across a channel's bandwidth, simple blank-division does not
fully cancel the illuminant, producing a nonlinear calibration curve.
The repository implements the forward model itself, a channel-ratio correction, and the figure and table generation for the paper and its
Supporting Information: two validation analytes (potassium ferricyanide and methylene
blue), RAW against the phone's native camera-app JPEG, bandpass-filter experiments that
remove the transparent tail, and a case study measuring iron in two wines as the
Fe(II)-phenanthroline complex by standard addition.

## Repository layout

```
scripts/            Analysis and figure-generation source (see scripts/README.md for the
                     script-to-figure/table index)
vendor/              Vendored third-party dependencies (see below) — not authored for this
                     paper, included so the scripts run standalone
figure_style.md      Canonical matplotlib style used by every figure script
requirements.txt     Python dependencies
ruff.toml            Lint/format configuration (PEP 8, 100-char lines)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Developed and run with Python 3.11–3.14; `requirements.txt` lists the minimum versions.

## Data

The RAW smartphone images, extracted absorbance/calibration CSVs, and UV-Vis reference
spectra that these scripts operate on are not included in this repository — they are
deposited in a public Zenodo dataset alongside the paper
(https://doi.org/10.5281/zenodo.23212497). Download it and place its contents in a
`Data/` folder at the repository root. Each script's module
docstring documents the exact `Data/...` input paths it expects, matching the Zenodo
deposit's folder layout, and the exact output paths it writes.

## Vendored dependency (`vendor/`)

Several scripts (`forward_model.py`, `mb_uvvis_analysis.py`, `figure_S1.py`,
`bandpass_filter_metrics.py`, `raw_image_histograms.py`, `channel_common_origin_check.py`,
`daylight_diffuser_figures*.py`, and the figure scripts that read the Ohlab curves) use a
small `core` package (Bayer ROI/naming
utilities, the Ohlab smartphone spectral-sensitivity loader) and a `roi_selector` module
from the corresponding author's general-purpose `raw-image-analysis` toolbox, which is
not itself part of this publication. The exact modules needed are vendored here under
`vendor/` so the scripts run without any external dependency; see `vendor/core/` and
`vendor/roi_selector.py`. Set the `COLORIMETRY_VENDOR` environment variable to point at a
different copy if needed — otherwise scripts default to the `vendor/` folder next to
`scripts/`.

`vendor/_resources/spectral_sensitivity/` contains the Ohlab smartphone spectral
sensitivity database used throughout (device: `average_all`, the 20-phone average).
**Citation required if you reuse this data** — see
`vendor/_resources/spectral_sensitivity/README.md`:

> Tominaga, S.; Nishi, S.; Ohtera, R. Measurement and Estimation of Spectral Sensitivity
> Functions for Mobile Phone Cameras. *Sensors* **2021**, 21(15), 4985.
> https://doi.org/10.3390/s21154985

## Running the scripts

Run from the repository root once `Data/` (from the Zenodo deposit) is in place, e.g.:

```bash
python scripts/figure_3.py
python scripts/forward_model.py --project /path/to/Ferricyanide --illuminant ... --uvvis ...
```

See `scripts/README.md` for the full script index and `figure_style.md` for the
publication figure style every figure script follows.

## License

Apache License 2.0 — see `LICENSE`.

## Contact

Corresponding authors: F. Javier del Campo (javier.delcampo@bcmaterials.net,
BCMaterials, Basque Center for Materials, Applications and Nanostructures) and
Jean-Yves Hihn (jean-yves.hihn@umlp.fr, Université Marie et Louis Pasteur).
