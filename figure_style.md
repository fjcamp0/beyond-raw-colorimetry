# figure_style.md — colorimetry_Project

Canonical style reference for all publication figures, manuscript and supplementary.
Revised 2026-09-30 (jdc: the type in several figures read too small for publication).

**The style is implemented once, in `scripts/figstyle.py`.** Every figure script imports it
and saves through `figstyle.save()`; no script carries its own rcParams block, colour hex
codes or font sizes. This file states the rules and the reasons; `figstyle.py` holds the
values. Change one and you change the other in the same edit.

Target journal: **ACS Measurement Science Au**. Sizes follow the ACS graphics-preparation
guidelines (https://pubs.acs.org/page/4authors/submission/graphics_prep.html) and the journal
author guidelines (`Manuscript/AuthorGuidelines.md` §12).

---

## The five rules

1. **Draw every figure at the width it will print: 3.33 in (single column) or 7.00 in
   (double column).** Font sizes and line widths in the script are then the printed sizes.
   A figure drawn wider and shrunk to fit shrinks its lettering with it: a 10 in figure
   placed at 7 in turns 8 pt into 5.6 pt. This is what made Figures 7 and S6 look small.
2. **A composite figure is laid out as one figure at 7.00 in**, with `plt.subplots` or a
   `GridSpec`. Never assemble it from single-column panels placed side by side at their
   own width (three 3.25 in panels make 9.75 in).
3. **No font size below 7 pt, anywhere.** Scripts never write a number after `fontsize=`;
   they use `fs.FONT_BASE` (9 pt), `fs.FONT_SMALL` (8 pt) or `fs.FONT_PANEL` (10 pt; all raised 1 pt by jdc, 2026-10-05). If a
   label does not fit at 7 pt, shorten the label, move it to the caption, or make the panel
   bigger — do not shrink the type.
4. **Save with `fs.save()`.** It measures every piece of text and every line at the placed
   width and refuses to write the figure if anything prints below 7 pt or 0.5 pt, or if the
   width or depth breaks the ACS limits. It also writes the companion CSV.
5. **Place the figure at its drawn width.** In `Manuscript/figures.yaml`, give the width in
   inches (`width: '7in'`, `width: '3.33in'`) rather than a fraction of `\linewidth`, so the
   draft PDF shows the figure at its true printed size.

`python scripts/check_figure_sizes.py` audits the PNGs already under
`Manuscript/_drafts/figures/`: drawn size, dpi, and what 8 pt and 7 pt lettering become once
placed. Run it before a submission build.

---

## ACS hard limits (do not exceed)

| Property | ACS limit | Our setting |
|---|---|---|
| Single-column width | ≤ 240 pt = 3.33 in (8.46 cm) | `SINGLE_COL` = 3.33 in |
| Double-column width | 300–504 pt = 4.17–7.00 in (10.6–17.8 cm) | `DOUBLE_COL` = 7.00 in |
| Widths between 3.33 and 4.17 in | not allowed | refused by `fs.save()` |
| Max depth incl. caption | ≤ 660 pt = 9.167 in | figure body ≤ 8.5 in |
| Min font (final size) | ≥ 4.5 pt (ACS recommends ~8 pt) | **≥ 7 pt**; primary text 9 pt |
| Min line thickness | ≥ 0.5 pt | 0.6 pt axes, 0.8 pt data lines |
| Resolution — colour | ≥ 300 dpi | 600 dpi |
| Resolution — grayscale | ≥ 600 dpi | 600 dpi |
| Resolution — B/W line art | ≥ 1200 dpi | `fs.DPI_LINEART` = 1200 dpi |
| TOC graphic | exactly 3.25 × 1.75 in | dedicated script |

Photographs and schematics that are not drawn in matplotlib (e.g. the Figure 2 instrument
schematic) obey the same limits: supply them at ≥ 300 dpi at their printed width, and check
that any lettering in them is ≥ 7 pt at that width.

---

## Canvas

| Constant | Value | Use |
|---|---|---|
| `SINGLE_COL` | 3.33 in | single-column figure, and every standalone single panel |
| `DOUBLE_COL` | 7.00 in | double-column and composite figures |
| `CAL_H` | 3.00 in | calibration plot height |
| `SPEC_H` | 2.60 in | spectra height |
| `ROW_H` | 2.00 in | one row of a multi-row composite (1.5–2.0 in) |
| `MAX_BODY_DEPTH` | 8.5 in | figure body; leaves room for the caption |

Standalone single panels (the `figure_3a.png` kind) are drawn at 3.33 in, not 3.43 or 3.44.

---

## Fonts

All sizes are printed sizes, because figures are drawn at printed size (rule 1).

| Element | Size | Constant |
|---|---|---|
| Axis tick labels | 9 pt | `FONT_BASE` |
| Axis labels | 9 pt | `FONT_BASE` |
| Axes titles (if any) | 9 pt | `FONT_BASE` |
| Legend text and title | 8 pt | `FONT_SMALL` |
| Colorbar label and ticks | 8 pt | `FONT_SMALL` |
| In-figure annotations (arrows, peak markers, fit equations) | 8 pt | `FONT_SMALL` |
| Panel letter (A, B, …) | 10 pt bold | `FONT_PANEL`, via `fs.panel_label()` |

- Font family: Arial, then Helvetica, then DejaVu Sans. **Mathematical text is set in
  Arial too** (`mathtext.fontset = custom`), so `$\mu$M` and `$\varepsilon$` match the
  surrounding labels instead of falling back to DejaVu.
- Use LaTeX math notation for symbols: `r"$\mu$M"`, `r"$\log_{10}$"`, `r"$\varepsilon$"`.
  Never use Unicode subscript/superscript characters (₁₀, ²): they render as missing glyphs
  on most backends.
- SVGs keep their text as text (`svg.fonttype = none`) so labels stay editable in
  Inkscape/Illustrator; PDFs embed TrueType (`pdf.fonttype = 42`).

---

## Using the module

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle as fs

fs.apply()

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(fs.DOUBLE_COL, fs.CAL_H))
ax_a.plot(x, y, color=fs.FERIC_COLOR)
ax_a.annotate("c*", xy=(500, 0.4), fontsize=fs.FONT_SMALL, color=fs.FIT_COLOR)
fs.panel_label(ax_a, "A")
fs.panel_label(ax_b, "B")
fs.save(fig, OUT_DIR / "figure_x", data=plotted_df)
```

- Run scripts from the project root; `scripts/` is on the path when a script there is run.
- Layout uses matplotlib's constrained layout (on by default via `fs.apply()`); do not call
  `fig.tight_layout()`. The saved figure is then exactly the width in `figsize`.
- `fs.save(..., tight=True)` crops with `bbox_inches="tight"`, which changes the saved width;
  use it only when something sits outside the axes grid, and `fs.save` checks the cropped
  size instead of `figsize`.
- `fs.save(..., placed_width=...)` is for the rare figure that must be drawn at one size and
  placed at another; the check then runs at the placed size.
- `fs.save(..., strict=False)` turns refusals into warnings. Use it only while drafting.
- `fs.audit(fig)` returns the same report without saving.

---

## Colors

Use the constants; do not retype hex codes. **Each colour has one meaning across the whole
paper, main text and SI** (jdc, 2026-10-06), so a reader who has learnt it in one figure can
read the next without the legend:

| Meaning | Colour | Where it may appear |
|---|---|---|
| Measurement route | UV-Vis grey/black, RAW blues, JPEG oranges (`ROUTE_COLORS`) | every calibration, LOD, recovery or comparison plot |
| 450/40 bandpass filter | dashed line; `ROUTE_HATCH` on bars | with the route colour |
| Model, fit, reference line; annotations | `FIT_COLOR` (= `MODEL_COLOR`), `GRAY` | everywhere |
| Bayer channel R/G/B | `CHANNEL_COLORS` | **only** on wavelength axes, where the channel itself is plotted. In calibration plots the channel goes in the axis label, not the colour |
| Concentration series | `conc_colors(n)`, greys, darker = more concentrated | spectra and integrands at several concentrations |
| Analyte | `FERIC_COLOR`, `MB_COLOR`, `FERROIN_COLOR` | **only** molar-absorptivity spectra (Figure 3d-f) |
| Replicate set / sample (Set 1-3, Wine 1-2) | `SET_COLORS` with `SET_MARKERS` | where sets or samples must be told apart |

Within a route, dark to light separates variants (filtered or ratio vs plain); measured
vs model-corrected data are open vs filled markers of the same route colour. Every figure
must stay readable printed in grayscale: check it by converting the PNG to grey.

### Measurement routes

| Route | Hex (dark → light) | Constant |
|---|---|---|
| UV-Vis | `#252525`, `#6B6B6B`, `#A0A0A0` | `ROUTE_COLORS["uv"]` |
| RAW | `#08306B`, `#2171B5`, `#4292C6` | `ROUTE_COLORS["raw"]` |
| JPEG | `#8C2D04`, `#D94801`, `#FD8D3C` | `ROUTE_COLORS["jpeg"]` |

### Model and annotation

| Use | Hex | Constant |
|---|---|---|
| Model / OLS fit line, annotations | `#2C3E50` | `FIT_COLOR`, `MODEL_COLOR` |
| Secondary annotations, ideal or reference lines | `#626567` | `GRAY` |

### Bayer channels

`CHANNEL_COLORS = {"R": "#C0392B", "G": "#27AE60", "B": "#2471A3"}`, wavelength axes only.

### Analytes (molar-absorptivity spectra only)

| Analyte | Hex | Constant |
|---|---|---|
| K₃[Fe(CN)₆] ferricyanide | `#9C7A00` (ochre, not the JPEG orange) | `FERIC_COLOR` |
| Methylene blue (665 nm, monomer) | `#7D3C98` | `MB_COLOR` |
| Methylene blue (610 nm, dimer) | `#1A5276` | `DIMER_COLOR` |
| Fe(II)-phenanthroline (ferroin) | `#A62A6E` | `FERROIN_COLOR` |

### Replicate sets and samples

| Set | Hex | Marker |
|---|---|---|
| Set 1 / Wine 1 | `#5E3C99` | `o` |
| Set 2 / Wine 2 | `#1B7F7A` | `s` |
| Set 3 | `#9C7A00` | `^` |

`SET_COLORS` and `SET_MARKERS`, in that order.

### Concentration series

`fs.conc_colors(n)` returns n greys from `CONC_CMAP = "Greys"` over `CONC_RANGE`, light at
the lowest concentration and dark at the highest. Normalise on `log10(c)` when the range
spans more than one order of magnitude. A blank (c = 0) is drawn as a `GRAY` dashed line.

---

## Axes and spines

Set by `fs.apply()`; listed here so that nobody overrides them by accident.

- Top and right spines off; bottom and left 0.6 pt.
- No grid. Add a subtle horizontal grid only where point density makes reading hard.
- Ticks point inward, 0.6 pt wide. Minor ticks off.

---

## Calibration (Beer-Lambert) plots

- Individual replicate points: `s=12`, `alpha=0.60`, marker and colour per set.
- Mean ± 1 SD: filled circle (`markersize=4.5`), route colour, `capsize=2.5`,
  `elinewidth=0.8`.
- OLS fit line: `FIT_COLOR`, `lw=0.9`, dashed.
- Legend inside the axes (frame alpha 0.85, no edge, set by `fs.apply()`).
- x-axis in µM up to 1000 µM; in mM from 2000 µM.
- y-axis label: `"Absorbance at XXX nm (AU)"`.

---

## Spectra plots

- One line per concentration, coloured by `log10(c)` with `conc_colors` / `CONC_CMAP`; `lw=0.8`, `alpha=0.9`.
- SD band: `alpha=0.18`, same colour as the line.
- Colorbar on the right, label `r"$\log_{10}(c\,/\,\mu\mathrm{M})$"` at `FONT_SMALL`.
- Wavelength markers (λ_max, shoulder): `FIT_COLOR`, dashed, `lw=0.8`, label at
  `FONT_SMALL`.
- x-axis 350–805 nm (the Cary 60 range measured); y-axis `"Absorbance (AU)"`, bottom −0.02.

---

## Saving and companion data

- `fs.save()` writes PNG (600 dpi) and SVG by default; pass `formats=("png", "svg", "pdf")`
  if a PDF is wanted, and `dpi=fs.DPI_LINEART` for pure black-and-white line art.
- Pass the plotted data as `data=`; it is written as `<figure_name>.csv` beside the figure.
  It holds the data drawn in the figure, not raw instrument files, so the figure can be
  reproduced and the data shared openly. `fs.save` warns when no CSV is given.
- Generated figures live in `Data/paper_figures/` (or `Data/analysis/`,
  `Data/spectra/`), never in `scripts/`.

---

## Migration status (2026-09-30)

`figstyle.py` and this revision were written before the revision-round figure changes, and
**no figure script has been migrated or regenerated yet**, by decision (jdc, 2026-09-30:
the figures are about to change for the revised version). Each script is moved onto
`figstyle` when it is next edited for the revision. The audit of the current PNGs
(`check_figure_sizes.py`) found, at ACS placement:

- **Figure 7** (drawn 10.0 in) and **Figure S6a** (9.8 in): 8 pt prints at 5.6–5.7 pt, and
  their 5.5–6.5 pt annotations lower still. Worst cases; fix first.
- **Figure 2** schematic: 9.25 in at 300 dpi, so its lettering prints at about 6 pt.
  **Table S5**: 9.0 in, 8 pt prints at 6.2 pt.
- Composites drawn at 7.06–7.68 in because of `bbox_inches="tight"` (Figures 4 and 5, S2,
  S4, S6, S7, S8): 7 pt prints at 6.4–6.9 pt.
- Standalone panels at 3.40–3.44 in instead of 3.33 in (Figures 3, S1, S2, S3, S5, S7, S8).
- Hard-coded 5.5–6.5 pt annotations in `figure_3`, `figure_5`, `figure_1b`, `figure_4_*`,
  `figure_S1`–`S3`, `figure_S6`, `figure_S6b`, `figure_S7`–`S9`,
  `figure_dominant_channel_routes`, and `uvvis_calibration_figures` (which also used a
  3.43 in width and a 7 pt base, and is no longer the reference implementation).
