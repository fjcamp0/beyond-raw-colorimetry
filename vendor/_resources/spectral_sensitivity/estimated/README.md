# Estimated Spectral Sensitivity Curves (SOURCE C)

**Predicted, not measured.** Use only when no measured curve exists for the
camera (Ohlab covers smartphones; colour-science covers two DSLRs). A measured
file with the same device key always wins at load time — see
[`core/spectral.py`](../../../core/spectral.py).

## Source & citation (REQUIRED when used)

> Solomatov, G.; Akkaynak, D.
> **Spectral Sensitivity Estimation Without a Camera.**
> *IEEE International Conference on Computational Photography (ICCP)* **2023**.
> arXiv:2304.11549 — https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation

`citation_for(device)` returns this string automatically for any device whose
`source_of(...)` is `estimated_eilat`, and it is flagged as ESTIMATED in
`analyte_channel_report`.

## License caveat

The upstream repository ships **no LICENSE file**. These curves are therefore
**fetched on demand and never committed** to this toolbox (see `.gitignore`).
Confirm reuse terms with the authors before publishing derived results.

## How to fetch

```bash
# Search the ~1000-camera dataset
python scripts/fetch_estimated_sensitivity.py --list "nikon d7"

# Fetch one — writes <device_key>.txt into this folder
python scripts/fetch_estimated_sensitivity.py --camera "Nikon D7000"
```

The fetcher resamples nothing (upstream grid is already 400–700 nm / 10 nm),
peak-normalises to ≈1 by the **global** R/G/B max (preserving cross-channel
ratios), and prints the `device_key` to put in `experiment.yaml`:

```yaml
device: nikon_d7000
```

## File format

Identical to the measured files one level up: **31 rows × 3 tab-separated
columns** (400–700 nm / 10 nm; R, G, B; peak ≈ 1). Lines beginning with `#`
are provenance headers and are ignored by the loader.
