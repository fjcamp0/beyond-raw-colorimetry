"""Recompute Table 2 LOD/LOQ with a pooled replicate sigma (Q-I, decided 2026-07-27).

Table 2 takes sigma as the standard deviation of the lowest calibration standard.
For the recommended ferricyanide B/G channel that point is about four times noisier
than every other level in the fit range, and an SD from three replicates is unstable
enough that the paper's headline detection limit rests on its least reliable number.

This replaces it with the replicate scatter pooled across the levels inside each
row's own fit range (root mean square of the per-level SDs; equal replicate counts,
so no weighting is needed). The physical meaning is preserved -- the scatter of real
repeated measurements, which is what a single determination is subject to -- and the
estimate no longer depends on one level.

Slopes are held at their published values so that sigma is the only thing that
changes. Every row first reproduces its published LOD from the lowest-standard
sigma; a row that fails to reproduce is reported as such rather than silently
carried forward.

Writes Data/analysis/lod_loq_pooled_2026-07-27.csv and prints the Markdown table.
"""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ANALYSIS = Path("Data/analysis")
OUT_CSV = ANALYSIS / "lod_loq_pooled_2026-07-27.csv"

# Levels pooled for the near-blank estimate. Replicate SD scales with absorbance for
# methylene blue, so pooling the whole fit range imports high-concentration noise into
# a quantity that is meant to describe the blank region; the lowest few levels stay in
# that regime while still giving more than one SD to average.
LOW_LEVELS = 3

TABLES = {
    ("ferricyanide", "raw"): ANALYSIS / "ferricyanide/beer_lambert/absorbance_table.csv",
    ("ferricyanide", "sRGB"): ANALYSIS / "ferricyanide/beer_lambert/sRGB/absorbance_table.csv",
    ("methylene_blue", "raw"): ANALYSIS / "methylene_blue/beer_lambert/absorbance_table.csv",
    ("methylene_blue", "sRGB"): ANALYSIS / "methylene_blue/beer_lambert/sRGB/absorbance_table.csv",
    ("ferricyanide", "daylight"): (
        ANALYSIS
        / "ferricyanide/daylight_diffuser/image_analysis/s02_calibrate/absorbance_table.csv"
    ),
}

# Table 2 (v3) rows. slope is per uM; lod/loq are the published values to reproduce.
# channel is either a stored channel name or a derived difference of absorbances --
# a ratio of transmittances is a difference in log space.
ROWS = [
    dict(
        analyte="ferricyanide",
        domain="sRGB",
        label="B (sRGB)",
        channel="B",
        lo=7.8,
        hi=500,
        slope=3.335e-5,
        lod=322,
        loq=974,
    ),
    dict(
        analyte="ferricyanide",
        domain="sRGB",
        label="B/G (sRGB)",
        channel="BdivG",
        lo=7.8,
        hi=1000,
        slope=2.070e-5,
        lod=218,
        loq=662,
    ),
    dict(
        analyte="ferricyanide",
        domain="raw",
        label="B (raw)",
        channel="B",
        lo=7.8,
        hi=500,
        slope=4.762e-5,
        lod=374,
        loq=1135,
    ),
    dict(
        analyte="ferricyanide",
        domain="raw",
        label="B/G (raw)",
        channel="BdivG",
        lo=7.8,
        hi=1000,
        slope=2.082e-5,
        lod=208,
        loq=629,
    ),
    dict(
        analyte="methylene_blue",
        domain="sRGB",
        label="R (sRGB)",
        channel="R",
        lo=0.78,
        hi=25,
        slope=6.606e-3,
        lod=1.60,
        loq=4.85,
    ),
    dict(
        analyte="methylene_blue",
        domain="sRGB",
        label="R/B (sRGB)",
        channel="RdivB",
        lo=0.78,
        hi=25,
        slope=6.117e-3,
        lod=0.57,
        loq=1.74,
    ),
    dict(
        analyte="methylene_blue",
        domain="sRGB",
        label="R/G (sRGB)",
        channel="RdivG",
        lo=0.78,
        hi=25,
        slope=5.224e-3,
        lod=0.42,
        loq=1.28,
    ),
    dict(
        analyte="methylene_blue",
        domain="raw",
        label="R (raw)",
        channel="R",
        lo=0.78,
        hi=50,
        slope=8.248e-3,
        lod=2.64,
        loq=8.01,
    ),
    dict(
        analyte="methylene_blue",
        domain="raw",
        label="R/B (raw)",
        channel=("R", "B"),
        lo=0.78,
        hi=50,
        slope=7.007e-3,
        lod=0.57,
        loq=1.72,
    ),
    dict(
        analyte="methylene_blue",
        domain="raw",
        label="R/G (raw)",
        channel=("R", "G"),
        lo=0.78,
        hi=50,
        slope=5.343e-3,
        lod=0.45,
        loq=1.38,
    ),
]

# Not in Table 2, but quoted in the prose and the SI, so they must move to the same
# convention or the table and the text will disagree. The two daylight rows were never
# on the lowest-standard convention at all: calibrate.py wrote them from its internal
# residual-based LOD, which is why they carry from_residual and skip the reproduction
# check. That mismatch is itself worth reporting -- see the note this script prints.
EXTRA_ROWS = [
    dict(
        analyte="ferricyanide",
        domain="raw",
        label="B (raw), 4000 range",
        channel="B",
        lo=7.8,
        hi=4000,
        slope=2.202e-5,
        lod=810,
        loq=2450,
        where="section 4.3 prose",
    ),
    dict(
        analyte="ferricyanide",
        domain="sRGB",
        label="B (sRGB), 4000 range",
        channel="B",
        lo=7.8,
        hi=4000,
        slope=1.9494e-5,
        lod=551,
        loq=1679,
        where="section 4.3 prose",
    ),
    dict(
        analyte="ferricyanide",
        domain="daylight",
        label="B/G (daylight)",
        channel="BdivG",
        lo=7.1,
        hi=500,
        slope=3.55804e-5,
        lod=40,
        loq=122,
        where="section 4.5, SI",
        from_residual=True,
    ),
    dict(
        analyte="ferricyanide",
        domain="daylight",
        label="B (daylight)",
        channel="B",
        lo=7.1,
        hi=500,
        slope=6.17046e-5,
        lod=142,
        loq=431,
        where="section 4.5",
        from_residual=True,
    ),
]


def load(analyte, domain):
    """Per-image absorbances as {filename: (concentration, {channel: absorbance})}.

    Keyed on filename, not on set: the methylene blue sRGB table labels every
    image "PNG", so a (set, concentration) key would collapse its replicates.
    """
    table = {}
    with open(TABLES[(analyte, domain)], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            conc, channels = table.setdefault(row["filename"], (float(row["concentration_um"]), {}))
            channels[row["channel"]] = float(row["absorbance"])
    return table


def value_of(channels, spec):
    """One replicate's value for a stored or derived channel.

    A tuple spec is a difference of absorbances. "G" means the mean of the two
    green Bayer planes, which is the definition that reproduces the published
    methylene blue R/G row.
    """
    if isinstance(spec, str):
        return channels[spec]
    numerator, denominator = spec
    if denominator == "G" and "G1" in channels:
        second = (channels["G1"] + channels["G2"]) / 2
    else:
        second = channels[denominator]
    return channels[numerator] - second


def per_level_sd(table, spec, lo, hi):
    """{concentration: replicate SD} for levels inside the fit range."""
    grouped = defaultdict(list)
    for conc, channels in table.values():
        if lo <= conc <= hi:
            grouped[conc].append(value_of(channels, spec))
    return {c: statistics.stdev(v) for c, v in sorted(grouped.items()) if len(v) > 1}


def rms(values):
    return math.sqrt(sum(v * v for v in values) / len(values))


def main():
    results = []
    for row in ROWS + EXTRA_ROWS:
        sds = per_level_sd(
            load(row["analyte"], row["domain"]), row["channel"], row["lo"], row["hi"]
        )
        levels = sorted(sds)
        lowest = sds[levels[0]]
        pooled_full = rms([sds[c] for c in levels])
        pooled_low = rms([sds[c] for c in levels[:LOW_LEVELS]])

        reproduced = 3.3 * lowest / row["slope"]
        drift = 100 * abs(reproduced - row["lod"]) / row["lod"]
        results.append(
            dict(
                row=row["label"],
                where=row.get("where", "Table 2"),
                from_residual=row.get("from_residual", False),
                range_um=f"{row['lo']:g}-{row['hi']:g}",
                n_levels=len(sds),
                sd_trend=sds[levels[-1]] / sds[levels[0]],
                sigma_lowest=lowest,
                sigma_pooled_low=pooled_low,
                sigma_pooled_full=pooled_full,
                lod_published=row["lod"],
                lod_reproduced=reproduced,
                reproduction_error_pct=drift,
                lod_pooled_low=3.3 * pooled_low / row["slope"],
                lod_pooled_full=3.3 * pooled_full / row["slope"],
                loq_published=row["loq"],
                loq_pooled_low=10 * pooled_low / row["slope"],
                loq_pooled_full=10 * pooled_full / row["slope"],
            )
        )

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        for r in results:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    print("Reproduction of published values from the lowest-standard sigma:")
    for r in results:
        if r["from_residual"]:
            flag = "n/a - published value came from the residual-SD convention"
        else:
            flag = "ok" if r["reproduction_error_pct"] < 2 else "MISMATCH"
        print(
            f"  {r['row']:<22} published {r['lod_published']:>7} "
            f"recomputed {r['lod_reproduced']:>8.1f}  "
            f"({r['reproduction_error_pct']:5.1f}%)  {flag}"
        )

    print(
        f"\n| Row | Where | Range (uM) | Levels | LOD now | LOD low-{LOW_LEVELS} | "
        f"LOQ now | LOQ low-{LOW_LEVELS} |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        fmt = "{:.2f}" if r["lod_pooled_low"] < 10 else "{:.0f}"
        print(
            f"| {r['row']} | {r['where']} | {r['range_um']} | {r['n_levels']} | "
            f"{r['lod_published']} | {fmt.format(r['lod_pooled_low'])} | "
            f"{r['loq_published']} | {fmt.format(r['loq_pooled_low'])} |"
        )
    print(
        "\nSD trend = replicate SD at the top of the fit range divided by the SD at the "
        "bottom.\nA large value means the noise is concentration-dependent, so pooling "
        "over the full\nrange overestimates the noise near zero."
    )
    print(f"\nwritten to {OUT_CSV}")


if __name__ == "__main__":
    main()
