import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def apply():
    out = ROOT / "Data/analysis/ferricyanide/native_nofilter_2026-09-29"
    repeat = ROOT / "Data/analysis/ferricyanide/native_nofilter_repeat_2026-09-30"
    archive = out / "original_2026-09-29"
    archive.mkdir(exist_ok=True)
    for name in [
        "concentration_summary.csv",
        "filter_comparison.csv",
        "frame_checks.csv",
        "findings.md",
    ]:
        if not (archive / name).exists():
            shutil.copy2(out / name, archive / name)
    old = pd.read_csv(archive / "concentration_summary.csv")
    new = pd.read_csv(repeat / "concentration_summary.csv")
    active = old.copy()
    active["source_date"] = "2026-09-29"
    active["reference_method"] = "interpolated_or_endpoint_water_2026-09-29"
    for r in new.itertuples():
        mask = active.concentration_uM == r.concentration_uM
        assert mask.sum() == 1
        active.loc[mask, ["n", "A_mean", "A_sd", "A_pooled", "mean_DN", "clipping_pct_max"]] = [
            r.n,
            r.A_mean,
            r.A_sd,
            r.A_mean,
            r.mean_blue_DN,
            r.clipping_max_pct,
        ]
        active.loc[mask, "source_date"] = "2026-09-30"
        active.loc[mask, "reference_method"] = "pooled_repeat_water_2026-09-30"
    active.to_csv(out / "concentration_summary.csv", index=False)
    filtered = pd.read_csv(
        ROOT / "Data/analysis/ferricyanide/native450_40_2026-09-29/concentration_summary.csv"
    )
    comparison = active.rename(
        columns={"A_mean": "A_mean_nofilter", "A_sd": "A_sd_nofilter"}
    ).merge(
        filtered[["concentration_uM", "A_mean", "A_sd"]].rename(
            columns={"A_mean": "A_mean_450_40", "A_sd": "A_sd_450_40"}
        ),
        on="concentration_uM",
        how="left",
    )
    comparison.to_csv(out / "filter_comparison.csv", index=False)
    frames = pd.read_csv(archive / "frame_checks.csv")
    frames["source_date"] = "2026-09-29"
    frames["reference_method"] = "interpolated_or_endpoint_water_2026-09-29"
    frames = frames[~((frames.kind == "sample") & frames.concentration_uM.isin([2000, 4000]))]
    fresh = pd.read_csv(repeat / "frame_checks.csv")
    fresh["source_date"] = "2026-09-30"
    fresh["reference_method"] = "pooled_repeat_water_2026-09-30"
    # A_interpolated remains empty for repeat frames; use A_selected for active analysis.
    frames["A_selected"] = frames.A_interpolated
    fresh["A_selected"] = fresh.A_pooled
    pd.concat([frames, fresh], ignore_index=True).to_csv(out / "frame_checks.csv", index=False)
    lines = [
        "# Ferricyanide unfiltered native JPEG — active selection updated 30 September 2026",
        "",
        "At Rajendra's explicit request, the original 2000 and 4000 uM sample results are replaced by the three repeat images per level acquired on 30 September. Lower concentrations remain from 29 September. Original results are retained in original_2026-09-29/; source JPEGs are unchanged.",
        "",
        "The replacement levels use their own three fresh water blanks (pooled), not the previous day's references. source_date and reference_method identify each row. A_mean and A_mean_nofilter contain the selected absorbance; A_selected is the corresponding per-frame field. A_interpolated is not populated for repeat frames.",
        "",
        "This is a mixed-day selected dataset, not a single-session calibration. The filtered comparison column remains from 29 September. The replacement does not resolve negative apparent absorbance at the two high concentrations.",
        "",
        "| uM | n | Selected mean A | SD | Source date |",
        "|---|---:|---:|---:|---|",
    ]
    for r in active.itertuples():
        lines.append(
            f"| {r.concentration_uM:g} | {r.n} | {r.A_mean:.6f} | {r.A_sd:.6f} | {r.source_date} |"
        )
    lines += [
        "",
        "Both source analyses found no blue-channel ceiling clipping. The original lower levels lack subsequent water blanks, and the repeat blanks precede the repeat samples. Native colour processing and illumination/reference differences remain unresolved; no validated linear calibration or filter gain is claimed.",
        "The 7.182 uM label is preserved. Full original diagnostic methods and qualifications remain in original_2026-09-29/findings.md and the separate repeat report.",
    ]
    (out / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    assert (active.loc[active.concentration_uM.isin([2000, 4000]), "n"] == 3).all()
    pd.testing.assert_frame_equal(
        active.loc[~active.concentration_uM.isin([2000, 4000]), old.columns].reset_index(drop=True),
        old.loc[~old.concentration_uM.isin([2000, 4000])].reset_index(drop=True),
    )
    print(comparison[comparison.concentration_uM.isin([2000, 4000])].to_string(index=False))


if __name__ == "__main__":
    apply()
