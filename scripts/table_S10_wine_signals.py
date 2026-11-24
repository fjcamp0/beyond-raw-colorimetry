"""Supplementary Table S10: every camera signal in the two wines.

Reads the standard-addition fits of wine_case_study.py and writes the table
as CSV and as the Markdown rows of Manuscript/_drafts/Table_S10_vN.md.
"""

from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
FOLDER = PROJ / "Data/analysis/wine_case_study_2026-09-30"
SIGNALS = ["G/R", "B/G", "G", "B"]


def sig(x, n):
    """Round to n significant figures, printed without exponent."""
    text = f"{float(f'{x:.{n}g}'):f}".rstrip("0").rstrip(".")
    digits = len(text.replace("-", "").replace(".", "").lstrip("0"))
    return text + (
        "0" * (n - digits) if "." in text else ("." + "0" * (n - digits) if digits < n else "")
    )


def main():
    r = pd.read_csv(FOLDER / "wine_case_study_results.csv")
    r = r[r.arm.isin(["RAW", "JPEG"])].copy()
    ref = r[r.signal == "G/R"].set_index(["wine", "arm"]).sa_rmse
    r["rmse_vs_GR"] = [row.sa_rmse / ref[(row.wine, row.arm)] for row in r.itertuples()]
    r["signal"] = pd.Categorical(r.signal, SIGNALS, ordered=True)
    r = r.sort_values(["arm", "wine", "signal"], ascending=[False, True, True])
    out = pd.DataFrame(
        {
            "arm": r.arm,
            "wine": r.wine,
            "signal": r.signal.astype(str),
            "levels_uM": r.sa_levels_uM.str.replace("13.0891", "13.1").str.replace(
                "52.3565", "52.4"
            ),
            "n": r.sa_n.astype(int),
            "slope_1e-3_AU_per_uM": (r.sa_slope_per_uM * 1e3).map(lambda x: sig(x, 3)),
            "r2": r.sa_r2.map(lambda x: f"{x:.4f}" if x > 0.99 else f"{x:.2f}"),
            "rmse_1e-3_AU": (r.sa_rmse * 1e3).map(lambda x: sig(x, 2)),
            "rmse_vs_GR": r.rmse_vs_GR.map(lambda x: sig(x, 2)),
            "recovery_52uM_pct": [
                f"{v:.0f}" if q >= 0.5 else "—" for v, q in zip(r.sa_recovery_32uM_pct, r.sa_r2)
            ],
            "ceiling_52uM_pct": r.saturated_pct_max_32uM.round(0).astype(int),
        }
    )
    out.to_csv(FOLDER / "table_S10_wine_signals.csv", index=False)
    cols = [
        "arm",
        "wine",
        "signal",
        "slope_1e-3_AU_per_uM",
        "r2",
        "rmse_1e-3_AU",
        "rmse_vs_GR",
        "recovery_52uM_pct",
        "ceiling_52uM_pct",
    ]
    for row in out[cols].itertuples(index=False):
        cells = [str(v).replace("Wine", "").replace("-", "−") for v in row]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
