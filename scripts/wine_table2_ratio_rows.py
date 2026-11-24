"""Table 2 rows for RAW G/R and RAW B/R, computed as in wine_case_study.table2_with_sd.

jdc, 2026-10-06: Table 2 shows B/R beside G/R to support the choice of green. Figure 8 and
the wine case-study CSVs are not touched; output is table2_raw_ratio_rows.csv in the same
folder. G/R is recomputed as a check against table2_with_sd.csv.
"""

import pandas as pd
import wine_case_study as w
from numfmt import pm


def rows():
    wines, std = w.camera_tables()
    for frame in (wines, std):
        frame["B/R"] = frame.B - frame.R
    s = std[(std.arm == "RAW") & (std.concentration_uM <= w.CAL_TOP + 1e-9)]
    levels = [0.0, w.SPIKES[8], w.SPIKES[32]]
    out = []
    for sig in ("G/R", "B/R"):
        fc = w.fit(s.concentration_uM, s[sig])
        for wine, g in wines[(wines.arm == "RAW") & (wines.kind == "wine")].groupby("wine"):
            p = g.rename(columns={"concentration_uM": "added_uM", sig: "A"})
            sp = p[p.added_uM.isin(levels)]
            fs_ = w.fit(sp.added_uM, sp.A)
            a0 = p[p.added_uM == 0].A
            c_cal = w.mg_per_L((a0 - fc["intercept"]) / fc["slope"])
            row = dict(
                wine=wine,
                arm="RAW",
                signal=sig,
                Fe_cal_mg_L=c_cal.mean(),
                Fe_cal_sd=c_cal.std(ddof=1),
                Fe_sa_mg_L=w.mg_per_L(fs_["intercept"] / fs_["slope"]),
                Fe_sa_sd=w.mg_per_L(w.intercept_conc_se(sp.added_uM, sp.A, fs_)),
                replicate_sd=p.groupby("added_uM").A.std(ddof=1).mean(),
            )
            for nom, lvl in w.SPIKES.items():
                rec = 100 * (p[p.added_uM == lvl].A - a0.mean()) / fc["slope"] / lvl
                row[f"rec_{nom}"], row[f"rec_{nom}_sd"] = rec.mean(), rec.std(ddof=1)
            out.append(row)
    return pd.DataFrame(out)


def main():
    t = rows()
    t.to_csv(w.OUT / "table2_raw_ratio_rows.csv", index=False)
    t["printed"] = [
        " | ".join(
            pm(a, b)
            for a, b in (
                (r.Fe_cal_mg_L, r.Fe_cal_sd),
                (r.Fe_sa_mg_L, r.Fe_sa_sd),
                (r.rec_8, r.rec_8_sd),
                (r.rec_32, r.rec_32_sd),
            )
        )
        for r in t.itertuples()
    ]
    print(t[["wine", "signal", "replicate_sd", "printed"]].to_string())


if __name__ == "__main__":
    main()
