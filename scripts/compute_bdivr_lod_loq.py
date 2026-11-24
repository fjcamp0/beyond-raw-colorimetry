"""Compute LOD/LOQ for the ferricyanide B/R ratio channel (missing from Table 2).

`beer_lambert/calibration_table.csv` never computed a BdivR channel, so no
per-concentration SD exists for it there (unlike B, BdivG, R, etc.). This
derives sigma directly from the per-session A_B - A_R difference at the
lowest calibration standard (7.8 uM), then applies the same IUPAC 3.3-sigma/
10-sigma convention already used for the other Table 2 rows.
"""

import csv
import statistics
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "Data" / "analysis" / "ferricyanide"
ABSORBANCE_TABLE = ANALYSIS_DIR / "beer_lambert" / "absorbance_table.csv"
OUTPUT_CSV = ANALYSIS_DIR / "beer_lambert" / "bdivr_lod_loq.csv"

LOWEST_STANDARD_UM = 7.8
BDIVR_SLOPE_PER_UM = 2.090e-5  # from forward_model_BdivR_vs_BdivG.md, see Table_2.md


def load_b_minus_r_by_session(path: Path, concentration_um: float) -> dict[str, float]:
    """Return {session: A_B - A_R} at the given concentration from the raw absorbance table."""
    per_session: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["concentration_um"]) != concentration_um:
                continue
            if row["channel"] not in ("B", "R"):
                continue
            per_session.setdefault(row["set"], {})[row["channel"]] = float(row["absorbance"])
    return {
        session: channels["B"] - channels["R"]
        for session, channels in per_session.items()
        if "B" in channels and "R" in channels
    }


def main() -> None:
    """Compute B/R sigma, LOD, and LOQ and write them to a CSV alongside the source table."""
    b_minus_r = load_b_minus_r_by_session(ABSORBANCE_TABLE, LOWEST_STANDARD_UM)
    values = list(b_minus_r.values())
    mean = statistics.mean(values)
    sigma = statistics.stdev(values)  # sample SD (ddof=1), matches calibration_table.csv convention
    lod = 3.3 * sigma / BDIVR_SLOPE_PER_UM
    loq = 10 * sigma / BDIVR_SLOPE_PER_UM

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["session", "A_B_minus_A_R_at_7.8uM"])
        for session, value in sorted(b_minus_r.items()):
            writer.writerow([session, f"{value:.6f}"])
        writer.writerow([])
        writer.writerow(["n_sessions", len(values)])
        writer.writerow(["mean", f"{mean:.6f}"])
        writer.writerow(["sigma_sample_sd", f"{sigma:.6f}"])
        writer.writerow(["slope_per_uM", BDIVR_SLOPE_PER_UM])
        writer.writerow(["LOD_uM", f"{lod:.1f}"])
        writer.writerow(["LOQ_uM", f"{loq:.1f}"])

    print(f"n_sessions={len(values)} sigma={sigma:.6f} LOD={lod:.1f} uM LOQ={loq:.1f} uM")
    print(f"written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
