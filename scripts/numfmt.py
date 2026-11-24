"""Number formatting for the paper (CLAUDE.md hard rule 10, Stephania, 2026-10-01).

A value with an uncertainty is printed with the uncertainty rounded to two significant
figures and the value rounded to the same decimal place:

    pm(0.23412, 0.04512)  -> "0.234 ± 0.045"
    pm(301.75, 27.36)     -> "302 ± 27"
    pm(0.72298, 0.00483)  -> "0.7230 ± 0.0048"

Every script that writes "value ± SD" into a paper table, figure or report uses this.
"""

from __future__ import annotations

import math


def decimals(sd: float) -> int:
    """Decimal places that leave `sd` with two significant figures."""
    d = 1 - math.floor(math.log10(abs(sd)))
    # Rounding can carry into a new digit (0.0996 -> 0.10); recheck at that precision.
    if abs(round(abs(sd), d)) >= 10 ** (2 - d):
        d -= 1
    return max(0, d)


def pm(value: float, sd: float, sep: str = " ± ", sign: bool = False) -> str:
    """'value ± sd' with sd to two significant figures and value to the same decimal.

    A zero, missing or non-finite sd cannot set a precision: the value is then given to
    four significant figures with no uncertainty, rather than as "± 0".
    """
    if sd is None or not math.isfinite(sd) or sd == 0:
        return f"{value:+.4g}" if sign else f"{value:.4g}"
    d = decimals(sd)
    v = f"{value:+.{d}f}" if sign else f"{value:.{d}f}"
    return f"{v}{sep}{abs(sd):.{d}f}"
