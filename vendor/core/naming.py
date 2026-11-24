# Copyright 2024 F. Javier del Campo
# Licensed under the Apache License, Version 2.0

"""
naming.py — filename conventions: blank detection and concentration parsing.
"""

from __future__ import annotations

import re

DEFAULT_BLANK_PREFIXES = (
    "water",
    "blank",
    "blanco",
    "agua",
    "ref_",
    "reference",
    "bkg",
    "control",
)

GROUP_TOL = 0.06  # two concentrations are "same group" if they differ < 6%


def is_blank(stem: str, prefixes: tuple[str, ...] | list[str] = DEFAULT_BLANK_PREFIXES) -> bool:
    """True if the filename stem marks a blank/reference (case-insensitive prefix)."""
    s = stem.lower()
    return any(s.startswith(p) for p in prefixes)


def parse_concentration_um(
    stem: str,
    blank_prefixes: tuple[str, ...] | list[str] = DEFAULT_BLANK_PREFIXES,
) -> float | None:
    """Return concentration in µM, or None if filename is blank/unparseable.

    Handles period and comma decimal separators: 7,812µM 7.812uM 1000µM 125uM,
    followed by _ or end-of-string. \\b fails before _ (which is \\w), so a
    lookahead is used instead.
    """
    if is_blank(stem, blank_prefixes):
        return None
    m = re.match(
        r"^(\d+)(?:[,.](\d+))?\s*([µu]M|mM|M)(?=_|\s|$)",
        stem,
        re.IGNORECASE,
    )
    if not m:
        return None
    integer_part = m.group(1)
    decimal_part = m.group(2) or ""
    unit = m.group(3).lower().replace("µ", "u")
    val = float(f"{integer_part}.{decimal_part}" if decimal_part else integer_part)
    if unit == "um":
        return val
    elif unit == "mm":
        return val * 1000.0
    elif unit == "m":
        return val * 1e6
    return val


def parse_sequence_value(stem: str) -> float | None:
    """First signed number in a filename stem — the experimental state value
    (potential, time, cycle number). Comma and period decimals accepted:
    '-0,8V' → -0.8, 't010s' → 10.0. None if no number present."""
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", stem)
    if not m:
        return None
    return float(m.group(0).replace(",", "."))


def group_conc(c: float, known: list[float], tol: float = GROUP_TOL) -> float:
    """Return nearest known group concentration, or c itself if it's new."""
    for g in known:
        if abs(c - g) / g < tol:
            return g
    return c
