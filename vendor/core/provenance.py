# Copyright 2024 F. Javier del Campo
# Licensed under the Apache License, Version 2.0

"""
provenance.py — run provenance records for reproducibility.

Every stage writes a run_config.yaml next to its outputs: resolved
parameters, tool version (git commit of doctor-mirabilis), and timestamp.
This is what a reviewer needs to reproduce a figure.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def tool_commit() -> str:
    """Short git SHA of the toolbox repo for provenance; 'unknown' if unavailable."""
    try:
        return (
            subprocess.run(
                ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _clean(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


def write_run_config(out_dir: Path, stage: str, params: dict) -> Path:
    """Write run_config.yaml with resolved parameters. Paths are stringified."""
    record = {
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "doctor-mirabilis/raw-image-analysis",
        "tool_commit": tool_commit(),
        "parameters": _clean(params),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "run_config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(record, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return path
