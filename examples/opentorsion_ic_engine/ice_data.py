"""Load cylinder pressure tables for the ICE example (path anchored to this file)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_EXAMPLE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _EXAMPLE_DIR / "ICE_data"


def ice_data_dir() -> Path:
    """Return the ICE_data directory next to this module."""
    if not _DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"ICE_data not found at {_DATA_DIR}. "
            "Expected examples/opentorsion_ic_engine/ICE_data/ in the repo."
        )
    return _DATA_DIR


def load_pressure_tables():
    """Load normalized pressure-vs-angle and peak-pressure-vs-RPM tables."""
    data_dir = ice_data_dir()
    pressure = np.genfromtxt(data_dir / "pressure_data.csv", delimiter=";")
    peak = np.genfromtxt(data_dir / "peak_data.csv", delimiter=";")
    return (
        pressure[:, 0],
        pressure[:, 1] / np.max(pressure[:, 1]),
        peak[:, 0],
        peak[:, 1],
    )
