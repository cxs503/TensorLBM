"""Mass conservation checker for green water simulations.

Validates that total mass (water + air) is conserved to within
the specified tolerance (<2%) throughout the simulation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MassConservationResult:
    """Result of mass conservation check."""
    max_drift_water: float
    max_drift_air: float
    max_drift_total: float
    passes: bool
    tolerance: float
    n_steps_checked: int
    details: list[dict]


def check_mass_conservation(
    mass_history_path: Path | str,
    tolerance: float = 0.02,
) -> MassConservationResult:
    """Check mass conservation from a mass_history.csv file."""
    mass_history_path = Path(mass_history_path)
    if not mass_history_path.exists():
        return MassConservationResult(
            max_drift_water=0.0, max_drift_air=0.0, max_drift_total=0.0,
            passes=False, tolerance=tolerance, n_steps_checked=0,
            details=[{"error": f"File not found: {mass_history_path}"}],
        )

    max_drift_w = 0.0
    max_drift_a = 0.0
    max_drift_t = 0.0
    details: list[dict] = []
    n_steps = 0
    m_w0 = None
    m_a0 = None

    with mass_history_path.open("r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_steps += 1
            m_w = float(row["mass_water"])
            m_a = float(row["mass_air"])
            if m_w0 is None:
                m_w0 = m_w
                m_a0 = m_a
            drift_w = abs(m_w - m_w0) / max(m_w0, 1e-12)
            drift_a = abs(m_a - m_a0) / max(m_a0, 1e-12)
            drift_t = abs((m_w + m_a) - (m_w0 + m_a0)) / max(m_w0 + m_a0, 1e-12)
            max_drift_w = max(max_drift_w, drift_w)
            max_drift_a = max(max_drift_a, drift_a)
            max_drift_t = max(max_drift_t, drift_t)
            details.append({
                "step": int(row["step"]),
                "drift_water": drift_w,
                "drift_air": drift_a,
                "drift_total": drift_t,
            })

    passes = max_drift_w < tolerance and max_drift_a < tolerance and max_drift_t < tolerance
    return MassConservationResult(
        max_drift_water=max_drift_w, max_drift_air=max_drift_a,
        max_drift_total=max_drift_t, passes=passes,
        tolerance=tolerance, n_steps_checked=n_steps, details=details,
    )


def check_mass_from_metadata(
    metadata_path: Path | str,
    tolerance: float = 0.02,
) -> MassConservationResult:
    """Check mass conservation from run_metadata.json."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return MassConservationResult(
            max_drift_water=0.0, max_drift_air=0.0, max_drift_total=0.0,
            passes=False, tolerance=tolerance, n_steps_checked=0,
            details=[{"error": f"File not found: {metadata_path}"}],
        )

    with metadata_path.open("r") as fh:
        meta = json.load(fh)

    max_drift_w = meta.get("max_mass_drift_water", 0.0)
    max_drift_a = meta.get("max_mass_drift_air", 0.0)
    max_drift_t = max(max_drift_w, max_drift_a)
    passes = max_drift_w < tolerance and max_drift_a < tolerance and max_drift_t < tolerance

    return MassConservationResult(
        max_drift_water=max_drift_w, max_drift_air=max_drift_a,
        max_drift_total=max_drift_t, passes=passes,
        tolerance=tolerance, n_steps_checked=meta.get("n_profiles", 0),
        details=[{"source": "metadata", "max_drift_water": max_drift_w, "max_drift_air": max_drift_a}],
    )
