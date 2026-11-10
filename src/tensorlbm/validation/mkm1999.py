"""Moser-Kim-Mansour (1999) turbulent-channel DNS reference profiles.

The three spectral DNS cases published by Moser, Kim & Mansour (1999,
*Phys. Fluids* 11(4), 943-945) are the standard anchors for wall-bounded
turbulence models at ``Re_tau = 180 / 395 / 590``.  The upstream statistical
database (``chandata.tar.gz`` from ``turbulence.oden.utexas.edu``) ships, per
Reynolds number, a ``.means`` file and a ``.reystress`` file; verbatim copies
live in ``tensorlbm/data/mkm1999/`` next to a ``provenance.json`` that records
the upstream URL, the archive SHA-256 and the per-file SHA-256.

Everything here is pure ``numpy``: the parser is deterministic, has no network
access and re-verifies the shipped SHA-256 so a silently edited reference file
cannot be used as a validation anchor (fail-closed reference data).

Normalisation used by the upstream files (and kept here):

* lengths  -> half channel height ``h`` (so ``y in [0, 1]``),
* velocities -> friction velocity ``u_tau``,
* ``R_ij`` -> ``u_tau**2`` (i.e. ``sqrt(R_uu)`` is ``u_rms+``).

The reference Reynolds numbers actually simulated upstream are
``178.12``, ``395`` and ``590``; the nominal node labels are 180/395/590.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "MKM_DIR",
    "MKMProfile",
    "available_cases",
    "bulk_velocity_plus",
    "friction_coefficient",
    "load_provenance",
    "load_profile",
    "nominal_to_reference_re_tau",
]

#: Directory holding the verbatim upstream profile files.
MKM_DIR = Path(__file__).resolve().parent.parent / "data" / "mkm1999"

#: Nominal node label -> Reynolds number actually simulated by MKM (1999).
_NOMINAL = {180: 178.12, 395: 395.0, 590: 590.0}


def available_cases() -> tuple[int, ...]:
    """Return the nominal Reynolds numbers with shipped reference files."""
    return tuple(sorted(_NOMINAL))


def nominal_to_reference_re_tau(nominal: int | float) -> float:
    """Map a nominal node Reynolds number (180/395/590) to MKM's ``Re_tau``."""
    key = int(round(float(nominal)))
    if key not in _NOMINAL:
        raise KeyError(f"no MKM-1999 reference for Re_tau={nominal!r}")
    return _NOMINAL[key]


def load_provenance() -> dict:
    """Load the shipped provenance record (upstream URL, hashes, date)."""
    path = MKM_DIR / "provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, record in payload["files"].items():
        actual = hashlib.sha256((MKM_DIR / name).read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise ValueError(
                f"reference file {name} does not match provenance "
                f"({actual} != {record['sha256']})"
            )
    return payload


@dataclass(frozen=True)
class MKMProfile:
    """One MKM-1999 case: mean profile plus the Reynolds-stress tensor."""

    nominal_re_tau: int
    re_tau: float
    y: np.ndarray
    y_plus: np.ndarray
    u_plus: np.ndarray
    r_uu: np.ndarray
    r_vv: np.ndarray
    r_ww: np.ndarray
    r_uv: np.ndarray
    r_uw: np.ndarray
    r_vw: np.ndarray
    nun: int
    source_files: tuple[str, str]

    # ---- derived statistics -------------------------------------------
    @property
    def u_rms_plus(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.r_uu, 0.0))

    @property
    def v_rms_plus(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.r_vv, 0.0))

    @property
    def w_rms_plus(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.r_ww, 0.0))

    @property
    def uv_plus(self) -> np.ndarray:
        """``<u'v'>+`` (negative in the whole channel)."""
        return self.r_uv

    def peak(self, quantity: str) -> dict:
        """Location and value of a profile peak in wall units.

        ``quantity`` is one of ``u_rms``, ``v_rms``, ``w_rms`` or
        ``minus_uv``.  The peak is searched over the resolved half channel
        (``y_plus <= re_tau``) and returned with the wall-unit abscissa.
        """
        table = {
            "u_rms": self.u_rms_plus,
            "v_rms": self.v_rms_plus,
            "w_rms": self.w_rms_plus,
            "minus_uv": -self.uv_plus,
        }
        if quantity not in table:
            raise KeyError(f"unknown quantity {quantity!r}")
        values = table[quantity]
        half = self.y_plus <= self.re_tau * 1.001
        index = int(np.argmax(np.where(half, values, -np.inf)))
        return {
            "quantity": quantity,
            "y_plus": float(self.y_plus[index]),
            "value": float(values[index]),
            "y": float(self.y[index]),
        }

    def u_plus_at(self, y_plus: np.ndarray | list[float]) -> np.ndarray:
        """Interpolate ``U+`` at the requested wall-unit stations."""
        return np.interp(np.asarray(y_plus, dtype=float), self.y_plus, self.u_plus)

    def bulk_velocity_plus(self) -> float:
        return bulk_velocity_plus(self.y, self.u_plus)

    def friction_coefficient(self) -> float:
        return friction_coefficient(self.bulk_velocity_plus())


def bulk_velocity_plus(y: np.ndarray, u_plus: np.ndarray) -> float:
    """Half-channel bulk velocity in wall units, ``U_b+ = int_0^1 U+ dy``."""
    y = np.asarray(y, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    if y.shape != u_plus.shape or y.size < 2:
        raise ValueError("y and u_plus must be 1-D arrays of equal length >= 2")
    order = np.argsort(y)
    return float(np.trapezoid(u_plus[order], y[order]) / (y.max() - y.min()))


def friction_coefficient(bulk_velocity_plus_value: float) -> float:
    """``C_f = 2 tau_w / (rho U_b^2) = 2 / U_b+^2``."""
    if bulk_velocity_plus_value <= 0.0:
        raise ValueError("bulk velocity in wall units must be positive")
    return 2.0 / float(bulk_velocity_plus_value) ** 2


def _parse_profile(path: Path) -> tuple[dict[str, str], list[str], np.ndarray]:
    header: dict[str, str] = {}
    columns: list[str] = []
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="latin-1").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            if "=" in body and not body.startswith("-"):
                key, _, value = body.partition("=")
                header[key.strip()] = value.strip()
            elif "," in body and "y+" in body.replace(" ", " "):
                columns = body.replace(",", " ").split()
            continue
        rows.append([float(item) for item in line.split()])
    if columns:
        # Header column names carry the "(y, y+, Umean, ...)" spelling; drop
        # stray characters so callers get plain identifiers.
        columns = [item.strip("()*')") for item in columns]
    return header, columns, np.asarray(rows, dtype=float)


def load_profile(nominal_re_tau: int) -> MKMProfile:
    """Load one MKM-1999 case (files are SHA-256 verified)."""
    load_provenance()
    key = int(round(float(nominal_re_tau)))
    re_tau = nominal_to_reference_re_tau(key)
    means_path = MKM_DIR / f"chan{key}.means"
    stress_path = MKM_DIR / f"chan{key}.reystress"
    means_header, means_columns, means = _parse_profile(means_path)
    stress_header, stress_columns, stress = _parse_profile(stress_path)
    measured = means_header.get("Re_tau") or stress_header.get("Re_tau")
    if measured is not None:
        re_tau = float(measured)
    if means.shape[1] < 3 or stress.shape[1] < 6:
        raise ValueError(f"unexpected MKM column layout for Re_tau={key}")
    if means.shape[0] != stress.shape[0]:
        raise ValueError("means and reystress files disagree on ny")
    y = stress[:, 0]
    y_plus = stress[:, 1]
    return MKMProfile(
        nominal_re_tau=key,
        re_tau=re_tau,
        y=y,
        y_plus=y_plus,
        u_plus=means[:, 2],
        r_uu=stress[:, 2],
        r_vv=stress[:, 3],
        r_ww=stress[:, 4],
        r_uv=stress[:, 5],
        r_uw=stress[:, 6],
        r_vw=stress[:, 7],
        nun=int(means.shape[0]),
        source_files=(means_path.name, stress_path.name),
    )