"""sha256-anchored literature envelope for gas-layer / microbubble drag reduction.

Provenance and honesty statement
--------------------------------
The reference values below are **published ranges**, transcribed from the
review-level statements named in each entry, not digitised data points.  No
offline access to the source figures was available inside the work tree, so a
point-by-point digitised comparison is **WITHHELD**; what is done here is an
*envelope* comparison: the lattice drag-reduction at a matched dimensionless gas
flow ``C_q`` must fall inside the published band (with its stated uncertainty).

``REFERENCE_TABLE_SHA256`` anchors the exact table text used by a run, so that a
later reader can detect any edit of the reference numbers.  The hash is written
into every artifact produced by this package.

The dimensionless gas-flow definition used for matching is the one fixed by the
mission, ``C_q = Q_g / (U L)`` (gas volume flow per unit width over the plate
length times the free-stream speed).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

#: Published drag-reduction envelope, one entry per configuration family.
#: ``C_q_range`` and ``DR_range`` are the *ranges* reported by the source;
#: ``DR_uncertainty`` is the transcription uncertainty assigned to the range
#: centre (percentage points, absolute in DR units).
REFERENCE_TABLE: dict[str, object] = {
    "definition": "C_q = Q_g / (U L); DR = 1 - tau_w,g(L)/tau_w,0(L)",
    "provenance_note": (
        "Published ranges transcribed from review-level statements; "
        "point-by-point digitisation WITHHELD (no offline figure access)."
    ),
    "entries": [
        {
            "id": "flat_plate_microbubble",
            "source": "Kodama et al. (2000) Int. J. Heat Fluid Flow 21:582-588",
            "configuration": "flat plate, 50 m, microbubble injection",
            "C_q_range": [0.02, 0.06],
            "DR_range": [0.30, 0.40],
            "DR_uncertainty": 0.10,
        },
        {
            "id": "flat_plate_high_flux",
            "source": "Madavan, Deutsch & Merkle (1984) J. Fluids Eng. 106:414",
            "configuration": "flat-plate boundary layer, high air flux",
            "C_q_range": [0.05, 0.12],
            "DR_range": [0.50, 0.80],
            "DR_uncertainty": 0.15,
        },
        {
            "id": "review_envelope",
            "source": "Ceccio (2010) Annu. Rev. Fluid Mech. 42:183-203 (review)",
            "configuration": "microbubble / air-layer, all configurations",
            "C_q_range": [0.005, 0.10],
            "DR_range": [0.10, 0.80],
            "DR_uncertainty": 0.10,
        },
        {
            "id": "mission_design_envelope",
            "source": "campaign goal statement (DR ~ 10-40 % band)",
            "configuration": "ship-bottom air layer, design range",
            "C_q_range": [0.005, 0.10],
            "DR_range": [0.10, 0.40],
            "DR_uncertainty": 0.10,
        },
    ],
}


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def table_sha256(table: Mapping[str, object] | None = None) -> str:
    """SHA-256 of the canonical JSON serialisation of the reference table."""
    return hashlib.sha256(_canonical_json(table or REFERENCE_TABLE).encode()).hexdigest()


#: Hash of :data:`REFERENCE_TABLE` as shipped.  Pinned so artifacts can prove
#: which reference numbers they were graded against.
REFERENCE_TABLE_SHA256 = table_sha256()


@dataclass(frozen=True)
class EnvelopePoint:
    """One envelope-comparison record."""

    entry_id: str
    C_q: float
    DR_lattice: float
    DR_center: float
    inside_band: bool
    rel_error_to_center: float
    band: tuple[float, float]


def literature_envelope_check(
    C_q: float,
    drag_reduction: float,
    table: Mapping[str, object] | None = None,
    rel_error_target: float = 0.15,
) -> dict[str, object]:
    """Compare one lattice point with the published envelope.

    For every entry whose ``C_q_range`` contains ``C_q`` the band centre and the
    relative deviation are reported.  ``verdict`` is

    * ``"inside_band"``   -- DR within the published range,
    * ``"outside_band"``  -- DR outside the range for all applicable entries,
    * ``"no_matching_cq"``-- the sweep did not reach a published ``C_q``.
    """
    tab = table or REFERENCE_TABLE
    entries = tab["entries"]  # type: ignore[index]
    matches: list[EnvelopePoint] = []
    for entry in entries:
        lo, hi = entry["C_q_range"]  # type: ignore[misc]
        if lo <= C_q <= hi:
            dr_lo, dr_hi = entry["DR_range"]  # type: ignore[misc]
            centre = 0.5 * (dr_lo + dr_hi)
            matches.append(
                EnvelopePoint(
                    entry_id=str(entry["id"]),  # type: ignore[index]
                    C_q=float(C_q),
                    DR_lattice=float(drag_reduction),
                    DR_center=float(centre),
                    inside_band=bool(dr_lo <= drag_reduction <= dr_hi),
                    rel_error_to_center=float(abs(drag_reduction - centre) / centre),
                    band=(float(dr_lo), float(dr_hi)),
                )
            )
    if not matches:
        verdict = "no_matching_cq"
    elif any(m.inside_band for m in matches):
        verdict = "inside_band"
    else:
        verdict = "outside_band"
    best = min(matches, key=lambda m: m.rel_error_to_center) if matches else None
    return {
        "verdict": verdict,
        "C_q": float(C_q),
        "drag_reduction": float(drag_reduction),
        "matches": [
            {
                "entry_id": m.entry_id,
                "band": list(m.band),
                "DR_center": m.DR_center,
                "inside_band": m.inside_band,
                "rel_error_to_center": m.rel_error_to_center,
                "rel_error_le_15pct": bool(m.rel_error_to_center <= rel_error_target),
            }
            for m in matches
        ],
        "best_entry": None if best is None else best.entry_id,
        "reference_table_sha256": table_sha256(tab),
        "note": (
            "Envelope comparison against published ranges; point-by-point "
            "digitised comparison WITHHELD (no offline figure access)."
        ),
    }


def envelope_check_sweep(
    C_q_values: Sequence[float],
    drag_reduction_values: Sequence[float],
    rel_error_target: float = 0.15,
) -> dict[str, object]:
    """Apply :func:`literature_envelope_check` over a whole sweep."""
    records = [
        literature_envelope_check(float(cq), float(dr), rel_error_target=rel_error_target)
        for cq, dr in zip(C_q_values, drag_reduction_values, strict=True)
    ]
    inside = [r for r in records if r["verdict"] == "inside_band"]
    return {
        "records": records,
        "n_points": len(records),
        "n_inside_band": len(inside),
        "all_inside": len(inside) == len(records),
        "reference_table_sha256": REFERENCE_TABLE_SHA256,
    }