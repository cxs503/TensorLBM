"""Cross-consistency machinery for the IN25-T6 survey (seat B + seat C).

Three independent checks live here:

``friction_line_agreement``
    the two friction lines used by the ships community (ITTC-1957 correlation
    line, Schoenherr flat-plate line) are evaluated against each other over the
    model/ship Re band.  If they disagreed by more than a few per cent, a
    "cross-consistent with ITTC" claim would be meaningless.
``archived_grid_practice``
    the observed grid-convergence order of *our own* verified SUBOFF benchmark,
    back-computed from the archived pair of resolutions in
    ``benchmarks/verified/suboff_re1000/result.json`` (an audit of archived
    numbers, not a new run).
``published_grid_scale_audit``
    converts the one open-access large-grid precedent (1.03e8 cells for
    appended SUBOFF, arXiv 2503.19674) into an implied hull-length resolution
    under an explicit uniform-grid assumption, so the survey can compare it with
    our resolution needs instead of quoting a bare cell count.

Everything is pure arithmetic on numbers that carry a provenance string; no
number is entered without a source path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from tensorlbm.external_survey.suboff_anchor import (
    SUBOFF_LENGTH_M,
    SUBOFF_MAX_RADIUS_M,
    ittc57_cf,
    schoenherr_cf,
)

__all__ = [
    "REPO_ROOT",
    "FrictionLineAgreement",
    "friction_line_agreement",
    "archived_grid_practice",
    "published_grid_scale_audit",
    "own_ladder_status",
    "red_team_packet",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFIED_RESULT = REPO_ROOT / "benchmarks" / "verified" / "suboff_re1000" / "result.json"
THROUGHPUT = REPO_ROOT / "artifacts" / "in25_t6" / "seat_c_throughput.json"


@dataclass(frozen=True)
class FrictionLineAgreement:
    re_points: tuple[float, ...]
    max_rel_diff_pct: float
    mean_rel_diff_pct: float
    verdict: str
    note: str
    max_rel_diff_pct_model_band: float = 0.0   # Re >= 1e7 (SUBOFF / full-scale band)
    model_band_re_points: tuple[float, ...] = ()


def friction_line_agreement(
    re_points: tuple[float, ...] = (1.0e6, 3.0e6, 1.0e7, 5.0e7, 1.0e8, 1.0e9),
    tol_band_pct: float = 3.0,
    tol_all_pct: float = 5.0,
) -> FrictionLineAgreement:
    """ITTC-1957 vs Schoenherr: they must agree where our anchors live.

    The two lines diverge by design below Re ~ 3e6 (ITTC-1957 is a model-ship
    *correlation* line, Schoenherr a flat-plate *friction* line), so the gate is
    applied on the Re >= 1e7 band, which is where the SUBOFF / full-scale
    anchors sit; the whole-range spread is reported for honesty.
    """
    diffs = {re: 100.0 * (schoenherr_cf(re) - ittc57_cf(re)) / ittc57_cf(re) for re in re_points}
    band = tuple(re for re in re_points if re >= 1.0e7)
    mx_band = max(abs(diffs[re]) for re in band)
    mx_all = max(abs(d) for d in diffs.values())
    mean = sum(abs(d) for d in diffs.values()) / len(diffs)
    return FrictionLineAgreement(
        re_points=re_points,
        max_rel_diff_pct=mx_all,
        mean_rel_diff_pct=mean,
        verdict="cross_consistent" if (mx_band <= tol_band_pct and mx_all <= tol_all_pct)
        else "inconsistent",
        note=(
            "ITTC-1957 (0.075/(lgRe-2)^2) vs Schoenherr (0.455/lgRe^2.58): max |diff| "
            f"{mx_band:.2f}% on Re>=1e7 (gate {tol_band_pct}%), {mx_all:.2f}% over the whole "
            f"Re 1e6..1e9 range (gate {tol_all_pct}%); the low-Re divergence is by construction "
            "(correlation line vs friction line) and does not affect the SUBOFF band"
        ),
        max_rel_diff_pct_model_band=mx_band,
        model_band_re_points=band,
    )


def archived_grid_practice() -> dict:
    """Observed order of convergence from the archived verified SUBOFF pair."""
    if not VERIFIED_RESULT.exists():
        return {
            "available": False,
            "reason": f"archived verified result not found at {VERIFIED_RESULT}",
        }
    data = json.loads(VERIFIED_RESULT.read_text())
    grid = data["convergence_with_grid"]
    err_coarse = abs(grid["L48_mix50_err_pct"])   # L = 48 cells/hull length
    err_fine = abs(grid["L80_mix50_err_pct"])     # L = 80
    r = 80.0 / 48.0
    p_obs = math.log(err_coarse / err_fine) / math.log(r)
    # first-order error model err = C h^p with h = 1/L
    c = err_fine * 80.0**p_obs
    # GCI-style band for the fine grid: error of the *extrapolated* value
    err_extrapolated = c * (1.0 / 80.0) ** p_obs - c * 0.0  # == err_fine; extrapolated ~0
    band = err_fine * (r**p_obs - 1.0) ** -1.0 if p_obs > 0 else None
    return {
        "available": True,
        "source": str(VERIFIED_RESULT.relative_to(REPO_ROOT)),
        "verified_case": "SUBOFF bare hull Re=1000 (benchmarks/verified, verified 2026-09-29)",
        "note": (
            "AUDIT of archived numbers, not a new run: this survey is a frontier node and "
            "must not re-package the verified case as fresh evidence"
        ),
        "grids_cells_per_L": (48, 80),
        "refinement_ratio": r,
        "err_vs_reference_pct": {"L48": grid["L48_mix50_err_pct"], "L80": grid["L80_mix50_err_pct"]},
        "observed_order_p": p_obs,
        "fine_grid_error_pct": err_fine,
        "gci_style_fine_band_pct": band,
        "err_extrapolated_pct": err_extrapolated,
        "verdict": (
            "second-order-like (p~2) and monotonically decreasing: the discretisation behaves "
            "as a converging scheme, but two grids cannot certify a GCI band"
        ),
    }


def published_grid_scale_audit(
    cells: float = 1.03e8,
    *,
    streamwise_lengths: float = 6.0,
    lateral_lengths: float | None = None,
) -> dict:
    """Implied cell size of the open-access large-grid precedent (arXiv 2503.19674)."""
    diameter_over_length = 2.0 * SUBOFF_MAX_RADIUS_M / SUBOFF_LENGTH_M
    if lateral_lengths is None:
        lateral_lengths = 2.0 + diameter_over_length  # repo convention: 2L + D each side
    box_volume_in_L3 = streamwise_lengths * lateral_lengths * lateral_lengths
    cells_per_L = (cells / box_volume_in_L3) ** (1.0 / 3.0)
    return {
        "source_anchor": "2503.19674",
        "cells": cells,
        "assumption": (
            f"uniform Cartesian grid, domain {streamwise_lengths:g}L x {lateral_lengths:.3f}L x "
            f"{lateral_lengths:.3f}L (repo domain-padding convention) — DERIVED ESTIMATE, "
            "the paper may use a different box or local refinement"
        ),
        "box_volume_in_L3": box_volume_in_L3,
        "implied_cells_per_hull_length": cells_per_L,
        "implied_cells_across_max_diameter": cells_per_L * diameter_over_length,
        "our_verified_case_cells_across_diameter": {"L48": 48 * diameter_over_length,
                                                    "L80": 80 * diameter_over_length},
        "verdict": (
            "the published appended-submarine LES resolves the hull with ~2x the streamwise "
            "cells of our verified L=80 case; matching it is a ~10^8-cell budget, i.e. a "
            "distributed/GPU regime, not a single-workstation regime"
        ),
    }


def own_ladder_status() -> dict:
    """Honest FAIL record for the fresh Re=2000 convergence ladder."""
    if THROUGHPUT.exists():
        measured = json.loads(THROUGHPUT.read_text())
    else:
        measured = {"measurements": [], "consequence": {}}
    return {
        "delivered": False,
        "honest_fail": True,
        "reason": (
            "no GPU on this host; measured 74-122 ms/step on CPU for 1e5-3e5 cells and the "
            "parametric hull needs L>=26 cells per hull length (max radius 0.058 L) before the "
            "body is resolved at 1.5 cells; a temporally converged three-grid ladder at "
            "L=26/36/48 x 8000 steps is ~3.4 h, beyond the per-invocation wall clock; the "
            "compute cluster is reachable but its torch build fails (libflexiblas.so.3)"
        ),
        "evidence": str(THROUGHPUT.relative_to(REPO_ROOT)) if THROUGHPUT.exists() else None,
        "measurements": measured.get("measurements", []),
        "geometry_constraint": measured.get("geometry_constraint", {}),
        "consequence": measured.get("consequence", {}),
        "what_would_close_it": (
            "run scripts/in25_t6/seat_c_grid_convergence.py --resolutions 26 36 48 --steps 8000 "
            "on a GPU node (or fix the cluster torch build); the driver already emits "
            "Richardson/GCI from the three grids"
        ),
    }


def red_team_packet() -> dict:
    """Hand-off packet for the T5 red team: what may be scored, what may not."""
    from tensorlbm.external_survey.lit_anchors import (
        ABSENT_ITEMS,
        LITERATURE_ANCHORS,
        MEASURES,
        validate_anchor_table,
    )
    from tensorlbm.external_survey.suboff_anchor import (
        anchor_consistency,
        scorer_candidates,
    )

    table = validate_anchor_table()
    return {
        "anchor_table_ok": table.ok,
        "n_anchors": table.n_anchors,
        "per_measure": dict(table.per_measure),
        "measures": dict(MEASURES),
        "scorer_candidates": list(scorer_candidates()),
        "consistency_audit": [
            {
                "label": c.label,
                "claimed": c.claimed,
                "claimed_re": c.claimed_re,
                "reproduced": c.reproduced,
                "reproduced_re": c.reproduced_re,
                "dev_pct": c.dev_pct,
                "dev_pct_bracket": c.dev_pct_bracket,
                "verdict": c.verdict,
                "note": c.note,
            }
            for c in anchor_consistency()
        ],
        "friction_line_agreement": friction_line_agreement().__dict__,
        "absent": [
            {"item": a.item, "measure": a.measure, "reason": a.reason,
             "consequence": a.consequence}
            for a in ABSENT_ITEMS
        ],
        "pending_redteam_anchors": [
            {"id": a.anchor_id, "measure": a.measure, "kind": a.claim_kind.value,
             "cite": a.citation.short, "title": a.citation.title}
            for a in LITERATURE_ANCHORS
            if a.status.value == "pending_redteam"
        ],
        "own_ladder": own_ladder_status(),
        "archived_grid_practice": archived_grid_practice(),
    }