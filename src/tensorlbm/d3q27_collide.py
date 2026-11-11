"""D3Q27 recommended configuration helper (NOT a collision kernel).

This module only *builds a parameter dict* for the D3Q27 + cascaded solver path
implemented elsewhere (``tensorlbm.advanced_collision``, ``solver3d``,
``wall_model``).  The name is historical; no collision operator lives here.

Usage:
    from tensorlbm.d3q27_collide import suboff_config_d3q27
    cfg = suboff_config_d3q27(nx=200, re=2e6)
    # → D3Q27 CASCADED, no Smagorinsky, wallfn, farfield

Evidence status (G21 verification, docs/G21_ORPHAN_VERIFICATION.md §4): the
earlier header claimed "Validated: SUBOFF bare_hull 200³ → Ct=0.00393 (2.9% vs
ref 0.00405)".  That number cannot be reproduced from any artifact in this
repository, and 0.00405 is the **ITTC-1957 flat-plate friction coefficient**
Cf(Re=2e6) = 0.004054, not a measured or CFD hull resistance coefficient — so
the quoted 2.9 % compares a total hull resistance coefficient with a flat-plate
friction line (a category error, not a validation).  The helper itself is
arithmetically self-consistent (Re = u·L/ν, τ = 0.5 + 3ν) and is exercised by
tests/test_g21_orphan_verification.py; treat it as UNVERIFIED for drag
prediction until a SUBOFF Ct artifact is registered.
"""

from __future__ import annotations

# Re-export for convenience


def suboff_config_d3q27(
    nx: int = 200,
    re: float = 2e6,
    hull_type: str = "bare_hull",
    collision: str = "cascaded",
    n_steps: int = 5000,
    u_in: float = 0.06,
) -> dict:
    """Recommended SUBOFF configuration using D3Q27.

    Args:
        nx: Grid points in streamwise direction (hull_length = 0.4 * nx).
        re: Reynolds number.
        hull_type: "bare_hull" | "full".
        collision: "cascaded" (best) | "cumulant" (also excellent).
        n_steps: Total steps.
        u_in: Free-stream velocity.

    Returns:
        Dict with all parameters needed by run_dg_lbm_suboff_flow or manual loops.
    """
    hl = 0.4 * nx
    nu_lat = u_in * hl / re
    return {
        "nx": nx,
        "ny": int(nx * 0.4),
        "nz": int(nx * 0.4),
        "hull_length": hl,
        "u_in": u_in,
        "re": re,
        "nu_lat": nu_lat,
        "tau": 3.0 * nu_lat + 0.5,
        "lattice": "D3Q27",
        "collision": collision,
        "smagorinsky_cs": 0.0,
        "wall_law": "log",
        "use_van_driest": False,
        "n_steps": n_steps,
        "warmup": n_steps // 3,
        "hull_type": hull_type,
    }
