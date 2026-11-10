#!/usr/bin/env python3
"""P127W2-N03 feasibility model: what does the campaign specification cost?

The campaign asks for a plane channel ``Lx x H x Lz = 2 pi h x 2h x pi h`` at
``Re_tau = 180 / 395 / 590`` with ``dy+ = 1..2``, ``dx+ = 10..15``, ``dz+ = 5..8``
and a statistical window of ``5000 h / U_bulk``.

Everything below is *arithmetic on that specification* — no fitted constants and
no simulation output.  It answers three questions with numbers:

1. How many lattice cells and time steps does each requested case need?
2. How many cell-updates (the only portable cost unit for LBM) is that?
3. What resolution is affordable for a fixed cell-update budget — and how far is
   it from the requested one?

The specification's own numbers also contain two internal inconsistencies that
are worth stating explicitly, because they are the root of the infeasibility:

* ``dy+ = 1..2`` with ``dz+ = 5..8`` and ``dx+ = 10..15`` is an **anisotropic**
  grid (aspect ratio ~1 : 8 : 4).  ``turbulent_channel3d`` / ``les_channel3d``
  use a **uniform** D3Q19 lattice, so on that solver all three spacings are
  equal in wall units and one cannot have ``dy+ = 1.5`` together with
  ``dz+ = 6.5``; the model below therefore evaluates both variants (uniform at
  ``dy+``, and the anisotropic grid the spec asks for).
* ``Re_tau`` and a fixed domain in wall units fix the node count; the time step
  is fixed by the LBM stability floor ``nu >= 0.5e-3`` (``tau >= 0.5015``),
  which in turn fixes ``u_tau`` for a given ``h`` and hence the number of steps
  to reach ``5000 h/U_bulk``.

Usage::

    python benchmarks/p127w2_n03_les_channel/feasibility.py --out artifacts/campaign/feasibility.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# --- campaign specification (metadata of node p127w2-n03) -------------------
RE_TAU = (180.0, 395.0, 590.0)
DOMAIN_OVER_H = (2.0 * math.pi, 2.0, math.pi)     # Lx, H, Lz in units of h
KAPPA = 0.41
B_LOG = 5.2

# --- numerical constraints of the D3Q19 TRT channel ------------------------
NU_FLOOR = 0.5e-3          # tau_plus = 0.5015: practical stability floor
MAX_MACH = 0.40            # compressibility ceiling used by the campaign runs
U_BULK_OVER_U_TAU = 18.5   # canonical channel-flow ratio (u_bulk/u_tau)
STAT_WINDOW = 5000.0       # h/U_bulk that the campaign asks for


def u_tau_from_mach(nu: float, h: float, re_tau: float) -> float:
    """Largest ``u_tau`` that keeps ``Ma = u_bulk/c_s`` below ``MAX_MACH``.

    ``u_tau = Re_tau nu / h`` (definition of ``Re_tau``) and
    ``u_bulk = U_BULK_OVER_U_TAU u_tau``; the LBM sound speed is ``1/sqrt(3)``.
    """
    u_tau_by_re = re_tau * nu / h
    u_tau_by_mach = MAX_MACH / math.sqrt(3.0) / U_BULK_OVER_U_TAU
    return min(u_tau_by_re, u_tau_by_mach)


def case_cost(re_tau: float, h: float, dy_plus_anisotropy: tuple[float, float, float] | None):
    """Cells, steps and cell-updates for one case at a given ``h``.

    ``dy_plus_anisotropy = (dx+, dy+, dz+)`` selects the anisotropic variant of
    the specification; ``None`` means the uniform lattice (all spacings equal to
    ``h+ / h = Re_tau / h``).
    """
    h_plus = re_tau
    if dy_plus_anisotropy is None:
        dx_plus = dy_plus = dz_plus = h_plus / h
        nx = max(4, int(round(DOMAIN_OVER_H[0] * h)))
        ny = max(4, int(round(DOMAIN_OVER_H[1] * h)) + 2)
        nz = max(4, int(round(DOMAIN_OVER_H[2] * h)))
    else:
        dx_plus, dy_plus, dz_plus = dy_plus_anisotropy
        nx = max(4, int(round(DOMAIN_OVER_H[0] * h_plus / dx_plus)))
        ny = max(4, int(round(DOMAIN_OVER_H[1] * h_plus / dy_plus)) + 2)
        nz = max(4, int(round(DOMAIN_OVER_H[2] * h_plus / dz_plus)))
    cells = nx * ny * nz
    # time: the lattice step is 1, so the window in steps is
    #   T = 5000 h / U_bulk  with  U_bulk = 18.5 u_tau,  u_tau = Re_tau nu / h
    # and nu is pushed to the stability floor (largest admissible viscosity
    # minimises the step count; a smaller nu only makes the run longer).
    nu = max(NU_FLOOR, 0.0)
    u_tau = u_tau_from_mach(nu, max(h, 1.0), re_tau)
    u_bulk = U_BULK_OVER_U_TAU * u_tau
    steps = STAT_WINDOW * h / max(u_bulk, 1e-12)
    steps = max(steps, 1.0)
    eddy_turnover_steps = h / max(u_tau, 1e-12)
    return {
        "h": float(h),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "cells": cells,
        "dx_plus": dx_plus,
        "dy_plus": dy_plus,
        "dz_plus": dz_plus,
        "nu": nu,
        "u_tau": u_tau,
        "u_bulk": u_bulk,
        "steps": steps,
        "eddy_turnover_steps": eddy_turnover_steps,
        "eddy_turnovers": steps / eddy_turnover_steps,
        "cell_updates": cells * steps,
    }


def affordable_h(re_tau: float, budget_cell_updates: float, lo: float = 3.0, hi: float = 2000.0) -> float:
    """Largest ``h`` whose 5000 h/U_bulk campaign fits a cell-update budget."""
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if case_cost(re_tau, mid, None)["cell_updates"] <= budget_cell_updates:
            lo = mid
        else:
            hi = mid
    return lo


def build(budget: float, measured_mlups: float) -> dict:
    spec_uniform = []
    for re_tau in RE_TAU:
        # wall-resolved on the uniform lattice: dy+ = dx+ = dz+ = 2 (the spec's
        # dy+ upper bound; dz+ = 5..8 is unreachable on a uniform grid)
        entry = case_cost(re_tau, re_tau / 2.0, None)
        entry["re_tau"] = re_tau
        entry["variant"] = "uniform_dy_plus_2"
        entry["years_at_measured_rate"] = entry["cell_updates"] / (measured_mlups * 1e6) / 3.156e7
        spec_uniform.append(entry)

    spec_aniso = []
    for re_tau in RE_TAU:
        entry = case_cost(re_tau, re_tau / 1.5, (12.0, 1.5, 6.5))
        entry["re_tau"] = re_tau
        entry["variant"] = "spec_anisotropic_12_1.5_6.5"
        entry["years_at_measured_rate"] = entry["cell_updates"] / (measured_mlups * 1e6) / 3.156e7
        spec_aniso.append(entry)

    affordable = []
    for re_tau in RE_TAU:
        h = affordable_h(re_tau, budget)
        entry = case_cost(re_tau, h, None)
        entry["re_tau"] = re_tau
        entry["budget_cell_updates"] = budget
        entry["dy_plus_gap_vs_spec"] = entry["dy_plus"] / 2.0
        affordable.append(entry)

    ladder = []
    for re_tau in RE_TAU:
        for h in (8, 10, 12, 16):
            entry = case_cost(re_tau, h, None)
            entry["re_tau"] = re_tau
            ladder.append(entry)

    return {
        "schema": "p127w2-n03-feasibility-v1",
        "spec": {
            "re_tau": list(RE_TAU),
            "domain_over_h": list(DOMAIN_OVER_H),
            "dy_plus": [1.0, 2.0],
            "dx_plus": [10.0, 15.0],
            "dz_plus": [5.0, 8.0],
            "statistics_over_h_over_u_bulk": STAT_WINDOW,
            "reference": "MKM1999 (Re_tau=180/395), Moser2021 / Lee-Moser (Re_tau=590)",
        },
        "numerics": {
            "nu_floor": NU_FLOOR,
            "max_mach": MAX_MACH,
            "u_bulk_over_u_tau": U_BULK_OVER_U_TAU,
            "sound_speed": 1.0 / math.sqrt(3.0),
        },
        "measured_mlups": measured_mlups,
        "budget_cell_updates": budget,
        "wall_resolved_uniform": spec_uniform,
        "wall_resolved_spec_anisotropic": spec_aniso,
        "affordable_within_budget": affordable,
        "affordable_ladder": ladder,
        "notes": [
            "cell_updates = cells * steps; the portable LBM cost unit.",
            "steps assume the statistical window of 5000 h/U_bulk and u_bulk/u_tau = 18.5.",
            "uniform-lattice cases cannot satisfy dz+ = 5..8 together with dy+ = 1..2: on a "
            "uniform lattice dx+ = dy+ = dz+ = Re_tau/h.",
            "MACH ceiling 0.40 caps u_tau, which lower-bounds the step count.",
        ],
    }


def markdown(payload: dict) -> str:
    rows = ["| case | variant | cells | steps | cell-updates | years @ measured rate |",
            "|---|---|---|---|---|---|"]
    for entry in payload["wall_resolved_uniform"] + payload["wall_resolved_spec_anisotropic"]:
        rows.append(
            f"| Re_tau={entry['re_tau']:.0f} | {entry['variant']} | "
            f"{entry['cells']:,} | {entry['steps']:,.0f} | {entry['cell_updates']:.3e} | "
            f"{entry['years_at_measured_rate']:.2f} |"
        )
    rows.append("")
    rows.append("| case | affordable h | dy+ | cells | cell-updates | dy+ vs spec (dy+<=2) |")
    rows.append("|---|---|---|---|---|---|")
    for entry in payload["affordable_within_budget"]:
        rows.append(
            f"| Re_tau={entry['re_tau']:.0f} | {entry['h']:.1f} | {entry['dy_plus']:.1f} | "
            f"{entry['cells']:,} | {entry['cell_updates']:.3e} | {entry['dy_plus_gap_vs_spec']:.1f}x |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=1.0e9,
                    help="cell-update budget available for one full campaign case")
    ap.add_argument("--measured-mlups", type=float, default=0.4,
                    help="measured throughput (10^6 cell-updates/s) of this host")
    ap.add_argument("--out", default="artifacts/campaign/feasibility.json")
    args = ap.parse_args(argv)
    payload = build(args.budget, args.measured_mlups)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(markdown(payload))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())