"""3D square-duct Poiseuille: resolution ladder + observed convergence order.

:mod:`tensorlbm.duct_poiseuille_validation` established *one* D3Q27 MRT
square-duct operating point against the Boussinesq series (0.4 %-class profile
error at ``L = 10``) and proved the wall/geometry/streaming composition is
executable.  What it does not establish is the thing a benchmark anchor is
really worth: **how the error behaves under refinement**, i.e. whether the
agreement at one resolution is an accident of the discretisation or a genuine
second-order approach to the continuum solution.

This module adds that ladder.  It runs the same physics (body-force driven,
periodic along ``x``, half-way bounce-back on two pairs of walls, MRT
collision) at four duct widths ``L in {6, 8, 10, 12}`` with the *dimensionless
problem held fixed*::

    body_force = 8 nu u_nominal / L^2

which keeps the flow speed (and hence the Reynolds number and the Mach number)
fixed to within a few percent across the ladder, so the sequence is a
discretisation study rather than a scan through physics.  The residual drift
of the *measured* peak velocity (0.0272 -> 0.0289 over the ladder) is itself
part of the discretisation error the ladder quantifies, and it is reported
rung by rung rather than assumed away.
For each rung it measures

* ``velocity_profile_error`` -- ``max|u_num - u_exact| / max(u_exact)`` on the
  mid-plane centreline, against the closed-form Boussinesq series;
* ``mass_conservation_error`` -- ``|sum(rho) - N| / N`` over the full lattice;
* the integral Poiseuille number ``f Re = 2 g L^2 / (nu u_mean)`` against the
  Shah & London (1978) laminar square-duct value ``56.91``;
* a steady-state residual so a non-converged rung cannot masquerade as a
  discretisation error.

Then it derives the observed order of accuracy two independent ways:
adjacent-pair orders ``log(e_i/e_{i+1}) / log(L_{i+1}/L_i)`` and a
least-squares slope of ``log e`` versus ``log L`` over the whole ladder
(reporting ``R^2`` so a non-power-law ladder is visible rather than asserted).

Control with discriminating power (the point of a *negative* control): the
textbook circular-pipe parabola ``u = u_c (1 - (y/a)^2)``, peak-matched to the
exact square-duct profile, is compared against the same measurement.  For a
square duct that profile is *wrong* (the true profile is a cosh/cosine series
with corner singularities), so if the parabola passed the 2 % gate the gate
would carry no discriminating power.  The ladder therefore also reports
``parabola_error`` and the test suite asserts it is *rejected*.

Scope (honest): single-phase, static geometry, ``cpu``, ``float64``,
``tau = 1`` (``nu = 1/6``), no turbulence / multiphase / moving boundary /
accelerator backend.  This is a cold-path evidence writer.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .duct_poiseuille_validation import (
    LITERATURE_POISEUILLE_NUMBER,
    _canonical_sha256,
    duct_coords,
    run_duct_case,
)

ARTIFACT_ID = "poiseuille-3d-resolution-r1"
ARTIFACT_VERSION = "poiseuille-3d-resolution-r1"

#: ``(length, steps)`` ladder.  Steps sit at the measured steady-state onset
#: (``steady_rel_change_tail < 1e-9`` on every rung; extra steps do not move
#: the error -- verified in ``tests/test_poiseuille_3d.py``).
DEFAULT_LADDER: tuple[tuple[int, int], ...] = ((6, 1200), (8, 1500), (10, 2000), (12, 2000))

#: Ecosystem acceptance thresholds (ship-performance-platform standard).
PROFILE_TOLERANCE = 0.02
MASS_TOLERANCE = 0.01
ORDER_TARGET = 1.5
LITERATURE_TOLERANCE = 0.03


def parabola_profile(length: int, u_peak: float) -> np.ndarray:
    """Peak-matched textbook parabola ``u = u_peak (1 - (y/a)^2)`` on the grid.

    ``a = L/2`` is the duct half-width and coordinates come from the same
    ``duct_coords`` used by the series reference, so the two profiles are
    sampled at identical points.
    """
    y = duct_coords(length).numpy()
    a = 0.5 * float(length)
    return u_peak * (1.0 - (y / a) ** 2)


def observed_order(errors: Sequence[float], lengths: Sequence[int]) -> dict[str, Any]:
    """Order of accuracy from adjacent pairs and from a global log-log fit."""
    err = np.asarray(errors, dtype=np.float64)
    h = np.asarray(lengths, dtype=np.float64)
    if err.size != h.size or err.size < 2:
        raise ValueError("need >= 2 matched (error, length) samples")
    if np.any(err <= 0.0):
        raise ValueError("errors must be strictly positive")
    adjacent = [
        float(math.log(err[i] / err[i + 1]) / math.log(h[i + 1] / h[i]))
        for i in range(err.size - 1)
    ]
    x = np.log(h)
    y = np.log(err)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "orders_adjacent": adjacent,
        "order_fit": float(-slope),
        "fit_r2": float(r2),
        "fit_prefactor": float(math.exp(intercept)),
        "order_min_adjacent": float(min(adjacent)),
        "order_max_adjacent": float(max(adjacent)),
    }


def run_ladder(
    ladder: Sequence[tuple[int, int]] = DEFAULT_LADDER,
    *,
    nx: int = 4,
    tau: float = 1.0,
    u_nominal: float = 0.05,
    collision: str = "mrt",
) -> dict[str, Any]:
    """Run the resolution ladder and assemble the acceptance metrics."""
    # Lattices here are tiny (<= 14^2 * 4 cells): torch's intra-op thread
    # dispatch then dominates the wall clock by an order of magnitude, so pin
    # to one thread and restore the caller's setting afterwards.
    saved_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _run_ladder_serial(
            ladder, nx=nx, tau=tau, u_nominal=u_nominal, collision=collision
        )
    finally:
        torch.set_num_threads(saved_threads)


def _run_ladder_serial(
    ladder: Sequence[tuple[int, int]],
    *,
    nx: int,
    tau: float,
    u_nominal: float,
    collision: str,
) -> dict[str, Any]:
    rungs: list[dict[str, Any]] = []
    for length, steps in ladder:
        t0 = time.time()
        case = run_duct_case(
            length=length,
            nx=nx,
            steps=steps,
            tau=tau,
            u_nominal=u_nominal,
            collision=collision,
            collect_profile=True,
        )
        u_num = np.asarray(case["profile_num_midz"], dtype=np.float64)
        u_exact = np.asarray(case["profile_exact_midz"], dtype=np.float64)
        u_peak = float(u_exact.max())
        para = parabola_profile(length, u_peak)
        rung = {
            "length": int(length),
            "steps": int(steps),
            "cells": int((length + 2) ** 2 * nx),
            "nu": case["nu"],
            "body_force": case["body_force"],
            "u_max_num": case["u_max_num"],
            "u_max_exact": case["u_max_exact"],
            "velocity_profile_error": case["rel_error_max"],
            "velocity_profile_error_mean": case["rel_error_mean"],
            "mass_conservation_error": case["mass_drift"],
            "steady_rel_change_tail": case["steady_rel_change_tail"],
            "poiseuille_number_num": case["poiseuille_number_num"],
            "poiseuille_number_exact": case["poiseuille_number_exact"],
            "poiseuille_number_rel_error": abs(
                case["poiseuille_number_num"] - case["poiseuille_number_exact"]
            )
            / case["poiseuille_number_exact"],
            "parabola_error": float(np.max(np.abs(u_num - para)) / u_peak),
            "parabola_error_mean": float(np.mean(np.abs(u_num - para)) / u_peak),
            # Independent *integral* observable that does not reuse the
            # pointwise max metric: the cross-section mean velocity (midpoint
            # rule over every fluid cell of the y-z plane), which is also what
            # the resistance / Poiseuille number is built from.
            "u_mean_num": case["u_mean_num"],
            "u_mean_exact": case["u_mean_exact"],
            "mean_velocity_rel_error": abs(case["u_mean_num"] - case["u_mean_exact"])
            / abs(case["u_mean_exact"]),
            # centreline samples, kept in the artifact so figures and tables can
            # be regenerated from the JSON alone (no re-simulation, no hand copy)
            "coords_y": duct_coords(length).tolist(),
            "profile_num": u_num.tolist(),
            "profile_exact": u_exact.tolist(),
            "profile_parabola": para.tolist(),
            "wall_clock_s": round(time.time() - t0, 3),
        }
        rungs.append(rung)

    errors = [r["velocity_profile_error"] for r in rungs]
    lengths = [r["length"] for r in rungs]
    order = observed_order(errors, lengths)
    mean_errors = [r["mean_velocity_rel_error"] for r in rungs]
    mean_order = observed_order(mean_errors, lengths)
    finest = rungs[-1]
    metrics = {
        "velocity_profile_error": finest["velocity_profile_error"],
        "velocity_profile_error_length": finest["length"],
        "velocity_profile_error_ladder": errors,
        "convergence_order": order["order_fit"],
        "convergence_order_min_adjacent": order["order_min_adjacent"],
        "mean_velocity_rel_error": finest["mean_velocity_rel_error"],
        "mean_velocity_error_ladder": mean_errors,
        "mean_velocity_order": mean_order["order_fit"],
        "mass_conservation_error": max(r["mass_conservation_error"] for r in rungs),
        "parabola_error": finest["parabola_error"],
        "poiseuille_number_num": finest["poiseuille_number_num"],
        "poiseuille_number_rel_error_vs_shah_london": abs(
            finest["poiseuille_number_num"] - LITERATURE_POISEUILLE_NUMBER
        )
        / LITERATURE_POISEUILLE_NUMBER,
    }
    acceptance = {
        "velocity_profile_error": metrics["velocity_profile_error"] <= PROFILE_TOLERANCE,
        "convergence_order": metrics["convergence_order"] >= ORDER_TARGET,
        "mean_velocity_error": metrics["mean_velocity_rel_error"] <= PROFILE_TOLERANCE,
        "mean_velocity_order": metrics["mean_velocity_order"] >= ORDER_TARGET,
        "mass_conservation_error": metrics["mass_conservation_error"] <= MASS_TOLERANCE,
        "monotone_error_decrease": all(
            errors[i + 1] < errors[i] for i in range(len(errors) - 1)
        ),
        "steady_at_every_rung": all(r["steady_rel_change_tail"] < 1e-9 for r in rungs),
        "poiseuille_number_vs_literature": metrics[
            "poiseuille_number_rel_error_vs_shah_london"
        ]
        <= LITERATURE_TOLERANCE,
        # negative control: a wrong-but-plausible profile must be rejected by
        # the same 2 % gate, otherwise the gate has no discriminating power.
        "parabola_negative_control_rejected": metrics["parabola_error"] > PROFILE_TOLERANCE,
    }
    return {
        "artifact_id": ARTIFACT_ID,
        "artifact_version": ARTIFACT_VERSION,
        "case": "square_duct_poiseuille_3d_resolution_ladder",
        "lattice": "D3Q27 MRT collision, half-way bounce-back walls",
        "collision": collision,
        "tau": tau,
        "u_nominal": u_nominal,
        "nx": nx,
        "ladder": [[int(L), int(s)] for L, s in ladder],
        "rungs": rungs,
        "order": order,
        "order_mean_velocity": mean_order,
        "metrics": metrics,
        "acceptance": acceptance,
        "thresholds": {
            "velocity_profile_error": PROFILE_TOLERANCE,
            "convergence_order": ORDER_TARGET,
            "mass_conservation_error": MASS_TOLERANCE,
            "poiseuille_number_literature_tolerance": LITERATURE_TOLERANCE,
        },
        "notes": (
            "Half-way bounce-back + MRT (s_e=1.19, s_eps=1.4, s_q=1.2), body "
            "force injected as w*3*ci_x*g on fluid cells only, float64, single "
            "thread. Reference = Boussinesq series (self-checked in "
            "duct_poiseuille_validation.check_series). 'parabola_error' is a "
            "NEGATIVE control: the circular-pipe parabola is not the square-duct "
            "solution, so it must fail the same 2 % gate."
        ),
    }


def build_artifact(**kwargs: Any) -> dict[str, Any]:
    payload = run_ladder(**kwargs)
    payload["ladder_sha256"] = _canonical_sha256(
        {"rungs": payload["rungs"], "order": payload["order"]}
    )
    payload["all_accepted"] = all(payload["acceptance"].values())
    return payload


def write_artifact(path: str | Path, payload: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = payload or build_artifact()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[2]
            / "artifacts"
            / "poiseuille_3d_resolution"
            / "results.json"
        ),
    )
    args = parser.parse_args(argv)
    payload = build_artifact()
    write_artifact(args.out, payload)
    print(
        json.dumps(
            {"metrics": payload["metrics"], "acceptance": payload["acceptance"]}, indent=2
        )
    )
    print(f"written: {args.out}")
    return 0 if payload["all_accepted"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())