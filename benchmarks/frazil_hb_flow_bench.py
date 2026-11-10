#!/usr/bin/env python3
"""Frazil-ice Herschel-Bulkley suspended-flow verification (campaign P519-N04).

Builds the *first* H-B non-Newtonian rheology verification surface for a
frazil-ice suspension in TensorLBM.  The physical chain is

    phi -> (gel closure) tau_y -> (W9-1215 root) K, n -> rho -> lattice case

(:mod:`tensorlbm.frazil_hb_rheology`); the numerical membrane is the existing
variable-viscosity BGK solver of :mod:`tensorlbm.herschel_bulkley_lbm`
(re-exported through :func:`tensorlbm.frazil_hb_lbm.run_frazil_hb_channel`,
which uses a shear-rate operator consistent with the node-based bounce-back
wall — see that module) with Papanastasiou
regularisation, Guo body force and half-way bounce-back walls.

Cases
-----
* ``poiseuille_phi*``: body-force driven plane channel, symmetric plug at the
  mid-plane, ``r_y = tau_y/tau_w = 0.40`` at ``phi = 0.30 / 0.40 / 0.50``
  (W9-1215 ``n(phi)`` = 0.81 / 0.70 / 0.60) at ``N = 256`` cells across the gap,
  plus a grid ladder ``N = 64 / 128 / 256`` at ``phi = 0.40`` for the GCI study.
* ``couette_phi040_*``: adverse-gradient Couette-Poiseuille (moving upper plate,
  stress ``sigma = beta y`` vanishing at the resting lower plate) so an interior
  yield surface exists at ``y_c = h tau_y/tau_w``; the *shear-rate distribution*
  is the gated quantity.
* ``bingham_*``: the Bingham branch ``n = 1`` with ``tau_y > 0`` (solver-branch
  verification; the frazil closure has ``n = 1`` only at ``phi = 0``, where
  ``tau_y = 0``, so this is *not* a frazil material point and is declared so).
* ``water_newtonian_*``: ``phi = 0`` clears yield stress and flow index
  (``tau_y = 0``, ``n = 1``): the solver must degenerate to the Newtonian
  Poiseuille parabola.
* ``poiseuille_phi040_n64_rest``: warm-start invariance control.
* ``couette_pure_plug``: negative control — pure Couette with ``tau_w < tau_y``
  is a *global* plug (no interior yield surface).

Outputs
-------
``runs/frazil_hb/<case>.json``    per-case payload (profiles + metrics)
``artifacts/frazil_hb_flow.json`` merged artifact with the gate ledger
``figures/frazil_hb/*.png``       report figures

Usage
-----
::

    python3 benchmarks/frazil_hb_flow_bench.py --plan
    python3 benchmarks/frazil_hb_flow_bench.py --all          # run + assemble
    python3 benchmarks/frazil_hb_flow_bench.py --case <id>    # one case
    python3 benchmarks/frazil_hb_flow_bench.py --assemble
    python3 benchmarks/frazil_hb_flow_bench.py --all --quick  # smoke run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.frazil_hb_rheology import (  # noqa: E402
    PHI_GEL,
    dimensional_consistency,
    hb_couette_adverse_shear_rate,
    frazil_hb_state,
    hb_couette_adverse_shear_rate,
    hb_plane_shear_rate,
    lattice_mapping,
    physical_flow_case,
    scaling_consistency,
    verification_case,
)
from tensorlbm.herschel_bulkley import (  # noqa: E402
    hb_couette_adverse_profile,
    hb_couette_adverse_wall_velocity,
    hb_plane_velocity_profile,
    validate_rheology,
)
from tensorlbm.frazil_hb_lbm import (  # noqa: E402
    run_frazil_hb_channel,
    shear_rate_channel_consistent,
)
from tensorlbm.herschel_bulkley_lbm import measure_yield_surface  # noqa: E402

RUNS = ROOT / "runs" / "frazil_hb"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures" / "frazil_hb"

# --- numerical controls (documented in the report) --------------------------
U_LAT_TARGET = 0.015  # max lattice velocity -> Ma = 0.026, compressibility-safe
NU_CHAR_TARGET = 0.03  # kinematic viscosity at the characteristic shear rate
C_REGULARISATION = 30.0  # Papanastasiou stiffness ratio m*gamma_char
TOL = 1.0e-8  # relative change of the mean velocity per CHECK_EVERY steps
CHECK_EVERY = 1000
MAX_STEPS = {32: 12000, 64: 14000, 128: 14000, 256: 16000, 512: 16000}
EPS_PROBE = 0.05  # yield-surface probe threshold (fraction of max shear rate)
R_Y = 0.40  # plastic ratio of the verification points
PHI_SWEEP = (0.30, 0.40, 0.50)


# ---------------------------------------------------------------------------
# Case plan
# ---------------------------------------------------------------------------


def plan(quick: bool = False) -> list[dict]:
    """Case list of the verification surface."""
    phis = (0.40,) if quick else PHI_SWEEP
    ladder = (32, 64) if quick else (32, 64, 128, 256, 512)  # 5 grids: GCI + derivative gates
    top = max(ladder)
    cases: list[dict] = []
    for phi in phis:
        n = float(frazil_hb_state(phi).n)
        for level in ladder:
            if phi != 0.40 and level != top:
                continue
            cases.append(
                {
                    "id": f"poiseuille_phi{int(round(phi * 100)):03d}_n{level}",
                    "kind": "poiseuille",
                    "phi": phi,
                    "n": n,
                    "r_y": R_Y,
                    "gap_cells": level,
                    "ladder": sorted({lv for lv in ladder if lv <= level}),
                }
            )
    cases.append(
        {
            # pure (plane) Couette: uniform stress, whole gap yielded, uniform
            # shear rate — the shear-rate distribution gate (G2)
            "id": f"couette_pure_phi040_n{top}",
            "kind": "couette_pure",
            "phi": 0.40,
            "n": float(frazil_hb_state(0.40).n),
            "r_y": R_Y,
            "gap_cells": top,
            "ladder": sorted(ladder),
        }
    )
    cases.append(
        {
            # adverse-gradient Couette-Poiseuille: interior yield surface at
            # y_c = H tau_y/tau_w (reported case; its plug has a long viscous
            # transient, see docs)
            "id": f"couette_phi040_n{top}",
            "kind": "couette",
            "phi": 0.40,
            "n": float(frazil_hb_state(0.40).n),
            "r_y": R_Y,
            "gap_cells": top,
            "ladder": sorted(ladder),
        }
    )
    cases.append(
        {
            "id": f"bingham_n{top}",
            "kind": "bingham",
            "phi": None,
            "n": 1.0,
            "r_y": R_Y,
            "gap_cells": top,
            "ladder": sorted(ladder),
        }
    )
    cases.append(
        {
            "id": f"water_newtonian_n{64 if quick else 128}",
            "kind": "water",
            "phi": 0.0,
            "n": 1.0,
            "r_y": 0.0,
            "gap_cells": 64 if quick else 128,
            "ladder": [32, 64] if quick else [32, 64, 128],  # transient-limited, not gated on GCI
        }
    )
    cases.append(
        {
            "id": "poiseuille_phi040_n64_rest",
            "kind": "poiseuille",
            "phi": 0.40,
            "n": float(frazil_hb_state(0.40).n),
            "r_y": R_Y,
            "gap_cells": 64,
            "ladder": [32, 64],
            "from_rest": True,
        }
    )
    cases.append({"id": "couette_pure_plug", "kind": "pure_couette", "gap_cells": 32})
    return cases


def find_case(case_id: str, quick: bool = False) -> dict:
    for case in plan(quick):
        if case["id"] == case_id:
            return case
    raise KeyError(f"unknown case id {case_id!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_VOLATILE_KEYS = ("wall_seconds", "generated_utc", "elapsed_s")


def _canonical(payload):
    """Payload with the volatile fields removed (wall-clock timing only).

    The reproducibility gate must hash the *physics* payload, not the wall time
    of the run that produced it.
    """
    if isinstance(payload, dict):
        return {
            k: _canonical(v)
            for k, v in payload.items()
            if k not in _VOLATILE_KEYS and k != "sha256"
        }
    if isinstance(payload, list):
        return [_canonical(v) for v in payload]
    return payload


def _hash(payload: dict) -> str:
    blob = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _rel_l2(num: torch.Tensor, ana: torch.Tensor) -> float:
    denom = float(torch.linalg.norm(ana))
    if denom == 0.0:
        return float("nan")
    return float(torch.linalg.norm(num - ana) / denom)


def _interp_profile(
    u: torch.Tensor, h_from: int, h_to: int, u_wall: float
) -> torch.Tensor:
    """Refine a channel profile ``N_from -> N_to`` (linear, walls pinned).

    Coordinates are the wall distance ``s = row index`` (the no-slip planes of
    the node-based bounce-back sit at ``s = 0`` and ``s = N+1``).  At a fixed
    wall stress the H-B profile is self-similar in ``s/H`` with
    ``u_max = C0 H tau_w^(1/n)/K^(1/n)``, so the velocity is rescaled by
    ``H_to/H_from`` as well as interpolated.
    """
    scale = (h_to + 1.0) / (h_from + 1.0)
    s_old = torch.arange(h_from + 2, dtype=torch.float64)
    s_new = torch.arange(h_to + 2, dtype=torch.float64)
    sq = (s_new / scale).clamp(min=float(s_old[0]), max=float(s_old[-1]))
    idx = torch.searchsorted(s_old, sq, right=True).clamp(1, len(s_old) - 1)
    s0, s1 = s_old[idx - 1], s_old[idx]
    u0, u1 = u[idx - 1], u[idx]
    w = ((sq - s0) / (s1 - s0)).clamp(0.0, 1.0)
    out = (u0 + w * (u1 - u0)) * scale
    out[0] = 0.0
    out[-1] = u_wall
    return out


def _fluid_mask(gap_cells: int) -> torch.Tensor:
    fluid = torch.ones((gap_cells + 2, 1), dtype=torch.bool)
    fluid[0, :] = False
    fluid[-1, :] = False
    return fluid


def _discrete_shear_rate(
    ux: torch.Tensor, gap_cells: int, u_wall: float = 0.0
) -> torch.Tensor:
    """The solver's own wall-consistent shear rate (same operator as the collision)."""
    gamma = shear_rate_channel_consistent(
        ux.to(torch.float64), _fluid_mask(gap_cells), 0.0, u_wall
    )
    return gamma[:, 0][1:-1]


def _level_record(level: int, res: dict, seconds: float) -> dict:
    """Ledger entry for one grid level.

    ``drift_last_window`` is the relative change of the mean velocity over the
    last ``CHECK_EVERY`` window; it is the steady-state criterion actually gated
    (the solver's own ``converged`` flag uses an extremely tight tolerance and
    can stay ``False`` while the solution no longer moves).
    """
    hist = res["history"]
    drift = float("nan")
    if len(hist) >= 2 and hist[-2][1] != 0.0:
        drift = abs(hist[-1][1] - hist[-2][1]) / abs(hist[-2][1])
    return {
        "gap_cells": level,
        "steps": int(res["steps"]),
        "converged": bool(res["converged"]),
        "drift_last_window": drift,
        "wall_seconds": round(seconds, 2),
        "mean_u_tail": [[int(s), float(v)] for s, v in res["history"][-3:]],
    }


def _case_payload(case: dict, base: dict, levels: list, metrics: dict) -> dict:
    params = {k: v for k, v in base.items() if k != "rheo"}
    params["rheo"] = base["rheo"].as_dict() if hasattr(base.get("rheo"), "as_dict") else None
    payload = {
        "case": case,
        "parameters": params,
        "levels": levels,
        "metrics": metrics,
        "converged_final": bool(levels[-1]["converged"]) if levels else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "dtype": "float64",
        "platform": platform.platform(),
    }
    return payload


# ---------------------------------------------------------------------------
# Channel driver (shared by Poiseuille / Bingham / Couette / water)
# ---------------------------------------------------------------------------


def _drive_channel(
    rheo,
    tau_w_lat: float,
    m: float,
    gap_cells: int,
    ladder: list[int],
    *,
    couette: bool = False,
    from_rest: bool = False,
    label: str = "",
) -> tuple[dict, list]:
    """Converge through the grid ladder at a *fixed* wall stress.

    The stress level ``tau_w_lat`` (hence ``r_y = tau_y/tau_w`` and the
    non-dimensional profile) is held fixed while the gap is refined, so every
    level solves the same problem: the body force and the Couette plate speed
    follow as ``a = 2 tau_w/N`` and ``U(N) = U(N_max) N/N_max``.  Coarse
    solutions warm-start the finer level after rescaling ``u`` by ``N_to/N_from``
    (the H-B channel is self-similar in ``y/H`` at fixed wall stress).
    """
    levels: list[dict] = []
    seq = sorted({lv for lv in ladder if lv <= gap_cells})
    if not seq or seq[-1] != gap_cells:
        seq.append(gap_cells)
    for level in seq:
        H_disc = float(level) + 1.0
        a_lat = 2.0 * tau_w_lat / level  # momentum balance over the N fluid rows
        if couette:
            beta_eff = tau_w_lat / H_disc  # analytic adverse gradient (gap H_disc)
            a_lat = -beta_eff * H_disc / level  # discrete force: tau_w/level
            u_wall = float(
                hb_couette_adverse_wall_velocity(beta_eff, H_disc, rheo)["U"]
            )
        else:
            u_wall = 0.0
        if from_rest:
            u_init = None
        else:
            # Warm start from the analytic solution of *this* level: it is closer
            # to the discrete steady state than the interpolated coarse-grid
            # solution (whose discretisation error would have to decay first).
            # The from-rest control shows the steady state does not depend on it.
            s_coord = torch.arange(level + 2, dtype=torch.float64)
            if couette:
                u_init = hb_couette_adverse_profile(
                    s_coord, tau_w_lat / H_disc, H_disc, rheo
                )
            else:
                u_init = hb_plane_velocity_profile(
                    s_coord - H_disc / 2.0, H_disc, tau_w_lat, rheo
                )
        t0 = time.time()
        res = run_frazil_hb_channel(
            level,
            rheo,
            a_lat,
            u_wall=u_wall,
            nx=1,
            n_steps=MAX_STEPS.get(level, 40000),
            m=m,
            check_every=CHECK_EVERY,
            tol=TOL,
            u_init=u_init,
            dtype=torch.float64,
        )
        levels.append(_level_record(level, res, time.time() - t0))
        print(
            f"  [{label}] N={level} a={a_lat:.3e} U={u_wall:.4e} steps={res['steps']} "
            f"conv={res['converged']} {levels[-1]['wall_seconds']}s",
            flush=True,
        )
    return res, levels


# ---------------------------------------------------------------------------
# Case runners
# ---------------------------------------------------------------------------


def run_poiseuille(case: dict) -> dict:
    gap = case["gap_cells"]
    base = verification_case(case["n"], case["r_y"], gap_cells=gap)
    rheo, m = base["rheo"], base["m"]
    res, levels = _drive_channel(
        rheo, base["tau_w_lat"], m, gap, case["ladder"],
        from_rest=case.get("from_rest", False), label=case["id"],
    )
    H = float(gap) + 1.0  # discrete gap of the node-based channel kernel
    y = res["y"].to(torch.float64)
    u_num = res["u_profile"].to(torch.float64)
    s_coord = torch.arange(gap + 2, dtype=torch.float64)  # wall distance
    s_fluid, u_n = s_coord[1:-1], u_num[1:-1]
    tau_w = base["tau_w_lat"]
    ana = hb_plane_velocity_profile(s_fluid - H / 2.0, H, tau_w, rheo)
    gamma_ana = hb_plane_shear_rate(s_fluid - H / 2.0, H, tau_w, rheo)
    gamma_num = _discrete_shear_rate(res["ux"], gap)
    g_max = float(gamma_ana.max())
    yielded = gamma_ana > 0.05 * g_max
    ys_num = measure_yield_surface(u_num, s_coord, rheo, m=m, eps=EPS_PROBE)
    ana_full = hb_plane_velocity_profile(s_coord - H / 2.0, H, tau_w, rheo)
    ys_ana = measure_yield_surface(ana_full, s_coord, rheo, m=m, eps=EPS_PROBE)
    # the probe scans from the lower wall, so the *first* crossing of the
    # symmetric Poiseuille flow is the upper yield surface s = H/2 + (H/2) r_y
    y_c_plug = 0.5 * H * base["r_y"]
    y_c = 0.5 * H * (1.0 + base["r_y"])
    met = {
        "u_max_num": float(u_n.max()),
        "u_max_ana": float(ana.max()),
        "u_max_rel_error": float(u_n.max() / ana.max()) - 1.0,
        "l2_profile": _rel_l2(u_n, ana),
        "max_abs_profile": float((u_n - ana).abs().max() / ana.abs().max()),
        "l2_shear_rate_yielded": _rel_l2(gamma_num[yielded], gamma_ana[yielded]),
        "max_abs_shear_rate_yielded": float(
            (gamma_num[yielded] - gamma_ana[yielded]).abs().max() / g_max
        ),
        "y_c_analytic": float(y_c),
        "y_c_plug_halfwidth": float(y_c_plug),
        "y_yield_probe_num": None if ys_num["y_yield"] is None else float(ys_num["y_yield"]),
        "y_yield_probe_ana": None if ys_ana["y_yield"] is None else float(ys_ana["y_yield"]),
        "y_yield_const_num": (
            None if ys_num["y_yield_const"] is None else float(ys_num["y_yield_const"])
        ),
        "y_yield_const_rel_error": (
            None
            if ys_num["y_yield_const"] is None
            else abs(float(ys_num["y_yield_const"]) - y_c) / y_c
        ),
        "y_yield_rel_error_vs_analytic": abs(float(ys_num["y_yield"]) - y_c) / y_c,
        "y_yield_rel_error_like_for_like": abs(
            float(ys_num["y_yield"]) - float(ys_ana["y_yield"])
        )
        / y_c,
        "profile_s": [round(float(v), 8) for v in s_fluid],
        "profile_u_num": [round(float(v), 10) for v in u_n],
        "profile_u_ana": [round(float(v), 10) for v in ana],
        "profile_gamma_num": [round(float(v), 10) for v in gamma_num],
        "profile_gamma_ana": [round(float(v), 10) for v in gamma_ana],
        "H_disc": H,
        "projection_cells": int(H * 0.4 * 0.5),
    }
    return _case_payload(case, base, levels, met)


def run_couette(case: dict) -> dict:
    gap = case["gap_cells"]
    base = verification_case(case["n"], case["r_y"], gap_cells=gap)
    rheo, m = base["rheo"], base["m"]
    H = float(gap) + 1.0
    beta_eff = base["tau_w_lat"] / H
    meta = hb_couette_adverse_wall_velocity(beta_eff, H, rheo)
    u_wall = float(meta["U"])
    res, levels = _drive_channel(
        rheo, base["tau_w_lat"], m, gap, case["ladder"], couette=True, label=case["id"],
    )
    y = res["y"].to(torch.float64)
    u_num = res["u_profile"].to(torch.float64)
    s_coord = torch.arange(gap + 2, dtype=torch.float64)
    s_fluid, u_n = s_coord[1:-1], u_num[1:-1]
    ana = hb_couette_adverse_profile(s_fluid, beta_eff, H, rheo)
    gamma_ana = hb_couette_adverse_shear_rate(s_fluid, beta_eff, rheo)
    gamma_num = _discrete_shear_rate(res["ux"], gap, u_wall=u_wall)
    g_max = float(gamma_ana.max())
    yielded = gamma_ana > 0.05 * g_max
    ys_num = measure_yield_surface(u_num, s_coord, rheo, m=m, eps=EPS_PROBE)
    ana_full = hb_couette_adverse_profile(s_coord, beta_eff, H, rheo)
    ys_ana = measure_yield_surface(ana_full, s_coord, rheo, m=m, eps=EPS_PROBE)
    y_c = float(meta["y_yield"])
    met = {
        "u_wall_num": float(u_num[-1]),
        "u_wall_ana": u_wall,
        "u_max_num": float(u_n.max()),
        "l2_profile": _rel_l2(u_n, ana),
        "l2_shear_rate_yielded": _rel_l2(gamma_num[yielded], gamma_ana[yielded]),
        "max_abs_shear_rate_yielded": float(
            (gamma_num[yielded] - gamma_ana[yielded]).abs().max() / g_max
        ),
        "y_c_analytic": y_c,
        "y_yield_probe_num": None if ys_num["y_yield"] is None else float(ys_num["y_yield"]),
        "y_yield_probe_ana": None if ys_ana["y_yield"] is None else float(ys_ana["y_yield"]),
        "y_yield_const_num": (
            None if ys_num["y_yield_const"] is None else float(ys_num["y_yield_const"])
        ),
        "y_yield_const_rel_error": (
            None
            if ys_num["y_yield_const"] is None
            else abs(float(ys_num["y_yield_const"]) - y_c) / y_c
        ),
        "y_yield_rel_error_vs_analytic": abs(float(ys_num["y_yield"]) - y_c) / y_c,
        "y_yield_rel_error_like_for_like": abs(
            float(ys_num["y_yield"]) - float(ys_ana["y_yield"])
        )
        / y_c,
        "beta_eff": beta_eff,
        "beta_discrete": base["tau_w_lat"] / gap,
        "u_wall": u_wall,
        "H_disc": H,
        "profile_s": [round(float(v), 8) for v in s_fluid],
        "profile_u_num": [round(float(v), 10) for v in u_n],
        "profile_u_ana": [round(float(v), 10) for v in ana],
        "profile_gamma_num": [round(float(v), 10) for v in gamma_num],
        "profile_gamma_ana": [round(float(v), 10) for v in gamma_ana],
    }
    return _case_payload(case, base, levels, met)


def run_couette_pure(case: dict) -> dict:
    """Pure plane Couette: uniform stress, no plug, uniform shear rate.

    With ``tau_w > tau_y`` the H-B material yields in the whole gap and the
    solution is ``gamma_dot = ((tau_w-tau_y)/K)**(1/n)`` everywhere, ``u = gamma_dot*s``
    with the upper plate at ``gamma_dot*H``.  There is no interior yield surface
    (the degeneracy documented in :mod:`tensorlbm.herschel_bulkley`), which makes
    this the clean gate on the *shear-rate distribution*.
    """
    gap = case["gap_cells"]
    H = float(gap) + 1.0
    # a pure Couette has *uniform* stress, so the plate speed is gamma_dot*H and
    # would grow linearly with the gap if the stress were held fixed.  The
    # similarity point (n, r_y) is therefore realised by fixing the shear rate
    # from the finest grid (u_wall = U_LAT_TARGET there) and rebuilding the stress:
    gamma_wall = U_LAT_TARGET / H
    K_lat = NU_CHAR_TARGET * gamma_wall ** (1.0 - case["n"])
    tau_w_lat = K_lat * gamma_wall ** case["n"] / (1.0 - case["r_y"])
    rheo = validate_rheology(case["r_y"] * tau_w_lat, K_lat, case["n"])
    c_reg = C_REGULARISATION / gamma_wall
    m = c_reg
    base = {
        "rheo": rheo,
        "tau_w_lat": tau_w_lat,
        "K_lat": K_lat,
        "m": float(m),
        "n": float(case["n"]),
        "r_y": float(case["r_y"]),
        "gamma_char": float(gamma_wall),
        "nu_char": NU_CHAR_TARGET,
        "H_disc": H,
        "gap_cells": gap,
    }
    u_wall_gap = gamma_wall * H
    levels = []
    seq = sorted({lv for lv in case["ladder"] if lv <= gap})
    if seq[-1] != gap:
        seq.append(gap)
    for level in seq:
        H_l = float(level) + 1.0
        u_wall_l = gamma_wall * H_l  # same shear rate, plate speed scales with the gap
        s_coord = torch.arange(level + 2, dtype=torch.float64)
        u_init = gamma_wall * s_coord
        t0 = time.time()
        res = run_frazil_hb_channel(
            level, rheo, 0.0, u_wall=u_wall_l, nx=1,
            n_steps=MAX_STEPS.get(level, 40000), m=m, check_every=CHECK_EVERY,
            tol=TOL, u_init=u_init, dtype=torch.float64,
        )
        levels.append(_level_record(level, res, time.time() - t0))
        print(
            f"  [{case['id']}] N={level} U={u_wall_l:.4e} steps={res['steps']} "
            f"conv={res['converged']} {levels[-1]['wall_seconds']}s",
            flush=True,
        )
    s_coord = torch.arange(gap + 2, dtype=torch.float64)
    u_num = res["u_profile"].to(torch.float64)
    s_fluid, u_n = s_coord[1:-1], u_num[1:-1]
    ana = gamma_wall * s_fluid
    gamma_ana = torch.full_like(s_fluid, gamma_wall)
    gamma_num = _discrete_shear_rate(res["ux"], gap, u_wall=u_wall_gap)
    met = {
        "gamma_wall_analytic": float(gamma_wall),
        "u_wall_analytic": float(u_wall_gap),
        "u_wall_num": float(u_num[-1]),
        "l2_profile": _rel_l2(u_n, ana),
        "l2_shear_rate_yielded": _rel_l2(gamma_num, gamma_ana),
        "max_abs_shear_rate_yielded": float(
            (gamma_num - gamma_ana).abs().max() / gamma_wall
        ),
        "yield_surface": None,  # uniform stress -> whole gap yielded
        "tau_w_lat": tau_w_lat,
        "m": float(m),
        "H_disc": H,
        "profile_s": [round(float(v), 8) for v in s_fluid],
        "profile_u_num": [round(float(v), 10) for v in u_n],
        "profile_u_ana": [round(float(v), 10) for v in ana],
        "profile_gamma_num": [round(float(v), 10) for v in gamma_num],
        "profile_gamma_ana": [round(float(v), 10) for v in gamma_ana],
    }
    return _case_payload(case, base, levels, met)


def run_water(case: dict) -> dict:
    """Newtonian degeneration: phi = 0 -> tau_y = 0, n = 1."""
    gap = case["gap_cells"]
    base = verification_case(1.0, 0.0, gap_cells=gap)
    rheo, m = base["rheo"], base["m"]
    res, levels = _drive_channel(
        rheo, base["tau_w_lat"], m, gap, case["ladder"], label=case["id"]
    )
    H = float(gap) + 1.0
    y = res["y"].to(torch.float64)
    s_coord = torch.arange(gap + 2, dtype=torch.float64)
    u_num = res["u_profile"].to(torch.float64)[1:-1]
    ana = hb_plane_velocity_profile(s_coord[1:-1] - H / 2.0, H, base["tau_w_lat"], rheo)
    met = {
        "l2_profile": _rel_l2(u_num, ana),
        "u_max_num": float(u_num.max()),
        "u_max_ana": float(ana.max()),
        "max_abs_profile": float((u_num - ana).abs().max() / ana.abs().max()),
        "tau_y_lat": rheo.tau_y,
        "plug_absent": rheo.tau_y == 0.0,
        "nu_lat": rheo.K,
        "a_lat": 2.0 * base["tau_w_lat"] / gap,
        "profile_s": [round(float(v), 8) for v in s_coord[1:-1]],
        "profile_u_num": [round(float(v), 10) for v in u_num],
        "profile_u_ana": [round(float(v), 10) for v in ana],
        "H_disc": H,
    }
    return _case_payload(case, base, levels, met)


def run_bingham(case: dict) -> dict:
    return run_poiseuille(case)


def run_pure_couette(case: dict) -> dict:
    """Negative control: pure Couette with tau_w < tau_y is a global plug."""
    gap = case["gap_cells"]
    rheo = validate_rheology(1.0e-3, 1.0, 0.7)
    u_wall = 0.01
    res = run_frazil_hb_channel(
        gap, rheo, 0.0, u_wall=u_wall, nx=1, n_steps=20000, m=1.0e4,
        check_every=CHECK_EVERY, tol=TOL, dtype=torch.float64,
    )
    u = res["u_profile"].to(torch.float64)[1:-1]
    spread = float((u - u_wall).abs().max() / u_wall)
    met = {
        "tau_y": rheo.tau_y,
        "u_wall": u_wall,
        "plug_uniformity_error": spread,
        "is_global_plug": bool(spread < 0.05),
        "interior_yield_surface": False,
    }
    base = {"rheo": rheo, "tau_w_lat": 0.0, "r_y": None, "m": 1.0e4}
    return _case_payload(case, base, [_level_record(gap, res, 0.0)], met)


RUNNERS = {
    "poiseuille": run_poiseuille,
    "couette_pure": run_couette_pure,
    "bingham": run_bingham,
    "couette": run_couette,
    "water": run_water,
    "pure_couette": run_pure_couette,
}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_case(case: dict) -> dict:
    payload = RUNNERS[case["kind"]](case)
    payload["sha256"] = _hash(payload)
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{case['id']}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"  wrote {path.relative_to(ROOT)} sha256={payload['sha256'][:16]}", flush=True)
    return payload


def run_all(quick: bool = False) -> int:
    cases = plan(quick)
    for case in cases:
        print(f"[case] {case['id']} ({case['kind']}, N={case['gap_cells']})", flush=True)
        run_case(case)
    return 0

# ---------------------------------------------------------------------------
# Gates and artifact assembly
# ---------------------------------------------------------------------------


def _gate(gate_id: str, description: str, value, threshold: float) -> dict:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return {
            "id": gate_id,
            "description": description,
            "value": None,
            "threshold": float(threshold),
            "passed": False,
        }
    return {
        "id": gate_id,
        "description": description,
        "value": float(value),
        "threshold": float(threshold),
        "passed": bool(abs(float(value)) <= threshold),
    }


def _gci(errors: dict[int, float], ratio: float = 2.0) -> dict:
    """Grid convergence index from error magnitudes on successive grids."""
    levels = sorted(errors)
    if len(levels) < 3:
        return {"gci_fine": float("inf"), "levels": levels, "note": "fewer than 3 grids"}
    e1, e2, e3 = (abs(errors[lv]) for lv in levels[-3:])
    if e1 <= 0.0 or e2 <= 0.0 or e3 <= 0.0:
        return {"gci_fine": float("inf"), "levels": levels, "note": "zero error level"}
    p = math.log(e1 / e2) / math.log(ratio) if e2 > 0 else float("nan")
    safe_p = p if (0.0 < p < 6.0) else 1.0
    gci = 1.25 * e2 / (ratio**safe_p - 1.0)
    return {
        "levels": levels[-3:],
        "errors": [e1, e2, e3],
        "observed_order": float(p),
        "observed_order_used": float(safe_p),
        "gci_fine": float(gci),
        "extrapolated_zero": float(errors[levels[-1]] / max(ratio**safe_p - 1.0, 1e-12)),
    }


def _physical_table() -> list[dict]:
    """phi -> tau_y, K, n, r_y, implied pressure gradient and velocity."""
    rows = []
    for phi in (0.0, PHI_GEL, 0.30, 0.40, 0.50, 0.55):
        st = frazil_hb_state(phi)
        case = physical_flow_case(phi, r_y=(R_Y if st.tau_y > 0.0 else None))
        rows.append(
            {
                "phi": phi,
                "rho": st.rho,
                "mu_r": st.mu_r,
                "n": st.n,
                "tau_y_Pa": st.tau_y,
                "K_Pa_sn": st.K,
                "tau_y_kin": st.tau_y_kin,
                "K_kin": st.K_kin,
                "r_y_at_H50mm_rY040": case["r_y"],
                "dPdx_Pa_per_m": case["dPdx_Pa_per_m"],
                "u_max_m_per_s": case["u_max_si"],
                "y_c_m": case["y_c_si"],
                "H_m": case["H"],
            }
        )
    return rows


def _repeat_check(case_id: str) -> dict:
    """Re-run a case in a fresh process and compare the payload sha256."""
    path = RUNS / f"{case_id}.json"
    if not path.exists():
        return {"case": case_id, "mismatch": float("inf"), "note": "case payload missing"}
    stored = json.loads(path.read_text())["sha256"]
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--case", case_id, "--hash-only"],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if proc.returncode != 0:
        return {
            "case": case_id,
            "mismatch": float("inf"),
            "note": proc.stderr.strip()[-400:],
        }
    fresh = proc.stdout.strip().splitlines()[-1].strip()
    return {
        "case": case_id,
        "sha256_stored": stored,
        "sha256_fresh_process": fresh,
        "mismatch": 0.0 if stored == fresh else 1.0,
        "identical": bool(stored == fresh),
    }


def assemble(quick: bool = False) -> dict:
    cases = plan(quick)
    results: dict[str, dict] = {}
    for case in cases:
        path = RUNS / f"{case['id']}.json"
        if not path.exists():
            print(f"  [skip] {case['id']} (no payload)", flush=True)
            continue
        results[case["id"]] = json.loads(path.read_text())

    top = max(c["gap_cells"] for c in cases)
    ladder = sorted(
        c["gap_cells"]
        for c in cases
        if c["kind"] == "poiseuille" and c.get("phi") == 0.40 and not c.get("from_rest")
    )

    # --- grid convergence index on the phi = 0.40 ladder -------------------
    l2_by_level = {
        lv: results[f"poiseuille_phi040_n{lv}"]["metrics"]["l2_profile"]
        for lv in ladder
        if f"poiseuille_phi040_n{lv}" in results
    }
    yc_by_level = {
        lv: results[f"poiseuille_phi040_n{lv}"]["metrics"]["y_yield_rel_error_vs_analytic"]
        for lv in ladder
        if f"poiseuille_phi040_n{lv}" in results
    }
    gci = {
        "profile_l2": _gci(l2_by_level),
        "yield_surface": _gci(yc_by_level),
        "grids": ladder,
        "l2_profile_by_grid": l2_by_level,
    }
    gci["gci_fine"] = max(
        gci["profile_l2"]["gci_fine"], gci["yield_surface"]["gci_fine"]
    )
    results["_gci"] = gci

    # --- shear-rate diagnostics: full yielded layer vs its smooth interior ---
    for cid, payload in results.items():
        if cid.startswith("_") or not isinstance(payload, dict):
            continue
        met = payload.get("metrics", {})
        if "profile_gamma_num" not in met:
            continue
        gn = torch.tensor(met["profile_gamma_num"], dtype=torch.float64)
        ga = torch.tensor(met["profile_gamma_ana"], dtype=torch.float64)
        s_coord = torch.tensor(met["profile_s"], dtype=torch.float64)
        if gn.numel() != ga.numel():
            continue
        keep = torch.ones_like(gn, dtype=torch.bool)
        keep[:4] = False  # wall boundary layer
        keep[-4:] = False
        y_c = met.get("y_c_analytic")
        if y_c is not None:
            # exclude three cells on both sides of the yield surface (the shear
            # rate kinks there and a second-order stencil cannot represent it)
            keep &= (s_coord - y_c).abs() > 3.0
        if int(keep.sum()) > 4:
            denom = float(torch.linalg.norm(ga[keep]))
            if denom > 0.0:
                met["l2_shear_rate_interior"] = float(
                    torch.linalg.norm(gn[keep] - ga[keep]) / denom
                )

    # --- dimensional + similarity ledger -----------------------------------
    dims = {}
    for phi in (0.0, 0.30, 0.40, 0.50):
        st = frazil_hb_state(phi)
        dims[f"phi_{int(round(phi * 100)):03d}"] = dimensional_consistency(st)
    st = frazil_hb_state(0.40)
    phys = physical_flow_case(0.40, r_y=R_Y)
    mapping = lattice_mapping(phys, gap_cells=64, u_lat_target=U_LAT_TARGET)
    ver = verification_case(st.n, R_Y, gap_cells=64)
    scaling = scaling_consistency(phys, mapping, ver)
    results["_dimensional"] = {
        "cases": dims,
        "worst_residual": max(v["worst_residual"] for v in dims.values()),
        "all_passed": all(v["passed"] for v in dims.values()),
    }
    results["_scaling"] = scaling
    results["_physical_table"] = _physical_table()
    results["_scaling_mapping"] = {
        "dx": mapping["dx"],
        "dt": mapping["dt"],
        "tau_w_lat": mapping["tau_w_lat"],
        "K_lat": mapping["K_lat"],
        "u_lat": mapping["u_lat"],
        "tau_minus_half": mapping["tau_minus_half"],
        "verification_case": {k: v for k, v in ver.items() if k != "rheo"}
        | {"rheo": ver["rheo"].as_dict()},
    }

    # --- reproducibility ---------------------------------------------------
    repro_case = "poiseuille_phi040_n64" if "poiseuille_phi040_n64" in results else None
    results["_repeat"] = _repeat_check(repro_case) if repro_case else {"mismatch": float("inf")}

    # --- warm-start invariance control -------------------------------------
    rest_id = "poiseuille_phi040_n64_rest"
    warm_id = "poiseuille_phi040_n64"
    if rest_id in results and warm_id in results:
        u_rest = torch.tensor(results[rest_id]["metrics"]["profile_u_num"])
        u_warm = torch.tensor(results[warm_id]["metrics"]["profile_u_num"])
        results["_warm_start_control"] = {
            "l2_profile_rest": results[rest_id]["metrics"]["l2_profile"],
            "l2_profile_warm": results[warm_id]["metrics"]["l2_profile"],
            "profile_difference_rel": float(
                torch.linalg.norm(u_rest - u_warm) / torch.linalg.norm(u_warm)
            ),
            "converged_rest": results[rest_id]["converged_final"],
        }

    # --- gate ledger -------------------------------------------------------
    gates = [
        _gate(
            "G1_poiseuille_profile_l2_phi040",
            f"plane Poiseuille, phi=0.40 (n=0.70, r_y=0.40), N={top}: relative L2 "
            "error of the velocity profile vs the exact H-B solution",
            results[f"poiseuille_phi040_n{top}"]["metrics"]["l2_profile"]
            if f"poiseuille_phi040_n{top}" in results
            else None,
            0.02,
        ),
        _gate(
            "G1b_poiseuille_profile_l2_phi_sweep",
            f"worst relative L2 velocity error of the phi = 0.30 / 0.50 sweep at N={top}",
            max(
                [
                    results[f"poiseuille_phi{int(round(phi*100)):03d}_n{top}"]["metrics"][
                        "l2_profile"
                    ]
                    for phi in (0.30, 0.50)
                    if f"poiseuille_phi{int(round(phi*100)):03d}_n{top}" in results
                ]
                or [float("inf")]
            ),
            0.02,
        ),
        _gate(
            "G2_couette_shear_rate",
            "pure Couette (uniform stress, whole gap yielded): relative L2 error of the "
            "shear-rate distribution (normalised by the analytic wall value)",
            results.get(f"couette_pure_phi040_n{top}", {}).get("metrics", {}).get(
                "l2_shear_rate_yielded"
            ),
            0.02,
        ),
        _gate(
            "G3a_yield_surface_poiseuille",
            "Poiseuille yield surface: the same shear-rate probe applied to the LBM and "
            "to the analytic profile, difference relative to analytic y_c = H/2 (1+r_y)",
            results.get(f"poiseuille_phi040_n{top}", {}).get("metrics", {}).get(
                "y_yield_rel_error_like_for_like"
            ),
            0.01,
        ),
        _gate(
            "G3b_yield_surface_couette_adverse",
            "adverse-gradient Couette-Poiseuille: interior yield-surface position, same "
            "probe on the LBM and the analytic profile (relative to y_c = H tau_y/tau_w)",
            results.get(f"couette_phi040_n{top}", {}).get("metrics", {}).get(
                "y_yield_rel_error_like_for_like"
            ),
            0.01,
        ),
        _gate(
            "G3c_yield_surface_bingham",
            "Bingham (n=1, tau_y>0) yield-surface position: same probe on the LBM and "
            "the analytic profile",
            results.get(f"bingham_n{top}", {}).get("metrics", {}).get(
                "y_yield_rel_error_like_for_like"
            ),
            0.01,
        ),
        _gate(
            "G4_newtonian_degeneration",
            "tau_y -> 0 and n -> 1 (phi = 0, clear water): L2 error of the profile vs "
            "the Newtonian Poiseuille parabola",
            next(
                (
                    results[k]["metrics"]["l2_profile"]
                    for k in results
                    if k.startswith("water_newtonian")
                ),
                None,
            ),
            0.02,
        ),
        _gate(
            "G5_grid_convergence_index",
            "GCI of the finest grid (the worst of the profile-error and yield-surface "
            "GCI, three grids, observed order)",
            gci["gci_fine"],
            0.01,
        ),
        _gate(
            "G6_dimensional_consistency",
            "worst residual of the phi -> SI -> kinematic unit-algebra / round-trip chain",
            results["_dimensional"]["worst_residual"],
            1e-9,
        ),
        _gate(
            "G7_reproducibility_sha256",
            "re-run the reproducibility case in a fresh process; 0 = identical sha256",
            results["_repeat"].get("mismatch", float("inf")),
            0.0,
        ),
        _gate(
            "G8_si_lattice_similarity",
            "worst residual of the SI <-> lattice mapping and the dimensionless "
            "profile similarity",
            scaling["worst_residual"],
            1e-9,
        ),
        _gate(
            "G9_bingham_profile",
            "Bingham (n=1) velocity-profile L2 error at the finest grid",
            results.get(f"bingham_n{top}", {}).get("metrics", {}).get("l2_profile"),
            0.02,
        ),
        _gate(
            "G10_steady_state",
            "worst relative drift of the mean velocity over the last 1000-step "
            "window of the gated cases (double precision round-off is ~1e-15)",
            max(
                [
                    results[cid]["levels"][-1].get("drift_last_window", float("inf"))
                    for cid in (
                        f"poiseuille_phi040_n{top}",
                        f"couette_pure_phi040_n{top}",
                        f"bingham_n{top}",
                    )
                    if cid in results
                ]
                or [float("inf")]
            ),
            1e-5,
        ),
    ]

    artifact = {
        "node": "P519-N04",
        "title": "Frazil-ice suspended flow: Herschel-Bulkley non-Newtonian "
        "rheology verification surface",
        "repository": "TensorLBM",
        "module": "tensorlbm.frazil_hb_rheology + herschel_bulkley_lbm",
        "benchmark": "benchmarks/frazil_hb_flow_bench.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "numerical_controls": {
            "u_lat_target": U_LAT_TARGET,
            "nu_char_target": NU_CHAR_TARGET,
            "c_regularisation": C_REGULARISATION,
            "tol": TOL,
            "check_every": CHECK_EVERY,
            "max_steps": MAX_STEPS,
            "probe_eps": EPS_PROBE,
            "r_y": R_Y,
            "phi_sweep": PHI_SWEEP,
        },
        "gates": gates,
        "all_passed": all(g["passed"] for g in gates),
        "gci": gci,
        "dimensional_consistency": results["_dimensional"],
        "si_lattice_similarity": scaling,
        "physical_chain": results["_physical_table"],
        "reproducibility": results["_repeat"],
        "warm_start_control": results.get("_warm_start_control"),
        "cases": {
            cid: {
                "case": results[cid]["case"],
                "converged_final": results[cid]["converged_final"],
                "sha256": results[cid]["sha256"],
                "levels": results[cid]["levels"],
                "metrics": {
                    k: v
                    for k, v in results[cid]["metrics"].items()
                    if not k.startswith("profile_")
                },
                "parameters": results[cid]["parameters"],
            }
            for cid in results
            if not cid.startswith("_")
        },
        "solver": {
            "lattice": "D2Q9 BGK, Guo body force, half-way bounce-back",
            "viscosity": "Papanastasiou-regularised Herschel-Bulkley, per-cell tau",
            "shear_rate": "wall-aware central gradient (solver operator)",
            "steady_state": f"mean-velocity tolerance {TOL}",
            "threads": 1,
            "torch": torch.__version__,
            "dtype": "float64",
        },
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "frazil_hb_flow.json"
    path.write_text(json.dumps(artifact, indent=1))
    print(f"\nartifact: {path.relative_to(ROOT)}  all_passed={artifact['all_passed']}")
    for g in gates:
        print(
            f"  {'PASS' if g['passed'] else 'FAIL'}  {g['id']:<34} "
            f"value={g['value']}  threshold={g['threshold']}"
        )
    _report_raw(results, artifact)
    make_figures(results, artifact)
    return artifact


def _report_raw(results: dict, artifact: dict) -> None:
    """Persist the raw (un-rounded) case payloads inside the artifact dir."""
    out = {cid: v for cid, v in results.items() if not cid.startswith("_")}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "frazil_hb_flow_raw.json").write_text(
        json.dumps(
            {
                "cases": out,
                "gci": results["_gci"],
                "scaling_mapping": results["_scaling_mapping"],
            },
            indent=1,
        )
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(results: dict, artifact: dict) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    top = max(c["gap_cells"] for c in plan() if c["kind"] == "poiseuille")

    # 1. velocity profiles of the phi sweep
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for phi, colour in zip(PH_SWEEP_FOR_PLOT, ("C0", "C1", "C2")):
        cid = f"poiseuille_phi{int(round(phi * 100)):03d}_n{top}"
        if cid not in results:
            continue
        met = results[cid]["metrics"]
        h = met["H_disc"]
        y = torch.tensor(met["profile_s"]) / h
        u_n = torch.tensor(met["profile_u_num"])
        u_a = torch.tensor(met["profile_u_ana"])
        umax = float(u_a.max())
        axes[0].plot(u_a / umax, y, "-", color=colour, lw=1.4,
                     label=f"analytic, phi={phi:.2f} (n={results[cid]['parameters']['rheo']['n']:.2f})")
        axes[0].plot(u_n / umax, y, "o", ms=2.6, mfc="none", color=colour, mew=0.8)
        axes[1].plot((u_n - u_a) / umax, y, "-", color=colour, lw=1.2,
                     label=f"phi={phi:.2f}")
    axes[0].set_xlabel("$u/u_{max}$")
    axes[0].set_ylabel("$y/H$")
    axes[0].set_title(f"H-B plane Poiseuille, $r_y=\\tau_y/\\tau_w={R_Y}$ (N={top})")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].grid(alpha=0.3)
    axes[1].axvline(0.0, color="k", lw=0.6)
    axes[1].set_xlabel("$(u_{LBM}-u_{analytic})/u_{max}$")
    axes[1].set_title("profile error (markers: none = converged)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = FIGURES / "velocity_profiles.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # 2. shear-rate distributions
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    cid = f"poiseuille_phi040_n{top}"
    if cid in results:
        met = results[cid]["metrics"]
        h = met["H_disc"]
        y = torch.tensor(met["profile_s"]) / h
        gn = torch.tensor(met["profile_gamma_num"])
        ga = torch.tensor(met["profile_gamma_ana"])
        gmax = float(ga.max())
        axes[0].plot(ga / gmax, y, "-", color="C0", lw=1.5, label="analytic")
        axes[0].plot(gn / gmax, y, "o", ms=2.6, mfc="none", color="C3", mew=0.8,
                     label="LBM (solver operator)")
        yc = met["y_c_analytic"] / h
        for sgn in (+1, -1):
            axes[0].axhline(0.5 + sgn * (0.5 - yc), color="k", ls=":", lw=0.8)
        axes[0].set_xlabel("$\\dot\\gamma/\\dot\\gamma_{wall}$")
        axes[0].set_ylabel("$y/H$")
        axes[0].set_title(f"Poiseuille shear rate, phi=0.40 (yield surface: dotted)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
    cid = f"couette_phi040_n{top}"
    if cid in results:
        met = results[cid]["metrics"]
        h = met["H_disc"]
        y = torch.tensor(met["profile_s"]) / h
        gn = torch.tensor(met["profile_gamma_num"])
        ga = torch.tensor(met["profile_gamma_ana"])
        gmax = float(ga.max())
        axes[1].plot(ga / gmax, y, "-", color="C0", lw=1.5, label="analytic")
        axes[1].plot(gn / gmax, y, "o", ms=2.6, mfc="none", color="C3", mew=0.8,
                     label="LBM")
        axes[1].axhline(met["y_c_analytic"] / h, color="k", ls=":", lw=0.8)
        axes[1].set_xlabel("$\\dot\\gamma/\\dot\\gamma_{wall}$")
        axes[1].set_ylabel("$y/h$")
        axes[1].set_title(
            "Couette-Poiseuille (adverse), shear rate; "
            f"L2={met['l2_shear_rate_yielded']*100:.2f}%"
        )
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = FIGURES / "shear_rate_distributions.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # 3. grid convergence
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    gci = artifact["gci"]
    lv = gci["grids"]
    l2 = [gci["l2_profile_by_grid"][str(k)] if str(k) in gci["l2_profile_by_grid"]
          else gci["l2_profile_by_grid"][k] for k in lv]
    axes[0].loglog(lv, l2, "o-", label="L2 velocity error")
    ysl = [
        results[f"poiseuille_phi040_n{k}"]["metrics"]["y_yield_rel_error_vs_analytic"]
        for k in lv
        if f"poiseuille_phi040_n{k}" in results
    ]
    if len(ysl) == len(lv):
        axes[0].loglog(lv, ysl, "s--", label="yield-surface rel. error")
    axes[0].axhline(0.02, color="C3", ls=":", lw=1.0, label="2% gate")
    axes[0].axhline(0.01, color="C4", ls=":", lw=1.0, label="1% gate")
    axes[0].set_xlabel("N (cells across the gap)")
    axes[0].set_ylabel("relative error")
    axes[0].set_title(
        "grid convergence"
        + (
            f", observed order p={gci['profile_l2']['observed_order']:.2f}"
            if "observed_order" in gci["profile_l2"]
            else ""
        )
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    axes[1].axis("off")
    def _fmt_gci(entry: dict, label: str) -> str:
        if not math.isfinite(entry.get("gci_fine", float("inf"))):
            return f"GCI({label}) = n/a"
        return f"GCI({label}) = {entry['gci_fine'] * 100:.3f}%"

    order_txt = (
        f"observed order (profile): {gci['profile_l2']['observed_order']:.2f}"
        if "observed_order" in gci["profile_l2"]
        else "(fewer than 3 grids: no order estimate)"
    )
    txt = "\n".join(
        [
            "grid convergence ledger",
            f"grids: {gci['grids']}",
            "L2 profile errors: "
            + ", ".join(f"{k}: {v * 100:.3f}%" for k, v in gci["l2_profile_by_grid"].items()),
            order_txt,
            _fmt_gci(gci["profile_l2"], "profile"),
            _fmt_gci(gci["yield_surface"], "yield surface"),
            f"gate: GCI < 1% -> {'PASS' if gci['gci_fine'] < 0.01 else 'FAIL'}",
            "",
            "SI <-> lattice mapping (phi=0.40)",
            *[
                f"{k}: {v:.3e}"
                for k, v in artifact["si_lattice_similarity"]["checks"].items()
            ],
            f"worst = {artifact['si_lattice_similarity']['worst_residual']:.3e}",
        ]
    )
    axes[1].text(0.0, 0.95, txt, va="top", fontsize=8, family="monospace")
    fig.tight_layout()
    p = FIGURES / "grid_convergence.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # 4. rheology chain
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    phis = [row["phi"] for row in artifact["physical_chain"]]
    axes[0].plot(phis, [row["tau_y_Pa"] for row in artifact["physical_chain"]], "o-")
    axes[0].axvline(0.20, color="C3", ls=":", label="gel threshold $\\phi_{gel}$=0.20")
    axes[0].set_xlabel("$\\phi$ (ice volume fraction)")
    axes[0].set_ylabel("$\\tau_y$ [Pa]")
    axes[0].set_title("yield stress closure")
    axes[0].legend(fontsize=8)
    axes[1].plot(phis, [row["n"] for row in artifact["physical_chain"]], "s-", color="C1")
    axes[1].set_xlabel("$\\phi$")
    axes[1].set_ylabel("n [-]")
    axes[1].set_title("W9-1215 flow index $n(\\phi)$")
    axes[2].semilogy(phis, [row["K_Pa_sn"] for row in artifact["physical_chain"]], "^-",
                     color="C2")
    axes[2].set_xlabel("$\\phi$")
    axes[2].set_ylabel("K [Pa s$^n$]")
    axes[2].set_title("W9-1215 consistency $K(\\rho\\nu_w\\mu_r)$")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIGURES / "rheology_chain.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)
    print(f"figures: {[str(q.relative_to(ROOT)) for q in paths]}")
    return paths


PH_SWEEP_FOR_PLOT = PHI_SWEEP


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="list the case plan")
    parser.add_argument("--case", help="run a single case id")
    parser.add_argument("--all", action="store_true", help="run every case and assemble")
    parser.add_argument("--assemble", action="store_true", help="assemble artifact + figures")
    parser.add_argument("--quick", action="store_true", help="coarse smoke configuration")
    parser.add_argument("--hash-only", action="store_true",
                        help="print the payload sha256 only (reproducibility probe)")
    args = parser.parse_args(argv)

    torch.set_default_dtype(torch.float64)
    # the channel grids are a few hundred cells wide: thread-pool overhead
    # dominates the 9-direction kernel above one thread (~5x slower with 128)
    torch.set_num_threads(1)

    if args.plan:
        for case in plan(args.quick):
            print(json.dumps(case))
        return 0
    if args.case:
        case = find_case(args.case, args.quick)
        if args.hash_only:
            payload = RUNNERS[case["kind"]](case)
            print(_hash(payload))
            return 0
        run_case(case)
        return 0
    if args.all:
        rc = run_all(args.quick)
        if rc != 0:
            return rc
    if args.all or args.assemble:
        assemble(args.quick)
    if not any((args.plan, args.case, args.all, args.assemble)):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
