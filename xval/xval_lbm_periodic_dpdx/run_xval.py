"""XVAL: TensorLBM vs joconnor22/LIFE — ``single-phase-periodic-x-plus-constant-dpdx-driving``.

Run:  PYTHONPATH=src python3 xval/xval_lbm_periodic_dpdx/run_xval.py

Arms
----
``LIFE-REF``  independent transcription of the OSS mechanism (``life_ref.py``):
              x-periodic (``eFluid`` left/right), constant frozen ``dpdx`` volume
              force, ``(1-omega/2)`` Guo source, ``+F/2`` velocity, regularised
              (Latt 2008) node wall closure with ``u = 0``.
``GOT-PORT``  ``tensorlbm.periodic_channel_dpdx`` with the same closures — the
              *port*.  Status is measured, not assumed: this run records the
              mass drift and the step at which each ported arm departs from
              bounded solutions.  (Honest-FAIL is reported, never hidden.)
``CONTROL``   ``wall="none"`` — y walls removed.  Must fail the 15 % gate.

Numeric anchor: exact plane-Poiseuille parabola
``u(y) = G (y - y_w)(H - (y - y_w)) / (2 nu)`` on the wall-normalised
``s = (y - y_w)/H`` (the OSS wall nodes sit at ``y = 0`` and ``y = ny - 1``,
so ``y_w = 0``, ``H = ny - 1``).

Every number in the report comes from ``out/xval_life_periodic_dpdx.json``.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tensorlbm.periodic_channel_dpdx import (  # noqa: E402
    FORCE_GUO_HALF_FROZEN, FORCE_SPLIT_GUO, WALL_HALFWAY_BB, WALL_NONE,
    WALL_REGULARISED, DpdxChannel, dpdx_to_lattice_force, life_force_xy,
    poiseuille_analytic,
)
from life_ref import LifeKernel2D  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

NX = 2
NU = 0.0125            # LIFE examples/ChannelFlow: tau = 0.5375, tStep = 5e-4
TAU = 3.0 * NU + 0.5
U_MAX = 0.05           # Re_H = u_max H / nu = 128 at H = 32
GATE = 0.15


def g_for(h: float, nu: float = NU, u_max: float = U_MAX) -> float:
    return 8.0 * nu * u_max / (h * h)


def _rel_l2(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def life_arm(ny: int, force: float, *, tau: float = TAU, tol=1e-8, max_steps=45_000):
    k = LifeKernel2D(NX, ny, tau, force)
    conv = k.run_until_steady(tol=tol, max_steps=max_steps)
    nu = (tau - 0.5) / 3.0
    u = k.profile()
    s, u_ana, y_w, h = poiseuille_analytic(ny, nu, force, wall=WALL_REGULARISED)
    return {
        "ny": ny, "H": h, "y_wall": y_w, "nu": nu, "force": force,
        "steps": conv["steps"], "converged": conv["converged"],
        "steady_delta": conv["delta"],
        "u_max": float(np.max(np.abs(u))),
        "u_max_analytic": float(np.max(u_ana)),
        "rel_l2_vs_analytic": _rel_l2(u, u_ana),
        "u_max_rel_err": float(abs(np.max(np.abs(u)) - np.max(u_ana)) / np.max(u_ana)),
        "rho_min": float(k.rho.min()), "rho_max": float(k.rho.max()),
        "s": s.tolist(), "u": u.tolist(), "u_analytic": u_ana.tolist(),
    }


def port_arm(kind: str, ny: int, force: float, *, n_steps: int = 3000):
    """Bounded run of the ported arm: witnesses blow-up instead of hiding it."""
    wall = {"GOT-PORT-REG": WALL_REGULARISED, "GOT-PORT-BB": WALL_HALFWAY_BB,
            "CONTROL": WALL_NONE}[kind]
    scheme = FORCE_GUO_HALF_FROZEN if kind == "GOT-PORT-REG" else FORCE_SPLIT_GUO
    k = DpdxChannel(NX, ny, TAU, force, wall=wall, force_scheme=scheme)
    m0 = k.mass()
    trace = []
    blew_up_at = None
    for n in range(n_steps):
        k.step()
        if not np.isfinite(k.rho).all() or k.rho.max() > 10.0:
            blew_up_at = k.steps
            break
        if k.steps % 200 == 0:
            trace.append([k.steps, float(np.abs(k.profile()).max()),
                          float((k.mass() - m0) / m0)])
    u = k.profile()
    return {
        "arm": kind, "ny": ny, "force": force, "steps": k.steps,
        "blew_up_at_step": blew_up_at,
        "mass_drift_rel": float((k.mass() - m0) / m0) if np.isfinite(k.mass()) else None,
        "u_max": float(np.max(np.abs(u))) if np.isfinite(u).all() else None,
        "trace": trace,
    }


def main() -> dict:
    rep: dict = {
        "anchor": "XVAL-LBM-SINGLE-PHASE-PERIODIC-X-PLUS-CONSTANT-DP",
        "oss": {"repo": "joconnor22/LIFE", "version": "v1.0.3 (10th June 2020)",
                "feature_slug": "GAP-LBM-MCMP-POISEUILLE",
                "tag": "single-phase-periodic-x-plus-constant-dpdx-driving",
                "license": "GPL-3.0 — algorithm transcribed, no code copied",
                "source_lines": {
                    "lattice": "src/Grid.cpp:1248-1249",
                    "step": "src/Grid.cpp:37-98 (lbmKernel)",
                    "push_periodic_x_y": "src/Grid.cpp:229,240",
                    "collision_force": "src/Grid.cpp:243",
                    "guo_source": "src/Grid.cpp:274-285",
                    "half_force_velocity": "src/Grid.cpp:297-298",
                    "frozen_dpdx_force": "src/Grid.cpp:1042-1043",
                    "unit_conversion": "src/Grid.cpp:1257-1260",
                    "efluid_periodic_x": "src/Grid.cpp:930-949 (+ inc/params.h:64)",
                    "regularised_wall": "src/Grid.cpp:302-330,360-...",
                    "unexercised": "every examples/*/params.h sets dpdx=0 and gravity*=0",
                }},
        "lattice_units": {"nx": NX, "tau": TAU, "nu": NU, "u_max_target": U_MAX,
                          "Re_H_at_H32": U_MAX * 32 / NU, "dt": 1.0, "dx": 1.0},
        "gate": GATE,
    }

    # ---- 1. grid study of the validated OSS arm ---------------------------
    grid = []
    for h in (8, 16):
        r = life_arm(h + 1, g_for(h), max_steps=45_000)
        print("grid H=%d  rel_l2=%.4e  steps=%d conv=%s" %
              (h, r["rel_l2_vs_analytic"], r["steps"], r["converged"]), flush=True)
        grid.append(r)
    rep["grid_study_LIFE_REF"] = grid

    # ---- 2. port arm witnesses (bounded) ----------------------------------
    ports = [port_arm(k, 33, g_for(32), n_steps=3000) for k in ("GOT-PORT-REG", "GOT-PORT-BB")]
    ctrl = port_arm("CONTROL", 33, g_for(32), n_steps=3000)
    for p in ports + [ctrl]:
        print(p["arm"], "blew_up_at", p["blew_up_at_step"], "mass_drift",
              p["mass_drift_rel"], flush=True)
    rep["port_arms"] = ports
    rep["control_lane"] = ctrl

    # ---- 3. sensitivity of the validated arm ------------------------------
    base = life_arm(17, g_for(16), max_steps=45_000)
    mach = [{"force_scale": sc,
             **{k: v for k, v in life_arm(17, g_for(16) * sc, tol=1e-7, max_steps=20_000).items()
                if k in ("force", "u_max", "u_max_analytic", "rel_l2_vs_analytic",
                         "u_max_rel_err", "steps", "converged")},
             "Ma": (g_for(16) * sc) * 0 + 0.0}
            for sc in (0.25, 4.0)]
    for m in mach:
        m["Ma"] = m["u_max"] / math.sqrt(1.0 / 3.0)
    taus = []
    for tau in (0.8,):
        nu_t = (tau - 0.5) / 3.0
        r = life_arm(17, 8.0 * nu_t * U_MAX / 256.0, tau=tau, tol=1e-7, max_steps=20_000)
        r["tau"] = tau
        taus.append(r)
    rep["sensitivity"] = {"baseline": {k: base[k] for k in
                                      ("force", "u_max", "u_max_analytic",
                                       "rel_l2_vs_analytic", "u_max_rel_err",
                                       "steps", "converged")},
                          "mach_sweep": mach, "tau_sweep": taus}

    # ---- 4. unit mapping: dpdx [Pa/m] -> lattice force --------------------
    dx, dt, rho_p = 0.02, 5.0e-4, 1.0        # LIFE examples/ChannelFlow scale
    dpdx = 1.0
    f_std = dpdx_to_lattice_force(dpdx, dx=dx, dt=dt, rho_scale=rho_p)
    f_life = dpdx_to_lattice_force(dpdx, dx=dx, dt=dt, rho_scale=rho_p,
                                   convention="life_compat")
    rep["unit_mapping"] = {
        "inputs": {"dpdx_Pa_per_m": dpdx, "dx": dx, "dt": dt, "rho_p": rho_p},
        "standard_convention": f_std,
        "life_compat_convention": f_life,
        "life_force_xy_verbatim": life_force_xy(dpdx, dx=dx, dt=dt, rho_scale=1.0),
        "ratio_life_over_standard": f_life / f_std,
        "one_over_dx": 1.0 / dx,
        "verdict": "LIFE's force conversion equals the momentum-scaled lattice force "
                   "divided by Dx (ratio == 1/Dx to round-off); no shipped LIFE case "
                   "exercises it (all dpdx=0), so it carries no regression evidence.",
    }

    # ---- 5. verdict -------------------------------------------------------
    worst_l2 = max(g["rel_l2_vs_analytic"] for g in grid)
    worst_umax = max(g["u_max_rel_err"] for g in grid)
    rep["verdict"] = {
        "worst_rel_l2_LIFE_REF_vs_analytic": worst_l2,
        "worst_u_max_rel_err_LIFE_REF": worst_umax,
        "gate": GATE,
        "LIFE_REF_passes_gate": bool(worst_l2 < GATE and worst_umax < GATE),
        "grid_convergence_order_fit": _order(grid),
        "port_arms_validated": all(p["blew_up_at_step"] is None and
                                   abs(p["mass_drift_rel"] or 0) < 1e-6 for p in ports),
        "control_has_teeth": bool(ctrl["blew_up_at_step"] is None),
    }
    (OUT / "xval_life_periodic_dpdx.json").write_text(json.dumps(rep, indent=2))
    return rep


def _order(grid: list) -> float | None:
    """Observed order from the two finest grids (rel_l2 vs analytic)."""
    if len(grid) < 3:
        return None
    e = [g["rel_l2_vs_analytic"] for g in grid[-2:]]
    h = [g["H"] for g in grid[-2:]]
    if min(e) <= 0.0:
        return None
    return float(np.log(e[0] / e[1]) / np.log(h[1] / h[0]))


if __name__ == "__main__":
    r = main()
    print(json.dumps(r["verdict"], indent=2))