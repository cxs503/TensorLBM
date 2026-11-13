"""Adversarial scaling probe for the Aitken under-relaxation guards.

The anchor XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXE (mission m17727)
replaced the divergent clip floor ``omega_min = 0.05`` of
``tensorlbm.fsi_partitioned.AitkenRelaxation`` by ``1e-4``.  That fixes the
measured ``beta_eff = 50`` case, but the guard is an *absolute* floor: for a
linear added-mass interface the fixed point converges only for
``|1 - omega (1+beta_eff)| < 1``, i.e. ``omega < 2/(1+beta_eff)``, so any floor
``omega_min`` re-introduces divergence as soon as
``beta_eff > 1/omega_min - 1``.  With ``omega_min = 1e-4`` that is
``beta_eff > 9999`` -- the failure threshold moved by a factor 256, it was not
removed.

The same defect class lives in the two sibling copies of the accelerator
(``fsi_added_mass``: floor ``1e-4``, ``fsi_partitioned_stability``: floor
``1e-3``, threshold ``beta_eff > 999``), which is the structural root cause
recorded in the m17727 report ("两处实现不一致是本缺陷的根因").

This script measures, on the bare 1-DOF fixed-point map
``r(a) = 1 - (1+beta) a`` (gain ``beta_eff = m_a/m_eff``, analytic optimum
``omega* = 1/(1+beta)``):

  A. the shipped drivers (:func:`solve_partitioned_1d`) over an extended
     ``beta_eff`` ladder -- divergence threshold before/after the guard change,
  B. the factor recovered by each of the three ``AitkenRelaxation`` copies on
     the same residual sequence (cross-module consistency),
  C. the number of sub-iterations needed at large ``beta_eff``.

Output: ``out/scale_guard.json`` (machine written, no hand-copied numbers).
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reference as R  # noqa: E402

from tensorlbm import fsi_partitioned as FP  # noqa: E402
from tensorlbm.fsi_added_mass import AitkenRelaxation as AitkenAddedMass  # noqa: E402
from tensorlbm.fsi_partitioned_stability import (  # noqa: E402
    AitkenRelaxation as AitkenStability,
)

OUT = os.path.join(HERE, "out")
BETAS = [0.2, 0.5, 2.0, 10.0, 25.0, 50.0, 200.0, 1_000.0, 4_000.0, 9_000.0,
         10_000.0, 20_000.0, 100_000.0]
TOL_REL = 1e-8
REF_LEN = 1.0
STEPS_PER_PERIOD = 100


def _case(beta: float, steps_per_period: int = STEPS_PER_PERIOD):
    p0 = R.Piston(m_struct=1.0, beta_eff=beta, stiffness=100.0, damping=0.3, dt=0.0)
    p = R.Piston(m_struct=1.0, beta_eff=beta, stiffness=100.0, damping=0.3,
                 dt=p0.period / steps_per_period)
    omega_f = 2.0 * math.pi / p.period
    f_ext = lambda t: math.sin(omega_f * t)  # noqa: E731
    model = FP.AddedMassModel1D(m_struct=p.m_struct, m_added=p.m_added,
                                stiffness=p.stiffness, damping=p.damping)
    scale = REF_LEN / (R.BETA_N * p.dt * p.dt)
    return p, f_ext, model, scale


def _run_driver(beta: float, n_steps: int = 60, **acc_kw):
    p, f_ext, model, scale = _case(beta)
    acc = FP.AitkenRelaxation(omega0=0.5, **acc_kw)
    res = FP.solve_partitioned_1d(model, n_steps=n_steps, dt=p.dt, variant="explicit",
                                  forcing=f_ext, accelerator=acc, tol=TOL_REL,
                                  max_iter=200, ref_scale=scale)
    mono = R.monolithic_newmark(p, n_steps=n_steps, f_ext=f_ext)
    ana = R.analytic_sdof(p, n_steps=n_steps, amp=1.0, freq=2.0 * math.pi / p.period)
    scale_d = float(np.max(np.abs(mono["d"])))
    d = res.displacement
    err_mono = (float(np.max(np.abs(d - mono["d"]))) / scale_d
                if np.all(np.isfinite(d)) else float("nan"))
    err_ana = (float(np.max(np.abs(d - ana["d"]))) / scale_d
               if np.all(np.isfinite(d)) else float("nan"))
    return {
        "err_vs_monolithic_pct": 100.0 * err_mono,
        "err_vs_analytic_pct": 100.0 * err_ana,
        "beta_eff": beta,
        "converged": bool(res.converged),
        "iters_mean": float(np.mean(res.iterations)),
        "iters_max": int(np.max(res.iterations)),
        "omega_final": float(acc.omega),
        "omega_star": float(p.omega_star),
        "finite": bool(np.all(np.isfinite(res.displacement))),
        "d_final": float(res.displacement[-1]) if np.all(np.isfinite(res.displacement)) else None,
    }


def _probe_module(kind: str, beta: float, iters: int = 200, tol: float = 1e-12):
    """Drive one of the three accelerator copies on the scalar fixed-point map.

    ``r(a) = 1 - (1+beta) a`` has gain ``-(1+beta)``; ``a <- a + omega r`` with
    ``omega = 1/(1+beta)`` lands on the fixed point in one update.
    """
    if kind == "partitioned":
        acc = FP.AitkenRelaxation(omega0=1.0)
        def step(a, r):
            x, w = acc.step(np.array([a]), np.array([r]))
            return float(x[0]), float(w)
    elif kind == "added_mass":
        acc = AitkenAddedMass(omega0=1.0)
        def step(a, r):
            inc, w = acc.step(np.array([a]), np.array([r]))
            return a + float(inc[0]), float(w)
    elif kind == "stability":
        acc = AitkenStability(omega0=1.0)
        def step(a, r):
            w = acc.update(np.array([r]), np.zeros(1))
            return a + w * r, float(acc.omega)
    else:  # pragma: no cover
        raise ValueError(kind)

    a, hist = 0.0, []
    for _ in range(iters):
        r = 1.0 - (1.0 + beta) * a
        if abs(r) < tol:
            break
        a, w = step(a, r)
        hist.append(w)
        if not math.isfinite(a) or abs(a) > 1e12:
            hist.append(float("nan"))
            break
    return {
        "kind": kind,
        "beta_eff": beta,
        "omega_star": 1.0 / (1.0 + beta),
        "omega_final": hist[-1] if hist else None,
        "updates": len(hist),
        "residual_final": abs(1.0 - (1.0 + beta) * a) if math.isfinite(a) else float("nan"),
        "converged": bool(math.isfinite(a) and abs(1.0 - (1.0 + beta) * a) < 1e-8),
        "finite": bool(math.isfinite(a)),
    }


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    out = {
        "note": "machine-written by scale_guard_probe.py; no hand-copied numbers",
        "tolerance_rel": TOL_REL,
        "steps_per_period": STEPS_PER_PERIOD,
        "driver_ladder": [],
        "module_factor": [],
    }
    for beta in BETAS:
        row = _run_driver(beta)
        row["arm"] = "shipped_after_patch"
        out["driver_ladder"].append(row)
        # pre-patch baseline floor (m17727 finding) and the unconditional
        # raw Aitken convention, for the before/after contrast
        for label, kw in (("pre_patch_floor_1e-4", dict(omega_min=1e-4)),
                          ("pre_patch_floor_5e-2", dict(omega_min=5e-2)),
                          ("raw_no_clip", dict(clip=False))):
            if label == "pre_patch_floor_1e-4" and abs(beta - 0.2) < 1e-12:
                pass
            r = _run_driver(beta, **kw)
            r["arm"] = label
            out["driver_ladder"].append(r)
        for kind in ("partitioned", "added_mass", "stability"):
            out["module_factor"].append(_probe_module(kind, beta))

    with open(os.path.join(OUT, "scale_guard.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"{'arm':22s} {'beta':>9s} {'conv':>5s} {'it_mean':>8s} {'omega':>10s} {'omega*':>10s}")
    for row in out["driver_ladder"]:
        print(f"{row['arm']:22s} {row['beta_eff']:9.1f} {str(row['converged']):>5s} "
              f"{row['iters_mean']:8.2f} {row['omega_final']:10.3e} {row['omega_star']:10.3e}")
    print()
    print(f"{'kind':12s} {'beta':>9s} {'conv':>5s} {'updates':>7s} {'omega':>10s} {'omega*':>10s}")
    for row in out["module_factor"]:
        print(f"{row['kind']:12s} {row['beta_eff']:9.1f} {str(row['converged']):>5s} "
              f"{row['updates']:7d} "
              f"{(row['omega_final'] if row['omega_final'] is not None else float('nan')):10.3e} "
              f"{row['omega_star']:10.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())