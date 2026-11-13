"""XVAL: displacement-Aitken under-relaxation fixed point (TensorLBM vs joconnor22/LIFE).

Anchor: XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXE.

Design (see ``predeclared.json``): a 1-DOF piston wetted by an incompressible
fluid column, i.e. the added-mass interface problem in which the Aitken
under-relaxation is the load-bearing numerical device.  Six interface arms are
run on the same physical target and judged against two independent references
(discrete monolithic Newmark = ``REF-D``, continuous closed form = ``REF-C``)
and against the analytic Aitken fixed point ``omega* = 1/(1+beta_eff)``.

Usage:  PYTHONPATH=src python3 xval/aitken_underrelaxation_life/run_xval.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from reference import BETA_N, Piston, analytic_sdof, life_run, monolithic_newmark  # noqa: E402
from tensorlbm.fsi_partitioned import (  # noqa: E402
    AddedMassModel1D,
    AitkenRelaxation,
    ConstantRelaxation,
    NoRelaxation,
    solve_partitioned_1d,
)

OUT = os.path.join(HERE, "out")
REF_LEN = 1.0                 # LIFE's ref_L (m); residual normalisation length
TOL_REL = 1e-8                # LIFE's subTol on the normalised residual
MAX_ITER = 200                # repo default; LIFE uses 20 (measured: 3 needed)
BETAS = (0.2, 0.5, 2.0, 10.0, 25.0, 50.0)
M_STRUCT = 1.0                                            # kg
STIFF = 100.0                                             # N/m
DAMP = 0.3                                                # N s/m
FORCE_AMP = 1.0                                           # N
FORCE_PERIODS = 3.0
STEPS_PER_PERIOD = 200


def make_case(beta: float, steps_per_period: int = STEPS_PER_PERIOD) -> tuple[Piston, callable]:
    p = Piston(m_struct=M_STRUCT, beta_eff=beta, stiffness=STIFF, damping=DAMP, dt=0.0)
    dt = p.period / steps_per_period
    p = Piston(m_struct=M_STRUCT, beta_eff=beta, stiffness=STIFF, damping=DAMP, dt=dt)
    # forcing at the coupled natural frequency (resonant drive = worst case)
    omega_f = 2.0 * math.pi / p.period
    f_ext = lambda t: FORCE_AMP * math.sin(omega_f * t)  # noqa: E731
    return p, f_ext


def model_of(p: Piston) -> AddedMassModel1D:
    return AddedMassModel1D(
        m_struct=p.m_struct, m_added=p.m_added, stiffness=p.stiffness, damping=p.damping
    )


def _accel_scale(p: Piston) -> float:
    """displacement residual -> acceleration residual conversion."""
    return REF_LEN / (BETA_N * p.dt * p.dt)


def run_arm(name: str, p: Piston, n_steps: int, f_ext, *, steps_per_period=STEPS_PER_PERIOD) -> dict:
    """Run one couple-numerics arm and return its KPI record."""
    model = model_of(p)
    acc = None
    kwargs = dict(n_steps=n_steps, dt=p.dt, variant="explicit", forcing=f_ext,
                  tol=TOL_REL, max_iter=MAX_ITER, ref_scale=_accel_scale(p))
    if name == "ours_old_default":
        # behaviour shipped before this mission: clip floor omega_min = 0.05
        acc = AitkenRelaxation(omega0=0.5, omega_min=0.05)
    elif name == "ours_new_default":
        acc = AitkenRelaxation(omega0=0.5)                       # fixed default (1e-4)
    elif name == "ours_noclip":
        acc = AitkenRelaxation(omega0=0.5, clip=False)
    elif name == "ours_life_conv":
        acc = AitkenRelaxation(omega0=1.0, clip=False)
        kwargs["predictor"] = "quadratic"
    elif name == "none_control":
        acc = NoRelaxation()
    elif name == "oracle_constant":
        acc = ConstantRelaxation(p.omega_star)
        kwargs["tol"] = TOL_REL
        kwargs["ref_scale"] = _accel_scale(p)
    else:
        raise ValueError(name)
    res = solve_partitioned_1d(model, accelerator=acc, **kwargs)

    mono = monolithic_newmark(p, n_steps=n_steps, f_ext=f_ext)
    ana = analytic_sdof(p, n_steps=n_steps, amp=FORCE_AMP, freq=2.0 * math.pi / p.period)
    d = np.asarray(res.displacement)
    scale_d = float(np.max(np.abs(mono["d"]))) or 1.0
    rec = {
        "arm": name,
        "beta_eff": p.beta_eff,
        "steps": int(n_steps),
        "iters_mean": float(np.mean(res.iterations)),
        "iters_max": int(np.max(res.iterations)),
        "iters_total": int(np.sum(res.iterations)),
        "converged": bool(res.converged),
        "omega_final": float(acc.omega),
        "omega_star": float(p.omega_star),
        "omega_rel_dev_pct": abs(float(acc.omega) - p.omega_star) / p.omega_star * 100.0,
        "err_vs_monolithic_pct": float(np.max(np.abs(d - mono["d"])) / scale_d * 100.0),
        "err_vs_analytic_pct": float(np.max(np.abs(d - ana["d"])) / np.max(np.abs(ana["d"])) * 100.0),
        "d_final": float(d[-1]),
        "d_monolithic_final": float(mono["d"][-1]),
        "d_analytic_final": float(ana["d"][-1]),
        "dt": float(p.dt),
        "dt_over_period": float(p.dt / p.period),
        "tol_rel_displacement": TOL_REL,
        "omega_per_step": [float(h[-1]) if h else 1.0 for h in res.omega_history],
        "iters_per_step": [int(x) for x in res.iterations],
    }
    return rec, res, mono, ana


def run_life_ref(p: Piston, n_steps: int, f_ext, steps_per_period=STEPS_PER_PERIOD) -> dict:
    r = life_run(p, n_steps=n_steps, f_ext=f_ext, tol=TOL_REL, max_iter=MAX_ITER, ref_len=REF_LEN)
    mono = monolithic_newmark(p, n_steps=n_steps, f_ext=f_ext)
    ana = analytic_sdof(p, n_steps=n_steps, amp=FORCE_AMP, freq=2.0 * math.pi / p.period)
    d = r["d"]
    scale_d = float(np.max(np.abs(mono["d"]))) or 1.0
    oms = [o for o in r["omega"] if math.isfinite(o)]
    omega_final = float(oms[-1]) if oms else float("nan")
    rec = {
        "arm": "life_ref",
        "beta_eff": p.beta_eff,
        "steps": int(n_steps),
        "iters_mean": float(np.mean(r["iters"])),
        "iters_max": int(np.max(r["iters"])),
        "iters_total": int(np.sum(r["iters"])),
        "converged": bool(r["converged"]),
        "omega_final": omega_final,
        "omega_star": float(p.omega_star),
        "omega_rel_dev_pct": abs(omega_final - p.omega_star) / p.omega_star * 100.0,
        "err_vs_monolithic_pct": float(np.max(np.abs(d - mono["d"])) / scale_d * 100.0),
        "err_vs_analytic_pct": float(np.max(np.abs(d - ana["d"])) / np.max(np.abs(ana["d"])) * 100.0),
        "d_final": float(d[-1]),
        "d_monolithic_final": float(mono["d"][-1]),
        "d_analytic_final": float(ana["d"][-1]),
        "dt": float(p.dt),
        "dt_over_period": float(p.dt / p.period),
        "tol_rel_displacement": TOL_REL,
        "omega_per_step": [float(o) for o in r["omega"]],
        "iters_per_step": [int(x) for x in r["iters"]],
    }
    return rec, r, mono, ana


ARMS = ("ours_old_default", "ours_new_default", "ours_noclip", "ours_life_conv",
        "none_control", "oracle_constant")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows: list[dict] = []
    hist_store: dict[str, dict] = {}
    cross_impl: list[dict] = []
    for beta in BETAS:
        p, f_ext = make_case(beta)
        n_steps = int(round(FORCE_PERIODS * STEPS_PER_PERIOD))
        for arm in ARMS:
            rec, res, mono, ana = run_arm(arm, p, n_steps, f_ext)
            rows.append(rec)
            if arm in ("ours_old_default", "ours_new_default", "ours_life_conv"):
                hist_store[f"{arm}@{beta}"] = {
                    "t": list(map(float, res.t)),
                    "d": list(map(float, res.displacement)),
                    "residuals_first_step": [float(x) for x in res.residual_history[0]],
                    "omega_first_step": [float(x) for x in res.omega_history[0]],
                }
            print(f"beta={beta:5.2f} {arm:20s} iters/step={rec['iters_mean']:7.2f} "
                  f"conv={rec['converged']} err_mono={rec['err_vs_monolithic_pct']:.2e}% "
                  f"err_ana={rec['err_vs_analytic_pct']:.3e}% omega={rec['omega_final']:.5f}")
        # LIFE reference path (independent transcription)
        rec, r, mono, ana = run_life_ref(p, n_steps, f_ext)
        rows.append(rec)
        ours = next(x for x in rows if x["arm"] == "ours_life_conv" and x["beta_eff"] == beta)
        w_ours = np.asarray(ours["omega_per_step"], dtype=float)
        w_life = np.asarray(rec["omega_per_step"], dtype=float)
        i_ours = np.asarray(ours["iters_per_step"], dtype=int)
        i_life = np.asarray(rec["iters_per_step"], dtype=int)
        nmin = min(len(w_ours), len(w_life))
        # only steps where BOTH arms actually ran an Aitken update (>= 2
        # sub-iterations) are comparable: at one sub-iteration no factor exists
        # >= 3 sub-iterations => at least one *Aitken* update was applied
        # (with 1-2 sub-iterations the unrelaxed start converged and the
        # factor was never updated, so there is nothing to compare)
        active = (i_ours[:nmin] >= 3) & (i_life[:nmin] >= 3)
        dev = (np.abs(w_ours[:nmin] - w_life[:nmin]) / np.maximum(np.abs(w_life[:nmin]), 1e-30))[active]
        cross_impl.append({
            "beta_eff": beta,
            "omega_ours_on_active_steps": float(np.median(w_ours[:nmin][active])) if active.any() else None,
            "omega_life_on_active_steps": float(np.median(w_life[:nmin][active])) if active.any() else None,
            "omega_star": float(p.omega_star),
            "active_step_fraction": float(np.mean(active)),
            "n_active_steps": int(np.sum(active)),
            "omega_max_rel_dev_pct_ours_vs_life": float(np.max(dev) * 100.0) if dev.size else 0.0,
            "omega_mean_rel_dev_pct_ours_vs_life": float(np.mean(dev) * 100.0) if dev.size else 0.0,
            "iters_mean_ours": ours["iters_mean"],
            "iters_mean_life": rec["iters_mean"],
            "iters_rel_diff_pct": abs(ours["iters_mean"] - rec["iters_mean"]) / rec["iters_mean"] * 100.0,
            "err_mono_ours_pct": ours["err_vs_monolithic_pct"],
            "err_mono_life_pct": rec["err_vs_monolithic_pct"],
        })
        hist_store[f"life_ref@{beta}"] = {
            "t": list(map(float, r["t"])),
            "d": list(map(float, r["d"])),
            "residuals_first_step": [float(x) for x in r["per_step"][0].residuals],
            "omega_first_step": [float(x) for x in r["per_step"][0].omegas],
        }
        # monolithic / analytic reference history for the figures
        hist_store[f"ref@{beta}"] = {
            "t": list(map(float, mono["t"])),
            "d_mono": list(map(float, mono["d"])),
            "d_ana": list(map(float, ana["d"])),
        }
        print(f"beta={beta:5.2f} {'life_ref':20s} iters/step={rec['iters_mean']:7.2f} "
              f"conv={rec['converged']} err_mono={rec['err_vs_monolithic_pct']:.2e}% "
              f"err_ana={rec['err_vs_analytic_pct']:.3e}% omega={rec['omega_final']:.5f}")

    # ---- recurrence identity: same residual sequence into both codes -----
    rng = np.random.default_rng(20261113)
    recurrence = []
    for scale in (1.0, 1e-3, 1e3):
        seq = rng.normal(size=(12, 3)) * scale
        acc_ours = AitkenRelaxation(omega0=1.0, clip=False)
        w_ours, w_life_ref, relax_ref, r_prev = [], [], 1.0, None
        for r in seq:
            _, w = acc_ours.step(np.zeros(3), r)
            if r_prev is not None:
                dr = r - r_prev
                den = float(dr @ dr)
                if den > 1e-300:
                    cand = -relax_ref * float(r_prev @ dr) / den
                    if math.isfinite(cand):
                        relax_ref = cand
                w_life_ref.append(relax_ref)
            r_prev = r
            w_ours.append(w)
        a = np.asarray(w_ours[1:], dtype=float)
        b = np.asarray(w_life_ref, dtype=float)
        recurrence.append({
            "residual_scale": scale,
            "n_updates": int(len(b)),
            "max_rel_dev_pct": float(np.max(np.abs(a - b) / np.abs(b)) * 100.0),
            "omega_last": float(b[-1]),
        })

    # ---- sensitivity 1: initial relaxation omega0 ------------------------
    sens_omega0 = []
    for beta in (10.0, 25.0):
        p, f_ext = make_case(beta)
        n_steps = int(round(1.5 * STEPS_PER_PERIOD))
        for w0 in (0.05, 0.3, 0.5, 1.0):
            for name, kw in (("ours_old_default", dict(omega0=w0, omega_min=0.05)),
                             ("ours_new_default", dict(omega0=w0)),
                             ("ours_noclip", dict(omega0=w0, clip=False))):
                acc = AitkenRelaxation(**kw)
                res = solve_partitioned_1d(
                    model_of(p), n_steps=n_steps, dt=p.dt, variant="explicit", forcing=f_ext,
                    accelerator=acc, tol=TOL_REL, max_iter=MAX_ITER, ref_scale=_accel_scale(p))
                sens_omega0.append({
                    "beta_eff": beta, "arm": name, "omega0": w0,
                    "iters_mean": float(np.mean(res.iterations)),
                    "iters_max": int(np.max(res.iterations)),
                    "converged": bool(res.converged),
                    "omega_final": float(acc.omega),
                })

    # ---- sensitivity 2: time resolution dt/T -----------------------------
    sens_dt = []
    for beta in (2.0, 25.0):
        for spp in (100, 200, 400):
            p, f_ext = make_case(beta, steps_per_period=spp)
            n_steps = int(round(2 * spp))
            rec, res, mono, ana = run_arm("ours_life_conv", p, n_steps, f_ext)
            rec2, r2, _, _ = run_life_ref(p, n_steps, f_ext)
            sens_dt.append({
                "beta_eff": beta, "steps_per_period": spp,
                "err_vs_analytic_pct_ours": rec["err_vs_analytic_pct"],
                "err_vs_analytic_pct_life": rec2["err_vs_analytic_pct"],
                "iters_mean": rec["iters_mean"],
                "converged": rec["converged"],
            })

    payload = {
        "anchor": "XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXE",
        "reference_oss": {
            "repo": "joconnor22/LIFE",
            "feature_slug": "NEW:LBM-FSI-BLOCK-GS-AITKEN",
            "oss_solutions_row": 258,
            "tag": "displacement-aitken-underrelaxation-fixed-point",
            "frozen_method_quote": (
                "do{predict 2.5U_n-2U_nm1+0.5U_nm2 -> interpolate -> FEM full re-solve "
                "(Newton-Raphson TOL=1e-10 MAXIT=20) -> residual} while(res>1e-8 and it<20); "
                "Aitken at k>=2: relax=-relax*(r_{k-1}.(r_k-r_{k-1}))/|r_k-r_{k-1}|^2, "
                "U=U_km1+relax*(U-U_km1); residual normalised by ref_L*sqrt(DOF); "
                "IBM force spread once after convergence"
            ),
        },
        "parameters": {
            "m_struct_kg": M_STRUCT, "stiffness_N_per_m": STIFF, "damping_N_s_per_m": DAMP,
            "force_amp_N": FORCE_AMP, "force_periods": FORCE_PERIODS,
            "steps_per_period": STEPS_PER_PERIOD, "tol_rel_displacement": TOL_REL,
            "max_iter": MAX_ITER, "ref_len_m": REF_LEN, "newmark_gamma": 0.5,
            "newmark_beta": BETA_N,
        },
        "rows": rows,
        "cross_implementation": cross_impl,
        "recurrence_identity": recurrence,
        "sensitivity_omega0": sens_omega0,
        "sensitivity_dt": sens_dt,
        "histories": hist_store,
    }
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=1)

    for r in rows:
        r.pop("omega_per_step", None)
        r.pop("iters_per_step", None)
    keys = ["arm", "beta_eff", "steps", "iters_mean", "iters_max", "converged", "omega_final",
            "omega_star", "omega_rel_dev_pct", "err_vs_monolithic_pct", "err_vs_analytic_pct",
            "d_final", "d_monolithic_final", "d_analytic_final", "dt", "dt_over_period"]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    with open(os.path.join(OUT, "cases.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["beta_eff", "m_added", "m_eff_solid", "dt", "period",
                                          "dt_over_period", "omega_star"])
        w.writeheader()
        for beta in BETAS:
            p, _ = make_case(beta)
            w.writerow({"beta_eff": beta, "m_added": p.m_added, "m_eff_solid": p.m_eff_solid,
                        "dt": p.dt, "period": p.period, "dt_over_period": p.dt / p.period,
                        "omega_star": p.omega_star})
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())