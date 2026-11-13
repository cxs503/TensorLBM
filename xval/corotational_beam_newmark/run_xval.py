#!/usr/bin/env python3
"""XVAL-LBM-COROTATIONAL-2NODE-BEAM-NEWMARK-NEWTON -- run all predeclared cases.

Two solver arms share geometry, mesh, mass matrix, Newmark parameters, Newton
loop and post-processing; only the element formulation changes:

  TL    TensorLBM energy-consistent corotational element (g = dU/dq, exact
        tangent; transverse nodal force at (M1+M2)/L)
  LIFE  line-faithful transplant of joconnor22/LIFE
        (src/FEMElement.cpp forceVector/stiffMatrix/setLocalMatrices +
        src/FEMBody.cpp Newmark/Newton; transverse nodal force at (M1+M2)/L0)

References are external to both arms (see ``elastica_reference.py``).

Outputs (all programmatic, nothing typed by hand):
  runs/raw/<case>.json   per-case structured results
  table.csv              flat KPI table (case, arm, param, kpi, got, ref, err%)
  summary.json           KPI-level pass/fail against the 15 % owner criterion

Usage:
  python xval/corotational_beam_newmark/run_xval.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from elastica_reference import (  # noqa: E402
    elastica_shape,
    pure_moment_arc,
    tip_load_elastica,
)
from tensorlbm.corotational_beam import (  # noqa: E402
    straight_beam,
    euler_bernoulli_frequency,
)

warnings.simplefilter("ignore")

# ---------------------------------------------------------------- case set-up
L = 1.0
EI = 1.0
RHOA = 1.0                 # rho*A for the dynamic cases
# Slender cross-section: A/I = 12/h^2 with h/L = 0.01  =>  A = 1.2e5
A_SECTION = 1.2e5
ARMS = {"TL": "consistent", "LIFE": "life"}
CLAMP = [(0, "x"), (0, "y"), (0, "theta")]
N_LADDER = (2, 4, 8, 16, 32)
OWNER_PCT = 15.0

rows: list[dict] = []


def beam(n_elem: int, arm: str, *, rho: float = 1.0, E: float = 1.0):
    return straight_beam(
        L,
        n_elem,
        E=E,
        A=A_SECTION,
        I=EI,
        rho=rho if rho > 0 else 1e-30,
        bc=CLAMP,
        formulation=ARMS[arm],
    )


def record(case, arm, param, kpi, got, ref, unit=""):
    """Append one KPI row.

    ``err_pct`` follows the predeclared rule |got-ref|/|ref|*100, with two
    documented exceptions that are marked in the ``note`` column:
    * ``unit == '%'``: the KPI *is* already a percentage deviation from an exact
      zero reference (Newmark period / amplitude / energy drift), so err_pct =
      got;
    * ``|ref| < 1e-8``: the exact reference itself vanishes (pure-moment case
      with phi = pi, where the tip returns to x = 0), so the error is reported
      as 100*|got-ref|, i.e. percent of the beam length L.
    """
    note = ""
    err = None
    if ref is not None and np.isfinite(ref) and unit == "%":
        err = got
        note = "kpi already a % deviation from an exact zero reference"
    elif ref is not None and np.isfinite(ref) and abs(ref) < 1e-8:
        err = 100.0 * (got - ref)
        note = "degenerate reference |ref|<1e-8: |Delta|/L * 100"
    elif ref is not None and np.isfinite(ref) and ref != 0.0:
        err = 100.0 * (got - ref) / ref
    elif ref is not None and ref == 0.0:
        err = 100.0 * got if np.isfinite(got) else np.inf
    rows.append(
        dict(
            case=case,
            arm=arm,
            param=param,
            kpi=kpi,
            got=float(got),
            ref=None if ref is None else float(ref),
            err_pct=None if err is None else float(err),
            pass15=None if err is None else bool(abs(err) < OWNER_PCT),
            unit=unit,
            note=note,
        )
    )
    return err


# ---------------------------------------------------------------------- S1
def case_s1():
    """Small-deflection cantilever: tip load P with P L^2 / EI = 0.004."""
    lam = 0.004
    P = lam * EI / L**2
    ref_dy = -lam / 3.0          # PL^3/(3EI)
    ref_rot = -lam / 2.0         # PL^2/(2EI)
    out = {}
    for arm in ARMS:
        b = beam(8, arm)
        q, rec = b.solve_static({b.n_node - 1: (0.0, -P, 0.0)}, n_steps=1)
        dy = b.tip_deflection(q)[1] / L
        rot = b.tip_rotation(q)
        out[arm] = dict(dy=dy, rot=rot, iters=rec.iterations,
                        res_last=rec.residuals[-1], rate=rec.rate)
        record("S1_LINEAR_STATIC", arm, f"lambda={lam}", "tip_defl_y_over_L", dy, ref_dy)
        record("S1_LINEAR_STATIC", arm, f"lambda={lam}", "tip_rotation_rad", rot, ref_rot)
    return out


# ---------------------------------------------------------------------- S2
def case_s2(lams=(1.0, 2.0, 5.0, 10.0)):
    """Large-deflection elastica under a tip transverse load."""
    out = {}
    for lam in lams:
        ref = tip_load_elastica(lam)
        out[str(lam)] = {"reference": ref, "arms": {}}
        for arm in ARMS:
            out[str(lam)]["arms"][arm] = {}
            for n in N_LADDER:
                b = beam(n, arm)
                q, rec = b.solve_static({b.n_node - 1: (0.0, -lam * EI / L**2, 0.0)},
                                        n_steps=20)
                d = b.tip_deflection(q)
                dy = -d[1] / L          # positive = in the load direction
                dx = (b.x0[-1, 0] + d[0]) / L
                rot = abs(b.tip_rotation(q))
                out[str(lam)]["arms"][arm][n] = dict(
                    dy_over_L=dy, dx_over_L=dx, theta0=rot,
                    iters=rec.iterations, converged=rec.converged,
                    res_last=rec.residuals[-1], rate=rec.rate,
                )
                if n == N_LADDER[-1]:
                    tag = f"lambda={lam},n={n}"
                    record("S2_ELASTICA", arm, tag, "tip_defl_y_over_L", dy, ref["dy_over_L"])
                    record("S2_ELASTICA", arm, tag, "tip_x_over_L", dx, ref["dx_over_L"])
                    record("S2_ELASTICA", arm, tag, "tip_rotation_rad", rot, ref["theta0"])
    return out


# ---------------------------------------------------------------------- S3
def case_s3(phis=(np.pi / 2, np.pi)):
    """Pure end moment: exact circular arc."""
    out = {}
    for phi in phis:
        ref = pure_moment_arc(phi)
        M = phi * EI / L
        out[str(phi)] = {"reference": ref, "arms": {}}
        for arm in ARMS:
            out[str(phi)]["arms"][arm] = {}
            for n in N_LADDER:
                b = beam(n, arm)
                q, rec = b.solve_static({b.n_node - 1: (0.0, 0.0, M)}, n_steps=20)
                pos = b.node_position(q) / L
                out[str(phi)]["arms"][arm][n] = dict(
                    x=float(pos[0]), y=float(pos[1]), theta0=abs(b.tip_rotation(q)),
                    iters=rec.iterations, converged=rec.converged,
                    res_last=rec.residuals[-1],
                )
                if n == N_LADDER[-1]:
                    tag = f"phi={phi:.6f},n={n}"
                    record("S3_PURE_MOMENT", arm, tag, "tip_x_over_L", pos[0], ref["x_over_L"])
                    record("S3_PURE_MOMENT", arm, tag, "tip_y_over_L", pos[1], ref["y_over_L"])
                    record("S3_PURE_MOMENT", arm, tag, "tip_rotation_rad",
                           abs(b.tip_rotation(q)), ref["theta0"])
    return out


# ---------------------------------------------------------------------- D1
def case_d1(n_modes=3):
    """Linearised clamped-free frequencies vs (kL)^2."""
    rhoA_model = RHOA * A_SECTION   # the beam helper sets A = A_SECTION
    ref = euler_bernoulli_frequency(n_modes) * np.sqrt(EI / (rhoA_model * L**4))
    out = {"reference_omega": ref.tolist(), "arms": {}}
    for arm in ARMS:
        out["arms"][arm] = {}
        for n in N_LADDER:
            b = beam(n, arm, rho=RHOA)
            w = b.mode_frequencies(n_modes)
            out["arms"][arm][n] = w.tolist()
            if n == N_LADDER[-1]:
                for i, wi in enumerate(w):
                    record("D1_MODAL", arm, f"mode={i+1},n={n}", f"omega_{i+1}",
                           wi, ref[i], "rad/s")
    return out


# ---------------------------------------------------------------------- D2
def _period_from_signal(t, s):
    """Period from upward zero crossings with linear interpolation."""
    zc = []
    for k in range(len(s) - 1):
        if s[k] < 0.0 <= s[k + 1]:
            frac = -s[k] / (s[k + 1] - s[k])
            zc.append(t[k] + frac * (t[k + 1] - t[k]))
    if len(zc) < 3:
        return float("nan"), len(zc)
    return float(np.mean(np.diff(zc))), len(zc)


def case_d2(n_elem=16, n_periods=5, dt_fractions=(50, 100, 200, 400), amp_load=0.004):
    """Newmark period/amplitude error vs dt (linear small-amplitude regime)."""
    out = {}
    for arm in ARMS:
        b = beam(n_elem, arm, rho=RHOA)
        w1 = b.mode_frequencies(1)[0]
        T1 = 2.0 * np.pi / w1
        q0, _ = b.solve_static({b.n_node - 1: (0.0, -amp_load, 0.0)}, n_steps=1)
        out[arm] = {"omega1_discrete": float(w1), "T1": float(T1), "ladder": {}}
        for frac in dt_fractions:
            dt = T1 / frac
            n_steps = int(round(n_periods * frac))
            res = b.solve_newmark(dt=dt, n_steps=n_steps, q0=q0)
            tip = res["q_hist"][:, 3 * (b.n_node - 1) + 1] / L
            t = res["t"]
            Tmeas, nzc = _period_from_signal(t, tip)
            # amplitude: half peak-to-peak of the last 10 % of the record
            tail = tip[int(0.9 * len(tip)):]
            head = tip[: int(0.1 * len(tip))]
            amp_tail = 0.5 * (tail.max() - tail.min())
            amp_head = 0.5 * (head.max() - head.min())
            lam = 2.0 * np.pi / w1
            err_T = 100.0 * (Tmeas - T1) / T1 if np.isfinite(Tmeas) else float("nan")
            d = dict(
                dt=dt, steps=n_steps, T_measured=Tmeas, zero_crossings=nzc,
                period_err_pct=err_T,
                amplitude_change_pct=100.0 * (amp_tail - amp_head) / amp_head,
                converged=res["converged"], max_newton_iters=res["max_iterations"],
            )
            out[arm]["ladder"][str(frac)] = d
            record("D2_NEWMARK_TIME", arm, f"dt=T1/{frac},n={n_elem}",
                   "period_err_pct", err_T, 0.0, "%")
            record("D2_NEWMARK_TIME", arm, f"dt=T1/{frac},n={n_elem}",
                   "amplitude_change_pct", d["amplitude_change_pct"], 0.0, "%")
            del lam
        # observed order in dt from the two finest points
        fr = out[arm]["ladder"]
        e1 = abs(fr[str(dt_fractions[-2])]["period_err_pct"])
        e2 = abs(fr[str(dt_fractions[-1])]["period_err_pct"])
        out[arm]["observed_order_p"] = (
            float(np.log(e2 / e1) / np.log(2.0)) if e1 > 0 and e2 > 0 else float("nan")
        )
    return out


# ---------------------------------------------------------------------- D3
def case_d3(n_elem=16, lam=5.0, periods=(1, 1), fracs=(200, 1000)):
    """Large-amplitude free vibration: relative drift of total energy."""
    out = {}
    for arm in ARMS:
        b = beam(n_elem, arm, rho=RHOA)
        w1 = b.mode_frequencies(1)[0]
        T1 = 2.0 * np.pi / w1
        q0, _ = b.solve_static({b.n_node - 1: (0.0, -lam * EI / L**2, 0.0)}, n_steps=20)
        out[arm] = {"T1": float(T1), "ladder": {}}
        for frac, nper in zip(fracs, periods):
            e_hist = []

            def sample(step, t, q, qdot, qddot, b=b, e_hist=e_hist):
                e_hist.append(b.kinetic_energy(qdot) + b.strain_energy(q))

            res = b.solve_newmark(dt=T1 / frac, n_steps=int(nper * frac), q0=q0,
                                  sample=sample)
            e = np.asarray(e_hist)
            diverged = bool(np.abs(res["q_hist"]).max() > 1e3 * L)
            d = dict(
                dt=float(T1 / frac), steps=int(nper * frac),
                energy0=float(e[0]), energy_end=float(e[-1]),
                energy_drift_pct=float(100.0 * abs(e[-1] - e[0]) / e[0]),
                max_abs_q=float(np.abs(res["q_hist"]).max()),
                converged=bool(res["converged"] and not diverged),
                new_iters_max=res["max_iterations"],
                energy_hist=e.tolist(),
            )
            out[arm]["ladder"][str(frac)] = d
            if d["converged"]:
                record("D3_NONLINEAR_ENERGY", arm, f"lambda={lam},dt=T1/{frac},n={n_elem}",
                       "energy_drift_pct", d["energy_drift_pct"], 0.0, "%")
    return out


# ------------------------------------------------------- element consistency
def case_consistency(n_elem=4, n_configs=8, seed=12345, strains=(1e-4, 1e-3, 1e-2)):
    """|g - grad(U)| for random configurations and for controlled pure axial
    strain, showing that the LIFE denominator error scales with (L-L0)/L0."""
    rng = np.random.default_rng(seed)
    out = {"random_configs": {}, "axial_scaling": {}}
    for arm in ARMS:
        b = beam(n_elem, arm)
        devs = []
        for _ in range(n_configs):
            q = np.zeros(b.n_dof)
            q[3:] = rng.normal(scale=0.08, size=b.n_dof - 3)
            g = b.internal_forces(q)
            eps = 1e-6
            gn = np.array(
                [
                    (
                        b.strain_energy(q + eps * np.eye(b.n_dof)[i])
                        - b.strain_energy(q - eps * np.eye(b.n_dof)[i])
                    )
                    / (2.0 * eps)
                    for i in range(b.n_dof)
                ]
            )
            devs.append(float(np.linalg.norm(g - gn) / np.linalg.norm(g)))
        out["random_configs"][arm] = dict(max_rel_dev=float(max(devs)), all=devs)
        # controlled: uniform axial stretch + a bending rotation field
        for s in strains:
            b2 = beam(n_elem, arm)
            q = np.zeros(b2.n_dof)
            for i in range(b2.n_node):
                q[3 * i + 0] = s * b2.x0[i, 0]
            # small rotational noise so that bending terms are non-zero
            q[2::3] = 1e-3 * np.linspace(-1.0, 1.0, b2.n_node)
            g = b2.internal_forces(q)
            eps = 1e-7
            gn = np.array(
                [
                    (
                        b2.strain_energy(q + eps * np.eye(b2.n_dof)[i])
                        - b2.strain_energy(q - eps * np.eye(b2.n_dof)[i])
                    )
                    / (2.0 * eps)
                    for i in range(b2.n_dof)
                ]
            )
            out["axial_scaling"].setdefault(arm, {})[f"{s:.0e}"] = float(
                np.linalg.norm(g - gn) / np.linalg.norm(g)
            )
    return out


# ------------------------------------------------------------- Newton quality
def case_newton_quality():
    """Iteration counts and observed convergence rates for a nonlinear step."""
    out = {}
    for arm in ARMS:
        b = beam(16, arm)
        f = b.external_load({b.n_node - 1: (0.0, -1.0 * EI / L**2, 0.0)})
        q_star, _ = b.solve_static({b.n_node - 1: (0.0, -1.0 * EI / L**2, 0.0)}, n_steps=20)
        # one Newton solve started 10 % away from the converged state: the
        # residual sequence then exposes the asymptotic rate directly
        q, it, rec = b._newton_static_step(0.9 * q_star, f, 30, 1e-10)
        res = np.asarray(rec.residuals)
        usable = res[res > 1e-9]          # drop the round-off floor
        rate = float("nan")
        if usable.size >= 3:
            e = np.log(usable)
            rate = float((e[-1] - e[-2]) / (e[-2] - e[-3]))
        out[arm] = dict(iterations_total=rec.iterations, res_first=rec.residuals[:8],
                        observed_rate=rate, floor_limited=rec.floor_limited)
    return out


# ------------------------------------------------------------------ arm vs arm
def arm_agreement(per_case):
    """|TL - LIFE| / |TL| for every KPI pair at the finest ladder point."""
    out = {}
    for case, ref, get in per_case:
        d = {}
        for key, tl, li in get():
            if tl == 0.0:
                continue
            d[key] = 100.0 * abs(li - tl) / abs(tl)
        out[case] = {
            "value": d,
            "max_pct": float(max(d.values())) if d else float("nan"),
            "pass15": bool(all(v < OWNER_PCT for v in d.values())) if d else None,
        }
        del ref
    return out


STAGES = {
    "statics": ("S1", "S2", "S3"),
    "d1": ("D1",),
    "d2": ("D2",),
    "d3": ("D3",),
    "cons": ("element_consistency", "newton_quality"),
}
STAGE_FN = {
    "S1": lambda: case_s1(),
    "S2": lambda: case_s2(),
    "S3": lambda: case_s3(),
    "D1": lambda: case_d1(),
    "D2": lambda: case_d2(),
    "D3": lambda: case_d3(),
    "element_consistency": lambda: case_consistency(),
    "newton_quality": lambda: case_newton_quality(),
}


def run_stage(stage: str):
    """Run one stage and persist its rows + case data (each stage < 2 min)."""
    raw = HERE / "runs" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    global rows
    rows = []
    cases = {}
    for name in STAGES[stage]:
        cases[name] = STAGE_FN[name]()
    (raw / f"stage_{stage}.json").write_text(
        json.dumps({"cases": cases, "rows": rows, "wall_clock_s": time.time() - t0},
                   indent=1, default=float)
    )
    print(f"stage {stage}: {len(rows)} KPI rows in {time.time() - t0:.1f}s")
    return 0


def assemble():
    raw = HERE / "runs" / "raw"
    t0 = time.time()
    all_rows, summary = [], {}
    for f in sorted(raw.glob("stage_*.json")):
        payload = json.loads(f.read_text())
        summary.update(payload["cases"])
        all_rows.extend(payload["rows"])

    for r in all_rows:
        if r.get("unit") == "%":
            r["err_pct"] = r["got"]
            r["note"] = "kpi already a % deviation from an exact zero reference"
        elif r["ref"] is not None and abs(r["ref"]) < 1e-8:
            r["err_pct"] = 100.0 * (r["got"] - r["ref"])
            r["note"] = "degenerate reference |ref|<1e-8: |Delta|/L*100"
        elif r["ref"] not in (None, 0.0):
            r["err_pct"] = 100.0 * (r["got"] - r["ref"]) / r["ref"]
            r["note"] = ""
        r["pass15"] = None if r["err_pct"] is None else bool(abs(r["err_pct"]) < OWNER_PCT)

    global rows
    rows = all_rows
    missing = [k for k in ("S1", "S2", "S3", "D1", "D2", "D3") if k not in summary]
    if missing:
        raise RuntimeError(f"stages missing: {missing}")

    # arm-vs-arm agreement at the finest discretisation
    pairs = []
    for lam, v in summary["S2"].items():
        tl, li = v["arms"]["TL"][str(N_LADDER[-1])], v["arms"]["LIFE"][str(N_LADDER[-1])]
        pairs.append(
            (
                f"S2_ELASTICA/lam={lam}",
                v["reference"],
                lambda tl=tl, li=li: [
                    (k, tl[k], li[k]) for k in ("dy_over_L", "dx_over_L", "theta0")
                ],
            )
        )
    for phi, v in summary["S3"].items():
        tl, li = v["arms"]["TL"][str(N_LADDER[-1])], v["arms"]["LIFE"][str(N_LADDER[-1])]
        pairs.append(
            (
                f"S3_PURE_MOMENT/phi={float(phi):.6f}",
                v["reference"],
                lambda tl=tl, li=li: [(k, tl[k], li[k]) for k in ("x", "y", "theta0")],
            )
        )
    summary["arm_vs_arm"] = arm_agreement(pairs)

    # convergence deltas (mesh)
    conv = {}
    for case, v in (("S2", summary["S2"]), ("S3", summary["S3"])):
        for key, vv in v.items():
            for arm in ARMS:
                ladder = vv["arms"][arm]
                kpi = "dy_over_L" if case == "S2" else "y"
                a, b_ = ladder[str(N_LADDER[-2])][kpi], ladder[str(N_LADDER[-1])][kpi]
                conv[f"{case}[{key}]/{arm}"] = 100.0 * abs(b_ - a) / abs(b_)
    summary["mesh_convergence_delta_pct"] = conv

    # write artifacts
    (raw / "summary_cases.json").write_text(json.dumps(summary, indent=1, default=float))
    raw = HERE / "runs" / "raw"

    import csv

    with (HERE / "table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    kpis = {
        "n_kpi_rows": len(rows),
        "n_fail15": sum(1 for r in rows if r["pass15"] is False),
        "max_abs_err_pct": max(abs(r["err_pct"]) for r in rows if r["err_pct"] is not None),
        "by_case": {},
        "wall_clock_s": None,
    }
    for r in rows:
        d = kpis["by_case"].setdefault(r["case"], {"n": 0, "n_fail": 0, "max_abs_err_pct": 0.0})
        d["n"] += 1
        if r["pass15"] is False:
            d["n_fail"] += 1
        if r["err_pct"] is not None:
            d["max_abs_err_pct"] = max(d["max_abs_err_pct"], abs(r["err_pct"]))
    kpis["wall_clock_s"] = time.time() - t0
    (HERE / "summary.json").write_text(json.dumps(kpis, indent=1))
    print(json.dumps(kpis, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="assemble",
                    choices=list(STAGES) + ["assemble", "all"])
    args = ap.parse_args(argv)
    if args.stage == "all":
        for st in STAGES:
            run_stage(st)
        return assemble()
    if args.stage != "assemble":
        return run_stage(args.stage)
    return assemble()


if __name__ == "__main__":
    raise SystemExit(main())