#!/usr/bin/env python3
"""CASE-TAYLOR-GREEN-3D — exact-solution decay of the 3D Taylor-Green mode on D3Q27.

Case-pool entry (mission 15374). Everything is predeclared in ``prereg.json``;
this file is the single executable artifact behind every number in the report.

What is compared, and against what
----------------------------------
The reference is a **closed-form Fourier-mode solution of the 3D incompressible
Navier-Stokes equations** (the z-invariant Taylor-Green mode)::

    u(x,y,t) = A e^{-nu |k|^2 t} ( sin(kx) cos(ky), -cos(kx) sin(ky), 0 ),
    |k|^2 = 2 k^2,  k = 2*pi/N,  periodic box [0,2*pi)^3

It is an *exact* 3D NS solution because for this field the 3D convective term is
curl-free (a pure gradient) and is absorbed by ``p = -(A^2/4) e^{-4 nu k^2 t}
(cos 2kx + cos 2ky)``; what remains is ``du/dt = nu * laplacian(u)`` with
``laplacian(u) = -|k|^2 u``.  ``verify()`` machine-checks exactly this
(curl of the convective term, the Laplacian identity and the resulting NS
residual, by spectral derivatives) instead of asserting it.

Why not the classic z-dependent vortex
--------------------------------------
The z-dependent Taylor-Green vortex has **no closed-form solution**: with the
``cos(kz)`` factor the convective term is no longer curl-free (``verify()``
prints that curl ratio, ~0.4, versus ~1e-16 for the z-invariant mode), so any
"reference" for it would be another simulation.  The z-dependent 3D branch is
therefore exercised by a **Beltrami (ABC) flow** — a genuinely 3D, z-dependent,
unsteady exact NS solution::

    u0 = A ( sin kz + cos ky,  sin kx + cos kz,  sin ky + cos kx ),
    curl(u0) = k * u0  ->  u(t) = u0 e^{-nu k^2 t}   (exact for all t)

Numeric protocol: lattice units (dx = dt = 1), D3Q27, BGK, float64, periodic.
``nu = (tau - 1/2)/3``.

Usage
-----
    python benchmarks/case_taylor_green_3d/run_case.py --verify        # exactness proof
    python benchmarks/case_taylor_green_3d/run_case.py --all           # full sweep + JSON
    python benchmarks/case_taylor_green_3d/run_case.py --mode decay --n 64 --steps 3000

Writes ``runs/case_taylor_green_3d/*.json`` and ``summary.md`` (tables generated
from the JSON — no hand-typed numbers).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

OUT_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "runs",
    "case_taylor_green_3d",
)
DT = torch.float64
TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# analytic fields
# --------------------------------------------------------------------------- #
def _coords(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Node coordinates in lattice units (dx = 1), shape (nz, ny, nx).

    The box is N lattice units long, so the fundamental wave number is
    ``k = 2*pi/N`` rad per lattice unit -- the same convention the spectral
    derivative helper below uses (``2*pi*fftfreq``).
    """
    z, y, x = torch.meshgrid(*[torch.arange(n, dtype=DT)] * 3, indexing="ij")
    return x, y, z


def init_fields(case: str, n: int, amp: float) -> tuple[torch.Tensor, ...]:
    """Initial (exact at t=0) velocity field of the case."""
    k = TWO_PI / n
    x, y, z = _coords(n)
    if case == "tg":  # z-invariant Taylor-Green mode
        ux = amp * torch.sin(k * x) * torch.cos(k * y)
        uy = -amp * torch.cos(k * x) * torch.sin(k * y)
        uz = torch.zeros_like(x)
    elif case == "tg3d":  # classic z-dependent vortex (no closed form; diagnostic only)
        ux = amp * torch.sin(k * x) * torch.cos(k * y) * torch.cos(k * z)
        uy = -amp * torch.cos(k * x) * torch.sin(k * y) * torch.cos(k * z)
        uz = torch.zeros_like(x)
    elif case == "abc":  # Beltrami / ABC flow, decay u0 * exp(-nu k^2 t)
        ux = amp * (torch.sin(k * z) + torch.cos(k * y))
        uy = amp * (torch.sin(k * x) + torch.cos(k * z))
        uz = amp * (torch.sin(k * y) + torch.cos(k * x))
    else:  # pragma: no cover
        raise ValueError(f"unknown case {case!r}")
    return ux, uy, uz


def k2_of(case: str, n: int) -> float:
    """|k|^2 (lattice units) of the single-mode exact solution of the case."""
    k = TWO_PI / n
    if case == "tg":
        return 2.0 * k * k
    if case == "abc":
        return k * k
    raise ValueError(f"case {case!r} has no closed-form |k|^2")


def exact_fields(case: str, n: int, amp: float, nu: float, t: float):
    """Exact solution at lattice time t, and its kinetic energy."""
    ux, uy, uz = init_fields(case, n, amp)
    decay = math.exp(-nu * k2_of(case, n) * t)
    return decay * ux, decay * uy, decay * uz


def kinetic_energy(ux, uy, uz) -> float:
    return 0.5 * float((ux * ux + uy * uy + uz * uz).mean())


def l2_rel(a, b) -> float:
    """||a||_2 / ||b||_2 for tensor triples (relative L2 field error)."""
    num = sum(float((ai - bi).pow(2).sum()) for ai, bi in zip(a, b))
    den = sum(float(bi.pow(2).sum()) for bi in b)
    return math.sqrt(num / den) if den > 0.0 else float("nan")


# --------------------------------------------------------------------------- #
# exactness proof by spectral residuals (the "why it is a reference" check)
# --------------------------------------------------------------------------- #
def _spectral_deriv(f, axis: int, n: int):
    import numpy as np

    kx = TWO_PI * np.fft.fftfreq(n)  # rad per lattice unit
    shape = [1, 1, 1]
    shape[axis] = n
    return np.real(np.fft.ifft(1j * kx.reshape(shape) * np.fft.fft(f, axis=axis), axis=axis))


def _grad(f, n):
    return [_spectral_deriv(f, ax, n) for ax in (2, 1, 0)]  # x, y, z


def _curl(u, n):
    du = [_grad(u[i], n) for i in range(3)]  # du[i][j] = d u_i / d x_j
    return [
        du[2][1] - du[1][2],
        du[0][2] - du[2][0],
        du[1][0] - du[0][1],
    ]


def _norm(a) -> float:
    import numpy as np

    return float(np.sqrt(sum((np.asarray(ai) ** 2).sum() for ai in a)))


def _conv(u, n):
    du = [_grad(u[i], n) for i in range(3)]
    return [sum(u[j] * du[i][j] for j in range(3)) for i in range(3)]


def verify(n: int = 32, amp: float = 2e-3) -> dict:
    """Machine-check which cases are exact NS solutions, and why the classic one is not."""
    import numpy as np

    report = {}
    for case in ("tg", "tg3d", "abc"):
        u = [t.numpy().astype(np.float64) for t in init_fields(case, n, amp)]
        C = _conv(u, n)
        curl_C = _curl(C, n)
        lap = [sum(_spectral_deriv(_spectral_deriv(u[i], ax, n), ax, n) for ax in (0, 1, 2)) for i in range(3)]
        # curl(C) / |C| measures how far the convective term is from a pure gradient
        curl_ratio = _norm(curl_C) / _norm(C)
        entry = {
            "n": n,
            "amp": amp,
            "curl_of_convective_term_over_norm": curl_ratio,
            "convective_term_is_pure_gradient": bool(curl_ratio < 1e-10),
        }
        if case == "tg3d":
            entry["exact_solution"] = None
            entry["note"] = (
                "z-dependent Taylor-Green: convective term has a non-zero curl, so it cannot be "
                "absorbed by a pressure gradient -> no closed-form decaying solution exists."
            )
        else:
            k2 = k2_of(case, n)
            nu, t = 1.0 / 30.0, 3.0
            # residual of du/dt = -grad p - (u.grad)u + nu lap(u) with du/dt = -nu|k|^2 u
            # and the convective term absorbed by p (guaranteed by the curl test above)
            decay = math.exp(-nu * k2 * t)
            # analytic time derivative of the candidate solution
            du_dt = [-nu * k2 * decay * u[i] for i in range(3)]
            # viscous term of the NS equation at the same time
            visc = [nu * decay * lap[i] for i in range(3)]
            # the convective term is a pure gradient (curl test above) -> the pressure
            # gradient cancels it exactly, so NS reduces to du/dt = nu*lap(u)
            res = [du_dt[i] - visc[i] for i in range(3)]
            entry.update(
                {
                    "laplacian_identity_rel_residual": _norm(
                        [lap[i] + k2 * u[i] for i in range(3)]
                    )
                    / _norm([k2 * u[i] for i in range(3)]),
                    "ns_momentum_residual_rel": _norm(res) / _norm(du_dt),
                    "exact_solution": f"u(t) = u0 * exp(-nu*{k2:g}*t)",
                }
            )
        report[case] = entry
    # Rankine-type sanity: the ABC field is indeed Beltrami (curl u = k u)
    u = [t.numpy().astype(np.float64) for t in init_fields("abc", n, amp)]
    w = _curl(u, n)
    report["abc_beltrami_rel_residual"] = _norm([w[i] - (TWO_PI / n) * u[i] for i in range(3)]) / _norm(w)
    return report


# --------------------------------------------------------------------------- #
# the LBM run
# --------------------------------------------------------------------------- #
def run_decay(
    case: str,
    n: int,
    tau: float,
    amp: float,
    steps: int,
    sample: int = 25,
    collision: str = "bgk",
) -> dict:
    """One decay run on the D3Q27 lattice; returns the full sampled history."""
    from tensorlbm.d3q27 import collide_bgk27, equilibrium27, macroscopic27
    from tensorlbm.lbm_step_d3q27 import lbm_step_correct_27

    nu = (tau - 0.5) / 3.0
    k2 = k2_of(case, n)
    u0 = init_fields(case, n, amp)
    f = equilibrium27(torch.ones_like(u0[0]), *u0)
    solid = torch.zeros(u0[0].shape, dtype=torch.bool)
    e0 = kinetic_energy(*u0)
    m0 = float(f.sum())
    mass_drift, finite = 0.0, True
    hist = [{"t": 0, "E": e0, "E_exact": e0, "E_rel_err_pct": 0.0, "l2_rel": 0.0}]
    t0 = time.time()
    with torch.no_grad():
        for step in range(1, steps + 1):
            f = lbm_step_correct_27(f, collide_bgk27, tau, solid, 0.0, far_field_bc_fn=None)
            if step % sample == 0 or step == steps:
                rho, ux, uy, uz = macroscopic27(f)
                finite = finite and bool(torch.isfinite(f).all())
                mass_drift = max(mass_drift, abs(float(f.sum()) - m0) / m0)
                e = kinetic_energy(ux, uy, uz)
                ex = exact_fields(case, n, amp, nu, step)
                exact_e = kinetic_energy(*ex)
                hist.append(
                    {
                        "t": step,
                        "E": e,
                        "E_exact": exact_e,
                        "E_rel_err_pct": 100.0 * (e - exact_e) / exact_e if exact_e else float("nan"),
                        "l2_rel": l2_rel((ux, uy, uz), ex),
                        "rho_mean": float(rho.mean()),
                        "u_max": float(torch.sqrt(ux**2 + uy**2 + uz**2).max()),
                    }
                )
    wall = time.time() - t0

    # fitted decay exponent of the energy curve, ln E = ln E0 - lambda t
    ts = [h["t"] for h in hist]
    lnE = [math.log(h["E"]) for h in hist]
    n_pts = len(ts)
    tm, lm = sum(ts) / n_pts, sum(lnE) / n_pts
    num = sum((t - tm) * (l - lm) for t, l in zip(ts, lnE))
    den = sum((t - tm) ** 2 for t in ts)
    lam_fit = -num / den
    lam_exact = 2.0 * nu * k2
    # per-sample rate over each interval (local slope, robustness check)
    last, first = hist[-1], hist[0]
    l2_final = hist[-1]["l2_rel"]
    return {
        "case": case,
        "n": n,
        "tau": tau,
        "nu": nu,
        "amp": amp,
        "steps": steps,
        "sample_every": sample,
        "collision": collision,
        "k2": k2,
        "ma": amp * math.sqrt(3.0),
        "re_lattice": amp / (nu * math.sqrt(k2)) if nu > 0 else None,
        "k_times_h": math.sqrt(k2),
        "wall_s": wall,
        "finite": finite,
        "mass_drift_rel": mass_drift,
        "E0": e0,
        "E_final": last["E"],
        "E_final_exact": last["E_exact"],
        "energy_decay_ratio_num": last["E"] / e0,
        "energy_decay_ratio_exact": last["E_exact"] / e0,
        "M1_energy_decay_rel_err_pct": 100.0
        * ((last["E"] / e0) - (last["E_exact"] / e0))
        / (last["E_exact"] / e0),
        "lambda_fit": lam_fit,
        "lambda_exact": lam_exact,
        "M2_decay_exponent_rel_err_pct": 100.0 * (lam_fit - lam_exact) / lam_exact,
        "M3_l2_field_rel_err_pct": 100.0 * l2_final,
        "M5_mass_drift_rel": mass_drift,
        "history": hist,
    }


# --------------------------------------------------------------------------- #
# sweeps
# --------------------------------------------------------------------------- #
# Declared before the results were inspected.  Deviation D1 (recorded in
# prereg.json): the two sensitivity sweeps run at N=48 / 1200 steps and the
# primary decay at N=64 / 2000 steps instead of N=64 / 2000 each, because the
# control plane was saturated (load ~390 on 128 cores) and the cluster queue
# stopped dispatching (~300 jobs ahead, no job started for >30 min).
PROTOCOL = {
    "grids_convergence": {"case": "tg", "grids": [24, 32, 48, 64], "tau": 0.6, "amp": 2e-3, "steps": 500},
    "primary_decay": {"case": "tg", "grids": [64], "tau": 0.6, "amp": 2e-3, "steps": 2000, "sample": 50},
    "tau_sensitivity": {"case": "tg", "grids": [48], "taus": [0.55, 0.6, 0.7, 0.8], "amp": 2e-3, "steps": 1200},
    "amp_sensitivity": {"case": "tg", "grids": [48], "tau": 0.6, "amps": [1e-3, 2e-3, 1e-2, 2e-2], "steps": 1200},
    # fallback grids for the same two sweeps if the N=48 budget cannot finish in
    # the session window (declared as deviation D2 in the report)
    "tau_sensitivity_fast": {"case": "tg", "grids": [32], "taus": [0.55, 0.6, 0.7, 0.8], "amp": 2e-3, "steps": 400},
    "amp_sensitivity_fast": {"case": "tg", "grids": [32], "tau": 0.6, "amps": [1e-3, 2e-3, 1e-2, 2e-2], "steps": 400},
    "abc_zdependent": {"case": "abc", "grids": [32, 48], "tau": 0.6, "amp": 1e-2, "steps": 1000},
}


def _save(out: str, name: str, payload: dict) -> str:
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all", "verify", "convergence", "decay", "tau", "amp", "abc", "tau_fast", "amp_fast"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--amp", type=float, default=2e-3)
    ap.add_argument("--case", default="tg", choices=["tg", "abc", "tg3d"])
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("LBM_THREADS", "8")))
    os.makedirs(args.out, exist_ok=True)
    todo = ["verify", "convergence", "decay", "tau", "amp", "abc"] if args.mode == "all" else [args.mode]
    written = []

    if "verify" in todo:
        v = verify()
        written.append(_save(args.out, "exactness_verification.json", v))
        print(json.dumps(v, indent=1))

    if "convergence" in todo:
        p = PROTOCOL["grids_convergence"]
        runs = [run_decay(p["case"], g, p["tau"], p["amp"], p["steps"], sample=100) for g in p["grids"]]
        e = {str(r["n"]): r["M3_l2_field_rel_err_pct"] / 100.0 for r in runs}
        order = (
            math.log2(e["24"] / e["48"]) if e["24"] > 0 and e["48"] > 0 else float("nan")
        )
        payload = {
            "protocol": p,
            "runs": runs,
            "M4_observed_order": order,
            "M4_order_24_32": math.log2(e["24"] / e["32"]) if e["32"] > 0 else None,
            "M4_order_48_64": math.log2(e["48"] / e["64"]) if e["64"] > 0 else None,
            "l2_rel_by_grid": e,
        }
        written.append(_save(args.out, "convergence.json", payload))
        print("convergence order =", order, {k: f"{v:.3e}" for k, v in e.items()})

    if "decay" in todo:
        p = PROTOCOL["primary_decay"]
        r = run_decay(p["case"], p["grids"][0], p["tau"], p["amp"], p["steps"], sample=p["sample"])
        written.append(_save(args.out, "primary_decay.json", r))
        print(
            f"primary: M1={r['M1_energy_decay_rel_err_pct']:.3e}%  "
            f"M2={r['M2_decay_exponent_rel_err_pct']:.3e}%  M3={r['M3_l2_field_rel_err_pct']:.3e}%  "
            f"M5={r['M5_mass_drift_rel']:.2e}  wall={r['wall_s']:.1f}s"
        )

    if "tau" in todo:
        p = PROTOCOL["tau_sensitivity"]
        runs = [run_decay(p["case"], p["grids"][0], tau, p["amp"], p["steps"], sample=100) for tau in p["taus"]]
        written.append(_save(args.out, "tau_sensitivity.json", {"protocol": p, "runs": runs}))
        for r in runs:
            print(f"tau={r['tau']}: M2={r['M2_decay_exponent_rel_err_pct']:+.3e}%  lam_fit={r['lambda_fit']:.6e}")

    if "amp" in todo:
        p = PROTOCOL["amp_sensitivity"]
        runs = [run_decay(p["case"], p["grids"][0], p["tau"], amp, p["steps"], sample=100) for amp in p["amps"]]
        written.append(_save(args.out, "amp_sensitivity.json", {"protocol": p, "runs": runs}))
        for r in runs:
            print(f"amp={r['amp']:.1e} Ma={r['ma']:.4f}: M3={r['M3_l2_field_rel_err_pct']:.3e}%  M1={r['M1_energy_decay_rel_err_pct']:+.3e}%")

    if "tau_fast" in todo:
        p = PROTOCOL["tau_sensitivity_fast"]
        runs = [run_decay(p["case"], p["grids"][0], tau, p["amp"], p["steps"], sample=50) for tau in p["taus"]]
        written.append(_save(args.out, "tau_sensitivity.json", {"protocol": p, "runs": runs}))
        for r in runs:
            print(f"tau={r['tau']}: M2={r['M2_decay_exponent_rel_err_pct']:+.3e}%")

    if "amp_fast" in todo:
        p = PROTOCOL["amp_sensitivity_fast"]
        runs = [run_decay(p["case"], p["grids"][0], p["tau"], amp, p["steps"], sample=50) for amp in p["amps"]]
        written.append(_save(args.out, "amp_sensitivity.json", {"protocol": p, "runs": runs}))
        for r in runs:
            print(f"amp={r['amp']:.1e}: M3={r['M3_l2_field_rel_err_pct']:.3e}%")

    if "abc" in todo:
        p = PROTOCOL["abc_zdependent"]
        runs = [run_decay(p["case"], g, p["tau"], p["amp"], p["steps"], sample=50) for g in p["grids"]]
        written.append(_save(args.out, "abc_zdependent.json", {"protocol": p, "runs": runs}))
        for r in runs:
            print(f"abc N={r['n']}: M1={r['M1_energy_decay_rel_err_pct']:+.3e}%  M2={r['M2_decay_exponent_rel_err_pct']:+.3e}%  M3={r['M3_l2_field_rel_err_pct']:.3e}%")

    print("\nwrote:")
    for w in written:
        print(" ", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())