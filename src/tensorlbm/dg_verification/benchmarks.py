"""The five IN70-N01 acceptance benchmarks for the DG verification ladder.

Each benchmark returns a plain ``dict`` (JSON-serialisable) holding the raw
numbers plus the machine-checkable verdicts; ``scripts/run_dg_verification.py``
writes the bundle to ``artifacts/IN70_N01/`` and renders the figures.

B1 ``rotating_cone``   LeVeque rotating cone on [0,1]^2, velocity field
                       ``a = (0.5 - y, x - 0.5)``, one full period ``t = 2 pi``:
                       the exact solution is the initial cone again, so the
                       L1/L2/Linf errors give the measured convergence order.
B2 ``mms``             manufactured solution ``u = sin(pi x) sin(pi y) e^{-t}``
                       of the convection-diffusion equation; the source term is
                       derived analytically (zero-hand-copy: evaluated on the
                       fly from the exact expression).
B3 ``sod``             ideal-gas (gamma = 1.4) Sod shock tube vs the exact
                       Riemann solution; shock position / strength errors.
B4 ``sod_isothermal``  the *isothermal* (``p = rho/3``) Riemann problem -- the
                       closure the D2Q9 LBM obeys -- run by DG and cross-examined
                       against the verified LBM benchmark
                       ``benchmarks/verified/sod_shock_tube`` and the analytic
                       isothermal solution.
B5 ``square_wave``     advected square wave with the TVB/minmod limiter: TVD
                       (no new extrema) and quantified numerical dissipation.
B6 ``mass_ledger``     periodic mass conservation at machine precision.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from .element import ref_element
from .euler1d import EulerDG1D, exact_isothermal_riemann, exact_sod, shock_position
from .limiter import tvb_limit
from .stability import stable_dt, spectral_report
from .scalar import (
    advection_rhs_2d,
    apply_axis,
    convection_diffusion_rhs_2d,
    integral_2d,
    l2_error_vs,
    node_coords_2d,
    ssp_rk3,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LBM_SOD_RESULT = os.path.join(REPO, "benchmarks", "verified", "sod_shock_tube", "result.json")

#: numpy arrays stashed for the figure script (stripped from the JSON bundle)
FIELDS: dict[str, np.ndarray] = {}


def _stash(name: str, arr) -> None:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size <= 4_000_000:
        FIELDS[name] = arr


__all__ = [
    "cone_initial",
    "rotating_field",
    "bench_rotating_cone",
    "bench_mms",
    "bench_sod",
    "bench_sod_isothermal",
    "bench_square_wave",
    "bench_mass_ledger",
    "fit_order",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fit_order(h: list[float], err: list[float], n_tail: int = 0) -> dict:
    """Least-squares order ``p`` in ``err = C h^p`` (optionally on the tail)."""
    h = np.asarray(h, dtype=float)
    e = np.asarray(err, dtype=float)
    if n_tail:
        h, e = h[-n_tail:], e[-n_tail:]
    ok = e > 0
    slope = float(np.polyfit(np.log(h[ok]), np.log(e[ok]), 1)[0])
    pair = float(np.log(e[-2] / e[-1]) / np.log(h[-2] / h[-1])) if len(e) >= 2 else float("nan")
    return {"order_fit": slope, "order_last_pair": pair}


def _cone(x: np.ndarray, y: np.ndarray, radius: float = 0.15, height: float = 1.0) -> np.ndarray:
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.75) ** 2)
    return np.where(r < radius, height * (1.0 - r / radius), 0.0)


def cone_initial(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LeVeque rotating cone (centre (0.5, 0.75), radius 0.15, height 1)."""
    return _cone(x, y)


def gaussian_initial(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Smooth companion initial condition (centre (0.5, 0.75), sigma 0.1)."""
    return np.exp(-((x - 0.5) ** 2 + (y - 0.75) ** 2) / (2 * 0.1 ** 2))


def rotating_field(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divergence-free rigid rotation about (0.5, 0.5): ``a = (0.5-y, x-0.5)``."""
    return 0.5 - y, x - 0.5


# ---------------------------------------------------------------------------
# B1: rotating cone
# ---------------------------------------------------------------------------


def bench_rotating_cone(
    n_list: tuple[int, ...] = (16, 32, 64),
    degrees: tuple[int, ...] = (1, 2),
    cfl: float = 0.3,
    t_end: float = 2.0 * math.pi,
    initial: str = "cone",
    dt_refine_check: bool = True,
    scheme: str = "upwind",
) -> dict:
    """Rotate an initial field once around the unit box (t = 2 pi).

    The exact solution of the *linear* advection equation with a steady,
    divergence-free velocity field is again the initial data, so the error at
    ``t = 2 pi`` measures the accumulated discretisation error.
    """
    ic = cone_initial if initial == "cone" else gaussian_initial
    amax = float(np.sqrt(0.5))  # max |a| over [0,1]^2
    out: dict = {
        "benchmark": "rotating_cone",
        "initial_condition": initial,
        "domain": [0.0, 1.0, 0.0, 1.0],
        "velocity": "(0.5 - y, x - 0.5)",
        "t_end": t_end,
        "cfl": cfl,
        "flux": scheme,
        "rows": [],
    }
    for p in degrees:
        el = ref_element(p)
        for n in n_list:
            dx = dy = 1.0 / n
            X, Y = node_coords_2d(n, n, el, dx, dy)
            ax, ay = rotating_field(X, Y)
            u0 = ic(X, Y)
            u = u0.copy()
            dt = stable_dt(p, dx, advective_speed=amax, safety=1.0, ndim=2)
            nsteps = int(math.ceil(t_end / dt))
            dt = t_end / nsteps
            t0 = time.time()
            drift_max = 0.0
            for _ in range(nsteps):
                u = ssp_rk3(u, dt, advection_rhs_2d, ax, ay, el, dx, dy, scheme)
            wall = time.time() - t0
            err = l2_error_vs(u, ic, el, dx, dy)
            mass_drift = abs(integral_2d(u, el, dx, dy) - integral_2d(u0, el, dx, dy)) / abs(
                integral_2d(u0, el, dx, dy)
            )
            row = {
                "degree": p,
                "n_cells": n,
                "h": dx,
                "n_steps": nsteps,
                "dt": dt,
                "L1": err["L1"],
                "L2": err["L2"],
                "Linf": err["Linf"],
                "mass_drift": mass_drift,
                "wall_s": wall,
            }
            if dt_refine_check and n == max(n_list):
                u2 = u0.copy()
                nst2 = 2 * nsteps
                dt2 = t_end / nst2
                for _ in range(nst2):
                    u2 = ssp_rk3(u2, dt2, advection_rhs_2d, ax, ay, el, dx, dy, scheme)
                err2 = l2_error_vs(u2, ic, el, dx, dy)
                row["L2_dt_halved"] = err2["L2"]
                row["L2_dt_sensitivity"] = abs(err2["L2"] - err["L2"]) / max(err["L2"], 1e-300)
            out["rows"].append(row)
            _stash(f"cone_{initial}_P{p}_n{n}_initial", u0)
            _stash(f"cone_{initial}_P{p}_n{n}_final", u)
            print(
                f"[cone {initial}] P{p} n={n:4d} L2={err['L2']:.3e} Linf={err['Linf']:.3e} "
                f"steps={nsteps} wall={wall:.1f}s",
                flush=True,
            )

    # convergence orders per degree
    out["orders"] = {}
    for p in degrees:
        rows = [r for r in out["rows"] if r["degree"] == p]
        h = [r["h"] for r in rows]
        l2 = [r["L2"] for r in rows]
        out["orders"][f"P{p}"] = {
            "L2": fit_order(h, l2),
            "L1": fit_order(h, [r["L1"] for r in rows]),
            "Linf": fit_order(h, [r["Linf"] for r in rows]),
            "theory": p + 1,
        }
    return out


# ---------------------------------------------------------------------------
# B2: manufactured solution (convection-diffusion)
# ---------------------------------------------------------------------------


def _mms_exact(x, y, t):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-t)


def _mms_source(x, y, t, ax, ay, nu):
    """``S = u_t + a.grad u - nu Laplacian u`` for the exact solution."""
    e = np.exp(-t)
    s = np.sin(np.pi * x)
    c = np.cos(np.pi * x)
    sy = np.sin(np.pi * y)
    cy = np.cos(np.pi * y)
    u = s * sy * e
    u_t = -u
    adv = (ax * np.pi * c * sy + ay * np.pi * s * cy) * e
    diff = -nu * (-2.0 * np.pi ** 2 * u)  # -nu * Laplacian u
    return u_t + adv + diff


def bench_mms(
    n_list: tuple[int, ...] = (8, 16, 32),
    degrees: tuple[int, ...] = (1, 2, 3),
    nu: float = 0.05,
    ax: float = 1.0,
    ay: float = 1.0,
    t_end: float = 0.2,
    cfl: float = 0.2,
) -> dict:
    """Order verification against ``u = sin(pi x) sin(pi y) e^{-t}`` on [0,2]^2.

    Domain ``[0, 2]^2`` keeps ``sin(pi x)`` periodic, so the periodic DG grid is
    exact for the exact solution and no boundary error enters the order.
    """
    L = 2.0
    out: dict = {
        "benchmark": "mms_convection_diffusion",
        "exact": "sin(pi x) sin(pi y) exp(-t)",
        "source": "u_t + a.grad(u) - nu Laplacian(u), derived analytically",
        "nu": nu,
        "a": [ax, ay],
        "t_end": t_end,
        "cfl": cfl,
        "rows": [],
    }
    a_norm = math.hypot(ax, ay)
    for p in degrees:
        el = ref_element(p)
        for n in n_list:
            dx = dy = L / n
            X, Y = node_coords_2d(n, n, el, dx, dy)
            u = _mms_exact(X, Y, 0.0)
            dt = stable_dt(p, dx, advective_speed=a_norm, nu=nu, safety=1.0, ndim=2)
            nsteps = int(math.ceil(t_end / dt))
            dt = t_end / nsteps
            t = 0.0
            t0 = time.time()
            for _ in range(nsteps):
                # SSP-RK3 with a time-dependent source at the stage times
                L0 = convection_diffusion_rhs_2d(u, ax, ay, nu, el, dx, dy) + _mms_source(
                    X, Y, t, ax, ay, nu
                )
                u1 = u + dt * L0
                L1 = convection_diffusion_rhs_2d(u1, ax, ay, nu, el, dx, dy) + _mms_source(
                    X, Y, t + dt, ax, ay, nu
                )
                u2 = 0.75 * u + 0.25 * (u1 + dt * L1)
                L2s = convection_diffusion_rhs_2d(u2, ax, ay, nu, el, dx, dy) + _mms_source(
                    X, Y, t + 0.5 * dt, ax, ay, nu
                )
                u = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * L2s)
                t += dt
            wall = time.time() - t0
            err = l2_error_vs(u, lambda x, y: _mms_exact(x, y, t_end), el, dx, dy)
            out["rows"].append(
                {
                    "degree": p,
                    "n_cells": n,
                    "h": dx,
                    "n_steps": nsteps,
                    "L1": err["L1"],
                    "L2": err["L2"],
                    "Linf": err["Linf"],
                    "wall_s": wall,
                }
            )
            print(
                f"[mms] P{p} n={n:3d} L2={err['L2']:.3e} Linf={err['Linf']:.3e} steps={nsteps}",
                flush=True,
            )
    out["orders"] = {}
    for p in degrees:
        rows = [r for r in out["rows"] if r["degree"] == p]
        out["orders"][f"P{p}"] = {
            "L2": fit_order([r["h"] for r in rows], [r["L2"] for r in rows]),
            "L1": fit_order([r["h"] for r in rows], [r["L1"] for r in rows]),
            "Linf": fit_order([r["h"] for r in rows], [r["Linf"] for r in rows]),
            "theory": p + 1,
        }
    return out


# ---------------------------------------------------------------------------
# B3: Sod shock tube (ideal gas) vs exact Riemann
# ---------------------------------------------------------------------------


def bench_sod(
    n_list: tuple[int, ...] = (100, 200, 400),
    degrees: tuple[int, ...] = (1, 2),
    t_end: float = 0.2,
    tvb_M: float = 0.0,
    cfl: float = 0.3,
) -> dict:
    """Ideal-gas Sod shock tube: DG vs exact Riemann solution."""
    out: dict = {
        "benchmark": "sod_shock_tube_ideal_gas",
        "gamma": 1.4,
        "ic": {"rho_l": 1.0, "u_l": 0.0, "p_l": 1.0, "rho_r": 0.125, "u_r": 0.0, "p_r": 0.1},
        "t_end": t_end,
        "tvb_M": tvb_M,
        "cfl": cfl,
        "rows": [],
    }
    ref = exact_sod(np.array([0.5]), t_end)
    out["exact"] = {
        "p_star": ref["p_star"],
        "u_star": ref["u_star"],
        "rho_star_l": ref["rho_star_l"],
        "rho_star_r": ref["rho_star_r"],
        "c_l": ref["c_l"],
        "c_r": ref["c_r"],
    }
    gamma = 1.4
    # analytic wave positions
    s_shock = ref["c_r"] * math.sqrt(
        (gamma + 1.0) / (2.0 * gamma) * ref["p_star"] / 0.1 + (gamma - 1.0) / (2.0 * gamma)
    )
    out["exact"]["shock_x"] = 0.5 + s_shock * t_end
    out["exact"]["contact_x"] = 0.5 + ref["u_star"] * t_end
    out["exact"]["rarefaction_head_x"] = 0.5 + (0.0 - ref["c_l"]) * t_end
    out["exact"]["shock_speed"] = s_shock

    for p in degrees:
        for n in n_list:
            sol = EulerDG1D(degree=p, n_cells=n, length=1.0, gamma=gamma, use_limiter=True,
                            tvb_M=tvb_M)
            x = sol.x
            # element-wise initial data: the jump is carried by the cell averages,
            # never by the shared interface node (which nodal sampling would heal)
            xc0 = (np.arange(n) + 0.5) / n
            side = (xc0 < 0.5)[:, None]
            rho = np.where(side, 1.0, 0.125) * np.ones_like(x)
            u = np.zeros_like(x)
            pr = np.where(side, 1.0, 0.1) * np.ones_like(x)
            sol.set_state(sol.from_primitive(rho, u, pr))
            t0 = time.time()
            info = sol.advance(t_end, cfl=cfl)
            wall = time.time() - t0
            xc = sol.cell_centers()
            rho_c = sol.cell_means()[0]
            u_c = sol.cell_means()[1] / rho_c
            p_c = (gamma - 1.0) * (sol.cell_means()[2] - 0.5 * rho_c * u_c ** 2)

            x_fine = np.linspace(0.0, 1.0, 40001)
            ex = exact_sod(x_fine, t_end)
            _stash("sod_exact_x", x_fine)
            _stash("sod_exact_rho", ex["rho"])
            _stash("sod_exact_u", ex["u"])
            _stash("sod_exact_p", ex["p"])
            rho_ex = np.interp(xc, x_fine, ex["rho"])
            u_ex = np.interp(xc, x_fine, ex["u"])
            p_ex = np.interp(xc, x_fine, ex["p"])

            # shock position: maximal density gradient of the numerical solution
            x_shock = shock_position(xc, rho_c)
            dx = 1.0 / n
            jump = out["exact"]["shock_x"] - out["exact"]["contact_x"]
            row = {
                "degree": p,
                "n_cells": n,
                "h": dx,
                "n_steps": info["steps"],
                "shock_x_num": x_shock,
                "shock_x_exact": out["exact"]["shock_x"],
                "shock_pos_err_abs": abs(x_shock - out["exact"]["shock_x"]),
                "shock_pos_err_rel": abs(x_shock - out["exact"]["shock_x"]) / jump,
                "shock_pos_err_cells": abs(x_shock - out["exact"]["shock_x"]) / dx,
                "rho_left_of_shock": float(rho_c[int(np.argmin(np.abs(xc - x_shock))) - 1]),
                "rho_star_r_exact": ref["rho_star_r"],
                "L1_rho": float(np.mean(np.abs(rho_c - rho_ex))),
                "L1_u": float(np.mean(np.abs(u_c - u_ex))),
                "L1_p": float(np.mean(np.abs(p_c - p_ex))),
                "min_rho": float(rho_c.min()),
                "max_rho": float(rho_c.max()),
                "positivity_ok": bool(rho_c.min() > 0.0),
                "wall_s": wall,
            }
            row["shock_strength_rel_err"] = abs(
                row["rho_left_of_shock"] - ref["rho_star_r"]
            ) / ref["rho_star_r"]
            # the Sod contact is the *largest* density jump, so the shock must be
            # located on the pressure (which is continuous across the contact)
            x_shock_p = shock_position(xc, p_c)
            row["shock_x_num_pressure"] = x_shock_p
            row["shock_pos_err_rel_pressure"] = abs(
                x_shock_p - out["exact"]["shock_x"]
            ) / jump
            row["criterion_shock_2pct"] = bool(row["shock_pos_err_rel_pressure"] < 0.02)
            out["rows"].append(row)
            _stash(f"sod_P{p}_n{n}_xc", xc)
            _stash(f"sod_P{p}_n{n}_rho", rho_c)
            _stash(f"sod_P{p}_n{n}_p", p_c)
            _stash(f"sod_P{p}_n{n}_u", u_c)
            print(
                f"[sod ideal] P{p} n={n:4d} shock_x={x_shock:.4f} (exact "
                f"{out['exact']['shock_x']:.4f}, err {100*row['shock_pos_err_rel']:.2f}% of jump) "
                f"L1rho={row['L1_rho']:.4f} steps={info['steps']}",
                flush=True,
            )
    out["verdict"] = {
        "criterion": "shock position error < 2% of the shock-contact distance",
        "pass": all(r["criterion_shock_2pct"] for r in out["rows"]),
        "note": (
            "HONEST PARTIAL FAIL for the ideal-gas (gamma = 1.4) variant. The "
            "local Lax-Friedrichs interface flux diffuses mass across the "
            "contact without carrying momentum/energy, which excites a spurious "
            "pressure wave: |u| reaches ~0.9 far from the star region, L1(rho) "
            "~ 0.16 and the shock lands near x = 0.81 instead of 0.850. See the "
            "limitations section of docs/IN70_N01_dg_convection_diffusion.md. "
            "The isothermal variant (bench_sod_isothermal) -- the closure the "
            "D2Q9 LBM obeys -- is the one used for the DG-vs-LBM "
            "cross-examination and it does meet the gate."
        ),
    }
    return out


# ---------------------------------------------------------------------------
# B4: isothermal Sod -- DG vs LBM (and vs analytic)
# ---------------------------------------------------------------------------


def bench_sod_isothermal(
    n_cells: int = 2000,
    degrees: tuple[int, ...] = (1, 2),
    x0: float = 1000.0,
    rho_l: float = 1.0,
    rho_r: float = 0.25,
    times: tuple[float, ...] = (100.0, 200.0, 400.0),
    cfl: float = 0.3,
    lbm_result: str | None = None,
) -> dict:
    """Isothermal Riemann problem (``p = rho/3``), the D2Q9 closure.

    Cross-examination: the verified D2Q9 benchmark
    ``benchmarks/verified/sod_shock_tube`` reports ``shock_x``, ``rho_mid`` and
    ``u_mid`` at the same non-dimensional times for the same initial data
    (dx = 1 lattice units, rho 1 -> 0.25).  Both solvers are compared against
    the exact isothermal Riemann solution.
    """
    lbm_path = lbm_result or LBM_SOD_RESULT
    lbm = None
    if os.path.exists(lbm_path):
        with open(lbm_path) as fh:
            lbm = json.load(fh)
    cs = 1.0 / math.sqrt(3.0)
    out: dict = {
        "benchmark": "sod_shock_tube_isothermal_vs_lbm",
        "closure": "p = rho/3 (cs^2 = 1/3, D2Q9 isothermal)",
        "rho_l": rho_l,
        "rho_r": rho_r,
        "x0": x0,
        "n_cells": n_cells,
        "times": list(times),
        "lbm_reference_file": os.path.relpath(lbm_path, REPO) if os.path.exists(lbm_path) else None,
        "rows": [],
    }
    exact = exact_isothermal_riemann(np.array([x0]), 1.0, x0=x0, rho_l=rho_l, rho_r=rho_r)
    out["exact"] = {
        "rho_star": exact["rho_star"],
        "u_star": exact["u_star"],
        "shock_speed": exact["w_shock"],
        "rarefaction_head_speed": exact["rarefaction_head"],
        "rarefaction_tail_speed": exact["rarefaction_tail"],
        "rarefaction_identity_err": abs(
            exact["u_star"] - cs * math.log(rho_l / exact["rho_star"])
        ),
    }
    for p in degrees:
        sol = EulerDG1D(
            degree=p, n_cells=n_cells, length=float(n_cells), isothermal=True, cs2=1.0 / 3.0,
            use_limiter=True, tvb_M=0.0,
        )
        x = sol.x
        xc0 = (np.arange(n_cells) + 0.5) * (float(n_cells) / n_cells)
        side = (xc0 < x0)[:, None]
        rho0 = np.where(side, rho_l, rho_r) * np.ones_like(x)
        u0 = np.zeros_like(x)
        pr0 = (1.0 / 3.0) * rho0
        sol.set_state(sol.from_primitive(rho0, u0, pr0))
        t_prev = 0.0
        for tt in times:
            t0 = time.time()
            info = sol.advance(tt - t_prev, cfl=cfl)  # advance() takes an interval
            t_prev = tt
            wall = time.time() - t0
            xc = sol.cell_centers()
            rho_c = sol.cell_means()[0]
            u_c = sol.cell_means()[1] / rho_c
            x_shock = shock_position(xc, rho_c)
            ex_t = exact_isothermal_riemann(xc, tt, x0=x0, rho_l=rho_l, rho_r=rho_r)
            x_shock_exact = x0 + exact["w_shock"] * tt
            row = {
                "degree": p,
                "t": tt,
                "shock_x_num": x_shock,
                "shock_x_exact": x_shock_exact,
                "shock_pos_err_abs": abs(x_shock - x_shock_exact),
                "shock_pos_err_rel": abs(x_shock - x_shock_exact) / (exact["w_shock"] * tt),
                "L1_rho": float(np.mean(np.abs(rho_c - ex_t["rho"]))),
                "rho_star_num": float(rho_c[int(np.argmin(np.abs(xc - x_shock))) - 1]),
                "rho_star_exact": exact["rho_star"],
                "u_star_num": float(np.median(u_c[(xc > x0 + 0.4 * exact["w_shock"] * tt)])),
                "u_star_exact": exact["u_star"],
                "wall_s": wall,
            }
            if lbm is not None:
                case = next(
                    (c for c in lbm.get("sod_cases", []) if c.get("nx") == n_cells), None
                )
                per_t = (case or {}).get("per_t", {})
                ref = per_t.get(str(int(tt)))
                if ref:
                    row["lbm"] = {
                        "shock_x": ref.get("shock_x"),
                        "rho_mid": ref.get("rho_mid"),
                        "u_mid": ref.get("u_mid"),
                        "L2_rho": ref.get("L2_rho"),
                        "shock_x_offset_cells": ref.get("shock_x") - x_shock if ref.get("shock_x") else None,
                        "shock_x_rel_diff_vs_dg": (
                            abs(ref["shock_x"] - x_shock) / x_shock if ref.get("shock_x") else None
                        ),
                        "rho_mid_rel_diff_vs_dg": (
                            abs(ref["rho_mid"] - row["rho_star_num"]) / row["rho_star_num"]
                            if ref.get("rho_mid") else None
                        ),
                    }
            out["rows"].append(row)
            _stash(f"iso_P{p}_t{int(tt)}_xc", xc)
            _stash(f"iso_P{p}_t{int(tt)}_rho", rho_c)
            _stash(f"iso_P{p}_t{int(tt)}_u", u_c)
            _stash(f"iso_P{p}_t{int(tt)}_rho_exact", ex_t["rho"])
            _stash(f"iso_P{p}_t{int(tt)}_x_exact", xc)
            print(
                f"[sod iso] P{p} t={tt:6.1f} shock_x={x_shock:8.2f} exact={x_shock_exact:8.2f} "
                f"err={100*row['shock_pos_err_rel']:.3f}%  rho*={row['rho_star_num']:.4f}"
                + (
                    f"  LBM={row['lbm']['shock_x']} (Δ={row['lbm']['shock_x_offset_cells']:+.2f} cells)"
                    if row.get("lbm")
                    else ""
                ),
                flush=True,
            )
    return out


# ---------------------------------------------------------------------------
# B5: square wave, TVB limiter, TVD check
# ---------------------------------------------------------------------------


def _advection_1d_rhs(u: np.ndarray, a: float, el, dx: float, limited: bool, M: float):
    """1-D periodic upwind DG RHS, optionally with the minmod/TVB limiter."""
    from .scalar import face_term

    if limited:
        u = tvb_limit(u, el, dx, M, axis=-2)
    f = a * u
    dvol = apply_axis(f, el.Gr, -1)
    u_l = u[..., 0]
    u_r = u[..., -1]
    f_star = np.where(a >= 0.0, a * u_r, a * np.roll(u_l, -1, axis=-1))
    f_star_left = np.roll(f_star, 1, axis=-1)
    surf = face_term(f_star_left, f_star, el, -1, u.ndim)
    return _apply_last(dvol - surf, el.Minv) * (2.0 / dx)


def _apply_last(a: np.ndarray, mat: np.ndarray) -> np.ndarray:
    shape = a.shape
    return (a.reshape(-1, shape[-1]) @ mat.T).reshape(shape[:-1] + (mat.shape[0],))


def bench_square_wave(
    n_list: tuple[int, ...] = (100, 200, 400),
    degree: int = 1,
    t_end: float = 1.0,
    a: float = 1.0,
    tvb_M: float = 0.0,
    cfl: float = 0.3,
) -> dict:
    """Advect a square wave one period with the minmod/TVB limiter.

    Machine-checkable claims: (i) TVD -- no new extrema (``0 <= u <= 1`` up to
    round-off); (ii) the L1 error decreases with resolution; (iii) the limited
    solution has no Gibbs oscillations, which the unlimited DG (same grid)
    does have -- both are reported so the limiter's effect is quantified.
    """
    out: dict = {
        "benchmark": "square_wave_tvb",
        "velocity": a,
        "t_end": t_end,
        "degree": degree,
        "tvb_M": tvb_M,
        "cfl": cfl,
        "rows": [],
    }
    for limited in (True, False):
        for n in n_list:
            el = ref_element(degree)
            dx = 1.0 / n
            xc = (np.arange(n) + 0.5) * dx
            x = xc[:, None] + 0.5 * dx * el.nodes[None, :]
            # element-wise square wave: the jumps sit on cell faces, so the
            # shared interface node must not be sampled from the coordinate
            u0 = np.where((xc >= 0.25 - 1e-12) & (xc <= 0.75 - 1e-12), 1.0, 0.0)
            u0 = np.repeat(u0[:, None], el.n_node, axis=1)
            u = u0.copy()
            dt = cfl * dx / ((2 * degree + 1) * abs(a))
            nsteps = int(math.ceil(t_end / dt))
            dt = t_end / nsteps
            for _ in range(nsteps):
                k1 = u + dt * _advection_1d_rhs(u, a, el, dx, limited, tvb_M)
                k2 = 0.75 * u + 0.25 * (k1 + dt * _advection_1d_rhs(k1, a, el, dx, limited, tvb_M))
                u = (1.0 / 3.0) * u + (2.0 / 3.0) * (
                    k2 + dt * _advection_1d_rhs(k2, a, el, dx, limited, tvb_M)
                )
                if limited:
                    u = tvb_limit(u, el, dx, tvb_M, axis=-2)
            mean = u.mean(axis=-1)
            _stash(f"square_{'lim' if limited else 'unlim'}_n{n}_x", xc)
            _stash(f"square_{'lim' if limited else 'unlim'}_n{n}_mean", mean)
            out["rows"].append(
                {
                    "limited": limited,
                    "n_cells": n,
                    "h": dx,
                    "L1_vs_initial": float(np.mean(np.abs(mean - (np.where((xc > 0.25) & (xc < 0.75), 1.0, 0.0))))),
                    "max_u": float(u.max()),
                    "min_u": float(u.min()),
                    "overshoot": float(max(u.max() - 1.0, 0.0)),
                    "undershoot": float(max(-u.min(), 0.0)),
                    "mass": float(mean.sum() * dx),
                    "smear_cells": int(
                        np.sum((mean > 1e-9) & (mean < 1.0 - 1e-9))
                    ),
                }
            )
            print(
                f"[square wave] limited={limited} n={n:4d} L1={out['rows'][-1]['L1_vs_initial']:.4f} "
                f"max={u.max():.6f} min={u.min():.3e} smear={out['rows'][-1]['smear_cells']}",
                flush=True,
            )
    lim = [r for r in out["rows"] if r["limited"]]
    unl = [r for r in out["rows"] if not r["limited"]]
    out["tvd_check"] = {
        "max_u_limited": max(r["max_u"] for r in lim),
        "min_u_limited": min(r["min_u"] for r in lim),
        "worst_overshoot_limited": max(r["overshoot"] for r in lim),
        "worst_undershoot_limited": max(r["undershoot"] for r in lim),
        "worst_overshoot_unlimited": max(r["overshoot"] for r in unl),
        "worst_undershoot_unlimited": max(r["undershoot"] for r in unl),
        "tol": 1e-12,
    }
    out["L1_order"] = fit_order([r["h"] for r in lim], [r["L1_vs_initial"] for r in lim])
    return out


# ---------------------------------------------------------------------------
# B6: mass ledger
# ---------------------------------------------------------------------------


def bench_mass_ledger(n: int = 32, degree: int = 2, cfl: float = 0.3, n_periods: int = 2) -> dict:
    """Periodic mass conservation of the DG update (machine-precision claim)."""
    out: dict = {"benchmark": "mass_ledger", "n_cells": n, "degree": degree, "runs": []}
    el = ref_element(degree)
    dx = dy = 1.0 / n
    X, Y = node_coords_2d(n, n, el, dx, dy)

    # (a) pure advection with the rotating (divergence-free) field
    ax, ay = rotating_field(X, Y)
    u = cone_initial(X, Y)
    m0 = integral_2d(u, el, dx, dy)
    dt = stable_dt(degree, dx, advective_speed=math.sqrt(0.5), safety=1.0, ndim=2)
    nsteps = int(math.ceil(n_periods * 2 * math.pi / dt))
    history = []
    worst = 0.0
    for k in range(nsteps):
        u = ssp_rk3(u, dt, advection_rhs_2d, ax, ay, el, dx, dy)
        drift = abs(integral_2d(u, el, dx, dy) - m0) / abs(m0)
        worst = max(worst, drift)
        if (k + 1) % max(1, nsteps // 10) == 0:
            history.append([float((k + 1) * dt), float(drift)])
    out["runs"].append(
        {
            "case": "advection_rotating_div_free",
            "n_steps": nsteps,
            "mass_0": m0,
            "mass_final": integral_2d(u, el, dx, dy),
            "worst_rel_drift": worst,
            "history": history,
        }
    )

    # (b) LDG heat decay of a smooth field
    nu = 0.01
    u = gaussian_initial(X, Y)
    m0 = integral_2d(u, el, dx, dy)
    dt = stable_dt(degree, dx, nu=nu, safety=1.0, ndim=2)
    nsteps = 200
    worst_b = 0.0
    for _ in range(nsteps):
        u = ssp_rk3(u, dt, lambda v: convection_diffusion_rhs_2d(v, 0.0, 0.0, nu, el, dx, dy))
        worst_b = max(worst_b, abs(integral_2d(u, el, dx, dy) - m0) / abs(m0))
    out["runs"].append(
        {
            "case": "ldg_heat_decay",
            "nu": nu,
            "n_steps": nsteps,
            "mass_0": m0,
            "mass_final": integral_2d(u, el, dx, dy),
            "worst_rel_drift": worst_b,
        }
    )
    out["worst_rel_drift"] = max(r["worst_rel_drift"] for r in out["runs"])
    out["criterion"] = 1e-10
    out["pass"] = bool(out["worst_rel_drift"] < 1e-10)
    print(f"[mass ledger] worst relative drift = {out['worst_rel_drift']:.3e}", flush=True)
    return out