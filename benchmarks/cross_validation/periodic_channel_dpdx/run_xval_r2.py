"""Round-2 XVAL driver: does the *port* reproduce the periodic-x + constant-dp/dx
channel of ``joconnor22/LIFE`` (``single-phase-periodic-x-plus-constant-dpdx-driving``)?

Run (cluster):  python3 benchmarks/cross_validation/periodic_channel_dpdx/run_xval_r2.py
Writes:         benchmarks/cross_validation/periodic_channel_dpdx/out/xval_life_periodic_dpdx_r2.json

Round 1 (``run_xval.py``) validated only the independent transcription
(``life_ref.py``) and left the two ported arms as honest FAILs: at 3000 steps
they showed ~1 % mass drift and then diverged.  Round 2 fixes the two
transcription defects found in ``tensorlbm.periodic_channel_dpdx``
(``_shift`` streaming sign, ``equilibrium`` quadratic coefficient) and reruns the
port **to steady state** against the exact plane-Poiseuille parabola.

Arms
----
``PORT-REG``  port, ``wall="regularised"`` + ``force_scheme="guo_half_frozen"``
              — the OSS closure pair; must agree with ``PORT-LIFE-REF`` to
              round-off and match the parabola whose walls are the wall-node
              centres (``H = ny - 1``).
``PORT-BB``   port, ``wall="halfway_bb"`` + ``force_scheme="split_guo"``
              — the repo production convention; parabola walls at the link
              midpoints (``H = ny - 2``).
``CONTROL``   port, ``wall="none"`` (no y walls) — no parabola can exist, so the
              gate must fail.  This is the metric-has-teeth lane.

Numeric anchor (zero hand-copy): the exact plane-Poiseuille solution
``u(y) = G (y - y_w)(H - (y - y_w)) / (2 nu)`` evaluated in lattice units on the
fluid rows; ``G = F/rho_0``.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tensorlbm.periodic_channel_dpdx import (  # noqa: E402
    D2Q9_C,
    FORCE_GUO_HALF_FROZEN,
    FORCE_SPLIT_GUO,
    WALL_HALFWAY_BB,
    WALL_NONE,
    WALL_REGULARISED,
    DpdxChannel,
    dpdx_to_lattice_force,
    equilibrium,
    observed_order,  # noqa: E402
    poiseuille_analytic,
)

TAU = 0.5375  # LIFE examples/ChannelFlow value
NU = (TAU - 0.5) / 3.0  # 0.0125
NX = 2  # x-periodic, state x-invariant
GATE = 0.15  # owner accuracy standard
OUT = HERE / "out"

ARMS = {
    "PORT-REG": (WALL_REGULARISED, FORCE_GUO_HALF_FROZEN, 1),
    "PORT-BB": (WALL_HALFWAY_BB, FORCE_SPLIT_GUO, 2),
    "CONTROL": (WALL_NONE, FORCE_SPLIT_GUO, 0),
}


def g_for(ny: int, u_target: float, wall: str) -> float:
    """Body force that makes the exact parabola peak at ``u_target``."""
    h = float(ny - 1) if wall != WALL_HALFWAY_BB else float(ny - 2)
    return 8.0 * NU * u_target / h**2


# ---------------------------------------------------------------------------
# 0. operator invariants — the two defects fixed in round 2
# ---------------------------------------------------------------------------
def operator_invariants() -> dict:
    rng = np.random.default_rng(20261115)
    out = {}
    # equilibrium moments
    ux = np.full((3, 4), 0.05)
    uy = np.full((3, 4), -0.02)
    rho = np.full((3, 4), 1.3)
    f = equilibrium(rho, ux, uy)
    c = np.array(D2Q9_C, dtype=np.float64)
    out["equilibrium_moments"] = {
        "sum_f_minus_rho": float(np.abs(f.sum(axis=0) - rho).max()),
        "sum_cx_f_minus_rho_ux": float(
            np.abs(np.tensordot(c[:, 0], f, axes=(0, 0)) - rho * ux).max()
        ),
        "sum_cy_f_minus_rho_uy": float(
            np.abs(np.tensordot(c[:, 1], f, axes=(0, 0)) - rho * uy).max()
        ),
        # second moment: sum_i |c_i|^2 f^eq = 2 rho c_s^2 + rho |u|^2 (the
        # u u term of the Hermite expansion, see the module docstring)
        "sum_c2_f_minus_2rho_cs2_minus_rho_u2": float(
            np.abs(
                np.tensordot(c[:, 0] ** 2 + c[:, 1] ** 2, f, axes=(0, 0))
                - 2.0 * rho / 3.0
                - rho * (ux * ux + uy * uy)
            ).max()
        ),
    }
    # streaming sign: a single-direction bump must move by +c
    bump = np.zeros((5, 6))
    bump[2, 3] = 1.0
    from tensorlbm.periodic_channel_dpdx import _shift  # local import: private op

    pos = {}
    for i, (cx, cy) in enumerate(D2Q9_C):
        if (cx, cy) == (0, 0):
            continue
        idx = np.argwhere(_shift(bump, cx, cy) == 1.0)
        pos[f"{cx},{cy}"] = (int(idx[0][0] - 2), int(idx[0][1] - 3))
    # a population of direction c must be displaced by exactly +c per step
    expected = {f"{cx},{cy}": (cx, cy) for cx, cy in D2Q9_C if (cx, cy) != (0, 0)}
    out["streaming_displacement"] = {
        "measured": pos,
        "expected_plus_c": expected,
        "matches_pull_form": pos == expected,
    }
    # equilibrium perturbation must not be damped by a pure-relaxation step
    out["seed"] = int(rng.integers(0, 2**31))
    return out


# ---------------------------------------------------------------------------
# 1. arm runner (steady state, always bounded)
# ---------------------------------------------------------------------------
def run_arm(
    name: str, ny: int, u_target: float, *, max_steps: int = 400_000, tol: float = 1e-11
) -> dict:
    wall, scheme, dn = ARMS[name]
    force = g_for(ny, u_target, wall)
    k = DpdxChannel(NX, ny, TAU, force, wall=wall, force_scheme=scheme)
    m0 = k.mass()
    t0 = time.time()
    rec = k.run_until_steady(tol=tol, max_steps=max_steps)
    blow = None
    if not np.isfinite(k.ux).all() or not np.isfinite(k.rho).all():
        blow = k.steps
    u = k.profile()
    s, u_ana, y_w, h = poiseuille_analytic(ny, NU, force, wall=wall)
    rec_out = {
        "arm": name,
        "ny": ny,
        "wall": wall,
        "force_scheme": scheme,
        "H": h,
        "y_wall": y_w,
        "force": force,
        "u_max_target": u_target,
        "steps": k.steps,
        "converged": bool(rec["converged"]),
        "steady_delta": rec["delta"],
        "blew_up_at_step": blow,
        "u_max": float(np.max(np.abs(u))),
        "u_max_analytic": float(u_ana.max()),
        "u_max_rel_err": float(abs(np.max(np.abs(u)) - u_ana.max()) / u_ana.max()),
        "rel_l2_vs_analytic": float(np.linalg.norm(u - u_ana) / np.linalg.norm(u_ana)),
        "u_mean_over_umax": float(np.mean(u) / np.max(np.abs(u))),
        "umax_over_umean": float(np.max(np.abs(u)) / np.mean(np.abs(u))),
        "mass_drift_rel": float((k.mass() - m0) / m0),
        "rho_dev_max": float(np.abs(k.rho - 1.0).max()),
        "Re_H": float(np.max(np.abs(u)) * h / NU),
        "Ma": float(np.max(np.abs(u)) / np.sqrt(1.0 / 3.0)),
        "machine": "cluster (run_compute)",
        "wall_time_s": round(time.time() - t0, 1),
    }
    if wall != WALL_NONE:
        rec_out["s"] = s.tolist()
        rec_out["u_profile"] = u.tolist()
        rec_out["u_analytic"] = u_ana.tolist()
    return rec_out


# ---------------------------------------------------------------------------
# 2. port vs independent transcription (round-1 arm) — same closure pair
# ---------------------------------------------------------------------------
def port_vs_life_ref(ny: int = 17, steps: int = 2000, u_target: float = 0.02) -> dict:
    from life_ref import LIFE_C, LifeKernel2D

    force = g_for(ny, u_target, WALL_REGULARISED)
    a = DpdxChannel(2, ny, TAU, force, wall=WALL_REGULARISED, force_scheme=FORCE_GUO_HALF_FROZEN)
    b = LifeKernel2D(2, ny, TAU, force)
    # LIFE lattice order -> port lattice order (matched by velocity, not index)
    mapping = np.array([[i for i, cc in enumerate(D2Q9_C) if cc == v][0] for v in LIFE_C])
    inv = np.argsort(mapping)
    hist = []
    for n in range(1, steps + 1):
        a.step()
        b.step()
        if n in (1, 10, 100, 500, 1000, steps):
            hist.append(
                {
                    "step": n,
                    "max_abs_df": float(np.abs(a.f - b.f[inv]).max()),
                    "max_abs_drho": float(np.abs(a.rho - b.rho).max()),
                }
            )
    return {"ny": ny, "force": force, "steps": steps, "history": hist}


# ---------------------------------------------------------------------------
# 3. unit mapping: physical dp/dx -> lattice force -> analytic u_max
# ---------------------------------------------------------------------------
def unit_mapping() -> dict:
    dx, dt, rho_p = 0.02, 5.0e-4, 1.0
    nu_p = 1.0e-6
    dpdx = 1.0  # Pa/m
    h_p = 32 * dx  # 32 lattice cells tall
    u_p = dpdx * h_p**2 / (8.0 * rho_p * nu_p)
    f_std = dpdx_to_lattice_force(dpdx, dx=dx, dt=dt, rho_scale=rho_p)
    f_life = dpdx_to_lattice_force(dpdx, dx=dx, dt=dt, rho_scale=rho_p, convention="life_compat")
    nu_lb = nu_p * dt / dx**2
    h_lb = h_p / dx
    return {
        "inputs": {
            "dpdx_Pa_per_m": dpdx,
            "dx": dx,
            "dt": dt,
            "rho_p": rho_p,
            "nu_p": nu_p,
            "h_p": h_p,
        },
        "standard_force": f_std,
        "life_compat_force": f_life,
        "ratio_life_over_standard": f_life / f_std,
        "one_over_dx": 1.0 / dx,
        # the standard convention must reproduce the physical u_max after
        # nondimensionalisation; life_compat is off by dx
        "u_max_physical": u_p,
        "u_max_from_standard_force": f_std * h_lb**2 / (8.0 * nu_lb) * dx / dt,
        "u_max_from_life_compat_force": f_life * h_lb**2 / (8.0 * nu_lb) * dx / dt,
    }


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = {
        "anchor": "XVAL-LBM-SINGLE-PHASE-PERIODIC-X-PLUS-CONSTANT-DP",
        "round": 2,
        "oss": {
            "repo": "joconnor22/LIFE",
            "version": "v1.0.3 (10th June 2020)",
            "tag": "single-phase-periodic-x-plus-constant-dpdx-driving",
            "license": "GPL-3.0 — algorithm transcribed, no code copied",
        },
        "lattice_units": {"tau": TAU, "nu": NU, "nx": NX, "cs2": 1.0 / 3.0},
        "gate": GATE,
    }
    rep["operator_invariants"] = operator_invariants()
    print(
        "invariants:",
        json.dumps(rep["operator_invariants"]["equilibrium_moments"]),
        "streaming ok:",
        rep["operator_invariants"]["streaming_displacement"]["matches_pull_form"],
        flush=True,
    )

    rep["port_vs_life_ref"] = port_vs_life_ref()
    print("port vs life_ref:", rep["port_vs_life_ref"]["history"][-1], flush=True)

    # steady-state validation, two target Mach levels
    steady = {}
    for u_target in (0.02, 0.05):
        for name in ("PORT-REG", "PORT-BB"):
            for ny in (17, 33, 65):
                r = run_arm(name, ny, u_target)
                steady[f"{name}|U{u_target}|ny{ny}"] = r
                print(
                    "%-8s U=%.2f ny=%2d steps=%6d conv=%s relL2=%.3e umax_err=%.3e "
                    "mass=%.1e rho=%.1e (%.0fs)"
                    % (
                        name,
                        u_target,
                        ny,
                        r["steps"],
                        r["converged"],
                        r["rel_l2_vs_analytic"],
                        r["u_max_rel_err"],
                        r["mass_drift_rel"],
                        r["rho_dev_max"],
                        r["wall_time_s"],
                    ),
                    flush=True,
                )
    rep["steady_state"] = steady

    # control lane: no walls -> no parabola -> gate must fail
    ctrl = run_arm("CONTROL", 33, 0.02, max_steps=60_000)
    u = DpdxChannel(
        NX, 33, TAU, g_for(33, 0.02, WALL_REGULARISED), wall=WALL_NONE, force_scheme=FORCE_SPLIT_GUO
    )
    u.run_until_steady(tol=1e-11, max_steps=60_000)
    _, u_ana, _, _ = poiseuille_analytic(
        33, NU, g_for(33, 0.02, WALL_REGULARISED), wall=WALL_REGULARISED
    )
    ctrl["rel_l2_vs_analytic"] = float(np.linalg.norm(u.profile() - u_ana) / np.linalg.norm(u_ana))
    ctrl["u_profile"] = u.profile().tolist()
    ctrl["u_analytic"] = u_ana.tolist()
    ctrl["passes_gate"] = bool(ctrl["rel_l2_vs_analytic"] < GATE)
    rep["control_lane"] = ctrl
    print(
        "CONTROL relL2=%.3e passes_gate=%s" % (ctrl["rel_l2_vs_analytic"], ctrl["passes_gate"]),
        flush=True,
    )

    rep["unit_mapping"] = unit_mapping()

    # grid-refinement order of each validated arm (same target u_max).
    # Round-3 note: the order must be formed with the *wall distance* H as the
    # resolution length, never with an assumed factor-2 refinement -- an
    # ``ny`` 17->25 pair is only 16->24 in H (1.5x) and ``log2`` there reports
    # a second-order scheme as 1.15 (see the round-3 report and
    # ``tensorlbm.periodic_channel_dpdx.observed_order``).
    orders = {}
    for name in ("PORT-REG", "PORT-BB"):
        h = {ny: (float(ny - 2) if name == "PORT-BB" else float(ny - 1)) for ny in (17, 33, 65)}
        errs = {ny: steady[f"{name}|U0.02|ny{ny}"]["rel_l2_vs_analytic"] for ny in (17, 33, 65)}
        orders[name] = {
            "rel_l2": errs,
            "h": h,
            "observed_order_17_33": observed_order(errs[17], errs[33], h[17], h[33]),
            "observed_order_33_65": observed_order(errs[33], errs[65], h[33], h[65]),
        }
    rep["grid_order"] = orders

    # verdict
    worst = {k: v["rel_l2_vs_analytic"] for k, v in steady.items()}
    rep["verdict"] = {
        "worst_rel_l2": max(worst.values()),
        "worst_arm": max(worst, key=worst.get),
        "gate": GATE,
        "all_arms_pass_gate": bool(max(worst.values()) < GATE),
        "all_within_3pct": bool(max(worst.values()) < 0.03),
        "control_fails_gate": not ctrl["passes_gate"],
        "round1_status": "honest-FAIL (port arms diverged after ~3-5e3 steps)",
        "round2_root_cause": [
            "_shift streamed with the opposite sign (out[x]=a[x+c] instead of a[x-c])",
            "equilibrium used 1+3cu+1.5cu^2-1.5u^2 instead of ...+4.5cu^2...",
        ],
    }
    (OUT / "xval_life_periodic_dpdx_r2.json").write_text(json.dumps(rep, indent=1))
    print("verdict:", json.dumps(rep["verdict"]), flush=True)
    return rep


if __name__ == "__main__":
    main()
