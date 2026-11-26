"""XVAL R5 -- why is the regularised wall closure 2x less accurate than bounce back
in a body-force channel, and can it be fixed?

Anchor: ``XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC``.

R4 left two open statements:

* "the regularised closure is ~2x *less* accurate than bounce back on steady
  Poiseuille ... because the projected node drops the genuine third moment of
  the parabolic non-equilibrium part" -- an *unmeasured* attribution;
* corners and ``CENTRAL_MOMENTS`` untransplanted (unchanged scope).

R5 replaces the attribution by a measurement.  Arms (all on the *production*
code path of :mod:`tensorlbm.regularised_bc`, no re-implementation):

``closure`` x ``wall_force``
    ``closure`` in {``regularised`` (shipped: order 2, stress from the
    anti-symmetric completion), ``regularised_neighbour_stress`` (order 2,
    stress extrapolated from the interior neighbour),
    ``regularised3_completed`` (order 3, third moment kept),
    ``regularised3_neighbour`` (order 3, third moment from the neighbour),
    ``bounce_back`` (control)}; ``wall_force`` in {0 (shipped), 1 (the
    boundary row also receives the Guo half-step force, which is what an
    interior node gets)}.

Experiments
    ``ladder``  body-force Poiseuille, ``ny`` in {16,24,32,48}, tau = 0.8
    ``tau``     body-force Poiseuille, ``ny`` = 32, tau in {0.6,0.8,1.0,1.5}
    ``force``   body-force Poiseuille, ``ny`` = 24, F in {1e-5,2e-5,4e-5}:
                tests whether the residual error is proportional to the force
                (a forcing/clamp artefact) or to the third moment (a
                regularisation artefact -- which does not know about F)
    ``osc``     oscillating Stokes layer (F = 0): the arm matrix must not damage
                the unsteady case R4 validated
    ``invariant`` third-moment representability of the order-3 map per lattice

Usage
    PYTHONPATH=src python development/verification/regularised_bc_noneq_stress/run_xval_r5.py --arm ladder
    ... --arm tau | --arm force | --arm osc | --arm invariant | --arm analyse

Each arm merges into ``artifacts_r5/*.json`` so the sweep can be split into
chunks; ``--arm analyse`` folds everything into ``summary_r5.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensorlbm import regularised_bc as rb

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts_r5"

CLOSURES = (
    "regularised",
    "regularised_neighbour_stress",
    "regularised3_completed",
    "regularised3_neighbour",
    "bounce_back",
)

TAU = 0.8
FORCE = 2.0e-5


def _load(name: str) -> dict:
    path = OUT / name
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _dump(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def steady_row(closure: str, ny: int, tau: float, force: float, wall_force: float) -> dict:
    cfg = rb.ChannelConfig(
        case="poiseuille",
        wall_bc=closure,
        ny=ny,
        nx=2,
        tau=tau,
        body_force=force,
        steps=400000,
        tol=1e-11,
        wall_force=wall_force,
    )
    t0 = time.time()
    rec = rb.run_channel(cfg)
    y = np.asarray(rec["y"])
    ux = np.asarray(rec["ux"])
    exact = np.asarray(rec["ux_exact"])
    err = ux - exact
    u_max = float(np.max(np.abs(exact)))
    # structural decomposition: a wall slip is a *constant* offset of the
    # profile (both walls slip), a viscosity/force error is proportional to
    # u_exact; both are measured separately so the attribution is not assumed.
    offset = float(err.mean())
    propto = float((err * exact).sum() / (exact * exact).sum())
    return {
        "closure": closure,
        "ny": ny,
        "tau": tau,
        "nu": cfg.nu,
        "force": force,
        "wall_force": wall_force,
        "steps_run": rec["steps_run"],
        "converged": rec["converged"],
        "final_rel_delta": rec["final_rel_delta"],
        "l2_rel": float(rec["l2_rel"]),
        "linf_abs": float(rec["linf_abs"]),
        "u_max": u_max,
        "offset_over_umax": offset / u_max,
        "offset_over_force": offset / force,
        "propto_over_umax": propto,
        "residual_after_offset_l2": float(
            np.linalg.norm(err - offset) / np.linalg.norm(exact)
        ),
        "fitted_wall_low": rec.get("fitted_wall_low"),
        "fitted_wall_high": rec.get("fitted_wall_high"),
        "wall_origin_declared": cfg.wall_origin,
        "height": cfg.height,
        "mass_drift": rec["mass_drift"],
        "runtime_s": round(time.time() - t0, 2),
    }


def oscillating_row(closure: str, delta_cells: float, tau: float) -> dict:
    nu = (tau - 0.5) / 3.0
    ny = int(round(7.0 * delta_cells)) + 2  # H/delta ~ 7
    period = 2.0 * np.pi * delta_cells**2 / (2.0 * nu)
    omega = 2.0 * np.pi / round(period)
    period = int(round(2.0 * np.pi / omega))
    steps = 5 * period
    cfg = rb.ChannelConfig(
        case="oscillating_wall",
        wall_bc=closure,
        ny=ny,
        nx=2,
        tau=tau,
        u_wall=0.05,
        omega=omega,
        steps=steps,
    )
    t0 = time.time()
    rec = rb.run_channel(cfg)
    return {
        "closure": closure,
        "ny": ny,
        "tau": tau,
        "delta_stokes": rec["delta_stokes"],
        "delta_over_dx": rec["delta_stokes"],
        "H_over_delta": rec["H_over_delta"],
        "period_steps": rec["period_steps"],
        "steps_run": rec["steps_run"],
        "l2_rel_complex": rec["l2_rel_complex"],
        "amp_max_rel": rec["amp_max_rel"],
        "phase_max_abs_masked": rec["phase_max_abs_masked"],
        "period_to_period_change": rec["period_to_period_change"],
        "mass_drift": rec["mass_drift"],
        "runtime_s": round(time.time() - t0, 2),
    }


def arm_ladder(nys: list[int], taus: list[float], forces: list[float], walls: list[float]) -> None:
    data = _load("ladder.json")
    for ny in nys:
        for tau in taus:
            for force in forces:
                for closure in CLOSURES:
                    for wf in walls:
                        key = f"ny{ny}_tau{tau:g}_F{force:g}_w{wf:g}_{closure}"
                        if key in data:
                            continue
                        data[key] = steady_row(closure, ny, tau, force, wf)
                        _dump("ladder.json", data)
                        print(
                            f"{key}: L2={data[key]['l2_rel'] * 100:.4f}% "
                            f"off/u_max={data[key]['offset_over_umax']:+.3e}",
                            flush=True,
                        )


def arm_osc(deltas: list[float], taus: list[float]) -> None:
    data = _load("oscillating.json")
    for d in deltas:
        for tau in taus:
            for closure in CLOSURES:
                key = f"delta{d:g}_tau{tau:g}_{closure}"
                if key in data:
                    continue
                data[key] = oscillating_row(closure, d, tau)
                _dump("oscillating.json", data)
                print(
                    f"{key}: L2={data[key]['l2_rel_complex'] * 100:.4f}% "
                    f"phase={data[key]['phase_max_abs_masked']:.2e}",
                    flush=True,
                )


def arm_invariant() -> None:
    payload = {"D2Q9": rb.third_moment_representability(rb.D2Q9)}
    payload["D3Q19"] = rb.third_moment_representability(rb.D3Q19, shape=(3, 3, 3))
    payload["note"] = (
        "max_rel_residual = |M3(order-3 output) - T_in| / |T_in| on random "
        "lattice distributions; rank_gram3/symmetric_dim = rank of the discrete "
        "Gram matrix over the symmetric third-moment directions."
    )
    _dump("invariant.json", payload)
    print(json.dumps(payload, indent=1))


def _order(rows: list[dict], key: str = "l2_rel") -> dict:
    """Least-squares order from |log l2| vs log(ny - 1)."""
    xs = np.log(np.asarray([r["height"] for r in rows], dtype=float))
    ys = np.log(np.asarray([r[key] for r in rows], dtype=float))
    if xs.size < 2:
        return {}
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    return {
        "fit_order": float(slope),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_points": int(xs.size),
    }


def arm_analyse() -> None:
    ladder = _load("ladder.json")
    osc = _load("oscillating.json")
    inv = _load("invariant.json")

    groups: dict[str, list[dict]] = {}
    for row in ladder.values():
        if row["tau"] != TAU or row["force"] != FORCE:
            continue
        key = f"w{row['wall_force']:g}/{row['closure']}"
        groups.setdefault(key, []).append(row)
    orders = {
        k: _order(sorted(v, key=lambda r: r["ny"]))
        for k, v in sorted(groups.items())
        if len(v) >= 2
    }

    tau_sweep: dict[str, dict] = {}
    for row in ladder.values():
        if row["ny"] != 32 or row["force"] != FORCE:
            continue
        tau_sweep[f"tau{row['tau']:g}/w{row['wall_force']:g}/{row['closure']}"] = {
            "l2_rel": row["l2_rel"],
            "offset_over_umax": row["offset_over_umax"],
        }

    force_sweep: dict[str, dict] = {}
    for row in ladder.values():
        if row["ny"] != 24 or row["tau"] != TAU or row["force"] == FORCE:
            continue
        force_sweep[f"F{row['force']:g}/w{row['wall_force']:g}/{row['closure']}"] = {
            "l2_rel": row["l2_rel"],
            "offset_over_umax": row["offset_over_umax"],
            "offset_over_force": row["offset_over_force"],
        }
    for row in ladder.values():
        if row["ny"] != 24 or row["tau"] != TAU or row["force"] != FORCE:
            continue
        force_sweep[f"F{row['force']:g}/w{row['wall_force']:g}/{row['closure']}"] = {
            "l2_rel": row["l2_rel"],
            "offset_over_umax": row["offset_over_umax"],
            "offset_over_force": row["offset_over_force"],
        }

    summary = {
        "arm": "R5",
        "anchor": "XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC",
        "ladder_tau": TAU,
        "ladder_force": FORCE,
        "order_fits": orders,
        "tau_sweep_ny32": tau_sweep,
        "force_sweep_ny24": force_sweep,
        "oscillating": {
            k: {
                "delta_over_dx": v["delta_over_dx"],
                "tau": v["tau"],
                "l2_rel_complex": v["l2_rel_complex"],
                "phase_max_abs_masked": v["phase_max_abs_masked"],
            }
            for k, v in sorted(osc.items())
            if isinstance(v, dict)
        },
        "third_moment": inv,
        "ladder_raw": {k: v for k, v in sorted(ladder.items())},
    }
    _dump("summary_r5.json", summary)

    print("=== steady Poiseuille ladder (tau=0.8, F=2e-5) ===")
    for k, fit in orders.items():
        rows = sorted(groups[k], key=lambda r: r["ny"])
        l2 = "  ".join(f"{r['ny']}:{r['l2_rel'] * 100:.4f}%" for r in rows)
        off = "  ".join(f"{r['offset_over_umax']:+.2e}" for r in rows)
        print(f"{k:42s} {l2}   | order={fit.get('fit_order', float('nan')):.2f}")
        print(f"{'':42s} offset/u_max {off}")
    print("\n=== oscillating Stokes layer (F=0) ===")
    for k, v in sorted(osc.items()):
        if isinstance(v, dict):
            print(
                f"{k:44s} delta={v['delta_over_dx']:.1f} L2={v['l2_rel_complex'] * 100:7.3f}% "
                f"phase={v['phase_max_abs_masked']:.2e}"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--arm", required=True, choices=("ladder", "tau", "force", "osc", "invariant", "analyse")
    )
    ap.add_argument("--nys", default="16,24,32,48")
    ap.add_argument("--taus", default=f"{TAU}")
    ap.add_argument("--forces", default=f"{FORCE}")
    ap.add_argument("--walls", default="0,1")
    ap.add_argument("--deltas", default="4,6")
    ap.add_argument("--osc-taus", default=f"{TAU}")
    args = ap.parse_args(argv)

    def _floats(text: str) -> list[float]:
        return [float(t) for t in text.split(",") if t]

    if args.arm == "ladder":
        arm_ladder([int(n) for n in args.nys.split(",") if n], _floats(args.taus), _floats(args.forces), _floats(args.walls))
    elif args.arm == "tau":
        arm_ladder([32], _floats(args.taus), _floats(args.forces), _floats(args.walls))
    elif args.arm == "force":
        arm_ladder([24], [TAU], _floats(args.forces), _floats(args.walls))
    elif args.arm == "osc":
        arm_osc(_floats(args.deltas), _floats(args.osc_taus))
    elif args.arm == "invariant":
        arm_invariant()
    else:
        arm_analyse()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())