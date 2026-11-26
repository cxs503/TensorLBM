"""R4 cross-validation of the regularised (non-equilibrium stress) wall closure.

Anchor: ``XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC``.

R1-R3 (m18423) proved the semantics transplant (identity to the OSS scalar
transcription) and measured two *single points* of accuracy: the steady
Couette/Poiseuille solutions and the oscillating Stokes layer at
H/delta = 18.6 and 26.3.  Two open claims were left unverified:

* the module docstring claims the closure is "second order and time-step
  independent at a wall";
* the oscillating-wall error (2.87 % / 5.62 %) was *attributed* to
  under-resolution (delta ~ 1.8-2.5 lattice spacings) on two points only.

This runner turns both into falsifiable tests:

``L1`` resolution ladder
    delta/Delta_x in {1.5, 2, 3, 4, 5, 6, 8, 10} at fixed tau = 0.8, both
    closures, H/delta ~ 10.  If the residual is a discretisation error the
    complex L2 error must fall like ``(Delta_x/delta)**p`` with p ~ 2; a
    closure bias would show up as a plateau (p -> 0).
``L2`` tau sweep
    same physical Stokes problem (delta/Delta_x = 6 fixed, omega = 2 nu/delta^2)
    at tau in {0.55 ... 2.0}.  "Time-step independent at a wall" predicts a flat
    error curve for the regularised closure; bounce back is expected to move
    with ``omega tau``.
``L3`` D3Q19 physical validation
    3-D plane channel (Couette + body-force Poiseuille, periodic x/z) driven
    through the same closure with the D3Q19 velocity set -- previously only the
    D2Q9 path had *physical* evidence (the D3Q19 tables were pinned by
    invariants only).  Bounce back is kept as the control arm.

Everything is float64 NumPy, single process, deterministic: the same command
reproduces the artifacts byte for byte.

CLI::

    PYTHONPATH=src python development/verification/regularised_bc_noneq_stress/run_xval_r4.py --arm all
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from tensorlbm import regularised_bc as rb

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts_r4"

TAU_REF = 0.8
NU_REF = (TAU_REF - 0.5) / 3.0
U0 = 0.05
DELTAS = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
TAUS = (0.55, 0.575, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _loglog_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope/intercept of log y vs log x (y, x > 0)."""
    p, c = np.polyfit(np.log(x), np.log(y), 1)
    pred = p * np.log(x) + c
    ss_res = float(np.sum((np.log(y) - pred) ** 2))
    ss_tot = float(np.sum((np.log(y) - np.log(y).mean()) ** 2))
    return float(p), 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _oscillating(delta_target: float, tau: float, wall_bc: str) -> dict:
    """One oscillating-wall run whose Stokes layer spans ``delta_target`` cells.

    Holding ``delta`` fixed in lattice units and raising the period refines the
    *layer* at Delta_x = 1; ``omega`` is set from ``nu`` so that
    ``delta = sqrt(2 nu / omega)``.
    """
    nu = (tau - 0.5) / 3.0
    omega = 2.0 * nu / delta_target**2
    period = 2.0 * np.pi / omega
    height = max(20.0, 10.0 * delta_target)
    ny = int(round(height)) + 2
    steps = int(round(4.0 * period))
    cfg = rb.ChannelConfig(
        case="oscillating_wall",
        wall_bc=wall_bc,
        ny=ny,
        nx=2,
        tau=tau,
        u_wall=U0,
        omega=omega,
        steps=steps,
        check_every=200,
        tol=1e-10,
        init="rest",
    )
    t0 = time.time()
    rec = rb.run_channel(cfg)
    rec["wall_time_s"] = time.time() - t0
    # The preregistered phase metric is masked to |A_ref| > 0.1 U0; the module
    # reports the unmasked maximum, which in the far field is the phase of two
    # near-zero complex numbers (O(1) rad regardless of the closure).  Keep both.
    amp_abs = np.asarray(rec["amp_abs"])
    ref_abs = np.asarray(rec["amp_ref_abs"])
    dphi = np.angle(
        np.exp(1j * (np.asarray(rec["amp_phase"]) - np.asarray(rec["amp_ref_phase"])))
    )
    mask = ref_abs > 0.1 * U0
    rec["phase_max_abs_masked"] = float(np.max(np.abs(dphi[mask]))) if mask.any() else float("nan")
    rec["phase_mask_points"] = int(mask.sum())
    rec["delta_target"] = delta_target
    rec["nu"] = nu
    rec["omega"] = omega
    rec["omega_tau"] = omega * tau
    return rec


# ---------------------------------------------------------------------------
# L1 + L2: D2Q9 oscillating wall
# ---------------------------------------------------------------------------
def ladder(deltas=DELTAS, tau: float = TAU_REF) -> list[dict]:
    rows = []
    for delta in deltas:
        for wall_bc in ("regularised", "bounce_back"):
            rec = _oscillating(delta, tau, wall_bc)
            rows.append(
                {
                    "arm": "L1",
                    "tau": tau,
                    "wall_bc": wall_bc,
                    "delta_over_dx": rec["delta_stokes"],
                    "ny": rec["config"]["ny"],
                    "H_over_delta": rec["H_over_delta"],
                    "period_steps": rec["period_steps"],
                    "steps_run": rec["steps_run"],
                    "l2_rel_complex": rec["l2_rel_complex"],
                    "amp_max_rel": rec["amp_max_rel"],
                    "phase_max_abs": rec["phase_max_abs"],
                    "phase_max_abs_masked": rec["phase_max_abs_masked"],
                    "phase_mask_points": rec["phase_mask_points"],
                    "period_to_period_change": rec["period_to_period_change"],
                    "mass_drift": rec["mass_drift"],
                    "wall_time_s": rec["wall_time_s"],
                }
            )
            print(
                f"[L1] tau={tau:4.2f} {wall_bc:11s} delta={rec['delta_stokes']:5.2f} "
                f"H/delta={rec['H_over_delta']:5.1f} T={rec['period_steps']:6d} "
                f"l2={100*rec['l2_rel_complex']:.4f}% "
                f"phase={rec['phase_max_abs']:.2e} drift={rec['period_to_period_change']:.2e}",
                flush=True,
            )
    return rows


def tau_sweep(taus=TAUS) -> list[dict]:
    rows = []
    delta = 6.0
    for tau in taus:
        for wall_bc in ("regularised", "bounce_back"):
            rec = _oscillating(delta, tau, wall_bc)
            rows.append(
                {
                    "arm": "L2",
                    "wall_bc": wall_bc,
                    "tau": tau,
                    "nu": rec["nu"],
                    "omega": rec["omega"],
                    "omega_tau": rec["omega_tau"],
                    "delta_over_dx": rec["delta_stokes"],
                    "H_over_delta": rec["H_over_delta"],
                    "period_steps": rec["period_steps"],
                    "steps_run": rec["steps_run"],
                    "l2_rel_complex": rec["l2_rel_complex"],
                    "amp_max_rel": rec["amp_max_rel"],
                    "phase_max_abs": rec["phase_max_abs"],
                    "phase_max_abs_masked": rec["phase_max_abs_masked"],
                    "phase_mask_points": rec["phase_mask_points"],
                    "period_to_period_change": rec["period_to_period_change"],
                    "mass_drift": rec["mass_drift"],
                    "wall_time_s": rec["wall_time_s"],
                }
            )
            print(
                f"[L2] {wall_bc:11s} tau={tau:5.3f} omega*tau={rec['omega_tau']:.2e} "
                f"T={rec['period_steps']:6d} l2={100*rec['l2_rel_complex']:.4f}% "
                f"phase={rec['phase_max_abs']:.2e}",
                flush=True,
            )
    return rows


# ---------------------------------------------------------------------------
# L3: D3Q19 3-D plane channel
# ---------------------------------------------------------------------------
def _stream3(spec, f: np.ndarray) -> np.ndarray:
    out = np.empty_like(f)
    axes = tuple(range(spec.dims))
    for q in range(spec.q):
        shifts = tuple(int(v) for v in spec.c[q])
        out[q] = np.roll(f[q], shifts, axis=axes)
    return out


def _bb_plane3(spec, f_pre: np.ndarray, out: np.ndarray, row: int, u_wall: float = 0.0) -> None:
    """Half-way bounce back on the y plane ``row`` of a (Q, nx, ny, nz) field."""
    rho = f_pre[:, :, row, :].sum(axis=0)
    u_vec = np.zeros(spec.dims)
    u_vec[0] = u_wall
    cu = spec.c.astype(f_pre.dtype) @ u_vec
    shape = (-1, 1, 1)
    out[:, :, row, :] = f_pre[spec.opposite][:, :, row, :] + (
        2.0 * spec.w.reshape(shape) * rho[None, :, :] * cu.reshape(shape)
    ) / spec.cs2


def _profile_exact_3d(case: str, y: np.ndarray, tau: float, *, u_wall: float, body_force: float,
                      height: float, lower: float) -> np.ndarray:
    nu = (tau - 0.5) / 3.0
    s = y - lower
    if case == "couette":
        return u_wall * s / height
    return body_force / (2.0 * nu) * s * (height - s)


def channel_3d(
    case: str,
    *,
    ny: int,
    tau: float,
    wall_bc: str,
    u_wall: float = U0,
    body_force: float = 0.0,
    nx: int = 2,
    nz: int = 2,
    tol: float = 1e-9,
    check_every: int = 100,
    max_steps: int = 60000,
    init: str = "analytic",
) -> dict:
    spec = rb.D3Q19
    lower = 0.0 if wall_bc == "regularised" else 0.5
    height = (ny - 1.0) if wall_bc == "regularised" else (ny - 2.0)
    u_init = np.zeros((3, nx, ny, nz))
    if init == "analytic":
        y_init = np.arange(ny, dtype=float)
        prof = _profile_exact_3d(
            case, y_init, tau, u_wall=u_wall, body_force=body_force, height=height, lower=lower
        )
        prof[: 1 if lower == 0.0 else 2] = 0.0
        u_init[0] = prof[:, None][None, :, :].repeat(nx, axis=0)
    f = rb.equilibrium(spec, np.ones((nx, ny, nz)), u_init)
    force = np.zeros((3, nx, ny, nz))
    if case == "poiseuille":
        force[0, :, 1:-1, :] = body_force
    moving = case == "couette"
    u_wall_vec = (u_wall, 0.0, 0.0)
    ux_prev = None
    converged = False
    delta = float("inf")
    step = 0
    t0 = time.time()
    for step in range(1, max_steps + 1):
        f_arrival = f
        f = rb._collide(spec, f, tau)
        if case == "poiseuille":
            f = rb._guo_body_force(spec, f, force)
        if wall_bc == "regularised":
            f = _stream3(spec, f)
            f = rb.apply_regularised_wall(
                spec, f, axis=1, side="high",
                u_wall=u_wall_vec if moving else 0.0, component_of_axis=(0, 1, 2),
            )
            f = rb.apply_regularised_wall(
                spec, f, axis=1, side="low", u_wall=0.0, component_of_axis=(0, 1, 2),
            )
        else:
            out = f.copy()
            _bb_plane3(spec, f_arrival, out, ny - 1, u_wall if moving else 0.0)
            _bb_plane3(spec, f_arrival, out, 0, 0.0)
            f = _stream3(spec, out)
        if step % check_every == 0:
            _, u = rb.macroscopic(spec, f)
            ux = u[0, :, 1:-1, :].mean(axis=(0, 2))
            if ux_prev is not None:
                delta = float(np.max(np.abs(ux - ux_prev)) / max(np.max(np.abs(ux)), 1e-30))
                if delta < tol:
                    converged = True
                    break
            ux_prev = ux
    rho, u = rb.macroscopic(spec, f)
    ux = u[0, :, 1:-1, :].mean(axis=(0, 2))
    y = np.arange(1, ny - 1, dtype=float)
    exact = _profile_exact_3d(
        case, y, tau, u_wall=u_wall, body_force=body_force, height=height, lower=lower
    )
    err = ux - exact
    return {
        "arm": "L3",
        "case": case,
        "wall_bc": wall_bc,
        "tau": tau,
        "ny": ny,
        "nx": nx,
        "nz": nz,
        "height": height,
        "wall_origin": lower,
        "body_force": body_force,
        "u_max_exact": float(np.max(np.abs(exact))),
        "steps_run": step,
        "converged": converged,
        "final_rel_delta": delta,
        "l2_rel": float(np.linalg.norm(err) / np.linalg.norm(exact)),
        "linf_abs": float(np.max(np.abs(err))),
        "mass_drift": float(rho.mean() - 1.0),
        "y": y.tolist(),
        "ux": ux.tolist(),
        "ux_exact": exact.tolist(),
        "wall_time_s": time.time() - t0,
    }


def d3q19_channel(cases=("couette", "poiseuille"), nys=(24, 32)) -> list[dict]:
    rows = []
    for case in cases:
        for ny in nys:
            nu = (TAU_REF - 0.5) / 3.0
            height = ny - 1.0
            g = 8.0 * nu * U0 / (height * height) if case == "poiseuille" else 0.0
            for wall_bc in ("regularised", "bounce_back"):
                rec = channel_3d(
                    case, ny=ny, tau=TAU_REF, wall_bc=wall_bc, body_force=g, init="analytic"
                )
                rec["delta_over_dx"] = None
                rows.append(rec)
                print(
                    f"[L3] {case:10s} {wall_bc:11s} ny={ny:3d} l2={100*rec['l2_rel']:.4f}% "
                    f"linf={rec['linf_abs']:.2e} steps={rec['steps_run']:6d} "
                    f"conv={rec['converged']} {rec['wall_time_s']:.1f}s",
                    flush=True,
                )
    return rows


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def analyse(l1: list[dict], l2: list[dict], l3: list[dict]) -> dict:
    out: dict = {}

    # L1: order of accuracy per closure (points with delta >= 2 to stay out of
    # the delta ~ 1.5 cell regime where the layer is not resolved at all).
    order = {}
    taus_l1 = sorted({float(r.get("tau", TAU_REF)) for r in l1})
    for tau in taus_l1:
      for wall_bc in ("regularised", "bounce_back"):
        sel = sorted(
            [r for r in l1
             if r["wall_bc"] == wall_bc
             and float(r.get("tau", TAU_REF)) == tau
             and r["delta_over_dx"] >= 2.0],
            key=lambda r: r["delta_over_dx"],
        )
        if len(sel) < 2:
            continue
        x = np.array([r["delta_over_dx"] for r in sel])
        y = np.array([r["l2_rel_complex"] for r in sel])
        p, r2 = _loglog_slope(x, y)
        order[f"tau={tau:.3f}/{wall_bc}"] = {
            "tau": tau,
            "wall_bc": wall_bc,
            "delta_over_dx": x.tolist(),
            "l2_rel_complex": y.tolist(),
            "slope": p,
            "r2": r2,
            "ratio_l2_coarse_over_fine": float(y[0] / y[-1]),
            "second_order_prediction": float((x[-1] / x[0]) ** 2),
            "l2_at_finest": float(y[-1]),
            "amp_max_rel_finest": float(sel[-1]["amp_max_rel"]),
            "phase_max_abs_finest": float(sel[-1]["phase_max_abs"]),
            "phase_max_abs_masked_finest": float(sel[-1]["phase_max_abs_masked"]),
            "max_period_to_period_change": float(max(r["period_to_period_change"] for r in sel)),
            "max_abs_mass_drift": float(max(abs(r["mass_drift"]) for r in sel)),
        }
    out["L1_order"] = order

    # L2: spread of the error over tau at a fixed physical Stokes problem.
    spread = {}
    for wall_bc in ("regularised", "bounce_back"):
        sel = [r for r in l2 if r["wall_bc"] == wall_bc]
        vals = np.array([r["l2_rel_complex"] for r in sel])
        taus = np.array([r["tau"] for r in sel])
        spread[wall_bc] = {
            "tau": taus.tolist(),
            "l2_rel_complex": vals.tolist(),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "max_over_min": float(vals.max() / vals.min()),
            "mean": float(vals.mean()),
            "corr_with_tau": float(np.corrcoef(taus, vals)[0, 1]),
            "max_period_to_period_change": float(max(r["period_to_period_change"] for r in sel)),
        }
    out["L2_tau_independence"] = spread

    # L3: D3Q19 physical error and its grid dependence.
    d3 = {}
    for case in ("couette", "poiseuille"):
        for wall_bc in ("regularised", "bounce_back"):
            sel = sorted([r for r in l3 if r["case"] == case and r["wall_bc"] == wall_bc],
                         key=lambda r: r["ny"])
            if len(sel) >= 2:
                p, _ = _loglog_slope(
                    np.array([r["height"] for r in sel]), np.array([r["l2_rel"] for r in sel])
                )
            else:
                p = float("nan")
            d3[f"{case}/{wall_bc}"] = {
                "ny": [r["ny"] for r in sel],
                "l2_rel": [r["l2_rel"] for r in sel],
                "linf_abs": [r["linf_abs"] for r in sel],
                "slope_vs_height": p,
                "converged": [r["converged"] for r in sel],
                "max_abs_mass_drift": float(max(abs(r["mass_drift"]) for r in sel)),
            }
    out["L3_d3q19"] = d3
    return out


def _merge(path: Path, rows: list[dict], key: tuple[str, ...]) -> list[dict]:
    """Accumulate sweep rows across invocations (identical keys overwritten)."""
    old: list[dict] = json.loads(path.read_text()) if path.exists() else []
    keep = [r for r in old if tuple(r.get(k) for k in key) not in {tuple(r2.get(k) for k in key) for r2 in rows}]
    merged = sorted(keep + rows, key=lambda r: tuple(str(r.get(k)) for k in key))
    path.write_text(json.dumps(merged, indent=1))
    return merged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="all", choices=("all", "L1", "L2", "L3", "analyse"))
    ap.add_argument("--taus", default=None, help="comma separated tau subset for L2")
    ap.add_argument("--l1-taus", default=None, help="comma separated tau subset for L1")
    ap.add_argument("--l1-deltas", default=None, help="comma separated delta subset for L1")
    ap.add_argument("--d3", default=None,
                    help="comma separated case:ny subset for L3, e.g. couette:24,poiseuille:32")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    l1: list[dict] = []
    l2: list[dict] = []
    l3: list[dict] = []
    if args.arm in ("all", "L1"):
        deltas = tuple(float(x) for x in args.l1_deltas.split(",")) if args.l1_deltas else DELTAS
        taus_l1 = tuple(float(x) for x in args.l1_taus.split(",")) if args.l1_taus else (TAU_REF,)
        rows = []
        for tau in taus_l1:
            rows.extend(ladder(deltas, tau))
        _merge(OUT / "ladder.json", rows, ("tau", "delta_over_dx", "wall_bc"))
    if args.arm in ("all", "L2"):
        taus = tuple(float(t) for t in args.taus.split(",")) if args.taus else TAUS
        rows = tau_sweep(taus)
        _merge(OUT / "tau_sweep.json", rows, ("tau", "wall_bc"))
    if args.arm in ("all", "L3"):
        cases, nys = ("couette", "poiseuille"), (24, 32)
        if args.d3:
            pairs = [item.split(":") for item in args.d3.split(",")]
            cases = tuple(dict.fromkeys(c for c, _ in pairs))
            nys = tuple(dict.fromkeys(int(n) for _, n in pairs))
            plates = {(c, int(n)) for c, n in pairs}
            rows = [
                r for r in d3q19_channel(cases, nys)
                if (r["case"], r["ny"]) in plates
            ]
        else:
            rows = d3q19_channel(cases, nys)
        for rec in rows:
            print(
                f"[L3] {rec['case']:10s} {rec['wall_bc']:11s} ny={rec['ny']:3d} "
                f"l2={100 * rec['l2_rel']:.4f}% linf={rec['linf_abs']:.2e} "
                f"steps={rec['steps_run']:6d} conv={rec['converged']} "
                f"{rec['wall_time_s']:.1f}s",
                flush=True,
            )
        l3 = _merge(OUT / "d3q19_channel.json", rows, ("case", "ny", "wall_bc"))
    if args.arm == "analyse":
        l1 = json.loads((OUT / "ladder.json").read_text())
        l2 = json.loads((OUT / "tau_sweep.json").read_text())
        l3 = json.loads((OUT / "d3q19_channel.json").read_text())
    if args.arm in ("all", "analyse"):
        summary = analyse(l1, l2, l3)
        summary["meta"] = {
            "anchor": "XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC",
            "arm": "R4",
            "delta_over_dx_ladder": list(DELTAS),
            "taus": list(TAUS),
            "tau_ref": TAU_REF,
            "u0": U0,
            "dtype": "float64",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "wall_time_s": time.time() - t_start,
        }
        (OUT / "summary_r4.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())