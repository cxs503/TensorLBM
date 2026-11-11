"""Runner: stratified (two-layer) Poiseuille flow of the SC-MCMP model.

This is the A-arm reparation harness for ``GAP-LBM-MCMP-POISEUILLE``.  It runs
one cell of the pre-declared experiment matrix (see
``docs/MCMP_STRATIFIED_POISEUILLE_SPEC.md``) and writes a JSON record with
every number the verdict needs, so nothing is hand-copied into the report.

Two modes:

``--mode two_phase`` (default)
    Two immiscible components in a channel with bounce-back walls; component 1
    (``tau_lower``) fills the lower half, component 2 (``tau_upper``) the upper
    half.  A uniform body force ``G_x`` per unit mixture volume drives both.

``--mode walls``
    Wall-convention calibration: one component only (``G_12 = 0``, uniform
    density), i.e. plain single-phase Poiseuille through *the same* collision /
    streaming / bounce-back code path, used to measure where the no-slip plane
    actually is (``H = ny - 1`` vs ``ny - 2``).

Usage (single cell)::

    PYTHONPATH=src python cases/mcmp_stratified_poiseuille/run_sweep.py \
        --ny 25 --steps 15000 --u-eq self --force-scheme guo \
        --out artifacts/mcmp_stratified_poiseuille/ny25_self_guo.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tensorlbm.boundary.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import collide_sc_two_component  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

from cases.mcmp_stratified_poiseuille.analytic import (  # noqa: E402
    CS2,
    layer_viscosities,
    layered_poiseuille,
)

torch.set_grad_enabled(False)


def _init_fields(ny, nx, half, rho_heavy, rho_light, width, single_phase):
    ys = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, nx)
    if single_phase:
        rho1 = torch.full((ny, nx), 0.5 * (rho_heavy + rho_light))
        rho2 = torch.full((ny, nx), 0.5 * (rho_heavy + rho_light))
    else:
        alpha = 0.5 * (1.0 - torch.tanh((ys - float(half)) / width))
        rho1 = rho_heavy * alpha + rho_light * (1.0 - alpha)
        rho2 = rho_light * alpha + rho_heavy * (1.0 - alpha)
    z = torch.zeros((ny, nx))
    return equilibrium(rho1, z, z), equilibrium(rho2, z, z), rho1, rho2


def _profile(x, nx):
    """x-averaged profile (flow is x-periodic and uniform in x by symmetry)."""
    return x.mean(dim=1)


def _interface_crossing(alpha_profile) -> float:
    """Interpolated y where alpha crosses 0.5 (alpha = heavy fraction)."""
    a = alpha_profile.tolist()
    for j in range(len(a) - 1):
        if (a[j] - 0.5) * (a[j + 1] - 0.5) <= 0 and a[j] != a[j + 1]:
            t = (0.5 - a[j]) / (a[j + 1] - a[j])
            return j + t
    return float("nan")


def run_case(
    ny: int = 25,
    nx: int = 4,
    half: int | None = None,
    tau_lower: float = 1.0,
    tau_upper: float = 0.75,
    G_x: float = 5e-5,
    steps: int = 15000,
    G_12: float = -1.0,
    u_eq: str = "self",
    force_scheme: str = "shift",
    rho_heavy: float = 0.7,
    rho_light: float = 0.3,
    interface_width: float = 3.0,
    mode: str = "two_phase",
    sample_every: int | None = None,
    state_path: str | None = None,
    time_budget_s: float | None = None,
    resume: bool = False,
) -> dict:
    half = ny // 2 if half is None else half
    single_phase = mode == "walls"
    g12 = 0.0 if single_phase else G_12
    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = True
    wall[-1, :] = True

    sample_every = sample_every or max(1, min(steps // 20, 2000))
    hist: list[dict] = []
    start_step = 1
    if resume and state_path and os.path.exists(state_path):
        # Resume a long run from a checkpoint (the transient of this case can
        # exceed a single scheduling window).  Only the solution state and the
        # sample history are carried over; every parameter comes from this call.
        st = torch.load(state_path, weights_only=False)
        if st["config_core"] != [ny, nx, half, tau_lower, tau_upper, g12, u_eq, force_scheme]:
            raise ValueError("checkpoint was produced with different parameters")
        f1, f2 = st["f1"], st["f2"]
        mass0 = tuple(st["mass0"])
        hist = st["hist"]
        start_step = st["step"] + 1
        print(f"resumed from {state_path} at step {st['step']}", flush=True)
    else:
        f1, f2, rho1_0, rho2_0 = _init_fields(
            ny, nx, half, rho_heavy, rho_light, interface_width, single_phase
        )
        mass0 = (float(rho1_0.sum()), float(rho2_0.sum()))

    t0 = time.time()
    for step in range(start_step, start_step + steps):
        f1, f2 = collide_sc_two_component(
            f1, f2, G_12=g12, tau1=tau_lower, tau2=tau_upper,
            gx=G_x, solid_mask=wall, u_eq=u_eq, force_scheme=force_scheme,
        )
        f1, f2 = stream(f1), stream(f2)
        f1, f2 = bounce_back_cells(f1, wall), bounce_back_cells(f2, wall)
        if step % sample_every == 0 or step == steps:
            r1, ux1, uy1 = macroscopic(f1)
            r2, ux2, uy2 = macroscopic(f2)
            finite = bool(torch.isfinite(f1).all() and torch.isfinite(f2).all())
            if not finite:
                hist.append(dict(step=step, finite=False))
                break
            Q1 = float(_profile(ux1, nx).sum())
            Q2 = float(_profile(ux2, nx).sum())
            hist.append(dict(
                step=step,
                Q1=Q1,
                Q2=Q2,
                u_max=max(float(ux1[1:-1].abs().max()), float(ux2[1:-1].abs().max())),
                spurious=max(float(uy1[1:-1].abs().max()), float(uy2[1:-1].abs().max())),
                finite=True,
            ))
        if (time_budget_s is not None and step % 200 == 0
                and (time.time() - t0) > time_budget_s):
            torch.save(dict(
                f1=f1, f2=f2, step=step, hist=hist, mass0=mass0,
                config_core=[ny, nx, half, tau_lower, tau_upper, g12, u_eq, force_scheme],
            ), state_path)
            print(f"checkpoint written at step {step} (time budget reached)", flush=True)
            return dict(interrupted=True, step=step, finite=True, mode=mode)
    wall_time = time.time() - t0

    r1, ux1, uy1 = macroscopic(f1)
    r2, ux2, uy2 = macroscopic(f2)
    prof1 = _profile(ux1, nx)
    prof2 = _profile(ux2, nx)
    Q1 = float(prof1.sum())
    Q2 = float(prof2.sum())
    finite = bool(torch.isfinite(f1).all() and torch.isfinite(f2).all())

    rho_tot = torch.clamp(r1 + r2, min=1e-12)
    u_mix = (r1 * ux1 + r2 * ux2) / rho_tot
    uy_mix = (r1 * uy1 + r2 * uy2) / rho_tot
    u_mix_prof = _profile(u_mix, nx)
    # Mass-flux preserving velocities: the per-component velocity of a *trace*
    # component is amplified by 1/rho in the force shift, so it is not a
    # continuum velocity and must not be flux-integrated on its own.
    mflux = _profile(r1 * ux1 + r2 * ux2, nx)  # = rho_tot * u_mix (rho_tot ~ 1)
    Q_mix = float(u_mix_prof.sum())

    steady = False
    if len(hist) >= 5 and all(h.get("finite") for h in hist[-3:]):
        q_tail = [h["Q1"] + h["Q2"] for h in hist[-3:]]
        denom = max(abs(q_tail[-1]), 1e-30)
        steady = abs(q_tail[-1] - q_tail[0]) / denom < 1e-3

    steps_total = hist[-1]["step"] if hist else 0
    record: dict = dict(
        mode=mode,
        steps_total=steps_total,
        config=dict(
            ny=ny, nx=nx, half=half, tau_lower=tau_lower, tau_upper=tau_upper,
            G_x=G_x, steps=steps, G_12=g12, u_eq=u_eq, force_scheme=force_scheme,
            rho_heavy=rho_heavy, rho_light=rho_light, interface_width=interface_width,
            wall="post-streaming bounce-back, walls at cell centres (rows 0 and ny-1)",
        ),
        runtime=dict(
            wall_seconds=wall_time, python=platform.python_version(),
            torch=torch.__version__, platform=platform.platform(),
            dtype="float32", device="cpu",
        ),
        finite=finite,
        steady=steady,
        Q1=Q1, Q2=Q2, Q_total=Q1 + Q2,
        u_profile_1=prof1.tolist(), u_profile_2=prof2.tolist(),
        u_profile_mix=u_mix_prof.tolist(),
        u_max=max(float(ux1[1:-1].abs().max()), float(ux2[1:-1].abs().max())),
        spurious=max(float(uy1[1:-1].abs().max()), float(uy2[1:-1].abs().max())),
        u_max_mix=float(u_mix[1:-1].abs().max()),
        spurious_mix=float(uy_mix[1:-1].abs().max()),
        Q_mix=Q_mix,
        mass_flux_profile=mflux.tolist(),
        mass_drift_1=float(r1.sum()) - mass0[0],
        mass_drift_2=float(r2.sum()) - mass0[1],
        history=hist,
    )

    if single_phase:
        nu = CS2 * (tau_lower - 0.5)
        ref = {}
        for label, H in (("H_ny_minus_1", float(ny - 1)), ("H_ny_minus_2", float(ny - 2))):
            ana = layered_poiseuille(ny, 0.5 * H, G_x, nu, nu, H=H)
            umax_ref = G_x * H * H / (8.0 * nu)
            ref[label] = dict(
                H=H,
                u_max_analytic=umax_ref,
                u_max_error_rel=(float(prof1.max()) - umax_ref) / umax_ref,
                Q_analytic=ana["Q_total"],
                Q_error_rel=(Q1 - ana["Q_total"]) / ana["Q_total"],
            )
        record["wall_calibration"] = ref
        record["nu"] = nu
    else:
        alpha_prof = _profile((r1 - rho_light) / (rho_heavy - rho_light), nx).clamp(0.0, 1.0)
        record["interface_crossing"] = _interface_crossing(alpha_prof)
        record["heavy_mass_per_width"] = float(alpha_prof.sum())
        record["alpha_profile"] = alpha_prof.tolist()
        # Stratification gate: a run whose phases have mixed is NOT a two-phase
        # (stratified) Poiseuille flow and its flux comparison is meaningless.
        bulk = alpha_prof[1:-1]
        record["phi_contrast"] = float(bulk.max() - bulk.min())
        record["stratified"] = bool(record["phi_contrast"] > 0.5)
        # Trace-component velocity blow-up diagnostic (1/rho amplification of
        # the force-shift velocity): max |u_sigma| where rho_sigma < 0.05.
        with torch.no_grad():
            trace1 = ux1[(r1 < 0.05) & (~wall)]
            trace2 = ux2[(r2 < 0.05) & (~wall)]
        record["trace_velocity_max"] = float(
            max(trace1.abs().max().item() if trace1.numel() else 0.0,
                trace2.abs().max().item() if trace2.numel() else 0.0)
        )
        low = slice(0, int(half))
        up = slice(int(half), ny)
        layer = dict(
            Q_mix_lower=float(u_mix_prof[low].sum()),
            Q_mix_upper=float(u_mix_prof[up].sum()),
            mass_flux_lower=float(mflux[low].sum()),
            mass_flux_upper=float(mflux[up].sum()),
        )
        record["layer_flux"] = layer
        refs = {}
        for rule in ("component", "mixture"):
            nu_lo, nu_up = layer_viscosities(tau_lower, tau_upper, rho_heavy, rho_light, rule)
            ana = layered_poiseuille(ny, float(half), G_x, nu_lo, nu_up)
            refs[rule] = dict(
                nu_lower=nu_lo, nu_upper=nu_up,
                Q_lower=ana["Q_lower"], Q_upper=ana["Q_upper"], Q_total=ana["Q_total"],
                u_profile=ana["profile"],
                err_Q1=(Q1 - ana["Q_lower"]) / ana["Q_lower"],
                err_Q2=(Q2 - ana["Q_upper"]) / ana["Q_upper"],
                err_Q_total=(Q1 + Q2 - ana["Q_total"]) / ana["Q_total"],
                err_profile_L2=float(
                    math.sqrt(sum((a - b) ** 2 for a, b in zip(u_mix_prof.tolist(), ana["profile"]))
                              / max(sum(b * b for b in ana["profile"]), 1e-300))
                ),
                err_interface_rel=(record["interface_crossing"] - half) / half,
                err_Q_mix=(Q_mix - ana["Q_total"]) / ana["Q_total"],
                err_Q_mix_lower=(layer["Q_mix_lower"] - ana["Q_lower"]) / ana["Q_lower"],
                err_Q_mix_upper=(layer["Q_mix_upper"] - ana["Q_upper"]) / ana["Q_upper"],
            )
        record["analytical"] = refs
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=25)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--half", type=int, default=None)
    ap.add_argument("--tau-lower", type=float, default=1.0)
    ap.add_argument("--tau-upper", type=float, default=0.75)
    ap.add_argument("--gx", type=float, default=5e-5)
    ap.add_argument("--g12", type=float, default=-1.0)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--u-eq", default="self", choices=["self", "mixture", "viscous"])
    ap.add_argument("--force-scheme", default="shift", choices=["shift", "guo"])
    ap.add_argument("--mode", default="two_phase", choices=["two_phase", "walls"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--state", default=None, help="checkpoint path (enables resume)")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="seconds; stop and checkpoint when exceeded")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    rec = run_case(
        ny=args.ny, nx=args.nx, half=args.half, tau_lower=args.tau_lower,
        tau_upper=args.tau_upper, G_x=args.gx, steps=args.steps, G_12=args.g12,
        u_eq=args.u_eq, force_scheme=args.force_scheme, mode=args.mode,
        state_path=args.state, time_budget_s=args.time_budget, resume=args.resume,
    )
    if rec.get("interrupted"):
        print(f"INTERRUPTED at step {rec['step']}")
        return 3
    text = json.dumps(rec, indent=1)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text)
    summary = dict(
        mode=rec["mode"], ny=rec["config"]["ny"], u_eq=rec["config"]["u_eq"],
        force_scheme=rec["config"]["force_scheme"], finite=rec["finite"],
        steady=rec["steady"], Q1=rec["Q1"], Q2=rec["Q2"], Q_total=rec["Q_total"],
        u_max=rec["u_max"], spurious=rec["spurious"],
    )
    if rec["mode"] == "two_phase":
        summary.update(dict(
            Q_mix=rec["Q_mix"], u_max_mix=rec["u_max_mix"],
            spurious_mix=rec["spurious_mix"], phi_contrast=rec["phi_contrast"],
            stratified=rec["stratified"], trace_velocity_max=rec["trace_velocity_max"],
        ))
        summary["interface_crossing"] = rec["interface_crossing"]
        summary["errors"] = {
            rule: {k: v for k, v in ref.items() if k.startswith("err_") or k.startswith("Q_")}
            for rule, ref in rec["analytical"].items()
        }
    else:
        summary["wall_calibration"] = rec["wall_calibration"]
    print("SUMMARY " + json.dumps(summary))
    return 0 if rec["finite"] else 2


if __name__ == "__main__":
    raise SystemExit(main())