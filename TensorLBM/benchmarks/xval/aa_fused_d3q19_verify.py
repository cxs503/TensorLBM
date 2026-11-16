"""XVAL driver: D3Q19 AA-pattern fused stream-collide against exact solutions.

Anchor: ``XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE`` (lanl/MF-LBM
``aa-pattern-fused-stream-collide``) — the reference practice is a 3D D3Q19
code, while the verified kernel in this package was D2Q9 only.

Gates, in order of logical dependence
-------------------------------------
``identity``    fused vs canonical materialised operator, bit for bit, on a
                non-uniform state with walls, a moving wall and a body force.
                (A tolerance comparison would pass for a wrong-but-close
                operator; equality does not.)
``couette``     plane Couette between halfway walls, exact linear profile —
                isolates the moving-wall term ``6 w_i rho (c_i . u_w)``, whose
                sign a fused-vs-materialised comparison cannot see (shared line).
``poiseuille``  Guo-forced plane channel, exact parabola, ``u_max = g H^2/8nu``
                — pins the D3Q19 weights and the forcing.
``taylor_green``2D-extended Taylor-Green vortex, an exact 3D Navier-Stokes
                solution with a non-zero convective term; pins the in-plane
                diagonals and the decay rate ``2 nu k^2``.

Usage
-----
    python benchmarks/xval/aa_fused_d3q19_verify.py                 # all gates
    python benchmarks/xval/aa_fused_d3q19_verify.py --gate identity  # one gate
    python benchmarks/xval/aa_fused_d3q19_verify.py --out artifacts/xval_aa_pattern/d3q19_verify.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import tensorlbm.aa_fused_push_pull_3d as M  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _random_state(nz, ny, nx, seed=0, amp=0.02):
    """Non-uniform, finite state: equilibrium of a random smooth velocity field."""
    g = torch.Generator().manual_seed(seed)
    rho = 1.0 + amp * torch.rand((nz, ny, nx), generator=g, dtype=torch.float64)
    ux = amp * torch.randn((nz, ny, nx), generator=g, dtype=torch.float64)
    uy = amp * torch.randn((nz, ny, nx), generator=g, dtype=torch.float64)
    uz = amp * torch.randn((nz, ny, nx), generator=g, dtype=torch.float64)
    return M.equilibrium(rho, ux, uy, uz)


def _profile_error(u_num: torch.Tensor, u_exact: torch.Tensor, u_scale: float):
    """L2 and max errors of a 1-D profile, normalised by the reference scale."""
    d = u_num - u_exact
    return {
        "l2": float(torch.sqrt(torch.mean(d * d))) / u_scale,
        "max": float(d.abs().max()) / u_scale,
    }


def _channel_initial(nz, ny, nx, u0=0.0):
    """Start from rest (rho = 1, u = 0) plus the given uniform stream in x."""
    rho = torch.ones((nz, ny, nx), dtype=torch.float64)
    ux = torch.full((nz, ny, nx), float(u0), dtype=torch.float64)
    uy = torch.zeros_like(ux)
    uz = torch.zeros_like(ux)
    return M.equilibrium(rho, ux, uy, uz)


# --------------------------------------------------------------------------- #
# gate 1 — operator identity (bit for bit)
# --------------------------------------------------------------------------- #
def gate_identity():
    nz, ny, nx = 5, 8, 7
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    solid[:, 3, 0:3] = True                      # an interior obstacle
    wall_u = M.wall_velocity_3d(solid, (0.05, 0.0, -0.02), row=-1)
    force = torch.zeros((3, nz, ny, nx), dtype=torch.float64)
    force[0] = 1e-4
    force[2] = -3e-5

    f0 = _random_state(nz, ny, nx, seed=11)
    tau = 0.63

    fused = M.fused_step(f0, tau, solid=solid, wall_u=wall_u, force=force)
    canon = M.collide_stream_step(f0, tau, solid=solid, wall_u=wall_u, force=force)

    result = {
        "shape": [nz, ny, nx],
        "tau": tau,
        "bit_equal": bool(torch.equal(fused, canon)),
        "max_abs_diff": float((fused - canon).abs().max()),
        "periodic_bit_equal": bool(torch.equal(
            M.aa_fused_stream_collide_3d(_random_state(nz, ny, nx, seed=3), tau),
            M.separate_stream_collide_3d(_random_state(nz, ny, nx, seed=3), tau),
        )),
        "n_differing_entries": int((fused != canon).sum()),
    }
    # vacuity control: a destination-colliding (C[S[f]]) operator must NOT match
    rho, ux, uy, uz = M.macroscopic(f0)
    streamed = torch.empty_like(f0)
    for i in range(M.ND):
        streamed[i] = M._pull(f0[i], i)
    r2, x2, y2, z2 = M.macroscopic(streamed)
    cs = streamed + (1.0 / tau) * (M.equilibrium(r2, x2, y2, z2) - streamed)
    result["stream_then_collide_max_abs_diff"] = float((cs - canon).abs().max())
    result["stream_then_collide_rel_diff"] = float(
        (cs - canon).abs().max() / f0.abs().max()
    )
    result["vacuity_control_rejected"] = result["stream_then_collide_rel_diff"] > 1e-6
    return result


# --------------------------------------------------------------------------- #
# gate 2 — plane Couette (moving wall, exact linear profile)
# --------------------------------------------------------------------------- #
def gate_couette(n_fluid=16, nx=4, nz=4, u_wall=0.05, tau=0.8,
                 steps=8000, tol=1e-10, check_every=200, min_steps=1000):
    ny = n_fluid + 2
    solid = M.channel_masks_3d(nz, ny, nx, solid_rows=1)
    wall_u = M.wall_velocity_3d(solid, (u_wall, 0.0, 0.0), row=-1)
    f = _channel_initial(nz, ny, nx)

    history = []
    t0 = time.time()
    prev = None
    converged_at = None
    step = 0
    while step < steps:
        f = M.fused_step(f, tau, solid=solid, wall_u=wall_u)
        step += 1
        if step % check_every == 0:
            _, ux, _, _ = M.macroscopic(f)
            prof = ux[:, 1:-1, :].mean(dim=(0, 2))
            if prev is not None:
                d = float((prof - prev).abs().max()) / u_wall
                history.append([step, d])
                if d < tol and step >= min_steps:
                    converged_at = step
                    break
            prev = prof
    wall_clock = time.time() - t0

    _, ux, _, _ = M.macroscopic(f)
    prof = ux[:, 1:-1, :].mean(dim=(0, 2))
    exact = M.analytic_couette(n_fluid, u_wall)
    err = _profile_error(prof, exact, u_wall)
    return {
        "n_fluid": n_fluid, "u_wall": u_wall, "tau": tau, "steps": step,
        "converged_at": converged_at, "wall_clock_s": wall_clock,
        "profile": prof.tolist(), "exact": exact.tolist(),
        "l2": err["l2"], "max": err["max"],
        "history_tail": history[-3:],
        "u_center": float(prof[n_fluid // 2]),
        "u_center_exact": float(exact[n_fluid // 2]),
    }


# --------------------------------------------------------------------------- #
# gate 3 — plane Poiseuille (Guo forcing, exact parabola)
# --------------------------------------------------------------------------- #
def gate_poiseuille(n_fluid=16, nx=4, nz=4, g=1.5625e-4, tau=0.8,
                    steps=40000, tol=1e-9, check_every=500, min_steps=2000):
    ny = n_fluid + 2
    nu = (tau - 0.5) / 3.0
    solid = M.channel_masks_3d(nz, ny, nx, solid_rows=1)
    force = torch.zeros((3, nz, ny, nx), dtype=torch.float64)
    force[0] = g
    f = _channel_initial(nz, ny, nx)

    history = []
    t0 = time.time()
    prev = None
    converged_at = None
    step = 0
    while step < steps:
        f = M.fused_step(f, tau, solid=solid, force=force)
        step += 1
        if step % check_every == 0:
            _, ux, _, _ = M.macroscopic(f, force)
            prof = ux[:, 1:-1, :].mean(dim=(0, 2))
            if prev is not None:
                d = float((prof - prev).abs().max())
                history.append([step, d])
                if d < tol and step >= min_steps:
                    converged_at = step
                    break
            prev = prof
    wall_clock = time.time() - t0

    _, ux, _, _ = M.macroscopic(f, force)
    prof = ux[:, 1:-1, :].mean(dim=(0, 2))
    exact, u_max = M.analytic_poiseuille(n_fluid, g, nu)
    err = _profile_error(prof, exact, u_max)
    return {
        "n_fluid": n_fluid, "g": g, "tau": tau, "nu": nu, "steps": step,
        "converged_at": converged_at, "wall_clock_s": wall_clock,
        "u_max_analytic": u_max, "u_max_measured": float(prof.max()),
        "profile": prof.tolist(), "exact": exact.tolist(),
        "l2": err["l2"], "max": err["max"],
        "history_tail": history[-3:],
    }


# --------------------------------------------------------------------------- #
# gate 4 — decaying shear wave: measures the transport coefficient nu
# --------------------------------------------------------------------------- #
def gate_shear_wave(ny=64, nx=2, nz=2, u0=0.02, tau=0.8, periods=1, steps=200,
                    sub_sample=(50, 100, 200)):
    """Amplitude decay of a transverse mode -> effective viscosity."""
    nu = (tau - 0.5) / 3.0
    k_lat = 2.0 * math.pi * periods / ny
    f = M.shear_wave_initial(nz, ny, nx, u0, k_lat)

    t0 = time.time()
    samples = []
    t = 0
    for T in sub_sample:
        while t < T:
            f = M.fused_step(f, tau)
            t += 1
        _, ux, _, _ = M.macroscopic(f)
        prof = ux[:, :, 0]                            # (nz, ny) — nx is uniform
        amp = float(torch.sqrt(torch.mean(prof ** 2)))
        exact = M.shear_wave_exact(ny, float(T), u0, k_lat, nu)
        amp_exact = float(torch.sqrt(torch.mean(exact ** 2)))
        samples.append({
            "t": T, "amp": amp, "amp_exact": amp_exact,
            "nu_eff": -math.log(amp / (u0 / math.sqrt(2))) / (k_lat ** 2 * T),
            "profile_l2": M.relative_l2(prof.mean(dim=0), exact),
        })
    wall_clock = time.time() - t0

    nu_eff = samples[-1]["nu_eff"]
    return {
        "ny": ny, "u0": u0, "tau": tau, "nu_analytic": nu,
        "k_lat": k_lat, "steps": t, "wall_clock_s": wall_clock,
        "samples": samples,
        "nu_eff": nu_eff,
        "nu_rel_error": abs(nu_eff - nu) / nu,
        "profile_l2": samples[-1]["profile_l2"],
        "profile_l2_at_early_time": samples[0]["profile_l2"],
    }


# --------------------------------------------------------------------------- #
# gate 5 — 2D-extended Taylor-Green vortex (exact 3D Navier-Stokes solution)
# --------------------------------------------------------------------------- #
def gate_taylor_green(n=64, nz=2, u0=0.02, tau=0.8, steps=None, decay_steps_at_32=90):
    """One grid of the TG sweep.

    The number of periods over the domain is held at one (``k_lat = 2 pi / n``),
    i.e. the physical problem is fixed and the grid is refined; because the
    viscous rate scales as ``k_lat^2``, the time window is scaled as ``n^2`` so
    that every run sees the *same* decay factor and can be compared at equal
    physical time.
    """
    nu = (tau - 0.5) / 3.0
    k_lat = 2.0 * math.pi / n
    if steps is None:
        steps = max(20, int(round(decay_steps_at_32 * (n / 32.0) ** 2)))
    f = M.taylor_green_initial(nz, n, u0, k_lat)
    t0 = time.time()
    for _ in range(steps):
        f = M.fused_step(f, tau)
    wall_clock = time.time() - t0

    rho, ux, uy, uz = M.macroscopic(f)
    ex, ey, ez = M.taylor_green_exact(n, float(steps), u0, k_lat, nu)
    num = torch.stack([ux[0], uy[0]])
    ana = torch.stack([ex, ey])
    amp_num = float(torch.sqrt(torch.mean(ux[0] ** 2 + uy[0] ** 2)))
    amp_ana = float(torch.sqrt(torch.mean(ex ** 2 + ey ** 2)))
    decay_exact = math.exp(-2.0 * nu * k_lat ** 2 * steps)
    return {
        "n": n, "nz": nz, "u0": u0, "tau": tau, "nu": nu, "steps": steps,
        "k_lat": k_lat, "decay_factor_exact": decay_exact,
        "wall_clock_s": wall_clock,
        "amp_measured": amp_num, "amp_exact": amp_ana,
        "decay_measured": amp_num / (u0 * math.sqrt(0.5)),
        "decay_rel_error": abs(amp_num - amp_ana) / amp_ana,
        "l2": M.relative_l2(num, ana),
        "max_norm_diff": float((num - ana).abs().max()),
        "rho_min": float(rho.min()), "rho_max": float(rho.max()),
        "uz_max": float(uz.abs().max()),
        "mass_drift": abs(float(rho[0].mean()) - 1.0),
    }


def gate_taylor_green_sweep(ns=(32, 64, 128), **kw):
    runs = [gate_taylor_green(n=n, **kw) for n in ns]
    orders = []
    for a, b in zip(runs[:-1], runs[1:]):
        if a["l2"] > 0 and b["l2"] > 0:
            orders.append({
                "pair": [a["n"], b["n"]],
                "order": math.log(a["l2"] / b["l2"]) / math.log(b["n"] / a["n"]),
            })
    return {"runs": runs, "orders": orders}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="all",
                    choices=["all", "identity", "couette", "poiseuille", "shear_wave",
                             "taylor_green"])
    ap.add_argument("--out", default="artifacts/xval_aa_pattern/d3q19_verify.json")
    ap.add_argument("--quick", action="store_true", help="short run for CI/smoke")
    args = ap.parse_args(argv)

    torch.set_grad_enabled(False)
    res = {"anchor": "XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE",
           "module": "tensorlbm.aa_fused_push_pull_3d",
           "quick": bool(args.quick)}

    if args.gate in ("all", "identity"):
        res["identity"] = gate_identity()
    if args.gate in ("all", "couette"):
        res["couette"] = gate_couette(
            steps=2000 if args.quick else 8000, tol=1e-6 if args.quick else 1e-10)
    if args.gate in ("all", "poiseuille"):
        res["poiseuille"] = gate_poiseuille(
            steps=4000 if args.quick else 40000, tol=1e-4 if args.quick else 1e-9)
    if args.gate in ("all", "shear_wave"):
        res["shear_wave"] = gate_shear_wave(
            steps=100 if args.quick else 200,
            sub_sample=(50, 100) if args.quick else (50, 100, 200))
    if args.gate in ("all", "taylor_green"):
        res["taylor_green"] = gate_taylor_green_sweep(
            ns=(32, 64) if args.quick else (32, 64, 128))

    # explicit gate verdicts (owner standard: < 15 % unacceptable, < 3 % target)
    verdicts = {}
    if "couette" in res:
        verdicts["couette_l2"] = res["couette"]["l2"]
    if "poiseuille" in res:
        verdicts["poiseuille_l2"] = res["poiseuille"]["l2"]
    if "shear_wave" in res:
        verdicts["shear_wave_nu_rel_error"] = res["shear_wave"]["nu_rel_error"]
        verdicts["shear_wave_profile_l2"] = res["shear_wave"]["profile_l2"]
    if "taylor_green" in res:
        verdicts["taylor_green_l2"] = {r["n"]: r["l2"] for r in res["taylor_green"]["runs"]}
    if "identity" in res:
        verdicts["bit_equal"] = res["identity"]["bit_equal"]
    res["verdicts"] = verdicts
    flat = {}
    for k, v in verdicts.items():
        if isinstance(v, float):
            flat[k] = v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}[{kk}]"] = vv
    res["gates_under_3pct"] = {k: (v < 0.03) for k, v in flat.items()}
    res["gates_under_15pct"] = {k: (v < 0.15) for k, v in flat.items()}

    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", args.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "taylor_green"}, indent=1)[:4000])
    if "taylor_green" in res:
        print(json.dumps({"taylor_green_summary": [
            {"n": r["n"], "l2": r["l2"], "decay_rel_error": r["decay_rel_error"],
             "rho_range": [r["rho_min"], r["rho_max"]]}
            for r in res["taylor_green"]["runs"]],
            "orders": res["taylor_green"]["orders"]}, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())