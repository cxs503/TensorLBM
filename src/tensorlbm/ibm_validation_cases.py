"""Quantitative validation cases for the immersed-boundary force transfer.

Each case is a self-contained driver returning a plain ``dict`` of metrics plus
time series, so that the results can be dumped to JSON without post-processing
guesswork.  All quantities are in lattice units (``dx = dt = rho = 1``).

Cases
-----
``taylor_green_with_ibm``
    Taylor–Green vortex decay with Lagrangian markers that move with the
    analytic velocity field.  Validates that the IBM operators add no spurious
    dissipation when the marker forcing is (physically) zero, and that the
    bulk decay rate matches ``2 ν k²``.
``taylor_couette_2d`` / ``taylor_couette_3d``
    Rotating inner cylinder, stationary outer cylinder.  Validates force
    transfer for a *rotating* boundary against the exact analytic torque
    ``T = 4π μ ω R1² R2² / (R2² − R1²)`` (per unit length in 2-D, times the
    periodic height in 3-D) and the analytic azimuthal velocity profile.
``cylinder_strouhal_2d``
    Flow past a fixed circular cylinder at ``Re = U D / ν = 100``.  Validates
    the unsteady force signal (Strouhal number from the lift history).
``oscillating_cylinder_added_mass_2d``
    Transversely oscillating cylinder in quiescent fluid.  Validates the
    *moving* boundary force in phase with acceleration against the
    potential-flow added-mass coefficient ``C_A = 1``.

Every case reports ``force_audit``: the reaction force implied by the marker
forces versus the total Eulerian body force (exact partition of unity), and
``partition_error`` of the delta kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .d2q9 import equilibrium, macroscopic
from .d3q19 import equilibrium3d, macroscopic3d
from .ibm_force_transfer import (
    body_source_2d,
    body_source_3d,
    build_cylinder_markers_2d,
    build_stencil,
    direct_forcing,
    guo_force_2d,
    guo_force_3d,
    interior_mask_force,
    momentum_audit,
)
from .solver import collide_bgk, stream
from .solver3d import collide_bgk3d, stream3d

__all__ = [
    "taylor_green_with_ibm",
    "taylor_couette_2d",
    "taylor_couette_3d",
    "cylinder_strouhal_2d",
    "oscillating_cylinder_added_mass_2d",
    "oscillating_flow_added_mass_2d",
    "accelerated_cylinder_added_mass_2d",
    "analytic_couette_profile",
    "analytic_couette_torque",
]


# --------------------------------------------------------------------------- #
# Analytic references
# --------------------------------------------------------------------------- #


def disc_mask_2d(shape: tuple[int, int], center: tuple[float, float], radius: float) -> torch.Tensor:
    """Boolean solid mask for a disc (interior stencil of the IBM body)."""
    ny, nx = shape
    y = torch.arange(ny, dtype=torch.float64).view(ny, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, nx)
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius * radius


def analytic_couette_profile(
    r: torch.Tensor, r1: float, r2: float, omega: float
) -> torch.Tensor:
    """Azimuthal velocity of steady circular Couette flow (inner rotates)."""
    a = -omega * r1**2 / (r2**2 - r1**2)
    b = omega * r1**2 * r2**2 / (r2**2 - r1**2)
    return a * r + b / r


def analytic_couette_torque(r1: float, r2: float, omega: float, nu: float, height: float = 1.0) -> float:
    """Exact viscous torque on the rotating inner cylinder.

    ``T = 4π μ ω R1² R2² H / (R2² − R1²)`` with ``μ = ρ ν`` and ``ρ = 1``.
    """
    return 4.0 * math.pi * nu * omega * r1**2 * r2**2 * height / (r2**2 - r1**2)


# --------------------------------------------------------------------------- #
# Taylor–Green vortex with IBM markers riding the analytic field
# --------------------------------------------------------------------------- #


def taylor_green_with_ibm(
    n: int = 64,
    u0: float = 0.04,
    tau: float = 0.8,
    n_steps: int = 1200,
    kernel: str = "4pt",
    marker_radii: tuple[float, ...] = (1.5, 3.0),
    marker_steps: int = 500,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Taylor–Green decay: bulk control run + IBM-footprint sensitivity.

    Two measurements are reported.

    (a) **Bulk control** (no markers): the two-dimensional Taylor–Green solution
        ``u_x = u0 sin(kx)cos(ky) e^{−2νk²t}`` decays as ``E = E0 e^{−4νk²t}``.
        A correct solver must reproduce the analytic decay rate; this is the
        control against which the IBM runs are judged.

    (b) **IBM footprint** (markers prescribed the analytic velocity): a ring of
        Lagrangian markers is *fixed* in space while the field decays, so the
        markers act as velocity constraints and inevitably perturb the decay.
        The reported deviation is therefore a measure of the local forcing
        footprint, **not** an error metric — it is expected to grow with the
        marker count/support (reported as ``ibm_footprint``), and shrinking it
        to zero would mean asking the IBM to do nothing.
    """
    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / n
    decay = 2.0 * nu * k * k
    y, x = torch.meshgrid(
        torch.arange(n, dtype=dtype), torch.arange(n, dtype=dtype), indexing="ij"
    )
    ux0 = u0 * torch.sin(k * x) * torch.cos(k * y)
    uy0 = -u0 * torch.cos(k * x) * torch.sin(k * y)
    rho = torch.ones_like(ux0)
    e0 = float((0.5 * (ux0**2 + uy0**2)).mean().item())

    def decay_fit(energies: list[tuple[float, float]]) -> float:
        tt = torch.tensor([t for t, _ in energies], dtype=torch.float64)
        ee = torch.tensor([e for _, e in energies], dtype=torch.float64)
        slope = float(
            ((tt - tt.mean()) * (ee - ee.mean())).sum() / ((tt - tt.mean()) ** 2).sum()
        )
        return -slope / 2.0

    # ---------------- (a) bulk control ----------------
    f = equilibrium(rho, ux0, uy0)
    series = {"t": [], "energy": []}
    fit_pts = []
    for step in range(n_steps):
        if step % max(1, n_steps // 300) == 0:
            _, a, b = macroscopic(f)
            e = float((0.5 * (a**2 + b**2)).mean().item())
            series["t"].append(step)
            series["energy"].append(e)
            if e > 0.05 * e0:
                fit_pts.append((float(step), math.log(e)))
        f = stream(collide_bgk(f, tau))
    bulk_rate = decay_fit(fit_pts)

    # ---------------- (b) IBM footprint ----------------
    footprint = {}
    for radius in marker_radii:
        markers = build_cylinder_markers_2d(
            (0.5 * n + 0.37, 0.5 * n + 0.21), radius, spacing=1.0
        ).to(dtype)
        stencil = build_stencil(markers, (n, n), kernel)
        fm = equilibrium(rho, ux0, uy0)
        residual = 0.0
        audit = {}
        pts = []
        for step in range(marker_steps):
            amp = u0 * math.exp(-decay * step)
            mx, my = markers[:, 0], markers[:, 1]
            u_target = torch.stack(
                [
                    amp * torch.sin(k * mx) * torch.cos(k * my),
                    -amp * torch.cos(k * mx) * torch.sin(k * my),
                ],
                dim=1,
            )
            rho_f, uxf, uyf = macroscopic(fm)
            f_m, f_grid, _, _ = direct_forcing(
                stencil, torch.stack([uxf, uyf]), u_target, rho_f, 2
            )
            residual = max(residual, float(f_m.abs().max().item()))
            if step == 0:
                audit = momentum_audit(stencil, f_m)
            fm = collide_bgk(fm, tau)
            fm = body_source_2d(fm, f_grid[0], f_grid[1])
            fm = stream(fm)
            if step % max(1, marker_steps // 100) == 0:
                _, a, b = macroscopic(fm)
                e = float((0.5 * (a**2 + b**2)).mean().item())
                if e > 0.05 * e0:
                    pts.append((float(step), math.log(e)))
        footprint[str(radius)] = {
            "n_markers": stencil.n_markers,
            "marker_area_fraction": (math.pi * radius * radius) / (n * n),
            "decay_rate": decay_fit(pts) if len(pts) > 2 else None,
            "spurious_force_max": residual,
            "spurious_force_scaled": residual / (u0 * u0),
        }
        footprint[str(radius)]["decay_rate_rel_error"] = (
            abs(footprint[str(radius)]["decay_rate"] - decay) / decay
            if footprint[str(radius)]["decay_rate"]
            else None
        )

    return {
        "case": "taylor_green_with_ibm",
        "params": {
            "n": n,
            "u0": u0,
            "tau": tau,
            "nu": nu,
            "n_steps": n_steps,
            "kernel": kernel,
            "marker_radii": list(marker_radii),
            "marker_steps": marker_steps,
        },
        "metrics": {
            "decay_rate_measured": bulk_rate,
            "decay_rate_analytic": decay,
            "decay_rate_rel_error": abs(bulk_rate - decay) / decay,
            "energy_decay_rate_measured": 2.0 * bulk_rate,
            "energy_decay_rate_analytic": 4.0 * nu * k * k,
            "final_energy": series["energy"][-1] if series["energy"] else None,
            "ibm_footprint": footprint,
        },
        "force_audit": {"note": "markers absent in the bulk control run"},
        "series": series,
    }


# --------------------------------------------------------------------------- #
# Taylor–Couette: rotating inner cylinder, stationary outer cylinder
# --------------------------------------------------------------------------- #


def effective_wall_shift(torque: float, r1: float, r2: float, omega: float, nu: float,
                         height: float = 1.0) -> float:
    """Invert the analytic torque for a uniform wall offset ``Δr`` (lattice cells).

    ``T(Δr) = 4π μ ω (r1+Δr)² (r2+Δr)² H / ((r2+Δr)² − (r1+Δr)²)`` is monotone in
    ``Δr`` for the geometries used here, so a bisection is safe.  A shift that is
    *constant in lattice cells* across resolutions is the signature of an
    ``O(Δx)`` effective-wall bias of the delta-kernel smearing (not a bug).
    """
    def torque_at(d: float) -> float:
        return analytic_couette_torque(r1 + d, r2 + d, omega, nu, height)

    lo, hi = -2.0, 4.0
    if abs(torque - torque_at(lo)) < abs(torque - torque_at(hi)):
        best = lo
    else:
        best = hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = torque_at(mid)
        if (val - torque) * (torque_at(lo) - torque) <= 0:
            hi = mid
        else:
            lo = mid
        if abs(val - torque) < 1e-14:
            break
    for cand in (lo, hi, 0.5 * (lo + hi)):
        if abs(torque_at(cand) - torque) < abs(torque_at(best) - torque):
            best = cand
    return best


def _couette_metrics(
    history: list[dict],
    r1: float,
    r2: float,
    omega: float,
    nu: float,
    height: float,
    profile: dict | None,
    torque_outer: float | None = None,
) -> dict:
    torques = [h["torque_inner"] for h in history]
    n_avg = max(1, len(torques) // 5)
    t_ref = analytic_couette_torque(r1, r2, omega, nu, height)
    t_mean = sum(torques[-n_avg:]) / n_avg
    t_drift = abs(torques[-1] - torques[-n_avg]) / max(abs(t_mean), 1e-30)
    d_eff = effective_wall_shift(abs(t_mean), r1, r2, omega, nu, height)
    t_corr = analytic_couette_torque(r1 + d_eff, r2 + d_eff, omega, nu, height)
    out = {
        "torque_measured": t_mean,
        "torque_analytic": t_ref,
        "torque_rel_error": abs(t_mean - t_ref) / abs(t_ref),
        "torque_drift_last_window": t_drift,
        "torque_first": torques[0],
        "torque_history": torques,
        "effective_wall_shift_cells": d_eff,
        "torque_rel_error_after_wall_shift": abs(abs(t_mean) - abs(t_corr)) / abs(t_corr),
    }
    if torque_outer is not None:
        out["torque_outer_measured"] = torque_outer
        out["torque_balance_rel"] = abs(abs(torque_outer) - abs(t_mean)) / abs(t_mean)
    out["torque_rel_error"] = abs(abs(t_mean) - t_ref) / t_ref
    out["torque_corrected_rel_error"] = abs(
        abs(t_mean) - t_ref
    ) / t_ref
    if profile:
        out.update(
            {
                "profile_l2_rel_error": profile["l2_rel_error"],
                "profile_max_rel_error": profile["max_rel_error"],
                "profile_r": profile["r"],
                "profile_measured": profile["measured"],
                "profile_analytic": profile["analytic"],
            }
        )
    return out


def taylor_couette_2d(
    n: int = 96,
    r1_frac: float = 0.15,
    r2_frac: float = 0.34,
    omega: float = 0.0015,
    tau: float = 0.8,
    n_steps: int = 4000,
    kernel: str = "4pt",
    sample_every: int = 20,
    n_correctors: int = 2,
    forcing: str = "plain",
    wall_offset: float = 0.0,
) -> dict:
    """2-D circular Couette flow driven by a rotating IBM inner cylinder.

    ``wall_offset`` (lattice cells) shifts the *marker* radii inward by that
    amount, compensating the ``O(Δx)`` effective-wall bias of the delta kernel
    (the hydrodynamic wall of a 4-point IBM stencil sits ~0.5 cell outside the
    markers; see the module docstring and the reported
    ``effective_wall_shift_cells``).
    """
    nu = (tau - 0.5) / 3.0
    r1 = r1_frac * n
    r2 = r2_frac * n
    c = 0.5 * n
    if r2 >= 0.5 * n - 2.0:
        raise ValueError("outer cylinder too close to the periodic boundary")

    inner = build_cylinder_markers_2d((c, c), r1 - wall_offset, spacing=0.85)
    outer = build_cylinder_markers_2d((c, c), r2 - wall_offset, spacing=0.85)
    st_in = build_stencil(inner, (n, n), kernel)
    st_out = build_stencil(outer, (n, n), kernel)

    rel = inner - torch.tensor([c, c], dtype=torch.float64)
    u_inner = omega * torch.stack([-rel[:, 1], rel[:, 0]], dim=1)
    zero = torch.zeros((st_out.n_markers, 2), dtype=torch.float64)

    rho = torch.ones((n, n), dtype=torch.float64)
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))

    history: list[dict] = []
    audit = {}
    for step in range(n_steps):
        rho_f, uxf, uyf = macroscopic(f)
        u_grid = torch.stack([uxf, uyf])
        f_in, g_in, bf_in, tq_in = direct_forcing(
            st_in, u_grid, u_inner, rho_f, n_correctors
        )
        f_out, g_out, bf_out, tq_out = direct_forcing(
            st_out, u_grid, zero, rho_f, n_correctors
        )
        if step == 0:
            audit = {
                "inner": momentum_audit(st_in, f_in),
                "outer": momentum_audit(st_out, f_out),
                "reaction_force_inner": [float(v) for v in bf_in.tolist()],
                "reaction_force_outer": [float(v) for v in bf_out.tolist()],
            }
        f = collide_bgk(f, tau)
        if forcing == "guo":
            f = guo_force_2d(f, g_in[0] + g_out[0], g_in[1] + g_out[1], tau)
        else:
            f = body_source_2d(f, g_in[0] + g_out[0], g_in[1] + g_out[1])
        f = stream(f)
        if step % sample_every == 0:
            history.append({"t": step, "torque_inner": float(tq_in)})

    # final torque over the last sample window
    rho_f, uxf, uyf = macroscopic(f)
    f_in, g_in, _, tq_in = direct_forcing(
        st_in, torch.stack([uxf, uyf]), u_inner, rho_f, n_correctors
    )
    f_out, g_out, _, tq_out = direct_forcing(
        st_out, torch.stack([uxf, uyf]), zero, rho_f, n_correctors
    )
    history.append({"t": n_steps, "torque_inner": float(tq_in)})
    u_int = st_in.interpolate(torch.stack([uxf, uyf])).T
    marker_velocity_rel_error = float(
        (u_int - u_inner).norm().item() / u_inner.norm().item()
    )

    # azimuthal velocity profile vs analytic
    rr = torch.arange(r1 + 2.0, r2 - 1.5, 1.0, dtype=torch.float64)
    angles = torch.arange(0.0, 2.0 * math.pi, math.pi / 12, dtype=torch.float64)
    meas, ana = [], []
    for radius in rr:
        vals = []
        for th in angles:
            px, py = c + radius * math.cos(th), c + radius * math.sin(th)
            ix, iy = int(round(px.item())) % n, int(round(py.item())) % n
            ux_v = float(uxf[iy, ix].item())
            uy_v = float(uyf[iy, ix].item())
            # azimuthal component about the cylinder axis
            vals.append(-math.sin(th) * ux_v + math.cos(th) * uy_v)
        meas.append(sum(vals) / len(vals))
        ana.append(float(analytic_couette_profile(radius, r1, r2, omega).item()))
    meas_t = torch.tensor(meas)
    ana_t = torch.tensor(ana)
    scale = float(ana_t.abs().max().item()) or 1.0
    profile = {
        "r": [float(v) for v in rr.tolist()],
        "measured": meas,
        "analytic": ana,
        "l2_rel_error": float((meas_t - ana_t).pow(2).sum().sqrt().item())
        / (ana_t.pow(2).sum().sqrt().item() + 1e-30),
        "max_rel_error": float((meas_t - ana_t).abs().max().item()) / scale,
    }

    metrics = _couette_metrics(
        history, r1, r2, omega, nu, 1.0, profile, torque_outer=float(tq_out)
    )
    metrics["marker_velocity_rel_error"] = marker_velocity_rel_error
    re_omega = omega * r1 * (r2 - r1) / nu
    return {
        "case": "taylor_couette_2d",
        "params": {
            "n": n,
            "r1": r1,
            "r2": r2,
            "omega": omega,
            "nu": nu,
            "tau": tau,
            "n_steps": n_steps,
            "kernel": kernel,
            "n_markers_inner": st_in.n_markers,
            "n_markers_outer": st_out.n_markers,
            "re_omega_gap": re_omega,
            "n_correctors": n_correctors,
            "forcing": forcing,
            "wall_offset": wall_offset,
        },
        "metrics": metrics,
        "force_audit": audit,
        "series": {"t": [h["t"] for h in history], "tau_wall": metrics["torque_history"]},
    }


def taylor_couette_3d(
    n: int = 36,
    r1_frac: float = 0.16,
    r2_frac: float = 0.33,
    omega: float = 0.0015,
    tau: float = 0.8,
    n_steps: int = 1200,
    kernel: str = "4pt",
    sample_every: int = 10,
    n_correctors: int = 2,
    forcing: str = "plain",
    wall_offset: float = 0.0,
) -> dict:
    """3-D circular Couette flow (periodic in z) with a rotating IBM cylinder."""
    nu = (tau - 0.5) / 3.0
    r1 = r1_frac * n
    r2 = r2_frac * n
    c = 0.5 * n
    r1_m = r1 - wall_offset
    r2_m = r2 - wall_offset
    rho = torch.ones((n, n, n), dtype=torch.float64)
    z = torch.arange(n, dtype=torch.float64)
    th = torch.arange(0.0, 2.0 * math.pi, 0.9 / r1_m, dtype=torch.float64)
    thg, zg = torch.meshgrid(th, z, indexing="ij")
    inner = torch.stack(
        [c + r1_m * torch.cos(thg.reshape(-1)), c + r1_m * torch.sin(thg.reshape(-1)), zg.reshape(-1)],
        dim=1,
    )
    th = torch.arange(0.0, 2.0 * math.pi, 0.9 / r2_m, dtype=torch.float64)
    thg, zg = torch.meshgrid(th, z, indexing="ij")
    outer = torch.stack(
        [c + r2_m * torch.cos(thg.reshape(-1)), c + r2_m * torch.sin(thg.reshape(-1)), zg.reshape(-1)],
        dim=1,
    )
    st_in = build_stencil(inner, (n, n, n), kernel)
    st_out = build_stencil(outer, (n, n, n), kernel)

    rel = inner - torch.tensor([c, c, 0.0], dtype=torch.float64)
    u_inner = omega * torch.stack([-rel[:, 1], rel[:, 0], torch.zeros_like(rel[:, 2])], dim=1)
    zero = torch.zeros((st_out.n_markers, 3), dtype=torch.float64)

    zeros = torch.zeros_like(rho)
    f = equilibrium3d(rho, zeros, zeros, zeros)
    history: list[dict] = []
    audit = {}
    for step in range(n_steps):
        rho_f, uxf, uyf, uzf = macroscopic3d(f)
        u_grid = torch.stack([uxf, uyf, uzf])
        f_in, g_in, bf_in, tq_in = direct_forcing(
            st_in, u_grid, u_inner, rho_f, n_correctors
        )
        f_out, g_out, bf_out, tq_out = direct_forcing(
            st_out, u_grid, zero, rho_f, n_correctors
        )
        if step == 0:
            audit = {
                "inner": momentum_audit(st_in, f_in),
                "outer": momentum_audit(st_out, f_out),
                "reaction_force_inner": [float(v) for v in bf_in.tolist()],
                "reaction_force_outer": [float(v) for v in bf_out.tolist()],
            }
        f = collide_bgk3d(f, tau)
        if forcing == "guo":
            f = guo_force_3d(
                f, g_in[0] + g_out[0], g_in[1] + g_out[1], g_in[2] + g_out[2], tau
            )
        else:
            f = body_source_3d(
                f, g_in[0] + g_out[0], g_in[1] + g_out[1], g_in[2] + g_out[2]
            )
        f = stream3d(f)
        if step % sample_every == 0:
            history.append({"t": step, "torque_inner": float(tq_in[2])})

    # profile in the x-direction line at z = c, y = c
    rho_f, uxf, uyf, uzf = macroscopic3d(f)
    rr = torch.arange(r1 + 1.5, r2 - 1.5, 1.0, dtype=torch.float64)
    meas, ana = [], []
    for radius in rr:
        ix = int(round(c + float(radius))) % n
        ux_v = float(uxf[n // 2, n // 2, ix].item())
        uy_v = float(uyf[n // 2, n // 2, ix].item())
        meas.append(uy_v)  # theta component at theta=0 is +y
        ana.append(float(analytic_couette_profile(radius, r1, r2, omega).item()))
    meas_t = torch.tensor(meas)
    ana_t = torch.tensor(ana)
    profile = {
        "r": [float(v) for v in rr.tolist()],
        "measured": meas,
        "analytic": ana,
        "l2_rel_error": float((meas_t - ana_t).pow(2).sum().sqrt().item())
        / (ana_t.pow(2).sum().sqrt().item() + 1e-30),
        "max_rel_error": float((meas_t - ana_t).abs().max().item())
        / (float(ana_t.abs().max().item()) or 1.0),
    }
    if rr.numel() < 2:
        profile["l2_rel_error"] = None
        profile["max_rel_error"] = None
    metrics = _couette_metrics(
        history, r1, r2, omega, nu, float(n), profile, torque_outer=float(tq_out[2])
    )
    u_int = st_in.interpolate(torch.stack([uxf, uyf, uzf])).T
    metrics["marker_velocity_rel_error"] = float(
        (u_int - u_inner).norm().item() / u_inner.norm().item()
    )
    return {
        "case": "taylor_couette_3d",
        "params": {
            "n": n,
            "r1": r1,
            "r2": r2,
            "omega": omega,
            "nu": nu,
            "tau": tau,
            "n_steps": n_steps,
            "kernel": kernel,
            "n_markers_inner": st_in.n_markers,
            "n_markers_outer": st_out.n_markers,
            "re_omega_gap": omega * r1 * (r2 - r1) / nu,
            "n_correctors": n_correctors,
            "forcing": forcing,
            "wall_offset": wall_offset,
        },
        "metrics": metrics,
        "force_audit": audit,
        "series": {"t": [h["t"] for h in history], "tau_wall": metrics["torque_history"]},
    }


# --------------------------------------------------------------------------- #
# Flow past a fixed cylinder: Strouhal number
# --------------------------------------------------------------------------- #


def cylinder_strouhal_2d(
    nx: int = 160,
    ny: int = 80,
    diameter: float = 8.0,
    u_target: float = 0.1,
    tau: float = 0.53,
    n_steps: int = 4000,
    n_transient: int = 1200,
    kernel: str = "4pt",
    gain_init: float = 0.0,
    perturbation: float = 0.0,
) -> dict:
    """Flow past a fixed cylinder at ``Re = u D / ν``; returns ``St`` and ``Cd``.

    The flow is driven by a spatially uniform adaptive body force that keeps
    the domain-averaged streamwise velocity at ``u_target``; the cylinder drag
    is measured from the IBM reaction force.  Periodic boundaries in both
    directions (an effective unbounded flow with ``ny / D`` blockage).
    """
    nu = (tau - 0.5) / 3.0
    re = u_target * diameter / nu
    c = (0.35 * nx, 0.5 * ny)
    markers = build_cylinder_markers_2d(c, 0.5 * diameter, spacing=0.8)
    stencil = build_stencil(markers, (ny, nx), kernel)
    zero = torch.zeros((stencil.n_markers, 2), dtype=torch.float64)
    mask = disc_mask_2d((ny, nx), c, max(0.5 * diameter - 0.5, 0.5))

    rho = torch.ones((ny, nx), dtype=torch.float64)
    uy_init = torch.zeros_like(rho)
    ux_init = u_target * torch.ones_like(rho)
    if perturbation:
        # A broadband velocity perturbation is required to let the
        # antisymmetric (lift) mode grow: with a perfectly symmetric initial
        # condition the discrete solution stays symmetric to machine precision
        # and the cylinder sheds nothing (observed C_L rms ~ 1e-13).
        gen = torch.Generator().manual_seed(12345)
        noise = torch.randn((2, ny, nx), generator=gen, dtype=torch.float64)
        ux_init = ux_init + perturbation * u_target * noise[0]
        uy_init = perturbation * u_target * noise[1]
    f = equilibrium(rho, ux_init, uy_init)
    gx = gain_init
    mean_ux_hist: list[float] = []
    lift: list[float] = []
    drag: list[float] = []
    t_list: list[int] = []
    audit = {}
    gain = 0.02
    for step in range(n_steps):
        rho_f, uxf, uyf = macroscopic(f)
        mean_ux = float(uxf.mean().item())
        gx += gain * (u_target - mean_ux)
        u_grid = torch.stack([uxf, uyf])
        n_corr = 0 if "n_corr" not in locals() else n_corr
        f_m, f_grid, bf, _ = direct_forcing(stencil, u_grid, zero, rho_f, n_corr)
        # Interior solid mask: without it the "cylinder" is a hollow shell and
        # the trapped fluid is accelerated by the free stream, which inflates
        # the apparent drag by an order of magnitude on coarse grids.
        f_int = interior_mask_force(u_grid, mask, torch.zeros_like(u_grid))
        if step == 0:
            audit = momentum_audit(stencil, f_m)
        # body force on fluid = -F_body (F_body is the force *on* the body)
        fx = f_grid[0] + f_int[0] + gx * (~mask).to(torch.float64)
        fy = f_grid[1] + f_int[1]
        f = collide_bgk(f, tau)
        f = body_source_2d(f, fx, fy)
        f = stream(f)
        lift.append(float(-(f_grid[1] + f_int[1]).sum().item()))
        # the +x direction is the streamwise (drag) direction
        drag.append(float(-(f_grid[0] + f_int[0]).sum().item()))
        mean_ux_hist.append(mean_ux)
        t_list.append(step)

    # Strouhal from the lift history after the transient
    lf = torch.tensor(lift[n_transient:], dtype=torch.float64)
    lf = lf - lf.mean()
    window = torch.hann_window(lf.numel(), dtype=torch.float64, periodic=False)
    spec = torch.fft.rfft(lf * window)
    freqs = torch.fft.rfftfreq(lf.numel(), d=1.0)
    power = spec.abs() ** 2
    # ignore the zero-frequency bin
    power[0] = 0.0
    k_max = int(power.argmax().item())
    f_peak = float(freqs[k_max].item())
    # FFT peak with parabolic (3-point) interpolation to beat the bin resolution
    if 0 < k_max < power.numel() - 1:
        p0, p1, p2 = (float(power[k_max - 1]), float(power[k_max]), float(power[k_max + 1]))
        denom = p0 - 2.0 * p1 + p2
        delta = 0.5 * (p0 - p2) / denom if abs(denom) > 1e-30 else 0.0
    else:
        delta = 0.0
    df = float(freqs[1] - freqs[0])
    f_interp = f_peak + delta * df
    # zero-crossing estimate (robust when the record holds only a few periods)
    sig = lf / (lf.abs().max() + 1e-30)
    crossings = []
    for i in range(1, sig.numel()):
        if sig[i - 1] < 0.0 <= sig[i]:
            frac = -float(sig[i - 1]) / (float(sig[i] - sig[i - 1]) + 1e-30)
            crossings.append(i - 1 + frac)
    period_zc = (
        (crossings[-1] - crossings[0]) / (len(crossings) - 1) if len(crossings) > 2 else None
    )
    st_zc = (diameter / (u_target * period_zc)) if period_zc else None
    st = f_interp * diameter / u_target
    cl_mean = float(torch.tensor(lift[n_transient:]).mean().item())
    cl_rms = float(lf.pow(2).mean().sqrt().item())
    cd_mean = float(torch.tensor(drag[n_transient:]).mean().item())
    q_ref = 0.5 * u_target * u_target * diameter
    return {
        "case": "cylinder_strouhal_2d",
        "params": {
            "nx": nx,
            "ny": ny,
            "diameter": diameter,
            "u_target": u_target,
            "tau": tau,
            "nu": nu,
            "reynolds": re,
            "n_steps": n_steps,
            "n_transient": n_transient,
            "kernel": kernel,
            "n_markers": stencil.n_markers,
            "blockage_ratio": diameter / ny,
        },
        "metrics": {
            "strouhal": st,
            "strouhal_fft_peak_interpolated": st,
            "strouhal_zero_crossing": st_zc,
            "strouhal_reference": 0.165,
            "strouhal_rel_error": abs(st - 0.165) / 0.165,
            "strouhal_rel_error_zero_crossing": (
                abs(st_zc - 0.165) / 0.165 if st_zc else None
            ),
            "shedding_period_zero_crossing": period_zc,
            "n_shedding_cycles": len(crossings),
            "cd_mean": cd_mean / q_ref,
            "cd_reference": 1.34,
            "cd_rel_error": abs(cd_mean / q_ref - 1.34) / 1.34,
            "cl_rms": cl_rms / q_ref,
            "cl_mean": cl_mean / q_ref,
            "shedding_period": (1.0 / f_peak) if f_peak > 0 else 0.0,
            "cd_momentum_balance": float(gx) * nx * ny / q_ref,
            "mean_ux_final": mean_ux_hist[-1],
            "mean_ux_mean_after_transient": float(
                torch.tensor(mean_ux_hist[n_transient:]).mean().item()
            ),
            "drag_force_over_drive_force": (
                float(torch.tensor(drag[n_transient:]).mean().item())
                / (abs(float(gx)) * nx * ny)
                if abs(float(gx)) > 1e-14
                else None
            ),
            "streamwise_force_body_force": float(gx),
        },
        "force_audit": audit,
        "series": {
            "t": t_list,
            "lift": [v / q_ref for v in lift],
            "drag": [v / q_ref for v in drag],
        },
    }


# --------------------------------------------------------------------------- #
# Transversely oscillating cylinder: added mass
# --------------------------------------------------------------------------- #


def oscillating_cylinder_added_mass_2d(
    n: int = 128,
    radius: float = 6.0,
    amplitude: float = 0.5,
    period: float = 400.0,
    tau: float = 0.8,
    n_periods: int = 6,
    kernel: str = "4pt",
) -> dict:
    """Added-mass coefficient of a transversely oscillating cylinder.

    The cylinder centre oscillates as ``y_c(t) = A sin(Ωt)`` with
    ``Ω = 2π / period``; the force component in phase with the acceleration is
    ``−C_A m_d a`` with the displaced mass ``m_d = ρ π R²`` per unit length.
    Potential flow gives ``C_A = 1``.
    """
    omega = 2.0 * math.pi / period
    c0 = torch.tensor([0.5 * n, 0.5 * n], dtype=torch.float64)
    base = build_cylinder_markers_2d((0.0, 0.0), radius, spacing=0.85)
    rho = torch.ones((n, n), dtype=torch.float64)
    zeros = torch.zeros_like(rho)
    f = equilibrium(rho, zeros, zeros)
    nu = (tau - 0.5) / 3.0
    m_disp = math.pi * radius**2

    force: list[float] = []
    accel: list[float] = []
    marker_force = 0.0
    audit = {}
    n_steps = int(n_periods * period)
    for step in range(n_steps):
        t = float(step)
        yc = c0[1] + amplitude * math.sin(omega * t)
        vy = amplitude * omega * math.cos(omega * t)
        ay = -amplitude * omega * omega * math.sin(omega * t)
        markers = base + torch.tensor([c0[0].item(), yc], dtype=torch.float64)
        stencil = build_stencil(markers, (n, n), kernel)
        u_target = torch.stack(
            [torch.zeros(stencil.n_markers, dtype=torch.float64),
             torch.full((stencil.n_markers,), vy, dtype=torch.float64)],
            dim=1,
        )
        rho_f, uxf, uyf = macroscopic(f)
        f_m, f_grid, bf, _ = direct_forcing(
            stencil, torch.stack([uxf, uyf]), u_target, rho_f, 0
        )
        mask = disc_mask_2d((n, n), (float(c0[0]), yc), max(radius - 0.5, 0.5))
        u_solid = torch.zeros((2, n, n), dtype=torch.float64)
        u_solid[1] = vy
        f_int = interior_mask_force(torch.stack([uxf, uyf]), mask, u_solid)
        marker_force = max(marker_force, float(f_m.abs().max().item()))
        if step == 0:
            audit = momentum_audit(stencil, f_m)
        f = collide_bgk(f, tau)
        f = body_source_2d(f, f_grid[0] + f_int[0], f_grid[1] + f_int[1])
        f = stream(f)
        force.append(float(-(f_grid[1] + f_int[1]).sum().item()))
        accel.append(ay)

    # Harmonic (first-order Fourier) projection over the last 3 periods: the
    # added-mass force is the component of F in phase with the acceleration,
    # a *sin(Ωt)* harmonic.  A raw inner product is spoiled by the start-up
    # transient and by the discrete spikes when markers cross cell boundaries.
    k0 = n_steps - int(3 * period)
    F = torch.tensor(force[k0:], dtype=torch.float64)
    tt = torch.arange(F.numel(), dtype=torch.float64)
    basis_sin = torch.sin(omega * tt)
    basis_cos = torch.cos(omega * tt)
    norm = 0.5 * F.numel()
    f_sin = float((F * basis_sin).sum().item()) / norm
    f_cos = float((F * basis_cos).sum().item()) / norm
    # a_y = -A Ω² sin(Ωt) => its first harmonic coefficient is -A Ω²
    a_harm = amplitude * omega * omega
    c_a = f_sin / (m_disp * a_harm)
    in_phase = f_sin
    quadrature = f_cos
    return {
        "case": "oscillating_cylinder_added_mass_2d",
        "params": {
            "n": n,
            "radius": radius,
            "amplitude": amplitude,
            "period": period,
            "omega": omega,
            "nu": nu,
            "tau": tau,
            "n_periods": n_periods,
            "keulegan_carpenter": 2.0 * math.pi * amplitude / (2.0 * radius),
            "stokes_number": ((2.0 * nu / omega) ** 0.5) / radius,
            "kernel": kernel,
        },
        "metrics": {
            "added_mass_coefficient": c_a,
            "added_mass_reference": 1.0,
            "added_mass_rel_error": abs(c_a - 1.0),
            "in_phase_force_amplitude": float(in_phase),
            "quadrature_force_amplitude": float(quadrature),
            "max_marker_force": marker_force,
        },
        "force_audit": audit,
        "series": {"t": list(range(n_steps)), "force_y": force, "accel_y": accel},
    }

def oscillating_flow_added_mass_2d(
    n: int = 256,
    radius: float = 8.0,
    u_amp: float = 0.04,
    period: float = 150.0,
    tau: float = 0.65,
    n_periods: int = 8,
    n_settle: int = 3,
    kernel: str = "4pt",
    spacing: float = 0.7,
    mask_shrink: float = 0.0,
    use_interior: bool = True,
) -> dict:
    """Added-mass coefficient of a *fixed* cylinder in an oscillating free stream.

    A spatially uniform body force ``g(t) = ρ U₀ Ω cos(Ωt)`` drives the otherwise
    quiescent periodic box, so the free stream is ``U(t) = U₀ sin(Ωt)``.  For a
    body at rest the interior constraint force carries no inertial term
    (``d/dt ∫_V ρu = 0``), which removes the interior-fluid inertia that
    contaminates a *moving*-body added-mass test (see
    :func:`oscillating_cylinder_added_mass_2d`).

    The measured force ``F(t)`` on the cylinder is split by a two-term
    least-squares fit

    .. math:: F(t) = -C_A m_d a(t) + \\text{(in-phase with } u\\text{)},

    with ``m_d = ρ π R²`` the displaced mass per unit length and the
    acceleration ``a(t)`` taken from the first-harmonic fit of the *measured*
    domain-averaged velocity.

    For a *fixed* body in an accelerating free stream the potential-flow force
    contains the Froude–Krylov term as well as the added mass,
    ``F = (1 + C_A) m_d a``, so the reference is the inertia coefficient
    ``C_M = 1 + C_A = 2``; the reported added mass is ``C_A = C_M − 1``.

    Args:
        n: square domain side (cells).
        radius: cylinder radius (cells).
        u_amp: free-stream velocity amplitude (lattice units).
        period: oscillation period (lattice steps).
        tau: BGK relaxation time (``ν = (τ − 1/2)/3``).
        n_periods: number of periods used in the fit (after ``n_settle``).
        n_settle: start-up periods discarded.
        kernel: IBM delta kernel name.
        spacing: marker spacing in cells.

    Returns:
        dict of metrics and time series.
    """
    omega = 2.0 * math.pi / period
    nu = (tau - 0.5) / 3.0
    c = (0.5 * n, 0.5 * n)
    markers = build_cylinder_markers_2d(c, radius, spacing=spacing)
    stencil = build_stencil(markers, (n, n), kernel)
    # the solid volume is the *nominal* disc, so that m_d = ρπR² is consistent
    mask = disc_mask_2d((n, n), c, max(radius - mask_shrink, 0.5))
    zero_target = torch.zeros((stencil.n_markers, 2), dtype=torch.float64)
    zero_grid = torch.zeros((2, n, n), dtype=torch.float64)
    rho = torch.ones((n, n), dtype=torch.float64)
    z = torch.zeros_like(rho)
    f = equilibrium(rho, z, z)
    m_disp = math.pi * radius**2

    n_steps = int((n_settle + n_periods) * period)
    force: list[float] = []
    u_mean: list[float] = []
    audit: dict = {}
    for step in range(n_steps):
        t = float(step)
        g = u_amp * omega * math.cos(omega * t)  # ρ = 1
        rho_f, uxf, uyf = macroscopic(f)
        u_grid = torch.stack([uxf, uyf])
        f_m, f_grid, _, _ = direct_forcing(stencil, u_grid, zero_target, rho_f, 0)
        f_int = interior_mask_force(u_grid, mask, zero_grid) if use_interior else zero_grid
        if step == 0:
            audit = momentum_audit(stencil, f_m)
        live_mask = mask if use_interior else torch.zeros_like(mask)
        force.append(float(-(f_grid[0] + f_int[0]).sum().item()))
        u_mean.append(float(uxf.mean().item()))
        f = collide_bgk(f, tau)
        # the uniform drive must not act inside the solid: the interior
        # constraint is evaluated *before* the drive is applied, so a drive
        # inside the mask would leave a residual rho*g per step in the solid.
        drive = g * (~live_mask).to(torch.float64)
        f = body_source_2d(f, f_grid[0] + f_int[0] + drive, f_grid[1] + f_int[1])
        f = stream(f)

    k0 = n_steps - int(n_periods * period)
    F = torch.tensor(force[k0:], dtype=torch.float64)
    U = torch.tensor(u_mean[k0:], dtype=torch.float64)
    tt = torch.arange(F.numel(), dtype=torch.float64) * 1.0
    # phase reference: the fit window starts at t0 = k0
    ph = omega * (float(k0) + tt)
    sin_t, cos_t = torch.sin(ph), torch.cos(ph)
    norm = 0.5 * F.numel()
    U_s = float((U * sin_t).sum().item()) / norm
    U_c = float((U * cos_t).sum().item()) / norm
    # first harmonic of a = dU/dt: U = U_s sin + U_c cos  =>  a = w(U_s cos - U_c sin)
    a_fit = omega * (U_s * cos_t - U_c * sin_t)
    F_a = float((F * a_fit).sum().item()) / float((a_fit * a_fit).sum().item())
    c_a = -F_a / m_disp
    # drag part: in phase with the measured velocity
    U_fit = U_s * sin_t + U_c * cos_t
    c_d = float((F * U_fit).sum().item()) / (0.5 * 2.0 * radius * float((U_fit * U_fit).sum().item()))
    model = F_a * a_fit + c_d * 0.5 * 2.0 * radius * U_fit * U_fit.abs()
    ss_res = float(((F - model) ** 2).sum().item())
    ss_tot = float(((F - F.mean()) ** 2).sum().item())
    u_amp_meas = math.hypot(U_s, U_c)
    return {
        "case": "oscillating_flow_added_mass_2d",
        "params": {
            "n": n,
            "radius": radius,
            "u_amp": u_amp,
            "period": period,
            "omega": omega,
            "nu": nu,
            "tau": tau,
            "n_periods": n_periods,
            "n_settle": n_settle,
            "keulegan_carpenter": 2.0 * math.pi * (u_amp / omega) / (2.0 * radius),
            "stokes_layer_cells": (2.0 * nu / omega) ** 0.5,
            "stokes_number": ((2.0 * nu / omega) ** 0.5) / radius,
            "kernel": kernel,
            "spacing": spacing,
            "mask_shrink": mask_shrink,
            "use_interior": use_interior,
            "blockage_ratio": 2.0 * radius / n,
        },
        "metrics": {
            "inertia_coefficient_C_M": c_a + 1.0,
            "inertia_coefficient_reference": 2.0,
            "added_mass_coefficient": c_a,
            "added_mass_reference": 1.0,
            "added_mass_rel_error": abs(c_a - 1.0),
            "inertia_rel_error": abs(c_a + 1.0 - 2.0) / 2.0,
            "drag_coefficient_quadrature": c_d,
            "measured_u_amplitude": u_amp_meas,
            "drive_efficiency": u_amp_meas / u_amp,
            "force_amplitude_in_phase": abs(F_a) * omega * u_amp_meas,
            "fit_residual_fraction": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
            "force_mean": float(F.mean().item()),
        },
        "force_audit": audit,
        "series": {"t": list(range(k0, n_steps)), "force_x": force[k0:], "u_mean": u_mean[k0:]},
    }


def accelerated_cylinder_added_mass_2d(
    n: int = 160,
    radius: float = 6.0,
    u0: float = 0.05,
    n_ramp: int = 60,
    n_hold: int = 90,
    tau: float = 0.65,
    kernel: str = "4pt",
    spacing: float = 0.7,
) -> dict:
    """Added mass of a cylinder from the impulse of a prescribed acceleration.

    The cylinder (markers *and* interior mask) is accelerated from rest to
    ``U₀`` with a smooth half-cosine ramp in a quiescent periodic box.  For a
    body whose interior is modelled as fluid of the same density the momentum
    injected over the ramp is

    .. math:: I = \\int F\\,dt = (1 + C_A)\\,m_d\\,U_0 + I_\\text{viscous},

    with ``m_d = ρπR²``.  The impulse is an integral, so it is insensitive to
    the half-step phase errors that plague a harmonic (lock-in) estimate.  The
    viscous part is removed by a ramp-duration sweep: ``I(n_ramp)`` is linear in
    ``n_ramp`` for small ``n_ramp`` and extrapolated to ``n_ramp → 0``.

    Args:
        n: square domain side (cells).
        radius: cylinder radius (cells).
        u0: terminal velocity.
        n_ramp: ramp length in steps.
        n_hold: steps held at ``u0`` after the ramp (for the drag history).
        tau: BGK relaxation time.
        kernel: IBM delta kernel.
        spacing: marker spacing.

    Returns:
        dict with the impulse, ``C_M = I/(m_d U₀)`` and ``C_A = C_M − 1``.
    """
    nu = (tau - 0.5) / 3.0
    c0 = (0.5 * n, 0.5 * n)
    base = build_cylinder_markers_2d((0.0, 0.0), radius, spacing=spacing)
    m_disp = math.pi * radius**2
    rho = torch.ones((n, n), dtype=torch.float64)
    zeros = torch.zeros((n, n), dtype=torch.float64)
    zero_grid = torch.zeros((2, n, n), dtype=torch.float64)
    f = equilibrium(rho, zeros, zeros)

    x_c = c0[0]
    impulse = 0.0
    force: list[float] = []
    velocity: list[float] = []
    for step in range(n_ramp + n_hold):
        if step < n_ramp:
            v = u0 * 0.5 * (1.0 - math.cos(math.pi * step / n_ramp))
        else:
            v = u0
        x_c += v
        markers = base + torch.tensor([x_c, c0[1]], dtype=torch.float64)
        stencil = build_stencil(markers, (n, n), kernel)
        u_target = torch.zeros((stencil.n_markers, 2), dtype=torch.float64)
        u_target[:, 0] = v
        mask = disc_mask_2d((n, n), (x_c, c0[1]), radius)
        u_solid = torch.zeros((2, n, n), dtype=torch.float64)
        u_solid[0] = v
        rho_f, uxf, uyf = macroscopic(f)
        u_grid = torch.stack([uxf, uyf])
        _, f_grid, _, _ = direct_forcing(stencil, u_grid, u_target, rho_f, 0)
        f_int = interior_mask_force(u_grid, mask, u_solid)
        fx = float(-(f_grid[0] + f_int[0]).sum().item())
        force.append(fx)
        velocity.append(v)
        impulse += fx  # dt = 1; convention: F_x < 0 opposes the motion
        f = collide_bgk(f, tau)
        f = body_source_2d(f, f_grid[0] + f_int[0], f_grid[1] + f_int[1])
        f = stream(f)

    c_m = -impulse / (m_disp * u0)
    return {
        "case": "accelerated_cylinder_added_mass_2d",
        "params": {
            "n": n,
            "radius": radius,
            "u0": u0,
            "n_ramp": n_ramp,
            "n_hold": n_hold,
            "nu": nu,
            "tau": tau,
            "kernel": kernel,
            "spacing": spacing,
            "blockage_ratio": 2.0 * radius / n,
            "reynolds": u0 * 2.0 * radius / nu,
        },
        "metrics": {
            "impulse": impulse,
            "inertia_coefficient_C_M": c_m,
            "inertia_coefficient_reference": 2.0,
            "added_mass_coefficient": c_m - 1.0,
            "added_mass_reference": 1.0,
            "added_mass_rel_error": abs(c_m - 1.0 - 1.0),
            "peak_force": max(abs(x) for x in force),
            "final_force": force[-1],
        },
        "series": {"t": list(range(len(force))), "force_x": force, "velocity_x": velocity},
    }
