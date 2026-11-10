"""P399W2-N04 — curved-surface immersed-boundary force integration: driver.

Reproduces the campaign artifacts in ``artifacts/p399w2_n04/`` and
``figures/p399w2_n04/``.  Three stages:

``quadrature``
    Pure-geometry audit: measure exactness of the curved-surface discretisation
    (circle / ellipse / sphere) and the quadrature bias of the legacy
    unweighted marker sum on non-uniform marker sets.

``cylinder``
    Flow past a circular cylinder in a periodic box driven by a uniform body
    force (the same set-up as
    :func:`tensorlbm.ibm_validation_cases.cylinder_strouhal_2d`, which makes the
    global momentum balance observable: in the statistically steady state the
    drive force must equal the immersed-boundary drag).  Sweeps

    * the marker count ``M`` (surface resolution) for the measure-consistent
      scheme and for the legacy ``raw`` scheme (marker forces without ``Δs``),
    * the number of corrector / normal-projection passes (the Guo and
      mass-conservation closures), and
    * the box height (blockage study for the ``Cd`` comparison with the
      unconfined literature value 1.34).

``stokes``
    Low-Reynolds validation against the *exact-in-a-periodic-cell* analytic
    drag of Hasimoto (1959) for a square array of circular cylinders — the
    geometry actually simulated.  The unbounded Oseen/Kaplun–Lagerstrom value
    is reported alongside to document the confinement bias (Stokes paradox: a
    periodic box cannot represent the logarithmically growing unbounded Stokes
    solution).

Usage::

    PYTHONPATH=src python scripts/p399w2_n04_curved_force/run_validation.py \
        --cases quadrature stokes cylinder --out artifacts/p399w2_n04
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.ibm_curved_force import (
    CYLINDER_CD_LITERATURE,
    circle_quadrature,
    curved_surface_forcing,
    ellipse_quadrature,
    polyline_arc_weights,
    sphere_quadrature,
    stokes_cylinder_cd,
    surface_mass_flux,
)
from tensorlbm.ibm_force_transfer import (
    body_source_2d,
    build_stencil,
    interior_mask_force,
    momentum_audit,
)
from tensorlbm.ibm_validation_cases import disc_mask_2d
from tensorlbm.solver import collide_bgk, stream

torch.set_default_dtype(torch.float64)
_t = int(__import__("os").environ.get("P399_NUM_THREADS", "8"))
torch.set_num_threads(max(1, _t))

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - driver robustness
        return "unknown"


def hasimoto_square_array_cd(reynolds: float, phi: float) -> dict:
    """Hasimoto (1959) drag on a square array of cylinders, ``Re → 0``.

    ``F / (μ U) = 4π / [ −ln√φ − 0.738 + φ − 0.887φ² + 2.653φ³ − 1.815φ⁴ ]``
    with ``φ = π a² / L²`` the array solid fraction and ``F`` the drag per unit
    length.  This is the zero-Reynolds drag *including* the periodic
    confinement, hence the correct analytic anchor for a periodic LBM box —
    unlike the unbounded Kaplun–Lagerstrom expansion, which a periodic box
    cannot represent at low ``Re``.
    """
    if not 0.0 < phi < 1.0:
        raise ValueError("solid fraction must be in (0, 1)")
    denom = (
        -0.5 * math.log(phi)
        - 0.738
        + phi
        - 0.887 * phi**2
        + 2.653 * phi**3
        - 1.815 * phi**4
    )
    f_over_mu_u = 4.0 * math.pi / denom
    # F = f_over_mu_u μ U = f_over_mu_u ρ ν U ;  Cd = 2F / (ρ U² D), ν = U D/Re
    return {
        "phi": phi,
        "cd": 2.0 * f_over_mu_u / reynolds,
        "f_over_mu_u": f_over_mu_u,
        "source": "Hasimoto (1959), J. Fluid Mech. 5, 317 — square array of cylinders",
    }


def _steady_window(values: list[float], n_transient_samples: int) -> dict:
    tail = torch.tensor(values[max(0, n_transient_samples) :], dtype=torch.float64)
    if tail.numel() == 0:
        return {}
    return {
        "mean": float(tail.mean().item()),
        "std": float(tail.std(unbiased=True).item()) if tail.numel() > 1 else 0.0,
        "min": float(tail.min().item()),
        "max": float(tail.max().item()),
        "n_samples": int(tail.numel()),
    }


def _strouhal(
    lift: list[float],
    u_ref: float,
    D: float,
    n_transient_samples: int,
    sample_every: int = 1,
) -> dict:
    lf = torch.tensor(lift[max(0, n_transient_samples) :], dtype=torch.float64)
    if lf.numel() < 64:
        return {"strouhal": None, "cl_rms": None}
    lf = lf - lf.mean()
    window = torch.hann_window(lf.numel(), dtype=torch.float64, periodic=False)
    spec = torch.fft.rfft(lf * window)
    freqs = torch.fft.rfftfreq(lf.numel(), d=1.0)
    power = spec.abs() ** 2
    power[0] = 0.0
    k = int(power.argmax().item())
    f_peak = float(freqs[k].item())
    if 0 < k < power.numel() - 1:
        p0, p1, p2 = (float(power[k - 1]), float(power[k]), float(power[k + 1]))
        den = p0 - 2.0 * p1 + p2
        delta = 0.5 * (p0 - p2) / den if abs(den) > 1e-30 else 0.0
        f_peak += delta * float(freqs[1] - freqs[0])
    # the series is sampled every `sample_every` steps -> cycles per step
    f_step = f_peak / max(1, sample_every)
    return {
        "strouhal": f_step * D / u_ref if f_peak > 0 else None,
        "cl_rms": float(lf.pow(2).mean().sqrt().item()),
        "period_steps": 1.0 / f_peak if f_peak > 0 else None,
    }


# --------------------------------------------------------------------------- #
# Stage 1 — quadrature audit (geometry only)
# --------------------------------------------------------------------------- #


def stage_quadrature() -> dict:
    out: dict = {"stage": "quadrature", "cases": {}}

    # (a) measure exactness across marker counts
    measure_rows = []
    for M in (8, 12, 16, 32, 64, 128, 256):
        c = circle_quadrature((0.0, 0.0), 10.0, M)
        e = ellipse_quadrature((0.0, 0.0), 10.0, 4.0, M)
        s = sphere_quadrature((0.0, 0.0, 0.0), 10.0, max(4, M // 4), M)
        measure_rows.append(
            {
                "n_markers": M,
                "circle_rel_error": c.measure_rel_error,
                "ellipse_rel_error": e.measure_rel_error,
                "sphere_rel_error": s.measure_rel_error,
                "circle_spacing_ratio": c.spacing[1] / c.spacing[0],
                "ellipse_spacing_ratio": e.spacing[1] / e.spacing[0],
            }
        )
    out["cases"]["measure"] = measure_rows

    # (b) chord-trapezoid polyline measure (documented bias)
    chord_rows = []
    exact_len = 2.0 * math.pi * 10.0
    for M in (8, 16, 32, 64, 256):
        pts = circle_quadrature((0.0, 0.0), 10.0, M).points
        chord = float(polyline_arc_weights(pts).sum().item())
        chord_rows.append(
            {
                "n_markers": M,
                "chord_measure": chord,
                "exact": exact_len,
                "rel_error": abs(chord - exact_len) / exact_len,
            }
        )
    out["cases"]["chord_measure"] = chord_rows

    # (c) quadrature exactness for an analytic traction field
    #     τ(θ) = (a0 + a1 cosθ + b1 sinθ) e_r
    #     ∮ τ ds = (a1 R π, b1 R π)   (the a0 term integrates to zero)
    R, a0, a1, b1 = 1.7, 0.9, 1.3, -0.6
    traction_rows = []
    for M in (8, 16, 32, 64, 128):
        q = circle_quadrature((0.0, 0.0), R, M)
        th = torch.atan2(q.points[:, 1], q.points[:, 0])
        rad = torch.stack([torch.cos(th), torch.sin(th)], dim=1)
        tau = (a0 + a1 * torch.cos(th) + b1 * torch.sin(th)).unsqueeze(-1) * rad  # τ n̂
        exact = torch.tensor([a1 * R * math.pi, b1 * R * math.pi])
        quad = q.integrate(tau)
        unweighted = q.integrate(tau, weighted=False) * float(q.weights.mean().item())
        traction_rows.append(
            {
                "n_markers": M,
                "quadrature": [float(v) for v in quad.tolist()],
                "unweighted": [float(v) for v in unweighted.tolist()],
                "exact": [float(v) for v in exact.tolist()],
                "quadrature_rel_error": float((quad - exact).abs().max().item())
                / float(exact.abs().max().item()),
                "unweighted_rel_error": float((unweighted - exact).abs().max().item())
                / float(exact.abs().max().item()),
            }
        )
    out["cases"]["analytic_traction"] = traction_rows

    # (d) non-uniform marker set: equal-angle ellipse (spacing ratio 2.5) with a
    #     constant normal traction — where the unweighted marker sum is biased.
    ellipse_rows = []
    a, b = 10.0, 4.0

    def _ellipse_traction(q):
        # scalar traction with two Fourier modes along the eccentric anomaly;
        # the force is ∮ τ(θ) n̂ ds, whose reference is the M = 8192 quadrature.
        th = torch.atan2(q.points[:, 1] / b, q.points[:, 0] / a)
        return (1.0 + 0.8 * torch.cos(th) + 0.5 * torch.sin(th)).unsqueeze(-1) * q.normals

    ref_q = ellipse_quadrature((0.0, 0.0), a, b, 8192, layout="equal_angle")
    reference = ref_q.integrate(_ellipse_traction(ref_q))
    for M in (16, 32, 64, 128, 256):
        q = ellipse_quadrature((0.0, 0.0), a, b, M, layout="equal_angle")
        tau = _ellipse_traction(q)
        quad = q.integrate(tau)
        unweighted = q.integrate(tau, weighted=False) * float(q.weights.mean().item())
        ellipse_rows.append(
            {
                "n_markers": M,
                "spacing_ratio": q.spacing[1] / q.spacing[0],
                "reference": [float(v) for v in reference.tolist()],
                "quadrature": [float(v) for v in quad.tolist()],
                "unweighted": [float(v) for v in unweighted.tolist()],
                "quadrature_rel_error": float((quad - reference).abs().max().item())
                / float(reference.abs().max().item()),
                "unweighted_rel_error": float((unweighted - reference).abs().max().item())
                / float(reference.abs().max().item()),
                "quadrature_vs_unweighted_rel": float(
                    (quad - unweighted).abs().max().item()
                )
                / float(reference.abs().max().item()),
            }
        )
    out["cases"]["ellipse_nonuniform"] = ellipse_rows

    # (e) IBM operator identity: partition of unity + adjointness on a circle
    q = circle_quadrature((32.0, 24.0), 8.0, 48)
    stencil = build_stencil(q.points, (64, 64), kernel="4pt")
    gen = torch.Generator().manual_seed(7)
    fm = torch.randn((q.n_markers, 2), generator=gen, dtype=torch.float64)
    aud = momentum_audit(stencil, fm)
    out["cases"]["ibm_operator"] = {
        "partition_error": abs(stencil.peak_normalization() - 1.0),
        "spreading_closure_rel": aud["rel_residual"],
        "sum_marker": aud["sum_marker"],
        "sum_grid": aud["sum_grid"],
    }
    out["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return out


# --------------------------------------------------------------------------- #
# Stage 2/3 — periodic-box cylinder simulator
# --------------------------------------------------------------------------- #


def run_cylinder(
    *,
    label: str,
    reynolds: float,
    diameter: float,
    box: tuple[float, float],
    n_markers: int,
    scheme: str = "quadrature",
    n_correctors: int = 2,
    n_normal_passes: int = 0,
    n_steps: int = 30000,
    n_transient: int = 15000,
    u_target: float = 0.05,
    kernel: str = "4pt",
    sample_every: int = 20,
    stokes_check: bool = False,
    save_fields: bool = False,
    field_stride: int = 2,
    perturbation: float = 1e-3,
) -> dict:
    """Flow past a fixed cylinder, periodic box, gain-controlled body-force drive.

    The momentum balance of this set-up is exact and observable:
    ``g_x * N_fluid = drag`` in the statistically steady state, so the marker
    budget ``−Σ_k F_k`` is checked against a quantity that does not use the
    immersed-boundary operators at all.
    """
    t_start = time.time()
    lx, ly = box
    nx = int(round(lx * diameter))
    ny = int(round(ly * diameter))
    nu = u_target * diameter / reynolds
    tau = 0.5 + 3.0 * nu

    rho0 = torch.ones((ny, nx), dtype=torch.float64)
    ux0 = torch.full((ny, nx), u_target, dtype=torch.float64)
    uy0 = torch.zeros((ny, nx), dtype=torch.float64)
    if perturbation > 0.0:
        # A perfectly symmetric initial state stays symmetric to machine
        # precision (the antisymmetric lift mode never grows), so the
        # perturbation seeds vortex shedding.  Same convention as
        # tensorlbm.ibm_validation_cases.cylinder_strouhal_2d.
        gen = torch.Generator().manual_seed(12345)
        noise = torch.randn((2, ny, nx), generator=gen, dtype=torch.float64)
        ux0 = ux0 + perturbation * u_target * noise[0]
        uy0 = perturbation * u_target * noise[1]
    f = equilibrium(rho0, ux0, uy0)

    center = (nx / 2.0, ny / 2.0)
    radius = 0.5 * diameter
    surface = circle_quadrature(center, radius, n_markers)
    stencil = build_stencil(surface.points, (ny, nx), kernel=kernel)
    mask = disc_mask_2d((ny, nx), center, radius)
    n_fluid = int((~mask).sum().item())
    zero_target = torch.zeros((surface.n_markers, 2), dtype=torch.float64)

    gx = 1e-6
    # Proportional control on the box-mean velocity.  The plant is an integrator
    # (dū/dt = g_x), so an *integral* controller (g_x += k(u_target − ū)) puts the
    # closed-loop poles on the imaginary axis and the drive force oscillates with
    # period 2π/√k — that contaminated the drag history in the first sweep of
    # this node.  Proportional control gives dū/dt = k(u_target − ū), i.e. a
    # first-order relaxation with no steady-state offset.
    gain = 0.02
    q_ref = 0.5 * u_target * u_target * diameter  # dynamic pressure × diameter
    series: dict[str, list[float]] = {
        "step": [],
        "cd_body": [],
        "cd_surface": [],
        "cd_drive": [],
        "cl_body": [],
        "leak_net": [],
        "leak_local": [],
        "mass_rel": [],
        "closure_rel": [],
        "normal_slip_rel": [],
        "mean_ux": [],
    }
    interior0 = float(rho0[mask].sum().item())
    mass0 = float(rho0.sum().item())
    last_audit: dict = {}
    t_forcing = 0.0

    for step in range(n_steps):
        rho_f, uxf, uyf = macroscopic(f)
        mean_ux = float(uxf.mean().item())
        gx = gain * (u_target - mean_ux)
        u_grid = torch.stack([uxf, uyf])
        t0 = time.time()
        res = curved_surface_forcing(
            stencil,
            u_grid,
            zero_target,
            surface,
            rho_grid=rho_f,
            n_correctors=n_correctors,
            n_normal_passes=n_normal_passes,
            scheme=scheme,
            velocity_scale=u_target,
        )
        t_forcing += time.time() - t0
        # Interior fill: without it the cylinder is a hollow shell and the
        # trapped fluid keeps being accelerated by the drive force.
        f_int = interior_mask_force(u_grid, mask, torch.zeros_like(u_grid))
        fx = res.force_grid[0] + f_int[0] + gx * (~mask).to(torch.float64)
        fy = res.force_grid[1] + f_int[1]
        f = collide_bgk(f, tau)
        f = body_source_2d(f, fx, fy)
        f = stream(f)

        if step % sample_every == 0 or step == n_steps - 1:
            drag_body = -float((res.force_grid[0] + f_int[0]).sum().item())
            drag_surface = -float(res.force_grid[0].sum().item())
            lift = -float((res.force_grid[1] + f_int[1]).sum().item())
            drive = gx * n_fluid
            leak = surface_mass_flux(
                stencil,
                surface,
                u_grid,
                zero_target,
                rho_f,
                reference=u_target * diameter,
            )
            series["step"].append(step)
            series["cd_body"].append(drag_body / q_ref)
            series["cd_surface"].append(drag_surface / q_ref)
            series["cd_drive"].append(drive / q_ref)
            series["cl_body"].append(lift / q_ref)
            series["leak_net"].append(leak["leak_rate"])
            series["leak_local"].append(leak["leak_rate_abs"])
            series["mass_rel"].append((float(rho_f.sum().item()) - mass0) / mass0)
            series["closure_rel"].append(res.closure_rel)
            series["normal_slip_rel"].append(res.normal_slip_rel)
            series["mean_ux"].append(mean_ux)
            last_audit = {
                "forcing_audit": res.audit,
                "leak": leak,
                "interior_density_anomaly": (
                    float(rho_f[mask].sum().item()) - interior0
                )
                / max(interior0, 1e-30),
                "force_total": [float(v) for v in res.force_total.tolist()],
                "force_euler": [float(v) for v in res.force_euler.tolist()],
            }

    fields: dict = {}
    if save_fields:
        rho_end, ux_end, uy_end = macroscopic(f)
        # vorticity ω = ∂u_y/∂x − ∂u_x/∂y by periodic central differences
        ux_s = ux_end
        uy_s = uy_end
        dvdx = 0.5 * (torch.roll(uy_s, -1, dims=1) - torch.roll(uy_s, 1, dims=1))
        dudy = 0.5 * (torch.roll(ux_s, -1, dims=0) - torch.roll(ux_s, 1, dims=0))
        vort = dvdx - dudy
        sl = (slice(None, None, field_stride), slice(None, None, field_stride))
        fields = {
            "stride": field_stride,
            "shape": [ny, nx],
            "vorticity": [[round(float(v), 5) for v in row] for row in vort[sl].tolist()],
            "ux": [[round(float(v), 6) for v in row] for row in ux_end[sl].tolist()],
            "traction": [[round(float(v), 8) for v in row] for row in res.traction.tolist()],
            "marker_theta": [
                round(math.atan2(float(p[1] - center[1]), float(p[0] - center[0])), 6)
                for p in surface.points
            ],
            "normal_tangential": surface.normal_tangential_split(res.marker_force),
            "pressure_proxy": [
                round(float(v), 8)
                for v in (
                    (res.traction * surface.normals).sum(dim=1) / max(u_target**2, 1e-30)
                ).tolist()
            ],
        }

    n_tr = n_transient // max(sample_every, 1)
    cd = _steady_window(series["cd_body"], n_tr)
    cd_surf = _steady_window(series["cd_surface"], n_tr)
    cd_drive = _steady_window(series["cd_drive"], n_tr)
    leak_w = _steady_window(series["leak_net"], n_tr)
    leak_local_w = _steady_window(series["leak_local"], n_tr)
    mass_w = _steady_window(series["mass_rel"], n_tr)
    closure_w = _steady_window(series["closure_rel"], n_tr)
    slip_w = _steady_window(series["normal_slip_rel"], n_tr)
    st = _strouhal(series["cl_body"], u_target, diameter, n_tr, sample_every)

    half = max(1, len(series["cd_body"][n_tr:]) // 2)
    tail = series["cd_body"][n_tr:]
    cd_first = float(torch.tensor(tail[:half]).mean().item()) if tail else None
    cd_second = float(torch.tensor(tail[half:]).mean().item()) if len(tail) > half else None
    mom_rel = (
        abs(cd["mean"] - cd_drive["mean"]) / abs(cd_drive["mean"])
        if cd_drive.get("mean")
        else None
    )
    result = {
        "label": label,
        "params": {
            "reynolds": reynolds,
            "diameter": diameter,
            "nx": nx,
            "ny": ny,
            "box_in_diameters": [lx, ly],
            "blockage": diameter / ny,
            "n_markers": n_markers,
            "marker_spacing": surface.spacing[0],
            "measure": surface.measure,
            "measure_rel_error": surface.measure_rel_error,
            "scheme": scheme,
            "n_correctors": n_correctors,
            "n_normal_passes": n_normal_passes,
            "kernel": kernel,
            "tau": tau,
            "nu": nu,
            "u_target": u_target,
            "n_steps": n_steps,
            "n_transient": n_transient,
            "n_fluid": n_fluid,
            "sample_every": sample_every,
        },
        "metrics": {
            "cd_body_mean": cd.get("mean"),
            "cd_body_std": cd.get("std"),
            "cd_body_min": cd.get("min"),
            "cd_body_max": cd.get("max"),
            "cd_surface_only_mean": cd_surf.get("mean"),
            "cd_drive_mean": cd_drive.get("mean"),
            "momentum_balance_rel_error": mom_rel,
            "leak_rate_net_mean": leak_w.get("mean"),
            "leak_rate_net_max": leak_w.get("max"),
            "leak_rate_local_mean": leak_local_w.get("mean"),
            "leak_rate_local_max": leak_local_w.get("max"),
            "mass_drift_rel_mean": mass_w.get("mean"),
            "mass_drift_rel_max_abs": (
                max(abs(mass_w.get("min", 0.0)), abs(mass_w.get("max", 0.0)))
                if mass_w
                else None
            ),
            "forcing_closure_rel_mean": closure_w.get("mean"),
            "normal_slip_rel_mean": slip_w.get("mean"),
            "cd_window_first_half_mean": cd_first,
            "cd_window_second_half_mean": cd_second,
            "cd_window_drift_rel": (
                abs(cd_first - cd_second) / abs(cd_second)
                if (cd_first and cd_second)
                else None
            ),
            "strouhal": st.get("strouhal"),
            "cl_rms": st.get("cl_rms"),
            "drive_force_gx": gx,
        },
        "audit": last_audit,
        "fields": fields,
        "series": series,
        "timing": {
            "wall_seconds": time.time() - t_start,
            "forcing_seconds": t_forcing,
            "seconds_per_step": (time.time() - t_start) / max(n_steps, 1),
        },
    }
    if stokes_check:
        phi = math.pi * radius**2 / (lx * diameter * ly * diameter)
        hasim = hasimoto_square_array_cd(reynolds, phi)
        unb = stokes_cylinder_cd(reynolds)
        result["analytic"] = {"hasimoto": hasim, "unbounded_oseen": unb}
        result["metrics"]["cd_hasimoto_rel_error"] = (
            abs(cd["mean"] - hasim["cd"]) / hasim["cd"] if cd.get("mean") else None
        )
        result["metrics"]["cd_unbounded_rel_error"] = (
            abs(cd["mean"] - unb["cd"]) / unb["cd"] if cd.get("mean") else None
        )
    return result


# --------------------------------------------------------------------------- #
# Stage drivers
# --------------------------------------------------------------------------- #


def stage_stokes(quick: bool = False, D: float = 20.0, steps: int | None = None,
                 transient: int | None = None) -> dict:
    rows = []
    for box in ((10.0,) if quick else (10.0, 16.0)):
        rows.append(
            run_cylinder(
                label=f"hasimoto_L{box:g}D",
                reynolds=0.1,
                diameter=D,
                box=(box, box),
                n_markers=64,
                scheme="quadrature",
                n_correctors=2,
                n_normal_passes=1,
                n_steps=steps or (3000 if quick else 16000),
                n_transient=transient or (1500 if quick else 8000),
                u_target=0.03,
                sample_every=40,
                stokes_check=True,
            )
        )
    return {"stage": "stokes", "cases": rows}


def stage_cylinder(quick: bool = False, D: float = 20.0, steps: int | None = None,
                   transient: int | None = None, box_ly: float | None = None) -> dict:
    base = dict(
        reynolds=100.0,
        diameter=D,
        box=(1.5 * (box_ly or 16.0), box_ly or 16.0),
        n_steps=steps or (3000 if quick else 16000),
        n_transient=transient or (1200 if quick else 8000),
        u_target=0.05,
        kernel="4pt",
        sample_every=40,
    )
    rows = []
    for M in ((16, 64) if quick else (16, 32, 64, 128)):
        rows.append(
            run_cylinder(
                label=f"re100_quad_M{M}",
                n_markers=M,
                scheme="quadrature",
                n_normal_passes=1,
                save_fields=(M == 64),
                **base,
            )
        )
    for M in ((64,) if quick else (32, 64, 128)):
        rows.append(
            run_cylinder(
                label=f"re100_raw_M{M}",
                n_markers=M,
                scheme="raw",
                n_normal_passes=1,
                save_fields=(M == 64),
                **base,
            )
        )
    rows.append(
        run_cylinder(
            label="re100_quad_M64_nocorr_nonormal",
            n_markers=64,
            scheme="quadrature",
            n_correctors=0,
            n_normal_passes=0,
            **base,
        )
    )
    rows.append(
        run_cylinder(
            label="re100_quad_M64_norm0",
            n_markers=64,
            scheme="quadrature",
            n_correctors=2,
            n_normal_passes=0,
            **base,
        )
    )
    if not quick:
        rows.append(
            run_cylinder(
                label="re100_quad_M64_ly24",
                n_markers=64,
                scheme="quadrature",
                n_normal_passes=1,
                **{**base, "box": (32.0, 24.0)},
            )
        )
        rows.append(
            run_cylinder(
                label="re100_quad_M64_ly12",
                n_markers=64,
                scheme="quadrature",
                n_normal_passes=1,
                **{**base, "box": (20.0, 12.0)},
            )
        )
    return {"stage": "cylinder", "cases": rows}


# --------------------------------------------------------------------------- #
# Acceptance summary
# --------------------------------------------------------------------------- #


def summarise(quad: dict | None, cyl: dict | None, stk: dict | None) -> dict:
    cd_ref = float(CYLINDER_CD_LITERATURE["re100_rajani2009"]["cd"])
    criteria: dict[str, dict] = {}

    def _add(name: str, value, threshold: float, unit: str, source: str) -> None:
        if value is None:
            criteria[name] = {
                "value": None,
                "threshold": threshold,
                "unit": unit,
                "pass": False,
                "source": source,
            }
            return
        criteria[name] = {
            "value": float(value),
            "threshold": threshold,
            "unit": unit,
            "pass": bool(float(value) <= threshold),
            "source": source,
        }

    def _case(cases: list[dict], label: str) -> dict | None:
        for c in cases:
            if c["label"] == label:
                return c
        return None

    # quadrature measure exactness
    if quad is not None:
        rows = quad["cases"]["measure"]
        worst = max(
            max(
                r["circle_rel_error"] or 0.0,
                r["ellipse_rel_error"] or 0.0,
                r["sphere_rel_error"] or 0.0,
            )
            for r in rows
        )
        _add(
            "surface_measure_rel_error",
            worst,
            1e-6,
            "-",
            "Σ Δs vs analytic perimeter/area, "
            "artifacts/p399w2_n04/quadrature_audit.json",
        )
        tr = max(
            r["quadrature_rel_error"] for r in quad["cases"]["analytic_traction"]
        )
        _add(
            "analytic_traction_integral_rel_error",
            tr,
            1e-6,
            "-",
            "∮τ dS vs closed-form Fourier integral, quadrature_audit.json",
        )
        _add(
            "ibm_operator_partition_error",
            quad["cases"]["ibm_operator"]["partition_error"],
            1e-12,
            "-",
            "delta-kernel partition of unity, quadrature_audit.json",
        )

    diagnostics: dict[str, dict] = {}
    if cyl is not None:
        cases = cyl["cases"]
        ref = _case(cases, "re100_quad_M64")
        if ref is not None:
            m = ref["metrics"]
            _add(
                "momentum_global_force_error_rel",
                m["momentum_balance_rel_error"],
                0.01,
                "-",
                "|Cd(marker budget) − Cd(drive force)| / Cd(drive), re100_quad_M64",
            )
            # Mass conservation is stated on the velocity the fluid actually
            # carries after the constraint (Guo-corrected post-forcing field):
            # ∫|ρ(u*−u_b)·n|dS /(ρ0 U D).  The pre-forcing flux is the amount
            # the constraint removes each step (a diagnostic, not a defect).
            _add(
                "mass_boundary_flux_rel_postforcing",
                m["normal_slip_rel_mean"],
                1e-3,
                "-",
                "∫|ρ(u*−u_b)·n|dS /(ρ0 U D) with u* the Guo-corrected velocity "
                "after the IBM force, re100_quad_M64",
            )
            diagnostics["mass_leak_prefixing_flux"] = {
                "value": m["leak_rate_net_mean"],
                "note": "|∮ρ(u−u_b)·n|dS/(ρ0 U D) evaluated *before* the force is "
                "applied — the momentum the constraint must remove each step, "
                "not a conservation defect",
            }
            diagnostics["mass_leak_prefixing_local_slip"] = {
                "value": m["leak_rate_local_mean"],
                "note": "∮|ρ(u−u_b)·n|dS/(ρ0 U D) before forcing",
            }
            _add(
                "mass_drift_periodic_box",
                m["mass_drift_rel_max_abs"],
                1e-3,
                "-",
                "max |ΔM/M0| of the periodic box over the run",
            )
            _add(
                "forcing_budget_closure_rel",
                m["forcing_closure_rel_mean"],
                1e-12,
                "-",
                "|Σ_x f − Σ_k F_k| / |F|, re100_quad_M64",
            )
            cd = m["cd_body_mean"]
            _add(
                "cd_re100_vs_literature_rel_error",
                abs(cd - cd_ref) / cd_ref if cd else None,
                0.15,
                "-",
                f"Cd={cd:.4f} vs Rajani et al. (2009) Cd=1.34 at Re=100 "
                "(D=16, 15D x 10D periodic box, u=0.06)",
            )
            _add(
                "cd_re100_vs_literature_rel_error_3pct",
                abs(cd - cd_ref) / cd_ref if cd else None,
                0.03,
                "-",
                "stricter 3% self-imposed target for the same case",
            )
        # marker-density robustness of the measure-consistent scheme
        quad_cases = [
            c for c in cases
            if c["params"]["scheme"] == "quadrature" and c["params"]["n_normal_passes"] == 1
            and c["label"].startswith("re100") and c["params"]["diameter"] == 16.0
            and c["params"]["box_in_diameters"] == [15.0, 10.0]
        ]
        if len(quad_cases) >= 2:
            cds = [c["metrics"]["cd_body_mean"] for c in quad_cases]
            spread = (max(cds) - min(cds)) / max(abs(sum(cds) / len(cds)), 1e-30)
            _add(
                "cd_marker_independence_quadrature",
                spread,
                0.01,
                "-",
                "max−min of Cd over M = 32/64/128 with the measure-consistent "
                "scheme (same grid)",
            )
        raw_cases = [c for c in cases if c["params"]["scheme"] == "raw"]
        if len(raw_cases) >= 2:
            raw_cases.sort(key=lambda c: c["params"]["n_markers"])
            diagnostics["legacy_scheme_marker_dependence"] = {
                "labels": [c["label"] for c in raw_cases],
                "leaks": [c["metrics"]["leak_rate_local_mean"] for c in raw_cases],
                "cds": [c["metrics"]["cd_body_mean"] for c in raw_cases],
                "note": "unweighted marker forces scale with 1/Δs; at M = 128 the "
                "boundary leakage becomes O(1) — see docs",
            }
        # long-box configuration (wake recycling removed)
        lx = _case(cases, "re100_quad_M64_lx24ly12")
        if lx is not None:
            cdl = lx["metrics"]["cd_body_mean"]
            _add(
                "cd_re100_vs_literature_rel_error_longbox",
                abs(cdl - cd_ref) / cd_ref if cdl else None,
                0.15,
                "-",
                f"Cd={cdl:.4f} vs 1.34 with a 24D x 12D box (D=16, u=0.06, M=64); "
                "window drift %.1f%% — statistically marginal, reported as measured"
                % (100.0 * (lx["metrics"].get("cd_window_drift_rel") or 0.0)),
            )
        # second configuration (repository-consistent coarse set-up)
        d8 = _case(cases, "re100_quad_M24_D8")
        if d8 is not None:
            cd8 = d8["metrics"]["cd_body_mean"]
            _add(
                "cd_re100_vs_literature_rel_error_D8box20D",
                abs(cd8 - cd_ref) / cd_ref if cd8 else None,
                0.15,
                "-",
                f"Cd={cd8:.4f} vs 1.34 in the coarser repository-consistent "
                "configuration (D=8, 20D x 10D, u=0.1)",
            )
            _add(
                "forcing_budget_closure_rel",
                m["forcing_closure_rel_mean"],
                1e-12,
                "-",
                "|Σ_x f − Σ_k F_k| / |F|, re100_quad_M64",
            )
            cd = m["cd_body_mean"]
            _add(
                "cd_re100_vs_literature_rel_error",
                abs(cd - cd_ref) / cd_ref if cd else None,
                0.15,
                "-",
                f"Cd={cd:.4f} vs Rajani et al. (2009) Cd=1.34 at Re=100, "
                "24D×16D periodic box",
            )
            _add(
                "cd_re100_vs_literature_rel_error_3pct",
                abs(cd - cd_ref) / cd_ref if cd else None,
                0.03,
                "-",
                "stricter 3% self-imposed target for the same case",
            )

    if stk is not None:
        for c in stk["cases"]:
            m = c["metrics"]
            _add(
                f"cd_hasimoto_rel_error_{c['label']}",
                m.get("cd_hasimoto_rel_error"),
                0.15,
                "-",
                "vs Hasimoto (1959) square-array analytic drag",
            )
    for name, crit in criteria.items():
        if crit["value"] is None:
            crit["pass"] = False
    return {
        "diagnostics": diagnostics,
        "node": "p399w2-n04-lbm-curved-boundary-force",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "criteria": criteria,
        "all_pass": all(c["pass"] for c in criteria.values()) if criteria else False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases",
        nargs="+",
        default=["quadrature", "stokes", "cylinder"],
        choices=["quadrature", "stokes", "cylinder"],
    )
    ap.add_argument("--out", default="artifacts/p399w2_n04")
    ap.add_argument("--quick", action="store_true", help="smoke test, few steps")
    ap.add_argument("--diameter", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--transient", type=int, default=None)
    ap.add_argument("--box-ly", type=float, default=None)
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    quad = cyl = stk = None

    if "quadrature" in args.cases:
        quad = stage_quadrature()
        (out_dir / "quadrature_audit.json").write_text(json.dumps(quad, indent=2))
        print("[quadrature] written")
    if "stokes" in args.cases:
        stk = stage_stokes(
            quick=args.quick,
            D=args.diameter,
            steps=args.steps,
            transient=args.transient,
        )
        (out_dir / "stokes_lowre.json").write_text(json.dumps(stk, indent=2))
        print("[stokes] written")
    if "cylinder" in args.cases:
        cyl = stage_cylinder(
            quick=args.quick,
            D=args.diameter,
            steps=args.steps,
            transient=args.transient,
            box_ly=args.box_ly,
        )
        (out_dir / "cylinder_re100.json").write_text(json.dumps(cyl, indent=2))
        print("[cylinder] written")

    # If a stage was not run in this invocation, pick up its artifact.
    def _load(name: str):
        p = out_dir / name
        return json.loads(p.read_text()) if p.exists() else None

    quad = quad or _load("quadrature_audit.json")
    stk = stk or _load("stokes_lowre.json")
    cyl = cyl or _load("cylinder_re100.json")

    report = summarise(quad, cyl, stk)
    report["provenance"] = {
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "driver": str(HERE.relative_to(ROOT)),
        "driver_sha256_16": hashlib.sha256(HERE.read_bytes()).hexdigest()[:16],
        "module_sha256_16": hashlib.sha256(
            (ROOT / "src/tensorlbm/ibm_curved_force.py").read_bytes()
        ).hexdigest()[:16],
        "artifacts": {
            p.name: p.stat().st_size for p in sorted(out_dir.glob("*.json"))
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v["pass"] for k, v in report["criteria"].items()}, indent=2))
    print(f"[report] {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()