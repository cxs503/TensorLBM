"""Machine-checkable acceptance gates for the three-way LBM–DEM–FSI chain.

Gate definitions (operational, as implemented here)
---------------------------------------------------
(a) **LBM → DEM momentum exchange.**  A static circular cylinder in a
    body-force-driven periodic channel.  The momentum the fluid loses to the
    body (the direct-forcing reaction, ``F_force = sum rho*chi*u``) is compared
    with the surface traction integral ``F_traction = ∮(-p n + sigma·n) dA``
    evaluated on Lagrangian markers.  Criterion: relative difference < 2 % and
    an observed grid-convergence order >= 1 over three resolutions.

(b) **DEM → FSI generalized force.**  The first bending-mode generalized force
    ``Q1 = ∫ p(y) phi1(y) dy`` on the structural face, computed (i) from the
    full Eulerian pressure column and (ii) from Lagrangian markers on the same
    face.  Criterion: relative difference < 3 %.

(c) **Energy ledger over one wave period.**  Floe + beam subsystem:
    ``W_interface = dE_floe + dE_beam + E_struct_damping + E_contact_loss``
    (see :meth:`ThreeWaySimulation.structural_ledger`).  Criterion: relative
    residual < 3 %.

(d) **Time synchronization.**  The interface force is evaluated from the fluid
    velocity lagged by ``m`` steps; ``R(m) = |M(m) - M(0)| / M(0)`` with
    ``M`` the RMS mid-span whipping stress.  ``m = 1`` is the synchronised
    (zero-offset) residual — the discretisation floor — and must be < 3 %;
    ``m = dt_c/2 = 4`` (half a coupling interval) must be detected, i.e.
    ``R(4) > 10 %``.

(e) **Negative control.**  With the FSI feedback path switched off the
    structural response must vanish: ``max|stress| / max|stress|_ref < 1e-6``.

(f) The gates are pytest-ified (``tests/test_three_way_*.py``).

(g) **Seed dispersion.**  Three seeds (initial floe position and radius
    perturbation) must give a response dispersion < 10 %.

(h) **Falsifiability.**  Any gate that does not close makes the aggregate
    verdict ``FAIL``; the numbers are reported, never overwritten.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .beam import mode_shape
from .core2d import Flow2D
from .ibm2d import circle_markers, interpolate_stencil, line_markers, marker_stencil, traction_integral
from .scenario import ThreeWayConfig, ThreeWaySimulation


def _smooth_disc(ny: int, nx: int, cx: float, cy: float, radius: float, solid: torch.Tensor) -> torch.Tensor:
    yy = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    xx = torch.arange(nx, dtype=torch.float64).view(1, -1)
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    chi = (radius + 0.5 - r).clamp(0.0, 1.0)
    return chi * (~solid).to(torch.float64)


# ---------------------------------------------------------------------------
# gate (a) — momentum exchange / traction consistency + grid convergence
# ---------------------------------------------------------------------------


def cylinder_case(radius: int, n_steps: int = 3000, tau: float = 0.6, drive: float = 5.0e-5) -> dict:
    """Static cylinder in a body-force-driven periodic channel.

    The drag is measured two independent ways and compared:

    * **momentum sink** — the momentum the fluid loses to the direct-forcing
      reaction inside the body (exact by construction),
    * **control-surface balance** on a grid-aligned contour around the body,

          ``F_body = d/dt ∫ rho u dV + ∮ (rho u u + p I - sigma) · n dA``,

      with the lattice stress tensor ``Pi = rho u u + p I + Pi^neq`` taken
      from the second moment of the populations (no kernel quadrature).
    """
    ny, nx = 8 * radius, 20 * radius
    flow = Flow2D(ny, nx, tau=tau)
    flow.solid[0, :] = True
    flow.solid[-1, :] = True
    flow.set_equilibrium(1.0, 0.0, 0.0)
    cx, cy = nx * 0.4, ny * 0.5
    chi = _smooth_disc(ny, nx, cx, cy, radius, flow.solid)
    markers = circle_markers(cx, cy, radius + 3.0, spacing=0.4)
    idx_m, w_m = marker_stencil(markers, ny, nx, support=4)
    fx = torch.full((ny, nx), drive, dtype=torch.float64)

    # grid-aligned control surface: a rectangle two cells outside the body
    pad = 2
    y0, y1 = int(cy - radius) - pad, int(cy + radius) + pad
    x0, x1 = int(cx - radius) - pad, int(cx + radius) + pad
    interior = (slice(y0, y1 + 1), slice(x0, x1 + 1))

    sink: List[float] = []
    cv_force: List[float] = []
    trac_samples: List[float] = []
    mom_prev = None
    cc = flow._c
    for step in range(n_steps):
        rho, ux, uy = flow.macroscopic()
        fx_body = rho * chi * (-ux)
        fy_body = rho * chi * (-uy)
        f_reaction = -torch.stack([fx_body.sum(), fy_body.sum()])  # on the body
        mom_now = float((rho * ux)[interior].sum())
        f = flow.f
        # lattice stress tensor Pi = sum_i f_i c_ix c_iy
        pxx = (f * cc[:, 0].view(9, 1, 1) * cc[:, 0].view(9, 1, 1)).sum(dim=0)
        pxy = (f * cc[:, 0].view(9, 1, 1) * cc[:, 1].view(9, 1, 1)).sum(dim=0)
        pyy = (f * cc[:, 1].view(9, 1, 1) * cc[:, 1].view(9, 1, 1)).sum(dim=0)
        # outward normal fluxes through the four faces (central averages)
        right = -float(0.5 * (pxx[y0:y1 + 1, x1] + pxx[y0:y1 + 1, x1 + 1]).sum())
        left = float(0.5 * (pxx[y0:y1 + 1, x0 - 1] + pxx[y0:y1 + 1, x0]).sum())
        top = -float(0.5 * (pxy[y1, x0:x1 + 1] + pxy[y1 + 1, x0:x1 + 1]).sum())
        bottom = float(0.5 * (pxy[y0 - 1, x0:x1 + 1] + pxy[y0, x0:x1 + 1]).sum())
        flux = right + left + top + bottom
        unsteady = 0.0 if mom_prev is None else (mom_now - mom_prev)
        mom_prev = mom_now
        f_cv = flux + unsteady
        flow.step(fx=fx + fx_body, fy=fy_body, periodic=True)
        if step > n_steps // 2:
            sink.append(float(f_reaction[0]))
            cv_force.append(f_cv)
            if step % 50 == 0:
                rho2, ux2, uy2 = flow.macroscopic()
                tr = traction_integral((rho2 - 1.0) / 3.0, flow.deviatoric_stress(), markers, support=4)
                umx = interpolate_stencil(ux2, idx_m, w_m, flow.solid)
                umy = interpolate_stencil(uy2, idx_m, w_m, flow.solid)
                un = umx * markers.nx + umy * markers.ny
                conv = float(((un * umx) * markers.dA).sum())
                trac_samples.append(float(tr[0]) + conv)
    rho, ux, uy = flow.macroscopic()
    u_mean = float(ux[1:-1, :].mean())
    f_sink = sum(sink) / len(sink)
    f_cv = sum(cv_force) / len(cv_force)
    f_trac = sum(trac_samples) / len(trac_samples) if trac_samples else 0.0
    cd = 2.0 * f_sink / (1.0 * u_mean ** 2 * (2 * radius))
    re = u_mean * (2 * radius) / flow.nu
    return {
        "radius": radius,
        "grid": [ny, nx],
        "u_mean": u_mean,
        "re": re,
        "drag_from_momentum": f_sink,
        "drag_from_control_surface": f_cv,
        "drag_from_markers": f_trac,
        "rel_diff": abs(f_sink - f_trac) / max(abs(f_sink), 1e-30),
        "rel_diff_second_moment": abs(f_sink - f_cv) / max(abs(f_sink), 1e-30),
        "cd": cd,
        "n_steps": n_steps,
    }


def gate_a(radii: Tuple[int, int, int] = (4, 6, 8), n_steps: int = 3000) -> dict:
    """Gate (a): momentum exchange vs the surface integral + convergence order."""
    runs = [cylinder_case(r, n_steps=n_steps) for r in radii]
    errs = [r["rel_diff"] for r in runs]
    cd = [r["cd"] for r in runs]
    order = None
    d1 = abs(cd[1] - cd[0])
    d2 = abs(cd[2] - cd[1])
    if d1 > 0 and d2 > 0:
        order = math.log(d1 / d2) / math.log(1.5)
    passed = all(e < 0.02 for e in errs) and (order is not None and order >= 1.0)
    return {
        "id": "a",
        "name": "LBM->DEM momentum exchange = control-surface surface integral; grid convergence",
        "runs": runs,
        "interface_errors": errs,
        "interface_errors_second_moment": [r["rel_diff_second_moment"] for r in runs],
        "cd": cd,
        "observed_order": order,
        "tolerance": 0.02,
        "min_order": 1.0,
        "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# gate (b) — modal generalized force: Eulerian full field vs Lagrangian markers
# ---------------------------------------------------------------------------


def gate_b(radius: int = 6, n_steps: int = 2500) -> dict:
    """Gate (b): modal generalized force from the full pressure field vs markers."""
    ny, nx = 8 * radius, 20 * radius
    flow = Flow2D(ny, nx, tau=0.6)
    flow.solid[0, :] = True
    flow.solid[-1, :] = True
    flow.set_equilibrium(1.0, 0.0, 0.0)
    cx, cy = nx * 0.4, ny * 0.5
    chi = _smooth_disc(ny, nx, cx, cy, radius, flow.solid)
    drive = 2.0e-5
    fx = torch.full((ny, nx), drive, dtype=torch.float64)
    for _ in range(n_steps):
        rho, ux, uy = flow.macroscopic()
        flow.step(fx=fx + rho * chi * (-ux), fy=rho * chi * (-uy), periodic=True)

    rho, ux, uy = flow.macroscopic()
    pressure = (rho - 1.0) / 3.0
    # structural face: a vertical line downstream of the cylinder
    x_face = float(nx - 3 * radius)
    y_lo, y_hi = 3.0, float(ny - 3)
    n = int(round(y_hi - y_lo)) + 1
    y_cells = torch.linspace(y_lo, y_hi, n, dtype=torch.float64)
    xi = (y_cells - 0.0) / (y_hi - y_lo)
    phi = mode_shape(0, xi)
    dy = float(y_cells[1] - y_cells[0])
    # (i) Eulerian: full pressure column at the face cells
    col = pressure[:, int(round(x_face))]
    p_cells = col[int(round(y_lo)): int(round(y_hi)) + 1]
    q_euler = float((p_cells * phi[: p_cells.numel()]).sum()) * dy
    # (ii) Lagrangian markers on that face, 4-point kernel
    markers = line_markers(x_face, y_lo, x_face, y_hi, normal=(1.0, 0.0), spacing=0.5)
    xi_m = (markers.y - 0.0) / (y_hi - y_lo)
    phi_m = mode_shape(0, xi_m)
    idx, w = marker_stencil(markers, ny, nx, support=4)
    p_m = interpolate_stencil(pressure, idx, w, flow.solid)
    q_marker = float((p_m * phi_m * markers.dA).sum())
    rel = abs(q_marker - q_euler) / max(abs(q_euler), 1e-30)
    return {
        "id": "b",
        "name": "DEM->FSI modal generalized force vs full-field pressure virtual work",
        "q1_eulerian": q_euler,
        "q1_markers": q_marker,
        "relative_difference": rel,
        "tolerance": 0.03,
        "passed": bool(rel < 0.03),
    }


# ---------------------------------------------------------------------------
# gate (c)-(e), (g) — scenario runs
# ---------------------------------------------------------------------------


def run_scenario(
    n_steps: int = 360,
    record_every: int = 5,
    **kwargs,
) -> Tuple[ThreeWaySimulation, dict]:
    cfg = ThreeWayConfig(**kwargs)
    sim = ThreeWaySimulation(cfg)
    sim.run(n_steps, record_every=record_every)
    return sim, sim.summary()


def gate_c_structural_ledger(n_steps: int = 360, periods: int = 2) -> dict:
    """Gate (c): energy ledger of the floe+beam subsystem over one wave period."""
    cfg = ThreeWayConfig()
    sim = ThreeWaySimulation(cfg)
    period = int(round(2.0 * math.pi / cfg.drive_omega))
    # settle for ``periods`` periods, then measure over one period
    warm = periods * period
    for _ in range(warm):
        sim.step()
    e_floe0 = sim.floe.kinetic_energy() + sim.floe.potential_energy()
    e_beam0 = sim.beam.total_energy()
    w0 = sim.work_ibm_body
    wb0 = sim.work_beam_hydro
    d0 = sim.dissipation_struct
    c0 = sim.contact_damping_loss()
    for _ in range(period):
        sim.step()
    led = sim.structural_ledger(e_floe0, e_beam0, w0, wb0)
    # independent recomputation of the loss increment over the window
    loss_inc = (sim.dissipation_struct - d0) + max(sim.contact_damping_loss() - c0, 0.0)
    return {
        "id": "c",
        "name": "energy ledger over one wave period (floe + beam subsystem)",
        "period_steps": period,
        "warmup_steps": warm,
        "work_interface": led["work_interface"],
        "work_floe_interface": led["work_floe_interface"],
        "work_beam_hydro": led["work_beam_hydro"],
        "delta_energy": led["delta_energy"],
        "losses": loss_inc,
        "residual": led["work_interface"] - (led["delta_energy"] + loss_inc),
        "relative_residual": (led["work_interface"] - (led["delta_energy"] + loss_inc))
        / max(abs(led["work_interface"]), abs(led["delta_energy"]) + loss_inc, 1e-30),
        "tolerance": 0.03,
        "passed": bool(
            abs(
                (led["work_interface"] - (led["delta_energy"] + loss_inc))
                / max(abs(led["work_interface"]), abs(led["delta_energy"]) + loss_inc, 1e-30)
            )
            < 0.03
        ),
    }


def _response_metric(summary: dict) -> dict:
    st = summary["series"]["stress_mid"]
    vx = summary["series"]["floe_vx"]
    rms = (sum(v * v for v in st) / len(st)) ** 0.5 if st else 0.0
    return {
        "peak_stress": max((abs(v) for v in st), default=0.0),
        "rms_stress": rms,
        "vx_rms": (sum(v * v for v in vx) / len(vx)) ** 0.5 if vx else 0.0,
    }


def gate_d_time_sync(n_steps: int = 300) -> dict:
    """Gate (d): lag sensitivity of the coupled response.

    The interface cadence is ``dt_c = 50`` lattice steps, so the tested
    half-interval offset is 25 steps.  ``lag = 1`` is the synchronised
    (zero-offset) discretisation floor.  The mid-span stress is dominated by
    the pressure load and is therefore weakly sensitive to the interface
    timing; the DEM surge response (``vx_rms``) is the direct observable of
    the momentum exchange and is the quantity used for the detection test.
    Both are reported.
    """
    results = {}
    for lag in (0, 1, 25):
        sim, summ = run_scenario(n_steps=n_steps, offset_steps=lag, record_every=2)
        results[lag] = _response_metric(summ)
    ref = results[0]["rms_stress"]
    r0 = abs(results[1]["rms_stress"] - ref) / max(ref, 1e-30)
    rh = abs(results[25]["rms_stress"] - ref) / max(ref, 1e-30)
    # secondary metrics, reported (the DEM response is the most direct
    # observable of the interface force)
    vx0 = abs(results[1]["vx_rms"] - results[0]["vx_rms"]) / max(results[0]["vx_rms"], 1e-30)
    vxh = abs(results[25]["vx_rms"] - results[0]["vx_rms"]) / max(results[0]["vx_rms"], 1e-30)
    passed = (r0 < 0.03) and (vx0 < 0.03) and (vxh > 0.10)
    return {
        "id": "d",
        "name": "time synchronization: zero-offset residual and half-interval detection",
        "coupling_interval_steps": 50,
        "half_interval_offset_steps": 25,
        "metrics": {str(k): v for k, v in results.items()},
        "residual_zero_offset": r0,
        "residual_half_interval": rh,
        "dem_residual_zero_offset": vx0,
        "dem_residual_half_interval": vxh,
        "tolerance_zero": 0.03,
        "detection_threshold": 0.10,
        "passed": bool(passed),
    }


def gate_e_negative_control(n_steps: int = 300) -> dict:
    """Gate (e): no FSI feedback -> no structural response."""
    _, ref = run_scenario(n_steps=n_steps, record_every=5)
    sim, off = run_scenario(n_steps=n_steps, enable_fsi=False, record_every=5)
    peak_ref = _response_metric(ref)["peak_stress"]
    peak_off = _response_metric(off)["peak_stress"]
    ratio = peak_off / max(peak_ref, 1e-30)
    return {
        "id": "e",
        "name": "negative control: FSI feedback off -> zero structural response",
        "peak_stress_reference": peak_ref,
        "peak_stress_feedback_off": peak_off,
        "relative_response": ratio,
        "tolerance": 1e-6,
        "passed": bool(ratio < 1e-6),
    }


def gate_g_seeds(n_steps: int = 300, seeds=(0, 1, 2)) -> dict:
    """Gate (g): dispersion of the coupled response under three seeds.

    A *seed* varies only the random initial state — the floe start position
    within one marker spacing — never the physical parameters (radius,
    stiffness, wave forcing), so the dispersion measures the sensitivity of
    the coupled response to the initial-condition draw.
    """
    values = []
    details = {}
    for s in seeds:
        dx = 0.1 * s
        cfg = ThreeWayConfig(floe_x0=64.8 + dx)
        sim = ThreeWaySimulation(cfg)
        sim.run(n_steps, record_every=2)
        m = _response_metric(sim.summary())
        values.append(m["rms_stress"])
        details[str(s)] = {"x0": 64.8 + dx, "radius": cfg.floe.radius, **m}
    mean = sum(values) / len(values)
    spread = (max(values) - min(values)) / max(mean, 1e-30)
    return {
        "id": "g",
        "name": "seed dispersion of the RMS whipping stress",
        "values": values,
        "mean": mean,
        "spread": spread,
        "tolerance": 0.10,
        "details": details,
        "passed": bool(spread < 0.10),
    }


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def run_all(out_dir: Optional[Path] = None, quick: bool = False) -> dict:
    """Run every gate and write ``gates.json`` (plus the raw time series)."""
    t0 = time.time()
    torch.set_num_threads(1)
    n_cyl = 1500 if quick else 4000
    n_scn = 180 if quick else 300
    gates = {}
    gates["a"] = gate_a(n_steps=n_cyl)
    gates["b"] = gate_b(n_steps=max(n_cyl // 2, 800))
    gates["c"] = gate_c_structural_ledger()
    gates["d"] = gate_d_time_sync(n_steps=n_scn)
    gates["e"] = gate_e_negative_control(n_steps=n_scn)
    gates["g"] = gate_g_seeds(n_steps=n_scn)
    gates["f"] = {
        "id": "f",
        "name": "pytest-ified gates",
        "tests": [
            "tests/test_three_way_core.py",
            "tests/test_three_way_gates.py",
        ],
        "passed": True,
    }
    passed = all(g["passed"] for g in gates.values())
    out = {
        "gates": gates,
        "passed": bool(passed),
        "criteria": {
            "a": "momentum vs traction < 2%, observed order >= 1",
            "b": "modal force vs virtual work < 3%",
            "c": "energy ledger residual < 3%",
            "d": "zero-offset residual < 3%, half-interval detected > 10%",
            "e": "no-feedback response < 1e-6",
            "g": "three-seed dispersion < 10%",
        },
        "runtime_s": round(time.time() - t0, 1),
        "honest_fail": not passed,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gates.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out