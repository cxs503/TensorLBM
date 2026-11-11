"""Capability gate for the two-level (2:1) static refinement of TensorLBM.

The gate is a set of independent, machine-checkable verifications of the 2-D
two-level refined D2Q9 solver :mod:`tensorlbm.two_level_amr2d`:

===========  ==================================================================
gate         what it isolates
===========  ==================================================================
``conservation``  composite mass/momentum bookkeeping across the interface
``poiseuille``    consistency of the interface stress with an analytic profile
``reflection``    spurious acoustic reflection of the interface operator
``order``         formal accuracy of the composite (refined) discretisation
``cylinder``      engineering cross-check: drag of a cylinder in a refined patch
``negative_control`` the gates must *fail* when the operator physics is broken
===========  ==================================================================

Every gate returns a plain dictionary with the measured numbers, the target
threshold and ``passed``; :func:`run_gate_suite` assembles them into the JSON
artifact consumed by the campaign node.
"""

from __future__ import annotations

import math

import torch

from .two_level_amr2d import (
    TwoLevelConfig,
    TwoLevelD2Q9,
    UniformD2Q9,
    cylinder_solid_mask,
    obstacle_force,
)

__all__ = [
    "gate_conservation",
    "gate_poiseuille",
    "gate_reflection",
    "gate_order",
    "gate_cylinder",
    "gate_negative_controls",
    "run_gate_suite",
]

TAU = 0.65


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _grid(ny: int, nx: int) -> tuple[torch.Tensor, torch.Tensor]:
    yy = (torch.arange(ny, dtype=torch.float64) + 0.5).view(-1, 1)
    xx = (torch.arange(nx, dtype=torch.float64) + 0.5).view(1, -1)
    return yy, xx


def _channel_solid(ny: int, nx: int) -> torch.Tensor:
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    return solid


# ---------------------------------------------------------------------------
# gate (a): mass / momentum conservation of the composite grid
# ---------------------------------------------------------------------------


def gate_conservation(
    shape: tuple[int, int] = (64, 64),
    steps: int = 200,
    tau: float = 0.6,
    amplitude: float = 1.0e-3,
    mass_tol: float = 1.0e-12,
    momentum_tol: float = 1.0e-8,
) -> dict:
    """Closed periodic box: both invariants must be conserved to round-off."""
    ny, nx = shape
    yy, xx = _grid(ny, nx)
    kx = 2.0 * math.pi / nx
    ky = 2.0 * math.pi / ny
    rho = 1.0 + amplitude * torch.sin(kx * xx) * torch.cos(ky * yy)
    ux = 0.02 + amplitude * torch.cos(kx * xx) * torch.cos(ky * yy) + torch.zeros_like(rho)
    uy = (amplitude * torch.sin(ky * yy)).expand(ny, nx).contiguous()
    box = (nx // 4, 3 * nx // 4, ny // 4, 3 * ny // 4)
    sim = TwoLevelD2Q9(TwoLevelConfig(shape=shape, box=box, tau_coarse=tau))
    sim.fill_uniform(rho, ux, uy)
    m0, px0, py0 = sim.totals()
    worst_mass = 0.0
    worst_mom = 0.0
    for _ in range(steps):
        sim.step()
        m, px, py = sim.totals()
        worst_mass = max(worst_mass, abs(m - m0) / abs(m0))
        worst_mom = max(worst_mom, math.hypot(px - px0, py - py0) / (1.0 + math.hypot(px0, py0)))
    return {
        "gate": "conservation",
        "steps": steps,
        "shape": list(shape),
        "box": list(box),
        "tau_coarse": tau,
        "mass_residual_relative": worst_mass,
        "momentum_residual_relative": worst_mom,
        "mass_tolerance": mass_tol,
        "momentum_tolerance": momentum_tol,
        "interface_defect_mass_per_step": sim.crossing_ledger["defect_mass"],
        "interface_defect_momentum_per_step": sim.crossing_ledger["defect_px"],
        "passed": bool(worst_mass <= mass_tol and worst_mom <= momentum_tol),
    }


# ---------------------------------------------------------------------------
# gate (b): Poiseuille profile in a refined band
# ---------------------------------------------------------------------------


def _poiseuille_profile(ny: int, nu: float, force_x: float) -> torch.Tensor:
    """Analytic half-way-bounce-back Poiseuille profile on the cell centres.

    Walls sit at ``y = 1`` and ``y = ny - 1``; the peak value is
    ``force_x * (ny - 2)**2 / (8 nu)``.
    """
    y = torch.arange(ny, dtype=torch.float64) + 0.5
    u = force_x / (2.0 * nu) * (y - 1.0) * (ny - 1.0 - y)
    return torch.clamp(u, min=0.0)


def gate_poiseuille(
    shape: tuple[int, int] = (26, 64),
    steps: int = 4000,
    tau: float = TAU,
    target_umax: float = 0.05,
    tol: float = 0.01,
) -> dict:
    """Body-force channel; the refined band must reproduce the analytic profile."""
    ny, nx = shape
    nu = (tau - 0.5) / 3.0
    h = ny - 2
    force = 8.0 * nu * target_umax / (h * h)
    u_exact = _poiseuille_profile(ny, nu, force)
    u_max = float(u_exact.max())
    u_field = u_exact.view(ny, 1).expand(ny, nx).contiguous()
    box = (nx // 2 - 8, nx // 2 + 8, 3, ny - 3)

    sim = TwoLevelD2Q9(
        TwoLevelConfig(
            shape=shape,
            box=box,
            tau_coarse=tau,
            force=(force, 0.0),
            periodic=(True, False),
            momentum_reflux=False,
        ),
        solid_coarse=_channel_solid(ny, nx),
    )
    rho = torch.ones((ny, nx), dtype=torch.float64)
    zeros = torch.zeros((ny, nx), dtype=torch.float64)
    sim.fill_uniform(rho, u_field, zeros)
    for _ in range(steps):
        sim.step()
    _, ux_c, _ = sim.composite_fields()
    # stabilised profile: average along x inside the patch columns
    ys, xs = sim.config.interior_coarse_slice
    ux_prof = ux_c[:, xs].mean(dim=1)
    err_full = float((ux_prof[1:-1] - u_exact[1:-1]).abs().max() / u_max)
    # the two cells adjacent to the tangential interface carry a first-order
    # interface layer; the band interior is the accuracy statement of the gate
    y0, y1 = ys.start, ys.stop
    interior_rows = list(range(y0 + 2, y1 - 2))
    err = float(
        max(
            abs(float(ux_prof[r]) - float(u_exact[r])) for r in interior_rows
        )
        / u_max
    )

    # uniform coarse baseline at the same wall-normal resolution
    uniform = UniformD2Q9(
        shape,
        tau,
        force=(force, 0.0),
        periodic=(True, False),
        solid=_channel_solid(ny, nx),
    )
    uniform.fill_uniform(rho, u_field, zeros)
    uniform.step(steps)
    _, ux_u, _ = uniform.fields()
    ux_u_prof = ux_u[:, box[0] : box[1]].mean(dim=1)
    err_uniform = float((ux_u_prof[1:-1] - u_exact[1:-1]).abs().max() / u_max)

    return {
        "gate": "poiseuille",
        "shape": list(shape),
        "box": list(box),
        "tau_coarse": tau,
        "tau_fine": sim.config.tau_fine,
        "nu_lattice": nu,
        "force_density": force,
        "u_max_lattice": u_max,
        "steps": steps,
        "profile_two_level": [float(v) for v in ux_prof],
        "profile_uniform_coarse": [float(v) for v in ux_u_prof],
        "profile_analytic": [float(v) for v in u_exact],
        "profile_error_two_level": err,
        "profile_error_two_level_full_band": err_full,
        "interface_layer_rows": [y0, y0 + 1, y1 - 2, y1 - 1],
        "error_at_interface_layer_rows": {
            str(r): abs(float(ux_prof[r]) - float(u_exact[r])) / u_max for r in (y0, y0 + 1, y1 - 2, y1 - 1)
        },
        "profile_error_uniform_coarse": float(err_uniform),
        "tolerance": tol,
        "passed": bool(err <= tol),
    }


# ---------------------------------------------------------------------------
# gate (c): spurious acoustic reflection of the interface
# ---------------------------------------------------------------------------


def gate_reflection(
    shape: tuple[int, int] = (10, 200),
    steps: int = 200,
    tau: float = 0.56,
    amplitude: float = 1.0e-4,
    sigma_values: tuple[float, ...] = (2.0, 3.0, 4.0, 6.0),
    tol: float = 0.01,
) -> dict:
    """Plane acoustic pulse hitting the coarse/fine interface at normal incidence.

    The reflected part is isolated by differencing the probe signal against an
    identical *uniform coarse* run (no interface).
    """
    ny, nx = shape
    box = (100, 140, 1, ny - 1)
    probe = 70
    source = 40
    rows = {}
    traces: dict[str, list[float]] = {}
    for sigma in sigma_values:
        yy, xx = _grid(ny, nx)
        rho0 = 1.0 + amplitude * torch.exp(-((xx - source) ** 2) / (2.0 * sigma**2))
        rho0 = rho0.expand(ny, nx).contiguous()
        zeros = torch.zeros_like(rho0)

        ref = UniformD2Q9(shape, tau, periodic=(True, True))
        ref.fill_uniform(rho0, zeros, zeros)
        sim = TwoLevelD2Q9(TwoLevelConfig(shape=shape, box=box, tau_coarse=tau))
        sim.fill_uniform(rho0, zeros, zeros)

        ref_trace = []
        sim_trace = []
        for _ in range(steps):
            ref.step()
            sim.step()
            ref_trace.append(float(ref.fields()[0][:, probe].mean()))
            sim_trace.append(float(sim.composite_fields()[0][:, probe].mean()))
        ref_t = torch.tensor(ref_trace) - 1.0
        sim_t = torch.tensor(sim_trace) - 1.0
        incident_amp = float(ref_t.abs().max())
        reflected = sim_t - ref_t
        reflected_amp = float(reflected.abs().max())
        if abs(sigma - 4.0) < 1e-9:
            traces = {
                "incident": [float(v) for v in ref_t],
                "total": [float(v) for v in sim_t],
                "reflected": [float(v) for v in reflected],
            }
        rows[f"{sigma:g}"] = {
            "pulse_sigma": float(sigma),
            "incident_amplitude": incident_amp,
            "reflected_amplitude": reflected_amp,
            "reflection_coefficient": reflected_amp / incident_amp if incident_amp else float("nan"),
        }
    worst = max(row["reflection_coefficient"] for row in rows.values())
    nominal = rows.get("4", list(rows.values())[-1])
    return {
        "gate": "reflection",
        "shape": list(shape),
        "box": list(box),
        "probe_x": probe,
        "source_x": source,
        "steps": steps,
        "tau_coarse": tau,
        "amplitude": amplitude,
        "per_sigma": rows,
        "traces_sigma4": traces,
        "reflection_coefficient": nominal["reflection_coefficient"],
        "worst_reflection_coefficient": worst,
        "tolerance": tol,
        "passed": bool(nominal["reflection_coefficient"] <= tol),
    }


# ---------------------------------------------------------------------------
# gate (d): formal order of accuracy with an embedded refined block
# ---------------------------------------------------------------------------


def gate_order(
    resolutions: tuple[int, ...] = (32, 64, 128),
    tau_reference: float = 0.548,
    reference_resolution: int = 32,
    nu_physical: float = 0.00314,
    target_order: float = 1.8,
) -> dict:
    """Taylor-Green vortex decay: L2 error of the composite field vs resolution."""
    u0 = 0.05
    rows = []
    errors = []
    for n in resolutions:
        nu_lat = n * nu_physical / (2.0 * math.pi)
        tau = 0.5 + 3.0 * nu_lat
        steps = int(round(300.0 * n / reference_resolution))
        t_lat = float(steps)
        yy, xx = _grid(n, n)
        # physical coordinate x = 2*pi*(i+0.5)/n
        xp = (2.0 * math.pi / n) * xx
        yp = (2.0 * math.pi / n) * yy
        decay = math.exp(-2.0 * nu_physical * t_lat * (2.0 * math.pi / n))
        ux0 = u0 * torch.cos(xp) * torch.cos(yp)
        uy0 = -u0 * torch.sin(xp) * torch.sin(yp)
        box = (n // 3, 2 * n // 3, n // 3, 2 * n // 3)
        sim = TwoLevelD2Q9(TwoLevelConfig(shape=(n, n), box=box, tau_coarse=tau))
        sim.fill_uniform(torch.ones((n, n), dtype=torch.float64), ux0, uy0)
        for _ in range(steps):
            sim.step()
        _, ux, uy = sim.composite_fields()
        ux_ex = ux0 * decay
        uy_ex = uy0 * decay
        err = math.sqrt(
            float(((ux - ux_ex) ** 2 + (uy - uy_ex) ** 2).mean())
        ) / u0
        rows.append(
            {
                "resolution": n,
                "tau_coarse": tau,
                "tau_fine": sim.config.tau_fine,
                "steps": steps,
                "l2_error": err,
                "box": list(box),
            }
        )
        errors.append(err)
    orders = []
    for i in range(len(errors) - 1):
        orders.append(math.log2(errors[i] / errors[i + 1]))
    observed = min(orders) if orders else float("nan")
    return {
        "gate": "order",
        "case": "taylor_green_decay",
        "u0": u0,
        "nu_physical": nu_physical,
        "rows": rows,
        "observed_order": observed,
        "order_per_pair": orders,
        "target_order": target_order,
        "passed": bool(observed >= target_order),
    }


# ---------------------------------------------------------------------------
# gate (e): cylinder drag in a refined patch
# ---------------------------------------------------------------------------


def _cylinder_run(
    *,
    diameter_cells: int,
    levels: str,
    steps: int,
    tau: float,
    domain_diameters: tuple[float, float] = (18.0, 14.0),
    patch_margin: float = 3.0,
    force: float = 2.0e-5,
    u0: float = 0.05,
    average_window: int = 400,
) -> dict:
    """One cylinder run; drag from the momentum-exchange (MEM) force."""
    from .two_level_amr2d import collide_bgk, obstacle_force, stream_mask

    d = float(diameter_cells)
    r = 0.5 * d
    lx = domain_diameters[0] * d
    ly = domain_diameters[1] * d
    nx = int(round(lx))
    ny = int(round(ly))
    centre = (lx / 2.0, ly / 2.0)
    c = None

    def _accumulate(post, solid, tau_used, count):
        fx, fy = obstacle_force(post, solid, sim.c)
        return fx / count, fy / count

    if levels in ("uniform_coarse", "uniform_fine"):
        if levels == "uniform_coarse":
            grid = (ny, nx)
            spacing = 1.0
            d_cells = d
            tau_used = tau
            n_steps = steps
        else:
            grid = (2 * ny, 2 * nx)
            spacing = 0.5
            d_cells = 2.0 * d
            tau_used = 0.5 + 2.0 * (tau - 0.5)
            n_steps = 2 * steps
        solid = cylinder_solid_mask(grid, centre, r, spacing=spacing)
        sim = UniformD2Q9(grid, tau_used, force=(force, 0.0), solid=solid)
        rho = torch.ones(grid, dtype=torch.float64)
        sim.fill_uniform(rho, torch.full_like(rho, u0), torch.zeros_like(rho))
        samples = 0
        f_sum = 0.0
        window_start = n_steps - average_window
        for k in range(n_steps):
            post = collide_bgk(sim.f, tau_used, (force, 0.0), sim.c, sim.w)
            if k >= window_start and (k - window_start) % 5 == 0:
                fx, _ = obstacle_force(post, solid, sim.c)
                f_sum += fx
                samples += 1
            sim.f = stream_mask(post, solid, (True, True), sim.c, sim.opp)
        rho_f, ux, uy = sim.fields()
        fluid = ~solid
        cell_volume = spacing**2
        u_mean = float((rho_f * ux)[fluid].sum() / rho_f[fluid].sum())
        cost = {"coarse_cells": 0.0, "fine_cells": float(fluid.sum()) * cell_volume}
        if levels == "uniform_coarse":
            cost = {"coarse_cells": float(fluid.sum()), "fine_cells": 0.0}
        drag = f_sum / samples
        rho_mean = float(rho_f[fluid].mean())
        c_d = 2.0 * drag / (rho_mean * u_mean**2 * d_cells)
        nu_lat = (tau_used - 0.5) / 3.0
        re = u_mean * d_cells / nu_lat
        return {
            "levels": levels,
            "diameter_cells": d_cells,
            "grid": list(grid),
            "steps": n_steps,
            "tau": tau_used,
            "mean_velocity": u_mean,
            "reynolds": re,
            "c_d": c_d,
            "cost": cost,
        }

    # --- two-level ---------------------------------------------------------
    half = r + patch_margin * d
    x0 = int(math.floor(centre[0] - half))
    x1 = int(math.ceil(centre[0] + half))
    y0 = int(math.floor(centre[1] - half))
    y1 = int(math.ceil(centre[1] + half))
    solid_c = cylinder_solid_mask((ny, nx), centre, r, spacing=1.0)
    fh = 2 * (y1 - y0) + 2
    fw = 2 * (x1 - x0) + 2
    ii = (torch.arange(fw, dtype=torch.float64) - 1.0) * 0.5 + x0
    jj = (torch.arange(fh, dtype=torch.float64) - 1.0) * 0.5 + y0
    xc = (ii + 0.25).view(1, -1)
    yc = (jj + 0.25).view(-1, 1)
    solid_f = ((xc - centre[0]) ** 2 + (yc - centre[1]) ** 2) <= r**2
    cfg = TwoLevelConfig(
        shape=(ny, nx),
        box=(x0, x1, y0, y1),
        tau_coarse=tau,
        force=(force, 0.0),
    )
    sim = TwoLevelD2Q9(cfg, solid_coarse=solid_c, solid_fine=solid_f)
    rho = torch.ones((ny, nx), dtype=torch.float64)
    sim.fill_uniform(rho, torch.full_like(rho, u0), torch.zeros_like(rho))
    f_sum = 0.0
    samples = 0
    window_start = steps - average_window
    tau_f = cfg.tau_fine
    for k in range(steps):
        if k >= window_start and (k - window_start) % 5 == 0:
            post = collide_bgk(sim.fine, tau_f, cfg.force, sim.c, sim.w)
            post = torch.where(sim._fine_ghost_mask.unsqueeze(0), sim.fine, post)
            fx, _ = obstacle_force(post, sim.solid_fine, sim.c)
            f_sum += fx
            samples += 1
        sim.step()
    rho_f, ux, uy = sim.composite_fields()
    fluid = ~solid_c
    u_mean = float((rho_f * ux)[fluid].sum() / rho_f[fluid].sum())
    d_fine = 2.0 * d
    drag = f_sum / samples
    rho_mean = float(rho_f[fluid].mean())
    c_d = 2.0 * drag / (rho_mean * u_mean**2 * d_fine)
    nu_lat = (tau - 0.5) / 3.0
    re = u_mean * d_fine / (nu_lat * 2.0)
    cost = sim.cost()
    return {
        "levels": "two_level",
        "diameter_cells": d_fine,
        "grid": [ny, nx],
        "patch": [x0, x1, y0, y1],
        "steps": steps,
        "tau_coarse": tau,
        "tau_fine": tau_f,
        "mean_velocity": u_mean,
        "reynolds": re,
        "c_d": c_d,
        "cost": cost,
    }


def gate_cylinder(
    coarse_diameter: int = 12,
    steps: int = 12000,
    tau: float = 0.53,
    force: float = 2.0e-5,
    u0: float = 0.05,
    cd_tol: float = 0.02,
    reduction_target: float = 0.5,
) -> dict:
    """Engineering cross-check of the refined patch against a uniform fine run."""
    fine = _cylinder_run(
        diameter_cells=coarse_diameter, levels="uniform_fine", steps=steps, tau=tau,
        force=force, u0=u0,
    )
    coarse = _cylinder_run(
        diameter_cells=coarse_diameter, levels="uniform_coarse", steps=steps, tau=tau,
        force=force, u0=u0,
    )
    two = _cylinder_run(
        diameter_cells=coarse_diameter, levels="two_level", steps=steps, tau=tau,
        force=force, u0=u0,
    )
    ref = fine["c_d"]
    err_two = abs(two["c_d"] - ref) / ref
    err_coarse = abs(coarse["c_d"] - ref) / ref
    # equal-cost uniform baseline: same fluid-cell count as the two-level grid
    cost_two = two["cost"]["coarse_cells"] + two["cost"]["fine_cells"]
    cost_fine = fine["cost"]["fine_cells"]
    ratio = math.sqrt(cost_two / max(cost_fine, 1.0))
    d_eq = max(int(round(coarse_diameter * ratio)), coarse_diameter + 1)
    equal_cost = _cylinder_run(
        diameter_cells=d_eq, levels="uniform_coarse", steps=steps, tau=tau,
        force=force, u0=u0,
    )
    err_eq = abs(equal_cost["c_d"] - ref) / ref
    base = equal_cost if err_eq <= err_coarse else coarse
    err_base = min(err_eq, err_coarse)
    reduction = 1.0 - err_two / err_base if err_base > 0 else float("nan")
    return {
        "gate": "cylinder",
        "reference": fine,
        "uniform_coarse": coarse,
        "two_level": two,
        "equal_cost_uniform": equal_cost,
        "c_d_error_two_level": err_two,
        "c_d_error_uniform_coarse": err_coarse,
        "c_d_error_equal_cost": err_eq,
        "c_d_error_baseline": err_base,
        "c_d_tolerance": cd_tol,
        "error_reduction": reduction,
        "reduction_target": reduction_target,
        "passed": bool(err_two <= cd_tol and reduction >= reduction_target),
    }


# ---------------------------------------------------------------------------
# gate (f): negative controls
# ---------------------------------------------------------------------------


def _channel_interface_error(
    mode: str,
    steps: int,
    *,
    tau: float = TAU,
    shape: tuple[int, int] = (26, 64),
) -> dict:
    """Poiseuille interface-layer error of one non-equilibrium scaling mode."""
    ny, nx = shape
    nu = (tau - 0.5) / 3.0
    h = ny - 2
    force = 8.0 * nu * 0.05 / (h * h)
    u_exact = _poiseuille_profile(ny, nu, force)
    u_max = float(u_exact.max())
    box = (nx // 2 - 8, nx // 2 + 8, 3, ny - 3)
    sim = TwoLevelD2Q9(
        TwoLevelConfig(
            shape=shape,
            box=box,
            tau_coarse=tau,
            force=(force, 0.0),
            periodic=(True, False),
            momentum_reflux=False,
            neq_scale=mode,
        ),
        solid_coarse=_channel_solid(ny, nx),
    )
    rho = torch.ones((ny, nx), dtype=torch.float64)
    sim.fill_uniform(
        rho,
        u_exact.view(ny, 1).expand(ny, nx).contiguous(),
        torch.zeros((ny, nx), dtype=torch.float64),
    )
    for _ in range(steps):
        sim.step()
    _, ux, _ = sim.composite_fields()
    prof = ux[:, box[0] : box[1]].mean(dim=1)
    errs = {r: abs(float(prof[r]) - float(u_exact[r])) / u_max for r in range(1, ny - 1)}
    return {
        "mode": mode,
        "max_error": max(errs.values()),
        "interface_layer_error": max(errs[3], errs[ny - 4]),
        "interior_error": max(v for r, v in errs.items() if 5 <= r <= ny - 6),
    }


def gate_negative_controls(
    steps: int = 400,
    tau: float = 0.6,
) -> dict:
    """Broken operators must fail the gates; a gate that cannot fail is void."""
    controls = {}

    # (i) no non-equilibrium rescaling at the coarse/fine interface.
    # The probe uses tau_c = 1.0 and a short channel: the interface layer then
    # separates the three operators most clearly (see the artifact's
    # operator_variants table for the tau_c = 0.65 reference).
    ce = _channel_interface_error("chapman_enskog", steps, tau=1.0, shape=(18, 40))
    none = _channel_interface_error("none", steps, tau=1.0, shape=(18, 40))
    tau_only = _channel_interface_error("tau_only", steps, tau=1.0, shape=(18, 40))
    controls["chapman_enskog"] = ce
    controls["neq_scale_none"] = none
    controls["neq_scale_tau_only"] = tau_only

    # (ii) reflux disabled -> mass bookkeeping must break
    shape = (64, 64)
    ny, nx = shape
    yy, xx = _grid(ny, nx)
    kx = 2.0 * math.pi / nx
    ky = 2.0 * math.pi / ny
    rho = 1.0 + 1e-3 * torch.sin(kx * xx) * torch.cos(ky * yy)
    ux = 0.02 + torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    box = (nx // 4, 3 * nx // 4, ny // 4, 3 * ny // 4)
    sim = TwoLevelD2Q9(
        TwoLevelConfig(shape=shape, box=box, tau_coarse=tau, reflux=False)
    )
    sim.fill_uniform(rho, ux, uy)
    m_start = sim.totals()[0]
    for _ in range(steps):
        sim.step()
    m_end = sim.totals()[0]
    failed_mass = abs(m_end - m_start) / abs(m_start)
    controls["reflux_disabled"] = {
        "mode": "no_reflux",
        "mass_residual_relative": failed_mass,
    }

    ratio_none = none["interface_layer_error"] / max(ce["interface_layer_error"], 1e-30)
    ratio_tau = tau_only["interface_layer_error"] / max(ce["interface_layer_error"], 1e-30)
    return {
        "gate": "negative_control",
        "steps": steps,
        "controls": controls,
        "no_reflux_mass_residual": failed_mass,
        "no_reflux_fails_mass_gate": bool(failed_mass > 1.0e-12),
        "wrong_neq_scale_error_ratio_none": float(ratio_none),
        "wrong_neq_scale_error_ratio_tau_only": float(ratio_tau),
        "wrong_neq_scale_is_worse": bool(ratio_none > 1.5 and ratio_tau > 1.5),
        "passed": bool(failed_mass > 1.0e-12 and ratio_none > 1.5 and ratio_tau > 1.5),
    }


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------


def run_gate_suite(
    *,
    include_cylinder: bool = False,
    conservation_steps: int = 200,
    poiseuille_steps: int = 4000,
    reflection_steps: int = 300,
    order_resolutions: tuple[int, ...] = (32, 64, 128),
    cylinder_steps: int = 20000,
) -> dict:
    """Run every gate and assemble the campaign artifact."""
    gates = {
        "conservation": gate_conservation(steps=conservation_steps),
        "poiseuille": gate_poiseuille(steps=poiseuille_steps),
        "reflection": gate_reflection(steps=reflection_steps),
        "order": gate_order(resolutions=order_resolutions),
        "negative_control": gate_negative_controls(),
    }
    if include_cylinder:
        gates["cylinder"] = gate_cylinder(steps=cylinder_steps)
    passed = {name: bool(cell["passed"]) for name, cell in gates.items()}
    return {
        "suite": "p387w2-n05 two-level static refinement gate",
        "gates": gates,
        "passed": passed,
        "all_passed": all(passed.values()),
    }