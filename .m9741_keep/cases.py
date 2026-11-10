"""The three invariance test cases.

Case 01 - inclined plane-Poiseuille channel (grid rotation)
    A periodic array of parallel channels tilted by ``theta`` w.r.t. the lattice,
    driven by a body force along the channel axis.  Exact solution
    ``u(s) = G / (2 nu) * (H^2/4 - s^2)``.  The physical problem does not depend
    on ``theta``; only its sampling on the lattice does.  Measured: the peak
    axial velocity, the volume flux, and the wall-normal profile, for four tilts
    and three resolutions - so that the *convergence order* can be separated from
    the *orientation* effect.

Case 02 - circular cylinder at Re = 100 / 200 (grid rotation)
    The same physical body and free stream, discretised on lattices rotated by
    ``theta``.  Measured: mean drag coefficient and Strouhal number.

Case 03 - Taylor-Green vortex with a Galilean boost (frame boost)
    The same decaying vortex observed from frames moving at
    ``U = Ma * c_s``.  Measured: the decay-rate defect (fitted against ``Ma^2``)
    and the drift of the vortex (a *strict* Galilean invariance statement: the
    solution must translate, not deform).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .galilean import (
    fit_ma2,
    DriftTracker,
    momentum_mach,
    uniform_flow_fixed_point,
    unwrap,
)
from .geometry import Circle, Frame, TiltedChannelArray, make_ray_distance, sample
from .lattice import CS, Simulation, build_wall_links, equilibrium, macroscopic

# Rational tilts: the doubly periodic tilted channel array requires
#   tan(theta) = k / m  rational,  perpendicular wall spacing d = L / sqrt(m^2+k^2)
# The requested 15 deg / 30 deg are therefore represented by the closest rational
# tilts 14.036 deg (= atan 1/4) and 26.565 deg (= atan 1/2); this is recorded in
# the report as a declared deviation and does not weaken the test, because the
# geometry is defined analytically for every tilt.
RATIONAL_TILTS: dict[float, tuple[int, int]] = {
    0.0: (1, 0),
    14.036: (4, 1),
    26.565: (2, 1),
    45.0: (1, 1),
}


# =============================================================================
# Case 01: inclined plane-Poiseuille channel
# =============================================================================
def inclined_channel(
    theta_deg: float,
    L: int = 128,
    width: float = 24.0,
    nu: float = 0.05,
    u_peak_target: float = 0.02,
    max_steps: int = 30000,
    tol: float = 1e-11,
    profile_bins: int = 21,
) -> dict:
    """Plane Poiseuille flow in a channel tilted by ``theta_deg``."""
    m, k = RATIONAL_TILTS[float(theta_deg)]
    spacing = L / math.hypot(m, k)
    if width >= spacing:
        raise ValueError(f"width {width} must stay below the wall spacing {spacing:.3f}")
    geom = TiltedChannelArray(theta_deg, width, spacing)
    phi = sample(geom, L, L)
    fluid = phi <= 0.0
    if fluid.sum() == 0:
        raise ValueError("empty fluid region - check the geometry")
    ray = make_ray_distance(geom.sdf)
    links = build_wall_links(phi, ray)

    g = 8.0 * nu * u_peak_target / (width * width)
    axis = geom.axis
    force = (g * axis[0], g * axis[1])
    tau = 3.0 * nu + 0.5
    sim = Simulation(L, L, tau, force=force)

    # start from the analytical profile: only the discretisation error has to be
    # forgotten, not the diffusive transient of the initial condition
    s = geom.normal[0] * np.arange(L)[None, :] + geom.normal[1] * np.arange(L)[:, None]
    s = np.mod(s + 0.5 * spacing, spacing) - 0.5 * spacing
    u_analytic = g / (2.0 * nu) * (0.25 * width**2 - s**2)
    u_analytic = np.where(fluid, u_analytic, 0.0)
    sim.f = equilibrium(
        np.ones((L, L)), u_analytic * axis[0], u_analytic * axis[1]
    )
    sim.add_wall_links(links)

    history: list[float] = []
    converged = False
    steps = 0
    _, ux_prev, uy_prev = macroscopic(sim.f)
    for steps in range(1, max_steps + 1):
        sim.step()
        if steps % 100 == 0:
            rho, ux, uy = macroscopic(sim.f)
            change = max(np.abs(ux - ux_prev).max(), np.abs(uy - uy_prev).max())
            ux_prev, uy_prev = ux, uy
            history.append(float(change))
            if change < tol:
                converged = True
                break

    rho, ux, uy = macroscopic(sim.f)
    u_axial = ux * axis[0] + uy * axis[1]
    u_peak = float(np.abs(u_axial[fluid]).max())
    u_mean = float(u_axial[fluid].mean())
    flux = width * u_mean  # volume flux per unit channel length
    u_peak_exact = g * width**2 / (8.0 * nu)
    flux_exact = g * width**3 / (12.0 * nu)

    # wall-normal profile: bin the fluid nodes by the transverse coordinate
    s_fluid = s[fluid]
    u_fluid = u_axial[fluid]
    edges = np.linspace(-0.5 * width, 0.5 * width, profile_bins + 1)
    idx = np.clip(np.digitize(s_fluid, edges) - 1, 0, profile_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prof_mean = np.zeros(profile_bins)
    prof_exact = np.zeros(profile_bins)
    for b in range(profile_bins):
        sel = idx == b
        prof_mean[b] = u_fluid[sel].mean() if sel.any() else np.nan
        sc = centers[b]
        prof_exact[b] = g / (2.0 * nu) * (0.25 * width**2 - sc**2)

    return {
        "theta_requested_deg": float(theta_deg),
        "theta_used_deg": math.degrees(math.atan2(k, m)),
        "L": int(L),
        "width": float(width),
        "spacing": float(spacing),
        "nu": float(nu),
        "tau": float(tau),
        "body_force": float(g),
        "steps": int(steps),
        "converged": bool(converged),
        "residual_velocity_change": float(history[-1]) if history else float("nan"),
        "fluid_fraction": float(fluid.sum()) / float(L * L),
        "fluid_fraction_exact": float(width / spacing),
        "n_wall_links": links.size,
        "u_peak": u_peak,
        "u_peak_exact": float(u_peak_exact),
        "u_peak_error": float((u_peak - u_peak_exact) / u_peak_exact),
        "flux": float(flux),
        "flux_exact": float(flux_exact),
        "flux_error": float((flux - flux_exact) / flux_exact),
        "profile_s": centers.tolist(),
        "profile_u": prof_mean.tolist(),
        "profile_exact": prof_exact.tolist(),
    }


# =============================================================================
# Case 02: circular cylinder, rotated lattice
# =============================================================================
def cylinder_flow(
    theta_deg: float,
    Re: float = 100.0,
    diameter: int = 16,
    length_over_d: float = 24.0,
    span_over_d: float = 16.0,
    u_in: float = 0.06,
    transient: int = 8000,
    sampling: int = 20000,
    seed: int = 0,
) -> dict:
    """Flow past a circular cylinder, physical body fixed, lattice rotated."""
    nx = int(round(length_over_d * diameter))
    ny = int(round(span_over_d * diameter))
    radius = 0.5 * diameter
    centre = (6.0 * diameter, 0.5 * ny)
    frame = Frame(theta_deg, origin=centre)
    circle = Circle(radius, frame)
    phi = sample(circle, nx, ny)
    ray = make_ray_distance(circle.sdf)
    links = build_wall_links(phi, ray, object_id=1)

    nu = u_in * diameter / float(Re)
    tau = 3.0 * nu + 0.5
    sim = Simulation(
        nx,
        ny,
        tau,
        bc={"w": "velocity", "e": "pressure", "s": "free_slip", "n": "free_slip"},
    )
    ulat = frame.to_lattice_velocity(u_in, 0.0)
    sim.inlet_velocity = np.array(ulat)
    sim.outlet_density = 1.0
    sim.add_wall_links(links)

    # symmetric initial field + a small transverse perturbation that seeds the
    # shedding instability identically for every orientation
    rho0, ux0, uy0 = macroscopic(sim.f)
    ux0 = np.full((ny, nx), ulat[0])
    uy0 = np.full((ny, nx), ulat[1])
    rng = np.random.default_rng(seed)
    wobble = 1e-3 * u_in * rng.standard_normal((ny, nx))
    sim.f = equilibrium(rho0, ux0, uy0 + wobble)

    rho_f = 1.0
    area = 0.5 * rho_f * u_in * u_in * diameter
    cds: list[float] = []
    cls: list[float] = []
    times: list[float] = []
    for step in range(1, transient + sampling + 1):
        sim.step()
        if step > transient:
            fx, fy = links.force
            fpx, fpy = frame.to_physical_velocity(fx, fy)
            # the sign convention of Ladd's momentum exchange gives the force the
            # fluid exerts on the solid along +c_i; reverse it for the force on
            # the body in the flow direction
            cds.append(float(-fpx / area))
            cls.append(float(-fpy / area))
            times.append(float(step))

    cds_a = np.asarray(cds)
    cls_a = np.asarray(cls)
    cd_mean = float(cds_a.mean())
    cl_rms = float(np.sqrt(np.mean(cls_a**2)))
    cl_amp = float(0.5 * (cls_a.max() - cls_a.min()))

    # Strouhal number from the dominant frequency of the lift history
    dt = times[1] - times[0]
    sig = cls_a - cls_a.mean()
    n = sig.size
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(sig * win))
    freqs = np.fft.rfftfreq(n, d=dt)
    spec[0] = 0.0
    kpk = int(np.argmax(spec))
    strouhal = float("nan")
    if 0 < kpk < spec.size - 1:
        # parabolic refinement of the peak location
        y0, y1, y2 = spec[kpk - 1], spec[kpk], spec[kpk + 1]
        denom = y0 - 2.0 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        f_dom = (kpk + shift) * (freqs[1] - freqs[0])
        strouhal = f_dom * diameter / u_in if f_dom > 0 else float("nan")

    # sanity: recirculation length / near-field diagnostics
    rho_fin, ux_fin, uy_fin = macroscopic(sim.f)
    u_mag = np.hypot(ux_fin, uy_fin)
    return {
        "theta_deg": float(theta_deg),
        "Re": float(Re),
        "diameter": int(diameter),
        "nx": nx,
        "ny": ny,
        "blockage": float(diameter / ny),
        "u_in": float(u_in),
        "nu": float(nu),
        "tau": float(tau),
        "transient": int(transient),
        "sampling": int(sampling),
        "n_wall_links": links.size,
        "Cd": cd_mean,
        "Cl_rms": cl_rms,
        "Cl_amplitude": cl_amp,
        "St": strouhal,
        "max_mach": float(u_mag.max() / CS),
        "density_min": float(rho_fin.min()),
        "density_max": float(rho_fin.max()),
        "cd_history": cds_a[-4000:].tolist(),
        "cl_history": cls_a[-4000:].tolist(),
        "time_history": times[-4000:],
    }


# =============================================================================
# Case 03: Taylor-Green vortex with a Galilean boost
# =============================================================================
def taylor_green_boost(
    n: int = 64,
    nu: float = 0.02,
    amplitude: float = 0.05,
    mach_list: tuple[float, ...] = (0.0, 0.025, 0.05, 0.075, 0.10),
    steps: int = 4000,
    sample_every: int = 20,
    discard: int = 400,
    boost_direction_deg: float = 0.0,
) -> dict:
    """Taylor-Green vortex seen from frames moving at ``U = Ma * c_s``."""
    k = 2.0 * math.pi / n
    tau = 3.0 * nu + 0.5
    x = np.arange(n, dtype=np.float64)[None, :]
    y = np.arange(n, dtype=np.float64)[:, None]
    th = math.radians(boost_direction_deg)
    results = []
    for mach in mach_list:
        u_boost = np.array([mach * CS * math.cos(th), mach * CS * math.sin(th)])
        ux = u_boost[0] + amplitude * np.cos(k * x) * np.sin(k * y)
        uy = u_boost[1] - amplitude * np.sin(k * x) * np.cos(k * y)
        sim = Simulation(n, n, tau)
        sim.f = equilibrium(np.ones((n, n)), np.broadcast_to(ux, (n, n)).copy(),
                            np.broadcast_to(uy, (n, n)).copy())
        tracker = DriftTracker((n, n), k)
        t_hist: list[float] = []
        amp: list[float] = []
        ph_a: list[float] = []
        ph_b: list[float] = []
        energy: list[float] = []
        for step in range(1, steps + 1):
            sim.step()
            if step % sample_every == 0:
                rho_s, ux_s, uy_s = macroscopic(sim.f)
                fluctu_x = ux_s - ux_s.mean()
                fluctu_y = uy_s - uy_s.mean()
                amp.append(float(np.sqrt(np.mean(fluctu_x**2 + fluctu_y**2))))
                energy.append(0.5 * float(np.mean(fluctu_x**2 + fluctu_y**2)))
                a, b = tracker(ux_s, uy_s)
                ph_a.append(a)
                ph_b.append(b)
                t_hist.append(float(step))
        t_arr = np.asarray(t_hist)
        sel = t_arr >= discard
        amp_arr = np.asarray(amp)
        # velocity amplitude of the (1,1) mode decays as exp(-2 nu k^2 t)
        log_amp = np.log(amp_arr[sel])
        slope, intercept = np.polyfit(t_arr[sel], log_amp, 1)
        resid = float(np.sqrt(np.mean((log_amp - (slope * t_arr[sel] + intercept)) ** 2)))
        # drift (strict Galilean invariance): the vortex centre must move at U
        a_unwrapped = unwrap(ph_a, 2.0 * math.pi / k)
        b_unwrapped = unwrap(ph_b, 2.0 * math.pi / k)
        drift_x = float(np.polyfit(t_arr[sel], a_unwrapped[sel], 1)[0])
        drift_y = float(np.polyfit(t_arr[sel], b_unwrapped[sel], 1)[0])
        results.append(
            {
                "mach": float(mach),
                "u_boost": u_boost.tolist(),
                "decay_exponent": float(-slope),
                "decay_exponent_exact": float(2.0 * nu * k**2),
                "fit_residual": resid,
                "drift_x": drift_x,
                "drift_y": drift_y,
                "drift_error_rel": float(
                    np.hypot(drift_x - u_boost[0], drift_y - u_boost[1]) / max(np.linalg.norm(u_boost), 1e-30)
                ) if mach > 0 else 0.0,
                "energy": energy,
                "t": t_hist,
            }
        )

    base = results[0]["decay_exponent"]
    mas = np.array([r["mach"] for r in results])
    defects = np.array([abs(r["decay_exponent"] / base - 1.0) if base > 0 else 0.0 for r in results])
    law = fit_ma2(mas, defects)
    for r, d in zip(results, defects):
        r["decay_defect"] = float(d)
    fixed_point = uniform_flow_fixed_point(n=32, tau=tau, boost=(0.05 * CS, 0.0), steps=200)
    return {
        "n": int(n),
        "nu": float(nu),
        "tau": float(tau),
        "k": float(k),
        "amplitude": float(amplitude),
        "steps": int(steps),
        "sample_every": int(sample_every),
        "mach_list": [float(m) for m in mach_list],
        "cases": results,
        "law": law,
        "uniform_flow_fixed_point": fixed_point,
    }