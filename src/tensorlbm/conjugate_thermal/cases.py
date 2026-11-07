"""Validation cases for the DDF thermal model (IN28-N03).

Three physics problems, each with its own machine-checkable figure of merit:

* :func:`cavity_natural_convection` — differentially heated square cavity,
  ``Ra = 1e3 ... 1e6``, ``Pr = 0.71``; Nusselt number against de Vahl Davis
  (1983) / Le Quere benchmark values.
* :func:`cylinder_forced_convection` — cross-flow over an isothermal circular
  cylinder, ``Re = 40 ... 200``, ``Pr = 0.7``; surface Nusselt distribution and
  mean Nusselt number against Churchill-Bernstein.
* :func:`conjugate_channel` — streamwise-homogeneous channel flow over a
  conductive solid slab: harmonic-mean interface coupling and the steady solid
  mid-plane temperature against the series thermal-resistance network.
* :func:`rayleigh_growth_rate` — linear growth/decay rate of a small
  perturbation, used to bracket the critical Rayleigh number.

All routines return plain numpy diagnostics so that the runner and the tests can
consume them without touching torch internals.
"""

from __future__ import annotations

import math

import torch

from .ddf2d import DDF2D, Conduction2D, bilinear_sample

__all__ = [
    "cavity_natural_convection",
    "cylinder_forced_convection",
    "conjugate_channel",
    "rayleigh_growth_rate",
    "nu_cylinder_polar",
    "churchill_bernstein",
    "de_vahl_davis_nu",
    "channel_nusselt_fully_developed",
]

#: de Vahl Davis (1983) benchmark Nusselt numbers, 2-D square cavity, Pr = 0.71
DE_VAHL_DAVIS = {1.0e3: 1.118, 1.0e4: 2.243, 1.0e5: 4.519, 1.0e6: 8.800}


def de_vahl_davis_nu(ra: float) -> float:
    """Benchmark cavity Nusselt number (de Vahl Davis 1983, Pr = 0.71)."""
    for key, val in DE_VAHL_DAVIS.items():
        if math.isclose(ra, key, rel_tol=1e-9):
            return val
    raise KeyError(f"no de Vahl Davis benchmark tabulated at Ra={ra}")


def churchill_bernstein(re: float, pr: float = 0.7) -> float:
    """Churchill-Bernstein (1977) mean Nusselt number of a cylinder in cross flow.

    ``Nu_D = 0.3 + 0.62 Re^0.5 Pr^(1/3) / (1 + (0.4/Pr)^(2/3))^(1/4)
    * (1 + (Re/282000)^(5/8))^(4/5)``
    """
    num = 0.62 * math.sqrt(re) * pr ** (1.0 / 3.0)
    den = (1.0 + (0.4 / pr) ** (2.0 / 3.0)) ** 0.25
    return 0.3 + num / den * (1.0 + (re / 282000.0) ** (5.0 / 8.0)) ** 0.8


def channel_nusselt_fully_developed() -> float:
    """Fully developed plane-Poiseuille Nusselt number, isothermal walls.

    ``Nu = 7.541`` based on the hydraulic diameter ``D_h = 2H``
    (Shah & London, *Laminar Flow Forced Convection in Ducts*, 1978).
    """
    return 7.541


# ---------------------------------------------------------------------------
# Natural convection in a differentially heated square cavity
# ---------------------------------------------------------------------------
def cavity_natural_convection(
    ra: float = 1.0e4,
    pr: float = 0.71,
    n: int = 66,
    nu: float = 0.02,
    nsteps: int = 60000,
    tol: float = 1e-6,
    check_every: int = 500,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Differentially heated square cavity with adiabatic top/bottom walls.

    Left wall ``T = 1``, right wall ``T = 0``, gravity along ``-y``;
    ``Ra = g beta dT H^3 / (nu alpha)`` with ``H = n - 2`` lattice units.
    """
    H = n - 2
    alpha = nu / pr
    gbeta = ra * nu * alpha / H**3
    d = DDF2D(n, n, nu=nu, alpha=alpha, gravity=gbeta, beta=1.0, T_ref=0.5,
              dtype=dtype)
    d.set_box_walls(1)
    left = torch.zeros(n, n, dtype=torch.bool)
    right = torch.zeros(n, n, dtype=torch.bool)
    left[:, 0] = True
    right[:, -1] = True
    d.set_temperature_bc(left, DDF2D.DIRICHLET, 1.0)
    d.set_temperature_bc(right, DDF2D.DIRICHLET, 0.0)
    d.build_thermal_links()

    xs = (torch.arange(n, dtype=dtype) - 0.5) / H
    T0 = (1.0 - xs).clamp(0.0, 1.0).reshape(1, -1).expand(n, n).clone()
    idx = torch.arange(1, n - 1, dtype=dtype)
    shape = torch.sin(math.pi * idx / H)
    T0[1:-1, 1:-1] += 1e-4 * shape.reshape(-1, 1) * shape.reshape(1, -1)
    d.init_fields(T=T0)

    hist = []
    nu_prev = None
    converged = False
    steps_done = 0
    while steps_done < nsteps:
        d.step(check_every)
        steps_done += check_every
        _, ux, uy, T = d.macroscopic()
        q = d.wall_flux_normal("x", 0)
        nu_hot = float(q[1:-1].mean()) * H / alpha
        u_max = float((ux ** 2 + uy ** 2).sqrt()[1:-1, 1:-1].max())
        hist.append({"steps": steps_done, "nu": nu_hot, "u_max": u_max})
        if nu_prev is not None and abs(nu_hot - nu_prev) < tol * max(abs(nu_hot), 1e-30):
            converged = True
            break
        nu_prev = nu_hot

    _, ux, uy, T = d.macroscopic()
    q_hot = d.wall_flux_normal("x", 0)
    q_cold = d.wall_flux_normal("x", n - 1)
    nu_hot = float(q_hot[1:-1].mean()) * H / alpha
    nu_cold = float(q_cold[1:-1].mean()) * H / alpha
    bal = abs(float(q_hot[1:-1].sum()) - float(q_cold[1:-1].sum())) / abs(
        float(q_hot[1:-1].sum())
    )
    v = (ux ** 2 + uy ** 2)[1:-1, 1:-1]
    vmax_idx = int(torch.argmax(v))
    iy = vmax_idx // (n - 2) + 1
    ix = vmax_idx % (n - 2) + 1
    bench = None
    for key in DE_VAHL_DAVIS:
        if math.isclose(ra, key, rel_tol=1e-9):
            bench = DE_VAHL_DAVIS[key]
    return {
        "case": "natural_convection_cavity",
        "ra": ra, "pr": pr, "nu": float(nu), "alpha": float(alpha), "H": H,
        "gbeta": float(gbeta), "resolution": n,
        "steps": steps_done, "converged": converged,
        "nu_hot": nu_hot, "nu_cold": nu_cold, "nu_mean": 0.5 * (nu_hot + nu_cold),
        "nu_benchmark": bench,
        "u_max": float(v.sqrt().max()), "v_max_loc": (iy, ix),
        "energy_balance_rel": bal,
        "history": hist,
        "fields": {
            "T": T.cpu().numpy().copy(),
            "ux": ux.cpu().numpy().copy(),
            "uy": uy.cpu().numpy().copy(),
        },
    }


# ---------------------------------------------------------------------------
# Forced convection over an isothermal circular cylinder
# ---------------------------------------------------------------------------
def nu_cylinder_polar(
    T: torch.Tensor, cx: float, cy: float, r_wall: float, d_eff: float,
    ntheta: int = 720, dtype: torch.dtype = torch.float64,
) -> dict:
    """Surface Nusselt distribution from polar sampling of the temperature field.

    The radial derivative at the half-cell wall radius ``r_wall`` comes from a
    three-point quadratic fit through ``r_wall + {0, 1, 2}`` (weights
    ``(-1.5, 2.0, -0.5)``); ``Nu(theta) = -dT/dr * d_eff / dT_ref``.
    """
    theta = torch.arange(ntheta, dtype=dtype) * (2.0 * math.pi / ntheta)
    ct, st = torch.cos(theta), torch.sin(theta)
    s0 = bilinear_sample(T, cx + r_wall * ct, cy + r_wall * st)
    s1 = bilinear_sample(T, cx + (r_wall + 1) * ct, cy + (r_wall + 1) * st)
    s2 = bilinear_sample(T, cx + (r_wall + 2) * ct, cy + (r_wall + 2) * st)
    dTdr = -1.5 * s0 + 2.0 * s1 - 0.5 * s2
    nu = -dTdr * d_eff
    # the inflow comes from -x, so theta = pi marks the stagnation point
    front = (theta - math.pi).abs() < 0.15
    rear = (theta < 0.15) | (theta > 2 * math.pi - 0.15)
    return {
        "theta": theta.cpu().numpy().copy(),
        "nu_theta": nu.cpu().numpy().copy(),
        "nu_mean": float(nu.mean()),
        "nu_front": float(nu[front].mean()),
        "nu_rear": float(nu[rear].mean()),
    }


def cylinder_forced_convection(
    re: float = 100.0,
    pr: float = 0.7,
    n_d: int = 20,
    length_d: float = 18.0,
    width_d: float = 8.0,
    nu: float = 0.02,
    nsteps: int = 40000,
    check_every: int = 250,
    tol: float = 1e-6,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Cross-flow over an isothermal heated cylinder in a plane channel.

    ``T_cyl = 1``, ``T_in = 0``, no-slip adiabatic top/bottom walls, plug inflow
    with the cylinder at one third of the length.  ``Re = U D / nu`` with
    ``D = n_d + 1`` (half-way wall convention), ``Pr = nu / alpha``.
    """
    D = n_d + 1.0
    U = re * nu / D
    alpha = nu / pr
    nx = max(int(round(length_d * D)), 60)
    ny = max(int(round(width_d * D)) + 2, 26)
    cx = nx / 3.0
    cy = (ny - 1) / 2.0
    R = n_d / 2.0

    d = DDF2D(ny, nx, nu=nu, alpha=alpha, dtype=dtype)
    d.set_box_walls(1, walls="y")
    yy = torch.arange(ny, dtype=torch.float64).reshape(-1, 1)
    xx = torch.arange(nx, dtype=torch.float64).reshape(1, -1)
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= R * R
    d.set_solid(disk)
    d.set_temperature_bc(disk, DDF2D.DIRICHLET, 1.0)
    d.build_thermal_links()
    d.init_uniform(ux=U, T=0.0)
    d.inflow = {"ux": U, "T": 0.0}
    d.outflow = True

    n_links = int(sum(int(v.numel()) for v in d._links.nodes))
    hist = []
    nu_prev = None
    converged = False
    steps_done = 0
    while steps_done < nsteps:
        d.step(check_every)
        steps_done += check_every
        _, ux, uy, T = d.macroscopic()
        res = nu_cylinder_polar(T, cx, cy, R + 0.5, D)
        nu_now = res["nu_mean"]
        hist.append({"steps": steps_done, "nu": nu_now})
        if nu_prev is not None and abs(nu_now - nu_prev) < tol * abs(nu_now):
            converged = True
            break
        nu_prev = nu_now

    rho, ux, uy, T = d.macroscopic()
    res = nu_cylinder_polar(T, cx, cy, R + 0.5, D)
    rows = slice(1, ny - 1)
    q_in = float((rho[rows, 1] * ux[rows, 1] * T[rows, 1]).sum())
    q_out = float((rho[rows, -2] * ux[rows, -2] * T[rows, -2]).sum())
    nu_int = (q_out - q_in) / (math.pi * alpha)
    return {
        "case": "cylinder_forced_convection",
        "re": re, "pr": pr, "nu": float(nu), "alpha": float(alpha),
        "U": float(U), "D": float(D), "R": float(R), "nx": nx, "ny": ny,
        "blockage": float(D / (ny - 2)),
        "steps": steps_done, "converged": converged,
        "nu_mean": res["nu_mean"], "nu_front": res["nu_front"],
        "nu_rear": res["nu_rear"], "nu_integral": float(nu_int),
        "nu_churchill_bernstein": churchill_bernstein(re, pr),
        "n_wall_links": n_links,
        "history": hist,
        "theta": res["theta"], "nu_theta": res["nu_theta"],
        "fields": {
            "T": T.cpu().numpy().copy(),
            "ux": ux.cpu().numpy().copy(),
            "uy": uy.cpu().numpy().copy(),
        },
    }


# ---------------------------------------------------------------------------
# Conjugate heat transfer: channel flow over a conductive solid slab
# ---------------------------------------------------------------------------
def conjugate_channel(
    hf: int = 32,
    ts: int = 10,
    length: int = 256,
    k_ratio: float = 5.0,
    re: float = 50.0,
    pr: float = 0.7,
    nu: float = 0.02,
    nsteps: int = 40000,
    check_every: int = 250,
    tol: float = 1e-12,
    dtype: torch.dtype = torch.float32,
    T_in: float = 1.0,
    T_cold: float = 0.0,
) -> dict:
    """Conjugate heat transfer in a plane channel above a solid slab.

    Geometry (lattice units, half-way wall convention)::

        y = hf + 0.5   adiabatic no-slip wall
        fluid          rows 1 .. hf   (row 0 is the interface ghost row)
        y = 0.5        fluid/solid interface (coupling plane)
        solid          rows 1 .. ts   (row 0 is the cold-wall ghost row)
        y = -ts + 0.5  isothermal cold wall, T = T_cold

    Interface temperature ``T_i = (k_s T_s + k_f T_f)/(k_s + k_f)`` gives the
    exact series resistance of the layered medium, so the flux seen by the fluid
    and by the solid are identical by construction.  The flow is a
    streamwise-homogeneous Poiseuille profile (body-force driven, initialised
    analytically) with the same profile imposed at the inlet at ``T = T_in``.
    """
    k_f = float(nu / pr)          # k = alpha * rho * cp, rho = cp = 1
    k_s = k_ratio * k_f
    alpha_f = nu / pr
    alpha_s = k_s

    nrows_f = hf + 2
    fluid = DDF2D(nrows_f, length, nu=nu, alpha=alpha_f, dtype=dtype)
    fluid.set_box_walls(1, walls="y")
    bottom = torch.zeros(nrows_f, length, dtype=torch.bool)
    bottom[0, :] = True
    fluid.set_temperature_bc(bottom, DDF2D.DIRICHLET, 0.0)
    fluid.build_thermal_links()

    H_chan = float(hf)
    D_h = 2.0 * H_chan          # hydraulic diameter of a plane channel
    u_max = re * nu / H_chan    # Re = u_max * H_chan / nu
    s = torch.arange(nrows_f, dtype=torch.float64).reshape(-1, 1)
    prof = 4.0 * u_max * ((s - 0.5) / D_h) * (1.0 - (s - 0.5) / D_h)
    prof[(s < 0.5).reshape(-1), :] = 0.0
    prof[(s > hf + 0.5).reshape(-1), :] = 0.0
    ux0 = prof.expand(nrows_f, length).contiguous().to(dtype).clone()
    T_init = torch.full((nrows_f, length), float(T_in), dtype=dtype)
    T_init[0, :] = 0.0
    fluid.init_fields(T=T_init, ux=ux0)
    # body force that sustains u_max for a channel of half-height hf/2
    fluid.body_fx = 8.0 * nu * u_max / H_chan ** 2
    fluid.inflow = {"ux": prof.reshape(-1).to(dtype), "T": T_in}
    fluid.outflow = True

    solid = Conduction2D(ts + 1, length, alpha=alpha_s, dtype=dtype)
    solid.init_uniform(T_cold)
    solid.add_dirichlet_link(4, 1, T_cold)  # cold wall, half a cell below row 1
    tw_iface = solid.add_dirichlet_link(3, ts, T_cold)  # interface above row ts
    _flat = torch.arange((ts + 1) * length, device=solid.device).reshape(ts + 1, length)
    solid.add_link(2, nodes=_flat[:, 0], T=None)     # adiabatic ends (no x wrap)
    solid.add_link(1, nodes=_flat[:, -1], T=None)

    k_sum = k_s + k_f
    hist = []
    prev = None
    converged = False
    steps_done = 0

    def _couple() -> None:
        solid.macroscopic()
        T_f1 = fluid.T[1, :].to(torch.float64)
        T_s = solid.T[ts, :].to(torch.float64)
        T_i = (k_s * T_s + k_f * T_f1) / k_sum
        fluid.T_wall[0, :] = T_i.to(fluid.T_wall.dtype)
        tw_iface[:] = T_i.to(tw_iface.dtype)

    while steps_done < nsteps:
        _couple()
        for _ in range(check_every):
            fluid.step(1)
            solid.step(1)
            _couple()
        steps_done += check_every
        solid.macroscopic()
        T_f1 = fluid.T[1, :].to(torch.float64)
        T_s = solid.T[ts, :].to(torch.float64)
        T_i = (k_s * T_s + k_f * T_f1) / k_sum
        q_f = 2.0 * k_f * (T_f1 - T_i)
        q_s = 2.0 * k_s * (T_i - T_s)
        resid = float((q_f - q_s).abs().max() / q_f.abs().max())
        T_center = solid.T[ts // 2 + 1, :].to(torch.float64)
        drift = 0.0 if prev is None else float(
            (T_center - prev).abs().max())
        hist.append({"steps": steps_done, "resid": resid, "drift": drift})
        if prev is not None and drift < tol:
            converged = True
            break
        prev = T_center.clone()

    Tf = fluid.T.to(torch.float64)
    ux = fluid.ux.to(torch.float64)
    rho = fluid.rho.to(torch.float64)
    Ts = solid.T.to(torch.float64)
    T_f1 = Tf[1, :]
    T_s = Ts[ts, :]
    T_i = (k_s * T_s + k_f * T_f1) / k_sum
    q_f = 2.0 * k_f * (T_f1 - T_i)
    q_s = 2.0 * k_s * (T_i - T_s)
    resid_iface = float((q_f - q_s).abs().max() / q_f.abs().max())
    flux = rho * ux * Tf
    T_bulk = flux[1:hf + 1, :].sum(0) / (rho * ux)[1:hf + 1, :].sum(0).clamp_min(1e-30)
    T_center = Ts[ts // 2 + 1, :]
    q_cold = 2.0 * k_s * (Ts[1, :] - T_cold)  # flux leaving through the cold wall
    # energy flux through a plane: enthalpy advection minus axial conduction
    cond_in = alpha_f * (Tf[1:hf + 1, 2] - Tf[1:hf + 1, 0]) / 2.0
    cond_out = alpha_f * (Tf[1:hf + 1, -1] - Tf[1:hf + 1, -3]) / 2.0
    q_in = float(flux[1:hf + 1, 1].sum() - cond_in.sum())
    q_out = float(flux[1:hf + 1, -2].sum() - cond_out.sum())
    q_wall_loss = float(q_cold[1:-1].sum())
    global_bal = abs(q_in - q_out - q_wall_loss) / max(abs(q_in), 1e-30)
    h_conv = channel_nusselt_fully_developed() * k_f / D_h
    r_tot = 1.0 / h_conv + ts / k_s
    T_center_net = T_cold + (T_bulk - T_cold) * (ts / (2.0 * k_s)) / r_tot
    dev = (T_center - T_center_net).abs() / (T_bulk - T_cold).abs().clamp_min(1e-30)
    h_loc = q_f / (T_bulk - T_i).clamp_min(1e-12)
    nu_loc = h_loc * D_h / k_f
    return {
        "case": "conjugate_channel",
        "hf": hf, "ts": ts, "length": length, "k_ratio": k_ratio,
        "k_f": k_f, "k_s": k_s, "re": re, "pr": pr, "nu": float(nu),
        "u_max": float(u_max), "steps": steps_done, "converged": converged,
        "resid_interface": resid_iface,
        "resid_global_energy": global_bal,
        "q_interface": q_f.cpu().numpy().copy(),
        "q_solid": q_s.cpu().numpy().copy(),
        "q_cold": q_cold.cpu().numpy().copy(),
        "T_interface": T_i.cpu().numpy().copy(),
        "T_center": T_center.cpu().numpy().copy(),
        "T_center_network": T_center_net.cpu().numpy().copy(),
        "T_bulk": T_bulk.cpu().numpy().copy(),
        "nu_local": nu_loc.cpu().numpy().copy(),
        "h_conv": float(h_conv), "r_total": float(r_tot),
        "dev_center_max": float(dev[1:-1].max()),
        "dev_center_tail": float(dev[-20:].mean()),
        "history": hist,
        "fields": {
            "T_fluid": Tf.cpu().numpy().copy(),
            "T_solid": Ts.cpu().numpy().copy(),
            "ux": ux.cpu().numpy().copy(),
        },
    }


# ---------------------------------------------------------------------------
# Linear growth rate of a perturbation of the conductive state
# ---------------------------------------------------------------------------
def rayleigh_growth_rate(
    ra: float,
    pr: float = 0.71,
    n: int = 34,
    nu: float = 0.02,
    nsteps: int = 30000,
    amp: float = 1e-6,
    sample_every: int = 200,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Exponential growth rate of a small perturbation in the square cavity.

    The base state is pure conduction plus a ``sin(pi x) sin(pi y)`` temperature
    perturbation of amplitude ``amp``; ``sigma = 0.5 d ln E/dt`` with
    ``E = sum(u^2 + v^2)`` fitted over the linear window.  ``sigma > 0`` marks a
    supercritical Rayleigh number.
    """
    H = n - 2
    alpha = nu / pr
    gbeta = ra * nu * alpha / H**3
    d = DDF2D(n, n, nu=nu, alpha=alpha, gravity=gbeta, beta=1.0, T_ref=0.5, dtype=dtype)
    d.set_box_walls(1)
    left = torch.zeros(n, n, dtype=torch.bool)
    right = torch.zeros(n, n, dtype=torch.bool)
    left[:, 0] = True
    right[:, -1] = True
    d.set_temperature_bc(left, DDF2D.DIRICHLET, 1.0)
    d.set_temperature_bc(right, DDF2D.DIRICHLET, 0.0)
    d.build_thermal_links()
    xs = (torch.arange(n, dtype=dtype) - 0.5) / H
    T0 = (1.0 - xs).clamp(0.0, 1.0).reshape(1, -1).expand(n, n).clone()
    # asymmetric random perturbation (a symmetric seed cannot excite the
    # symmetry-breaking roll that becomes unstable at Ra_c)
    gen = torch.Generator(device="cpu").manual_seed(int(round(ra)) + 12345)
    noise = torch.rand(n - 2, n - 2, generator=gen, dtype=torch.float64)
    noise = (noise - noise.mean()) / noise.std().clamp_min(1e-30)
    T0[1:-1, 1:-1] += amp * noise.to(dtype)
    d.init_fields(T=T0)

    ts, Es = [], []
    for k in range(nsteps):
        d.step(1)
        if (k + 1) % sample_every == 0:
            _, ux, uy, _ = d.macroscopic()
            E = float((ux[1:-1, 1:-1] ** 2 + uy[1:-1, 1:-1] ** 2).sum())
            ts.append(k + 1)
            Es.append(E)
    Es_t = torch.tensor(Es, dtype=torch.float64)
    Emax = float(Es_t.max())
    if Emax <= 0.0 or len(Es) < 6:
        sigma = float("-inf")
        window = (0, 0)
    else:
        # late window: the early samples carry the impulsive start, not the
        # linear eigenmode
        k0 = int(0.4 * len(Es))
        tt = torch.tensor(ts[k0:], dtype=torch.float64)
        yy = torch.log(Es_t[k0:].clamp_min(1e-300))
        A = torch.stack([tt, torch.ones_like(tt)], dim=1)
        coef = torch.linalg.lstsq(A, yy.reshape(-1, 1)).solution
        sigma = float(coef[0]) * 0.5
        window = (int(tt.min()), int(tt.max()))
    return {
        "case": "rayleigh_growth",
        "ra": ra, "pr": pr, "resolution": n, "nu": float(nu),
        "sigma": sigma, "e_final": float(Es[-1]) if Es else 0.0,
        "e_max": float(Es_t.max()) if Es else 0.0,
        "fit_window": window,
        "t": ts, "E": Es,
    }