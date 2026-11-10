"""The three verification cases of the sediment / scour face.

Case 1 -- Shields threshold (:func:`shields_case`)
    Uniform sand d = 0.2 mm.  The LBM plane channel supplies the velocity field
    and, through the exact discrete momentum balance ``rho g H = tau_w``, the
    friction velocity; the Shields curve (Soulsby & Whitehouse 1991 fit of
    Shields 1936) supplies the threshold.  The acceptance number is
    ``theta_cr(d = 0.2 mm)`` against the classical 0.045.

Case 2 -- Rouse equilibrium profile (:func:`rouse_case`)
    Open channel, ``u_* = 0.05 m/s``, h = 0.3 m.  The LBM channel supplies
    ``u_*``, then the equilibrium suspended-sediment profile is obtained three
    ways -- closed form (Rouse 1937), a second-order finite-volume
    advection-diffusion solve, and a Lagrangian random walk of 5e4 grains -- and
    the three are compared.

Case 3 -- single-pile equilibrium scour (:func:`pile_scour_case`)
    D = 0.1 m, h = 0.3 m, live bed.  Plan-view depth-averaged D2Q9 flow around
    the pile (bottom-drag closure) coupled to a total-load Exner bed update
    (Meyer-Peter-Mueller bed load + van Rijn/Rouse suspended load) with an
    angle-of-repose limiter.

    Documented model-form limitation: a depth-averaged model cannot resolve the
    horseshoe vortex / downflow, which is the dominant mechanism of pile scour
    in 3-D.  The case therefore also reports the ``depth_feedback=False``
    variant, so the sensitivity to that closure is visible instead of hidden,
    and the runner marks C3 against the Sumer & Fredsoe (2002) band 1.0-1.5
    without any fitting to it.

Everything returns plain dictionaries in SI units so that the test-suite can
assert on numbers and the runner can serialise them.
"""

from __future__ import annotations

import numpy as np

from .lattice import D2Q9Channel, equilibrium
from .sediment import (
    G,
    KAPPA,
    Sediment,
    avalanche_relax,
    rouse_profile,
    shields_critical_soulsby,
    shields_curve_digitised,
    total_load_transport,
    total_load_transport_vec,
)

__all__ = ["shields_case", "rouse_case", "pile_scour_case", "log_law_fit"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def log_law_fit(y: np.ndarray, u: np.ndarray, y_min: float = 8.0, y_max: float | None = None):
    """Least-squares fit of ``u = (u_*/kappa) ln(y/z0)`` to an inner profile.

    Returns ``(u_star, z0, rms)``.  ``z0`` is *fitted*, so ``u_*`` is independent
    of the solver's sublayer treatment and can be compared against the
    momentum-balance friction velocity.
    """
    mask = y > y_min
    if y_max is not None:
        mask &= y <= y_max
    yy, uu = y[mask], u[mask]
    if yy.size < 5:
        return float("nan"), float("nan"), float("nan")
    a, b = np.polyfit(np.log(yy), uu, 1)
    u_star = a * KAPPA
    z0 = float(np.exp(-b / a)) if abs(a) > 1e-30 else float("nan")
    rms = float(np.sqrt(np.mean((np.polyval((a, b), np.log(yy)) - uu) ** 2)))
    return float(u_star), z0, rms


def _turbulent_channel(nx: int, h_cells: int, *, re_h: float, u_lat: float, cs: float):
    """LBM channel driven by a body force giving ``u_* = u_lat / 20``.

    The friction velocity follows from the *exact* discrete momentum balance
    ``g H = u_*^2`` of the periodic driven channel, not from a fitted profile.
    """
    ny = h_cells + 2
    nu_lat = u_lat * h_cells / re_h
    u_star_lat = u_lat / 20.0
    g_lat = u_star_lat**2 / h_cells
    s = D2Q9Channel(nx, ny, u_lat=u_lat, nu_lat=nu_lat, g_lat=g_lat, cs_smag=cs, top="slip")
    # Seed the mean profile with the log law instead of rest: the viscous spin-up
    # time H^2/nu is ~10^6 steps, so a resting start would never reach the
    # turbulent state inside a verification run.
    y = (np.arange(1, h_cells + 1) + 0.5) - 1.0
    u0 = np.maximum(
        u_star_lat * (np.log(np.maximum(y * u_star_lat / nu_lat, 1.01)) / KAPPA + 5.5), 0.0
    )
    rng = np.random.default_rng(12345)
    ux = np.zeros((nx, ny))
    ux[:, 1 : h_cells + 1] = u0[None] * (1.0 + 0.02 * rng.standard_normal((nx, h_cells)))
    s.ux = ux
    s.f = equilibrium(np.ones((nx, ny)), s.ux, s.uy)
    # mixing-length SGS: nu_t = (kappa y)^2 |S|, capped at the outer scale
    lm = KAPPA * np.minimum(y, 0.2 * h_cells)
    s.mixing_length = np.zeros((nx, ny))
    s.mixing_length[:, 1 : h_cells + 1] = lm[None]
    s.mixing_length[:, 0] = lm[0]
    s.mixing_length[:, -1] = lm[-1]
    return s, nu_lat, u_star_lat


def _theta0(cf: float, sed: Sediment, U: float) -> float:
    tau0 = 0.5 * cf * sed.rho_f * U * U
    return tau0 / ((sed.rho_s - sed.rho_f) * G * sed.d)


# ---------------------------------------------------------------------------
# Case 1 -- Shields
# ---------------------------------------------------------------------------
def shields_case(
    *,
    d: float = 2.0e-4,
    h: float = 0.3,
    re_h: float = 6.0e3,
    nx: int = 16,
    h_cells: int = 64,
    steps: int = 60000,
    u_lat: float = 0.05,
    cs_smag: float = 0.12,
    theta_reference: float = 0.045,
) -> dict:
    """Shields threshold for uniform sand: LBM hydraulics + critical Shields number."""
    sed = Sediment(d=d)
    d_star = sed.d_star
    theta_cr = float(shields_critical_soulsby(d_star))

    s, nu_lat, _ = _turbulent_channel(nx, h_cells, re_h=re_h, u_lat=u_lat, cs=cs_smag)
    s.step(steps)
    _, ux, _ = s.macros()
    y_lat = (np.arange(1, h_cells + 1) + 0.5) - 1.0  # height above the bed plane
    prof = ux[:, 1 : h_cells + 1].mean(axis=0)
    u_star_momentum = float(np.sqrt(s.g_lat * h_cells))
    # inner-layer fit band in wall units: below y+ = 100 the buffer layer is not
    # resolved at ~5 wall units per cell, so it is excluded and reported.
    y_plus = y_lat * u_star_momentum / nu_lat
    u_star_fit, z0_fit, fit_rms = log_law_fit(
        y_lat, prof, y_min=float(y_lat[y_plus >= 100].min()) if (y_plus >= 100).any() else 1.0,
        y_max=0.6 * h_cells,
    )
    u_star_wallfn = float(np.sqrt(s.bed_shear().mean()))
    u_mean = float(prof.mean())
    r_ustar_over_U = u_star_momentum / u_mean

    u_star_cr = float(np.sqrt(theta_cr * (sed.s - 1.0) * G * d))
    tau_cr = sed.critical_shear_stress(theta_cr)
    U_crit = u_star_cr / r_ustar_over_U

    d_sweep = np.logspace(np.log10(6e-5), np.log10(2e-2), 60)
    theta_sweep = shields_critical_soulsby(np.array([Sediment(d=float(dd)).d_star for dd in d_sweep]))
    ds_data, th_data = shields_curve_digitised()
    theta_at_data = shields_critical_soulsby(ds_data)
    curve_dev = float(np.max(np.abs(theta_at_data / th_data - 1.0)))

    return {
        "case": "shields",
        "parameters": {
            "d_m": d,
            "h_m": h,
            "rho_s": sed.rho_s,
            "rho_f": sed.rho_f,
            "nu": sed.nu,
            "d_star": d_star,
            "re_h": re_h,
            "grid_cells_across_depth": h_cells,
        },
        "lbm": {
            "u_star_over_U": r_ustar_over_U,
            "u_star_lattice_momentum": u_star_momentum,
            "u_star_lattice_logfit": float(u_star_fit),
            "u_star_lattice_wallfunction": u_star_wallfn,
            "y_plus_first_cell": float(y_plus[0]),
            "logfit_z0_cells": float(z0_fit),
            "logfit_rms_over_U": float(fit_rms / max(u_mean, 1e-30)),
            "u_mean_lattice": u_mean,
            "momentum_vs_logfit_rel": float(abs(u_star_momentum - u_star_fit) / u_star_momentum),
            "momentum_vs_wallfn_rel": float(abs(u_star_momentum - u_star_wallfn) / u_star_momentum),
        },
        "theta_cr_model": theta_cr,
        "theta_cr_classical": theta_reference,
        "theta_cr_deviation": abs(theta_cr - theta_reference) / theta_reference,
        "u_star_critical_m_s": u_star_cr,
        "tau_critical_pa": float(tau_cr),
        "U_depth_mean_critical_m_s": float(U_crit),
        "shields_curve": {
            "d_m": d_sweep.tolist(),
            "d_star": [float(Sediment(d=float(dd)).d_star) for dd in d_sweep],
            "theta_cr": theta_sweep.tolist(),
            "digitised_d_star": ds_data.tolist(),
            "digitised_theta_cr": th_data.tolist(),
            "max_rel_deviation_vs_digitised": curve_dev,
        },
        "profile": {
            "y_over_h": (y_lat / h_cells).tolist(),
            "u_over_Umean": (prof / u_mean).tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Case 2 -- Rouse profile
# ---------------------------------------------------------------------------
def rouse_case(
    *,
    u_star: float = 0.05,
    h: float = 0.3,
    d: float = 2.0e-4,
    z_a_over_h: float = 0.05,
    re_h: float = 6.0e3,
    h_cells: int = 64,
    nx: int = 16,
    steps: int = 40000,
    u_lat: float = 0.05,
    cs_smag: float = 0.12,
    n_particles: int = 50_000,
    lpt_steps: int = 12_000,
    lpt_dtau: float = 2.0e-3,
    seed: int = 20261109,
) -> dict:
    """Rouse (1937) equilibrium concentration profile against LBM hydraulics."""
    sed = Sediment(d=d)
    w_s = sed.settling_velocity()
    z_rouse = w_s / (KAPPA * u_star)
    z_a = z_a_over_h * h

    s, nu_lat, _ = _turbulent_channel(nx, h_cells, re_h=re_h, u_lat=u_lat, cs=cs_smag)
    s.step(steps)
    _, ux, _ = s.macros()
    y_lat = (np.arange(1, h_cells + 1) + 0.5) - 1.0
    prof = ux[:, 1 : h_cells + 1].mean(axis=0)
    u_star_momentum = float(np.sqrt(s.g_lat * h_cells))
    y_plus = y_lat * u_star_momentum / nu_lat
    u_star_fit, z0_fit, fit_rms = log_law_fit(
        y_lat, prof,
        y_min=float(y_lat[y_plus >= 100].min()) if (y_plus >= 100).any() else 1.0,
        y_max=0.6 * h_cells,
    )
    u_mean_lat = float(prof.mean())
    r = u_star_momentum / u_mean_lat  # u_*/U of the simulated channel
    U_phys = u_star / r
    z0_phys = h * float(z0_fit) / h_cells

    z = np.linspace(z_a, 0.98 * h, 200)
    c_ana = rouse_profile(z, h, z_a, 1.0, z_rouse)

    c_fv, z_fv = _advection_diffusion(h, z_a, z_rouse, n=400)

    hist, edges = _rouse_random_walk(
        z_rouse, n_particles=n_particles, n_steps=lpt_steps, dtau=lpt_dtau, seed=seed
    )
    zc_zeta = 0.5 * (edges[1:] + edges[:-1])
    zc = zc_zeta * h  # bins are in zeta = z/h, convert to absolute height
    density = hist / np.maximum(np.diff(edges), 1e-12) * 1.0

    # Fix the arbitrary density normalisation using the *statistics* only:
    # scale so that the LPT profile matches the analytic one at the reference
    # height (the analytic profile is normalised to 1 there as well).
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.interp(z_a, zc, density)
        c_lpt = density / scale if scale > 0 else density * np.nan
    c_ana_bins = rouse_profile(np.maximum(zc, 1e-9), h, z_a, 1.0, z_rouse)

    win = (zc >= 0.04 * h) & (zc <= 0.72 * h) & (c_ana_bins > 0.01) & (hist > 100)
    if win.any():
        dev_lpt = float(np.max(np.abs(c_lpt[win] / c_ana_bins[win] - 1.0)))
        log_rms_lpt = float(
            np.sqrt(np.mean((np.log(c_lpt[win]) - np.log(c_ana_bins[win])) ** 2))
        )
        c_fv_at = np.interp(zc[win], z_fv, c_fv)
        dev_fv = float(np.max(np.abs(c_fv_at / c_ana_bins[win] - 1.0)))
        log_rms_fv = float(
            np.sqrt(np.mean((np.log(c_fv_at) - np.log(c_ana_bins[win])) ** 2))
        )
    else:  # pragma: no cover - defensive
        dev_lpt = log_rms_lpt = dev_fv = log_rms_fv = float("nan")

    return {
        "case": "rouse",
        "parameters": {
            "u_star_m_s": u_star,
            "h_m": h,
            "d_m": d,
            "z_a_over_h": z_a_over_h,
            "w_settling_m_s": w_s,
            "rouse_number": z_rouse,
            "beta_diffusivity": 1.0,
            "re_h": re_h,
            "grid_cells_across_depth": h_cells,
            "n_particles": n_particles,
        },
        "lbm": {
            "u_star_over_U": r,
            "u_mean_m_s": U_phys,
            "u_star_logfit_lattice": float(u_star_fit),
            "u_star_momentum_lattice": u_star_momentum,
            "logfit_rms_over_U": float(fit_rms / u_mean_lat),
            "z0_m": z0_phys,
        },
        "profile": {
            "z_over_h": zc_zeta.tolist(),
            "C_analytic": c_ana_bins.tolist(),
            "C_lpt": c_lpt.tolist(),
            "C_fv": np.interp(zc, z_fv, c_fv).tolist(),
            "counts": hist.astype(int).tolist(),
            "comparison_window": win.astype(bool).tolist(),
        },
        "deviation_lpt": dev_lpt,
        "deviation_fv": dev_fv,
        "log_rms_lpt": log_rms_lpt,
        "log_rms_fv": log_rms_fv,
        "deviation": dev_lpt,
    }


def _advection_diffusion(h: float, z_a: float, z_rouse: float, n: int = 400):
    """Steady ``dC/dz = -Z C / (z (1 - z/h))`` by second-order finite volumes.

    Cell-centred, ``C(z_a) = 1``.  The solution must equal the closed-form Rouse
    profile to discretisation error; this validates the suspended-load integral
    used by the scour case.
    """
    edges = np.linspace(z_a, h, n + 1)
    zc = 0.5 * (edges[1:] + edges[:-1])
    dz = np.diff(edges)
    g = z_rouse / (zc * (1.0 - zc / h))  # = -d ln C / dz
    c = np.empty(n)
    c[0] = 1.0
    for i in range(n - 1):
        c[i + 1] = c[i] * np.exp(-0.5 * (g[i] + g[i + 1]) * dz[i])
    return c, zc


def _rouse_random_walk(
    z_rouse: float,
    *,
    n_particles: int,
    n_steps: int,
    dtau: float,
    seed: int,
    n_bins: int = 40,
    sample_every: int = 20,
):
    """Lagrangian grain tracking for the Rouse equilibrium profile.

    Dimensionless height ``zeta = z/h``, time ``tau = t kappa u_* / h``.  The
    grains follow the drift-diffusion process::

        dzeta = -Z dtau + sqrt(2 Z zeta (1 - zeta) dtau) xi

    discretised with the exponential-fitted (Scharfetter-Gummel) jump chain on a
    uniform grid: between cells ``i`` and ``i+1`` the upward jump rate is
    ``p_i^+ = eps_{i+1/2}/dz^2 * Pe/(e^Pe - 1)`` with ``Pe = Z dz / eps`` and the
    downward rate follows from detailed balance ``p_{i+1}^- = p_i^+ e^{Pe}``.
    That discretisation is *exact* for piecewise-constant coefficients, so the
    stationary distribution of the chain is the Rouse profile rather than an
    O(dtau) approximation of it -- which matters because the profile spans three
    decades and a naive Euler-Maruyama step biases it by ~15-60 % (measured, see
    the node report).

    ``dtau`` is only used to cap the per-step jump probability (``<= 1/3``); the
    effective time step is ``1/(3 max(p^+ + p^-))``.
    """
    rng = np.random.default_rng(seed)
    dz = 1.0 / n_bins
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    eps_half = edges[1:-1] * (1.0 - edges[1:-1])  # dimensionless eps, i+1/2
    eps_half = np.maximum(eps_half, 1e-12)
    pe = z_rouse * dz / eps_half
    pup = eps_half / dz**2 * pe / np.expm1(np.clip(pe, 1e-12, 700))
    pdn = np.zeros(n_bins)
    pdn[1:] = pup * np.exp(np.clip(pe, -700, 700))
    p_up = np.zeros(n_bins)
    p_up[:-1] = pup
    p_dn = pdn
    rate_max = float((p_up + p_dn).max())
    dt = min(dtau, 0.33 / max(rate_max, 1e-30))
    n_steps = max(int(n_steps), int(np.ceil(2.5 / dt)))

    cell = rng.integers(0, n_bins, n_particles)
    hist = np.zeros(n_bins)
    n_sample = 0
    for step in range(n_steps):
        u = rng.random(n_particles)
        up = u < p_up[cell] * dt
        dn = (u >= p_up[cell] * dt) & (u < (p_up[cell] + p_dn[cell]) * dt)
        cell = cell + up - dn
        np.clip(cell, 0, n_bins - 1, out=cell)
        if step > n_steps // 2 and step % sample_every == 0:
            hist += np.bincount(cell, minlength=n_bins)
            n_sample += 1
    hist /= max(n_sample, 1)
    return hist, edges


# ---------------------------------------------------------------------------
# Case 3 -- single pile scour
# ---------------------------------------------------------------------------
def _exner_2d(
    z_b: np.ndarray,
    q: np.ndarray,
    dx: float,
    porosity: float,
    erodible: np.ndarray,
    *,
    dz_max: float,
) -> tuple[np.ndarray, float, float]:
    """Conservative 2-D Exner update ``dz_b/dt = -(1/(1-lambda)) div q``.

    Face fluxes are arithmetic means of the adjacent cell-centre fluxes, so the
    update telescopes: ``sum(dz_b) * dx^2 == -dt/(1-lambda) * (net boundary
    flux)`` to round-off.  The time step is chosen so that the largest
    increment is ``dz_max`` metres; the limiter is a single positive scalar, so
    it preserves the telescoping property (a per-cell clip would not).
    ``q`` is ``(nx, ny, 2)`` in m^2/s.
    """
    qx = 0.5 * (q[:, :, 0] + np.roll(q[:, :, 0], -1, axis=0))
    qy = 0.5 * (q[:, :, 1] + np.roll(q[:, :, 1], -1, axis=1))
    div = (qx - np.roll(qx, 1, axis=0)) / dx + (qy - np.roll(qy, 1, axis=1)) / dx
    scale = float(np.abs(div[erodible]).max()) if erodible.any() else 0.0
    if scale <= 0.0:
        return z_b.copy(), 0.0, 0.0
    dt = dz_max * (1.0 - porosity) / scale
    dz = -div * dt / (1.0 - porosity)
    dz = np.where(erodible, dz, 0.0)
    return z_b + dz, dt, float(div.sum())


def _avalanche_2d(z_b: np.ndarray, dx: float, repose_deg: float, erodible: np.ndarray, iters: int = 6):
    """Mass-conserving 2-D angle-of-repose limiter (x-sweeps then y-sweeps)."""
    z = z_b.copy()
    tan_phi = np.tan(np.deg2rad(repose_deg))
    for _ in range(iters):
        if not erodible.any():
            break
        z = avalanche_relax(z, dx, repose_deg, iters=1)  # x direction
        zt = z.T.copy()
        et = erodible.T
        # y direction: operate column-wise through the same 1-D operator
        for i in range(zt.shape[0]):
            col = zt[i]
            if not et[i].any():
                continue
            col_new = avalanche_relax(col, dx, repose_deg, iters=1)
            # keep non-erodible columns untouched
            col_new = np.where(et[i], col_new, col)
            zt[i] = col_new
        z = zt.T
        slope = np.maximum(
            np.abs(np.diff(z, axis=1)).max(initial=0.0),
            np.abs(np.diff(z.T, axis=1)).max(initial=0.0),
        ) / dx
        if slope <= tan_phi * 1.05:
            break
    return z


def pile_scour_case(
    *,
    D: float = 0.1,
    h: float = 0.3,
    U: float = 0.5,
    d: float = 2.0e-4,
    cells_per_D: int = 12,
    length_D: float = 16.0,
    width_D: float = 8.0,
    u_lat: float = 0.05,
    nu_lat: float = 2.0e-5,
    cs_smag: float = 0.17,
    n_relax: int = 6000,
    n_cycle: int = 30,
    n_per_cycle: int = 400,
    depth_feedback: bool = True,
    dz_max_cells: float = 0.35,
    max_scour_D: float = 3.0,
) -> dict:
    """Plan-view depth-averaged LBM + total-load Exner scour around a pile.

    Model form and its documented limitations:

    * depth-averaged (2-D plan view) -- the horseshoe vortex and the downflow
      that set the scour depth in 3-D are not resolved, only their footprint on
      the depth-averaged shear field;
    * bottom friction from the log law, ``tau_b = cf/2 rho |U| U``;
    * total load = Meyer-Peter-Mueller bed load + van Rijn reference
      concentration x Rouse suspended load; both vanish at ``theta_cr``, so the
      transport stops where the bed shear falls below threshold;
    * Exner update, finite volume, telescoping (conservative to round-off), with
      an angle-of-repose limiter;
    * ``depth_feedback`` switches on the continuity closure ``u_eff = U h0/h``,
      the mechanism by which a depth-averaged model can terminate a scour hole.
      ``depth_feedback=False`` (no feedback, hole grows until the transport
      threshold is met) is reported as the other bound.
    """
    sed = Sediment(d=d)
    dx = D / cells_per_D
    dt = u_lat * dx / U
    h_lat = h / dx
    cf = 2.0 * (KAPPA / np.log(11.0 * h / max(2.5 * d, 1e-9))) ** 2
    nx = int(round(length_D * D / dx))
    ny = int(round(width_D * D / dx))
    if ny % 2:
        ny += 1
    n_pile = max(int(round(D / dx)), 3)

    solid = np.zeros((nx, ny), dtype=bool)
    solid[:, 0] = solid[:, -1] = True  # lateral flume walls, no-slip
    xc, yc = nx / 3.0, (ny - 1) / 2.0
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    solid |= ((ii - xc) ** 2 + (jj - yc) ** 2) <= (0.5 * n_pile) ** 2
    erodible = ~solid

    drag = cf / (2.0 * h_lat)
    g_lat = drag * u_lat**2
    s = D2Q9Channel(nx, ny, u_lat=u_lat, nu_lat=nu_lat, g_lat=g_lat, cs_smag=cs_smag, top="wall")
    s.set_solid(solid)
    s.drag_coeff = drag
    # Seed the approach flow at the equilibrium depth-averaged velocity: the
    # spin-up time from rest is g^-1 * U = 2e5 steps, far beyond a verification
    # budget, while the interesting transient (wake / scour) is O(D/U) = 1e2.
    ux0 = np.zeros((nx, ny))
    ux0[erodible] = u_lat
    s.ux = ux0
    s.f = equilibrium(np.ones((nx, ny)), s.ux, s.uy)
    s.step(n_relax)

    theta_cr = float(shields_critical_soulsby(sed.d_star))
    a_ref = max(2.0 * d, 0.02 * h)
    z_b = np.zeros((nx, ny))
    zs_hist: list[float] = []
    cycle_log = []
    mass_err = 0.0
    u_bar = s.macros()[1].copy() * (dx / dt)
    for cyc in range(n_cycle):
        acc = np.zeros((nx, ny))
        n_acc = 0
        for k in range(0, n_per_cycle, 4):
            s.step(4)
            acc += s.macros()[1]  # lattice velocity of the current state
            n_acc += 1
        u_bar_lat = acc / max(n_acc, 1)
        u_bar = u_bar_lat * (dx / dt)  # lattice -> physical depth-averaged velocity
        h_loc = h - z_b
        if depth_feedback:
            factor = np.where(erodible, h / np.maximum(h_loc, 0.25 * h), 1.0)
        else:
            factor = 1.0
        u_eff = u_bar * factor
        tau = 0.5 * cf * sed.rho_f * np.abs(u_eff) * u_eff
        theta = np.abs(tau) / ((sed.rho_s - sed.rho_f) * G * d)
        u_star = np.sqrt(np.abs(tau) / sed.rho_f)
        qs_mag = total_load_transport_vec(
            sed,
            theta=theta,
            theta_cr=theta_cr,
            u_star=u_star,
            h=np.maximum(h_loc, 0.05 * h),
            a_ref=a_ref,
        )
        qs_mag = np.where(erodible, qs_mag, 0.0)
        q = np.stack([qs_mag * np.sign(u_eff), np.zeros_like(qs_mag)], axis=-1)
        prev = z_b.copy()
        z_exner, dt_m, div_sum = _exner_2d(
            z_b, q, dx, sed.porosity, erodible, dz_max=dz_max_cells * (D / cells_per_D)
        )
        # discrete Exner conservation: the telescoping sum of the update must
        # equal -dt/(1-lambda) * (net face flux through the domain boundary)
        dv_ex = float((z_exner - prev)[erodible].sum()) * dx * dx
        gross_ex = float(np.abs(z_exner - prev)[erodible].sum()) * dx * dx
        pred = -div_sum * dx * dx * dt_m / (1.0 - sed.porosity)
        mass_err = max(mass_err, abs(dv_ex - pred) / max(abs(pred), gross_ex, 1e-30))
        z_b_new = _avalanche_2d(z_exner, dx, sed.repose_deg, erodible)
        z_b_new = np.where(erodible, z_b_new, 0.0)
        z_b = np.maximum(z_b_new, -max_scour_D * D)
        zs = float(-z_b.min())
        zs_hist.append(zs)
        cycle_log.append({"cycle": cyc, "z_s_m": zs, "z_s_over_D": zs / D, "dt_morpho_s": dt_m})
        if len(zs_hist) > 8 and abs(zs - zs_hist[-5]) < 2e-3 * D:
            break

    zs = float(-z_b.min())
    n_tail = max(len(zs_hist) // 3, 2)
    growth = (zs_hist[-1] - zs_hist[-n_tail]) / (n_tail * D)
    return {
        "case": "pile_scour",
        "parameters": {
            "D_m": D,
            "h_m": h,
            "U_m_s": U,
            "d_m": d,
            "h_over_D": h / D,
            "friction_coeff_cf": float(cf),
            "theta_0": float(_theta0(cf, sed, U)),
            "theta_cr": theta_cr,
            "re_D": float(U * D / sed.nu),
            "re_D_numerical": float(u_lat * n_pile / nu_lat),
            "cells_per_D": cells_per_D,
            "dx_m": dx,
            "dt_s": dt,
            "grid": [nx, ny],
            "depth_feedback": bool(depth_feedback),
        },
        "z_s_m": zs,
        "z_s_over_D": zs / D,
        "converged": bool(growth < 1e-3),
        "growth_per_cycle": float(growth),
        "history": zs_hist,
        "cycles": cycle_log,
        "mass_balance_rel_error": float(mass_err),
        "bed": z_b.tolist(),
        "u_mean_field": u_bar.tolist(),
        "tau_pa": (0.5 * cf * sed.rho_f * np.abs(u_bar) * u_bar).tolist(),
        "pile_mask": solid.tolist(),
        "max_scour_cap_hit": bool(zs >= max_scour_D * D - 1e-6),
    }