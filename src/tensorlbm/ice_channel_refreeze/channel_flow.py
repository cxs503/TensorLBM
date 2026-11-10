"""Two-dimensional wind-driven circulation and heat transport in a channel.

What this module does
---------------------
D2Q9 lattice Boltzmann for momentum and D2Q5 advection-diffusion for a passive
scalar, on the same grid, in the **similarity (dimensionless) regime**: the
lattice Reynolds number is kept small enough for the BGK operator to be stable
(``u_lid = 0.04``, ``nu_lat = 0.05`` -> ``Re_lat ~ 10``), so the solver resolves
the *shape* of the lid-driven circulation and of the scalar field, not the
physical Reynolds number.  The physical magnitude of the transport is carried by
the bulk closures in :mod:`surface` (drag law for the drift, Stanton number for
the water-side heat flux); the report states this limitation explicitly.

What is verified here
---------------------
* momentum: the steady mid-channel profile of the moving-lid Couette flow
  ``u(y) = U_lid (H - y)/H`` (:func:`validate_flow_solver`);
* scalar: the D2Q5 advection-diffusion operator against the analytic solution of
  a decaying sinusoidal profile (:func:`validate_scalar_solver`);
* the surface flux *shape factor* ``S = (theta[2] - theta[1]) / (h/H)`` used to
  modulate the bulk ocean heat flux across the channel width.

No result in this module is used as a quantitative ocean heat flux without the
bulk closure -- this is deliberate and is the main documented limitation of the
node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import DEFAULT_PROPERTIES, IceWaterProperties
from .surface import drag_law_current, ocean_heat_flux, wind_stress

C9 = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=int,
)
W9 = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
C5 = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=int)
W5 = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])


def _shift(field: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.roll(np.roll(field, dy, axis=0), dx, axis=1)


def _equilibrium9(rho, ux, uy):
    cu = 3.0 * (C9[:, 0, None, None] * ux + C9[:, 1, None, None] * uy)
    usq = ux * ux + uy * uy
    return W9[:, None, None] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)


def _equilibrium5(temp, ux, uy):
    cu = 3.0 * (C5[:, 0, None, None] * ux + C5[:, 1, None, None] * uy)
    return W5[:, None, None] * temp * (1.0 + cu)


@dataclass
class ChannelFlowResult:
    W: float
    H: float
    u_wind: float
    u_lid_phys: float
    nu_lat: float
    kappa_lat: float
    nx: int
    ny: int
    n_steps: int
    ux_lat: np.ndarray
    uy_lat: np.ndarray
    theta: np.ndarray          # dimensionless superheat, 1 = bulk, 0 = interface
    q_shape: np.ndarray        # d(theta)/d(y/H) at the interface, per column
    q_shape_mean: float
    q_shape_history: np.ndarray
    surface_drift_lat: float

    def summary(self) -> dict:
        return {
            "W_m": self.W,
            "H_m": self.H,
            "u_wind_m_s": self.u_wind,
            "u_lid_phys_m_s": self.u_lid_phys,
            "Re_lat": float(self.u_lid_phys and self.surface_drift_lat / self.nu_lat),
            "nu_lat": self.nu_lat,
            "kappa_lat": self.kappa_lat,
            "nx": self.nx,
            "ny": self.ny,
            "n_steps": int(self.n_steps),
            "surface_drift_lat": float(self.surface_drift_lat),
            "q_shape_mean": float(self.q_shape_mean),
            "q_shape_max": float(np.max(self.q_shape)),
            "q_shape_min": float(np.min(self.q_shape)),
            "mode": "similarity (lattice Re ~ 10); bulk closure carries the magnitude",
        }


def simulate_channel(
    W: float = 100.0,
    H: float = 20.0,
    u_wind: float = 5.0,
    nx: int = 64,
    n_steps: int = 12000,
    record_every: int = 500,
    u_lid_lat: float = 0.04,
    nu_lat: float = 0.05,
    kappa_lat: float = 0.02,
    buoyancy: bool = False,
    lateral_periodic: bool = False,
    props: IceWaterProperties = DEFAULT_PROPERTIES,
) -> ChannelFlowResult:
    """Lid-driven cavity in similarity form with a scalar (superheat) field.

    Row 0 is the moving lid (ice/air interface, lattice speed ``u_lid_lat``),
    row ``ny-1`` the no-slip bed, the lateral columns are no-slip ice walls.
    The scalar is 1 in the bulk and 0 on the first fluid row (the ice
    interface), so ``theta`` is the dimensionless superheat.
    """
    nx = max(int(nx), 8)
    h_phys = float(W) / nx
    ny = max(10, int(round(float(H) / h_phys)))
    H_phys = ny * h_phys
    tau_f = 0.5 + 3.0 * nu_lat
    tau_t = 0.5 + 3.0 * kappa_lat

    fluid = np.ones((ny, nx), dtype=bool)
    fluid[0, :] = False
    fluid[-1, :] = False
    if not lateral_periodic:
        fluid[:, 0] = False
        fluid[:, -1] = False

    rho = np.ones((ny, nx))
    ux = np.zeros((ny, nx))
    ux[0, :] = u_lid_lat
    uy = np.zeros((ny, nx))
    f = _equilibrium9(rho, ux, uy).copy()
    theta = np.ones((ny, nx))
    theta[0, :] = 0.0
    theta[1, :] = 0.0
    g = _equilibrium5(theta, ux, uy).copy()

    u_lid = u_lid_lat
    hist = []
    for step_i in range(int(n_steps)):
        rho = f.sum(axis=0)
        ux = (f * C9[:, 0, None, None]).sum(axis=0) / rho
        uy = (f * C9[:, 1, None, None]).sum(axis=0) / rho
        if buoyancy:
            fy = -0.01 * (theta - 0.5)
            fy[0, :] = 0.0
            fy[-1, :] = 0.0
            uy = uy + 0.5 * fy
        else:
            fy = None
        feq = _equilibrium9(rho, ux, uy)
        f = np.where(fluid[None, :, :], f - (f - feq) / tau_f, f)
        if fy is not None:
            f += (1.0 - 0.5 / tau_f) * 3.0 * W9[:, None, None] * (
                C9[:, 1, None, None] * fy
            )
        for k in range(9):
            f[k] = _shift(f[k], int(C9[k, 0]), int(C9[k, 1]))
        # moving lid: reflect only the populations that stream back into the
        # fluid on the next step (top: cy=+1, bottom: cy=-1)
        f[2, 0, :] = f[4, 0, :]
        f[5, 0, :] = f[7, 0, :] + u_lid / 6.0
        f[6, 0, :] = f[8, 0, :] - u_lid / 6.0
        f[4, -1, :] = f[2, -1, :]
        f[7, -1, :] = f[5, -1, :]
        f[8, -1, :] = f[6, -1, :]
        if not lateral_periodic:
            f[1, 1:-1, 0] = f[3, 1:-1, 0]
            f[5, 1:-1, 0] = f[7, 1:-1, 0]
            f[8, 1:-1, 0] = f[6, 1:-1, 0]
            f[3, 1:-1, -1] = f[1, 1:-1, -1]
            f[7, 1:-1, -1] = f[5, 1:-1, -1]
            f[6, 1:-1, -1] = f[8, 1:-1, -1]

        rho = f.sum(axis=0)
        ux = (f * C9[:, 0, None, None]).sum(axis=0) / rho
        uy = (f * C9[:, 1, None, None]).sum(axis=0) / rho
        theta = g.sum(axis=0)
        geq = _equilibrium5(theta, ux, uy)
        g -= (g - geq) / tau_t
        for k in range(5):
            g[k] = _shift(g[k], int(C5[k, 0]), int(C5[k, 1]))
        theta = g.sum(axis=0)
        theta[0, :] = 0.0
        theta[1, :] = 0.0
        theta[-1, :] = 1.0  # mixed-layer reservoir below the resolved layer
        if not lateral_periodic:
            theta[:, 0] = theta[:, 1]
            theta[:, -1] = theta[:, -2]
        for k in range(5):
            g[k, 0, :] = 0.0
            g[k, 1, :] = 0.0
            g[k, -1, :] = W5[k] * theta[-1, :]
            g[k, :, 0] = W5[k] * theta[:, 0]
            g[k, :, -1] = W5[k] * theta[:, -1]
        if (step_i + 1) % max(int(record_every), 1) == 0:
            q_shape = (theta[2, :] - theta[1, :]) * (ny - 1)
            hist.append(float(np.mean(q_shape[1:-1])))

    rho = f.sum(axis=0)
    ux = (f * C9[:, 0, None, None]).sum(axis=0) / rho
    uy = (f * C9[:, 1, None, None]).sum(axis=0) / rho
    q_shape = (theta[2, :] - theta[1, :]) * (ny - 1)
    tau_a = wind_stress(u_wind, props)
    return ChannelFlowResult(
        W=float(W),
        H=float(H_phys),
        u_wind=float(u_wind),
        u_lid_phys=drag_law_current(tau_a, props),
        nu_lat=float(nu_lat),
        kappa_lat=float(kappa_lat),
        nx=nx,
        ny=ny,
        n_steps=int(n_steps),
        ux_lat=ux,
        uy_lat=uy,
        theta=theta,
        q_shape=q_shape,
        q_shape_mean=float(np.mean(q_shape[1:-1])),
        q_shape_history=np.asarray(hist),
        surface_drift_lat=float(np.mean(ux[1, 1:-1])),
    )


def physical_ocean_flux(
    result: ChannelFlowResult,
    superheat: float = 0.05,
    kappa_t: float = 1.0e-3,
    props: IceWaterProperties = DEFAULT_PROPERTIES,
) -> np.ndarray:
    """Map the resolved shape factor onto a bulk-consistent flux, W/m^2.

    ``q_o(x) = rho cp kappa_t (T_w - T_f) dtheta/dy`` with ``dtheta/dy`` taken
    from the lattice field divided by the physical cell height.  The bulk drift
    is the drag-law value carried in ``result.u_lid_phys``.
    """
    dy = result.H / (result.ny - 1)
    grad = (result.theta[2, :] - result.theta[1, :]) / dy
    return props.rho_w * props.cp_w * kappa_t * float(superheat) * grad


def validate_flow_solver(
    ny: int = 20, nx: int = 30, n_steps: int = 8000, u_lid_lat: float = 0.04,
    nu_lat: float = 0.05,
) -> dict:
    """LBM lid-driven Couette profile vs ``u(y) = U_lid (H - y)/H``."""
    res = simulate_channel(
        W=float(nx), H=float(ny - 1), nx=nx, n_steps=n_steps,
        u_lid_lat=u_lid_lat, nu_lat=nu_lat, kappa_lat=0.02,
        lateral_periodic=True,
    )
    u = res.ux_lat.mean(axis=1)
    y = np.arange(res.ny) / (res.ny - 1)
    analytic = u_lid_lat * (1.0 - y)
    sl = slice(2, res.ny - 2)
    err = np.abs(u[sl] - analytic[sl]) / max(abs(u_lid_lat), 1e-12)
    return {
        "u_lid_lat": float(u_lid_lat),
        "u_row1_lat": float(u[1]),
        "u_analytic_row1_lat": float(analytic[1]),
        "max_rel_error_prof": float(np.max(err)),
        "mean_rel_error_prof": float(np.mean(err)),
        "stable": bool(np.isfinite(u).all()),
    }


def validate_scalar_solver(
    ny: int = 40, nx: int = 8, n_steps: int = 400, kappa_lat: float = 0.02
) -> dict:
    """D2Q5 diffusion vs the analytic decay of a sinusoidal profile.

    A quiescent scalar field ``sin(pi (j+0.5)/ny)`` decays as
    ``exp(-kappa pi^2 t / H^2)``; the LBM is compared after ``n_steps``.
    """
    tau_t = 0.5 + 3.0 * kappa_lat
    j = np.arange(ny) + 0.5
    theta0 = np.sin(np.pi * j / ny)
    theta = np.tile(theta0[:, None], (1, nx))
    ux = np.zeros((ny, nx))
    uy = np.zeros((ny, nx))
    g = _equilibrium5(theta, ux, uy).copy()
    for _ in range(int(n_steps)):
        theta = g.sum(axis=0)
        geq = _equilibrium5(theta, ux, uy)
        g -= (g - geq) / tau_t
        for k in range(5):
            g[k] = _shift(g[k], int(C5[k, 0]), int(C5[k, 1]))
        theta = g.sum(axis=0)
        theta[0, :] = theta[1, :]
        theta[-1, :] = theta[-2, :]
        for k in range(5):
            g[k, 0, :] = W5[k] * theta[0, :]
            g[k, -1, :] = W5[k] * theta[-1, :]
            g[k, :, 0] = W5[k] * theta[:, 0]
            g[k, :, -1] = W5[k] * theta[:, -1]
    t = float(n_steps)
    analytic = theta0 * np.exp(-kappa_lat * np.pi ** 2 * t / ny ** 2)
    num = theta[:, nx // 2]
    err = np.linalg.norm(num - analytic) / np.linalg.norm(analytic)
    sl = slice(2, ny - 2)
    err_int = np.linalg.norm(num[sl] - analytic[sl]) / np.linalg.norm(analytic[sl])
    return {
        "n_steps": int(n_steps),
        "kappa_lat": float(kappa_lat),
        "l2_rel_error": float(err),
        "l2_rel_error_interior": float(err_int),
        "amplitude_numeric": float(num.max()),
        "amplitude_analytic": float(analytic.max()),
    }


__all__ = [
    "ChannelFlowResult",
    "simulate_channel",
    "physical_ocean_flux",
    "validate_flow_solver",
    "validate_scalar_solver",
    "ocean_heat_flux",
]