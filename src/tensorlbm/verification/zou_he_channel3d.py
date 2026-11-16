"""Zou/He-driven 3D plane channel: solver + analytic (plane Poiseuille) reference.

Anchor: ``XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION``.

This module exists to answer one question with a number: **does the D3Q19
inlet/outlet reconstruction in** :mod:`tensorlbm.boundary.zou_he_3d` **actually
produce the continuum channel flow when it is the only thing driving it?**

The setup is deliberately minimal so that the boundary closure is the sole
active ingredient:

* D3Q19, BGK, ``tau = 0.65``;
* two solid ``y`` rows with half-way bounce-back, so the effective width
  between wall mid-planes is ``H = ny - 2`` (the ``y = 0.5`` and
  ``y = ny - 1.5`` planes);
* periodic in ``z`` (``nz`` is a thin slab, no ``z`` dependence develops);
* **no body force**.  The flow is driven by
  ``zou_he_inlet_pressure_3d(rho_in)`` on ``x = 0`` plus
  ``zou_he_outlet_pressure_3d(rho_out)`` on ``x = nx - 1``, i.e. the
  pressure/pressure pair.  A purely velocity-driven arm
  (``inlet="velocity-profile"``) is also provided, which exercises the
  per-cell profile reconstruction with a parabolic inflow.

The analytic reference for the same driving is the plane-Poiseuille parabola

    u(y) = G y' (H - y') / (2 nu),   y' = y - 1/2,   G = -(dp/dx) = drho / (3 L)

with ``L = nx - 1`` the lattice distance between the two boundary planes and
``dp = drho / 3`` (``p = rho / 3``).  ``G`` is taken **a priori** from the
nominal driving; a least-squares ``G_fit`` from the measured profile is
reported separately as the "did the driving survive transport" diagnostic --
it is not used to normalise the error.

Because a soft pressure outlet only weakly damps acoustics, the steady state is
defined by block time averaging; the window-to-window drift of the peak
velocity is reported instead of hidden behind a pointwise tolerance.

Nothing in this module imports the campaign scripts: the tests call
:func:`simulate_channel_3d` directly and the ``xval`` case wraps it for the
report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import functools

import numpy as np
import torch

from tensorlbm.boundary.zou_he_3d import (
    zou_he_inlet_pressure_3d,
    zou_he_inlet_velocity_3d,
    zou_he_outlet_pressure_3d,
)
from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d, macroscopic3d

_C = C.cpu().numpy()
_OPP = OPPOSITE.cpu().numpy()


def pressure_gradient(drho: float, nx: int) -> float:
    """A-priori body-force-equivalent gradient ``G = drho / (3 (nx-1))``."""
    return drho / (3.0 * (nx - 1))


def plane_poiseuille_profile(ny: int, G: float, nu: float) -> np.ndarray:
    """Exact plane-Poiseuille velocity on the fluid rows ``1 .. ny-2``."""
    H = ny - 2
    yt = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    return G * yt * (H - yt) / (2.0 * nu)


_UP = [k for k in range(19) if int(_C[k, 1]) > 0]
_DOWN = [k for k in range(19) if int(_C[k, 1]) < 0]
_UP_OPP_GATHER = [int(_OPP[k]) for k in _UP]
_DOWN_OPP_GATHER = [int(_OPP[k]) for k in _DOWN]


@functools.lru_cache(maxsize=8)
def _stream_plan(nz: int, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat gather indices for the free streaming, plus the fluid-cell mask.

    ``out[k] = f_post[k][:, (z - c_z) % nz, j - c_y, i - c_x]`` wherever the
    source cell is a fluid cell; entries whose source is a solid wall row or
    lies outside the streamwise ends are masked to zero and then either fixed
    by the half-way bounce-back reflection or overwritten by the inlet/outlet
    closure.  Built once per geometry so the time loop has no index arithmetic.
    """
    n_cells = nz * ny * nx
    zz, jj, ii = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    idx = np.empty((19, nz, ny, nx), dtype=np.int64)
    valid = np.zeros((19, nz, ny, nx), dtype=np.float64)
    fluid_rows = (jj >= 1) & (jj <= ny - 2)
    for k in range(19):
        cx, cy, cz = int(_C[k, 0]), int(_C[k, 1]), int(_C[k, 2])
        z_src = (zz - cz) % nz
        j_src = jj - cy
        i_src = ii - cx
        ok = fluid_rows & (j_src >= 1) & (j_src <= ny - 2) & (i_src >= 0) & (i_src <= nx - 1)
        flat = (
            np.clip(z_src, 0, nz - 1) * (ny * nx)
            + np.clip(j_src, 0, ny - 1) * nx
            + np.clip(i_src, 0, nx - 1)
        )
        idx[k] = np.where(ok, flat, 0)
        valid[k] = ok
    return idx.reshape(19, n_cells), valid.reshape(19, n_cells)


def _stream_3d(f_post: torch.Tensor) -> torch.Tensor:
    """Stream with half-way bounce-back on the two ``y`` walls, periodic ``z``.

    ``x`` is *not* periodic: populations that would enter from outside the
    domain are zeroed, and the inlet/outlet planes are then overwritten by the
    boundary closure, so a forgotten boundary update cannot masquerade as an
    inflow (it produces ``rho = 0`` cells, which the run metrics reveal).
    """
    nz, ny, nx = f_post.shape[1], f_post.shape[2], f_post.shape[3]
    idx, valid = _stream_plan(nz, ny, nx)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=f_post.device)
    valid_t = torch.as_tensor(valid, dtype=f_post.dtype, device=f_post.device)
    flat = f_post.reshape(19, nz * ny * nx)
    streamed = torch.gather(flat, 1, idx_t) * valid_t
    out = streamed.reshape(19, nz, ny, nx)
    # half-way bounce-back: the two fluid rows adjacent to a solid row reflect
    # their *own* opposite population at the wall mid-plane.
    out[_UP, :, 1, :] = f_post[_UP_OPP_GATHER, :, 1, :]
    out[_DOWN, :, ny - 2, :] = f_post[_DOWN_OPP_GATHER, :, ny - 2, :]
    return out


@dataclass
class Channel3DResult:
    """Outcome of one channel run (all velocities in lattice units)."""

    nx: int
    ny: int
    nz: int
    nu: float
    tau: float
    drho: float
    steps: int
    inlet: str
    H: float
    u_profile: np.ndarray  # streamwise/z-averaged u_x on the fluid rows
    u_exact: np.ndarray
    u_profile_inlet_plane: np.ndarray
    metrics: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "nu": self.nu,
            "tau": self.tau,
            "drho": self.drho,
            "steps": self.steps,
            "inlet": self.inlet,
            "H": self.H,
            "metrics": self.metrics,
            # Raw field data, so a reviewer can recompute every error in the
            # metrics without re-running the solver (a metrics-only artifact
            # cannot be audited).
            "u_profile": [float(v) for v in self.u_profile],
            "u_exact": [float(v) for v in self.u_exact],
            "u_profile_inlet_plane": [float(v) for v in self.u_profile_inlet_plane],
            "trace": {k: [float(v) for v in vals] for k, vals in self.trace.items()},
        }


def simulate_channel_3d(
    *,
    nx: int = 64,
    ny: int = 18,
    nz: int = 2,
    drho: float = 0.004,
    tau: float = 0.65,
    steps: int = 12000,
    inlet: str = "pressure",
    block: int = 500,
    u_peak_inlet: float | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> Channel3DResult:
    """Run a Zou/He-driven D3Q19 plane channel to a block-averaged steady state.

    Args:
        nx: streamwise cells (inlet plane ``0``, outlet plane ``nx-1``).
        ny: wall-normal cells; the two outer rows are solid.
        nz: spanwise cells (periodic).
        drho: ``rho_in - rho_out``; sets the a-priori gradient.
        tau: BGK relaxation time; ``nu = (tau - 1/2) / 3``.
        steps: total time steps.
        inlet: ``"pressure"`` (pressure/pressure pair) or
            ``"velocity-profile"`` (parabolic velocity inlet +
            pressure outlet, exercising the per-cell reconstruction).
        block: block length for time averaging and drift reporting.
        u_peak_inlet: peak velocity of the prescribed inflow parabola for
            ``inlet="velocity-profile"`` (defaults to the a-priori
            ``G H^2 / (8 nu)``).
        dtype: torch dtype (``float64`` keeps the moment checks meaningful).
        device: torch device.

    Returns:
        :class:`Channel3DResult` with the mean profile, the analytic profile
        and the error metrics.
    """
    device = torch.device("cpu") if device is None else device
    previous_threads = torch.get_num_threads()
    if ny < 4:
        raise ValueError("ny must be >= 4 (two solid rows plus fluid)")
    if not (0.5 < tau < 5.0):
        raise ValueError("tau outside the stable range for BGK")

    nu = (tau - 0.5) / 3.0
    H = float(ny - 2)
    G = pressure_gradient(drho, nx)
    u_exact = plane_poiseuille_profile(ny, G, nu)
    u_peak = float(u_exact.max()) if u_peak_inlet is None else float(u_peak_inlet)

    rho_in = 1.0 + drho / 2.0
    rho_out = 1.0 - drho / 2.0

    rho0 = torch.ones((nz, ny, nx), dtype=dtype, device=device)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      torch.zeros_like(rho0), device=device)

    profile_field = torch.zeros((nz, ny), dtype=dtype, device=device)
    if inlet == "velocity-profile":
        prof = torch.as_tensor(plane_poiseuille_profile(ny, G, nu), dtype=dtype, device=device)
        profile_field[:, 1:-1] = prof.unsqueeze(0)
    elif inlet != "pressure":
        raise ValueError(f"unknown inlet mode {inlet!r}")

    # These grids are tiny (O(10^3) cells): intra-op threading costs more than
    # it saves, and the default pool can be 100+ threads on the cluster nodes.
    if device.type == "cpu":
        torch.set_num_threads(min(2, previous_threads))
    trace = {"step": [], "u_peak": [], "ld2": []}
    blocks: list[np.ndarray] = []
    peak_trace: list[float] = []

    u_mean_acc = np.zeros(ny - 2, dtype=np.float64)
    n_acc = 0
    for step in range(steps):
        rho, ux, uy, uz = macroscopic3d(f, device=device)
        feq = equilibrium3d(rho, ux, uy, uz, device=device)
        f_post = f - (f - feq) / tau
        f = _stream_3d(f_post)

        if inlet == "pressure":
            f = zou_he_inlet_pressure_3d(f, rho_in)
        else:
            f = zou_he_inlet_velocity_3d(f, profile_field)
        f = zou_he_outlet_pressure_3d(f, rho_out)

        if step >= steps - 2 * block:
            _, ux_s, _, _ = macroscopic3d(f, device=device)
            u_mean_acc += ux_s[:, 1:-1, :].mean(dim=(0, 2)).cpu().numpy()
            n_acc += 1
            peak_trace.append(float(ux_s[:, 1:-1, :].mean(dim=(0, 2)).abs().max()))
        if step % max(1, steps // 12) == 0:
            _, ux_s, _, _ = macroscopic3d(f, device=device)
            mid = slice(nx // 4, 3 * nx // 4)
            prof_now = ux_s[:, 1:-1, mid].mean(dim=(0, 2)).cpu().numpy()
            trace["step"].append(step)
            trace["u_peak"].append(float(prof_now.max()))
            trace["ld2"].append(float(np.linalg.norm(prof_now - u_exact) / np.linalg.norm(u_exact)))

    if device.type == "cpu":
        torch.set_num_threads(previous_threads)
    u_profile = u_mean_acc / max(1, n_acc)

    # drift between the two half-blocks of the averaging window
    half = len(peak_trace) // 2
    drift = (
        abs(peak_trace[-1] - peak_trace[half]) / max(abs(peak_trace[-1]), 1e-30)
        if half > 0 and peak_trace[-1] != 0.0
        else float("inf")
    )

    _, ux_f, _, _ = macroscopic3d(f, device=device)
    u_profile_inlet = ux_f[:, 1:-1, 0].mean(dim=0).cpu().numpy()
    # streamwise mean over the middle half (away from both closures)
    mid = slice(nx // 4, 3 * nx // 4)
    u_streamwise = ux_f[:, 1:-1, mid].mean(dim=(0, 2)).cpu().numpy()

    rho_m, ux_m, _, _ = macroscopic3d(f, device=device)
    # Diagnostics are restricted to fluid rows: the closures also write the
    # solid cells of their plane, where the reconstruction has no physical
    # meaning (a pressure face implies u_x = 1 there, since both column sums
    # vanish).  Including those cells would fake a huge end-to-end flux.
    flux = (rho_m * ux_m)[:, 1:-1, :].sum(dim=(0, 1))  # (nx,) mass flux through y-z
    interior = flux[nx // 4: 3 * nx // 4]
    flux_defect = float(abs(flux[int(0)] - flux[nx // 4]) / max(abs(interior.mean()), 1e-30))
    flux_uniformity = float((interior.max() - interior.min()) / max(abs(interior.mean()), 1e-30))

    u_max_sim = float(u_streamwise.max())
    u_max_exact = float(G * H * H / (8.0 * nu))
    phi = plane_poiseuille_profile(ny, 1.0, nu)  # unit-gradient shape
    G_fit = float(np.dot(phi, u_streamwise) / np.dot(phi, phi))

    metrics = {
        "nu": nu,
        "G_nominal": G,
        "G_fit": G_fit,
        "G_fit_rel_err": abs(G_fit - G) / G,
        "u_max_sim": u_max_sim,
        "u_max_exact": u_max_exact,
        "amp_rel_err": abs(u_max_sim - u_max_exact) / u_max_exact,
        "linf_rel_err": float(np.abs(u_streamwise - u_exact).max() / u_max_exact),
        "shape_l2_rel": float(
            np.linalg.norm(u_streamwise / u_max_sim - u_exact / u_max_exact)
            / np.linalg.norm(u_exact / u_max_exact)
        ),
        "drift_rel": float(drift),
        "converged": bool(drift < 1e-3),
        "flux_end_to_interior_defect": flux_defect,
        "flux_interior_uniformity": flux_uniformity,
        "inlet_plane_profile_rel_err": float(
            np.abs(u_profile_inlet - u_exact).max() / u_max_exact
        ),
        "rho_min": float(rho_m[:, 1:-1, :].min()),
        "rho_max": float(rho_m[:, 1:-1, :].max()),
    }
    trace["u_exact"] = u_exact.tolist()
    return Channel3DResult(
        nx=nx,
        ny=ny,
        nz=nz,
        nu=nu,
        tau=tau,
        drho=drho,
        steps=steps,
        inlet=inlet,
        H=H,
        u_profile=u_streamwise,
        u_exact=u_exact,
        u_profile_inlet_plane=u_profile_inlet,
        metrics=metrics,
        trace=trace,
    )


def run_verification(
    *,
    widths: tuple[int, ...] = (8, 16, 24),
    nx: int = 64,
    steps: int = 12000,
    drho: float = 0.004,
    tau: float = 0.65,
    include_velocity_arm: bool = True,
    **kwargs,
) -> dict:
    """Resolution ladder of the pressure/pressure pair (+ one profile inlet arm).

    Returns:
        ``{"arms": [result.as_dict(), ...], "runs": [Channel3DResult, ...]}``.
    """
    runs: list[Channel3DResult] = []
    for ny in widths:
        runs.append(
            simulate_channel_3d(nx=nx, ny=ny + 2, drho=drho, tau=tau, steps=steps, **kwargs)
        )
    if include_velocity_arm:
        runs.append(
            simulate_channel_3d(
                nx=nx,
                ny=max(widths) + 2,
                drho=drho,
                tau=tau,
                steps=steps,
                inlet="velocity-profile",
                **kwargs,
            )
        )
    return {"arms": [r.as_dict() for r in runs], "runs": runs}