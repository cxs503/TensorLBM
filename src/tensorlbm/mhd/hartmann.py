"""Hartmann (low-Rm MHD) channel flow: closed form + D2Q9 solver with Lorentz force.

Physical problem
----------------
A plane channel spans ``y in [-H, H]`` (2D, periodic in ``x``, no-slip walls).
A uniform magnetic field ``B0`` points along ``y`` and a uniform electric field
``E_z = -K B0`` points along the out-of-plane ``z`` axis.  With a low magnetic
Reynolds number the induced field is neglected, the current density follows
Ohm's law ``j_z = sigma (E_z + u_x B0) = sigma B0 (u_x - K)`` and the Lorentz
force per unit volume is

    F_x = (j x B)_x = -j_z B0 = sigma B0 (K - u_x).

``K`` is the loading parameter (``K > u`` means the device works as a pump).
Adding an optional axial load gradient ``dp/dx = load_grad`` (an adverse
pressure gradient - the pump head) and a plain body force ``drive`` (used for
the ``B0 = 0`` Poiseuille negative control) the steady momentum balance is

    mu u'' = -F_x + load_grad - drive
           = -sigma B0 (K - u) + load_grad - drive.

With ``U0 = K - (load_grad - drive) / (sigma B0)`` and the Hartmann number
``Ha = B0 H sqrt(sigma / mu)`` the exact solution is

    u(y) / U0 = 1 - cosh(Ha y/H) / cosh(Ha),
    u_mean   = U0 * (1 - tanh(Ha)/Ha).

``Ha -> 0`` degenerates exactly onto plane Poiseuille flow with the effective
body force ``sigma B0 K - load_grad + drive`` and mean/centre ratio ``2/3``.

Lattice units: ``H`` is the channel half-width in lattice spacings, ``nu`` the
kinematic viscosity (``mu = rho0 nu``), ``sigma`` the electric conductivity and
``B0`` the field, both in lattice units.  Consistently with a low-Rm model the
Lorentz force is a body force: it enters the LBE through the second-order Guo
forcing scheme, so the discrete momentum that the force injects is exactly the
prescribed value even at ``O(u F)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# --- D2Q9 lattice -----------------------------------------------------------
C_D2Q9: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
)
W_D2Q9: tuple[float, ...] = (
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
)
OPP_D2Q9: tuple[int, ...] = (0, 3, 4, 1, 2, 7, 8, 5, 6)
#: directions pointing into the bottom wall (wall lies below node row 0)
_BOTTOM_HIT: tuple[int, ...] = (4, 7, 8)
#: directions pointing into the top wall
_TOP_HIT: tuple[int, ...] = (2, 5, 6)


# ---------------------------------------------------------------------------
# Closed-form solutions
# ---------------------------------------------------------------------------
def hartmann_mean_factor(ha: float) -> float:
    """Return ``u_mean / U0 = 1 - tanh(Ha)/Ha`` (Taylor series for small ``Ha``).

    This is the *mean of the shape function*: ``u(y) = U0 A(y/H)`` with
    ``A = 1 - cosh(Ha y/H)/cosh(Ha)``, hence ``u_mean = U0 m(Ha)``.  ``m`` is
    ``O(Ha^2)`` for small ``Ha`` (the profile collapses as ``U0`` grows like
    ``1/Ha^2`` in the pressure-driven normalisation).
    """
    ha = float(ha)
    if ha < 0.0:
        raise ValueError(f"Hartmann number must be >= 0, got {ha}")
    if ha < 0.1:
        h2 = ha * ha
        # 1 - tanh(x)/x = x^2/3 - 2x^4/15 + 17x^6/315 - ...  (truncation O(x^8))
        return h2 * (1.0 / 3.0 - 2.0 * h2 / 15.0 + 17.0 * h2 * h2 / 315.0)
    return 1.0 - math.tanh(ha) / ha


def hartmann_mean_velocity_coefficient(ha: float) -> float:
    """Return the campaign normalisation ``ubar* = (1/Ha^2)(1 - tanh(Ha)/Ha)``.

    This is the mean velocity of the classical Hartmann solution written as
    ``u(y) = (G H^2/mu) u*(y/H)``, i.e. ``m(Ha)/Ha^2``.  ``Ha -> 0`` gives
    ``1/3`` (plane Poiseuille) and ``Ha -> inf`` gives ``Ha^-2``; the large-``Ha``
    asymptotic test therefore acts on this coefficient.
    """
    ha = float(ha)
    if ha < 0.0:
        raise ValueError(f"Hartmann number must be >= 0, got {ha}")
    if ha < 1e-4:
        return 1.0 / 3.0
    return hartmann_mean_factor(ha) / (ha * ha)


def hartmann_shape_factor(y_hat: Any, ha: float) -> np.ndarray:
    """Return ``1 - cosh(Ha y/H) / cosh(Ha)`` for ``y_hat = y/H in [-1, 1]``.

    Uses a Taylor expansion ``[Ha^2 (1-y^2)/2 + Ha^4 (1-y^4)/24 + ...]/cosh(Ha)``
    for ``Ha < 0.1`` so that the ``Ha -> 0`` limit does not lose digits to
    cancellation in ``1 - cosh(Ha y/H)``.
    """
    y = np.asarray(y_hat, dtype=float)
    ha = float(ha)
    if ha < 0.0:
        raise ValueError(f"Hartmann number must be >= 0, got {ha}")
    if ha < 0.1:
        h2 = ha * ha
        num = (
            h2 * (1.0 - y * y) / 2.0
            + h2 * h2 * (1.0 - y**4) / 24.0
            + h2 * h2 * h2 * (1.0 - y**6) / 720.0
        )
        return num / math.cosh(ha)
    return 1.0 - np.cosh(ha * y) / math.cosh(ha)


def hartmann_profile(
    y_hat: Any,
    ha: float,
    u0: float = 1.0,
) -> np.ndarray:
    """Dimensional Hartmann velocity profile ``u = U0 (1 - cosh(Ha y/H)/cosh(Ha))``."""
    return u0 * hartmann_shape_factor(y_hat, ha)


def hartmann_u0_normalised_profile(y_hat: Any, ha: float) -> np.ndarray:
    """Return ``A(y/H, Ha)/Ha^2 = (1/Ha^2)(1 - cosh(Ha y/H)/cosh(Ha))``.

    This is the campaign normalisation ``u*`` (``Ha -> 0`` gives ``(1-y^2)/2``).
    A dedicated series branch keeps the ``Ha -> 0`` limit free of ``0/0``.
    """
    y = np.asarray(y_hat, dtype=float)
    ha = float(ha)
    if ha < 0.0:
        raise ValueError(f"Hartmann number must be >= 0, got {ha}")
    if ha < 0.1:
        h2 = ha * ha
        num = (1.0 - y * y) / 2.0 + h2 * (1.0 - y**4) / 24.0 + h2 * h2 * (1.0 - y**6) / 720.0
        return num / math.cosh(ha)
    return hartmann_shape_factor(y, ha) / (ha * ha)


def hartmann_normalised_profile(y_hat: Any, ha: float) -> np.ndarray:
    """Profile normalised by its own mean: ``u(y)/u_mean`` (equals ``1.5(1-y^2)`` at ``Ha=0``)."""
    y = np.asarray(y_hat, dtype=float)
    ha = float(ha)
    if ha < 0.1:
        # ratio of the two series expansions: no 0/0 at Ha = 0
        h2 = ha * ha
        num = (1.0 - y * y) / 2.0 + h2 * (1.0 - y**4) / 24.0 + h2 * h2 * (1.0 - y**6) / 720.0
        den = (1.0 / 3.0 - 2.0 * h2 / 15.0 + 17.0 * h2 * h2 / 315.0) * math.cosh(ha)
        return num / den
    return hartmann_shape_factor(y, ha) / hartmann_mean_factor(ha)


def poiseuille_profile(y_hat: Any, u_mean: float = 1.0) -> np.ndarray:
    """Plane Poiseuille profile normalised to the given mean velocity."""
    y = np.asarray(y_hat, dtype=float)
    return 1.5 * u_mean * (1.0 - y * y)


def poiseuille_centre_ratio() -> float:
    """``u_mean / u_centre = 2/3`` for plane Poiseuille flow."""
    return 2.0 / 3.0


def hartmann_wall_shear_coefficient(ha: float) -> float:
    """Dimensionless wall shear ``tau_w H / (mu u_mean) = Ha tanh(Ha) / m(Ha)``.

    The ``Ha -> 0`` limit is the Poiseuille value ``3`` (``tau_w = 3 mu u_mean/H``).
    """
    ha = float(ha)
    if ha <= 0.0:
        return 3.0
    return ha * math.tanh(ha) / hartmann_mean_factor(ha)


def ha_from_params(
    b0: float,
    half_width: float,
    sigma: float,
    nu: float,
    rho0: float = 1.0,
) -> float:
    """Return ``Ha = B0 H sqrt(sigma/mu)`` with ``mu = rho0 nu``."""
    if nu <= 0.0:
        raise ValueError(f"nu must be > 0, got {nu}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    return float(b0) * float(half_width) * math.sqrt(float(sigma) / (rho0 * float(nu)))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HartmannConfig:
    """Configuration of the D2Q9 Hartmann/Poiseuille channel.

    Args:
        ny: number of fluid rows across the channel (``H = ny/2`` lattice spacings).
        nx: number of periodic streamwise cells (``4`` is enough: the flow is
            streamwise uniform, more cells only add cost).
        ha: Hartmann number ``B0 H sqrt(sigma/mu)``. ``ha = 0`` switches the
            Lorentz coupling off (pure body-force Poiseuille negative control).
        u_mean: target mean velocity; it fixes the applied loading ``K``.
        nu: kinematic viscosity in lattice units (``tau = 1/2 + 3 nu``).
        rho0: reference density.
        b0: magnetic flux density in lattice units.
        load_grad: adverse axial pressure gradient ``dp/dx`` (pump load), force
            density in lattice units.
        drive: plain body force density (negative-control driver). ``None`` (or 0)
            with ``ha = 0`` derives it from ``u_mean``: ``F = 3 mu u_mean / H^2``.
        steps: hard cap on the number of lattice steps.
        check_every: steady-state check interval.
        tol: convergence tolerance on the change of the mean velocity per check.
        tol: steady-state tolerance on the relative change of the whole profile.
        dtype: floating point dtype ("float64" is required for verification runs).
        u_init: optional initial streamwise velocity profile on the node centres
            (grid-sequenced start-up from a coarser level; ``None`` = start at rest).
    """

    ny: int = 64
    nx: int = 4
    ha: float = 5.0
    u_mean: float = 0.02
    nu: float = 0.05
    rho0: float = 1.0
    b0: float = 1.0
    load_grad: float = 0.0
    drive: float | None = None
    steps: int = 60000
    check_every: int = 500
    tol: float = 1e-11
    dtype: str = "float64"
    u_init: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.ny < 4 or self.ny % 2:
            raise ValueError(f"ny must be even and >= 4, got {self.ny}")
        if self.nx < 1:
            raise ValueError(f"nx must be >= 1, got {self.nx}")
        if self.ha < 0.0:
            raise ValueError(f"ha must be >= 0, got {self.ha}")
        if self.nu <= 0.0:
            raise ValueError(f"nu must be > 0, got {self.nu}")
        if self.u_mean <= 0.0:
            raise ValueError(f"u_mean must be > 0, got {self.u_mean}")
        if self.u_mean > 0.1:
            raise ValueError(
                f"u_mean={self.u_mean} exceeds the low-Mach verification window (<= 0.1)"
            )

    @property
    def half_width(self) -> float:
        """Channel half-width ``H = ny/2`` in lattice spacings."""
        return 0.5 * self.ny

    @property
    def mu(self) -> float:
        """Dynamic viscosity ``rho0 nu``."""
        return self.rho0 * self.nu

    @property
    def tau(self) -> float:
        """BGK relaxation time for the configured viscosity."""
        return 0.5 + 3.0 * self.nu

    @property
    def sigma(self) -> float:
        """Electric conductivity that realises ``ha`` at the configured ``B0`` and ``H``."""
        h = self.half_width
        if self.b0 == 0.0:
            return 0.0
        return self.mu * self.ha * self.ha / (self.b0 * self.b0 * h * h)

    @property
    def body_force(self) -> float:
        """Plain body force density actually applied.

        For the ``Ha = 0`` negative control the force is derived from the target
        mean velocity of plane Poiseuille flow (``u_mean = F H^2 / (3 mu)``), so
        the same physical case is run on every grid.
        """
        if self.drive is not None and self.drive > 0.0:
            return float(self.drive)
        if self.ha == 0.0:
            return 3.0 * self.mu * self.u_mean / (self.half_width**2)
        return 0.0

    @property
    def sigma_b0(self) -> float:
        """Product ``sigma B0`` (current density per unit velocity mismatch)."""
        return self.sigma * self.b0

    @property
    def magnetic_force_coefficient(self) -> float:
        """Force density per unit velocity mismatch, ``sigma B0^2 = mu Ha^2 / H^2``.

        The Lorentz force is ``F_x = (j x B)_x = -j_z B0 = sigma B0^2 (K - u)``;
        the two factors of ``B0`` are physically distinct (Ohm's law and the
        cross product), so the coefficient must not be collapsed into ``sigma B0``.
        """
        return self.sigma * self.b0 * self.b0

    @property
    def u0_amplitude(self) -> float:
        """Exact profile amplitude ``U0 = K - (load_grad - drive)/(sigma B0^2)``."""
        if self.magnetic_force_coefficient <= 0.0:
            # Poiseuille limit: u_mean = 2/3 u_max
            return 1.5 * self.u_mean
        return self.u_mean / hartmann_mean_factor(self.ha)

    @property
    def loading(self) -> float:
        """Applied electric loading ``K = -E_z/B0`` that yields ``u_mean``."""
        if self.magnetic_force_coefficient <= 0.0:
            return 0.0
        return self.u0_amplitude + (self.load_grad - self.body_force) / (
            self.magnetic_force_coefficient
        )

    def y_hat(self) -> np.ndarray:
        """Node-centre coordinates in ``[-1, 1]`` (walls at ``y = +-H``)."""
        j = np.arange(self.ny, dtype=float)
        return (j + 0.5 - self.half_width) / self.half_width

    def validate(self) -> None:
        """Raise if the numerical set-up leaves the verification envelope."""
        if self.tau <= 0.5:
            raise ValueError(f"tau={self.tau} must exceed 0.5")
        if not 0.0 <= self.ha < 1e4:
            raise ValueError(f"ha={self.ha} out of range")
        if self.ha == 0.0 and self.body_force <= 0.0:
            raise ValueError("ha=0 requires a positive body force for the Poiseuille control")


@dataclass
class HartmannResult:
    """Outcome of one Hartmann/Poiseuille channel run."""

    config: HartmannConfig
    y_hat: np.ndarray
    u_profile: np.ndarray
    u_profile_analytic: np.ndarray
    u_mean: float
    u_mean_analytic: float
    u_centre: float
    tau_wall: float
    tau_wall_analytic: float
    lorentz_force: float
    drive_force: float
    wall_momentum_flux: float
    closure_residual: float
    steps: int
    converged: bool
    rel_linf: float
    rel_l2: float
    rel_max_pointwise: float
    rel_local_core: float
    max_mach: float
    u_field: "np.ndarray | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def wall_shear_coefficient(self) -> float:
        """Measured ``tau_w H / (mu u_mean)``."""
        return self.tau_wall * self.config.half_width / (self.config.mu * self.u_mean)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (profiles excluded: use :meth:`profile_rows`)."""
        return {
            "config": {
                "ny": self.config.ny,
                "nx": self.config.nx,
                "ha": self.config.ha,
                "u_mean_target": self.config.u_mean,
                "nu": self.config.nu,
                "half_width": self.config.half_width,
                "sigma": self.config.sigma,
                "b0": self.config.b0,
                "loading_K": self.config.loading,
                "load_grad": self.config.load_grad,
                "drive": self.config.body_force,
                "tau": self.config.tau,
            },
            "u_mean": self.u_mean,
            "u_mean_analytic": self.u_mean_analytic,
            "u_centre": self.u_centre,
            "tau_wall": self.tau_wall,
            "tau_wall_analytic": self.tau_wall_analytic,
            "wall_shear_coefficient": self.wall_shear_coefficient,
            "lorentz_force": self.lorentz_force,
            "drive_force": self.drive_force,
            "wall_momentum_flux": self.wall_momentum_flux,
            "closure_residual": self.closure_residual,
            "rel_linf": self.rel_linf,
            "rel_l2": self.rel_l2,
            "rel_max_pointwise": self.rel_max_pointwise,
            "rel_local_core": self.rel_local_core,
            "max_mach": self.max_mach,
            "steps": self.steps,
            "converged": self.converged,
            "metadata": self.metadata,
        }

    def profile_rows(self) -> list[dict[str, float]]:
        """Per-row profile table (node coordinate, LBM velocity, closed form, error)."""
        rows: list[dict[str, float]] = []
        for y, un, ua in zip(self.y_hat, self.u_profile, self.u_profile_analytic):
            rows.append(
                {
                    "y_over_H": float(y),
                    "u_lbm": float(un),
                    "u_analytic": float(ua),
                    "abs_error": float(abs(un - ua)),
                    "rel_error": float(abs(un - ua) / abs(ua)) if ua != 0.0 else 0.0,
                }
            )
        return rows


_INDEX_CACHE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _stream_indices(ny: int, nx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cached gather indices for one-shot (periodic) streaming of a (9, ny, nx) field."""
    key = (ny, nx)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    c = np.asarray(C_D2Q9, dtype=np.int64)
    q_idx = torch.arange(9, dtype=torch.int64).view(9, 1, 1).expand(9, ny, nx)
    y_idx = (
        torch.arange(ny, dtype=torch.int64).view(1, ny) - torch.as_tensor(c[:, 1]).view(9, 1)
    ) % ny
    x_idx = (
        torch.arange(nx, dtype=torch.int64).view(1, nx) - torch.as_tensor(c[:, 0]).view(9, 1)
    ) % nx
    indices = (
        q_idx.contiguous(),
        y_idx.unsqueeze(2).expand(9, ny, nx).contiguous(),
        x_idx.unsqueeze(1).expand(9, ny, nx).contiguous(),
    )
    _INDEX_CACHE[key] = indices
    return indices


def _stream(f_post: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Periodic streaming (single gather); the y wrap is overwritten by the walls."""
    ny, nx = f_post.shape[1], f_post.shape[2]
    q_idx, y_idx, x_idx = _stream_indices(ny, nx)
    return f_post[q_idx, y_idx, x_idx]


def run_hartmann_channel(config: HartmannConfig) -> HartmannResult:
    """Integrate the D2Q9 duct flow to steady state and compare with the closed form.

    The Lorentz force ``sigma B0 (K - u_x) - load_grad + drive`` is applied with
    the second-order Guo scheme (the ``1 - 1/(2 tau)`` prefactor is included).
    The stationary solution is compared pointwise with
    ``U0 (1 - cosh(Ha y/H)/cosh Ha)`` evaluated on the same node centres, and the
    total Lorentz force is checked against the momentum balance
    ``F_lorentz = load_grad * A + wall momentum flux + drive * A``.
    """
    config.validate()
    torch_dtype = getattr(torch, config.dtype)
    device = torch.device("cpu")
    c, w, opp = _tables(device, torch_dtype)

    ny, nx = config.ny, config.nx
    cx = c[:, 0].reshape(9, 1, 1).to(torch_dtype)
    cy = c[:, 1].reshape(9, 1, 1).to(torch_dtype)
    wv = w.reshape(9, 1, 1)
    tau = config.tau
    force_coeff = config.magnetic_force_coefficient
    loading = config.loading
    load_grad = config.load_grad
    drive = config.body_force

    rho = torch.ones((ny, nx), dtype=torch_dtype, device=device)
    ux = torch.zeros((ny, nx), dtype=torch_dtype, device=device)
    uy = torch.zeros((ny, nx), dtype=torch_dtype, device=device)
    if config.u_init is not None:
        if len(config.u_init) != ny:
            raise ValueError(f"u_init must have ny={ny} entries, got {len(config.u_init)}")
        ux = (
            torch.as_tensor(config.u_init, dtype=torch_dtype, device=device)
            .view(ny, 1)
            .expand(ny, nx)
            .contiguous()
        )
    f = _equilibrium(rho, ux, uy, cx, cy, wv)

    u_prev: torch.Tensor | None = None
    converged = False
    step = 0
    for step in range(1, config.steps + 1):
        f_post, ux, uy, rho = _collide(
            f, config, cx, cy, wv, tau, force_coeff, loading, load_grad, drive
        )
        f = _stream(f_post, c)
        # half-way bounce-back walls at y = +-H (row 0 and row ny-1)
        for i in _BOTTOM_HIT:
            f[opp[i], 0, :] = f_post[i, 0, :]
        for i in _TOP_HIT:
            f[opp[i], ny - 1, :] = f_post[i, ny - 1, :]
        if step % config.check_every == 0:
            u_now = ux.mean(dim=1)
            if u_prev is not None:
                change = float((u_now - u_prev).abs().max()) / max(float(u_now.abs().max()), 1e-30)
                if change <= config.tol:
                    converged = True
                    u_prev = u_now
                    break
            u_prev = u_now

    f_post, ux, uy, rho = _collide(f, config, cx, cy, wv, tau, force_coeff, loading, load_grad, drive)
    u_row = ux.mean(dim=1).cpu().numpy()
    y_hat = config.y_hat()
    u_mean = float(ux.mean())
    u0 = config.u0_amplitude
    u_an = hartmann_profile(y_hat, config.ha, u0)
    u_mean_an = u0 * hartmann_mean_factor(config.ha)
    tau_an = config.mu * u0 * config.ha * math.tanh(config.ha) / config.half_width
    if config.ha == 0.0:
        # Poiseuille: u_max = 3/2 u_mean, tau_w = 3 mu u_mean / H
        u_mean_an = config.u_mean
        u_an = poiseuille_profile(y_hat, u_mean_an)
        tau_an = 3.0 * config.mu * u_mean_an / config.half_width

    err = u_row - u_an
    scale = max(abs(u_mean_an), 1e-30)
    core = np.abs(u_an) >= 0.1 * np.abs(u_mean_an)
    core_half = np.abs(u_an) >= 0.5 * np.abs(u_mean_an)
    with np.errstate(divide="ignore", invalid="ignore"):
        pointwise = np.where(core, np.abs(err) / np.maximum(np.abs(u_an), 1e-300), 0.0)
        eval_local_core = np.where(core_half, np.abs(err) / np.maximum(np.abs(u_an), 1e-300), 0.0)

    # Momentum bookkeeping (per unit span, per lattice time^2 normalisation dropped)
    lorentz = float(force_coeff * (loading - ux).mean()) * ny * nx
    drive_force = float(drive) * ny * nx
    wall_flux = _wall_momentum_flux(f_post, config)
    denom = max(abs(lorentz) + abs(drive_force), 1e-30)
    closure = abs(lorentz + drive_force - wall_flux) / denom

    # Wall shear from the momentum exchange at the top wall (same value as bottom
    # wall by symmetry) converted to a physical stress: F / area with area = Lx*1.
    tau_wall = 0.5 * abs(wall_flux) / (nx * 1.0) / 1.0

    return HartmannResult(
        config=config,
        y_hat=y_hat,
        u_profile=u_row,
        u_profile_analytic=u_an,
        u_mean=u_mean,
        u_mean_analytic=u_mean_an,
        u_centre=float(u_row[ny // 2 - 1 : ny // 2 + 1].mean()),
        tau_wall=tau_wall,
        tau_wall_analytic=tau_an,
        lorentz_force=lorentz,
        drive_force=drive_force,
        wall_momentum_flux=wall_flux,
        closure_residual=closure,
        steps=step,
        converged=converged,
        rel_linf=float(np.max(np.abs(err)) / scale),
        rel_l2=float(np.sqrt(np.mean(err * err)) / scale),
        rel_max_pointwise=float(np.max(pointwise)),
        rel_local_core=float(np.max(eval_local_core)),
        max_mach=float(torch.sqrt(ux * ux + uy * uy).max()),
        u_field=ux.cpu().numpy().copy(),
        metadata={
            "device": "cpu",
            "dtype": config.dtype,
            "scheme": "BGK + Guo forcing (2nd order), half-way bounce-back walls",
            "tau": tau,
            "sigma_b0_sq_times_H2_over_mu": (
                config.magnetic_force_coefficient * config.half_width**2 / config.mu
            ),
        },
    )


def _tables(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = torch.tensor(C_D2Q9, dtype=torch.int64, device=device)
    w = torch.tensor(W_D2Q9, dtype=dtype, device=device)
    opp = torch.tensor(OPP_D2Q9, dtype=torch.int64, device=device)
    return c, w, opp


def _equilibrium(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    usq = (ux * ux + uy * uy).unsqueeze(0)
    return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def _collide(
    f: torch.Tensor,
    config: HartmannConfig,
    cx: torch.Tensor,
    cy: torch.Tensor,
    w: torch.Tensor,
    tau: float,
    force_coeff: float,
    loading: float,
    load_grad: float,
    drive: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """BGK collision with the Guo-forced Lorentz body force. Returns (f_post, ux, uy, rho)."""
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-30)
    ux_star = (f * cx).sum(dim=0) / rho_safe
    uy_star = (f * cy).sum(dim=0) / rho_safe
    # Lorentz force from Ohm's law with the half-step corrected velocity:
    # u = u* + F/(2 rho) solved implicitly for the x component.
    force_const = torch.as_tensor(drive - load_grad, dtype=f.dtype, device=f.device)
    if force_coeff > 0.0:
        a = force_coeff / (2.0 * rho)
        fx = (force_coeff * (loading - ux_star) + force_const) / (1.0 + a)
    else:
        fx = force_const.expand_as(ux_star).clone()
    ux = ux_star + 0.5 * fx / rho_safe
    uy = uy_star
    feq = _equilibrium(rho, ux, uy, cx, cy, w)
    # Guo source term S_i = (1 - 1/(2tau)) w_i [3 (c_i - u).F + 9 (c_i.u)(c_i.F)]
    cuf = cx * fx.unsqueeze(0)
    cidotu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    src = (
        (1.0 - 0.5 / tau)
        * w
        * (3.0 * (cuf - ux.unsqueeze(0) * fx.unsqueeze(0)) + 9.0 * cidotu * cuf)
    )
    f_post = f - (f - feq) / tau + src
    return f_post, ux, uy, rho


def _wall_momentum_flux(f_post: torch.Tensor, config: HartmannConfig) -> float:
    """Momentum exchanged with the two walls per step (Ladd momentum-exchange).

    Negative sign convention: the value returned is the x-momentum the *fluid
    loses* to the walls.  For the aligned half-way bounce-back walls used here
    this is exact up to the boundary-layer closure of the scheme.
    """
    ny = config.ny
    total = 0.0
    for i in _BOTTOM_HIT:
        total += 2.0 * C_D2Q9[i][0] * float(f_post[i, 0, :].mean())
    for i in _TOP_HIT:
        total += 2.0 * C_D2Q9[i][0] * float(f_post[i, ny - 1, :].mean())
    return total * config.nx
