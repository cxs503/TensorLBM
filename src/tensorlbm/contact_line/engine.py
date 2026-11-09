"""Wall-bounded Shan-Chen single-component engine for moving-contact-line work.

Why this module exists
----------------------
``tensorlbm.multiphase`` implements the pseudopotential (Shan-Chen, SC) force and
collision kernels, but every existing benchmark drives them in a *periodic* box
(``benchmarks/verified/laplace_droplet``) or behind a bounce-back mask without a
wetting wall parameter.  A contact line needs three ingredients that were not
packaged anywhere in the library:

1. a **no-slip wall** that reflects the populations (half-way bounce-back) while
   the SC force sum replaces the pseudopotential inside the solid with a tunable
   wall value ``psi_w`` (this is what selects the static contact angle);
2. a **reproducible initial state** for the two geometries of interest —
   a sessile droplet on the wall and a meniscus spanning a plate channel;
3. a **body-force drive** whose magnitude sets the capillary number
   ``Ca = mu U / sigma`` of the (steady) sliding state.

The stepper here is deliberately thin: it composes the *existing* library
kernels (:func:`tensorlbm.multiphase.collide_sc_single_component`,
:func:`tensorlbm.solver.stream`) instead of re-implementing the physics, so any
improvement made to the SC kernels propagates to the contact-line studies
automatically.

Conventions (locked by the library, see ``multiphase`` module docstring)
-----------------------------------------------------------------------
* ``G > 0`` is attractive (backward-gather sign convention).  ``G = 5.0`` with
  ``psi_exp`` and ``tau = 1`` gives the discrete coexistence analysed in
  ``sc_eos_coexistence`` / ``benchmarks/verified/laplace_droplet``:
  ``rho_l = 1.957``, ``rho_v = 0.1596`` (density ratio ~12.3) and
  ``sigma = 0.0561`` lattice units.
* The wall value ``psi_w`` is expressed here as a **fraction of the bulk liquid
  pseudopotential** ``psi(\\rho_l)``: ``f_w = psi_w / psi(rho_l)``.  ``f_w = 0``
  is the historical "dry" wall (a fully non-wetting, density-depleted wall that
  is numerically fragile for a closed droplet), ``f_w -> 1`` drives the wall
  towards complete wetting.  ``f_w`` and the resulting static angle are related
  by a *measured* calibration (see
  :func:`tensorlbm.contact_line.capillary_driven.calibrate_static_angle`), never
  by an assumed formula.
* The bottom wall plane is the half-way bounce-back location
  ``y_wall = wall_rows - 0.5``; the top wall plane is
  ``y_top = ny - wall_rows - 0.5``.  All contact angles are referenced to
  ``y_wall``.

Discretisation of the drive
---------------------------
``gx`` is a uniform body-force acceleration applied to ``rho`` (i.e. genuine
gravity-like forcing, ``F = rho g``), exactly as in the gravity-driven sessile
drop contact-angle literature.  Because the channel is periodic in ``x`` the
drop slides, wraps around, and can be tracked indefinitely while the wall
bounce-back absorbs the injected streamwise momentum — a steady state exists.

References
----------
Shan & Chen (1993) Phys. Rev. E 47, 1815.
Benzi, Succi & Vergassola (1992) Phys. Rep. 222, 145 (SC review).
Bouzidi, Firdaouss & Lallemand (2001) Phys. Fluids 13, 3319 (half-way BB).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from ..d2q9 import OPPOSITE, equilibrium, macroscopic
from ..multiphase import collide_sc_single_component, psi_exp
from ..solver import stream

__all__ = [
    "Channel",
    "WallStepper",
    "default_coexistence",
    "meniscus_state",
    "sessile_drop_state",
    "tanh_profile",
]

#: Discrete liquid/vapour coexistence of the ``G = 5`` + ``psi_exp`` + ``tau = 1``
#: SCMP scheme (measured in ``benchmarks/verified/laplace_droplet``).
RHO_L_DEFAULT = 1.957
RHO_V_DEFAULT = 0.1596
#: Surface tension of the same scheme (lattice units), measured from the Laplace
#: law of static droplets (``benchmarks/verified/laplace_droplet/result.json``).
SIGMA_DEFAULT = 0.05611


def default_coexistence() -> tuple[float, float, float]:
    """Return ``(rho_l, rho_v, sigma)`` for the default SCMP parameter set."""
    return RHO_L_DEFAULT, RHO_V_DEFAULT, SIGMA_DEFAULT


def tanh_profile(
    shape: tuple[int, int],
    distance: torch.Tensor,
    rho_in: float,
    rho_out: float,
    width: float = 1.5,
) -> torch.Tensor:
    """``tanh`` two-phase profile across a signed distance field.

    Args:
        shape:     Output shape ``(ny, nx)`` (only used for validation).
        distance:  Signed distance to the interface in lattice units
                   (negative = inside the ``rho_in`` phase).
        rho_in:    Density of the inner (liquid) phase.
        rho_out:   Density of the outer (vapour) phase.
        width:     Diffuse-interface half-width (lattice units).

    Returns:
        Density field of shape ``shape``.
    """
    if distance.shape != shape:
        raise ValueError(f"distance shape {tuple(distance.shape)} != {shape}")
    if width <= 0.0:
        raise ValueError("width must be positive")
    return rho_out + (rho_in - rho_out) * 0.5 * (1.0 - torch.tanh(distance / width))


@dataclass(frozen=True)
class Channel:
    """Rectangular channel with no-slip walls on the bottom and top.

    The channel is periodic in ``x`` so that a sliding droplet can be tracked
    over arbitrarily long times (its centroid wraps).

    Attributes:
        nx:         Streamwise extent (lattice units, periodic).
        ny:         Wall-normal extent (lattice units).
        wall_rows:  Number of solid rows at the bottom *and* at the top.
        device:     Torch device string.
    """

    nx: int = 160
    ny: int = 72
    wall_rows: int = 2
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.nx < 8 or self.ny < 8:
            raise ValueError("channel must be at least 8x8")
        if self.wall_rows < 1:
            raise ValueError("wall_rows must be >= 1")
        if 2 * self.wall_rows >= self.ny - 4:
            raise ValueError("wall_rows too large for ny")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def n_fluid_y(self) -> int:
        """Number of fluid rows between the two walls."""
        return self.ny - 2 * self.wall_rows

    def y_wall(self) -> float:
        """Height of the bottom wall plane (half-way bounce-back location)."""
        return self.wall_rows - 0.5

    def y_top(self) -> float:
        """Height of the top wall plane."""
        return self.ny - self.wall_rows - 0.5

    def solid_mask(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Boolean ``(ny, nx)`` mask, ``True`` inside the solid rows."""
        dev = torch.device(device if device is not None else self.device)
        solid = torch.zeros(self.shape, dtype=torch.bool, device=dev)
        solid[: self.wall_rows, :] = True
        solid[self.ny - self.wall_rows :, :] = True
        return solid

    def grid(self, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(xx, yy)`` coordinate fields of shape ``(ny, nx)`` (cell centres)."""
        dev = torch.device(device if device is not None else self.device)
        yy, xx = torch.meshgrid(
            torch.arange(self.ny, dtype=torch.float32, device=dev),
            torch.arange(self.nx, dtype=torch.float32, device=dev),
            indexing="ij",
        )
        return xx, yy


def sessile_drop_state(
    channel: Channel,
    radius: float,
    center_x: float | None = None,
    height: float | None = None,
    *,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    width: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialise a circular droplet resting on the bottom wall.

    Args:
        channel:  Channel geometry.
        radius:   Initial droplet radius (lattice units).
        center_x: Streamwise centre (defaults to the channel centre).
        height:   Height of the droplet centre above the wall plane (defaults to
                  ``radius``, i.e. a droplet tangent to the wall).
        rho_l:    Liquid density of the coexistence pair.
        rho_v:    Vapour density.
        width:    Diffuse-interface half-width.

    Returns:
        ``(f, rho)`` with ``f`` the D2Q9 distribution at rest, shape
        ``(9, ny, nx)``.
    """
    dev = torch.device(channel.device)
    solid = channel.solid_mask(dev)
    xx, yy = channel.grid(dev)
    cx = channel.nx / 2.0 if center_x is None else float(center_x)
    cy = channel.y_wall() + (radius if height is None else float(height))
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - float(radius)
    rho = tanh_profile(channel.shape, dist, rho_l, rho_v, width)
    # Solid cells carry the liquid density so that the SC wall sum is well defined
    # before the first ``wall_psi`` substitution (the value is overwritten anyway).
    rho = torch.where(solid, torch.full_like(rho, rho_l), rho)
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero), rho


def meniscus_state(
    channel: Channel,
    fill_x: float,
    *,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    width: float = 1.5,
    cap_height: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialise a liquid plug occupying ``x < fill_x`` with a flat meniscus.

    This is the second contact-line geometry of the node: a meniscus spanning a
    plate channel.  The meniscus is a straight (flat) interface in the vapour,
    which relaxes to the wall-imposed capillary shape within a few thousand
    steps when ``cap_height == 0``.  A non-zero ``cap_height`` adds a parabolic
    bow of that height (sign selects advancing/receding curvature) so a run can
    start closer to the equilibrium shape.

    Args:
        channel:    Channel geometry.
        fill_x:     Streamwise position of the liquid/vapour interface.
        rho_l:      Liquid density.
        rho_v:      Vapour density.
        width:      Diffuse-interface half-width.
        cap_height: Parabolic bow amplitude (lattice units, + bows downstream).

    Returns:
        ``(f, rho)``.
    """
    dev = torch.device(channel.device)
    solid = channel.solid_mask(dev)
    xx, yy = channel.grid(dev)
    dist = xx - float(fill_x)
    if cap_height != 0.0:
        y0, y1 = channel.y_wall(), channel.y_top()
        half = 0.5 * (y1 - y0)
        eta = (yy - 0.5 * (y0 + y1)) / half
        dist = dist + float(cap_height) * (1.0 - eta**2)
    rho = tanh_profile(channel.shape, dist, rho_l, rho_v, width)
    rho = torch.where(solid, torch.full_like(rho, rho_l), rho)
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero), rho


class WallStepper:
    """Advance an SCMP distribution in a wall-bounded channel.

    The stepper owns the solid mask, the (per-cell) wall pseudopotential field
    and the forcing/relaxation parameters.  A single call to :meth:`step`
    performs collision + streaming + half-way bounce-back on the walls.

    Args:
        channel:      :class:`Channel` geometry.
        G:            SC coupling (library convention, ``> 0`` attractive).
        tau:          BGK relaxation time.
        rho_l:        Liquid density used to normalise ``wall_psi`` and the
                      level set.
        rho_v:        Vapour density.
        psi_fn:       Pseudopotential callable.
        forcing:      Forcing scheme (``"edm"`` recommended for the driven drop).
        scheme:       Interaction-force discretisation (``"standard"`` or
                      ``"pressure_tensor"``).
        wall_fraction: Initial uniform wall wetting ``psi_w / psi(rho_l)``.
        dtype:        Torch dtype of the distribution.
        compile_step: Wrap the fused step in ``torch.compile``.
        drive:        How a body force is distributed over the phases.
                      ``"uniform"`` applies ``F = rho g`` everywhere (gravity on
                      both liquid and vapour).  ``"liquid"`` applies
                      ``F = rho g * w(rho)`` with
                      ``w = clip((rho - rho_m)/(rho_l - rho_m), 0, 1)`` so that
                      only the dense phase is driven.  ``"liquid"`` is the mode
                      used for the driven-drop sweeps: with ``"uniform"`` the
                      vapour (which has no wall friction to speak of) is
                      accelerated and drags the droplet, so the droplet speed is
                      set by the vapour and not by the contact line.  The
                      phase-selective force is the lattice analogue of a dense
                      drop settling in a light gas, where the buoyancy acts on
                      the mass that is actually there.
    """

    def __init__(
        self,
        channel: Channel,
        *,
        G: float = 5.0,
        tau: float = 1.0,
        rho_l: float = RHO_L_DEFAULT,
        rho_v: float = RHO_V_DEFAULT,
        psi_fn: Callable[[torch.Tensor], torch.Tensor] = psi_exp,
        forcing: str = "edm",
        scheme: str = "standard",
        wall_fraction: float = 0.6,
        dtype: torch.dtype = torch.float32,
        compile_step: bool = False,
        drive: str = "uniform",
    ) -> None:
        self.channel = channel
        self.device = torch.device(channel.device)
        self.dtype = dtype
        self.G = float(G)
        self.tau = float(tau)
        self.rho_l = float(rho_l)
        self.rho_v = float(rho_v)
        self.rho_mid = 0.5 * (self.rho_l + self.rho_v)
        self.psi_fn = psi_fn
        self.forcing = forcing
        self.scheme = scheme
        if drive not in ("uniform", "liquid"):
            raise ValueError(f"drive must be 'uniform' or 'liquid', got {drive!r}")
        self.drive = drive
        self.solid = channel.solid_mask(self.device)
        self.psi_l = float(psi_fn(torch.tensor(self.rho_l, dtype=dtype)))
        self.wall_psi = self.uniform_wall(wall_fraction)
        self._step_fn = self._raw_step
        if compile_step:  # pragma: no cover - depends on the torch build
            try:
                self._step_fn = torch.compile(self._raw_step, dynamic=False)
            except Exception:  # pragma: no cover
                self._step_fn = self._raw_step

    def drive_field(self, rho: torch.Tensor, gx: torch.Tensor) -> torch.Tensor:
        """Per-cell streamwise acceleration for the configured drive mode."""
        if self.drive == "uniform":
            return gx
        w = torch.clamp((rho - self.rho_mid) / (self.rho_l - self.rho_mid), 0.0, 1.0)
        return gx * w

    def _raw_step(
        self, f: torch.Tensor, wall_psi: torch.Tensor, gx: torch.Tensor
    ) -> torch.Tensor:
        """One collision + stream + wall bounce-back (the compiled unit)."""
        if self.drive != "uniform":
            rho = macroscopic(f)[0]
            gx = self.drive_field(rho, gx)
        f = collide_sc_single_component(
            f,
            self.G,
            tau=self.tau,
            psi_fn=self.psi_fn,
            gx=gx,
            solid_mask=self.solid,
            wall_psi=wall_psi,
            forcing=self.forcing,
            scheme=self.scheme,
        )
        f = stream(f)
        return torch.where(self.solid.unsqueeze(0), f[OPPOSITE], f)

    # -- wall wetting -------------------------------------------------------
    def uniform_wall(self, fraction: float) -> torch.Tensor:
        """Per-cell wall-pseudopotential field for a uniform wetting fraction."""
        if fraction < 0.0:
            raise ValueError("wall wetting fraction must be >= 0")
        return torch.full(
            self.channel.shape,
            float(fraction) * self.psi_l,
            dtype=self.dtype,
            device=self.device,
        )

    def wall_fraction_field(self, field: torch.Tensor) -> torch.Tensor:
        """Convert a wetting-fraction field into a ``wall_psi`` field."""
        return field.to(dtype=self.dtype, device=self.device) * self.psi_l

    def set_wall_psi(self, wall_psi: torch.Tensor | float) -> None:
        """Install a (uniform or per-cell) wall pseudopotential field."""
        if isinstance(wall_psi, torch.Tensor):
            if wall_psi.shape != self.channel.shape:
                raise ValueError(
                    f"wall_psi shape {tuple(wall_psi.shape)} != {self.channel.shape}"
                )
            self.wall_psi = wall_psi.to(dtype=self.dtype, device=self.device)
        else:
            self.wall_psi = self.uniform_wall(float(wall_psi) / self.psi_l)

    def gradient_wall(self, fraction_left: float, fraction_right: float) -> torch.Tensor:
        """Linear wetting-gradient wall (``x = 0 -> left``, ``x = nx-1 -> right``)."""
        xx, _ = self.channel.grid(self.device)
        ramp = xx / max(self.channel.nx - 1, 1)
        field = fraction_left + (fraction_right - fraction_left) * ramp
        return self.wall_fraction_field(field)

    # -- state --------------------------------------------------------------
    def density(self, f: torch.Tensor) -> torch.Tensor:
        rho, _, _ = macroscopic(f)
        return rho

    def velocity(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, ux, uy = macroscopic(f)
        return ux, uy

    def step(self, f: torch.Tensor, gx: float | torch.Tensor = 0.0) -> torch.Tensor:
        """Advance one lattice step (collision + streaming + wall bounce-back).

        Args:
            f:  Distribution, shape ``(9, ny, nx)``.
            gx: Streamwise body-force acceleration (gravity-like, ``F = rho gx``).

        Returns:
            Updated distribution of the same shape.
        """
        gx_t = gx if isinstance(gx, torch.Tensor) else torch.as_tensor(
            float(gx), dtype=self.dtype, device=self.device
        )
        return self._step_fn(f, self.wall_psi, gx_t)

    def run(
        self,
        f: torch.Tensor,
        n_steps: int,
        *,
        gx: float = 0.0,
        callback: Callable[[int, torch.Tensor], None] | None = None,
        every: int = 100,
    ) -> torch.Tensor:
        """Advance ``n_steps`` steps, invoking ``callback(step, f)`` every ``every``."""
        for n in range(1, n_steps + 1):
            f = self.step(f, gx)
            if callback is not None and every > 0 and n % every == 0:
                callback(n, f)
        return f

    # -- diagnostics --------------------------------------------------------
    def spurious_current(self, f: torch.Tensor, liquid_only: bool = True) -> dict[str, float]:
        """Max/mean velocity magnitude away from the walls.

        Used to establish the *velocity floor* of the scheme: a dynamic angle or
        drift measurement is only meaningful when the driven velocity exceeds
        the residual (spurious) current of the static state.
        """
        rho = self.density(f)
        ux, uy = self.velocity(f)
        speed = torch.sqrt(ux**2 + uy**2)
        fluid = ~self.solid
        if liquid_only:
            fluid = fluid & (rho > 1.5)
        if not bool(fluid.any()):
            return {"max": float("nan"), "mean": float("nan")}
        return {
            "max": float(speed[fluid].max()),
            "mean": float(speed[fluid].mean()),
        }