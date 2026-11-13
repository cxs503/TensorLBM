"""Case specification for XVAL-LBM-CARTESIAN-MPI: 2D Poiseuille flow.

2D Poiseuille (plane channel) flow has a closed-form analytical solution:

    u(y) = (4 * U_max / H^2) * y * (H - y)     [parabolic profile]
    dp/dx = -8 * mu * U_max / H^2              [constant pressure gradient]

where H is the channel height and U_max is the centreline velocity.

In LBM lattice units (dx = dt = 1, rho = 1):
    nu = tau - 0.5 / 3
    Re = U_max * H / nu

The analytical solution provides an exact reference for velocity profile
comparison at any cross-section once the flow is fully developed.

This spec is consumed by both the monolithic and domain-decomposed runners,
ensuring identical physics, geometry, and non-dimensional numbers.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PoiseuilleSpec:
    """Frozen specification for 2D Poiseuille flow cross-validation."""

    # Geometry
    nx: int = 256          # streamwise resolution
    ny: int = 64           # wall-normal resolution
    u_max: float = 0.05    # centreline velocity (lattice units, Ma < 0.1)

    # Physics
    re: float = 100.0      # Reynolds number Re = U_max * H / nu

    # LBM parameters
    tau: float | None = None  # relaxation time (None = auto from Re)

    # Simulation control
    steps: int = 8000      # total time steps
    sample_every: int = 200  # sampling interval for history
    warmup_steps: int = 2000  # steps before sampling (flow development)

    # Domain decomposition (emulating pylbm cartesian-MPI)
    npx: int = 2           # number of subdomains in x
    npy: int = 2           # number of subdomains in y
    halo: int = 1          # halo width (D2Q9 needs 1 layer)

    # Metadata
    name: str = "poiseuille_2d"
    ref_id: str = "XVAL-LBM-CARTESIAN-MPI"

    # ---- derived quantities ----

    @property
    def H(self) -> float:
        """Channel height in lattice units."""
        return float(self.ny)

    @property
    def nu(self) -> float:
        """Kinematic viscosity from Re = U_max * H / nu."""
        return self.u_max * self.H / self.re

    @property
    def tau_computed(self) -> float:
        """Relaxation time: tau = 0.5 + 3*nu."""
        if self.tau is not None:
            return self.tau
        return 0.5 + 3.0 * self.nu

    @property
    def mach(self) -> float:
        """Mach number (lattice): Ma = U_max * sqrt(3)."""
        return self.u_max * math.sqrt(3.0)

    @property
    def knudsen(self) -> float:
        """Knudsen number estimate: Kn = sqrt(pi/2) * Ma / Re."""
        return math.sqrt(math.pi / 2) * self.mach / self.re

    def analytical_velocity(self, y: float) -> float:
        """Analytical Poiseuille velocity profile at wall-normal position y.

        Parameters
        ----------
        y : float
            Wall-normal coordinate (0 <= y <= H).

        Returns
        -------
        float
            Streamwise velocity u(y).
        """
        return 4.0 * self.u_max / (self.H ** 2) * y * (self.H - y)

    def analytical_pressure_gradient(self) -> float:
        """Analytical pressure gradient dp/dx."""
        mu = self.nu  # rho = 1 in lattice units
        return -8.0 * mu * self.u_max / (self.H ** 2)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PoiseuilleSpec":
        """Deserialize from dict."""
        known = {k: v for k, v in d.items() if k in (f.name for f in dataclasses.fields(cls))}
        return cls(**known)


# Default specification for the cross-validation campaign
DEFAULT_SPEC = PoiseuilleSpec(
    nx=256,
    ny=64,
    u_max=0.05,
    re=100.0,
    steps=8000,
    sample_every=200,
    warmup_steps=2000,
    npx=2,
    npy=2,
    halo=1,
)
