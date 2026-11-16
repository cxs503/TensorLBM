"""Single-point configuration for D2Q9 lattice Boltzmann runs.

Absorbed architectural point
----------------------------
lssfau/lbmpy builds a simulation from **one** ``LBMConfig`` dataclass
(``lbmpy/creationfunctions.py``): stencil, method (collision family),
relaxation rate, force model and force vector are declared once, and the
equilibrium that is relaxed towards, the force source term, the macroscopic
velocity output and the streaming step are all *derived* from that single
object.  Nothing about the scheme can be configured at a second place, so the
solver, the initialisation and the diagnostics cannot disagree about the
relaxation time or about whether the velocity contains the half-step force
correction.

Before this module, TensorLBM D2Q9 runs configured the same physical scheme
through scattered call arguments: ``collide_bgk(f, tau)``,
``collide_trt(f, tau_plus, lambda_trt)``, a hand written Guo source term,
``macroscopic(f)`` (which returns the *raw* momentum velocity) and one of
several velocity-output conventions depending on the caller.  ``nu``, ``tau``,
the discrete-lattice force correction and the velocity convention therefore
had to be re-derived (correctly!) at every call site.

The two Guo variants are a concrete example of why one place matters.  The
textbook scheme (``force_model="guo"``, Guo, Zheng & Shi 2002) relaxes
towards the equilibrium built with the *mechanical* velocity
``u = j/rho + F/(2 rho)`` and uses the ``(1 - omega/2)`` corrected source
term -- the two halves inject exactly ``F`` of momentum per node and step.
Pairing the corrected source with a raw-velocity equilibrium instead injects
only ``(1 - omega/2) F`` and turns a plane-Poiseuille channel into a profile
that is short by exactly that factor (measured: 70 % error at
``omega = 1.4``; see ``docs/XVAL_LBM_LBMCONFIG_SINGLE_POINT_CONFIG.md``).
The configuration below pairs the two halves itself and *rejects* the
inconsistent declaration, so that defect is not expressible.

This module is the production home of the pattern for 2D problems.  It is a
pure, frozen value object: all derived quantities are properties, so the
scheme is a mathematical function of the configuration (unit-testable without
a grid).  No code is copied from lbmpy; only the architectural pattern is
transplanted (see ``docs/XVAL_LBM_LBMCONFIG_SINGLE_POINT_CONFIG.md``).

Conventions (documented once, here)
-----------------------------------
* ``relaxation_rate`` is the lbmpy convention ``omega = 1 / tau`` for the
  shear mode -- **not** the "velocity relaxation" ``omega = nu``.  The
  kinematic viscosity follows ``nu = cs^2 (tau - 1/2)`` with ``cs^2 = 1/3``.
* ``method="TRT"``: the anti-symmetric relaxation time is derived from the
  magic parameter ``Lambda`` (Ginzburg 2008):
  ``tau_minus = 1/2 + Lambda / (tau_plus - 1/2)``.
* ``force_model="guo"`` (textbook): equilibrium and source use the mechanical
  velocity ``u = j/rho + F/(2 rho)``; source scale ``1 - omega/2``; the
  macroscopic velocity output carries the half step force.  Restricted to the
  SRT/TRT families (an MRT collision leaves the conserved momentum moments
  unrelaxed, so the half-step momentum bookkeeping cannot be expressed).
* ``force_model="guo_split"``: the in-repo ``channel_kernel_numpy``
  convention -- equilibrium uses the raw velocity ``j/rho`` and the source
  term carries the **full** force (no ``1 - omega/2`` factor); the velocity
  output is the raw velocity.  Momentum injection per node and step is again
  exactly ``F``.  Selectable for every collision family.
* ``force_model="simple"``: first-order source ``S_i = 3 w_i (c_i . F)`` with
  no velocity coupling.  Kept as the documented naive reference; it is
  momentum consistent (injects ``F``) but drops the ``O(u F)`` source
  moments, so it is *not* validated for unsteady flows.
* Half-way bounce-back walls sit half a lattice unit outside the first and
  last fluid row, i.e. at ``y = 1/2`` and ``y = H + 1/2`` for fluid rows
  ``1..H``.

References
----------
* Guo, Z., Zheng, C., Shi, B. (2002). Discrete lattice effects on the forcing
  term in the lattice Boltzmann method. *Phys. Rev. E* 65, 046308.
* Ginzburg, I. (2008). Two-relaxation-time lattice Boltzmann scheme.
  *Commun. Comput. Phys.* 3(2), 427-478.
* Krueger, T. et al. (2017). *The Lattice Boltzmann Method: Principles and
  Practice*. Springer, ch. 9 (force terms and their momentum bookkeeping).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import torch

from . import d2q9
from .solver import _get_d2q9_mrt_matrices, stream

__all__ = ["LBMConfig", "DEFAULT_MRT_RATES", "FORCE_MODELS", "METHODS"]

#: Supported collision families.
METHODS: tuple[str, ...] = ("SRT", "TRT", "MRT")
#: Supported force models (``none`` means no body force term at all).
FORCE_MODELS: tuple[str, ...] = ("guo", "guo_split", "simple", "none")
#: Force models whose momentum injection requires the half-step velocity
#: bookkeeping of the textbook Guo scheme, and the collision families for
#: which that bookkeeping can be expressed.
_HALF_STEP_MODELS: tuple[str, ...] = ("guo",)
#: Non-hydrodynamic relaxation rates for MRT (solver.collide_mrt defaults).
DEFAULT_MRT_RATES: Mapping[str, float] = {
    "s_e": 1.64,
    "s_eps": 1.54,
    "s_q": 1.7,
}
#: Lattice speed of sound squared for the D2Q9 stencil.
CS2: float = 1.0 / 3.0


@dataclass(frozen=True)
class LBMConfig:
    """One object that fully determines a D2Q9 scheme.

    >>> cfg = LBMConfig(method="SRT", relaxation_rate=1.5, force=(1e-5, 0.0),
    ...                 force_model="guo_split")
    >>> round(cfg.tau, 12), round(cfg.nu, 12)
    (0.666666666667, 0.0555555555556)
    """

    stencil: str = "D2Q9"
    method: str = "SRT"
    #: omega = 1/tau for the shear mode (lbmpy convention).
    relaxation_rate: float = 1.5
    #: TRT magic parameter Lambda (only used when method == "TRT").
    lambda_trt: float = 3.0 / 16.0
    #: Body force per unit mass (lattice units), applied on every fluid node.
    force: tuple[float, float] = (0.0, 0.0)
    #: ``guo`` | ``guo_split`` | ``simple`` | ``none``.
    force_model: str = "none"
    #: Diagnostics convention: does the macroscopic velocity carry the half
    #: step force term?  ``None`` resolves from ``force_model``.
    velocity_output_includes_force: bool | None = None
    #: MRT rates for the non-hydrodynamic moments.
    mrt_rates: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_MRT_RATES))

    # -- validation: the single point where a bad scheme can be declared ------
    def __post_init__(self) -> None:
        if self.stencil != "D2Q9":
            raise ValueError(
                f"LBMConfig currently supports stencil='D2Q9', got {self.stencil!r}"
            )
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if not (0.0 < self.relaxation_rate < 2.0):
            raise ValueError(
                "relaxation_rate = 1/tau must lie in (0, 2) (tau > 1/2 for a "
                f"positive viscosity); got {self.relaxation_rate!r}"
            )
        if self.lambda_trt <= 0.0:
            raise ValueError(f"lambda_trt must be > 0, got {self.lambda_trt!r}")
        if self.force_model not in FORCE_MODELS:
            raise ValueError(
                f"force_model must be one of {FORCE_MODELS}, got {self.force_model!r}"
            )
        if len(tuple(self.force)) != 2:
            raise ValueError(f"force must have 2 components, got {self.force!r}")
        fx, fy = (float(v) for v in self.force)
        if not (math.isfinite(fx) and math.isfinite(fy)):
            raise ValueError(f"force must be finite, got {self.force!r}")
        if self.force_model == "none" and (fx != 0.0 or fy != 0.0):
            raise ValueError(
                "force_model='none' but a non-zero force was declared; pass "
                "force_model='guo' (or 'guo_split'/'simple') to apply the force"
            )
        unknown = set(self.mrt_rates) - set(DEFAULT_MRT_RATES)
        if unknown:
            raise ValueError(f"unknown MRT rates {sorted(unknown)}")

        half_step = self.force_model in _HALF_STEP_MODELS
        if self.velocity_output_includes_force is None:
            object.__setattr__(self, "velocity_output_includes_force", half_step)
        elif bool(self.velocity_output_includes_force) != half_step:
            raise ValueError(
                "inconsistent force bookkeeping: force_model="
                f"{self.force_model!r} implies velocity_output_includes_force="
                f"{half_step}, got {self.velocity_output_includes_force!r}. "
                "Pairing the (1 - omega/2) corrected source with a raw-velocity "
                "equilibrium injects only (1 - omega/2) F of momentum per step "
                "(measured: 70 % plane-Poiseuille error at omega = 1.4); use "
                "force_model='guo_split' for the raw-velocity convention."
            )
        if half_step and self.method == "MRT":
            raise ValueError(
                "force_model='guo' cannot be combined with method='MRT': an MRT "
                "collision leaves the conserved momentum moments unrelaxed, so "
                "the half-step momentum bookkeeping of the textbook Guo scheme "
                "cannot be expressed; use force_model='guo_split'"
            )

    # -- derived single-point quantities ------------------------------------
    @property
    def tau(self) -> float:
        """Shear relaxation time ``tau = 1/omega``."""
        return 1.0 / self.relaxation_rate

    @property
    def nu(self) -> float:
        """Kinematic viscosity ``nu = cs^2 (tau - 1/2)``."""
        return CS2 * (self.tau - 0.5)

    @property
    def tau_minus(self) -> float:
        """TRT anti-symmetric relaxation time from the magic parameter."""
        return 0.5 + self.lambda_trt / (self.tau - 0.5)

    @property
    def omega_minus(self) -> float:
        """TRT anti-symmetric relaxation rate ``1 / tau_minus``."""
        return 1.0 / self.tau_minus

    @property
    def has_force(self) -> bool:
        return self.force_model != "none"

    @property
    def momentum_relaxation_rate(self) -> float:
        """Relaxation rate governing the momentum-carrying mode.

        In a TRT collision the symmetric part carries no momentum
        (``sum_i c_i f_i^+ = 0``), so the momentum half step of the textbook
        Guo bookkeeping is relaxed at ``omega_minus``; for SRT the single rate
        applies.
        """
        return self.omega_minus if self.method == "TRT" else self.relaxation_rate

    @property
    def force_source_scale(self) -> float:
        """Discrete-lattice correction factor on the Guo source term.

        ``1 - omega_momentum/2`` pairs with the half step velocity so that
        collision + source inject exactly ``F`` of momentum per node and step.
        """
        if self.force_model == "guo":
            return 1.0 - 0.5 * self.momentum_relaxation_rate
        return 1.0

    def scheme(self) -> dict[str, Any]:
        """All derived scheme parameters, as an auditable dict."""
        return {
            "stencil": self.stencil,
            "method": self.method,
            "relaxation_rate": self.relaxation_rate,
            "tau": self.tau,
            "nu": self.nu,
            "cs2": CS2,
            "lambda_trt": self.lambda_trt if self.method == "TRT" else None,
            "tau_minus": self.tau_minus if self.method == "TRT" else None,
            "omega_minus": self.omega_minus if self.method == "TRT" else None,
            "force_model": self.force_model,
            "force": tuple(float(v) for v in self.force),
            "momentum_relaxation_rate": self.momentum_relaxation_rate,
            "force_source_scale": self.force_source_scale,
            "velocity_output_includes_force": bool(self.velocity_output_includes_force),
            "mrt_rates": dict(self.mrt_rates) if self.method == "MRT" else None,
        }

    # -- scheme primitives --------------------------------------------------
    def equilibrium(
        self,
        rho: torch.Tensor,
        ux: torch.Tensor,
        uy: torch.Tensor,
    ) -> torch.Tensor:
        """Second-order D2Q9 equilibrium implied by this configuration."""
        return d2q9.equilibrium(rho, ux, uy)

    def momentum(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``rho, j_x, j_y`` from the raw populations."""
        c = d2q9.C.to(f.device)
        rho = f.sum(dim=0)
        jx = (f * c[:, 0].view(9, 1, 1)).sum(dim=0)
        jy = (f * c[:, 1].view(9, 1, 1)).sum(dim=0)
        return rho, jx, jy

    def velocity(
        self,
        f: torch.Tensor,
        *,
        include_force: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Macroscopic ``rho, u_x, u_y`` under this configuration's convention."""
        if include_force is None:
            include_force = bool(self.velocity_output_includes_force)
        rho, jx, jy = self.momentum(f)
        rho_safe = torch.clamp(rho, min=1e-12)
        ux = jx / rho_safe
        uy = jy / rho_safe
        if include_force and self.has_force:
            fx, fy = (float(v) for v in self.force)
            ux = ux + 0.5 * fx / rho_safe
            uy = uy + 0.5 * fy / rho_safe
        return rho, ux, uy

    def relax(self, f: torch.Tensor, feq: torch.Tensor) -> torch.Tensor:
        """Relax ``f`` towards the supplied equilibrium under this method."""
        if self.method == "SRT":
            return f - self.relaxation_rate * (f - feq)
        opp = d2q9.OPPOSITE.to(f.device)
        if self.method == "TRT":
            f_plus = 0.5 * (f + f[opp])
            f_minus = 0.5 * (f - f[opp])
            e_plus = 0.5 * (feq + feq[opp])
            e_minus = 0.5 * (feq - feq[opp])
            return (
                f
                - (f_plus - e_plus) * self.relaxation_rate
                - (f_minus - e_minus) * self.omega_minus
            )
        matrix, matrix_inv = _get_d2q9_mrt_matrices(f.device, f.dtype)
        ny, nx = f.shape[1], f.shape[2]
        rates = dict(self.mrt_rates)
        s_vec = torch.tensor(
            [0.0, rates["s_e"], rates["s_eps"], 0.0, rates["s_q"], 0.0,
             rates["s_q"], self.relaxation_rate, self.relaxation_rate],
            dtype=f.dtype,
            device=f.device,
        )
        m = matrix @ f.reshape(9, -1)
        m_eq = matrix @ feq.reshape(9, -1)
        return (matrix_inv @ (m - s_vec.unsqueeze(1) * (m - m_eq))).reshape(9, ny, nx)

    def collide(self, f: torch.Tensor) -> torch.Tensor:
        """Post-collision populations (no force source, no streaming).

        The equilibrium is built with the velocity this configuration declares
        (mechanical velocity for ``guo``, raw velocity otherwise), which is
        what makes the collision and the source term momentum-consistent.
        """
        rho, ux, uy = self.velocity(f)
        return self.relax(f, self.equilibrium(rho, ux, uy))

    def force_source(self, f: torch.Tensor) -> torch.Tensor:
        """Force source term to be added to the post-collision populations."""
        if not self.has_force:
            return torch.zeros_like(f)
        fx, fy = (float(v) for v in self.force)
        device = f.device
        c = d2q9.C.to(device).to(f.dtype)
        w = d2q9._w_for(f.dtype, device).view(9, 1, 1)
        # Velocity entering the source term: the mechanical velocity for the
        # textbook scheme, the raw velocity for the split/naive variants.
        _, ux, uy = self.velocity(f)
        cx = c[:, 0].view(9, 1, 1)
        cy = c[:, 1].view(9, 1, 1)
        if self.force_model in ("guo", "guo_split"):
            cu = cx * ux + cy * uy
            cf = cx * fx + cy * fy
            uf = ux * fx + uy * fy
            return 3.0 * w * self.force_source_scale * (cf - uf + 3.0 * cu * cf)
        # first-order "simple" model: S_i = 3 w_i (c_i . F)
        # (a uniform force is node-independent; broadcast to the field shape)
        return torch.broadcast_to(3.0 * w * (cx * fx + cy * fy), f.shape)

    def step(self, f: torch.Tensor) -> torch.Tensor:
        """One collide + force + stream step (boundaries are the caller's job)."""
        return stream(self.collide(f) + self.force_source(f))

    # -- serialisation ------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        """JSON round-trippable declaration (only the declared knobs)."""
        return {
            "stencil": self.stencil,
            "method": self.method,
            "relaxation_rate": self.relaxation_rate,
            "lambda_trt": self.lambda_trt,
            "force": [float(v) for v in self.force],
            "force_model": self.force_model,
            "velocity_output_includes_force": bool(self.velocity_output_includes_force),
            "mrt_rates": {k: float(v) for k, v in self.mrt_rates.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LBMConfig":
        payload = dict(data)
        if "force" in payload:
            payload["force"] = tuple(float(v) for v in payload["force"])
        return cls(**payload)

    def with_(self, **changes: Any) -> "LBMConfig":
        """Copy with overrides -- the supported way to build variants."""
        if "force_model" in changes and "velocity_output_includes_force" not in changes:
            changes["velocity_output_includes_force"] = None
        return replace(self, **changes)