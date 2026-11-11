"""Single-source case specification for the XVAL-LBM-TAYLORGREEN cross-validation.

One spec object is consumed by *every* tool (in-house TensorLBM, palabos,
OpenFOAM icoFoam) and by the analytic reference, so the three runs are
guaranteed to share grid, non-dimensional numbers, initial condition, boundary
treatment and sampling convention.  ``TG2DSpec.to_dict()`` is what ends up in
``result.case_spec``.

Physics (2D Taylor-Green vortex, lattice units dx = dt = 1)
-----------------------------------------------------------
    u(x,y,t) = -U0 cos(kx) sin(ky) e^{-2 nu k^2 t}
    v(x,y,t) = +U0 sin(kx) cos(ky) e^{-2 nu k^2 t}
    K(t)     = 0.5 <u^2+v^2> = (U0^2/4) e^{-4 nu k^2 t}

The TG vortex is an *exact* solution of the incompressible Navier-Stokes
equations (the convective term is a pure gradient, absorbed by the pressure),
so ``gamma_K = 4 nu k^2`` and ``gamma_vel = 2 nu k^2`` are known in closed
form -- there is no interpolation and no series truncation involved.  The
"series solution" in the acceptance criteria is the Fourier series of this
closed form, which for a single mode collapses to the two expressions above.

Boosted variant (``boost > 0``) superposes a uniform translation ``U_b`` along
x.  Galilean invariance of the Navier-Stokes equations makes
``u_tot = U_b + u(x - U_b t, y, t)`` an exact solution as well, which turns the
otherwise degenerate "peak-velocity phase" criterion into a real measurement:
the vortex pattern must travel at exactly ``U_b`` (lattice/BGK schemes only
reproduce this up to their Galilean defect, which is what we want to measure).
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TG2DSpec:
    """Frozen case specification (identical for every tool)."""

    n: int = 128
    re: float = 100.0
    u0: float = 0.05
    boost: float = 0.0
    steps: int = 4900
    sample_every: int = 100
    field_steps: tuple[int, ...] = (1000, 2500)
    name: str = "taylor_green_2d_decay"
    ref_id: str = "XVAL-TAYLORGREEN"

    # ---- derived quantities (single definition, used by all tools) --------
    @property
    def L(self) -> float:
        return float(self.n)

    @property
    def k(self) -> float:
        return 2.0 * math.pi / self.L

    @property
    def nu(self) -> float:
        """Lattice kinematic viscosity from Re = U0 L / nu."""
        return self.u0 * self.L / self.re

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def mach(self) -> float:
        return self.u0 * math.sqrt(3.0)

    @property
    def gamma_k_theory(self) -> float:
        """Kinetic-energy decay rate of the exact solution: 4 nu k^2."""
        return 4.0 * self.nu * self.k * self.k

    @property
    def gamma_vel_theory(self) -> float:
        """Velocity decay rate of the exact solution: 2 nu k^2."""
        return 2.0 * self.nu * self.k * self.k

    @property
    def k0_theory(self) -> float:
        return 0.25 * self.u0 * self.u0

    @property
    def is_boosted(self) -> bool:
        return self.boost != 0.0

    def with_overrides(self, **kwargs: Any) -> "TG2DSpec":
        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """The ``result.case_spec`` payload (one spec, three uses)."""
        return {
            "ref_id": self.ref_id,
            "case": self.name,
            "geometry": {
                "kind": "2D periodic box [0,L)^2",
                "n": self.n,
                "L_lattice": self.L,
                "dx": 1.0,
                "dt": 1.0,
                "cell_centre": "x_i = i + 0.5 (identical in all tools)",
            },
            "nondim": {
                "Re": self.re,
                "U0": self.u0,
                "Ma": self.mach,
                "nu_lattice": self.nu,
                "tau_bgk": self.tau,
                "k": self.k,
                "wave_lengths": self.L,
                "boost_x": self.boost,
            },
            "initial_condition": {
                "rho": 1.0,
                "u": "-U0 cos(kx) sin(ky) + boost",
                "v": "+U0 sin(kx) cos(ky)",
                "f": "f = f_eq(rho, u) exactly (no density perturbation)",
            },
            "lattice": {"model": "D2Q9", "collision": "BGK (SRT)", "bc": "periodic"},
            "sampling": {
                "steps": self.steps,
                "sample_every": self.sample_every,
                "field_steps": list(self.field_steps),
                "K": "K = 0.5 * mean(u^2 + v^2) over all nodes",
                "umax": "max |u| over all nodes",
            },
            "analytic_reference": {
                "source": "closed-form 2D Taylor-Green (exact Navier-Stokes)",
                "gamma_K": self.gamma_k_theory,
                "gamma_vel": self.gamma_vel_theory,
                "K0": self.k0_theory,
            },
            "criteria": {
                "cross_tool_gamma_K_pct": 5.0,
                "vs_exact_gamma_K_pct": 8.0,
                "phase_pct": 3.0,
            },
        }

    # canonical command-line form for the palabos binary --------------------
    def palabos_argv(self, out_dir: str) -> list[str]:
        return [
            out_dir,
            str(self.n),
            repr(float(self.re)),
            repr(float(self.u0)),
            str(self.steps),
            str(self.sample_every),
            ",".join(str(s) for s in self.field_steps),
            repr(float(self.boost)),
        ]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TG2DSpec":
        return cls(
            n=int(payload["geometry"]["n"]),
            re=float(payload["nondim"]["Re"]),
            u0=float(payload["nondim"]["U0"]),
            boost=float(payload["nondim"].get("boost_x", 0.0)),
            steps=int(payload["sampling"]["steps"]),
            sample_every=int(payload["sampling"]["sample_every"]),
            field_steps=tuple(int(s) for s in payload["sampling"]["field_steps"]),
            name=str(payload.get("case", "taylor_green_2d_decay")),
            ref_id=str(payload.get("ref_id", "XVAL-TAYLORGREEN")),
        )

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
            fh.write("\n")


DEFAULT_SPEC = TG2DSpec()


def boosted_spec(spec: TG2DSpec | None = None, boost: float | None = None) -> TG2DSpec:
    """Galilean-boosted companion case (phase criterion with teeth)."""
    base = spec or DEFAULT_SPEC
    ub = base.u0 if boost is None else float(boost)
    return base.with_overrides(boost=ub, name="taylor_green_2d_boosted")