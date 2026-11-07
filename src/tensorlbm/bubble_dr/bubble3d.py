"""Single-bubble rise gate: LBM terminal velocity vs the Hadamard-Rybczynski law.

Campaign node IN57-N02, criterion (1).

Physical problem
----------------
A spherical gas bubble of radius ``R`` rises at terminal velocity ``U`` through
a quiescent liquid of dynamic viscosity ``mu_c`` in the creeping-flow limit.
Hadamard (1911) / Rybczynski (1911) give, for viscosity ratio
``lam = mu_d/mu_c``,

    U_HR = (2/3) (drho g) R^2 (1 + lam) / (mu_c (2 + 3 lam)) .

The lattice runs here use a *single* relaxation time, i.e. the kinematic
viscosity is uniform, so ``lam = mu_gas/mu_liquid = rho_gas/rho_liquid`` is
fixed by the equation of state and is **not** a free knob -- exactly the trap
that killed the m155 attempt (``benchmarks/pending/hadamard_rybczynski_droplet/
terminal_verdict_m155.md``: G2 "viscosity ratio slaved to density ratio").  The
reference is therefore evaluated at the *realised* ``lam``, not at ``lam = 0``.

Two further confounders are handled explicitly rather than tuned away:

* **confinement** -- a periodic box of side ``L`` confines the bubble at
  ``R/H = 2R/L``; three box sizes are run and ``U(R/H)`` is extrapolated
  linearly to ``R/H -> 0`` (the lesson of m155: the container inflated the drag
  by 19-31 % at ``R/H = 0.1``);
* **finite Re** -- HR is the ``Re -> 0`` asymptote.  ``Re`` is reported for
  every run and the first-order finite-Re drag correction
  ``(1 + Re/8)`` (Taylor & Acrivos 1964, clean bubble) is applied as a
  *stated, separate* retraction so the reader can see both numbers.

Buoyancy is applied as a **net-zero** body force ``F = (rho - rho_ref) g``
(``rho_ref`` = box mean density).  With periodic boundaries a uniform gravity
cannot be balanced hydrostatically, so the net-zero form is the only
momentum-conserving way to impose buoyancy in a closed periodic box.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..hadamard_rybczynski import hr_velocity
from ..multiphase3d import (
    equilibrium3d,
    macroscopic3d,
    psi_exp,
    _sc_neighbor_weighted_sum_3d,
)
from ..solver3d import stream3d

CS2 = 1.0 / 3.0


@dataclass
class BubbleConfig:
    """One rising-bubble run (lattice units)."""

    L: int = 48
    radius: float = 6.0
    G: float = 5.0
    tau: float = 1.0
    psi_rho0: float = 1.0
    #: discrete coexistence densities of the shipped D3Q19 SC scheme at
    #: ``G = 5`` / ``psi_exp(rho0 = 1)`` (measured in
    #: ``benchmarks/verified/laplace_droplet``: 1.957 / 0.1596).
    rho_liquid: float = 1.957
    rho_gas: float = 0.1596
    g_buoy: float = 2.0e-3
    steps: int = 1400
    burn_in: int = 200
    #: interface width of the tanh initialisation (cells).  Must be narrower than
    #: ``radius``; the SC interface relaxes to its own profile within ~100 steps.
    W_int: float = 4.0
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    @property
    def nu(self) -> float:
        """Uniform kinematic viscosity implied by ``tau``."""
        return CS2 * (self.tau - 0.5)

    def validate(self) -> None:
        if self.L < 8 or not 2.0 <= self.radius < 0.35 * self.L:
            raise ValueError("need L >= 8 and 2 <= radius < 0.35 L")
        if self.tau <= 0.55:
            raise ValueError("tau must exceed 0.55")
        if self.g_buoy <= 0.0:
            raise ValueError("g_buoy must be positive")


def _buoyant_collide(
    f: torch.Tensor,
    rho_ref: torch.Tensor | float,
    cfg: BubbleConfig,
) -> torch.Tensor:
    """Shan-Chen BGK collision with a net-zero buoyancy body force.

    Force ``F = -G psi (x) sum_i w_i c_i psi(x - c_i) + (rho - rho_ref) g e_z``,
    stencilled exactly like :func:`tensorlbm.multiphase3d.collide_sc_single_component_3d`
    so that the phase-separation physics is unchanged; only the body force differs.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    psi = psi_exp(rho, cfg.psi_rho0)
    sx, sy, sz = _sc_neighbor_weighted_sum_3d(psi, None)
    rho_s = torch.clamp(rho, min=1e-12)
    fz = -cfg.G * psi * sz + (rho - rho_ref) * cfg.g_buoy
    fx = -cfg.G * psi * sx
    fy = -cfg.G * psi * sy
    feq = equilibrium3d(
        rho, ux + cfg.tau * fx / rho_s, uy + cfg.tau * fy / rho_s, uz + cfg.tau * fz / rho_s
    )
    return f - (f - feq) / cfg.tau


class RisingBubble3D:
    """A single SC bubble in a periodic box under net-zero buoyancy."""

    def __init__(self, cfg: BubbleConfig | None = None):
        cfg = cfg or BubbleConfig()
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        L = cfg.L
        self.shape = (L, L, L)
        zz, yy, xx = torch.meshgrid(
            *[torch.arange(L, dtype=cfg.dtype, device=self.device)] * 3, indexing="ij"
        )
        self.zz, self.yy, self.xx = zz, yy, xx
        self.rr = torch.sqrt(
            (xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2 + (zz - L / 2.0) ** 2
        )
        # Mass-consistent coexistence initialisation: filling a fraction ``f`` of
        # the box with gas at ``rho_gas`` and the rest with liquid puts the box
        # mean density at ``f rho_gas + (1-f) rho_liquid``, i.e. exactly the
        # two-phase state at coexistence.  Initialising at ``rho > rho_liquid``
        # instead supersaturates the liquid and the bubble dissolves (measured:
        # R_eq 6 -> 0 within 600 steps) -- that trap cost one pilot run.
        # tanh profile: the volume-weighted mean of this field is exactly the
        # coexistence mean density for the chosen radius, so the run starts
        # neither supersaturated nor undersaturated (see the module docstring and
        # the report's trap list: a sharp 2.0/0.2 init dissolved a R=6 bubble).
        rho = cfg.rho_gas + 0.5 * (cfg.rho_liquid - cfg.rho_gas) * (
            1.0 + torch.tanh((self.rr - cfg.radius) / cfg.W_int)
        )
        rho = rho.clamp(min=1e-3)
        zero = torch.zeros_like(rho)
        self.f = equilibrium3d(rho, zero, zero, zero)
        self.rho_ref = float(rho.mean())
        self.mass0 = float(rho.sum())
        self.t = 0
        self.history: list[dict[str, float]] = []

    # -- diagnostics ---------------------------------------------------------
    def fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return macroscopic3d(self.f)

    def coexist(self, rho: torch.Tensor) -> tuple[float, float]:
        """Liquid / gas densities from the outer / inner radial bands."""
        R = self.cfg.radius
        liquid = float(rho[self.rr > 1.5 * R].mean())
        gas = float(rho[self.rr < 0.6 * R].mean())
        return liquid, gas

    def gas_fraction(self, rho: torch.Tensor) -> torch.Tensor:
        liquid, gas = self.coexist(rho)
        alpha = (liquid - rho) / max(liquid - gas, 1e-12)
        return torch.clamp(alpha, 0.0, 1.0)

    def measure(self) -> dict[str, float]:
        rho, ux, uy, uz = self.fields()
        liquid, gas = self.coexist(rho)
        alpha = self.gas_fraction(rho)
        vol = float(alpha.sum())
        r_eq = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0) if vol > 0 else float("nan")
        w = alpha.clamp_min(0.0)
        wsum = float(w.sum())
        if wsum > 0:
            u_gas = float((w * uz).sum() / wsum)
            z_c = float((w * self.zz).sum() / wsum)
        else:
            u_gas = z_c = float("nan")
        return {
            "step": float(self.t),
            "rho_liquid": liquid,
            "rho_gas": gas,
            "density_ratio": liquid / max(gas, 1e-12),
            "R_eq": r_eq,
            "u_gas_z": u_gas,
            "z_centroid": z_c,
            "mass_drift": abs(float(rho.sum()) - self.mass0) / self.mass0,
            "max_u": float(torch.sqrt(ux**2 + uy**2 + uz**2).max()),
        }

    # -- time integration ----------------------------------------------------
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.f = stream3d(_buoyant_collide(self.f, self.rho_ref, self.cfg))
            self.t += 1
            if self.t % 20 == 0:
                self.history.append(self.measure())

    def run(self) -> dict[str, object]:
        t0 = time.perf_counter()
        self.step(self.cfg.steps)
        hist = self.history
        tail = hist[-max(3, len(hist) // 4):]
        u_mean = float(np.mean([h["u_gas_z"] for h in tail]))
        u_std = float(np.std([h["u_gas_z"] for h in tail]))
        # Centroid-drift estimate: least-squares slope of z_centroid(t).  This is
        # the estimator used for the gate -- the Shan-Chen interface carries
        # steady spurious currents of order 1e-1 (measured), which bias the
        # phase-averaged velocity but do not translate the bubble.
        if len(tail) >= 3:
            tt = np.array([h["step"] for h in tail])
            zz = np.array([h["z_centroid"] for h in tail])
            u_c = float(np.polyfit(tt, zz, 1)[0])
        else:
            u_c = float("nan")
        last = hist[-1]
        cfg = self.cfg
        liquid = last["rho_liquid"]
        gas = last["rho_gas"]
        mu_c = liquid * cfg.nu
        lam = gas / liquid
        drho_eff = self.rho_ref - gas
        R = last["R_eq"]
        u_hr = hr_velocity(drho_eff, cfg.g_buoy, R, mu_c, lam)
        u_gate = u_c if np.isfinite(u_c) else u_mean
        re = u_gate * 2.0 * R / cfg.nu
        # first-order finite-Re retraction for a clean bubble (Taylor & Acrivos 1964)
        u_stokes = u_gate / (1.0 + re / 8.0)
        return {
            "L": cfg.L,
            "R_init": cfg.radius,
            "R_eq": R,
            "confinement": 2.0 * R / cfg.L,
            "rho_liquid": liquid,
            "rho_gas": gas,
            "density_ratio": last["density_ratio"],
            "lambda": lam,
            "tau": cfg.tau,
            "nu": cfg.nu,
            "mu_liquid": mu_c,
            "g_buoy": cfg.g_buoy,
            "u_lbm": u_mean,
            "u_lbm_std": u_std,
            "u_centroid": u_c,
            "u_hr": u_hr,
            "error_raw": u_gate / u_hr - 1.0,
            "Re": re,
            "u_stokes_corrected": u_stokes,
            "error_finite_Re_corrected": u_stokes / u_hr - 1.0,
            "mass_drift": last["mass_drift"],
            "max_u": last["max_u"],
            "steps": cfg.steps,
            "dt_s": time.perf_counter() - t0,
        }


def confinement_extrapolation(records: list[dict[str, object]]) -> dict[str, object]:
    """Linear fit of ``U`` against ``R/H = 2R/L`` and extrapolation to ``R/H = 0``.

    Confinement inflates the drag (``U`` falls with ``R/H``); the intercept is
    the unbounded-domain velocity estimate and the fit residual is its honest
    error bar.
    """
    if len(records) < 2:
        raise ValueError("need at least two confinement levels")
    x = np.array([float(r["confinement"]) for r in records])
    y = np.array([float(r["u_lbm"]) for r in records])
    y_corr = np.array([float(r["u_stokes_corrected"]) for r in records])
    y = np.where(np.isfinite(y), y, np.array([float(r["u_lbm"]) for r in records]))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    slope_c, intercept_c = np.linalg.lstsq(A, y_corr, rcond=None)[0]
    resid = y - (slope * x + intercept)
    u_hr = float(np.mean([float(r["u_hr"]) for r in records]))
    return {
        "confinement_levels": x.tolist(),
        "u_lbm": y.tolist(),
        "slope": float(slope),
        "u_unbounded": float(intercept),
        "u_unbounded_finite_Re_corrected": float(intercept_c),
        "fit_residual_relative": float(np.max(np.abs(resid)) / max(abs(intercept), 1e-12)),
        "u_hr": u_hr,
        "error_unbounded": float(intercept) / u_hr - 1.0,
        "error_unbounded_corrected": float(intercept_c) / u_hr - 1.0,
    }