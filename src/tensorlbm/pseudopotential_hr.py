"""Pseudopotential (Shan-Chen) two-component drop settling vs the
Hadamard-Rybczynski drag law (node IN31-N12, part 2).

Physical problem
----------------
A spherical fluid drop (dynamic viscosity ``mu_d``, density ``rho_d``) settles at
terminal velocity ``U`` through a quiescent fluid (``mu_c``, ``rho_c``) in the
creeping-flow limit.  Hadamard (1911) and Rybczynski (1911):

    F = 2 pi mu_c R U (2 + 3 lam) / (1 + lam),      lam = mu_d / mu_c        (1)
    U = (2/3) drho g R^2 (1 + lam) / (mu_c (2 + 3 lam))                      (2)

with ``lam -> 0`` giving the Rybczynski bubble and ``lam -> inf`` the Stokes
rigid-sphere limit.  (The analytic law itself and its axisymmetric Stokes
verification already live in ``tensorlbm.hadamard_rybczynski``; this module is
the *lattice* side: what a two-component Shan-Chen engine actually produces.)

Why a decoupling helper is needed
---------------------------------
In a single-relaxation-time kernel the dynamic viscosity of a cell is
``mu = rho cs^2 (tau - 1/2)``.  If ``tau`` were shared by both phases, the
viscosity ratio would be slaved to the density ratio, ``lam = rho_d/rho_c``,
and the HR family (parameterised by ``lam`` at fixed density contrast) would be
un-runnable.  The two-component Shan-Chen kernels
(:func:`tensorlbm.multiphase3d.collide_sc_two_component_3d`) relax each component
with its own relaxation time, so ``lam`` and ``rho_d/rho_c`` *are* independent:
:func:`decoupled_relaxation` implements the inversion

    tau_d = 1/2 + lam (rho_c/rho_d) (tau_c - 1/2),                           (3)

and :func:`verify_decoupling` measures the viscosity each phase actually
realises on the lattice (shear-wave decay) instead of asserting (3).

Lattice set-up
--------------
``ny x ny x nz`` D3Q19 grid, fully periodic; component 2 is the drop, component 1
the ambient fluid.  Gravity is applied to the drop component only and a
compensating uniform force to the ambient component so that the total momentum
injection is zero (mandatory in a periodic box: otherwise the whole box
accelerates and no terminal state exists).  The relative velocity is measured as
the drop's drift minus the ambient drift, and the effective buoyancy is measured
from the *applied force field* rather than from nominal densities, so that the
comparison with (2) does not depend on assumptions about the Shan-Chen phase
diagram.

Honest limitation (measured, not asserted): with ``u_eq="self"`` the components
carry independent velocities and the interface transmits tangential momentum only
through the Shan-Chen force, so the internal circulation of the drop is not
resisted by a viscosity contrast in the way (1) assumes; with ``u_eq="mixture"``
the components relax to the barycentric velocity (viscous coupling) but the
kernel then carries a momentum source of order ``(1/tau_2 - 1/tau_1)`` whenever
``tau_1 != tau_2``.  Both conventions are run and reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .hadamard_rybczynski import drag_coefficient_factor, hr_velocity

CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Viscosity ratio / density ratio decoupling
# ---------------------------------------------------------------------------
def viscosity_from_tau(rho: float, tau: float, cs2: float = CS2) -> float:
    """Dynamic viscosity of a BGK cell, ``mu = rho cs^2 (tau - 1/2)``."""
    return float(rho) * cs2 * (float(tau) - 0.5)


def decoupled_relaxation(
    lam: float,
    rho_c: float,
    rho_d: float,
    tau_c: float = 1.0,
    cs2: float = CS2,
) -> Dict[str, float]:
    """Relaxation times realising ``lam = mu_d/mu_c`` at density ratio ``rho_d/rho_c``.

    Returns ``{"tau_c", "tau_d", "lam", "density_ratio", "mu_c", "mu_d"}``.
    Raises ``ValueError`` when the requested combination is not realisable with
    a stable BGK relaxation time (``0.5 < tau < 2``).
    """
    lam = float(lam)
    rho_c = float(rho_c)
    rho_d = float(rho_d)
    if lam < 0.0 or not np.isfinite(lam):
        raise ValueError(f"lam must be finite and >= 0, got {lam}")
    for name, val in (("rho_c", rho_c), ("rho_d", rho_d), ("tau_c", tau_c)):
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {val}")
    if not (0.5 < tau_c < 2.0):
        raise ValueError(f"tau_c must lie in (0.5, 2), got {tau_c}")
    mu_c = viscosity_from_tau(rho_c, tau_c, cs2)
    mu_d = lam * mu_c
    tau_d = 0.5 + mu_d / (rho_d * cs2)
    if not (0.5 < tau_d < 2.0):
        raise ValueError(
            f"lam={lam} at density ratio {rho_d/rho_c} needs tau_d={tau_d:.4f}, "
            "outside the stable BGK window (0.5, 2); choose a different tau_c"
        )
    return {
        "tau_c": float(tau_c),
        "tau_d": float(tau_d),
        "lam": lam,
        "density_ratio": rho_d / rho_c,
        "mu_c": mu_c,
        "mu_d": mu_d,
    }


def verify_decoupling(lam_list: Sequence[float], density_ratios: Sequence[float], tau_c: float = 1.0,
                      steps: int = 600) -> List[Dict[str, float]]:
    """Measure the viscosity ratio each (lam, density ratio) pair realises.

    The viscosity of a *single-component* periodic BGK shear wave is measured by
    ``tensorlbm.two_phase_viscosity.shear_wave_decay`` in each phase separately
    (pure component at its bulk density); the ratio of the two measurements is
    compared with the prescribed ``lam``.  Because the measurement uses the real
    lattice kernel it is a test of (3), not a restatement of it.
    """
    from .two_phase_viscosity import shear_wave_decay

    rows: List[Dict[str, float]] = []
    ny = 64
    for lam in lam_list:
        for dr in density_ratios:
            rho_c = 1.0
            rho_d = float(dr)
            par = decoupled_relaxation(lam, rho_c, rho_d, tau_c=tau_c)
            # kinematic viscosity of each phase, measured on the real D2Q9 BGK
            # kernel (cs^2 is the same for D2Q9/D3Q19, so nu transfers directly);
            # the dynamic viscosity ratio then carries the density weighting.
            nu_c = float(
                shear_wave_decay(np.full(ny, par["tau_c"]), ny=ny, steps=steps)["nu"]
            )
            nu_d = float(
                shear_wave_decay(np.full(ny, par["tau_d"]), ny=ny, steps=steps)["nu"]
            )
            mu_c = rho_c * nu_c
            mu_d = rho_d * nu_d
            rows.append(
                {
                    "lam": float(lam),
                    "density_ratio": float(dr),
                    "tau_c": par["tau_c"],
                    "tau_d": par["tau_d"],
                    "nu_c_measured": nu_c,
                    "nu_d_measured": nu_d,
                    "mu_c_measured": mu_c,
                    "mu_d_measured": mu_d,
                    "lam_measured": mu_d / mu_c,
                    "lam_rel_err": mu_d / mu_c / lam - 1.0,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Lattice drop settling
# ---------------------------------------------------------------------------
@dataclass
class DropSetup:
    """Parameters of one two-component Shan-Chen drop settling run."""

    n: int = 32  # grid edge (cube)
    R: float = 6.0  # drop radius in lattice units
    rho_c: float = 1.0  # ambient bulk density of component 1
    rho_d: float = 1.0  # drop bulk density of component 2
    lam: float = 1.0  # viscosity ratio mu_d/mu_c
    tau_c: float = 1.0
    G_12: float = -1.0
    g: float = 2.0e-4  # gravity acceleration applied to the drop component
    steps: int = 4000
    interface: float = 1.5  # diffuse interface half-width
    background: float = 0.05  # residual density of each component (numerical)
    u_eq: str = "self"
    sample_every: int = 20
    dtype: str = "float64"

    def relaxations(self) -> Dict[str, float]:
        return decoupled_relaxation(self.lam, self.rho_c, self.rho_d, tau_c=self.tau_c)


def _initial_fields(s: DropSetup, device, dtype):
    """Diffuse spherical drop of component 2 in component 1."""
    import torch

    n = s.n
    z = torch.arange(n, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, z, z, indexing="ij")
    c = 0.5 * (n - 1)
    r = torch.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2)
    phi = torch.sigmoid((s.R + 0.5 * s.interface - r) / s.interface)  # 1 inside drop
    # Each component keeps a small background density everywhere: the Shan-Chen
    # kernel is an ideal-gas EOS, and tau F/rho diverges in a component that is
    # expelled to zero density (measured: the run blows up within 20 steps with a
    # vanishing background; see the report's "pitfalls" section).
    bg = s.background
    rho2 = s.rho_d * phi + bg
    rho1 = s.rho_c * (1.0 - phi) + bg
    zero = torch.zeros_like(rho1)
    from .d3q19 import equilibrium3d

    f1 = equilibrium3d(rho1, zero, zero, zero, device=device)
    f2 = equilibrium3d(rho2, zero, zero, zero, device=device)
    return f1, f2, phi


def run_settling(s: DropSetup, device: str = "cpu", verbose: bool = False) -> Dict[str, object]:
    """Run one drop settling experiment and return the measured quantities.

    Returns a dict with the terminal relative velocity, the HR prediction at the
    *measured* radius and effective buoyancy, the viscosity ratio realised by the
    relaxation times, and the time histories (so the convergence can be plotted).
    """
    import torch

    from .d3q19 import macroscopic3d
    from .solver3d import stream3d
    from .two_phase_viscosity import shear_wave_decay

    dtype = getattr(torch, s.dtype)
    device = torch.device(device)
    par = s.relaxations()
    tau1, tau2 = par["tau_c"], par["tau_d"]

    f1, f2, phi0 = _initial_fields(s, device, dtype)
    # force field: gravity on the drop component, momentum-neutral compensation
    phi_mean = float(phi0.mean().item())
    gz1 = s.g * phi_mean / (1.0 - phi_mean)  # opposite sign, momentum neutral
    hist: List[Dict[str, float]] = []
    t0 = __import__("time").time()

    def _macros(f):
        rho, ux, uy, uz = macroscopic3d(f)
        return rho, ux, uy, uz

    with torch.no_grad():
        for step in range(1, int(s.steps) + 1):
            f1, f2 = collide_two_component_with_gravity(
                f1,
                f2,
                G_12=s.G_12,
                tau1=tau1,
                tau2=tau2,
                gz1=gz1,  # ambient component: compensating (upward) force
                gz2=-s.g,  # drop component: gravity (downward in -z)
                u_eq=s.u_eq,
            )
            f1 = stream3d(f1)
            f2 = stream3d(f2)
            if step % int(s.sample_every) == 0 or step == int(s.steps):
                rec = _diagnostics(f1, f2, phi0, s, step, shear_wave_decay, device, dtype)
                if not np.isfinite(rec["U_rel"]) or not np.isfinite(rec["R_eff"]):
                    raise FloatingPointError(f"drop run diverged at step {step}")
                hist.append(rec)
                if verbose and (step % (20 * int(s.sample_every)) == 0):
                    h = hist[-1]
                    print(
                        f"    step {step:6d} U_rel={h['U_rel']:+.4e} rho2_in={h['rho2_in']:.3f} "
                        f"rho1_out={h['rho1_out']:.3f}",
                        flush=True,
                    )

    tail = hist[len(hist) // 2 :]
    U_rel = float(np.mean([h["U_rel"] for h in tail]))
    U_spread = float(np.std([h["U_rel"] for h in tail]))
    R_eff = float(np.mean([h["R_eff"] for h in tail]))
    rho2_in = float(np.mean([h["rho2_in"] for h in tail]))
    rho1_out = float(np.mean([h["rho1_out"] for h in tail]))
    drho_eff = float(np.mean([h["drho_eff"] for h in tail]))

    mu_c = viscosity_from_tau(rho1_out, tau1)
    mu_d = viscosity_from_tau(rho2_in, tau2)
    lam_measured = mu_d / mu_c
    U_hr = hr_velocity(drho_eff, s.g, R_eff, mu_c, lam_measured)
    return {
        "setup": {
            "n": s.n,
            "R0": s.R,
            "rho_c": s.rho_c,
            "rho_d": s.rho_d,
            "lam_nominal": s.lam,
            "tau_c": tau1,
            "tau_d": tau2,
            "G_12": s.G_12,
            "g": s.g,
            "u_eq": s.u_eq,
            "steps": s.steps,
            "interface": s.interface,
        },
        "R_eff": R_eff,
        "rho2_in": rho2_in,
        "rho1_out": rho1_out,
        "density_ratio_measured": rho2_in / rho1_out,
        "mu_c": mu_c,
        "mu_d": mu_d,
        "lam_measured": lam_measured,
        "drho_eff": drho_eff,
        "U_rel": U_rel,
        "U_rel_spread": U_spread,
        "U_hr": U_hr,
        "rel_err_vs_hr": U_rel / U_hr - 1.0,
        "drag_factor": drag_coefficient_factor(lam_measured),
        "history": hist,
        "elapsed_s": round(__import__("time").time() - t0, 1),
    }


def collide_two_component_with_gravity(
    f1,
    f2,
    G_12: float,
    tau1: float,
    tau2: float,
    gz1: float,
    gz2: float,
    u_eq: str = "self",
):
    """Shan-Chen two-component D3Q19 collision with *per-component* body force.

    ``multiphase3d.collide_sc_two_component_3d`` accepts a single ``gz`` shared by
    both components, which cannot express "gravity on the drop only"
    (a buoyancy problem needs that: the ambient phase must receive the
    compensating force).  This function mirrors the library convention exactly
    (interaction force from ``sc_two_component_force_3d``, velocity-shift forcing
    ``u_eq = u + tau F/rho``) and adds ``rho_sigma gz_sigma`` per component;
    ``test_pseudopotential_hr.py`` asserts that it is identical to the library
    kernel when the two gravities coincide.
    """
    import torch

    from .d3q19 import equilibrium3d, macroscopic3d
    from .multiphase3d import sc_two_component_force_3d

    rho1, ux1, uy1, uz1 = macroscopic3d(f1)
    rho2, ux2, uy2, uz2 = macroscopic3d(f2)
    Fx1, Fy1, Fz1, Fx2, Fy2, Fz2 = sc_two_component_force_3d(rho1, rho2, G_12)
    Fz1 = Fz1 + rho1 * gz1
    Fz2 = Fz2 + rho2 * gz2

    rho1_s = torch.clamp(rho1, min=1e-12)
    rho2_s = torch.clamp(rho2, min=1e-12)
    if u_eq == "self":
        feq1 = equilibrium3d(
            rho1,
            ux1 + tau1 * Fx1 / rho1_s,
            uy1 + tau1 * Fy1 / rho1_s,
            uz1 + tau1 * Fz1 / rho1_s,
            device=f1.device,
        )
        feq2 = equilibrium3d(
            rho2,
            ux2 + tau2 * Fx2 / rho2_s,
            uy2 + tau2 * Fy2 / rho2_s,
            uz2 + tau2 * Fz2 / rho2_s,
            device=f2.device,
        )
    elif u_eq == "mixture":
        rt = rho1_s + rho2_s
        mx = (rho1_s * ux1 + rho2_s * ux2) / rt
        my = (rho1_s * uy1 + rho2_s * uy2) / rt
        mz = (rho1_s * uz1 + rho2_s * uz2) / rt
        feq1 = equilibrium3d(
            rho1, mx + tau1 * Fx1 / rho1_s, my + tau1 * Fy1 / rho1_s, mz + tau1 * Fz1 / rho1_s,
            device=f1.device,
        )
        feq2 = equilibrium3d(
            rho2, mx + tau2 * Fx2 / rho2_s, my + tau2 * Fy2 / rho2_s, mz + tau2 * Fz2 / rho2_s,
            device=f2.device,
        )
    else:
        raise ValueError(f"u_eq must be 'self' or 'mixture', got {u_eq!r}")

    return f1 - (f1 - feq1) / tau1, f2 - (f2 - feq2) / tau2


def _diagnostics(f1, f2, phi0, s: DropSetup, step: int, shear_wave_decay, device, dtype) -> Dict[str, float]:
    """Instantaneous drop kinematics, geometry and the effective buoyancy."""
    import torch

    from .d3q19 import macroscopic3d

    rho1, ux1, uy1, uz1 = macroscopic3d(f1)
    rho2, ux2, uy2, uz2 = macroscopic3d(f2)
    # excess of component 2 over its ambient value -> drop weight field
    rho2_out = max(float(rho2.min().item()), 1e-9)
    w = torch.clamp(rho2 - rho2_out, min=0.0)
    W = float(w.sum().item())
    uz_drop = float((w * uz2).sum().item() / max(W, 1e-30))
    ux_drop = float((w * ux2).sum().item() / max(W, 1e-30))
    uy_drop = float((w * uy2).sum().item() / max(W, 1e-30))
    rho1_s = torch.clamp(rho1, min=1e-12)
    W1 = float(rho1_s.sum().item())
    uz_amb = float((rho1_s * uz1).sum().item() / W1)
    # drop volume from the excess density -> equivalent radius
    vol = W / max(s.rho_d - rho2_out, 1e-9)
    R_eff = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
    # effective buoyancy from the applied force field:
    # net force on the drop region minus the far-field force density times its volume
    # gravity is applied to component 2 (+g downward in -z) and to component 1 (upwards)
    phi_mean = float(phi0.mean().item())
    gz1 = s.g * phi_mean / (1.0 - phi_mean)
    fz = (rho2 * (-s.g)) + (rho1 * gz1)  # total body-force density
    far = float(fz.mean().item())
    net_force = float(((fz - far) * (w > 0).to(dtype)).sum().item())
    drho_eff = net_force / max(vol, 1e-30) / s.g
    return {
        "step": int(step),
        "U_rel": uz_drop - uz_amb,
        "U_drop": uz_drop,
        "U_amb": uz_amb,
        "U_x_drop": ux_drop,
        "U_y_drop": uy_drop,
        "R_eff": R_eff,
        "rho2_in": float((w * rho2).sum().item() / max(W, 1e-30)),
        "rho1_out": float(rho1_s.min().item()),
        "drho_eff": drho_eff,
        "div_rms": float(
            torch.sqrt(
                torch.mean(
                    (torch.gradient(ux1, dim=2)[0] + torch.gradient(uy1, dim=1)[0]
                     + torch.gradient(uz1, dim=0)[0]) ** 2
                )
            ).item()
        ),
    }