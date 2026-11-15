"""Free-energy pressure-tensor (Korteweg stress) body force for the binary-fluid
free-energy LBM — cross-validation port of ludwig-cf/ludwig
``FE_FORCE_METHOD_STRESS_DIVERGENCE``.

Physics / conventions (identical to ludwig ``symmetric.c``)
-----------------------------------------------------------
Bulk free energy of the symmetric (double-well) binary fluid::

    f(phi) = a/2 * phi^2 + b/4 * phi^4 + kappa/2 * |grad phi|^2      (a < 0, b > 0)

Chemical potential (ludwig ``fe_symm_mu``)::

    mu = a*phi + b*phi^3 - kappa * laplacian(phi)

Bulk (thermodynamic) pressure ``p0 = phi f'(phi) - f``
(ludwig ``fe_symm_str``: ``0.5*a*phi^2 + 0.75*b*phi^4``), generalised pressure
and pressure tensor::

    d   = p0 - kappa*phi*laplacian(phi) - kappa/2*|grad phi|^2
    P_ab = d*delta_ab + kappa * d_a phi * d_b phi

The capillary body force is the divergence of that tensor,

    F_a = -d_b P_ab

which is analytically equal to the Korteweg /chemical-potential form
``F = -phi*grad(mu)`` that :func:`tensorlbm.multiphase.free_energy_step`
currently hard-codes; the two differ only by the discretisation.

Interfacial tension of this model (ludwig ``fe_symm_interfacial_tension``
evaluates ``sqrt(-8*kappa*a^3/(9*b^2))``)::

    sigma = (2*sqrt(2)/3) * sqrt(kappa * A^3) / b,      A = -a

and the equilibrium interface width ``xi = sqrt(2*kappa/A)``.

All routines are 2-D periodic (the TensorLBM free-energy solver is 2-D); the 3-D
port only changes the gradient stencils and the divergence loop.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "FreeEnergyParams",
    "GRAD_SCHEMES",
    "grad",
    "laplacian",
    "chemical_potential",
    "bulk_pressure",
    "pressure_tensor",
    "stress_divergence_force",
    "grad_mu_force",
    "surface_tension_analytic",
    "interface_width",
    "planar_profile",
    "droplet_profile",
    "DEFAULT_RELAX_TOL",
    "relax_ch",
    "measure_planar_interface",
    "surface_tension_pressure_integral",
    "surface_tension_energy_integral",
]

GRAD_SCHEMES = ("5pt", "iso9")


class FreeEnergyParams:
    """Free-energy parameters in ludwig's sign convention (``a < 0``)."""

    def __init__(self, A: float = 1.0, B: float = 1.0, kappa: float = 2.0):
        if A <= 0.0 or B <= 0.0 or kappa <= 0.0:
            raise ValueError("A, B, kappa must be positive (a = -A < 0)")
        self.A = float(A)
        self.B = float(B)
        self.kappa = float(kappa)

    @property
    def a(self) -> float:  # ludwig param[0]
        return -self.A

    @property
    def b(self) -> float:  # ludwig param[1]
        return self.B

    def __repr__(self) -> str:  # pragma: no cover
        return f"FreeEnergyParams(A={self.A}, B={self.B}, kappa={self.kappa})"


def _roll(field: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    return torch.roll(torch.roll(field, -dy, dims=-2), -dx, dims=-1)


def grad(field: torch.Tensor, scheme: str = "iso9") -> tuple[torch.Tensor, torch.Tensor]:
    """Periodic gradient of a 2-D field (unit lattice spacing).

    ``"5pt"``   central differences, O(h^2), anisotropic truncation.
    ``"iso9"``  isotropic 9-point (Tomita) gradient, O(h^2) with the leading
                anisotropic term cancelled -- the default fluid gradient in
                ludwig (``gradient_2d_tomita_fluid``).
    """
    if scheme == "5pt":
        gx = 0.5 * (_roll(field, 0, -1) - _roll(field, 0, 1))
        gy = 0.5 * (_roll(field, -1, 0) - _roll(field, 1, 0))
        return gx, gy
    if scheme == "iso9":
        # Tomita / isotropic: 4/12 on axis, 1/12 on diagonals (sum = 1 along x)
        gx = (
            4.0 / 12.0 * (_roll(field, 0, -1) - _roll(field, 0, 1))
            + 1.0 / 12.0 * (_roll(field, -1, -1) - _roll(field, -1, 1))
            + 1.0 / 12.0 * (_roll(field, 1, -1) - _roll(field, 1, 1))
        )
        gy = (
            4.0 / 12.0 * (_roll(field, -1, 0) - _roll(field, 1, 0))
            + 1.0 / 12.0 * (_roll(field, -1, -1) - _roll(field, 1, -1))
            + 1.0 / 12.0 * (_roll(field, -1, 1) - _roll(field, 1, 1))
        )
        return gx, gy
    raise ValueError(f"unknown gradient scheme {scheme!r}; expected one of {GRAD_SCHEMES}")


def laplacian(field: torch.Tensor) -> torch.Tensor:
    """5-point periodic Laplacian (ludwig ``field->delsq`` for the 5-pt stencil)."""
    return (
        _roll(field, 0, -1) + _roll(field, 0, 1)
        + _roll(field, -1, 0) + _roll(field, 1, 0)
        - 4.0 * field
    )


def chemical_potential(phi: torch.Tensor, p: FreeEnergyParams) -> torch.Tensor:
    """``mu = a*phi + b*phi^3 - kappa*laplacian(phi)`` (ludwig fe_symm_mu)."""
    return p.a * phi + p.b * phi ** 3 - p.kappa * laplacian(phi)


def bulk_pressure(phi: torch.Tensor, p: FreeEnergyParams) -> torch.Tensor:
    """``p0 = phi f'(phi) - f = a/2 phi^2 + 3b/4 phi^4`` (ludwig fe_symm_str)."""
    return 0.5 * p.a * phi ** 2 + 0.75 * p.b * phi ** 4


def pressure_tensor(
    phi: torch.Tensor, p: FreeEnergyParams, scheme: str = "iso9"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Free-energy pressure tensor ``P_ab`` (ludwig ``fe_symm_str``).

    Returns ``(Pxx, Pyy, Pxy)``; ``P_ab = d*delta_ab + kappa d_a phi d_b phi``
    with ``d = p0 - kappa*phi*laplacian(phi) - kappa/2*|grad phi|^2``.
    """
    gx, gy = grad(phi, scheme)
    d = (
        bulk_pressure(phi, p)
        - p.kappa * phi * laplacian(phi)
        - 0.5 * p.kappa * (gx * gx + gy * gy)
    )
    Pxx = d + p.kappa * gx * gx
    Pyy = d + p.kappa * gy * gy
    Pxy = p.kappa * gx * gy
    return Pxx, Pyy, Pxy


def stress_divergence_force(
    phi: torch.Tensor, p: FreeEnergyParams, scheme: str = "iso9"
) -> tuple[torch.Tensor, torch.Tensor]:
    """``F = -div P`` with central differences on the discrete tensor.

    This is the ludwig ``FE_FORCE_METHOD_STRESS_DIVERGENCE`` body force: the
    tensor is evaluated at every site, then differenced once (the O(h^2)
    wide-stencil divergence of ``phi_force_stress.c``).
    """
    Pxx, Pyy, Pxy = pressure_tensor(phi, p, scheme)
    Fx = -0.5 * (_roll(Pxx, 0, -1) - _roll(Pxx, 0, 1)) - 0.5 * (
        _roll(Pxy, -1, 0) - _roll(Pxy, 1, 0)
    )
    Fy = -0.5 * (_roll(Pxy, 0, -1) - _roll(Pxy, 0, 1)) - 0.5 * (
        _roll(Pyy, -1, 0) - _roll(Pyy, 1, 0)
    )
    return Fx, Fy


def grad_mu_force(
    phi: torch.Tensor, p: FreeEnergyParams, scheme: str = "iso9"
) -> tuple[torch.Tensor, torch.Tensor]:
    """``F = -phi*grad(mu)`` — TensorLBM's existing free-energy body force."""
    mu = chemical_potential(phi, p)
    gmx, gmy = grad(mu, scheme)
    return -phi * gmx, -phi * gmy


def surface_tension_analytic(p: FreeEnergyParams) -> float:
    """``sigma = sqrt(-8*kappa*a^3/(9*b^2)) = (2 sqrt2/3) sqrt(kappa A^3)/b``."""
    return math.sqrt(-8.0 * p.kappa * p.a ** 3 / (9.0 * p.b ** 2))


def interface_width(p: FreeEnergyParams) -> float:
    """Equilibrium interface thickness parameter ``xi = sqrt(2*kappa/A)``."""
    return math.sqrt(2.0 * p.kappa / p.A)


def planar_profile(ny: int, nx: int, p: FreeEnergyParams, x0: float | None = None) -> torch.Tensor:
    """Equilibrium planar interface ``phi = phi0 tanh((x-x0)/xi)``."""
    x0 = 0.5 * nx if x0 is None else x0
    phi0 = math.sqrt(p.A / p.B)
    xi = interface_width(p)
    x = torch.arange(nx, dtype=torch.float64)
    prof = phi0 * torch.tanh((x - x0) / xi)
    return prof.unsqueeze(0).expand(ny, nx).contiguous()


def droplet_profile(
    ny: int, nx: int, p: FreeEnergyParams, radius: float | None = None
) -> torch.Tensor:
    """Equilibrium droplet ``phi = -phi0 tanh((r-R)/xi)`` (phi>0 outside)."""
    R = (0.25 * min(ny, nx) if radius is None else radius)
    phi0 = math.sqrt(p.A / p.B)
    xi = interface_width(p)
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    r = torch.sqrt((y - 0.5 * ny) ** 2 + (x - 0.5 * nx) ** 2)
    return phi0 * torch.tanh((r - R) / xi)


# ---------------------------------------------------------------------------
# Convergence-controlled Cahn-Hilliard relaxation + interface measurement
# ---------------------------------------------------------------------------
# Why these helpers exist: the first cross-validation round of
# XVAL-LBM-R4-COEXISTENCE (docs/XVAL-LBM-R4-COEXISTENCE.md) measured the
# surface tension of the transplanted functional after a *fixed* number of
# gradient-flow steps and reported the thin-interface case (kappa = 0.5) as a
# 15-18 % "discretisation limit".  Re-running the same case with a convergence
# criterion shows the deviation is dominated by an unfinished relaxation
# (mu is not yet uniform), not by the lattice: the sigma error keeps falling
# 18 % -> 12 % between 3 k and 24 k steps.  Measuring a *relaxed* interface
# therefore needs an explicit convergence test, and the tension itself should
# be cross-checked along two independent routes (Korteweg pressure integral vs
# the thermodynamic excess free energy) so that an implementation defect and a
# lattice-resolution effect cannot be confused.

DEFAULT_RELAX_TOL = 1e-3
"""Relative drift of the discrete surface tension below which the CH
relaxation is declared converged (default 1e-3 = 0.1 %)."""


def surface_tension_pressure_integral(
    phi: torch.Tensor, p: FreeEnergyParams, scheme: str = "iso9", n_interfaces: int = 1
) -> float:
    """``sigma = (1/n) * int (P_NN - P_TT) dy`` — the Korteweg (mechanical) route.

    The integrand of a planar equilibrium interface is ``kappa*(dphi/dy)^2``, so
    the column integral equals ``n_interfaces * sigma``; the result is divided by
    ``n_interfaces`` to give a single-interface tension.  Pass ``scheme`` exactly
    as the relaxation used it: mixed stencils (relax with one, measure with the
    other) break the discrete stress balance and bias sigma.
    """
    if n_interfaces < 1:
        raise ValueError("n_interfaces must be >= 1")
    Pxx, Pyy, _ = pressure_tensor(phi, p, scheme=scheme)
    return float((Pyy - Pxx).mean(dim=1).sum()) / float(n_interfaces)


def surface_tension_energy_integral(
    phi: torch.Tensor, p: FreeEnergyParams, scheme: str = "iso9", n_interfaces: int = 1
) -> float:
    """``sigma = (1/n) * int [domega + kappa/2*(dphi/dy)^2] dy`` — thermodynamic route.

    ``domega = f(phi) - f(+-phi0)`` is the excess bulk free energy of the double
    well ``f = a/2 phi^2 + b/4 phi^4`` (equal at both bulk phases here), and the
    gradient term is the Korteweg energy density.  This route shares no
    intermediate quantity with :func:`surface_tension_pressure_integral`, so
    agreement of the two is evidence for the implementation, independently of
    the lattice resolution.
    """
    if n_interfaces < 1:
        raise ValueError("n_interfaces must be >= 1")
    col = phi if phi.dim() == 1 else phi[:, 0]
    _, gy = grad(col.view(-1, 1), scheme)
    gy = gy[:, 0]
    omega = 0.5 * p.a * col**2 + 0.25 * p.b * col**4
    phi0_sq = p.A / p.B
    omega_bulk = 0.5 * p.a * phi0_sq + 0.25 * p.b * phi0_sq**2
    return float((omega - omega_bulk + 0.5 * p.kappa * gy * gy).sum()) / float(n_interfaces)


def relax_ch(
    phi: torch.Tensor,
    p: FreeEnergyParams,
    dt: float = 0.05,
    mobility: float = 1.0,
    steps: int = 40000,
    tol: float = DEFAULT_RELAX_TOL,
    mu_tol: float = 2e-3,
    check_every: int = 500,
    n_interfaces: int = 2,
    clip: float | None = 1.3,
    scheme: str = "iso9",
) -> dict:
    """Cahn-Hilliard gradient flow ``dphi/dt = M lap(mu)`` with a convergence test.

    Evolves the periodic order-parameter field until the *discrete* surface
    tension stops drifting by more than ``tol`` (relative) between two
    checkpoints, or until ``steps`` is exhausted.  The returned ``converged``
    flag is the honest statement about the state that was measured — a caller
    must not report a sigma that comes from an unconverged field as an
    equilibrium result.

    Args:
        phi: Initial field ``(ny, nx)`` (or ``(ny,)``), double precision.
        p: Free-energy parameters.
        dt: Explicit Euler step.  Stability needs ``dt*M*kappa <~ h^2/4``;
            violations show up as a clamped (|phi| = clip) plateau.
        mobility: CH mobility ``M``.
        steps: Maximum number of steps.
        tol: Relative surface-tension drift accepted as converged.
        mu_tol: Acceptance bound on ``std(mu)/A``.  The CH gradient flow
            approaches equilibrium algebraically, so the sigma drift alone can
            fall below ``tol`` while ``mu`` is still visibly non-uniform; both
            conditions are required.
        check_every: Checkpoint interval (steps).
        n_interfaces: Interfaces per column (2 for a periodic strip, 1 otherwise).
        clip: Optional |phi| clamp used as a last-resort stability guard, or
            ``None``.  A large share of cells sitting on the clamp invalidates
            the run (reported through ``clipped_fraction``).
        scheme: Gradient stencil for the tension check (must match the caller's
            measurement scheme).

    Returns:
        ``{"phi", "converged", "steps", "sigma", "sigma_history", "drift",
        "mu_std_over_A", "clipped_fraction"}``.
    """
    if phi.dim() == 1:
        phi = phi.view(-1, 1).clone()
    if phi.dim() != 2:
        raise ValueError("phi must be 1-D or 2-D")
    phi = phi.clone()
    phi0 = math.sqrt(p.A / p.B)
    sigma_prev: float | None = None
    history: list[dict] = []
    drift = float("inf")
    converged = False
    step = 0
    for step in range(1, steps + 1):
        mu = chemical_potential(phi, p)
        phi = phi + dt * mobility * laplacian(mu)
        if clip is not None:
            phi.clamp_(-clip * phi0, clip * phi0)
        if step % check_every == 0 or step == steps:
            sigma = surface_tension_pressure_integral(
                phi, p, scheme=scheme, n_interfaces=n_interfaces
            )
            mu_std = float(chemical_potential(phi, p).std() / p.A)
            if sigma_prev is not None and sigma != 0.0:
                drift = abs(sigma - sigma_prev) / abs(sigma)
            history.append({"step": step, "sigma": sigma, "mu_std_over_A": mu_std,
                            "drift": drift})
            sigma_prev = sigma
            if drift < tol and mu_std < mu_tol:
                converged = True
                break
    clipped = 0.0
    if clip is not None:
        clipped = float((phi.abs() >= 0.999 * clip * phi0).double().mean())
    return {
        "phi": phi,
        "converged": converged,
        "steps": step,
        "sigma": sigma_prev,
        "sigma_history": history,
        "drift": drift,
        "mu_std_over_A": history[-1]["mu_std_over_A"] if history else float("nan"),
        "clipped_fraction": clipped,
    }


def _crossing(seg: torch.Tensor, lo: int, level: float) -> float:
    """First interpolated grid coordinate where ``seg`` crosses ``level``."""
    s = seg.tolist()
    for i in range(len(s) - 1):
        if (s[i] - level) * (s[i + 1] - level) <= 0:
            denom = s[i] - s[i + 1]
            w = (s[i] - level) / (denom if denom != 0 else 1e-30)
            return lo + i + w
    return float("nan")


def measure_planar_interface(
    phi: torch.Tensor,
    p: FreeEnergyParams,
    scheme: str = "iso9",
    n_interfaces: int = 2,
    window: int | None = None,
) -> dict:
    """Measure xi and sigma of a relaxed planar interface, two independent routes.

    The interface is located at the steepest local gradient, then a window of
    +/-``window`` cells around it is fitted with ``phi = c tanh((y0 - y)/xi)``.
    Working on a *single* window (instead of a superposition of both interfaces
    of a periodic strip) removes the bias the round-1 report carried: there the
    tanh model was fitted to the full column containing two interfaces.

    ``sigma`` is integrated along two routes that share no intermediate quantity
    — the Korteweg pressure-tensor integral and the thermodynamic excess free
    energy — so their spread is evidence about the implementation, while the
    deviation from the analytic value mixes implementation *and* lattice
    resolution and must be read together with ``xi/h``.

    Returns ``{"xi_10_90", "xi_lsq", "xi_analytic", "sigma_pressure",
    "sigma_energy", "sigma_analytic", "sigma_route_spread", "max_abs_divP",
    "mu_std_over_A", "phi0_measured", "interface_at"}``.
    """
    if phi.dim() == 1:
        phi = phi.view(-1, 1)
    col = phi[:, 0]
    ny = col.shape[0]
    phi0 = math.sqrt(p.A / p.B)
    # --- locate the interface: steepest periodic central difference
    d = 0.5 * (torch.roll(col, -1) - torch.roll(col, 1))
    idx = int(torch.argmax(d.abs()))
    w = max(8, ny // 8) if window is None else int(window)
    # unwrap around idx so the window may straddle the periodic boundary
    unrolled = torch.roll(col, w - idx)
    seg = unrolled[: 2 * w + 1]
    yy = torch.arange(seg.shape[0], dtype=torch.float64)
    levels = (float(seg.min()), float(seg.max()))
    span = levels[1] - levels[0]
    y90 = _crossing(seg, 0, levels[0] + 0.9 * span)
    y10 = _crossing(seg, 0, levels[0] + 0.1 * span)
    xi_1090 = abs(y10 - y90) / (2.0 * math.atanh(0.8))
    y0 = _crossing(seg, 0, 0.5 * (levels[0] + levels[1]))
    sign = -1.0 if float(seg[0]) > float(seg[-1]) else 1.0
    amp = sign * 0.5 * span
    best = (float("inf"), float("nan"))
    for k in range(400):
        xi = 0.20 + 0.01 * k
        model = amp * torch.tanh((yy - y0) / xi)
        res_i = float(((model - seg) ** 2).sum())
        if res_i < best[0]:
            best = (res_i, xi)
    sigma_p = surface_tension_pressure_integral(phi, p, scheme=scheme, n_interfaces=n_interfaces)
    sigma_e = surface_tension_energy_integral(phi, p, scheme=scheme, n_interfaces=n_interfaces)
    Fx, Fy = stress_divergence_force(phi, p, scheme=scheme)
    mu = chemical_potential(phi, p)
    sigma_a = surface_tension_analytic(p)
    return {
        "xi_10_90": float(xi_1090),
        "xi_lsq": float(best[1]),
        "xi_analytic": float(interface_width(p)),
        "xi_relative_err": float(abs(best[1] - interface_width(p)) / interface_width(p)),
        "sigma_pressure": float(sigma_p),
        "sigma_energy": float(sigma_e),
        "sigma_analytic": float(sigma_a),
        "sigma_route_spread": float(abs(sigma_p - sigma_e) / abs(sigma_a)),
        "max_abs_divP": float(torch.sqrt(Fx**2 + Fy**2).max()),
        "mu_std_over_A": float(mu.std() / p.A),
        "phi0_measured": float(amp),
        "interface_at": int(idx),
    }
