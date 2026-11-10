"""Mechanical pressure tensor for the Shan-Chen two-component (SCMC) model.

Why this module exists
----------------------
``tensorlbm.multiphase_benchmarks.run_static_droplet`` measures the droplet
pressure jump from the **ideal-gas density proxy** ``p* = cs² ρ`` and labels the
result ``laplace_physical_validation: NOT_VALIDATED``.  The label is honest: the
pseudopotential (Shan-Chen) interaction contributes a *non-ideal* pressure term
that lives inside the diffuse interface, so ``cs² ρ`` is not the mechanical
pressure and the Young-Laplace law ``ΔP = σ/R`` cannot be tested with it.

This module supplies the missing piece: the **discrete mechanical pressure
tensor** of the two-component Shan-Chen model in the form

    P_ab(x) = cs² (ρ₁ + ρ₂) δ_ab
              + (G/2) Σ_i w_i c_ia c_ib [ρ₁(x) ρ₂(x + c_i) + ρ₂(x) ρ₁(x + c_i)]

(the symmetric two-point form of the Shan-Chen pressure tensor, Shan 2006;
Landau-Lifshitz sign convention: a *positive* P means compression, so the
surface tension follows from ``σ = ∫ (P_NN − P_TT) dn`` across a *flat*
interface and, for a circular droplet, from the exact mechanical relation
``ΔP = σ/R``).

The pairing orientation (forward gather ``x + c_i`` versus backward gather
``x − c_i``) changes the tensor by O(h²) but **not** its isotropic part, so the
module keeps both and lets the physics decide: in static equilibrium the normal
component ``P_NN`` of a flat interface must be constant across the interface
(``∂_n P_NN = 0``) and it is exactly this condition that discriminates a correct
pressure formula from ``cs²ρ``.  :func:`flat_interface_pressure_profile` returns
the profile and its flatness residual for every candidate.

Everything is pure ``torch`` + ``numpy`` on the existing D2Q9 kernels of
:mod:`tensorlbm.multiphase` / :mod:`tensorlbm.solver`; no new solver is
introduced and no existing benchmark output is modified.

References
----------
Shan, X. (2006). Analysis and reduction of the spurious current in a class of
    multiphase lattice Boltzmann models. Phys. Rev. E 73, 047701.
Shan, X. & Chen, H. (1993). Lattice Boltzmann model for simulating flows with
    multiple phases and components. Phys. Rev. E 47, 1815.
Young (1805) Phil. Trans. R. Soc. 95, 65; Laplace (1806) Mécanique Céleste.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .d2q9 import C, W, equilibrium, macroscopic
from .multiphase import collide_sc_two_component
from .solver import stream

__all__ = [
    "CS2",
    "SCMCPressureConfig",
    "scmc_pressure_tensor",
    "mechanical_pressure",
    "surface_tension_flat",
    "flat_interface_pressure_profile",
    "simulate_flat_interface",
    "simulate_scmc_droplet",
    "laplace_sweep",
]

CS2 = 1.0 / 3.0  # lattice speed of sound squared for D2Q9

PAIRINGS = ("fwd", "bwd", "sym", "isotropic")


# ---------------------------------------------------------------------------
# Pressure tensor
# ---------------------------------------------------------------------------


def _shift(field: torch.Tensor, i: int) -> torch.Tensor:
    """Periodic shift of a 2-D field by lattice vector ``c_i`` (``x + c_i``)."""
    dy = int(C[i, 1].item())
    dx = int(C[i, 0].item())
    return torch.roll(torch.roll(field, -dy, dims=0), -dx, dims=1)


def _pair_term(rho1: torch.Tensor, rho2: torch.Tensor, pairing: str) -> torch.Tensor:
    """Return ``(9, ny, nx)`` pair products ``ρ₁ρ₂ + ρ₂ρ₁`` on the stencil."""
    fwd = rho1.unsqueeze(0) * torch.stack([_shift(rho2, i) for i in range(9)])
    fwd = fwd + rho2.unsqueeze(0) * torch.stack([_shift(rho1, i) for i in range(9)])
    if pairing == "fwd":
        return fwd
    bwd = rho1.unsqueeze(0) * torch.stack([_shift(rho2, -i if i else 0) for i in range(9)])
    bwd = bwd + rho2.unsqueeze(0) * torch.stack([_shift(rho1, -i if i else 0) for i in range(9)])
    if pairing == "bwd":
        return bwd
    if pairing == "sym":
        return 0.5 * (fwd + bwd)
    raise ValueError(f"unknown pairing {pairing!r}; expected one of {PAIRINGS}")


def scmc_pressure_tensor(
    f1: torch.Tensor,
    f2: torch.Tensor,
    G12: float,
    pairing: str = "sym",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Discrete mechanical pressure tensor of the SCMC model.

    Args:
        f1, f2: component distributions, shape ``(9, ny, nx)``.
        G12:    Shan-Chen coupling (``< 0`` ↔ repulsion ↔ phase separation).
        pairing: ``"fwd"``, ``"bwd"``, ``"sym"`` (two-point Shan-Chen form) or
            ``"isotropic"`` (the pure gradient part ``cs²ρ + G cs² ρ₁ρ₂``,
            included as the *negative control*: it discards the anisotropic
            part of the interaction stress).

    Returns:
        ``(Pxx, Pyy, Pxy)``, each ``(ny, nx)``, Landau-Lifshitz convention
        (positive = compression).
    """
    rho1, _, _ = macroscopic(f1)
    rho2, _, _ = macroscopic(f2)
    rho_tot = rho1 + rho2
    device = f1.device

    if pairing == "isotropic":
        p_iso = CS2 * rho_tot + G12 * CS2 * rho1 * rho2
        zero = torch.zeros_like(p_iso)
        return p_iso, p_iso.clone(), zero

    q = _pair_term(rho1, rho2, pairing)  # (9, ny, nx)
    c = C.to(device).float()
    w = W.to(device).float()
    cxx = (c[:, 0] * c[:, 0]).view(9, 1, 1)
    cyy = (c[:, 1] * c[:, 1]).view(9, 1, 1)
    cxy = (c[:, 0] * c[:, 1]).view(9, 1, 1)
    w3 = w.view(9, 1, 1)

    pref = 0.5 * G12
    Pxx = CS2 * rho_tot + pref * (w3 * cxx * q).sum(0)
    Pyy = CS2 * rho_tot + pref * (w3 * cyy * q).sum(0)
    Pxy = pref * (w3 * cxy * q).sum(0)
    return Pxx, Pyy, Pxy


def mechanical_pressure(
    f1: torch.Tensor,
    f2: torch.Tensor,
    G12: float,
    pairing: str = "sym",
) -> torch.Tensor:
    """Isotropic part ``(Pxx + Pyy) / 2`` of the SCMC mechanical pressure."""
    Pxx, Pyy, _ = scmc_pressure_tensor(f1, f2, G12, pairing)
    return 0.5 * (Pxx + Pyy)


# ---------------------------------------------------------------------------
# Flat interface: the discriminating physical check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SCMCPressureConfig:
    """Configuration for the flat-interface / droplet pressure experiments."""

    ny: int = 64
    nx: int = 8
    n_steps: int = 3000
    G12: float = -0.9
    tau: float = 1.0
    rho_heavy: float = 0.7
    rho_light: float = 0.3
    device: str = "cpu"

    def validate(self) -> None:
        if self.ny < 16 or self.nx < 4:
            raise ValueError("flat-interface domain must be at least 4 x 16")
        if self.G12 >= 0:
            raise ValueError("G12 must be negative for phase separation")


def simulate_flat_interface(
    config: SCMCPressureConfig,
    *,
    steep: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relax a flat (planar) SCMC interface to static equilibrium.

    The domain is periodic; the interface normal is ``y`` and the field is
    uniform in ``x`` (``nx`` is only wide enough to keep the D2Q9 stencil
    valid).  ``steep=True`` starts from a one-cell step (the hardest initial
    data) instead of a linear ramp, which is used for a resolution check.
    """
    config.validate()
    device = torch.device(config.device)
    ny, nx = config.ny, config.nx
    ys = torch.arange(ny, dtype=torch.float32, device=device).view(ny, 1)
    frac = (ys / ny).expand(ny, nx).clone()
    if steep:
        frac = (frac >= 0.5).float()
    rho1 = config.rho_light + (config.rho_heavy - config.rho_light) * frac
    rho2 = config.rho_heavy - (config.rho_heavy - config.rho_light) * frac
    zero = torch.zeros_like(rho1)
    f1 = equilibrium(rho1, zero, zero)
    f2 = equilibrium(rho2, zero, zero)
    for _ in range(config.n_steps):
        f1, f2 = collide_sc_two_component(f1, f2, G_12=config.G12, tau1=config.tau, tau2=config.tau)
        f1 = stream(f1)
        f2 = stream(f2)
    return f1, f2


def flat_interface_pressure_profile(
    f1: torch.Tensor,
    f2: torch.Tensor,
    G12: float,
    pairing: str = "sym",
) -> dict[str, torch.Tensor | float]:
    """Normal-pressure profile of a flat interface and its flatness residual.

    In static equilibrium the normal stress ``P_yy`` must be constant across a
    planar interface.  ``flatness = max|P_NN − mean(P_NN)| / mean(P_NN)`` over
    the whole column therefore measures whether the pressure *formula* is the
    mechanical pressure at all; ``cs²ρ`` (the legacy proxy) fails this test.

    Also returns the mechanical surface tension of the flat interface,
    ``σ = ∫ (P_NN − P_TT) dy`` (sum over the column, 2-D).
    """
    Pxx, Pyy, _ = scmc_pressure_tensor(f1, f2, G12, pairing)
    p_n = Pyy.mean(dim=1)  # (ny,) — x-uniform, average removes round-off
    p_t = Pxx.mean(dim=1)
    rho_tot = (f1.sum(0) + f2.sum(0)).mean(dim=1)
    rho1 = f1.sum(0).mean(dim=1)
    rho2 = f2.sum(0).mean(dim=1)
    proxy = CS2 * rho_tot
    mean_n = float(p_n.mean().item())
    flatness = float((p_n - p_n.mean()).abs().max().item() / max(abs(mean_n), 1e-30))
    sigma = float((p_n - p_t).sum().item())
    return {
        "p_normal": p_n,
        "p_tangential": p_t,
        "rho_total": rho_tot,
        "rho1": rho1,
        "rho2": rho2,
        "proxy_cs2rho": proxy,
        "mean_normal": mean_n,
        "flatness": flatness,
        "sigma_flat": sigma,
        "proxy_flatness": float(
            (proxy - proxy.mean()).abs().max().item() / max(abs(float(proxy.mean())), 1e-30)
        ),
    }


def surface_tension_flat(f1: torch.Tensor, f2: torch.Tensor, G12: float, pairing: str = "sym") -> float:
    """``σ = ∫ (P_NN − P_TT) dy`` of a flat, static SCMC interface."""
    prof = flat_interface_pressure_profile(f1, f2, G12, pairing)
    return float(prof["sigma_flat"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Circular droplet: Young-Laplace law
# ---------------------------------------------------------------------------


def _circular_mask(ny: int, nx: int, r: float, device: torch.device) -> torch.Tensor:
    ys = torch.arange(ny, dtype=torch.float32, device=device)
    xs = torch.arange(nx, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return (xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2 <= r**2


def simulate_scmc_droplet(
    r: float,
    config: SCMCPressureConfig,
    *,
    nx: int | None = None,
    ny: int | None = None,
    n_steps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relax a circular SCMC droplet of the heavy component to steady state."""
    config.validate()
    device = torch.device(config.device)
    ny = ny or max(config.ny, int(4 * r))
    nx = nx or max(config.nx, int(4 * r))
    inside = _circular_mask(ny, nx, r, device)
    zero = torch.zeros((ny, nx), device=device)
    rho1 = torch.where(inside, torch.full_like(zero, config.rho_heavy), zero + config.rho_light)
    rho2 = torch.where(inside, torch.full_like(zero, config.rho_light), zero + config.rho_heavy)
    f1 = equilibrium(rho1, zero, zero)
    f2 = equilibrium(rho2, zero, zero)
    for _ in range(n_steps or config.n_steps):
        f1, f2 = collide_sc_two_component(f1, f2, G_12=config.G12, tau1=config.tau, tau2=config.tau)
        f1 = stream(f1)
        f2 = stream(f2)
    return f1, f2


def _max_velocity(f1: torch.Tensor, f2: torch.Tensor) -> float:
    rho = (f1.sum(0) + f2.sum(0)).clamp(min=1e-12)
    c = C.to(f1.device).float()
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    ux = (f1 * cx).sum(0) + (f2 * cx).sum(0)
    uy = (f1 * cy).sum(0) + (f2 * cy).sum(0)
    return float(torch.sqrt((ux / rho) ** 2 + (uy / rho) ** 2).max().item())


def droplet_pressure_jump(
    f1: torch.Tensor,
    f2: torch.Tensor,
    r: float,
    G12: float,
    pairing: str = "sym",
) -> dict[str, float]:
    """Young-Laplace measurement with the **mechanical** pressure.

    ``ΔP = <P>_{r<R/2} − <P>_{r>1.5R}`` with ``P = (Pxx+Pyy)/2`` (the isotropic
    part of the mechanical pressure tensor); the two sampling regions exclude
    the diffuse interface.  ``σ = ΔP·R`` for a 2-D circular interface.
    """
    Pxx, Pyy, _ = scmc_pressure_tensor(f1, f2, G12, pairing)
    p = 0.5 * (Pxx + Pyy)
    ny, nx = p.shape
    ys = torch.arange(ny, dtype=torch.float32, device=p.device)
    xs = torch.arange(nx, dtype=torch.float32, device=p.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    rf = torch.sqrt((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2)
    inside = rf <= 0.5 * r
    outside = rf >= 1.5 * r
    p_in = float(p[inside].mean().item())
    p_out = float(p[outside].mean().item())
    dp = p_in - p_out
    proxy_in = float((CS2 * (f1.sum(0) + f2.sum(0))[inside]).mean().item())
    proxy_out = float((CS2 * (f1.sum(0) + f2.sum(0))[outside]).mean().item())
    return {
        "r": r,
        "p_inside": p_in,
        "p_outside": p_out,
        "delta_p": dp,
        "sigma_laplace": dp * r,
        "delta_p_proxy": proxy_in - proxy_out,
        "sigma_proxy": (proxy_in - proxy_out) * r,
        "max_spurious_u": _max_velocity(f1, f2),
    }


def _fit_through_origin(xs: list[float], ys: list[float]) -> float:
    num = sum(x * y for x, y in zip(xs, ys, strict=True))
    den = sum(x * x for x in xs)
    return num / den if den > 0 else float("nan")


def laplace_sweep(
    radii: tuple[float, ...] = (8.0, 10.0, 12.0, 14.0),
    config: SCMCPressureConfig | None = None,
    *,
    pairing: str = "sym",
    n_steps: int | None = None,
) -> dict[str, object]:
    """Run the droplet sweep and return the Young-Laplace law check.

    Reported quantities:
    * ``per_radius``: raw per-radius measurements (mechanical and proxy);
    * ``sigma_laplace``: slope of ``ΔP`` vs ``1/R`` through the origin;
    * ``laplace_linearity``: relative spread of ``ΔP·R`` over the radii
      (``(max−min)/mean``) — the law predicts a *constant*;
    * ``spurious_current``: mean/max of ``|u|`` in the static droplet.
    """
    cfg = config or SCMCPressureConfig()
    rows = []
    for r in radii:
        f1, f2 = simulate_scmc_droplet(r, cfg, n_steps=n_steps)
        rows.append(droplet_pressure_jump(f1, f2, r, cfg.G12, pairing))
    inv_r = [1.0 / float(row["r"]) for row in rows]
    dp = [float(row["delta_p"]) for row in rows]
    sigma_r = [float(row["r"]) * float(row["delta_p"]) for row in rows]
    proxy_sigma = [float(row["r"]) * float(row["delta_p_proxy"]) for row in rows]
    mean_sigma = sum(sigma_r) / len(sigma_r)
    spread = (max(sigma_r) - min(sigma_r)) / abs(mean_sigma) if mean_sigma else float("inf")
    return {
        "per_radius": rows,
        "sigma_laplace": _fit_through_origin(inv_r, dp),
        "sigma_per_radius": sigma_r,
        "sigma_mean": mean_sigma,
        "laplace_linearity": spread,
        "sigma_proxy_per_radius": proxy_sigma,
        "sigma_proxy_sign_flips": sum(1 for s in proxy_sigma if not math.isfinite(s) or s <= 0),
        "mean_spurious_u": sum(float(r_["max_spurious_u"]) for r_ in rows) / len(rows),
        "pairing": pairing,
        "G12": cfg.G12,
        "tau": cfg.tau,
        "rho_heavy": cfg.rho_heavy,
        "rho_light": cfg.rho_light,
    }