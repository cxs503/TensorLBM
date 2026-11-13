r"""Ternary conservative Allen-Cahn phase field, semantics-traced to CudaPhaseField.

Why this module exists
----------------------
The XVAL anchor ``XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOT`` classifies
``AminZar69/CudaPhaseField`` (OSS, GPL-3) as *not applicable* to TensorLBM's MCMP
Shan-Chen/Poiseuille line: the OSS multiphase physics is a **ternary
conservative Allen-Cahn phase field**, not a pseudo-potential model
(OSS ``src/global_subroutines.f90``: no ``G``, no interaction strength, no
colour gradient, no component-resolved pressure boundary; the ``(1000,500,100)``
triple in ``src/device_var.f90:25`` enters only ``rho = sum_k phi_k rho_k``).

That classification is a claim about *semantics*, and a claim about semantics is
only worth something if the semantics can be re-executed.  This module is the
executable form of the OSS phase-field model: every coefficient below is traced
to a source line (vendored at
``xval/XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOT/oss_src/``), and the
driver ``scripts/xval_cudaphasefield_not_pseudopot.py`` measures the module
against analytic references (interface width, surface tension, mass closure,
density independence).

Provenance map (OSS = ``AminZar69/CudaPhaseField``)
---------------------------------------------------
======================  ====================================================  ==============
quantity                OSS expression                                        source line
======================  ====================================================  ==============
phase mass densities    ``density(3) = [1000, 500, 100]``                     device_var.f90:25
interface thickness     ``w = 4``                                              device_var.f90:29
mobility                ``m = 0.01``                                           device_var.f90:30
over-relaxation         ``w_c = 1/(0.5 + 3 m)``                                device_var.f90:31
binary relaxation       ``tau_k = 0.9``                                        device_var.f90:32-34
shipped tension matrix  ``sigma(3,3) = 0.1`` (all entries)                     device_var.f90:35
phase coefficients      ``lambda_k = sigma_jk + sigma_ki - sigma_ij``          device_var.f90:36-40
harmonic coupling       ``lambdat = 3/(1/l1 + 1/l2 + 1/l3)``                  device_var.f90:41-42
isotropic Laplacian     ``(S_diag + 4 S_edge - 20 phi)/6``                     global_subroutines.f90:318-326
chemical potential      ``mu_k = (12/w)[l_k phi_k(1-phi_k)(1-2phi_k)``         global_subroutines.f90:329-339
                        ``          - 2 l_t phi1 phi2(1-phi1-phi2)]``
                        ``        - (3/4) w l_k lap(phi_k)``
unit normals            ``n_k = grad phi_k / |grad phi_k|`` (1e-32 guard)      global_subroutines.f90:200-244
compression source      ``efh_i = 2 a_k (e_i.n_k) - sum_{j != k} a_j (e_i.n_j)``  global_subroutines.f90:600-628
                        with ``a_k = (4/w)|phi_k (1-phi_k)|``
source populations      ``hlph_i = w_i efh_i / 3``                            global_subroutines.f90:655-661
equilibrium             ``h_i^eq = phi_k Gamma_i - 0.5 hlph_i``                global_subroutines.f90:665-676
relaxation              ``h_i <- (1-w_c) h_i + w_c h_i^eq + hlph_i``           global_subroutines.f90:688-700
algebraic third phase   ``phi3 = 1 - phi1 - phi2``                            global_subroutines.f90:43
populations -> field    ``phi_k = sum_i h_{k,i}``                             global_subroutines.f90:38-39
======================  ====================================================  ==============

Coefficient convention (measured, not assumed)
---------------------------------------------
With ``Gamma_i = w_i`` at rest and the pair ``(12/w, (3/4) w lambda)`` above,
the flat-interface solution of ``mu = const`` is

    phi(s) = 0.5 (1 + tanh(2 s / w))          (10-90 width = w * atanh(0.8))

and the gradient-energy integral gives

    sigma = (3/4) w lambda * integral (dphi/ds)^2 ds = lambda / 2 .

So the OSS coefficient pair is the *standard* CAC free energy with
``lambda = 2 sigma``: the shipped ``sigma(3,3) = 0.1`` therefore corresponds to a
true lattice tension of ``0.05`` unless the caller doubles ``lambda``.  That is
exactly TensorLBM's own convention ``beta = 12 sigma / W``, ``k = 1.5 sigma W``
(see :func:`tensorlbm.cac_lbm2d.free_energy_params`), and the driver checks it
both analytically and from a converged profile (:func:`tension_from_lambda`,
:func:`tension_from_profile`).

Scope / honesty
---------------
* The phase field is validated **on its own** (``u = 0``): the OSS fluid step
  (``g``-LBE with ``rho = sum_k phi_k density_k`` and the Korteweg stress
  ``mu_k grad phi_k``) is *not* transplanted here, so no claim is made about the
  coupled hydrodynamics of the OSS code.  The interface physics (equilibrium
  profile, tension closure, mass closure, density independence of the phase
  equation) is decidable without the fluid step and is what the anchor needs.
* The module is a semantics transplant, not a copy: it is written over
  TensorLBM's own :mod:`tensorlbm.d2q9` lattice and reuses
  :func:`tensorlbm.cac_lbm2d.iso_laplacian_2d` /
  :func:`tensorlbm.cac_lbm2d.iso_gradient_2d` after proving the stencils are
  identical (:func:`oss_laplacian_is_stencil_identity`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from ..cac_lbm2d import iso_gradient_2d, iso_laplacian_2d
from ..d2q9 import C, W_EXACT

__all__ = [
    "OSS_PHASE_MASS_DENSITIES",
    "OSS_SHIPPED_SIGMA",
    "TernaryAC2D",
    "analytic_interface_width",
    "flat_interface_profile",
    "flat_interface_state",
    "interface_width_10_90",
    "measure_sigma_from_frame",
    "lambda_from_sigma",
    "lambda_from_sigma_matrix",
    "oss_laplacian_is_stencil_identity",
    "oss_relaxation_factor",
    "rho_of_phases",
    "ternary_ac_step_terms",
    "tension_from_lambda",
    "tension_from_profile",
    "ternary_chemical_potential",
    "ternary_compression_source",
]

#: ``src/device_var.f90:25`` -- per-component *mass* densities (inertial only).
OSS_PHASE_MASS_DENSITIES: tuple[float, float, float] = (1000.0, 500.0, 100.0)

#: ``src/device_var.f90:35`` -- the shipped (degenerate) tension matrix.
OSS_SHIPPED_SIGMA: float = 0.1


# ---------------------------------------------------------------------------
# Source constants
# ---------------------------------------------------------------------------
def oss_relaxation_factor(m: float) -> float:
    """``w_c = 1 / (0.5 + 3 m)`` (``src/device_var.f90:31``).

    For ``m = 0.01`` this is ``1.8867924528...``: an *over-relaxed* scheme whose
    amplification factor is ``1 - w_c = -0.88679...`` (stable, ``|1-w_c| < 1``).
    The corresponding lattice relaxation time is ``tau_phi = 1/w_c = 0.5 + 3 m``
    (``0.53`` at ``m = 0.01``), which is the number the OSS study quotes.
    """
    m = float(m)
    if m <= 0.0:
        raise ValueError("mobility m must be positive")
    w_c = 1.0 / (0.5 + 3.0 * m)
    if not abs(1.0 - w_c) < 1.0:
        raise ValueError(f"over-relaxation factor {w_c} is unstable (|1-w_c| >= 1)")
    return w_c


def lambda_from_sigma(sigma_12: float, sigma_31: float, sigma_23: float) -> tuple[tuple[float, float, float], float]:
    """``lambda_k`` and the harmonic mean ``lambdat`` (``src/device_var.f90:36-42``).

    ``lambda_1 = s12 + s31 - s23``, ``lambda_2 = s12 + s23 - s31``,
    ``lambda_3 = s31 + s23 - s12``, ``lambdat = 3 / sum_k 1/lambda_k``.
    """
    l1 = sigma_12 + sigma_31 - sigma_23
    l2 = sigma_12 + sigma_23 - sigma_31
    l3 = sigma_31 + sigma_23 - sigma_12
    lam = (l1, l2, l3)
    if any(v == 0.0 for v in lam):
        raise ValueError(f"degenerate lambda {lam}: harmonic mean undefined")
    lambdat = 3.0 / sum(1.0 / v for v in lam)
    return lam, lambdat


def lambda_from_sigma_matrix(sigma: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], float]:
    """3x3 symmetric tension matrix (zero diagonal) -> ``(lambda, lambdat)``.

    The OSS rule is *not* a triangle-inequality projection: for an inadmissible
    tension triple (``s_ij > s_jk + s_ki``) it returns a negative ``lambda``,
    which the harmonic mean then propagates.  The driver reports this as a
    transplant caveat rather than silently clipping.
    """
    if len(sigma) != 3 or any(len(row) != 3 for row in sigma):
        raise ValueError("sigma must be 3x3")
    for i in range(3):
        if abs(float(sigma[i][i])) > 0.0:
            raise ValueError("sigma diagonal must be zero")
        for j in range(i + 1, 3):
            if abs(float(sigma[i][j]) - float(sigma[j][i])) > 1e-15:
                raise ValueError("sigma must be symmetric")
    return lambda_from_sigma(float(sigma[0][1]), float(sigma[0][2]), float(sigma[1][2]))


# ---------------------------------------------------------------------------
# Stencils
# ---------------------------------------------------------------------------
def _w(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Exact D2Q9 weights (:data:`tensorlbm.d2q9.W_EXACT`).

    ``tensorlbm.d2q9.W`` is stored in float32; its float64 cast sums to
    ``1 + 7.45e-9`` and a ternary AC collision would then amplify the phase mass
    by ``(1 + 7.45e-9 * w_c)`` per step (a linear ramp, not a drift-free scheme).
    The exact literals keep ``sum_i w_i == 1`` to the last bit, which is what the
    mass-closure claim of this module rests on.
    """
    return torch.tensor(W_EXACT, dtype=dtype, device=device)


def _iso_laplacian(phi: torch.Tensor) -> torch.Tensor:
    """OSS 9-point isotropic Laplacian (``global_subroutines.f90:318-326``).

    Written directly as ``(S_diag + 4 S_edge - 20 phi)/6`` so the source form is
    visible; equals ``6 sum_i w_i (phi(x+c_i) - phi(x))`` for the exact weights.
    """
    acc = -20.0 * phi
    for q in range(1, 9):
        dx = int(C[q, 0].item())
        dy = int(C[q, 1].item())
        acc = acc + (4.0 if q <= 4 else 1.0) * torch.roll(phi, shifts=(-dy, -dx), dims=(-2, -1))
    return acc / 6.0


def _iso_gradient(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """OSS isotropic gradient (``global_subroutines.f90:177-186``).

    ``dphi/dx = (phi(x+1)-phi(x-1))/3
              + (phi(x+1,y-1)+phi(x+1,y+1)-phi(x-1,y-1)-phi(x-1,y+1))/12``
    i.e. ``3 sum_i w_i c_ix phi(x+c_i)``.
    """
    gx = torch.zeros_like(phi)
    gy = torch.zeros_like(phi)
    for q in range(1, 9):
        dx = int(C[q, 0].item())
        dy = int(C[q, 1].item())
        nb = torch.roll(phi, shifts=(-dy, -dx), dims=(-2, -1))
        gx = gx + (dx / 3.0 if q <= 4 else dx / 12.0) * nb
        gy = gy + (dy / 3.0 if q <= 4 else dy / 12.0) * nb
    return gx, gy


def _c(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return C.to(device=device, dtype=dtype)


def oss_laplacian_is_stencil_identity(n: int = 12) -> float:
    """Max |OSS stencil - :func:`iso_laplacian_2d`| on a pseudo-random field.

    OSS (``global_subroutines.f90:318-326``) writes
    ``(S_diag + 4 S_edge - 20 phi)/6`` = ``(1/6) S_diag + (2/3) S_edge - (10/3) phi``,
    which is the same 9-point isotropic operator as
    ``6 sum_i w_i (phi(x+c_i) - phi(x))``.  The check makes the reuse of the
    repo operator a measured identity instead of an assumption.
    """
    g = torch.Generator().manual_seed(20261113)
    phi = torch.rand(n, n, dtype=torch.float64, generator=g)
    # explicit OSS gather
    acc = -20.0 * phi
    for q in range(1, 9):
        dx = int(C[q, 0].item())
        dy = int(C[q, 1].item())
        nb = torch.roll(phi, shifts=(-dy, -dx), dims=(-2, -1))
        acc = acc + (4.0 if q <= 4 else 1.0) * nb
    oss = acc / 6.0
    repo = iso_laplacian_2d(phi, axes="pp")
    return float((oss - repo).abs().max().item())


def oss_unit_normal(gx: torch.Tensor, gy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``n = grad phi / |grad phi|`` with the OSS ``1e-32`` guard (lines 200-244)."""
    mag = torch.sqrt(gx * gx + gy * gy)
    safe = torch.where(mag > 0.0, mag, torch.sqrt(mag + 1e-32))
    return gx / safe, gy / safe


# ---------------------------------------------------------------------------
# Model terms
# ---------------------------------------------------------------------------
def ternary_chemical_potential(
    phi: torch.Tensor,
    lap: torch.Tensor,
    lam: Sequence[float],
    lambdat: float,
    w: float,
) -> torch.Tensor:
    """``mu_k`` for ``k = 1,2,3`` (``global_subroutines.f90:329-339``).

    ``phi`` and ``lap`` have shape ``(3, ...)``; returns ``(3, ...)``.
    ``phi[2] = 1 - phi[0] - phi[1]`` is *not* enforced here (the OSS applies the
    closure when reconstructing the field from populations).
    """
    if phi.shape[0] != 3 or lap.shape[0] != 3:
        raise ValueError("phi and lap must be stacked as (3, ...)")
    lam_t = torch.as_tensor(lam, dtype=phi.dtype, device=phi.device).reshape(3, *([1] * (phi.dim() - 1)))
    p1, p2, p3 = phi[0], phi[1], phi[2]
    cross = 2.0 * float(lambdat) * p1 * p2 * p3
    bulk = (12.0 / float(w)) * (lam_t * phi * (1.0 - phi) * (1.0 - 2.0 * phi) - cross)
    return bulk - 0.75 * float(w) * lam_t * lap


def ternary_compression_source(
    phi: torch.Tensor,
    nx: torch.Tensor,
    ny: torch.Tensor,
    w: float,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """``efh_i`` for the nine D2Q9 directions (``global_subroutines.f90:600-628``).

    ``a_k = (4/w)|phi_k (1 - phi_k)|``; ``efh_i = 2 a_k (e_i.n_k) -
    sum_{j != k} a_j (e_i.n_j)`` for phase ``k``.  Returns ``(3, 9, ...)``:
    the per-phase, per-direction source before the ``w_i / 3`` weighting.
    """
    dtype = dtype or phi.dtype
    device = device or phi.device
    a = (4.0 / float(w)) * torch.abs(phi * (1.0 - phi))  # (3, ...)
    c = _c(dtype, device)
    ex = c[:, 0].reshape(9, *([1] * (phi.dim() - 1)))
    ey = c[:, 1].reshape(9, *([1] * (phi.dim() - 1)))
    dot = (
        ex.unsqueeze(0) * nx.unsqueeze(1) + ey.unsqueeze(0) * ny.unsqueeze(1)
    )  # (3, 9, ...)
    weighted = a.unsqueeze(1) * dot  # (3, 9, ...)
    total = weighted.sum(dim=0, keepdim=True)  # sum_j a_j (e_i.n_j)
    # efh_k = 2 a_k (e_i.n_k) - sum_{j != k} a_j (e_i.n_j) = 3 a_k (e_i.n_k) - total
    # (the "j != k" exclusion is what makes sum_k efh_k == 0, hence mass exact)
    return 3.0 * weighted - total


def rho_of_phases(
    phi: torch.Tensor,
    density: Sequence[float] = OSS_PHASE_MASS_DENSITIES,
) -> torch.Tensor:
    """``rho = sum_k phi_k density_k`` (``global_subroutines.f90:47``).

    Inertial mass only -- the OSS study itself notes this is *not* a
    thermodynamic density ratio (there is no EOS in the code).
    """
    r = torch.zeros_like(phi[0])
    for k in range(3):
        r = r + float(density[k]) * phi[k]
    return r


# ---------------------------------------------------------------------------
# Analytic references
# ---------------------------------------------------------------------------
def analytic_interface_width(w: float) -> float:
    """10-90 width of ``phi = 0.5 (1 + tanh(2 s / w))``: ``w * atanh(0.8)``."""
    return float(w) * math.atanh(0.8)


def flat_interface_profile(x: torch.Tensor, w: float, x0: float = 0.0) -> torch.Tensor:
    """Analytic equilibrium profile of the transplanted free energy."""
    return 0.5 * (1.0 + torch.tanh(2.0 * (x - x0) / float(w)))


def tension_from_lambda(lam: float) -> float:
    """``sigma = lambda / 2`` for the OSS coefficient pair (see module docstring)."""
    return 0.5 * float(lam)


def tension_from_profile(
    phi: torch.Tensor,
    w: float,
    lam: float,
    half_width: float | None = None,
) -> float:
    """Gradient-energy integral ``sigma = kappa * integral (dphi/dn)^2 dn``.

    ``kappa = (3/4) w lambda`` (the coefficient of the OSS chemical potential's
    Laplacian term).  ``phi`` must be a 1-D profile with unit node spacing.

    A periodic box holds two interfaces, so ``half_width`` crops a single
    interface around the steepest-gradient node -- without it the integral counts
    both interfaces and returns ``2 sigma``.  Pass the *same* ``half_width`` for
    the analytic profile to compare like with like.
    """
    dphi = torch.gradient(phi.reshape(-1))[0]
    if half_width is not None:
        n = dphi.numel()
        i = int(dphi.abs().argmax().item())
        h = int(round(float(half_width)))
        lo, hi = max(0, i - h), min(n, i + h + 1)
        dphi = dphi[lo:hi]
    kappa = 0.75 * float(w) * float(lam)
    return float((kappa * dphi * dphi).sum().item())


def interface_width_10_90(phi: torch.Tensor) -> float:
    """Local 10-90 width of the interface nearest the steepest gradient.

    A periodic box holds at least two interfaces, so the measurement is *local*:
    start at the steepest-gradient node, walk left/right until the normalised
    phase crosses 0.1/0.9 and interpolate linearly.  Raises ``ValueError`` when
    the band is not bracketed inside the array (a non-equilibrium state is an
    error, not a number).
    """
    p = phi.detach().reshape(-1).double()
    n = p.numel()
    d = torch.zeros_like(p)
    d[1:-1] = 0.5 * (p[2:] - p[:-2])
    i = int(d.abs().argmax().item())
    q = p if bool(d[i] > 0.0) else 1.0 - p

    def cross(step: int, level: float) -> float:
        j = i
        while 0 <= j < n:
            hit = q[j] >= level if step > 0 else q[j] <= level
            if hit:
                j0 = j - step
                if not (0 <= j0 < n):
                    break
                frac = (level - float(q[j0])) / float(q[j] - q[j0])
                return float(j0) + step * frac
            j += step
        raise ValueError(f"interface band {level} not bracketed within the profile")

    return abs(cross(+1, 0.9) - cross(-1, 0.1))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass
class TernaryAC2D:
    """D2Q9 ternary conservative Allen-Cahn solver (OSS semantics).

    Two population sets ``h1``, ``h2`` carry phases 1 and 2; phase 3 is algebraic
    (``phi3 = 1 - phi1 - phi2``).  ``u`` is held at zero: the OSS fluid step is
    explicitly out of scope here (see the module docstring), and every quantity
    the anchor needs (interface profile, tension closure, mass closure, density
    independence) is decidable in the pure phase-field limit.
    """

    shape: tuple[int, int]
    w: float = 4.0
    m: float = 0.01
    sigma: Sequence[Sequence[float]] | None = None
    density: Sequence[float] = OSS_PHASE_MASS_DENSITIES
    dtype: torch.dtype = torch.float64
    device: torch.device | str = "cpu"
    #: diagnostic knob, predeclared per arm: 1.0 = literal OSS normalisation
    #: (``2 a_k`` with the ``/3`` weighting); 1.5 restores the textbook
    #: ``w_i (4/W) phi(1-phi) (c_i.n)`` strength.  Never fitted.
    source_scale: float = 1.0

    lam: tuple[float, float, float] = field(init=False)
    lambdat: float = field(init=False)
    w_c: float = field(init=False)
    h1: torch.Tensor = field(init=False)
    h2: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        sig = self.sigma
        if sig is None:
            sig = [[0.0, OSS_SHIPPED_SIGMA, OSS_SHIPPED_SIGMA],
                   [OSS_SHIPPED_SIGMA, 0.0, OSS_SHIPPED_SIGMA],
                   [OSS_SHIPPED_SIGMA, OSS_SHIPPED_SIGMA, 0.0]]
        self.sigma = [list(map(float, row)) for row in sig]
        self.lam, self.lambdat = lambda_from_sigma_matrix(self.sigma)
        self.w_c = oss_relaxation_factor(self.m)
        ny, nx = self.shape
        self.h1 = torch.zeros((9, ny, nx), dtype=self.dtype, device=self.device)
        self.h2 = torch.zeros((9, ny, nx), dtype=self.dtype, device=self.device)

    # -- state ------------------------------------------------------------
    @property
    def tau_phi(self) -> float:
        """``1/w_c = 0.5 + 3 m`` -- the relaxation time the OSS study quotes."""
        return 1.0 / self.w_c

    def initialize(self, phi1: torch.Tensor, phi2: torch.Tensor | None = None) -> None:
        """Populations at rest: ``h_{k,i} = phi_k w_i``."""
        phi1 = phi1.to(dtype=self.dtype, device=self.device)
        if phi2 is None:
            phi2 = torch.zeros_like(phi1)
        phi2 = phi2.to(dtype=self.dtype, device=self.device)
        if tuple(phi1.shape) != tuple(self.shape):
            raise ValueError(f"phi1 shape {tuple(phi1.shape)} != {self.shape}")
        w = _w(self.dtype, self.device).reshape((9,) + (1,) * phi1.dim())
        self.h1 = phi1.unsqueeze(0) * w
        self.h2 = phi2.unsqueeze(0) * w

    def fields(self) -> torch.Tensor:
        """``(phi1, phi2, phi3)`` stacked, ``phi3 = 1 - phi1 - phi2`` (line 43)."""
        p1 = self.h1.sum(dim=0)
        p2 = self.h2.sum(dim=0)
        p3 = 1.0 - p1 - p2
        return torch.stack([p1, p2, p3])

    # -- one lattice step -------------------------------------------------
    def step(self) -> None:
        phi = self.fields()  # (3, ny, nx)
        grads = [_iso_gradient(phi[k]) for k in range(3)]
        laps = torch.stack([_iso_laplacian(phi[k]) for k in range(3)])
        nrm = [oss_unit_normal(gx, gy) for (gx, gy) in grads]
        nx = torch.stack([n[0] for n in nrm])
        ny = torch.stack([n[1] for n in nrm])

        efh = ternary_compression_source(phi, nx, ny, self.w, dtype=self.dtype, device=self.device)
        wv = _w(self.dtype, self.device).reshape((1, 9) + (1,) * (phi.dim() - 1))
        w9 = wv.reshape((9,) + (1,) * (phi.dim() - 1))
        hlph = self.source_scale * wv * efh / 3.0  # (3, 9, ny, nx)

        # Gamma_i = w_i at rest (u = 0); h_i^eq = phi_k Gamma_i - 0.5 hlph_i
        heq1 = phi[0].unsqueeze(0) * w9 - 0.5 * hlph[0]
        heq2 = phi[1].unsqueeze(0) * w9 - 0.5 * hlph[1]

        w_c = self.w_c
        self.h1 = (1.0 - w_c) * self.h1 + w_c * heq1 + hlph[0]
        self.h2 = (1.0 - w_c) * self.h2 + w_c * heq2 + hlph[1]

        # streaming (periodic, pull form: population arriving at x came from x - c_i)
        for q in range(1, 9):
            dx = int(C[q, 0].item())
            dy = int(C[q, 1].item())
            self.h1[q] = torch.roll(self.h1[q], shifts=(dy, dx), dims=(-2, -1))
            self.h2[q] = torch.roll(self.h2[q], shifts=(dy, dx), dims=(-2, -1))

        self._mu = ternary_chemical_potential(phi, laps, self.lam, self.lambdat, self.w)

    # -- diagnostics ------------------------------------------------------
    def chemical_potential(self) -> torch.Tensor:
        """``mu_k`` of the current field (the coupled-Korteweg caller's input)."""
        phi = self.fields()
        laps = torch.stack([_iso_laplacian(phi[k]) for k in range(3)])
        return ternary_chemical_potential(phi, laps, self.lam, self.lambdat, self.w)

    def total_phase(self) -> torch.Tensor:
        return self.fields().sum(dim=0)

    def phase_mass(self) -> torch.Tensor:
        """``sum_x phi_k`` for ``k = 1,2,3``."""
        return self.fields().sum(dim=(1, 2))

    def darcy_velocity(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Korteweg stress gradient ``sum_k mu_k grad phi_k`` (OSS ``fsx/fsy``).

        ``global_subroutines.f90:462-463``.  Exposed for the coupled caller; not
        used by the phase equation at ``u = 0``.
        """
        phi = self.fields()
        mu = self.chemical_potential()
        grads = [_iso_gradient(phi[k]) for k in range(3)]
        fx = sum(mu[k] * grads[k][0] for k in range(3))
        fy = sum(mu[k] * grads[k][1] for k in range(3))
        return fx, fy

    def run(
        self,
        n_steps: int,
        sample_every: int = 0,
        on_sample: Callable[["TernaryAC2D", int], None] | None = None,
    ) -> None:
        for it in range(1, int(n_steps) + 1):
            self.step()
            if sample_every and on_sample is not None and it % sample_every == 0:
                on_sample(self, it)


# ---------------------------------------------------------------------------
# Driver helpers used by the XVAL entry point
# ---------------------------------------------------------------------------
def ternary_ac_step_terms(
    phi: torch.Tensor,
    w: float,
    lam: Sequence[float],
    lambdat: float,
    nx: torch.Tensor,
    ny: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pure-function view of one collision: ``mu``, ``efh``, ``hlph`` (no state).

    Lets tests and the XVAL driver assert individual model terms against the OSS
    expressions without running a lattice step.
    """
    laps = torch.stack([_iso_laplacian(phi[k]) for k in range(3)])
    mu = ternary_chemical_potential(phi, laps, lam, lambdat, w)
    efh = ternary_compression_source(phi, nx, ny, w, dtype=phi.dtype, device=phi.device)
    wv = _w(phi.dtype, phi.device).reshape((1, 9) + (1,) * (phi.dim() - 1))
    return {"mu": mu, "efh": efh, "hlph": wv * efh / 3.0}


def flat_interface_state(
    shape: tuple[int, int] = (8, 128),
    w: float = 4.0,
    initial_width: float | None = None,
    x0: float | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Periodic binary flat-interface state (phase 3 absent, ``phi2 = 0``).

    Two counter-oriented interfaces at ``x0`` and ``x0 + nx/2`` make the field
    periodic; inside a plateau the second factor is ``-1`` so the profile at
    ``x0`` is the analytic ``0.5 (1 + tanh(2 (x-x0)/w))``.
    """
    ny, nx = shape
    x = torch.arange(nx, dtype=dtype)
    if x0 is None:
        x0 = 0.25 * nx
    width = float(initial_width if initial_width is not None else w)
    prof = 0.5 * (1.0 + torch.tanh(2.0 * (x - x0) / width) * torch.tanh(2.0 * (x - x0 - 0.5 * nx) / width))
    return prof.unsqueeze(0).expand(ny, nx).contiguous()


def measure_sigma_from_frame(phi1: torch.Tensor, w: float, lam: float) -> dict[str, float]:
    """Measure interface tension and width on the steepest-gradient row.

    ``phi1`` is a 2-D phase field; the row with the largest mean |dphi/dx| is
    taken as the interface cut (a flat-interface frame is invariant along y).
    """
    dphi_x = (torch.roll(phi1, -1, dims=1) - torch.roll(phi1, 1, dims=1)) * 0.5
    i = int(dphi_x.abs().mean(dim=1).argmax().item())
    prof = phi1[i]
    return {
        "sigma": tension_from_profile(prof, w, lam, half_width=3.0 * w),
        "row_index": float(i),
        "width_10_90": interface_width_10_90(prof),
        "profile_min": float(prof.min().item()),
        "profile_max": float(prof.max().item()),
    }
