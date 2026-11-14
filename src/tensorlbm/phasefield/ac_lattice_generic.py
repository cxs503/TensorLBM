r"""Lattice-generic conservative Allen-Cahn phase-field LBE (D3Q15 / D3Q19).

What this module is
-------------------
The conservative Allen-Cahn (CAC) dual-distribution phase-field LBE of
:mod:`tensorlbm.cac_lbm` (frozen, D3Q19-only, verified), rewritten so that the
*velocity set* is a parameter: ``lattice="D3Q15"`` or ``"D3Q19"``.  Everything
except the lattice tables and the two derivative stencils is imported from the
frozen module unchanged, so the D3Q19 path is a re-derivation rather than a
new scheme (equivalence is asserted bit-for-bit in
``tests/test_ac_lattice_generic.py``).

Governing equations (lattice units, dx = dt = 1, cs^2 = 1/3)
------------------------------------------------------------
    d_t phi + div(phi u) = M lap(mu_phi)                               (AC)
    mu_phi = 4 beta phi(phi-1)(phi-1/2) - kappa lap(phi)
    beta = 12 sigma / W ,  kappa = 1.5 sigma W

transported by an h-population, while pressure/momentum are carried by an HCH
g-population -- Liang, Xu, Chen, Wang, Chai, Shi, Phys. Rev. E 97, 033309
(2018), Eq.(1-28).  :func:`tensorlbm.cac_lbm.free_energy_params` supplies
``(beta, kappa)``.

OSS counterpart learned from (anchor XVAL-LBM-PHASE-FIELD-ALLEN-CAHN-D3Q15-D3Q19)
----------------------------------------------------------------------------------
``lbmpy.phasefield_allen_cahn`` -- the phase-field/Allen-Cahn app that ships with
lssfau/walberla -- is the reference.  Its structure is a **mixed-lattice** scheme:
the phase field (h-population) is advanced on **D3Q15** while the hydrodynamics
(g-population) runs on **D3Q19**
(``apps/benchmarks/PhaseFieldAllenCahn/multiphase_codegen.py``, as recorded in
``oss_solutions`` row 363); lbmpy itself is stencil-parametric and derives the
isotropic derivative stencils per lattice
(``lbmpy/phasefield_allen_cahn/derivatives.py``).  This module therefore exposes
``lattice_phase`` and ``lattice_hydro`` independently, so that the walberla
configuration (15/19), the all-D3Q15 and the all-D3Q19 configurations all run
through the *same* code path:

* parameter calibration: ``beta = 12 sigma / W``, ``kappa = 3 sigma W / 2``
  (lbmpy/phasefield_allen_cahn/parameter_calculation.py, ``beta``/``kappa``
  properties) -- numerically identical to our ``free_energy_params``;
* lattice-specific second-order isotropic derivative stencils used for the
  phase field come from lbmpy's own symbolic derivation
  (lbmpy/phasefield_allen_cahn/derivatives.py: ``isotropic_gradient_symbolic``,
  ``laplacian_symbolic``; the ``Q == 15`` / ``Q == 19`` branches).  Evaluated
  for the two stencils, lbmpy produces

      gradient  (Q=15 and Q=19):   3 w_i c_i          (= velocity weights)
      laplacian Q=19:  -4 (centre) + 1/3 (axis) + 1/6 (face-diagonal)
      laplacian Q=15:  -32/9 (centre) + 7/18 (axis) + 11/72 (corner)

  The D3Q15 Laplacian is *not* the velocity-weighted Laplacian
  ``6 sum_{i!=0} w_i (phi(x+c_i) - phi(x))`` (which gives -14/3, 2/3, 1/12):
  both are second-order consistent, they differ only in the O(dx^2) error
  constant.  Both are available here (``lap_scheme="lbmpy"`` / ``"weights"``)
  so the choice can be measured instead of asserted.

No code is transcribed from lbmpy: only the two published stencil coefficient
sets are used (they are also derivable by hand from the isotropy conditions
``sum a_i c_ix^2 = 2``, ``sum a_i = 0`` plus one free weight; the unit tests
check polynomial exactness and second-order convergence on both lattices).

Bounded scope
-------------
Periodic domains only (no wall/bounce-back path yet -- the frozen module keeps
that responsibility).  ``float64`` is the default compute dtype because the
D3Q15 arm has no prior stability evidence; ``float32`` is supported and used by
the smoke tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tensorlbm.cac_lbm import (
    f_equilibrium,
    f_source,
    free_energy_params,
    g_equilibrium,
    g_force,
    lambda_phi,
    normal_from_gradient,
    rho_of_phi,
    s_hermite,
    viscosity_of_phi,
)
from tensorlbm.cac_lbm import CS2

__all__ = [
    "LatticeSpec",
    "ACPhaseFieldSim",
    "lattice_spec",
    "shifted_neighbors",
    "iso_gradient",
    "iso_laplacian",
    "chemical_potential",
    "macroscopic_g",
    "stream_pull",
    "init_phi_droplet",
    "init_phi_layers",
    "droplet_pressure_jump",
    "diagnostics",
]

# ---------------------------------------------------------------------------
# Velocity sets
# ---------------------------------------------------------------------------

# D3Q15: rest + 6 axis + 8 body diagonals.  Row order is [cx, cy, cz]
# (same row convention as tensorlbm.d3q19.C); tensor dims are (z, y, x).
_D3Q15_C = torch.tensor(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ],
    dtype=torch.int64,
)
_W0_D3Q15 = 2.0 / 9.0
_W1_D3Q15 = 1.0 / 9.0
_W3_D3Q15 = 1.0 / 72.0
_D3Q15_W = (_W0_D3Q15, _W1_D3Q15, _W1_D3Q15, _W1_D3Q15, _W1_D3Q15, _W1_D3Q15, _W1_D3Q15) + (
    _W3_D3Q15,
) * 8

# Isotropic Laplacian coefficients derived from lbmpy (see module docstring).
_LAP_D3Q15 = (-32.0 / 9.0,) + (7.0 / 18.0,) * 6 + (11.0 / 72.0,) * 8


@dataclass(frozen=True)
class LatticeSpec:
    """Velocity set + derivative stencils for one lattice."""

    name: str
    C: torch.Tensor  # (Q, 3) int64, row = [cx, cy, cz]
    W: torch.Tensor  # (Q,)    float64
    lap: torch.Tensor  # (Q,)    float64 laplacian coefficients, [0] = centre

    @property
    def Q(self) -> int:
        return int(self.C.shape[0])

    @property
    def w0(self) -> float:
        return float(self.W[0])

    def wv(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.W.to(device=device, dtype=dtype).view(self.Q, 1, 1, 1)

    def cf(self, dtype: torch.dtype, device: torch.device) -> list[torch.Tensor]:
        c = self.C.to(device=device, dtype=dtype)
        return [c[:, a].view(self.Q, 1, 1, 1) for a in range(3)]


def _classify(C: torch.Tensor) -> torch.Tensor:
    """1 for axis neighbours, 2 for face diagonals, 3 for body diagonals."""
    nz = (C != 0).sum(dim=1)
    return nz


def lattice_spec(name: str, *, lap_scheme: str = "lbmpy") -> LatticeSpec:
    """Build the :class:`LatticeSpec` for ``name`` in {"D3Q19", "D3Q15"}.

    ``lap_scheme``: ``"lbmpy"`` (the stencil lbmpy derives for that lattice) or
    ``"weights"`` (the velocity-weighted Laplacian ``2/cs^2 sum w_i (phi_i-phi)``,
    available for both lattices; identical for D3Q19 by construction).
    """
    key = name.upper()
    if key == "D3Q19":
        from tensorlbm.d3q19 import C, W_EXACT64

        C = C.clone()
        W = W_EXACT64.clone()
        cls = _classify(C)
        if lap_scheme == "lbmpy":
            lap = torch.where(
                cls == 0,
                torch.tensor(-4.0, dtype=torch.float64),
                torch.where(
                    cls == 1,
                    torch.tensor(1.0 / 3.0, dtype=torch.float64),
                    torch.tensor(1.0 / 6.0, dtype=torch.float64),
                ),
            )
        elif lap_scheme == "weights":
            lap = 6.0 * W
            lap[0] = -6.0 * (1.0 - float(W[0]))
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unknown lap_scheme {lap_scheme!r}")
        return LatticeSpec(name="D3Q19", C=C, W=W, lap=lap)

    if key == "D3Q15":
        C = _D3Q15_C.clone()
        W = torch.tensor(_D3Q15_W, dtype=torch.float64)
        cls = _classify(C)
        if lap_scheme == "lbmpy":
            lap = torch.tensor(_LAP_D3Q15, dtype=torch.float64)
        elif lap_scheme == "weights":
            lap = 6.0 * W
            lap[0] = -6.0 * (1.0 - float(W[0]))
        else:  # pragma: no cover
            raise ValueError(f"unknown lap_scheme {lap_scheme!r}")
        return LatticeSpec(name="D3Q15", C=C, W=W, lap=lap)

    raise ValueError(f"unsupported lattice {name!r} (expected 'D3Q15' or 'D3Q19')")


# ---------------------------------------------------------------------------
# Shifts / stencils
# ---------------------------------------------------------------------------


def _roll(f: torch.Tensor, spec_shifts: tuple[int, int, int]) -> torch.Tensor:
    """out[z,y,x] = f[z-cz, y-cy, x-cx]  (pull shift, periodic)."""
    cz, cy, cx = spec_shifts
    if cx == 0 and cy == 0 and cz == 0:
        return f
    return torch.roll(f, shifts=(cz, cy, cx), dims=(0, 1, 2))


def shifted_neighbors(phi: torch.Tensor, spec: LatticeSpec) -> torch.Tensor:
    """Stack (Q-1, nz, ny, nx) of ``phi(x + c_i)``, i = 1..Q-1 (rest skipped)."""
    nz, ny, nx = phi.shape
    out = torch.empty((spec.Q - 1, nz, ny, nx), dtype=phi.dtype, device=phi.device)
    for j, row in enumerate(spec.C[1:].tolist()):
        cx, cy, cz = row
        # phi(x + c): pull with the opposite shift
        out[j] = _roll(phi, (-cz, -cy, -cx))
    return out


def iso_gradient(nb: torch.Tensor, spec: LatticeSpec):
    """(gx, gy, gz) = 3 sum_{i!=0} w_i c_i phi(x+c_i)  (2nd-order isotropic).

    Identical on D3Q15 and D3Q19 (lbmpy ``isotropic_gradient_symbolic`` gives the
    velocity weights for both lattices).
    """
    w = spec.W[1:].to(device=nb.device, dtype=nb.dtype).view(spec.Q - 1, 1, 1, 1)
    c = spec.C[1:].to(device=nb.device, dtype=nb.dtype)
    return tuple(3.0 * (w * c[:, a].view(spec.Q - 1, 1, 1, 1) * nb).sum(dim=0) for a in range(3))


def iso_laplacian(phi: torch.Tensor, nb: torch.Tensor, spec: LatticeSpec) -> torch.Tensor:
    """lap(phi) with the lattice's isotropic Laplacian stencil (see docstring)."""
    a = spec.lap[1:].to(device=phi.device, dtype=phi.dtype).view(spec.Q - 1, 1, 1, 1)
    a0 = float(spec.lap[0])
    return a0 * phi + (a * nb).sum(dim=0)


def chemical_potential(
    phi: torch.Tensor, nb: torch.Tensor, beta: float, kappa: float, spec: LatticeSpec
) -> torch.Tensor:
    """mu_phi = 4 beta phi(phi-1)(phi-1/2) - kappa lap(phi)  (CAC Eq.(6))."""
    return 4.0 * beta * phi * (phi - 1.0) * (phi - 0.5) - kappa * iso_laplacian(phi, nb, spec)


def stream_pull(f: torch.Tensor, spec: LatticeSpec) -> torch.Tensor:
    """Periodic pull-stream: out[q](x) = f[q](x - c_q)."""
    out = torch.empty_like(f)
    for q, row in enumerate(spec.C.tolist()):
        cx, cy, cz = row
        out[q] = _roll(f[q], (cz, cy, cx))
    return out


def macroscopic_g(g, rho, grad_phi, drho, F, spec: LatticeSpec):
    """HCH pressure/velocity recovery, CAC Eq.(21) (lattice-generic version of
    :func:`tensorlbm.cac_lbm.macroscopic_g`)."""
    cf = spec.cf(g.dtype, g.device)
    mom = [(cf[a] * g).sum(dim=0) for a in range(3)]
    rho_safe = torch.where(rho != 0, rho, torch.ones_like(rho))
    u_star = [mom[a] / rho_safe for a in range(3)]
    u = [u_star[a] + 0.5 * F[a] / rho_safe for a in range(3)]
    ugr = u[0] * grad_phi[0] + u[1] * grad_phi[1] + u[2] * grad_phi[2]
    u2 = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
    w0 = spec.w0
    p = (CS2 / (1.0 - w0)) * (g[1:].sum(dim=0) + drho * ugr + rho * w0 * (-1.5 * u2))
    return u, p, u_star


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class ACPhaseFieldSim:
    """CAC h-LBE + HCH g-LBE on a chosen lattice, periodic box."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        *,
        lattice: str = "D3Q19",
        lattice_phase: str | None = None,
        lattice_hydro: str | None = None,
        lap_scheme: str = "lbmpy",
        rho_l: float = 1.0,
        rho_g: float = 0.1,
        mu_l: float = 0.1,
        mu_g: float | None = None,
        sigma: float = 0.01,
        width: float = 4.0,
        mobility: float = 0.05,
        mu_law: str = "harmonic",
        gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.shape = shape
        self.device = device
        self.dtype = dtype
        # ``lattice`` sets both populations; ``lattice_phase``/``lattice_hydro``
        # override them (walberla's Allen-Cahn app uses phase=D3Q15, hydro=D3Q19).
        self.spec_f = lattice_spec(lattice_phase or lattice, lap_scheme=lap_scheme)
        self.spec_g = lattice_spec(lattice_hydro or lattice, lap_scheme=lap_scheme)
        #: alias kept for the single-lattice reading of the state
        self.spec = self.spec_f
        self.rho_l, self.rho_g = rho_l, rho_g
        # mu_g defaults to mu_l * rho_g / rho_l: equal KINEMATIC viscosity across
        # the interface (the convention of the frozen D3Q19 benchmark).
        self.mu_l = mu_l
        self.mu_g = mu_l * rho_g / rho_l if mu_g is None else mu_g
        self.sigma, self.width = sigma, width
        self.beta, self.kappa = free_energy_params(sigma, width)
        self.mobility = mobility
        self.mu_law = mu_law
        self.gravity = gravity
        self.drho = rho_l - rho_g
        self.tau_f = 0.5 + 3.0 * mobility
        self.inv_tau_f = 1.0 / self.tau_f
        # ``fm_f`` in the frozen module is the *fluid-mask* multiplier (not the
        # Guo (1-1/(2 tau)) factor, which f_source/g_force already carry).
        # This module is periodic-only, so the mask is everywhere 1.
        self.fm_f = 1.0
        self.wv = self.spec_f.wv(dtype, device)
        self.cf = self.spec_f.cf(dtype, device)
        self.w0 = self.spec_f.w0
        self.wv_g = self.spec_g.wv(dtype, device)
        self.cf_g = self.spec_g.cf(dtype, device)
        self.w0_g = self.spec_g.w0
        self.f = torch.zeros((self.spec_f.Q, *shape), dtype=dtype, device=device)
        self.g = torch.zeros((self.spec_g.Q, *shape), dtype=dtype, device=device)
        self.phi_u_prev = tuple(
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(3)
        )
        self.step_count = 0
        self.initialised = False

    @property
    def label(self) -> str:
        """Phase/hydro lattice pair, e.g. ``D3Q19/D3Q19`` or ``D3Q15/D3Q19``."""
        return f"{self.spec_f.name}/{self.spec_g.name}"

    # -- initialisation -----------------------------------------------------
    def initialize(self, phi0: torch.Tensor, p0: float = 0.01) -> None:
        phi = phi0.to(device=self.device, dtype=self.dtype)
        rho = rho_of_phi(phi, self.rho_l, self.rho_g)
        zeros = torch.zeros_like(phi)
        u = (zeros, zeros.clone(), zeros.clone())
        self.f = f_equilibrium(phi, u, self.wv, self.cf).clone()
        self.g = g_equilibrium(
            torch.full_like(phi, p0), rho, u, self.wv_g, self.cf_g, self.w0_g
        ).clone()
        self.phi_u_prev = tuple(torch.zeros_like(phi) for _ in range(3))
        self.step_count = 0
        self.initialised = True

    # -- one time step ------------------------------------------------------
    def collide(self) -> None:
        phi = self.f.sum(dim=0)
        rho = rho_of_phi(phi, self.rho_l, self.rho_g)
        nb = shifted_neighbors(phi, self.spec_f)
        grad = iso_gradient(nb, self.spec_f)
        mu_phi = chemical_potential(phi, nb, self.beta, self.kappa, self.spec_f)
        mu = viscosity_of_phi(
            phi, self.rho_l, self.rho_g, self.mu_l, self.mu_g, self.mu_law
        )
        rho_safe = torch.where(rho != 0, rho, torch.ones_like(rho))
        tau_g = 0.5 + 3.0 * mu / rho_safe
        inv_tau_g = 1.0 / tau_g
        n_hat = normal_from_gradient(*grad)
        lam = lambda_phi(phi, self.width)
        F = (
            mu_phi * grad[0] + self.gravity[0] * rho,
            mu_phi * grad[1] + self.gravity[1] * rho,
            mu_phi * grad[2] + self.gravity[2] * rho,
        )
        u, p, _u_star = macroscopic_g(self.g, rho, grad, self.drho, F, self.spec_g)
        phi_u_now = tuple(phi * u[a] for a in range(3))
        phi_u_dt = tuple(phi_u_now[a] - self.phi_u_prev[a] for a in range(3))
        self.phi_u_prev = phi_u_now

        feq = f_equilibrium(phi, u, self.wv, self.cf)
        fsrc = f_source(phi_u_dt, lam, n_hat, self.tau_f, self.wv, self.cf)
        self.f = self.f + self.inv_tau_f * (feq - self.f) + self.fm_f * fsrc

        geq = g_equilibrium(p, rho, u, self.wv_g, self.cf_g, self.w0_g)
        gsrc = g_force(F, u, grad, self.drho, inv_tau_g, self.wv_g, self.cf_g)
        self.g = self.g + inv_tau_g * (geq - self.g) + self.fm_f * gsrc

    def stream(self) -> None:
        self.f = stream_pull(self.f, self.spec_f)
        self.g = stream_pull(self.g, self.spec_g)

    def step(self) -> None:
        self.collide()
        self.stream()
        self.step_count += 1

    # -- diagnostics --------------------------------------------------------
    def macro(self):
        """fp64 recompute of the macro chain (phi, rho, grad, mu_phi, u, p)."""
        f64 = torch.float64
        phi = self.f.to(f64).sum(dim=0)
        rho = rho_of_phi(phi, self.rho_l, self.rho_g)
        nb = shifted_neighbors(phi, self.spec_f)
        grad = iso_gradient(nb, self.spec_f)
        mu_phi = chemical_potential(phi, nb, self.beta, self.kappa, self.spec_f)
        F = tuple(mu_phi * grad[a] + self.gravity[a] * rho for a in range(3))
        u, p, _ = macroscopic_g(self.g.to(f64), rho, grad, self.drho, F, self.spec_g)
        return phi, rho, grad, mu_phi, u, p


# ---------------------------------------------------------------------------
# Initial conditions (CAC Eq.(30)/(33) profiles)
# ---------------------------------------------------------------------------


def init_phi_droplet(
    shape: tuple[int, int, int],
    radius: float,
    width: float,
    center: tuple[float, float, float] | None = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """phi = 0.5 + 0.5 tanh(2 (R - r) / W)  (phi = 1 inside the drop)."""
    nz, ny, nx = shape
    cz, cy, cx = center if center is not None else (nz / 2.0, ny / 2.0, nx / 2.0)
    z = torch.arange(nz, device=device, dtype=dtype).view(-1, 1, 1)
    y = torch.arange(ny, device=device, dtype=dtype).view(1, -1, 1)
    x = torch.arange(nx, device=device, dtype=dtype).view(1, 1, -1)
    r = ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2).sqrt()
    return 0.5 + 0.5 * torch.tanh(2.0 * (radius - r) / width)


def init_phi_layers(
    shape: tuple[int, int, int],
    width: float,
    y0: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """phi = 0.5 + 0.5 tanh(2 (y0 - y) / W)  (planar interface normal to y)."""
    nz, ny, nx = shape
    y = torch.arange(ny, device=device, dtype=dtype).view(1, -1, 1)
    return (0.5 + 0.5 * torch.tanh(2.0 * (y0 - y) / width)).expand(nz, ny, nx).contiguous()


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def droplet_pressure_jump(
    phi: torch.Tensor,
    p: torch.Tensor,
    *,
    radius: float,
    inner_band: tuple[float, float] = (0.95, 1.05),
    outer_band: tuple[float, float] = (1.6, 2.2),
) -> dict:
    """Young-Laplace readout: sample p inside/outside the diffuse interface.

    ``inner_band``/``outer_band`` are radial bands in units of ``radius``;
    the bands skip the interface (|phi - 0.5| < 0.45) so the sampling never
    sits on the transition layer.
    """
    nz, ny, nx = phi.shape
    z = torch.arange(nz, dtype=torch.float64, device=phi.device).view(-1, 1, 1)
    y = torch.arange(ny, dtype=torch.float64, device=phi.device).view(1, -1, 1)
    x = torch.arange(nx, dtype=torch.float64, device=phi.device).view(1, 1, -1)
    cz, cy, cx = nz / 2.0, ny / 2.0, nx / 2.0
    r = ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2).sqrt() / radius
    inner = (r >= inner_band[0]) & (r <= inner_band[1])
    outer = (r >= outer_band[0]) & (r <= outer_band[1])
    p_in = p[inner].mean().item()
    p_out = p[outer].mean().item()
    return {
        "p_in": p_in,
        "p_out": p_out,
        "dp": p_in - p_out,
        "n_inner": int(inner.sum().item()),
        "n_outer": int(outer.sum().item()),
    }


def interface_width_fit(
    phi: torch.Tensor, width_guess: float, axis: int = 1, lo: float = 0.05, hi: float = 0.95
) -> dict:
    """Fit ``phi = 0.5 - 0.5 tanh(2 (y - y0) / W_fit)`` to a 1-D profile.

    Returns the fitted interface position ``y0``, half-thickness ``W_fit`` and
    the maximum profile deviation from the analytic tanh form (a dimensionless
    measure of how well the simulated interface keeps the model's own
    equilibrium shape).
    """
    phi64 = phi.to(torch.float64)
    # average over the transverse planes -> 1-D profile
    dims = tuple(d for d in range(3) if d != 0)  # (nz, ny, nx): axis 0 of the line
    prof = phi64.mean(dim=dims) if axis == 0 else None
    if prof is None:
        nz, ny, nx = phi64.shape
        if axis == 1:
            prof = phi64.mean(dim=(0, 2))
        else:
            prof = phi64.mean(dim=(0, 1))
    n = prof.numel()
    s = torch.arange(n, dtype=torch.float64, device=phi.device)
    # y0 = where phi crosses 0.5 (linear interpolation)
    below = torch.nonzero(prof <= 0.5)
    y0 = s[below[0]].item() if len(below) else float(n) / 2.0
    if len(below):
        i = int(below[0].item())
        if i > 0:
            p_hi, p_lo = prof[i - 1].item(), prof[i].item()
            frac = (p_hi - 0.5) / (p_hi - p_lo) if p_hi != p_lo else 0.0
            y0 = (i - 1) + frac
    # least-squares W from the inverse-tanh transform over the interface band
    mask = (prof > lo) & (prof < hi)
    if int(mask.sum()) < 3:
        return {"y0": y0, "W_fit": float("nan"), "n_band": int(mask.sum()), "max_dev": float("nan")}
    yy = s[mask]
    arg = ((0.5 - prof[mask]) / 0.5).clamp(-0.999999, 0.999999)
    artanh = 0.5 * torch.log((1.0 + arg) / (1.0 - arg))
    # artanh = 2 (y - y0)/W  ->  slope = 2/W
    yc = yy - yy.mean()
    slope = (yc * (artanh - artanh.mean())).sum() / (yc * yc).sum()
    W_fit = float(2.0 / slope.abs())
    y0_fit = float(yy.mean() - (artanh.mean() / slope)) if slope != 0 else y0
    analytic = 0.5 - 0.5 * torch.tanh(2.0 * (s - y0_fit) / W_fit)
    max_dev = float((prof - analytic).abs()[mask].max())
    return {"y0": y0_fit, "W_fit": W_fit, "n_band": int(mask.sum()), "max_dev": max_dev}


def diagnostics(sim: ACPhaseFieldSim) -> dict:
    phi, rho, grad, mu_phi, u, p = sim.macro()
    gnorm = (grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2).sqrt()
    umax = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt().max().item()
    return {
        "step": sim.step_count,
        "lattice": sim.label,
        "lap_scheme": "lbmpy" if float(sim.spec_f.lap[1]) != 6.0 * float(sim.spec_f.W[1]) else "weights",
        "phi_integral": phi.sum().item(),
        "phi_min": phi.min().item(),
        "phi_max": phi.max().item(),
        "p_min": p.min().item(),
        "p_max": p.max().item(),
        "u_max": umax,
        "grad_phi_max": gnorm.max().item(),
        "nan_f": int(torch.isnan(sim.f).sum().item()),
        "nan_g": int(torch.isnan(sim.g).sum().item()),
    }


def run(sim: ACPhaseFieldSim, n_steps: int, sample_every: int = 500, on_sample=None):
    """Advance ``sim`` and collect :func:`diagnostics` every ``sample_every``."""
    samples: list[dict] = []
    for it in range(1, n_steps + 1):
        sim.step()
        if it % sample_every == 0:
            d = diagnostics(sim)
            samples.append(d)
            if on_sample is not None:
                on_sample(sim, d)
            if d["nan_f"] or d["nan_g"]:
                break
    return samples


def analytic_sigma_from_dp(dp: float, radius: float) -> float:
    """Young-Laplace inversion for a 3-D sphere: sigma = dp R / 2."""
    return dp * radius / 2.0


def sigma_relative_error(sigma_rec: float, sigma_ref: float) -> float:
    return abs(sigma_rec - sigma_ref) / abs(sigma_ref) * 100.0


def cs2() -> float:
    """Lattice speed of sound squared (exposed for report tables)."""
    return CS2


def lattice_invariants(spec: LatticeSpec) -> dict:
    """Moment checks used by the unit tests / report: sum w = 1, sum w c c = cs2 I."""
    W = spec.W
    worst = 0.0
    for a in range(3):
        for b in range(a, 3):
            m = float((W * spec.C[:, a].double() * spec.C[:, b].double()).sum())
            target = CS2 if a == b else 0.0
            worst = max(worst, abs(m - target))
    return {
        "lattice": spec.name,
        "sum_w": float(W.sum()),
        "fourth_moment_x": float((W * spec.C[:, 0].double() ** 4).sum()),
        "max_second_moment_dev": worst,
        "lap_coeff_sum": float(spec.lap.sum()),
        "grad_scale": math.isclose(float((3.0 * W * spec.C[:, 0].abs()).sum()), 1.0, rel_tol=1e-12),
    }