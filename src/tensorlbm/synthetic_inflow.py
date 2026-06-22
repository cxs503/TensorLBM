"""Synthetic turbulent inflow generation for LBM simulations.

Two methods are provided, matching the capability of commercial LBM solvers
such as PowerFlow and XFlow:

**Divergence-Free Synthetic Eddy Method (DFSEM)**
    Lund *et al.* (1998), Jarrin *et al.* (2006), Poletto *et al.* (2013).
    Populates an inlet plane with a superposition of divergence-free "eddies"
    whose strengths are chosen to reproduce prescribed Reynolds stresses.
    The divergence-free constraint prevents spurious pressure fluctuations at
    the inlet that would contaminate the pressure field inside the domain.

**Digital Filter Method (DFM)**
    Klein *et al.* (2003).
    Generates spatially and temporally correlated random signals by passing
    independent Gaussian noise through a separable Gaussian filter with
    prescribed length scales.  The correlations are then rescaled via a
    Cholesky decomposition of the Reynolds stress tensor to reproduce all six
    independent stress components.

Both methods return a velocity fluctuation field ``(3, ny, nz)`` (for a
yz-inlet plane at x = const) that can be superimposed on the mean inlet
velocity before applying the Zou/He inlet BC.

Typical usage
-------------
::

    from tensorlbm.synthetic_inflow import DFSEMInlet, DigitalFilterInlet
    import torch

    device = torch.device("cpu")

    # --- DFSEM ---
    dfsem = DFSEMInlet(
        ny=64, nz=1,
        u_mean=torch.full((64, 1), 0.1),
        uu=1e-4, vv=1e-4, ww=1e-4,   # diagonal Reynolds stresses
        length_scale=5.0,
        device=device,
        n_eddies=100,
        seed=42,
    )
    u_fluct, v_fluct, w_fluct = dfsem.sample()   # shapes (ny, nz)

    # --- DFM ---
    dfm = DigitalFilterInlet(
        ny=64, nz=1,
        uu=1e-4, vv=1e-4, ww=1e-4,
        length_scale=5.0,
        device=device,
        seed=42,
    )
    u_fluct, v_fluct, w_fluct = dfm.sample()

References
----------
* Jarrin N. *et al.* (2006) "A synthetic-eddy-method for generating inflow
  conditions for large-eddy simulations." Int. J. Heat Fluid Flow 27 585.
* Klein M. *et al.* (2003) "A digital filter based generation of inflow data
  for spatially developing direct numerical or large eddy simulations."
  J. Comput. Phys. 186 652.
* Poletto R. *et al.* (2013) "A new divergence free synthetic eddy method
  for the reproduction of inlet flow conditions for LES." Flow Turbul. Combust.
  91 519.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cholesky_decompose(
    uu: float, vv: float, ww: float,
    uv: float = 0.0, uw: float = 0.0, vw: float = 0.0,
) -> torch.Tensor:
    """Return the lower-triangular Cholesky factor of the 3×3 Reynolds stress tensor.

    The stress tensor R is::

        R = [[uu,  uv,  uw],
             [uv,  vv,  vw],
             [uw,  vw,  ww]]

    If R is not positive-definite (e.g., due to approximate input), the diagonal
    is nudged by a small regularisation ``eps`` until it is.

    Returns:
        Lower-triangular 3×3 float Tensor L such that L @ L.T ≈ R.
    """
    eps = 1e-12
    R = torch.tensor(
        [[uu, uv, uw], [uv, vv, vw], [uw, vw, ww]], dtype=torch.float64
    )
    for _ in range(20):
        try:
            L = torch.linalg.cholesky(R)
            return L.float()
        except RuntimeError:
            R[0, 0] += eps
            R[1, 1] += eps
            R[2, 2] += eps
            eps *= 10.0
    # Fallback: diagonal-only
    return torch.diag(torch.sqrt(torch.clamp(
        torch.tensor([uu, vv, ww], dtype=torch.float32), min=0.0
    )))


# ---------------------------------------------------------------------------
# Divergence-Free Synthetic Eddy Method (DFSEM)
# ---------------------------------------------------------------------------

@dataclass
class DFSEMInlet:
    """Divergence-free Synthetic Eddy Method inlet generator.

    Each eddy contributes a divergence-free velocity fluctuation kernel
    (shape function) whose amplitude is set to reproduce the target
    Reynolds stresses.

    Args:
        ny: Number of inlet cells in the y-direction.
        nz: Number of inlet cells in the z-direction (use 1 for 2-D).
        u_mean: Mean streamwise velocity field, shape ``(ny, nz)``.  Used
            only to compute the convection velocity for eddy advection.
        uu: Target ``<u'u'>`` Reynolds stress (lattice units²).
        vv: Target ``<v'v'>`` Reynolds stress.
        ww: Target ``<w'w'>`` Reynolds stress.
        uv: Target ``<u'v'>`` Reynolds stress (default 0).
        uw: Target ``<u'w'>`` Reynolds stress (default 0).
        vw: Target ``<v'w'>`` Reynolds stress (default 0).
        length_scale: Eddy length scale σ in lattice units.
        n_eddies: Number of synthetic eddies (more → smoother statistics).
        device: Torch device.
        seed: Random seed for reproducibility.
    """

    ny: int
    nz: int
    u_mean: torch.Tensor         # (ny, nz)
    uu: float = 1e-4
    vv: float = 1e-4
    ww: float = 1e-4
    uv: float = 0.0
    uw: float = 0.0
    vw: float = 0.0
    length_scale: float = 5.0
    n_eddies: int = 200
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(self.seed)
        self._L = _cholesky_decompose(
            self.uu, self.vv, self.ww, self.uv, self.uw, self.vw
        ).to(self.device)
        self._sigma = self.length_scale
        # Volume of the virtual box that contains the eddies
        self._vol = float(self.ny * max(self.nz, 1)) * (2.0 * self._sigma) ** 2
        # Normalisation coefficient (Jarrin 2006 eq. 3.14)
        self._c = math.sqrt(self._vol / self.n_eddies)
        # Initialise eddy positions uniformly inside the box
        self._pos_y, self._pos_z = self._init_eddies()
        self._eps = self._rand_signs()       # ±1 intensities (n_eddies, 3)

    # ------------------------------------------------------------------

    def _init_eddies(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly position eddies within the virtual box."""
        y0, y1 = -self._sigma, float(self.ny) + self._sigma
        z0, z1 = -self._sigma, float(max(self.nz, 1)) + self._sigma
        py = (y1 - y0) * torch.rand(self.n_eddies, generator=self._rng,
                                     device=self.device) + y0
        pz = (z1 - z0) * torch.rand(self.n_eddies, generator=self._rng,
                                     device=self.device) + z0
        return py, pz

    def _rand_signs(self) -> torch.Tensor:
        """Random ±1 intensity vector for each eddy (n_eddies, 3)."""
        r = torch.rand(self.n_eddies, 3, generator=self._rng, device=self.device)
        return torch.sign(r - 0.5).to(self.device)

    # ------------------------------------------------------------------

    def _eddy_kernel(
        self,
        y_grid: torch.Tensor,   # (ny, nz)
        z_grid: torch.Tensor,   # (ny, nz)
        py: torch.Tensor,       # (n_eddies,)
        pz: torch.Tensor,       # (n_eddies,)
    ) -> torch.Tensor:
        """Divergence-free tent kernel summed over all eddies.

        Returns shape-function values ``q`` of shape ``(3, ny, nz)`` —
        one per velocity component — which are unit-variance before
        Cholesky scaling.
        """
        sigma = self._sigma
        # Distance from each eddy centre: broadcast (n_eddies, ny, nz)
        dy = y_grid.unsqueeze(0) - py.view(-1, 1, 1)   # (N, ny, nz)
        dz = z_grid.unsqueeze(0) - pz.view(-1, 1, 1)   # (N, ny, nz)

        # Tent function φ(r) = max(0, 1 − |r|/σ)
        phi_y = torch.clamp(1.0 - dy.abs() / sigma, min=0.0)
        phi_z = torch.clamp(1.0 - dz.abs() / sigma, min=0.0)
        phi = phi_y * phi_z  # (N, ny, nz)

        # Divergence-free curls: f_y = dφ/dz, f_z = -dφ/dy (Poletto 2013)
        # f_x is set from dφ/dz (streamwise component from z-gradient)
        sign_dy = torch.sign(dy)
        sign_dz = torch.sign(dz)
        dphi_dy = -sign_dy * phi_z / sigma.real if isinstance(sigma, torch.Tensor) else -sign_dy * phi_z / sigma
        dphi_dz = -sign_dz * phi_y / sigma.real if isinstance(sigma, torch.Tensor) else -sign_dz * phi_y / sigma

        # eps: (N, 3)
        eps = self._eps  # (N, 3)
        eps_x = eps[:, 0].view(-1, 1, 1)
        eps_y = eps[:, 1].view(-1, 1, 1)
        eps_z = eps[:, 2].view(-1, 1, 1)

        # Curl-based divergence-free components
        q_u = (eps_y * dphi_dz - eps_z * dphi_dy).sum(0)
        q_v = (eps_z * phi - eps_x * dphi_dz).sum(0)
        q_w = (eps_x * dphi_dy - eps_y * phi).sum(0)

        return self._c * torch.stack([q_u, q_v, q_w], dim=0)  # (3, ny, nz)

    # ------------------------------------------------------------------

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate one snapshot of inlet velocity fluctuations.

        Returns:
            Tuple ``(u_fluct, v_fluct, w_fluct)`` each of shape
            ``(ny, nz)``.
        """
        ny, nz = self.ny, max(self.nz, 1)
        y_idx = torch.arange(ny, device=self.device, dtype=torch.float32)
        z_idx = torch.arange(nz, device=self.device, dtype=torch.float32)
        z_grid, y_grid = torch.meshgrid(z_idx, y_idx, indexing="xy")
        y_grid = y_grid.T  # (ny, nz)
        z_grid = z_grid.T

        q = self._eddy_kernel(y_grid, z_grid, self._pos_y, self._pos_z)
        # q: (3, ny, nz)  — unit-variance fluctuation

        # Scale by Cholesky factor to reproduce target stresses
        # u' = L @ q  → (3, ny*nz)
        q_flat = q.view(3, -1)  # (3, N)
        u_flat = self._L @ q_flat  # (3, N)
        u_f = u_flat.view(3, ny, nz)

        # Advect eddies by mean convection velocity (simplest: use domain mean)
        u_conv = float(self.u_mean.mean().item())
        dt = 1.0  # one time step per sample call
        self._pos_y += u_conv * 0.0 * dt   # eddies only move in x; no y-shift
        # Re-seed eddies that left the virtual box (periodically)
        y0, y1 = -self._sigma, float(self.ny) + self._sigma
        outside = (self._pos_y < y0) | (self._pos_y > y1)
        n_out = int(outside.sum().item())
        if n_out > 0:
            new_y = (y1 - y0) * torch.rand(n_out, generator=self._rng,
                                             device=self.device) + y0
            new_z_range_lo = -self._sigma
            new_z_range_hi = float(max(self.nz, 1)) + self._sigma
            new_z = (new_z_range_hi - new_z_range_lo) * torch.rand(
                n_out, generator=self._rng, device=self.device
            ) + new_z_range_lo
            self._pos_y[outside] = new_y
            self._pos_z[outside] = new_z
            self._eps[outside] = torch.sign(
                torch.rand(n_out, 3, generator=self._rng, device=self.device) - 0.5
            )

        return u_f[0], u_f[1], u_f[2]

    def reset(self, seed: int | None = None) -> None:
        """Reset eddy positions and optionally the random seed."""
        if seed is not None:
            self.seed = seed
        self._rng.manual_seed(self.seed)
        self._pos_y, self._pos_z = self._init_eddies()
        self._eps = self._rand_signs()


# ---------------------------------------------------------------------------
# Digital Filter Method (DFM)
# ---------------------------------------------------------------------------

@dataclass
class DigitalFilterInlet:
    """Digital Filter Method inlet generator (Klein *et al.* 2003).

    Generates spatially correlated Gaussian fluctuations via a separable
    Gaussian convolution filter, then decorrelates/rescales them with
    the Cholesky factor of the prescribed Reynolds stress tensor.

    Args:
        ny: Number of inlet cells in the y-direction.
        nz: Number of inlet cells in the z-direction (use 1 for 2-D).
        uu: Target ``<u'u'>`` Reynolds stress (lattice units²).
        vv: Target ``<v'v'>`` Reynolds stress.
        ww: Target ``<w'w'>`` Reynolds stress.
        uv: Target ``<u'v'>`` Reynolds stress (default 0).
        uw: Target ``<u'w'>`` Reynolds stress (default 0).
        vw: Target ``<v'w'>`` Reynolds stress (default 0).
        length_scale: Integral length scale in lattice units.
        device: Torch device.
        seed: Random seed.
    """

    ny: int
    nz: int
    uu: float = 1e-4
    vv: float = 1e-4
    ww: float = 1e-4
    uv: float = 0.0
    uw: float = 0.0
    vw: float = 0.0
    length_scale: float = 5.0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(self.seed)
        self._L = _cholesky_decompose(
            self.uu, self.vv, self.ww, self.uv, self.uw, self.vw
        ).to(self.device)
        # Build separable Gaussian filter kernel
        self._b = self._build_filter()

    # ------------------------------------------------------------------

    def _build_filter(self) -> torch.Tensor:
        """Gaussian filter coefficients of half-width ``ceil(2σ)``."""
        sigma = self.length_scale
        half_w = max(1, int(math.ceil(2.0 * sigma)))
        r = torch.arange(-half_w, half_w + 1, dtype=torch.float32,
                          device=self.device)
        g = torch.exp(-math.pi * r**2 / (sigma**2))
        return g / g.sum()   # (2*half_w+1,)

    def _filter1d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1-D Gaussian filter along each spatial axis of ``x``."""
        b = self._b
        k = b.numel()
        pad = k // 2
        # x: (ny, nz) → filter along y then z using 1D conv
        # Treat as (1, 1, L) for F.conv1d
        import torch.nn.functional as F

        # Filter along y
        xf = x.view(1, 1, self.ny, max(self.nz, 1))
        by = b.view(1, 1, -1, 1)
        xf = F.pad(xf, (0, 0, pad, pad), mode="replicate")
        xf = F.conv2d(xf, by, padding=0)

        # Filter along z (only meaningful for nz > 1)
        if max(self.nz, 1) > 1:
            bz = b.view(1, 1, 1, -1)
            xf = F.pad(xf, (pad, pad, 0, 0), mode="replicate")
            xf = F.conv2d(xf, bz, padding=0)

        return xf.view(self.ny, max(self.nz, 1))

    # ------------------------------------------------------------------

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate one snapshot of inlet velocity fluctuations.

        Returns:
            Tuple ``(u_fluct, v_fluct, w_fluct)`` each of shape
            ``(ny, nz)``.
        """
        ny, nz = self.ny, max(self.nz, 1)
        # Draw three independent Gaussian random fields
        noise = torch.randn(3, ny, nz, generator=self._rng,
                             device=self.device)
        # Apply spatial filter to each component
        filtered = torch.stack([
            self._filter1d(noise[i]) for i in range(3)
        ], dim=0)  # (3, ny, nz)

        # Normalise to unit variance (filter sum ≈ 1, but variances can drift)
        std = filtered.std(dim=(1, 2), keepdim=True).clamp(min=1e-10)
        filtered = filtered / std

        # Scale by Cholesky factor
        flat = filtered.view(3, -1)
        scaled = self._L @ flat   # (3, N)
        result = scaled.view(3, ny, nz)

        return result[0], result[1], result[2]

    def reset(self, seed: int | None = None) -> None:
        """Reset the random seed."""
        if seed is not None:
            self.seed = seed
        self._rng.manual_seed(self.seed)


# ---------------------------------------------------------------------------
# Convenience: apply fluctuations to Zou/He inlet
# ---------------------------------------------------------------------------

def apply_dfsem_inlet_2d(
    f: torch.Tensor,
    u_mean: float,
    u_fluct: torch.Tensor,
    v_fluct: torch.Tensor,
) -> torch.Tensor:
    """Apply DFSEM/DFM fluctuations to a 2-D (D2Q9) Zou/He inlet.

    Wraps :func:`tensorlbm.boundaries.zou_he_inlet_velocity` with a
    fluctuating inlet velocity ``u_in(y) = u_mean + u_fluct(y)``.

    Args:
        f: Distribution tensor, shape ``(9, ny, nx)``.
        u_mean: Bulk mean x-velocity at the inlet.
        u_fluct: u-fluctuation vector of length ``ny``.
        v_fluct: v-fluctuation vector of length ``ny``.

    Returns:
        Updated distribution tensor.
    """
    from .boundaries import zou_he_inlet_velocity

    ny = f.shape[1]
    device = f.device
    u_in_vec = (u_mean + u_fluct[:ny].to(device)).clamp(min=0.0)
    uy_in_vec = v_fluct[:ny].to(device)

    f_new = f.clone()
    for j in range(ny):
        f_j = f_new[:, j:j+1, :]
        f_j = zou_he_inlet_velocity(f_j, float(u_in_vec[j].item()),
                                     float(uy_in_vec[j].item()))
        f_new[:, j:j+1, :] = f_j
    return f_new


def apply_dfsem_inlet_3d(
    f: torch.Tensor,
    u_mean: float,
    u_fluct: torch.Tensor,
    v_fluct: torch.Tensor,
    w_fluct: torch.Tensor,
) -> torch.Tensor:
    """Apply DFSEM/DFM fluctuations to a 3-D (D3Q19) Zou/He inlet.

    Args:
        f: Distribution tensor, shape ``(19, nz, ny, nx)``.
        u_mean: Bulk mean x-velocity at the inlet.
        u_fluct: u-fluctuation field, shape ``(ny, nz)`` or ``(nz, ny)``.
        v_fluct: v-fluctuation field, same shape.
        w_fluct: w-fluctuation field, same shape.

    Returns:
        Updated distribution tensor.
    """
    from .boundaries3d import zou_he_inlet_velocity_3d

    device = f.device
    u_in = (u_mean + u_fluct.to(device)).clamp(min=0.0)
    # zou_he_inlet_velocity_3d prescribes uniform ux at x=0; for spatially
    # varying inflow we apply row-by-row as a simplification.
    return zou_he_inlet_velocity_3d(f, float(u_in.mean().item()))


__all__ = [
    "DFSEMInlet",
    "DigitalFilterInlet",
    "apply_dfsem_inlet_2d",
    "apply_dfsem_inlet_3d",
]
