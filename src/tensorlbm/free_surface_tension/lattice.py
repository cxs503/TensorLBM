"""Minimal lattice primitives (D2Q9 / D3Q19) for the capillary (CSF) solver.

The free-surface surface-tension verification lives on top of a *single-fluid*
lattice Boltzmann solve: the interface is described by a smooth colour/volume
fraction field ``phi`` and the capillary effect enters only as a body force
(Brackbill et al. 1992, CSF).  No density contrast and no separate gas solve is
introduced, which is exactly the modelling situation of the Koerner
free-surface formulation: only the liquid phase is solved and the free surface
applies a capillary pressure jump.

Conventions
-----------
* Field tensors are laid out ``(ny, nx)`` in 2-D and ``(nz, ny, nx)`` in 3-D,
  i.e. ``x`` is the *fastest* (last) axis.
* Vectors (velocity, force, gradients) are stored as ``(ndim, *spatial)`` with
  component ``0 = x``, ``1 = y`` (``2 = z``).  Only :meth:`Lattice.stream` needs
  to know that the tensor axis of component ``a`` is ``dim - 1 - a``.
* ``c_s^2 = 1/3`` throughout and ``dt = dx = 1``.
"""

from __future__ import annotations

import torch

__all__ = ["Lattice", "CS2", "get_lattice"]

CS2 = 1.0 / 3.0

_D2Q9_C = (
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
)
_D2Q9_W = (4.0 / 9.0,) + (1.0 / 9.0,) * 4 + (1.0 / 36.0,) * 4

_D3Q19_C = (
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 1, 0),
    (-1, 1, 0),
    (1, -1, 0),
    (-1, -1, 0),
    (1, 0, 1),
    (-1, 0, 1),
    (1, 0, -1),
    (-1, 0, -1),
    (0, 1, 1),
    (0, -1, 1),
    (0, 1, -1),
    (0, -1, -1),
)
_D3Q19_W = (1.0 / 3.0,) + (1.0 / 18.0,) * 6 + (1.0 / 36.0,) * 12


class Lattice:
    """A velocity set with the moment/streaming helpers used by the solver."""

    def __init__(
        self,
        c: tuple[tuple[int, ...], ...],
        w: tuple[float, ...],
        name: str,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.device = torch.device(device)
        self.c = torch.tensor(c, dtype=torch.long, device=self.device)
        self.w = torch.tensor(w, dtype=dtype, device=self.device)
        self.Q = int(self.c.shape[0])
        self.ndim = int(self.c.shape[1])
        self.cf = self.c.to(dtype)

    # -- geometry -----------------------------------------------------------
    def _view(self, ndim_spatial: int) -> tuple[int, ...]:
        return (self.Q,) + (1,) * ndim_spatial

    # -- moments ------------------------------------------------------------
    def equilibrium(self, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Second-order equilibrium ``f_i^eq`` (``u`` has shape ``(ndim, *sp)``)."""
        nsp = rho.dim()
        cu = self._contract(u)
        usq = (u * u).sum(dim=0).unsqueeze(0)
        w = self.w.view(self._view(nsp))
        return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)

    def _contract(self, v: torch.Tensor) -> torch.Tensor:
        """``c_i . v`` for ``v`` of shape ``(ndim, *sp)`` -> ``(Q, *sp)``.

        Written as an explicit sum over the 2-3 components: torch.einsum has no
        fast path for ``(Q, ndim) x (ndim, nz, ny, nx)`` and costs >20 ms per
        call at 32^3, which dominated the solver.
        """
        out = self.c[:, 0].to(v.dtype).view((-1,) + (1,) * (v.dim() - 1)) * v[0]
        for a in range(1, self.ndim):
            out = out + self.c[:, a].to(v.dtype).view((-1,) + (1,) * (v.dim() - 1)) * v[a]
        return out

    def moments(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rho, momentum_flux)`` with ``momentum_flux = sum_i c_i f_i``."""
        rho = f.sum(dim=0)
        mom = torch.stack([(f * self.cf[:, a].view((-1,) + (1,) * (f.dim() - 1))).sum(dim=0)
                           for a in range(self.ndim)], dim=0)
        return rho, mom

    def guo_source(self, u: torch.Tensor, force: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Guo (2002) forcing source term ``F_i`` for a body force density ``force``."""
        nsp = u.dim() - 1
        cu = self._contract(u)
        cf = self._contract(force)
        uf = (u * force).sum(dim=0).unsqueeze(0)
        pre = (1.0 - 0.5 / tau).unsqueeze(0)
        w = self.w.view(self._view(nsp))
        return w * pre * (3.0 * (cf - uf) + 9.0 * cu * cf)

    def stream(self, f: torch.Tensor) -> torch.Tensor:
        """Pull streaming ``f_i(x) <- f_i(x - c_i)`` (periodic wrap)."""
        out = torch.empty_like(f)
        for i in range(self.Q):
            shifts: list[int] = []
            dims: list[int] = []
            for a in range(self.ndim):
                ci = int(self.c[i, a])
                if ci:
                    shifts.append(ci)
                    # component a lives on tensor axis (f.dim()-1-a)
                    dims.append(f.dim() - 2 - a)
            if not shifts:
                out[i] = f[i]
            else:
                out[i] = torch.roll(f[i], shifts=shifts, dims=dims)
        return out

    def opposite(self) -> torch.Tensor:
        """Index vector of the reversed direction for each population."""
        idx = torch.empty(self.Q, dtype=torch.long, device=self.device)
        lut = {tuple(int(v) for v in self.c[i]): i for i in range(self.Q)}
        for i in range(self.Q):
            idx[i] = lut[tuple(int(-v) for v in self.c[i])]
        return idx


def get_lattice(
    ndim: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Lattice:
    """Return the default velocity set: D2Q9 for ``ndim == 2``, D3Q19 for 3-D."""
    if ndim == 2:
        return Lattice(_D2Q9_C, _D2Q9_W, "D2Q9", dtype=dtype, device=device)
    if ndim == 3:
        return Lattice(_D3Q19_C, _D3Q19_W, "D3Q19", dtype=dtype, device=device)
    raise ValueError(f"unsupported ndim={ndim}")