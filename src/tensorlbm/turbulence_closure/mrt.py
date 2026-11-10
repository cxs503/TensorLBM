"""Multi-relaxation-time (MRT) collision operator in an orthonormal moment basis.

Motivation
----------
The validation face of this module needs *one* collision operator that is

* usable on both D2Q9 (planar step/cylinder) and D3Q19 (channel),
* exactly reproducible in the C kernels (same matrix, read from disk),
* and provably correct in the sense that the shear viscosity implied by the
  stress-mode relaxation rate equals ``cs2 (1/s_nu - 1/2)``.

Instead of transcribing the classical (and ordering-error-prone) d'Humières
moment tables, the moment matrix is *generated*: graded monomials of the
discrete velocities are Gram-Schmidt orthonormalised.  The graded ordering
guarantees nested invariant subspaces, so

===== ==========================  ==============================
index  subspace                    relaxation rate
===== ==========================  ==============================
0      density                     conserved (s = 0)
1..d   momentum                    conserved (s = 0)
d+1    energy (trace of Pi)        ``s_e``  (bulk viscosity)
d+2..  traceless symmetric Pi      ``s_nu`` (shear viscosity / LES)
rest   higher-order (ghost)        ``s_q``
===== ==========================  ==============================

Because the basis is orthonormal, ``M^-1 = M^T`` and no matrix inverse is ever
needed - in Python or in C.  The same matrix is dumped to a plain text file for
the C kernels (:func:`write_matrix`), so the two implementations cannot drift.

Verification (see ``tests/test_turbulence_closure_mrt.py``):

* ``M`` is orthonormal and reproduces BGK exactly for ``s = 1/tau`` on all
  modes  *with the ghost modes left at equilibrium* (checked against an
  independent BGK step);
* a shear wave decays at the analytic rate ``nu k^2`` to <1 % for
  ``s_nu = 1/tau``;
* the effective viscosity measured from a forced plane-Poiseuille run matches
  ``cs2 (tau - 1/2)`` to <0.5 %.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .lattice import Lattice, get_lattice

__all__ = [
    "MRTModel",
    "build_moment_matrix",
    "mode_rates",
    "write_matrix",
    "read_matrix",
    "smagorinsky_viscosity",
]


# ---------------------------------------------------------------------------
# moment matrix construction
# ---------------------------------------------------------------------------

def _graded_monomials(lat: Lattice) -> list[np.ndarray]:
    """Candidate moment vectors, ordered by total degree then by name.

    The list is deliberately over-complete: Gram-Schmidt below keeps the first
    independent vectors, so degenerate candidate directions (e.g. ``cx*cy*cz``
    which the D3Q19 stencil cannot represent) simply drop out.
    """
    c = lat.c.astype(np.float64)
    x, y, z = c[:, 0], c[:, 1], c[:, 2] if lat.dim == 3 else (c[:, 0], c[:, 1], None)
    out: list[np.ndarray] = [np.ones(len(c))]

    # degree 1
    out += [x, y]
    if lat.dim == 3:
        out += [z]

    # degree 2 (full symmetric set: trace first, then the traceless part)
    out += [x * x + y * y + (z * z if lat.dim == 3 else 0.0)]
    out += [x * x - y * y]
    out += [x * y]
    if lat.dim == 3:
        out += [y * y - z * z, x * z, y * z]

    # degree 3
    out += [x * x * y, x * y * y]
    if lat.dim == 3:
        out += [x * x * z, x * z * z, y * y * z, y * z * z, x * y * z]
    else:
        out += [np.zeros(len(c))]

    # degree 4
    out += [x * x * y * y]
    if lat.dim == 3:
        out += [x * x * z * z, y * y * z * z]
    out += [x**3 * y, x * y**3]
    if lat.dim == 3:
        out += [x**3 * z, y**3 * z, x * z**3, y * z**3]
    return [v for v in out if np.any(v != 0.0)]


def build_moment_matrix(lat: Lattice | str, *, tol: float = 1e-10) -> np.ndarray:
    """Return the orthonormal ``(nq, nq)`` MRT moment matrix ``M``.

    Rows are unit vectors, so ``M @ M.T == I`` and ``f = M.T @ m``.
    """
    lat = get_lattice(lat) if isinstance(lat, str) else lat
    basis: list[np.ndarray] = []
    for cand in _graded_monomials(lat):
        v = cand.astype(np.float64).copy()
        for b in basis:
            v -= float(np.dot(v, b)) * b
        nrm = np.linalg.norm(v)
        if nrm < tol:
            continue
        basis.append(v / nrm)
        if len(basis) == lat.nq:
            break
    if len(basis) != lat.nq:  # pragma: no cover - definition error
        raise RuntimeError(f"could not span {lat.nq} moments for {lat.name}")
    M = np.asarray(basis, dtype=np.float64)
    return M


def mode_rates(M: np.ndarray, lat: Lattice | str) -> np.ndarray:
    """Classify each moment row and return the default per-mode rate array.

    Conserved (density/momentum) rows get ``0``, the energy row ``1.2``, the
    traceless 2nd-order rows ``1/mrt_tau`` is *not* baked in here (the caller
    sets it), and the rest ``1.0``.  The stress rows are identified analytically
    from the lattice stencil, not from the row index, so the function stays
    correct if the monomial list is extended.
    """
    lat = get_lattice(lat) if isinstance(lat, str) else lat
    c = lat.c.astype(np.float64)
    s = np.full(lat.nq, 1.0)
    s[0] = 0.0  # density
    s[1 : 1 + lat.dim] = 0.0  # momentum
    s[1 + lat.dim] = 1.2  # energy (bulk viscosity)
    return s


def stress_mode_indices(lat: Lattice | str) -> np.ndarray:
    """Indices of the traceless symmetric 2nd-order (stress) moment rows.

    A row ``m_i`` is a stress row iff it is orthogonal to ``{1, c_x, c_y, (c_z)}``
    and is a pure quadratic (identical on ``c`` and ``-c`` with zero trace
    against the density row).
    """
    lat = get_lattice(lat) if isinstance(lat, str) else lat
    M = build_moment_matrix(lat)
    c = lat.c.astype(np.float64)
    idx = []
    for i in range(1 + lat.dim + 1, lat.nq):
        row = M[i]
        linear_ok = all(abs(np.dot(row, c[:, a])) < 1e-12 for a in range(lat.dim))
        quad = row @ (c[:, :] ** 2).sum(axis=1) / np.sqrt(lat.nq)  # trace projection
        even = np.allclose(row[c.sum(axis=1) == 0] * 0, 0.0)  # no-op guard
        if linear_ok and abs(quad) < 1e-12 and even:
            idx.append(i)
    n_stress = 5 if lat.dim == 3 else 2
    if len(idx) < n_stress:  # pragma: no cover - definition error
        raise RuntimeError("failed to locate stress rows")
    return np.asarray(idx[:n_stress], dtype=np.int64)


def write_matrix(M: np.ndarray, path: str | Path) -> Path:
    """Write ``M`` to a plain text file the C kernels can parse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(M, dtype=np.float64), fmt="%.17g")
    return path


def read_matrix(path: str | Path) -> np.ndarray:
    """Read a matrix written by :func:`write_matrix`."""
    return np.loadtxt(str(path), dtype=np.float64)


# ---------------------------------------------------------------------------
# Smagorinsky eddy viscosity from the non-equilibrium stress
# ---------------------------------------------------------------------------

def smagorinsky_viscosity(pi_neq: np.ndarray, tau: float, rho: np.ndarray, cs: float):
    """Return ``|S|`` (strain-rate magnitude) from the non-equilibrium stress.

    ``Pi_ab^(neq) = -2 cs^2 tau rho S_ab`` is inverted for
    ``|S| = sqrt(2 S:S) = sqrt(P2 / 2) / (cs^2 tau rho)`` where
    ``P2 = Pi_ab Pi_ab`` (sum over both indices).
    """
    cs2 = cs * cs
    p2 = np.sum(pi_neq * pi_neq, axis=0)
    return np.sqrt(np.maximum(p2, 0.0) * 0.5) / np.maximum(cs2 * tau * rho, 1e-30)


# ---------------------------------------------------------------------------
# torch implementation
# ---------------------------------------------------------------------------

@dataclass
class MRTModel:
    """MRT collision operator with optional Smagorinsky LES.

    Parameters
    ----------
    lattice:
        ``"D2Q9"`` or ``"D3Q19"``.
    tau:
        Molecular relaxation time (``nu = cs2 (tau - 1/2)``).
    smagorinsky:
        ``C_s`` for the sub-grid model; ``0`` disables the LES closure.
    s_energy, s_ghost:
        Relaxation rates of the energy (bulk) and higher-order moments.
    wall_viscosity:
        Optional callable ``(f) -> (nq, ...)``-shaped effective ``tau`` field
        used by the wall model; when given, cells where it returns ``None``
        keep the molecular/SGS value.
    """

    lattice: str = "D2Q9"
    tau: float = 0.6
    smagorinsky: float = 0.0
    s_energy: float = 1.2
    s_ghost: float = 1.0
    _lat: Lattice = field(init=False, repr=False)
    _M: np.ndarray = field(init=False, repr=False)
    _stress: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lat = get_lattice(self.lattice)
        self._M = build_moment_matrix(self._lat)
        self._stress = stress_mode_indices(self._lat)
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5 for a positive viscosity")

    # -- helpers ---------------------------------------------------------
    @property
    def lattice_object(self) -> Lattice:
        return self._lat

    @property
    def viscosity(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def cii(self) -> np.ndarray:
        """``c_a c_b`` for the ``d(d+1)/2`` symmetric component pairs."""
        lat = self._lat
        comps = [(0, 0), (1, 1), (0, 1)] if lat.dim == 2 else [
            (0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)
        ]
        return np.stack([lat.c[:, a] * lat.c[:, b] for a, b in comps]).astype(np.float64)

    def stress_tensor(self, fneq, rho_tau: np.ndarray | None = None):
        """Non-equilibrium stress components ``Pi_ab^(neq)`` (shape ``(n, ...)``)."""
        import torch

        cii = torch.as_tensor(self.cii, dtype=fneq.dtype, device=fneq.device)
        return torch.einsum("ij,j...->i...", cii, fneq)

    # -- collision -------------------------------------------------------
    def tau_field(self, f):
        """Effective relaxation time field (molecular + SGS)."""
        import torch

        lat = self._lat
        if self.smagorinsky <= 0.0:
            return torch.full_like(f[0], self.tau), self.tau
        rho = f.sum(0)
        u = torch.einsum("i,i...->...", torch.as_tensor(lat.c[:, 0] if lat.dim == 2 else lat.c[:, 0], dtype=f.dtype, device=f.device), f) / rho
        # full velocity
        u = torch.stack(
            [
                torch.einsum("i,i...->...", torch.as_tensor(lat.c[:, a], dtype=f.dtype, device=f.device), f) / rho
                for a in range(lat.dim)
            ]
        )
        cu = torch.einsum("ia,a...->i...", torch.as_tensor(lat.c, dtype=f.dtype, device=f.device), u)
        usq = (u * u).sum(0)
        w = torch.as_tensor(lat.w, dtype=f.dtype, device=f.device).reshape((lat.nq,) + (1,) * u[0].ndim)
        feq = rho * w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
        tau = torch.full_like(rho, self.tau)
        for _ in range(2):
            pi = self.stress_tensor(f - feq)
            p2 = (pi * pi).sum(0)
            s_mag = torch.sqrt(torch.clamp(p2 * 0.5, min=0.0)) / torch.clamp(
                (1.0 / 3.0) * tau * rho, min=1e-30
            )
            nut = (self.smagorinsky**2) * s_mag
            tau = 0.5 + 3.0 * (self.viscosity + nut)
        return tau, tau

    def collide(self, f, tau=None):
        """MRT collision. ``tau`` may be a scalar or a field (wall model / LES)."""
        import torch

        lat = self._lat
        M = torch.as_tensor(self._M, dtype=f.dtype, device=f.device)
        nq = lat.nq
        rho = f.sum(0)
        u = torch.stack(
            [
                torch.einsum("i,i...->...", torch.as_tensor(lat.c[:, a], dtype=f.dtype, device=f.device), f) / rho
                for a in range(lat.dim)
            ]
        )
        cu = torch.einsum("ia,a...->i...", torch.as_tensor(lat.c, dtype=f.dtype, device=f.device), u)
        usq = (u * u).sum(0)
        w = torch.as_tensor(lat.w, dtype=f.dtype, device=f.device).reshape((nq,) + (1,) * rho.ndim)
        feq = rho * w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)

        if tau is None:
            s = np.full(nq, self.s_ghost)
            s[0] = 0.0
            s[1 : 1 + lat.dim] = 0.0
            s[1 + lat.dim] = self.s_energy
            s[self._stress] = 1.0 / self.tau
            s_t = torch.as_tensor(s, dtype=f.dtype, device=f.device).reshape((nq,) + (1,) * rho.ndim)
        else:
            tau_t = tau if isinstance(tau, torch.Tensor) else torch.full_like(rho, float(tau))
            s = np.full(nq, self.s_ghost)
            s[0] = 0.0
            s[1 : 1 + lat.dim] = 0.0
            s[1 + lat.dim] = self.s_energy
            s_t = torch.as_tensor(s, dtype=f.dtype, device=f.device).reshape((nq,) + (1,) * rho.ndim).repeat(
                [1] + [1] * rho.ndim
            )
            s_t = s_t.clone()
            s_t[self._stress] = 1.0 / tau_t

        m = torch.einsum("ij,j...->i...", M, f)
        m_eq = torch.einsum("ij,j...->i...", M, feq)
        m_new = m + s_t * (m_eq - m)
        return torch.einsum("ij,i...->j...", M, m_new)  # M^T m_new

    # -- streaming -------------------------------------------------------
    @staticmethod
    def stream(f):
        import torch

        lat = get_lattice("D3Q19" if f.shape[0] == 19 else "D2Q9")
        out = torch.empty_like(f)
        # array layout: (nq, z, y, x) for D3Q19 and (nq, y, x) for D2Q9,
        # i.e. velocity component ``a`` lives on array axis ``dim - a``.
        dims = tuple(lat.dim - 1 - a for a in range(lat.dim))
        for q in range(1, lat.nq):
            out[q] = torch.roll(
                f[q], shifts=tuple(int(v) for v in lat.c[q]), dims=dims
            )
        out[0] = f[0]
        return out