"""Conservative Allen-Cahn **h-population** kernel with TCLB parity (XVAL m17678).

Why this module exists
----------------------
TensorLBM already carries a conservative Allen-Cahn (CAC) h-population kernel
(:mod:`tensorlbm.cac_lbm`, :mod:`tensorlbm.cac_lbm2d`) written after
Liang, Xu, Chen, Wang, Chai & Shi, *PRE* **97**, 033309 (2018).  The mature
open-source implementation CFD-GO/``TCLB`` solves the *same* conservative
Allen-Cahn equation with a visibly different discrete recipe
(``models/multiphase/d3q27_pf_velocity``, Fakhari & Mitchell lineage).  This
module is the clean-room TensorLBM counterpart of that recipe, so that the two
can be cross-validated field-by-field (see
``docs/XVAL_M17678_TCLB_CONSERVATIVE_AC_H_POPULATION.md``).

Nothing here is transcribed from the GPL-3.0 TCLB sources: the discrete scheme
is the published conservative phase-field / Chiu-Lin form, and every TCLB-side
number used in the cross-validation is extracted from their source *text* at
run time by ``scripts/xval_m17678_tclb_cac_h_population.py``.

The discrete scheme (per TCLB ``Dynamics.c.Rt``; line numbers in the XVAL script)
------------------------------------------------------------------------------
phase field           ``phi = sum_i h_i``
equilibrium           ``h_i^eq = w_i phi [1 + 3 c_i.u + 4.5 (c_i.u)^2 - 1.5 |u|^2]``
                      (equilibrium order ``eq_order=1`` drops the O(u^2) terms)
normal                ``n = grad(phi)/|grad(phi)|``, zeroed where ``|grad| < min_gradient``
counter term          ``F_phi,i = w_i (1 - 4 (phi-1/2)^2) / W (c_i . n)``
                      (identically ``w_i 4 phi (1-phi)/W (c_i . n)``)
collision             ``h_i <- h_i - omega (h_i - h_i^eq + 0.5 F_phi,i) + F_phi,i``
relaxation            ``omega = 1/(3 M + 1/2)``  (``M`` = mobility)
transport             periodic pull streaming, ``h_i(x, t+1) = h_i^*(x - c_i, t)``

The ``(1 - omega/2)`` split and the ``omega = 1/(3M+1/2)`` pairing make the
numerical mobility ``cs^2 (1/omega - 1/2) = M`` and the counter-gradient flux
amplitude ``cs^2 * 4 phi (1-phi) / W``; the equilibrium interface is therefore

    phi = 1/2 [1 - tanh(2 (r - R) / W_eff)],      W_eff = 3 M W,

i.e. ``W`` is an *anti-diffusivity coefficient*, not the realized interface
width, unless ``M = 1/3`` (see the XVAL report, P8 sensitivity).

Operators (stencil-exact, D2Q9 / D3Q19 / D3Q27)
-----------------------------------------------
isotropic gradient   ``grad_a phi = 3 sum_i w_i c_ia phi(x + c_i)``
isotropic laplacian  ``lap phi = 6 sum_{i != 0} w_i (phi(x + c_i) - phi)``

Both reduce to TCLB's explicitly written stencils (``model.R`` IsotropicGrad:
coefficients 16 / 4 / 1 over 72; ``myLaplace``: 16 / 4 / 1 over 36); the
equivalence is checked numerically against the *textual* R formula in the XVAL
test, not assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "Stencil",
    "STENCIL_NAMES",
    "get_stencil",
    "isotropic_gradient",
    "isotropic_laplacian",
    "interface_normal",
    "counter_term",
    "h_equilibrium",
    "CACHPopulation",
]

STENCIL_NAMES = ("d2q9", "d3q19", "d3q27")

# exact lattice weights keyed by (dim, n_nonzero_components, |c|^2)
_WEIGHT_TABLE = {
    2: {(0, 0): (4.0, 9.0), (1, 1): (1.0, 9.0), (2, 2): (1.0, 36.0)},
    3: {
        (0, 0): (8.0, 27.0),
        (1, 1): (2.0, 27.0),
        (2, 2): (1.0, 54.0),
        (3, 3): (1.0, 216.0),
    },
}


@dataclass(frozen=True)
class Stencil:
    """Velocity set + exact (rational) weights, ordering-agnostic."""

    name: str
    C: torch.Tensor  # (Q, dim) int64
    W: torch.Tensor  # (Q,)    float64

    @property
    def dim(self) -> int:
        return int(self.C.shape[1])

    @property
    def q(self) -> int:
        return int(self.C.shape[0])

    @property
    def cs2(self) -> float:
        return 1.0 / 3.0

    def velocities(self) -> list[tuple[int, ...]]:
        return [tuple(int(v) for v in row) for row in self.C.tolist()]


def get_stencil(name: str) -> Stencil:
    """Velocity set from the library lattices, weights re-derived exactly.

    The weights are rebuilt from the velocity *classes* in ``float64`` instead of
    casting the library's ``float32`` tables, so that second moments carry no
    ``1e-9`` asymmetry; the pairing with each velocity is checked here.
    """
    if name == "d2q9":
        from ..d2q9 import C as C2

        C = C2
    elif name == "d3q19":
        from ..d3q19 import C as C3

        C = C3
    elif name == "d3q27":
        from ..d3q27 import C as C27

        C = C27
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown stencil {name!r}; expected one of {STENCIL_NAMES}")

    C = C.to(dtype=torch.int64).clone()
    dim = int(C.shape[1])
    table = _WEIGHT_TABLE[dim]
    w = []
    for row in C.tolist():
        c2 = sum(v * v for v in row)
        key = (sum(1 for v in row if v != 0), c2)
        num, den = table[key]
        w.append(num / den)
    W = torch.tensor(w, dtype=torch.float64)
    st = Stencil(name=name, C=C, W=W)
    mom = torch.einsum("q,qa,qb->ab", W, C.to(torch.float64), C.to(torch.float64))
    eye = torch.eye(dim, dtype=torch.float64) / 3.0
    if not torch.allclose(mom, eye, atol=1e-14):
        raise AssertionError(f"{name}: second moment != cs2*I (bad weight pairing)")
    return st


# ---------------------------------------------------------------------------
# discrete operators
# ---------------------------------------------------------------------------


def _shift(field: torch.Tensor, offsets: tuple[int, ...]) -> torch.Tensor:
    """Return ``g`` with ``g[pos] = field[pos + offsets]`` (periodic).

    Array axes follow the lattice row order reversed: ``(z, y, x)`` in 3D,
    ``(y, x)`` in 2D; ``offsets`` is given in lattice order ``(x, y[, z])``.
    """
    shifts = tuple(-int(c) for c in reversed(offsets))
    dims = tuple(range(field.dim()))
    return torch.roll(field, shifts=shifts, dims=dims)


def _neighbours(field: torch.Tensor, st: Stencil) -> torch.Tensor:
    """Stack ``(Q, *shape)`` of ``field(x + c_i)`` (periodic)."""
    out = torch.empty((st.q,) + tuple(field.shape), dtype=field.dtype, device=field.device)
    for q in range(st.q):
        out[q] = _shift(field, tuple(int(v) for v in st.C[q].tolist()))
    return out


def isotropic_gradient(
    phi: torch.Tensor,
    stencil: Stencil,
    neighbours: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """``grad_a phi = 3 sum_i w_i c_ia phi(x + c_i)`` (isotropic, stencil-exact)."""
    nb = _neighbours(phi, stencil) if neighbours is None else neighbours
    w = stencil.W.to(device=phi.device, dtype=phi.dtype).view(-1, *([1] * phi.dim()))
    out = []
    for a in range(stencil.dim):
        ca = stencil.C[:, a].to(device=phi.device, dtype=phi.dtype).view(-1, *([1] * phi.dim()))
        out.append(3.0 * (w * ca * nb).sum(dim=0))
    return tuple(out)


def isotropic_laplacian(
    phi: torch.Tensor,
    stencil: Stencil,
    neighbours: torch.Tensor | None = None,
) -> torch.Tensor:
    """``lap phi = 6 sum_{i != 0} w_i (phi(x + c_i) - phi)``."""
    nb = _neighbours(phi, stencil) if neighbours is None else neighbours
    w = stencil.W.to(device=phi.device, dtype=phi.dtype).view(-1, *([1] * phi.dim()))
    acc = torch.zeros_like(phi)
    for q in range(stencil.q):
        if not stencil.C[q].any():
            continue
        acc = acc + w[q] * (nb[q] - phi)
    return 6.0 * acc


def interface_normal(
    phi: torch.Tensor,
    stencil: Stencil,
    *,
    neighbours: torch.Tensor | None = None,
    min_gradient: float = 0.0,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Unit normal ``n = grad(phi)/|grad(phi)|`` with a TCLB-style magnitude cutoff."""
    nb = _neighbours(phi, stencil) if neighbours is None else neighbours
    grad = isotropic_gradient(phi, stencil, nb)
    mag = torch.zeros_like(phi)
    for g in grad:
        mag = mag + g * g
    mag = mag.sqrt()
    safe = torch.where(mag > max(min_gradient, 0.0), mag, torch.ones_like(mag))
    n = tuple(g / safe for g in grad)
    return n, mag


def counter_term(
    phi: torch.Tensor,
    n_hat: tuple[torch.Tensor, ...],
    width: float,
    stencil: Stencil,
) -> torch.Tensor:
    """``F_phi,i = w_i (1 - 4 (phi - 1/2)^2) / W (c_i . n)``.

    Identically ``w_i 4 phi (1 - phi) / W (c_i . n)``: the anti-diffusion term of
    the conservative Allen-Cahn equation in population form.  ``sum_i F_phi,i = 0``
    (zero net mass) and ``sum_i c_i F_phi,i = cs^2 (4 phi (1-phi)/W) n``.
    """
    shape = (stencil.q,) + (1,) * phi.dim()
    w = stencil.W.to(device=phi.device, dtype=phi.dtype).view(shape)
    cdotn = torch.zeros(shape, dtype=phi.dtype, device=phi.device)
    for a in range(stencil.dim):
        ca = stencil.C[:, a].to(device=phi.device, dtype=phi.dtype).view(shape)
        cdotn = cdotn + ca * n_hat[a].unsqueeze(0)
    coef = (1.0 - 4.0 * (phi - 0.5) ** 2) / float(width)
    return w * coef.unsqueeze(0) * cdotn


def h_equilibrium(
    phi: torch.Tensor,
    u: tuple[torch.Tensor, ...] | None,
    stencil: Stencil,
    *,
    eq_order: int = 2,
) -> torch.Tensor:
    """``h_i^eq = w_i phi [1 + 3 c_i.u (+ 4.5 (c_i.u)^2 - 1.5 |u|^2)]``."""
    shape = (stencil.q,) + (1,) * phi.dim()
    w = stencil.W.to(device=phi.device, dtype=phi.dtype).view(shape)
    if u is None:
        u = tuple(torch.zeros_like(phi) for _ in range(stencil.dim))
    cdotu = torch.zeros(shape, dtype=phi.dtype, device=phi.device)
    u2 = torch.zeros_like(phi)
    for a in range(stencil.dim):
        ca = stencil.C[:, a].to(device=phi.device, dtype=phi.dtype).view(shape)
        cdotu = cdotu + ca * u[a].unsqueeze(0)
        u2 = u2 + u[a] * u[a]
    eq = 1.0 + 3.0 * cdotu
    if int(eq_order) >= 2:
        eq = eq + 4.5 * cdotu * cdotu - 1.5 * u2.unsqueeze(0)
    return w * (phi.unsqueeze(0) * eq)


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


class CACHPopulation:
    """Conservative Allen-Cahn h-population solver (TCLB-parity recipe).

    Parameters
    ----------
    shape:
        Grid shape, ``(ny, nx)`` for ``d2q9`` and ``(nz, ny, nx)`` for the 3D sets.
    width:
        Anti-diffusivity coefficient ``W`` (TCLB ``IntWidth``).  The realized
        equilibrium interface width is ``3 * mobility * width``.
    mobility:
        TCLB ``M``; the relaxation rate is ``omega = 1/(3M + 1/2)``.
    stencil:
        ``"d2q9"``, ``"d3q19"`` or ``"d3q27"`` (TCLB default lattice is d3q27).
    min_gradient:
        ``|grad(phi)|`` below which the normal is set to zero (TCLB
        ``minGradient``); ``0`` disables the cutoff.
    eq_order:
        ``2`` = TCLB ``calcGamma`` (full ``O(u^2)`` Maxwellian),
        ``1`` = first-order equilibrium used by :mod:`tensorlbm.cac_lbm*`.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        width: float,
        mobility: float,
        stencil: str = "d3q27",
        min_gradient: float = 0.0,
        eq_order: int = 2,
        dtype: torch.dtype = torch.float64,
        device=torch.device("cpu"),
    ) -> None:
        if width <= 0.0:
            raise ValueError("width must be positive")
        if mobility <= 0.0:
            raise ValueError("mobility must be positive")
        if int(eq_order) not in (1, 2):
            raise ValueError("eq_order must be 1 or 2")
        self.st = get_stencil(stencil)
        if len(shape) != self.st.dim:
            raise ValueError(f"shape {shape} does not match stencil dim {self.st.dim}")
        self.shape = tuple(int(s) for s in shape)
        self.width = float(width)
        self.mobility = float(mobility)
        self.omega = 1.0 / (3.0 * self.mobility + 0.5)
        self.min_gradient = float(min_gradient)
        self.eq_order = int(eq_order)
        self.dtype = dtype
        self.device = device
        self.C = self.st.C.to(device=device)
        self.W = self.st.W.to(device=device, dtype=dtype)
        self.h: torch.Tensor | None = None
        self.step_count = 0

    # -- state ---------------------------------------------------------
    @property
    def phi(self) -> torch.Tensor:
        if self.h is None:
            raise RuntimeError("call initialize(phi0) first")
        return self.h.sum(dim=0)

    def initialize(self, phi0: torch.Tensor) -> None:
        phi0 = phi0.to(dtype=self.dtype, device=self.device)
        if tuple(phi0.shape) != self.shape:
            raise ValueError(f"phi0 shape {tuple(phi0.shape)} != {self.shape}")
        view = (self.st.q,) + (1,) * self.st.dim
        self.h = self.W.view(view) * phi0.unsqueeze(0)
        self.step_count = 0

    def mass(self) -> float:
        return float(self.phi.sum().item())

    def first_moment(self) -> tuple[float, ...]:
        """``int phi x / int phi`` in lattice-axis order ``(x, y[, z])``."""
        phi = self.phi
        idx = [torch.arange(n, dtype=self.dtype, device=self.device) for n in phi.shape]
        mesh = torch.meshgrid(*idx, indexing="ij")
        m0 = phi.sum()
        # array axes are (z, y, x) in 3D / (y, x) in 2D -> flip to lattice order
        return tuple(float((phi * g).sum().item() / m0.item()) for g in reversed(mesh))

    # -- one LBE step ---------------------------------------------------
    def collision(self, u: tuple[torch.Tensor, ...] | None = None) -> None:
        phi = self.phi
        nb = _neighbours(phi, self.st)
        n_hat, _ = interface_normal(phi, self.st, neighbours=nb, min_gradient=self.min_gradient)
        f_phi = counter_term(phi, n_hat, self.width, self.st)
        h_eq = h_equilibrium(phi, u, self.st, eq_order=self.eq_order)
        self.h = self.h - self.omega * (self.h - h_eq + 0.5 * f_phi) + f_phi

    def stream(self) -> None:
        """Periodic pull streaming: ``h_i(x, t+1) = h_i^*(x - c_i, t)``."""
        out = torch.empty_like(self.h)
        for q in range(self.st.q):
            out[q] = _shift(self.h[q], tuple(-int(v) for v in self.C[q].tolist()))
        self.h = out

    def step(self, u: tuple[torch.Tensor, ...] | None = None) -> None:
        self.collision(u)
        self.stream()
        self.step_count += 1

    def run(self, n_steps: int, u: tuple[torch.Tensor, ...] | None = None) -> None:
        for _ in range(int(n_steps)):
            self.step(u)


# ---------------------------------------------------------------------------
# analytic helpers shared by the benchmarks
# ---------------------------------------------------------------------------


def equilibrium_profile(width_eff: float) -> float:
    """Slope factor of the tanh interface: ``phi = 1/2 (1 - tanh(2 d / W_eff))``."""
    return float(width_eff)


def predicted_width(width: float, mobility: float, cs2: float = 1.0 / 3.0) -> float:
    """``W_eff = W * (numerical mobility)/cs^2 = W * 3 M`` (see module docstring)."""
    return float(width) * (3.0 * float(mobility) / (3.0 * cs2))


def circle_phi(
    shape: tuple[int, int],
    radius: float,
    width: float,
    center: tuple[float, float] | None = None,
    dtype: torch.dtype = torch.float64,
    device=torch.device("cpu"),
) -> torch.Tensor:
    """Initial condition in TCLB's own convention (``Dynamics.c.Rt`` init)."""
    ny, nx = shape
    cy, cx = center if center is not None else (ny / 2.0, nx / 2.0)
    y = torch.arange(ny, dtype=dtype, device=device).view(-1, 1)
    x = torch.arange(nx, dtype=dtype, device=device).view(1, -1)
    d = radius - ((y - cy) ** 2 + (x - cx) ** 2).sqrt()
    return 0.5 + 0.5 * torch.tanh(2.0 * d / width)


def sphere_phi(
    shape: tuple[int, int, int],
    radius: float,
    width: float,
    center: tuple[float, float, float] | None = None,
    dtype: torch.dtype = torch.float64,
    device=torch.device("cpu"),
) -> torch.Tensor:
    nz, ny, nx = shape
    cz, cy, cx = center if center is not None else (nz / 2.0, ny / 2.0, nx / 2.0)
    z = torch.arange(nz, dtype=dtype, device=device).view(-1, 1, 1)
    y = torch.arange(ny, dtype=dtype, device=device).view(1, -1, 1)
    x = torch.arange(nx, dtype=dtype, device=device).view(1, 1, -1)
    d = radius - ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2).sqrt()
    return 0.5 + 0.5 * torch.tanh(2.0 * d / width)


def update_from_tclb(
    h: torch.Tensor,
    h_eq: torch.Tensor,
    f_phi: torch.Tensor,
    omega: float,
) -> torch.Tensor:
    """The TCLB collision line, kept here as the algebra under test (P9)."""
    return h - omega * (h - h_eq + 0.5 * f_phi) + f_phi


def width_from_profile(
    phi,
    radius: float,
    center: tuple[float, ...] | None = None,
) -> float:
    """Least-squares tanh width ``W`` with ``phi = 1/2 (1 - tanh(2 d/W))``.

    ``d`` is the signed distance to the circle/sphere of the given ``radius``;
    samples are restricted to ``0.05 < phi < 0.95`` (the fitting band).
    """
    dim = phi.dim()
    idx = [torch.arange(n, dtype=phi.dtype, device=phi.device) for n in phi.shape]
    mesh = torch.meshgrid(*idx, indexing="ij")
    if center is None:
        center = tuple(float(n) / 2.0 for n in phi.shape)
    center_rev = tuple(reversed(center))
    r2 = None
    for g, c0 in zip(mesh, center_rev):
        term = (g - c0) ** 2
        r2 = term if r2 is None else r2 + term
    d = radius - r2.sqrt()
    mask = (phi > 0.05) & (phi < 0.95)
    if int(mask.sum().item()) < 4:
        return float("nan")
    phi_s = phi[mask]
    d_s = d[mask]
    # phi = 1/2 (1 - tanh(2 d / W)) -> atanh(1 - 2 phi) = 2 d / W -> linear fit
    arg = (2.0 * phi_s - 1.0).clamp(-0.999999, 0.999999)
    lhs = torch.atanh(arg)
    A = torch.stack([d_s], dim=1)
    sol = torch.linalg.lstsq(A, lhs.unsqueeze(1)).solution
    slope = float(sol[0, 0].item())  # slope = 2/W
    return 2.0 / slope if slope != 0.0 else float("inf")