"""Upgraded 3-D pseudopotential (SCMP) engine: isotropic high-order interaction
gradients and the Kupershtokh exact-difference forcing -- node ``p97w3-n04``.

Motivation
----------
The production Shan-Chen single-component engine discretises the interaction
force with the *lattice* D3Q19 weights folded into the gradient,

    F = -G cs^2 psi grad(psi)                                        (classic)

and folds the force into the equilibrium velocity (``feq(rho, u + tau F/rho)``,
the "velocity shift" scheme).  Two independent upgrades are implemented and
measured here:

1. **Forcing axis -- exact difference method (Kupershtokh 2004/2009).**  Instead
   of shifting the equilibrium velocity by ``tau F/rho``, relax to
   ``feq(rho, u)`` and add ``feq(rho, u + F/rho) - feq(rho, u)``.  This injects
   exactly ``F`` of momentum per step (asserted in the tests) and is free of the
   ``tau``-dependent spurious stress carried by the velocity-shift scheme.
   Nomenclature note: the *spatial* "exact difference" form
   ``psi(x) sum_i W_i c_i [psi(x+c_i) - psi(x)]`` is **algebraically identical**
   to the product form on every symmetric stencil used here, because
   ``sum_i W_i c_i = 0`` identically (also asserted in the tests).  The genuine
   Kupershtokh exact-difference method is therefore a *forcing* scheme, and that
   is the axis this module upgrades.
2. **Isotropy axis -- high-order isotropic gradient (Li, Luo, Li & Kang 2012/13).**
   A gradient ``sum_i W_i c_i psi(x+c_i)`` is isotropic to order ``n`` iff the
   even moment tensors of the weight set up to rank ``n`` are isotropic.  D3Q19
   weights satisfy rank 2 and rank 4; the leading anisotropy is rank 6
   (``sum W c_x^6 - 5 sum W c_x^4 c_y^2 = -2/3``).  Removing it needs further
   shells: this module solves the four isotropy conditions exactly for the
   ``|c|^2 = 1,2,3,4`` set (32 neighbours, weights 2/15, 1/15, 1/60, 1/120) and
   for ``|c|^2 = 1,3,5`` (38 neighbours, weights 1/6, 1/24, 1/120).

Conventions
-----------
* ``(nz, ny, nx)`` field layout, D3Q19 lattice, ``cs^2 = 1/3``.
* ``G_eff < 0`` is attractive (the library convention is ``G = -G_eff``).
* ``p_mech = rho cs^2 + (G_eff/2) cs^2 psi^2`` (Shan-Chen mechanical pressure;
  identical to ``tensorlbm.multiphase_edm.pressure_mech``).
* Viscosity contrast is imposed per cell through
  ``tau(x) = 1/2 + mu(phi(x)) / (rho(x) cs^2)``, so ``lambda = mu_l/mu_v`` is
  independent of the density ratio (cf. ``tensorlbm.two_phase_viscosity``).
* Lattice constants are built from exact rationals: the library's ``d3q19.W``
  constant is float32, and casting it to float64 leaves a ~1e-8 relative weight
  error which breaks mass conservation at the 1e-8 per-collision level -- far
  above the drift this module resolves.
"""

from __future__ import annotations

import inspect
import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np
import torch

from .d3q19 import macroscopic3d
from .hadamard_rybczynski import hr_velocity
from .multiphase import psi_exp
from .multiphase_edm import CS2, pressure_mech
from .solver3d import stream3d

__all__ = [
    "CS2",
    "Stencil3D",
    "SHELLS",
    "STENCIL_NAMES",
    "FORMATS",
    "FORCINGS",
    "isotropic_weights_3d",
    "stencil3d",
    "equilibrium3d_exact",
    "psi_gradient_3d",
    "sc_interaction_force_3d",
    "collide_sc_force_3d",
    "step_sc3d",
    "maxwell_coexistence",
    "droplet_init3d",
    "droplet_state",
    "tau_field_from_mu",
    "static_droplet",
    "settling_droplet",
    "richardson_limit",
    "pressure_mech",
    "psi_exp",
]

# ---------------------------------------------------------------------------
# Exact D3Q19 lattice constants
# ---------------------------------------------------------------------------

#: D3Q19 velocity set in the library's order (rest vector first).
_D3Q19_VECTORS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
)

#: D3Q19 weights from exact rationals (float64, no float32 round trip).
_D3Q19_WEIGHTS: tuple[float, ...] = tuple(
    float(Fraction(v))
    for v in ([Fraction(1, 3)] + [Fraction(1, 18)] * 6 + [Fraction(1, 36)] * 12)
)

_LATTICE_CACHE: dict = {}


def _lattice(device=None, dtype=torch.float64):
    """Cached exact ``(c_x, c_y, c_z, w)`` tensors, each shaped ``(19,1,1,1)``."""
    key = (str(device), str(dtype))
    if key not in _LATTICE_CACHE:
        c = torch.tensor(_D3Q19_VECTORS, dtype=torch.float64, device=device)
        w = torch.tensor(_D3Q19_WEIGHTS, dtype=torch.float64, device=device)
        _LATTICE_CACHE[key] = (
            c[:, 0].to(dtype).view(19, 1, 1, 1),
            c[:, 1].to(dtype).view(19, 1, 1, 1),
            c[:, 2].to(dtype).view(19, 1, 1, 1),
            w.to(dtype).view(19, 1, 1, 1),
        )
    return _LATTICE_CACHE[key]


def equilibrium3d_exact(rho: torch.Tensor, ux, uy, uz) -> torch.Tensor:
    """D3Q19 equilibrium with exact float64 weights (mass conserving to round-off)."""
    cx, cy, cz, w = _lattice(rho.device, rho.dtype)
    u_sq = ux * ux + uy * uy + uz * uz
    cu = cx * ux + cy * uy + cz * uz
    return w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))


# ---------------------------------------------------------------------------
# Gradient stencils and the exact isotropic weight solver
# ---------------------------------------------------------------------------


def _shell(name: str) -> tuple[tuple[int, int, int], ...]:
    """Integer neighbour shells used by the gradient operators."""
    table = {
        "axis1": ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        "diag": tuple(
            v for v in itertools.product((-1, 0, 1), repeat=3) if sum(abs(q) for q in v) == 2
        ),
        "corner": tuple(
            v for v in itertools.product((-1, 0, 1), repeat=3) if sum(abs(q) for q in v) == 3
        ),
        "axis2": ((2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2)),
        "diag21": tuple(
            v
            for v in itertools.product((-2, -1, 0, 1, 2), repeat=3)
            if sum(q * q for q in v) == 5
        ),
    }
    return table[name]


SHELLS: dict[str, tuple[tuple[int, int, int], ...]] = {
    k: _shell(k) for k in ("axis1", "diag", "corner", "axis2", "diag21")
}

#: name -> (shells, isotropy order guaranteed by the solved weights)
STENCIL_SHELLS: dict[str, tuple[tuple[str, ...], int]] = {
    "d3q19": (("axis1", "diag"), 4),
    "iso6_32": (("axis1", "diag", "corner", "axis2"), 6),
    "iso6_38": (("axis1", "corner", "diag21"), 6),
}

STENCIL_NAMES: tuple[str, ...] = tuple(STENCIL_SHELLS)

#: Force-format ladder: baseline (production) -> upgraded (EDM forcing + iso6).
FORMATS: dict[str, tuple[str, str, str]] = {
    "F0_classic": ("d3q19", "product", "velocity_shift"),
    "F1_classic_edmforce": ("d3q19", "product", "kupershtokh"),
    "F2_edm_d3q19": ("d3q19", "difference", "kupershtokh"),
    "F3_edm_iso6": ("iso6_32", "difference", "kupershtokh"),
}


def _moment_rows(shell: Iterable[tuple[int, int, int]]) -> list[Fraction]:
    """The four independent isotropy conditions of one shell.

    Isotropic to rank 6 requires (``M_p`` = shell sum of the monomial):

        rank 2:  M_xx                   = S2   (normalisation, S2 = 1)
        rank 4:  M_xxxx - 3 M_xxyy      = 0
        rank 6:  M_xxxxxx - 5 M_xxxxyy  = 0
        rank 6:  M_xxxxyy - 3 M_xxyyzz  = 0
    """
    v = [(Fraction(c[0]), Fraction(c[1]), Fraction(c[2])) for c in shell]
    m2 = sum(cx * cx for cx, _, _ in v)
    m4a = sum(cx ** 4 for cx, _, _ in v)
    m4b = sum(cx * cx * cy * cy for cx, cy, _ in v)
    m6a = sum(cx ** 6 for cx, _, _ in v)
    m6b = sum(cx ** 4 * cy * cy for cx, cy, _ in v)
    m6c = sum(cx * cx * cy * cy * cz * cz for cx, cy, cz in v)
    return [m2, m4a - 3 * m4b, m6a - 5 * m6b, m6b - 3 * m6c]


def _solve_exact(a: list[list[Fraction]], b: list[Fraction]) -> list[Fraction]:
    """Exact Gaussian elimination (Fraction arithmetic) with singularity check."""
    n = len(a)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if piv is None:
            raise ValueError("isotropy conditions are singular for this shell set")
        aug[col], aug[piv] = aug[piv], aug[col]
        p = aug[col][col]
        aug[col] = [val / p for val in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [aug[r][k] - f * aug[col][k] for k in range(n + 1)]
    return [aug[i][n] for i in range(n)]


def isotropic_weights_3d(shell_names: Sequence[str], max_order: int = 6) -> tuple[float, ...]:
    """Shell weights making ``sum_i W_i c_i psi(x+c_i)`` isotropic to ``max_order``.

    ``max_order=4`` uses the first two conditions (sensible only for exactly two
    shells; it reproduces the D3Q19 ``w/cs^2`` set ``(1/6, 1/12)``).
    ``max_order=6`` uses all four conditions and needs four independent shells
    (or three, when the extra condition is linearly dependent -- the
    ``|c|^2 = 1,3,5`` set).  Normalisation ``sum_i W_i c_x^2 = 1`` makes the
    operator a *gradient* rather than a ``cs^2``-scaled finite difference.
    """
    if max_order not in (4, 6):
        raise ValueError(f"max_order must be 4 or 6, got {max_order}")
    shells = tuple(shell_names)
    n_cond = 2 if max_order == 4 else 4
    rows = [_moment_rows(SHELLS[s]) for s in shells]
    a = [[rows[j][i] for j in range(len(shells))] for i in range(n_cond)]
    b = [Fraction(1)] + [Fraction(0)] * (n_cond - 1)
    if len(shells) == n_cond:
        return tuple(float(x) for x in _solve_exact(a, b))
    if len(shells) < n_cond:
        # fewer shells than conditions: over-determined but solvable when the
        # extra conditions are linearly dependent (the |c|^2 = 1,3,5 case, where
        # the |c|^2 = 2 weight of the 1,2,3,5 solution vanishes identically)
        sub = _solve_exact([row[: len(shells)] for row in a[: len(shells)]], b[: len(shells)])
        resid = [
            sum(Fraction(row[k]) * sub[k] for k in range(len(shells))) - b[i]
            for i, row in enumerate(a)
        ]
        if any(abs(r) > Fraction(1, 10 ** 12) for r in resid):
            raise ValueError(
                f"{len(shells)} shells cannot be made isotropic to order {max_order} "
                f"(residual {[float(r) for r in resid]})"
            )
        return tuple(float(x) for x in sub)
    raise ValueError(
        f"{len(shells)} shells cannot be made isotropic to order {max_order} with a "
        f"unique weight set (need exactly {n_cond} independent shells)"
    )


@dataclass(frozen=True)
class Stencil3D:
    """A gradient stencil: integer vectors, shell weights, guaranteed isotropy order."""

    name: str
    vectors: tuple[tuple[int, int, int], ...]
    weights: tuple[float, ...]
    max_order: int

    @property
    def n_neighbours(self) -> int:
        return len(self.vectors)

    def tensors(self, device=None, dtype=torch.float64):
        vec = torch.tensor(self.vectors, dtype=dtype, device=device)
        wt = torch.tensor(self.weights, dtype=dtype, device=device)
        return vec, wt


def stencil3d(name: str = "d3q19") -> Stencil3D:
    """Build one of :data:`STENCIL_NAMES`."""
    if name not in STENCIL_SHELLS:
        raise ValueError(f"unknown stencil {name!r}; expected one of {STENCIL_NAMES}")
    shells, order = STENCIL_SHELLS[name]
    w = isotropic_weights_3d(shells, max_order=order)
    vectors: list[tuple[int, int, int]] = []
    weights: list[float] = []
    for sh, wsh in zip(shells, w):
        for v in SHELLS[sh]:
            vectors.append(v)
            weights.append(wsh)
    return Stencil3D(name=name, vectors=tuple(vectors), weights=tuple(weights), max_order=order)


# ---------------------------------------------------------------------------
# Gradient and interaction force
# ---------------------------------------------------------------------------


def _shift(field: torch.Tensor, c: Sequence[int], periodic: bool = True) -> torch.Tensor:
    """``field(x + c)`` for ``c = (cx, cy, cz)``, dims ``(..., nz, ny, nx)``."""
    cx, cy, cz = int(c[0]), int(c[1]), int(c[2])
    if periodic:
        return torch.roll(field, shifts=(-cz, -cy, -cx), dims=(-3, -2, -1))
    pad = torch.nn.functional.pad(
        field,
        (max(0, cx), max(0, -cx), max(0, cy), max(0, -cy), max(0, cz), max(0, -cz)),
        mode="constant",
        value=0.0,
    )
    nx, ny, nz = field.shape[-1], field.shape[-2], field.shape[-3]
    x0, y0, z0 = max(0, -cx), max(0, -cy), max(0, -cz)
    return pad[..., z0:z0 + nz, y0:y0 + ny, x0:x0 + nx]


def _neighbour_stack(psi: torch.Tensor, st: Stencil3D, periodic: bool = True) -> torch.Tensor:
    """All ``psi(x + c_i)`` in one tensor of shape ``(N, nz, ny, nx)``."""
    return torch.stack([_shift(psi, v, periodic) for v in st.vectors], dim=0)


def psi_gradient_3d(
    psi: torch.Tensor,
    stencil: str | Stencil3D = "d3q19",
    form: str = "difference",
    periodic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Discrete gradient of a scalar field on the requested stencil.

    ``form="product"``     ``g = sum_i W_i c_i psi(x+c_i)``  (classic)
    ``form="difference"``  ``g = sum_i W_i c_i [psi(x+c_i) - psi(x)]``

    On every symmetric stencil of this module ``sum_i W_i c_i = 0``, so the two
    forms are algebraically identical (asserted in the tests); ``"difference"``
    is kept for parity with the 2-D module and for stencils whose weight sum
    does not vanish.
    """
    if form not in ("product", "difference"):
        raise ValueError(f"psi_gradient_3d supports form='product'|'difference', got {form!r}")
    st = stencil3d(stencil) if isinstance(stencil, str) else stencil
    vec, wt = st.tensors(device=psi.device, dtype=psi.dtype)
    stack = _neighbour_stack(psi, st, periodic)
    if form == "difference":
        stack = stack - psi.unsqueeze(0)
    wvec = wt.view(-1, 1) * vec  # (N, 3)
    g = torch.einsum("nc,nzyx->czyx", wvec, stack)
    return g[0], g[1], g[2]


def sc_interaction_force_3d(
    psi: torch.Tensor,
    G_eff: float,
    stencil: str | Stencil3D = "d3q19",
    form: str = "difference",
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    rho: torch.Tensor | None = None,
    periodic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shan-Chen interaction force (+ optional body force ``g * rho``).

    ``form="product"``     ``F = -G cs^2 psi grad psi``
    ``form="difference"``  ``F = -G cs^2 psi * sum_i W_i c_i [psi(x+c_i)-psi(x)]``
    ``form="psi2"``        ``F = -(G/2) cs^2 grad(psi^2)`` (pressure-tensor form:
                           exactly curl free, hence exactly momentum conserving)
    """
    if form == "psi2":
        dpx, dpy, dpz = psi_gradient_3d(psi * psi, stencil, "product", periodic)
        Fx = -0.5 * G_eff * CS2 * dpx
        Fy = -0.5 * G_eff * CS2 * dpy
        Fz = -0.5 * G_eff * CS2 * dpz
    elif form in ("product", "difference"):
        dpx, dpy, dpz = psi_gradient_3d(psi, stencil, form, periodic)
        Fx = -G_eff * CS2 * psi * dpx
        Fy = -G_eff * CS2 * psi * dpy
        Fz = -G_eff * CS2 * psi * dpz
    else:
        raise ValueError(f"unknown form {form!r}")
    if gx or gy or gz:
        r = rho if rho is not None else torch.ones_like(psi)
        Fx = Fx + gx * r
        Fy = Fy + gy * r
        Fz = Fz + gz * r
    return Fx, Fy, Fz


# ---------------------------------------------------------------------------
# Collision: EDM (Kupershtokh) vs velocity shift vs Guo
# ---------------------------------------------------------------------------

FORCINGS = ("kupershtokh", "velocity_shift", "guo")


def collide_sc_force_3d(
    f: torch.Tensor,
    Fx: torch.Tensor,
    Fy: torch.Tensor,
    Fz: torch.Tensor,
    tau: float | torch.Tensor = 1.0,
    forcing: str = "kupershtokh",
) -> torch.Tensor:
    """D3Q19 BGK collision with a body force under three forcing schemes.

    ``forcing="kupershtokh"``  exact-difference method (Kupershtokh 2004/2009):
        relax to ``feq(rho,u)`` and add ``feq(rho,u+F/rho) - feq(rho,u)``;
        momentum exact (``sum_i c_i df_i = F``) and ``tau`` independent.
    ``forcing="velocity_shift"``  ``feq(rho, u + tau F/rho)`` (historical default;
        carries a ``tau``-dependent spurious stress).
    ``forcing="guo"``  Guo (2002) source term with the half-force velocity.

    ``tau`` may be a scalar or a per-cell field ``(nz,ny,nx)`` (viscosity contrast).
    """
    cx, cy, cz, w = _lattice(f.device, f.dtype)
    rho = f.sum(0)
    rho_s = torch.clamp(rho, min=1e-12)
    ux = (f * cx).sum(0) / rho_s
    uy = (f * cy).sum(0) / rho_s
    uz = (f * cz).sum(0) / rho_s
    tau_t: float | torch.Tensor = torch.clamp(tau, min=0.5005) if torch.is_tensor(tau) else tau
    cu = cx * ux + cy * uy + cz * uz
    u_sq = ux * ux + uy * uy + uz * uz
    feq_u = w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))
    if forcing == "kupershtokh":
        dux, duy, duz = Fx / rho_s, Fy / rho_s, Fz / rho_s
        A = cx * dux + cy * duy + cz * duz
        u_du = ux * dux + uy * duy + uz * duz
        du_sq = dux * dux + duy * duy + duz * duz
        delta = w * rho * (
            3.0 * A + 4.5 * (2.0 * A * cu + A * A) - 1.5 * (2.0 * u_du + du_sq).unsqueeze(0)
        )
        return f - (f - feq_u) / tau_t + delta
    if forcing == "velocity_shift":
        usx = ux + tau_t * Fx / rho_s
        usy = uy + tau_t * Fy / rho_s
        usz = uz + tau_t * Fz / rho_s
        cus = cx * usx + cy * usy + cz * usz
        us_sq = usx * usx + usy * usy + usz * usz
        feq = w * rho * (1.0 + 3.0 * cus + 4.5 * cus * cus - 1.5 * us_sq.unsqueeze(0))
        return f - (f - feq) / tau_t
    if forcing == "guo":
        uxp, uyp, uzp = ux + 0.5 * Fx / rho_s, uy + 0.5 * Fy / rho_s, uz + 0.5 * Fz / rho_s
        cus = cx * uxp + cy * uyp + cz * uzp
        us_sq = uxp * uxp + uyp * uyp + uzp * uzp
        feq = w * rho * (1.0 + 3.0 * cus + 4.5 * cus * cus - 1.5 * us_sq.unsqueeze(0))
        cf = cx * Fx.unsqueeze(0) + cy * Fy.unsqueeze(0) + cz * Fz.unsqueeze(0)
        uf = (uxp * Fx + uyp * Fy + uzp * Fz).unsqueeze(0)
        src = w * (1.0 - 0.5 / tau_t) * ((cf - uf) / CS2 + cus * cf / CS2 ** 2)
        return f - (f - feq) / tau_t + src
    raise ValueError(f"unknown forcing {forcing!r}; expected one of {FORCINGS}")


def step_sc3d(
    f: torch.Tensor,
    G_eff: float,
    tau: float | torch.Tensor = 1.0,
    stencil: str | Stencil3D = "d3q19",
    form: str = "difference",
    forcing: str = "kupershtokh",
    gz: float = 0.0,
    gravity_mode: str = "density",
    rho_ref: float | None = None,
    psi_fn=None,
) -> torch.Tensor:
    """One collide-and-stream step of the upgraded SCMP scheme.

    ``gravity_mode="density"``  body force ``(rho - rho_ref) gz`` -- buoyancy with
        the mean subtracted, so a periodic box gains no net momentum and a
        terminal settling state exists.
    ``gravity_mode="none"``     pure interaction force.
    """
    psi_fn = psi_exp if psi_fn is None else psi_fn
    rho = macroscopic3d(f)[0]
    psi = psi_fn(rho)
    Fx, Fy, Fz = sc_interaction_force_3d(psi, G_eff, stencil, form)
    if gravity_mode == "density":
        if gz != 0.0:
            r0 = float(rho.mean()) if rho_ref is None else float(rho_ref)
            Fz = Fz + gz * (rho - r0)
    elif gravity_mode != "none":
        raise ValueError(f"unknown gravity_mode {gravity_mode!r}")
    f = collide_sc_force_3d(f, Fx, Fy, Fz, tau, forcing)
    return stream3d(f)

# ---------------------------------------------------------------------------
# EOS coexistence (Maxwell construction)
# ---------------------------------------------------------------------------


def _accepts_rho0(fn) -> bool:
    try:
        return "rho0" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _p_mech_scalar(r, G_eff: float, psi_fn, rho0: float = 1.0):
    r_t = torch.as_tensor(np.asarray(r, dtype=np.float64), dtype=torch.float64)
    psi = psi_fn(r_t, rho0) if _accepts_rho0(psi_fn) else psi_fn(r_t)
    return CS2 * np.asarray(r, dtype=float) + 0.5 * G_eff * CS2 * np.asarray(psi) ** 2


def maxwell_coexistence(
    G_eff: float,
    rho0: float = 1.0,
    psi_fn=None,
    brackets: tuple[float, float] = (0.02, 2.5),
) -> dict:
    """Coexistence densities of the Shan-Chen EOS by equal-area (Maxwell) construction.

    ``p(rho) = cs^2 rho + (G_eff/2) cs^2 psi(rho)^2``; the pair ``(rho_v, rho_l)``
    satisfies ``p(rho_v) = p(rho_l)`` and ``int_{v_v}^{v_l} (p - p_sat) dv = 0``
    with ``v = 1/rho``.  The trivial root ``rho_v == rho_l`` (which satisfies both
    conditions identically) is excluded: the pair with the largest density ratio
    is returned.
    """
    from scipy.integrate import quad
    from scipy.optimize import root

    psi_fn = psi_exp if psi_fn is None else psi_fn

    def p_of(r: float) -> float:
        return float(_p_mech_scalar(r, G_eff, psi_fn, rho0))

    def residual(x):
        rv, rl = float(x[0]), float(x[1])
        if not (brackets[0] < rv < rl < brackets[1]) or (rl - rv) < 1e-3:
            return [1e3 * (rl - rv) + 1e3, 1e3]
        f = lambda r: (p_of(r) - p_of(rl)) / r ** 2  # noqa: E731
        area = quad(f, rv, rl, limit=400)[0]
        return [p_of(rl) - p_of(rv), 1e2 * area]

    best = None
    for guess in ((0.2, 1.5), (0.15, 1.9), (0.3, 1.2), (0.1, 2.0)):
        sol = root(residual, list(guess))
        rv, rl = float(sol.x[0]), float(sol.x[1])
        if sol.success and brackets[0] < rv < rl - 1e-3 < brackets[1]:
            res = residual([rv, rl])
            err = max(abs(res[0]), abs(res[1]))
            if err < 1e-6 and (best is None or rl / rv > best[3]):
                best = (err, rv, rl, rl / rv)
    if best is None:
        return {
            "rho_v": float("nan"),
            "rho_l": float("nan"),
            "p_sat": float("nan"),
            "density_ratio": float("nan"),
            "converged": False,
        }
    err, rv, rl, ratio = best
    return {
        "rho_v": rv,
        "rho_l": rl,
        "p_sat": p_of(rl),
        "density_ratio": ratio,
        "converged": bool(err < 1e-6),
        "residual": err,
    }


# ---------------------------------------------------------------------------
# Droplet initialisation and diagnostics
# ---------------------------------------------------------------------------


def _grid(n: int, device=None, dtype=torch.float64):
    z = torch.arange(n, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, z, z, indexing="ij")
    c = 0.5 * (n - 1)
    r = torch.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2)
    return r, c


def droplet_init3d(
    n: int,
    R: float,
    rho_l: float,
    rho_v: float,
    width: float = 2.0,
    device=None,
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rest D3Q19 distribution of a tanh liquid droplet in its vapour.

    ``width`` is the tanh half-width in lattice units; the equilibrium interface
    width depends on ``G_eff``, so a prescribed profile only sets the transient
    the run relaxes through.
    """
    r, _ = _grid(n, device, dtype)
    phi = 0.5 * (1.0 - torch.tanh((r - R) / width))
    rho = rho_v + (rho_l - rho_v) * phi
    zero = torch.zeros_like(rho)
    return equilibrium3d_exact(rho, zero, zero, zero), r


def droplet_state(f: torch.Tensor) -> dict:
    """Volume-equivalent radius, bulk densities, velocity field and ``|u|max``."""
    rho, ux, uy, uz = macroscopic3d(f)
    rho_mid = 0.5 * (float(rho.max()) + float(rho.min()))
    inside = rho > rho_mid
    v = float(inside.sum())
    r_eq = (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0) if v > 0 else float("nan")
    return {
        "rho": rho,
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "rho_l": float(rho[inside].mean()) if v > 0 else float("nan"),
        "rho_v": float(rho[~inside].mean()),
        "R_eq": r_eq,
        "u_max": float(torch.sqrt(ux * ux + uy * uy + uz * uz).max()),
        "inside": inside,
    }


def tau_field_from_mu(
    rho: torch.Tensor,
    rho_l: float,
    rho_v: float,
    mu_l: float,
    mu_v: float,
) -> torch.Tensor:
    """Per-cell relaxation time realising ``mu_l`` / ``mu_v`` in the two bulks.

    ``tau(x) = 1/2 + mu(phi(x)) / (rho(x) cs^2)`` with
    ``phi = clip((rho - rho_v)/(rho_l - rho_v), 0, 1)`` and ``mu`` linear in
    ``phi``.  Because the BGK viscosity of a bulk is ``rho cs^2 (tau - 1/2)``,
    this keeps ``lambda = mu_l/mu_v`` independent of the density ratio.
    """
    phi = ((rho - rho_v) / (rho_l - rho_v)).clamp(0.0, 1.0)
    mu = mu_v + (mu_l - mu_v) * phi
    rho_s = torch.clamp(rho, min=1e-12)
    return 0.5 + mu / (rho_s * CS2)


# ---------------------------------------------------------------------------
# Static droplet: Young-Laplace law and spurious currents
# ---------------------------------------------------------------------------


def static_droplet(
    R: float = 8.0,
    n: int | None = None,
    G_eff: float = -5.0,
    tau: float = 1.0,
    stencil: str = "d3q19",
    form: str = "difference",
    forcing: str = "kupershtokh",
    rho_l: float | None = None,
    rho_v: float | None = None,
    width: float = 2.0,
    steps: int = 3000,
    check_every: int = 100,
    steady_tol: float = 1e-4,
    steady_window: int = 3,
    thread_count: int = 4,
    dtype=torch.float64,
) -> dict:
    """March a static droplet to steady state; measure Laplace pressure and |u|max.

    ``dp`` is the difference of the Shan-Chen mechanical pressure between the
    drop core (``r < 0.6 R_eq``) and the far field (``r > 1.5 R_eq``) -- the
    interface band is excluded from both because ``p`` dips strongly negative
    inside it.  ``sigma = dp R_eq / 2`` follows the 3-D law ``dp = 2 sigma / R``.
    """
    if rho_l is None or rho_v is None:
        coex = maxwell_coexistence(G_eff)
        rho_l = coex["rho_l"] if rho_l is None else rho_l
        rho_v = coex["rho_v"] if rho_v is None else rho_v
    n = int(max(round(4 * R) + 8, 32)) if n is None else int(n)
    torch.set_num_threads(thread_count)
    f, r_grid = droplet_init3d(n, R, rho_l, rho_v, width, dtype=dtype)
    mass0 = float(f.sum()) / n ** 3
    hist: list[dict] = []
    prev = None
    n_steady = 0
    converged = False
    step = 0
    while step < steps:
        for _ in range(min(check_every, steps - step)):
            f = step_sc3d(f, G_eff, tau, stencil, form, forcing)
        step += check_every
        st = droplet_state(f)
        psi = psi_exp(st["rho"])
        p = pressure_mech(st["rho"], psi, G_eff)
        core = r_grid < 0.6 * st["R_eq"]
        far = r_grid > 1.5 * st["R_eq"]
        dp = float(p[core].mean() - p[far].mean()) if bool(far.any()) else float("nan")
        u = torch.sqrt(st["ux"] ** 2 + st["uy"] ** 2 + st["uz"] ** 2)
        band = (r_grid - st["R_eq"]).abs() < 2.0 * width
        rec = {
            "step": step,
            "dp": dp,
            "R_eq": st["R_eq"],
            "u_max": float(u.max()),
            "u_max_iface": float(u[band].max()) if bool(band.any()) else float("nan"),
            "u_max_core": float(u[core].max()) if bool(core.any()) else float("nan"),
            "mass": float(f.sum()) / n ** 3,
            "rho_min": float(st["rho"].min()),
            "rho_max": float(st["rho"].max()),
        }
        hist.append(rec)
        if not (math.isfinite(rec["dp"]) and math.isfinite(rec["u_max"])):
            break
        if prev is not None and abs(dp - prev) / max(abs(dp), 1e-30) < steady_tol:
            n_steady += 1
        else:
            n_steady = 0
        prev = dp
        if n_steady >= steady_window:
            converged = True
            break
    tail = hist[-steady_window:]
    dp_m = float(np.mean([h["dp"] for h in tail]))
    req_m = float(np.mean([h["R_eq"] for h in tail]))
    return {
        "R0": R,
        "n": n,
        "G_eff": G_eff,
        "G_lib": -G_eff,
        "tau": tau,
        "stencil": stencil,
        "form": form,
        "forcing": forcing,
        "rho_l": rho_l,
        "rho_v": rho_v,
        "dp": dp_m,
        "R_eq": req_m,
        "sigma": 0.5 * dp_m * req_m,
        "u_max": float(np.mean([h["u_max"] for h in tail])),
        "u_max_iface": float(np.nanmean([h["u_max_iface"] for h in tail])),
        "u_max_core": float(np.nanmean([h["u_max_core"] for h in tail])),
        "mass_drift_rel": abs(float(f.sum()) / n ** 3 - mass0) / mass0,
        "rho_min": min(h["rho_min"] for h in hist),
        "rho_max": max(h["rho_max"] for h in hist),
        "steps": step,
        "converged": converged,
        "finite": bool(all(math.isfinite(h["dp"]) for h in hist)),
        "history": hist,
    }


# ---------------------------------------------------------------------------
# Buoyant droplet settling vs Hadamard-Rybczynski
# ---------------------------------------------------------------------------


def settling_droplet(
    R: float = 8.0,
    n: int | None = None,
    lam: float = 1.0,
    G_eff: float = -5.0,
    g: float = 1.2e-6,
    tau_v: float = 1.5,
    rho_l: float | None = None,
    rho_v: float | None = None,
    stencil: str = "d3q19",
    form: str = "difference",
    forcing: str = "kupershtokh",
    width: float = 2.0,
    steps: int = 6000,
    sample_every: int = 100,
    thread_count: int = 4,
    dtype=torch.float64,
) -> dict:
    """Settle a buoyant liquid droplet and compare its terminal speed with HR.

    Gravity acts along ``-z`` as ``(rho - <rho>) g``: the mean part is removed so
    the periodic box gains no net momentum and a terminal *relative* velocity
    exists.  ``U_rel`` is the drop's rigid-body drift (mean over the core shell
    ``r < 0.6 R_eq``, where the spurious interface recirculation is weak) minus
    the far-field (``r > 1.5 R_eq``) drift.

    ``lam = mu_l/mu_v`` is imposed through the per-cell relaxation time;
    ``mu_v = rho_v cs^2 (tau_v - 1/2)`` sets the dynamic viscosity scale.
    """
    if rho_l is None or rho_v is None:
        coex = maxwell_coexistence(G_eff)
        rho_l = coex["rho_l"] if rho_l is None else rho_l
        rho_v = coex["rho_v"] if rho_v is None else rho_v
    n = int(max(5 * R, 32)) if n is None else int(n)
    torch.set_num_threads(thread_count)
    mu_v = rho_v * CS2 * (tau_v - 0.5)
    mu_l = lam * mu_v
    f, r_grid = droplet_init3d(n, R, rho_l, rho_v, width, dtype=dtype)
    rho0 = float(macroscopic3d(f)[0].mean())
    mass0 = float(f.sum()) / n ** 3
    hist: list[dict] = []
    t0 = __import__("time").time()
    u_prev = None
    stalled = 0
    for step in range(1, steps + 1):
        rho = macroscopic3d(f)[0]
        tau_cell = tau_field_from_mu(rho, rho_l, rho_v, mu_l, mu_v)
        f = step_sc3d(
            f, G_eff, tau_cell, stencil, form, forcing,
            gz=-abs(g), gravity_mode="density", rho_ref=rho0,
        )
        if step % sample_every == 0 or step == steps:
            st = droplet_state(f)
            inside = st["inside"]
            phi = ((st["rho"] - rho_v) / (rho_l - rho_v)).clamp(0.0, 1.0)
            core = r_grid < 0.6 * st["R_eq"]
            w = phi[core]
            u_drop = float((st["uz"][core] * w).sum() / torch.clamp(w.sum(), min=1e-12))
            far = r_grid > 1.5 * st["R_eq"]
            u_amb = float(st["uz"][far].mean()) if bool(far.any()) else 0.0
            drho_eff = float((st["rho"][inside] - rho0).mean())
            rec = {
                "step": step,
                "U_rel": u_drop - u_amb,
                "u_drop": u_drop,
                "u_amb": u_amb,
                "R_eq": st["R_eq"],
                "rho_l_in": st["rho_l"],
                "rho_v_out": st["rho_v"],
                "drho_eff": drho_eff,
                "u_max": st["u_max"],
                "mass_drift_rel": abs(float(f.sum()) / n ** 3 - mass0) / mass0,
            }
            hist.append(rec)
            if not (math.isfinite(rec["U_rel"]) and math.isfinite(rec["R_eq"])):
                break
            if u_prev is not None and abs(rec["U_rel"] - u_prev) / max(abs(u_prev), 1e-30) < 2e-3:
                stalled += 1
            else:
                stalled = 0
            u_prev = rec["U_rel"]
            if stalled >= 3:
                break
    if not hist:
        raise RuntimeError("settling_droplet produced no sample (diverged immediately)")
    tail = hist[len(hist) // 2:]
    U_rel = float(np.mean([h["U_rel"] for h in tail]))
    R_eq = float(np.mean([h["R_eq"] for h in tail]))
    drho = float(np.mean([h["drho_eff"] for h in tail]))
    lam_meas = mu_l / mu_v
    U_hr = float(hr_velocity(drho, abs(g), R_eq, mu_v, lam_meas))
    nu_v = CS2 * (tau_v - 0.5)
    return {
        "R0": R,
        "n": n,
        "lam_nominal": lam,
        "lam_measured": lam_meas,
        "tau_v": tau_v,
        "tau_l": 0.5 + mu_l / (rho_l * CS2),
        "mu_v": mu_v,
        "mu_l": mu_l,
        "nu_v": nu_v,
        "Re_v": abs(U_rel) * R_eq / nu_v if nu_v > 0 else float("nan"),
        "G_eff": G_eff,
        "g": -abs(g),
        "stencil": stencil,
        "form": form,
        "forcing": forcing,
        "rho_l": rho_l,
        "rho_v": rho_v,
        "steps_run": hist[-1]["step"],
        "U_rel": U_rel,
        "U_rel_spread": float(np.std([h["U_rel"] for h in tail])),
        "R_eq": R_eq,
        "drho_eff": drho,
        "U_hr": U_hr,
        "rel_err_vs_hr": (abs(U_rel) / abs(U_hr) - 1.0) if U_hr else float("nan"),
        "u_max": float(np.mean([h["u_max"] for h in tail])),
        "u_max_over_U_hr": (float(np.mean([h["u_max"] for h in tail])) / abs(U_hr))
        if U_hr else float("nan"),
        "mass_drift_rel": hist[-1]["mass_drift_rel"],
        "finite": bool(all(math.isfinite(h["U_rel"]) for h in hist)),
        "elapsed_s": round(__import__("time").time() - t0, 1),
        "history": hist,
    }


# ---------------------------------------------------------------------------
# Convergence / extrapolation helpers
# ---------------------------------------------------------------------------


def richardson_limit(h: Sequence[float], y: Sequence[float], grid: int = 2001) -> dict:
    """Richardson extrapolation ``y(h) = y0 + A h^p`` with ``p`` fitted.

    ``h`` is the mesh spacing (``h ~ 1/R`` for a fixed-domain-refinement study).
    ``p`` minimises the least-squares residual over a scan; the return value
    reports the fitted order, the extrapolated limit ``y0``, the coefficient of
    determination and (for exactly three points) the classic three-point estimate
    of the observed order.
    """
    h_arr = np.asarray(h, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if h_arr.size != y_arr.size or h_arr.size < 3:
        raise ValueError("need at least three (h, y) pairs")
    best = None
    for p in np.linspace(0.1, 6.0, grid):
        A = np.vstack([np.ones_like(h_arr), h_arr ** p]).T
        coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
        resid = y_arr - A @ coef
        ss = float((resid ** 2).sum())
        if best is None or ss < best[0]:
            best = (ss, float(p), float(coef[0]), float(coef[1]))
    ss, p, y0, amp = best
    ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum())
    r2 = 1.0 - ss / ss_tot if ss_tot > 0 else float("nan")
    p3 = float("nan")
    if h_arr.size == 3:
        d1 = y_arr[1] - y_arr[0]
        d2 = y_arr[2] - y_arr[1]
        r = h_arr[1] / h_arr[0]
        if abs(d2) > 0 and abs(d1) > 0 and r > 1:
            ratio = d1 / d2
            if ratio > 0:
                p3 = float(np.log(ratio) / np.log(r))
    return {
        "p": p,
        "y0": y0,
        "amplitude": amp,
        "r2": r2,
        "p_three_point": p3,
        "n_points": int(h_arr.size),
        "h": h_arr.tolist(),
        "y": y_arr.tolist(),
    }
