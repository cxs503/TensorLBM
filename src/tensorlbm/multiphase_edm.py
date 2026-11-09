"""Exact-difference pseudopotential force + high-order isotropic gradients (SCMP).

This module upgrades the *interaction-force discretisation* of the single-
component Shan–Chen (SCMP) model.  It is orthogonal to the *forcing scheme*
(how the force enters the collision, ``velocity_shift`` / Kupershtokh-EDM /
Guo), which already lives in :mod:`tensorlbm.multiphase`.

Two independent axes are provided
---------------------------------
``form`` — how the directional difference is assembled
    ``"product"``      Σ_i W_i c_i ψ(x+c_i)              (classic Shan–Chen)
    ``"difference"``   Σ_i W_i c_i [ψ(x+c_i) − ψ(x)]     (exact difference)
    ``"psi2"``         −(G/2)Σ_i W_i c_i ψ²(x+c_i)       (pressure-tensor /
                       divergence form: an exact discrete gradient of ψ²,
                       hence exactly curl-free and exactly momentum conserving)

``stencil`` — which neighbour set / weights are used (see :func:`stencil`)

Why the ``"difference"`` form is not merely cosmetic
----------------------------------------------------
On a *symmetric, complete* stencil Σ_i W_i c_i = 0 identically, so

    ψ(x) Σ_i W_i c_i [ψ(x+c_i) − ψ(x)]  ≡  ψ(x) Σ_i W_i c_i ψ(x+c_i)

i.e. the exact-difference form is **algebraically identical** to the classic
product form (asserted bit-exactly in ``tests/test_multiphase_edm.py``).  The
two differ — and the difference form is strictly better — as soon as the
stencil is *incomplete/inhomogeneous*, which is precisely what happens at
walls and on truncated stencils: there Σ_i W_i c_i ≠ 0 and the product form
then produces a non-zero force in a perfectly uniform fluid (a spurious wall
force, the seed of wall spurious currents and contact-angle artefacts) while
the difference form still returns exactly zero, because the *interaction* of
a uniform field with itself must vanish by symmetry.  The difference form is
therefore the stencil-independent ("EOS-covariant") way of writing the SC
force, and it is the form that survives when the stencil is extended.

Why the stencil must be extended (the isotropy result)
-----------------------------------------------------
A gradient operator ``g_α ψ = Σ_i W_i c_iα ψ(x+c_i)`` is isotropic to order n
iff the even moment tensors of the weight set up to rank n are isotropic.  On
the 8-neighbour D2Q9 set the two conditions

    Σ W c_x²  = 1                        (2nd order)
    Σ W c_x⁴  = 3 Σ W c_x² c_y²          (4th order)

have the **unique** solution {axis |c|=1: 1/3, diagonal |c|=√2: 1/12}, i.e.
exactly the standard D2Q9 weights (``w_i/cs²``).  The 8-neighbour stencil is
therefore already 4th-order isotropic and there is *no* freedom left in it;
the leading anisotropy of the classic SC gradient is 6th order
(Σ W c_x⁶ ≠ 5 Σ W c_x⁴ c_y², residual −2/3 for D2Q9).  Removing it requires
the next shell — this is the "8-neighbourhood + 16-neighbourhood" mixture:

    ``"iso6"``     |c| = 1, √2, 2                    (12 neighbours, 3 shells)
    ``"iso6_25"``  |c| = 1, √2, 2, √5   ("8 + 16")   (24 neighbours, 4 shells)

with weights obtained from the linear moment conditions
(``isotropic_weights``).  The measured reduction of the interface anisotropy
is reported in ``benchmarks/pending/pseudopotential_edm_upgrade``.

Conventions
-----------
*Physical (this module)*::

    F(x) = −G_eff · cs² · ψ(x) · ∇ψ(x)          (G_eff < 0 → attraction)
    p_mech = ρ cs² + (G_eff/2) cs² ψ² = ρ/3 + (G_eff/6) ψ²

*Library* (``tensorlbm.multiphase``, backward-gather convention)::

    F_lib = −G_lib ψ(x) Σ_i w_i c_i ψ(x−c_i),   G_eff = −G_lib,  Σ_i w_i c_i = cs² ∇

so ``G_lib = −G_eff``; use :func:`library_coupling` / :func:`physical_coupling`.
With ``stencil="d2q9", form="product"`` this module reproduces
``multiphase.sc_single_component_force(..., scheme="standard")`` and with
``form="psi2"`` it reproduces ``scheme="pressure_tensor"`` (both asserted in
the tests).

References
----------
Kupershtokh (2004) *Exact difference method*; Sbragaglia et al. (2007) PRE 75
026702 (multi-range potentials, higher-order isotropy); Lee & Lin (2005) JCP
206 16 (high-order isotropic finite differences for two-phase LBM);
Shan (2006) PRE 73 047701 (isotropic stencils for the pseudopotential force).
"""

from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass

import torch

from .d2q9 import C as _D2Q9_C
from .d2q9 import W as _D2Q9_W
from .d2q9 import equilibrium, macroscopic
from .solver import stream

CS2 = 1.0 / 3.0

__all__ = [
    "CS2",
    "Stencil",
    "stencil",
    "stencil_names",
    "isotropic_weights",
    "psi_gradient",
    "sc_interaction_force",
    "collide_sc_force",
    "stream_step",
    "pressure_mech",
    "library_coupling",
    "physical_coupling",
    "FORMS",
    "FORCINGS",
]


# ---------------------------------------------------------------------------
# Stencils
# ---------------------------------------------------------------------------

FORMS = ("product", "difference", "psi2")
FORCINGS = ("velocity_shift", "kupershtokh", "guo")


def _shell(vectors: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return list(vectors)


_SHELLS: dict[str, list[tuple[int, int]]] = {
    # |c| = 1, axial
    "axis1": [(1, 0), (-1, 0), (0, 1), (0, -1)],
    # |c| = sqrt(2), diagonal (the rest of the 8-neighbourhood)
    "diag": [(1, 1), (-1, 1), (-1, -1), (1, -1)],
    # |c| = 2, axial
    "axis2": [(2, 0), (-2, 0), (0, 2), (0, -2)],
    # |c| = sqrt(5)  –  the 16-neighbourhood of the 25-point stencil
    "l21": [(2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)],
}

_STENCIL_SHELLS: dict[str, tuple[str, ...]] = {
    "d2q9": ("axis1", "diag"),
    "iso6": ("axis1", "diag", "axis2"),
    "iso6_25": ("axis1", "diag", "axis2", "l21"),
}


def _moment_rows(shells: tuple[str, ...]) -> list[list[float]]:
    """Moment-condition rows (one row per condition, one column per shell).

    Conditions for an isotropic gradient operator (Σ W c_x² = 1 normalisation):

    * 2nd order: ``Σ W c_x² = 1``
    * 4th order: ``Σ W c_x⁴ − 3 Σ W c_x² c_y² = 0``
    * 6th order: ``Σ W c_x⁶ − 5 Σ W c_x⁴ c_y² = 0``

    The numerical factors 3 and 5 are the 2-D isotropic tensor contractions
    (T_xxxx = 3 T_xxyy for rank 4, T_xxxxxx = 5 T_xxxxyy for rank 6).
    """
    rows: list[list[float]] = [[], [], []]
    for name in shells:
        vecs = _SHELLS[name]
        m2 = sum(cx * cx for cx, cy in vecs)
        m4 = sum(cx**4 for cx, cy in vecs) - 3 * sum(cx * cx * cy * cy for cx, cy in vecs)
        m6 = sum(cx**6 for cx, cy in vecs) - 5 * sum(cx**4 * cy * cy for cx, cy in vecs)
        rows[0].append(float(m2))
        rows[1].append(float(m4))
        rows[2].append(float(m6))
    return rows


def _inv(mat: list[list[float]]) -> list[list[float]]:
    """Gauss–Jordan inverse of a small square matrix (float64, pivoted)."""
    n = len(mat)
    aug = [list(map(float, mat[i])) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-14:
            raise ValueError("singular moment system")
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            f = aug[r][col]
            if f:
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def isotropic_weights(shells: tuple[str, ...], max_order: int = 6) -> tuple[float, ...]:
    """Per-shell weights making the gradient operator isotropic up to ``max_order``.

    The isotropy conditions are *linear* in the weights, so:

    * ``A`` square (e.g. the three-shell ``iso6`` set with 3 conditions) — the
      solution is unique: ``(4/15, 1/10, 1/120)`` for ``|c| = 1, √2, 2``;
    * ``A`` over-determined in the unknowns' null space (e.g. the four-shell
      ``iso6_25`` set: 4 shells, 3 conditions) — the minimum-norm solution
      ``w = Aᵀ(AAᵀ)⁻¹b`` is used.

    Args:
        shells:    Shell names from :data:`_SHELLS`.
        max_order: Highest isotropy order to impose (2, 4 or 6).

    Returns:
        One weight per shell.
    """
    n_order = {2: 1, 4: 2, 6: 3}[max_order]
    rows = _moment_rows(shells)[:n_order]
    rhs = [1.0] + [0.0] * (n_order - 1)
    n = len(shells)
    A = [rows[i][:] for i in range(n_order)]  # (n_order, n)
    if n_order > n:
        raise ValueError(f"{len(shells)} shells cannot satisfy {n_order} conditions")
    AAT = [[sum(A[i][k] * A[j][k] for k in range(n)) for j in range(n_order)]
           for i in range(n_order)]
    y = [sum(_inv(AAT)[i][j] * rhs[j] for j in range(n_order)) for i in range(n_order)]
    return tuple(sum(A[i][k] * y[i] for i in range(n_order)) for k in range(n))


@dataclass(frozen=True)
class Stencil:
    """A gradient stencil: neighbour vectors, per-vector weights, isotropy order."""

    name: str
    vectors: tuple[tuple[int, int], ...]
    weights: tuple[float, ...]
    max_order: int

    @property
    def shell_weights(self) -> dict[str, float]:
        """Per-shell weight (one entry per shell, vectors are stored shell-major)."""
        out: dict[str, float] = {}
        idx = 0
        for sh in _STENCIL_SHELLS[self.name]:
            out[sh] = self.weights[idx]
            idx += len(_SHELLS[sh])
        return out

    def tensors(self, device=None, dtype=torch.float32):
        vec = torch.tensor(self.vectors, dtype=dtype, device=device)
        w = torch.tensor(self.weights, dtype=dtype, device=device)
        return vec, w


@functools.cache
def stencil(name: str = "d2q9") -> Stencil:
    """Return a named gradient stencil.

    ``"d2q9"``    |c| = 1, √2 — the classic 8-neighbour D2Q9 set, weights
                  (1/3, 1/12): unique solution of the 2nd+4th order isotropy
                  conditions, i.e. isotropic to 4th order, anisotropic at 6th.
    ``"iso6"``    |c| = 1, √2, 2 — 12 neighbours; 6th-order isotropic weights
                  (4/15, 1/10, 1/120).
    ``"iso6_25"`` |c| = 1, √2, 2, √5 — the full "8 + 16" neighbourhood
                  (24 neighbours); 6th-order isotropic, minimum-norm weights.
    """
    if name not in _STENCIL_SHELLS:
        raise ValueError(f"unknown stencil {name!r}; expected one of {stencil_names()}")
    shells = _STENCIL_SHELLS[name]
    if name == "d2q9":
        shell_w = {"axis1": 1.0 / 3.0, "diag": 1.0 / 12.0}
    else:
        w = isotropic_weights(shells, max_order=6)
        shell_w = dict(zip(shells, w))
    vectors: list[tuple[int, int]] = []
    weights: list[float] = []
    for sh in shells:
        for v in _SHELLS[sh]:
            vectors.append(v)
            weights.append(shell_w[sh])
    return Stencil(name=name, vectors=tuple(vectors), weights=tuple(weights),
                   max_order=4 if name == "d2q9" else 6)


def stencil_names() -> tuple[str, ...]:
    return tuple(_STENCIL_SHELLS)


# ---------------------------------------------------------------------------
# Gradient / force
# ---------------------------------------------------------------------------


def _shift(field: torch.Tensor, cx: int, cy: int, periodic: bool) -> torch.Tensor:
    """Return ``field(x + c)`` with ``c = (cx, cy)`` (dims: ``(..., ny, nx)``)."""
    if periodic:
        if cy:
            field = torch.roll(field, -cy, dims=-2)
        if cx:
            field = torch.roll(field, -cx, dims=-1)
        return field
    ny, nx = field.shape[-2], field.shape[-1]
    pad = field
    top, bottom = max(0, cy), max(0, -cy)
    left, right = max(0, cx), max(0, -cx)
    # zero padding == truncated stencil (outside the box psi = 0: a dry wall)
    if top or bottom or left or right:
        pad = torch.nn.functional.pad(field, (left, right, top, bottom),
                                      mode="constant", value=0.0)
    y0, x0 = max(0, -cy), max(0, -cx)
    return pad[..., y0:y0 + ny, x0:x0 + nx]


@functools.lru_cache(maxsize=64)
def _neighbour_index(
    ny: int, nx: int, vectors: tuple[tuple[int, int], ...],
    device_type: str, device_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached advanced-indexing tensors gathering all ``ψ(x+c)`` in one call."""
    dev = torch.device(device_type) if device_index < 0 else torch.device(device_type, device_index)
    cy = torch.tensor([v[1] for v in vectors], dtype=torch.int64)
    cx = torch.tensor([v[0] for v in vectors], dtype=torch.int64)
    # ψ(x + c): index (i + c) mod n on both axes
    y = (torch.arange(ny, dtype=torch.int64).unsqueeze(0) + cy.unsqueeze(1)) % ny
    x = (torch.arange(nx, dtype=torch.int64).unsqueeze(0) + cx.unsqueeze(1)) % nx
    return y.to(dev).view(-1, ny, 1), x.to(dev).view(-1, 1, nx)


def psi_gradient(
    psi: torch.Tensor,
    stencil: str | "Stencil" = "d2q9",
    form: str = "difference",
    periodic: bool = True,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discrete gradient of a scalar field on the requested stencil.

    Args:
        psi:        Scalar field ``(ny, nx)``.
        stencil:    Stencil name (see :func:`stencil`) or a :class:`Stencil`.
        form:       ``"product"`` (Σ W c ψ(x+c)) or ``"difference"``
                    (Σ W c [ψ(x+c) − ψ(x)], the exact-difference/EDM form).
        periodic:   ``True`` for a periodic (toroidal) box; ``False`` treats
                    cells outside the domain as ``ψ = 0`` (dry wall), which
                    truncates the stencil and makes Σ W c ≠ 0.
        solid_mask: Optional boolean ``(ny, nx)`` wall mask; ψ is replaced at
                    wall cells before differencing.
        wall_psi:   ψ attributed to wall cells (``None`` → 0, dry wall).

    Returns:
        ``(gx, gy)``, each of shape ``(ny, nx)``, approximating ``∇ψ``.
    """
    if form not in ("product", "difference"):
        raise ValueError(f"psi_gradient supports form='product'|'difference', got {form!r}")
    if isinstance(stencil, str):
        st = globals()["stencil"](stencil)
    else:
        st = stencil
    if solid_mask is not None:
        psi = psi.masked_fill(solid_mask, 0.0 if wall_psi is None else float(wall_psi))
    vec, wt = st.tensors(device=psi.device, dtype=psi.dtype)
    n = vec.shape[0]
    if periodic:
        dev = psi.device
        y_idx, x_idx = _neighbour_index(
            psi.shape[-2], psi.shape[-1], st.vectors,
            dev.type, -1 if dev.index is None else dev.index,
        )
        shifted = psi[y_idx, x_idx]  # (n, ny, nx)
    else:
        shifted = torch.stack([_shift(psi, int(vec[k, 0]), int(vec[k, 1]), False)
                               for k in range(n)], dim=0)
    if form == "difference":
        shifted = shifted - psi.unsqueeze(0)
    wcx = (wt * vec[:, 0]).view(n, 1, 1)
    wcy = (wt * vec[:, 1]).view(n, 1, 1)
    gx = (shifted * wcx).sum(0)
    gy = (shifted * wcy).sum(0)
    return gx, gy


def sc_interaction_force(
    psi: torch.Tensor,
    G_eff: float,
    stencil: str | Stencil = "d2q9",
    form: str = "difference",
    gx: float = 0.0,
    gy: float = 0.0,
    rho: torch.Tensor | None = None,
    periodic: bool = True,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shan–Chen self-interaction force (+ optional gravity).

    ``form="product"``     F = −G_eff cs² ψ ∇ψ          (classic Shan–Chen)
    ``form="difference"``  F = −G_eff cs² ψ Σ W c [ψ(x+c) − ψ(x)]   (exact diff.)
    ``form="psi2"``        F = −(G_eff/2) cs² ∇(ψ²)     (pressure-tensor form,
                           exactly curl-free / exactly momentum conserving)

    Args:
        psi:        Pseudopotential field ``(ny, nx)``.
        G_eff:      Physical coupling (negative → attraction).
        stencil:    Stencil name or :class:`Stencil`.
        form:       See above.
        gx, gy:     Body-force accelerations (multiplied by ``rho`` when given).
        rho:        Density field for the gravity term (defaults to ψ's shape
                    and unit density — pass ``rho`` for physical gravity).
        periodic:   Periodic box or truncated (wall) stencil.
        solid_mask, wall_psi: as in :func:`psi_gradient`.

    Returns:
        ``(Fx, Fy)`` each ``(ny, nx)``.
    """
    if form == "psi2":
        psi_w = psi
        if solid_mask is not None:
            psi_w = psi_w.masked_fill(solid_mask, 0.0 if wall_psi is None else float(wall_psi))
        dpx, dpy = psi_gradient(psi_w * psi_w, stencil, "product", periodic)
        Fx = -0.5 * G_eff * CS2 * dpx
        Fy = -0.5 * G_eff * CS2 * dpy
    elif form in ("product", "difference"):
        dpx, dpy = psi_gradient(psi, stencil, form, periodic, solid_mask, wall_psi)
        Fx = -G_eff * CS2 * psi * dpx
        Fy = -G_eff * CS2 * psi * dpy
    else:
        raise ValueError(f"unknown form {form!r}; expected one of {FORMS}")
    if gx or gy:
        r = rho if rho is not None else torch.ones_like(psi)
        Fx = Fx + r * gx
        Fy = Fy + r * gy
    return Fx, Fy


# ---------------------------------------------------------------------------
# Collision / stepping
# ---------------------------------------------------------------------------


def collide_sc_force(
    f: torch.Tensor,
    Fx: torch.Tensor,
    Fy: torch.Tensor,
    tau: float = 1.0,
    forcing: str = "kupershtokh",
    solid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """BGK collision of a D2Q9 single-component distribution with a given force.

    ``forcing="kupershtokh"``  exact-difference (EDM) forcing of Kupershtokh
        (2004): relax to ``feq(ρ,u)`` and add ``feq(ρ, u+F/ρ) − feq(ρ, u)``.
        Momentum-exact and mass-conserving — identical to ``forcing="edm"`` in
        :func:`tensorlbm.multiphase.collide_sc_single_component`.
    ``forcing="velocity_shift"``  fold the force into the equilibrium velocity
        (historical library behaviour, injects a τ-dependent spurious force).
    ``forcing="guo"``  Guo (2002) source term with the half-force velocity.

    Args:
        f:      Distribution tensor ``(9, ny, nx)``.
        Fx, Fy: Force components ``(ny, nx)`` (the *full* force: interaction +
                gravity).
        tau:    BGK relaxation time.
        forcing, solid_mask: see above.

    Returns:
        Collided distribution, same shape.
    """
    rho, ux, uy = macroscopic(f)
    rho_s = torch.clamp(rho, min=1e-12)
    if forcing == "kupershtokh":
        feq_u = equilibrium(rho, ux, uy)
        feq_d = equilibrium(rho, ux + Fx / rho_s, uy + Fy / rho_s)
        f_out = f - (f - feq_u) / tau + (feq_d - feq_u)
    elif forcing == "velocity_shift":
        feq = equilibrium(rho, ux + tau * Fx / rho_s, uy + tau * Fy / rho_s)
        f_out = f - (f - feq) / tau
    elif forcing == "guo":
        uxp = ux + 0.5 * Fx / rho_s
        uyp = uy + 0.5 * Fy / rho_s
        feq = equilibrium(rho, uxp, uyp)
        c = _D2Q9_C.to(f.device).float()
        w = _D2Q9_W.to(f.device).float().view(9, 1, 1)
        cx = c[:, 0].view(9, 1, 1)
        cy = c[:, 1].view(9, 1, 1)
        cf = cx * Fx.unsqueeze(0) + cy * Fy.unsqueeze(0)
        cu = cx * uxp.unsqueeze(0) + cy * uyp.unsqueeze(0)
        uf = uxp * Fx + uyp * Fy
        S = w * (1.0 - 1.0 / (2.0 * tau)) * ((cf - uf.unsqueeze(0)) / CS2
                                             + cu * cf / CS2**2)
        f_out = f - (f - feq) / tau + S
    else:
        raise ValueError(f"unknown forcing {forcing!r}; expected one of {FORCINGS}")
    if solid_mask is not None:
        f_out = torch.where(solid_mask.unsqueeze(0), f, f_out)
    return f_out


def stream_step(f: torch.Tensor) -> torch.Tensor:
    """Periodic streaming on the D2Q9 stencil (delegates to ``solver.stream``)."""
    return stream(f)


def step(
    f: torch.Tensor,
    G_eff: float,
    tau: float = 1.0,
    psi_fn=None,
    stencil: str | Stencil = "d2q9",
    form: str = "difference",
    forcing: str = "kupershtokh",
    gx: float = 0.0,
    gy: float = 0.0,
) -> torch.Tensor:
    """One full collide-and-stream step of the upgraded SCMP scheme.

    Args:
        f:       Distribution ``(9, ny, nx)``.
        G_eff:   Physical coupling (<0 attractive).
        tau:     Relaxation time.
        psi_fn:  Pseudopotential, default :func:`tensorlbm.multiphase.psi_exp`.
        stencil, form, forcing, gx, gy: see the individual functions.

    Returns:
        The streamed post-collision distribution.
    """
    if psi_fn is None:
        from .multiphase import psi_exp as psi_fn
    rho, _, _ = macroscopic(f)
    psi = psi_fn(rho)
    Fx, Fy = sc_interaction_force(psi, G_eff, stencil, form, gx, gy, rho=rho)
    f = collide_sc_force(f, Fx, Fy, tau, forcing)
    return stream_step(f)


# ---------------------------------------------------------------------------
# Diagnostics / helpers
# ---------------------------------------------------------------------------


def pressure_mech(rho: torch.Tensor, psi: torch.Tensor, G_eff: float) -> torch.Tensor:
    """Mechanical (bulk) pressure of the SCMP model.

    ``p = ρ cs² + (G_eff/2) cs² ψ² = ρ/3 + (G_eff/6) ψ²`` — the standard
    Shan–Chen equation of state, used to measure the Laplace pressure jump.
    """
    return rho * CS2 + 0.5 * G_eff * CS2 * psi * psi


def library_coupling(G_eff: float) -> float:
    """Convert a physical coupling to the ``tensorlbm.multiphase`` convention."""
    return -G_eff


def physical_coupling(G_lib: float) -> float:
    """Convert a ``tensorlbm.multiphase`` coupling to the physical convention."""
    return -G_lib


def max_velocity(f: torch.Tensor) -> float:
    """``max |u|`` over the domain (the spurious-current metric)."""
    _, ux, uy = macroscopic(f)
    return float(torch.sqrt(ux * ux + uy * uy).max())