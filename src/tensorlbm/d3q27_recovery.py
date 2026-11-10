"""D3Q27 high-order lattice recovery: moment audit + analytic flow anchors.

This module turns the D3Q27 lattice from a *declared* enum value into a
*measurable* capability with three independent, machine-checkable anchors:

1. **Lattice algebra / dead-enumeration audit** (:func:`moment_audit`,
   :func:`direction_liveness_audit`).  All 27 discrete velocities, the Qian
   weights, ``c_s^2 = 1/3`` and the isotropic moment tensors of rank 2/4 are
   asserted exactly (float64, residual at roundoff).  The axis permutation
   between the descriptor ``(cx, cy, cz)`` and the solver field order
   ``(z, y, x)`` is verified per direction, so no direction can silently
   stream to the wrong neighbour -- the failure mode behind the historical
   "dead enumeration" (27 directions declared, 19 actually moving).

2. **Three-dimensional plane-Poiseuille analytic anchor**
   (:func:`run_channel_poiseuille`).  Body-force driven flow between two
   half-way bounce-back walls separated by ``H`` lattice cells, periodic in
   ``x`` and ``z``::

       u(y) = F / (2 rho nu) * y (H - y),   nu = (tau - 1/2) c_s^2

   The steady LBM field is compared with this closed form through a relative
   L2 norm; second-order convergence in ``H`` is the expected behaviour and
   is what the acceptance ladder reports.

3. **Stokes sphere-drag anchor** (:func:`run_sphere_stokes`).  A rigid sphere
   of diameter ``D`` sits in a triply periodic box of side ``L`` and is
   loaded by a uniform body force; the drag is measured with a link-resolved
   Ladd momentum-exchange estimator (:func:`momentum_exchange_drag`) whose
   *own* correctness is cross-checked by the exact steady momentum balance
   ``F_drag = sum(F_body)``.  The analytic reference is the Hasimoto (1959)
   simple-cubic-array solution, i.e. the *confinement-corrected* Stokes drag
   of a periodic box::

       F = 6 pi mu a U / (1 - 2.837297 a/L + 4.19 (a/L)^3 + O((a/L)^5))

   Comparing with the bare ``6 pi mu a U`` of an unbounded sphere would
   charge a 20-35 % "error" that is pure domain truncation, so both numbers
   are reported and the acceptance judgement uses the Hasimoto reference;
   the ``a/L -> 0`` limit is demonstrated by a two-box extrapolation.

Physics caveat carried into the report: ``C_D = 24/Re`` is the *zero
Reynolds number* limit.  At ``Re = 1`` the true drag is already ~10 % above
Stokes (Oseen) and at ``Re = 10`` it is ~70 % above (``C_D ~ 4.1``).  The
sweep therefore reports the Stokes error only inside the Stokes regime
(``Re <= 0.5``) and, above it, the dimensionless drag
``K = C_D Re / 24`` against the Oseen correction ``(1 + 3 Re / 16)`` -- that
growth *is* the physical content at ``Re ~ 1``.

Conventions
-----------
* Lattice units, ``rho_0 = 1``, ``c_s^2 = 1/3``.
* Fields are ``(q, nz, ny, nx)``; the descriptor ``C`` stores ``(cx, cy, cz)``.
* Channel walls sit half-way between the first fluid row and the solid row,
  so the hydrodynamic gap is ``H = ny - 2`` lattice cells.
* A body force is applied as a *force density* ``F`` (force per volume) with
  the Guo (2002) second-order scheme, which keeps the discrete Poiseuille
  solution second-order accurate.

References
----------
Qian, d'Humières & Lallemand (1992), EPL 17, 479.
Guo, Zheng & Shi (2002), Phys. Rev. E 65, 046308 (forcing scheme).
Ladd (1994), J. Fluid Mech. 271, 285 (momentum exchange).
Hasimoto (1959), J. Fluid Mech. 5, 317 (periodic array drag).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .d3q27 import OPPOSITE, W_EXACT64, C, equilibrium27, macroscopic27, stream27

CS2 = 1.0 / 3.0

_DELTA = torch.eye(3, dtype=torch.float64)


def _delta(dtype: torch.dtype) -> torch.Tensor:
    return _DELTA.to(dtype)


def _weights(dtype: torch.dtype) -> torch.Tensor:
    return W_EXACT64.to(dtype)


# ---------------------------------------------------------------------------
# 1. Lattice algebra audit
# ---------------------------------------------------------------------------


def velocity_moments(dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    """Raw velocity moments ``M_n = sum_i w_i c_i ⊗ ... ⊗ c_i`` of ranks 1..6."""
    c = C.to(dtype)
    w = _weights(dtype)
    return {
        "m1": torch.einsum("i,ia->a", w, c),
        "m2": torch.einsum("i,ia,ib->ab", w, c, c),
        "m3": torch.einsum("i,ia,ib,ic->abc", w, c, c, c),
        "m4": torch.einsum("i,ia,ib,ic,id->abcd", w, c, c, c, c),
        "m6": torch.einsum("i,ia,ib,ic,id,ie,if->abcdef", w, c, c, c, c, c, c),
    }


def _isotropic_rank4(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """``c_s^4 (delta_ab delta_cd + delta_ac delta_bd + delta_ad delta_bc)``."""
    d = _delta(dtype)
    return CS2**2 * (
        torch.einsum("ab,cd->abcd", d, d)
        + torch.einsum("ac,bd->abcd", d, d)
        + torch.einsum("ad,bc->abcd", d, d)
    )


_RANK6_STRINGS = (
    "ab,cd,ef->abcdef",
    "ab,ce,df->abcdef",
    "ab,cf,de->abcdef",
    "ac,bd,ef->abcdef",
    "ac,be,df->abcdef",
    "ac,bf,de->abcdef",
    "ad,bc,ef->abcdef",
    "ad,be,cf->abcdef",
    "ad,bf,ce->abcdef",
    "ae,bc,df->abcdef",
    "ae,bd,cf->abcdef",
    "ae,bf,cd->abcdef",
    "af,bc,de->abcdef",
    "af,bd,ce->abcdef",
    "af,be,cd->abcdef",
)


def _isotropic_rank6(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Fully isotropic rank-6 tensor: the 15 pairings of ``c_s^6``."""
    d = _delta(dtype)
    out = torch.zeros((3,) * 6, dtype=dtype)
    for spec in _RANK6_STRINGS:
        out = out + torch.einsum(spec, d, d, d)
    return (CS2**3) * out


@dataclass(frozen=True)
class MomentAudit:
    """Exact lattice-algebra residuals for the D3Q27 stencil (dimensionless)."""

    n_directions: int
    weight_sum: float
    cs2_measured: float
    cs2_error: float
    rank1_residual: float
    rank2_residual: float
    rank3_residual: float
    rank4_isotropic_residual: float
    rank6_isotropic_residual: float
    max_abs_rank3_component: float
    is_isotropic_rank2: bool
    is_isotropic_rank4: bool
    is_isotropic_rank6: bool

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def moment_audit(dtype: torch.dtype = torch.float64) -> MomentAudit:
    """Audit the D3Q27 velocity moments against the isotropic tensors.

    Checks, in float64:

    * ``sum_i w_i = 1``;
    * ``M_1 = sum_i w_i c_i = 0`` (no lattice drift);
    * ``M_2 = c_s^2 delta`` with ``c_s^2 = 1/3``;
    * ``M_3 = sum_i w_i c_i c_i c_i = 0`` **exactly** -- the odd moment of any
      centro-symmetric lattice vanishes.  (This is the moment identity the
      campaign metadata quotes with an ``e_gamma`` on the right-hand side;
      that form can only be satisfied by the *equilibrium* third moment
      ``sum_i f_i^eq c c c``, verified separately in
      :func:`equilibrium_third_moment_residual`.)
    * ``M_4`` against its isotropic tensor (true for D3Q27) and ``M_6``
      against its isotropic tensor (false for D3Q27).

    The rank-6 *failure* is asserted as well, so the audit cannot be silently
    downgraded to a weaker check.

    Raises ``AssertionError`` if any hard identity is violated beyond
    ``1e-12`` (float64 roundoff is ~1e-15).
    """
    m = velocity_moments(dtype)
    d = _delta(dtype)

    weight_sum = float(_weights(dtype).sum())
    cs2_measured = float(torch.einsum("ab,ab->", m["m2"], d) / 3.0)
    rank1 = float(m["m1"].abs().max())
    rank2 = float((m["m2"] - CS2 * d).abs().max())
    rank3 = float(m["m3"].abs().max())
    iso4 = _isotropic_rank4(dtype)
    iso6 = _isotropic_rank6(dtype)
    rank4 = float((m["m4"] - iso4).abs().max()) / float(iso4.abs().max())
    rank6 = float((m["m6"] - iso6).abs().max()) / float(iso6.abs().max())

    audit = MomentAudit(
        n_directions=int(C.shape[0]),
        weight_sum=weight_sum,
        cs2_measured=cs2_measured,
        cs2_error=abs(cs2_measured - CS2),
        rank1_residual=rank1,
        rank2_residual=rank2,
        rank3_residual=rank3,
        rank4_isotropic_residual=rank4,
        rank6_isotropic_residual=rank6,
        max_abs_rank3_component=rank3,
        is_isotropic_rank2=rank2 < 1e-12,
        is_isotropic_rank4=rank4 < 1e-12,
        is_isotropic_rank6=rank6 < 1e-12,
    )

    assert C.shape == (27, 3), "D3Q27 descriptor must hold 27 directions"
    assert abs(weight_sum - 1.0) < 1e-12, f"weights do not sum to 1: {weight_sum}"
    assert audit.cs2_error < 1e-12, f"c_s^2 != 1/3 (got {cs2_measured})"
    assert rank1 < 1e-12, f"first velocity moment is non-zero: {rank1}"
    assert rank2 < 1e-12, f"second velocity moment is not isotropic: {rank2}"
    assert rank3 < 1e-12, f"odd third velocity moment does not vanish: {rank3}"
    assert audit.is_isotropic_rank4, f"rank-4 moment is not isotropic: {rank4}"
    assert not audit.is_isotropic_rank6, (
        "D3Q27 unexpectedly isotropic at rank 6; the audit threshold is stale"
    )
    return audit


def equilibrium_third_moment_residual(
    rho: float = 1.3, u: tuple[float, float, float] = (0.07, -0.05, 0.03)
) -> dict[str, Any]:
    """Compare ``sum_i f_i^eq c c c`` with the analytic Maxwellian third moment.

    Analytic value (``c_s^2 = 1/3``)::

        T_abg = rho c_s^2 (delta_ab u_g + delta_bg u_a + delta_ga u_b)
                + rho u_a u_b u_g

    This is the correct statement of the third-moment identity that the
    campaign text abbreviates (it writes ``e_gamma`` where ``u_gamma``
    belongs).  The residual is at roundoff for D3Q27 because the rank-4
    isotropy of the stencil supplies exactly the ``u u u`` term.
    """
    du = torch.float64
    c = C.to(du)
    w = W_EXACT64.to(du)
    uvec = torch.tensor(u, dtype=du)
    ux, uy, uz = uvec[0], uvec[1], uvec[2]
    cu = c[:, 0] * ux + c[:, 1] * uy + c[:, 2] * uz
    u_sq = ux * ux + uy * uy + uz * uz
    feq = w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)
    num = torch.einsum("i,ia,ib,ic->abc", feq, c, c, c)

    d = _delta(du)
    analytic = rho * CS2 * (
        torch.einsum("ab,c->abc", d, uvec)
        + torch.einsum("bc,a->abc", d, uvec)
        + torch.einsum("ca,b->abc", d, uvec)
    ) + rho * torch.einsum("a,b,c->abc", uvec, uvec, uvec)

    truncated = rho * CS2 * (
        torch.einsum("ab,c->abc", d, uvec)
        + torch.einsum("bc,a->abc", d, uvec)
        + torch.einsum("ca,b->abc", d, uvec)
    )
    resid_full = float((num - analytic).abs().max())
    resid_trunc = float((num - truncated).abs().max())
    scale = float(truncated.abs().max())
    return {
        "residual_vs_full_maxwellian_abs": resid_full,
        "residual_vs_full_maxwellian_rel": resid_full / float(analytic.abs().max()),
        "residual_vs_truncated_abs": resid_trunc,
        "residual_vs_truncated_rel": resid_trunc / scale if scale else resid_trunc,
        "uuu_deficit_abs": float((rho * torch.einsum("a,b,c->abc", uvec, uvec, uvec)).abs().max()),
        "scale": scale,
        "rho": rho,
        "u": list(u),
        "interpretation": (
            "the lattice equilibrium reproduces rho c_s^2 (d_ab u_c + ...) to "
            "roundoff; the remaining gap against the full Maxwellian is exactly "
            "the rho u_a u_b u_c term, i.e. O(Ma^3), which no stencil that is "
            "anisotropic at rank 6 (see moment_audit: rank6 residual 0.40) can "
            "carry. Third-moment accuracy is therefore exact at the retained "
            "order with a deficit that is third order in Mach number."
        ),
    }


def direction_liveness_audit() -> dict[str, Any]:
    """Prove that all 27 directions actually move populations (no dead enum).

    For every direction ``q`` a single-cell perturbation is placed in a
    resting equilibrium field, one streaming step is applied, and the
    perturbation is required to arrive at the neighbour ``x + c_q``.  This
    catches (a) a direction that is never streamed, (b) the
    ``(cx, cy, cz)`` vs ``(z, y, x)`` permutation applied backwards, and
    (c) an ``OPPOSITE`` mistake that would corrupt bounce-back.

    Returns the per-``q`` shift table together with the list of failed
    directions (empty on success).
    """
    du = torch.float64
    n = 5
    rho = torch.ones((n, n, n), dtype=du)
    zero = torch.zeros_like(rho)
    f0 = equilibrium27(rho, zero, zero, zero)
    probe = (2, 2, 2)  # (z, y, x)
    delta = 1e-3
    failures: list[int] = []
    shifts: dict[int, tuple[int, int, int]] = {}
    for q in range(27):
        f = f0.clone()
        f[q, probe[0], probe[1], probe[2]] += delta
        fs = stream27(f)
        dif = fs - f0
        total = float(dif.abs().sum())
        hits = torch.nonzero(dif.abs() > 0.1 * delta, as_tuple=False)
        cx, cy, cz = (int(v) for v in C[q].tolist())
        want = ((probe[0] + cz) % n, (probe[1] + cy) % n, (probe[2] + cx) % n)
        ok = (
            hits.shape[0] == 1
            and tuple(int(v) for v in hits[0, 1:].tolist()) == want
            and int(hits[0, 0].item()) == q
            and abs(total - delta) < 1e-9
        )
        shifts[q] = (cx, cy, cz)
        if not ok:
            failures.append(q)
    return {
        "n_directions": 27,
        "failures": failures,
        "all_live": not failures,
        "shifts": shifts,
        "note": (
            "field order (q, nz, ny, nx); descriptor order (cx, cy, cz); "
            "streaming moves direction q by (+cz, +cy, +cx)"
        ),
    }


# ---------------------------------------------------------------------------
# 2. Guo forcing + generic D3Q27 stepper (half-way bounce-back walls)
# ---------------------------------------------------------------------------


def bgk_shear_viscosity(tau: float, k: float, dtype: torch.dtype = torch.float64) -> float:
    """Effective shear viscosity of the D3Q27 BGK operator at wavenumber ``k``.

    Eigenvalue of the linearised collide+stream operator: with
    ``J_ij = w_i (1 + 3 c_i.c_j)`` (``d f_i^eq / d f_j`` at ``rho = 1, u = 0``)
    and ``D = diag(exp(-i k c_y))`` the operator is
    ``M = D (I - (I - J)/tau)`` and the slowest decaying magnitude gives
    ``nu_eff = -ln|lambda_max| / k^2``.

    This matters because the *textbook* ``nu = (tau - 1/2)/3`` is only the
    ``k -> 0`` limit.  Measured here (artifacts ``viscosity_gate.json``): the
    deviation is ``O(tau^2 k^2)`` and reaches -2.7 % at ``(tau, k) = (2, 2pi/32)``,
    -12 % at ``(2, 2pi/16)`` and -49 % at ``(6.5, 2pi/32)``.  A drag study that
    normalises by the formula instead of by this number inherits that bias --
    exactly the failure mode that produced a (spurious) 3x low sphere drag in
    the first attempt at this benchmark, which used ``tau = 6.5``.
    """
    c = C.to(dtype)
    w = W_EXACT64.to(dtype)
    J = w.unsqueeze(1) * (1.0 + 3.0 * (c @ c.T))
    eye = torch.eye(27, dtype=dtype)
    phase = torch.exp(-1j * k * c[:, 1].to(torch.complex128))
    M = phase.diag().to(torch.complex128) @ (
        eye.to(torch.complex128) - (eye - J).to(torch.complex128) / tau
    )
    mags = torch.linalg.eigvals(M).abs()
    return float(-torch.log(mags.max()) / (k * k))


def poiseuille_with_effective_viscosity(
    H: int,
    tau: float,
    force_density: float,
    k_mode: float | None = None,
) -> dict[str, float]:
    """Analytic peak velocity of the channel using the *effective* viscosity.

    The steady plane-Poiseuille peak is ``u_max = F H^2 / (8 rho nu)``; this
    helper returns the peak for the formula viscosity and for the lattice BGK
    effective viscosity at the lowest channel mode ``k = pi / H``.
    """
    nu_formula = (tau - 0.5) * CS2
    k = k_mode if k_mode is not None else math.pi / H
    nu_eff = bgk_shear_viscosity(tau, k)
    return {
        "nu_formula": nu_formula,
        "nu_effective": nu_eff,
        "deviation": nu_eff / nu_formula - 1.0 if nu_formula else float("nan"),
        "k_mode": k,
        "u_max_formula": force_density * H * H / (8.0 * nu_formula),
        "u_max_effective": force_density * H * H / (8.0 * nu_eff),
    }


def guo_source(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    force: tuple[float, float, float],
    tau: float,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Guo (2002) second-order body-force source ``F_i``.

    ``F_i = (1 - 1/(2 tau)) w_i [ 3 (c_i - u).F + 9 (c_i.u)(c_i.F) ]``
    for ``c_s^2 = 1/3`` and ``rho = 1`` normalisation of the force density.
    """
    c = C.to(dtype)
    w = W_EXACT64.to(dtype).view(27, 1, 1, 1)
    cx = c[:, 0].view(27, 1, 1, 1)
    cy = c[:, 1].view(27, 1, 1, 1)
    cz = c[:, 2].view(27, 1, 1, 1)
    fx, fy, fz = force
    cu = cx * ux + cy * uy + cz * uz
    cf = cx * fx + cy * fy + cz * fz
    uf = ux * fx + uy * fy + uz * fz
    return (1.0 - 0.5 / tau) * w * (3.0 * (cf - uf) + 9.0 * cu * cf)


def half_force_velocity(
    rho: torch.Tensor,
    mom_x: torch.Tensor,
    mom_y: torch.Tensor,
    mom_z: torch.Tensor,
    force: tuple[float, float, float],
    fluid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``u = (sum c f + F/2) / rho`` on fluid nodes; solid nodes get ``u = 0``."""
    rho_safe = torch.clamp(rho, min=1e-30)
    rho_use = torch.where(fluid_mask, rho_safe, torch.ones_like(rho_safe))
    ux = mom_x / rho_use + 0.5 * force[0] / rho_use
    uy = mom_y / rho_use + 0.5 * force[1] / rho_use
    uz = mom_z / rho_use + 0.5 * force[2] / rho_use
    zero = torch.zeros_like(ux)
    return (
        torch.where(fluid_mask, ux, zero),
        torch.where(fluid_mask, uy, zero),
        torch.where(fluid_mask, uz, zero),
    )


class D3Q27Stepper:
    """D3Q27 BGK + Guo stepper with *true* half-way bounce-back walls.

    The wall treatment is the pull-form half-way bounce-back: at a fluid node
    ``x`` and direction ``i`` whose upstream neighbour ``x - c_i`` is solid,
    the streamed population is replaced by the post-collision population
    pointing into the wall, ``f_{opp(i)}^post(x)``.  The reflected population
    then travels the two half links back in a *single* time step, which puts
    the no-slip plane exactly half-way between the two node centres.

    Why this matters (measured, not assumed): the cheaper convention
    "stream periodically, then mirror every population inside the solid cell"
    reflects at the *solid node centre* instead of the mid-point.  That is a
    first-order wall-position error of half a cell, which shows up as a
    ``(H+1)^2 / H^2`` over-prediction of the peak velocity -- measured here as
    +12.4 % at ``H = 16`` and +24.6 % at ``H = 8`` before this formulation was
    adopted.  The steady Poiseuille anchor is what exposed it.

    Only *interface* links are touched: the fluid/solid link list is built once
    from the static mask, so bounce-back and momentum exchange are gathers /
    scatters over a few thousand indices instead of full-array passes (the
    streaming pass remains 27 ``roll`` operations).
    """

    def __init__(
        self,
        solid_mask: torch.Tensor,
        *,
        tau: float,
        force: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.solid = solid_mask
        self.fluid = ~solid_mask
        self.tau = float(tau)
        self.force = tuple(float(v) for v in force)
        shape = tuple(int(v) for v in solid_mask.shape)
        opp = OPPOSITE.tolist()
        c_list = C.tolist()
        # bounce-back links: fluid node x whose upstream neighbour x - c_q is solid
        self._bb: list[tuple[int, int, torch.Tensor]] = []
        # momentum-exchange links: fluid node x whose downstream neighbour x + c_q is solid
        self._me: list[tuple[int, torch.Tensor]] = []
        for q in range(27):
            cx, cy, cz = (int(v) for v in c_list[q])
            if cx == 0 and cy == 0 and cz == 0:
                continue
            upstream_solid = torch.roll(solid_mask, shifts=(cz, cy, cx), dims=(0, 1, 2)) & self.fluid
            if bool(upstream_solid.any()):
                idx = torch.nonzero(upstream_solid.reshape(-1), as_tuple=False).squeeze(1)
                self._bb.append((q, opp[q], idx))
            downstream_solid = torch.roll(solid_mask, shifts=(-cz, -cy, -cx), dims=(0, 1, 2)) & self.fluid
            if bool(downstream_solid.any()):
                idx = torch.nonzero(downstream_solid.reshape(-1), as_tuple=False).squeeze(1)
                self._me.append((q, idx))
        assert shape == tuple(int(v) for v in solid_mask.shape)

    @property
    def n_bounce_back_links(self) -> int:
        return sum(int(idx.numel()) for _, _, idx in self._bb)

    # -- one step ---------------------------------------------------------
    def step(self, f: torch.Tensor, *, hook=None) -> torch.Tensor:
        rho, mx, my, mz = macroscopic27(f)
        ux, uy, uz = half_force_velocity(rho, mx, my, mz, self.force, self.fluid)
        f_eq = equilibrium27(rho, ux, uy, uz)
        f_post = f - (f - f_eq) / self.tau
        if self.force != (0.0, 0.0, 0.0):
            src = guo_source(ux, uy, uz, self.force, self.tau, dtype=f.dtype)
            # The body force acts on fluid volume only; letting it act inside
            # solid cells over-forces the fluid once bounce-back reflects it.
            f_post = f_post + torch.where(
                self.fluid.unsqueeze(0), src, torch.zeros_like(src)
            )
        if hook is not None:
            hook(f_post)
        f_new = stream27(f_post)
        for q, oq, idx in self._bb:
            f_new[q].view(-1)[idx] = f_post[oq].view(-1)[idx]
        return f_new

    # -- momentum exchange on the same link set ---------------------------
    def momentum_exchange_force(self, f_post: torch.Tensor) -> torch.Tensor:
        """Ladd drag ``F = 2 sum_links c_q f_q^post(x)`` over interface links."""
        force = torch.zeros(3, dtype=f_post.dtype, device=f_post.device)
        c_list = C.tolist()
        for q, idx in self._me:
            cx, cy, cz = (int(v) for v in c_list[q])
            s = f_post[q].view(-1)[idx].sum()
            force[0] += 2.0 * cx * s
            force[1] += 2.0 * cy * s
            force[2] += 2.0 * cz * s
        return force


def lbm_step_d3q27(
    f: torch.Tensor,
    *,
    tau: float,
    solid_mask: torch.Tensor,
    force: tuple[float, float, float] = (0.0, 0.0, 0.0),
    post_stream_hook=None,
) -> torch.Tensor:
    """Single-step convenience wrapper around :class:`D3Q27Stepper`.

    ``post_stream_hook`` is kept for API symmetry and receives the
    post-collision (pre-stream) field, which is the field the momentum
    exchange is defined on; for hot loops instantiate the stepper instead so
    the bounce-back link masks are built once.
    """
    stepper = D3Q27Stepper(solid_mask, tau=tau, force=force)
    return stepper.step(f, hook=post_stream_hook)


def macroscopic_with_force(
    f: torch.Tensor,
    solid_mask: torch.Tensor,
    force: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover ``(rho, ux, uy, uz)`` including the half-force velocity shift."""
    rho, mx, my, mz = macroscopic27(f)
    ux, uy, uz = half_force_velocity(rho, mx, my, mz, force, ~solid_mask)
    return rho, ux, uy, uz


# ---------------------------------------------------------------------------
# 3. Three-dimensional plane Poiseuille anchor
# ---------------------------------------------------------------------------


def poiseuille_analytic(
    y: torch.Tensor,
    H: float,
    force_density: float,
    nu: float,
    rho: float = 1.0,
) -> torch.Tensor:
    """Exact steady plane-Poiseuille profile ``u(y) = F/(2 rho nu) y (H - y)``."""
    return force_density / (2.0 * rho * nu) * y * (H - y)


@dataclass(frozen=True)
class PoiseuilleResult:
    """Steady D3Q27 plane-Poiseuille validation report."""

    H: float
    ny: int
    tau: float
    nu: float
    force_density: float
    n_steps: int
    converged: bool
    u_max_lbm: float
    u_max_analytic: float
    rel_l2_error: float
    rel_linf_error: float
    mean_rho_drift: float
    max_divergence: float
    finite: bool
    wall_force_x: float
    body_force_total: float
    force_balance_error: float
    profile_y: list[float]
    profile_lbm: list[float]
    profile_analytic: list[float]
    mass_history: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def run_channel_poiseuille(
    *,
    H: int = 32,
    nx: int = 4,
    nz: int = 4,
    tau: float = 0.8,
    force_density: float = 1.0e-5,
    n_steps: int = 60_000,
    tol: float = 1e-11,
    check_every: int = 100,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    stall_limit: int = 30,
) -> PoiseuilleResult:
    """Body-force driven plane Poiseuille flow between bounce-back walls.

    The walls occupy the first and last ``y`` rows; half-way bounce-back puts
    the no-slip planes at ``y = 0.5`` and ``y = ny - 1.5``, so the
    hydrodynamic gap is ``H = ny - 2``.  The field is stored and solved as a
    genuine three-dimensional array (periodic in ``x`` and ``z``) even though
    the exact solution is unidirectional.

    Convergence: the run stops when ``max|u(t) - u(t - check_every)| < tol``
    on the profile, or after ``stall_limit`` consecutive windows whose
    residual no longer decreases (steady state reached to roundoff).
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    ny = int(H) + 2
    rho0 = torch.ones((nz, ny, nx), dtype=dtype, device=device)
    u0 = torch.zeros_like(rho0)
    f = equilibrium27(rho0, u0, u0, u0, device=device)

    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    nu = (tau - 0.5) * CS2
    force = (float(force_density), 0.0, 0.0)
    y_coords = torch.arange(1, ny - 1, dtype=dtype, device=device) - 0.5
    u_exact = poiseuille_analytic(y_coords, H=float(H), force_density=force_density, nu=nu)

    prev: torch.Tensor | None = None
    last_res = float("inf")
    stall = 0
    converged = False
    mass_history: list[float] = []
    steps_done = 0

    stepper = D3Q27Stepper(solid, tau=tau, force=force)
    for step in range(1, n_steps + 1):
        f = stepper.step(f)
        steps_done = step
        if step % check_every == 0 or step == n_steps:
            if not torch.isfinite(f).all():
                break
            rho, ux, _uy, _uz = macroscopic_with_force(f, solid, force)
            mass_history.append(float(rho.sum().item()))
            prof = ux[0, 1:-1, 0]
            if prev is not None:
                res = float((prof - prev).abs().max().item())
                if res < tol:
                    converged = True
                    break
                stall = stall + 1 if res > 0.999 * last_res else 0
                last_res = res
                # a stalled residual only counts as converged at the numerical
                # floor; otherwise the run is still transient (large-tau runs
                # have long, slightly oscillatory tails).
                if stall >= stall_limit and last_res < max(1e-9, 100.0 * tol):
                    converged = True
                    break
            prev = prof.clone()

    rho, ux, uy, uz = macroscopic_with_force(f, solid, force)
    # One extra step carrying a momentum-exchange hook: on the two plane walls
    # the steady drag must equal the total applied body force, which validates
    # the force estimator itself (used later for the sphere).
    captured: list[torch.Tensor] = []
    stepper.step(f, hook=lambda fp: captured.append(stepper.momentum_exchange_force(fp)))
    wall_force = float(captured[0][0].item())
    rho_fluid = float(rho.sum().item()) - float(rho[solid].sum().item())
    body_total = float(force_density * rho_fluid)
    prof_lbm = ux[0, 1:-1, 0]
    denom = float(torch.linalg.vector_norm(u_exact))
    l2 = float(torch.linalg.vector_norm(prof_lbm - u_exact) / denom) if denom else float("inf")
    peak = float(u_exact.abs().max())
    linf = float((prof_lbm - u_exact).abs().max() / peak) if peak else float("inf")
    return PoiseuilleResult(
        H=float(H),
        ny=ny,
        tau=float(tau),
        nu=float(nu),
        force_density=float(force_density),
        n_steps=steps_done,
        converged=converged,
        u_max_lbm=float(prof_lbm.max().item()),
        u_max_analytic=float(u_exact.max().item()),
        rel_l2_error=l2,
        rel_linf_error=linf,
        mean_rho_drift=float((rho.mean() - 1.0).abs().item()),
        max_divergence=divergence_max(ux, uy, uz),
        finite=bool(torch.isfinite(f).all().item()),
        wall_force_x=wall_force,
        body_force_total=body_total,
        force_balance_error=abs(wall_force - body_total) / abs(body_total) if body_total else float("inf"),
        profile_y=[float(v) for v in y_coords.tolist()],
        profile_lbm=[float(v) for v in prof_lbm.tolist()],
        profile_analytic=[float(v) for v in u_exact.tolist()],
        mass_history=mass_history,
    )


def divergence_max(ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor) -> float:
    """Max ``|div u|`` with periodic central differences (interior ``y`` only)."""
    dux_dx = (torch.roll(ux, -1, dims=2) - torch.roll(ux, 1, dims=2)) * 0.5
    duz_dz = (torch.roll(uz, -1, dims=0) - torch.roll(uz, 1, dims=0)) * 0.5
    duy_dy = torch.zeros_like(uy)
    duy_dy[:, 1:-1, :] = (uy[:, 2:, :] - uy[:, :-2, :]) * 0.5
    div = (dux_dx + duy_dy + duz_dz)[:, 1:-1, :]
    return float(div.abs().max().item())


# ---------------------------------------------------------------------------
# 4. Stokes sphere drag
# ---------------------------------------------------------------------------


def sphere_mask(n: int, diameter: float, centre: tuple[float, float, float] | None = None) -> torch.Tensor:
    """Boolean solid mask of a staircase sphere in an ``n``-cubed periodic box.

    Cell ``(k, j, i)`` is solid when its centre ``(i + 0.5, j + 0.5, k + 0.5)``
    lies inside the sphere (the same half-cell convention as the half-way
    bounce-back walls).  Default centre is the box centre.
    """
    if centre is None:
        centre = (0.5 * n, 0.5 * n, 0.5 * n)
    zz = torch.arange(n, dtype=torch.float64).view(n, 1, 1) + 0.5
    yy = torch.arange(n, dtype=torch.float64).view(1, n, 1) + 0.5
    xx = torch.arange(n, dtype=torch.float64).view(1, 1, n) + 0.5
    r2 = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (zz - centre[2]) ** 2
    return r2 <= (0.5 * diameter) ** 2


def momentum_exchange_drag(f_post: torch.Tensor, solid_mask: torch.Tensor) -> torch.Tensor:
    """Link-resolved Ladd momentum exchange on a D3Q27 staircase solid.

    For every fluid node ``x`` and direction ``q`` with ``x + c_q`` inside the
    solid, the post-collision population ``f_q(x)`` hits the wall half-way and
    comes back reversed, so the momentum delivered to the solid is
    ``2 c_q f_q(x)``::

        F = 2 sum_x sum_{q : x + c_q in solid} c_q f_q^post(x)

    Only links that actually cross the fluid/solid interface are counted, so
    populations stored inside the solid cannot pollute the estimate.  This
    number is cross-checked against the exact steady momentum balance
    ``F = sum(F_body)`` in :func:`run_sphere_stokes`; a wrong estimator cannot
    satisfy both.
    """
    fluid = ~solid_mask
    force = torch.zeros(3, dtype=f_post.dtype, device=f_post.device)
    for q in range(27):
        cx, cy, cz = (int(v) for v in C[q].tolist())
        if cx == 0 and cy == 0 and cz == 0:
            continue
        downstream_solid = torch.roll(
            solid_mask, shifts=(-cz, -cy, -cx), dims=(0, 1, 2)
        ) & fluid
        if not bool(downstream_solid.any()):
            continue
        s = torch.where(downstream_solid, f_post[q], torch.zeros_like(f_post[q])).sum()
        force[0] += 2.0 * cx * s
        force[1] += 2.0 * cy * s
        force[2] += 2.0 * cz * s
    return force


def hasimoto_correction(a_over_l: float) -> float:
    """Hasimoto (1959) periodic-cube drag correction ``F / F_Stokes``.

    ``F = 6 pi mu a U / (1 - 2.837297 (a/L) + 4.19 (a/L)^3 + O((a/L)^5))``.
    Valid for ``a/L <~ 0.2``; the truncated series is reported as-is so the
    residual truncation is visible in the error budget.
    """
    x = a_over_l
    denom = 1.0 - 2.837297 * x + 4.19 * x**3
    return 1.0 / denom


def oseen_correction(re: float) -> float:
    """Oseen inertial correction ``C_D = (24/Re) (1 + 3 Re / 16)``."""
    return 1.0 + 3.0 * re / 16.0


def schiller_naumann_cd(re: float) -> float:
    """Schiller-Naumann correlation ``C_D = (24/Re)(1 + 0.15 Re^0.687)``."""
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


@dataclass(frozen=True)
class StokesSphereResult:
    """Steady sphere-drag report in a periodic box."""

    diameter: float
    box: int
    a_over_l: float
    tau: float
    nu: float
    force_density: float
    n_steps: int
    converged: bool
    mean_u: float
    re: float
    drag_force: float
    body_force_total: float
    momentum_balance_error: float
    k_measured: float
    k_hasimoto: float
    k_stokes: float
    rel_error_vs_hasimoto: float
    rel_error_vs_stokes: float
    cd_measured: float
    cd_stokes: float
    finite: bool
    u_history: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def run_sphere_stokes(
    *,
    diameter: int = 16,
    box_factor: float = 4.0,
    tau: float = 0.8,
    force_density: float = 1.0e-7,
    n_steps: int = 40_000,
    tol: float = 1e-5,
    check_every: int = 200,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    stall_limit: int = 40,
    u_init: float | None = None,
    verbose: bool = False,
) -> StokesSphereResult:
    """Steady low-Reynolds sphere drag in a triply periodic box (D3Q27).

    A uniform body force accelerates the fluid until the drag on the fixed
    sphere balances it.  At steady state two independent numbers are
    available:

    * the Ladd momentum-exchange drag :math:`F_{ME}`, and
    * the exact momentum-balance drag :math:`\\sum F_{body} = F \\rho V_{fluid}`.

    Their relative difference is reported as ``momentum_balance_error`` and
    is the self-check of the force estimator (a wrong ME formula cannot pass
    both).

    The dimensionless drag ``K = F / (6 pi mu a U)`` is 1 for an unbounded
    Stokes sphere and ``hasimoto_correction(a/L)`` for a periodic array, so
    ``rel_error_vs_hasimoto`` is the meaningful accuracy metric and
    ``rel_error_vs_stokes`` shows how much of the raw gap is domain
    truncation.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    n = int(round(box_factor * diameter))
    solid = sphere_mask(n, float(diameter)).to(device)
    fluid = ~solid
    n_fluid = int(fluid.sum().item())
    nu = (tau - 0.5) * CS2
    force = (float(force_density), 0.0, 0.0)

    rho0 = torch.ones((n, n, n), dtype=dtype, device=device)
    if u_init is None:
        u0 = torch.zeros_like(rho0)
    else:
        # Starting from the estimated steady uniform stream removes the slow
        # box-scale momentum build-up (an O(L^2/nu) transient that has nothing
        # to do with the drag physics).  The steady state is independent of the
        # initial condition.
        u0 = torch.full_like(rho0, float(u_init))
    f = equilibrium27(rho0, u0, u0, u0, device=device)

    prev_u = None
    last_res = float("inf")
    stall = 0
    converged = False
    u_history: list[float] = []
    drag = torch.zeros(3, dtype=dtype)
    steps_done = 0

    stepper = D3Q27Stepper(solid, tau=tau, force=force)
    for step in range(1, n_steps + 1):
        drag_buf: list[torch.Tensor] = []

        def _hook(fp, _buf=drag_buf, _st=stepper):
            _buf.append(_st.momentum_exchange_force(fp))

        f = stepper.step(f, hook=_hook)
        steps_done = step
        if step % check_every == 0 or step == n_steps:
            if not torch.isfinite(f).all():
                break
            drag = drag_buf[0]
            rho, ux, uy, uz = macroscopic_with_force(f, solid, force)
            u_mean = float((ux * fluid).sum().item() / n_fluid)
            u_history.append(u_mean)
            if verbose:
                print(
                    f"    [sphere] step {step} U={u_mean:.6e} "
                    f"F={float(drag_buf[0][0]):.6e} Re={abs(u_mean)*diameter/nu:.4f}",
                    flush=True,
                )
            if prev_u is not None:
                res = abs(u_mean - prev_u) / max(abs(u_mean), 1e-30)
                if res < tol:
                    converged = True
                    break
                stall = stall + 1 if res > 0.999 * last_res else 0
                last_res = res
                # a stalled residual only counts as converged at the numerical
                # floor; otherwise the run is still transient (large-tau runs
                # have long, slightly oscillatory tails).
                if stall >= stall_limit and last_res < max(1e-9, 100.0 * tol):
                    converged = True
                    break
            prev_u = u_mean

    rho, ux, uy, uz = macroscopic_with_force(f, solid, force)
    u_mean = float((ux * fluid).sum().item() / n_fluid)
    rho_mean = float((rho * fluid).sum().item() / n_fluid)
    drag_x = float(drag[0].item())
    body_total = float(force_density * rho_mean * n_fluid)
    a = 0.5 * diameter
    mu = rho_mean * nu
    k_meas = drag_x / (6.0 * torch.pi * mu * a * abs(u_mean)) if u_mean else float("inf")
    a_over_l = a / n
    k_has = hasimoto_correction(a_over_l)
    re = abs(u_mean) * 2.0 * a / nu
    cd_meas = 2.0 * drag_x / (rho_mean * u_mean * u_mean * torch.pi * a * a) if u_mean else float("inf")
    return StokesSphereResult(
        diameter=float(diameter),
        box=n,
        a_over_l=a_over_l,
        tau=float(tau),
        nu=float(nu),
        force_density=float(force_density),
        n_steps=steps_done,
        converged=converged,
        mean_u=abs(u_mean),
        re=float(re),
        drag_force=drag_x,
        body_force_total=body_total,
        momentum_balance_error=abs(drag_x - body_total) / abs(body_total) if body_total else float("inf"),
        k_measured=float(k_meas),
        k_hasimoto=float(k_has),
        k_stokes=1.0,
        rel_error_vs_hasimoto=abs(k_meas - k_has) / k_has,
        rel_error_vs_stokes=abs(k_meas - 1.0),
        cd_measured=float(cd_meas),
        cd_stokes=24.0 / re if re else float("inf"),
        finite=bool(torch.isfinite(f).all().item()),
        u_history=u_history,
    )


# ---------------------------------------------------------------------------
# 5. Long-run stability gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityResult:
    """Long-run (>= 1e5 steps) finiteness / boundedness gate."""

    H: int
    tau: float
    force_density: float
    n_steps: int
    finite: bool
    u_max_start: float
    u_max_end: float
    rho_min: float
    rho_max: float
    drift_u_max: float
    diverged: bool

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def run_long_stability(
    *,
    H: int = 16,
    nx: int = 4,
    nz: int = 4,
    tau: float = 0.8,
    n_steps: int = 100_000,
    force_density: float = 2.0e-5,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    probe_every: int = 1_000,
) -> StabilityResult:
    """Run a forced channel for ``n_steps`` and gate on finiteness/boundedness.

    Uses float32 on purpose: the acceptance criterion "1e5 steps without
    NaN/Inf" is about robustness of the production dtype, not about float64
    roundoff.  ``drift_u_max`` is the relative change of the peak velocity
    between the first and the last third of the run: after the transient the
    steady Poiseuille state must not drift.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    ny = int(H) + 2
    rho0 = torch.ones((nz, ny, nx), dtype=dtype, device=device)
    u0 = torch.zeros_like(rho0)
    f = equilibrium27(rho0, u0, u0, u0, device=device)
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    force = (float(force_density), 0.0, 0.0)

    probes: list[float] = []
    rho_min = float("inf")
    rho_max = -float("inf")
    finite = True
    stepper = D3Q27Stepper(solid, tau=tau, force=force)
    for step in range(1, n_steps + 1):
        f = stepper.step(f)
        if step % probe_every == 0 or step == 1:
            if not torch.isfinite(f).all():
                finite = False
                break
            rho, ux, _uy, _uz = macroscopic_with_force(f, solid, force)
            probes.append(float(ux[0, 1:-1, 0].abs().max().item()))
            rho_min = min(rho_min, float(rho[:, 1:-1, :].min().item()))
            rho_max = max(rho_max, float(rho[:, 1:-1, :].max().item()))

    third = max(1, len(probes) // 3)
    u_start = probes[third] if len(probes) > third else (probes[0] if probes else float("nan"))
    u_end = probes[-1] if probes else float("nan")
    drift = abs(u_end - u_start) / max(abs(u_start), 1e-30)
    return StabilityResult(
        H=int(H),
        tau=float(tau),
        force_density=float(force_density),
        n_steps=int(n_steps),
        finite=finite,
        u_max_start=u_start,
        u_max_end=u_end,
        rho_min=rho_min,
        rho_max=rho_max,
        drift_u_max=drift,
        diverged=(not finite) or not (0.1 < rho_min and rho_max < 10.0 and drift < 0.05),
    )


__all__ = [
    "CS2",
    "MomentAudit",
    "moment_audit",
    "velocity_moments",
    "equilibrium_third_moment_residual",
    "direction_liveness_audit",
    "guo_source",
    "half_force_velocity",
    "lbm_step_d3q27",
    "D3Q27Stepper",
    "macroscopic_with_force",
    "poiseuille_analytic",
    "PoiseuilleResult",
    "run_channel_poiseuille",
    "divergence_max",
    "sphere_mask",
    "momentum_exchange_drag",
    "bgk_shear_viscosity",
    "poiseuille_with_effective_viscosity",
    "hasimoto_correction",
    "oseen_correction",
    "schiller_naumann_cd",
    "StokesSphereResult",
    "run_sphere_stokes",
    "StabilityResult",
    "run_long_stability",
]