"""Conformation-tensor algebra, constitutive models and realizability projection.

The conformation tensor is stored as a ``(4, ny, nx)`` tensor with component
order ``[Cxx, Cxy, Cyy, Czz]`` (the 2-D ``zz`` component is carried because it
enters ``tr C`` and the FENE-P stretch function ``f(tr C)``).  All fields live
on the D2Q9 lattice, rows ``1..H`` are fluid and rows ``0``/``H+1`` are walls.

Constitutive evolution (upper-convected, conformation-tensor form)::

    dC/dt + u.grad(C) = L C + C L^T - (f(tr C) C - a I) / lambda

with ``L = grad(u)``, ``f = 1`` and ``a = 1`` for Oldroyd-B.  Note the strain
term is ``L C + C L^T`` (not ``L^T C + C L``): only this form gives the textbook
steady simple-shear solution ``C_xx = 1 + 2 Wi^2``, ``C_xy = Wi``, ``C_yy = 1``
with ``Wi = lambda*gdot``; the transposed variant silently produces
``C_yy = 1 + 2 Wi^2`` instead, which is caught by
``tests/test_viscoelastic_toms.py::test_oldroyd_b_steady_shear_conformation``.

Time integration: the strain + relaxation part is solved *implicitly* (a 3x3
linear solve per node, unconditionally stable for any ``lambda`` including the
Newtonian limit ``lambda -> 0``); advection is explicit (central differences --
it vanishes identically in the validated plane-Poiseuille case, which is
x-uniform, so it is not exercised by the accuracy gates).  See
``docs/VISCOELASTIC_P50_N06.md`` for the scope statement: a distribution-function
(Fokker-Planck) LBM for ``C`` is *not* implemented; ``C`` is transported on the
LBM lattice with a finite-difference advection + implicit local solve, coupled to
the LBM momentum solver through the divergence of the polymer stress.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "COMP_XX",
    "COMP_XY",
    "COMP_YY",
    "COMP_ZZ",
    "ConformationState",
    "ViscoelasticDivergenceError",
    "advance_conformation",
    "check_realizability",
    "conformation_from_components",
    "ddx_periodic",
    "ddy_2nd",
    "elastic_energy",
    "fene_p_f",
    "fene_p_steady_shear_conformation",
    "fene_p_stress",
    "min_eigenvalue",
    "oldroyd_b_stress",
    "project_realizable",
    "trace",
    "velocity_gradients_2d",
]

COMP_XX, COMP_XY, COMP_YY, COMP_ZZ = 0, 1, 2, 3


class ViscoelasticDivergenceError(RuntimeError):
    """Raised when the viscoelastic state leaves its physical domain.

    Used for the "Wi above the limit" negative control: instead of returning a
    silently non-physical field the solver reports divergence explicitly.
    """


# ---------------------------------------------------------------------------
# gradients on the channel lattice
# ---------------------------------------------------------------------------


def ddy_2nd(field: torch.Tensor) -> torch.Tensor:
    """Second-order ``d/dy`` on rows 1..H with one-sided stencils at the edges.

    Interior rows use the central difference; the first/last fluid rows use the
    three-point one-sided stencil (``(-3 f0 + 4 f1 - f2) / 2``).  For a
    quadratic field (plane Poiseuille) the operator is *exact*, which is what
    lets the Oldroyd-B fixed point reproduce the analytic parabola to round-off.
    Rows ``0``/``-1`` are walls; their entries are not used by the solver.
    """
    f = field
    g = torch.zeros_like(f)
    g[..., 1:-1, :] = 0.5 * (f[..., 2:, :] - f[..., :-2, :])
    # second-order one-sided stencils on the *first/last fluid rows* (the wall
    # rows 0 / -1 are never used as stencil centres because their derivative is
    # not needed; using them in the central stencil of row 1 would inject the
    # wall value and destroy the exactness on a parabola).
    g[..., 1, :] = 0.5 * (-3.0 * f[..., 1, :] + 4.0 * f[..., 2, :] - f[..., 3, :])
    g[..., -2, :] = 0.5 * (3.0 * f[..., -2, :] - 4.0 * f[..., -3, :] + f[..., -4, :])
    return g


def ddx_periodic(field: torch.Tensor) -> torch.Tensor:
    """Second-order central ``d/dx`` with periodic wrap (channel is x-periodic)."""
    return 0.5 * (torch.roll(field, -1, dims=-1) - torch.roll(field, 1, dims=-1))


def velocity_gradients_2d(
    ux: torch.Tensor, uy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``L_ij = d u_i / d x_j`` as ``(Lxx, Lxy, Lyx, Lyy)``."""
    return (
        ddx_periodic(ux),
        ddy_2nd(ux),
        ddx_periodic(uy),
        ddy_2nd(uy),
    )


# ---------------------------------------------------------------------------
# tensor algebra
# ---------------------------------------------------------------------------


def conformation_from_components(
    cxx, cxy, cyy, czz
) -> torch.Tensor:
    """Stack components into the ``(4, ny, nx)`` conformation tensor."""
    return torch.stack([cxx, cxy, cyy, czz], dim=0)


def trace(C: torch.Tensor) -> torch.Tensor:
    """``tr C = C_xx + C_yy + C_zz``."""
    return C[COMP_XX] + C[COMP_YY] + C[COMP_ZZ]


def _eigenvalues_2x2(cxx, cxy, cyy):
    """Eigenvalues of the 2x2 block ``[[cxx, cxy], [cxy, cyy]]`` (analytic)."""
    half_trace = 0.5 * (cxx + cyy)
    disc = torch.sqrt(torch.clamp(0.25 * (cxx - cyy) ** 2 + cxy**2, min=0.0))
    return half_trace + disc, half_trace - disc


def min_eigenvalue(C: torch.Tensor) -> torch.Tensor:
    """Smallest eigenvalue of ``C`` (``C_zz`` decouples in 2-D)."""
    lam_hi, lam_lo = _eigenvalues_2x2(C[COMP_XX], C[COMP_XY], C[COMP_YY])
    return torch.minimum(lam_lo, C[COMP_ZZ])


def check_realizability(
    C: torch.Tensor, *, floor: float = 0.0, eps: float = 1e-9
) -> dict:
    """Realizability gate for the conformation tensor.

    The realizable set is

    * ``C`` symmetric positive semi-definite: ``min eigenvalue >= floor - eps``
      (a Hookean / FENE dumbbell requires ``C = <R R> >= 0``; note that for
      Oldroyd-B in steady shear one eigenvalue drops *below unity*
      (``1 + Wi^2 - Wi sqrt(1 + Wi^2)``, e.g. 0.586 at ``Wi = 1``), so
      ``C >= I`` is **not** a valid constraint -- enforcing it corrupts the
      constitutive solution, which is exactly what an earlier version of this
      gate did), and
    * ``tr C >= 3 - eps``: the kinematic consequence of ``<R R>`` for a dumbbell
      (at equilibrium ``C = I`` and ``tr C = 3``).

    ``tr C >= 3`` alone is *not* sufficient (a tensor can have ``tr C = 3.5`` and
    a negative eigenvalue), which is why both numbers are returned and tested.
    """
    eig_min = min_eigenvalue(C)
    tr = trace(C)
    eig_min_val = float(eig_min.min())
    tr_min_val = float(tr.min())
    ok = bool(eig_min_val >= floor - eps and tr_min_val >= 3.0 - eps)
    return {
        "ok": ok,
        "min_eigenvalue": eig_min_val,
        "min_trace": tr_min_val,
        "floor": float(floor),
        "eps": float(eps),
    }


def project_realizable(
    C: torch.Tensor,
    *,
    floor: float = 0.0,
    trace_cap: float | None = None,
) -> torch.Tensor:
    """Eigen-projection of ``C`` onto the realizable set.

    Steps (all of them are algebraic, no iteration):

    1. eigendecompose the symmetric 2x2 ``(xx, xy, yy)`` block analytically and
       clamp every eigenvalue from below at ``floor`` (default ``0``: positive
       semi-definiteness).  ``C_zz`` is clamped the same way.
    2. if ``trace_cap`` is given (FENE-P: ``L^2``) rescale the deviation from
       ``I`` so that ``tr C < trace_cap``.
    3. enforce the kinematic trace floor ``tr C >= 3`` by scaling the deviatoric
       part up when needed (this is what makes "projected => ``tr C >= 3 - 1e-9``
       everywhere" hold by construction, per the mission criterion).

    Returns a new tensor; the input is not modified.
    """
    cxx, cxy, cyy = C[COMP_XX], C[COMP_XY], C[COMP_YY]
    lam_hi, lam_lo = _eigenvalues_2x2(cxx, cxy, cyy)
    lam_hi = torch.clamp(lam_hi, min=floor)
    lam_lo = torch.clamp(lam_lo, min=floor)
    # eigenvector angle of the 2x2 block
    theta = 0.5 * torch.atan2(2.0 * cxy, cxx - cyy)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    new_cxx = lam_hi * cos_t**2 + lam_lo * sin_t**2
    new_cyy = lam_hi * sin_t**2 + lam_lo * cos_t**2
    new_cxy = (lam_hi - lam_lo) * sin_t * cos_t
    new_czz = torch.clamp(C[COMP_ZZ], min=floor)
    out = conformation_from_components(new_cxx, new_cxy, new_cyy, new_czz)

    dev = out.clone()
    if trace_cap is not None:
        limit = float(trace_cap) * (1.0 - 1e-12)
        tr = trace(out)
        scale = torch.where(tr > limit, limit / torch.clamp(tr, min=1e-30), torch.ones_like(tr))
        dev[COMP_XX] = (out[COMP_XX] - 1.0) * scale
        dev[COMP_XY] = out[COMP_XY] * scale
        dev[COMP_YY] = (out[COMP_YY] - 1.0) * scale
        dev[COMP_ZZ] = (out[COMP_ZZ] - 1.0) * scale
        out = conformation_from_components(
            dev[COMP_XX] + 1.0, dev[COMP_XY], dev[COMP_YY] + 1.0, dev[COMP_ZZ] + 1.0
        )
    # kinematic trace floor
    tr = trace(out)
    need = tr < 3.0
    if bool(need.any()):
        scale_up = torch.where(need, 2.0 / torch.clamp(tr - 1.0, min=1e-30), torch.ones_like(tr))
        out = conformation_from_components(
            (out[COMP_XX] - 1.0) * scale_up + 1.0,
            out[COMP_XY] * scale_up,
            (out[COMP_YY] - 1.0) * scale_up + 1.0,
            (out[COMP_ZZ] - 1.0) * scale_up + 1.0,
        )
    return out


# ---------------------------------------------------------------------------
# stresses
# ---------------------------------------------------------------------------


def oldroyd_b_stress(
    C: torch.Tensor, eta_p: float, lam: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``tau_p = (eta_p / lambda) (C - I)`` -> ``(tau_xx, tau_xy, tau_yy)``.

    ``N1 = tau_xx - tau_yy``; in steady shear ``tau_xy = eta_p * gdot``
    (shear-rate independent, the Oldroyd-B signature) and
    ``N1 = 2 eta_p lambda gdot^2``.
    """
    scale = float(eta_p) / float(lam)
    tau_xx = scale * (C[COMP_XX] - 1.0)
    tau_xy = scale * C[COMP_XY]
    tau_yy = scale * (C[COMP_YY] - 1.0)
    return tau_xx, tau_xy, tau_yy


def fene_p_f(C: torch.Tensor, L2: float) -> torch.Tensor:
    """Peterlin function ``f(tr C) = 1 / (1 - tr C / L^2)`` (``tr C < L^2``)."""
    tr = trace(C)
    denom = torch.clamp(1.0 - tr / float(L2), min=1.0 / float(L2))
    return 1.0 / denom


def fene_p_parameters(L2: float) -> float:
    """``a = 1 / (1 - 3 / L^2)`` (Peterlin offset keeping ``tau_p = 0`` at rest)."""
    if L2 <= 3.0:
        msg = f"FENE-P requires L^2 > 3 (got L^2={L2})"
        raise ValueError(msg)
    return 1.0 / (1.0 - 3.0 / float(L2))


def fene_p_stress(
    C: torch.Tensor, eta_p: float, lam: float, L2: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``tau_p = (eta_p / lambda) (f(tr C) C - a I)``."""
    f = fene_p_f(C, L2)
    a = fene_p_parameters(L2)
    scale = float(eta_p) / float(lam)
    tau_xx = scale * (f * C[COMP_XX] - a)
    tau_xy = scale * (f * C[COMP_XY])
    tau_yy = scale * (f * C[COMP_YY] - a)
    return tau_xx, tau_xy, tau_yy


# ---------------------------------------------------------------------------
# evolution
# ---------------------------------------------------------------------------


@dataclass
class ConformationState:
    """Conformation tensor plus the constitutive parameters it is evolved with."""

    tensor: torch.Tensor
    model: str = "oldroyd_b"
    lam: float = 1.0
    eta_p: float = 0.0
    L2: float | None = None

    def trace(self) -> torch.Tensor:
        return trace(self.tensor)

    def min_eigenvalue(self) -> torch.Tensor:
        return min_eigenvalue(self.tensor)

    def realizability(self, eps: float = 1e-9) -> dict:
        return check_realizability(self.tensor, eps=eps)

    def stress(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model == "fene_p":
            if self.L2 is None:
                msg = "FENE-P stress requires L2"
                raise ValueError(msg)
            return fene_p_stress(self.tensor, self.eta_p, self.lam, self.L2)
        return oldroyd_b_stress(self.tensor, self.eta_p, self.lam)


def _advective_increment(
    C: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor
) -> torch.Tensor:
    """Explicit ``-u.grad(C)`` for every component, ``(4, ny, nx)``."""
    # vectorised over the four components (single roll/slice pass each)
    return -(ux.unsqueeze(0) * ddx_periodic(C) + uy.unsqueeze(0) * ddy_2nd(C))


def advance_conformation(
    C: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    *,
    dt: float,
    lam: float,
    model: str = "oldroyd_b",
    L2: float | None = None,
    projection: bool = True,
    implicit_strain: bool = True,
    strain_scale: float = 1.0,
    advection: bool = True,
    trace_cap: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """One constitutive time step (implicit strain/relaxation by default).

    Args:
        C: conformation tensor ``(4, ny, nx)``.
        ux, uy: velocity fields ``(ny, nx)``.
        dt: time step (lattice units).
        lam: relaxation time ``lambda`` (> 0).
        model: ``"oldroyd_b"`` or ``"fene_p"``.
        L2: finite extensibility ``L^2`` (FENE-P only).
        projection: apply :func:`project_realizable` after the update.
        implicit_strain: solve the strain + relaxation part implicitly.  Setting
            this to ``False`` is the negative-control path (forward Euler loses
            positive semi-definiteness at high Weissenberg number).
        strain_scale: multiplies the upper-convective strain term; ``0`` is the
            negative control for "upper-convective derivative switched off".
        advection: include ``-u.grad(C)``.
        trace_cap: absolute trace ceiling used by the projection (FENE-P: ``L^2``).

    Returns:
        ``(C_new, info)`` where ``info`` carries the realizability diagnostics and
        a ``diverged`` flag.  Raises :class:`ViscoelasticDivergenceError` when the
        state becomes non-finite or leaves the model's trace domain.
    """
    if lam <= 0.0:
        msg = f"relaxation time must be positive (got lambda={lam})"
        raise ValueError(msg)
    if model == "fene_p" and L2 is None:
        msg = "FENE-P requires L2"
        raise ValueError(msg)

    floor = 0.0  # SPD floor; ``C >= I`` is NOT physical for Oldroyd-B (see check_realizability)
    C_star = C - dt * _advective_increment(C, ux, uy) if advection else C
    Lxx, Lxy, Lyx, Lyy = velocity_gradients_2d(ux, uy)
    if strain_scale != 1.0:
        Lxx = Lxx * strain_scale
        Lxy = Lxy * strain_scale
        Lyx = Lyx * strain_scale
        Lyy = Lyy * strain_scale

    # FENE-P relaxation factor f(tr C) is lagged/iterated; Oldroyd-B has f = 1.
    f = torch.ones_like(C_star[COMP_XX]) if model == "oldroyd_b" else fene_p_f(C_star, L2)  # type: ignore[arg-type]
    a = 1.0 if model == "oldroyd_b" else fene_p_parameters(L2)  # type: ignore[arg-type]

    def _update(f_field: torch.Tensor) -> torch.Tensor:
        if not implicit_strain:
            # explicit forward Euler (negative-control path)
            cxx, cxy, cyy, czz = (
                C_star[COMP_XX],
                C_star[COMP_XY],
                C_star[COMP_YY],
                C_star[COMP_ZZ],
            )
            s_xx = 2.0 * (Lxx * cxx + Lxy * cxy)
            s_xy = Lxy * cyy + Lyx * cxx + cxy * (Lxx + Lyy)
            s_yy = 2.0 * (Lyy * cyy + Lyx * cxy)
            new_xx = cxx + dt * (s_xx - (f_field * cxx - a) / lam)
            new_xy = cxy + dt * (s_xy - (f_field * cxy) / lam)
            new_yy = cyy + dt * (s_yy - (f_field * cyy - a) / lam)
            new_zz = czz + dt * (-(f_field * czz - a) / lam)
            return conformation_from_components(new_xx, new_xy, new_yy, new_zz)

        relax = 1.0 / dt + f_field / lam
        # Solve for the *deviatoric* part Y = C - I.  For lambda -> 0 the
        # unknown becomes O(lambda) instead of O(1), which removes the
        # catastrophic cancellation between the 1/lambda relaxation and the
        # strain term (the C-form loses ~3 digits at Wi = 1e-3).
        a11 = relax - 2.0 * Lxx
        a12 = -2.0 * Lxy
        a21 = -Lyx
        a22 = relax - (Lxx + Lyy)
        a23 = -Lxy
        a32 = -2.0 * Lyx
        a33 = relax - 2.0 * Lyy
        shift = (f_field - a) / lam
        b0 = (C_star[COMP_XX] - 1.0) / dt + 2.0 * Lxx - shift
        b1 = C_star[COMP_XY] / dt + Lxy + Lyx
        b2 = (C_star[COMP_YY] - 1.0) / dt + 2.0 * Lyy - shift
        # Cramer's rule on the 3x3 upper-convected block (cheaper than a batched
        # LAPACK solve for a lattice of 3x3 systems)
        m22 = a22 * a33 - a23 * a32
        r1 = b1 * a33 - a23 * b2
        det = a11 * m22 - a12 * a21 * a33
        yxx = (b0 * m22 - a12 * r1) / det
        yxy = (a11 * r1 - b0 * a21 * a33) / det
        yyy = (a11 * (a22 * b2 - b1 * a32) - a12 * a21 * b2 + b0 * a21 * a32) / det
        yzz = ((C_star[COMP_ZZ] - 1.0) / dt - shift) / relax
        return conformation_from_components(yxx + 1.0, yxy, yyy + 1.0, yzz + 1.0)

    C_new = _update(f)
    if model == "fene_p":
        for _ in range(4):  # fixed point on f(tr C)
            C_new = _update(fene_p_f(C_new, L2))  # type: ignore[arg-type]

    if projection:
        cap = trace_cap
        if cap is None and model == "fene_p":
            cap = L2
        C_new = project_realizable(C_new, floor=floor, trace_cap=cap)

    info = check_realizability(C_new, floor=floor)
    finite = bool(torch.isfinite(C_new).all())
    if not finite:
        wi = float(dt) * float(lam)
        msg = (
            "viscoelastic state diverged (non-finite conformation tensor) at "
            f"lambda={lam:g}, dt={dt:g} (lambda/dt={wi:g}, explicit_strain={not implicit_strain})"
        )
        raise ViscoelasticDivergenceError(msg)
    tr_max = float(trace(C_new).max())
    info["max_trace"] = tr_max
    if model == "fene_p" and tr_max >= float(L2):
        msg = (
            f"FENE-P finite extensibility exceeded: tr C={tr_max:g} >= L^2={L2:g} "
            f"at lambda={lam:g}"
        )
        raise ViscoelasticDivergenceError(msg)
    info["diverged"] = False
    return C_new, info


# ---------------------------------------------------------------------------
# homogeneous-shear reference solutions
# ---------------------------------------------------------------------------


def oldroyd_b_steady_shear_conformation(Wi: float) -> tuple[float, float, float, float]:
    """Exact Oldroyd-B steady simple-shear conformation ``(Cxx, Cxy, Cyy, Czz)``."""
    return 1.0 + 2.0 * Wi * Wi, Wi, 1.0, 1.0


def fene_p_steady_shear_conformation(
    Wi: float, L2: float, *, n_iter: int = 400
) -> tuple[float, float, float, float]:
    """Steady simple-shear FENE-P conformation by damped fixed-point iteration.

    Solves ``0 = 2 Wi C_xy - (f C_xx - a)``, ``0 = Wi C_yy - f C_xy``,
    ``0 = -(f C_yy - a)`` (Weissenberg-scaled), which for ``L^2 -> inf`` recovers
    ``(1 + 2 Wi^2, Wi, 1, 1)``.
    """
    a = fene_p_parameters(L2)
    cxx, cxy, cyy = 1.0 + 2.0 * Wi * Wi, Wi, 1.0
    for _ in range(n_iter):
        tr = cxx + cyy + 1.0
        f = 1.0 / max(1.0 - tr / L2, 1.0 / L2)
        # fixed point of the algebraic system
        cyy_new = a / f
        cxy_new = Wi * cyy_new / f
        cxx_new = (a + 2.0 * Wi * cxy_new) / f
        cxx, cxy, cyy = (0.5 * (cxx + cxx_new), 0.5 * (cxy + cxy_new), 0.5 * (cyy + cyy_new))
    tr = cxx + cyy + 1.0
    if tr >= L2:
        msg = f"FENE-P steady shear not realizable at Wi={Wi:g}, L^2={L2:g} (tr C={tr:g})"
        raise ViscoelasticDivergenceError(msg)
    return cxx, cxy, cyy, 1.0

def elastic_energy(
    C: torch.Tensor, *, nu_p: float, lam: float, x_average: bool = True
) -> float:
    """Elastic free energy per unit width, ``sum_j (nu_p / (2 lambda)) tr(C - I)``.

    Wall rows are excluded (they carry ``C = I`` by construction).  Its time
    derivative vanishes at steady state, which is the term that closes the
    energy budget in :mod:`tensorlbm.viscoelastic.energy`.
    """
    per_cell = (float(nu_p) / (2.0 * float(lam))) * (trace(C) - 3.0)[1:-1]
    if x_average and per_cell.dim() == 2:
        per_cell = per_cell.mean(dim=1)
    return float(per_cell.sum())
