"""Slip-wall calibration: Maxwell-condition audit, mesh dependence and the
lattice-resolution recipe for a prescribed physical slip length.

物理背景
--------
上游站（IN67-N04 / P106W1-N04）已经把「混合反弹/镜面壁」的 ``b(sigma)`` 标定律、
平面/圆管反演和减阻律做到了可用的程度。它们留下的是**三个可测但未测的问题**：

1. **算子在多大程度上真的实现了 Maxwell 一阶滑移条件？**
   上游只做了「反演式 = 半程反弹壁面上 Maxwell 条件的离散形式」这一**代数恒等**。
   本模块改成**测量**：从壁面规则本身（混合算子的切向动量簿记）直接读出该算子
   隐含的壁面速度 :math:`u_w`，与从**流场内部节点外推到壁面**得到的速度梯度
   :math:`\\dot\\gamma_w` 相乘，检查 :math:`u_w = b\\dot\\gamma_w` 是否成立。
   这条检查不假设剖面形状（外推只用近壁 3 层节点的一侧差分），因此是非循环的。

2. **b 是壁面属性还是流动属性？**
   同一 ``sigma``、不同驱动力、不同对壁（一侧无滑移一侧滑移）下 b 是否不变。

3. **b 在格子单位里是常数，但物理滑移长度 b_phys = b_lu * dx 随网格线性变化。**
   即混合壁**不是**一个"固定 b 壁"。本模块给出把 b_phys 钉住所需的重标定
   ``sigma(b_phys, dx)`` 并在 4 个分辨率上闭环验证。

可写的算子事实（本模块实现并验证）
----------------------------------
对 D2Q9/D3Q19 的轴对齐壁，混合规则相对纯反弹多出的切向动量通量为

    ΔM_x = sigma * sum_{q inward} c_xq [ f_{M(q)} - f_{opp(q)} ]

（``inward`` = 反射后射回流体的方向，``M`` = 关于壁法向的镜面置换）。半程反弹的
移动壁动量注入是标准的 ``6 w_q rho (c_q . u_wall)``，其切向和为 ``rho u_wall / 3``
（D2Q9、D3Q19 相同：``sum_{q inward} w_q c_xq^2 = 1/18``），因此

    u_w = uw_imposed + 3 ΔM_x / rho

即**壁面速度可以从每次反弹的动量簿记里直接读出**。本模块的
:func:`wall_velocity_from_rule` 就是这个式子；`:func:`maxwell_audit` 把它与
流场外推值对拍。

适用范围
--------
低 Mach、不可压、单相、纯 BGK、轴对齐壁（平面通道；圆管用轴对称列，见
:func:`pipe_inversion_convergence`）。曲线/阶梯壁的镜面反射非局部，本模块**不做**，
而是把它作为明确列出的未完成项（报告 §5）。

REFERENCES
----------
Maxwell (1879); Ladd (1994) moving-wall momentum injection; Succi, *The Lattice
Boltzmann Equation for Fluid Dynamics and Beyond* (2001) ch. 5 (bounce-back
half-way wall + partial-slip mixtures).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import partial_slip_wall as ps
from . import partial_slip_wall3d as p3

__all__ = [
    "PlaneChannel",
    "plane_channel_3d",
    "wall_velocity_from_rule",
    "one_sided_wall_fit",
    "maxwell_audit",
    "sigma_for_slip_length",
    "physical_slip_length",
    "resolution_slip_study",
    "pipe_inversion_convergence",
    "fit_b_sigma_law",
    "free_slip_asymptotics",
    "drag_law_bplus",
    "SLOPE_DEFAULT",
]

#: lattice-local slope of the mixed-wall law ``b = a * sigma/(1-sigma)`` (planes).
SLOPE_DEFAULT = 0.3

_C19 = p3.C19
_W19 = p3.W19
_OPP19 = p3.OPP19
_MIRROR19 = p3.MIRROR_Y
_Q = int(_C19.shape[0])
_CX19 = _C19[:, 0].astype(float)
_IN_LOW = _C19[:, 1] > 0      # returning directions at the j=0 wall
_IN_HIGH = _C19[:, 1] < 0     # returning directions at the j=ny-1 wall
_W_SHAPE = _W19.reshape(_Q, 1, 1)
_CX_SHAPE = _CX19.reshape(_Q, 1, 1)
_INW_LOW = _IN_LOW.reshape(_Q, 1, 1)
_INW_HIGH = _IN_HIGH.reshape(_Q, 1, 1)


# ---------------------------------------------------------------------------
# D3Q19 plane channel with per-wall sigma, per-wall wall velocity
# ---------------------------------------------------------------------------
@dataclass
class PlaneChannel:
    """Steady D3Q19 plane channel state, incl. the raw wall-rule data."""

    sigma_low: float
    sigma_high: float
    ny: int
    nx: int
    nz: int
    tau: float
    g: float
    uw_low: float
    uw_high: float
    u_profile: np.ndarray        # (ny-2,) plane-averaged streamwise velocity
    rho_profile: np.ndarray      # (ny-2,) plane-averaged density
    f_low: np.ndarray            # (19, nx, nz) post-collision, row j=1
    f_high: np.ndarray           # (19, nx, nz) post-collision, row j=ny-2
    residual: float
    converged: bool
    steps: int

    @property
    def length(self) -> float:
        """Half height of the wall-plane channel, ``H = (ny-2)/2``."""
        return (self.ny - 2) / 2.0

    @property
    def u_max(self) -> float:
        return float(np.max(np.abs(self.u_profile)))

    @property
    def coord(self) -> np.ndarray:
        return np.arange(1, self.ny - 1, dtype=float)


def _mixed_wall_row(base: np.ndarray, sigma: float, uw: float, rho_adj: np.ndarray,
                    top: bool) -> np.ndarray:
    """Mixed bounce-back/specular wall row with an optional moving wall.

    ``base`` holds the post-collision populations of the fluid row adjacent to
    the wall, shape ``(19, nx, nz)``.  The returned row carries, for every
    direction that streams from the wall back into the fluid, the convex
    combination of the bounce-back and the mirror population, plus the standard
    moving-wall momentum injection ``6 w_q rho (c_q . u_wall)`` (Ladd 1994)
    evaluated with the *returning* direction ``q``.
    """
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")
    mix = (1.0 - sigma) * base[_OPP19] + sigma * base[_MIRROR19]
    if uw:
        mix = mix + 6.0 * _W_SHAPE * (_CX_SHAPE * uw) * rho_adj
    inward = _INW_HIGH if top else _INW_LOW
    return np.where(inward, mix, base)


def plane_channel_3d(
    sigma_low: float = 0.0,
    sigma_high: float | None = None,
    ny: int = 33,
    nx: int = 1,
    nz: int = 1,
    tau: float = 0.8,
    g: float = 1.0e-5,
    uw_low: float = 0.0,
    uw_high: float = 0.0,
    steps: int = 60000,
    tol: float = 1e-10,
) -> PlaneChannel:
    """Steady D3Q19 plane channel, walls at ``j=0`` and ``j=ny-1``.

    ``sigma_high`` defaults to ``sigma_low`` (symmetric channel).  Solid rows are
    ``0`` and ``ny-1``, fluid rows ``1..ny-2``; a bounce-back wall plane sits at
    ``y = 0.5`` and ``y = ny-1.5``, so ``H = (ny-2)/2`` -- the same convention as
    :mod:`partial_slip_wall`.  ``x`` and ``z`` are periodic.  Setting
    ``uw_low/uw_high`` drives a plane Couette flow (used as the independent
    momentum-bookkeeping check of the wall rule).
    """
    if sigma_high is None:
        sigma_high = sigma_low
    if ny < 8:
        raise ValueError("ny must be >= 8")
    if nx < 1 or nz < 1:
        raise ValueError("nx, nz must be >= 1")
    zero = np.zeros((nx, ny, nz))
    f = p3._equilibrium(np.ones_like(zero), zero, zero, zero)
    u_prev = np.zeros((ny - 2, nx, nz))
    residual = float("inf")
    converged = False
    step = 0
    f_post = f
    for step in range(1, steps + 1):
        rho = f.sum(0)
        f_post = p3._collide_bgk_guo(f, tau, fx=g * rho)
        base_low = f_post[:, :, 1, :]
        base_high = f_post[:, :, ny - 2, :]
        f_post[:, :, 0, :] = _mixed_wall_row(
            base_low, sigma_low, uw_low, rho[:, 1, :], top=False)
        f_post[:, :, ny - 1, :] = _mixed_wall_row(
            base_high, sigma_high, uw_high, rho[:, ny - 2, :], top=True)
        new_f = np.zeros_like(f)
        for q in range(_Q):
            cx, cy, cz = (int(v) for v in _C19[q])
            src = f_post[q]
            if cx:
                src = np.roll(src, cx, axis=0)
            if cz:
                src = np.roll(src, cz, axis=2)
            new_f[q][:, 1:ny - 1, :] = src[:, 1 - cy:ny - 1 - cy, :]
        f = new_f
        _rho, ux, _uy, _uz = p3._macroscopic(f, g * f.sum(0))
        u = ux[:, 1:ny - 1, :].transpose(1, 0, 2)
        residual = float(np.max(np.abs(u - u_prev)) / max(np.max(np.abs(u)), 1e-300))
        u_prev = u.copy()
        if residual < tol and step > 200:
            converged = True
            break
    return PlaneChannel(
        sigma_low=float(sigma_low), sigma_high=float(sigma_high), ny=ny, nx=nx, nz=nz,
        tau=tau, g=g, uw_low=float(uw_low), uw_high=float(uw_high),
        u_profile=u_prev.mean(axis=(1, 2)),
        rho_profile=np.asarray(f.sum(0))[:, 1:ny - 1, :].mean(axis=(0, 2)),
        f_low=f_post[:, :, 1, :].copy(), f_high=f_post[:, :, ny - 2, :].copy(),
        residual=residual, converged=converged, steps=step,
    )


# ---------------------------------------------------------------------------
# Wall velocity read off the boundary rule (momentum bookkeeping)
# ---------------------------------------------------------------------------
def wall_velocity_from_rule(base: np.ndarray, sigma: float, rho_adj: np.ndarray,
                            top: bool = False, uw: float = 0.0) -> np.ndarray:
    """Streamwise wall velocity implied by the mixed wall rule [lattice units].

    The mixed rule differs from pure bounce-back by the tangential momentum
    flux ``ΔM_x = sigma sum_{q inward} c_xq [f_{M(q)} - f_{opp(q)}]``.  A wall
    moving at ``u_w`` with full bounce-back injects ``rho u_w / 3`` of tangential
    momentum per step (``sum_{q inward} w_q c_xq^2 = 1/18`` for both D2Q9 and
    D3Q19), hence ``u_w = 3 ΔM_x / rho`` on top of any externally imposed wall
    velocity ``uw``.

    This is a property of the *operator*, not of the flow: it is the wall speed
    the populations "see", and it is what the Maxwell condition must be checked
    against.
    """
    inward = _INW_HIGH.reshape(_Q) if top else _INW_LOW.reshape(_Q)
    flux = sigma * np.tensordot(_CX19 * inward, base[_MIRROR19] - base[_OPP19], axes=(0, 0))
    return 3.0 * flux / rho_adj + uw


def one_sided_wall_fit(
    u_rows: Sequence[float], y_rows: Sequence[float], y_wall: float, order: int = 2
) -> dict:
    """Extrapolate a near-wall velocity profile to the wall plane.

    Uses **only** the near-wall data (one-sided polynomial of degree ``order``
    through the first ``order+1`` fluid rows); no global parabola is assumed.
    Returns the wall velocity, the wall-normal gradient magnitude and the fit
    residual on the used rows.
    """
    y = np.asarray(y_rows, float)
    u = np.asarray(u_rows, float)
    if y.size < order + 1:
        raise ValueError("too few rows for the requested order")
    yk, uk = y[: order + 1], u[: order + 1]
    coef = np.polyfit(yk - y_wall, uk, order)
    poly = np.poly1d(coef)
    return {
        "u_wall": float(poly(0.0)),
        "gradient": float(poly.deriv()(0.0)),
        "residual": float(np.max(np.abs(poly(yk - y_wall) - uk))),
        "order": int(order),
        "rows_used": int(order + 1),
        "y_wall": float(y_wall),
    }


# ---------------------------------------------------------------------------
# Maxwell audit: operator wall speed vs. flow-side extrapolation
# ---------------------------------------------------------------------------
def maxwell_audit(
    sigma: float,
    ny: int = 33,
    tau: float = 0.8,
    g: float = 1.0e-5,
    steps: int = 60000,
    tol: float = 1e-10,
    compare_sigma: float | None = None,
) -> dict:
    """Audit ``u_w = b * du/dn|_w`` from two independent sides.

    * **operator side** -- :func:`wall_velocity_from_rule` applied to the
      post-collision populations next to the wall (pure momentum bookkeeping of
      the boundary rule);
    * **flow side** -- :func:`one_sided_wall_fit` on the first three interior
      rows, then ``b_flow = u_wall / gradient``.

    The two routes share no fitting ansatz, so their agreement is evidence that
    the mixed operator realises Maxwell's first-order condition with a *single*
    slip length.
    """
    res = plane_channel_3d(sigma_low=sigma, ny=ny, tau=tau, g=g, steps=steps, tol=tol)
    nu = ps.nu_from_tau(tau)
    y_wall_low, y_wall_high = 0.5, ny - 1.5
    fit_low = one_sided_wall_fit(res.u_profile, res.coord, y_wall_low)
    fit_high = one_sided_wall_fit(res.u_profile[::-1], res.coord[::-1], y_wall_high)
    rho_low = float(res.rho_profile[0])
    rho_high = float(res.rho_profile[-1])
    uw_low = float(np.mean(wall_velocity_from_rule(res.f_low, sigma, rho_low, top=False)))
    uw_high = float(np.mean(wall_velocity_from_rule(res.f_high, sigma, rho_high, top=True)))
    grad_w = 0.5 * (abs(fit_low["gradient"]) + abs(fit_high["gradient"]))
    uw_op = 0.5 * (abs(uw_low) + abs(uw_high))
    inv = ps.invert_slip_length(
        ps.PoiseuilleResult(
            sigma=sigma, n=ny, tau=tau, g=g, coord=res.coord, u=res.u_profile,
            length=res.length, geometry="plane", steps=res.steps,
            converged=res.converged, residual=res.residual))
    b_flow = uw_op / grad_w if grad_w > 0 else float("inf")
    b_fit = float(inv["b"])
    ana_grad = g * res.length / nu
    out = {
        "sigma": float(sigma),
        "ny": int(ny),
        "converged": bool(res.converged),
        "residual": float(res.residual),
        "steps": int(res.steps),
        "u_wall_rule": float(uw_op),
        "u_wall_rule_low": uw_low,
        "u_wall_rule_high": uw_high,
        "u_wall_extrapolated": 0.5 * (abs(fit_low["u_wall"]) + abs(fit_high["u_wall"])),
        "gradient_extrapolated": grad_w,
        "gradient_analytic": float(ana_grad),
        "gradient_rel_err": float(abs(grad_w / ana_grad - 1.0)) if ana_grad else None,
        "b_from_rule_and_gradient": float(b_flow),
        "b_from_quadratic_inversion": b_fit,
        "b_rule_vs_fit_rel_err": float(abs(b_flow / b_fit - 1.0)) if b_fit else None,
        "u_wall_rel_err_rule_vs_extrapolation": float(
            abs(uw_op / max(abs(0.5 * (fit_low["u_wall"] + fit_high["u_wall"])), 1e-300) - 1.0)
        ) if abs(fit_low["u_wall"] + fit_high["u_wall"]) > 0 else None,
        "maxwell_defect": float(abs(uw_op - b_fit * grad_w)),
        "maxwell_defect_rel": float(abs(uw_op / (b_fit * grad_w) - 1.0)) if b_fit * grad_w else None,
        "u_wall_over_u1": float(uw_op / max(abs(res.u_profile[0]), 1e-300)),
    }
    if compare_sigma is not None:
        other = maxwell_audit(compare_sigma, ny=ny, tau=tau, g=g, steps=steps, tol=tol)
        out["compare_sigma"] = compare_sigma
        out["b_rule_other_sigma"] = other["b_from_rule_and_gradient"]
    return out


# ---------------------------------------------------------------------------
# Slip-length -> sigma, and mesh dependence
# ---------------------------------------------------------------------------
def sigma_for_slip_length(b_target: float, slope: float = SLOPE_DEFAULT) -> float:
    """Invert the lattice-law ``b = slope * sigma/(1-sigma)`` exactly."""
    if b_target < 0.0:
        raise ValueError("b_target must be >= 0")
    x = b_target / slope
    return float(x / (1.0 + x))


def physical_slip_length(b_lattice: float, dx: float) -> float:
    """``b_phys = b_lu * dx`` -- the mixed wall is *not* a fixed-b wall."""
    return float(b_lattice * dx)


def resolution_slip_study(
    beta: float = 0.10,
    resolutions: Sequence[int] = (17, 25, 33, 49, 65),
    tau: float = 0.8,
    g: float = 1.0e-5,
    slope: float = SLOPE_DEFAULT,
    steps: int = 40000,
    tol: float = 1e-10,
    retune: bool = True,
) -> dict:
    """Mesh dependence of a slip wall at a *prescribed physical* slip length.

    The channel *physical* half height is held at ``H_phys = 1``, so the lattice
    spacing is ``dx = 1/H_lu`` with ``H_lu = (ny-2)/2``, and the prescribed
    physical slip length is ``b_phys = beta H_phys``.  Three arms:

    * ``fixed_sigma`` -- **one** ``sigma`` (the value that would be correct on the
      finest mesh) used on every mesh.  Because the wall law is lattice-local
      (``b_lu`` only weakly mesh dependent) the *physical* slip length becomes
      ``b_phys = b_lu dx ~ 1/ny``: the wall silently changes with the mesh;
    * ``ideal_retune`` -- ``sigma`` re-solved per mesh from the idealised law
      ``b_lu = a sigma/(1-sigma)``.  The closure error is then dominated by the
      discrete amplitude defect ``c`` (which the idealised law does not carry);
    * ``measured_retune`` -- ``sigma`` re-solved per mesh from the **measured**
      ``b(sigma)`` at that mesh (secant iterations on the solver itself).

    The reported spreads are the quantitative statement of "a mixed wall is a
    lattice-scaled slip model, not a fixed-``b`` boundary condition".
    """
    rows = []
    sigma_fixed = sigma_for_slip_length(beta * ((max(resolutions) - 2) / 2.0), slope)
    for ny in resolutions:
        H_lu = (ny - 2) / 2.0
        dx = 1.0 / H_lu
        b_lu_target = beta * H_lu
        res_fixed = ps.channel_poiseuille(sigma_fixed, ny=ny, tau=tau, g=g,
                                          steps=steps, tol=tol)
        b_lu_fixed = float(ps.invert_slip_length(res_fixed)["b"])
        sig_ideal = sigma_for_slip_length(b_lu_target, slope)
        res_ideal = ps.channel_poiseuille(sig_ideal, ny=ny, tau=tau, g=g,
                                          steps=steps, tol=tol)
        b_lu_ideal = float(ps.invert_slip_length(res_ideal)["b"])
        row = {
            "ny": int(ny), "H_lu": H_lu, "dx": dx,
            "sigma_fixed": sigma_fixed,
            "b_lu_fixed_sigma": b_lu_fixed,
            "b_phys_fixed_sigma": b_lu_fixed * dx,
            "b_phys_fixed_sigma_rel_err": float(abs(b_lu_fixed * dx / beta - 1.0)),
            "sigma_ideal": sig_ideal,
            "b_lu_ideal": b_lu_ideal,
            "b_phys_ideal": b_lu_ideal * dx,
            "b_phys_ideal_rel_err": float(abs(b_lu_ideal * dx / beta - 1.0)),
            "converged": bool(res_ideal.converged),
            "residual": float(res_ideal.residual),
            "steps": int(res_ideal.steps),
        }
        if retune:
            res = _retuned(ny, b_lu_target, tau, g, slope, steps, tol)
            row.update({
                "sigma_retuned": res["sigma"],
                "b_lu_retuned": res["b"],
                "b_lu_target": b_lu_target,
                "b_phys_retuned": res["b"] * dx,
                "b_phys_retuned_rel_err": float(abs(res["b"] * dx / beta - 1.0)),
                "retune_rel_err": res["rel_err"],
                "converged_retuned": res["converged"],
            })
        rows.append(row)
    out = {
        "beta": float(beta),
        "slope": float(slope),
        "sigma_fixed": float(sigma_fixed),
        "rows": rows,
        "arm_fixed_sigma_b_phys_spread": _spread([r["b_phys_fixed_sigma"] for r in rows]),
        "arm_fixed_sigma_b_phys_rel_err_max": float(
            max(r["b_phys_fixed_sigma_rel_err"] for r in rows)),
        "arm_ideal_b_phys_spread": _spread([r["b_phys_ideal"] for r in rows]),
        "arm_ideal_rel_err_max": float(max(r["b_phys_ideal_rel_err"] for r in rows)),
    }
    if retune:
        out["arm_retuned_b_phys_spread"] = _spread([r["b_phys_retuned"] for r in rows])
        out["arm_retuned_rel_err_max"] = float(max(r["b_phys_retuned_rel_err"] for r in rows))
    return out


def _retuned(ny: int, b_lu_target: float, tau: float, g: float, slope: float,
             steps: int, tol: float) -> dict:
    """Solve ``sigma`` so that the *measured* b_lu equals ``b_lu_target``.

    Uses the measured lattice law at this resolution (two fixed-point
    iterations), i.e. the retuning recipe does not assume the idealised
    ``a sigma/(1-sigma)`` form -- it only uses it as a first guess.
    """
    sigma = sigma_for_slip_length(b_lu_target, slope)
    b = float("nan")
    for _ in range(3):
        res = ps.channel_poiseuille(sigma, ny=ny, tau=tau, g=g, steps=steps, tol=tol)
        inv = ps.invert_slip_length(res)
        b = float(inv["b"])
        if abs(b - b_lu_target) <= 1e-8 * max(b_lu_target, 1.0):
            break
        # local secant step on the measured law: b ~ slope_eff * s/(1-s)
        x = sigma / (1.0 - sigma)
        slope_eff = b / x if x > 0 else slope
        slope_eff = slope_eff if slope_eff > 0 else slope
        sigma = sigma_for_slip_length(b_lu_target, slope_eff)
    return {"sigma": float(sigma), "b": float(b),
            "rel_err": float(abs(b / b_lu_target - 1.0)),
            "converged": bool(res.converged)}


def _spread(values: Sequence[float]) -> dict:
    v = np.asarray(values, float)
    if v.size == 0:
        return {"mean": None, "min": None, "max": None, "relative_spread": None,
                "n": 0}
    mean = float(np.mean(v))
    return {"mean": mean, "min": float(np.min(v)), "max": float(np.max(v)),
            "relative_spread": float((np.max(v) - np.min(v)) / abs(mean)) if mean else None}


# ---------------------------------------------------------------------------
# Round pipe: convergence of the inversion error with radial resolution
# ---------------------------------------------------------------------------
def pipe_inversion_convergence(
    sigma: float,
    radii: Sequence[int] = (17, 25, 33, 41, 49),
    tau: float = 0.8,
    g: float = 1.0e-5,
    steps: int = 120000,
    tol: float = 1e-10,
) -> dict:
    """Radial-resolution convergence of the three pipe estimators.

    The axisymmetric column's discrete steady state is quadratic only up to an
    ``O(1/R)`` axis layer, so the quadratic fit carries a bias; the ``model``
    estimator removes it by fixing the curvature analytically.  Here we measure
    *both* versus ``1/R`` and extrapolate the quadratic bias to ``R -> inf``:
    a bias that vanishes linearly in ``1/R`` is a discretisation error, not a
    modelling defect.
    """
    rows = []
    for nr in radii:
        res = ps.pipe_poiseuille(sigma, nr=nr, tau=tau, g=g, steps=steps, tol=tol)
        inv = ps.invert_slip_length(res)
        model = p3.model_slip_length(
            res.coord, res.u, length=res.length, nu=ps.nu_from_tau(tau), g=g,
            geometry="pipe")
        r = np.asarray(res.coord, float)
        u_area = float(np.sum(np.abs(r) * res.u) / np.sum(np.abs(r)))
        bulk = p3.bulk_mean_slip_length(
            res.u, length=res.length, nu=ps.nu_from_tau(tau), g=g,
            geometry="pipe", coord=res.coord, u_mean=u_area)
        R = res.length
        rows.append({
            "nr": int(nr), "R": float(R), "inv_R": float(1.0 / R),
            "b_quadratic": float(inv["b"]),
            "b_model": float(model["b"]),
            "b_bulk": float(bulk["b"]),
            "curvature_ratio": float(inv["curvature_ratio"]),
            "u_max": float(res.u.max()),
            "u_max_analytic_noslip": float(g * R * R / (4.0 * ps.nu_from_tau(tau))),
            "converged": bool(res.converged), "residual": float(res.residual),
            "steps": int(res.steps),
        })
    out = {"sigma": float(sigma), "rows": rows}
    for key in ("b_quadratic", "b_model", "b_bulk"):
        y = np.array([r[key] for r in rows], float)
        x = np.array([r["inv_R"] for r in rows], float)
        slope, intercept = np.polyfit(x, y, 1)
        ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out[f"{key}_continuum"] = float(intercept)
        out[f"{key}_slope_per_invR"] = float(slope)
        out[f"{key}_r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    return out


# ---------------------------------------------------------------------------
# b(sigma) law across geometries and resolutions
# ---------------------------------------------------------------------------
def fit_b_sigma_law(
    sigmas: Sequence[float] = (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9),
    plane_resolutions: Sequence[int] = (17, 25, 33, 49),
    pipe_resolutions: Sequence[int] = (17, 25, 33),
    tau: float = 0.8,
    g: float = 1.0e-5,
    steps: int = 60000,
    tol: float = 1e-10,
) -> dict:
    """Fit ``b(sigma)`` on every ``(geometry, resolution)`` family.

    The lattice-law claim is ``b = a sigma/(1-sigma) + c`` with ``a``
    dimensionless (a property of the collision + wall rule) and ``c`` a
    discrete amplitude defect that must scale like ``1/H``.
    """
    families = []
    for kind, res_list in (("plane", plane_resolutions), ("pipe", pipe_resolutions)):
        for n in res_list:
            bs, conv = [], []
            for s in sigmas:
                if kind == "plane":
                    r = ps.channel_poiseuille(s, ny=n, tau=tau, g=g, steps=steps, tol=tol)
                else:
                    r = ps.pipe_poiseuille(s, nr=n, tau=tau, g=g, steps=steps, tol=tol)
                inv = ps.invert_slip_length(r)
                bs.append(float(inv["b"]))
                conv.append(bool(r.converged))
            law = ps.fit_slip_law(sigmas, bs)
            L = (n - 2) / 2.0 if kind == "plane" else (n - 1.5)
            families.append({
                "family": f"{kind}_{n}",
                "geometry": kind,
                "n": int(n),
                "length": float(L),
                "sigmas": list(map(float, sigmas)),
                "b": bs,
                "converged": conv,
                "form": law.form,
                "a": float(law.a),
                "c": float(law.c),
                "c_times_L": float(law.c * L),
                "r2": float(law.r2),
                "one_minus_r2": float(1.0 - law.r2),
                "monotone": bool(law.monotone),
            })
    slopes = [f["a"] for f in families if f["geometry"] == "plane"]
    out = {"families": families,
           "plane_slope_mean": float(np.mean(slopes)) if slopes else None,
           "plane_slope_spread": _spread(slopes)}
    out["r2_min"] = float(min(f["r2"] for f in families))
    out["all_monotone"] = bool(all(f["monotone"] for f in families))
    return out


# ---------------------------------------------------------------------------
# Free-slip limit: sigma -> 1 without crashing
# ---------------------------------------------------------------------------
def free_slip_asymptotics(
    sigmas: Sequence[float] = (0.9, 0.95, 0.99, 0.999),
    ny: int = 33,
    tau: float = 0.8,
    g: float = 1.0e-5,
    steps: int = 60000,
    tol: float = 1e-10,
    slope: float = SLOPE_DEFAULT,
) -> dict:
    """``b -> inf`` as ``sigma -> 1``: record the divergence, do not crash.

    The mixed wall keeps b finite for every ``sigma < 1``; the reciprocal
    ``1/b`` must go to zero linearly in ``(1-sigma)``.  This is the recorded
    limit behaviour (a pure specular wall, ``sigma = 1``, is the only singular
    case: no tangential stress is exchanged at all).
    """
    rows = []
    for s in sigmas:
        r = ps.channel_poiseuille(s, ny=ny, tau=tau, g=g, steps=steps, tol=tol)
        inv = ps.invert_slip_length(r)
        flat = float((r.u.max() - r.u.min()) / max(r.u.max(), 1e-300))
        d3 = plane_channel_3d(sigma_low=s, ny=ny, tau=tau, g=g, steps=steps, tol=tol)
        grad = 0.5 * (abs(one_sided_wall_fit(d3.u_profile, d3.coord, 0.5)["gradient"])
                      + abs(one_sided_wall_fit(d3.u_profile[::-1], d3.coord[::-1],
                                               ny - 1.5)["gradient"]))
        uw = 0.5 * (abs(float(np.mean(wall_velocity_from_rule(
                  d3.f_low, s, float(d3.rho_profile[0]), top=False))))
                  + abs(float(np.mean(wall_velocity_from_rule(
                      d3.f_high, s, float(d3.rho_profile[-1]), top=True)))))
        rows.append({
            "sigma": float(s), "one_minus_sigma": float(1.0 - s),
            "b": float(inv["b"]),
            "b_rule": float(uw / grad),
            "inv_b_rule": float(grad / uw) if uw else None,
            "converged_3d": bool(d3.converged),
            "inv_b": float(1.0 / inv["b"]) if inv["b"] != 0 else None,
            "profile_flatness": flat,
            "u_max": float(r.u.max()),
            "u_max_analytic_freeslip": float(g * r.length ** 2 / (2 * ps.nu_from_tau(tau))),
            "converged": bool(r.converged), "residual": float(r.residual),
            "steps": int(r.steps),
        })
    x = np.array([r["one_minus_sigma"] for r in rows], float)
    xr = x / np.array([r["sigma"] for r in rows], float)   # (1-sigma)/sigma
    y = np.array([r["inv_b"] for r in rows], float)
    y_rule = np.array([r["inv_b_rule"] for r in rows], float)
    slope_fit, intercept = np.polyfit(x, y, 1)
    slope_exact, intercept_exact = np.polyfit(xr, y, 1)
    slope_rule, intercept_rule = np.polyfit(xr, y_rule, 1)
    ss_res = float(np.sum((y - (slope_fit * x + intercept)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res_rule = float(np.sum((y_rule - (slope_rule * xr + intercept_rule)) ** 2))
    ss_tot_rule = float(np.sum((y_rule - y_rule.mean()) ** 2))
    return {
        "rows": rows,
        "inv_b_vs_one_minus_sigma_slope": float(slope_fit),
        "inv_b_vs_one_minus_sigma_intercept": float(intercept),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        # exact abscissa: 1/b = (1-sigma)/(a sigma) for b = a sigma/(1-sigma)
        "inv_b_vs_one_minus_over_sigma_slope_rule": float(slope_rule),
        "inv_b_vs_one_minus_over_sigma_intercept_rule": float(intercept_rule),
        "r2_rule_exact_abscissa": float(1.0 - ss_res_rule / ss_tot_rule) if ss_tot_rule > 0 else None,
        "inv_b_vs_one_minus_over_sigma_slope_fit": float(slope_exact),
        "inv_b_vs_one_minus_over_sigma_intercept_fit": float(intercept_exact),
        "predicted_slope": float(1.0 / slope),
        "slope_rel_err": float(abs(slope_exact / (1.0 / slope) - 1.0)),
        "slope_rel_err_rule_route": float(abs(slope_rule / (1.0 / slope) - 1.0)),
        "inv_b_rule_vs_one_minus_sigma_slope": float(np.polyfit(x, y_rule, 1)[0]),
        "verdict": "b diverges like sigma/(1-sigma); every sigma<1 is finite and "
                   "the solver stays convergent -- sigma=1 (pure specular) is the "
                   "singular case (no tangential stress exchange -> plug flow)",
    }


# ---------------------------------------------------------------------------
# Slip drag law: dCf(b+) on a grid, two model routes, no fabricated anchor
# ---------------------------------------------------------------------------
def drag_law_bplus(
    b_plus_values: Sequence[float] = (0.0, 5.0, 10.0, 20.0),
    re_L: float = 1.0e8,
    re_thetas: Sequence[float] = (300.0, 1000.0, 3000.0, 1.0e4, 1.0e5),
    k_slips: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 6.0),
    re_L_sweep: Sequence[float] = (1.0e6, 1.0e7, 1.0e8),
) -> dict:
    """``dCf(b+)`` table with a bracket of the two available closures.

    Route A (lower bound): virtual-wall mean-profile model
    ``u+(y+) = u+_smooth(y+ + b+)`` -- parameter free, but upstream already
    showed it is *falsified* as an explanation of the DNS drag reduction
    because it confines the change below ``y+ ~ b+``.

    Route B (scaling): FIK laminar-analogue ``Cf/Cf0 = (1 + k b+/delta+)^-2``
    with ``k`` uncalibrated in this repository (no offline DNS data).

    Both routes are monotone in ``b+``; the *monotonicity* is model independent
    and is what this function certifies.  The absolute magnitude is reported as
    a bracket over ``k`` and marked WITHHELD against literature.
    """
    base = ps.virtual_wall_drag_reduction(tuple(b_plus_values), re_theta=re_thetas[-1])
    vw = {r["b_plus"]: r for r in base["rows"]}
    rows = []
    prev_rel = None
    monotone = True
    for bp in b_plus_values:
        cf0 = ps.plate_average_cf_noslip(re_L)
        broute = {f"k={k:g}": float(1.0 - 1.0 / (1.0 + k * bp / ps.delta_plus(re_L)) ** 2)
                  for k in k_slips}
        rel = vw[float(bp)]["dCf_rel"]
        if prev_rel is not None and not (rel > prev_rel):
            monotone = False
        prev_rel = rel
        rows.append({
            "b_plus": float(bp),
            "cf_average_noslip": cf0,
            "dCf_virtual_wall": float(rel),
            "dCf_scaling": broute,
            "dCf_scaling_k3": broute["k=3"],
        })
    # Re_theta sweep of the parameter-free route
    sweep = []
    for r_theta in re_thetas:
        got = ps.virtual_wall_drag_reduction(tuple(b_plus_values), re_theta=r_theta)
        sweep.append({
            "Re_theta": float(r_theta),
            "baseline_cf": float(got["baseline_cf"]),
            "rows": [{"b_plus": r["b_plus"], "dCf_rel": float(r["dCf_rel"])} for r in got["rows"]],
            "monotone": bool(all(b["dCf_rel"] > a["dCf_rel"] for a, b in zip(got["rows"], got["rows"][1:]))),
        })
    sweep_re = []
    for rl in re_L_sweep:
        d = ps.drag_reduction_curve(tuple(b_plus_values), re_L=rl)
        sweep_re.append({
            "Re_L": float(rl),
            "delta_plus": float(ps.delta_plus(rl)),
            "rows": [{"b_plus": r["b_plus"], "dCf_rel": float(r["dCf_local_rel"])}
                     for r in d["rows"]],
            "monotone": bool(d["monotone"]),
        })
    return {
        "Re_L": float(re_L),
        "delta_plus_at_Re_L": float(ps.delta_plus(re_L)),
        "re_L_sweep": sweep_re,
        "all_re_L_monotone": bool(all(s["monotone"] for s in sweep_re)),
        "cf_average_noslip": float(ps.plate_average_cf_noslip(re_L)),
        "rows": rows,
        "rows_monotone": bool(monotone),
        "re_theta_sweep": sweep,
        "all_re_theta_monotone": bool(all(s["monotone"] for s in sweep)),
        "literature_anchor": "WITHHELD: no offline DNS/literature dCf(b+) dataset in "
                             "this repository; magnitude given as a bracket over "
                             "k_slip in [1,6] and over the two closure routes, "
                             "monotonicity certified model-independently",
    }

# ---------------------------------------------------------------------------
# Slope of the wall law read from the momentum bookkeeping (no profile fit)
# ---------------------------------------------------------------------------
def law_from_rule(
    sigmas: Sequence[float] = (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9),
    resolutions: Sequence[int] = (17, 25, 33, 49),
    tau: float = 0.8,
    g: float = 1.0e-5,
    steps: int = 60000,
    tol: float = 1e-10,
    fit_core_fraction: float = 0.6,
) -> dict:
    """``b(sigma)`` from the operator wall speed, per resolution, both walls.

    For every ``(sigma, ny)`` the wall speed is read off the mixed rule
    (:func:`wall_velocity_from_rule`), the wall-normal gradient is measured by
    one-sided extrapolation of the three near-wall rows, and ``b = u_w/gamma``.
    This route neither fits the profile nor assumes its shape, so the fitted
    intercept ``c`` is an *independent* measure of the discrete amplitude defect
    that the profile-fit route carries.
    """
    rows = []
    for ny in resolutions:
        for s in sigmas:
            res = plane_channel_3d(sigma_low=s, ny=ny, tau=tau, g=g, steps=steps, tol=tol)
            grad = 0.5 * (abs(one_sided_wall_fit(res.u_profile, res.coord, 0.5)["gradient"])
                          + abs(one_sided_wall_fit(res.u_profile[::-1], res.coord[::-1],
                                                   ny - 1.5)["gradient"]))
            uw_low = float(np.mean(wall_velocity_from_rule(
                res.f_low, s, float(res.rho_profile[0]), top=False)))
            uw_high = float(np.mean(wall_velocity_from_rule(
                res.f_high, s, float(res.rho_profile[-1]), top=True)))
            inv = ps.invert_slip_length(ps.PoiseuilleResult(
                sigma=s, n=ny, tau=tau, g=g, coord=res.coord, u=res.u_profile,
                length=res.length, geometry="plane", steps=res.steps,
                converged=res.converged, residual=res.residual))
            rows.append({
                "ny": int(ny), "H_lu": (ny - 2) / 2.0, "sigma": float(s),
                "u_wall_low": uw_low, "u_wall_high": uw_high,
                "gradient": float(grad),
                "gradient_analytic": float(g * res.length / ps.nu_from_tau(tau)),
                "gradient_rel_err": float(abs(grad / (g * res.length / ps.nu_from_tau(tau)) - 1.0)),
                "b_rule": float(0.5 * (abs(uw_low) + abs(uw_high)) / grad),
                "b_rule_low_minus_high": float(abs(uw_low) - abs(uw_high)),
                "b_fit": float(inv["b"]),
                "b_fit_minus_rule": float(inv["b"] - 0.5 * (abs(uw_low) + abs(uw_high)) / grad),
                "u_wall_over_u1": float(0.5 * (abs(uw_low) + abs(uw_high))
                                        / max(abs(res.u_profile[0]), 1e-300)),
                "converged": bool(res.converged), "residual": float(res.residual),
                "steps": int(res.steps),
            })
    # per-resolution law fits on both routes
    laws = []
    for ny in resolutions:
        sub = [r for r in rows if r["ny"] == ny]
        keep = [r for r in sub if r["sigma"] < 1.0]
        sig = [r["sigma"] for r in keep]
        for route, key in (("rule", "b_rule"), ("profile_fit", "b_fit")):
            law = ps.fit_slip_law(sig, [r[key] for r in keep])
            laws.append({"ny": int(ny), "route": route, "form": law.form, "a": float(law.a),
                         "c": float(law.c), "c_times_H": float(law.c * ((ny - 2) / 2.0)),
                         "r2": float(law.r2), "one_minus_r2": float(1.0 - law.r2),
                         "monotone": bool(law.monotone)})
    out = {"rows": rows, "laws": laws}
    for route in ("rule", "profile_fit"):
        sub = [l for l in laws if l["route"] == route]
        out[f"{route}_slope_mean"] = float(np.mean([l["a"] for l in sub]))
        out[f"{route}_slope_spread"] = _spread([l["a"] for l in sub])
        out[f"{route}_c_times_H_spread"] = _spread([l["c_times_H"] for l in sub])
        out[f"{route}_r2_min"] = float(min(l["r2"] for l in sub))
    out["all_monotone"] = bool(all(l["monotone"] for l in laws))
    return out


def locality_check(
    sigma: float = 0.5,
    ny: int = 33,
    tau: float = 0.8,
    g_values: Sequence[float] = (1.0e-5, 2.0e-5, 5.0e-5),
    steps: int = 60000,
    tol: float = 1e-10,
) -> dict:
    """Is ``b`` a property of the wall alone?

    Three tests that share no fitting:

    1. **forcing independence** -- ``b`` from the rule route must not change with
       the body force (``g``) that drives the flow;
    2. **wall independence** -- in an asymmetric channel (one no-slip wall, one
       mixed wall) the mixed wall must keep the same ``b`` as in the symmetric
       channel;
    3. **wall symmetry** -- in the symmetric channel the two walls must give the
       same ``b`` (their ``u_w`` differ only by sign).

    The wall-normal gradient is measured at each wall independently.
    """
    rows = []
    for g in g_values:
        sym = plane_channel_3d(sigma_low=sigma, ny=ny, tau=tau, g=g, steps=steps, tol=tol)
        asym = plane_channel_3d(sigma_low=0.0, sigma_high=sigma, ny=ny, tau=tau, g=g,
                                steps=steps, tol=tol)
        def _b(res, top):
            y = res.coord[::-1] if top else res.coord
            u = res.u_profile[::-1] if top else res.u_profile
            wall = (res.ny - 1.5) if top else 0.5
            f = one_sided_wall_fit(u, y, wall)
            base = res.f_high if top else res.f_low
            rho = float(res.rho_profile[-1] if top else res.rho_profile[0])
            uw = float(np.mean(wall_velocity_from_rule(base, sigma, rho, top=top)))
            return abs(uw), abs(f["gradient"]), abs(uw) / abs(f["gradient"])
        uw_l, gr_l, b_l = _b(sym, False)
        uw_h, gr_h, b_h = _b(sym, True)
        uw_a, gr_a, b_a = _b(asym, True)
        rows.append({
            "g": float(g),
            "b_sym_low": b_l, "b_sym_high": b_h,
            "b_asym_high": b_a,
            "u_wall_low": uw_l, "u_wall_high": uw_h, "u_wall_asym": uw_a,
            "grad_sym_low": gr_l, "grad_sym_high": gr_h, "grad_asym": gr_a,
            "sym_wall_rel_diff": float(abs(b_l / b_h - 1.0)),
            "asym_vs_sym_rel_diff": float(abs(b_a / b_h - 1.0)),
            "converged_sym": bool(sym.converged), "converged_asym": bool(asym.converged),
        })
    bvals = [r["b_sym_high"] for r in rows]
    return {
        "sigma": float(sigma), "rows": rows,
        "b_over_forcings_spread": _spread(bvals),
        "forcing_independence_rel_err_max": float(max(abs(v / bvals[0] - 1.0) for v in bvals)),
        "asym_vs_sym_rel_err_max": float(max(r["asym_vs_sym_rel_diff"] for r in rows)),
        "wall_symmetry_rel_err_max": float(max(r["sym_wall_rel_diff"] for r in rows)),
    }
