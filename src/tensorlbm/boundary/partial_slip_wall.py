"""Partial-slip (mixed bounce-back / specular) walls: sigma->b law, Poiseuille
inversion, turbulent drag-reduction law and net-benefit ledger.

物理背景
--------
真实壁面的切向动量适应系数介于 0（完美滑移）与 1（无滑移）之间。LBM 里最便宜的
连续可调实现是**混合反弹/镜面**（mixed bounce-back / specular，亦称
partial-slip 或 fractional BB）::

    f_q(wall) = (1 - sigma) f_{opp(q)}(wall) + sigma f_{M(q)}(wall)

``q`` 取指向流体的方向，``opp(q)`` 为反方向（反弹：完全吸收切向动量），
``M(q)`` 为关于壁法向的镜面方向（保存切向动量）。``sigma`` = 镜面分数。

这正是 Maxwell 一阶滑移条件

    u_t|_w = b * (du_t/dn)|_w

在离散层面的一个实现：每次碰撞以概率 ``sigma`` 完整保存切向动量、以概率
``1-sigma`` 完整吸收，Knudsen 层被折算进有效滑移长度 ``b``。本模块**不预设**
``b(sigma)`` 的函数形状，而是从数值实验反标定：给定 ``sigma`` 跑平面 Poiseuille，
从速度剖面反演 ``b``，再对 ``b``-``sigma`` 数据拟合标定律。

可写出的解析事实
----------------
1. **代数层**：``sigma=0`` 时算子逐位等价于半程反弹（壁面位于壁格点与相邻
   流格点之间的中面，``y = 0.5``）；``sigma=1`` 时等价于纯镜面（自由滑移）。
2. **剖面层**：定常强迫平面 Poiseuille 的离散解在体区**精确为二次式**（中心
   差分对二次式无截断误差），故二次拟合反演 ``b`` 是代数恒等而非近似。
3. **圆管层**：轴对称的 ``(1/r) du/dr`` 项用中心差分滞后源施加，定常不动点同样
   精确满足二次式（``B = -g/(4 nu)``），因此无滑移圆管可用
   ``u_max = g R^2/(4 nu)`` 做**正控**。

滑移长度反演（平面通道半高 ``H`` ／圆管半径 ``R``）
--------------------------------------------------
对剖面做最小二乘 ``u = A + B (x - x_c)^2``。解析解

    u = g/(2 nu) (H^2 - (y-y_c)^2) + g H b / nu

给出 ``b = (A + B L^2) nu / (g L)``（圆管把 ``H -> R``；``A + B L^2`` 的结构
两种几何一致，只有曲率 ``B`` 的期望值不同：平面 ``-g/(2 nu)``、圆管 ``-g/(4 nu)``）。

减阻律与净收益
--------------
湍流平板带滑移的减阻用**虚拟壁面位移（负粗糙度类比）**模型：滑移长度 ``b``
等效于把壁面沿法向**下移** ``b``，即 ``u+(y+) = u+_smooth(y+ + b+)``。该模型自动
给出近壁 ``u+ = y+ + b+``（正是 Maxwell 滑移条件）与对数区上移，是气层/超疏水
减阻文献里的标准一阶模型。本模块用 Spalding 复合律实现它，再经 von Kármán 动量
积分得到平板 ``Cf(b+, Re_x)``，并以 ``b+=0`` 复现教科书平板摩擦线作正控。

净收益账本把减阻省下的功率与维持滑移（气层注气）的代价对比，给出盈亏平衡
``b+``。注气通量的标定常数在仓内无离线文献锚点，故显式标 WITHHELD，盈亏平衡
以"关于该常数的一族曲线"给出，而不是伪造一个点值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

# ---------------------------------------------------------------------------
# D2Q9 lattice (1D-in-y reduction: the target flows are x-uniform)
# ---------------------------------------------------------------------------
C2 = (
    (0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
    (1, 1), (-1, 1), (-1, -1), (1, -1),
)
W2 = (
    4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
)
OPP2 = (0, 3, 4, 1, 2, 7, 8, 5, 6)
#: mirror about a wall whose normal is ``y`` (``c_y -> -c_y``, ``c_x`` kept)
MIRROR_Y = (0, 1, 4, 3, 2, 8, 7, 6, 5)

_CY = np.array([c[1] for c in C2])
_CX = np.array([c[0] for c in C2])
_W = np.array(W2)


def nu_from_tau(tau: float) -> float:
    """Lattice kinematic viscosity ``nu = (tau - 1/2)/3``."""
    if tau <= 0.5:
        raise ValueError(f"tau must exceed 0.5 for stability, got {tau}")
    return (tau - 0.5) / 3.0


def tau_from_nu(nu: float) -> float:
    """BGK relaxation time from lattice viscosity."""
    return 3.0 * nu + 0.5


# ---------------------------------------------------------------------------
# D2Q9 kernels on a 1D (y / r) column  (numpy: the column is O(10^2) wide, so a
# pure-numpy kernel is ~50x faster than 20 torch kernel launches per step)
# ---------------------------------------------------------------------------
_CX9 = _CX.reshape(9, 1)
_CY9 = _CY.reshape(9, 1)
_W9 = _W.reshape(9, 1)
_OPP2 = np.array(OPP2)
_MIRROR_Y = np.array(MIRROR_Y)


def macroscopic_1d(f, fx=0.0, fy=0.0):
    """Density and velocity with the Guo half-force correction (shape ``(9,N)``)."""
    rho = f.sum(0)
    ux = (_CX9 * f).sum(0)
    uy = (_CY9 * f).sum(0)
    safe = np.where(rho == 0.0, 1.0, rho)      # wall rows carry no mass
    return rho, (ux + 0.5 * fx) / safe, (uy + 0.5 * fy) / safe


def equilibrium_1d(rho, ux, uy):
    """D2Q9 equilibrium on a 1D column, shape ``(9, N)``."""
    cu = 3.0 * (_CX9 * ux + _CY9 * uy)
    usq = ux * ux + uy * uy
    return _W9 * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)


def collide_bgk_guo_1d(f, tau, fx=0.0, fy=0.0):
    """BGK collision with Guo forcing on a 1D column."""
    rho, ux, uy = macroscopic_1d(f, fx, fy)
    feq = equilibrium_1d(rho, ux, uy)
    f_post = f - (f - feq) / tau
    cu = _CX9 * ux + _CY9 * uy
    src = 3.0 * ((_CX9 - ux) * fx + (_CY9 - uy) * fy) + 9.0 * cu * (_CX9 * fx + _CY9 * fy)
    return f_post + (1.0 - 0.5 / tau) * _W9 * src


def mixed_wall_row(f_post, sigma, j_fluid, top):
    """Mixed bounce-back / specular values for one solid wall row.

    ``j_fluid`` is the adjacent *fluid* row index.  Only the directions that
    point back into the fluid carry a reflection; the other entries are inert
    padding (they never stream inward).
    """
    row = f_post[:, j_fluid]
    mix = (1.0 - sigma) * row[_OPP2] + sigma * row[_MIRROR_Y]
    # directions travelling from the wall back INTO the fluid
    inward = (_CY < 0) if top else (_CY > 0)
    return np.where(inward, mix, row)


def sigma_limits():
    """The algebraic limits of the mixed wall operator."""
    return {
        "sigma=0": "half-way bounce-back; wall plane at y = j_wall -/+ 0.5",
        "sigma=1": "pure specular; free slip (tangential momentum conserved)",
        "operator": "f_q(wall) = (1-sigma) f_opp(q)(wall) + sigma f_mirror(q)(wall)",
    }


# ---------------------------------------------------------------------------
# Plane channel with mixed walls
# ---------------------------------------------------------------------------
@dataclass
class PoiseuilleResult:
    """Steady (or terminal) 1D Poiseuille state with a mixed-slip wall."""

    sigma: float
    n: int
    tau: float
    g: float
    coord: np.ndarray          # fluid-row coordinate (y or r)
    u: np.ndarray              # streamwise velocity profile
    length: float              # half-height H (plane) or radius R (pipe)
    geometry: str
    steps: int
    converged: bool
    residual: float
    momentum_growth: float | None = None

    @property
    def u_center(self) -> float:
        return float(self.u[0] if self.geometry == "pipe" else self.u.min())


def channel_poiseuille(
    sigma: float,
    ny: int = 33,
    tau: float = 0.8,
    g: float = 1e-5,
    steps: int = 40000,
    tol: float = 1e-10,
    track_momentum: bool = False,
) -> PoiseuilleResult:
    """Steady plane Poiseuille flow between two mixed-slip walls.

    Wall rows are ``j=0`` and ``j=ny-1`` (solid); fluid rows are ``1..ny-2``.
    The bounce-back (``sigma=0``) wall plane therefore sits at ``y = 0.5`` and
    ``y = ny-1.5``; that plane is the reference surface for the inverted slip
    length.  A constant body-force density ``fx = rho*g`` drives the flow along
    x; x-uniformity makes a single column sufficient.
    """
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")
    if ny < 8:
        raise ValueError("ny must be >= 8")
    f = equilibrium_1d(np.ones(ny), np.zeros(ny), np.zeros(ny))
    u_prev = np.zeros(ny - 2)
    converged = False
    residual = float("inf")
    mom_hist = []
    step = 0
    for step in range(1, steps + 1):
        fx = g * f.sum(0)
        f_post = collide_bgk_guo_1d(f, tau, fx=fx, fy=0.0)
        full = f_post
        full[:, 0] = mixed_wall_row(f_post, sigma, 1, top=False)
        full[:, ny - 1] = mixed_wall_row(f_post, sigma, ny - 2, top=True)
        new_f = np.zeros_like(f)
        for q in range(9):
            cy = _CY[q]
            new_f[q, 1:ny - 1] = full[q, 1 - cy:ny - 1 - cy]
        f = new_f
        _rho, ux, _uy = macroscopic_1d(f, g * f.sum(0), 0.0)
        u = ux[1:ny - 1].copy()
        residual = float(np.max(np.abs(u - u_prev)) / max(np.max(np.abs(u)), 1e-300))
        u_prev = u
        if track_momentum and step % 1000 == 0:
            mom_hist.append(float(np.abs(u).sum()))
        if residual < tol and step > 200:
            converged = True
            break
    mom_growth = None
    if track_momentum and len(mom_hist) >= 2:
        mom_growth = (mom_hist[-1] - mom_hist[0]) / (len(mom_hist) - 1)
    return PoiseuilleResult(
        sigma=sigma, n=ny, tau=tau, g=g, coord=np.arange(1, ny - 1, dtype=float),
        u=u, length=(ny - 2) / 2.0, geometry="plane", steps=step,
        converged=converged, residual=residual, momentum_growth=mom_growth,
    )


# ---------------------------------------------------------------------------
# Profile inversion
# ---------------------------------------------------------------------------
def slip_length_from_quadratic_fit(
    coord: Sequence[float],
    u: Sequence[float],
    length: float,
    nu: float,
    g: float,
    center: float | None = None,
    geometry: str = "plane",
    which: str | None = None,
    core_fraction: float = 0.6,
    inner_fraction: float = 0.0,
) -> dict:
    """Invert the slip length ``b`` from a Poiseuille velocity profile.

    ``which`` is a deprecated alias of ``geometry`` kept for readability at
    call sites that pass ``length`` directly.  The pipe carries an extra
    factor 2 because ``A + B R^2 = -4 B R b`` there (its centreline velocity
    scales like ``g R^2/(4 nu)`` while the slip term scales like ``g R b/(2 nu)``).
    """
    if which is not None:
        geometry = which
    coord = np.asarray(coord, dtype=float)
    u = np.asarray(u, dtype=float)
    xc = float(np.mean(coord)) if center is None else float(center)
    d = np.abs(coord - xc)
    keep = (d <= core_fraction * length) & (d >= inner_fraction * length)
    if keep.sum() < 6:
        raise ValueError("not enough profile points for the quadratic fit")
    X = np.stack([np.ones(int(keep.sum())), (coord[keep] - xc) ** 2], axis=1)
    (A, B), *_ = np.linalg.lstsq(X, u[keep], rcond=None)
    B_plane = -g / (2.0 * nu)
    B_pipe = -g / (4.0 * nu)
    # plane: A + B L^2 = -2 B L b -> b = (A+B L^2) nu/(g L)
    # pipe : A + B L^2 = -4 B L b (because u has g/(4 nu)) -> extra factor 2
    factor = 2.0 if geometry == "pipe" else 1.0
    b = (A + B * length ** 2) * factor * nu / (g * length)
    resid = float(np.max(np.abs(X @ np.array([A, B]) - u[keep])))
    return {
        "b": float(b),
        "A_centerline": float(A),
        "B_fit": float(B),
        "B_plane_expected": float(B_plane),
        "B_pipe_expected": float(B_pipe),
        "curvature_ratio": float(B / (B_pipe if geometry == "pipe" else B_plane)),
        "fit_residual": resid,
        "points": int(keep.sum()),
        "geometry": geometry,
    }


def invert_slip_length(
    result: PoiseuilleResult, core_fraction: float = 0.6,
    inner_fraction: float | None = None,
) -> dict:
    """Slip length of a :class:`PoiseuilleResult`.

    For the pipe the axis is itself a boundary, so its O(1) discrete layer must
    be excluded from the quadratic fit (otherwise the fitted curvature carries
    a bias that the ``A + B L^2`` cancellation amplifies by ``L^2``).  The
    default annulus is ``0.2 R .. 0.6 R``.
    """
    nu = nu_from_tau(result.tau)
    center = 0.0 if result.geometry == "pipe" else float(np.mean(result.coord))
    if inner_fraction is None:
        inner_fraction = 0.2 if result.geometry == "pipe" else 0.0
    out = slip_length_from_quadratic_fit(
        result.coord, result.u, length=result.length, nu=nu, g=result.g,
        center=center, geometry=result.geometry, core_fraction=core_fraction,
        inner_fraction=inner_fraction,
    )
    out.update(sigma=result.sigma, n=result.n, geometric_length=result.length)
    return out


def analytic_plane_profile(y, half_height, nu, g, b=0.0, center=None):
    """Analytic slip Poiseuille ``u = g/(2 nu)(H^2 - s^2) + g H b/nu``."""
    y = np.asarray(y, dtype=float)
    yc = float(np.mean(y)) if center is None else float(center)
    s = y - yc
    return g / (2.0 * nu) * (half_height ** 2 - s ** 2) + g * half_height * b / nu


def analytic_pipe_profile(r, radius, nu, g, b=0.0):
    """Analytic slip Hagen-Poiseuille ``u = g/(4 nu)(R^2 - r^2) + g R b/(2 nu)``."""
    r = np.asarray(r, dtype=float)
    return g / (4.0 * nu) * (radius ** 2 - r ** 2) + g * radius * b / (2.0 * nu)


# ---------------------------------------------------------------------------
# Axisymmetric (circular pipe) Poiseuille with a mixed-slip wall
# ---------------------------------------------------------------------------
def pipe_poiseuille(
    sigma: float,
    nr: int = 49,
    tau: float = 0.8,
    g: float = 1e-5,
    steps: int = 80000,
    tol: float = 1e-10,
) -> PoiseuilleResult:
    """Slip Hagen-Poiseuille on an axisymmetric (r-only) D2Q9 column.

    Node ``j`` sits at radius ``r = j``; ``j=0`` lies **on** the axis (symmetry
    plane realised by specular reflection at the node plane), the outermost row
    ``j=nr-1`` is solid so the bounce-back wall plane is at ``R = nr - 1.5``.

    The cylindrical geometry enters through the lagged source
    ``S = nu (1/r) du/dr`` added to the body force; its discrete fixed point
    satisfies ``nu (u_rr + u_r/r) + g = 0`` exactly for a quadratic profile
    (central differences are exact on quadratics), so the no-slip limit is
    checkable against ``u_max = g R^2/(4 nu)``.
    """
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")
    if nr < 10:
        raise ValueError("nr must be >= 10")
    nu = nu_from_tau(tau)
    f = equilibrium_1d(np.ones(nr), np.zeros(nr), np.zeros(nr))
    r = np.arange(nr, dtype=float)
    u_prev = np.zeros(nr - 1)
    converged = False
    residual = float("inf")
    step = 0
    u_new = u_prev
    for step in range(1, steps + 1):
        rho = f.sum(0)
        u = (_CX9 * f).sum(0) / np.where(rho == 0.0, 1.0, rho)
        uf = u[: nr - 1]
        s = np.zeros(nr - 1)
        s[0] = 2.0 * nu * (uf[1] - uf[0])                     # L'Hopital at r=0
        s[1:nr - 2] = nu * (uf[2:] - uf[:-2]) / (2.0 * r[1:nr - 2])
        s[nr - 2] = nu * (3.0 * uf[-1] - 4.0 * uf[-2] + uf[-3]) / (2.0 * r[nr - 2])
        fx = np.zeros(nr)
        fx[: nr - 1] = g + s
        fx = fx * rho
        f_post = collide_bgk_guo_1d(f, tau, fx=fx, fy=0.0)
        full = f_post
        full[:, nr - 1] = mixed_wall_row(f_post, sigma, nr - 2, top=True)
        new_f = np.zeros_like(f)
        for q in range(9):
            cy = _CY[q]
            if cy > 0:
                # ghost node at r=-1 is the mirror of node r=+1 across r=0
                new_f[q, 0] = full[_MIRROR_Y[q], 1]
            else:
                new_f[q, 0] = full[q, 1 - cy]
        for q in range(9):
            cy = _CY[q]
            new_f[q, 1:nr - 1] = full[q, 1 - cy:nr - 1 - cy]
        f = new_f
        _rho, ux, _uy = macroscopic_1d(f, fx, 0.0)
        u_new = ux[: nr - 1].copy()
        residual = float(np.max(np.abs(u_new - u_prev)) / max(np.max(np.abs(u_new)), 1e-300))
        u_prev = u_new
        if residual < tol and step > 200:
            converged = True
            break
    return PoiseuilleResult(
        sigma=sigma, n=nr, tau=tau, g=g, coord=r[: nr - 1], u=u_new,
        length=nr - 1.5, geometry="pipe", steps=step,
        converged=converged, residual=residual,
    )


# ---------------------------------------------------------------------------
# b(sigma) calibration law
# ---------------------------------------------------------------------------
@dataclass
class SlipLaw:
    """Calibrated ``b(sigma)`` law on a finite ``sigma`` grid."""

    form: str
    a: float
    c: float
    r2: float
    monotone: bool
    sigmas: np.ndarray
    b: np.ndarray
    b_fit: np.ndarray
    extra: dict = field(default_factory=dict)

    def b_of_sigma(self, sigma):
        s = np.asarray(sigma, dtype=float) if not np.isscalar(sigma) else float(sigma)
        if self.form == "a*s/(1-s)+c":
            if np.any(np.asarray(s) >= 1.0):
                return np.inf if np.isscalar(sigma) else np.full(np.shape(s), np.inf)
            return self.a * s / (1.0 - s) + self.c
        if self.form == "a*s/(1-s)":
            if np.any(np.asarray(s) >= 1.0):
                return np.inf if np.isscalar(sigma) else np.full(np.shape(s), np.inf)
            return self.a * s / (1.0 - s)
        raise ValueError(self.form)

    def sigma_of_b(self, b_target: float) -> float:
        """Invert the law (``b = k*(1-s)`` style inversion)."""
        if self.form == "a*s/(1-s)+c":
            denom = self.a + (b_target - self.c)
            if denom <= 0:
                raise ValueError("target b below the attainable range of the law")
            s = (b_target - self.c) / denom
        elif self.form == "a*s/(1-s)":
            s = b_target / (self.a + b_target)
        else:
            raise ValueError(self.form)
        if not 0.0 <= s < 1.0:
            raise ValueError(f"inverted sigma={s} outside [0,1)")
        return float(s)


def _r2(y, yhat) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_slip_law(
    sigmas: Sequence[float], b_values: Sequence[float], forms: Sequence[str] = None
) -> SlipLaw:
    """Fit ``b(sigma)`` and return the best form by ``R^2``.

    Default candidate forms (all forced to ``b(0) = 0`` unless an offset is
    explicitly fitted):

    * ``a*s/(1-s)``      -- 1 parameter, the classical partial-slip scaling
    * ``a*s/(1-s)+c``    -- 2 parameters, absorbs a lattice offset

    Selection is by ``R^2``; the monotonicity of the *raw data* is reported
    separately so a good fit cannot hide a non-monotone dataset.
    """
    forms = tuple(forms) if forms is not None else ("a*s/(1-s)", "a*s/(1-s)+c")
    s = np.asarray(sigmas, dtype=float)
    b = np.asarray(b_values, dtype=float)
    keep = s < 1.0
    s_k, b_k = s[keep], b[keep]
    if s_k.size < 3:
        raise ValueError("need at least 3 points with sigma < 1")
    x = s_k / (1.0 - s_k)
    best: SlipLaw | None = None
    for form in forms:
        if form == "a*s/(1-s)":
            a = float(np.sum(x * b_k) / np.sum(x * x))
            c = 0.0
        elif form == "a*s/(1-s)+c":
            X = np.stack([x, np.ones_like(x)], axis=1)
            (a, c), *_ = np.linalg.lstsq(X, b_k, rcond=None)
            a, c = float(a), float(c)
        else:
            raise ValueError(form)
        yhat = a * x + c
        r2 = _r2(b_k, yhat)
        law = SlipLaw(
            form=form, a=a, c=c, r2=r2,
            monotone=bool(np.all(np.diff(b_k) > 0)),
            sigmas=s, b=b, b_fit=a * x + c,
            extra={"n_points": int(s_k.size)},
        )
        if best is None or law.r2 > best.r2:
            best = law
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Turbulent flat plate with slip: virtual-wall (negative-roughness) model
# ---------------------------------------------------------------------------
KAPPA = 0.41
B_LOG = 5.0


def spalding_yplus(u_plus: float, b_plus: float = 0.0) -> float:
    """Spalding composite law with a virtual-wall shift ``b+``.

    ``y+ + b+ = u+ + e^{-kappa B}[ e^{kappa u+} - 1 - k u+ - (k u+)^2/2 - (k u+)^3/6 ]``

    At ``u+ -> 0`` this gives ``y+ = -b+ + O(u+^2)``; the slip velocity is
    therefore the root ``y+(u+) = 0`` i.e. ``u+ ~ b+``, exactly Maxwell's
    first-order slip condition with a constant-stress sublayer.
    """
    k = KAPPA
    ku = k * u_plus
    return (
        u_plus
        + math.exp(-k * B_LOG)
        * (math.exp(ku) - 1.0 - ku - 0.5 * ku * ku - ku ** 3 / 6.0)
        - b_plus
    )


def u_plus_of_yplus(y_plus: float, b_plus: float = 0.0) -> float:
    """Invert :func:`spalding_yplus` (monotone in ``u+``)."""
    lo, hi = 0.0, 60.0
    target = y_plus + b_plus
    # bracket
    while spalding_yplus(hi, b_plus) + b_plus < target:
        hi *= 1.5
        if hi > 1e6:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if spalding_yplus(mid, b_plus) + b_plus < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def spalding_yplus_np(u_plus, b_plus: float = 0.0):
    """Vectorised :func:`spalding_yplus` (exponential capped, still monotone)."""
    k = KAPPA
    ku = np.clip(k * np.asarray(u_plus, dtype=float), -500.0, 690.0)
    return (
        np.asarray(u_plus, dtype=float)
        + math.exp(-k * B_LOG)
        * (np.exp(ku) - 1.0 - ku - 0.5 * ku * ku - ku ** 3 / 6.0)
        - b_plus
    )


def _spalding_inv(y_plus, b_plus: float = 0.0, iters: int = 70):
    """Vectorised inverse of the Spalding law: ``y+ + b+ = Phi(u+)`` -> ``u+``."""
    target = np.asarray(y_plus, dtype=float) + b_plus
    lo = np.zeros_like(target)
    hi = np.minimum(2.0 * target + 20.0, 130.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = spalding_yplus_np(mid, b_plus) + b_plus < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def re_theta_table(
    b_plus: float = 0.0, n_ue: int = 240, n_lin: int = 240, n_log: int = 420,
    ue_max: float = 60.0,
):
    """Lookup table ``Re_theta(u_e+)`` for the slip composite profile.

    The momentum thickness is integrated over ``y+`` on a two-block grid:
    a linear block ``0 .. 12`` that resolves the viscous sublayer and a
    geometric block ``12 .. delta+`` that resolves the log region.  The
    composite velocity is the Spalding law with the virtual-wall shift ``b+``,
    so the wall condition ``u+ = b+`` is built in.

    Returns ``(u_e_plus, Re_theta, cf_local)`` with ``cf_local = 2/u_e+^2``.
    """
    ue = np.geomspace(10.0, ue_max, n_ue)
    delta = np.asarray(spalding_yplus_np(ue, b_plus)) + b_plus
    lin = np.broadcast_to(np.linspace(0.0, 12.0, n_lin)[None, :], (n_ue, n_lin))
    logd = np.linspace(0.0, 1.0, n_log)[None, :]   # y+ = 12 (delta/12)^xi
    log_block = 12.0 * (delta[:, None] / 12.0) ** logd
    yp = np.concatenate([lin, log_block], axis=1)
    up = _spalding_inv(yp, b_plus)
    integrand = up * (1.0 - up / ue[:, None])
    re_theta = np.trapezoid(integrand, yp, axis=1)
    return ue, re_theta, 2.0 / ue ** 2


def cf_from_re_theta(re_theta: float, b_plus: float = 0.0) -> dict:
    """Local skin-friction coefficient at a given ``Re_theta`` and ``b+``."""
    ue, rt, _cf = re_theta_table(b_plus)
    u_e = float(np.interp(re_theta, rt, ue))
    return {"Re_theta": float(re_theta), "b_plus": float(b_plus),
            "u_e_plus": u_e, "cf": float(2.0 / u_e ** 2)}


def validate_loglaw_closure() -> dict:
    """Validate the composite (Spalding) closure against the log friction law.

    At ``b+ = 0`` the composite profile must reproduce the log friction law
    ``sqrt(2/Cf) = (1/kappa) ln(Re_theta / C)``; the power-law form
    ``Cf ~ 0.0258 Re_theta^{-1/4}`` (the 1/7-law fit used by the engineering
    correlations) is quoted as the comparison column.  The two differ by the
    wake, which the Spalding law does not carry -- that residual is the
    documented model error of the mean-profile route.
    """
    rows = []
    for r_theta in (300.0, 1000.0, 3000.0, 1.0e4, 1.0e5):
        got = cf_from_re_theta(r_theta, 0.0)
        rows.append({
            "Re_theta": r_theta,
            "cf_composite": got["cf"],
            "u_e_plus": got["u_e_plus"],
            "cf_powerlaw": 0.0258 * r_theta ** -0.25,
            "rel_err_vs_powerlaw": got["cf"] / (0.0258 * r_theta ** -0.25) - 1.0,
        })
    return {"rows": rows}


def virtual_wall_drag_reduction(b_plus_values=(10.0, 30.0), re_theta=1.0e4) -> dict:
    """DR predicted by the *mean-profile* virtual-wall model (falsified here).

    Under ``u+(y+) = u+_smooth(y+ + b+)`` the near-wall shift is confined to
    ``y+ <~ b+``, i.e. to a fraction ``b+/delta+`` of the layer, so the drag
    change at fixed ``Re_theta`` must be O(b+/delta+).  Measuring it is what
    disqualifies the model as an explanation of the DNS drag reduction.
    """
    base = cf_from_re_theta(re_theta, 0.0)["cf"]
    rows = []
    for bp in b_plus_values:
        got = cf_from_re_theta(re_theta, float(bp))
        rows.append({
            "b_plus": float(bp),
            "Re_theta": float(re_theta),
            "cf_virtual_wall": got["cf"],
            "dCf_rel": 1.0 - got["cf"] / base,
        })
    return {"Re_theta": float(re_theta), "baseline_cf": base, "rows": rows}


# ---------------------------------------------------------------------------
# Turbulent flat plate: FIK / laminar-analogue slip scaling
#
# Textbook smooth-plate anchors (fully turbulent from the leading edge):
#   local  Cf0(Re_x) = 0.0592 Re_x^{-1/5}
#   average Cf0(Re_L) = 0.074 Re_L^{-1/5}
#   TBL thickness   delta/x = 0.37 Re_x^{-1/5},  delta+ = (delta/x) Re_x sqrt(Cf0/2)
# The slip enters as the exact laminar channel/pipe analogue: for a fixed
# pressure gradient the bulk velocity gains the slip contribution u_s, hence
#   Cf/Cf0 = (1 + k b+/delta+)^{-2},  k = 3 for the laminar channel (k = 4 pipe).
# k is the only free constant; it is NOT fitted to DNS here (no offline DNS
# data in this repository -> WITHHELD) and its sensitivity is reported.
# ---------------------------------------------------------------------------
def plate_local_cf_noslip(re_x: float) -> float:
    """Smooth turbulent flat-plate local friction, ``0.0592 Re_x^{-1/5}``."""
    return 0.0592 * float(re_x) ** -0.2


def plate_average_cf_noslip(re_L: float) -> float:
    """Smooth turbulent flat-plate average friction, ``0.074 Re_L^{-1/5}``."""
    return 0.074 * float(re_L) ** -0.2


def delta_plus(re_x: float) -> float:
    """Friction Reynolds number of the TBL, ``delta+ = delta u_tau / nu``."""
    re_x = float(re_x)
    delta_over_x = 0.37 * re_x ** -0.2
    return delta_over_x * re_x * math.sqrt(plate_local_cf_noslip(re_x) / 2.0)


def slip_drag_reduction(b_plus: float, re_x: float, k_slip: float = 3.0) -> float:
    """Relative skin-friction reduction ``1 - Cf/Cf0`` from b+ at station Re_x.

    ``Cf/Cf0 = (1 + k b+/delta+)^{-2}`` -- the exact laminar slip-channel
    scaling written in wall units (``3 b/h = 3 b+/delta+``).
    """
    dp = delta_plus(re_x)
    return 1.0 - 1.0 / (1.0 + k_slip * float(b_plus) / dp) ** 2


def drag_reduction_curve(
    b_plus_values: Sequence[float], re_L: float = 1.0e8, k_slip: float = 3.0
) -> dict:
    """``Delta Cf(b+)`` (local + plate average) plus a monotonicity verdict."""
    rows = []
    prev = None
    monotone = True
    cf0_loc = plate_local_cf_noslip(re_L)
    cf0_avg = plate_average_cf_noslip(re_L)
    for bp in b_plus_values:
        dr_loc = slip_drag_reduction(bp, re_L, k_slip)
        if prev is not None and not (dr_loc > prev):
            monotone = False
        prev = dr_loc
        rows.append({
            "b_plus": float(bp),
            "delta_plus": delta_plus(re_L),
            "cf_local_noslip": cf0_loc,
            "cf_local": cf0_loc * (1.0 - dr_loc),
            "cf_average": cf0_avg * (1.0 - dr_loc),
            "dCf_local_rel": dr_loc,
            "dCf_average_rel": dr_loc,
            "u_plus_slip": float(bp),
        })
    return {
        "Re_L": float(re_L),
        "k_slip": float(k_slip),
        "delta_plus_at_Re_L": delta_plus(re_L),
        "baseline": {"cf_local": cf0_loc, "cf_average": cf0_avg},
        "rows": rows,
        "monotone": bool(monotone),
    }


# ---------------------------------------------------------------------------
# Net-benefit ledger
# ---------------------------------------------------------------------------
SHIP_CASE = {
    "L_m": 200.0,
    "B_m": 32.0,
    "T_m": 12.0,
    "C_B": 0.82,
    "U_kn": 15.0,
    "rho_w": 1025.0,
    "nu_w": 1.19e-6,
    "eta_pump": 0.60,
}


def wetted_area_m2(ship: dict | None = None) -> float:
    """Wetted area, ``S = L (1.7 T + C_B B)`` (Mumford-type approximation)."""
    s = dict(SHIP_CASE if ship is None else ship)
    return s["L_m"] * (1.7 * s["T_m"] + s["C_B"] * s["B_m"])


def net_benefit_ledger(
    b_plus_values: Sequence[float],
    cost_coeffs: Sequence[float] = (2e-5, 5e-5, 1e-4, 2e-4),
    ship: dict | None = None,
    k_slip: float = 3.0,
) -> dict:
    """Drag saving vs. slip-maintenance cost, and the break-even condition.

    Saving: ``dP = dCf * 0.5 rho U^3 S`` with ``dCf`` the relative reduction at
    ``Re_L`` (FIK / laminar-analogue law, see :func:`slip_drag_reduction`).

    Cost (model, **not** literature-anchored in this repo -> WITHHELD): the gas
    layer is sustained by an injection flux ``j = cost_coeff * b+`` [m/s] per
    unit wetted area, delivered against the hydrostatic head at depth ``T``
    with pump efficiency ``eta``: ``P_cost = rho g T j S / eta``.

    Because the cost is linear in ``b+`` while the saving saturates, the
    ledger is best expressed through the **balance coefficient**
    ``cc_balance(b+) = dP(b+) * eta / (rho g T b+ S)``: a given cost model
    ``cost_coeff`` breaks even at the ``b+`` where ``cc_balance = cost_coeff``,
    and the technology is viable at all only if
    ``cost_coeff < max cc_balance``.
    """
    s = dict(SHIP_CASE if ship is None else ship)
    U = s["U_kn"] * 0.514444
    re_l = U * s["L_m"] / s["nu_w"]
    area = wetted_area_m2(s)
    dr = drag_reduction_curve(b_plus_values, re_l, k_slip)
    base_avg = dr["baseline"]["cf_average"]
    p_friction = 0.5 * s["rho_w"] * U ** 3 * area * base_avg
    head = s["rho_w"] * 9.81 * s["T_m"]
    rows = []
    for r in dr["rows"]:
        saved = r["dCf_average_rel"] * p_friction
        cost_unit = head * area / s["eta_pump"]          # W per (m/s of flux)
        cc_balance = (saved * s["eta_pump"]) / (head * r["b_plus"] * area)             if r["b_plus"] > 0 else float("inf")
        rows.append({
            **r,
            "P_saved_W": saved,
            "P_friction_W": p_friction,
            "b_physical_m": _physical_slip_length(r["b_plus"], U, base_avg, s),
            "cost_coeff_balance": cc_balance,
            "P_cost_unit_W": cost_unit,
        })
    cc_crit = max(r["cost_coeff_balance"] for r in rows if math.isfinite(r["cost_coeff_balance"]))
    breakeven = {}
    for kq in cost_coeffs:
        curve = []
        for r in rows:
            cost = head * kq * r["b_plus"] * area / s["eta_pump"]
            curve.append({"b_plus": r["b_plus"], "P_cost_W": cost,
                          "P_net_W": r["P_saved_W"] - cost})
        be = None
        for a, b in zip(curve[:-1], curve[1:]):
            if a["b_plus"] <= 0.0:
                continue
            if a["P_net_W"] >= 0.0 > b["P_net_W"]:
                t = a["P_net_W"] / (a["P_net_W"] - b["P_net_W"])
                be = a["b_plus"] + t * (b["b_plus"] - a["b_plus"])
                break
        breakeven[f"{kq:g}"] = {
            "cost_coeff_m_s_per_bplus": kq,
            "break_even_b_plus": be,
            "viable": be is not None,
            "curve": curve,
        }
    return {
        "ship": s,
        "U_ms": U,
        "Re_L": re_l,
        "delta_plus_at_Re_L": delta_plus(re_l),
        "wetted_area_m2": area,
        "P_friction_W": p_friction,
        "P_friction_kW": p_friction / 1e3,
        "injection_head_Pa": head,
        "k_slip": float(k_slip),
        "rows": rows,
        "cc_critical_m_s_per_bplus": cc_crit,
        "breakeven": breakeven,
        "cost_model_status": "WITHHELD: injection-flux coefficient has no offline "
                            "literature anchor in this repository; the ledger is "
                            "reported as a curve over that coefficient",
    }


def _physical_slip_length(b_plus: float, U: float, cf: float, ship: dict) -> float:
    """``b = b+ nu / u_tau`` with ``u_tau = U sqrt(Cf/2)`` [m]."""
    u_tau = U * math.sqrt(max(cf, 1e-12) / 2.0)
    return b_plus * ship["nu_w"] / u_tau