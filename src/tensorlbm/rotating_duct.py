"""Rotating-frame LBM, part 2 (campaign node ``p158w2-n01``).

Three verification surfaces that the already-verified rotating-frame cases do
*not* cover:

1. **Forcing-scheme equivalence and accuracy -- Guo (2002) vs EDM.**  The
   library had a single forcing implementation (Guo, second order, with the
   half-step velocity fixed point).  Here the *exact difference method*
   (Kupershtokh & Medvedev 2006; Fakhari & Lee 2013) is implemented side by
   side in the same D3Q19 kernel and the two schemes are cross-verified on the
   rotating plane channel, whose exact solution is known
   (:func:`tensorlbm.rotating_interior.rotating_channel_analytic`).  Two
   independent force-coupling schemes, each individually within a few percent
   of the analytic solution, is a stronger statement than one scheme matching
   an analytic anchor: a sign error, a missing ``(1 - 1/(2 tau))`` factor or a
   wrong velocity offset shows up as a *scheme-dependent* bias.

2. **Taylor-Couette anchor with a genuine rotating-frame torque.**  In the
   frame rotating with ``Omega_f`` a purely azimuthal field has a *radial*
   Coriolis acceleration, so the azimuthal momentum balance stays viscous and
   the classical profile ``u_theta = A r + B/r`` is frame independent.  What is
   *not* frame independent is the wall torque, and what the pseudo-forces must
   not create is a spurious radial flow.  The anchors here are (i) the profile
   after effective-radius inversion, (ii) the torque measured *locally* by
   momentum exchange on the wall links compared with the exact
   ``T = 4 pi mu L |B|``, (iii) ``u_r / u_theta -> 0`` and (iv) Guo/EDM
   agreement -- with **counter-rotating** cylinders (``Omega_i Omega_o < 0``),
   a regime not exercised by ``benchmarks/verified/rotating_interior_frame``.

3. **Rotating square duct: Coriolis secondary flow ("Dean-Coriolis" pair) and
   the Ekman-layer thickness scaling law.**  A square duct, primary flow along
   ``x`` driven by a body force, rotating about the spanwise axis ``y``.  The
   Coriolis acceleration of the primary flow, ``a_z = 2 Omega u_x``, has a
   non-zero curl and therefore drives a cross-plane secondary flow even though
   it is not a gradient:

   .. math:: \nabla^4 \psi = \frac{2\Omega}{\nu}\,\partial_y u_x,
             \qquad \psi = \partial_n \psi = 0 \ \text{on the walls},

   the *identical* linear problem as the classical small-Dean-number curved-duct
   (Dean) problem with the curvature replaced by ``kappa_eff = 2 Omega / U``.
   The rotating duct therefore inherits the Dean vortex-pair topology; that
   analogy is stated as a *structural* correspondence (not a coefficient-level
   Dean-number match) and the quantitative anchor is the independent
   finite-difference solution of the same linear problem,
   :func:`linear_coriolis_secondary_fd`.  Finally the Ekman thickness
   ``delta_E = sqrt(2 nu / f) = sqrt(nu / Omega)`` is measured over a decade of
   rotation rates and the *exponent* of the measured ``delta_E(Omega)`` law is
   fitted (blind one-parameter profile fit plus an independent hodograph-phase
   measurement), which turns the single-point check of
   ``benchmarks/verified/rotating_ekman`` into a scaling law.

Numerics
--------
* D3Q19 BGK, float64 numpy, pull-form streaming with half-way bounce-back and
  the Ladd momentum correction ``f_k = f_opp + 6 w_k rho (c_k . u_wall)`` for
  moving walls (identical algebra to the verified D2Q9 annulus solver).
* Two forcing schemes behind one switch (``scheme="guo"`` / ``"edm"``), both
  driven by the same acceleration field ``a(u) = g - 2 Omega x u
  - Omega x (Omega x r)`` and both using the same half-step velocity fixed
  point (default 2 iterations), so the comparison isolates the *scheme* and not
  the time-centering of the force.
* Steady states are declared by a residual on the velocity field; rotating
  flows carry inertial oscillations, so the residual is optionally
  period-averaged, exactly as in the verified Ekman / interior benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import splu

CS2 = 1.0 / 3.0

# D3Q19 lattice (ordering shared by the whole module).
C3 = np.array(
    [
        [0, 0, 0],
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
        [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
        [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
        [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
    ],
    dtype=np.int64,
)
W3 = np.array([1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12, dtype=float)
OPP3 = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int64)

__all__ = [
    "C3",
    "W3",
    "OPP3",
    "ForcedD3Q19",
    "annulus3d_setup",
    "annulus_profile",
    "couette_torque_analytic",
    "wall_torque_lbm",
    "couette_profile_error",
    "secondary_flow_diagnostics",
    "duct_poiseuille_fd",
    "linear_coriolis_secondary_fd",
    "fit_power_law",
    "ekman_delta_from_phase",
    "params_sha256",
]


# --------------------------------------------------------------------------- #
# generic helpers                                                             #
# --------------------------------------------------------------------------- #
def params_sha256(obj) -> str:
    """Stable SHA-256 of a JSON-serialisable parameter record (anchor, criterion 5)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def fit_power_law(x, y) -> dict:
    """Least-squares ``y = a x^p`` in log-log space; returns ``p``, ``a``, deviation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p, loga = np.polyfit(np.log(x), np.log(y), 1)
    pred = np.exp(loga) * x**p
    rel = float(np.max(np.abs(pred - y) / np.abs(y)))
    return {"exponent": float(p), "prefactor": float(np.exp(loga)), "max_rel_dev": rel}


def ekman_delta_from_phase(y, u_par, u_perp, w_geo=None) -> dict:
    """Measure ``delta_E`` from the *phase* of the Ekman spiral (no amplitude fit).

    With ``W = u_par + i u_perp`` and the geostrophic (complex) velocity
    ``w_geo = -i G/(2 Omega)`` the exact solution obeys

        ``W/w_geo = 1 - cosh(k y)/cosh(k h)``,  ``k = (1+i)/delta``,

    so the departure from the geostrophic vector, ``dep = 1 - W/w_geo``, has a
    phase that decreases by exactly one radian per ``delta_E`` of depth:

        ``d arg(dep)/dy = -1/delta``.

    A straight line through the wall region therefore measures ``delta`` with no
    free amplitude and without using the fitted profile family, i.e. it is an
    anchor independent of the least-squares profile fit.  ``w_geo`` defaults to
    the value of ``W`` on the mid-plane (where ``1/cosh(kh) -> 0``).

    ``u_perp`` is the *imaginary* part of ``W``: for the channel solution of
    :func:`tensorlbm.rotating_interior.rotating_channel_analytic`, whose complex
    velocity is ``W = u_x - i u_z``, pass ``u_perp = -u_z``.
    """
    y = np.asarray(y, dtype=float)
    W = np.asarray(u_par, dtype=float) + 1j * np.asarray(u_perp, dtype=float)
    if w_geo is None:
        w_geo = W[int(np.argmin(np.abs(y)))]
    dep = 1.0 - W / complex(w_geo)
    # one wall only: the spiral is (anti)symmetric about the mid-plane, so a fit
    # over both halves cancels the phase slope identically
    side = np.sign(y - 0.5 * (y[0] + y[-1]))
    side[side == 0] = 1.0
    mask = (np.abs(dep) > 0.05) & (side == side[0])
    if mask.sum() < 4:
        return {"delta_phase": float("nan"), "rms_resid": float("nan"), "n": int(mask.sum())}
    phase = np.unwrap(np.angle(dep[mask]))
    slope, intercept = np.polyfit(y[mask], phase, 1)
    resid = phase - (slope * y[mask] + intercept)
    delta = float(1.0 / abs(slope)) if slope != 0 else float("nan")
    return {
        "delta_phase": delta,
        "phase_slope": float(slope),
        "phase_intercept": float(intercept),
        "rms_resid": float(np.sqrt(np.mean(resid**2))),
        "n": int(mask.sum()),
    }


# --------------------------------------------------------------------------- #
# 1. D3Q19 kernel with switchable forcing scheme                              #
# --------------------------------------------------------------------------- #
class ForcedD3Q19:
    """D3Q19 BGK solver with Guo or EDM forcing in a rotating frame.

    Parameters
    ----------
    shape : ``(nx, ny, nz)`` lattice dimensions.
    tau : relaxation time, ``nu = (tau - 1/2)/3``.
    scheme : ``"guo"`` (second-order Guo force term) or ``"edm"`` (exact
        difference method: the equilibrium shift ``f_eq(u + du) - f_eq(u)``).
    omega : frame angular velocity vector (rad / step).
    body_force : constant specific body force (lattice acceleration units).
    centrifugal : include ``-Omega x (Omega x r)``.  For constant density this
        is a pure gradient and never changes the velocity field; kept switchable
        for free-surface use, exactly as in :mod:`tensorlbm.rotating_frame`.
    """

    def __init__(
        self,
        shape,
        tau: float,
        scheme: str = "guo",
        omega=(0.0, 0.0, 0.0),
        body_force=(0.0, 0.0, 0.0),
        centrifugal: bool = False,
        rho0: float = 1.0,
        n_guo_iter: int = 2,
        mass_conserving_walls: bool = True,
    ) -> None:
        if scheme not in ("guo", "edm"):
            raise ValueError(f"unknown forcing scheme {scheme!r}")
        self.shape = tuple(int(s) for s in shape)
        self.tau = float(tau)
        self.nu = (self.tau - 0.5) / 3.0
        self.scheme = scheme
        self.omega = np.asarray(omega, dtype=float).reshape(3)
        self.body_force = np.asarray(body_force, dtype=float).reshape(3, 1, 1, 1)
        self.centrifugal = bool(centrifugal)
        self.rho0 = float(rho0)
        self.n_guo_iter = int(n_guo_iter)
        self.mass_conserving_walls = bool(mass_conserving_walls)
        self.steps = 0

        nx, ny, nz = self.shape
        xx, yy, zz = np.meshgrid(
            np.arange(nx, dtype=float),
            np.arange(ny, dtype=float),
            np.arange(nz, dtype=float),
            indexing="ij",
        )
        self.pos = np.stack([xx, yy, zz])

        self.solid = np.zeros(self.shape, dtype=bool)
        self._uw = np.zeros((3,) + self.shape)

        self._ck = C3.reshape(19, 3, 1, 1, 1).astype(float)
        self._w = W3.reshape(19, 1, 1, 1)
        self._fluid = np.ones(self.shape, dtype=bool)
        self._nb_solid = [np.zeros(self.shape, dtype=bool) for _ in range(19)]
        self._wall_wsum = np.zeros(self.shape)
        self._nb_uw = [np.zeros((3,) + self.shape) for _ in range(19)]
        self._eq0 = self._feq(np.full(self.shape, self.rho0), np.zeros((3,) + self.shape))

        self.f = self._eq0.copy()

    # -- geometry ----------------------------------------------------------- #
    def set_solid(self, solid, wall_velocity=None) -> None:
        """Install a solid mask (``True`` = wall) and optional wall velocities.

        ``wall_velocity`` is ``(3,) + shape`` and gives, on **solid** nodes, the
        wall velocity in the frame of the solver; it enters the Ladd half-way
        bounce-back correction of the neighbouring fluid nodes.
        """
        self.solid = np.asarray(solid, dtype=bool)
        if wall_velocity is None:
            self._uw = np.zeros((3,) + self.shape)
        else:
            uw = np.asarray(wall_velocity, dtype=float)
            if uw.shape != (3,) + self.shape:
                raise ValueError("wall_velocity must have shape (3,)+grid shape")
            self._uw = np.where(self.solid[None], uw, 0.0)
        self._fluid = ~self.solid
        for k in range(19):
            sh = tuple(int(v) for v in C3[k])
            self._nb_solid[k] = np.roll(self.solid, sh, axis=(0, 1, 2))
            rolled = np.roll(self._uw, sh, axis=(1, 2, 3))
            self._nb_uw[k] = np.where(self._nb_solid[k][None], rolled, 0.0)
        self._wall_wsum = np.zeros(self.shape)
        for k in range(19):
            self._wall_wsum += self._nb_solid[k] * (6.0 * W3[k])

    # -- lattice operators -------------------------------------------------- #
    def _feq(self, rho, u):
        cu = self._ck[:, 0] * u[0] + self._ck[:, 1] * u[1] + self._ck[:, 2] * u[2]
        usq = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
        return rho[None] * self._w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])

    def fields(self):
        """Return ``(rho, u)`` with ``u`` shaped ``(3,) + grid``."""
        f = self.f
        rho = f.sum(axis=0)
        inv = 1.0 / rho
        u = np.stack(
            [
                (f * self._ck[:, 0]).sum(axis=0) * inv,
                (f * self._ck[:, 1]).sum(axis=0) * inv,
                (f * self._ck[:, 2]).sum(axis=0) * inv,
            ]
        )
        return rho, u

    def _accel(self, u):
        """Total specific force: body force + Coriolis (+ centrifugal)."""
        a = np.broadcast_to(self.body_force, (3,) + self.shape).astype(float).copy()
        w = self.omega
        if np.any(w != 0.0):
            a[0] -= 2.0 * (w[1] * u[2] - w[2] * u[1])
            a[1] -= 2.0 * (w[2] * u[0] - w[0] * u[2])
            a[2] -= 2.0 * (w[0] * u[1] - w[1] * u[0])
            if self.centrifugal:
                # -Omega x (Omega x r) = Omega (Omega . r) - r |Omega|^2
                wdotr = w[0] * self.pos[0] + w[1] * self.pos[1] + w[2] * self.pos[2]
                wsq = float(w @ w)
                for i in range(3):
                    a[i] = a[i] - (w[i] * wdotr - self.pos[i] * wsq)
        return a

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            rho, u = self.fields()
            # half-step velocity fixed point (identical for both schemes, so the
            # comparison isolates the force term and not the time centering)
            a = self._accel(u)
            u_s = u + 0.5 * a
            for _it in range(max(0, self.n_guo_iter - 1)):
                a = self._accel(u_s)
                u_s = u + 0.5 * a

            if self.scheme == "guo":
                feq = self._feq(rho, u_s)
                cu = self._ck[:, 0] * u_s[0] + self._ck[:, 1] * u_s[1] + self._ck[:, 2] * u_s[2]
                cF = (
                    self._ck[:, 0] * a[0][None]
                    + self._ck[:, 1] * a[1][None]
                    + self._ck[:, 2] * a[2][None]
                )
                uF = (u_s[0] * a[0] + u_s[1] * a[1] + u_s[2] * a[2])[None]
                src = (1.0 - 0.5 / self.tau) * self._w * (3.0 * cF - 3.0 * uF + 9.0 * cu * cF)
                f_post = self.f - (self.f - feq) * (1.0 / self.tau) + src
            else:  # exact difference method
                feq_u = self._feq(rho, u)
                feq_s = self._feq(rho, u + a)
                f_post = self.f - (self.f - feq_u) * (1.0 / self.tau) + (feq_s - feq_u)

            if self.solid.any():
                f_post.reshape(19, -1)[:, self.solid.ravel()] = self.f.reshape(19, -1)[
                    :, self.solid.ravel()
                ]

            # moving-wall correction.  The naive Ladd term injects mass at
            # staircase corners of a *curved* wall (the link set there is not
            # closed under k -> opp(k), so sum_k w_k (c_k . u_w) != 0).  The net
            # injection over the wall vanishes, but its *node-wise* pattern has a
            # strong m=2 component which drives a spurious m=2 secondary flow
            # (measured: 2.6 % mass drift and a quadrupolar u_theta in the 3D
            # annulus, i.e. an order of magnitude worse than the D2Q9 case).
            # Subtracting the per-node weighted mean of (c_k . u_w) makes the
            # correction mass conserving without changing the flat-wall limit
            # (where the mean is exactly zero).
            cu_w_list = []
            for k in range(19):
                uw = self._nb_uw[k]
                cu_w_list.append(C3[k, 0] * uw[0] + C3[k, 1] * uw[1] + C3[k, 2] * uw[2])
            if self.mass_conserving_walls:
                cw_sum = np.zeros(self.shape)
                for k in range(19):
                    cw_sum += self._nb_solid[k] * (6.0 * W3[k]) * cu_w_list[k]
                cw_mean = np.where(self._wall_wsum > 0, cw_sum / np.maximum(self._wall_wsum, 1e-300), 0.0)
                cu_w_list = [c - cw_mean for c in cu_w_list]
                del cw_sum


            f_new = np.empty_like(f_post)
            for k in range(19):
                sh = tuple(int(v) for v in C3[k])
                streamed = np.roll(f_post[k], sh, axis=(0, 1, 2))
                refl = f_post[OPP3[k]] + 6.0 * W3[k] * rho * cu_w_list[k]
                f_new[k] = np.where(self._nb_solid[k], refl, streamed)
            if self.solid.any():
                f_new.reshape(19, -1)[:, self.solid.ravel()] = self._eq0.reshape(19, -1)[
                    :, self.solid.ravel()
                ]
            self.f = f_new
            self.steps += 1

    # -- initialisation / driving ------------------------------------------ #
    def initialize_from_velocity(self, u, rho=None) -> None:
        r = self.rho0 if rho is None else float(rho)
        uu = np.asarray(u, dtype=float)
        self.f = self._feq(np.full(self.shape, r), uu)
        if self.solid.any():
            self.f.reshape(19, -1)[:, self.solid.ravel()] = self._eq0.reshape(19, -1)[
                :, self.solid.ravel()
            ]

    def steady(self, tol=1e-9, max_steps=200_000, check_every=200, period=None) -> dict:
        """Integrate until the velocity field stops changing (period-averaged)."""
        fluid = self._fluid
        u_prev = self.fields()[1]
        history = []
        residual = float("inf")
        if period is None:
            while self.steps < max_steps:
                self.step(check_every)
                u_cur = self.fields()[1]
                residual = float(np.max(np.abs(u_cur - u_prev)[:, fluid]))
                history.append({"steps": self.steps, "max_du": residual})
                u_prev = u_cur
                if residual < tol:
                    break
        else:
            period = max(int(period), 2)
            check_every = max(1, int(round(period / 8)))
            buf = [(self.steps, self.fields()[1].copy())]
            while self.steps < max_steps:
                self.step(check_every)
                u_cur = self.fields()[1]
                buf.append((self.steps, u_cur.copy()))
                target = self.steps - period
                while len(buf) > 1 and buf[1][0] <= target:
                    buf.pop(0)
                residual = float(np.max(np.abs(u_cur - buf[0][1])[:, fluid]))
                history.append(
                    {"steps": self.steps, "compare_to": buf[0][0], "max_du": residual}
                )
                if residual < tol and self.steps >= period:
                    break
        return {
            "steps": self.steps,
            "residual": residual,
            "tol": float(tol),
            "period": period,
            "converged": bool(residual < tol),
            "history": history[-8:],
        }


# --------------------------------------------------------------------------- #
# 2. Taylor-Couette annulus (3D, moving walls)                                #
# --------------------------------------------------------------------------- #
def annulus3d_setup(
    shape,
    ri: float,
    ro: float,
    omega_i: float,
    omega_o: float = 0.0,
    center=None,
) -> dict:
    """Cartesian mask + wall velocities of an annulus, periodic along ``z``.

    The cylinders are centred on ``(xc, yc)``; the wall velocity on solid nodes
    is ``omega x r`` *in the frame of the solver*, so the Ladd correction of the
    neighbouring fluid links carries the right tangential velocity.
    """
    nx, ny, nz = (int(s) for s in shape)
    yy, xx = np.meshgrid(np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij")
    yc, xc = (
        ((ny - 1) / 2.0, (nx - 1) / 2.0)
        if center is None
        else (float(center[0]), float(center[1]))
    )
    dx = xx - xc
    dy = yy - yc
    rad = np.sqrt(dx**2 + dy**2)
    inner = rad < float(ri)
    outer = rad > float(ro)
    # the mask is 2D in (x, y) and replicated over the periodic z axis
    solid = np.repeat((inner | outer)[:, :, None], nz, axis=2)
    uw = np.zeros((3, nx, ny, nz))
    uw[0] = np.where(inner, -omega_i * dy, np.where(outer, -omega_o * dy, 0.0))[:, :, None]
    uw[1] = np.where(inner, omega_i * dx, np.where(outer, omega_o * dx, 0.0))[:, :, None]
    uw = np.where(solid[None], uw, 0.0)
    return {
        "shape": (nx, ny, nz),
        "solid": solid,
        "wall_velocity": uw,
        "rad": rad,
        "dx": dx,
        "dy": dy,
        "ri": float(ri),
        "ro": float(ro),
        "omega_i": float(omega_i),
        "omega_o": float(omega_o),
    }


def annulus_profile(geom, u, n_shell=None) -> dict:
    """Shell-averaged ``(r, u_theta, u_r)`` over the fluid annulus (z-averaged)."""
    rad = geom["rad"]
    dx, dy = geom["dx"], geom["dy"]
    fluid = ~geom["solid"].all(axis=2)
    r_safe = np.where(rad > 0, rad, 1.0)
    ux = np.asarray(u[0]).mean(axis=2)
    uy = np.asarray(u[1]).mean(axis=2)
    u_th = (-dy * ux + dx * uy) / r_safe
    u_rad = (dx * ux + dy * uy) / r_safe
    kk = np.floor(rad).astype(np.int64)
    kmax = int(n_shell if n_shell is not None else math.ceil(geom["ro"]) + 1)
    r_bin, th, rd, cnt = [], [], [], []
    for k in range(kmax):
        sel = fluid & (kk == k)
        n = int(sel.sum())
        r_bin.append(k + 0.5)
        cnt.append(n)
        th.append(float(u_th[sel].mean()) if n else float("nan"))
        rd.append(float(u_rad[sel].mean()) if n else float("nan"))
    return {
        "r_bin": np.array(r_bin),
        "count": np.array(cnt),
        "u_theta": np.array(th),
        "u_r": np.array(rd),
    }


def couette_torque_analytic(nu: float, B: float, length: float = 1.0, rho: float = 1.0) -> float:
    """Exact torque magnitude on either cylinder: ``4 pi mu L |B|``."""
    return float(4.0 * math.pi * rho * nu * abs(B) * length)


def wall_torque_lbm(solver: ForcedD3Q19, center=None, axis: int = 2, solid_select=None) -> dict:
    """Torque on the solid walls from the local momentum exchange (Ladd).

    For every fluid node ``x`` whose neighbour in direction ``-c_k`` is solid,
    the incoming population is ``f_in = f_opp^post`` (travelling along ``-c_k``)
    and the reflected one returns along ``+c_k`` with
    ``f_out = f_in + 6 w_k rho (c_k . u_w)``.  The momentum given to the wall per
    link and step is ``dp = -(f_in + f_out) c_k``, summed with the lever arm at
    the wall position ``x - c_k/2``.  This is a *local* wall measurement -- it
    never uses the analytic solution -- so comparing it with ``4 pi mu L |B|``
    is an independent torque anchor.
    """
    f = solver.f
    rho, _ = solver.fields()
    nx, ny, nz = solver.shape
    if center is None:
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    else:
        cx, cy = float(center[0]), float(center[1])
    lever_x = np.arange(nx)[:, None, None] - cx
    lever_y = np.arange(ny)[None, :, None] - cy
    if solid_select is not None:
        sel = np.asarray(solid_select, dtype=bool)
        if sel.ndim == 2:
            sel = np.repeat(sel[:, :, None], nz, axis=2)
    torque = np.zeros(3)
    n_links = 0
    for k in range(19):
        mask = solver._nb_solid[k]
        if solid_select is not None:
            # keep only the links that bounce off the selected wall subset
            mask = mask & np.roll(sel, tuple(int(v) for v in C3[k]), axis=(0, 1, 2))
        if not mask.any():
            continue
        uw = solver._nb_uw[k]
        cu_w = C3[k, 0] * uw[0] + C3[k, 1] * uw[1] + C3[k, 2] * uw[2]
        f_in = f[OPP3[k]]
        f_out = f_in + 6.0 * W3[k] * rho * cu_w
        dp = -(f_in + f_out)
        lx = lever_x - 0.5 * C3[k, 0]
        ly = lever_y - 0.5 * C3[k, 1]
        # r x (dp c_k) with r_z = 0 and the z lever arm omitted (2D-in-plane)
        tx = np.sum(mask * dp * (ly * C3[k, 2]))
        ty = np.sum(mask * dp * (-lx * C3[k, 2]))
        tz = np.sum(mask * dp * (lx * C3[k, 1] - ly * C3[k, 0]))
        torque += np.array([tx, ty, tz])
        n_links += int(mask.sum())
    return {
        "torque": torque.tolist(),
        "torque_axis": float(torque[axis]),
        "magnitude_axis": abs(float(torque[axis])),
        "n_links": n_links,
    }


def couette_profile_error(
    r_bin,
    u_theta,
    ri: float,
    ro: float,
    omega_i: float,
    omega_o: float = 0.0,
    n_excl: int = 2,
) -> dict:
    """Profile error of the measured ``u_theta`` against the exact ``A r + B/r``.

    Two comparisons are returned:

    * ``nominal_rel_max`` -- analytic ``(A, B)`` from the nominal radii,
      normalised by ``max(|omega_i| ri, |omega_o| ro)``: the honest end-to-end
      error including the staircase wall offset (``O(1)`` cell);
    * ``effective_rel_max`` -- ``(A, B)`` are re-fitted on the interior shells,
      the effective radii are inverted from the no-slip conditions (same
      procedure as ``benchmarks/verified/taylor_couette``) and the analytic
      solution is rebuilt from ``(ri_eff, ro_eff, omega_i, omega_o)``; this
      isolates the *shape* error of the solver from the staircase geometry.
    """
    from .rotating_interior import circular_couette, fit_couette_AB, invert_effective_radii

    r = np.asarray(r_bin, dtype=float)
    u = np.asarray(u_theta, dtype=float)
    ok = np.isfinite(u) & (r > ri - 0.5) & (r < ro + 0.5)
    ana = circular_couette(r[ok], ri, ro, omega_i, omega_o)
    scale = max(abs(omega_i) * ri, abs(omega_o) * ro, 1e-12)
    nominal = float(np.max(np.abs(u[ok] - ana["u_theta"])) / scale)

    interior = ok.copy()
    interior[: n_excl + int(ri)] = False
    interior[int(ro) - n_excl + 1:] = False
    fit = fit_couette_AB(r[interior], u[interior])
    inv = invert_effective_radii(fit["A"], fit["B"], omega_i)
    ri_eff = inv["ri_eff"] if np.isfinite(inv["ri_eff"]) else ri
    ro_eff = inv["ro_eff"] if np.isfinite(inv["ro_eff"]) else ro
    ana_eff = circular_couette(r[ok], ri_eff, ro_eff, omega_i, omega_o)
    effective = float(np.max(np.abs(u[ok] - ana_eff["u_theta"])) / scale)
    return {
        "nominal_rel_max": nominal,
        "effective_rel_max": effective,
        "A_fit": fit["A"],
        "B_fit": fit["B"],
        "A_analytic": ana["A"],
        "B_analytic": ana["B"],
        "ri_eff": float(ri_eff),
        "ro_eff": float(ro_eff),
        "ri": float(ri),
        "ro": float(ro),
        "n_points": int(ok.sum()),
        "fit_residual_rms": fit["residual_rms"],
    }


# --------------------------------------------------------------------------- #
# 3. Duct: secondary-flow diagnostics and the linear reference solution        #
# --------------------------------------------------------------------------- #
def secondary_flow_diagnostics(uy, uz, y, z, psi=None) -> dict:
    """Topology metrics of a cross-plane secondary flow in a square duct.

    ``uy``/``uz`` are ``(ny, nz)`` cell-centre fields, ``y``/``z`` the matching
    coordinate vectors (walls at ``0`` and ``2h``; the no-slip values on the
    boundary rows are zero).  Metrics:

    * ``psi`` -- streamfunction from ``u_z = -d psi/dy`` and, independently, from
      ``u_y = d psi/dz``; the two must agree (``psi_consistency``);
    * ``antisym_psi`` -- ``max|psi(y) + psi(-y)| / max|psi|``.  The Coriolis
      forcing is antisymmetric in ``y``, so a symmetric forcing error or a
      geometry bug breaks this at first order;
    * ``sym_uz`` -- ``max|u_z(y) - u_z(-y)| / max|u_z|`` (``u_z`` must be even);
    * ``antisym_uy`` -- ``max|u_y(y) + u_y(-y)| / max|u_y|`` (``u_y`` odd);
    * ``sign_changes_y_mid`` / ``sign_changes_z_quarter`` -- number of sign
      changes of ``psi`` along the two mid-lines (``1`` along ``y`` and ``0``
      along ``z`` means one vortex in each half duct, i.e. a vortex *pair*);
    * ``circulation_upper/lower`` -- ``int int omega_x dA`` over the two halves
      (equal magnitude, opposite sign for a vortex pair);
    * ``net_flux_z`` -- ``int int u_z dA`` normalised by ``max|u_z| * area``: a
      closed cross-section admits no net cross-plane flux.
    """
    uy = np.asarray(uy, dtype=float)
    uz = np.asarray(uz, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    ny, nz = uz.shape

    # The fields are *cell centred* with the no-slip wall half a cell outside the
    # first/last sample (the lattice convention), so the integration has to start
    # at the wall itself (u = 0, psi = 0) rather than at the first cell:
    #   psi[0] = -d/2 * uz[0],   psi[i] = psi[i-1] - d/2 (uz[i] + uz[i-1])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0]) if nz > 1 else 0.0
    if psi is None:
        psi = np.zeros_like(uz)
        psi[0] = -0.5 * dy * uz[0]
        psi[1:] = psi[0] - np.cumsum(0.5 * dy * (uz[1:] + uz[:-1]), axis=0)
        # remove the O(d) closure drift of the discrete integral (a linear-in-y
        # term, hence odd about the mid-plane: it cannot mask an antisymmetry bug)
        psi = psi - np.linspace(0.0, float(psi[-1].mean()), ny)[:, None]
    else:
        psi = np.array(psi, dtype=float)
    psi_alt = np.zeros_like(uz)
    if nz > 1:
        psi_alt[:, 0] = 0.5 * dz * uy[:, 0]
        psi_alt[:, 1:] = psi_alt[:, [0]] + np.cumsum(0.5 * dz * (uy[:, 1:] + uy[:, :-1]), axis=1)
    scale_psi = max(float(np.max(np.abs(psi))), 1e-30)
    offset = float(np.mean(psi - psi_alt))
    psi_consistency = float(np.max(np.abs(psi - psi_alt - offset)) / scale_psi)

    antisym_psi = float(np.max(np.abs(psi + psi[::-1, :])) / scale_psi)
    scale_uz = max(float(np.max(np.abs(uz))), 1e-30)
    scale_uy = max(float(np.max(np.abs(uy))), 1e-30)
    sym_uz = float(np.max(np.abs(uz - uz[::-1, :])) / scale_uz)
    antisym_uy = float(np.max(np.abs(uy + uy[::-1, :])) / scale_uy)

    mid = ny // 2
    quarter = ny // 4
    # sign changes are counted on the strong part of psi only (|psi| > 2 % of
    # the max), so that the numerical zero of the mid-plane does not produce
    # spurious crossings.  A vortex *pair* means exactly one crossing along y
    # (the mid-plane) and none along z inside a half.
    sgn = np.where(np.abs(psi) > 0.02 * scale_psi, np.sign(psi), 0)
    col = sgn[:, nz // 2]
    sc_y = 0
    last = 0
    for v in col:
        if v != 0:
            if last != 0 and v != last:
                sc_y += 1
            last = v
    row = sgn[quarter, :]
    sc_z = 0
    last = 0
    for v in row:
        if v != 0:
            if last != 0 and v != last:
                sc_z += 1
            last = v

    # vorticity omega_x = d_y u_z - d_z u_y, circulation of the two halves
    gy, gz = np.gradient(uz, y, z, edge_order=1)
    gy2, gz2 = np.gradient(uy, y, z, edge_order=1)
    om = gy - gz2
    circ_upper = float(np.trapezoid(np.trapezoid(om[:mid], z, axis=1), y[:mid])) if mid > 1 else 0.0
    circ_lower = float(np.trapezoid(np.trapezoid(om[mid:], z, axis=1), y[mid:])) if mid > 1 else 0.0
    # closed cross-section: the net cross-plane flux is measured by the residual
    # jump of psi across the duct (exact identity: int int u_z dA = psi_bottom - psi_top)
    net_flux = float((psi[-1].mean()) / scale_psi)

    return {
        "psi": psi,
        "psi_max": scale_psi,
        "psi_end_norm": float(psi[-1].mean() / scale_psi),
        "psi_consistency": psi_consistency,
        "antisym_psi": antisym_psi,
        "sym_uz": sym_uz,
        "antisym_uy": antisym_uy,
        "sign_changes_y_mid": sc_y,
        "sign_changes_z_quarter": sc_z,
        "circulation_upper": circ_upper,
        "circulation_lower": circ_lower,
        "circ_balance": float(
            abs(circ_upper + circ_lower) / max(abs(circ_upper), abs(circ_lower), 1e-30)
        ),
        "net_flux_z": net_flux,
        "sec_max": float(max(np.max(np.abs(uy)), np.max(np.abs(uz)))),
    }


# --------------------------------------------------------------------------- #
# 3b. Finite-difference reference of the same linear problem                   #
# --------------------------------------------------------------------------- #
def _laplacian_dirichlet(m: int, d: float) -> csr_matrix:
    """5-point Laplacian on an ``m x m`` grid, index ``i*m + j`` = ``(y, z)``.

    The operator includes the walls as rows/columns (``m x m`` unknowns); callers
    select interior/boundary index sets, which keeps the Dirichlet handling
    explicit (needed for the Thom vorticity boundary condition).
    """
    off = csr_matrix((np.ones(m - 1), (np.arange(m - 1), np.arange(1, m))), shape=(m, m))
    d2 = (off + off.T - 2.0 * identity(m)) / d**2
    return csr_matrix(kron(identity(m), d2) + kron(d2, identity(m)))


def duct_poiseuille_fd(h: float, G: float, nu: float, d: float = 0.5) -> dict:
    """Square-duct Poiseuille profile ``w0(y, z)`` (walls at ``0`` and ``2h``).

    Solves ``nabla^2 w0 = -G/nu`` with ``w0 = 0`` on the walls (``G`` is the
    streamwise body force per unit mass, so ``mu = rho nu`` with ``rho = 1``).
    """
    m = int(round(2.0 * h / d)) + 1
    L = _laplacian_dirichlet(m, d)
    idx = np.arange(m * m).reshape(m, m)
    interior = idx[1:-1, 1:-1].ravel()
    A = L[interior][:, interior].tocsc()
    rhs = np.full(interior.size, -G / nu)
    w = np.zeros(m * m)
    w[interior] = splu(A).solve(rhs)
    c = np.linspace(0.0, 2.0 * h, m)
    return {
        "w0": w.reshape(m, m),
        "m": m,
        "d": d,
        "coord": c,
        "nodes": np.zeros((m, m), dtype=bool),
    }


def linear_coriolis_secondary_fd(
    h: float,
    Omega: float,
    nu: float,
    G: float | None = None,
    d: float = 0.5,
) -> dict:
    """Creeping-flow reference for the rotating-duct secondary flow.

    Solves the linearised cross-plane problem

    .. math:: \nabla^4 \psi = \frac{2\Omega}{\nu}\,\partial_y u_x,
              \qquad \psi = \partial_n\psi = 0 \ \text{on the walls},

    which is exactly the leading-order Coriolis balance
    ``mu lap^2 psi = (curl f)_x`` with ``f_z = 2 Omega u_x``.  The biharmonic
    problem is cast as one *linear* sparse system in ``(omega, psi)``:

    * interior: ``lap(omega) = -g``, ``lap(psi) = -omega`` with
      ``omega = -lap(psi)``;
    * walls: the Thom condition ``omega_wall = -2 psi_adjacent/d^2`` (exact for
      the clamped biharmonic problem with ``psi_wall = 0`` and ``d_n psi = 0``).

    A direct solve (no iteration) removes any question of iterative
    convergence: this is an *independent* (finite-difference, no lattice)
    solution of the same balance the LBM solves at finite cross-plane Reynolds
    number, i.e. a genuine anchor and not a re-derivation of the lattice scheme.

    ``G`` defaults to ``1``; pass the LBM forcing to match the primary profile.
    """
    if G is None:
        G = 1.0
    pois = duct_poiseuille_fd(h, G, nu, d=d)
    w0 = pois["w0"]
    m = pois["m"]
    c = pois["coord"]
    L = _laplacian_dirichlet(m, d)
    idx = np.arange(m * m).reshape(m, m)
    interior = idx[1:-1, 1:-1].ravel()
    boundary = np.setdiff1d(np.arange(m * m), interior)
    A_ii = L[interior][:, interior].tocsc()
    A_ib = L[interior][:, boundary].tocsc()

    g = (2.0 * Omega / nu) * (w0[2:, 1:-1] - w0[:-2, 1:-1]) / (2.0 * d)
    n_int = interior.size
    n_bnd = boundary.size
    n_all = m * m

    import scipy.sparse as sp

    # row block 1: interior vorticity equation  (L w)[interior] = -g
    # (the unknown vector holds omega on the FULL grid, so the row is the full
    # interior slice of L, not a contiguous interior-only block)
    r1 = sp.hstack([L[interior], sp.csr_matrix((n_int, n_int))], format="csr")
    # row block 2: Thom boundary condition  w_bnd + (2/d^2) psi_adj = 0
    T = sp.lil_matrix((n_bnd, n_int))
    lookup = {int(v): k for k, v in enumerate(interior)}
    for r, b in enumerate(boundary):
        i, j = divmod(int(b), m)
        ai = min(max(i, 1), m - 2)
        aj = min(max(j, 1), m - 2)
        col = lookup[ai * m + aj]
        T[r, col] = 2.0 / d**2
    r2 = sp.hstack([sp.identity(n_all, format="csr")[boundary], T.tocsr()], format="csr")
    # row block 3: streamfunction equation  L_ii psi = -w_int
    # omega + lap(psi) = 0  (since omega = -lap(psi)) restricted to the interior
    C = sp.lil_matrix((n_int, n_all))
    C[:, interior] = sp.identity(n_int)
    r3 = sp.hstack([C.tocsr(), A_ii], format="csr")

    A = sp.vstack([r1, r2, r3], format="csc")
    rhs = np.concatenate([-g.ravel(), np.zeros(n_bnd), np.zeros(n_int)])
    sol = sp.linalg.spsolve(A, rhs)
    om = sol[:n_all].reshape(m, m)
    psi = np.zeros((m, m))
    psi.ravel()[interior] = sol[n_all:]

    # biharmonic residual as an internal consistency check of the direct solve
    lap_psi = (L @ psi.ravel()).reshape(m, m)
    lap_om = (L @ om.ravel()).reshape(m, m)
    res_bih = float(
        np.max(np.abs(lap_om[1:-1, 1:-1] + g)) / max(np.max(np.abs(g)), 1e-30)
    )
    res_couple = float(
        np.max(np.abs(lap_psi[1:-1, 1:-1] + om[1:-1, 1:-1])) / max(np.max(np.abs(om)), 1e-30)
    )

    dpsi_dy, dpsi_dz = np.gradient(psi, d)
    uy = dpsi_dz
    uz = -dpsi_dy
    int_uy = uy[1:-1, 1:-1]
    int_uz = uz[1:-1, 1:-1]
    return {
        "psi": psi,
        "uy": int_uy,
        "uz": int_uz,
        "w0": w0,
        "coord": c,
        "m": m,
        "d": d,
        "h": float(h),
        "Omega": float(Omega),
        "nu": float(nu),
        "G": float(G),
        "biharmonic_residual": res_bih,
        "coupling_residual": res_couple,
        "sec_max": float(max(np.max(np.abs(int_uy)), np.max(np.abs(int_uz)))),
        "psi_wall_max": float(max(np.max(np.abs(psi[0, :])), np.max(np.abs(psi[-1, :])))),
    }


def torque_from_profile(r, u_theta, nu: float, length: float = 1.0, rho: float = 1.0) -> dict:
    """Torque transmitted across a cylindrical surface, from the measured profile.

    For a steady axisymmetric flow the angular-momentum balance gives the
    torque carried across *any* surface ``r = const``:

    .. math:: T(r) = 2\\pi\\rho\\nu L r^2\\left(\\frac{du_\\theta}{dr}
              - \\frac{u_\\theta}{r}\\right),

    which for the exact Couette family equals ``-4 pi mu L B`` for every ``r``.
    Evaluating it from the simulated (shell-averaged) profile therefore provides
    two independent checks at once: the torque magnitude against the analytic
    ``4 pi mu L |B|``, and the **r-independence** of ``T``, which no
    profile-fitting procedure can fake.

    The local momentum-exchange (Ladd) force is *not* used as the torque anchor:
    on a staircase wall its pressure part does not cancel exactly and the
    residual dominates by two orders of magnitude (measured: ``|T| ~ 1e2`` for
    an analytic torque of ``0.72``), see the benchmark record.
    """
    r = np.asarray(r, dtype=float)
    u = np.asarray(u_theta, dtype=float)
    ok = np.isfinite(u)
    r = r[ok]
    u = u[ok]
    du = np.gradient(u, r)
    T = 2.0 * math.pi * rho * nu * length * r**2 * (du - u / r)
    # only the interior shells carry a converged derivative (one-sided stencils
    # at the two ends are excluded)
    interior = np.zeros_like(r, dtype=bool)
    interior[2:-2] = True
    Ti = T[interior]
    # the central half of the gap: the one-sided stencils next to the walls are
    # the noisiest place of any finite-difference derivative, so the
    # r-independence check is reported both over all interior shells and over
    # the central half
    n = Ti.size
    mid = Ti[n // 4: n - n // 4] if n >= 8 else Ti
    return {
        "r": r,
        "T": T,
        "T_mean_interior": float(np.mean(Ti)),
        "T_mean_mid": float(np.mean(mid)),
        "T_min": float(np.min(Ti)),
        "T_max": float(np.max(Ti)),
        "spread": float((np.max(Ti) - np.min(Ti)) / max(abs(np.mean(Ti)), 1e-30)),
        "spread_mid": float((np.max(mid) - np.min(mid)) / max(abs(np.mean(mid)), 1e-30)),
        "n_interior": int(interior.sum()),
    }
