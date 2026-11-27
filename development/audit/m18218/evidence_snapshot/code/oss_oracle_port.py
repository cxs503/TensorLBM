"""Faithful NumPy port of the OSS ``oracle-minimal-d2q9-fs``
(Maarten-vd-Sande/lbm, D2Q9 mass-tracking free surface, Thürey 2007).

Anchor: ``XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS``.

Zero-hand-copy provenance
-------------------------
Every rule below was read from the inventoried snapshot
``/root/oss-cache/Maarten-vd-Sande_lbm`` at the cited file/construct; no
number in this module comes from memory:

============================  =============================================
OSS location                  what is reproduced here
============================  =============================================
``lbm/util.py``               ``v`` (v[0]=(0,0), v[1]=(0,-1), v[2]=(0,1),
                              v[3]=(-1,0), ...), ``t`` weights
                              (4/9,1/9,1/9,1/9,1/36,1/36,1/9,1/36,1/36),
                              ``v_inv``, ``CT`` bit flags
                              (FLUID 1, INTERFACE 2, GAS 4, OBSTACLE 8,
                              INLET 16, OUTLET 32, NO_FLUID_NEIGH 64,
                              NO_EMPTY_NEIGH 128, NO_IFACE_NEIGH 256,
                              TO_FLUID 512, TO_GAS 1024)
``lbm/init.py``               float32 field allocation
``lbm/streamcollide.py``      ``lbm_collision`` (skip GAS|OBSTACLE,
                              ``update_rho_u``, ``set_equi``, ``collide``),
                              ``lbm_streaming`` (fluid branch, interface
                              branch, ABB on gas neighbours, normal
                              correction, ``mass`` bookkeeping, fluid
                              ``rho = mass``), ``get_epsilon``,
                              ``get_normal``, ``interface_mass_exchange``
``lbm/update_types.py``       ``update_types`` (``fill_offset = 0.003``,
                              ``lonely_tresh = 0.1``, envelope promotion /
                              demotion, excess-mass distribution along the
                              surface normal, neighbour-flag sweep) and
                              ``average_surround``
``lbm/time_evolve.py``        ``evolve`` step order and the
                              ``rho_prev = np.copy(mass)`` aliasing
``examples/dambreak.py``      reference case constants (nx=ny=170,
                              ``omega = [1.0]``, ``gravity = (0,-0.1)``,
                              block ``x in [0,70), y in [40,170)``,
                              ``mass = 1`` fluid / ``0.5`` interface,
                              ``rho += mass``, ``fin = equi.copy()``)
============================  =============================================

Upstream quirks reproduced bit-for-bit (measured from the source, not
guessed; each matters before transplanting anything):

Q1 ``evolve`` calls ``rho_prev = np.copy(mass)`` (not ``rho``).  During
   streaming this makes ``get_epsilon`` return ``mass/mass = 1`` for cells
   with ``mass > 0`` and ``0.5`` otherwise, so the interface weight
   ``(eps + eps_nei)/2`` degenerates and the surface normal driving the
   normal correction is identically zero.  Kept with
   ``rho_prev_alias=True`` (as shipped); ``False`` is the intended
   behaviour and is scanned as a sensitivity arm.
Q2 ``get_epsilon`` contains ``elif cell_type & CT.GAS == 2``; ``==`` binds
   tighter than ``&`` so the test is ``cell_type & False`` and never fires.
   GAS cells therefore fall into the ``else`` branch and return ``0.5``.
Q3 In the interface branch of ``lbm_streaming`` only ``n = 1..8`` are
   written, so ``f0`` of an interface cell keeps its pre-collision value.
Q4 ``examples/dambreak.py`` puts the column top on the obstacle ceiling and
   initialises the interface ring with ``rho = 0.5`` (``rho += mass``), not
   with the liquid density.
Q5 ``evolve`` leaves ``u_prev`` unbound until the end of step 0; harmless
   only because step 0 performs no GAS->INTERFACE conversion.

Usage
-----
>>> st = make_dambreak(nx=160, ny=96, a=32, h0=64)
>>> for _ in range(100):
...     step(st, omega=1.0, gravity=(0.0, -1e-4))
>>> st.front_cells(), st.liquid_inventory()
"""

from __future__ import annotations

import numpy as np

# --- lbm/util.py -----------------------------------------------------------
V = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [-1, -1], [-1, 1], [1, 0],
              [1, -1], [1, 1]], dtype=np.int64)
T = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
              1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float64)
V_INV = [int(np.flatnonzero((V == -V[i]).all(axis=1))[0]) for i in range(9)]

FLUID = 1
INTERFACE = 2
GAS = 4
OBSTACLE = 8
INLET = 16
OUTLET = 32
NO_FLUID_NEIGH = 64
NO_EMPTY_NEIGH = 128
NO_IFACE_NEIGH = 256
TO_FLUID = 512
TO_GAS = 1024

FILL_OFFSET = 0.003          # update_types: fill_offset
LONELY_TRESH = 0.1           # update_types: lonely_tresh
U_MAX = (2.0 / 3.0) ** 0.5   # update_rho_u: maxvel

_MOVING = tuple(range(1, 9))
_SL = (slice(1, -1), slice(1, -1))


def I(a):
    """Interior view (padded arrays carry a one-cell OBSTACLE border)."""
    return a[..., 1:-1, 1:-1]


class State:
    """All fields padded by one OBSTACLE cell (border reads stay in range)."""

    def __init__(self, nx: int, ny: int, dtype=np.float32):
        self.nx, self.ny, self.dtype = nx, ny, dtype
        sh = (nx + 2, ny + 2)
        self.fin = np.zeros((9,) + sh, dtype=dtype)
        self.fout = np.zeros((9,) + sh, dtype=dtype)
        self.equi = np.zeros((9,) + sh, dtype=dtype)
        self.fdist = np.zeros((9,) + sh, dtype=dtype)
        self.u = np.zeros((2,) + sh, dtype=dtype)
        self.rho = np.zeros(sh, dtype=dtype)
        self.mass = np.zeros(sh, dtype=dtype)
        self.ct = np.full(sh, GAS, dtype=np.int32)
        self.ct[0, :] = self.ct[-1, :] = OBSTACLE
        self.ct[:, 0] = self.ct[:, -1] = OBSTACLE

    # -- observables -------------------------------------------------------
    def flags(self):
        return I(self.ct)

    def liquid_inventory(self) -> float:
        ct, rho, m = self.flags(), I(self.rho), I(self.mass)
        return float(rho[(ct & FLUID) != 0].sum() + m[(ct & INTERFACE) != 0].sum())

    def front_cells(self) -> int:
        """Rightmost interior column holding liquid (0 = wall column)."""
        fluid = (self.flags() & (FLUID | INTERFACE)) != 0
        cols = np.flatnonzero(fluid.any(axis=1))
        return int(cols[-1]) if cols.size else 0

    def surface_profile(self):
        """Top-most liquid row + 1 per column, in lattice cells."""
        fluid = (self.flags() & (FLUID | INTERFACE)) != 0
        h = np.zeros(fluid.shape[0], dtype=np.float64)
        for i in range(fluid.shape[0]):
            rows = np.flatnonzero(fluid[i])
            h[i] = (rows[-1] + 1) if rows.size else 0.0
        return h

    def velocity_magnitude(self):
        fluid = (self.flags() & (FLUID | INTERFACE)) != 0
        u = np.sqrt((I(self.u) ** 2).sum(axis=0))
        return np.where(fluid, u, 0.0)

    def snaps(self):
        """JSON-friendly snapshot of the instantaneous state."""
        return {
            "front_cells": self.front_cells(),
            "inventory": self.liquid_inventory(),
            "profile": self.surface_profile().tolist(),
            "u_max": float(np.abs(I(self.u)).max()),
        }


# ---------------------------------------------------------------------------
# neighbour access (on padded arrays, result is interior shaped)
# ---------------------------------------------------------------------------
def pull(field, k):
    """``field`` sampled at the pull source ``x - v[k]``."""
    vx, vy = int(V[k, 0]), int(V[k, 1])
    nx, ny = field.shape[-2] - 2, field.shape[-1] - 2
    if field.ndim == 2:
        return field[1 - vx:1 - vx + nx, 1 - vy:1 - vy + ny]
    return field[:, 1 - vx:1 - vx + nx, 1 - vy:1 - vy + ny]


def push(field, k):
    """``field`` sampled at ``x + v[k]`` (input must be a padded array)."""
    return pull(field, V_INV[k])


def pushi(field, k, fill=0.0):
    """``x + v[k]`` for an interior-shaped field; outside = ``fill``."""
    return push(np.pad(field, 1, constant_values=fill), k)


def pulli(field, k, fill=0.0):
    """``x - v[k]`` for an interior-shaped field; outside = ``fill``.

    ``lbm_streaming`` and ``update_types`` sweep the lattice with
    ``ix_next = ix - v[n]`` (population source), whereas
    ``average_surround`` and the excess-mass split use ``ix + v[n]``
    (receiver side); mixing the two silently pairs the wrong neighbours.
    """
    return pull(np.pad(field, 1, constant_values=fill), k)


# ---------------------------------------------------------------------------
# streamcollide.py helpers
# ---------------------------------------------------------------------------
def get_epsilon(ct, rho, mass):
    """``get_epsilon`` incl. quirk Q2 (``& CT.GAS == 2`` can never fire)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.where(rho > 0, mass / np.where(rho > 0, rho, 1.0), 0.5)
    val = np.clip(val, 0.0, 1.0)
    return np.where((ct & (FLUID | OBSTACLE)) != 0, 1.0, val)


def get_normal(ct, rho, mass):
    """``get_normal``: 0.5*(eps[x-1] - eps[x+1]) per axis, wall border eps = 1."""
    eps = np.pad(get_epsilon(ct, rho, mass), 1, constant_values=1.0)
    nx_ = 0.5 * (eps[0:-2, 1:-1] - eps[2:, 1:-1])
    ny_ = 0.5 * (eps[1:-1, 0:-2] - eps[1:-1, 2:])
    return nx_, ny_


def interface_mass_exchange(ct_self, ct_nei, fout_self, fout_nei):
    """``interface_mass_exchange`` (Thürey 4-way selector), vectorised."""
    nf_self = (ct_self & NO_FLUID_NEIGH) != 0
    nf_nei = (ct_nei & NO_FLUID_NEIGH) != 0
    ne_self = (ct_self & NO_EMPTY_NEIGH) != 0
    ne_nei = (ct_nei & NO_EMPTY_NEIGH) != 0
    out = np.zeros_like(fout_self, dtype=np.float64)
    b1 = nf_self
    out = np.where(b1 & nf_nei, fout_nei - fout_self, out)
    out = np.where(b1 & ~nf_nei, -fout_self, out)
    b2 = (~nf_self) & ne_self
    out = np.where(b2 & ne_nei, fout_nei - fout_self, out)
    out = np.where(b2 & ~ne_nei, fout_nei, out)
    b3 = (~nf_self) & (~ne_self)
    out = np.where(b3 & nf_nei, fout_nei, out)
    out = np.where(b3 & (~nf_nei) & ne_nei, fout_self, out)
    out = np.where(b3 & (~nf_nei) & (~ne_nei), fout_nei - fout_self, out)
    return out


def equilibrium(rho, ux, uy):
    """``set_equi``: feq[k] = rho t_k (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 |u|^2)."""
    usqr = 1.5 * (ux * ux + uy * uy)
    out = np.empty((9,) + rho.shape, dtype=np.float64)
    for k in range(9):
        vu = 3.0 * (V[k, 0] * ux + V[k, 1] * uy)
        out[k] = rho * T[k] * (1.0 + vu + 0.5 * vu * vu - usqr)
    return out


def _update_rho_u(st):
    """``update_rho_u`` on ``fin`` (sequential accumulation as upstream)."""
    ct, fin = st.flags(), st.fin
    act = (ct & (GAS | OBSTACLE)) == 0
    rho = np.zeros(ct.shape, dtype=np.float64)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    for k in range(9):
        f = I(fin)[k]
        rho += f
        ux += V[k, 0] * f
        uy += V[k, 1] * f
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(rho != 0, ux / np.where(rho != 0, rho, 1.0), 0.0)
        uy = np.where(rho != 0, uy / np.where(rho != 0, rho, 1.0), 0.0)
    vel = np.sqrt(ux * ux + uy * uy)
    scale = np.where(vel > U_MAX, U_MAX / np.where(vel > 0, vel, 1.0), 1.0)
    ux, uy = ux * scale, uy * scale
    r_view, ux_view, uy_view = I(st.rho), I(st.u)[0], I(st.u)[1]
    r_view[act] = rho[act].astype(st.dtype)
    ux_view[act] = ux[act].astype(st.dtype)
    uy_view[act] = uy[act].astype(st.dtype)


def collision(st, omega: float, gravity):
    """``lbm_collision``: BGK plus the linearised gravity population term."""
    ct = st.flags()
    if (ct & (INLET | OUTLET)).any():
        raise NotImplementedError("INLET/OUTLET cells are outside this XVAL case")
    _update_rho_u(st)
    act = (ct & (GAS | OBSTACLE)) == 0
    eq = equilibrium(I(st.rho), I(st.u)[0], I(st.u)[1])
    eq_view = I(st.equi)
    eq_view[:] = np.where(act, eq, eq_view)
    gx, gy = float(gravity[0]), float(gravity[1])
    fo_view, fi = I(st.fout), I(st.fin)
    for k in range(9):
        grav = V[k, 0] * gx + V[k, 1] * gy
        new = fi[k] + omega * (eq_view[k] - fi[k]) - I(st.rho) * T[k] * grav
        fo_view[k] = np.where(act, new, fo_view[k])
    return st


def streaming(st, rho_prev, mass_prev):
    """``lbm_streaming``: fluid and interface branches + ABB + normal fix."""
    ct, ct_full = st.flags(), st.ct
    rho_prev, mass_prev = I(rho_prev), I(mass_prev)
    fin_view, fout = I(st.fin), st.fout
    fin = fin_view.copy()
    eq_view = I(st.equi)

    fluid = (ct & (FLUID | INLET | OUTLET)) != 0
    iface = (ct & INTERFACE) != 0

    # interface cells carry the purity-1 equilibrium with the current velocity
    eq_if = equilibrium(np.ones(ct.shape, dtype=np.float64), I(st.u)[0], I(st.u)[1])
    eq_view[:] = np.where(iface, eq_if, eq_view)

    eps_prev = get_epsilon(ct, rho_prev, mass_prev)
    dmass = np.zeros(ct.shape, dtype=np.float64)

    for k in range(9):
        pulled = pull(fout, k)[k]
        opp = V_INV[k]
        f_self = I(fout)[opp]
        nb_ok_full = (ct_full & (FLUID | INTERFACE | INLET | OUTLET)) != 0
        nb_ok = pull(nb_ok_full, k)

        # ---- fluid branch (k = 0 included: pull from itself) --------------
        ok_f = fluid & nb_ok
        if k == 0:
            fin[k] = np.where(ok_f, pulled, fin[k])
            dmass += np.where(ok_f, pulled - f_self, 0.0)
            continue
        fin[k] = np.where(ok_f, pulled, fin[k])
        fin[k] = np.where(fluid & ~nb_ok, f_self, fin[k])          # bounce-back
        dmass += np.where(ok_f, pulled - f_self, 0.0)

        # ---- interface branch --------------------------------------------
        f_at = pull((ct_full & FLUID) != 0, k)
        i_at = pull((ct_full & INTERFACE) != 0, k)
        g_at = pull((ct_full & GAS) != 0, k)
        m_f = iface & f_at
        m_i = iface & i_at
        m_g = iface & g_at
        m_o = iface & ~f_at & ~i_at & ~g_at

        fin[k] = np.where(m_f | m_i, pulled, fin[k])
        dmass += np.where(m_f, pulled - f_self, 0.0)
        eps_nei = pulli(eps_prev, k, 1.0)
        exch = interface_mass_exchange(ct, pull(ct_full, k), f_self, pulled)
        dmass += np.where(m_i, exch * 0.5 * (eps_prev + eps_nei), 0.0)
        fin[k] = np.where(m_g, eq_view[k] + eq_view[opp] - f_self, fin[k])
        fin[k] = np.where(m_o, f_self, fin[k])

    # surface-normal correction (inert when rho_prev is aliased to mass, Q1)
    nrm_x, nrm_y = get_normal(ct, rho_prev, mass_prev)
    for k in _MOVING:
        opp = V_INV[k]
        sel = iface & ((V[opp, 0] * nrm_x + V[opp, 1] * nrm_y) > 0)
        fin[k] = np.where(sel, eq_view[k] + eq_view[opp] - I(fout)[opp], fin[k])

    fin_view[:] = fin.astype(st.dtype)
    I(st.mass)[:] = (I(st.mass) + dmass).astype(st.dtype)

    # after streaming: rho,u from fin; fluid cells take rho = mass
    _update_rho_u(st)
    fl = (ct & FLUID) != 0
    I(st.rho)[fl] = I(st.mass)[fl]
    return st, dmass


def average_surround(st, promote, rho_prev, mass_prev, u_prev, ct_prev):
    """``average_surround`` on the promoted cells (vectorised)."""
    live = (I(ct_prev) & (FLUID | INTERFACE)) != 0
    cnt = np.zeros(promote.shape, dtype=np.float64)
    rho_s = np.zeros_like(cnt)
    ux_s = np.zeros_like(cnt)
    uy_s = np.zeros_like(cnt)
    up = I(u_prev)
    for k in _MOVING:
        m = pushi(live, k, False)
        cnt += m
        rho_s += np.where(m, pushi(I(rho_prev), k, 0.0), 0.0)
        ux_s += np.where(m, pushi(up[0], k, 0.0), 0.0)
        uy_s += np.where(m, pushi(up[1], k, 0.0), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(cnt > 0, cnt, 1.0)
        ok = cnt > 0
        rho_n = np.where(ok, rho_s / c, 0.0)
        ux_n = np.where(ok, ux_s / c, 0.0)
        uy_n = np.where(ok, uy_s / c, 0.0)
    I(st.mass)[promote] = 0.0
    I(st.rho)[promote] = rho_n[promote].astype(st.dtype)
    I(st.u)[0][promote] = ux_n[promote].astype(st.dtype)
    I(st.u)[1][promote] = uy_n[promote].astype(st.dtype)
    I(st.equi)[:, promote] = equilibrium(I(st.rho), I(st.u)[0], I(st.u)[1])[:, promote]
    return st


def update_types(st, rho_prev, mass_prev, u_prev, ct_prev):
    """``update_types``: conversion marks, envelope, excess mass, flags."""
    ct = st.ct.copy()
    ctI = I(ct)
    rho, mass = I(st.rho), I(st.mass)

    # --- sweep 1: mark conversions -----------------------------------------
    iface = (ctI & INTERFACE) != 0
    no_empty = (ctI & NO_EMPTY_NEIGH) != 0
    no_fluid = (ctI & NO_FLUID_NEIGH) != 0
    no_iface = (ctI & NO_IFACE_NEIGH) != 0
    to_fluid = iface & ((mass > (1.0 + FILL_OFFSET) * rho)
                        | ((mass >= (1.0 - LONELY_TRESH) * rho) & no_empty))
    to_gas = iface & (~to_fluid) & ((mass < -FILL_OFFSET * rho)
                                    | ((mass <= LONELY_TRESH * rho) & no_fluid)
                                    | no_iface | no_fluid)
    ctI = np.where(to_fluid, TO_FLUID, ctI)
    ctI = np.where(to_gas, TO_GAS, ctI)
    ctI = ctI & ~(NO_FLUID_NEIGH + NO_EMPTY_NEIGH + NO_IFACE_NEIGH)
    I(ct)[:] = ctI

    # --- sweep 2: interface -> fluid envelope (GAS neighbours refilled) ----
    tf = I(ct) & TO_FLUID != 0
    promote = np.zeros(tf.shape, dtype=bool)
    for k in _MOVING:
        promote |= pulli(tf, k, False) & ((I(ct) & GAS) != 0)
    if promote.any():
        I(ct)[promote] = INTERFACE
        average_surround(st, promote, rho_prev, mass_prev, u_prev, ct_prev)

    # --- sweep 3: interface -> gas envelope (FLUID neighbours demoted) -----
    tg = (I(ct) & TO_GAS) != 0
    demote = np.zeros(tg.shape, dtype=bool)
    for k in _MOVING:
        demote |= pulli(tg, k, False) & ((I(ct) & FLUID) != 0)
    I(ct)[demote] = INTERFACE

    # --- sweep 4: distribute excess mass -----------------------------------
    nrm_x, nrm_y = get_normal(I(ct_prev), I(rho_prev), I(mass_prev))
    nrm_x = np.where(tg, -nrm_x, nrm_x)
    nrm_y = np.where(tg, -nrm_y, nrm_y)
    mex = np.zeros(nrm_x.shape, dtype=np.float64)
    mex = np.where(tf, mass - rho, mex)
    mex = np.where(tg, mass, mex)
    mass = np.where(tf, rho, mass)
    mass = np.where(tg, 0.0, mass)

    eta = np.zeros((9,) + mass.shape, dtype=np.float64)
    is_if = np.zeros((9,) + mass.shape, dtype=bool)
    ct_now = I(ct)
    for k in _MOVING:
        nb_if = pushi((ct_now & INTERFACE) != 0, k, False)
        e = V[k, 0] * nrm_x + V[k, 1] * nrm_y
        eta[k] = np.where(nb_if, np.maximum(e, 0.0), 0.0)
        is_if[k] = nb_if
    eta_total = eta.sum(axis=0)
    if_total = is_if.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = eta_total > 0
        fdist = np.where(ok, eta * mex / np.where(ok, eta_total, 1.0), 0.0)
        fb = (~ok) & (if_total > 0)
        per = np.where(if_total > 0, mex / np.where(if_total > 0, if_total, 1.0), 0.0)
        fdist = np.where(fb, np.where(is_if, per, 0.0), fdist)
    add = np.zeros(mass.shape, dtype=np.float64)
    for k in _MOVING:
        add += pushi(fdist[k], k, 0.0)
    mass = np.where((ct_now & INTERFACE) != 0, mass + add, mass)
    ct_now = np.where(tf, FLUID, ct_now)
    ct_now = np.where(tg, GAS, ct_now)

    # --- sweep 5: neighbourhood flags --------------------------------------
    not_obs = (ct_now & OBSTACLE) == 0
    has_f = np.zeros(not_obs.shape, dtype=bool)
    has_g = np.zeros_like(has_f)
    has_i = np.zeros_like(has_f)
    for k in _MOVING:
        has_f |= pulli((ct_now & FLUID) != 0, k, False)
        has_g |= pulli((ct_now & GAS) != 0, k, False)
        has_i |= pulli((ct_now & INTERFACE) != 0, k, False)
    base = NO_FLUID_NEIGH | NO_EMPTY_NEIGH | NO_IFACE_NEIGH
    ct_now = np.where(not_obs, ct_now | base, ct_now)
    ct_now = np.where(not_obs & has_f, ct_now & ~NO_FLUID_NEIGH, ct_now)
    ct_now = np.where(not_obs & has_g, ct_now & ~NO_EMPTY_NEIGH, ct_now)
    ct_now = np.where(not_obs & has_i, ct_now & ~NO_IFACE_NEIGH, ct_now)

    I(ct)[:] = ct_now.astype(np.int32)
    st.ct[:] = ct                      # commit flags (copy -> state)
    I(st.mass)[:] = mass.astype(st.dtype)
    return st


def step(st, omega: float = 1.0, gravity=(0.0, -0.1), rho_prev_alias: bool = True):
    """One ``evolve`` iteration: collision -> stream -> update_types."""
    collision(st, omega, gravity)
    mass_prev = st.mass.copy()
    rho_prev = (st.mass.copy() if rho_prev_alias else st.rho.copy())  # Q1
    streaming(st, rho_prev, mass_prev)
    rho_prev2 = st.rho.copy()
    mass_prev2 = st.mass.copy()
    u_prev = st.u.copy()
    ct_prev = st.ct.copy()
    update_types(st, rho_prev2, mass_prev2, u_prev, ct_prev)
    return st


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def make_dambreak(nx: int, ny: int, a: int, h0: int,
                  interface_rho: str = "oss", dtype=np.float32) -> State:
    """OSS dam-break initialisation on a walled box.

    Block occupies interior columns ``1..a`` and rows ``1..h0`` (the wall is
    column/row 0).  ``interface_rho="oss"`` reproduces
    ``examples/dambreak.py`` (``mass`` 1 / 0.5, ``rho += mass`` -> interface
    ring at ``rho = 0.5``, quirk Q4); ``"liquid"`` keeps ``mass = 0.5`` but
    starts the ring at the liquid density.
    """
    st = State(nx, ny, dtype=dtype)
    ct = I(st.ct)
    ct[:] = GAS
    ct[0, :] = OBSTACLE
    ct[-1, :] = OBSTACLE
    ct[:, 0] = OBSTACLE
    ct[:, -1] = OBSTACLE
    ct[1:a + 1, 1:h0 + 1] = INTERFACE
    ct[2:a, 2:h0] = FLUID
    ring_rho = 0.5 if interface_rho == "oss" else 1.0
    I(st.mass)[:] = np.where(ct == FLUID, 1.0, np.where(ct == INTERFACE, 0.5, 0.0))
    I(st.rho)[:] = np.where(ct == FLUID, 1.0, np.where(ct == INTERFACE, ring_rho, 0.0))
    st.u[:] = 0.0
    st.equi[:] = equilibrium(st.rho, st.u[0], st.u[1])
    st.fin[:] = st.equi
    st.fout[:] = 0.0
    return st


def make_oss_block(nx: int, ny: int, a: int, h0: int, dtype=np.float32) -> State:
    """Bit-faithful ``make_state`` in :mod:`oss_original` (upstream block).

    Block spans interior columns ``0..a-1`` and rows ``ny-h0..ny-1`` with an
    INTERFACE ring; the four walls are then stamped over it (this replicates
    ``examples/dambreak.py``, where the block top sits on the ceiling).
    Used for the port-equivalence test only.
    """
    st = State(nx, ny, dtype=dtype)
    ct = I(st.ct)
    ct[:] = GAS
    xx, yy = np.meshgrid(np.arange(0, a), np.arange(ny - h0, ny), indexing="ij")
    ct[xx, yy] = INTERFACE
    ct[xx[1:-1, 1:-1], yy[1:-1, 1:-1]] = FLUID
    ct[:, 0] = OBSTACLE
    ct[:, -1] = OBSTACLE
    ct[0, :] = OBSTACLE
    ct[-1, :] = OBSTACLE
    I(st.mass)[:] = np.where(ct == FLUID, 1.0, np.where(ct == INTERFACE, 0.5, 0.0))
    I(st.rho)[:] = I(st.mass)
    st.u[:] = 0.0
    st.equi[:] = equilibrium(st.rho, st.u[0], st.u[1])
    st.fin[:] = st.equi
    st.fout[:] = 0.0
    return st


def make_oss_example(nx: int = 170, ny: int = 170, dtype=np.float32) -> State:
    """Exact ``examples/dambreak.py`` initial state (block x<70, y>=40)."""
    st = State(nx, ny, dtype=dtype)
    ct = I(st.ct)
    ct[:] = GAS
    xs, ys = np.arange(0, 70), np.arange(40, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    ct[xx, yy] = INTERFACE
    ct[xx[1:-1, 1:-1], yy[1:-1, 1:-1]] = FLUID
    ct[:, 0] = OBSTACLE
    ct[:, -1] = OBSTACLE
    ct[0, :] = OBSTACLE
    ct[-1, :] = OBSTACLE
    I(st.mass)[:] = np.where(ct == FLUID, 1.0, np.where(ct == INTERFACE, 0.5, 0.0))
    I(st.rho)[:] = I(st.mass)
    st.u[:] = 0.0
    st.equi[:] = equilibrium(st.rho, st.u[0], st.u[1])
    st.fin[:] = st.equi
    st.fout[:] = 0.0
    return st