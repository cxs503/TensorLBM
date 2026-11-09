"""3D floe DEM in the ship frame: the first-principles ice-resistance engine.

The floe field of :mod:`floe_field` is advanced in the **ship frame**: the hull is
at rest and the broken-ice field drifts past it at the ship speed.  Every floe
carries its own vertical position, so buoyancy, hull-induced submergence and the
ice draft are resolved; the in-plane motion carries the floe-floe and floe-hull
contact mechanics.

Forces on a floe
----------------
* buoyancy / weight (cylinder of thickness ``h``, submerged fraction from ``z``)
* water drag on the floe's frontal area and on its submerged plan area, computed
  from the **local water velocity** (onset flow slowed down in the hull boundary
  layer, i.e. the flow -> ice branch of the two-way coupling)
* floe-floe contacts: linear spring-dashpot + Coulomb friction (in-plane)
* floe-hull contacts: the hull is the offset surface
  ``|y| = b(x, z)`` of the KVLCC2 geometry, so the contact normal carries the
  hull's local obliquity (which is what generates the longitudinal resistance);
  linear spring-dashpot + Coulomb friction

Known limitations (see the node report, section 5)
-----------------------------------------------
A floe can be pushed laterally into the hull footprint between two projection
steps by its neighbours, i.e. the hull contact is not a hard invariant; the
contact-dominated statistics therefore need a smaller time step and a proper
multi-body split-impulse solver before the resistance can be declared converged.

Boundary conditions
-------------------
Periodic recycling in ``x`` (floes leaving downstream re-enter upstream with the
onset velocity), confining walls in ``y`` at ``+-W/2`` (a brash-ice *channel* of
width W -- the resistance therefore depends on W, which is quantified in the
report), rigid lid on the vertical coordinate (floes cannot leave the surface).

Energy (resistance) audit
-------------------------
The module accumulates every dissipative term so that the resistance can be
closed against energy:

    R_ice * U = -P_water + D_contacts + sum(F_hull,x (U - v_x)) + sum(F_hull . v_perp) + ...

which is the exact steady-state power balance of the floe phase (see
:meth:`DEMField.energy_audit`).  ``U`` is the ship speed.

References
----------
- Cundall & Strack (1979), DEM contact model.
- Schiller & Naumann (1935) / standard drag closures for the floe-water drag.
- KVLCC2 offset geometry: SIMMAN 2008; see ``ship_hull_geometry``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from ..ship_hull_geometry import KVLCC2Geometry, kvlcc2_halfbreadth
from .floe_field import FloeField, FloeFieldConfig, generate_floe_field


@dataclass
class DEMConfig:
    """Configuration of the ship-frame floe DEM (SI units)."""

    # domain
    channel_width: float = 100.0     # lateral extent W [m]
    buffer_upstream: float = 60.0    # undisturbed inflow length upstream of FP [m]
    buffer_downstream: float = 60.0  # wake length downstream of AP [m]
    # hull
    L_pp: float = 230.0
    B: float = 32.29
    T: float = 14.5
    # ice
    concentration: float = 0.60
    thickness: float = 0.15
    d50: float = 1.0
    sigma_g: float = 1.35
    rho_ice: float = 917.0
    rho_water: float = 1025.0
    # operation
    ship_speed: float = 1.5          # [m/s]
    dt: float = 1.0e-3               # [s]
    n_steps: int = 6000
    # contacts
    k_normal: float = 5.0e4          # [N/m] ice-ice contact stiffness (stability-limited)
    c_normal: float = 3.0e3          # [N s/m] ice-ice contact damping
    mu_ice_ice: float = 0.15
    mu_ice_hull: float = 0.15
    # longitudinal extent of the hull that can be in contact with ice, as a
    # fraction of L_pp measured from AP.  The aft 35 % of a full hull is excluded:
    # behind the maximum-section the hull narrows again, so a laterally confined
    # ice field would be squeezed there and produce a spurious thrust; in reality
    # that region sits in the ship's own cleared wake.
    ice_contact_span: tuple = (0.35, 1.0)
    max_velocity_factor: float = 4.0  # numerical safety limiter (x ship speed)
    max_surface_slope: float = 2.0    # clamp |db/dx| of the parametric offsets
    zeta_vertical: float = 0.8       # vertical damping ratio
    # water
    drag_coeff_frontal: float = 1.0
    drag_coeff_planar: float = 5.0e-3
    blockage_length: float = 4.0     # hull boundary-layer thickness for the flow->ice coupling [m]
    # numerics
    neighbour_radius: float = 2.2    # contact search radius [m]
    rebuild_every: int = 10
    seed: int = 20261108

    def __post_init__(self) -> None:
        if self.channel_width <= self.B:
            raise ValueError("channel_width must exceed the beam")
        if self.dt <= 0.0 or self.n_steps <= 0:
            raise ValueError("dt / n_steps must be positive")
        if not (0.0 < self.concentration < 1.0):
            raise ValueError("concentration must be in (0, 1)")


class _HullSurface:
    """Tabulated KVLCC2 offset surface ``b(x, z)`` plus its gradients."""

    def __init__(self, cfg: DEMConfig):
        self.cfg = cfg
        self.hull = KVLCC2Geometry(scale=1.0, L_pp=cfg.L_pp, B=cfg.B, T=cfg.T)
        self.dx = 0.5
        self.dz = 0.05
        self.nx = int(round(cfg.L_pp / self.dx)) + 1
        self.nz = int(round(cfg.T / self.dz)) + 1
        self.x0 = -(cfg.buffer_upstream)          # AP at x = 0, FP at x = L_pp
        b = np.zeros((self.nz, self.nx))
        for i in range(self.nx):
            # the domain x axis runs from the bow (FP) to the stern (AP), because
            # in the ship frame the broken-ice field drifts from the bow aftwards
            x = cfg.L_pp - i * self.dx
            for k in range(self.nz):
                z = k * self.dz
                b[k, i] = kvlcc2_halfbreadth(x, min(z, 0.999 * cfg.T), self.hull)
        self.b = b
        # NOTE: the parametric half-breadth formula has a square-root entry at the
        # very bow tip, so db/dx diverges there.  The real waterline entry is a
        # finite angle, so the tabulated slope is clamped (documented modelling
        # choice) instead of letting a singular contact normal appear.
        self.dbdx = np.clip(np.gradient(b, self.dx, axis=1), -cfg.max_surface_slope, cfg.max_surface_slope)
        self.dbdz = np.clip(np.gradient(b, self.dz, axis=0), -1.5, 1.5)
        # distance from the hull boundary for the flow->ice coupling
        self.z_wl = cfg.T

    def sample(self, x: np.ndarray, z: np.ndarray):
        """Bilinear interpolation of b, db/dx, db/dz at (x, z)."""
        cfg = self.cfg
        xi = np.clip(x / self.dx, 0.0, self.nx - 1.000001)
        zi = np.clip(z / self.dz, 0.0, self.nz - 1.000001)
        i0 = np.floor(xi).astype(np.int64)
        k0 = np.floor(zi).astype(np.int64)
        fx = xi - i0
        fz = zi - k0
        def bil(arr):
            return ((1 - fz) * ((1 - fx) * arr[k0, i0] + fx * arr[k0, i0 + 1])
                    + fz * ((1 - fx) * arr[k0 + 1, i0] + fx * arr[k0 + 1, i0 + 1]))
        return bil(self.b), bil(self.dbdx), bil(self.dbdz)


class DEMField:
    """3D floe DEM of a broken-ice field drifting past the KVLCC2 hull."""

    def __init__(self, cfg: DEMConfig, field_: Optional[FloeField] = None):
        self.cfg = cfg
        self.hull = _HullSurface(cfg)
        self.U = cfg.ship_speed
        # longitudinal placement: the hull (AP at x=0 in its own frame) sits
        # ``buffer_upstream`` downstream of the inlet, so that the recycled floes
        # enter an undisturbed region instead of being injected into the hull
        self.hull_x0 = cfg.buffer_upstream

        Lx = cfg.L_pp + cfg.buffer_upstream + cfg.buffer_downstream
        self.Lx = Lx
        self.W = cfg.channel_width
        fc = FloeFieldConfig(
            length=Lx, width=self.W, concentration=cfg.concentration,
            thickness=cfg.thickness, d50=cfg.d50, sigma_g=cfg.sigma_g,
            rho_ice=cfg.rho_ice, rho_water=cfg.rho_water, seed=cfg.seed,
        )
        self.field_cfg = fc
        self.field = field_ if field_ is not None else generate_floe_field(fc)

        # ship frame: hull centred laterally, AP at x = 0, FP at x = L_pp.
        # Floes that start inside (or immediately against) the hull footprint are
        # removed: they would otherwise be created with a finite overlap and the
        # penalty contact would explode in the first step.  The recycling inflow
        # refills the cleared area naturally as the run develops.
        xs = self.field.x.copy()
        ys = self.field.y - 0.5 * self.W
        rs = 0.5 * self.field.d
        self._rng = np.random.default_rng(cfg.seed)
        hull = self.hull
        xh = xs - self.hull_x0
        in_span = (xh > 0.0) & (xh < cfg.L_pp)
        # clearance is checked against the *waterline* half-breadth, which is the
        # widest section the floe can meet (checking mid-draft would leave floes
        # already overlapping the waterline hull footprint at t = 0)
        b_at, _, _ = hull.sample(np.clip(xh, 0.0, cfg.L_pp), np.full_like(xs, 0.999 * cfg.T))
        clear = np.where(in_span, np.abs(ys) - b_at, 1.0e3) > (rs + 0.05)
        keep = clear
        self.initial_clear_fraction = float(keep.mean())
        n = int(keep.sum())
        self.n = n
        self.x = xs[keep]
        self.y = ys[keep]
        self.d = self.field.d[keep]
        self.r = rs[keep]
        self.m = self.field.mass[keep]
        self.area = self.field.area[keep]
        self.h = cfg.thickness
        self.vx = np.full(n, cfg.ship_speed)
        self.vy = np.zeros(n)
        self.vz = np.zeros(n)
        # buoyancy equilibrium: submerged depth = h rho_i / rho_w
        self.z = np.full(n, -0.5 * self.h + self.h * cfg.rho_ice / cfg.rho_water)
        self.k_buoy = cfg.rho_water * 9.81 * self.area
        self.c_vert = cfg.zeta_vertical * 2.0 * np.sqrt(self.k_buoy * self.m)

        # diagnostics
        self.R_hist: list[float] = []
        self.R_stern_hist: list[float] = []
        self.decomp_hist: list[Dict[str, float]] = []
        self._pairs_ii = np.empty((2, 0), dtype=np.int64)
        self._n_contacts = 0
        self.energy = {
            "P_hull_dot_v": 0.0,       # work rate of the hull force on the floes
            "P_hull_slip": 0.0,        # hull-x slip work  sum F_hx (U - vx)
            "P_hull_perp": 0.0,        # lateral/vertical hull work sum F_h . v_perp
            "P_water": 0.0,            # work rate of the water drag on the floes
            "D_contacts": 0.0,         # dissipative part of all contacts
            "KE_flux": 0.0,            # kinetic-energy flux through the recycling BCs
            "R_mean": 0.0,
        }
        self._ke_in = 0.0
        self._ke_out = 0.0
        self._t_accum = 0.0

    # ------------------------------------------------------------- helpers
    def _water_velocity(self, x, y, z):
        """Ship-frame water velocity seen by a floe (onset flow + hull blockage).

        The hull displaces water, so in the ship frame the water next to the hull
        is nearly still and recovers to the onset velocity ``U`` over the
        boundary-layer scale ``cfg.blockage_length``.  This is the flow -> ice
        branch of the FSI coupling used by the DEM (the Eulerian branch is solved
        in :mod:`fsi`).
        """
        xh = x - self.hull_x0
        b, _, _ = self.hull.sample(np.clip(xh, 0.0, self.cfg.L_pp),
                                   np.clip(self.cfg.T + z, 1e-3, 0.999 * self.cfg.T))
        in_hull_span = (xh > 0.0) & (xh < self.cfg.L_pp)
        gap = np.where(in_hull_span, np.abs(np.abs(y) - b), 1.0e3)
        gap = np.where(in_hull_span & (np.abs(y) <= b), 0.0, gap)
        red = np.exp(-gap / self.cfg.blockage_length)
        ux = self.U * (1.0 - red)
        return ux, np.zeros_like(ux), np.zeros_like(ux)

    def _rebuild_pairs(self) -> None:
        pts = np.column_stack([self.x, self.y])
        tree = cKDTree(pts)
        pairs = tree.query_pairs(self.cfg.neighbour_radius, output_type="ndarray")
        if pairs.size:
            # keep only true (non-overlapping) pairs - touching floes included
            i, j = pairs[:, 0], pairs[:, 1]
            dist = np.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j])
            keep = dist <= (self.r[i] + self.r[j]) * 1.001
            pairs = pairs[keep]
        if pairs.size:
            i, j = pairs[:, 0], pairs[:, 1]
            dist = np.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j])
            keep = dist < (self.r[i] + self.r[j])
            pairs = pairs[keep]
        self._pairs_ii = np.ascontiguousarray(pairs.T) if pairs.size else np.empty((2, 0), dtype=np.int64)

    def _contact_forces(self):
        """Return (force_on_floes xyz, dissipation rate, hull force accumulator)."""
        cfg = self.cfg
        n = self.n
        fx = np.zeros(n)
        fy = np.zeros(n)
        fz = np.zeros(n)
        diss = 0.0

        # ---------------- floe-floe (in-plane) ----------------
        if self._pairs_ii.shape[1]:
            i, j = self._pairs_ii
            dx = self.x[j] - self.x[i]
            dy = self.y[j] - self.y[i]
            dist = np.hypot(dx, dy)
            dist = np.where(dist < 1e-9, 1e-9, dist)
            nx = dx / dist
            ny = dy / dist
            delta = (self.r[i] + self.r[j]) - dist
            act = delta > 0.0
            if np.any(act):
                i, j, nx, ny, delta = i[act], j[act], nx[act], ny[act], delta[act]
                rel_vx = self.vx[j] - self.vx[i]
                rel_vy = self.vy[j] - self.vy[i]
                vn = rel_vx * nx + rel_vy * ny
                fn = cfg.k_normal * delta - cfg.c_normal * vn
                fn = np.maximum(fn, 0.0)
                # tangential (Coulomb)
                tvx = rel_vx - vn * nx
                tvy = rel_vy - vn * ny
                tmag = np.hypot(tvx, tvy)
                tmag_safe = np.where(tmag < 1e-9, 1e-9, tmag)
                ft = cfg.mu_ice_ice * fn
                ftx = -ft * tvx / tmag_safe
                fty = -ft * tvy / tmag_safe
                FX = fn * nx + ftx
                FY = fn * ny + fty
                np.add.at(fx, j, FX)
                np.add.at(fy, j, FY)
                np.add.at(fx, i, -FX)
                np.add.at(fy, i, -FY)
                diss += float(np.sum(cfg.c_normal * np.maximum(-vn, 0.0) * delta)
                              + np.sum(ft * tmag))
                self._n_contacts = int(i.size)

        # ---------------- floe-hull (impulse / projection) ----------------
        # A stiff penalty contact against the hull is numerically fragile once the
        # floe field jams (floes sit in the oblique bow corner and the penalty
        # force diverges).  The hull contact is therefore solved at impulse level:
        # a penetrating floe is projected back onto the offset surface and its
        # normal velocity is cancelled (restitution 0), with a Coulomb friction
        # impulse.  The equivalent contact force over the step is J / dt, and the
        # dissipated energy is available in closed form, which is what the energy
        # audit of criterion C4 needs.
        xh = self.x - self.hull_x0
        in_span = (xh > 0.0) & (xh < cfg.L_pp)
        self._counted = ((xh > cfg.ice_contact_span[0] * cfg.L_pp)
                         & (xh < cfg.ice_contact_span[1] * cfg.L_pp))
        self._hull_impulse = None
        if np.any(in_span):
            idx = np.nonzero(in_span)[0]
            zz = np.clip(cfg.T + self.z[idx], 1e-3, 0.999 * cfg.T)
            b, dbdx, dbdz = self.hull.sample(xh[idx], zz)
            ay = np.abs(self.y[idx])
            pen = self.r[idx] - (ay - b)
            act = pen > 0.0
            if np.any(act):
                idx = idx[act]
                pen = pen[act]
                b = b[act]
                dbdx = dbdx[act]
                dbdz = dbdz[act]
                sgn = np.sign(self.y[idx])
                sgn = np.where(sgn == 0.0, 1.0, sgn)
                nvec = np.stack([-sgn * dbdx, sgn * np.ones_like(sgn), -sgn * dbdz])
                nvec = nvec / np.linalg.norm(nvec, axis=0)
                v = np.stack([self.vx[idx], self.vy[idx], self.vz[idx]])
                vn = (v * nvec).sum(axis=0)
                # position projection onto the surface (no tunnelling)
                corr = pen + 1e-9
                self.y[idx] -= nvec[1] * corr * 0.5
                self.x[idx] -= nvec[0] * corr * 0.5
                self.z[idx] = np.clip(self.z[idx] - nvec[2] * corr * 0.5, -2.0 * self.h, 0.5 * self.h)
                # normal impulse removes the approaching normal velocity
                Jn = -self.m[idx] * np.minimum(vn, 0.0)
                vt = v - vn * nvec
                tmag = np.linalg.norm(vt, axis=0)
                tmag_safe = np.where(tmag < 1e-9, 1e-9, tmag)
                Jt = np.minimum(cfg.mu_ice_hull * Jn, self.m[idx] * tmag)
                dv = (Jn * nvec - Jt * vt / tmag_safe) / self.m[idx]
                self.vx[idx] += dv[0]
                self.vy[idx] += dv[1]
                self.vz[idx] += dv[2]
                # equivalent force over the step + dissipated energy
                F = (Jn * nvec - Jt * vt / tmag_safe) / cfg.dt
                np.add.at(fx, idx, F[0])
                np.add.at(fy, idx, F[1])
                np.add.at(fz, idx, F[2])
                diss += float(np.sum(Jt * tmag) + np.sum(0.5 * self.m[idx] * np.minimum(vn, 0.0) ** 2))
                self._hull_impulse = (idx, F, Jn / cfg.dt, Jt / cfg.dt, vt, tmag, nvec)

        return fx, fy, fz, diss

    # ------------------------------------------------------------------ step
    def step(self) -> None:
        cfg = self.cfg
        if not hasattr(self, "_step_count"):
            self._step_count = 0
        if self._step_count % cfg.rebuild_every == 0:
            self._rebuild_pairs()
        self._step_count += 1

        # water drag
        uw_x, uw_y, uw_z = self._water_velocity(self.x, self.y, self.z)
        rvx = uw_x - self.vx
        rvy = uw_y - self.vy
        rvz = uw_z - self.vz
        speed = np.sqrt(rvx ** 2 + rvy ** 2 + rvz ** 2)
        a_front = self.d * self.h
        a_plan = self.area
        drag = (0.5 * cfg.rho_water
                * (cfg.drag_coeff_frontal * a_front + cfg.drag_coeff_planar * a_plan) * speed)
        fwx = drag * rvx
        fwy = drag * rvy
        fwz = drag * rvz

        # buoyancy + weight (vertical), with added-mass damping
        sub = np.clip(-(self.z - 0.5 * self.h) / self.h, 0.0, 1.0)
        fb = cfg.rho_water * 9.81 * a_plan * (sub * self.h) - self.m * 9.81
        fbz = fb - self.c_vert * self.vz

        # contacts (returns hull force too)
        self._hull_impulse = None
        fcx, fcy, fcz, diss_ii = self._contact_forces()

        # integrate (semi-implicit Euler)
        ax = (fwx + fcx) / self.m
        ay = (fwy + fcy) / self.m
        az = (fwz + fbz + fcz) / self.m
        self.vx += ax * cfg.dt
        self.vy += ay * cfg.dt
        self.vz += az * cfg.dt
        self.x += self.vx * cfg.dt
        self.y += self.vy * cfg.dt
        self.z += self.vz * cfg.dt
        self.z = np.clip(self.z, -2.0 * self.h, 0.5 * self.h)
        # numerical safety limiter: a contact can never eject a floe faster than a
        # few ship speeds; anything above is a penalty-contact transient
        vmax = cfg.max_velocity_factor * self.U
        sp = np.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)
        over = sp > vmax
        if np.any(over):
            sc = vmax / sp[over]
            self.vx[over] *= sc
            self.vy[over] *= sc
            self.vz[over] *= sc

        # ---- energy bookkeeping (all rates in W) --------------------------
        e = self.energy
        if self._hull_impulse is not None:
            idx, F, fn, ft, vt, tmag, nvec = self._hull_impulse
            e["P_hull_dot_v"] += float(np.sum(F * np.stack([self.vx[idx], self.vy[idx], self.vz[idx]])))
            e["P_hull_slip"] += float(np.sum(F[0] * (self.U - self.vx[idx])))
            e["P_hull_perp"] += float(np.sum(F[1] * self.vy[idx] + F[2] * self.vz[idx]))
            self.decomp_hist.append({
                "Fn_x": float(np.sum(fn * nvec[0])),
                "Ft_x": float(np.sum(-ft * vt[0] / np.where(tmag < 1e-9, 1e-9, tmag))),
                "n_contact": int(idx.size),
                "fn_total": float(np.sum(fn)),
            })
        e["P_water"] += float(np.sum(fwx * self.vx + fwy * self.vy + fwz * self.vz))
        e["D_contacts"] += diss_ii
        self._t_accum += cfg.dt

        # ---- lateral walls (confining channel) ---------------------------
        half = 0.5 * self.W
        for sign in (+1.0, -1.0):
            hit = sign * self.y > half - self.r
            if np.any(hit):
                self.y[hit] = sign * (half - self.r[hit])
                self.vy[hit] = 0.0

        # ---- x-recycling --------------------------------------------------
        out = self.x > self.Lx - self.r
        if np.any(out):
            self._ke_out += float(np.sum(0.5 * self.m[out]
                                         * (self.vx[out] ** 2 + self.vy[out] ** 2)))
            for i in np.nonzero(out)[0]:
                for _ in range(20):
                    yy = self._rng.uniform(-half + self.r[i], half - self.r[i])
                    xx = self.r[i] + self._rng.uniform(0.0, 3.0 * self.r[i])
                    dd = np.hypot(self.x - xx, self.y - yy)
                    if np.all(dd > (self.r + self.r[i]) * 0.95):
                        break
                self.x[i] = xx
                self.y[i] = yy
                self.vx[i] = self.U
                self.vy[i] = 0.0
                self.vz[i] = 0.0
                self.z[i] = -0.5 * self.h + self.h * cfg.rho_ice / cfg.rho_water
                self._ke_in += float(0.5 * self.m[i] * self.U ** 2)

        # resistance on the hull = -(hull force on the floes)_x
        if self._hull_impulse is not None:
            idx, F, _, _, _, _, _ = self._hull_impulse
            counted = self._counted[idx]
            self.R_hist.append(-float(np.sum(F[0][counted])))   # resistance = -F_hull_on_ice
            self.R_stern_hist.append(-float(np.sum(F[0][~counted])))
        else:
            self.R_stern_hist.append(0.0)

    def run(self, n_steps: Optional[int] = None, warmup_fraction: float = 0.4) -> Dict[str, object]:
        n_steps = n_steps or self.cfg.n_steps
        n_warm = int(warmup_fraction * n_steps)
        for k in range(n_steps):
            self.step()
            if k == n_warm:
                self.R_hist.clear()
                self.R_stern_hist.clear()
                self.decomp_hist.clear()
        return self.diagnostics()

    # ------------------------------------------------------------- diagnostics
    def diagnostics(self) -> Dict[str, object]:
        R = np.asarray(self.R_hist) if self.R_hist else np.zeros(1)
        dec = self.decomp_hist
        fric = np.array([d["Ft_x"] for d in dec]) if dec else np.zeros(1)
        pres = np.array([d["Fn_x"] for d in dec]) if dec else np.zeros(1)
        ncont = np.array([d["n_contact"] for d in dec]) if dec else np.zeros(1)
        return {
            "R_ice_mean": float(R.mean()),       # resistance on the hull (+x)
            "R_aft_diagnostic": float(np.mean(self.R_stern_hist)) if self.R_stern_hist else 0.0,
            "R_ice_RMS": float(np.sqrt((R ** 2).mean())),
            "R_ice_std": float(R.std()),
            "R_friction_x": float(-fric.mean()),
            "R_normal_x": float(-pres.mean()),
            "n_contacts_mean": float(ncont.mean()),
            "n_floes": int(self.n),
            "energy": dict(self.energy),
            "physical_time": float(self._t_accum),
        }

    def energy_audit(self) -> Dict[str, float]:
        """Close the steady resistance against the floe-phase power balance.

        Exact steady-state identity (all terms time-averaged):

            R_ice U = -P_water + D_contacts + P_hull_slip + P_hull_perp + KE_flux

        with ``P_hull_slip = sum F_hx (U - v_x)`` (hull sliding work) and
        ``P_hull_perp`` the lateral/vertical work of the hull on the floes.
        """
        e = dict(self.energy)
        e["P_hull_slip"] /= max(self._t_accum, 1e-30)
        e["P_hull_perp"] /= max(self._t_accum, 1e-30)
        e["P_hull_dot_v"] /= max(self._t_accum, 1e-30)
        e["P_water"] /= max(self._t_accum, 1e-30)
        e["D_contacts"] /= max(self._t_accum, 1e-30)
        ke_flux = (self._ke_out - self._ke_in) / max(self._t_accum, 1e-30)
        e["KE_flux"] = ke_flux
        R = float(np.mean(self.R_hist)) if self.R_hist else 0.0
        e["R_mean"] = R
        e["P_resistance"] = R * self.U
        e["P_balance"] = (-e["P_water"] + e["D_contacts"] + e["P_hull_slip"]
                          + e["P_hull_perp"] + ke_flux)
        denom = max(abs(e["P_resistance"]), 1e-30)
        e["energy_residual_rel"] = abs(e["P_balance"] - e["P_resistance"]) / denom
        return e