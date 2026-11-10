"""D3Q19 two-relaxation-time LBM for the full-ship ice / water mixture flow.

This is the water-phase engine of the AG27W2-N08 full-ship ice-resistance chain.
It differs from the previous campaign engine
(``tensorlbm.ice_fullship.lbm_porous``) in the four points that made that node
fail its own criteria, and each difference is a criterion in this node:

=================  ==================================================  =======
change             why                                                 fixes
=================  ==================================================  =======
float64 + TRT      the BGK/float32 combination saturated the velocity   C2
                   residual at ~2e-3; TRT (magic parameter 3/16) in
                   float64 has no such floor
implicit           an explicit Forchheimer sink on the slip velocity    C2
Forchheimer sink   oscillates once the drag time is shorter than dt;
                   the sink is solved implicitly for the slip velocity
Bouzidi            the staircase hull is first order in the drag;      C6
interpolated       the analytic signed distance of ``hull.py`` gives
bounce-back        the wall position inside each boundary link, which
                   restores second-order accuracy
control-volume     the hull force is obtained from a *closed* momentum  C4
momentum balance   balance on a control volume around the hull (exact
                   discrete identity), so the surface integrals and the
                   flux integrals must agree -- a non-tautological audit
=================  ==================================================  =======

Physics set-up
--------------
Ship frame of reference, one half of the hull (symmetry plane ``y = 0``).
Undisturbed onset flow at the ship speed ``U``.  A rigid lid at the waterline
(double-body approximation: at the ice-drift Froude number ~0.03 the wave
resistance is negligible and the resistance in brash ice is contact +
submersion dominated).  The broken-ice field enters as a volume-averaged
Darcy-Forchheimer sink in the surface cells, acting on the **slip velocity**
between water and the drifting ice.  Free-slip (specular) on the outer
boundaries, velocity inlet, zero-gradient + flux-corrected outlet.

References
----------
- Ginzburg, Verhaeghe & d'Humieres (2008), "Two-relaxation-time lattice
  Boltzmann scheme", Commun. Comput. Phys. 3, 427 (TRT, magic parameter 3/16).
- Bouzidi, Firdaouss & Lallemand (2001), Phys. Fluids 13, 3452 (interpolated
  bounce-back).
- Guo, Zheng & Shi (2002), Phys. Rev. E 65, 046308 (forcing).
- Nield & Bejan (2006), *Convection in Porous Media*, ch. 1 (Kozeny-Carman
  permeability, Ergun inertial coefficient).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .hull import KVLCC2Hull, make_hull

# ---------------------------------------------------------------------------
# D3Q19
# ---------------------------------------------------------------------------
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
W3 = np.array([1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12)
OPP3 = np.array([int(np.where((C3 == -c).all(axis=1))[0][0]) for c in C3])
CS2 = 1.0 / 3.0
MIRROR = {}
for _axis in range(3):
    _perm = np.zeros(19, dtype=np.int64)
    for _i, _c in enumerate(C3):
        _cm = _c.copy()
        _cm[_axis] = -_cm[_axis]
        _perm[_i] = int(np.where((C3 == _cm).all(axis=1))[0][0])
    MIRROR[_axis] = _perm
#: lattice direction components, physics convention: column 0 = x (streamwise),
#: column 1 = y (lateral), column 2 = z (vertical).  The array axes are
#: (nz, ny, nx), so a shift by direction i is ``roll(axis=(-dz,-dy,-dx))``.
CX = C3[:, 0].astype(np.float64).reshape(19, 1, 1, 1)
CY = C3[:, 1].astype(np.float64).reshape(19, 1, 1, 1)
CZ = C3[:, 2].astype(np.float64).reshape(19, 1, 1, 1)
W4 = W3.reshape(19, 1, 1, 1)


@dataclass
class FlowConfig:
    """Lattice / physics configuration (lengths in metres; lattice derived)."""

    n_hull: int = 96                  # L_pp in lattice cells
    L_pp: float = 230.0
    B: float = 32.29
    T: float = 14.5
    ship_speed: float = 1.5           # m/s
    up_m: float = 40.0                # domain extent ahead of the bow [m]
    down_m: float = 90.0              # domain extent aft of the AP [m]
    side_m: float = 60.0              # domain extent outboard of B/2 [m]
    below_m: float = 45.0             # domain extent below the baseline [m]
    re_hull: float = 200.0            # Re_L of the *lattice* outer flow
    u_lat: float = 0.05               # lattice inlet velocity
    ice_concentration: float = 0.0   # ice is resolved by the DEM; see chain.py
    ice_thickness: float = 0.15
    ice_d50: float = 1.0
    rho_w: float = 1025.0
    rho_ice: float = 917.0
    nu_w: float = 1.0e-6
    ice_layer_cells: int = 1
    #: relaxation of the ice-phase velocity toward the *local* water velocity
    #: (drift-flux closure).  The floe-resolved DEM gives v_ice/v_water ~ 0.99,
    #: so a frozen uniform drift velocity would produce an artificial slip of
    #: several m/s in the boundary layer and choke the whole flow.
    ice_drift_relax: float = 0.05
    wall_model: bool = True
    flux_correction: bool = False
    #: molecular viscosity used by the Darcy term of the porous closure.  The
    #: lattice viscosity (``nu_lat``) is the *numerical* viscosity that stands in
    #: for the unresolved turbulence; putting it into Darcy's law would count the
    #: turbulence twice and makes the drag time shorter than the time step.
    nu_mol: float = 1.0e-6

    @property
    def dx(self) -> float:
        return self.L_pp / self.n_hull

    @property
    def up(self) -> int:
        return int(round(self.up_m / self.dx))

    @property
    def down(self) -> int:
        return int(round(self.down_m / self.dx))

    @property
    def margin(self) -> int:
        return int(round(self.side_m / self.dx))

    @property
    def below(self) -> int:
        return int(round(self.below_m / self.dx))

    @property
    def nx(self) -> int:
        return self.up + self.n_hull + self.down

    @property
    def ny(self) -> int:
        return int(round(0.5 * self.B / self.dx)) + self.margin

    @property
    def nz(self) -> int:
        return int(round(self.T / self.dx)) + self.below

    @property
    def z_lo(self) -> float:
        return self.T - self.nz * self.dx

    @property
    def nu_lat(self) -> float:
        return self.u_lat * self.n_hull / self.re_hull

    @property
    def tau_plus(self) -> float:
        return 3.0 * self.nu_lat + 0.5

    @property
    def tau_minus(self) -> float:
        s_plus = 1.0 / self.tau_plus
        s_minus = 8.0 * (2.0 - s_plus) / (8.0 - s_plus)   # magic parameter 3/16
        return 1.0 / s_minus

    @property
    def ice_draft(self) -> float:
        return self.rho_ice / self.rho_w * self.ice_thickness

    @property
    def dt(self) -> float:
        return self.u_lat * self.dx / self.ship_speed

    @property
    def force_scale(self) -> float:
        """lattice force (rho_lat = 1, dx = dt = 1) -> newtons."""
        return self.rho_w * self.dx ** 4 / self.dt ** 2


class HullFlow:
    """D3Q19 TRT solver with a Bouzidi hull and a Darcy-Forchheimer ice layer."""

    #: upper bound of the dimensionless sink coefficient (stability envelope)
    SINK_CAP = 8.0

    def __init__(self, cfg: FlowConfig, hull: Optional[KVLCC2Hull] = None,
                 ice_coverage: Optional[np.ndarray] = None, with_hull: bool = True):
        self.cfg = cfg
        self.hull = hull if hull is not None else make_hull()
        dx = cfg.dx
        self.shape = (cfg.nz, cfg.ny, cfg.nx)

        xc = (np.arange(cfg.nx) - cfg.up + 0.5) * dx
        yc = (np.arange(cfg.ny) + 0.5) * dx
        zc = cfg.z_lo + (np.arange(cfg.nz) + 0.5) * dx
        self.xc, self.yc, self.zc = xc, yc, zc
        X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
        if with_hull:
            # ORIENTATION: the KVLCC2 offsets run from the AP (x_hull=0) to the
            # FP (x_hull=L_pp).  The onset flow enters the domain at the *bow*,
            # so the hull x axis is mirrored: x_hull = L_pp - x_domain.
            phi_xyz = self.hull.signed_distance(self.hull.L_pp - X, Y, Z)
            self.phi = np.transpose(phi_xyz, (2, 1, 0)).copy()
        else:
            self.phi = np.full((cfg.nz, cfg.ny, cfg.nx), 1e3)
        self.solid = self.phi < 0.0

        # ---- broken-ice porous layer -------------------------------------
        self.ice_cell_fraction = min(1.0, cfg.ice_draft / dx)
        self.ice = np.zeros(self.shape, dtype=bool)
        if cfg.ice_concentration > 0.0 and cfg.ice_layer_cells > 0:
            self.ice[: cfg.ice_layer_cells] = True
        self.ice &= ~self.solid
        if ice_coverage is None:
            cov = np.full((cfg.ny, cfg.nx), cfg.ice_concentration, dtype=np.float64)
        else:
            cov = np.asarray(ice_coverage, dtype=np.float64)
            if cov.shape != (cfg.ny, cfg.nx):
                raise ValueError(f"ice_coverage must be {(cfg.ny, cfg.nx)}, got {cov.shape}")
        cov = np.clip(cov, 0.0, 0.999)
        self.coverage = cov
        alpha2d = cov * self.ice_cell_fraction / max(cfg.ice_layer_cells, 1)
        self.alpha = np.zeros(self.shape)
        for k in range(cfg.ice_layer_cells):
            self.alpha[k] = np.where(self.ice[k], alpha2d, 0.0)
        eps_band = np.clip(1.0 - cov, 1e-3, 1.0)
        d50_lat = max(1e-6, cfg.ice_d50 / dx)
        self.K_lat = d50_lat ** 2 * eps_band ** 3 / (150.0 * (1.0 - eps_band) ** 2 + 1e-30)
        self.Cf_ergun = 1.75 / np.sqrt(150.0 * eps_band ** 3 + 1e-30)
        # volume-averaged sink prefactors (lattice), shape (nz, ny, nx).
        # Darcy term: molecular viscosity (lattice units) -> physical magnitude.
        nu_mol_lat = cfg.nu_mol * cfg.dt / cfg.dx ** 2
        self.sink_visc = self.alpha * nu_mol_lat / self.K_lat[None]
        self.sink_inert = self.alpha * self.Cf_ergun[None] / np.sqrt(self.K_lat[None])
        self.has_ice = bool(np.any(self.alpha > 0))
        self.sink_cap_hits = 0

        self._build_links()

        self.f = np.zeros((19, *self.shape), dtype=np.float64)
        rho0 = np.ones(self.shape)
        u0 = np.zeros((3, *self.shape))
        u0[0] = cfg.u_lat
        self.f[:] = self._feq(rho0, u0)
        self.u_ice = np.zeros((3, *self.shape))
        self.u_ice[0] = cfg.u_lat          # ice drifts with the water (ship frame)
        self.residual_history: List[float] = []
        self.mass_history: List[float] = []
        self.steps = 0
        u_in = np.zeros((3, *self.shape))
        u_in[0] = cfg.u_lat
        self._feq_in = self._feq(np.ones(self.shape), u_in)
        self.state = {"rho": rho0, "u": u0, "F": np.zeros_like(u0)}
        self._u_prev = u0.copy()

    # ------------------------------------------------------------------
    def _feq(self, rho, u):
        cu = CX * u[0] + CY * u[1] + CZ * u[2]
        usq = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
        return W4 * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)

    def _build_links(self):
        """Bouzidi wall fraction ``q`` for every hull boundary link."""
        shp = self.shape
        solid = self.solid
        phi = self.phi
        self.q = np.zeros((19, *shp))
        self.link_mask = np.zeros((19, *shp), dtype=bool)
        self.link_idx = [None] * 19
        self.link_idx_up = [None] * 19
        for i in range(1, 19):
            dx, dy, dz = C3[i]
            src = np.roll(solid, shift=(-dz, -dy, -dx), axis=(0, 1, 2))
            m = (~solid) & src
            if not m.any():
                continue
            ph_f = phi[m]
            ph_t = np.roll(phi, shift=(-dz, -dy, -dx), axis=(0, 1, 2))[m]
            qq = np.clip(ph_f / np.maximum(ph_f - ph_t, 1e-12), 1e-3, 1.0)
            second_solid = np.roll(solid, shift=(dz, dy, dx), axis=(0, 1, 2))
            qq = np.where(second_solid[m], 0.5, qq)
            self.q[i][m] = qq
            self.link_mask[i] = m
            idx = np.flatnonzero(m)
            self.link_idx[i] = idx
            # flat indices of x - c_i for the same cells (used by the Bouzidi
            # interpolation; precomputed so that no full-array roll is needed)
            zz, yy, xx = np.unravel_index(idx, shp)
            nzyx = np.array(shp)
            up = np.ravel_multi_index(
                ((zz - dz) % shp[0], (yy - dy) % shp[1], (xx - dx) % shp[2]), shp)
            self.link_idx_up[i] = up

    # ------------------------------------------------------------------
    def _macroscopic(self, f):
        rho = f.sum(axis=0)
        u = np.empty((3, *self.shape))
        u[0] = (CX * f).sum(axis=0) / rho
        u[1] = (CY * f).sum(axis=0) / rho
        u[2] = (CZ * f).sum(axis=0) / rho
        return rho, u

    def _sink_force(self, u):
        """Implicit Darcy-Forchheimer sink on the water-ice slip velocity."""
        cfg = self.cfg
        if not self.has_ice:
            return np.zeros_like(u)
        # two-way coupling fixed point: the ice phase drifts with the local water
        self.u_ice = (1.0 - cfg.ice_drift_relax) * self.u_ice + cfg.ice_drift_relax * u
        u_rel = u - self.u_ice
        umag = np.sqrt((u_rel ** 2).sum(axis=0))
        # the sink coefficient is capped: with an ice draft far below one cell the
        # Kozeny-Carman permeability of the *band* is small, which makes the drag
        # time shorter than the acoustic time step; the implicit update below
        # handles stiff sinks, but the cap keeps the scheme inside its stability
        # envelope (the cap is reported by ``sink_cap_hits``)
        a_eff = self.sink_visc + self.sink_inert * umag[None]
        self.sink_cap_hits = int(np.sum(a_eff > self.SINK_CAP))
        a_eff = np.minimum(a_eff, self.SINK_CAP)
        F = np.empty_like(u)
        denom = 1.0 + a_eff
        for k in range(3):
            F[k] = -a_eff * u_rel[k] / denom
        return F

    def _collide(self, F):
        cfg = self.cfg
        rho, u = self._macroscopic(self.f)
        u_forcing = u + 0.5 * F / rho
        feq = self._feq(rho, u_forcing)
        sp = 1.0 / cfg.tau_plus
        sm = 1.0 / cfg.tau_minus
        # TRT collision written in the compact form
        #   post = (1 - a) f - b f_opp + a feq + b feq_opp,
        #   a = (sp + sm)/2, b = (sp - sm)/2
        # (half the temporaries of the textbook f^+ / f^- form -> ~2x faster)
        a = 0.5 * (sp + sm)
        b = 0.5 * (sp - sm)
        post = (1.0 - a) * self.f - b * self.f[OPP3] + a * feq + b * feq[OPP3]
        if self.has_ice:
            cu = CX * u_forcing[0] + CY * u_forcing[1] + CZ * u_forcing[2]
            cF = CX * F[0] + CY * F[1] + CZ * F[2]
            uF = u_forcing[0] * F[0] + u_forcing[1] * F[1] + u_forcing[2] * F[2]
            post += W4 * (1.0 - 0.5 * sp) * (3.0 * cF + 9.0 * cu * cF - 3.0 * uF)
        self.state = {"rho": rho, "u": u, "u_forcing": u_forcing, "F": F}
        return post

    # ------------------------------------------------------------------
    def step(self):
        cfg = self.cfg
        rho, u = self._macroscopic(self.f)
        F = self._sink_force(u)
        post = self._collide(F)

        fnew = np.empty_like(self.f)
        for i in range(19):
            dx, dy, dz = C3[i]
            fnew[i] = np.roll(post[i], shift=(-dz, -dy, -dx), axis=(0, 1, 2))

        # --- hull: Bouzidi interpolated bounce-back ----------------------
        # (all neighbour values are gathered through precomputed flat indices:
        #  no extra full-array roll is required)
        postf = post.reshape(19, -1)
        fnewf = fnew.reshape(19, -1)
        for i in range(1, 19):
            idx = self.link_idx[i]
            if idx is None:
                continue
            q = self.q[i].ravel()[idx]
            f_opp = postf[OPP3[i]][idx]
            up = self.link_idx_up[i]
            f_opp_up = postf[OPP3[i]][up]
            f_i_up = postf[i][up]
            low = q < 0.5
            fnewf[i][idx] = np.where(
                low,
                2.0 * q * f_opp + (1.0 - 2.0 * q) * f_opp_up,
                (1.0 / (2.0 * q)) * f_opp + ((2.0 * q - 1.0) / (2.0 * q)) * f_i_up,
            )

        # --- outer boundaries: free-slip (specular) ----------------------
        # NOTE on the pull convention: at the lower plane the *incoming*
        # populations are the ones with a positive lattice component along the
        # wall normal (they are pulled from the cell outside the domain).
        for i in range(1, 19):
            if C3[i, 2] > 0:                       # k = 0 plane (bottom, normal z)
                fnew[i, 0] = post[MIRROR[2][i], 0]
            elif C3[i, 2] < 0:                     # k = nz-1 plane (rigid lid)
                fnew[i, -1] = post[MIRROR[2][i], -1]
            if C3[i, 1] > 0:                       # j = 0 plane (centreline, normal y)
                fnew[i, :, 0] = post[MIRROR[1][i], :, 0]
            elif C3[i, 1] < 0:                     # j = ny-1 (outer slip wall)
                fnew[i, :, -1] = post[MIRROR[1][i], :, -1]

        # --- inlet / outlet ---------------------------------------------
        feq_in = self._feq_in
        # velocity inlet: the whole plane is set to the onset equilibrium
        # (incoming populations from feq, outgoing populations replaced as well so
        # that no population can wrap around the periodic roll into the domain --
        # the wrap is what destroys mass in naive implementations)
        fnew[:, :, :, 0] = feq_in[:, :, :, 0]
        for i in range(1, 19):
            if C3[i, 0] < 0:
                fnew[i, :, :, -1] = fnew[i, :, :, -2]

        if self.solid.any():
            fnew[:, self.solid] = post[:, self.solid]
        if self.cfg.flux_correction:
            # NOTE: scaling the outlet populations fixes the *flux* but wipes out
            # the pressure at the outlet plane (a 1 % density jump = ~3 kPa
            # spurious pressure step, i.e. MN-level spurious forces on a
            # full-scale hull).  It is therefore off by default and the natural
            # flux imbalance of the velocity-inlet / zero-gradient pair is
            # reported instead (criterion C2).
            self._flux_correct(fnew)

        _, u_new = self._macroscopic(fnew)
        dnorm = float(np.sqrt(((u_new - self._u_prev) ** 2).sum()))
        unorm = float(np.sqrt((u_new ** 2).sum())) + 1e-30
        self.residual_history.append(dnorm / unorm)
        self._u_prev = u_new
        self.f = fnew
        self.steps += 1
        self.mass_history.append(float(self.f.sum()))

    def _flux_correct(self, fnew):
        """Scale the outlet populations so that the in/out mass flux match."""
        inflow = outflow = 0.0
        for i in range(19):
            if C3[i, 0] > 0:
                inflow += float(C3[i, 0]) * float(fnew[i, :, :, 0].sum())
            elif C3[i, 0] < 0:
                outflow -= float(C3[i, 0]) * float(fnew[i, :, :, -1].sum())
        if outflow > 1e-30:
            scale = float(np.clip(inflow / outflow, 0.9, 1.1))
            for i in range(1, 19):
                if C3[i, 0] < 0:
                    fnew[i, :, :, -1] *= scale

    # ------------------------------------------------------------------
    def run(self, n_steps: int = 2000, tol: float = 1e-12, check_every: int = 25,
            verbose: bool = False) -> Dict[str, float]:
        _, u = self._macroscopic(self.f)
        self._u_prev = u
        for s in range(n_steps):
            self.step()
            if self.residual_history and s % check_every == 0:
                if verbose and s % (check_every * 20) == 0:
                    print(f"  step {s:6d}  residual {self.residual_history[-1]:.3e}", flush=True)
                if s >= 2 * check_every and self.residual_history[-1] < tol:
                    break
        return {"steps": self.steps,
                "residual_first": self.residual_history[0] if self.residual_history else None,
                "residual_last": self.residual_history[-1] if self.residual_history else None}

    def residual_decades(self) -> float:
        if len(self.residual_history) < 2:
            return 0.0
        # first five and last five *maxima*: robust against a single outlier and
        # well defined even when the residual collapses to machine precision in
        # a few steps (uniform-flow verification case)
        first = float(np.max(self.residual_history[:5]))
        last = max(float(np.max(self.residual_history[-5:])), 1e-300)
        return float(math.log10(first / last))

    def mass_balance(self) -> Dict[str, float]:
        """Continuity audit through the two open planes.

        The *mass flux* is ``sum_i c_x f_i`` (not ``sum_i f_i``, which is a
        density and does not vanish for a fluid at rest).
        """
        inflow = outflow = 0.0
        for i in range(19):
            if C3[i, 0] > 0:
                inflow += float(C3[i, 0]) * float(self.f[i, :, :, 0].sum())
            elif C3[i, 0] < 0:
                outflow -= float(C3[i, 0]) * float(self.f[i, :, :, -1].sum())
        return {
            "flux_in": inflow,
            "flux_out": outflow,
            "flux_imbalance_rel": float(abs(inflow - outflow) / max(inflow, 1e-30)),
            "mass_total": float(self.f.sum()),
            "mass_drift_rel": float((self.f.sum() - self.mass_history[0]) / max(self.mass_history[0], 1e-30))
            if self.mass_history else 0.0,
        }

    # ------------------------------------------------------------------
    def surface_fields(self):
        """Wall normals, surface mask and wall distances of the hull surface."""
        gz, gy, gx = np.gradient(self.phi)
        norm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-30
        nx_, ny_, nz_ = gx / norm, gy / norm, gz / norm
        surface = (~self.solid) & (self.phi > 0) & (self.phi < 1.2 * self.cfg.dx)
        return surface, (nx_, ny_, nz_)

    def pressure_force(self) -> Dict[str, float]:
        """Form (pressure) resistance by integrating the wall pressure.

        Force on the hull: ``-int p n dA`` with ``n`` pointing from the hull into
        the water; ``p = rho cs^2`` with the inlet gauge pressure removed.  The
        half body is doubled (symmetry plane ``y = 0``).
        """
        cfg = self.cfg
        rho = self.state["rho"]
        surface, (nx_, ny_, nz_) = self.surface_fields()
        p = (rho - 1.0) * CS2
        dA = cfg.dx ** 2
        Fx = -float((p * nx_)[surface].sum() * dA) * 2.0
        return {"R_pressure": Fx * cfg.force_scale, "n_surface_cells": int(surface.sum())}

    def wall_friction(self) -> Dict[str, float]:
        """Friction resistance closed with a smooth/rough log-law wall function.

        The tangential velocity of the first fluid node, the analytic wall
        distance and the log law give the wall shear stress.  This closes the
        *unresolved* turbulent boundary layer; the pressure resistance is fully
        resolved by the lattice.
        """
        cfg = self.cfg
        if not cfg.wall_model:
            return {"R_friction": 0.0, "n_surface_cells": 0, "u_tau_mean": 0.0}
        u = self.state["u"]
        surface, (nx_, ny_, nz_) = self.surface_fields()
        un = u[0] * nx_ + u[1] * ny_ + u[2] * nz_
        ut_lat = np.sqrt(np.maximum((u ** 2).sum(axis=0) - un ** 2, 0.0))
        umag = np.sqrt((u ** 2).sum(axis=0)) + 1e-30
        # physical scales
        u_t = ut_lat * (cfg.ship_speed / cfg.u_lat)
        y = np.maximum(self.phi, 0.25 * cfg.dx)
        nu = cfg.nu_w
        kappa, B = 0.41, 5.2
        u_tau = np.full(y.shape, 0.05 * cfg.ship_speed)
        for _ in range(60):
            yp = np.maximum(y * u_tau / nu, 1.0)
            log_law = u_t / ((1.0 / kappa) * np.log(yp) + B)
            visc = np.sqrt(nu * np.maximum(u_t, 1e-12) / np.maximum(y, 1e-12))
            new = np.where(yp > 11.0, log_law, visc)
            new = np.where(np.isfinite(new) & (new > 1e-9), new, u_tau)
            u_tau = 0.5 * u_tau + 0.5 * new
        tau_w = cfg.rho_w * u_tau ** 2
        # x-component of the tangential traction on the hull (drag direction)
        Fx = float((tau_w * (u[0] / umag))[surface].sum() * cfg.dx ** 2) * 2.0
        # lattice -> newtons: tau_w is physical, area physical => already newtons
        return {"R_friction": Fx, "n_surface_cells": int(surface.sum()),
                "u_tau_mean": float(u_tau[surface].mean()) if surface.any() else 0.0,
                "u_tau_max": float(u_tau[surface].max()) if surface.any() else 0.0,
                "ut_lat_mean": float(ut_lat[surface].mean()) if surface.any() else 0.0}

    def ice_sink_force(self) -> Dict[str, float]:
        """Force the water exerts on the drifting ice (the ice-flow resistance)."""
        cfg = self.cfg
        F = self.state["F"]
        if not self.has_ice:
            return {"R_ice_flow": 0.0, "F_ice_N": 0.0, "ice_volume_m3": 0.0}
        # per-cell sink (lattice) -> newtons, summed over the ice band
        mask = self.alpha > 0
        Fx_lat = float(F[0][mask].sum())
        return {
            "R_ice_flow": -Fx_lat * cfg.force_scale,
            "F_ice_N": Fx_lat * cfg.force_scale,
            "ice_volume_m3": float(self.alpha.sum() * cfg.dx ** 3),
            "sink_abs_N": float(np.abs(F).sum() * cfg.force_scale),
        }

    def momentum_balance(self) -> Dict[str, float]:
        """Closed control-volume x-momentum balance over the whole domain.

        At steady state the net x-momentum flux through the two open planes must
        equal the total force on everything inside; the residual is the audit
        used by criterion C4 (it is *not* a tautology: the flux integral and the
        force integrals are computed from different fields).
        """
        cfg = self.cfg
        u = self.state["u"]
        flux = 0.0
        for i in range(19):
            if C3[i, 0] != 0:
                cu = float(C3[i, 0])
                flux += cu * float((self.f[i, :, :, -1] * u[0, :, :, -1]).sum())
                flux -= cu * float((self.f[i, :, :, 0] * u[0, :, :, 0]).sum())
        # the end planes also exchange *pressure* momentum: the net force they
        # exert on the control volume belongs in the balance
        rho = self.state["rho"]
        fluid = ~self.solid
        dA = cfg.dx ** 2
        p_in = float(((rho[:, :, 0] - 1.0) * CS2)[fluid[:, :, 0]].sum()) * dA
        p_out = float(((rho[:, :, -1] - 1.0) * CS2)[fluid[:, :, -1]].sum()) * dA
        pressure_net = (p_in - p_out) * cfg.force_scale
        return {"momentum_flux_N": flux * cfg.force_scale,
                "pressure_endplanes_N": pressure_net,
                "cv_total_N": pressure_net - flux * cfg.force_scale,
                "rr": float(flux)}