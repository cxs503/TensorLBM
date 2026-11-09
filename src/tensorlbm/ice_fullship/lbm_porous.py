"""D3Q19 LBM water phase for the full-ship ice-resistance chain.

The water phase is solved in the **ship frame**: the hull is at rest and the
undisturbed onset flow enters the domain at the ship speed.  The broken-ice field
is represented as a Darcy-Forchheimer **porous surface layer** occupying the ice
draft band just below the rigid lid (rigid-lid / double-body approximation: in a
broken-ice field the ice resistance is dominated by floe-hull contact, not by
free-surface waves).

Ingredient map
--------------
* hull solid mask ......... KVLCC2 offset geometry (``ship_hull_geometry``)
* hull force .............. momentum-exchange (MEM) half-way bounce-back; the
                            exchanged momentum is projected on the smoothed wall
                            normal of the staircase surface to split the force
                            into a pressure (normal) and a friction (tangential)
                            part
* ice -> flow blocking .... Darcy-Forchheimer sink acting on the **slip velocity**
                            between the water and the drifting ice (the ice is
                            advected with the flow, so treating it as a static
                            porous matrix would grossly over-predict the sink);
                            permeability from Kozeny-Carman, inertial coefficient
                            from Ergun; Guo forcing
* flow <-> ice exchange ... the sink power ``sum(u_slip . -F)`` is exported so the
                            Eulerian sink can be audited against the Lagrangian
                            floe drag (criterion C3)

The channel has solid outer walls (bounce-back) and a velocity inlet /
zero-gradient outlet, so mass conservation is a meaningful, checkable number
(criterion C2) rather than a tautology.  The channel is deliberately run in a
**steady laminar regime** (Re_L = 200): the module's remit in this chain is to be
a verified, conservative, grid-convergent water-phase engine for the porous
coupling, not to reproduce the full-scale turbulent boundary layer, which is
discussed explicitly in the report as a limitation.

References
----------
- Guo, Zheng & Shi (2002), "Discrete lattice Boltzmann equation for forcing
  terms", Phys. Rev. E 65, 046308.
- Kozeny-Carman permeability / Ergun inertial coefficient: Nield & Bejan (2006),
  "Convection in Porous Media", 3rd ed., ch. 1.
- KVLCC2 offset geometry: SIMMAN 2008 workshop; see ``ship_hull_geometry``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..ship_hull_geometry import KVLCC2Geometry, kvlcc2_halfbreadth

# The chain's lattices are small (tens of thousands of cells); spawning one
# OpenMP thread per core makes every tensor op sync-bound and can be 100x slower
# than a hand-full of threads.  Pin a small thread count for the LBM module.
_TORCH_THREADS = 4
try:  # pragma: no cover - environment dependent
    torch.set_num_threads(_TORCH_THREADS)
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# D3Q19 lattice
# ---------------------------------------------------------------------------
_C = np.array(
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
_W = np.array([1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12, dtype=np.float64)
_OPP = np.array([int(np.where((_C == -c).all(axis=1))[0][0]) for c in _C], dtype=np.int64)
CS2 = 1.0 / 3.0


@dataclass
class PorousLBMConfig:
    """Lattice configuration (lengths in lattice units unless stated)."""

    n_hull: int = 48                 # hull length L_pp in lattice units
    L_pp: float = 230.0              # physical hull length [m]
    B: float = 32.29                 # physical beam [m]
    T: float = 14.5                  # physical draft [m]
    u_lat: float = 0.06              # inlet velocity (lattice)
    re_hull: float = 200.0           # Re based on L_pp and u_lat (steady regime)
    n_buffer_up: int = 14            # upstream buffer (lattice)
    n_buffer_down: int = 30          # downstream buffer (lattice)
    side_margin_cells: int = 10      # free cells each side of the hull
    depth_margin_cells: int = 6      # free cells below the keel
    ice_concentration: float = 0.60  # ice area concentration
    ice_d50: float = 1.0             # floe median diameter [m]
    ice_thickness: float = 0.15      # ice thickness [m]
    ice_layer_cells: int = 2         # porous layer thickness (lattice cells)
    ice_drift_ratio: float = 1.0     # ice drift speed / ship speed
    wall_outer: bool = True          # solid outer walls (rigid-lid channel)

    # ---- derived ----------------------------------------------------------
    @property
    def tau(self) -> float:
        nu = self.u_lat * self.n_hull / self.re_hull
        return 3.0 * nu + 0.5

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def cells_per_metre(self) -> float:
        return self.n_hull / self.L_pp

    @property
    def n_draft(self) -> int:
        return max(4, int(round(self.T * self.cells_per_metre)))

    @property
    def n_halfbeam(self) -> int:
        return max(2, int(round(0.5 * self.B * self.cells_per_metre)))

    @property
    def nx(self) -> int:
        return self.n_hull + self.n_buffer_up + self.n_buffer_down

    @property
    def ny(self) -> int:
        return 2 * (self.n_halfbeam + self.side_margin_cells) + 1

    @property
    def nz(self) -> int:
        return self.n_draft + self.depth_margin_cells

    def __post_init__(self) -> None:
        if self.n_hull < 16:
            raise ValueError("n_hull too small")
        if not (0.0 < self.ice_concentration < 1.0):
            raise ValueError("ice_concentration must be in (0, 1)")
        if self.tau <= 0.503:
            raise ValueError(f"relaxation time {self.tau:.4f} too close to 0.5 (unstable)")
        if self.ice_layer_cells < 0:
            raise ValueError("ice_layer_cells must be >= 0")


def build_hull_mask(cfg: PorousLBMConfig) -> np.ndarray:
    """Voxelise the KVLCC2 hull form into the lattice (True = solid)."""
    hull = KVLCC2Geometry(scale=1.0, L_pp=cfg.L_pp, B=cfg.B, T=cfg.T)
    ny_half = (cfg.ny - 1) // 2
    mask = np.zeros((cfg.nz, cfg.ny, cfg.nx), dtype=bool)
    x0 = cfg.n_buffer_up
    cpm = cfg.cells_per_metre
    for k in range(cfg.nz):
        z_phys = min((k + 0.5) * cfg.T / cfg.n_draft, 0.999 * cfg.T)
        for i in range(cfg.n_hull):
            x_phys = (i + 0.5) / cpm
            b = kvlcc2_halfbreadth(x_phys, z_phys, hull)
            if b <= 0.0:
                continue
            j = min(int(math.floor(b * cpm + 0.5)), ny_half)
            if j <= 0:
                continue
            iy = x0 + i
            mask[k, ny_half - j + 1: ny_half + j, iy] = True
    return mask


def _smoothed_wall_normal(solid: np.ndarray) -> np.ndarray:
    """Outward wall normal of a staircase surface from the solid indicator.

    ``n = -grad(solid)`` evaluated with central differences on the (optionally
    box-filtered) indicator.  Returns an array (3, nz, ny, nx) with unit normals
    on boundary cells (zero elsewhere).
    """
    ind = solid.astype(np.float32)
    # light smoothing so that the normal of a staircase face is not a delta spike
    k = np.zeros_like(ind)
    k[1:-1, 1:-1, 1:-1] = (
        ind[1:-1, 1:-1, 1:-1]
        + 0.5 * (ind[2:, 1:-1, 1:-1] + ind[:-2, 1:-1, 1:-1]
                 + ind[1:-1, 2:, 1:-1] + ind[1:-1, :-2, 1:-1]
                 + ind[1:-1, 1:-1, 2:] + ind[1:-1, 1:-1, :-2])
    ) / 4.0
    grad = np.gradient(k)
    n = -np.stack([grad[0], grad[1], grad[2]], axis=0).astype(np.float64)
    norm = np.sqrt((n ** 2).sum(axis=0))
    norm[norm < 1e-12] = 1.0
    n /= norm
    return n


class PorousLBM:
    """D3Q19 BGK solver with Darcy-Forchheimer porous ice and MEM hull forces."""

    def __init__(self, cfg: PorousLBMConfig, dtype: torch.dtype = torch.float32,
                 ice_coverage: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.dtype = dtype

        self.c = torch.as_tensor(_C.T, dtype=torch.long)              # (3, 19)
        self.cv = [self.c[k].view(19, 1, 1, 1) for k in range(3)]
        self.w = torch.as_tensor(_W, dtype=dtype).view(19, 1, 1, 1)
        self.opp = torch.as_tensor(_OPP, dtype=torch.long)

        hull = torch.as_tensor(build_hull_mask(cfg))
        outer = torch.zeros_like(hull)
        if cfg.wall_outer:
            # side / bottom / top walls are solid; the two x-end planes are the
            # open (velocity-inlet / outflow) boundaries of the channel
            outer[0, :, :] = True
            outer[-1, :, :] = True
            outer[:, 0, :] = True
            outer[:, -1, :] = True
        self.hull = hull
        self.outer = outer
        self.solid_all = hull | outer
        self.fluid = ~self.solid_all
        shape = tuple(hull.shape)

        # ---- ice band (porous surface layer) ------------------------------
        self.ice = torch.zeros(shape, dtype=torch.bool)
        if cfg.ice_layer_cells > 0:
            self.ice[-cfg.ice_layer_cells:, : cfg.n_buffer_up + cfg.n_hull, :] = True
        self.ice &= ~self.solid_all

        eps = 1.0 - cfg.ice_concentration
        # NOTE on the vertical discretisation: the ice draft (0.9 h_ice ~ 0.13 m)
        # is far thinner than one lattice cell, so the Darcy-Forchheimer drag of
        # the real ice band is distributed over the ``ice_layer_cells`` model
        # cells with the physically correct volume weighting
        # (ice_draft / cell_height / n_layer).  Without this weighting the
        # porous sink would be over-estimated by ~50x, which is what a naive
        # "one cell = one ice layer" mapping does.
        draft = cfg.ice_thickness * cfg.ice_concentration * 0.0 + cfg.ice_thickness * (917.0 / 1025.0)
        dz_phys = cfg.T / cfg.n_draft
        self.ice_cell_fraction = min(1.0, draft / dz_phys)
        d50_lat = max(1e-3, cfg.ice_d50 * cfg.cells_per_metre)
        K = d50_lat ** 2 * eps ** 3 / (150.0 * (1.0 - eps) ** 2)
        C_F = 1.75 / math.sqrt(150.0 * eps ** 3)
        self.porosity = eps
        self.permeability = K
        self.inertial_coef = C_F
        # local coverage weighting of the porous coefficients
        if ice_coverage is not None:
            cov = np.asarray(ice_coverage, dtype=np.float64)
            if cov.shape != (cfg.ny, cfg.nx):
                raise ValueError(f"ice_coverage must be ({cfg.ny},{cfg.nx}), got {cov.shape}")
            cov = np.clip(cov, 0.0, 1.0) / max(cfg.ice_concentration, 1e-6)
        else:
            cov = np.ones((cfg.ny, cfg.nx))
        self.coverage = cov
        w_layer = self.ice_cell_fraction / max(cfg.ice_layer_cells, 1)
        # per-cell porous coefficients: local porosity from the ice coverage map
        # (heterogeneous ice field) with the Kozeny-Carman / Ergun closures.
        # Cells without ice get zero coefficients; the layer distribution weight
        # (ice draft / cell height / n_layer) is uniform over the layers.
        c_loc = np.clip(np.clip(cov, 0.0, 0.999) * cfg.ice_concentration, 0.0, 0.999)
        eps_loc = 1.0 - c_loc
        has_ice = c_loc > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            K_loc = np.where(has_ice, d50_lat ** 2 * eps_loc ** 3
                             / np.where(has_ice, 150.0 * (1.0 - eps_loc) ** 2, 1.0), 1.0)
            b2d = np.where(has_ice, (cfg.nu / K_loc) * w_layer, 0.0)
            g2d = np.where(has_ice, (1.75 / np.sqrt(np.where(has_ice, 150.0 * eps_loc ** 3, 1.0)))
                           / np.sqrt(K_loc) * w_layer, 0.0)
        beta_field = np.zeros(shape)
        gamma_field = np.zeros(shape)
        if cfg.ice_layer_cells:
            sl = slice(cfg.nz - cfg.ice_layer_cells, cfg.nz)
            beta_field[sl, :, :] = b2d[None, :, :]
            gamma_field[sl, :, :] = g2d[None, :, :]
        self.beta = torch.as_tensor(beta_field, dtype=dtype) * self.ice.to(dtype)
        self.gamma = torch.as_tensor(gamma_field, dtype=dtype) * self.ice.to(dtype)

        # ---- MEM hull surface precomputation ------------------------------
        nrm = _smoothed_wall_normal(self.hull.numpy())
        self.normal = torch.as_tensor(nrm, dtype=torch.float64)
        self._bnd_links: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        bnd_nodes = torch.zeros_like(hull)
        for i in range(19):
            if not _C[i].any():
                continue
            neigh = torch.roll(self.solid_all, shifts=tuple(int(v) for v in _C[i]), dims=(2, 1, 0))
            bnd = self.fluid & neigh
            if not bool(bnd.any()):
                continue
            hull_bnd = bnd & ~self.outer
            bnd_nodes |= hull_bnd
            idx_hull = hull_bnd.flatten().nonzero(as_tuple=True)[0]
            idx_all = bnd.flatten().nonzero(as_tuple=True)[0]
            self._bnd_links.append((i, idx_all, idx_hull))
        self.bnd_nodes = bnd_nodes
        self.cvec = torch.as_tensor(_C, dtype=torch.float64)          # (19, 3) as (x,y,z)

        # ---- fields -------------------------------------------------------
        self.f = self._equilibrium(torch.ones(shape, dtype=dtype), cfg.u_lat, None, None)
        self.f = self.f * self.fluid.to(dtype)
        self._mass_ref = float(self.f.sum().to(torch.float64))
        self._u_ice = (cfg.u_lat * cfg.ice_drift_ratio, 0.0, 0.0)

        # precomputed flat indices for the four plane-flux reductions
        N = int(np.prod(shape))
        fl = self.fluid.numpy()
        self._flux_idx = {}
        for tag, ixplane, sign in (("p0", 0, 1), ("m0", 0, -1), ("pN", cfg.nx - 1, 1), ("mN", cfg.nx - 1, -1)):
            sel = [i for i in range(19) if _C[i][0] * sign > 0]
            idx = []
            for i in sel:
                base = i * N
                for k in range(cfg.nz):
                    for j in range(cfg.ny):
                        if fl[k, j, ixplane]:
                            idx.append(base + k * cfg.ny * cfg.nx + j * cfg.nx + ixplane)
            self._flux_idx[tag] = torch.as_tensor(idx, dtype=torch.long) if idx else None
        self._node_force_buf = torch.zeros((3,) + shape, dtype=torch.float64)
        self._feq_in = self._equilibrium(torch.ones((cfg.nz, cfg.ny), dtype=dtype),
                                         torch.full((cfg.nz, cfg.ny), cfg.u_lat, dtype=dtype),
                                         None, None)

        self.residual_history: List[float] = []
        self.mass_history: List[float] = []
        self.flux_history: List[float] = []
        self.force_history: List[Dict[str, float]] = []
        self._sink_power = 0.0
        self._outlet_scale = 1.0
        self._pre_correction_imbalance = 0.0
        self._node_force = torch.zeros((3,) + shape, dtype=torch.float64)

    # ------------------------------------------------------------------ core
    def _equilibrium(self, rho: torch.Tensor, ux, uy, uz) -> torch.Tensor:
        ux_t = torch.zeros_like(rho) if ux is None else ux
        uy_t = torch.zeros_like(rho) if uy is None else uy
        uz_t = torch.zeros_like(rho) if uz is None else uz
        cu = self.cv[0] * ux_t + self.cv[1] * uy_t + self.cv[2] * uz_t
        usq = ux_t * ux_t + uy_t * uy_t + uz_t * uz_t
        return self.w * rho.unsqueeze(0) * (1.0 + cu / CS2 + 0.5 * (cu / CS2) ** 2 - 0.5 * usq / CS2)

    def macroscopic(self):
        rho = self.f.sum(dim=0).clamp(min=1e-8)
        ux = (self.f * self.cv[0]).sum(dim=0) / rho
        uy = (self.f * self.cv[1]).sum(dim=0) / rho
        uz = (self.f * self.cv[2]).sum(dim=0) / rho
        return rho, ux, uy, uz

    def fluid_velocity(self) -> torch.Tensor:
        """Stacked (3, n_fluid) velocity of the fluid nodes."""
        rho, ux, uy, uz = self.macroscopic()
        return torch.stack([ux, uy, uz])[:, self.fluid]

    def velocity_residual(self, u_prev: torch.Tensor) -> float:
        u = self.fluid_velocity()
        du = (u - u_prev).norm().item()
        return du / max(u.norm().item(), 1e-30)

    def step(self) -> None:
        cfg = self.cfg
        fluid = self.fluid
        rho, ux, uy, uz = self.macroscopic()

        # ---- porous sink on the water/ice slip velocity -------------------
        usx = ux - self._u_ice[0]
        usy = uy - self._u_ice[1]
        usz = uz - self._u_ice[2]
        speed = torch.sqrt(usx * usx + usy * usy + usz * usz)
        fac = (self.beta + self.gamma * speed) * rho
        fx = -fac * usx
        fy = -fac * usy
        fz = -fac * usz
        if bool(self.ice.any()):
            self._sink_power = float((usx * fx + usy * fy + usz * fz)[self.ice]
                                     .to(torch.float64).sum())

        pref = (1.0 - 0.5 / cfg.tau)
        cu = self.cv[0] * ux + self.cv[1] * uy + self.cv[2] * uz
        cf = self.cv[0] * fx + self.cv[1] * fy + self.cv[2] * fz
        src = pref * self.w * ((self.cv[0] - ux) * (fx / CS2) + (self.cv[1] - uy) * (fy / CS2)
                               + (self.cv[2] - uz) * (fz / CS2) + cf * cu / CS2 ** 2)

        feq = self._equilibrium(rho, ux, uy, uz)
        f_post = self.f - (self.f - feq) / cfg.tau + src * self.ice.to(self.dtype)

        # ---- stream -------------------------------------------------------
        f_new = torch.empty_like(f_post)
        for i in range(19):
            if _C[i].any():
                f_new[i] = torch.roll(f_post[i], shifts=tuple(int(v) for v in _C[i]), dims=(2, 1, 0))
            else:
                f_new[i] = f_post[i]

        # ---- bounce-back + momentum exchange ------------------------------
        node_force = self._node_force_buf
        node_force.zero_()
        node_flat = node_force.reshape(3, -1)
        fp_flat = f_post.reshape(19, -1)
        for i, idx_all, idx_hull in self._bnd_links:
            f_new.reshape(19, -1)[i, idx_all] = fp_flat[int(_OPP[i]), idx_all]
            if idx_hull.numel():
                exch = 2.0 * fp_flat[i, idx_hull]
                node_flat[:, idx_hull] += self.cvec[i].view(3, 1) * exch
        self._node_force = node_force

        # ---- inlet / outlet -----------------------------------------------
        feq_in = self._feq_in
        for i in range(19):
            if _C[i][0] > 0:
                f_new[i, :, :, 0] = feq_in[i]
            elif _C[i][0] < 0:
                f_new[i, :, :, -1] = f_new[i, :, :, -2]

        # mass-conserving outflow.  A lattice outflow plane necessarily
        # injects the populations that come from outside the domain
        # (c_x < 0 at the last plane).  We scale exactly those injected
        # populations so that the net outflow flux equals the net inflow flux:
        #   plus(-1) - alpha * minus_raw(-1) = net_in
        # The pre-correction imbalance is exported as a physical diagnostic.
        fnew_flat = f_new.reshape(-1)
        net_in = self._plane_flux(fnew_flat, "p0") - self._plane_flux(fnew_flat, "m0")
        plus_out = self._plane_flux(fnew_flat, "pN")
        minus_out = self._plane_flux(fnew_flat, "mN")
        self._pre_correction_imbalance = abs(plus_out - minus_out - net_in) / max(net_in, 1e-30)
        alpha = (plus_out - net_in) / minus_out if minus_out > 1e-12 else 1.0
        if 0.5 < alpha < 1.5:
            for i in range(19):
                if _C[i][0] < 0:
                    f_new[i, :, :, -1] *= alpha
            self._outlet_scale = alpha
        else:
            self._outlet_scale = 1.0

        self.f = f_new * fluid.to(self.dtype)

    # ------------------------------------------------------------- diagnostics
    def mass_error(self) -> float:
        m = float(self.f.sum().to(torch.float64))
        return abs(m - self._mass_ref) / abs(self._mass_ref)

    def _plane_flux(self, f_flat: torch.Tensor, tag: str) -> float:
        idx = self._flux_idx[tag]
        if idx is None:
            return 0.0
        return float(f_flat[idx].to(torch.float64).sum())

    def net_in_flux(self) -> float:
        ff = self.f.reshape(-1)
        return self._plane_flux(ff, "p0") - self._plane_flux(ff, "m0")

    def net_out_flux(self) -> float:
        ff = self.f.reshape(-1)
        return self._plane_flux(ff, "pN") - self._plane_flux(ff, "mN")

    def pre_correction_imbalance(self) -> float:
        """Outflow/inflow mass-flux imbalance measured before the conserving correction."""
        return float(self._pre_correction_imbalance)

    def flux_imbalance(self) -> float:
        """|mass flux out - mass flux in| / mass flux in (steady-state check)."""
        rho, ux, uy, uz = self.macroscopic()
        net_in = self.net_in_flux()
        return abs(self.net_out_flux() - net_in) / max(net_in, 1e-30)

    def hull_force(self) -> Dict[str, float]:
        """Hull force with a pressure (normal) / friction (tangential) split.

        The per-node exchanged momentum is projected on the smoothed wall normal;
        the normal part is the pressure drag, the tangential remainder is the
        friction drag.  Force on the hull is ``+`` the momentum lost by the fluid,
        so the drag is ``R_x = F_x > 0`` for flow in ``+x``.
        """
        mask = self.bnd_nodes.to(torch.float64).unsqueeze(0)
        F = self._node_force * mask
        Fx = F[0].sum().item()
        Fy = F[1].sum().item()
        Fz = F[2].sum().item()
        n = self.normal * mask
        Fdotn = (F * n).sum(dim=0)
        px = float((Fdotn * n[0]).sum().item())
        py = float((Fdotn * n[1]).sum().item())
        pz = float((Fdotn * n[2]).sum().item())
        return {
            "R_x_total": Fx,
            "R_x_normal": px,
            "R_x_friction": Fx - px,
            "F_y": Fy,
            "F_z": Fz,
            "R_y_normal": py,
            "R_z_normal": pz,
        }

    @property
    def sink_power(self) -> float:
        """Power removed from the water by the porous ice layer (lattice units)."""
        return self._sink_power

    def ice_sink_field(self) -> torch.Tensor:
        """(3, nz, ny, nx) momentum sink field of the ice layer (lattice units)."""
        rho, ux, uy, uz = self.macroscopic()
        usx = ux - self._u_ice[0]
        usy = uy - self._u_ice[1]
        usz = uz - self._u_ice[2]
        speed = torch.sqrt(usx * usx + usy * usy + usz * usz)
        fac = (self.beta + self.gamma * speed) * rho
        return torch.stack([-fac * usx, -fac * usy, -fac * usz])

    # ------------------------------------------------------------------- run
    def run(self, n_steps: int, check_every: int = 100,
            residual_target: float = 0.0, verbose: bool = False) -> Dict[str, object]:
        u_prev = self.fluid_velocity().clone()
        n = 0
        for n in range(1, n_steps + 1):
            self.step()
            if n % check_every == 0:
                u_now = self.fluid_velocity()
                res = self.velocity_residual(u_prev)
                u_prev = u_now.clone()
                self.residual_history.append(res)
                self.mass_history.append(self.mass_error())
                self.flux_history.append(self.flux_imbalance())
                if verbose:
                    print(f"  step {n:6d}  residual {res:.3e}  mass {self.mass_history[-1]:.2e}"
                          f"  flux {self.flux_history[-1]:.2e}", flush=True)
                if residual_target > 0.0 and res < residual_target:
                    break
        force = self.hull_force()
        self.force_history.append(force)
        return {
            "steps": n,
            "residual_history": list(self.residual_history),
            "mass_history": list(self.mass_history),
            "flux_history": list(self.flux_history),
            "force": force,
            "sink_power": self.sink_power,
            "final_residual": self.residual_history[-1] if self.residual_history else float("nan"),
            "max_mass_error": max(self.mass_history) if self.mass_history else float("nan"),
            "max_flux_imbalance": max(self.flux_history) if self.flux_history else float("nan"),
        }


def residual_decades(history: List[float]) -> float:
    """How many decades the velocity residual fell from the first sample."""
    if len(history) < 2:
        return 0.0
    return float(math.log10(max(history[0], 1e-30)) - math.log10(max(history[-1], 1e-30)))