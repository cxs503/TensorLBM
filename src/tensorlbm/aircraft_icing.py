"""Aircraft icing: rime ice accretion on a NACA 0012 airfoil (Phase 2a).

Minimal physically credible rime-icing model ("-10 C, everything freezes
on impact, no thermodynamics"):

* Air flow: D3Q19 BGK on a 2-D slice (``nz = 1``) with equilibrium inlet
  (``x = 0``), zero-gradient outlet (``x = nx-1``), free-stream lateral
  walls and full-way bounce-back on airfoil + accreted ice.
* Supercooled droplets: Lagrangian point particles seeded at the inlet at
  the *physical* LWC/MVD mass flux and integrated with Stokes relaxation
  ``du_d/dt = (u_f - u_d) / tau_d``.  No ad-hoc velocity mixing and no
  impact bias: the collection-efficiency (beta) distribution is produced
  by droplet inertia alone.
* Freezing: a droplet that enters a solid cell deposits its mass ``m_d``
  in the last fluid cell it occupied; that cell converts to ice once it
  holds ``rho_rime * dx_phys**3`` of water.  A full mass audit
  (seeded = frozen + exited + airborne + pending) closes to < 1 %.
* Ice feeds back on the flow through the bounce-back solid mask, so the
  drag/lift history shows the icing drift.

Reference case (Ruff & Wright / NASA Glenn IRT, NACA 0012 rime)
    chord 0.5334 m, V = 67 m/s, Re = 2.5e6, LWC = 0.5 g/m^3, MVD = 20 um,
    T = -10 C, t = 360 s, AoA = 4 deg.  Target: LEWICE / IRT leading-edge
    ice shape (qualitative in Phase 2a; quantitative comparison needs
    digitised reference data — deferred to Phase 2b).

Unit mapping (lattice <-> SI)
-----------------------------
Length   ``dx_phys = chord_phys / chord_lu``  [m per lattice unit]
Velocity ``u_phys = u_lu * dx_phys / dt_phys``  ->  choose ``u_lu = u_in``
         matching ``V_inf``:
         ``dt_phys = dx_phys * u_in / v_inf``  [s per lattice time step]

Droplet seeding calibration (derivation)
----------------------------------------
Single droplet mass::

    m_d = rho_w * pi * MVD**3 / 6
        = 1000 * pi * (20e-6)**3 / 6 = 4.19e-12 kg

(a 20 um droplet cloud carries ~1.2e9 droplets/m^3 — far too many to
track individually, see *Parcels* below).

Inlet area of the 2-D slab (one cell deep in z)::

    A_in = ny * dx_phys * dx_phys      [m^2]

Water mass flux through the inlet (SI)::

    Q = LWC * V_inf * A_in             [kg/s]

Mass seeded per LBM step and droplets per LBM step::

    dm/step = LWC * V_inf * A_in * dt_phys
    N/step  = LWC * V_inf * A_in * dt_phys / m_d

so the local impinging water flux — and therefore the collection
efficiency beta — is dimensionally correct, not a free parameter.

Parcels (representative droplets)
---------------------------------
The simulator tracks Lagrangian *parcels*: each parcel represents
``parcel_multiplier`` identical physical droplets (mass ``m_parcel =
N * m_d``).  Because rime droplets do not interact (dilute: volume
fraction LWC/rho_w ~ 5e-7 even after acceleration), parcel statistics
are exact for trajectories; only the sampling noise of beta improves
with more parcels.  ``parcel_multiplier = None`` picks the smallest N
that keeps the seeding rate below ``parcels_per_step_target`` (default
2000/step).  All mass bookkeeping (audit, freezing threshold, beta) is
in kg of *water*, so the parcel choice never biases the physics.

Cloud initialisation (prefill)
------------------------------
The exposure window opens with the cloud already in place, so at t=0 the
run seeds the steady-state inventory ``parcels_per_step * kill_x / u_in``
parcels uniformly upstream of the kill plane (droplet speed ~ u_in:
tau_d >> leading-edge transit, so the line density is rate/u_in).
Velocities start at the local (warmed-up) flow value and positions inside
the airfoil are rejected.  Without prefill the leading edge would be
starved for the first ~x_le/u_in steps of the window.  Set
``prefill_cloud=False`` for pure inlet-flux seeding.
``disable_droplets=True`` runs the clean-airfoil twin (no droplets, no
freezing) for an A/B measurement of the ice feedback on cd/cl.

Time acceleration (rime only)
-----------------------------
LBM time steps are tiny: with the mapping above the 360 s IRT exposure
corresponds to O(10^7) lattice steps.  Rime accretion is *linear* in LWC
(all collected water freezes where it hits, droplets do not interact, the
quasi-steady flow is independent of concentration), so the run uses an
accelerated effective cloud ``LWC_eff = k * LWC`` and reports the
equivalent physical exposure::

    t_equiv = steps * dt_phys * k

This is exact for rime and deliberately *not* used for glaze (runback and
heat balance are non-linear in LWC) — glaze is Phase 3.

Droplet relaxation
------------------
Stokes relaxation time::

    tau_d = rho_w * MVD**2 / (18 * mu_air)      [s]
    tau_d_lu = tau_d / dt_phys                   [lattice steps]

Inertia (Stokes) number, invariant between unit systems::

    St = tau_d * V_inf / chord_phys = tau_d_lu * u_in / chord_lu

With the default ``mu_air = 1.8e-5 Pa s`` the reference case gives
``St = 0.155`` (IRT-consistent).  The explicit Euler relaxation factor
per LBM step is ``dt/tau_d_lu ~ 1/400``, i.e. far below any stability
limit, so a single sub-step per LBM step is sufficient by default; the
sub-step count is configurable and auto-scales if ``tau_d_lu`` shrinks.

Rime density
------------
Ice volume — and hence horn height — scales as 1/rho_rime, so the
density model matters quantitatively.  Implemented options
(``rime_density_mode``):

* ``"macklin"`` (default) — Macklin (1962) correlation
  ``rho [g/cm^3] = 0.110 * R**0.76`` capped at 0.917, with
  ``R = -(mvd[um] * V_inf[m/s]) / (2 * T_s[C])``.  Without the Phase 3
  energy balance the surface temperature is approximated by the static
  temperature plus optional aerodynamic recovery heating
  ``T_s = T_inf + 0.5 * V^2 / (2 cp)`` (r = 0.5, cp = 1005 J/kg/K;
  ~1.1 K for the IRT case — negligible here, kept for honesty).
* ``"jones"`` — Jones (1990): ``rho = 0.0443 * (mvd V)**0.75 / |T_inf|``,
  avoids the surface-temperature iteration.
* ``"const"`` — fixed ``rho_rime`` (measured rime spans ~100-900 kg/m^3,
  glaze 917).

For the IRT reference case Macklin gives R ~ 67-75, i.e. the cap: the
-10 C / 67 m /s / 20 um point is *hard* rime at essentially solid
density (917 kg/m^3) — consistent with it sitting near the glaze
boundary.  Colder/slower/smaller-droplet conditions give the fluffy
100-300 kg/m^3 values (e.g. 20 um, 10 m/s, -30 C -> ~270 kg/m^3).

Phase 2b scope (not in this module)
-----------------------------------
Re = 2.5e6 operation needs cumulant collision + LES (and/or wall models);
Phase 2a deliberately runs a moderate lattice Reynolds number
(``Re_lu ~ 400``, ``u_in = 0.05``, ``tau >= 0.53``) where BGK is stable
and the leading-edge flow field is essentially potential, which is what
governs the impingement statistics at matched St.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any

import numpy as np
import torch

from tensorlbm.compile_utils import compile_step, validate_compile_mode
from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

__all__ = [
    "IcingConfig",
    "RimeIcingSimulation",
    "run_rime_icing",
    "naca0012_mask_2d",
    "surface_arc_length",
    "collection_efficiency_curve",
    "ice_shape_metrics",
    "seed_counts_total",
    "rime_density_macklin",
    "rime_density_jones",
]


# ---------------------------------------------------------------------------
# Configuration + unit mapping
# ---------------------------------------------------------------------------
@dataclass
class IcingConfig:
    """Parameters for the 2-D rime icing simulation.

    Lattice and physical parameter sets are tied together by the unit
    mapping derived in the module docstring; all derived quantities are
    exposed as properties so every printed number is traceable.
    """

    # --- lattice ---
    nx: int = 320
    ny: int = 160
    chord_frac: float = 0.4
    u_in: float = 0.05
    tau: float = 0.55
    aoa_deg: float = 4.0
    cx_frac: float = 0.3
    cy_frac: float = 0.5
    naca_t: float = 0.12

    # --- physical reference case (NASA Glenn IRT NACA 0012 rime) ---
    chord_phys: float = 0.5334  # m
    v_inf: float = 67.0  # m/s
    lwc: float = 5.0e-4  # kg/m^3 (0.5 g/m^3)
    mvd: float = 20.0e-6  # m
    rho_water: float = 1000.0  # kg/m^3
    mu_air: float = 1.8e-5  # Pa s (gives St = 0.155 for this case)
    rime_density_mode: str = "macklin"  # "const" | "macklin" | "jones"
    rho_rime: float = 917.0  # kg/m^3, used when rime_density_mode == "const"
    t_static_c: float = -10.0  # static air temperature [C]
    recovery_heating: bool = True  # T_s proxy: T_inf + 0.5 V^2/(2 cp)
    t_exposure: float = 360.0  # s of cloud exposure to represent

    # --- run control ---
    steps: int = 3000
    warmup_steps: int = 2000
    substeps: int | None = None  # None -> auto from tau_d_lu
    parcel_multiplier: int | None = None  # None -> auto (target below)
    parcels_per_step_target: float = 2000.0
    kill_frac: float = 0.15  # droplet kill plane: LE + kill_frac*chord
    # Steady-state cloud inventory at t=0 of the exposure window (the real
    # cloud is already everywhere when icing starts; without prefill the
    # first ~x_le/u_in steps of the window would collect nothing and the
    # leading-edge flux would be underestimated).
    prefill_cloud: bool = True
    beta_window_frac: float = 0.5  # trailing fraction of steps used for beta
    disable_droplets: bool = False  # clean-airfoil twin for A/B cd drift
    uniform_flow: bool = False  # True: static uniform field (tests, no LBM)
    device: str = "cpu"
    seed: int = 0
    log_every: int = 250
    accel_override: float | None = None  # explicit k; else derived from t_exposure
    compile_mode: str | None = None  # flow step: None | "default" |
    #                                      "max-autotune-no-cudagraphs"

    # ------------------------------------------------------------------
    # lattice-side derived quantities
    @property
    def chord_lu(self) -> float:
        return self.nx * self.chord_frac

    @property
    def nu_lu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def re_lu(self) -> float:
        return self.u_in * self.chord_lu / self.nu_lu

    # ------------------------------------------------------------------
    # unit mapping
    @property
    def dx_phys(self) -> float:
        """Metres per lattice unit."""
        return self.chord_phys / self.chord_lu

    @property
    def dt_phys(self) -> float:
        """Seconds per lattice time step (u_lu = u_in <-> v_inf)."""
        return self.dx_phys * self.u_in / self.v_inf

    # ------------------------------------------------------------------
    # droplet physics
    @property
    def m_droplet(self) -> float:
        """Single droplet mass [kg]."""
        return self.rho_water * math.pi * self.mvd**3 / 6.0

    @property
    def tau_d_phys(self) -> float:
        """Stokes relaxation time [s]."""
        return self.rho_water * self.mvd**2 / (18.0 * self.mu_air)

    @property
    def tau_d_lu(self) -> float:
        """Stokes relaxation time in lattice steps."""
        return self.tau_d_phys / self.dt_phys

    @property
    def stokes(self) -> float:
        """Inertia number (unit-system invariant)."""
        return self.tau_d_phys * self.v_inf / self.chord_phys

    @property
    def t_surface_eff_c(self) -> float:
        """Surface-temperature proxy for the rime density correlations.

        Static temperature plus (optionally) aerodynamic recovery heating
        ``0.5 * V^2 / (2 cp)`` with cp = 1005 J/kg/K (r = 0.5).
        """
        if self.recovery_heating:
            return self.t_static_c + 0.5 * self.v_inf**2 / (2.0 * 1005.0)
        return self.t_static_c

    @property
    def rime_R_macklin(self) -> float:
        """Macklin parameter R = -(mvd[um] * V) / (2 T_s)."""
        if self.t_surface_eff_c >= 0.0:
            return float("inf")
        return -(self.mvd * 1e6 * self.v_inf) / (2.0 * self.t_surface_eff_c)

    @property
    def rho_rime_eff(self) -> float:
        """Effective rime density [kg/m^3] under the configured mode."""
        if self.rime_density_mode == "const":
            return self.rho_rime
        if self.rime_density_mode == "macklin":
            return rime_density_macklin(self.mvd, self.v_inf, self.t_surface_eff_c)
        if self.rime_density_mode == "jones":
            return rime_density_jones(self.mvd, self.v_inf, self.t_static_c)
        raise ValueError(
            f"rime_density_mode must be 'const', 'macklin' or 'jones'; "
            f"got {self.rime_density_mode!r}"
        )

    @property
    def m_cell_ice(self) -> float:
        """Ice mass that fills one cell [kg]."""
        return self.rho_rime_eff * self.dx_phys**3

    @property
    def inlet_area(self) -> float:
        """Inlet cross-section of the 2-D slab (one cell deep) [m^2]."""
        return self.ny * self.dx_phys * self.dx_phys

    @property
    def lwc_accel(self) -> float:
        """LWC acceleration factor k (see docstring: exact for rime)."""
        if self.accel_override is not None:
            return self.accel_override
        return self.t_exposure / (self.steps * self.dt_phys)

    @property
    def lwc_eff(self) -> float:
        return self.lwc * self.lwc_accel

    @property
    def droplets_per_step(self) -> float:
        """Analytic seeding rate [physical droplets / LBM step] (docstring)."""
        return self.lwc_eff * self.v_inf * self.inlet_area * self.dt_phys / self.m_droplet

    @property
    def effective_parcel_multiplier(self) -> int:
        """Physical droplets represented by one simulated parcel."""
        if self.parcel_multiplier is not None:
            return max(1, int(self.parcel_multiplier))
        n = math.ceil(self.droplets_per_step / max(self.parcels_per_step_target, 1.0))
        return max(1, int(n))

    @property
    def m_parcel(self) -> float:
        """Mass of one simulated parcel [kg]."""
        return self.effective_parcel_multiplier * self.m_droplet

    @property
    def parcels_per_step(self) -> float:
        """Simulated parcels seeded per LBM step (mass flux is exact)."""
        return self.droplets_per_step / self.effective_parcel_multiplier

    @property
    def t_equiv(self) -> float:
        """Equivalent physical cloud exposure achieved by the run [s]."""
        return self.steps * self.dt_phys * self.lwc_accel

    @property
    def n_substeps(self) -> int:
        if self.substeps is not None:
            return max(1, int(self.substeps))
        # explicit Euler relaxation needs dt/tau << 1; 8 sub-steps margin.
        return max(1, int(math.ceil(8.0 / max(self.tau_d_lu, 1e-12))))

    def mapping_report(self) -> dict[str, Any]:
        """All mapping/physics numbers in one dictionary (printed + JSON)."""
        return {
            "nx": self.nx,
            "ny": self.ny,
            "chord_lu": self.chord_lu,
            "u_in": self.u_in,
            "tau": self.tau,
            "nu_lu": self.nu_lu,
            "re_lu": self.re_lu,
            "dx_phys": self.dx_phys,
            "dt_phys": self.dt_phys,
            "chord_phys": self.chord_phys,
            "v_inf": self.v_inf,
            "lwc": self.lwc,
            "lwc_eff": self.lwc_eff,
            "lwc_accel": self.lwc_accel,
            "mvd": self.mvd,
            "m_droplet": self.m_droplet,
            "m_cell_ice": self.m_cell_ice,
            "rho_rime_eff": self.rho_rime_eff,
            "rime_density_mode": self.rime_density_mode,
            "t_static_c": self.t_static_c,
            "t_surface_eff_c": self.t_surface_eff_c,
            "rime_R_macklin": self.rime_R_macklin,
            "tau_d_phys": self.tau_d_phys,
            "tau_d_lu": self.tau_d_lu,
            "stokes": self.stokes,
            "droplets_per_step": self.droplets_per_step,
            "parcel_multiplier": self.effective_parcel_multiplier,
            "m_parcel": self.m_parcel,
            "parcels_per_step": self.parcels_per_step,
            "inlet_area": self.inlet_area,
            "t_exposure_target": self.t_exposure,
            "t_equiv": self.t_equiv,
            "steps_realtime": self.t_exposure / self.dt_phys,
            "n_substeps": self.n_substeps,
            "compile_mode": self.compile_mode,
            "uniform_flow": self.uniform_flow,
        }


def seed_counts_total(rate: float, steps: int) -> int:
    """Deterministic integer seeding over ``steps`` steps.

    Fractional droplet-per-step rates are accumulated in a carry so the
    expected total ``rate * steps`` is reproduced exactly (to one droplet)
    without stochastic rounding.  Used by the simulator and by the
    seeding-calibration unit test.
    """
    total = 0
    carry = 0.0
    for _ in range(steps):
        carry += rate
        n = int(carry)
        carry -= n
        total += n
    return total


# ---------------------------------------------------------------------------
# Rime density correlations (see module docstring)
# ---------------------------------------------------------------------------
RIME_DENSITY_CAP_G_CM3 = 0.917  # solid ice


def rime_density_macklin(mvd: float, v_inf: float, t_surface_c: float) -> float:
    """Macklin (1962) rime density [kg/m^3].

    ``rho [g/cm^3] = min(0.917, 0.110 * R**0.76)`` with the Macklin
    parameter ``R = -(mvd[um] * v_inf[m/s]) / (2 * T_s[C]) >= 0``.
    """
    if t_surface_c >= 0.0:
        return RIME_DENSITY_CAP_G_CM3 * 1000.0
    r_param = -(mvd * 1e6 * v_inf) / (2.0 * t_surface_c)
    rho = min(RIME_DENSITY_CAP_G_CM3, 0.110 * r_param**0.76)
    return rho * 1000.0


def rime_density_jones(mvd: float, v_inf: float, t_air_c: float) -> float:
    """Jones (1990) rime density [kg/m^3] (static temperature, no T_s)."""
    rho = min(RIME_DENSITY_CAP_G_CM3, 0.0443 * (mvd * 1e6 * v_inf) ** 0.75 / abs(t_air_c))
    return rho * 1000.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def naca0012_mask_2d(
    nx: int,
    ny: int,
    chord: float,
    aoa_deg: float = 4.0,
    cx: float | None = None,
    cy: float | None = None,
    t: float = 0.12,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Vectorised NACA 4-digit symmetric airfoil solid mask (ny, nx).

    Implicit fill: a cell is solid when its airfoil-aligned coordinates
    (xr, yr) satisfy 0 <= xr <= 1 and |yr| <= y_t(xr), with
    ``y_t = 5 t (0.2969 sqrt(x) - 0.126 x - 0.3516 x^2 + 0.2843 x^3 - 0.1015 x^4)``.
    Positive ``aoa_deg`` rotates the airfoil nose-up (LE above the TE).
    """
    if cx is None:
        cx = nx / 3.0
    if cy is None:
        cy = ny / 2.0
    dev = torch.device(device)
    xx = torch.arange(nx, device=dev, dtype=torch.float32)
    yy = torch.arange(ny, device=dev, dtype=torch.float32)
    gx, gy = torch.meshgrid(xx, yy, indexing="xy")  # (ny, nx)
    dx = gx - cx
    dy = gy - cy
    aoa = math.radians(aoa_deg)
    cos_a, sin_a = math.cos(aoa), math.sin(aoa)
    # positive AoA = nose-up (trailing edge below the leading edge),
    # matching the convention of tensorlbm.airfoil_benchmark
    xr = (dx * cos_a - dy * sin_a) / chord
    yr = (dx * sin_a + dy * cos_a) / chord
    xc = xr.clamp(0.0, 1.0)
    yt = (
        5.0
        * t
        * (0.2969 * torch.sqrt(xc) - 0.126 * xc - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4)
    )
    return (xr >= 0.0) & (xr <= 1.0) & (yr.abs() <= yt)


def _dilate4(mask: torch.Tensor) -> torch.Tensor:
    return (
        torch.roll(mask, 1, 0)
        | torch.roll(mask, -1, 0)
        | torch.roll(mask, 1, 1)
        | torch.roll(mask, -1, 1)
    )


# ---------------------------------------------------------------------------
# Surface arc-length machinery (numpy, CPU, run once at the end)
# ---------------------------------------------------------------------------
def surface_arc_length(airfoil: np.ndarray) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """Signed arc distance from the leading-edge surface cell.

    Returns ``(s_grid, stag_xy, surf)``:

    * ``surf`` — surface cells of the *original* airfoil (solid cells with
      a 4-connected fluid neighbour);
    * ``stag_xy`` — leading-edge surface cell (minimum x; the s = 0 origin);
    * ``s_grid`` — for every grid cell, the signed arc distance (in cells)
      from the LE cell to its nearest surface cell, positive on the upper
      surface (y > y_stag).  Distances between surface cells are graph
      distances on the 8-connected surface-cell graph (1 for orthogonal
      neighbours, sqrt(2) for diagonal).

    Impact cells are mapped onto this coordinate so the beta curve is a
    function of surface position, independent of how thick the ice has
    grown during the run.
    """
    ny, nx = airfoil.shape
    fluid = ~airfoil
    surf = airfoil & _dilate4(torch.from_numpy(fluid)).numpy()
    ys, xs = np.nonzero(surf)
    n_surf = len(ys)
    sid = -np.ones((ny, nx), dtype=np.int64)
    sid[ys, xs] = np.arange(n_surf)

    # leading edge: minimum x, tie-break closest to domain centreline
    y_cy = ny / 2.0
    order = np.lexsort((np.abs(ys - y_cy), xs))
    stag_id = int(order[0])
    stag = (int(ys[stag_id]), int(xs[stag_id]))

    # Dijkstra over the 8-connected surface-cell graph
    dist = np.full(n_surf, np.inf)
    dist[stag_id] = 0.0
    heap: list[tuple[float, int]] = [(0.0, stag_id)]
    nbrs = [
        (dys, dxs, math.hypot(dys, dxs))
        for dys in (-1, 0, 1)
        for dxs in (-1, 0, 1)
        if (dys, dxs) != (0, 0)
    ]
    coord2id = {(int(ys[i]), int(xs[i])): i for i in range(n_surf)}
    while heap:
        d, u = heappop(heap)
        if d > dist[u]:
            continue
        yu, xu = int(ys[u]), int(xs[u])
        for dys, dxs, w in nbrs:
            v = coord2id.get((yu + dys, xu + dxs))
            if v is None:
                continue
            nd = d + w
            if nd < dist[v] - 1e-12:
                dist[v] = nd
                heappush(heap, (nd, v))

    sign = np.where(ys > stag[0], 1.0, np.where(ys < stag[0], -1.0, 0.0))
    s_surf = dist * sign

    # multi-source BFS: nearest surface cell id for every grid cell
    nearest = -np.ones((ny, nx), dtype=np.int64)
    queue: deque[tuple[int, int]] = deque()
    for i in range(n_surf):
        nearest[ys[i], xs[i]] = i
        queue.append((int(ys[i]), int(xs[i])))
    while queue:
        y, x = queue.popleft()
        i0 = nearest[y, x]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yn, xn = y + dy, x + dx
            if 0 <= yn < ny and 0 <= xn < nx and nearest[yn, xn] < 0:
                nearest[yn, xn] = i0
                queue.append((yn, xn))

    s_grid = s_surf[nearest]
    return s_grid, stag, surf


def collection_efficiency_curve(
    s_grid: np.ndarray,
    impact_mass: np.ndarray,
    lwc_eff: float,
    v_inf: float,
    dx_phys: float,
    chord_phys: float,
    t_window: float,
    bin_width: float = 1.0,
) -> dict[str, np.ndarray]:
    """Local collection efficiency beta versus surface position.

    ``beta_i = dm_i / (LWC_eff * V_inf * A_cell * t_lat)`` with the cell
    face area ``A_cell = dx_phys**2`` (one-cell-deep slab), where
    ``t_lat = n_window_steps * dt_phys`` is the *lattice* time of the
    window (NOT the accelerated physical time ``n*dt*k``): the impact mass
    is measured in the accelerated run, so the streamtube that carries it
    is ``LWC_eff * V * dx^2 * dt * n_steps``.  Equivalently
    ``= LWC * V * dx^2 * t_phys`` in real units — the acceleration factor
    cancels.  Values are averaged in arc-length bins of ``bin_width`` cells.
    """
    ys, xs = np.nonzero(impact_mass > 0)
    if len(ys) == 0:
        return {"s_over_c": np.array([]), "beta": np.array([]), "n_cells": np.array([])}
    s = s_grid[ys, xs]
    m = impact_mass[ys, xs]
    denom = lwc_eff * v_inf * dx_phys**2 * t_window
    beta_cell = m / denom
    lo = math.floor(s.min() - 0.5)
    hi = math.ceil(s.max() + 0.5)
    edges = np.arange(lo, hi + bin_width, bin_width)
    idx = np.digitize(s, edges) - 1
    nb = len(edges) - 1
    s_out, b_out, n_out = [], [], []
    for b in range(nb):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        s_out.append(float(s[sel].mean()) * dx_phys / chord_phys)
        b_out.append(float(beta_cell[sel].mean()))
        n_out.append(n)
    return {"s_over_c": np.array(s_out), "beta": np.array(b_out), "n_cells": np.array(n_out)}


# ---------------------------------------------------------------------------
# Ice shape metrics
# ---------------------------------------------------------------------------
def _layer_depth(target: np.ndarray, base: np.ndarray, max_layers: int = 64) -> np.ndarray:
    """Layer distance (in cells) of ``target`` cells from the ``base`` set."""
    ny, nx = target.shape
    layers = np.zeros((ny, nx), dtype=np.int32)
    cur = base.copy()
    rem = target & ~base
    for layer in range(1, max_layers + 1):
        if not rem.any():
            break
        grown = np.zeros_like(cur)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            grown |= np.roll(cur, shift, axis=axis)
        hit = rem & grown
        layers[hit] = layer
        cur |= hit
        rem &= ~hit
    return layers


def ice_shape_metrics(
    airfoil: np.ndarray,
    solid: np.ndarray,
    dx_phys: float,
    chord_phys: float,
    chord_lu: float,
    stag: tuple[int, int],
) -> dict[str, Any]:
    """Quantitative ice-shape descriptors.

    * ice area (cells and m^2, % of chord^2)
    * upper/lower horn height: max layer depth of ice cells above/below the
      stagnation row, in cells, % chord and metres
    * ice streamwise extent (where ice actually grows relative to the LE)
    * upper/lower symmetry of the horns
    """
    ice_only = solid & ~airfoil
    n_ice = int(ice_only.sum())
    layers = _layer_depth(ice_only, airfoil)
    y_stag, x_le = stag
    y_idx, x_idx = np.nonzero(ice_only)
    metrics: dict[str, Any] = {
        "n_ice_cells": n_ice,
        "ice_area_m2": n_ice * dx_phys**2,
        "ice_area_pct_chord2": 100.0 * n_ice * dx_phys**2 / chord_phys**2,
        "x_le": int(x_le),
        "y_stag": int(y_stag),
    }
    if n_ice == 0:
        return metrics
    up = y_idx > y_stag
    lo = y_idx < y_stag
    if up.any():
        h = int(layers[y_idx[up], x_idx[up]].max())
        k = int(np.argmax(layers[y_idx[up], x_idx[up]]))
        metrics["upper_horn_cells"] = h
        metrics["upper_horn_pct_chord"] = 100.0 * h / chord_lu
        metrics["upper_horn_m"] = h * dx_phys
        metrics["upper_horn_xy"] = [int(y_idx[up][k]), int(x_idx[up][k])]
    if lo.any():
        h = int(layers[y_idx[lo], x_idx[lo]].max())
        k = int(np.argmax(layers[y_idx[lo], x_idx[lo]]))
        metrics["lower_horn_cells"] = h
        metrics["lower_horn_pct_chord"] = 100.0 * h / chord_lu
        metrics["lower_horn_m"] = h * dx_phys
        metrics["lower_horn_xy"] = [int(y_idx[lo][k]), int(x_idx[lo][k])]
    if "upper_horn_cells" in metrics and "lower_horn_cells" in metrics:
        hu, hl = metrics["upper_horn_cells"], metrics["lower_horn_cells"]
        metrics["horn_symmetry_pct"] = 100.0 * (hu - hl) / (hu + hl)
    x_off = (x_idx - x_le) / chord_lu
    metrics["ice_x_offset_min"] = float(x_off.min())
    metrics["ice_x_offset_max"] = float(x_off.max())
    metrics["ice_max_layer"] = int(layers.max())
    return metrics


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class RimeIcingSimulation:
    """Coupled BGK flow + Lagrangian droplet + rime accretion loop."""

    def __init__(self, cfg: IcingConfig, log: Any = print):
        self.cfg = cfg
        self.log = log
        self.dev = torch.device(cfg.device)
        self.nx, self.ny = cfg.nx, cfg.ny
        self.chord = cfg.chord_lu

        self.airfoil = naca0012_mask_2d(
            self.nx,
            self.ny,
            self.chord,
            cfg.aoa_deg,
            cx=self.nx * cfg.cx_frac,
            cy=self.ny * cfg.cy_frac,
            t=cfg.naca_t,
            device=self.dev,
        )
        self.solid = self.airfoil.clone()  # airfoil | ice
        self.opp = OPPOSITE.to(self.dev)

        # whole-step compilation (shared module: single source of truth
        # for the mode whitelist; cudagraph-class modes are rejected there)
        validate_compile_mode(cfg.compile_mode)
        hint = (
            "two guarded graphs: plain flow step + flow step with the "
            "post-stream/pre-bounce force probe"
        )
        self._step_plain = compile_step(self._flow_step_plain, cfg.compile_mode, warmup_hint=hint)
        self._step_probe = compile_step(self._flow_step_probe, cfg.compile_mode, warmup_hint=hint)

        rho0 = torch.ones((1, self.ny, self.nx), device=self.dev)
        u0 = torch.full((1, self.ny, self.nx), cfg.u_in, device=self.dev)
        z0 = torch.zeros_like(u0)
        self.f = equilibrium3d(rho0, u0, z0.clone(), z0.clone())
        self.feq_in = self.f.clone()

        # droplet state (empty; lazily concatenated torch tensors)
        self.px = torch.zeros(0, device=self.dev)
        self.py = torch.zeros(0, device=self.dev)
        self.vx = torch.zeros(0, device=self.dev)
        self.vy = torch.zeros(0, device=self.dev)
        self.age = torch.zeros(0, device=self.dev, dtype=torch.int64)

        # accretion + measurement fields
        self.m_w = torch.zeros((self.ny, self.nx), device=self.dev)  # kg per cell
        self.impact_mass = torch.zeros((self.ny, self.nx), device=self.dev)  # kg (all run)

        # geometry anchors
        ys, xs = torch.nonzero(self.airfoil, as_tuple=True)
        self.x_le = int(xs.min().item())
        self.y_le = int(ys[xs.argmin()].item())
        self.kill_x = float(self.x_le + cfg.kill_frac * self.chord)
        self.max_age = int(6.0 * self.nx / max(cfg.u_in, 1e-3))

        # mass audit (kg, float64)
        self.aud = {
            "seeded": 0.0,
            "frozen": 0.0,
            "exited": 0.0,
            "trapped": 0.0,
            "airborne": 0.0,
            "pending": 0.0,
        }
        self.n_impacts = 0
        self._seed_carry = 0.0
        self.history: dict[str, list[Any]] = {
            "step": [], "t_phys": [], "cd": [], "cl": [], "ice_cells": [], "drops_alive": []
        }
        self.cd0 = self.cl0 = None
        self.cd_end = self.cl_end = None

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _flow_step_plain(
        f: torch.Tensor,
        solid: torch.Tensor,
        feq_in: torch.Tensor,
        opp: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        """Whole LBM step (collide, stream, bounce-back, BCs) — pure tensors.

        This is the unit of compilation (compile_utils lesson 1: whole
        step, not hot op).  No step index, no host sync, no data-dependent
        branch — those live in the eager driver (lesson 2).
        """
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = torch.where(solid[None, None], f[opp], f)
        f[:, :, :, 0] = feq_in[:, :, :, 0]  # inlet equilibrium
        f[:, :, :, -1] = f[:, :, :, -2]  # outlet zero-gradient
        f[:, :, 0, :] = feq_in[:, :, 0, :]  # free-stream lateral walls
        f[:, :, -1, :] = feq_in[:, :, -1, :]
        return f

    @staticmethod
    def _flow_step_probe(
        f: torch.Tensor,
        solid: torch.Tensor,
        feq_in: torch.Tensor,
        opp: torch.Tensor,
        tau: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same step, additionally returning post-stream pre-bounce f_pre.

        Compiled as a *separate variant* (compile_utils lesson 2: one
        variant per branch pattern; the driver picks plain vs probe).
        """
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f_pre = f.clone()  # post-stream, pre-bounce-back (force probe)
        f = torch.where(solid[None, None], f[opp], f)
        f[:, :, :, 0] = feq_in[:, :, :, 0]
        f[:, :, :, -1] = f[:, :, :, -2]
        f[:, :, 0, :] = feq_in[:, :, 0, :]
        f[:, :, -1, :] = feq_in[:, :, -1, :]
        return f, f_pre

    @staticmethod
    def _sample_bilinear(field: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        ny, nx = field.shape
        x0 = torch.floor(xs).long().clamp(0, nx - 2)
        y0 = torch.floor(ys).long().clamp(0, ny - 2)
        fx = xs - x0.float()
        fy = ys - y0.float()
        return (
            field[y0, x0] * (1 - fx) * (1 - fy)
            + field[y0, x0 + 1] * fx * (1 - fy)
            + field[y0 + 1, x0] * (1 - fx) * fy
            + field[y0 + 1, x0 + 1] * fx * fy
        )

    def _flow_step(self, want_force: bool) -> torch.Tensor | None:
        """One flow step via the (optionally compiled) whole-step variants.

        Droplet advection/freezing stay eager on purpose: the droplet
        phase has per-step host syncs (mass audit) and dynamic tensor
        shapes (parcel filtering) that structurally reject torch.compile;
        the flow step is the hot path and is the compiled unit.
        """
        if want_force:
            f, f_pre = self._step_probe(self.f, self.solid, self.feq_in, self.opp, self.cfg.tau)
            self.f = f
            return f_pre
        self.f = self._step_plain(self.f, self.solid, self.feq_in, self.opp, self.cfg.tau)
        return None

    def _force_coeffs(self, f_pre: torch.Tensor) -> tuple[float, float]:
        solid = self.solid
        surf = solid & _dilate4(~solid)
        df = (f_pre - self.f) * surf[None, None].float()
        c = C.to(self.dev).float()
        fx = float((df * c[:, 0].view(19, 1, 1, 1)).sum().item())
        fy = float((df * c[:, 1].view(19, 1, 1, 1)).sum().item())
        q = 0.5 * self.cfg.u_in**2
        return fx / (q * self.chord), fy / (q * self.chord)

    def _seed(self) -> None:
        self._seed_carry += self.cfg.parcels_per_step
        n = int(self._seed_carry)
        self._seed_carry -= n
        if n <= 0:
            return
        gen_dev = self.dev
        px = 0.5 + 0.4 * torch.rand(n, device=gen_dev)
        py = 0.5 + (self.ny - 1.0) * torch.rand(n, device=gen_dev)
        vx = torch.full((n,), self.cfg.u_in, device=gen_dev)
        vy = torch.zeros(n, device=gen_dev)
        ag = torch.zeros(n, device=gen_dev, dtype=torch.int64)
        self.px = torch.cat([self.px, px])
        self.py = torch.cat([self.py, py])
        self.vx = torch.cat([self.vx, vx])
        self.vy = torch.cat([self.vy, vy])
        self.age = torch.cat([self.age, ag])
        self.aud["seeded"] += n * self.cfg.m_parcel

    def _prefill(self, ux: torch.Tensor, uy: torch.Tensor) -> None:
        """Seed the steady-state cloud inventory at t=0 of the exposure.

        The real cloud is already everywhere when the exposure window opens;
        seeding only at the inlet would leave the leading edge starved for
        the first ~x_le/u_in steps.  Steady line density is
        ``parcels_per_step / u_d`` per cell of x (droplet speed, which stays
        ~u_in through the LE deceleration because tau_d >> LE transit), so
        the inventory over [inlet, kill plane] is
        ``parcels_per_step * kill_x / u_in``.  Positions are uniform in x,
        velocities are sampled from the warmed-up flow (local equilibrium),
        and positions inside the airfoil are rejected (that water was never
        there).
        """
        cfg = self.cfg
        n_pre = int(cfg.parcels_per_step * self.kill_x / cfg.u_in)
        if n_pre <= 0:
            return
        px = 0.5 + (self.kill_x - 1.0) * torch.rand(n_pre, device=self.dev)
        py = 0.5 + (self.ny - 1.0) * torch.rand(n_pre, device=self.dev)
        ix = px.floor().long().clamp(0, self.nx - 1)
        iy = py.floor().long().clamp(0, self.ny - 1)
        ok = ~self.solid[iy, ix]
        n_kept = int(ok.sum().item())
        if n_kept == 0:
            return
        px, py = px[ok], py[ok]
        vx = self._sample_bilinear(ux, px, py)
        vy = self._sample_bilinear(uy, px, py)
        ag = torch.zeros(n_kept, device=self.dev, dtype=torch.int64)
        self.px = torch.cat([self.px, px])
        self.py = torch.cat([self.py, py])
        self.vx = torch.cat([self.vx, vx])
        self.vy = torch.cat([self.vy, vy])
        self.age = torch.cat([self.age, ag])
        self.aud["seeded"] += n_kept * cfg.m_parcel
        self.log(
            f"  [icing] prefill cloud: {n_kept} parcels "
            f"(~{cfg.parcels_per_step * self.kill_x / cfg.u_in:.0f} analytic, "
            f"{n_kept * cfg.m_parcel:.3e} kg) in x<[{self.kill_x:.0f}]"
        )

    def _droplet_step(self, ux: torch.Tensor, uy: torch.Tensor) -> None:
        cfg = self.cfg
        self._seed()
        n_sub = cfg.n_substeps
        dt_sub = 1.0 / n_sub
        relax = dt_sub / cfg.tau_d_lu
        m_p = cfg.m_parcel
        solid = self.solid
        nx, ny = self.nx, self.ny
        for _ in range(n_sub):
            if self.px.numel() == 0:
                return
            ufx = self._sample_bilinear(ux, self.px, self.py)
            ufy = self._sample_bilinear(uy, self.px, self.py)
            self.vx += (ufx - self.vx) * relax
            self.vy += (ufy - self.vy) * relax
            ix = self.px.floor().long().clamp(0, nx - 1)
            iy = self.py.floor().long().clamp(0, ny - 1)
            new_px = self.px + self.vx * dt_sub
            new_py = self.py + self.vy * dt_sub
            nix = new_px.floor().long().clamp(0, nx - 1)
            niy = new_py.floor().long().clamp(0, ny - 1)

            moved_into_solid = solid[niy, nix]
            encased = solid[iy, ix] & ~moved_into_solid  # frozen under a droplet

            # deposits: impactors freeze in their last fluid cell; droplets
            # encased by fresh ice freeze where they sit.
            dep_iy = torch.cat([iy[moved_into_solid], iy[encased]])
            dep_ix = torch.cat([ix[moved_into_solid], ix[encased]])
            hit = moved_into_solid | encased
            if hit.any():
                n_hit = int(hit.sum().item())
                self.n_impacts += n_hit
                vals = torch.full((n_hit,), m_p, device=self.dev)
                flat = dep_iy * nx + dep_ix
                self.m_w.view(-1).index_add_(0, flat, vals)
                self.impact_mass.view(-1).index_add_(0, flat, vals)

            left = (
                (new_px >= self.kill_x)
                | (new_px > nx - 0.5)
                | (new_px < -0.5)
                | (new_py < 0.5)
                | (new_py > ny - 0.5)
            )
            n_out = int((left & ~hit).sum().item())
            self.aud["exited"] += n_out * m_p

            self.px = new_px
            self.py = new_py
            keep = ~(hit | left)
            self.age = self.age + 1
            too_old = self.age > self.max_age
            self.aud["trapped"] += int((too_old & keep).sum().item()) * m_p
            keep &= ~too_old
            self.px = self.px[keep]
            self.py = self.py[keep]
            self.vx = self.vx[keep]
            self.vy = self.vy[keep]
            self.age = self.age[keep]

    def _freeze(self) -> int:
        cfg = self.cfg
        freeze = (self.m_w >= cfg.m_cell_ice) & (~self.solid)
        n = int(freeze.sum().item())
        if n > 0:
            self.m_w[freeze] -= cfg.m_cell_ice
            self.solid |= freeze
            self.aud["frozen"] += n * cfg.m_cell_ice
        return n

    # -- main loop -------------------------------------------------------
    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        log = self.log
        torch.manual_seed(cfg.seed)

        log(
            f"  [icing] chord={self.chord:.0f} lu AoA={cfg.aoa_deg} deg "
            f"airfoil={int(self.airfoil.sum().item())} cells LE=({self.x_le},{self.y_le})"
        )
        log(
            f"  [icing] dx={cfg.dx_phys:.4e} m dt={cfg.dt_phys:.4e} s "
            f"St={cfg.stokes:.4f} tau_d={cfg.tau_d_lu:.1f} steps "
            f"Re_lu={cfg.re_lu:.0f}"
        )
        log(
            f"  [icing] k(accel)={cfg.lwc_accel:.3e} "
            f"phys.drops/step={cfg.droplets_per_step:.3e} "
            f"parcels/step={cfg.parcels_per_step:.2f} (x{cfg.effective_parcel_multiplier}) "
            f"t_equiv={cfg.t_equiv:.2f} s (target {cfg.t_exposure:.0f} s, "
            f"realtime would need {cfg.t_exposure / cfg.dt_phys:.2e} steps)"
        )

        # ---- flow warmup on the clean airfoil ----
        if not cfg.uniform_flow:
            for _ in range(cfg.warmup_steps):
                last = _ == cfg.warmup_steps - 1
                f_pre = self._flow_step(want_force=last)
                if last and f_pre is not None:
                    self.cd0, self.cl0 = self._force_coeffs(f_pre)
            if self.cd0 is not None:
                log(f"  [icing] warmup done: cd0={self.cd0:.5f} cl0={self.cl0:.5f}")

        ux_c = torch.full((self.ny, self.nx), cfg.u_in, device=self.dev)
        uy_c = torch.zeros((self.ny, self.nx), device=self.dev)
        beta_w0 = cfg.steps - int(cfg.beta_window_frac * cfg.steps)
        impact_w0: torch.Tensor | None = None

        if cfg.prefill_cloud and not cfg.disable_droplets:
            if cfg.uniform_flow:
                ux0, uy0 = ux_c, uy_c
            else:
                _, ux3, uy3, _ = macroscopic3d(self.f)
                ux0, uy0 = ux3[0], uy3[0]
            self._prefill(ux0, uy0)

        for step in range(1, cfg.steps + 1):
            want_force = (
                step == 1 or step == cfg.steps or step % cfg.log_every == 0 or step == beta_w0
            )
            if cfg.uniform_flow:
                f_pre = None
            else:
                f_pre = self._flow_step(want_force=want_force)
            if not cfg.disable_droplets:
                if cfg.uniform_flow:
                    ux, uy = ux_c, uy_c
                else:
                    _, ux3, uy3, _ = macroscopic3d(self.f)
                    ux, uy = ux3[0], uy3[0]
                self._droplet_step(ux, uy)
                self._freeze()

            if impact_w0 is None and step >= beta_w0:
                impact_w0 = self.impact_mass.clone()

            if want_force and f_pre is not None:
                cd, cl = self._force_coeffs(f_pre)
                self.cd_end, self.cl_end = cd, cl
            n_ice = int(self.solid.sum().item()) - int(self.airfoil.sum().item())
            t_phys = step * cfg.dt_phys * cfg.lwc_accel
            self.history["step"].append(step)
            self.history["t_phys"].append(t_phys)
            self.history["cd"].append(self.cd_end)
            self.history["cl"].append(self.cl_end)
            self.history["ice_cells"].append(n_ice)
            self.history["drops_alive"].append(int(self.px.numel()))
            if step % cfg.log_every == 0 or step == cfg.steps:
                cd_s = "n/a" if self.cd_end is None else f"{self.cd_end:.5f}"
                log(
                    f"  [icing] step {step:5d}/{cfg.steps} t_eq={t_phys:8.2f} s "
                    f"cd={cd_s} ice={n_ice:4d} drops={int(self.px.numel()):7d} "
                    f"impacts={self.n_impacts}"
                )

        # ---- final audit ----
        self.aud["airborne"] = float(self.px.numel()) * cfg.m_parcel
        self.aud["pending"] = float(self.m_w.double().sum().item())
        accounted = (
            self.aud["frozen"] + self.aud["exited"] + self.aud["trapped"]
            + self.aud["airborne"] + self.aud["pending"]
        )
        err = abs(self.aud["seeded"] - accounted) / self.aud["seeded"] if self.aud["seeded"] else 0.0
        self.aud["closure_error"] = err

        # ---- beta + metrics ----
        airfoil_np = self.airfoil.cpu().numpy()
        solid_np = self.solid.cpu().numpy()
        s_grid, stag, _surf = surface_arc_length(airfoil_np)
        dm = self.impact_mass.clone()
        if impact_w0 is not None:
            dm = dm - impact_w0
        # lattice time of the beta window (acceleration cancels, see docstring)
        t_win = (cfg.steps - beta_w0) * cfg.dt_phys
        beta = collection_efficiency_curve(
            s_grid, dm.cpu().numpy(), cfg.lwc_eff, cfg.v_inf, cfg.dx_phys,
            cfg.chord_phys, t_win,
        )
        metrics = ice_shape_metrics(
            airfoil_np, solid_np, cfg.dx_phys, cfg.chord_phys, cfg.chord_lu, stag
        )
        impact_np = self.impact_mass.cpu().numpy()
        if impact_np.any():
            metrics["max_impact_x_frac"] = float(
                (np.nonzero(impact_np)[1].max() - self.x_le) / cfg.chord_lu
            )
        cd_drift = None
        if self.cd0 is not None and self.cd_end is not None and self.cd0 != 0:
            cd_drift = 100.0 * (self.cd_end - self.cd0) / self.cd0

        return {
            "config": cfg,
            "mapping": cfg.mapping_report(),
            "airfoil": airfoil_np,
            "solid": solid_np,
            "ice_only": solid_np & ~airfoil_np,
            "m_w": self.m_w.cpu().numpy(),
            "impact_mass": impact_np,
            "s_grid": s_grid,
            "stag": stag,
            "kill_x": self.kill_x,
            "beta": beta,
            "beta_grid": (dm / (cfg.lwc_eff * cfg.v_inf * cfg.dx_phys**2 * t_win)).cpu().numpy(),
            "metrics": metrics,
            "audit": dict(self.aud),
            "history": self.history,
            "cd0": self.cd0,
            "cl0": self.cl0,
            "cd_end": self.cd_end,
            "cl_end": self.cl_end,
            "cd_drift_pct": cd_drift,
            "n_impacts": self.n_impacts,
        }


def mass_audit_report(result: dict[str, Any]) -> str:
    a = result["audit"]
    lines = [
        f"seeded  = {a['seeded']:.6e} kg",
        f"frozen  = {a['frozen']:.6e} kg",
        f"exited  = {a['exited']:.6e} kg",
        f"trapped = {a['trapped']:.6e} kg",
        f"airborne= {a['airborne']:.6e} kg",
        f"pending = {a['pending']:.6e} kg",
        f"closure error = {a['closure_error'] * 100:.4f} %",
    ]
    return "\n".join(lines)


def run_rime_icing(cfg: IcingConfig, log: Any = print) -> dict[str, Any]:
    """Convenience wrapper: build and run the simulation, return results."""
    return RimeIcingSimulation(cfg, log=log).run()


# ---------------------------------------------------------------------------
# Artifacts: ice profile, CSV, PNG, JSON, NPZ
# ---------------------------------------------------------------------------
def ice_profile(airfoil: np.ndarray, ice_only: np.ndarray) -> dict[str, np.ndarray]:
    """Per-column (streamwise) ice extent for CSV export."""
    ny, nx = airfoil.shape
    ys_a = np.full(nx, -1, dtype=np.int64)
    for x in range(nx):
        col = np.nonzero(airfoil[:, x])[0]
        if len(col):
            ys_a[x] = col.min()
    up = np.zeros(nx, dtype=np.int64)
    lo = np.zeros(nx, dtype=np.int64)
    for x in range(nx):
        col = np.nonzero(ice_only[:, x])[0]
        if not len(col) or ys_a[x] < 0:
            continue
        up[x] = max(0, int((col < ys_a[x]).sum()))
        lo[x] = max(0, int((col > ys_a[x]).sum()))
    return {"x": np.arange(nx), "upper_cells": up, "lower_cells": lo}


def save_icing_artifacts(result: dict[str, Any], out_dir: str) -> dict[str, str]:
    """Write ice-shape CSV/PNG, beta CSV/PNG, history PNG, JSON and NPZ.

    matplotlib is imported lazily so the physics module stays usable
    without it.
    """
    import json
    import os
    from pathlib import Path

    cfg = result["config"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    # --- CSV: beta curve ---
    beta = result["beta"]
    p = out / "beta_curve.csv"
    with open(p, "w") as fh:
        fh.write("s_over_c,beta,n_cells\n")
        for s, b, n in zip(beta["s_over_c"], beta["beta"], beta["n_cells"]):
            fh.write(f"{s:.6f},{b:.6f},{int(n)}\n")
    files["beta_csv"] = str(p)

    # --- CSV: ice profile ---
    prof = ice_profile(result["airfoil"], result["ice_only"])
    x_le = float(result["metrics"].get("x_le", 0.0))
    p = out / "ice_profile.csv"
    with open(p, "w") as fh:
        fh.write("x_lu,x_over_c,upper_cells,lower_cells\n")
        for x, u, l in zip(prof["x"], prof["upper_cells"], prof["lower_cells"]):
            if u or l:
                fh.write(
                    f"{int(x)},{(x - x_le) / cfg.chord_lu:.6f},{int(u)},{int(l)}\n"
                )
    files["profile_csv"] = str(p)

    # --- JSON: mapping + audit + metrics ---
    p = out / "icing_metrics.json"
    blob = {
        "mapping": result["mapping"],
        "audit": result["audit"],
        "metrics": result["metrics"],
        "cd0": result["cd0"],
        "cl0": result["cl0"],
        "cd_end": result["cd_end"],
        "cl_end": result["cl_end"],
        "cd_drift_pct": result["cd_drift_pct"],
        "beta_max": float(beta["beta"].max()) if len(beta["beta"]) else 0.0,
        "n_impacts": result["n_impacts"],
    }
    with open(p, "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    files["json"] = str(p)

    # --- NPZ: everything extract_ice_shape.py needs ---
    p = out / "result.npz"
    np.savez_compressed(
        p,
        airfoil=result["airfoil"],
        solid=result["solid"],
        m_w=result["m_w"],
        impact_mass=result["impact_mass"],
        s_grid=result["s_grid"],
        beta_grid=result["beta_grid"],
        stag=np.array(result["stag"]),
        hist_step=np.array(result["history"]["step"], dtype=np.float64),
        hist_t=np.array(result["history"]["t_phys"], dtype=np.float64),
        hist_cd=np.array([np.nan if v is None else v for v in result["history"]["cd"]]),
        hist_cl=np.array([np.nan if v is None else v for v in result["history"]["cl"]]),
        hist_ice=np.array(result["history"]["ice_cells"], dtype=np.float64),
    )
    files["npz"] = str(p)

    # --- PNGs ---
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return files

    x_le = result["metrics"].get("x_le", 0)
    chord = cfg.chord_lu
    ny, nx = result["airfoil"].shape

    # 1. ice shape over airfoil
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    yy, xx = np.mgrid[0:ny, 0:nx]
    ax.pcolormesh(xx, yy, result["airfoil"].astype(float), cmap="gray", vmin=0, vmax=2)
    ax.pcolormesh(xx, yy, result["ice_only"].astype(float) * 2.0, cmap="cool", vmin=0, vmax=2)
    m = result["metrics"]
    ax.set_xlim(x_le - 0.15 * chord, x_le + 0.55 * chord)
    ax.set_aspect("equal")
    ax.set_title(
        f"Rime ice, t_eq={cfg.t_equiv:.0f} s (LWC_eff={cfg.lwc_eff * 1e3:.2f} g/m3, "
        f"St={cfg.stokes:.3f}, AoA={cfg.aoa_deg:.0f} deg)\n"
        f"ice cells={m.get('n_ice_cells', 0)}, "
        f"horns U/L={m.get('upper_horn_pct_chord', 0):.1f}/{m.get('lower_horn_pct_chord', 0):.1f}%c"
    )
    ax.set_xlabel("x [lu]")
    ax.set_ylabel("y [lu]")
    p = out / "ice_shape.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files["ice_png"] = str(p)

    # 2. beta curve
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if len(beta["s_over_c"]):
        up = beta["s_over_c"] >= 0
        ax.plot(beta["s_over_c"][up] * 100, beta["beta"][up], "o-", label="upper", ms=3)
        ax.plot(beta["s_over_c"][~up] * 100, beta["beta"][~up], "s-", label="lower", ms=3)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("surface arc from LE s/c [% chord] (+ upper)")
    ax.set_ylabel("collection efficiency beta [-]")
    bmax = float(beta["beta"].max()) if len(beta["beta"]) else 0.0
    ax.set_title(
        f"Collection efficiency (beta_max={bmax:.3f}, St={cfg.stokes:.3f})\n"
        "rime: inertia-driven impingement, no bias terms"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    p = out / "beta_curve.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files["beta_png"] = str(p)

    # 3. history (cd drift + ice growth)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    t = np.array(result["history"]["t_phys"])
    cd = np.array([np.nan if v is None else v for v in result["history"]["cd"]])
    cl = np.array([np.nan if v is None else v for v in result["history"]["cl"]])
    axes[0].plot(t, cd, label="cd")
    axes[0].plot(t, cl, label="cl")
    axes[0].set_xlabel("equivalent physical time [s]")
    axes[0].set_title(
        f"aero response (cd drift {result['cd_drift_pct']:+.1f}%)"
        if result["cd_drift_pct"] is not None
        else "aero response (uniform-flow mode)"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, np.array(result["history"]["ice_cells"]))
    axes[1].set_xlabel("equivalent physical time [s]")
    axes[1].set_ylabel("ice cells")
    axes[1].set_title("ice growth")
    axes[1].grid(alpha=0.3)
    p = out / "history.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files["history_png"] = str(p)

    return files
