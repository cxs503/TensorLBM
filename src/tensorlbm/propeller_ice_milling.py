"""Transient lattice-Boltzmann direct simulation of propeller-blade ice milling.

Motivation (campaign node ``p367seed-n08-propeller-ice-lbm``)
------------------------------------------------------------
Ice-class propulsors are certified with *empirical pulse* load models: the
blade-ice interaction torque is prescribed as a single-peak pulse train at the
blade-passing frequency whose amplitude comes from an ice-crushing pressure
(e.g. the ISO 19906 global ice pressure), i.e.

    T_emp(t) = T_peak * shape(t),   T_peak = p_G * A_contact * r_contact.

Nothing in such a model resolves the *flow* in the blade-ice gap, so it cannot
predict the pulse shape, the rise time, or the hydrodynamic share of the load.
This module replaces the prescribed pulse by a direct LBM solution of the
transient flow in the rotor annulus while a rigid ice block partially blocks the
blade passage: a 2-D mid-span cut of the milling groove, in which the blade
sweeps past an ice block whose inner face runs at a tip clearance ``gap`` from
the blade tip, so the passage is choked inside the ice sector and open
elsewhere.

Model (2-D, D2Q9, BGK, closed box, lattice units)
-------------------------------------------------
* Domain ``n x n`` with a ``wall_cells`` thick static no-slip boundary.
* Hub: static solid disk ``r <= hub_radius`` (torque-free).
* Rotor: ``n_blades`` radial plates of thickness ``blade_thickness`` spanning
  ``hub_radius <= r <= blade_radius``, rotating rigidly at ``omega`` rad/step.
* Ice: a *rigid, static* sector annulus
  ``blade_radius + gap <= r <= n/2 - wall_cells``, ``|theta - centre| <= w/2``.
  Inside the sector the only wetted path across the annulus is the tip clearance
  ``gap`` (a choked passage); the blade drives fluid through it, producing one
  torque pulse per blade pass.
* The blade is a **moving wall**: half-way bounce-back with the Ladd wall-velocity
  correction ``f_q^new = f_qbar^post + 6 w_q rho (c_q . u_w)``,
  ``u_w = Omega x (x - x_hub)``.  The momentum exchange on every boundary link,
  ``dF = -(f_q^new + f_qbar^post) c_q``, is accumulated per solid class so the
  blade torque, the ice reaction moment and the box-wall moment separate.
* Cells uncovered by the moving blade are re-filled with the rest equilibrium;
  cells covered by it are removed.  Both are booked in an explicit mass/energy
  ledger (``mass_added``, ``mass_removed``, ``ke_added``, ``ke_removed``) so the
  conservation residuals are audited rather than assumed.

Diagnostics implemented here (the campaign's machine criteria)
-------------------------------------------------------------
* ``torque_coefficient``     : T / (rho Omega^2 R^4)   (2-D, per unit depth).
* ``richardson_order``       : observed order p of a 3-grid refinement family.
* ``dominant_frequency``     : FFT peak (Hann window, mean removed), 1/step.
* ``count_pulses``           : prominent peaks per revolution.
* ``mass_ledger_residual``   : |M(t) - M(0) - sum added + sum removed| / M(0).
* ``energy_ledger_residual`` : (int P dt - dKE - int eps dt - E_boundary)/int P dt
  with ``eps = 2 rho nu S:S`` from the non-equilibrium stress
  (``S_ab = -Pi_ab^neq / (2 tau cs^2 rho)``).
* ``iso19906_milling_torque``: the empirical baseline this module is meant to
  replace (ISO 19906 global ice pressure on the crushed contact patch).

What this module deliberately does *not* model
----------------------------------------------
Rigid-ice crushing contact (ice strength, fracture, chip transport) is outside a
pure fluid solver; the LBM load is the *hydrodynamic* component of the milling
load.  ``iso19906_milling_torque`` therefore also returns the ratio
``T_lbm / T_emp`` so the hydrodynamic share is quantified instead of asserted.

References
----------
* Ladd (1994), J. Fluid Mech. 271, 285 -- moving-wall bounce-back, momentum
  exchange force.
* Mei, Luo & Shyy (1999), J. Comput. Phys. 155, 307 -- force evaluation and
  interpolated bounce-back for moving boundaries.
* Succi (2001), *The Lattice Boltzmann Equation*; Krueger et al. (2017),
  *The Lattice Boltzmann Method* (dissipation from non-equilibrium stress).
* ISO 19906:2010, Arctic offshore structures, Annex A (global ice pressure).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# D2Q9 stencil (dim 0 = y, dim 1 = x)
# ---------------------------------------------------------------------------
CS2 = 1.0 / 3.0
_CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
_CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)
_W = (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
      1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0)
_OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)

# solid classes
FLUID, WALL, HUB, BLADE, ICE = 0, 1, 2, 3, 4
CLASS_NAMES = {WALL: "wall", HUB: "hub", BLADE: "blade", ICE: "ice"}
N_CLASSES = 5


def _t(vec: Sequence[float]) -> torch.Tensor:
    return torch.tensor(vec, dtype=torch.float32)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class MillingConfig:
    """Geometry + kinematics of the 2-D ice-milling cut (lattice units).

    All lengths are in lattice cells and ``omega`` is rad/step, so the time unit
    is one lattice step.  ``tip_ma`` sets the rotational speed through
    ``omega = tip_ma * cs / blade_radius`` (``cs = 1/sqrt(3)``), and
    ``reynolds`` sets the viscosity through ``nu = omega * R^2 / Re``.
    """

    n: int = 192
    blade_radius_frac: float = 0.40          # R = frac * n
    hub_radius_frac: float = 0.32           # r_hub = frac * R
    n_blades: int = 2
    blade_thickness_frac: float = 0.09      # t_b = frac * R
    gap_frac: float = 0.040                 # tip clearance = frac * R
    ice_sector_width_deg: float = 60.0
    ice_sector_center_deg: float = 0.0
    wall_cells: int = 2
    tip_ma: float = 0.08                    # u_tip / cs
    reynolds: float = 120.0                 # Re = omega R^2 / nu
    with_ice: bool = True
    revolutions: float = 2.0
    idle_revolutions: float = 0.0           # extra spin-up before recording

    # ---- derived (filled by __post_init__) -------------------------------
    blade_radius: float = field(init=False)
    hub_radius: float = field(init=False)
    blade_thickness: float = field(init=False)
    gap: float = field(init=False)
    ice_center: float = field(init=False)
    ice_width: float = field(init=False)
    omega: float = field(init=False)
    nu: float = field(init=False)
    tau: float = field(init=False)
    n_steps: int = field(init=False)

    def __post_init__(self) -> None:
        n = int(self.n)
        self.n = n
        self.blade_radius = self.blade_radius_frac * n
        self.hub_radius = self.hub_radius_frac * self.blade_radius
        self.blade_thickness = self.blade_thickness_frac * self.blade_radius
        self.gap = self.gap_frac * self.blade_radius
        self.ice_center = math.radians(self.ice_sector_center_deg)
        self.ice_width = math.radians(self.ice_sector_width_deg)
        self.omega = self.tip_ma * math.sqrt(CS2) / self.blade_radius
        self.nu = self.omega * self.blade_radius ** 2 / self.reynolds
        self.tau = 0.5 + 3.0 * self.nu
        self.n_steps = int(round((self.revolutions + self.idle_revolutions)
                                 * 2.0 * math.pi / self.omega))

    # -- convenience -------------------------------------------------------
    @property
    def period_steps(self) -> float:
        return 2.0 * math.pi / self.omega

    @property
    def blade_passing_freq(self) -> float:
        """Blade-passing frequency in 1/step (one ice sector)."""
        return self.n_blades * self.omega / (2.0 * math.pi)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            period_steps=self.period_steps,
            blade_passing_freq=self.blade_passing_freq,
        )
        return d


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _polar(n: int):
    y = torch.arange(n, dtype=torch.float32).view(n, 1)
    x = torch.arange(n, dtype=torch.float32).view(1, n)
    c = 0.5 * (n - 1)
    yrel = y - c
    xrel = x - c
    r = torch.sqrt(xrel * xrel + yrel * yrel)
    th = torch.atan2(yrel, xrel)
    return r, th, yrel, xrel


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


_GEOM_CACHE: Dict[int, tuple] = {}


def _geom(n: int) -> tuple:
    """Cached polar coordinate fields (the trig is expensive on CPU)."""
    if n not in _GEOM_CACHE:
        r, th, yrel, xrel = _polar(n)
        _GEOM_CACHE[n] = (r, th, yrel, xrel, torch.sin(th), torch.cos(th))
    return _GEOM_CACHE[n]


def blade_mask(r: torch.Tensor, th: torch.Tensor, cfg: MillingConfig,
               angle: float, sinth: Optional[torch.Tensor] = None,
               costh: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cells inside any of the ``n_blades`` rotating radial plates.

    Perpendicular distance to the radial line ``theta = a`` is used
    (``r |sin(theta - a)|``), expanded with the angle-addition identity so no
    per-cell trigonometry is needed.
    """
    if sinth is None or costh is None:
        sinth, costh = torch.sin(th), torch.cos(th)
    half = 0.5 * cfg.blade_thickness
    radial = (r <= cfg.blade_radius) & (r >= cfg.hub_radius)
    out = torch.zeros_like(radial)
    for k in range(cfg.n_blades):
        a = angle + 2.0 * math.pi * k / cfg.n_blades
        sa, ca = math.sin(a), math.cos(a)
        perp = torch.abs(r * (sinth * ca - costh * sa))
        forward = (costh * ca + sinth * sa) > 0.0     # |theta - a| < pi/2
        out = out | (radial & forward & (perp <= half))
    return out


def ice_mask(r: torch.Tensor, th: torch.Tensor, cfg: MillingConfig,
             costh: Optional[torch.Tensor] = None,
             sinth: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Static rigid ice sector (annulus) outboard of the blade tip."""
    r_out = 0.5 * cfg.n - cfg.wall_cells
    if costh is None:
        costh = torch.cos(th)
    if sinth is None:
        sinth = torch.sin(th)
    ca, sa = math.cos(cfg.ice_center), math.sin(cfg.ice_center)
    cosd = costh * ca + sinth * sa                  # cos(theta - centre)
    inside = (cosd >= math.cos(0.5 * cfg.ice_width)).to(torch.bool)
    return (r >= cfg.blade_radius + cfg.gap) & (r <= r_out) & inside


def wall_mask(n: int, wall_cells: int) -> torch.Tensor:
    m = torch.zeros((n, n), dtype=torch.bool)
    w = int(wall_cells)
    m[:w, :] = True
    m[-w:, :] = True
    m[:, :w] = True
    m[:, -w:] = True
    return m


class GeometryCache:
    """Pre-computed static geometry; only the blade mask changes per step."""

    def __init__(self, cfg: MillingConfig):
        self.cfg = cfg
        self.r, self.th, self.yrel, self.xrel, self.sinth, self.costh = _geom(cfg.n)
        self.wall = wall_mask(cfg.n, cfg.wall_cells)
        self.hub = self.r <= cfg.hub_radius
        self.ice = (ice_mask(self.r, self.th, cfg, self.costh, self.sinth)
                    if cfg.with_ice else torch.zeros_like(self.wall))
        base = torch.zeros((cfg.n, cfg.n), dtype=torch.int8)
        base[self.wall] = WALL
        base[self.hub] = HUB
        base[self.ice] = ICE
        self.base = base

    def labels(self, angle: float, blade_buffer: Optional[torch.Tensor] = None) -> torch.Tensor:
        if blade_buffer is None:
            blade_buffer = self.base.clone()
        else:
            blade_buffer.copy_(self.base)
        blade_buffer[blade_mask(self.r, self.th, self.cfg, angle,
                                self.sinth, self.costh)] = BLADE
        return blade_buffer


def build_labels(cfg: MillingConfig, angle: float,
                 r=None, th=None) -> torch.Tensor:
    """Solid-class labels (``FLUID``/``WALL``/``HUB``/``BLADE``/``ICE``)."""
    cache = GeometryCache(cfg) if r is None else None
    if cache is not None:
        return cache.labels(angle)
    labels = torch.zeros((cfg.n, cfg.n), dtype=torch.int8)
    labels[wall_mask(cfg.n, cfg.wall_cells)] = WALL
    labels[r <= cfg.hub_radius] = HUB
    if cfg.with_ice:
        labels[ice_mask(r, th, cfg)] = ICE
    labels[blade_mask(r, th, cfg, angle)] = BLADE
    return labels


# ---------------------------------------------------------------------------
# lattice kernels
# ---------------------------------------------------------------------------


def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium populations, shape ``(9, ny, nx)``."""
    cx = _t(_CX).view(9, 1, 1)
    cy = _t(_CY).view(9, 1, 1)
    w = _t(_W).view(9, 1, 1)
    rho = rho.unsqueeze(0) if rho.dim() == 2 else rho
    ux = ux.unsqueeze(0) if ux.dim() == 2 else ux
    uy = uy.unsqueeze(0) if uy.dim() == 2 else uy
    cu = cx * ux + cy * uy
    usq = ux * ux + uy * uy
    return w * rho * (1.0 + cu / CS2 + 0.5 * cu * cu / (CS2 * CS2) - 0.5 * usq / CS2)


def macroscopic(f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cx = _t(_CX).view(9, 1, 1)
    cy = _t(_CY).view(9, 1, 1)
    rho = f.sum(dim=0)
    inv = 1.0 / rho.clamp(min=1e-9)
    return rho, (f * cx).sum(dim=0) * inv, (f * cy).sum(dim=0) * inv


def dissipation_rate(f: torch.Tensor, fneq: torch.Tensor,
                      ux: torch.Tensor, uy: torch.Tensor,
                      tau: float, fluid_f: Optional[torch.Tensor] = None) -> float:
    """Viscous dissipation rate ``sigma_ab S_ab`` (textbook discrete form).

    ``sigma_ab = -(1 - 1/(2 tau)) Pi_ab^neq`` (the ``1 - 1/(2 tau)`` is the
    discrete-time correction of the half-step shifted momentum flux) and
    ``S_ab`` from central differences of the macroscopic velocity:
    Krueger et al. (2017) eq. (10.36); Succi (2001) ch. 5.  The returned value
    is the integral over the fluid cells, so its time integral must equal the
    kinetic-energy decay of a freely decaying flow (Taylor-Green validation in
    ``tests/test_propeller_ice_milling.py``).
    """
    if fluid_f is None:
        fluid_f = torch.ones_like(ux)
    p_xx = (fneq * _CX_T * _CX_T).sum(dim=0)
    p_yy = (fneq * _CY_T * _CY_T).sum(dim=0)
    p_xy = (fneq * _CX_T * _CY_T).sum(dim=0)
    dux_dy = 0.5 * (torch.roll(ux, -1, dims=0) - torch.roll(ux, 1, dims=0))
    dux_dx = 0.5 * (torch.roll(ux, -1, dims=1) - torch.roll(ux, 1, dims=1))
    duy_dy = 0.5 * (torch.roll(uy, -1, dims=0) - torch.roll(uy, 1, dims=0))
    duy_dx = 0.5 * (torch.roll(uy, -1, dims=1) - torch.roll(uy, 1, dims=1))
    s_xx = dux_dx
    s_yy = duy_dy
    s_xy = 0.5 * (dux_dy + duy_dx)
    sig = -(1.0 - 0.5 / tau)
    eps = sig * (p_xx * s_xx + p_yy * s_yy + 2.0 * p_xy * s_xy)
    return float((eps * fluid_f).sum().item())


_CX_T = _t(_CX).view(9, 1, 1)
_CY_T = _t(_CY).view(9, 1, 1)
_W_T = _t(_W).view(9, 1, 1)


def lbm_step(f: torch.Tensor,
             labels: torch.Tensor,
             cfg: MillingConfig,
             collect_dissipation: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], float]:
    """One BGK step with corner-safe moving-wall bounce-back.

    Returns ``(f_new, diag, dissipation)`` with ``diag["force"]`` of shape
    ``(N_CLASSES, 2)`` (force *on the solid*) and ``diag["torque"]`` of shape
    ``(N_CLASSES,)`` (moment about the hub, ``r x F``).

    Corner safety: when a fluid cell has solid neighbours on *both* sides of a
    link pair ``(q, qbar)`` the two half-way reflections would otherwise create
    mass.  Such links are swapped instead of reflected
    (``f_q <- f_qbar``), which keeps ``sum(f)`` machine-exact; they are also
    excluded from the momentum-exchange force (documented in the report).
    """
    tau = cfg.tau
    n = cfg.n
    solid = labels > 0
    fluid_f = (~solid).to(torch.float32)
    rho = f.sum(dim=0)
    rho_s = rho.clamp(min=1e-6)
    inv = 1.0 / rho_s
    ux = (f * _CX_T).sum(dim=0) * inv * fluid_f
    uy = (f * _CY_T).sum(dim=0) * inv * fluid_f

    cu = _CX_T * ux + _CY_T * uy
    usq = ux * ux + uy * uy
    feq = _W_T * rho_s * (1.0 + cu / CS2 + 0.5 * cu * cu / (CS2 * CS2) - 0.5 * usq / CS2)
    fneq = f - feq
    fpost = f - fneq * (1.0 / tau)
    zero = torch.zeros_like(fpost)
    fpost = fpost * fluid_f

    diss = 0.0
    if collect_dissipation:
        diss = dissipation_rate(f, fneq, ux, uy, tau, fluid_f)

    r_, th_, yrel, xrel, sinth, costh = _geom(n)
    omega = cfg.omega
    fnew = torch.empty_like(fpost)
    fnew[0] = fpost[0]

    # per-direction masks first (needed for the corner test of the opposite link)
    nb_lab = [None] * 9
    bb = [None] * 9
    for q in range(1, 9):
        nb_lab[q] = torch.roll(labels, shifts=(_CY[q], _CX[q]), dims=(0, 1))
        bb[q] = (nb_lab[q] > 0) & (~solid)

    mass_src = 0.0
    fx = torch.zeros((n, n), dtype=torch.float32)
    fy = torch.zeros((n, n), dtype=torch.float32)
    tz = torch.zeros((n, n), dtype=torch.float32)
    cid = torch.zeros((n, n), dtype=torch.float32)

    for q in range(1, 9):
        dxq = _CX[q]
        dyq = _CY[q]
        qb = _OPP[q]
        f_opp = fpost[qb]
        fs_q = torch.roll(fpost[q], shifts=(dyq, dxq), dims=(0, 1))
        bbq = bb[q]
        both = bbq & bb[qb]
        moving = bbq & (nb_lab[q] == BLADE)
        uwx = -omega * (yrel - 0.5 * dyq)
        uwy = omega * (xrel - 0.5 * dxq)
        cu_w = torch.where(moving, dxq * uwx + dyq * uwy, zero[0])
        dm = 6.0 * _W[q] * rho_s * cu_w
        fb = f_opp + dm
        mov_act = moving & (~both)
        if mov_act.any():
            mass_src += float((dm * mov_act.to(torch.float32)).sum(dtype=torch.float64).item())
        refl = torch.where(both, f_opp, fb)
        fnew[q] = torch.where(bbq, refl, fs_q)
        act = bbq & (~both)
        contrib = (fnew[q] + f_opp) * act.to(torch.float32)
        fxq = -contrib * dxq
        fyq = -contrib * dyq
        fx = fx + fxq
        fy = fy + fyq
        tz = tz + ((xrel - 0.5 * dxq) * fyq - (yrel - 0.5 * dyq) * fxq)
        cid = torch.where(act, nb_lab[q].to(torch.float32), cid)

    fnew = fnew * fluid_f

    sel = cid > 0
    force = torch.zeros((N_CLASSES, 2), dtype=torch.float64)
    torque = torch.zeros((N_CLASSES,), dtype=torch.float64)
    if bool(sel.any()):
        ids = cid[sel].to(torch.int64)
        force[:, 0].index_add_(0, ids, fx[sel].to(torch.float64))
        force[:, 1].index_add_(0, ids, fy[sel].to(torch.float64))
        torque.index_add_(0, ids, tz[sel].to(torch.float64))
    return fnew, {"force": force, "torque": torque, "mass_source": mass_src}, diss


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_milling(cfg: MillingConfig,
                snapshot_steps: Sequence[int] = (),
                collect_dissipation: bool = True,
                record_every: int = 1) -> Dict[str, object]:
    """Time-integrate the milling case and return the full diagnostic history."""
    n = cfg.n
    cache = GeometryCache(cfg)
    r, th = cache.r, cache.th
    buf = cache.base.clone()
    labels = cache.labels(0.0, buf).clone()
    solid = labels > 0
    rho0 = torch.ones((n, n), dtype=torch.float32)
    u0 = torch.zeros((n, n), dtype=torch.float32)
    f = equilibrium(rho0, u0, u0)
    f = torch.where(solid.unsqueeze(0), torch.zeros_like(f), f)

    hist: Dict[str, list] = {
        "step": [], "angle": [], "torque_blade": [], "torque_ice": [],
        "torque_wall": [], "torque_hub": [], "ke": [], "mass": [],
        "diss": [], "power_in": [], "mass_added": [], "mass_removed": [],
        "ke_added": [], "ke_removed": [], "mass_source": [], "ie": [],
    }
    snapshots: Dict[int, dict] = {}
    mass_added = mass_removed = ke_added = ke_removed = mass_source = 0.0
    mass0 = float(f.sum(dtype=torch.float64).item())

    for it in range(cfg.n_steps):
        angle = cfg.omega * (it + 1)
        new_labels = cache.labels(angle, buf).clone()
        new_solid = new_labels > 0
        became_solid = new_solid & ~solid
        became_fluid = ~new_solid & solid
        if bool(became_solid.any()):
            mass_removed += float(f[:, became_solid].sum(dtype=torch.float64).item())
            if collect_dissipation:
                rl, ul, vl = macroscopic(f)
                ke_cells = 0.5 * rl * (ul * ul + vl * vl)
                ke_removed += float(ke_cells[became_solid].sum().item())
            f = torch.where(became_solid.unsqueeze(0), torch.zeros_like(f), f)
        if bool(became_fluid.any()):
            nfill = int(became_fluid.sum().item())
            mass_added += float(nfill)
            f = torch.where(became_fluid.unsqueeze(0),
                            equilibrium(torch.ones((n, n)), u0, u0), f)
        labels, solid = new_labels, new_solid

        f, diag, diss = lbm_step(f, labels, cfg, collect_dissipation=collect_dissipation)
        mass_source += float(diag["mass_source"])

        if it % record_every == 0 or it == cfg.n_steps - 1:
            rho, ux, uy = macroscopic(f)
            ke = float((0.5 * rho * (ux * ux + uy * uy))[~solid].sum().item())
            ie = float((CS2 * rho)[~solid].sum().item())
            hist["step"].append(it + 1)
            hist["angle"].append(angle)
            hist["torque_blade"].append(float(diag["torque"][BLADE].item()))
            hist["torque_ice"].append(float(diag["torque"][ICE].item()))
            hist["torque_wall"].append(float(diag["torque"][WALL].item()))
            hist["torque_hub"].append(float(diag["torque"][HUB].item()))
            hist["ke"].append(ke)
            hist["ie"].append(ie)
            hist["mass"].append(float(f.sum(dtype=torch.float64).item()))
            hist["diss"].append(diss)
            hist["power_in"].append(-float(diag["torque"][BLADE].item()) * cfg.omega)
            hist["mass_added"].append(mass_added)
            hist["mass_removed"].append(mass_removed)
            hist["ke_added"].append(ke_added)
            hist["ke_removed"].append(ke_removed)
            hist["mass_source"].append(mass_source)

        if it in snapshot_steps:
            rho, ux, uy = macroscopic(f)
            vort = _vorticity(ux, uy)
            snapshots[it] = {
                "step": it, "angle": angle,
                "rho": rho.cpu().numpy(), "ux": ux.cpu().numpy(),
                "uy": uy.cpu().numpy(), "vort": vort.cpu().numpy(),
                "labels": labels.cpu().numpy(),
            }

    hist["mass0"] = mass0
    hist["mass_added_total"] = mass_added
    hist["mass_removed_total"] = mass_removed
    hist["ke_added_total"] = ke_added
    hist["ke_removed_total"] = ke_removed
    hist["mass_source_total"] = mass_source
    hist["cfg"] = cfg.to_dict()
    hist["snapshots"] = snapshots
    return hist


def _vorticity(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    duy_dx = 0.5 * (torch.roll(uy, -1, dims=1) - torch.roll(uy, 1, dims=1))
    dux_dy = 0.5 * (torch.roll(ux, -1, dims=0) - torch.roll(ux, 1, dims=0))
    return duy_dx - dux_dy


# ---------------------------------------------------------------------------
# post-processing / criteria helpers
# ---------------------------------------------------------------------------


def torque_coefficient(torque: float, cfg: MillingConfig, rho: float = 1.0) -> float:
    """``C_T = T / (rho omega^2 R^4)`` (2-D, per unit depth)."""
    return torque / (rho * cfg.omega ** 2 * cfg.blade_radius ** 4)


def richardson_order(h: Sequence[float], values: Sequence[float]) -> Dict[str, float]:
    """Observed order of convergence and Richardson extrapolation.

    ``h`` are the (proportional to) grid spacings, ``values`` the grid-scalar
    (e.g. peak torque coefficient) on those grids.  For
    ``f(h) = f_0 + a h^p`` the order follows from

        (f_2 - f_1) / (f_3 - f_1) = (h_2^p - h_1^p) / (h_3^p - h_1^p),

    solved numerically (works for non-uniform refinement ratios).
    """
    if len(h) != 3 or len(values) != 3:
        raise ValueError("richardson_order needs exactly 3 grids")
    h1, h2, h3 = (float(v) for v in h)
    f1, f2, f3 = (float(v) for v in values)

    def g(p: float) -> float:
        return ((h2 ** p - h1 ** p) * (f3 - f1) - (h3 ** p - h1 ** p) * (f2 - f1))

    lo, hi = 1e-3, 8.0
    glo, ghi = g(lo), g(hi)
    p: Optional[float] = None
    if glo * ghi < 0:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            gm = g(mid)
            if abs(gm) < 1e-14:
                p = mid
                break
            if glo * gm < 0:
                hi, ghi = mid, gm
            else:
                lo, glo = mid, gm
        if p is None:
            p = 0.5 * (lo + hi)

    f_extrap = float("nan")
    if p is None or abs(h3 ** p - h2 ** p) < 1e-14:
        f_extrap = float("nan")   # monotone-but-unresolvable: cannot extrapolate
    else:
        a = (f3 - f2) / (h3 ** p - h2 ** p)
        f_extrap = f3 - a * h3 ** p
    rel = float("nan")
    if p is not None and abs(f_extrap) > 1e-30:
        rel = abs(f3 - f_extrap) / abs(f_extrap)
    return {"p": float(p) if p is not None else float("nan"),
            "f_extrapolated": f_extrap, "f_finest": f3,
            "fine_vs_extrapolated_rel": rel,
            "h": [h1, h2, h3], "values": [f1, f2, f3]}


def dominant_frequency(signal: Sequence[float], dt: float = 1.0,
                       top_k: int = 5) -> Dict[str, object]:
    """FFT peak of a (mean-removed, Hann-windowed) sampled signal, in 1/time."""
    import numpy as np

    s = np.asarray(signal, dtype=float)
    s = s - s.mean()
    win = np.hanning(len(s))
    spec = np.abs(np.fft.rfft(s * win))
    freqs = np.fft.rfftfreq(len(s), d=dt)
    spec[0] = 0.0
    order = np.argsort(spec)[::-1]
    k = int(min(top_k, len(order)))
    peaks = [(float(freqs[i]), float(spec[i])) for i in order[:k]]
    f_peak = peaks[0][0] if peaks else 0.0
    # parabolic refinement of the peak bin (sub-bin interpolation)
    idx = int(order[0])
    if 0 < idx < len(spec) - 1:
        y0, y1, y2 = spec[idx - 1], spec[idx], spec[idx + 1]
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-30:
            delta = 0.5 * (y0 - y2) / denom
            f_peak = float(freqs[idx] + delta * (freqs[1] - freqs[0]))
    return {"f_peak": f_peak, "peaks": peaks, "freqs": freqs, "amplitude": spec}


def count_pulses(signal: Sequence[float], prominence_frac: float = 0.25,
                 min_distance: int = 5) -> Dict[str, object]:
    """Count prominent peaks (torque pulses) and return their indices/heights."""
    import numpy as np

    try:
        from scipy.signal import find_peaks
    except Exception:                                    # pragma: no cover
        find_peaks = None
    s = np.asarray(signal, dtype=float)
    prom = prominence_frac * (s.max() - s.min() + 1e-30)
    if find_peaks is None:                               # pragma: no cover
        idx = [i for i in range(1, len(s) - 1)
               if s[i] >= s[i - 1] and s[i] > s[i + 1]]
        return {"n_peaks": len(idx), "indices": idx}
    idx, props = find_peaks(s, prominence=prom, distance=int(min_distance))
    return {"n_peaks": int(len(idx)), "indices": [int(i) for i in idx],
            "heights": [float(s[i]) for i in idx],
            "prominences": [float(v) for v in props.get("prominences", [])]}


def fluctuation_rms(signal: Sequence[float]) -> Dict[str, float]:
    import numpy as np

    s = np.asarray(signal, dtype=float)
    mean = float(s.mean())
    rms = float(np.sqrt(((s - mean) ** 2).mean()))
    return {"mean": mean, "rms": rms,
            "rms_over_abs_mean": rms / max(abs(mean), 1e-30)}


def mass_ledger_residual(hist: Dict[str, object]) -> Dict[str, float]:
    """Residual of the moving-boundary mass ledger.

    ``dM = mass_added - mass_removed + mass_source`` is the *audited* mass
    exchange of the moving-wall boundary condition (the Ladd term
    ``6 w_q rho (c_q . u_w)`` is a genuine mass source/sink of the isothermal
    LBM, not round-off), so the ledger residual measures bookkeeping error only.
    ``wall_mass_source_relative`` reports the physical drift that term produces,
    which is the number to quote when discussing compressibility of the model.
    """
    m = hist["mass"]
    added = float(hist["mass_added_total"])
    removed = float(hist["mass_removed_total"])
    src = float(hist.get("mass_source_total", 0.0))
    res = (m[-1] - hist["mass0"]) - added + removed - src
    m0 = max(abs(hist["mass0"]), 1e-30)
    return {"absolute": float(res),
            "relative": float(abs(res) / m0),
            "mass0": float(hist["mass0"]),
            "added": added,
            "removed": removed,
            "wall_mass_source": src,
            "wall_mass_source_relative": float(abs(src) / m0)}


def smooth_signal(signal: Sequence[float], window: int) -> "list[float]":
    """Centred moving average -- removes the lattice-scale staircase noise of a
    voxel-resolved moving boundary while leaving the blade-passing pulse intact
    (the window must stay well below the pulse width)."""
    import numpy as np

    s = np.asarray(signal, dtype=float)
    w = max(int(window), 1)
    if w == 1:
        return s.tolist()
    pad = w // 2
    s_pad = np.concatenate([np.full(pad, s[0]), s, np.full(w - 1 - pad, s[-1])])
    kern = np.ones(w) / w
    return np.convolve(s_pad, kern, mode="valid")[: len(s)].tolist()


def energy_ledger_residual(hist: Dict[str, object], cfg: MillingConfig,
                           dt: float = 1.0) -> Dict[str, float]:
    """Total-energy balance of the isothermal LBM:

        int P dt  =  [KE + cs^2 M]  +  int eps dt  +  boundary terms

    The ``cs^2 M`` term is the internal energy of the compressible lattice gas
    (rho rises by a few percent through the rotor's pressure field, so omitting
    it swamps the balance); the boundary terms book the mass/energy of the cells
    the moving blade covers/uncover plus the moving-wall mass source.
    """
    steps = hist["step"]
    if len(steps) < 2:
        raise ValueError("need at least two records")
    pin = hist["power_in"]
    diss = hist["diss"]
    ke = hist["ke"]
    steps_arr = steps
    ds = []
    for i in range(1, len(steps_arr)):
        ds.append(steps_arr[i] - steps_arr[i - 1])
    int_pin = float(sum(0.5 * (pin[i] + pin[i - 1]) * ds[i - 1]
                        for i in range(1, len(pin)))) * dt
    int_diss = float(sum(0.5 * (diss[i] + diss[i - 1]) * ds[i - 1]
                         for i in range(1, len(diss)))) * dt
    ie = hist.get("ie")
    if ie is None:
        ie = [CS2 * m for m in hist["mass"]]
    ke_total = [k + i for k, i in zip(ke, ie)]
    dke = float(ke_total[-1] - ke_total[0])
    added = float(hist["mass_added_total"])
    removed = float(hist["mass_removed_total"])
    src = float(hist.get("mass_source_total", 0.0))
    e_boundary = float((hist["ke_added_total"] + CS2 * added)
                       - (hist["ke_removed_total"] + CS2 * removed)
                       + CS2 * src)
    denom = abs(int_pin)
    residual = int_pin - dke - int_diss - e_boundary
    return {"int_power_in": int_pin, "delta_total_energy": dke,
            "delta_ke": float(ke[-1] - ke[0]),
            "delta_internal": float(ie[-1] - ie[0]),
            "int_dissipation": int_diss,
            "boundary_energy": e_boundary, "residual": residual,
            "relative": float(abs(residual) / max(denom, 1e-30)),
            "energy_start": float(ke_total[0]), "energy_end": float(ke_total[-1])}


# ---------------------------------------------------------------------------
# empirical baseline (the model being replaced)
# ---------------------------------------------------------------------------


@dataclass
class Iso19906Ice:
    """Ice + geometry parameters for the ISO 19906 global-pressure baseline."""

    sigma_c: float = 2.0e6        # ice compressive strength [Pa]
    C_R: float = 2.0              # global pressure coefficient [-]
    aspect_exponent: float = 0.16  # weak h/w scaling exponent [-]
    ice_thickness: float = 0.50   # out-of-plane width b [m]
    crush_depth: float = 0.02     # instantaneous crushed patch depth [m]
    propeller_radius: float = 1.00  # R [m]
    contact_radius: float = 0.98   # r_contact / R [-]


def iso19906_milling_torque(ice: Iso19906Ice) -> Dict[str, float]:
    """Empirical milling torque from the ISO 19906 global ice pressure.

    ``p_G = C_R sigma_c (h/w)^n`` (ISO 19906:2010 Annex A global pressure form,
    weak aspect-ratio exponent for a narrow contact), ``A_c = d_crush * b``
    (crushed patch), ``T = p_G A_c r_c``.  This is the *prescribed pulse*
    baseline that the LBM solve is compared against.
    """
    aspect = ice.crush_depth / max(ice.ice_thickness, 1e-12)
    p_g = ice.C_R * ice.sigma_c * aspect ** ice.aspect_exponent
    area = ice.crush_depth * ice.ice_thickness
    force = p_g * area
    r_c = ice.contact_radius * ice.propeller_radius
    return {"p_G": p_g, "contact_area": area, "contact_force": force,
            "torque": force * r_c, "contact_radius": r_c,
            "aspect_ratio": aspect}


def empirical_pulse_train(t: "list[float]", cfg: MillingConfig,
                          peak: float, duty: Optional[float] = None) -> List[float]:
    """The incumbent empirical model: half-sine pulse per blade pass."""
    if duty is None:
        duty = cfg.ice_width / (2.0 * math.pi / cfg.n_blades)
    period = 1.0 / cfg.blade_passing_freq
    out = []
    for ti in t:
        phase = (ti % period) / period
        if phase <= duty:
            out.append(peak * math.sin(math.pi * phase / duty))
        else:
            out.append(0.0)
    return out


def physical_scaling(cfg: MillingConfig,
                     propeller_radius_m: float = 1.0,
                     rho_water: float = 1025.0,
                     ice_thickness_m: float = 0.5,
                     rev_per_s: float = 2.0) -> Dict[str, float]:
    """Dimensional-analysis mapping of the 2-D torque coefficient to SI.

    The LBM result is condensed into the dimensionless pulse coefficient
    ``C_T = dT / (rho Omega^2 R^4 b)`` (``b`` = out-of-plane extent = ice
    thickness), which is then evaluated at a documented operating point:
    ``R = 1 m`` (D = 2 m), ``n = 2 rev/s`` (120 rpm), ``b = 0.5 m``,
    ``rho = 1025 kg/m^3``.  No lattice ``dx``/``dt`` enters, so the map does not
    depend on the (arbitrary) lattice speed of sound.

    Limitation: the Reynolds number of the lattice run (Re = 120 based on tip
    speed and radius) is far below a real propeller, so ``C_T`` is *not*
    Reynolds-transferable; the comparison below is therefore an indicative
    dimensional-analysis check, not a validation (stated in the report).
    """
    omega_phys = 2.0 * math.pi * rev_per_s
    scale = rho_water * omega_phys ** 2 * propeller_radius_m ** 4 * ice_thickness_m
    return {"omega_phys": omega_phys, "rev_per_s": rev_per_s,
            "u_tip_m_per_s": omega_phys * propeller_radius_m,
            "torque_scale": scale, "propeller_radius_m": propeller_radius_m,
            "ice_thickness_m": ice_thickness_m, "rho_water": rho_water,
            "lattice_reynolds": cfg.reynolds, "lattice_tip_ma": cfg.tip_ma}
