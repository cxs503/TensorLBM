"""Under-ice keel drag: skin / form decomposition of C_d in a periodic channel.

Physics and scope
-----------------
The under-ice water boundary layer drags the ice cover through two mechanisms
that a drifting-ice model lumps into one coefficient:

* **skin friction** on the level-ice underside (and on the ice-side faces of the
  keels), and
* **form (pressure) drag** on the frontal/lee faces of the keels.

Observations (ice drift buoys, AIDJEX/Pacific gyre inversions) bracket the total
ice-water drag coefficient at ``C_d = 0.004 ... 0.02`` when it is defined by
``tau_w = rho_w C_d |U_rel| U_rel``.  Because ``Cf = 2 tau_w / (rho U^2)`` in the
channel convention used here, the observational band maps to
``C_d = Cf / 2``.  Nothing in this module hard-codes a target value: the band is
only used to adjudicate the computed result.

Geometry (the "keel array")
---------------------------
Keels are triangular ridges of depth ``h_keel`` and base width ``w`` repeated
with pitch ``p``; the areal keel density is ``lambda = w / p`` (fraction of the
plan area occupied by keel bases) and the shape parameter is ``h_keel / h_i``
(keel draft over level-ice thickness).  Ridges are placed at angle ``theta`` to
the free-stream x-axis, may be buckled along their crest
(``height_modulation``) and may be interrupted by leads of areal fraction
``porosity`` -- the latter is the connection to the ice-morphology generator of
the upstream node (a areally distributed ice thickness ``gamma(h -> h')`` plus
open-water porosity ``phi``): ``phi`` maps onto the lead fraction and a
distribution of ``h_keel / h_i`` maps onto a superposition of ridge fields.
Ridges are voxelised into solid cells on both walls (mirrored), so the periodic
channel has two rough ice walls and doubles the statistics.

Nondimensionalisation and the exact closure
-------------------------------------------
The channel is force driven (Guo 2002 forcing) with body force ``g`` chosen from
the target friction velocity and the fluid half-height,
``g = u_tau^2 / h_eff``, exactly as in :mod:`tensorlbm.rough_channel3d`.  The
global momentum balance of a periodic channel is then exact and independent of
the flow details,

``tau_total = rho g h_eff``,   ``h_eff = N_fluid / (2 nx ny)``,

which is used as an independent closure check on the surface integrals.

Drag decomposition (three *independent* routes, all reported)
------------------------------------------------------------
1. ``total_momentum_exchange``: sum of the halfway bounce-back momentum
   exchange over every wall-crossing link,
   ``F_x = sum_links c_ix (f*_i + f*_opp)``.  This is a numerical identity for
   the discrete scheme and is compared against ``rho g N_fluid`` (residual is
   the momentum-balance criterion).
2. ``form``: first-order wall-pressure integral over the *x-normal* faces of
   the voxelised ice surface, ``F_form,x = sum p_wall (+/-1) A`` with
   ``p_wall = rho/3`` taken from the adjacent fluid cell.  Only x-normal faces
   contribute an x-projection, so a keel field perfectly aligned with the flow
   gives ``F_form ~ 0`` (which is the physically correct answer, and is the
   negative control of the decomposition).
3. ``skin``: viscous wall shear on the *horizontal* (z-normal) wetted faces,
   ``F_skin,x = sum 2 rho nu u_x`` (one-sided difference over the half-cell to
   the bounce-back wall location).  The y-normal keel flanks carry a mean-zero
   ``tau_xy`` at homogeneity and are omitted (documented limitation).

The primary split is (2)+(3), compared against (1); ``F_form + F_skin`` must
close on ``rho g h_eff`` (criterion: <= 2 %).  A fourth, purely geometric split
from projecting each link exchange onto the local wall normal is also returned
(``split_projection``) as a cross-check that cannot by construction help the
closure.

Analytic reference
------------------
For triangular ridges the frontal area per plan area (normalised by ``h_eff``)
is ``0.5 lambda h_keel / h_eff``, so the analytic form-drag estimate is

``C_d^form = 0.5 C_D,keel (S_fr / S_plan)``

with ``C_D,keel`` the (unknown, order-one) drag coefficient of one ridge.  The
*scaling exponent* is therefore a prediction with no free parameter:
``C_d^form ~ (S_fr/S_plan) ~ lambda (h_keel/h_i)``; it is fitted from the
computed data and compared with the analytic value 1.

References
----------
* Hibler (1979) dynamic-thermodynamic sea-ice model (ice-water drag ``C_d``).
* McPhee (1980, 2008) under-ice boundary layer / "ice-ocean drag" inversions.
* Lu, Li & Lepparanta (2011) form drag of keels; Keitzl et al. (2013).
* Ladd (1994) / Mei et al. (2002) momentum-exchange wall force in LBM.
* Guo, Zheng & Shi (2002) forcing scheme for LBM.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .rough_channel3d import C, OPPOSITE, TRT_LAMBDA, W, build_link_masks
from .roughness_laws import B_SMOOTH, KAPPA, fit_log_law

try:  # torch is optional at import time (the C/numpy path only needs numpy)
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

__all__ = [
    "KeelArraySpec",
    "triangular_keel_mask",
    "keel_geometry_stats",
    "analytic_frontal_area_ratio",
    "UnderIceChannelConfig",
    "run_underice_channel",
    "cd_from_cf",
    "cf_log_law",
    "dean_cf",
    "fit_power_law",
    "OBSERVED_CD_BAND",
    "LEVEL_ICE_TO_HALF_CLEARANCE",
]

#: Observational bracket of the ice-water drag coefficient (Hibler 1979;
#: McPhee 1980/2008 drift inversions): ``tau = rho C_d U^2``.
OBSERVED_CD_BAND: tuple[float, float] = (0.004, 0.02)

#: Model anchor between the morphological scale ``h_i`` (level ice thickness)
#: and the hydrodynamic scale ``h_eff`` (half of the under-ice clearance):
#: ``h_eff = LEVEL_ICE_TO_HALF_CLEARANCE * h_i``.  12 half-clearances = 6 level
#: ice thicknesses of water column under the ice, which is the low end of an
#: Arctic mixed layer (~10-40 m for h_i = 1.5-2 m) and is the *conservative*
#: choice: a larger anchor pushes the keels deeper into a deeper layer, which
#: lowers the drag per unit plan area.  Reported as a model assumption, not as
#: a measured quantity.
LEVEL_ICE_TO_HALF_CLEARANCE: float = 12.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeelArraySpec:
    """Voxelised triangular keel field on a ``(nz, ny, nx)`` lattice.

    Parameters
    ----------
    nx, ny, nz:
        Grid size; ``z`` is wall-normal with solid wall layers at ``k = 0`` and
        ``k = nz - 1``.
    h_keel:
        Keel depth in lattice units measured from the level-ice plane.
    coverage:
        ``lambda``: fraction of the plan area occupied by keel bases.
    angle_deg:
        Ridge orientation: ``90`` = ridges span ``y`` (transverse to the flow),
        ``0`` = ridges run along ``x`` (aligned with the flow).
    porosity:
        Areal fraction of leads (ridge crests interrupted) -- the ``phi`` of the
        upstream ice-morphology generator.
    height_modulation:
        Relative crest buckling amplitude (0 = perfectly straight ridges);
        implemented as a sinusoid along the crest with zero mean, so the mean
        keel depth is preserved.
    seed:
        Deterministic seed for the ridge phase jitter.
    """

    nx: int
    ny: int
    nz: int
    h_keel: float
    coverage: float
    angle_deg: float = 90.0
    porosity: float = 0.0
    height_modulation: float = 0.25
    seed: int = 20261109
    #: number of ridge crests across the span (fixed per sweep so that every
    #: case sits in the same periodic box)
    n_ridges: int = 4

    def __post_init__(self) -> None:
        if not (0.0 < self.coverage < 1.0):
            raise ValueError("coverage (lambda) must be in (0, 1)")
        if not (0.0 <= self.porosity < 1.0):
            raise ValueError("porosity must be in [0, 1)")
        if self.h_keel < 1.0:
            raise ValueError("h_keel must be at least one lattice cell")
        if self.h_keel > 0.45 * (self.nz - 2):
            raise ValueError(
                f"h_keel={self.h_keel} exceeds 45% of the clearance {self.nz - 2}; "
                "the channel degenerates into an obstacle array"
            )


def _ridge_coordinate(spec: KeelArraySpec) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(v, u)``: across-ridge and along-ridge coordinates on the plane."""
    xs = np.arange(spec.nx, dtype=float)[None, :]
    ys = np.arange(spec.ny, dtype=float)[:, None]
    th = math.radians(float(spec.angle_deg))
    v = -math.sin(th) * xs + math.cos(th) * ys
    u = math.cos(th) * xs + math.sin(th) * ys
    return np.broadcast_to(v, (spec.ny, spec.nx)).copy(), np.broadcast_to(u, (spec.ny, spec.nx)).copy()


def triangular_keel_mask(spec: KeelArraySpec) -> np.ndarray:
    """Voxelise the triangular keel field (roughness voxels only, ``1 <= k <= nz-2``).

    One ridge period covers ``p = w / lambda`` across-ridge cells; the crest of
    each ridge sits at the centre of the period, and the depth profile is
    ``d(t) = h (1 - 2|t|/w)`` for ``|t| <= w/2`` -- a triangle with base ``w``.
    Leads of areal fraction ``porosity`` are cut as bands along the crest with
    deterministic phase.
    """
    v, u = _ridge_coordinate(spec)
    # A fixed number of ridges across the span keeps every case in one periodic
    # box statistically comparable; the *base width* then carries lambda, and the
    # realised coverage is re-measured from the voxels (reported as
    # coverage_geometric) because a one-cell quantisation cannot hit an
    # arbitrary lambda exactly.
    n_ridge = max(2.0, float(spec.n_ridges))
    pitch = spec.ny / n_ridge
    width = float(min(max(1.0, round(spec.coverage * pitch)), pitch - 1.0))
    phase = (spec.seed % 997) / 997.0 * pitch
    t = np.mod(v + phase, pitch) - 0.5 * pitch  # centred on the crest
    frac = np.clip(1.0 - 2.0 * np.abs(t) / width, 0.0, None)
    depth = spec.h_keel * frac
    if spec.height_modulation > 0.0:
        lam_u = max(4.0, n_ridge * 2.0)
        buckle = 1.0 + spec.height_modulation * np.sin(2.0 * math.pi * u / lam_u + 0.7)
        depth = depth * buckle
    depth = np.clip(depth, 0.0, 0.45 * (spec.nz - 2))

    if spec.porosity > 0.0:
        rng = np.random.default_rng(int(spec.seed))
        # leads: bands along the crest, each with probability ~ porosity
        lead = rng.random(spec.ny) < float(spec.porosity)
        depth = np.where(lead[:, None], 0.0, depth)

    mask = np.zeros((spec.nz, spec.ny, spec.nx), dtype=bool)
    k = np.arange(spec.nz)[:, None, None]
    d = np.rint(depth).astype(int)  # top-most void depth in cells
    # Keel voxels hang from the upper wall: k >= nz - 1 - d ; the mirror wall is
    # added by ``mirrored_solid`` below, so only one side is generated here.
    mask = (k >= (spec.nz - 1 - d)[None, :, :]) & (k < spec.nz - 1)
    return mask


def mirrored_solid(mask: np.ndarray, *, mirror: bool = True) -> np.ndarray:
    """Fold roughness voxels into the two channel walls (optionally mirrored)."""
    solid = np.zeros_like(mask, dtype=bool)
    solid[0] = True
    solid[-1] = True
    solid |= mask
    if mirror:
        solid |= np.flip(mask, axis=0)
    return solid


def keel_geometry_stats(spec: KeelArraySpec | None, mask: np.ndarray, solid: np.ndarray) -> dict[str, Any]:
    """Geometry audit: coverage, frontal area ratio, mean depth, face census."""
    nz, ny, nx = mask.shape
    n_fluid = int((~solid).sum())
    h_eff = n_fluid / (2.0 * nx * ny)
    depth = mask.sum(axis=(1, 2)) / float(nx * ny)  # void depth per k-slab
    mean_depth_one_wall = float(depth.sum())
    plan = float(nx * ny)
    # x-normal and y-normal faces of the (single-wall) keel surface: these are
    # the faces that project onto the drag direction.
    fr = mask & ~np.roll(mask, -1, axis=2)  # x-normal faces with +x outward
    fl = mask & ~np.roll(mask, 1, axis=2)
    fy = mask & ~np.roll(mask, -1, axis=1)
    fyb = mask & ~np.roll(mask, 1, axis=1)
    base_cells = int((mask.sum(axis=0) > 0).sum())
    stats = {
        "n_fluid": n_fluid,
        "h_eff": h_eff,
        "solid_fraction": float(solid.mean()),
        "keel_volume_fraction": float(mask.mean()),
        "coverage_geometric": base_cells / plan,
        "mean_keel_depth": mean_depth_one_wall,
        "h_keel_spec": None if spec is None else float(spec.h_keel),
        "angle_deg": None if spec is None else float(spec.angle_deg),
        "porosity": None if spec is None else float(spec.porosity),
        # frontal area seen by the x-flow (per unit plan area, in lattice cells)
        "S_fr_x_over_plan": float(fr.sum() + fl.sum()) / plan,
        "S_fr_y_over_plan": float(fy.sum() + fyb.sum()) / plan,
        "S_horizontal_over_plan": float((mask.sum(axis=0) > 0).sum()) / plan,
        "analytic_S_fr_over_plan": analytic_frontal_area_ratio(
            coverage=base_cells / plan,
            h_keel=0.0 if spec is None else spec.h_keel,
            h_eff=h_eff,
        ),
        "coverage_nominal": None if spec is None else float(spec.coverage),
        "n_ridges": None if spec is None else int(spec.n_ridges),
    }
    return stats


def analytic_frontal_area_ratio(coverage: float, h_keel: float, h_eff: float) -> float:
    """Analytic ``S_fr / (S_plan h_eff)`` for triangular keels: ``0.5 lambda h_keel``.

    A triangular ridge of base ``w`` and depth ``h`` has side-profile area
    ``0.5 w h``; with ``lambda = w / p`` ridge pitch ``p`` the frontal area per
    unit plan area is ``0.5 lambda h``, normalised here by ``h_eff`` so that the
    number is dimensionless and comparable with ``C_d``.
    """
    if h_eff <= 0.0:
        raise ValueError("h_eff must be positive")
    return 0.5 * float(coverage) * float(h_keel) / float(h_eff)


# ---------------------------------------------------------------------------
# Channel configuration and solver
# ---------------------------------------------------------------------------


def _s_minus(tau: float) -> float:
    """Second TRT relaxation rate used by this module: ``s- = s+ = 1/tau``.

    The Ginzburg magic parameter (``Lambda = 3/16``, i.e. ``s- ~ 0.02`` at
    ``tau ~ 0.5``) blows up on a *staircase keel wall* at ``Re_tau = 200``:
    the antisymmetric (ghost) modes are then almost unrelaxed and the corner
    singularity of a voxel keel feeds them without bound.  Measured on the
    reference case (16x32x34, h_keel = 3, Re_tau = 200): ``s- = 0.019`` ->
    NaN within 400 steps, ``s- = 1/tau`` -> stable (see
    ``tests/test_underice_drag.py::test_staircase_keel_stability``).
    Relaxting the antisymmetric part at the same rate is the BGK-equivalent
    limit; the penalty is the classical BGK wall-location ambiguity (a fraction
    of a lattice cell), which the momentum-balance closure is insensitive to.
    """
    return 1.0 / float(tau)


@dataclass(frozen=True)
class UnderIceChannelConfig:
    """Numerical settings of one forced under-ice channel case."""

    nx: int = 48
    ny: int = 48
    nz: int = 98
    re_tau: float = 200.0
    u_tau: float = 0.02
    steps: int = 12000
    warmup: int = 4000
    sample_interval: int = 10
    seed: int = 20261109
    threads: int = 8
    smagorinsky_cs: float = 0.0
    init_scale: float = 0.5
    noise_amplitude: float = 0.03
    note: str = ""

    def with_grid(self, nx: int, ny: int, nz: int, **kw: Any) -> "UnderIceChannelConfig":
        d = asdict(self)
        d.update(nx=nx, ny=ny, nz=nz)
        d.update(kw)
        return UnderIceChannelConfig(**d)


def cd_from_cf(cf: float) -> float:
    """Observational convection: ``tau = rho C_d U^2`` vs ``Cf = 2 tau/(rho U^2)``."""
    return 0.5 * float(cf)


def cf_log_law(re_tau: float, kappa: float = KAPPA, b: float = B_SMOOTH) -> float:
    """Channel ``Cf`` implied by the log law ``U+ = (1/kappa) ln(Re_tau) + B``.

    ``Cf = 2 / U+^2`` in the bulk convention used by the momentum balance, with
    ``U+_bulk`` approximated by the log law at the channel centre
    (``y+ = Re_tau``).  This is the "对数律" target of acceptance criterion (a).
    """
    u_plus = math.log(re_tau) / kappa + b
    return 2.0 / u_plus**2


def dean_cf(re_h: float) -> float:
    """Dean (1978) empirical channel correlation ``Cf = 0.073 Re_h^{-0.25}``.

    ``Re_h = U_b h / nu`` with ``h`` the channel half-height; valid for
    ``10^4 < Re_h < 10^5``.  Used as the second, independent "经验相关" target.
    """
    if re_h <= 0:
        raise ValueError("Re_h must be positive")
    return 0.073 * re_h ** (-0.25)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares ``log y = a + n log x``; returns ``(n, a, r2)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    lx = np.log(x[good])
    ly = np.log(y[good])
    n, a = np.polyfit(lx, ly, 1)
    resid = ly - (a + n * lx)
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return float(n), float(a), float(r2)


_CS2 = 1.0 / 3.0


def _tensors() -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch back-end unavailable; underice_drag needs torch")
    cx = torch.from_numpy(C[:, 0]).view(19, 1, 1, 1).float()
    cy = torch.from_numpy(C[:, 1]).view(19, 1, 1, 1).float()
    cz = torch.from_numpy(C[:, 2]).view(19, 1, 1, 1).float()
    w = torch.from_numpy(W).view(19, 1, 1, 1).float()
    opp = torch.from_numpy(OPPOSITE).long()
    return {"cx": cx, "cy": cy, "cz": cz, "w": w, "opp": opp}


def _equilibrium(rho: Any, ux: Any, uy: Any, uz: Any, ten: dict[str, Any]) -> Any:
    cu = ten["cx"] * ux + ten["cy"] * uy + ten["cz"] * uz
    usq = ux * ux + uy * uy + uz * uz
    return rho * ten["w"] * (1.0 + cu / _CS2 + 0.5 * cu * cu / _CS2**2 - 0.5 * usq / _CS2)


def _momentum(f: Any, ten: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    rho = f.sum(0)
    mx = (f * ten["cx"]).sum(0)
    my = (f * ten["cy"]).sum(0)
    mz = (f * ten["cz"]).sum(0)
    return rho, mx, my, mz


def _step(f: Any, solid_t: Any, links: Any, gx: Any, tau: float, s_minus: float,
          smag2: float, ten: dict[str, Any], shifts: tuple[tuple[int, int, int], ...]) -> tuple[Any, Any, Any]:
    """One TRT + Guo-forced + link-wise-halfway-bounce-back step.

    Returns ``(f_new, ux_meas, fstar)``; ``fstar`` (post-collision state) is
    returned because the wall momentum-exchange integral needs it.
    """
    rho, mx, my, mz = _momentum(f, ten)
    # Solid nodes carry no populations; sanitise rho so that collisions there
    # cannot produce inf/NaN that would poison a 0 * NaN mask product.
    rho = torch.where(rho > 1e-12, rho, torch.ones_like(rho))
    inv_rho = 1.0 / rho
    ux = mx * inv_rho
    uy = my * inv_rho
    uz = mz * inv_rho
    s_plus = 1.0 / tau
    feq = _equilibrium(rho, ux, uy, uz, ten)
    f_opp = f[ten["opp"]]
    feq_opp = feq[ten["opp"]]
    fs = 0.5 * (f + f_opp)
    fa = 0.5 * (f - f_opp)
    fs_eq = 0.5 * (feq + feq_opp)
    # TRT relaxes the antisymmetric part toward *its own* equilibrium part
    # (feq^- = 3 w (c.u)); relaxing it to zero silently destroys momentum.
    fa_eq = 0.5 * (feq - feq_opp)
    fstar = f - s_plus * (fs - fs_eq) - s_minus * (fa - fa_eq)
    # Momentum-exact body-force source: d f_i = 3 w_i c_ix F with F = rho g.
    # Sum_i c_i d f_i = 3 F sum_i w_i c_ix^2 = F and sum_i d f_i = 0, so the
    # discrete momentum balance is exact to machine precision (this is the
    # property the wave/total-drag closure check relies on).  The Guo split
    # forcing was tried first and rejected: with TRT the second relaxation
    # rate also acts on the forced half-momentum, so the injected momentum is
    # (1 - s- /2 + ...) F instead of F (measured 0.38 F at tau ~ 0.5).
    fstar = fstar + 3.0 * gx * ten["w"] * ten["cx"] * rho

    f_new = torch.empty_like(fstar)
    f_new[0] = fstar[0]
    for i in range(1, 19):
        cx, cy, cz = shifts[i]
        streamed = torch.roll(torch.roll(torch.roll(fstar[i], cz, 0), cy, 1), cx, 2)
        f_new[i] = torch.where(links[i] > 0.5, fstar[ten["opp"][i]], streamed)
    rr = f_new.sum(0)
    rr = torch.where(rr > 1e-12, rr, torch.ones_like(rr))
    ux_meas = (f_new * ten["cx"]).sum(0) / rr
    return f_new, ux_meas, fstar


def _wall_masks(solid: np.ndarray) -> dict[str, np.ndarray]:
    """Fluid cells adjacent to a solid through each axis-aligned face."""
    fluid = ~solid
    sx_pos = np.roll(solid, -1, axis=2)  # solid at x+1
    sx_neg = np.roll(solid, 1, axis=2)
    sz_pos = np.roll(solid, -1, axis=0)  # solid at z+1 (above)
    sz_neg = np.roll(solid, 1, axis=0)
    sy_pos = np.roll(solid, -1, axis=1)
    sy_neg = np.roll(solid, 1, axis=1)
    return {
        "xn_pos": (fluid & sx_pos).astype(np.float32),
        "xn_neg": (fluid & sx_neg).astype(np.float32),
        "zn_pos": (fluid & sz_pos).astype(np.float32),
        "zn_neg": (fluid & sz_neg).astype(np.float32),
        "yn_pos": (fluid & sy_pos).astype(np.float32),
        "yn_neg": (fluid & sy_neg).astype(np.float32),
    }


def run_underice_channel(
    cfg: UnderIceChannelConfig,
    mask: np.ndarray | None = None,
    *,
    spec: KeelArraySpec | None = None,
    mirror: bool = True,
) -> dict[str, Any]:
    """Run one forced under-ice channel case with the skin/form decomposition.

    ``mask=None`` gives the smooth (level-ice) control case, which is the
    negative control of the decomposition and the ``Cf`` reference used to
    non-dimensionalise the keel cases.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch back-end unavailable")
    torch.set_num_threads(int(cfg.threads))
    ten = _tensors()
    shifts = tuple((int(C[i, 0]), int(C[i, 1]), int(C[i, 2])) for i in range(19))

    if mask is None:
        solid = np.zeros((cfg.nz, cfg.ny, cfg.nx), dtype=bool)
        solid[0] = True
        solid[-1] = True
    else:
        if mask.shape != (cfg.nz, cfg.ny, cfg.nx):
            raise ValueError(f"mask shape {mask.shape} != grid {(cfg.nz, cfg.ny, cfg.nx)}")
        solid = mirrored_solid(mask, mirror=mirror)

    n_fluid = int((~solid).sum())
    h_eff = n_fluid / (2.0 * cfg.nx * cfg.ny)
    nu = cfg.u_tau * h_eff / cfg.re_tau
    tau = 0.5 + 3.0 * nu
    if not (0.5 < tau < 1.2):
        raise ValueError(f"unstable tau={tau:.4f} for re_tau={cfg.re_tau}, u_tau={cfg.u_tau}")
    s_minus = _s_minus(tau)
    g = cfg.u_tau**2 / h_eff
    smag2 = float(cfg.smagorinsky_cs) ** 2

    solid_t = torch.from_numpy(solid.astype(np.float32))
    links_np = build_link_masks(solid)
    links = torch.from_numpy(links_np.astype(np.float32))
    fluid = torch.from_numpy((~solid).astype(np.float32))
    wm = {k: torch.from_numpy(v) for k, v in _wall_masks(solid).items()}
    cx_t = ten["cx"][:, 0, 0, 0]
    gx = torch.tensor(float(g), dtype=torch.float32)

    # initial condition: log-law profile scaled down + filtered noise
    k = np.arange(cfg.nz, dtype=float)
    y_plus = np.maximum((k - 0.5) * cfg.u_tau / nu, 0.5)
    u_prof = float(cfg.init_scale) * np.minimum(
        cfg.u_tau * (np.log(y_plus) / KAPPA + B_SMOOTH), 0.25
    )
    u_prof[k < 1.0] = 0.0
    u_prof[k > cfg.nz - 2] = 0.0
    ux0 = torch.from_numpy(np.repeat(u_prof[:, None, None], cfg.ny, 1).repeat(cfg.nx, 2).astype(np.float32))
    rng = np.random.default_rng(int(cfg.seed))
    amp = float(cfg.noise_amplitude) * cfg.u_tau * (B_SMOOTH + math.log(cfg.re_tau) / KAPPA)
    noise = rng.standard_normal((2, cfg.nz, cfg.ny, cfg.nx))
    for ax in (1, 2, 3):
        noise = 0.5 * noise + 0.25 * (np.roll(noise, 1, ax) + np.roll(noise, -1, ax))
    uy0 = torch.from_numpy((0.25 * amp * noise[0]).astype(np.float32))
    uz0 = torch.from_numpy((0.25 * amp * noise[1]).astype(np.float32))
    rho0 = torch.ones((cfg.nz, cfg.ny, cfg.nx), dtype=torch.float32)
    f = _equilibrium(rho0, ux0, uy0, uz0, ten) * fluid[None]

    acc = {kk: 0.0 for kk in ("me_x", "form_x", "skin_x", "proj_form_x", "form_x_abs",
                              "skin_yface_x", "u_bulk", "rho", "ux2", "mass_in")}
    prof = np.zeros(cfg.nz)
    k_count = fluid.sum(dim=(1, 2)).numpy()
    n_samples = 0
    max_ux = 0.0
    # local normal for the projection cross-check: smoothed solid indicator
    import time

    t0 = time.time()
    for t in range(1, int(cfg.steps)):
        f, ux_m, fstar = _step(f, solid_t, links, gx, tau, s_minus, smag2, ten, shifts)
        if t >= cfg.warmup and (t - cfg.warmup) % cfg.sample_interval == 0:
            # (1) momentum exchange on every wall-crossing link
            force_x = torch.zeros_like(ux_m)
            force_norm_proj = torch.zeros_like(ux_m)
            for i in range(1, 19):
                if float(C[i, 0]) == 0.0:
                    continue
                # wall momentum gain per link: c_i (f*_i - f*_opp); the two terms
                # are the incoming and the reflected population, and only their
                # difference is the force (the sum is the symmetric/pressure part
                # that cancels pairwise on a flat wall).
                contrib = (fstar[i] - fstar[int(OPPOSITE[i])]) * links[i] * float(C[i, 0])
                force_x = force_x + contrib
            acc["me_x"] += float(force_x.sum())
            # (2) pressure on x-normal faces (form drag)
            p = (f.sum(0) / 3.0) * fluid  # p = rho / 3
            acc["form_x"] += float((p * wm["xn_pos"]).sum() - (p * wm["xn_neg"]).sum())
            acc["form_x_abs"] += float((p * wm["xn_pos"]).sum() + (p * wm["xn_neg"]).sum())
            # (3) viscous shear on horizontal faces (skin drag)
            shear = 2.0 * nu * ux_m * (wm["zn_pos"] + wm["zn_neg"])
            acc["skin_x"] += float(shear.sum())
            # y-normal flanks: mean tau_xy (expected ~0 by homogeneity)
            du_dy = 0.5 * (torch.roll(ux_m, -1, 1) - torch.roll(ux_m, 1, 1))
            acc["skin_yface_x"] += float((nu * du_dy * (wm["yn_pos"] + wm["yn_neg"])).sum())
            # projection split (normal vs tangential per link, using the local
            # staircase normal from the solid mask)
            proj = _projection_form(force_x, solid_t, ten, shifts, fluid)
            acc["proj_form_x"] += proj
            uxf = ux_m * fluid
            acc["u_bulk"] += float(uxf.sum())
            acc["ux2"] += float((uxf * ux_m).sum())
            acc["rho"] += float((f.sum(0) * fluid).sum())
            acc["mass_in"] += float(fluid.sum())
            prof += uxf.sum(dim=(1, 2)).double().numpy()
            n_samples += 1
            max_ux = max(max_ux, float(ux_m.abs().max()))
    wall = time.time() - t0
    if n_samples <= 0:
        raise RuntimeError("no samples collected; increase steps")

    area = float(cfg.nx * cfg.ny)
    scale = 1.0 / (n_samples * n_fluid)
    u_bulk = acc["u_bulk"] * scale
    rho_mean = acc["rho"] * scale
    u_rms = math.sqrt(max(acc["ux2"] * scale - u_bulk**2, 0.0))
    u_tau_meas = math.sqrt(g * h_eff * rho_mean)
    u_plus_bulk = u_bulk / u_tau_meas
    cf = 2.0 / u_plus_bulk**2

    # time-averaged forces (per unit wall-plan area -> "tau" in rho u^2 units)
    me_tau = acc["me_x"] / (n_samples * area)
    form_tau = acc["form_x"] / (n_samples * area)
    skin_tau = acc["skin_x"] / (n_samples * area)
    tau_balance = rho_mean * g * h_eff
    proj_tau = acc["proj_form_x"] / (n_samples * area)

    c_total = cd_from_cf(cf)
    c_skin = skin_tau / (rho_mean * u_bulk**2)
    c_form = form_tau / (rho_mean * u_bulk**2)
    c_me = me_tau / (rho_mean * u_bulk**2)
    c_balance = tau_balance / (rho_mean * u_bulk**2)

    prof_mean = np.zeros(cfg.nz)
    valid = k_count > 0
    prof_mean[valid] = prof[valid] / (n_samples * k_count[valid]) / u_tau_meas
    k_idx = np.arange(cfg.nz)
    y = np.maximum(k_idx - 0.5, 0.5)
    y_plus = y * u_tau_meas / nu
    lo, hi = max(30.0, 0.0), 0.35 * cfg.re_tau
    window = (y_plus >= lo) & (y_plus <= hi) & valid
    fit = None
    if window.sum() >= 3:
        fit = fit_log_law(y_plus[window], prof_mean[window], kappa=KAPPA)

    geom = keel_geometry_stats(spec, mask if mask is not None else np.zeros_like(solid), solid)
    rec = {
        "grid": {"nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz},
        "config": {kk: _plain(v) for kk, v in asdict(cfg).items()},
        "geometry": geom,
        "n_fluid": n_fluid,
        "h_eff": h_eff,
        "nu": nu,
        "tau": tau,
        "s_minus": s_minus,
        "u_tau_target": cfg.u_tau,
        "u_tau_meas": u_tau_meas,
        "re_tau": cfg.re_tau,
        "re_h": u_plus_bulk * cfg.re_tau,
        "u_bulk": u_bulk,
        "u_rms": u_rms,
        "u_plus_bulk": u_plus_bulk,
        "cf": cf,
        "cd_total": c_total,
        "cd_skin": c_skin,
        "cd_form": c_form,
        "cd_me": c_me,
        "cd_balance": c_balance,
        "tau_total": tau_balance,
        "tau_me": me_tau,
        "tau_skin": skin_tau,
        "tau_form": form_tau,
        "tau_projection_form": proj_tau,
        "closure_residual": (skin_tau + form_tau - me_tau) / me_tau,
        "closure_residual_balance": (skin_tau + form_tau - tau_balance) / tau_balance,
        "momentum_balance_residual": (me_tau - tau_balance) / tau_balance,
        "cf_log_law": cf_log_law(cfg.re_tau),
        "dean_cf": dean_cf(u_plus_bulk * cfg.re_tau) if u_plus_bulk > 0 else float("nan"),
        "log_fit": fit,
        "profile": {
            "k": k_idx.tolist(),
            "y_plus": y_plus.tolist(),
            "u_plus": prof_mean.tolist(),
            "n_fluid_per_layer": k_count.tolist(),
        },
        "n_samples": n_samples,
        "wall_seconds": float(wall),
        "max_mach": max_ux * math.sqrt(1.0 / 3.0),
        "backend": "torch-trt-guo-bb",
        "note": cfg.note,
    }
    return rec


def _projection_form(force_x: Any, solid_t: Any, ten: dict[str, Any],
                     shifts: tuple[tuple[int, int, int], ...], fluid: Any) -> float:
    """Cross-check split: project the total exchange onto the local wall normal.

    Only the x-projection of the *pressure-like* part is returned; the normal is
    the normalised multi-axis indicator of the solid neighbours, which equals a
    coordinate axis for flat faces and a mixed direction at staircase corners.
    """
    nx_n = torch.zeros_like(force_x)
    nz_n = torch.zeros_like(force_x)
    for i in range(1, 19):
        cx, cy, cz = shifts[i]
        if cx == 0 and cz == 0:
            continue
        nb = torch.roll(torch.roll(torch.roll(solid_t, -cz, 0), -cy, 1), -cx, 2)
        nb = nb * torch.roll(torch.roll(torch.roll(fluid, -cz, 0), -cy, 1), -cx, 2)
        nx_n = nx_n + nb * float(cx)
        nz_n = nz_n + nb * float(cz)
    norm = torch.sqrt(nx_n**2 + nz_n**2) + 1e-12
    nx_u = nx_n / norm
    # force_x already holds c_ix (f*_i + f*_opp); approximate the pressure part
    # of the exchange by its normal projection times the normal x-component
    return float((force_x * nx_u * nx_u).sum())


def _plain(value: Any) -> Any:
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


# ---------------------------------------------------------------------------
# Chained physical model: keel array -> C_d closure
# ---------------------------------------------------------------------------


def keel_height_from_ratio(ratio: float, h_eff: float,
                           anchor: float = LEVEL_ICE_TO_HALF_CLEARANCE) -> float:
    """Lattice keel depth for a morphological ratio ``h_keel/h_i``.

    ``h_eff = anchor * h_i`` (see :data:`LEVEL_ICE_TO_HALF_CLEARANCE`), hence
    ``h_keel = ratio * h_eff / anchor``.
    """
    h = float(ratio) * float(h_eff) / float(anchor)
    return max(1.0, float(round(h)))


def cd_closure_terms(rec_total: dict[str, Any], rec_smooth: dict[str, Any]) -> dict[str, Any]:
    """Excess-drag view: keel case relative to the smoothed reference ``Cf,0``.

    ``Cf_0`` is the skin reference: the *smoothed* geometry (a flat ice
    underside at the same nominal clearance) computed with the same solver, so
    the comparison isolates the keel contribution from any solver bias.
    """
    cf0 = float(rec_smooth["cf"])
    cf = float(rec_total["cf"])
    return {
        "cf_total": cf,
        "cf_smooth": cf0,
        "cf_skin_reference": cf0,
        "cf_ratio": cf / cf0,
        "delta_cf": cf - cf0,
        "cd_total": cd_total,
        "cd_skin": cd_skin,
        "cd_form": cd_form,
        "cd_skin_gradient": cd_skin_grad,
        "cd_skin_yface": skin_y_tau / q,
        "cd_me_diagnostic": me_tau / q,
        "cd_balance": tau_balance / q,
        "tau_total": tau_balance,
        "tau_form": form_tau,
        "tau_skin_gradient": skin_grad_tau,
        "tau_skin_yface": skin_y_tau,
        "tau_me_diagnostic": me_tau,
        "f_wall_budget": f_wall_budget,
        "dP_dt": dP_dt,
        "injected_force": input_force,
        "momentum_balance_residual": (f_wall_budget - tau_balance) / tau_balance,
        "stationarity_drift": drift,
        "closure_residual_gradient": (form_tau + skin_grad_tau - tau_balance) / tau_balance,
        "cf_log_law": cf_log_law(cfg.re_tau),
        "dean_cf": dean_cf(u_plus_bulk * cfg.re_tau) if u_plus_bulk > 0 else float("nan"),
        "log_fit": fit,
        "profile": {
            "k": k_idx.tolist(),
            "y_plus": y_plus.tolist(),
            "u_plus": prof_mean.tolist(),
            "n_fluid_per_layer": k_count.tolist(),
        },
        "n_samples": n_samples,
        "wall_seconds": float(raw.get("wall_seconds", 0.0)),
        "max_mach": float(raw.get("max_ux", 0.0)) * math.sqrt(1.0 / 3.0),
        "backend": backend,
        "note": cfg.note,
    }

# ---------------------------------------------------------------------------
# Native (C/OpenMP) back-end: same scheme, ~100x faster, raw sums only
# ---------------------------------------------------------------------------

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_NATIVE_SRC = Path(__file__).resolve().parent / "native" / "underice_channel.c"
_BUILD_CACHE: dict[str, str] = {}


def native_available() -> bool:
    """``True`` when a C compiler with OpenMP is on PATH (no build attempted)."""
    return shutil.which("gcc") is not None or shutil.which("cc") is not None


def build_native(rebuild: bool = False) -> str:
    """Compile the kernel once and cache the binary in the user cache dir."""
    key = str(_NATIVE_SRC)
    cache_hit = _BUILD_CACHE.get(key)
    if not rebuild and cache_hit and Path(cache_hit).exists() and (
            Path(cache_hit).stat().st_mtime >= _NATIVE_SRC.stat().st_mtime):
        return cache_hit
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:  # pragma: no cover
        raise RuntimeError("no C compiler (gcc/cc) on PATH")
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tensorlbm"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / "underice_channel"
    cmd = [compiler, "-O3", "-march=native", "-funroll-loops", "-fopenmp", "-o", str(out),
           str(_NATIVE_SRC), "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:  # pragma: no cover
        raise RuntimeError(f"native build failed: {proc.stderr[-1500:]}")
    _BUILD_CACHE[key] = str(out)
    return str(out)


def _level_velocity(cfg: "UnderIceChannelConfig", nz: int, nu: float) -> np.ndarray:
    k = np.arange(nz, dtype=float)
    y_plus = np.maximum((k - 0.5) * cfg.u_tau / nu, 0.5)
    u_prof = float(cfg.init_scale) * np.minimum(
        cfg.u_tau * (np.log(y_plus) / KAPPA + B_SMOOTH), 0.25
    )
    u_prof[k < 1.0] = 0.0
    u_prof[k > nz - 2] = 0.0
    return u_prof


def run_underice_channel_native(
    cfg: UnderIceChannelConfig,
    mask: np.ndarray | None = None,
    *,
    spec: KeelArraySpec | None = None,
    mirror: bool = True,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Run one case with the C/OpenMP kernel; returns the same record as torch."""
    binary = build_native(rebuild=rebuild)
    if mask is None:
        solid = np.zeros((cfg.nz, cfg.ny, cfg.nx), dtype=bool)
        solid[0] = True
        solid[-1] = True
    else:
        if mask.shape != (cfg.nz, cfg.ny, cfg.nx):
            raise ValueError(f"mask shape {mask.shape} != grid {(cfg.nz, cfg.ny, cfg.nx)}")
        solid = mirrored_solid(mask, mirror=mirror)
    n_fluid = int((~solid).sum())
    h_eff = n_fluid / (2.0 * cfg.nx * cfg.ny)
    nu = cfg.u_tau * h_eff / cfg.re_tau
    tau = 0.5 + 3.0 * nu
    if not (0.5 < tau < 1.2):
        raise ValueError(f"unstable tau={tau:.4f} for re_tau={cfg.re_tau}, u_tau={cfg.u_tau}")
    s_minus = _s_minus(tau)
    g = cfg.u_tau**2 / h_eff

    u_prof = _level_velocity(cfg, cfg.nz, nu)
    rng = np.random.default_rng(int(cfg.seed))
    amp = float(cfg.noise_amplitude) * cfg.u_tau * (B_SMOOTH + math.log(cfg.re_tau) / KAPPA)
    noise = rng.standard_normal((2, cfg.nz, cfg.ny, cfg.nx))
    for ax in (1, 2, 3):
        noise = 0.5 * noise + 0.25 * (np.roll(noise, 1, ax) + np.roll(noise, -1, ax))
    fl = (~solid).astype(np.float32)
    ux0 = (np.repeat(u_prof[:, None, None], cfg.ny, 1).repeat(cfg.nx, 2) + amp * noise[0] * fl) * fl
    uy0 = (0.25 * amp * noise[0]) * fl
    uz0 = (0.25 * amp * noise[1]) * fl
    init = np.concatenate([ux0.ravel(), uy0.ravel(), uz0.ravel()]).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "solid.raw"
        ip = Path(td) / "init.f32"
        op = Path(td) / "out.json"
        sp.write_bytes(np.ascontiguousarray(solid).astype(np.uint8).tobytes())
        ip.write_bytes(init.tobytes())
        cmd = [binary, str(cfg.nx), str(cfg.ny), str(cfg.nz), f"{tau:.17g}",
               f"{s_minus:.17g}", f"{g:.17g}", f"{nu:.17g}", str(cfg.steps), str(cfg.warmup),
               str(cfg.sample_interval), str(cfg.threads), str(sp), str(op), str(ip)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=12 * 3600)
        if proc.returncode != 0:  # pragma: no cover
            raise RuntimeError(f"native run failed: {proc.stderr[-1500:]}")
        raw = json.loads(op.read_text())

    import json as _json  # noqa: F401  (module-level import kept local for clarity)
    return _assemble_from_sums(cfg, solid, mask, spec, raw, h_eff, nu, tau, s_minus, g, "native-c-omp")


def _assemble_from_sums(cfg: UnderIceChannelConfig, solid: np.ndarray, mask: np.ndarray | None,
                        spec: KeelArraySpec | None, raw: dict[str, Any], h_eff: float,
                        nu: float, tau: float, s_minus: float, g: float,
                        backend: str) -> dict[str, Any]:
    """Turn raw diagnostic sums into the physical record (shared by both back-ends)."""
    import json as _json  # noqa: F401
    n_samples = int(raw["n_samples"])
    if n_samples <= 0:
        raise RuntimeError("no samples collected; increase steps")
    n_fluid = int(raw["n_fluid"])
    area = float(cfg.nx * cfg.ny)
    sc = 1.0 / (n_samples * n_fluid)
    u_bulk = raw["sum_ux"] * sc
    rho_mean = raw["sum_rho"] * sc
    u_rms = math.sqrt(max(raw["sum_ux2"] * sc - u_bulk**2, 0.0))
    u_tau_meas = math.sqrt(g * h_eff * rho_mean)
    u_plus_bulk = u_bulk / u_tau_meas
    cf = 2.0 / u_plus_bulk**2

    # --- drag accounting -------------------------------------------------
    # total: the global momentum balance of a periodic force-driven channel is
    # exact, tau_total = rho g h_eff (two walls share the input momentum).
    # Its *measured* counterpart is the budget route
    #     F_wall = (injected momentum) - dP/dt,
    # with dP/dt from the sum of the populations at the first/last sample; at
    # stationarity the two agree and their difference is the momentum-balance
    # residual (criterion d).
    tau_balance = rho_mean * g * h_eff
    input_force = g * n_fluid * rho_mean
    dt_samples = float(cfg.sample_interval * max(n_samples - 1, 1))
    dP_dt = (float(raw.get("mom_last", 0.0)) - float(raw.get("mom_first", 0.0))) / dt_samples
    f_wall_budget = input_force - dP_dt
    # stationarity of the averaging window: drift of the bulk velocity between
    # the first and the last sample, relative to the mean bulk velocity
    sux_first = raw.get("sux_first")
    sux_last = raw.get("sux_last")
    drift = (float("nan") if sux_first is None else
             abs(float(sux_last) - float(sux_first)) / max(abs(raw["sum_ux"]), 1e-30))
    form_tau = raw["sum_form"] / (n_samples * area)
    skin_grad_tau = raw["sum_skin"] / (n_samples * area)
    skin_y_tau = raw["sum_skin_y"] / (n_samples * area)
    me_tau = raw["sum_me"] / (n_samples * area)  # diagnostic only (see note)
    q = rho_mean * u_bulk**2

    # primary split: form drag is measured directly (wall-pressure integral on
    # the x-normal faces); the friction part is the remainder of the exact
    # momentum balance.  The viscous-shear surface integral is carried alongside
    # as an *independent* cross-check of the split.
    cd_total = cd_from_cf(cf)
    cd_form = form_tau / q
    cd_skin = cd_total - cd_form
    cd_skin_grad = skin_grad_tau / q

    k_count = np.asarray(raw["k_count"], dtype=float)
    prof_sum = np.asarray(raw["prof_sum"], dtype=float)
    prof_mean = np.zeros(cfg.nz)
    valid = k_count > 0
    prof_mean[valid] = prof_sum[valid] / (n_samples * k_count[valid]) / u_tau_meas
    k_idx = np.arange(cfg.nz)
    y = np.maximum(k_idx - 0.5, 0.5)
    y_plus = y * u_tau_meas / nu
    window = (y_plus >= 30.0) & (y_plus <= 0.35 * cfg.re_tau) & valid
    fit = fit_log_law(y_plus[window], prof_mean[window], kappa=KAPPA) if window.sum() >= 3 else None

    geom = keel_geometry_stats(spec, mask if mask is not None else np.zeros_like(solid), solid)
    return {
        "grid": {"nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz},
        "config": {kk: _plain(v) for kk, v in asdict(cfg).items()},
        "geometry": geom,
        "n_fluid": n_fluid,
        "h_eff": h_eff,
        "nu": nu,
        "tau": tau,
        "s_minus": s_minus,
        "u_tau_target": cfg.u_tau,
        "u_tau_meas": u_tau_meas,
        "re_tau": cfg.re_tau,
        "re_h": u_plus_bulk * cfg.re_tau,
        "u_bulk": u_bulk,
        "u_rms": u_rms,
        "u_plus_bulk": u_plus_bulk,
        "cf": cf,
        "cd_total": cd_total,
        "cd_skin": cd_skin,
        "cd_form": cd_form,
        "cd_skin_gradient": cd_skin_grad,
        "cd_skin_yface": skin_y_tau / q,
        "cd_me_diagnostic": me_tau / q,
        "cd_balance": tau_balance / q,
        "tau_total": tau_balance,
        "tau_form": form_tau,
        "tau_skin_gradient": skin_grad_tau,
        "tau_skin_yface": skin_y_tau,
        "tau_me_diagnostic": me_tau,
        "f_wall_budget": f_wall_budget,
        "dP_dt": dP_dt,
        "injected_force": input_force,
        "momentum_balance_residual": (f_wall_budget - tau_balance) / tau_balance,
        "stationarity_drift": drift,
        "closure_residual_gradient": (form_tau + skin_grad_tau - tau_balance) / tau_balance,
        "cf_log_law": cf_log_law(cfg.re_tau),
        "dean_cf": dean_cf(u_plus_bulk * cfg.re_tau) if u_plus_bulk > 0 else float("nan"),
        "log_fit": fit,
        "profile": {
            "k": k_idx.tolist(),
            "y_plus": y_plus.tolist(),
            "u_plus": prof_mean.tolist(),
            "n_fluid_per_layer": k_count.tolist(),
        },
        "n_samples": n_samples,
        "wall_seconds": float(raw.get("wall_seconds", 0.0)),
        "max_mach": float(raw.get("max_ux", 0.0)) * math.sqrt(1.0 / 3.0),
        "backend": backend,
        "note": cfg.note,
    }
