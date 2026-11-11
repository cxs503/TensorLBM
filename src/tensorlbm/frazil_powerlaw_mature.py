"""Mature verification face for the power-law rheology of frazil ice slurries.

Scope of this face (what it measures, what it does not)
-------------------------------------------------------
Frazil -- millimetre ice crystals suspended in supercooled water -- makes the
carrier of a ship in an ice channel behave as a generalised Newtonian fluid
``tau = K gdot**n``.  The resistance / thrust-loss models in this platform take
``(K, n)`` as *input*, so those two numbers are physics that has to be measured
or referenced, not asserted.  :mod:`tensorlbm.frazil_powerlaw_shear` built the
first (structure-resolved, D3Q19 shear cell) measurement of that flow curve and
reported three of its five acceptance criteria as passed; the two it could not
close were

* the concentrated point (``c = 0.20``) -- random sequential placement with a
  one-cell clearance jammed at ``phi = 0.155``, so the requested concentration
  was never realised, and
* grid convergence of ``(K, n)`` -- the two resolutions were separate random
  geometries, so the comparison mixed discretisation error with a ~10 % seed to
  seed scatter of the microstructure itself.

This module closes both, and does it in a way that keeps the comparison honest:

1. **Physical-space microgeometry, voxelised at several resolutions.**  The
   crystals are placed once, in continuous coordinates of the unit cell
   (:func:`physical_suspension`).  Each resolution then rasterises *the same*
   physical object (:func:`voxelise`).  A coarse/fine pair therefore differs
   only in discretisation -- which is what a grid-convergence study is supposed
   to measure -- and the placement clearance is a physical length (a fraction of
   the crystal radius), not one lattice cell, so it stops setting the achievable
   concentration.
2. **The kinematic condition is measured, not assumed.**  The production
   protocol advects the crystals affinely with the imposed shear rate (the
   standard frozen-structure assumption for a statistically homogeneous
   suspension: the mean velocity profile of a uniformly sheared suspension is
   linear with slope ``U/H`` whatever its viscosity).  Because that assumption
   has real leverage on the answer, the module also provides the self-consistent
   fixed point (:func:`self_consistent_point`) and the campaign runs an explicit
   scale scan, so the size of the kinematic ambiguity is *reported* rather than
   hidden.
3. **Stress-estimator selection by evidence.**  Three estimators are computed
   for every point (plane momentum flux in the particle-free band next to the
   wall, the moving-wall momentum exchange, and the momentum-exchange force on
   the particles).  The pure-fluid control has the exact solution
   ``tau = rho nu gdot`` and the dilute limit has Einstein's ``1 + 5/2 c``;
   :func:`select_stress_estimator` picks the estimator that closes the first and
   is reported against the second.  (The previous face subtracted the particle
   force from the wall flux; the momentum balance says the flux at a
   particle-free plane *is* the total suspension stress, so that subtraction is
   double counting -- see :func:`moment_balance_note`.)
4. **Bootstrap confidence intervals.**  Every fit is repeated over independent
   microstructures (:func:`bootstrap_power_law`), so a statement like
   "``n < 0.95`` at ``c >= 0.10``" carries a 95 % interval rather than a point
   value from one frozen configuration.
5. **Closure comparison over the whole range.**  ``c in [0, 0.3]`` is compared
   with Einstein (``1 + 2.5 c``), Einstein-Batchelor-Green
   (``1 + 2.5 c + 6.2 c**2``, Batchelor & Green 1972) and Krieger-Dougherty
   (``(1 - c/phi_max)**(-2.5 phi_max)``), not just at the two points the
   acceptance criteria ask for.

Physics of the two structural families
--------------------------------------
* ``dispersed`` -- one crystal per particle.  This is the hard-sphere state for
  which the three closures above are the correct ones, and it is also the
  built-in negative control for shear thinning: a frozen rigid microstructure in
  Stokes flow is Newtonian by construction, so any measured ``n < 1`` here would
  be a solver artefact, not rheology.
* ``flocculated`` -- crystals bonded into fractal aggregates whose size is set by
  the shear-breakup balance ``R_eq = sqrt(F_b / (mu gdot))`` (Sonntag & Russel
  1986; Spicer & Pratsinis 1996), i.e. ``N_eq = (F_b/(mu gdot a**2))**(df/2)``.
  Aggregates trap water, so their hydrodynamic volume fraction exceeds their
  solids fraction, and they break up as the shear rate rises -- the accepted
  physical origin of the shear thinning of flocculated slurries (Barnes 1997).
  ``F_b`` is *anchored* (declared ``N_eq(gdot_min) = N_cap`` before any run) and
  the anchor is reported, not fitted.

Declared limitations (see the campaign report for the full list)
----------------------------------------------------------------
* Frozen-structure protocol (Ladd 1994): the geometry inside a run is rigid and
  advected kinematically with the imposed shear, so shear-rate dependence enters
  through the equilibrium structure selected *between* runs.  No dynamic
  aggregation, no population balance, no floc erosion inside a run.
* Primaries are spheres; real frazil crystals are discs of aspect ratio ~10.
* The cell is bounded, and the walls are a *depletion* boundary for a
  suspension: the apparent viscosity of a narrow gap is below the bulk value.
  A finite-size scan in ``H/D`` with a ``1/(H/D)`` extrapolation is part of the
  campaign, and the confined and extrapolated numbers are reported together.
* Contact lubrication is not resolved below one lattice cell; the near-contact
  resistance of the bounce-back scheme is the usual one (Ladd & Verberg 2001).

References
----------
* Einstein (1906, 1911); Batchelor & Green (1972) *J. Fluid Mech.* 56, 401;
  Batchelor (1977) *J. Fluid Mech.* 83, 97; Krieger & Dougherty (1959).
* Ladd (1994) *J. Fluid Mech.* 271, 285; Ladd & Verberg (2001) *J. Comput.
  Phys.* 171, 704.
* Barnes (1997) *J. Non-Newtonian Fluid Mech.* 70, 1.
* Sonntag & Russel (1986) *J. Colloid Interface Sci.* 113, 399;
  Spicer & Pratsinis (1996) *Water Res.* 30, 1049.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np

from .ice_suspension_rheology import closures, shapes
from .ice_suspension_rheology.couette import ShearCell, ShearCellConfig
from .frazil_powerlaw_shear import (
    DEFAULT_DF,
    DEFAULT_N_CAP,
    bond_force_from_reference,
    equilibrium_floc_size,
    fractal_cluster_cca,
)

__all__ = [
    "MatureRheologyError",
    "PhysicalSuspension",
    "physical_suspension",
    "voxelise",
    "resolutions",
    "measure_point",
    "bulk_shear_rate",
    "self_consistent_point",
    "select_stress_estimator",
    "pure_reference",
    "fit_power_law",
    "bootstrap_power_law",
    "einstein_batchelor_green",
    "krieger_dougherty",
    "hydrodynamic_volume_fraction",
    "grid_convergence_pair",
    "evaluate_acceptance",
]

#: Primary-crystal radius in units of the *cell side*.  Fixed for *every*
#: resolution, which is what makes the coarse/fine pair the same physical
#: suspension: ``3 / 32`` gives a 6-lattice-cell crystal in the 32-cell cell and
#: a 9-lattice-cell crystal in the 48-cell one (ratio 1.5).
RADIUS_FRACTION = 3.0 / 32.0

#: The two lattice resolutions of the convergence study: (name, n_xy, n_z).
#: The physical cell is a cube of side ``n_xy`` cells whose wall-normal extent is
#: mapped onto ``[2, n_xy + 2]``, so ``n_z >= n_xy + 4`` and the two wall bands
#: (``z < 2`` and ``z > n_xy + 2``) are crystal-free by construction.  The
#: effective gap of the moving-wall pair is ``n_z - 2``.
RESOLUTIONS = (("coarse", 32, 36), ("fine", 48, 52))

#: Inertia parameter of the production ladder, ``Re_p = gdot D**2 / nu``.
#: Kept at ``<= 1e-2`` so the O(Re_p) inertial correction to the drag (and
#: therefore to eta_r) stays far below the 5 % convergence tolerance.  This is
#: the fix for the previous face's ``Re_p`` up to 0.36, which showed up as
#: spurious shear *thickening* of the frozen (dispersed) family.
RE_P_LADDER = (2.0e-3, 4.0e-3, 8.0e-3)


class MatureRheologyError(ValueError):
    """Raised on inputs outside the validated envelope of this face."""


# ---------------------------------------------------------------------------
# Physical-space microgeometry
# ---------------------------------------------------------------------------


@dataclass
class PhysicalSuspension:
    """A suspension declared in continuous coordinates of the unit cell.

    ``clusters[k]`` is an ``(M, 3)`` array of primary-crystal centres in units of
    the cell size (``x, y`` periodic on ``[0, 1)``, ``z in [z_lo, z_hi]``) and
    ``radius`` is the primary radius, also in cell units.  Nothing here is
    lattice-specific: the same object is handed to :func:`voxelise` at any
    resolution.
    """

    family: str
    phi_target: float
    radius: float                       # primary radius, units of cell size
    clusters: list = field(default_factory=list)   # list of (M,3) float arrays
    n_primary: int = 1
    df: float = DEFAULT_DF
    seed: int = 0
    stats: dict = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def n_crystals(self) -> int:
        return int(sum(c.shape[0] for c in self.clusters))

    @property
    def solid_volume_fraction(self) -> float:
        """Analytic solids fraction ``N * 4/3 pi a**3`` of the declared object."""
        return float(self.n_crystals * 4.0 / 3.0 * math.pi * self.radius ** 3)


def _cluster_offsets(n_primary: int, df: float, seed: int) -> np.ndarray:
    """Primary-centre offsets of one aggregate, in units of the primary radius."""
    cl = fractal_cluster_cca(int(n_primary), 1.0, seed=seed, df_declared=df, max_attempts=12)
    off = np.asarray(cl.centres, dtype=float)
    off = off - off.mean(axis=0)
    return off, cl


def physical_suspension(
    phi: float,
    *,
    family: str = "dispersed",
    n_primary: int = 1,
    df: float = DEFAULT_DF,
    radius_fraction: float = RADIUS_FRACTION,
    seed: int = 0,
    min_gap_fraction: float = 0.02,
    wall_margin_fraction: float = 0.06,
    max_attempts: int = 400000,
) -> PhysicalSuspension:
    """Random sequential placement in continuous coordinates of the unit cell.

    The clearance between two crystals is ``min_gap_fraction * a`` (a *physical*
    fraction of the crystal radius, in contrast with the previous face's one
    lattice cell, which is a different physical distance at every resolution and
    jammed the placement at ``phi ~ 0.155``).  ``wall_margin_fraction`` keeps the
    crystals clear of both walls by the same kind of physical margin.

    Placement stops when the analytic solids fraction of the placed crystals
    reaches ``phi``; the achieved fraction is ``solid_volume_fraction`` after
    voxelisation it is the lattice-union fraction, which is what the closures are
    evaluated at.

    Args:
        phi: Target solids volume fraction in ``[0, 0.5]``.
        family: ``"dispersed"`` (one crystal per particle) or ``"flocculated"``
            (``n_primary`` crystals per aggregate).
        n_primary: Crystals per aggregate for the flocculated family.
        df: Fractal dimension used to build the aggregates.
        radius_fraction: Primary radius in units of the cell size.
        seed: RNG seed; the microstructure is a deterministic function of it.
        min_gap_fraction: Minimum surface-to-surface clearance, in units of ``a``.
        wall_margin_fraction: Minimum crystal-to-wall clearance, in units of ``a``.

    Returns:
        :class:`PhysicalSuspension`.

    Raises:
        MatureRheologyError: for ``phi`` outside ``[0, 0.5]``, a non-positive
            radius, an unknown family, or a cell too small to host one crystal.
    """
    if not (0.0 <= float(phi) <= 0.5):
        raise MatureRheologyError(f"phi must be in [0, 0.5], got {phi!r}")
    if not (radius_fraction > 0.0):
        raise MatureRheologyError(f"radius_fraction must be > 0, got {radius_fraction!r}")
    if family not in ("dispersed", "flocculated"):
        raise MatureRheologyError(f"unknown family {family!r}")
    n_primary = 1 if family == "dispersed" else max(1, int(n_primary))
    a = float(radius_fraction)
    margin = wall_margin_fraction * a
    gap = min_gap_fraction * a
    if 2.0 * margin + 2.0 * a >= 1.0:
        raise MatureRheologyError("cell too small for the declared crystal radius")

    rng = np.random.default_rng(seed)
    if phi == 0.0:
        return PhysicalSuspension(family=family, phi_target=0.0, radius=a,
                                  clusters=[], n_primary=n_primary, df=df, seed=seed,
                                  stats={"attempts": 0, "achieved_phi": 0.0})

    off, _cl = _cluster_offsets(n_primary, df, seed)
    v_cry = 4.0 / 3.0 * math.pi * a ** 3
    target_n = int(max(1, math.ceil(float(phi) / v_cry)))   # crystals to place

    # primary centres already placed, as a (M, 3) array in cell units
    placed = np.zeros((0, 3))
    clusters: list[np.ndarray] = []
    attempts = 0
    need = 2.0 * a + gap
    while sum(c.shape[0] for c in clusters) < target_n and attempts < max_attempts:
        attempts += 1
        c = rng.uniform(0.0, 1.0, size=3)
        cand = off * a + c
        # keep the whole aggregate inside the wall margins (z only; x,y periodic)
        z_lo = float(cand[:, 2].min())
        z_hi = float(cand[:, 2].max())
        if z_lo < margin + a or z_hi > 1.0 - margin - a:
            continue
        if placed.shape[0]:
            d = cand[:, None, :] - placed[None, :, :]
            d[:, :, 0] -= np.round(d[:, :, 0])
            d[:, :, 1] -= np.round(d[:, :, 1])
            dist2 = np.einsum("ijk,ijk->ij", d, d)
            if float(dist2.min()) < need ** 2:
                continue
        clusters.append(cand)
        placed = np.vstack([placed, cand])
    if not clusters:
        raise MatureRheologyError(
            f"no crystal could be placed at phi={phi}, a={a} (cell too small)")

    susp = PhysicalSuspension(family=family, phi_target=float(phi), radius=a,
                              clusters=clusters, n_primary=n_primary, df=df, seed=seed)
    susp.stats = {
        "attempts": attempts,
        "achieved_phi": susp.solid_volume_fraction,
        "n_clusters": susp.n_clusters,
        "n_crystals": susp.n_crystals,
        "min_gap_a": gap / a,
        "wall_margin_a": margin / a,
    }
    return susp


def _lattice_columns(n_xy: int, n_z: int) -> dict:
    """Map from cell coordinates ``[0,1]`` to lattice coordinates.

    ``x, y`` are periodic with period ``n_xy``; ``z`` maps ``[0, 1]`` onto the
    *effective gap* ``[2, n_z - 2]`` so that a physical z margin survives at
    every resolution and the walls stay free of solid nodes.
    """
    if int(n_z) < int(n_xy) + 4:
        raise MatureRheologyError(
            f"n_z={n_z} must be >= n_xy + 4 = {n_xy + 4}: the physical cell is a cube "
            f"of side n_xy with a two-cell wall band at each wall")
    z_lo, z_hi = 2.0, 2.0 + float(n_xy)
    return {"n_xy": int(n_xy), "n_z": int(n_z), "z_lo": z_lo, "z_hi": z_hi,
            "z_span": z_hi - z_lo, "x_span": float(n_xy), "y_span": float(n_xy)}


def voxelise(susp: PhysicalSuspension, *, n_xy: int, n_z: int,
             min_wall_clear: float = 1.5) -> tuple[list, dict]:
    """Rasterise a :class:`PhysicalSuspension` onto an ``n_z x n_xy x n_xy`` lattice.

    The crystal radius in lattice cells follows from ``radius / z_span * z_span``
    -- i.e. the *same physical* crystal has ``a_cells = radius * (n_z - 4)``
    (the physical z extent is the effective gap ``n_z - 4`` because both the wall
    margin and the wall band are part of the map).  Every sphere is rasterised
    with :func:`tensorlbm.ice_suspension_rheology.shapes.sphere_mask`; a node
    claimed by two overlapping masks (touching crystals) is assigned to the
    nearer centre, so no node is owned twice and the fluid sees a single
    occupied set.

    Returns:
        ``(particles, info)`` with ``particles`` a list of
        :class:`~tensorlbm.ice_suspension_rheology.shapes.Particle` and ``info``
        reporting the lattice radius, the achieved (union) volume fraction, the
        number of nodes reassigned at contacts and the minimum wall clearance in
        cells.
    """
    col = _lattice_columns(n_xy, n_z)
    a_cells = susp.radius * col["z_span"]
    if a_cells < 1.5:
        raise MatureRheologyError(
            f"crystal radius {a_cells:.3f} cells is too small to discretise "
            f"(increase n_z or radius_fraction)")
    nz, ny, nx = int(n_z), int(n_xy), int(n_xy)
    zz, yy, xx = shapes.box_grid(nx, ny, nz)

    centres_cells: list[tuple[float, float, float]] = []
    cluster_of: list[int] = []      # cluster index of every crystal
    for icl, cl in enumerate(susp.clusters):
        for c in cl:
            centres_cells.append((
                col["z_lo"] + float(c[2]) * col["z_span"],
                float(c[1]) * col["y_span"],
                float(c[0]) * col["x_span"],
            ))
            cluster_of.append(icl)

    # raw masks first: the number of nodes claimed by more than one crystal is a
    # contact diagnostic (touching crystals share their boundary shell)
    claim = np.zeros((nz, ny, nx), dtype=np.int16)
    masks: list[np.ndarray] = []
    for (cz, cy, cx) in centres_cells:
        mask = shapes.sphere_mask(nx, ny, nz, (cz, cy, cx), a_cells)
        claim += mask.astype(np.int16)
        masks.append(mask)
    reassigned = int((claim > 1).sum())

    # single owner per node: nearest centre (ties broken by index order)
    owner = np.full((nz, ny, nx), -1, dtype=np.int16)
    dist_best = np.full((nz, ny, nx), np.inf, dtype=np.float32)
    for idx, (cz, cy, cx) in enumerate(centres_cells):
        mask = masks[idx]
        if not mask.any():
            continue
        dz = zz - cz
        dy = yy - cy
        dx = xx - cx
        dy -= ny * np.round(dy / ny)          # periodic images in x, y
        dx -= nx * np.round(dx / nx)
        d2 = (dz * dz + dy * dy + dx * dx).astype(np.float32)
        take = mask & (d2 < dist_best)
        owner[take] = idx
        dist_best[take] = d2[take]

    particles = []
    for idx, ctr in enumerate(centres_cells):
        m = owner == idx
        if not m.any():
            continue
        particles.append(shapes.Particle(
            mask=m, centre=ctr, half_extent=float(a_cells), shape="sphere",
            volume=int(m.sum()),
            meta={"crystal": idx, "cluster": cluster_of[idx],
                  "radius_cells": float(a_cells)}))

    occupied = owner >= 0
    solid_z = np.flatnonzero(occupied.any(axis=(1, 2)))
    wall_clear = float(min(solid_z.min(), (nz - 1) - solid_z.max())) if solid_z.size else float(nz)
    if wall_clear < min_wall_clear:
        raise MatureRheologyError(
            f"solid node within {wall_clear} cells of a wall (need {min_wall_clear}); "
            f"lower the radius or the wall margin")
    # The solids fraction is referred to the *physical* cell volume n_xy**3 (the
    # cube that the wall-normal map [2, n_xy+2] covers), not to the lattice box,
    # which carries two extra wall rows and would deflate phi by
    # n_xy**3 / (n_xy**2 * n_z) -- 11 % at the coarse resolution, and a
    # different 11 % at every resolution.
    phi_lattice = float(occupied.sum()) / float(nx * ny * nx)
    info = {
        "radius_cells": float(a_cells),
        "phi_lattice": phi_lattice,
        "phi_lattice_full_box": float(occupied.sum()) / float(nx * ny * nz),
        "phi_declared": susp.solid_volume_fraction,
        "phi_ratio": float(phi_lattice / susp.solid_volume_fraction)
        if susp.solid_volume_fraction > 0 else 1.0,
        "n_particles": len(particles),
        "n_crystals": susp.n_crystals,
        "nodes": int(occupied.sum()),
        "wall_clearance_cells": wall_clear,
        "radius_over_gap": float(a_cells / (nz - 2)),
        "contact_nodes": reassigned,
    }
    return particles, info# ---------------------------------------------------------------------------
# Stress measurement
# ---------------------------------------------------------------------------


def _per_wall_exchange(cell: ShearCell, which: str = "bottom") -> float:
    """Momentum-exchange force on **one** wall, per unit wall area.

    The in-tree :class:`ShearCell` lumps the links of *both* walls into one
    accumulator (``f_wall_mem``), and in a symmetric Couette cell the two walls
    carry opposite forces, so that accumulator cancels to zero identically -- it
    cannot be used as a stress estimator (measured: exactly ``0.0`` for pure
    Couette at every resolution).  This helper repeats the half-way bounce-back
    population exchange of :meth:`ShearCell.step` for the link rows of one wall
    only, so the wall force can be compared with the exact pure-fluid value
    ``rho nu gdot A`` and with the plane-flux estimator.

    Args:
        cell: a :class:`ShearCell` whose state has been advanced at least once.
        which: ``"bottom"`` (``z = 0`` wall, links at the ``z = 1`` row) or
            ``"top"`` (``z = nz-1`` wall, links at the ``nz-2`` row).

    Returns:
        ``F_x`` on that wall divided by ``nx * ny``, i.e. a stress.
    """
    from .d3q19 import equilibrium3d, macroscopic3d
    from .solver3d import collide_bgk3d

    if which not in ("bottom", "top"):
        raise MatureRheologyError(f"which must be 'bottom' or 'top', got {which!r}")
    cfg = cell.cfg
    rho, ux, uy, uz = macroscopic3d(cell.f)
    solid = cell.solid
    rho = _where(solid, 1.0, rho)
    ux = _where(solid, cell.usx, ux)
    uy = _where(solid, cell.usy, uy)
    uz = _where(solid, cell.usz, uz)
    f = _where(solid.unsqueeze(0), equilibrium3d(rho, ux, uy, uz), cell.f)
    fstar = collide_bgk3d(f, cfg.tau)
    row = 1 if which == "bottom" else cfg.nz - 2
    acc = 0.0
    for q, opp, cvec in cell._dirs:
        if cvec[0] == 0:
            continue
        wl = cell.wall_links[opp][row]
        if not bool(wl.any()):
            continue
        vals = 2.0 * fstar[q][row] + cell.coefs[opp][row] * rho[row]
        acc += float(cvec[0]) * float(vals[wl].sum())
    return acc / float(cfg.nx * cfg.ny)


def _where(mask, a, b):
    """``torch.where(mask, a, b)`` with a scalar or tensor ``a``."""
    import torch

    if not torch.is_tensor(a):
        a = torch.as_tensor(a, dtype=b.dtype, device=b.device)
    return torch.where(mask, a, b)


def measure_point(
    particles,
    *,
    n_xy: int,
    n_z: int,
    nu: float = 0.2,
    gamma_dot: float,
    steps: int = 1800,
    warmup: int = 900,
    sample_every: int = 10,
    band: tuple[int, int] = (1, 4),
    labels: dict | None = None,
) -> dict:
    """Run one shear-cell point and time-average the stress diagnostics.

    ``u_wall`` follows from the *declared* shear rate through the effective gap
    ``n_z - 2`` (half-way bounce-back walls); the *measured* rate is the slope of
    a least-squares line through the mean velocity profile and is what the
    viscosity uses, so a mis-set gap shows up in ``profile_r2`` rather than in
    ``eta``.

    Three stress estimators are returned (see the module docstring):

    ``tau_flux``   plane momentum flux in the particle-free wall band -- the
                   production estimator,
    ``tau_wall``   per-wall momentum-exchange force, :func:`_per_wall_exchange`,
    ``f_particle`` momentum-exchange force on the particles.

    Returns:
        A JSON-able record: ``tau_flux``, ``tau_wall``, ``f_particle``,
        ``gamma_dot`` (measured), ``gamma_dot_nominal``, ``profile_r2``,
        ``eta_eff``, ``eta_flux_over_nu``, ``tau_sd``, ``n_samples`` and the
        geometry bookkeeping passed through ``labels``.
    """
    from .ice_suspension_rheology.couette import ShearCellConfig

    if not (gamma_dot > 0.0):
        raise MatureRheologyError(f"gamma_dot must be > 0, got {gamma_dot!r}")
    u_wall = 0.5 * gamma_dot * (n_z - 2)
    cfg = ShearCellConfig(nx=n_xy, ny=n_xy, nz=n_z, nu=nu, u_wall=u_wall)
    cell = ShearCell(cfg, particles)
    tau_flux, tau_wall, f_part = [], [], []
    for it in range(int(steps)):
        sample = it >= warmup and (it - warmup) % max(1, int(sample_every)) == 0
        out = cell.step(sample=sample, band=band)
        if sample:
            tau_flux.append(float(out["tau_near"]))
            tau_wall.append(_per_wall_exchange(cell, "bottom"))
            f_part.append(float(out["f_particle_mem"][0]) / (cfg.nx * cfg.ny))
    prof = cell.velocity_profile()
    g = float(prof["slope"])
    rec = {
        "gamma_dot": g,
        "gamma_dot_nominal": float(cfg.gamma_dot),
        "u_wall": float(u_wall),
        "profile_r2": float(prof["r2"]),
        "tau_flux": float(np.mean(tau_flux)),
        "tau_flux_sd": float(np.std(tau_flux)),
        "tau_wall": float(np.mean(tau_wall)),
        "f_particle": float(np.mean(f_part)),
        "n_samples": len(tau_flux),
        "n_steps": int(steps),
        "n_warmup": int(warmup),
        "nu": float(nu),
        "phi_lattice": float(cell.phi_lattice),
        "node_overlaps": int(cell.node_overlaps),
        "n_particles": int(cell.n_particles),
        "n_xy": int(n_xy),
        "n_z": int(n_z),
        "re_p": float(cfg.gamma_dot * (2.0 * getattr(cell.particles[0], "half_extent", 1.0)) ** 2 / nu)
        if cell.particles else 0.0,
    }
    rec["eta_eff"] = rec["tau_flux"] / g
    rec["eta_flux_over_nu"] = rec["eta_eff"] / nu
    rec["eta_wall_over_nu"] = rec["tau_wall"] / g / nu
    if labels:
        rec.update(labels)
    return rec


def pure_reference(*, n_xy: int, n_z: int, nu: float = 0.2, gamma_dot: float,
                   steps: int = 900, warmup: int = 450, sample_every: int = 10,
                   band: tuple[int, int] = (1, 4)) -> dict:
    """The ``c = 0`` control: pure Couette, whose exact solution is ``tau = rho nu gdot``.

    This is criterion 1 and simultaneously the stress-estimator calibration: the
    measured flux must reproduce ``rho nu gdot`` (and the per-wall exchange
    force must too) or the estimator is not trustworthy at that resolution.
    """
    rec = measure_point([], n_xy=n_xy, n_z=n_z, nu=nu, gamma_dot=gamma_dot,
                        steps=steps, warmup=warmup, sample_every=sample_every, band=band,
                        labels={"case": "pure"})
    rec["nu_gamma_dot"] = float(nu * rec["gamma_dot"])
    rec["flux_rel_err"] = float(rec["tau_flux"] / rec["nu_gamma_dot"] - 1.0)
    rec["wall_rel_err"] = float(rec["tau_wall"] / rec["nu_gamma_dot"] - 1.0)
    return rec


def select_stress_estimator(pure_cases) -> dict:
    """Pick the stress estimator by the pure-fluid momentum budget.

    Args:
        pure_cases: iterable of :func:`pure_reference` records.

    Returns:
        ``{"chosen": name, "median_abs_err": {...}, "note": str}``.  The
        production estimator is the one with the smaller median ``|tau/(nu
        gamma_dot) - 1|`` over the controls; ties and both-fail cases are
        reported rather than hidden.
    """
    flux = [abs(float(c["flux_rel_err"])) for c in pure_cases]
    wall = [abs(float(c["wall_rel_err"])) for c in pure_cases]
    if not flux:
        raise MatureRheologyError("no pure-fluid controls supplied")
    mf, mw = float(np.median(flux)), float(np.median(wall))
    chosen = "tau_flux" if mf <= mw else "tau_wall"
    return {"chosen": chosen, "median_abs_err": {"tau_flux": mf, "tau_wall": mw},
            "n_controls": len(flux),
            "note": "both estimators are reported for every point; eta_r is always a "
                    "ratio of the same estimator with and without crystals"}


# ---------------------------------------------------------------------------
# Closures
# ---------------------------------------------------------------------------


def einstein_batchelor_green(phi) -> np.ndarray:
    """``eta_r = 1 + 2.5 phi + 6.2 phi**2`` (Einstein + Batchelor-Green pair term)."""
    return closures.batchelor_relative_viscosity(phi)


def krieger_dougherty(phi, *, phi_max: float = 0.6, intrinsic: float = 2.5) -> np.ndarray:
    """``eta_r = (1 - phi/phi_max)**(-intrinsic phi_max)``."""
    return closures.krieger_dougherty_relative_viscosity(
        phi, phi_max=phi_max, intrinsic=intrinsic)


def hydrodynamic_volume_fraction(phi: float, *, n_primary: float, df: float = DEFAULT_DF) -> float:
    """Hydrodynamic volume fraction of a fractal aggregate suspension.

    An aggregate of ``N`` touching primaries of radius ``a`` occupies a
    hydrodynamic radius ``R = a N**(1/df)``, so it traps
    ``(R/a)**(3-df) = N**((3-df)/df)`` times its solids volume of carrier::

        phi_h = phi * N ** ((3 - df) / df)

    For ``df = 2`` and ``N = 8`` this is a factor ``8**0.5 = 2.83`` -- the reason
    a flocculated slurry thickens far more than a hard-sphere suspension at the
    same solids fraction, and the reason the flocculated family is *not* compared
    with the hard-sphere closures directly.
    """
    if not (1.0 <= float(n_primary)):
        return float(phi)
    return float(phi) * float(n_primary) ** ((3.0 - float(df)) / float(df))


# ---------------------------------------------------------------------------
# Power-law fits
# ---------------------------------------------------------------------------


def fit_power_law(gamma_dot, tau) -> dict:
    """Least-squares fit of ``tau = K gamma_dot**n`` in log-log space.

    Returns ``{"K", "n", "r2", "n_points", "residuals"}``.  ``r2`` is computed in
    log space, which is the space the fit lives in; the residuals are returned in
    log space too so a reader can see whether a two-parameter law is even
    adequate for the measured curve.

    Raises:
        MatureRheologyError: for fewer than two points or non-positive data.
    """
    g = np.asarray(gamma_dot, dtype=float)
    t = np.asarray(tau, dtype=float)
    if g.size < 2 or g.size != t.size:
        raise MatureRheologyError("need >= 2 paired (gamma_dot, tau) points")
    if np.any(g <= 0) or np.any(t <= 0):
        raise MatureRheologyError("gamma_dot and tau must be > 0")
    x, y = np.log(g), np.log(t)
    A = np.vstack([x, np.ones_like(x)]).T
    (n, lnK), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([n, lnK])
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"K": float(math.exp(lnK)), "n": float(n), "r2": r2,
            "n_points": int(g.size), "residuals": (y - pred).tolist()}


def bootstrap_power_law(samples, *, n_boot: int = 4000, seed: int = 0,
                        alpha: float = 0.05) -> dict:
    """Bootstrap the ``(K, n)`` pair over independent microstructures.

    Each entry of ``samples`` is one independent realisation of the *whole* rate
    ladder -- ``{"gamma_dot": [...], "tau": [...]}`` for one seed -- so resampling
    seeds propagates the configuration-to-configuration scatter (which the
    previous face measured at ~10 % in ``eta_r``) into the fit, instead of quoting
    a point value from a single frozen geometry.

    Returns:
        ``{"K", "n", "K_ci", "n_ci", "n_ci_excludes_one", "n_seeds", "n_boot"}``
        with percentile intervals at level ``1 - alpha``.
    """
    samples = [s for s in samples if s and len(s.get("gamma_dot", ())) >= 2]
    if not samples:
        raise MatureRheologyError("bootstrap needs at least one ladder sample")
    rng = np.random.default_rng(seed)
    ns, Ks = [], []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(samples), size=len(samples))
        g = np.concatenate([np.asarray(samples[i]["gamma_dot"], dtype=float) for i in pick])
        t = np.concatenate([np.asarray(samples[i]["tau"], dtype=float) for i in pick])
        try:
            f = fit_power_law(g, t)
        except MatureRheologyError:
            continue
        ns.append(f["n"])
        Ks.append(f["K"])
    if not ns:
        raise MatureRheologyError("every bootstrap resample was degenerate")
    ns = np.asarray(ns)
    Ks = np.asarray(Ks)
    lo, hi = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    n_ci = (float(np.percentile(ns, lo)), float(np.percentile(ns, hi)))
    K_ci = (float(np.percentile(Ks, lo)), float(np.percentile(Ks, hi)))
    point = fit_power_law(
        np.concatenate([np.asarray(s["gamma_dot"], float) for s in samples]),
        np.concatenate([np.asarray(s["tau"], float) for s in samples]))
    return {"K": point["K"], "n": point["n"], "r2": point["r2"],
            "K_ci": K_ci, "n_ci": n_ci,
            "n_ci_excludes_one": bool(n_ci[1] < 1.0 or n_ci[0] > 1.0),
            "n_seeds": len(samples), "n_boot": int(len(ns)),
            "n_median": float(np.median(ns)), "n_sd": float(np.std(ns))}


def grid_convergence_pair(coarse: dict, fine: dict) -> dict:
    """Relative differences of ``(K, n)`` between two resolutions.

    ``coarse`` / ``fine`` are :func:`bootstrap_power_law` outputs (or plain
    ``{"K", "n"}`` dictionaries).  The pair is only meaningful when both
    resolutions saw the *same physical* microstructure, which is what
    :func:`voxelise` guarantees.
    """
    dn = abs(float(fine["n"]) - float(coarse["n"])) / abs(float(coarse["n"]))
    dK = abs(float(fine["K"]) - float(coarse["K"])) / abs(float(coarse["K"]))
    return {"n_coarse": float(coarse["n"]), "n_fine": float(fine["n"]),
            "K_coarse": float(coarse["K"]), "K_fine": float(fine["K"]),
            "n_rel_diff": float(dn), "K_rel_diff": float(dK),
            "both_below_5pct": bool(dn < 0.05 and dK < 0.05)}


# ---------------------------------------------------------------------------
# Self-consistent kinematic protocol
# ---------------------------------------------------------------------------


def bulk_shear_rate(cell, *, lo: float = 0.25, hi: float = 0.75) -> dict:
    """Shear rate of the **bulk** band of the gap, and of the whole gap.

    In a bounded Couette cell the crystal-free layers next to the walls carry a
    higher shear rate than the crystal-rich middle (the walls are a depletion
    boundary).  The *global* slope is the average of the two and is therefore
    neither the bulk rate nor the wall-layer rate.  For the closures one wants
    the bulk rate, which is the slope fitted over the inner band ``[lo, hi]`` of
    the gap.

    Returns:
        ``{"gamma_bulk", "gamma_all", "z_band", "r2_all"}``.
    """
    prof = cell.velocity_profile()
    u = np.asarray(prof["u"], dtype=float)
    n = u.size
    if n < 8:
        return {"gamma_bulk": float(prof["slope"]), "gamma_all": float(prof["slope"]),
                "z_band": [0, n], "r2_all": float(prof["r2"])}
    i0, i1 = int(lo * n), int(hi * n)
    z = np.asarray(prof["z"], dtype=float)[i0:i1]
    A = np.vstack([z, np.ones_like(z)]).T
    sol, *_ = np.linalg.lstsq(A, u[i0:i1], rcond=None)
    return {"gamma_bulk": float(sol[0]), "gamma_all": float(prof["slope"]),
            "z_band": [int(z[0]), int(z[-1])], "r2_all": float(prof["r2"])}


def self_consistent_point(
    particles,
    *,
    n_xy: int,
    n_z: int,
    nu: float = 0.2,
    gamma_dot: float,
    steps: int = 1300,
    warmup: int = 650,
    sample_every: int = 10,
    band: tuple[int, int] = (1, 4),
    max_iter: int = 3,
    tol: float = 0.01,
    labels: dict | None = None,
) -> dict:
    """Point measurement with the *self-consistent* frozen-structure kinematics.

    Why this exists
    ---------------
    The frozen-structure protocol advects every crystal with the **imposed**
    shear rate.  A suspension that thickens shears more slowly in the bulk than
    the carrier would, so crystals moving at the nominal rate drag the fluid
    along and the measured stress is biased low -- measured here as an apparent
    ``eta_r`` below the hard-sphere closures even in the dilute limit.  The
    physically consistent condition is that the crystal rigid-body field equals
    the **measured bulk** rate, i.e. the fixed point

        s* = gamma_bulk(s*) / gamma_dot_imposed,   u_particle = s* gamma_dot (z - z_mid)

    which is :meth:`ShearCell`'s ``particle_shear_scale``.  The iteration starts
    at ``s = 1`` (the plain protocol), and both passes are kept in the record so
    a reader can see the size of the correction instead of taking it on trust.

    Returns:
        The pass-2 record plus ``scale``, ``scale_history``, ``pass1`` (the
        nominal-protocol record) and ``correction_ratio`` = eta(pass2)/eta(pass1).
    """
    hist: list[dict] = []
    s = 1.0
    rec = None
    prev = None                                  # (s, f(s)) of the previous pass
    for _ in range(max(1, int(max_iter))):
        rec = _point_with_scale(particles, n_xy=n_xy, n_z=n_z, nu=nu, gamma_dot=gamma_dot,
                               scale=s, steps=steps, warmup=warmup,
                               sample_every=sample_every, band=band)
        f_s = rec["gamma_bulk"] / rec["gamma_dot_nominal"]
        hist.append({"scale": s, "eta_r": rec["eta_r"], "gamma_bulk": rec["gamma_bulk"],
                     "gamma_all": rec["gamma_all"], "map_value": f_s})
        if abs(f_s - s) <= tol * s:
            break
        if prev is None:
            prev = (s, f_s)
            s = f_s
            continue
        # secant step on the fixed point s = f(s): with a locally linear map this
        # lands on s* directly instead of contracting by |df/ds| per pass
        s0, f0 = prev
        d0, d1 = f0 - s0, f_s - s
        prev = (s, f_s)
        if abs(d1 - d0) > 1e-9:
            s_star = s + d1 * (s - s0) / (d1 - d0)
            s = float(np.clip(s_star, 0.2 * s, 3.0 * s))
        else:
            s = f_s
    out = dict(rec)
    out["scale"] = float(hist[-1]["scale"])
    out["scale_history"] = hist
    out["self_consistent"] = True
    if len(hist) > 1:
        out["pass1"] = {k: hist[0][k] for k in ("scale", "eta_r", "gamma_bulk", "gamma_all")}
        out["correction_ratio"] = float(hist[-1]["eta_r"] / hist[0]["eta_r"])
    if labels:
        out.update(labels)
    return out


def _point_with_scale(particles, *, n_xy, n_z, nu, gamma_dot, scale, steps, warmup,
                      sample_every, band) -> dict:
    """One shear-cell run at a fixed ``particle_shear_scale``."""
    from .ice_suspension_rheology.couette import ShearCellConfig

    u_wall = 0.5 * gamma_dot * (n_z - 2)
    cfg = ShearCellConfig(nx=n_xy, ny=n_xy, nz=n_z, nu=nu, u_wall=u_wall,
                          particle_shear_scale=float(scale))
    cell = ShearCell(cfg, particles)
    tau_flux, tau_wall = [], []
    for it in range(int(steps)):
        sample = it >= warmup and (it - warmup) % max(1, int(sample_every)) == 0
        out = cell.step(sample=sample, band=band)
        if sample:
            tau_flux.append(float(out["tau_near"]))
            tau_wall.append(_per_wall_exchange(cell, "bottom"))
    prof = cell.velocity_profile()
    bul = bulk_shear_rate(cell)
    g_all = float(prof["slope"])
    g_b = float(bul["gamma_bulk"])
    tau = float(np.mean(tau_flux))
    rec = {
        "tau_flux": tau,
        "tau_flux_sd": float(np.std(tau_flux)),
        "tau_wall": float(np.mean(tau_wall)),
        "gamma_all": g_all,
        "gamma_bulk": g_b,
        "gamma_dot_nominal": float(cfg.gamma_dot),
        "profile_r2": float(prof["r2"]),
        "profile": {"z": [float(v) for v in prof["z"]],
                    "u": [float(v) for v in prof["u"]]},
        "eta_eff": tau / g_b,
        "eta_r": (tau / g_b) / nu,
        "eta_r_all": (tau / g_all) / nu,
        "u_wall": float(u_wall),
        "n_samples": len(tau_flux),
        "n_steps": int(steps),
        "n_warmup": int(warmup),
        "scale": float(scale),
        "phi_lattice": float(cell.phi_lattice),
        "n_particles": int(cell.n_particles),
        "n_xy": int(n_xy),
        "n_z": int(n_z),
        "nu": float(nu),
        "gap_cells": int(n_z - 2),
    }
    if cell.particles:
        a = float(cell.particles[0].half_extent)
        rec["radius_cells"] = a
        rec["h_over_d"] = float((n_z - 2) / (2.0 * a))
        rec["re_p"] = float(cfg.gamma_dot * (2.0 * a) ** 2 / nu)
    else:
        rec["radius_cells"] = 0.0
        rec["h_over_d"] = 0.0
        rec["re_p"] = 0.0
    return rec

# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def _pick(records, **kw):
    out = []
    for r in records:
        if all(abs(float(r.get(k, -1)) - float(v)) < 1e-9 if isinstance(v, (int, float))
               else r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def evaluate_acceptance(*, pure_cases, ladder_fits, convergence, pytest_result=None,
                        phi_max_kd: float = 0.6) -> dict:
    """Judge the campaign against the node's six pre-declared criteria.

    Args:
        pure_cases: ``c = 0`` controls (:func:`pure_reference` records).
        ladder_fits: mapping ``(family, phi) -> {"eta_r_mean", "n", "K", "n_ci",
            "phi_lattice", ...}`` produced by the campaign assembler.
        convergence: mapping ``state -> :func:`grid_convergence_pair` output``.
        pytest_result: ``{"n_tests": int, "n_passed": int}`` or ``None``.
        phi_max_kd: maximum packing fraction of the Krieger-Dougherty closure.

    Returns:
        ``{"criteria": {...}, "n_pass": int, "n_defined": int, "all_pass": bool}``
        with the numbers, thresholds and an explicit boolean per criterion.
    """
    crit = {}

    # (1) zero-concentration negative control
    if not pure_cases:
        crit["c1_zero_control"] = {"passed": False, "why": "no c=0 case"}
    else:
        dev_n, dev_eta = [], []
        for r in pure_cases:
            dev_n.append(abs(float(r.get("n", 1.0)) - 1.0))
            dev_eta.append(abs(float(r["tau_flux"] / r["nu_gamma_dot"]) - 1.0))
        crit["c1_zero_control"] = {
            "passed": bool(max(dev_n) <= 0.01 and max(dev_eta) <= 0.02),
            "n": [float(r.get("n", 1.0)) for r in pure_cases],
            "max_n_dev": float(max(dev_n)),
            "max_mu_dev": float(max(dev_eta)),
            "thresholds": {"n_dev": 0.01, "mu_dev": 0.02},
            "n_cases": len(pure_cases),
        }

    # (2) dilute Einstein
    dilute = ladder_fits.get(("dispersed", 0.05))
    if not dilute:
        crit["c2_dilute_einstein"] = {"passed": False, "why": "no dispersed c=0.05"}
    else:
        pred = 1.0 + 2.5 * 0.05
        err = float(dilute["eta_r"]) - pred
        crit["c2_dilute_einstein"] = {
            "passed": bool(abs(err) <= 0.15 * pred),
            "eta_r": float(dilute["eta_r"]), "prediction": pred,
            "rel_err": float(err / pred), "threshold": 0.15,
        }

    # (3) concentrated Krieger-Dougherty
    conc = ladder_fits.get(("dispersed", 0.20))
    if not conc:
        crit["c3_krieger_dougherty"] = {"passed": False, "why": "no dispersed c=0.20"}
    else:
        phi = float(conc.get("phi_lattice") or 0.20)
        pred = float(krieger_dougherty(phi, phi_max=phi_max_kd))
        err = float(conc["eta_r"]) - pred
        crit["c3_krieger_dougherty"] = {
            "passed": bool(abs(err) <= 0.15 * pred),
            "phi_lattice": phi, "eta_r": float(conc["eta_r"]), "prediction": pred,
            "rel_err": float(err / pred), "threshold": 0.15,
            "phi_max": float(phi_max_kd),
        }

    # (4) shear thinning with a bootstrap interval that excludes n = 1
    thin = {}
    for (family, phi), fit in ladder_fits.items():
        if family == "flocculated" and float(phi) >= 0.10 - 1e-12:
            ci = fit.get("n_ci")
            thin[f"{phi:g}"] = {
                "n": float(fit["n"]), "n_ci": ci,
                "below_0p95": bool(float(fit["n"]) < 0.95),
                "ci_excludes_one": bool(fit.get("n_ci_excludes_one")),
                "passed": bool(float(fit["n"]) < 0.95 and fit.get("n_ci_excludes_one")),
            }
    crit["c4_shear_thinning"] = {
        "passed": bool(thin) and all(v["passed"] for v in thin.values()),
        "per_concentration": thin,
        "thresholds": {"n_max": 0.95, "ci": 0.95},
        "defined": bool(thin),
    }

    # (5) grid convergence
    conv = {k: v for k, v in (convergence or {}).items() if v}
    if not conv:
        crit["c5_grid_convergence"] = {"passed": False, "why": "second resolution not run"}
    else:
        crit["c5_grid_convergence"] = {
            "passed": bool(all(v["both_below_5pct"] for v in conv.values())),
            "per_state": {k: {"n_rel_diff": v["n_rel_diff"], "K_rel_diff": v["K_rel_diff"]}
                          for k, v in conv.items()},
            "threshold": 0.05,
        }

    # (6) pytest
    if not pytest_result:
        crit["c6_pytest"] = {"passed": False, "why": "pytest not run"}
    else:
        crit["c6_pytest"] = {
            "passed": bool(pytest_result.get("n_tests", 0) >= 10
                           and pytest_result.get("n_passed", 0) == pytest_result.get("n_tests")),
            **pytest_result, "threshold": {"n_tests": 10},
        }

    n_pass = sum(1 for v in crit.values() if v.get("passed"))
    return {"criteria": crit, "n_pass": n_pass, "n_defined": len(crit),
            "all_pass": bool(n_pass == len(crit))}