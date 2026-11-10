"""Structure-resolved power-law rheology of frazil ice slurries (D3Q19 shear cell).

Why this module exists
----------------------
The frazil closure shipped in :mod:`tensorlbm.frazil_rheology` prescribes
``K(phi)`` and ``n(phi)`` from literature fits; :mod:`tensorlbm.frazil_lbm`
then verifies that the *solver* transports that viscosity surface correctly.
Both stop short of the question a resistance/thrust-loss model actually cares
about: **given a suspension of ice crystals at concentration ``c``, what is the
measured shear stress ``tau(gamma_dot)``, and what power-law ``(K, n)`` does the
flow curve have?**  If that answer is injected from the closure it is a
restatement, not evidence.

This module supplies the missing face.  The chain is

    equilibrium floc structure(gamma_dot) -> frozen microgeometry
        -> D3Q19 moving-wall Couette with momentum-exchange coupling
        -> measured tau, gamma_dot -> least-squares log-log fit -> (K, n)

Two structural families (the central distinction)
-------------------------------------------------
* ``"dispersed"`` -- every primary crystal is an independent rigid particle
  (``n_primary = 1``).  This is the hard-sphere state, the state for which
  Einstein's ``[eta] = 5/2`` and Krieger-Dougherty are the correct closures, so
  the acceptance comparison against those closed forms is made here.
* ``"flocculated"`` -- crystals are bonded into fractal aggregates whose size is
  set by the shear-breakup balance of :func:`equilibrium_floc_size`.  Aggregates
  trap water (hydrodynamic volume fraction > solids fraction) and *break up as
  the shear rate rises*, which is the physical origin of frazil shear thinning
  (Barnes 1997).  The flow curve and its ``n < 1`` are measured here.

Reporting both families side by side shows *where* the shear thinning comes
from: in the dispersed family the measured exponent stays at ``n ~ 1`` (a
structure-frozen Newtonian suspension) and the deviation appears only once the
structure is allowed to follow the shear rate.  That is a falsifiable statement
about the mechanism and it is checked in the acceptance block of the campaign.

Equilibrium floc size (the single closure assumption, stated once)
------------------------------------------------------------------
An aggregate of ``N`` primaries of radius ``a`` and fractal dimension ``df`` has
hydrodynamic radius ``R = a N^(1/df)``.  The Stokes force a shear of rate
``gamma_dot`` exerts on it scales as ``mu gamma_dot R^2`` and it breaks when that
exceeds the cohesive force ``F_b``::

    mu gamma_dot R^2 = F_b  ->  R_eq = sqrt(F_b / (mu gamma_dot))
                                N_eq = (R_eq/a)^df = (F_b / (mu gamma_dot a^2))^(df/2)

i.e. ``R_eq ~ gamma_dot^(-1/2)``, the classical shear-limited equilibrium floc
radius (Sonntag & Russel 1986; Spicer & Pratsinis 1996).  ``F_b`` is not fitted:
it is *anchored* by declaring ``N_eq(gamma_ref) = N_cap`` (see
:func:`bond_force_from_reference`), and the anchored value is reported so a
reader can judge it.

Frozen-structure protocol (declared limitation)
-----------------------------------------------
The cell's particles are kinematic: each aggregate is advected rigidly with the
imposed shear (translation ``gamma_dot (z_c - z_mid)`` plus the half-vorticity
rotation ``Omega = gamma_dot/2`` about the *aggregate* centre, applied
identically to all its primaries so the aggregate stays rigid).  The geometry
does not re-arrange during a run, so the flow sees a stationary random
microstructure -- the standard frozen-configuration protocol for suspension
viscosity (Ladd 1994).  The shear-rate dependence therefore enters only through
the equilibrium structure, which is exactly what is being tested.

Conventions
-----------
Lattice units throughout (``rho_f = 1``).  ``tau = K gamma_dot^n`` with ``tau``
the plane momentum flux at the wall band, corrected by the
:class:`~tensorlbm.ice_suspension_rheology.couette.ShearCell` for the momentum
absorbed by the particles.  Relative viscosity is always a ratio of the *same*
measurement with and without particles at the same wall speed, so a residual
multiplicative bias of the stress estimator cancels.

References
----------
* Ladd, A. J. C. (1994) *J. Fluid Mech.* 271, 285.
* Einstein, A. (1906) *Ann. Phys.* 19, 289; (1911) *Ann. Phys.* 34, 591.
* Krieger, I. M. & Dougherty, T. J. (1959) *Trans. Soc. Rheol.* 3, 137.
* Barnes, H. A. (1997) *J. Non-Newtonian Fluid Mech.* 70, 1.
* Sonntag, R. C. & Russel, W. B. (1986) *J. Colloid Interface Sci.* 113, 399.
* Spicer, P. T. & Pratsinis, S. E. (1996) *Water Res.* 30, 1049.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

# The shear cell pulls in torch, and torch's device-backend autoload can fail on
# hosts where the SDAA/other extension is only half installed (observed as
# "Failed to load the backend extension: torch_sdaa").  This face is CPU-only, so
# the autoload is switched off unless the caller has already decided otherwise.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np

from .ice_suspension_rheology import shapes
from .ice_suspension_rheology.couette import ShearCell, ShearCellConfig

__all__ = [
    "DEFAULT_DF",
    "DEFAULT_N_CAP",
    "FrazilShearError",
    "FractalCluster",
    "fractal_cluster_cca",
    "equilibrium_floc_size",
    "bond_force_from_reference",
    "build_suspension",
    "pure_case",
    "run_shear_case",
    "fit_power_law",
    "relative_viscosity",
    "grid_convergence",
    "lattice_to_si",
    "evaluate_acceptance",
]

#: Fractal dimension of a diffusion-limited cluster-cluster aggregate of ice
#: crystals (reported frazil/colloidal flocs sit in ``1.8 ... 2.2``).
DEFAULT_DF = 2.0

#: Cap on the number of primaries per aggregate -- the largest floc the cell can
#: host at the lowest shear rate.  Anchors the bond-force constant.
DEFAULT_N_CAP = 8

_CONTACT_EPS = 1.0e-3


class FrazilShearError(ValueError):
    """Raised on inputs outside the validated envelope of the shear protocol."""


# ---------------------------------------------------------------------------
# Microgeometry: fractal aggregates
# ---------------------------------------------------------------------------


@dataclass
class FractalCluster:
    """A rigid aggregate of touching primary spheres (centres only)."""

    centres: np.ndarray  # (N, 3), centre of mass at the origin
    radius: float  # primary radius, lattice units
    df_declared: float = DEFAULT_DF

    @property
    def n_primary(self) -> int:
        return int(self.centres.shape[0])

    @property
    def rg(self) -> float:
        """Radius of gyration."""
        if self.n_primary <= 1:
            return 0.0
        return float(np.sqrt(np.mean(np.sum(self.centres ** 2, axis=1))))

    @property
    def rg_over_a(self) -> float:
        return self.rg / self.radius

    @property
    def hydrodynamic_radius(self) -> float:
        """``R = a N^(1/df)`` -- the radius the closure assigns to this mass."""
        return self.radius * self.n_primary ** (1.0 / self.df_declared)

    @property
    def df_implied(self) -> float:
        """``ln N / ln(Rg/a)`` -- the fractal dimension the geometry really has.

        Undefined (``nan``) for a single primary, where ``Rg = 0``.
        """
        if self.n_primary <= 1 or self.rg_over_a <= 1.05:
            # Rg/a <= 1 cannot support an estimate (N = 2 gives Rg = a exactly):
            # returning a number there would be a fabricated dimension.
            return float("nan")
        return float(math.log(self.n_primary) / math.log(self.rg_over_a))

    @property
    def bounding_radius(self) -> float:
        if self.n_primary <= 0:
            return 0.0
        return float(np.max(np.linalg.norm(self.centres, axis=1)) + self.radius)


def _unit_vector(rng) -> np.ndarray:
    v = rng.normal(size=3)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0])


def _random_rotation(rng) -> np.ndarray:
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _min_pair_distance(p: np.ndarray, q: np.ndarray) -> float:
    d = p[:, None, :] - q[None, :, :]
    return float(np.sqrt(np.min(np.sum(d * d, axis=2))))


def _merge_clusters(a: np.ndarray, b: np.ndarray, radius: float, rng) -> np.ndarray:
    """Slide cluster ``b`` along a random direction until it touches ``a``."""
    u = _unit_vector(rng)
    b_rot = b @ _random_rotation(rng).T
    shift = float((a @ u).max() - (b_rot @ u).min() + 2.0 * radius + _CONTACT_EPS) * u
    for _ in range(200):
        cand = b_rot + shift
        if _min_pair_distance(a, cand) >= 2.0 * radius - 1.0e-6:
            return np.vstack([a, cand])
        shift = shift + 0.05 * radius * u
    return np.vstack([a, b_rot + shift])


def fractal_cluster_cca(
    n_primary: int,
    radius: float,
    *,
    seed: int = 0,
    df_declared: float = DEFAULT_DF,
    max_attempts: int = 24,
) -> FractalCluster:
    """Hierarchical cluster-cluster aggregate of ``n_primary`` touching spheres.

    Two clusters are merged by rotating one randomly and sliding it along a
    random direction until its nearest primary touches the other's farthest
    primary -- the standard cluster-cluster recipe, which yields open
    (``df ~ 2``) aggregates rather than compact ball-like ones.  ``max_attempts``
    realisations are drawn and the one whose *implied* fractal dimension is
    closest to ``df_declared`` is kept, so the geometry matches the dimension the
    closure assumes.  The achieved ``df_implied`` is always reported.

    Args:
        n_primary: Number of primaries in the aggregate (>= 1).
        radius: Primary radius in lattice units.
        seed: RNG seed (reproducibility).
        df_declared: Target fractal dimension.
        max_attempts: Number of realisations to draw and score.

    Raises:
        FrazilShearError: if ``n_primary < 1`` or ``radius <= 0``.
    """
    n_primary = int(n_primary)
    if n_primary < 1:
        raise FrazilShearError(f"n_primary must be >= 1, got {n_primary}")
    if not (radius > 0.0):
        raise FrazilShearError(f"primary radius must be > 0, got {radius!r}")
    if n_primary == 1:
        return FractalCluster(centres=np.zeros((1, 3)), radius=radius, df_declared=df_declared)

    best: FractalCluster | None = None
    fallback: FractalCluster | None = None
    best_err = math.inf
    for attempt in range(max(1, int(max_attempts))):
        rng = np.random.default_rng(seed * 1000003 + attempt)
        clusters = [np.zeros((1, 3), dtype=float) for _ in range(n_primary)]
        sizes = [1] * n_primary
        while len(clusters) > 1:
            i, j = rng.choice(len(clusters), size=2, replace=False)
            merged = _merge_clusters(clusters[i], clusters[j], radius, rng)
            new_sizes = sizes[i] + sizes[j]
            for k in sorted([int(i), int(j)], reverse=True):
                clusters.pop(k)
                sizes.pop(k)
            clusters.append(merged)
            sizes.append(new_sizes)
            if len(clusters) == 1:
                break
        centres = clusters[0] - clusters[0].mean(axis=0)
        cl = FractalCluster(centres=centres, radius=radius, df_declared=df_declared)
        if fallback is None:
            fallback = cl
        if not math.isfinite(cl.df_implied):
            # a two-crystal aggregate has Rg ~ a, so its dimension is genuinely
            # undefined; keep it as a fallback rather than rejecting the mass
            continue
        err = abs(cl.df_implied - df_declared)
        if err < best_err:
            best, best_err = cl, err
    if best is not None:
        return best
    if fallback is not None:
        return fallback        # df_implied is nan: reported, not invented
    raise FrazilShearError("could not build any cluster geometry")


# ---------------------------------------------------------------------------
# Equilibrium floc size: the single closure assumption
# ---------------------------------------------------------------------------


def bond_force_from_reference(
    gamma_ref: float, radius: float, mu: float, n_ref: float, df: float = DEFAULT_DF
) -> float:
    """Anchor the aggregate cohesive force ``F_b`` on a declared reference state.

    Solves ``N_eq(gamma_ref) = n_ref`` for ``F_b``::

        F_b = mu * gamma_ref * (a n_ref^(1/df))^2

    This is an *anchoring*, not a fit: the reference state (largest floc at the
    lowest shear rate of the campaign) is declared before any simulation runs,
    and the implied bond force is reported in lattice units so a reader can
    judge whether it is physically sensible.

    Args:
        gamma_ref: Reference shear rate (lattice units).
        radius: Primary radius ``a`` (lattice units).
        mu: Dynamic viscosity of the carrier ``rho nu`` (lattice units).
        n_ref: Number of primaries the aggregate has at ``gamma_ref``.
        df: Fractal dimension.

    Returns:
        ``F_b`` in lattice force units.

    Raises:
        FrazilShearError: on non-positive inputs.
    """
    for name, value in (("gamma_ref", gamma_ref), ("radius", radius), ("mu", mu), ("n_ref", n_ref)):
        if not (float(value) > 0.0):
            raise FrazilShearError(f"{name} must be > 0, got {value!r}")
    r_ref = radius * float(n_ref) ** (1.0 / df)
    return float(mu) * float(gamma_ref) * r_ref ** 2


def equilibrium_floc_size(
    gamma_dot: float,
    *,
    radius: float,
    mu: float,
    bond_force: float,
    df: float = DEFAULT_DF,
    n_cap: float = DEFAULT_N_CAP,
) -> dict:
    """Equilibrium number of primaries per aggregate at shear rate ``gamma_dot``.

    ``R_eq = sqrt(F_b / (mu gamma_dot))`` and ``N_eq = (R_eq/a)^df``, clipped to
    ``[1, n_cap]``.  The clip is physical, not cosmetic: ``N < 1`` means the
    shear would have to break the primary crystal, so the aggregate has fully
    dispersed into single crystals there.

    Returns:
        ``{"n_primary": int, "radius": float, "gamma_dot": float,
        "regime": "flocculated"|"dispersed", "raw_n": float}``.
    """
    if not (float(gamma_dot) > 0.0):
        raise FrazilShearError(f"gamma_dot must be > 0, got {gamma_dot!r}")
    r_eq = math.sqrt(float(bond_force) / (float(mu) * float(gamma_dot)))
    raw = (r_eq / float(radius)) ** float(df)
    clipped = min(max(raw, 1.0), float(n_cap))
    n_int = max(1, int(round(clipped)))
    return {
        "gamma_dot": float(gamma_dot),
        "n_primary": n_int,
        "raw_n": float(raw),
        "radius": float(r_eq),
        "regime": "dispersed" if n_int <= 1 else "flocculated",
    }


# ---------------------------------------------------------------------------
# Suspension construction and placement
# ---------------------------------------------------------------------------


def _periodic_offsets(dx: float, dy: float, lx: float, ly: float):
    """Minimum-image offsets in the two periodic directions."""
    dx = dx - lx * np.round(dx / lx)
    dy = dy - ly * np.round(dy / ly)
    return dx, dy


def build_suspension(
    nx: int,
    ny: int,
    nz: int,
    phi_target: float,
    radius: float,
    *,
    n_primary: int = 1,
    seed: int = 0,
    wall_margin: float = 4.0,
    min_gap: float = 1.0,
    df: float = DEFAULT_DF,
    pool_size: int = 24,
    max_attempts: int = 6000,
):
    """Random sequential placement of (possibly flocculated) ice crystals.

    Each aggregate is placed as a rigid unit: primaries keep their relative
    positions and share the aggregate centre, which is what the shear cell needs
    in order to advect them as one rigid body.

    Non-overlap is enforced on the **primary centres** (``|r_i - r_j| >=
    2a + min_gap``, minimum image in the periodic ``x``/``y`` directions), not on
    the aggregate bounding spheres.  That distinction matters for open
    (``df = 2``) flocs, whose bounding spheres overlap almost immediately even
    though the crystals themselves are far apart; rejecting on bounding spheres
    would cap the achievable concentration at a few percent and the cap would be
    an artefact of the placement rule rather than physics.

    A fixed pool of ``pool_size`` aggregate realisations is drawn first and then
    reused, so the geometric cost is paid once instead of once per attempt.

    Returns:
        ``(particles, stats)`` where ``particles`` are
        :class:`~tensorlbm.ice_suspension_rheology.shapes.Particle` objects and
        ``stats`` reports the **achieved** volume fraction from the union of the
        masks (never the requested one), the aggregate count, the mean size,
        ``Rg/a``, the implied fractal dimension and the hydrodynamic volume
        fraction ``phi (R/a)^(3-df)`` the closure assigns to the structure.

    Raises:
        FrazilShearError: if the box cannot host even one aggregate, or if
            ``phi_target`` is outside ``[0, 1)``.
    """
    if not (0.0 <= float(phi_target) < 1.0):
        raise FrazilShearError(f"phi_target must be in [0, 1), got {phi_target!r}")
    if not (radius > 0.0):
        raise FrazilShearError(f"radius must be > 0, got {radius!r}")
    n_primary = max(1, int(n_primary))
    if nz <= 2 * (wall_margin + radius):
        raise FrazilShearError(
            f"box too small in z for radius={radius}: nz={nz}, wall_margin={wall_margin}"
        )
    rng = np.random.default_rng(seed)

    pool = [fractal_cluster_cca(n_primary, radius, seed=seed * 6151 + k, df_declared=df,
                                max_attempts=6)
            for k in range(max(1, int(pool_size)))]

    box_vol = float(nx * ny * nz)
    target_vol = float(phi_target) * box_vol
    v_prim = 4.0 / 3.0 * math.pi * radius ** 3

    placed: list[np.ndarray] = []      # absolute primary centres, (M, 3)
    centres: list[np.ndarray] = []     # aggregate centres, (K, 3)
    clusters: list[FractalCluster] = []
    solid_vol = 0.0
    attempts = 0
    need_sep = 2.0 * radius + float(min_gap)

    while solid_vol < target_vol and attempts < max_attempts:
        attempts += 1
        cl = pool[attempts % len(pool)]
        cz = rng.uniform(wall_margin + radius, nz - 1.0 - wall_margin - radius)
        cy = rng.uniform(0.0, ny)
        cx = rng.uniform(0.0, nx)
        cand = cl.centres + np.array([cz, cy, cx])
        z_lo = float(cand[:, 0].min())
        z_hi = float(cand[:, 0].max())
        if z_lo < wall_margin + radius * 0.0 or z_hi > nz - 1.0 - wall_margin:
            continue
        if placed:
            ex = np.vstack(placed)                      # (M, 3)
            d = cand[:, None, :] - ex[None, :, :]       # (N, M, 3)
            d[:, :, 1] -= ny * np.round(d[:, :, 1] / ny)
            d[:, :, 2] -= nx * np.round(d[:, :, 2] / nx)
            dist2 = np.sum(d * d, axis=2)
            if float(dist2.min()) < need_sep ** 2:
                continue
        placed.append(cand)
        centres.append(np.array([cz, cy, cx]))
        clusters.append(cl)
        solid_vol += cl.n_primary * v_prim

    particles = []
    for cl, centre in zip(clusters, centres):
        agg_centre = (float(centre[0]), float(centre[1]), float(centre[2]))
        for offset in cl.centres:
            prim_centre = (agg_centre[0] + float(offset[0]),
                           agg_centre[1] + float(offset[1]),
                           agg_centre[2] + float(offset[2]))
            mask = shapes.sphere_mask(nx, ny, nz, prim_centre, radius)
            if not mask.any():
                continue
            particles.append(
                shapes.Particle(
                    mask=mask,
                    centre=agg_centre,   # rigid-body velocity of the whole aggregate
                    half_extent=radius,
                    shape="sphere",
                    meta={"primary_centre": prim_centre, "n_primary": cl.n_primary},
                )
            )

    phi_lattice = shapes.measure_effective_fraction(particles)
    df_implied = [c.df_implied for c in clusters if math.isfinite(c.df_implied)]
    factor = [(c.hydrodynamic_radius / c.radius) ** (3.0 - df) for c in clusters]
    stats = {
        "phi_target": float(phi_target),
        "phi_lattice": float(phi_lattice),
        "n_aggregates": len(clusters),
        "n_primaries": int(sum(c.n_primary for c in clusters)),
        "mean_n_primary": float(np.mean([c.n_primary for c in clusters])) if clusters else 0.0,
        "rg_over_a_mean": float(np.mean([c.rg_over_a for c in clusters])) if clusters else 0.0,
        "df_implied_mean": float(np.mean(df_implied)) if df_implied else float("nan"),
        "df_declared": float(df),
        "hydrodynamic_phi": float(phi_lattice * float(np.mean(factor))) if factor else 0.0,
        "solid_volume_fraction_rsa": float(solid_vol / box_vol),
        "attempts": int(attempts),
        "radius": float(radius),
        "placement_saturated": bool(solid_vol < target_vol),
    }
    return particles, stats


# ---------------------------------------------------------------------------
# Shear-cell runs
# ---------------------------------------------------------------------------


def pure_case(
    *,
    nx: int,
    ny: int,
    nz: int,
    nu: float,
    u_wall: float,
    n_steps: int = 3200,
    n_warmup: int = 2000,
    sample_every: int = 20,
    wall_band: int = 4,
    dtype: str = "float32",
) -> dict:
    """Particle-free reference run -- the denominator of ``eta_r``."""
    cfg = ShearCellConfig(nx=nx, ny=ny, nz=nz, nu=nu, u_wall=u_wall, dtype=dtype)
    cell = ShearCell(cfg, [], wall_band=wall_band)
    r = cell.run(n_steps, n_warmup=n_warmup, sample_every=sample_every)
    r.pop("profile", None)
    r.pop("tau_rows_mean", None)
    # canonical names so a record from either entry point is consumed identically
    r["tau"] = float(r["tau_corrected"])
    r["gamma_dot"] = float(r["gamma_dot_measured"])
    r["re_p"] = float(cfg.gamma_dot * (2.0 * cfg.nu) ** 2 / cfg.nu) if False else 0.0
    r.update({"family": "pure", "u_wall": float(u_wall), "nu": float(nu),
              "box": [int(nx), int(ny), int(nz)], "re_p": 0.0})
    return r


def run_shear_case(
    *,
    phi: float,
    u_wall: float,
    nx: int,
    ny: int,
    nz: int,
    radius: float,
    nu: float,
    n_primary: int = 1,
    seed: int = 0,
    df: float = DEFAULT_DF,
    n_steps: int = 3200,
    n_warmup: int = 2000,
    sample_every: int = 20,
    wall_margin: int = 4,
    placement_attempts: int = 40000,
    dtype: str = "float32",
) -> dict:
    """One shear-cell case: build the suspension, run, return measured quantities.

    Returns a JSON-serialisable record holding ``tau`` (plane-flux stress with
    the particle momentum sink removed), the *measured* shear rate (slope of the
    linear fit to the mean velocity profile), the profile ``R^2``, the achieved
    lattice volume fraction and the structure statistics.
    """
    import torch  # local import keeps module import cheap for analytic users

    particles, stats = build_suspension(
        nx, ny, nz, phi, radius, n_primary=n_primary, seed=seed, wall_margin=wall_margin,
        df=df, max_attempts=placement_attempts,
    )
    cfg = ShearCellConfig(nx=nx, ny=ny, nz=nz, nu=nu, u_wall=u_wall, dtype=dtype)
    cell = ShearCell(cfg, particles, wall_band=max(2, int(wall_margin) - 1))
    r = cell.run(n_steps, n_warmup=n_warmup, sample_every=sample_every)
    r.pop("profile", None)
    r.pop("tau_rows_mean", None)
    rec = {
        "family": "flocculated" if n_primary > 1 else "dispersed",
        "phi": float(phi),
        "u_wall": float(u_wall),
        "nu": float(nu),
        "radius": float(radius),
        "seed": int(seed),
        "n_primary_planned": int(n_primary),
        "tau": float(r["tau_corrected"]),
        "tau_near": float(r["tau_near"]),
        "f_particle_per_area": float(r["f_particle_per_area"]),
        "gamma_dot": float(r["gamma_dot_measured"]),
        "gamma_dot_nominal": float(r["gamma_dot_nominal"]),
        "profile_r2": float(r["profile_r2"]),
        "n_particles": int(r["n_particles"]),
        "node_overlaps": int(r["node_overlaps"]),
        "re_p": float(cfg.gamma_dot * (2.0 * radius) ** 2 / nu),
        "box": [int(nx), int(ny), int(nz)],
        "steps": int(n_steps),
        "warmup": int(n_warmup),
        "samples": int(r["n_samples"]),
        "torch_threads": int(torch.get_num_threads()),
    }
    rec.update({k: v for k, v in stats.items() if k not in ("phi_target",)})
    return rec


def relative_viscosity(case: dict, pure: dict) -> dict:
    """``eta_r = eta_susp / eta_pure`` from the two measured flow curves points.

    Both sides use the *measured* shear rate, so a systematic error of the
    velocity-profile fit cancels to first order.
    """
    if not (pure["gamma_dot"] > 0 and case["gamma_dot"] > 0):
        raise FrazilShearError("measured shear rate must be > 0 for both runs")
    eta_s = case["tau"] / case["gamma_dot"]
    eta_p = pure["tau"] / pure["gamma_dot"]
    return {
        "eta_susp": float(eta_s),
        "eta_pure": float(eta_p),
        "eta_r": float(eta_s / eta_p),
        "tau_ratio": float(case["tau"] / pure["tau"]),
        "gamma_ratio": float(case["gamma_dot"] / pure["gamma_dot"]),
    }


# ---------------------------------------------------------------------------
# Flow-curve fitting
# ---------------------------------------------------------------------------


def fit_power_law(gamma_dot, tau, *, weights=None) -> dict:
    """Least-squares fit of ``tau = K gamma_dot^n`` in log-log space.

    Args:
        gamma_dot: Shear rates (all > 0).
        tau: Measured stresses (all > 0).
        weights: Optional non-negative weights (e.g. inverse variance).

    Returns:
        ``{"K", "n", "logK", "r2", "n_stderr", "n_points", "gamma_range"}``.
        ``n_stderr`` is the standard error of the slope (``nan`` for < 3 points,
        where it is not defined rather than invented).

    Raises:
        FrazilShearError: on mismatched lengths, fewer than 2 points, or
            non-positive data (a log-log fit is undefined there).
    """
    g = np.asarray(gamma_dot, dtype=float).ravel()
    t = np.asarray(tau, dtype=float).ravel()
    if g.size != t.size:
        raise FrazilShearError(f"gamma_dot and tau differ in length ({g.size} vs {t.size})")
    if g.size < 2:
        raise FrazilShearError("a power-law fit needs at least 2 points")
    if np.any(g <= 0) or np.any(t <= 0):
        raise FrazilShearError("log-log fit requires strictly positive gamma_dot and tau")
    x = np.log(g)
    y = np.log(t)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.size != x.size:
        raise FrazilShearError("weights length mismatch")
    sw = w.sum()
    xm = float((w * x).sum() / sw)
    ym = float((w * y).sum() / sw)
    sxx = float((w * (x - xm) ** 2).sum())
    sxy = float((w * (x - xm) * (y - ym)).sum())
    if sxx <= 0.0:
        raise FrazilShearError("all shear rates are identical: slope is undetermined")
    n = sxy / sxx
    logK = ym - n * xm
    pred = logK + n * x
    ss_res = float((w * (y - pred) ** 2).sum())
    ss_tot = float((w * (y - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n_stderr = (
        float(math.sqrt(ss_res / (sw * (g.size - 2) / sw) / sxx)) if g.size > 2 and sxx > 0 else float("nan")
    )
    return {
        "K": float(math.exp(logK)),
        "n": float(n),
        "logK": float(logK),
        "r2": float(r2),
        "n_stderr": n_stderr,
        "n_points": int(g.size),
        "gamma_range": [float(g.min()), float(g.max())],
        "tau_range": [float(t.min()), float(t.max())],
    }


def grid_convergence(coarse: dict, fine: dict) -> dict:
    """Relative differences of ``(K, n)`` between two resolutions.

    ``coarse``/``fine`` are :func:`fit_power_law` results.  Returns the signed
    relative differences and the boolean the acceptance criterion needs.
    """
    dn = abs(fine["n"] - coarse["n"]) / abs(fine["n"]) if fine["n"] else float("inf")
    dK = abs(fine["K"] - coarse["K"]) / abs(fine["K"]) if fine["K"] else float("inf")
    return {
        "n_coarse": coarse["n"],
        "n_fine": fine["n"],
        "K_coarse": coarse["K"],
        "K_fine": fine["K"],
        "n_rel_diff": float(dn),
        "K_rel_diff": float(dK),
        "both_below_5pct": bool(dn < 0.05 and dK < 0.05),
    }


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def lattice_to_si(K_lattice: float, n: float, *, dx: float, rho: float, nu_si: float,
                  nu_lattice: float) -> dict:
    """Map a lattice-unit power-law pair ``(K, n)`` to SI.

    The lattice-to-physical map is fixed by three declared scales: the grid
    spacing ``dx`` [m] (set by the primary crystal radius), the density
    ``rho`` [kg/m^3] and the kinematic viscosity ``nu_si`` [m^2/s] of the carrier
    at the simulated ``nu_lattice``.  Time follows from the viscosity match::

        dt = nu_lattice * dx^2 / nu_si
        gamma_dot_SI = gamma_dot_lattice / dt
        tau_SI = tau_lattice * rho * dx^2 / dt^2  = K_SI gamma_dot_SI^n
        -> K_SI = K_lattice * rho * dx^2 / dt^2 * dt^n

    Returns the SI pair plus every intermediate scale, so the conversion is
    auditable rather than hidden.
    """
    dt = float(nu_lattice) * dx ** 2 / nu_si
    u_scale = dx / dt
    tau_scale = rho * dx ** 2 / dt ** 2
    K_si = K_lattice * tau_scale * dt ** n
    return {
        "dx_m": float(dx),
        "dt_s": float(dt),
        "u_scale_m_s": float(u_scale),
        "tau_scale_Pa": float(tau_scale),
        "K_SI": float(K_si),
        "K_SI_unit": f"Pa s^{n:.3f}",
        "n": float(n),
        "nu_si_m2_s": float(nu_si),
    }


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def _at(table: dict, phi: float):
    """Fetch a concentration entry by value, whatever the key formatting is."""
    for key, val in (table or {}).items():
        try:
            if abs(float(key) - float(phi)) < 1e-9:
                return val
        except (TypeError, ValueError):
            continue
    return None


def evaluate_acceptance(*, dispersed: dict, flocculated: dict, convergence: dict | None) -> dict:
    """Judge the campaign against the five pre-declared criteria.

    Args:
        dispersed: ``{phi: {"n": ..., "K": ..., "eta_r": ..., "eta_r_ref": ...}}``
            for the non-aggregated (hard-sphere) family.
        flocculated: same mapping for the structure-following family.
        convergence: :func:`grid_convergence` output, or ``None`` when the
            second resolution has not been run yet.

    Returns:
        ``{"criteria": {...}, "all_pass": bool, "n_pass": int}`` where each
        criterion carries its numbers, its threshold and an explicit boolean.
    """
    crit = {}

    # (1) zero-concentration negative control
    zero = _at(dispersed, 0.0) or _at(flocculated, 0.0)
    if zero is None:
        crit["c1_zero_control"] = {"defined": False, "passed": False,
                                   "why": "no c=0 case in the artifact"}
    else:
        n_dev = abs(zero["n"] - 1.0)
        mu_dev = abs(zero.get("eta_r", float("nan")) - 1.0)
        crit["c1_zero_control"] = {
            "defined": True,
            "n": zero["n"],
            "n_dev": n_dev,
            "eta_r": zero.get("eta_r"),
            "eta_r_dev": mu_dev,
            "n_passed": bool(n_dev <= 0.01),
            "mu_passed": bool(mu_dev <= 0.02),
            "passed": bool(n_dev <= 0.01 and mu_dev <= 0.02),
        }

    # (2) dilute Einstein slope
    c05 = _at(dispersed, 0.05)
    if c05 is None:
        crit["c2_dilute_einstein"] = {"defined": False, "passed": False,
                                      "why": "no c=0.05 dispersed case"}
    else:
        pred = 1.0 + 2.5 * 0.05
        err = (c05["eta_r"] - pred) / pred
        crit["c2_dilute_einstein"] = {
            "defined": True,
            "eta_r": c05["eta_r"],
            "prediction": pred,
            "rel_err": float(err),
            "passed": bool(abs(err) <= 0.15),
        }

    # (3) concentrated Krieger-Dougherty
    c20 = _at(dispersed, 0.20)
    if c20 is None:
        crit["c3_krieger_dougherty"] = {"defined": False, "passed": False,
                                        "why": "no c=0.20 dispersed case"}
    else:
        # the KD comparison is a statement about the *hydrodynamic* volume
        # fraction; when a drag calibration is present the mask fraction is
        # converted with phi_h = phi_mask (a_h/a_eff)^3 measured in the same cell
        phi_h = c20.get("phi_hydrodynamic")
        phi = phi_h or c20.get("phi_lattice") or 0.20
        pred = (1.0 - phi / 0.60) ** -1.5 if phi_h else c20.get(
            "kd_prediction", (1.0 - phi / 0.60) ** -1.5)
        err = (c20["eta_r"] - pred) / pred
        crit["c3_krieger_dougherty"] = {
            "defined": True,
            "phi_lattice": c20.get("phi_lattice"),
            "phi_hydrodynamic": phi_h,
            "phi_used": phi,
            "eta_r": c20["eta_r"],
            "prediction": pred,
            "rel_err": float(err),
            "passed": bool(abs(err) <= 0.15),
        }

    # (4) shear thinning of the structure-following family
    thin = {}
    for key, val in flocculated.items():
        if float(key) >= 0.10 - 1e-12:
            thin[key] = {"n": val["n"], "passed": bool(val["n"] < 0.95)}
    crit["c4_shear_thinning"] = {
        "defined": bool(thin),
        "per_concentration": thin,
        "passed": bool(thin) and all(v["passed"] for v in thin.values()),
    }

    # (5) grid convergence
    if convergence is None:
        crit["c5_grid_convergence"] = {"defined": False, "passed": False,
                                       "why": "second resolution not run"}
    else:
        crit["c5_grid_convergence"] = {
            "defined": True,
            "n_rel_diff": convergence["n_rel_diff"],
            "K_rel_diff": convergence["K_rel_diff"],
            "passed": bool(convergence["both_below_5pct"]),
        }

    n_pass = sum(1 for v in crit.values() if v.get("passed"))
    return {"criteria": crit, "n_pass": int(n_pass), "all_pass": bool(n_pass == len(crit))}