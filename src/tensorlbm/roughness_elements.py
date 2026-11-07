"""Voxelised k-type roughness elements for wall-resolved 3-D LBM channels.

The module generates *geometric* roughness — the discrete solid voxels that a
wall-resolved lattice-Boltzmann channel actually feels — for three morphologies
that matter for ship hydrodynamics:

``"sand"``
    Randomly packed sand-grain asperities (Nikuradse-type granular roughness).
    Grains are spheres of nominal diameter ``ks`` with jittered radius,
    jittered position and occasional clustering, grown on the wall layer and
    clipped to the domain.  This is the morphology behind the classical
    ``ks``-based friction lines (Nikuradse 1933, Colebrook 1939, ITTC-78).

``"ribs"``
    Transverse (spanwise) square ribs of height and width ``ks`` with pitch
    ``p``.  This is the canonical *k*-type roughness of Perry, Schofield &
    Joubert (1969) and the closest idealisation of a machined/scribed hull
    surface; it isolates the "skimming flow" behaviour from grain randomness.

``"ice_gouges"``
    Directional ridges (ice scoring / abrasion marks) elongated at an angle
    :math:`\\theta` to the freestream, with quasi-periodic spacing and along-
    ridge height modulation.  This morphology is what an ice-abraded hull or a
    propeller-blade leading edge actually looks like (see ITTC 7.5-02-03-01.4
    " roughness and fouling" and ice-class hull surveys).

Conventions
-----------
* Arrays are ``(nz, ny, nx)`` with ``z`` the wall-normal direction, ``z = 0``
  the bottom solid wall layer and ``z = nz - 1`` the top solid wall layer;
  ``x`` (streamwise) and ``y`` (spanwise) are periodic.
* :func:`element_mask` returns **only the roughness voxels** (``z >= 1`` and
  ``z <= nz - 2``); :func:`channel_solid_mask` folds them into the channel
  walls and (optionally) mirrors them onto the opposite wall so that both
  walls are rough, which doubles the available statistics of a forced channel.
* ``ks_nominal`` is the *geometric* element height (sphere diameter, rib
  height, ridge height).  The hydrodynamic ("equivalent sand-grain") height
  ``ks_eff`` is **not** assumed to equal it; it is measured from the computed
  roughness function (see :mod:`tensorlbm.roughness_laws`).  Keeping the two
  separate is what makes the morphology sensitivity (sand vs ribs vs gouges)
  visible instead of being hidden in a single fudge factor.

All generators are deterministic functions of their ``seed``, so a reported
``ks+`` is reproducible from ``(morphology, ks, seed, grid)`` alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

__all__ = [
    "KINDS",
    "RoughnessElementSet",
    "sand_grain_elements",
    "transverse_rib_elements",
    "ice_gouge_elements",
    "element_mask",
    "channel_solid_mask",
    "roughness_height_stats",
    "fluid_half_height",
]

KINDS: tuple[str, ...] = ("sand", "ribs", "ice_gouges")

#: Element height may not exceed this fraction of the channel height, otherwise
#: the "channel" degenerates into an obstacle array and no log layer exists.
MAX_KS_OVER_H = 0.45


@dataclass(frozen=True)
class RoughnessElementSet:
    """A voxelised roughness geometry plus the provenance needed to report it.

    Attributes
    ----------
    kind:
        One of :data:`KINDS`.
    mask:
        Boolean ``(nz, ny, nx)`` array, ``True`` on roughness voxels only.
    ks_nominal:
        Geometric element height in lattice units (sphere diameter, rib
        height, ridge height).
    params:
        Generator parameters actually used (pitch, angle, coverage, seed ...).
        Stored so that a case is reproducible from its JSON record alone.
    """

    kind: str
    mask: np.ndarray
    ks_nominal: float
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.mask.shape)  # type: ignore[return-value]

    @property
    def solid_fraction(self) -> float:
        return float(self.mask.mean())

    def stats(self) -> dict[str, Any]:
        """Geometry descriptors (heights, coverage, surface area)."""
        st = roughness_height_stats(self.mask, self.nx, self.ny)
        st["kind"] = self.kind
        st["ks_nominal"] = float(self.ks_nominal)
        st["params"] = {k: _jsonable(v) for k, v in self.params.items()}
        return st

    # ``nx``/``ny`` convenience accessors keep call sites readable.
    @property
    def nx(self) -> int:
        return int(self.mask.shape[2])

    @property
    def ny(self) -> int:
        return int(self.mask.shape[1])

    @property
    def nz(self) -> int:
        return int(self.mask.shape[0])


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _check_grid(nx: int, ny: int, nz: int) -> None:
    if min(int(nx), int(ny), int(nz)) < 4:
        raise ValueError(f"grid too small for a rough channel: nx={nx}, ny={ny}, nz={nz}")
    if int(nz) < 5:
        raise ValueError("nz must be >= 5 (two wall layers + at least three fluid layers)")


def _check_ks(ks: float, nz: int) -> float:
    ks = float(ks)
    if not math.isfinite(ks) or ks <= 0.0:
        raise ValueError(f"ks must be a positive finite length, got {ks!r}")
    half = (int(nz) - 2) / 2.0
    if ks > MAX_KS_OVER_H * half:
        raise ValueError(
            f"ks={ks:g} exceeds {MAX_KS_OVER_H:.2f} x half-channel height ({half:g}); "
            "the log layer would vanish"
        )
    return ks


def _sphere_offsets(radius: float) -> np.ndarray:
    """Voxel offsets of a digital sphere of radius ``radius`` (lattice units).

    A radius below 0.5 still returns a single voxel so that sub-cell ``ks``
    requests degrade gracefully to a one-cell asperity instead of vanishing.
    """
    r = max(float(radius), 0.5)
    m = int(math.ceil(r))
    g = np.arange(-m, m + 1)
    zz, yy, xx = np.meshgrid(g, g, g, indexing="ij")
    keep = zz**2 + yy**2 + xx**2 <= r * r + 1e-9
    return np.stack([zz[keep], yy[keep], xx[keep]], axis=1)


def sand_grain_elements(
    nx: int,
    ny: int,
    nz: int,
    ks: float,
    *,
    seed: int = 0,
    coverage: float = 0.35,
    radius_jitter: float = 0.25,
    height_jitter: float = 0.15,
) -> RoughnessElementSet:
    """Randomly packed sand grains of nominal diameter ``ks`` on the wall layer.

    Grains are hemispheres-to-spheres of radius ``ks/2`` (jittered) whose tops
    sit at ``ks * (1 +/- height_jitter)`` above the wall layer.  Placement uses
    a stratified random sample of the ``nx x ny`` footprint with ``coverage``
    the target fraction of wetted columns carrying at least one grain, so the
    packing density is grid independent (a physically meaningful knobs) rather
    than a raw grain count.
    """
    _check_grid(nx, ny, nz)
    ks = _check_ks(ks, nz)
    if not (0.0 < coverage <= 1.0):
        raise ValueError(f"coverage must be in (0, 1], got {coverage!r}")

    rng = np.random.default_rng(int(seed))
    r_mean = max(0.5 * ks, 0.5)
    area_grain = math.pi * r_mean**2
    n_target = int(round(coverage * nx * ny / max(area_grain, 1.0)))
    n_target = max(n_target, 1)

    mask = np.zeros((nz, ny, nx), dtype=bool)
    # Stratified (jittered-grid) sampling: better mixing than purely random
    # placement and far fewer unfilled holes at low coverage.
    cols = nx * ny
    n_cells = min(n_target, cols)
    cells = rng.permutation(cols)[:n_cells]
    for c in cells:
        iy, ix = divmod(int(c), nx)
        cx = ix + rng.uniform(-0.5, 0.5)
        cy = iy + rng.uniform(-0.5, 0.5)
        r = r_mean * (1.0 + radius_jitter * rng.uniform(-1.0, 1.0))
        r = max(r, 0.5)
        top = ks * (1.0 + height_jitter * rng.uniform(-1.0, 1.0))
        # Sphere centre placed so that the crest reaches ``top`` above the
        # bottom wall layer; the sphere is clipped by the wall (z >= 1).
        cz = top - r
        off = _sphere_offsets(r)
        zz = off[:, 0] + int(round(cz))
        yy = (off[:, 1] + int(round(cy))) % ny
        xx = (off[:, 2] + int(round(cx))) % nx
        sel = (zz >= 1) & (zz <= nz - 2)
        mask[zz[sel], yy[sel], xx[sel]] = True

    return RoughnessElementSet(
        kind="sand",
        mask=mask,
        ks_nominal=ks,
        params={
            "seed": int(seed),
            "coverage": float(coverage),
            "radius_jitter": float(radius_jitter),
            "height_jitter": float(height_jitter),
            "n_grains": int(n_cells),
            "nx": int(nx),
            "ny": int(ny),
            "nz": int(nz),
        },
    )


def transverse_rib_elements(
    nx: int,
    ny: int,
    nz: int,
    ks: float,
    *,
    pitch_over_ks: float = 8.0,
    width_over_ks: float = 1.0,
    phase: int = 0,
) -> RoughnessElementSet:
    """Spanwise square ribs (height ``ks``, width ``w``, pitch ``p``).

    ``p/ks = 8`` (default) places the element in the classical *k*-type regime
    ``p/ks >~ 4`` where the roughness function collapses onto the sand-grain
    curve with ``ks_eff ~ ks`` (Perry et al. 1969); smaller pitches approach
    *d*-type behaviour and are rejected by the sampler used in the campaign.
    """
    _check_grid(nx, ny, nz)
    ks = _check_ks(ks, nz)
    if pitch_over_ks < 1.5:
        raise ValueError("pitch must be >= 1.5 ks (otherwise ribs merge into a wall)")
    if not (0.1 <= width_over_ks <= 2.0):
        raise ValueError("width/ks must be in [0.1, 2.0]")

    pitch = pitch_over_ks * ks
    width = max(1, int(round(width_over_ks * ks)))
    height = max(1, int(round(ks)))
    mask = np.zeros((nz, ny, nx), dtype=bool)
    x0 = int(phase) % nx
    x = 0
    n_ribs = 0
    while x < nx:
        i0 = (x0 + x) % nx
        idx = (np.arange(i0, i0 + width) % nx)[:, None]
        mask[1 : 1 + height, :, idx] = True
        n_ribs += 1
        x += max(1, int(round(pitch)))

    return RoughnessElementSet(
        kind="ribs",
        mask=mask,
        ks_nominal=ks,
        params={
            "pitch_over_ks": float(pitch_over_ks),
            "width_over_ks": float(width_over_ks),
            "pitch": float(pitch),
            "width": int(width),
            "height": int(height),
            "n_ribs": int(n_ribs),
            "phase": int(phase),
            "nx": int(nx),
            "ny": int(ny),
            "nz": int(nz),
        },
    )


def ice_gouge_elements(
    nx: int,
    ny: int,
    nz: int,
    ks: float,
    *,
    angle_deg: float = 30.0,
    spacing_over_ks: float = 4.0,
    width_over_ks: float = 0.6,
    waviness: float = 0.25,
    seed: int = 0,
) -> RoughnessElementSet:
    """Directional ice-scoring ridges at ``angle_deg`` to the freestream.

    Ice abrasion leaves quasi-parallel scoring marks; each mark is modelled as
    a ridge of height ``ks`` (modulated along its length by ``waviness``) and
    width ``width_over_ks * ks``, repeated with spacing ``spacing_over_ks*ks``
    measured normal to the ridge direction.  The directionality is the point:
    a grooved surface is *not* equivalent to a sand surface of the same height,
    and this generator lets that difference be measured instead of assumed.
    """
    _check_grid(nx, ny, nz)
    ks = _check_ks(ks, nz)
    if not (0.0 <= angle_deg < 90.0):
        raise ValueError("angle_deg must be in [0, 90) (angle to the x/freestream axis)")
    if spacing_over_ks < 1.5:
        raise ValueError("ridge spacing must be >= 1.5 ks")
    if not (0.1 <= width_over_ks <= 1.5):
        raise ValueError("ridge width/ks must be in [0.1, 1.5]")

    rng = np.random.default_rng(int(seed))
    theta = math.radians(float(angle_deg))
    # Normal to the ridge direction; distance along the normal selects a ridge.
    nrm = np.array([-math.sin(theta), math.cos(theta)])
    along = np.array([math.cos(theta), math.sin(theta)])
    spacing = spacing_over_ks * ks
    width = max(width_over_ks * ks, 0.6)
    height = max(1, int(round(ks)))

    xs = np.arange(nx, dtype=float) + 0.5
    ys = np.arange(ny, dtype=float) + 0.5
    XX, YY = np.meshgrid(xs, ys, indexing="xy")  # (ny, nx)
    dist = XX * nrm[0] + YY * nrm[1]
    phase_shift = rng.uniform(0.0, spacing)
    s = XX * along[0] + YY * along[1]

    # Height field: crest height per column, modulated along the ridge.
    ridge_idx = np.floor((dist - phase_shift) / spacing)
    frac = (dist - phase_shift) / spacing - ridge_idx
    in_ridge = np.abs(frac - 0.5) * spacing <= 0.5 * width
    # Random per-ridge height offset keeps neighbouring scores unequal.
    ridge_hash = (ridge_idx.astype(np.int64) * 2654435761) % 1000 / 1000.0
    h_along = 1.0 + waviness * np.sin(2.0 * math.pi * s / (3.0 * spacing))
    h_along = h_along * (0.9 + 0.2 * ridge_hash)
    crest = np.where(in_ridge, ks * h_along, 0.0)
    n_col = np.clip(np.round(crest).astype(int), 0, height)

    mask = np.zeros((nz, ny, nx), dtype=bool)
    for k in range(1, height + 1):
        mask[k] = n_col >= k

    return RoughnessElementSet(
        kind="ice_gouges",
        mask=mask,
        ks_nominal=ks,
        params={
            "angle_deg": float(angle_deg),
            "spacing_over_ks": float(spacing_over_ks),
            "spacing": float(spacing),
            "width_over_ks": float(width_over_ks),
            "waviness": float(waviness),
            "seed": int(seed),
            "phase_shift": float(phase_shift),
            "nx": int(nx),
            "ny": int(ny),
            "nz": int(nz),
        },
    )


def element_mask(
    kind: str,
    nx: int,
    ny: int,
    nz: int,
    ks: float,
    *,
    seed: int = 0,
    **kwargs: Any,
) -> RoughnessElementSet:
    """Dispatch to the generator named by ``kind`` (thin registry helper)."""
    if kind == "sand":
        return sand_grain_elements(nx, ny, nz, ks, seed=seed, **kwargs)
    if kind == "ribs":
        return transverse_rib_elements(nx, ny, nz, ks, **kwargs)
    if kind == "ice_gouges":
        return ice_gouge_elements(nx, ny, nz, ks, seed=seed, **kwargs)
    raise ValueError(f"unknown roughness kind {kind!r}; expected one of {KINDS}")


def channel_solid_mask(
    elements: RoughnessElementSet | np.ndarray,
    *,
    mirror_to_both_walls: bool = True,
) -> np.ndarray:
    """Fold roughness voxels into the two solid wall layers of the channel.

    With ``mirror_to_both_walls`` the element pattern is point-reflected about
    the mid-plane (``z -> nz-1-z``), giving a rough/rough channel.  The mirror
    plane is exact, so the mean profile stays symmetric and the sampling error
    of the mean velocity roughly halves for the same number of steps.
    """
    mask = elements.mask if isinstance(elements, RoughnessElementSet) else np.asarray(elements)
    if mask.ndim != 3:
        raise ValueError(f"element mask must be 3-D (nz, ny, nx), got shape {mask.shape}")
    nz = mask.shape[0]
    solid = np.zeros_like(mask, dtype=bool)
    solid[0, :, :] = True
    solid[-1, :, :] = True
    solid |= mask
    if mirror_to_both_walls:
        solid |= mask[::-1, :, :]
        solid[0, :, :] = True
        solid[-1, :, :] = True
    return solid


def roughness_height_stats(mask: np.ndarray, nx: int | None = None, ny: int | None = None) -> dict[str, Any]:
    """Geometric descriptors of a voxel roughness field.

    Returns column-resolved heights (``h_mean``/``h_rms``/``h_max`` in lattice
    units measured above the bottom wall layer), the wetted-column coverage and
    a voxel surface-area estimate (needed to report the wetted-area penalty of
    each morphology).
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3:
        raise ValueError("mask must be 3-D")
    nz, ny_, nx_ = m.shape
    nx = nx_ if nx is None else int(nx)
    ny = ny_ if ny is None else int(ny)
    if nx != nx_ or ny != ny_:
        raise ValueError("nx/ny do not match mask shape")

    heights = np.zeros((ny, nx), dtype=float)
    for k in range(nz - 1, 0, -1):
        layer = m[k] & (heights == 0.0)
        heights[layer] = k  # top-most occupied layer index == height above wall
    wetted = heights > 0
    h_max = float(heights.max()) if m.any() else 0.0
    h_mean = float(heights[wetted].mean()) if wetted.any() else 0.0
    h_rms = float(np.sqrt((heights[wetted] ** 2).mean())) if wetted.any() else 0.0
    # Exposed voxel faces (surface area in lattice units^2).  Periodic in x/y.
    solid = m.copy()
    surf = 0
    for axis in (0, 1, 2):
        nb = np.roll(solid, 1, axis=axis)
        if axis == 0:
            # wall-normal: no wrapping across the domain ends (they are walls)
            nb[0] = True
            nb[-1] = True
        surf += int((solid & ~nb).sum())
    return {
        "n_voxels": int(m.sum()),
        "solid_fraction": float(m.mean()),
        "coverage": float(wetted.mean()),
        "h_mean": h_mean,
        "h_rms": h_rms,
        "h_max": h_max,
        "exposed_area_lattice": surf,
        "exposed_area_ratio": float(surf / (2.0 * nx * ny)) if nx and ny else 0.0,
    }


def fluid_half_height(solid: np.ndarray) -> float:
    """Mean fluid half-height ``h_eff = N_fluid / (2 nx ny)`` (lattice units).

    This is the height that makes the global momentum balance
    ``tau_w = rho g h_eff`` exact for a force-driven rough channel, so it is
    the correct length to fix the friction velocity and ``Re_tau`` with.
    """
    m = np.asarray(solid, dtype=bool)
    if m.ndim != 3:
        raise ValueError("solid mask must be 3-D")
    nz, ny, nx = m.shape
    n_fluid = float((~m).sum())
    if n_fluid <= 0:
        raise ValueError("no fluid cells left")
    return n_fluid / (2.0 * nx * ny)