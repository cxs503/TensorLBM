"""B4-P3a — CAD slider streaming echo: design params -> C_D + UQ + verdict.

The first end-to-end interactive loop of the ship-design surrogate roadmap:

    design params (suboff_cad 4 hull-form axes + sail/fin scale + u_in/Re)
        -> solid mask (suboff_cad predicates)
        -> v3-protocol condition vector (EXACT fit-time construction)
        -> DragSurrogateService (ensemble C_D + UQ + guardrail verdict)

:class:`GeometryEchoPipeline` wraps any
:class:`tensorlbm.ai.inference_service.DragSurrogateService` and adds the
geometry front-end the service itself does not have: config-aware
(hull-form-variant) condition construction, per-slider-move latency
instrumentation, a batched "slider curve" sweep (one ensemble forward per
geometry batch, not per point) and an arbitrary-STL demo path via
:mod:`tensorlbm.voxelize`.

Fit-time exactness contract
---------------------------
The serving checkpoints (``b4_serve_20260824``, arm C_full) were fit on
``cache_v4.npz`` with ``condition_v3(re, uin, sail, fin, geo)`` where the
``geo`` block came from the CAD-predicate decomposition behind
:func:`tensorlbm.ai.drag_cond.suboff_geometry_features` (pinned bitwise
against the cache by ``tests/test_geometry_echo.py``).  The echo pipeline
rebuilds that same decomposition — hull/sail/fin point predicates, same
operations, same order, CPU — so a mother design reproduces the fit-time
channels **bit-identically**, and extends it with the ``SuboffConfig``
hull-form axes the drag_cond builder cannot express.  Hull-form variants
are outside the C_full training corpus by construction; the guardrail owns
that honesty (measured by :meth:`validate_against_cache`).

Geometry-feature seam (SDF-v2 drop-in)
--------------------------------------
All geometry knowledge enters through exactly two functions —
:func:`suboff_component_counts` (v3 hand channels) and
:func:`generalised_mask_counts` (any-mask descriptive channels) — plus the
``geo`` block of :class:`EchoGeometry`.  A future SDF encoder
(``tensorlbm.ai.geom_encoder``, PR #235 lineage) drops in at
``GeometryEchoPipeline._geometry``: replace the ``geo`` block (and the STL
``unsupported_channels`` downgrade) with encoder latents plus a
latent-space guard; nothing else in the pipeline touches geometry.

The honesty contract
--------------------
Out-of-family input must never silently produce confident numbers:

- hull-form variants are guarded in the same condition space the guard was
  fit on **and then explicitly downgraded**: the v3 hand channels barely
  move under the hull-form axes, so the channel-envelope guard alone can
  answer ``ok`` for a geometry the served corpus never contained (measured
  2026-08-25 on the b4_serve ensemble: an ``l_over_d_mult`` sweep scored
  1.14-1.23 — flag ``ok`` — while the served C_D trend ran OPPOSITE to the
  B4-fam family cache).  :func:`_downgrade_hullform` therefore forces at
  least ``review`` on every non-mother design, with the underlying guard
  verdict preserved in the reasons, and
  :attr:`EchoResult.confident <EchoResult.confident>` requires ``ok``;
- an arbitrary STL cannot express the v3 geometry channels at all — the
  result lists every channel that was NOT computed in
  ``unsupported_channels``, the condition block is an explicit
  mother-geometry proxy (labelled ``cond_proxy`` in ``info``) and the guard
  verdict is forced to ``reject`` with the underlying guard score preserved
  in the reasons.
"""

from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..suboff_cad import (
    SuboffConfig,
    SuboffHullType,
    build_suboff_mask,
    suboff_fins_contain_points,
    suboff_hull_mask,
    suboff_sail_contains_points,
)
from ..voxelize import is_watertight, load_stl, mask_from_stl, place_on_grid
from .drag_cond import (
    GEOMETRY_CHANNEL_NAMES,
    PRODUCTION_GRID,
    SuboffGrid,
    condition_v3,
    geometry_channels,
    suboff_geometry_features,
)
from .inference_service import (
    FLAG_OK,
    FLAG_REJECT,
    FLAG_REVIEW,
    BackendQueryError,
    DragSurrogateService,
    GuardVerdict,
    ModelEnsembleBackend,
    ensemble_stats,
)

__all__ = [
    "EchoGeometry",
    "EchoResult",
    "EchoValidationReport",
    "GeometryEchoPipeline",
    "HULLFORM_AXIS_NAMES",
    "PARAM_AXIS_NAMES",
    "SWEEP_AXIS_NAMES",
    "generalised_mask_counts",
    "suboff_component_counts",
]

#: Continuous design axes of the geometry (slider-drivable, CAD-rebuilt).
PARAM_AXIS_NAMES = (
    "sail_scale",
    "fin_scale",
    "l_over_d_mult",
    "nose_len_mult",
    "stern_len_mult",
    "sail_x_mult",
)

#: The 2026-08-24 hull-form (lines-plan) subset of :data:`PARAM_AXIS_NAMES`.
HULLFORM_AXIS_NAMES = ("l_over_d_mult", "nose_len_mult", "stern_len_mult", "sail_x_mult")

#: Axes :meth:`GeometryEchoPipeline.sweep_axis` accepts (geometry axes + the
#: condition-only inlet speed, which reuses the cached geometry).
SWEEP_AXIS_NAMES = PARAM_AXIS_NAMES + ("u_in",)

_DEFAULT_U_IN = 0.1
_DEFAULT_HULL = "full"
_NON_GEOMETRY_KEYS = ("hull_type", "u_in")


def _config_from_params(params: dict[str, Any]) -> SuboffConfig:
    """Build the CAD config from a params dict (unknown keys rejected)."""
    unknown = sorted(set(params) - set(PARAM_AXIS_NAMES) - set(_NON_GEOMETRY_KEYS))
    if unknown:
        raise ValueError(f"unknown design params: {unknown}; supported axes {PARAM_AXIS_NAMES}")
    return SuboffConfig(
        sail_scale=float(params.get("sail_scale", 1.0)),
        fin_scale=float(params.get("fin_scale", 1.0)),
        l_over_d_mult=float(params.get("l_over_d_mult", 1.0)),
        nose_len_mult=float(params.get("nose_len_mult", 1.0)),
        stern_len_mult=float(params.get("stern_len_mult", 1.0)),
        sail_x_mult=float(params.get("sail_x_mult", 1.0)),
    )


def _params_key(hull_type: str, params: dict[str, Any]) -> tuple[Any, ...]:
    """Hash key of the geometry-defining params (u_in/re excluded)."""
    return (str(hull_type),) + tuple(
        round(float(params.get(axis, 1.0)), 9) for axis in PARAM_AXIS_NAMES
    )


def suboff_component_counts(
    hull_type: str,
    sail_scale: float,
    fin_scale: float,
    grid: SuboffGrid,
    *,
    config: SuboffConfig | None = None,
    device: str = "cpu",
) -> tuple[int, int, int, int, int, int]:
    """Disjoint hull/sail/fin voxel decomposition (the v3 channel source).

    Evaluates the CAD point predicates over ``grid`` exactly the way the
    fit-time builder (``drag_cond._component_counts``) does — same
    operations, same order — so for mother designs (``config`` all-1.0 or
    ``None``) the counts are bit-identical to the training-cache values.
    ``config`` additionally enables the hull-form axes: the same predicates
    then see the deformed axial frame (see :mod:`tensorlbm.suboff_cad`).

    Returns ``(v_bare, v_sail, v_fin, v_solid, aproj, aproj_bare)``:
    ``v_*`` are the disjoint component voxel counts (``v_solid == v_bare +
    v_sail + v_fin`` exactly) and ``aproj`` / ``aproj_bare`` the
    x-projected lattice areas of the full solid and of the bare hull.
    """
    ht = SuboffHullType(hull_type)
    cfg = SuboffConfig() if config is None else config
    dev = torch.device(device)
    zz, yy, xx = torch.meshgrid(
        torch.arange(grid.nz, device=dev, dtype=torch.float32),
        torch.arange(grid.ny, device=dev, dtype=torch.float32),
        torch.arange(grid.nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    center = (grid.cx, grid.cy, grid.cz)
    hull = suboff_hull_mask(
        grid.nx, grid.ny, grid.nz, grid.cx, grid.cy, grid.cz, grid.length, 0.0, dev, cfg
    )
    v_bare = int(hull.sum().item())
    aproj_bare = int((hull.max(dim=2).values > 0).sum().item())

    solid = hull
    v_sail = 0
    v_fin = 0
    if ht in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL):
        sail = suboff_sail_contains_points(
            xx, yy, zz, center=center, length=grid.length, scale=sail_scale, config=cfg
        )
        add = sail & ~solid
        v_sail = int(add.sum().item())
        solid = solid | add
    if ht == SuboffHullType.FULL:
        fins = suboff_fins_contain_points(
            xx, yy, zz, center=center, length=grid.length, scale=fin_scale, config=cfg
        )
        add = fins & ~solid
        v_fin = int(add.sum().item())
        solid = solid | add
    v_solid = int(solid.sum().item())
    aproj = int((solid.max(dim=2).values > 0).sum().item())
    return v_bare, v_sail, v_fin, v_solid, aproj, aproj_bare


def _geo_from_counts(counts: tuple[int, int, int, int, int, int]) -> np.ndarray:
    """The (4,) v3 geometry-channel block from decomposition counts."""
    v_bare, v_sail, v_fin, v_solid, aproj, aproj_bare = counts
    if v_bare <= 0 or aproj_bare <= 0:
        raise ValueError(f"degenerate geometry (v_bare={v_bare}, aproj_bare={aproj_bare})")
    return np.array(
        [
            math.log10(aproj / aproj_bare),
            v_sail / v_bare,
            v_fin / v_bare,
            v_solid / v_bare,
        ],
        dtype=np.float64,
    )


def generalised_mask_counts(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Any-mask descriptive counts ``(A_proj, A_side, A_top, V)``.

    Same definitions as the B4-fam family corpus (``build_cache_fam.py``):
    x-projected lattice area (max over the streamwise axis), z-x side and
    y-x top projections, and the solid voxel count.  Computable from any
    boolean solid mask — no CAD predicate involved.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"mask must be 3-D (nz, ny, nx), got shape {m.shape}")
    return (
        int(m.max(axis=2).sum()),
        int(m.max(axis=1).sum()),
        int(m.max(axis=0).sum()),
        int(m.sum()),
    )


@lru_cache(maxsize=8)
def _mother_general_ref(grid: SuboffGrid) -> tuple[int, int, int, int]:
    """Generalised-channel reference counts (mother ``with_sail``, scale 1).

    Mirrors ``build_cache_fam.py``: the reference is the mother with_sail
    mask at ``grid`` built with :func:`build_suboff_mask` on CPU.
    """
    mask, _ = build_suboff_mask(
        hull_type="with_sail",
        nx=grid.nx,
        ny=grid.ny,
        nz=grid.nz,
        cx=grid.cx,
        cy=grid.cy,
        cz=grid.cz,
        length=grid.length,
        config=SuboffConfig(),
        device="cpu",
    )
    return generalised_mask_counts(mask.cpu().numpy())


def _generalised_channels(mask: np.ndarray, grid: SuboffGrid) -> np.ndarray:
    """log10 ratios of mask counts vs the mother with_sail reference."""
    counts = np.asarray(generalised_mask_counts(mask), dtype=np.float64)
    ref = np.asarray(_mother_general_ref(grid), dtype=np.float64)
    if not (ref > 0).all():
        raise ValueError("mother reference counts must be positive")
    return np.log10(counts / ref)


@dataclass(frozen=True)
class EchoGeometry:
    """One design point's geometry front-end (channels + condition).

    ``geo`` is the (4,) v3 geometry-channel block in fit-time construction
    (CPU decomposition); :meth:`condition_rows` appends the Re / u_in /
    scale logs to give the exact (N, 8) ``condition_v3`` matrix the serving
    checkpoints were fit with.
    """

    hull_type: str
    sail_scale: float
    fin_scale: float
    u_in: float
    config: SuboffConfig
    grid: SuboffGrid
    geo: np.ndarray
    counts: tuple[int, int, int, int, int, int]
    is_mother: bool

    def build_mask(self, *, device: str = "cpu") -> np.ndarray:
        """Voxelise the solid mask (bool ``(nz, ny, nx)``, numpy)."""
        mask, _ = build_suboff_mask(
            hull_type=self.hull_type,
            nx=self.grid.nx,
            ny=self.grid.ny,
            nz=self.grid.nz,
            cx=self.grid.cx,
            cy=self.grid.cy,
            cz=self.grid.cz,
            length=self.grid.length,
            config=self.config,
            device=device,
        )
        return mask.cpu().numpy().astype(bool)

    def condition_rows(self, re_list: Sequence[float] | np.ndarray) -> np.ndarray:
        """(N, 8) condition_v3 rows for this design over ``re_list``."""
        re = np.asarray(re_list, dtype=np.float64).ravel()
        return condition_v3(
            re,
            np.full(re.shape, self.u_in),
            np.full(re.shape, self.sail_scale),
            np.full(re.shape, self.fin_scale),
            np.broadcast_to(self.geo, (re.size, 4)),
        )

    def channels_dict(self) -> dict[str, float]:
        return {name: float(v) for name, v in zip(GEOMETRY_CHANNEL_NAMES, self.geo)}


@dataclass(frozen=True)
class EchoResult:
    """One served echo answer: C_D curve + ensemble UQ + guard verdict."""

    re: np.ndarray
    cd: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    std: np.ndarray
    guard: GuardVerdict
    params: dict[str, Any]
    backend: str
    members: tuple[str, ...]
    unsupported_channels: tuple[str, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def confident(self) -> bool:
        """True only when the guard said ``ok`` and nothing was unsupported."""
        return self.guard.flag == FLAG_OK and not self.unsupported_channels

    def uq_dict(self) -> dict[str, Any]:
        return {
            "lo": self.lo.tolist(),
            "hi": self.hi.tolist(),
            "mean_std": float(np.mean(self.std)) if self.std.size else 0.0,
            "std": self.std.tolist(),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": dict(self.params),
            "re": self.re.tolist(),
            "cd": self.cd.tolist(),
            "uq": self.uq_dict(),
            "guard": self.guard.as_dict(),
            "confident": self.confident,
            "backend": self.backend,
            "members": list(self.members),
            "unsupported_channels": list(self.unsupported_channels),
            "info": _jsonable(self.info),
        }


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of an info dict to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class GeometryEchoPipeline:
    """Interactive params/STL -> C_D front-end over a DragSurrogateService.

    Parameters
    ----------
    service:
        Any served :class:`DragSurrogateService` (model or replay backend).
        The service guard and (for model backends) the corpus field cache
        are reused verbatim — the echo pipeline never refits them.
    grid:
        Voxel grid for the geometry front-end.  Defaults to the service
        grid.  The serving checkpoints expect the production grid
        ``(nz, ny, nx) = (64, 64, 128)`` (resolution 128); any other grid
        changes the channel values and is intended for tests only.
    device:
        Torch device for mask building.  Channel counts always run on CPU —
        the fit-time device — so served conditions stay bit-exact.
    cache_size:
        LRU slots for rebuilt geometries (decomposition + channels).  A
        slider move that only changes Re or u_in reuses the cached entry;
        geometry changes rebuild.
    """

    def __init__(
        self,
        service: DragSurrogateService,
        *,
        grid: SuboffGrid | None = None,
        device: str = "cpu",
        cache_size: int = 16,
    ) -> None:
        if cache_size < 1:
            raise ValueError(f"cache_size must be >= 1, got {cache_size}")
        self.service = service
        self.grid = grid if grid is not None else (service.grid or PRODUCTION_GRID)
        self.device = str(device)
        self._cache: OrderedDict[tuple[Any, ...], EchoGeometry] = OrderedDict()
        self._cache_size = int(cache_size)

    # -- geometry front-end ------------------------------------------------------

    def _geometry(self, params: dict[str, Any]) -> EchoGeometry:
        """Geometry bundle for the params (LRU-cached, channels on CPU).

        ``u_in`` is *not* part of the geometry: cached bundles are reused
        across Re / u_in changes and rebound with :func:`dataclasses.replace`.
        """
        hull_type = str(params.get("hull_type", _DEFAULT_HULL))
        key = _params_key(hull_type, params)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        config = _config_from_params(params)
        sail = float(params.get("sail_scale", 1.0))
        fin = float(params.get("fin_scale", 1.0))
        counts = suboff_component_counts(hull_type, sail, fin, self.grid, config=config)
        is_mother = all(float(params.get(axis, 1.0)) == 1.0 for axis in HULLFORM_AXIS_NAMES)
        bundle = EchoGeometry(
            hull_type=hull_type,
            sail_scale=sail,
            fin_scale=fin,
            u_in=_DEFAULT_U_IN,
            config=config,
            grid=self.grid,
            geo=_geo_from_counts(counts),
            counts=counts,
            is_mother=is_mother,
        )
        self._cache[key] = bundle
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return bundle

    # -- reference field -----------------------------------------------------------

    def _reference_field(
        self, bundle: EchoGeometry, mask: np.ndarray | None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """(5, ny, nx) model input field + provenance info.

        Prefers the corpus field of the closest cached design — the model
        was trained on simulation fields, whose wake structure carries most
        of the signal.  An exact ``(hull, sail, fin, u_in)`` match wins
        (nearest log-Re, like the service); otherwise the nearest design in
        log-(sail, fin, u_in) of the same hull type serves slider values
        between corpus points (``field_source`` says which happened).
        Hull-form variants resolve to their mother design: the variant
        enters through its condition channels, the field is a documented
        approximation.  Without a corpus cache a free-stream + solid
        mid-plane field is synthesised from the mask.
        """
        svc = self.service
        if (
            svc.corpus_cache is not None
            and svc.cache_designs is not None
            and svc.cache_re is not None
        ):
            log_hint = math.log10(100.0)
            best_exact: tuple[float, int] | None = None
            best_near: tuple[float, float, int] | None = None
            for row, key in enumerate(svc.cache_designs):
                if key[0] != bundle.hull_type:
                    continue
                d_re = abs(math.log10(max(float(svc.cache_re[row]), 1e-12)) - log_hint)
                d_axes = (
                    abs(math.log10(max(key[1], 1e-12)) - math.log10(bundle.sail_scale))
                    + abs(math.log10(max(key[2], 1e-12)) - math.log10(bundle.fin_scale))
                    + abs(math.log10(max(key[3], 1e-12)) - math.log10(bundle.u_in))
                )
                if d_axes <= 1e-12 and (best_exact is None or d_re < best_exact[0]):
                    best_exact = (d_re, row)
                if best_near is None or (d_axes, d_re) < best_near[:2]:
                    best_near = (d_axes, d_re, row)
            chosen = best_exact[1] if best_exact is not None else None
            if chosen is None and best_near is not None:
                chosen = best_near[2]
            if chosen is not None:
                arr = np.asarray(svc.corpus_cache[int(chosen)], dtype=np.float32)
                if arr.shape != (5, self.grid.ny, self.grid.nx):
                    raise BackendQueryError(
                        f"corpus field shape {arr.shape} does not match the pipeline grid "
                        f"(5, {self.grid.ny}, {self.grid.nx})"
                    )
                src = (
                    "cache_design_exact_nearest_re"
                    if best_exact is not None
                    else "cache_design_nearest_axes"
                )
                return arr, {
                    "field_source": src,
                    "field_row": int(chosen),
                    "field_re": float(svc.cache_re[int(chosen)]),
                }
        if mask is None:
            raise BackendQueryError(
                "model backend needs a corpus field cache (none attached or no row of the "
                "right hull type) or an explicitly built mask to synthesise the reference field"
            )
        return _synthetic_field(mask), {"field_source": "synthetic_freestream_midplane"}

    # -- prediction ----------------------------------------------------------------

    def predict_from_params(
        self,
        params: dict[str, Any],
        re_list: Sequence[float],
    ) -> EchoResult:
        """One design point swept over ``re_list`` (one slider move)."""
        re = _check_re(re_list)
        u_in = float(params.get("u_in", _DEFAULT_U_IN))
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        base = self._geometry(params)
        bundle = replace(base, u_in=u_in)
        mask: np.ndarray | None = None
        if self.service.corpus_cache is None:
            tm = time.perf_counter()
            mask = bundle.build_mask(device=self.device)
            timings["mask_s"] = time.perf_counter() - tm
        timings["geometry_s"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        cond = bundle.condition_rows(re)
        timings["condition_s"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        verdict = _downgrade_hullform(self.service.guard.check(cond), bundle)
        timings["guard_s"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        member_cd, extra_info = self._run_backend([bundle], [cond], [mask])
        timings["ensemble_s"] = time.perf_counter() - t3
        timings["total_s"] = time.perf_counter() - t0

        mean, std, lo, hi = ensemble_stats(member_cd)
        info: dict[str, Any] = {
            "geometry": bundle.channels_dict(),
            "counts": dict(
                zip(("v_bare", "v_sail", "v_fin", "v_solid", "aproj", "aproj_bare"), bundle.counts)
            ),
            "grid": {"nz": bundle.grid.nz, "ny": bundle.grid.ny, "nx": bundle.grid.nx},
            "hull_form_variant": not bundle.is_mother,
            "cached_geometry": True,
            "timings_ms": {k: v * 1e3 for k, v in timings.items()},
            **extra_info,
        }
        return EchoResult(
            re=re,
            cd=mean,
            lo=lo,
            hi=hi,
            std=std,
            guard=verdict,
            params=_norm_params(params, u_in=u_in),
            backend=self.service.backend.kind,
            members=tuple(self.service.backend.member_labels()),
            info=info,
        )

    def sweep_axis(
        self,
        axis: str,
        values: Sequence[float],
        base_params: dict[str, Any],
        re_list: Sequence[float],
    ) -> list[EchoResult]:
        """Slider curve: sweep one axis over ``values`` from ``base_params``.

        Geometry axes rebuild the front-end per value (each value is a
        different solid) but the whole sweep — all geometries x all
        Reynolds points — goes through the ensemble as **one batched
        forward per member**; sweeping ``u_in`` reuses the cached geometry.
        """
        if axis not in SWEEP_AXIS_NAMES:
            raise ValueError(f"axis must be one of {SWEEP_AXIS_NAMES}, got {axis!r}")
        vals = [float(v) for v in values]
        if not vals:
            raise ValueError("values must be non-empty")
        re = _check_re(re_list)

        t0 = time.perf_counter()
        u_in = float(base_params.get("u_in", _DEFAULT_U_IN))
        geom_params = {k: v for k, v in dict(base_params).items() if k != "u_in"}
        if axis == "u_in":
            base = self._geometry(geom_params)
            bundles = [replace(base, u_in=float(v)) for v in vals]
        else:
            bundles = [self._geometry({**geom_params, axis: v}) for v in vals]
            bundles = [replace(b, u_in=u_in) for b in bundles]
        masks: list[np.ndarray | None] = [None] * len(bundles)
        if self.service.corpus_cache is None:
            masks = [b.build_mask(device=self.device) for b in bundles]
        t_geom = time.perf_counter() - t0

        t1 = time.perf_counter()
        conds = [b.condition_rows(re) for b in bundles]
        t_cond = time.perf_counter() - t1

        t2 = time.perf_counter()
        verdicts = [
            _downgrade_hullform(self.service.guard.check(c), b) for c, b in zip(conds, bundles)
        ]
        t_guard = time.perf_counter() - t2

        t3 = time.perf_counter()
        member_cd, extra_info = self._run_backend(bundles, conds, masks)
        t_ens = time.perf_counter() - t3
        total = time.perf_counter() - t0

        results: list[EchoResult] = []
        for i, bundle in enumerate(bundles):
            rows = member_cd[:, i * re.size : (i + 1) * re.size]
            mean, std, lo, hi = ensemble_stats(rows)
            info: dict[str, Any] = {
                "geometry": bundle.channels_dict(),
                "grid": {"nz": bundle.grid.nz, "ny": bundle.grid.ny, "nx": bundle.grid.nx},
                "hull_form_variant": not bundle.is_mother,
                "timings_ms": {
                    "geometry_s": 1e3 * t_geom / len(bundles),
                    "condition_s": 1e3 * t_cond / len(bundles),
                    "guard_s": 1e3 * t_guard / len(bundles),
                    "ensemble_s": 1e3 * t_ens / len(bundles),
                    "total_s": 1e3 * total / len(bundles),
                    "sweep_total_s": 1e3 * total,
                },
                **extra_info,
            }
            merged = dict(base_params)
            merged[axis] = vals[i]
            results.append(
                EchoResult(
                    re=re.copy(),
                    cd=mean,
                    lo=lo,
                    hi=hi,
                    std=std,
                    guard=verdicts[i],
                    params=_norm_params(merged, u_in=bundle.u_in)
                    | {"axis": axis, "value": vals[i]},
                    backend=self.service.backend.kind,
                    members=tuple(self.service.backend.member_labels()),
                    info=info,
                )
            )
        return results

    def predict_from_stl(
        self,
        path: str | Path,
        re_list: Sequence[float],
        *,
        u_in: float = _DEFAULT_U_IN,
        hull_type: str = _DEFAULT_HULL,
    ) -> EchoResult:
        """Arbitrary-STL demo path (voxelize -> mask -> honest downgrade).

        The v3 geometry channels require the CAD hull/sail/fin decomposition
        and are NOT derivable from an arbitrary mask, so the condition block
        uses an explicit mother-geometry proxy, the result names every
        unsupported channel in ``unsupported_channels`` and the guard
        verdict is forced to ``reject`` (underlying guard score preserved in
        the reasons).  Requires the model ensemble backend.
        """
        re = _check_re(re_list)
        if not isinstance(self.service.backend, ModelEnsembleBackend):
            raise BackendQueryError(
                "STL echo requires the model ensemble backend (replay serves archived "
                "CAD designs only)"
            )
        shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        t0 = time.perf_counter()
        mesh = load_stl(path)
        watertight = is_watertight(mesh)
        placement = place_on_grid(mesh.vertices, shape)
        mask = mask_from_stl(
            placement.tris, shape, origin=placement.origin, spacing=placement.spacing
        )
        t_vox = time.perf_counter() - t0

        gen = _generalised_channels(mask, self.grid)

        t1 = time.perf_counter()
        proxy_geo = geometry_channels(suboff_geometry_features(hull_type, 1.0, 1.0, grid=self.grid))
        cond = condition_v3(
            re,
            np.full(re.shape, u_in),
            np.ones(re.shape),
            np.ones(re.shape),
            np.broadcast_to(proxy_geo, (re.size, 4)),
        )
        t_cond = time.perf_counter() - t1

        t2 = time.perf_counter()
        underlying = self.service.guard.check(cond)
        verdict = GuardVerdict(
            flag=FLAG_REJECT,
            score=underlying.score,
            reasons=(
                "geometry channels not derivable from an arbitrary mask: "
                + ", ".join(GEOMETRY_CHANNEL_NAMES)
                + " (condition uses a mother-geometry proxy; the SDF encoder is the "
                "planned replacement for this seam)",
                f"stl_watertight={watertight}",
                *underlying.reasons,
            ),
        )
        t_guard = time.perf_counter() - t2

        t3 = time.perf_counter()
        member_cd = self.service.backend.predict_batch(
            _synthetic_field(mask)[None, ...], cond, np.array([re.size])
        )
        t_ens = time.perf_counter() - t3
        total = time.perf_counter() - t0

        mean, std, lo, hi = ensemble_stats(member_cd)
        counts = generalised_mask_counts(mask)
        info: dict[str, Any] = {
            "stl": {
                "path": str(path),
                "watertight": bool(watertight),
                "n_triangles": int(mesh.vertices.shape[0]),
                "placement_scale": float(placement.scale),
                "streamwise_extent_vox": float(placement.streamwise_extent),
            },
            "generalised_channels": dict(
                zip(
                    (
                        "log10_aproj_ratio_ref",
                        "log10_aside_ratio_ref",
                        "log10_atop_ratio_ref",
                        "log10_volume_ratio_ref",
                    ),
                    (float(v) for v in gen),
                )
            ),
            "mask_counts": dict(zip(("aproj", "aside", "atop", "v"), (int(c) for c in counts))),
            "cond_proxy": "mother_geometry",
            "geometry": {n: float(v) for n, v in zip(GEOMETRY_CHANNEL_NAMES, proxy_geo)},
            "grid": {"nz": self.grid.nz, "ny": self.grid.ny, "nx": self.grid.nx},
            "hull_form_variant": False,
            "field_source": "synthetic_freestream_midplane",
            "timings_ms": {
                "mask_s": 1e3 * t_vox,
                "geometry_s": 1e3 * t_vox,
                "condition_s": 1e3 * t_cond,
                "guard_s": 1e3 * t_guard,
                "ensemble_s": 1e3 * t_ens,
                "total_s": 1e3 * total,
            },
        }
        return EchoResult(
            re=re,
            cd=mean,
            lo=lo,
            hi=hi,
            std=std,
            guard=verdict,
            params={
                "source": "stl",
                "path": str(path),
                "u_in": float(u_in),
                "hull_type": hull_type,
            },
            backend=self.service.backend.kind,
            members=tuple(self.service.backend.member_labels()),
            unsupported_channels=tuple(GEOMETRY_CHANNEL_NAMES),
            info=info,
        )

    # -- backend dispatch ------------------------------------------------------------

    def _run_backend(
        self,
        bundles: Sequence[EchoGeometry],
        conds: Sequence[np.ndarray],
        masks: Sequence[np.ndarray | None],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Member matrix ``(M, sum N_g)`` + shared info for a geometry batch."""
        backend = self.service.backend
        if isinstance(backend, ModelEnsembleBackend):
            resolved = [self._reference_field(b, m) for b, m in zip(bundles, masks)]
            fields = np.stack([f for f, _ in resolved]).astype(np.float32)
            cond = np.concatenate(list(conds), axis=0)
            counts = np.array([c.shape[0] for c in conds], dtype=np.int64)
            member_cd = backend.predict_batch(fields, cond, counts)
            info: dict[str, Any] = {
                "field_source": resolved[0][1]["field_source"],
                "field_rows": [int(s.get("field_row", -1)) for _, s in resolved],
            }
            return member_cd, info
        # Replay backend: archived CAD designs only, exact design match.
        for bundle in bundles:
            if not bundle.is_mother:
                raise BackendQueryError(
                    "hull-form axes are not served by the replay backend (no archived "
                    "variant geometries); use the model ensemble backend"
                )
        member_rows = []
        for bundle, cond in zip(bundles, conds):
            re = 10.0 ** np.asarray(cond[:, 0], dtype=np.float64)
            rows, _ = backend.predict(
                bundle.hull_type,
                bundle.sail_scale,
                bundle.fin_scale,
                re,
                u_in=bundle.u_in,
            )
            member_rows.append(rows)
        if len({r.shape for r in member_rows}) != 1:
            raise BackendQueryError("replay rows have inconsistent lengths across designs")
        return np.concatenate(member_rows, axis=1), {"field_source": "replay_archive"}

    # -- validation --------------------------------------------------------------------

    def validate_against_cache(
        self,
        cache_path: str | Path,
        *,
        max_points: int = 28,
    ) -> EchoValidationReport:
        """Rebuild B4-fam family points and check parity + service MAPE.

        For up to ``max_points`` family points of the B4-fam cache (spread
        over all four hull-form families, CPU rebuild at the pipeline grid):

        - the CAD-rebuilt mask's generalised counts ``(A_proj, A_side,
          A_top, V)`` must equal the archived per-point counts exactly
          (integer equality) and the rebuilt generalised channel row must
          equal the cached ``geom`` row bitwise (max abs diff == 0);
        - the service C_D at the archived Re is compared against the cached
          C_D (MAPE).  These points sit outside the C_full training corpus
          by construction, so the MAPE quantifies the hull-form
          extrapolation gap the guardrail is responsible for flagging.
        """
        cache_path = Path(cache_path)
        meta_path = cache_path.parent / (cache_path.stem + "_meta.json")
        if not cache_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"cache/meta pair not found: {cache_path}, {meta_path}")
        z = np.load(cache_path)
        meta = json.loads(meta_path.read_text())
        fam_rows = np.nonzero(np.asarray(z["dsi"]) >= 6)[0]
        if fam_rows.size != len(meta):
            raise ValueError(
                f"family rows {fam_rows.size} != meta entries {len(meta)}; not a B4-fam cache?"
            )
        step = max(1, len(meta) // max_points)
        picked = list(range(0, len(meta), step))[:max_points]

        ref = _mother_general_ref(self.grid)
        count_mismatches = 0
        geom_bitwise_rows = 0
        max_abs_diff = 0.0
        errs: list[float] = []
        flags: dict[str, int] = {}
        per_point: list[dict[str, Any]] = []
        for i in picked:
            row = int(fam_rows[i])
            m = meta[i]
            hull = str(m["hull"])
            params = {
                "hull_type": hull,
                "u_in": float(m["u_in"]),
                "sail_scale": float(m["sail"]),
                "fin_scale": float(m["fin"]),
                "l_over_d_mult": float(m["l_over_d_mult"]),
                "nose_len_mult": float(m["nose_len_mult"]),
                "stern_len_mult": float(m["stern_len_mult"]),
                "sail_x_mult": float(m["sail_x_mult"]),
            }
            cfg = _config_from_params(params)
            mask, _ = build_suboff_mask(
                hull_type=hull,
                nx=self.grid.nx,
                ny=self.grid.ny,
                nz=self.grid.nz,
                cx=self.grid.cx,
                cy=self.grid.cy,
                cz=self.grid.cz,
                length=self.grid.length,
                config=cfg,
                device="cpu",
            )
            counts = generalised_mask_counts(mask.cpu().numpy())
            cached_counts = (int(m["aproj"]), int(m["aside"]), int(m["atop"]), int(m["v"]))
            counts_ok = counts == cached_counts
            count_mismatches += 0 if counts_ok else 1
            rebuilt = np.log10(
                np.asarray(counts, dtype=np.float64) / np.asarray(ref, dtype=np.float64)
            )
            cached_geom = np.asarray(z["geom"][row], dtype=np.float64)
            diff = float(np.max(np.abs(rebuilt - cached_geom)))
            max_abs_diff = max(max_abs_diff, diff)
            if counts_ok and diff == 0.0:
                geom_bitwise_rows += 1

            res = self.predict_from_params(params, [float(z["re"][row])])
            cd_pred = float(res.cd[0])
            cd_true = float(z["cd"][row])
            rel = abs(cd_pred - cd_true) / cd_true if cd_true > 0 else float("nan")
            if not math.isnan(rel):
                errs.append(rel)
            flags[res.guard.flag] = flags.get(res.guard.flag, 0) + 1
            per_point.append(
                {
                    "row": row,
                    "family": str(m.get("fam", "")),
                    "l_over_d_mult": float(m["l_over_d_mult"]),
                    "re": float(z["re"][row]),
                    "counts_equal": bool(counts_ok),
                    "geom_max_abs_diff": diff,
                    "cd_pred": cd_pred,
                    "cd_cached": cd_true,
                    "abs_rel_err": rel,
                    "guard_flag": res.guard.flag,
                }
            )
        return EchoValidationReport(
            cache=str(cache_path),
            n_points=len(per_point),
            n_family_total=len(meta),
            counts_bitwise_rows=geom_bitwise_rows,
            counts_mismatch_rows=count_mismatches,
            max_abs_geom_diff=max_abs_diff,
            service_mape=float(np.mean(errs)) if errs else float("nan"),
            guard_flags=flags,
            per_point=per_point,
        )


def _downgrade_hullform(verdict: GuardVerdict, bundle: EchoGeometry) -> GuardVerdict:
    """Force at least ``review`` for hull-form variants (honesty contract).

    The served C_full corpus contains no hull-form variants, and the v3 hand
    channels barely move under the deformation axes, so the channel-space
    guard alone can answer ``ok`` for a geometry the model never saw
    (measured 2026-08-25 on the b4_serve ensemble: ``l_over_d_mult`` sweep
    scored 1.14-1.23, flag ``ok``, while the served trend direction
    disagreed with the B4-fam family cache).  Any non-mother design is
    therefore downgraded to at least ``review`` with the underlying verdict
    preserved in the reasons; ``reject`` is never weakened.
    """
    if bundle.is_mother or verdict.flag in (FLAG_REJECT, FLAG_REVIEW):
        return verdict
    changed = tuple(
        axis for axis in HULLFORM_AXIS_NAMES if float(getattr(bundle.config, axis, 1.0)) != 1.0
    )
    if not changed:
        return verdict
    reason = (
        "hull-form axes ("
        + ", ".join(changed)
        + ") are outside the served training corpus; channel-space guard said "
        + f"flag=ok score={verdict.score:.2f} (downgraded to review)"
    )
    return GuardVerdict(
        flag=FLAG_REVIEW,
        score=verdict.score,
        reasons=(reason,) + tuple(verdict.reasons),
    )


def _synthetic_field(mask: np.ndarray) -> np.ndarray:
    """Free-stream + solid mid-plane reference field (5, ny, nx) float32."""
    mid = mask[mask.shape[0] // 2].astype(np.float32)
    return np.stack(
        [
            np.ones_like(mid),
            np.zeros_like(mid),
            np.zeros_like(mid),
            np.ones_like(mid),
            mid,
        ]
    ).astype(np.float32)


def _check_re(re_list: Sequence[float]) -> np.ndarray:
    re = np.asarray(re_list, dtype=np.float64).ravel()
    if re.size == 0:
        raise ValueError("re_list must be non-empty")
    if not np.isfinite(re).all() or not (re > 0).all():
        raise ValueError("re_list entries must be finite and positive")
    return re


def _norm_params(params: dict[str, Any], *, u_in: float) -> dict[str, Any]:
    """JSON-safe normalised params echo (hull type + axes + u_in)."""
    out: dict[str, Any] = {
        "hull_type": str(params.get("hull_type", _DEFAULT_HULL)),
        "u_in": float(u_in),
    }
    for axis in PARAM_AXIS_NAMES:
        if axis in params:
            out[axis] = float(params[axis])
    return out


@dataclass(frozen=True)
class EchoValidationReport:
    """Result of :meth:`GeometryEchoPipeline.validate_against_cache`."""

    cache: str
    n_points: int
    n_family_total: int
    counts_bitwise_rows: int
    counts_mismatch_rows: int
    max_abs_geom_diff: float
    service_mape: float
    guard_flags: dict[str, int]
    per_point: list[dict[str, Any]]

    @property
    def channels_bitwise(self) -> bool:
        """True iff every rebuilt channel row equalled the cache bitwise."""
        return self.counts_mismatch_rows == 0 and self.max_abs_geom_diff == 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache": self.cache,
            "n_points": self.n_points,
            "n_family_total": self.n_family_total,
            "channels_bitwise": self.channels_bitwise,
            "counts_bitwise_rows": self.counts_bitwise_rows,
            "counts_mismatch_rows": self.counts_mismatch_rows,
            "max_abs_geom_diff": self.max_abs_geom_diff,
            "service_mape": self.service_mape,
            "guard_flags": self.guard_flags,
            "per_point": self.per_point,
        }
