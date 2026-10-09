"""B4-P3c — gradient-guided acquisition + trend-margin calibration (2026-08-27).

Second slice of the active-learning loop on top of
:mod:`tensorlbm.ai.active_learning` (#243) and
:mod:`tensorlbm.diff_voxelize` (#248):

1. **Batched end-to-end gradients.**
   :func:`tensorlbm.diff_voxelize.drag_gradients` is point-wise: it
   reloads the ensemble and rebuilds the whole SDF/autograd graph per
   design.  :func:`batch_drag_gradients` loads the ensemble once and
   evaluates ``B`` designs per backward pass — the candidates of an
   acquisition step share the frozen reference field, so the per-sample
   graphs are block-diagonal in the parameter leaves and ONE
   ``sum().backward()`` yields every gradient.
2. **The ``gradient`` acquisition strategy** (steepness-seeking).
   ``coverage`` picks *spatially empty* candidates; the gradient strategy
   picks candidates where the surrogate response is *steepest*
   (``|d(log10 C_D)/d theta|`` large — where the surrogate is most likely
   to be probed again and wrong).  :func:`propose_acquisition_grad`
   delegates the three #243 strategies to
   :func:`tensorlbm.ai.active_learning.propose_acquisition` **verbatim**
   (bitwise; no registration point needed in ``active_learning.py``) and
   adds ``'gradient'`` as a fourth strategy whose candidate pool is
   EXACTLY the coverage pool (same accept rules, same labelable
   corner-x-Re construction), ranked by a composite
   gradient-plus-optional-coverage score instead of round-robin.
3. **Trend-margin calibration** (#243 leftover: the retrained ensemble
   under-shoots the cache-truth C_D trend amplitude ~2.6x).
   :func:`trend_slopes` / :func:`margin_ratio` measure
   ``slope_pred / slope_truth`` per axis; :func:`calibrate_trend_margin`
   fits a per-axis scale from a fit split of measured ratios and reports
   honestly whether the ratio is stable enough to calibrate at all
   (dispersion and sign-flip gates) — it never manufactures a scale for
   an unstable axis.

Honest handling of the #247/#248 gradient pathologies is built into the
strategy (see :func:`axis_stability` / :func:`gradient_scores`):

- axes whose member gradients disagree in SIGN across the ensemble
  (``fin_scale`` is a measured near-cancellation) are excluded from the
  score and reported, not silently averaged;
- axes are normalised by the pool-median magnitude before aggregation so
  the out-of-support hull-form axes (``l_over_d_mult`` STE ~1e0) do not
  drown the appendage axes (~1e-2) purely by uncalibrated scale.

Everything /nfs-related (checkpoints, reference fields, caches) enters
as an explicit argument; the module is import-safe and testable on
CPU-only machines with synthetic checkpoints.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..diff_voxelize import (
    DEFAULT_SMOOTH_K_FT,
    DEFAULT_TAU_FT,
    DIFF_PARAM_NAMES,
    DiffDragEnsemble,
    DiffParams,
    soft_mask,
    straight_through_mask,
    suboff_component_sdfs,
)
from ..suboff_cad import SuboffConfig, SuboffHullType
from .active_learning import (
    STRATEGY_NAMES,
    AcquisitionPoint,
    FlaggedQuery,
    hullform_condition_rows,
    propose_acquisition,
)
from .drag_cond import PRODUCTION_GRID, SuboffGrid
from .inference_service import EnvelopeMahalanobisGuardrail

__all__ = [
    "GRADIENT_STRATEGY_NAMES",
    "GradBatch",
    "GradientScoreTable",
    "MarginRatioStat",
    "TrendMarginCalibration",
    "apply_trend_calibration",
    "axis_stability",
    "batch_drag_gradients",
    "calibrate_trend_margin",
    "gradient_scores",
    "margin_ratio",
    "propose_acquisition_grad",
    "trend_slopes",
]

#: All four strategies :func:`propose_acquisition_grad` dispatches on (the
#: three #243 names plus the new one).
GRADIENT_STRATEGY_NAMES = (*STRATEGY_NAMES, "gradient")

#: Default chunk size of the batched gradient engine.  The SDF autograd
#: graph keeps O(grid-size) float tensors per design alive; 16 fits in a
#: few GB at the production grid (see the run report for the measured
#: throughput/chunk table).
DEFAULT_CHUNK = 16

#: Pool cap used to materialise the FULL coverage candidate pool (the
#: gradient strategy ranks the same pool the round-robin coverage
#: strategy draws from, so the comparison is apples-to-apples).
_POOL_CAP = 4096

_MOTHER_DEFAULTS: dict[str, float] = {
    name: float(getattr(SuboffConfig(), name)) for name in DIFF_PARAM_NAMES
}


# ---------------------------------------------------------------------------
# Batched differentiable forward (candidates -> d(log10 C_D)/dtheta batch)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GradBatch:
    """Batched :func:`~tensorlbm.diff_voxelize.drag_gradients` result.

    ``grads[i, a]`` is the ensemble ``d(log10 C_D)/d theta_a`` of design
    ``i``; ``member_grads[m, i, a]`` the per-checkpoint-member component;
    ``log10_cd[i]`` / ``channels[i]`` are forward diagnostics.
    """

    point_keys: tuple[str, ...]
    axis_names: tuple[str, ...]
    grads: np.ndarray  # (N, A)
    member_grads: np.ndarray  # (M, N, A)
    member_labels: tuple[str, ...]
    log10_cd: np.ndarray  # (N,)
    channels: np.ndarray  # (N, 4)

    @property
    def n_points(self) -> int:
        return len(self.point_keys)


def _per_design(value: float | Sequence[float], b: int, name: str) -> np.ndarray:
    """Broadcast a scalar-or-per-design argument to ``(b,)``."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(b, float(arr[0]))
    if arr.size != b:
        raise ValueError(f"{name} must be scalar or length-{b}, got shape {arr.shape}")
    return arr.astype(np.float64, copy=True)


def _batch_leaves(
    designs: Sequence[Mapping[str, float]], device: torch.device, dtype: torch.dtype
) -> DiffParams:
    """Leaf bundle with one ``(B, 1, 1, 1)`` tensor per axis.

    The grid-broadcast shape lets the unmodified
    :func:`~tensorlbm.diff_voxelize.suboff_component_sdfs` evaluate the
    whole batch at once (every SDF operation broadcasts against the
    ``(nz, ny, nx)`` coordinate grid).
    """
    tensors: dict[str, torch.Tensor] = {}
    for name in DIFF_PARAM_NAMES:
        default = _MOTHER_DEFAULTS[name]
        vals = [float(d.get(name, default)) for d in designs]
        # built directly in grid-broadcast shape so the tensor IS a leaf
        # (a .view of a flat leaf is a non-leaf whose .grad stays empty)
        tensors[name] = torch.tensor(
            [[[[v]]] for v in vals], device=device, dtype=dtype, requires_grad=True
        )
    return DiffParams(**tensors)


def _occupancy(sdf: torch.Tensor, tau: float, ste: bool) -> torch.Tensor:
    """STE occupancy (default) or pure soft occupancy, as in #248."""
    return straight_through_mask(sdf, tau) if ste else soft_mask(sdf, tau)


def _batch_mask_channels(
    grid: SuboffGrid,
    params: DiffParams,
    hull_type: str,
    smooth_k: float,
    tau: float,
    ste: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched mirror of :func:`tensorlbm.diff_voxelize.mask_channels`.

    Returns ``(channel_vector (B, 4), midplane (B, ny, nx))`` with sums
    over the grid dims ``(1, 2, 3)`` so each batch row is one design.
    Forward semantics are identical to the point-wise version (STE masks
    are hard in the forward pass).
    """
    sdfs = suboff_component_sdfs(grid, params, smooth_k=smooth_k, device=device, dtype=dtype)
    variant = SuboffHullType(hull_type)
    m_hull = _occupancy(sdfs["hull"], tau, ste)
    if variant in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL):
        m_sail = _occupancy(sdfs["sail"], tau, ste)
    else:
        m_sail = torch.zeros_like(m_hull)
    if variant == SuboffHullType.FULL:
        m_fin = _occupancy(sdfs["fin"], tau, ste)
    else:
        m_fin = torch.zeros_like(m_hull)

    not_hull = 1.0 - m_hull
    v_bare = m_hull.sum(dim=(1, 2, 3))
    v_sail = (m_sail * not_hull).sum(dim=(1, 2, 3))
    v_fin = (m_fin * not_hull * (1.0 - m_sail)).sum(dim=(1, 2, 3))
    v_solid = v_bare + v_sail + v_fin

    def _projected(m: torch.Tensor) -> torch.Tensor:
        # prod over nx (dim 3) -> (B, nz, ny); a column is solid if any
        # x-cell is (per-point version: prod over dim 2 then total sum).
        return (1.0 - torch.prod(1.0 - m, dim=3)).sum(dim=(1, 2))

    aproj_bare = _projected(m_hull)
    cover = 1.0 - (1.0 - m_hull) * (1.0 - m_sail) * (1.0 - m_fin)
    aproj = _projected(cover)

    channels = torch.stack(
        (
            torch.log10(aproj / aproj_bare),
            v_sail / v_bare,
            v_fin / v_bare,
            v_solid / v_bare,
        ),
        dim=1,
    )
    return channels, cover[:, grid.nz // 2]


def _batch_condition_rows(
    params: DiffParams,
    re: np.ndarray,
    u_in: np.ndarray,
    channel_vector: torch.Tensor,
) -> torch.Tensor:
    """Batched ``condition_v3`` rows ``(B, 8)`` — logs use param tensors."""
    b = channel_vector.shape[0]
    dev, dt = channel_vector.device, channel_vector.dtype
    logs = torch.stack(
        (
            torch.log10(torch.as_tensor(re, dtype=dt, device=dev)),
            torch.log10(torch.as_tensor(u_in, dtype=dt, device=dev)),
            torch.log10(params.sail_scale.reshape(b)),
            torch.log10(params.fin_scale.reshape(b)),
        ),
        dim=1,
    )
    return torch.cat((logs, channel_vector), dim=1)


def _batch_member_log10_cd(
    ensemble: DiffDragEnsemble, fields: torch.Tensor, conds: torch.Tensor
) -> list[torch.Tensor]:
    """Per-member log10 C_D ``(B,)`` (batched member forward)."""
    outs: list[torch.Tensor] = []
    for model, norm in zip(ensemble.models, ensemble.norms):
        ch_m = torch.as_tensor(norm["ch_mean"], dtype=torch.float32, device=ensemble.device)
        ch_s = torch.as_tensor(norm["ch_std"], dtype=torch.float32, device=ensemble.device)
        p_m = torch.as_tensor(norm["p_mean"], dtype=torch.float32, device=ensemble.device)
        p_s = torch.as_tensor(norm["p_std"], dtype=torch.float32, device=ensemble.device)
        x_norm = (fields.float() - ch_m.view(1, -1, 1, 1)) / ch_s.view(1, -1, 1, 1)
        p_norm = (conds.float() - p_m) / p_s
        z = model(x_norm, p_norm)
        outs.append(z.double().reshape(-1) * float(norm["y_std"]) + float(norm["y_mean"]))
    return outs


def _run_chunk(
    designs: Sequence[Mapping[str, float]],
    re: np.ndarray,
    u_in: np.ndarray,
    ensemble: DiffDragEnsemble,
    field_row: np.ndarray,
    grid: SuboffGrid,
    hull_type: str,
    smooth_k: float,
    tau: float,
    ste: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """One batched forward+backward chunk (the whole graph lives here)."""
    b = len(designs)
    params = _batch_leaves(designs, device, dtype)
    channels, midplane = _batch_mask_channels(
        grid, params, hull_type, smooth_k, tau, ste, device, dtype
    )
    conds = _batch_condition_rows(params, re, u_in, channels)
    frozen = torch.from_numpy(np.asarray(field_row, dtype=np.float32)).to(device)
    if frozen.ndim != 3 or frozen.shape[0] != 5:
        raise ValueError(f"field_row must be (5, ny, nx), got {tuple(frozen.shape)}")
    if tuple(frozen.shape[1:]) != tuple(midplane.shape[1:]):
        raise ValueError(
            f"field plane {tuple(frozen.shape[1:])} does not match mask plane "
            f"{tuple(midplane.shape[1:])}"
        )
    fields = frozen.unsqueeze(0).repeat(b, 1, 1, 1).to(dtype)
    fields[:, 4] = midplane
    member = _batch_member_log10_cd(ensemble, fields, conds)
    member_cd = torch.stack(tuple(10.0**m for m in member))  # (M, B)
    log10_cd = torch.log10(member_cd.mean(dim=0))  # (B,)

    axes = list(DIFF_PARAM_NAMES)
    leaves = [getattr(params, name) for name in axes]
    member_rows: list[torch.Tensor] = []
    for m_log in member:
        gm = torch.autograd.grad(m_log.sum(), leaves, retain_graph=True, allow_unused=True)
        member_rows.append(
            torch.stack(
                tuple(
                    g.reshape(b) if g is not None else torch.zeros(b, device=device, dtype=dtype)
                    for g in gm
                ),
                dim=1,
            )
        )  # (B, A) per member
    ge = torch.autograd.grad(log10_cd.sum(), leaves, allow_unused=True)
    ens_grad = torch.stack(
        tuple(
            g.reshape(b) if g is not None else torch.zeros(b, device=device, dtype=dtype)
            for g in ge
        ),
        dim=1,
    )  # (B, A)
    return {
        "grads": ens_grad.detach().cpu().numpy().astype(np.float64),
        "member_grads": torch.stack(member_rows).detach().cpu().numpy().astype(np.float64),
        "log10_cd": log10_cd.detach().cpu().numpy().astype(np.float64),
        "channels": channels.detach().cpu().numpy().astype(np.float64),
    }


def batch_drag_gradients(
    designs: Sequence[Mapping[str, float]],
    ckpt_paths: Sequence[str],
    field_row: np.ndarray,
    *,
    re: float | Sequence[float] = 200.0,
    u_in: float | Sequence[float] = 0.1,
    grid: SuboffGrid | None = None,
    hull_type: str = "full",
    smooth_k: float = DEFAULT_SMOOTH_K_FT,
    tau: float = DEFAULT_TAU_FT,
    ste: bool = True,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    chunk: int = DEFAULT_CHUNK,
) -> GradBatch:
    """Batched ``d(log10 C_D)/d theta`` for ``B`` designs in one pass.

    Same semantics as :func:`tensorlbm.diff_voxelize.drag_gradients`
    (STE occupancy chain, frozen reference field with the CAD mask in
    channel 4, log10 of the linear-space ensemble mean C_D) but the
    ensemble is loaded once and ``chunk`` designs share each autograd
    graph — the per-sample graphs are block-diagonal in the parameter
    leaves, so a single ``backward`` of the batch sum yields every
    per-design gradient.  ``re`` / ``u_in`` may be per-design sequences.

    Row ``i`` of the output equals ``drag_gradients(designs[i], ...)``
    up to float round-off: measured <= 4e-7 relative on CPU (reduction
    order only; the geometry channels are bitwise equal) and <= 3e-3 on
    CUDA (float32 kernel layout differs between batch shapes).  Exact
    bitwise reruns require the same ``chunk`` / device / thread setup.
    """
    if not designs:
        raise ValueError("designs must be non-empty")
    b = len(designs)
    re_arr = _per_design(re, b, "re")
    u_arr = _per_design(u_in, b, "u_in")
    if not (re_arr > 0).all() or not (u_arr > 0).all():
        raise ValueError("re and u_in entries must be positive")
    ensemble = DiffDragEnsemble.from_checkpoints(ckpt_paths, device=device)
    g = PRODUCTION_GRID if grid is None else grid
    dev = torch.device(device)
    step = max(int(chunk), 1)
    grads: list[np.ndarray] = []
    member_grads: list[np.ndarray] = []
    log10_cd: list[np.ndarray] = []
    channels: list[np.ndarray] = []
    for lo in range(0, b, step):
        hi = min(lo + step, b)
        out = _run_chunk(
            list(designs[lo:hi]),
            re_arr[lo:hi],
            u_arr[lo:hi],
            ensemble,
            field_row,
            g,
            hull_type,
            smooth_k,
            tau,
            ste,
            dev,
            dtype,
        )
        grads.append(out["grads"])
        member_grads.append(out["member_grads"])
        log10_cd.append(out["log10_cd"])
        channels.append(out["channels"])
    keys = [
        "|".join(f"{float(d.get(a, _MOTHER_DEFAULTS[a])):.9g}" for a in DIFF_PARAM_NAMES)
        for d in designs
    ]
    return GradBatch(
        point_keys=tuple(keys),
        axis_names=tuple(DIFF_PARAM_NAMES),
        grads=np.concatenate(grads, axis=0),
        member_grads=np.concatenate(member_grads, axis=1),
        member_labels=tuple(ensemble.member_labels),
        log10_cd=np.concatenate(log10_cd),
        channels=np.concatenate(channels),
    )


# ---------------------------------------------------------------------------
# Axis stability + composite gradient score
# ---------------------------------------------------------------------------


def axis_stability(batch: GradBatch, *, grad_floor: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis ensemble sign-agreement fraction and median magnitude.

    ``stability[a]`` is the fraction of ``(member, point)`` gradient
    entries whose sign agrees with the ensemble gradient of the same
    point (points with ``|ensemble grad| <= grad_floor`` are skipped —
    their sign is round-off).  ``magnitude[a]`` is the pool-median
    ``|ensemble grad|``.  Both feed the honest-axis gate of
    :func:`gradient_scores`: near-cancellation axes (measured:
    ``fin_scale``) fail the sign vote, dead axes fail the floor.
    """
    ens = batch.grads  # (N, A)
    mem = batch.member_grads  # (M, N, A)
    _n, n_axes = ens.shape
    stability = np.zeros(n_axes, dtype=np.float64)
    magnitude = np.zeros(n_axes, dtype=np.float64)
    for a in range(n_axes):
        magnitude[a] = float(np.median(np.abs(ens[:, a])))
        active = np.abs(ens[:, a]) > grad_floor
        if not active.any():
            stability[a] = 0.0
            continue
        ens_sign = np.sign(ens[active, a])  # (n_active,)
        mem_sign = np.sign(mem[:, active, a])  # (M, n_active)
        stability[a] = float((mem_sign == ens_sign[None, :]).mean())
    return stability, magnitude


@dataclass(frozen=True)
class GradientScoreTable:
    """Full score table of a candidate pool under the gradient strategy."""

    point_keys: tuple[str, ...]
    axes_used: tuple[str, ...]
    axes_excluded: tuple[str, ...]
    exclusion_reasons: dict[str, str]
    scores_gradient: np.ndarray  # (N,) z-scored log-magnitude composite
    scores_coverage: np.ndarray  # (N,) z-scored Mahalanobis channel score
    scores_final: np.ndarray  # (N,) (1 - w) * grad + w * coverage
    weight_coverage: float
    fallback_coverage_order: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_keys": list(self.point_keys),
            "axes_used": list(self.axes_used),
            "axes_excluded": list(self.axes_excluded),
            "exclusion_reasons": dict(self.exclusion_reasons),
            "scores_gradient": self.scores_gradient.tolist(),
            "scores_coverage": self.scores_coverage.tolist(),
            "scores_final": self.scores_final.tolist(),
            "weight_coverage": self.weight_coverage,
            "fallback_coverage_order": self.fallback_coverage_order,
        }


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if sd <= 1e-12:
        return np.zeros_like(x)
    return (x - float(x.mean())) / sd


def gradient_scores(
    points: Sequence[AcquisitionPoint],
    batch: GradBatch,
    existing_cond: np.ndarray,
    *,
    w_coverage: float = 0.0,
    min_stability: float = 0.75,
    axes: Sequence[str] | None = None,
    grid: SuboffGrid = PRODUCTION_GRID,
    grad_floor: float = 1e-12,
) -> GradientScoreTable:
    """Composite acquisition score of a candidate pool.

    - per-axis honest gate: axes failing the member sign vote
      (``stability < min_stability``) or with pool-median magnitude at
      the floor are EXCLUDED from the score and reported (the #248
      near-cancellation / out-of-support pathologies — never silently
      averaged in);
    - per-axis normalisation: ``|grad_axis| / median_pool(|grad_axis|)``
      so uncalibrated out-of-support axes cannot dominate by raw scale;
    - ``scores_gradient = log10(L2 norm of the normalised stable-axis
      vector)`` — the log-|grad| norm of the task brief;
    - ``scores_coverage`` = z-scored Mahalanobis channel score of each
      candidate against the existing corpus cloud;
    - ``scores_final = (1 - w_coverage) * z(grad) + w_coverage * z(mahal)``:
      ``w_coverage=0`` is the pure gradient strategy, ``0 < w < 1`` the
      mixed strategy (weight exposed as a parameter, not hidden).

    If EVERY axis fails the gate the table degrades honestly: the flag
    ``fallback_coverage_order`` is set and the final score reduces to the
    coverage term alone (:func:`propose_acquisition_grad` then returns
    the pool in the coverage order itself) — a reported degradation, not
    a crash and not a silently meaningless ranking.
    """
    if len(points) != batch.n_points:
        raise ValueError(f"{len(points)} points but batch has {batch.n_points}")
    if not 0.0 <= w_coverage <= 1.0:
        raise ValueError(f"w_coverage must be in [0, 1], got {w_coverage}")
    stability, magnitude = axis_stability(batch, grad_floor=grad_floor)
    names = list(batch.axis_names)
    requested = list(axes) if axes is not None else names
    unknown = sorted(set(requested) - set(names))
    if unknown:
        raise ValueError(f"unknown axes {unknown}; batch has {names}")
    excluded: dict[str, str] = {}
    used: list[int] = []
    for i, name in enumerate(names):
        if name not in requested:
            excluded[name] = "not requested"
            continue
        if magnitude[i] <= grad_floor:
            excluded[name] = f"dead axis (pool-median |grad| {magnitude[i]:.2e} <= floor)"
            continue
        if stability[i] < min_stability:
            excluded[name] = (
                f"sign-unstable across members (stability {stability[i]:.2f} < {min_stability})"
            )
            continue
        used.append(i)

    guard = EnvelopeMahalanobisGuardrail(np.asarray(existing_cond, dtype=np.float64))
    u_in = float(points[0].params.get("u_in", 0.1)) if points else 0.1
    mahal = np.asarray(
        [
            float(
                guard.row_scores(hullform_condition_rows(p.params, [p.re], grid=grid, u_in=u_in))[0]
            )
            for p in points
        ],
        dtype=np.float64,
    )
    z_cov = _zscore(mahal)

    if used:
        med = np.median(np.abs(batch.grads[:, used]), axis=0)
        med = np.where(med <= grad_floor, 1.0, med)  # redundant with the gate, kept safe
        norm = np.abs(batch.grads[:, used]) / med
        grad_score = np.log10(np.sqrt((norm**2).sum(axis=1) + 1e-300))
        z_grad = _zscore(grad_score)
        fallback = False
        final = (1.0 - w_coverage) * z_grad + w_coverage * z_cov
    else:
        z_grad = np.zeros(len(points), dtype=np.float64)
        fallback = True
        final = z_cov  # no gradient signal: reduce to the coverage term
    return GradientScoreTable(
        point_keys=tuple(p.key for p in points),
        axes_used=tuple(names[i] for i in used),
        axes_excluded=tuple(sorted(excluded)),
        exclusion_reasons=excluded,
        scores_gradient=z_grad,
        scores_coverage=z_cov,
        scores_final=final,
        weight_coverage=float(w_coverage),
        fallback_coverage_order=fallback,
    )


# ---------------------------------------------------------------------------
# The gradient acquisition strategy (delegating dispatcher)
# ---------------------------------------------------------------------------


def propose_acquisition_grad(
    queries: Sequence[FlaggedQuery],
    *,
    strategy: str,
    budget: int,
    existing_cond: np.ndarray,
    grad_fn: Callable[[list[AcquisitionPoint]], GradBatch] | None = None,
    axis_ranges: dict[str, tuple[float, float]] | None = None,
    member_std_fn: Callable[[list[AcquisitionPoint]], np.ndarray] | None = None,
    grid: SuboffGrid = PRODUCTION_GRID,
    seed: int = 0,
    n_candidates: int | None = None,
    n_re_levels: int = 8,
    w_coverage: float = 0.0,
    min_stability: float = 0.75,
    axes: Sequence[str] | None = None,
) -> list[AcquisitionPoint]:
    """Propose ``budget`` points; the #243 strategies verbatim + ``gradient``.

    ``strategy in {'envelope_shell', 'max_disagreement', 'coverage'}``
    delegates to
    :func:`tensorlbm.ai.active_learning.propose_acquisition` with the
    identical arguments (bitwise-identical output, pinned by tests at
    the key level), so the three existing strategies are untouched.

    ``strategy='gradient'`` materialises the FULL coverage candidate pool
    (identical corner-x-Re construction and accept rules — the
    apples-to-apples guarantee) and returns the top ``budget`` candidates
    by :func:`gradient_scores` with weight ``w_coverage`` (0 = pure
    steepness, 0.5 = the mixed strategy of the comparison experiment).
    Ranking is descending score with the canonical key as tie-break.
    Fewer points are returned only when the pool itself is smaller
    (never silently padded).  When no axis survives the honest-axis gate
    the proposal degrades to the plain coverage order (the first
    ``budget`` pool points) — reported, not hidden.
    """
    if strategy not in GRADIENT_STRATEGY_NAMES:
        raise ValueError(f"strategy must be one of {GRADIENT_STRATEGY_NAMES}, got {strategy!r}")
    if strategy in STRATEGY_NAMES:
        return propose_acquisition(
            queries,
            strategy=strategy,
            budget=budget,
            existing_cond=existing_cond,
            axis_ranges=axis_ranges,
            member_std_fn=member_std_fn,
            grid=grid,
            seed=seed,
            n_candidates=n_candidates,
            n_re_levels=n_re_levels,
        )
    if grad_fn is None:
        raise ValueError("strategy='gradient' requires grad_fn (batched gradients)")
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    pool = [
        AcquisitionPoint(params=dict(p.params), re=p.re, strategy="gradient")
        for p in propose_acquisition(
            queries,
            strategy="coverage",
            budget=_POOL_CAP,
            existing_cond=existing_cond,
            axis_ranges=axis_ranges,
            member_std_fn=None,
            grid=grid,
            seed=seed,
            n_candidates=n_candidates,
            n_re_levels=n_re_levels,
        )
    ]
    if not pool or budget >= len(pool):
        return pool
    table = gradient_scores(
        pool,
        grad_fn(pool),
        existing_cond,
        w_coverage=w_coverage,
        min_stability=min_stability,
        axes=axes,
        grid=grid,
    )
    if table.fallback_coverage_order:
        return pool[:budget]
    order = sorted(range(len(pool)), key=lambda i: (-float(table.scores_final[i]), pool[i].key))
    return [pool[i] for i in order[:budget]]


# ---------------------------------------------------------------------------
# Trend-margin calibration (#243 leftover: ensemble under-shoots ~2.6x)
# ---------------------------------------------------------------------------


def trend_slopes(values: Sequence[float], cd_rows: np.ndarray) -> np.ndarray:
    """Per-Re OLS slope of ``log10 C_D`` against the swept axis values.

    ``cd_rows`` is ``(n_values, n_re)`` — one C_D curve per swept value
    (the layout :func:`tensorlbm.ai.active_learning.trend_stat`
    consumes).  For ``n_values == 2`` the OLS slope degenerates to the
    two-point difference quotient (exact).  This is the magnitude
    companion of ``trend_stat`` (which measures sign only).
    """
    v = np.asarray(values, dtype=np.float64)
    cd = np.asarray(cd_rows, dtype=np.float64)
    if cd.ndim != 2 or cd.shape[0] != v.size or cd.shape[1] < 1:
        raise ValueError(f"cd_rows must be (n_values={v.size}, n_re>=1), got {cd.shape}")
    if v.size < 2 or not np.isfinite(v).all() or not np.isfinite(cd).all():
        raise ValueError("need >= 2 finite axis values and finite C_D rows")
    if not (cd > 0).all():
        raise ValueError("C_D rows must be positive (log10 is taken)")
    y = np.log10(cd)
    v_c = v - v.mean()
    denom = float(v_c @ v_c)
    if denom <= 0.0:
        raise ValueError("axis values are constant; slope undefined")
    y_c = y - y.mean(axis=0)[None, :]
    return (v_c[:, None] * y_c).sum(axis=0) / denom


@dataclass(frozen=True)
class MarginRatioStat:
    """``slope_pred / slope_truth`` of one trend sweep, per Re column."""

    axis: str
    ratios: tuple[float, ...]
    median_ratio: float
    sign_agree: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "ratios": list(self.ratios),
            "median_ratio": self.median_ratio,
            "sign_agree": self.sign_agree,
        }


def margin_ratio(
    axis: str,
    values: Sequence[float],
    truth_cd: np.ndarray,
    pred_cd: np.ndarray,
) -> MarginRatioStat:
    """Ratio of predicted to cache-truth trend slope along one axis.

    The #243 leftover finding in one number: ratio ~0.38 means the
    retrained ensemble carries ~38 % of the true trend amplitude
    (under-shot ~2.6x).  A negative ratio means the predicted trend runs
    the OPPOSITE way — a sign failure no scale can calibrate
    (``sign_agree`` is False then).
    """
    st = trend_slopes(values, truth_cd)
    sp = trend_slopes(values, pred_cd)
    if not (np.abs(st) > 0).all():
        raise ValueError("truth slope has an exactly-zero column; ratio undefined")
    ratios = sp / st
    med = float(np.median(ratios))
    return MarginRatioStat(
        axis=str(axis),
        ratios=tuple(float(r) for r in ratios),
        median_ratio=med,
        sign_agree=bool(np.all(np.asarray(ratios) > 0.0)),
    )


@dataclass(frozen=True)
class TrendMarginCalibration:
    """Per-axis trend-margin scale + the honesty verdict of the fit.

    ``scales[axis] = 1 / median(fit ratios)`` so that
    :meth:`apply` maps a damped predicted slope onto the truth
    amplitude; ``calibratable[axis]`` records whether the fit ratios
    were stable enough for that to mean anything (enough samples, one
    sign, dispersion ``IQR / |median| <= max_dispersion``).
    """

    scales: dict[str, float]
    fit_median: dict[str, float]
    fit_dispersion: dict[str, float]
    n_fit: dict[str, int]
    calibratable: dict[str, bool]
    notes: dict[str, str] = field(default_factory=dict)

    def apply(self, axis: str, slope: float | np.ndarray) -> float | np.ndarray:
        """Scale a predicted trend slope onto the truth amplitude."""
        if axis not in self.scales:
            raise KeyError(f"axis {axis!r} not in calibration {sorted(self.scales)}")
        return self.scales[axis] * np.asarray(slope, dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scales": dict(self.scales),
            "fit_median": dict(self.fit_median),
            "fit_dispersion": dict(self.fit_dispersion),
            "n_fit": dict(self.n_fit),
            "calibratable": dict(self.calibratable),
            "notes": dict(self.notes),
        }


def calibrate_trend_margin(
    ratios_by_axis: Mapping[str, Sequence[float]],
    *,
    max_dispersion: float = 0.30,
    min_samples: int = 3,
) -> TrendMarginCalibration:
    """Fit per-axis trend-margin scales from measured slope ratios.

    Input: the ``median_ratio`` entries of :class:`MarginRatioStat`
    grouped by axis (one per retrain / rep / budget cell).  A
    multiplicative scale is only meaningful when the ratios are
    (a) enough in number, (b) of ONE sign (a sign-flipped trend is a
    coverage failure, not a calibration one) and (c) tight
    (``IQR / |median| <= max_dispersion``).  Axes failing any gate come
    back with ``calibratable=False`` and an explicit note — no scale is
    forced on unstable data.
    """
    scales: dict[str, float] = {}
    medians: dict[str, float] = {}
    dispersions: dict[str, float] = {}
    counts: dict[str, int] = {}
    verdicts: dict[str, bool] = {}
    notes: dict[str, str] = {}
    for axis, raw in ratios_by_axis.items():
        r = np.asarray([float(x) for x in raw if np.isfinite(float(x))], dtype=np.float64)
        counts[axis] = int(r.size)
        if r.size == 0:
            scales[axis] = 1.0
            medians[axis] = float("nan")
            dispersions[axis] = float("inf")
            verdicts[axis] = False
            notes[axis] = "no finite ratios"
            continue
        med = float(np.median(r))
        medians[axis] = med
        iqr = float(np.quantile(r, 0.75) - np.quantile(r, 0.25))
        disp = float("inf") if med == 0.0 else abs(iqr / med)
        dispersions[axis] = disp
        same_sign = bool(np.all(np.sign(r) == np.sign(med)))
        reasons: list[str] = []
        if r.size < min_samples:
            reasons.append(f"only {r.size} samples (< {min_samples})")
        if not same_sign:
            reasons.append("fit ratios mix signs (sign-flip regime; a scale cannot fix a sign)")
        if disp > max_dispersion:
            reasons.append(f"dispersion IQR/|median| {disp:.2f} > {max_dispersion}")
        if med == 0.0:
            reasons.append("median ratio is exactly zero; no scale defined")
            scales[axis] = 1.0
            verdicts[axis] = False
        else:
            scales[axis] = 1.0 / med
            verdicts[axis] = not reasons
        notes[axis] = "; ".join(reasons) if reasons else "calibratable"
    return TrendMarginCalibration(
        scales=scales,
        fit_median=medians,
        fit_dispersion=dispersions,
        n_fit=counts,
        calibratable=verdicts,
        notes=notes,
    )


def apply_trend_calibration(
    calibration: TrendMarginCalibration,
    ratios_by_axis: Mapping[str, Sequence[float]],
    *,
    band: float = 0.30,
) -> dict[str, dict[str, Any]]:
    """Apply a fitted calibration to VALIDATION ratios (the holdout check).

    Returns per axis the raw and calibrated ratios plus the in-band
    fraction (``|calibrated - 1| <= band`` — the task's 1 +/- 0.3
    target).  Axes the fit flagged uncalibratable keep their raw ratios
    and ``in_band_fraction = 0`` (honest: an unstable scale is reported,
    not applied silently).
    """
    out: dict[str, dict[str, Any]] = {}
    for axis, raw in ratios_by_axis.items():
        r = np.asarray([float(x) for x in raw if np.isfinite(float(x))], dtype=np.float64)
        entry: dict[str, Any] = {"n": int(r.size)}
        if r.size == 0 or axis not in calibration.scales:
            entry.update({"raw_median": None, "calibrated_median": None, "in_band_fraction": 0.0})
            out[axis] = entry
            continue
        usable = bool(calibration.calibratable.get(axis, False))
        cal = calibration.scales[axis] * r if usable else r
        entry.update(
            {
                "raw_median": float(np.median(r)),
                "calibrated_median": float(np.median(cal)),
                "calibrated_ratios": [float(x) for x in cal],
                "calibratable": usable,
                "in_band_fraction": float(np.mean(np.abs(cal - 1.0) <= band)),
            }
        )
        out[axis] = entry
    return out
