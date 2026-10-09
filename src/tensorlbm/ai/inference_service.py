"""B4-P1d — SUBOFF drag-surrogate inference service: ensemble UQ + guardrails.

Serving layer over the B4-v3/v4 conditional-FNO drag surrogate
(:class:`tensorlbm.ai.drag_cond.CondFNODrag`).  Three concerns are bundled
here because they only make sense together at the service boundary:

1. **Prediction** — one design point ``(hull_type, sail_scale, fin_scale)``
   swept over a batch of Reynolds numbers, tensorised into one forward
   pass per ensemble member (:class:`ModelEnsembleBackend`), or replayed
   from archived per-seed run predictions
   (:class:`ReplayEnsembleBackend`) when no checkpoint was archived.

2. **Deep-ensemble UQ** — per-point mean / std / min-max band over the seed
   ensemble members (both backends expose the same ``(M, N)`` member
   matrix, so downstream statistics are backend-agnostic).  Calibration
   helpers (:func:`ensemble_picp`, :func:`error_std_spearman`) quantify
   the band against archived LOHO truths.  :func:`resolve_uq_temperature`
   is the #251 serving knob (``uq_temperature`` argument /
   ``TENSORLBM_DRAG_UQ_TEMPERATURE``): it rescales the *reported* std only
   — verdicts and the min-max band never see it (audit in
   ``docs/uq_temperature_serving_20260827.md``).

3. **Extrapolation guardrails** — :class:`Guardrail` is a pluggable
   interface over an arbitrary feature space.  The default
   :class:`EnvelopeMahalanobisGuardrail` fits per-dimension envelopes plus
   a shrunk-covariance Mahalanobis distance on the *manual* ``condition_v3``
   space and is deliberately feature-agnostic: constructing it with the
   SDF-latent matrix of unmerged PR #235 (``latents.npz``, key
   ``sdf_joint``) yields the latent-distance guard without importing any
   unmerged code.  :func:`guard_threshold_sweep` turns archived LOHO
   errors into a flag-coverage vs large-error-capture table for threshold
   choice.

Known blind spot (measured, see ``docs/inference_service_20260824.md``):
the manual v3 features are identical for geometry corners that voxelise
identically (the bare-corner ladder of the G2b campaign), so a manual-space
guard cannot flag in-envelope geometry extrapolation there — PR #235's SDF
latent space is the planned fix and this interface accepts it unchanged.

4. **Measured-curve Re fallback (opt-in)** — the ``re_policy="quad3_fallback"
   flag of :meth:`DragSurrogateService.predict` serves the CACHED measured
   curve of an exactly-matched design (quad3: exact quadratic through the 3
   nearest cached Re levels in log10-Re / log10-C_D space) for query
   Reynolds numbers outside the corpus Re window, where the network
   extrapolates poorly (~14 % median on unseen geometries) but the measured
   curve is still the ground truth (campaign 2026-08-27: quad3 <= 0.21 %
   over the full tested range, ``global_lin`` 4 %+ systematic — banned).
   Default OFF; the network path stays byte-identical.  See
   ``docs/quad3_fallback_20260828.md``.

5. **Field borrowing for new geometries (opt-in)** — the
   ``field_policy="field_borrow"`` flag of :meth:`DragSurrogateService.predict`
   serves a query whose design is absent from the attached field cache by
   borrowing an in-manifold reference field from a
   :class:`~tensorlbm.ai.field_provider.FieldProvider` pool, keyed on the
   query SDF (``sdf=``); the response carries the retrieval provenance and
   guard flags, never silently.  Default OFF; the cached-field path stays
   byte-identical.  See ``docs/field_borrow_20260904.md``.

The module is import-safe without fastapi/onnx (the HTTP router lives in
``app/backend/routers/drag_surrogate.py``; ONNX helpers degrade to honest
failure reports).
"""

from __future__ import annotations

import copy
import math
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from .drag_cond import (
    COND_V3_CHANNEL_NAMES,
    PRODUCTION_GRID,
    CondFNODrag,
    SuboffGrid,
    condition_v3,
    geometry_channels,
    suboff_geometry_features,
)
from .field_borrow import (
    FIELD_POLICY_BORROW,
    FIELD_POLICY_CACHE,
    borrow_serving_field,
    param_cond_rows,
    resolve_field_policy,
)
from .field_provider import FieldProvider

__all__ = [
    "ENV_UQ_TEMPERATURE",
    "FIELD_POLICY_BORROW",
    "FIELD_POLICY_CACHE",
    "FLAG_OK",
    "FLAG_REJECT",
    "FLAG_REVIEW",
    "RE_POLICY_NETWORK",
    "RE_POLICY_QUAD3_FALLBACK",
    "BackendQueryError",
    "CalibrationRow",
    "CondDragCheckpoint",
    "DragCurveResult",
    "DragSurrogateService",
    "EnvelopeMahalanobisGuardrail",
    "Guardrail",
    "GuardVerdict",
    "ModelEnsembleBackend",
    "ReplayDesign",
    "ReplayEnsembleBackend",
    "SpectralConv2dMatmul",
    "ensemble_picp",
    "quad3_loo_std",
    "quad3_nearest3",
    "resolve_re_policy",
    "resolve_field_policy",
    "resolve_uq_temperature",
    "ensemble_stats",
    "error_std_spearman",
    "export_cond_fno_onnx",
    "guard_threshold_sweep",
    "load_checkpoint",
    "save_checkpoint",
    "to_matmul_spectral",
]

#: Guard verdict levels.
FLAG_OK = "ok"
FLAG_REVIEW = "review"
FLAG_REJECT = "reject"

#: Hull identity order of the B4 caches (``cache.npz['hull']`` ints).
HULL_ORDER = ("bare_hull", "with_sail", "full")

_DEFAULT_DEVICE = "cpu"

#: Default Re serving policy of :meth:`DragSurrogateService.predict`: the
#: ensemble network path only (the pre-flag behaviour, byte-identical).
RE_POLICY_NETWORK = "network"

#: Opt-in Re serving policy: for a query design that EXACTLY matches a
#: cached ``(hull, sail, fin, u_in)`` key with >= 3 cached Re rows, points
#: outside the corpus Re window are served by the quad3 measured-curve
#: extrapolation instead of the network (fresh-Re campaign 2026-08-27
#: adjudication; ``docs/quad3_fallback_20260828.md``).
RE_POLICY_QUAD3_FALLBACK = "quad3_fallback"

#: Environment variable overriding the *reported* ensemble-std temperature
#: (serving calibration knob, landing of the #251 UQ audit).  Default 1.0
#: reports the raw deep-ensemble std — the pre-knob behaviour, bit for bit.
ENV_UQ_TEMPERATURE = "TENSORLBM_DRAG_UQ_TEMPERATURE"


def resolve_uq_temperature(explicit: float | str | None = None) -> float:
    """Reported-sigma temperature: explicit argument > env var > 1.0.

    Mirrors :func:`tensorlbm.ai.service_backends.resolve_backend_kind`
    (``arg > TENSORLBM_DRAG_UQ_BACKEND > torch``) so all serving knobs of
    the drag service follow one precedence convention.  The value scales
    only the **reported** ensemble std of :meth:`DragSurrogateService.predict`
    — the guard verdict (Mahalanobis + envelope) and the member min-max
    band never see it, so no threshold semantics can shift with it.  Must
    be finite and strictly positive; a blank env var counts as unset.
    """
    raw = explicit if explicit is not None else os.environ.get(ENV_UQ_TEMPERATURE)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return 1.0
    try:
        t = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"uq temperature {raw!r} is not a number "
            f"({ENV_UQ_TEMPERATURE} or the uq_temperature argument)"
        ) from exc
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(
            f"uq temperature must be finite and positive, got {t!r} "
            f"({ENV_UQ_TEMPERATURE} or the uq_temperature argument)"
        )
    return t


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardVerdict:
    """Result of one guardrail check.

    ``flag`` is one of ``ok`` / ``review`` / ``reject``; ``score`` is the
    scalar severity (Mahalanobis distance in the default guard, any
    monotone severity in custom guards); ``reasons`` are human-readable
    violation strings for the worst offending row.
    """

    flag: str
    score: float
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "flag": self.flag,
            "score": float(self.score),
            "reasons": list(self.reasons),
        }


class Guardrail(Protocol):
    """Pluggable extrapolation guard interface.

    Implementations fit an in-distribution model over an ``(N, D)`` feature
    matrix (any space: manual condition channels, SDF latents, ...) and
    score query rows in the same space.  The service maps a design query
    to feature rows before calling :meth:`check`, so a guard never needs
    to know about hull types or Reynolds numbers.
    """

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of the D feature dimensions (for reasons/reporting)."""
        ...

    def row_scores(self, features: np.ndarray) -> np.ndarray:
        """Per-row severity score, shape ``(N,)``."""
        ...

    def row_reasons(self, features: np.ndarray) -> list[str]:
        """Per-row violation strings (empty when the row is unremarkable)."""
        ...

    def check(self, features: np.ndarray) -> GuardVerdict:
        """Aggregate a verdict over all rows (worst row drives the flag)."""
        ...


@dataclass(frozen=True)
class _EnvelopeStats:
    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    cov_inv: np.ndarray
    n_fit: int


def _chi2_quantile_wilson_hilferty(p: float, dof: int) -> float:
    """Chi-square quantile via the Wilson-Hilferty normal approximation.

    Accurate to ~0.5 % across the degrees of freedom used here (8 manual
    channels, 32 SDF latents); used only to derive default guard
    thresholds, so scipy is not pulled in as a dependency.
    """
    z = {0.99: 2.326348, 0.999: 3.090232}[p]
    c = 2.0 / (9.0 * dof)
    return dof * (1.0 - c + z * math.sqrt(c)) ** 3


class EnvelopeMahalanobisGuardrail:
    """Default guard: per-dim envelopes + shrunk-covariance Mahalanobis.

    Fit on an ``(N, D)`` feature matrix (rows = training corpus points in
    the served feature space).  A query row is *inside* iff every dimension
    lies within the fitted envelope (widened by ``margin`` as a fraction of
    the per-dim range) **and** its Mahalanobis distance to the fit mean is
    below ``mahal_threshold``; distances above ``review_threshold`` flag
    for review without rejecting.

    The default distance thresholds are chi-square calibrated for the
    feature dimensionality — for ``d = mahalanobis`` of an in-distribution
    row, ``d**2`` follows ``chi2(D)``, so the defaults are the square
    roots of the 0.99 / 0.999 quantiles (Wilson-Hilferty approximation,
    e.g. 4.48 / 5.13 at D=8).  Without this, a fixed cutoff would either
    flag half the training rows (small D) or almost nothing (large D).

    The Mahalanobis covariance is shrunk towards a scaled identity
    (``cov = (1 - shrinkage) * S + shrinkage * tr(S)/D * I``) because the
    manual condition space is strongly collinear (``solid_frac`` is an
    exact linear function of ``sail_frac``/``fin_frac``) and D is small
    relative to corpus size.

    The class is intentionally feature-agnostic: pass the SDF-latent matrix
    of PR #235 as ``features`` and ``("z0", ..., "z31")`` as ``names`` to
    obtain the latent-distance guard without any unmerged import.
    """

    def __init__(
        self,
        features: np.ndarray,
        names: Sequence[str] | None = None,
        *,
        mahal_threshold: float | None = None,
        review_threshold: float | None = None,
        margin: float = 0.05,
        shrinkage: float = 0.1,
    ) -> None:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError(f"features must be (N>=2, D), got {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError("features contain non-finite values")
        if names is None:
            if features.shape[1] == len(COND_V3_CHANNEL_NAMES):
                names = COND_V3_CHANNEL_NAMES
            else:
                names = tuple(f"f{i}" for i in range(features.shape[1]))
        if len(names) != features.shape[1]:
            raise ValueError(f"{len(names)} names for {features.shape[1]} dims")
        self._names = tuple(str(n) for n in names)
        d = features.shape[1]
        if review_threshold is None:
            review_threshold = math.sqrt(_chi2_quantile_wilson_hilferty(0.99, d))
        if mahal_threshold is None:
            mahal_threshold = math.sqrt(_chi2_quantile_wilson_hilferty(0.999, d))
        self.mahal_threshold = float(mahal_threshold)
        self.review_threshold = float(review_threshold)
        self.margin = float(margin)

        mean = features.mean(axis=0)
        lo = features.min(axis=0)
        hi = features.max(axis=0)
        pad = self.margin * (hi - lo)
        cov = np.atleast_2d(np.cov(features, rowvar=False))
        d = features.shape[1]
        trace = float(np.trace(cov))
        shrunk = (1.0 - shrinkage) * cov + (shrinkage * trace / d) * np.eye(d)
        # Jitter guard: strongly degenerate latent/feature spaces stay invertible.
        shrunk += 1e-10 * max(trace, 1.0) * np.eye(d)
        self._stats = _EnvelopeStats(
            lo=lo - pad,
            hi=hi + pad,
            mean=mean,
            cov_inv=np.linalg.inv(shrunk),
            n_fit=int(features.shape[0]),
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def n_fit(self) -> int:
        return self._stats.n_fit

    def row_scores(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        diff = features - self._stats.mean
        maha = np.einsum("ij,jk,ik->i", diff, self._stats.cov_inv, diff)
        scores: np.ndarray = np.sqrt(np.maximum(maha, 0.0))
        return scores

    def row_reasons(self, features: np.ndarray) -> list[str]:
        features = np.asarray(features, dtype=np.float64)
        scores = self.row_scores(features)
        reasons: list[str] = []
        for i, row in enumerate(features):
            bits: list[str] = []
            below = np.nonzero(row < self._stats.lo)[0]
            above = np.nonzero(row > self._stats.hi)[0]
            for j in below:
                bits.append(
                    f"{self._names[j]}={row[j]:.4g} below envelope "
                    f"[{self._stats.lo[j]:.4g}, {self._stats.hi[j]:.4g}]"
                )
            for j in above:
                bits.append(
                    f"{self._names[j]}={row[j]:.4g} above envelope "
                    f"[{self._stats.lo[j]:.4g}, {self._stats.hi[j]:.4g}]"
                )
            if scores[i] >= self.mahal_threshold:
                bits.append(
                    f"mahalanobis={scores[i]:.2f} >= reject_threshold {self.mahal_threshold:.2f}"
                )
            elif scores[i] >= self.review_threshold:
                bits.append(
                    f"mahalanobis={scores[i]:.2f} >= review_threshold {self.review_threshold:.2f}"
                )
            reasons.append("; ".join(bits))
        return reasons

    def check(self, features: np.ndarray) -> GuardVerdict:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self._names):
            raise ValueError(f"features must be (N, {len(self._names)}), got {features.shape}")
        scores = self.row_scores(features)
        reasons = self.row_reasons(features)
        worst = int(np.argmax(scores))
        inside = np.all((features >= self._stats.lo) & (features <= self._stats.hi), axis=1)
        if scores[worst] >= self.mahal_threshold or not inside.all():
            flag = FLAG_REJECT
        elif scores[worst] >= self.review_threshold:
            flag = FLAG_REVIEW
        else:
            flag = FLAG_OK
        worst_reasons = tuple(r for r in (reasons[worst],) if r)
        return GuardVerdict(flag=flag, score=float(scores[worst]), reasons=worst_reasons)


class NullGuardrail:
    """Pass-through guard: every query is ``ok`` (score 0).

    For A/B latency measurement of the guard overhead and as an explicit,
    visible opt-out — never a silent default (the service always reports
    which guard class produced the verdict).
    """

    def __init__(self, names: Sequence[str] = COND_V3_CHANNEL_NAMES) -> None:
        self._names = tuple(str(n) for n in names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def n_fit(self) -> int:
        return 0

    def row_scores(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        return np.zeros(features.shape[0])

    def row_reasons(self, features: np.ndarray) -> list[str]:
        features = np.asarray(features, dtype=np.float64)
        return [""] * features.shape[0]

    def check(self, features: np.ndarray) -> GuardVerdict:
        _ = np.asarray(features, dtype=np.float64)
        return GuardVerdict(flag=FLAG_OK, score=0.0, reasons=())


# ---------------------------------------------------------------------------
# Checkpoint container for real model members
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CondDragCheckpoint:
    """One trained ensemble member plus the normalisation it was fit with.

    The v3/v4 training protocol z-scores the field channels and the
    condition vector on the fit split and regresses z-scored ``log10 C_D``
    — none of that is recoverable from a bare ``state_dict``, so the
    service checkpoint format bundles: ``arch`` (CondFNODrag kwargs),
    ``state_dict``, ``norm`` (the six fit-stat arrays) and free-form
    ``meta`` (arm, seed, split, corpus tag).
    """

    arch: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    norm: dict[str, np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_model(self, device: str | torch.device = _DEFAULT_DEVICE) -> CondFNODrag:
        model = CondFNODrag(**self.arch)
        model.load_state_dict(self.state_dict)
        model.to(device)
        model.eval()
        return model


def save_checkpoint(
    ckpt: CondDragCheckpoint, path: str | Path, *, extra: dict[str, Any] | None = None
) -> str:
    """Serialise a member checkpoint (torch.save of a plain dict)."""
    payload: dict[str, Any] = {
        "format": "tensorlbm.ai.inference_service.CondDragCheckpoint",
        "version": 1,
        "arch": dict(ckpt.arch),
        "state_dict": {k: v.detach().cpu() for k, v in ckpt.state_dict.items()},
        "norm": {k: np.asarray(v) for k, v in ckpt.norm.items()},
        "meta": dict(ckpt.meta),
    }
    if extra:
        payload["extra"] = dict(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)
    return str(p.resolve())


def load_checkpoint(
    path: str | Path, device: str | torch.device = _DEFAULT_DEVICE
) -> CondDragCheckpoint:
    """Load a member checkpoint written by :func:`save_checkpoint`."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "arch" not in payload or "state_dict" not in payload:
        raise ValueError(f"{path} is not a CondDragCheckpoint file")
    return CondDragCheckpoint(
        arch=dict(payload["arch"]),
        state_dict=payload["state_dict"],
        norm={k: np.asarray(v) for k, v in payload.get("norm", {}).items()},
        meta=dict(payload.get("meta", {})),
    )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class BackendQueryError(RuntimeError):
    """Raised when a backend cannot serve the requested design/Re query."""


_REQUIRED_NORM_KEYS = ("ch_mean", "ch_std", "p_mean", "p_std", "y_mean", "y_std")


class ModelEnsembleBackend:
    """Real-model backend: one batched forward per checkpoint member.

    ``predict`` takes a reference field ``(5, ny, nx)`` (the mid-plane
    ``[ux/u, uy/u, uz/u, rho, solid_mask]`` stack of the B4 cache
    convention, from the corpus cache or supplied by the caller) plus
    condition rows — ``(N, 8)`` v3/v4 vectors or ``(N, 9)`` v5 (sail
    axial-position channel) — normalises both with each member's fit
    statistics, runs all N Reynolds points in one batch and returns the
    member matrix ``member_cd[m, n]`` in linear C_D space.  The accepted
    condition width is read from the members' architecture (all members
    must agree), so v3/v4/v5 ensembles serve through the same call.
    """

    def __init__(self, ckpts: Sequence[CondDragCheckpoint], device: str = _DEFAULT_DEVICE) -> None:
        if not ckpts:
            raise ValueError("ensemble needs at least one checkpoint")
        for c in ckpts:
            missing = [k for k in _REQUIRED_NORM_KEYS if k not in c.norm]
            if missing:
                raise ValueError(f"checkpoint norm missing keys: {missing}")
        self._ckpts = list(ckpts)
        self._norms = [c.norm for c in ckpts]
        self.device = torch.device(device)
        self._models = [c.to_model(self.device) for c in ckpts]
        dims = {int(getattr(m.cond_embed[0], "in_features")) for m in self._models}
        if len(dims) != 1:
            raise ValueError(f"ensemble members disagree on cond_dim: {sorted(dims)}")
        self._cond_dim = dims.pop()

    @property
    def cond_dim(self) -> int:
        """Condition-vector width the members were trained with (8 or 9)."""
        return self._cond_dim

    @property
    def n_members(self) -> int:
        return len(self._models)

    @property
    def kind(self) -> str:
        return "model"

    def member_labels(self) -> list[str]:
        return [str(c.meta.get("member", f"m{i}")) for i, c in enumerate(self._ckpts)]

    def predict(self, fields: np.ndarray, cond: np.ndarray) -> np.ndarray:
        fields = np.asarray(fields, dtype=np.float32)
        cond = np.asarray(cond, dtype=np.float64)
        if fields.ndim != 3 or fields.shape[0] != 5:
            raise ValueError(f"fields must be (5, ny, nx), got {fields.shape}")
        if cond.ndim != 2 or cond.shape[1] != self._cond_dim:
            raise ValueError(f"cond must be (N, {self._cond_dim}), got {cond.shape}")
        x = torch.from_numpy(fields).to(self.device)
        p = torch.from_numpy(cond.astype(np.float32)).to(self.device)
        xn = x.unsqueeze(0).expand(p.shape[0], -1, -1, -1)  # (N, 5, ny, nx)
        outs = []
        with torch.no_grad():
            for model, norm in zip(self._models, self._norms):
                ch_m = torch.as_tensor(norm["ch_mean"], dtype=torch.float32, device=self.device)
                ch_s = torch.as_tensor(norm["ch_std"], dtype=torch.float32, device=self.device)
                p_m = torch.as_tensor(norm["p_mean"], dtype=torch.float32, device=self.device)
                p_s = torch.as_tensor(norm["p_std"], dtype=torch.float32, device=self.device)
                y_m = float(norm["y_mean"])
                y_s = float(norm["y_std"])
                x_norm = (xn - ch_m.view(1, -1, 1, 1)) / ch_s.view(1, -1, 1, 1)
                p_norm = (p - p_m) / p_s
                z = model(x_norm, p_norm)
                outs.append(10.0 ** (z.double().cpu().numpy() * y_s + y_m))
        return np.stack(outs, axis=0)  # (M, N)

    def predict_batch(
        self,
        fields: np.ndarray,
        cond: np.ndarray,
        counts: np.ndarray,
    ) -> np.ndarray:
        """Batched multi-geometry variant of :meth:`predict` (B4-P3a sweep).

        ``fields`` is ``(G, 5, ny, nx)`` (one reference field per geometry),
        ``cond`` the ``(N, cond_dim)`` condition rows of all geometries
        concatenated and ``counts`` the per-geometry row counts
        (``counts.sum() == N``).  Row ``n`` is evaluated on the field of the
        geometry it belongs to (``torch.repeat_interleave`` expansion), so a
        whole slider sweep runs as **one forward per ensemble member**
        instead of one per geometry).  Same normalisation and de-scaling as
        :meth:`predict`; ``G == 1`` reproduces it bit-identically, and at
        ``G > 1`` individual rows match a per-geometry :meth:`predict` call
        only to float32 batch-kernel noise (measured max rel ~2e-8).
        """
        fields = np.asarray(fields, dtype=np.float32)
        cond = np.asarray(cond, dtype=np.float64)
        counts = np.asarray(counts, dtype=np.int64)
        if fields.ndim != 4 or fields.shape[1] != 5:
            raise ValueError(f"fields must be (G, 5, ny, nx), got {fields.shape}")
        if cond.ndim != 2 or cond.shape[1] != self._cond_dim:
            raise ValueError(f"cond must be (N, {self._cond_dim}), got {cond.shape}")
        if counts.ndim != 1 or counts.size != fields.shape[0] or not (counts > 0).all():
            raise ValueError(
                f"counts must be positive with one entry per field, got {counts!r} "
                f"for {fields.shape[0]} fields"
            )
        if int(counts.sum()) != cond.shape[0]:
            raise ValueError(f"counts sum {int(counts.sum())} != condition rows {cond.shape[0]}")
        x = torch.from_numpy(fields).to(self.device)
        p = torch.from_numpy(cond.astype(np.float32)).to(self.device)
        reps = torch.as_tensor(counts, device=self.device)
        xn = torch.repeat_interleave(x, reps, dim=0)  # (N, 5, ny, nx)
        outs = []
        with torch.no_grad():
            for model, norm in zip(self._models, self._norms):
                ch_m = torch.as_tensor(norm["ch_mean"], dtype=torch.float32, device=self.device)
                ch_s = torch.as_tensor(norm["ch_std"], dtype=torch.float32, device=self.device)
                p_m = torch.as_tensor(norm["p_mean"], dtype=torch.float32, device=self.device)
                p_s = torch.as_tensor(norm["p_std"], dtype=torch.float32, device=self.device)
                y_m = float(norm["y_mean"])
                y_s = float(norm["y_std"])
                x_norm = (xn - ch_m.view(1, -1, 1, 1)) / ch_s.view(1, -1, 1, 1)
                p_norm = (p - p_m) / p_s
                z = model(x_norm, p_norm)
                outs.append(10.0 ** (z.double().cpu().numpy() * y_s + y_m))
        return np.stack(outs, axis=0)  # (M, N)


@dataclass(frozen=True)
class ReplayDesign:
    """A matched design family inside the replay archive."""

    hull_type: str
    sail_scale: float
    fin_scale: float
    u_in: float
    re: np.ndarray  # ascending
    rows: np.ndarray  # corpus row indices aligned with re


class ReplayEnsembleBackend:
    """Lightweight replay backend over an archived B4 run directory.

    Loads the archived per-seed test predictions (``preds_v3.npz`` /
    ``preds_v4.npz`` layout: ``"{fold}::{arm}[::s{k}]::{true|pred|idx}"``)
    plus the point cache, and serves a queried ``(design, Re grid)`` by
    exact Reynolds match or log10-Re interpolation *per seed member* —
    the ensemble statistics then flow through the identical code path as
    the real-model backend.  Members are the seed variants of one arm
    (seed 0 is the unsuffixed key, seeds k >= 1 the ``s{k}`` keys).

    Exists because the v3/v4 training runs archived predictions but not
    checkpoints; this is the offline-UQ data path, not a deployable
    surrogate.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        arm: str = "C_full",
        fold: str = "loho::full",
        preds_name: str = "preds_v4.npz",
        cache_name: str = "cache.npz",
    ) -> None:
        run_dir = Path(run_dir)
        preds_path = run_dir / preds_name
        cache_path = run_dir / cache_name
        if not preds_path.is_file():
            raise FileNotFoundError(f"preds archive not found: {preds_path}")
        if not cache_path.is_file():
            # The v4 run dir names its combined cache ``cache_v4.npz``; fall
            # back to it only when the caller kept the default name.
            if cache_name == "cache.npz":
                cache_path = run_dir / "cache_v4.npz"
            if not cache_path.is_file():
                raise FileNotFoundError(f"point cache not found: {run_dir / cache_name}")
        self.run_dir = str(run_dir)
        self.arm = arm
        self.fold = fold

        preds = np.load(preds_path)
        if f"{fold}::{arm}::pred" not in preds.files:
            raise KeyError(f"arm {arm!r} not present in {preds_path} for fold {fold!r}")
        member_tags = [""]
        k = 1
        while f"{fold}::{arm}::s{k}::pred" in preds.files:
            member_tags.append(f"s{k}")
            k += 1
        self._member_tags = member_tags

        self._idx = np.asarray(preds[f"{fold}::{arm}::idx"], dtype=np.int64)
        self._member_preds = {
            tag: np.asarray(
                preds[f"{fold}::{arm}::pred" if tag == "" else f"{fold}::{arm}::{tag}::pred"],
                dtype=np.float64,
            )
            for tag in member_tags
        }
        self._truth = np.asarray(preds[f"{fold}::{arm}::true"], dtype=np.float64)

        cache = np.load(cache_path)
        hull_ids = np.asarray(cache["hull"], dtype=np.int64)
        self._table = {
            "hull": np.array([HULL_ORDER[int(h)] for h in hull_ids]),
            "sail": np.asarray(cache["sail"], dtype=np.float64),
            "fin": np.asarray(cache["fin"], dtype=np.float64),
            "uin": np.asarray(cache["uin"], dtype=np.float64),
            "re": np.asarray(cache["re"], dtype=np.float64),
        }

    @property
    def kind(self) -> str:
        return "replay"

    @property
    def n_members(self) -> int:
        return len(self._member_tags)

    @property
    def truth(self) -> np.ndarray:
        """Archived ground truth aligned with the fold test rows."""
        return self._truth

    def member_labels(self) -> list[str]:
        return [tag if tag else "s0" for tag in self._member_tags]

    def member_matrix(self) -> np.ndarray:
        """``(M, N_fold)`` member predictions over the archived test rows."""
        return np.stack([self._member_preds[t] for t in self._member_tags], axis=0)

    def fold_rows(self) -> np.ndarray:
        return self._idx

    def designs(self) -> list[ReplayDesign]:
        """Design families present in this fold, each with >= 1 point."""
        out: dict[tuple[str, float, float, float], list[tuple[float, int]]] = {}
        table = self._table
        for row, re in zip(self._idx, table["re"][self._idx]):
            key = (
                str(table["hull"][row]),
                float(table["sail"][row]),
                float(table["fin"][row]),
                float(table["uin"][row]),
            )
            out.setdefault(key, []).append((float(re), int(row)))
        designs = []
        for (hull, sail, fin, uin), pts in sorted(out.items()):
            pts.sort()
            designs.append(
                ReplayDesign(
                    hull_type=hull,
                    sail_scale=sail,
                    fin_scale=fin,
                    u_in=uin,
                    re=np.array([p[0] for p in pts]),
                    rows=np.array([p[1] for p in pts]),
                )
            )
        return designs

    def _lookup_design(
        self, hull_type: str, sail_scale: float, fin_scale: float, u_in: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Row indices + Re values of the matching design, sorted by Re."""
        table = self._table
        match = np.zeros(len(table["re"]), dtype=bool)
        for row in self._idx:
            if (
                str(table["hull"][row]) == hull_type
                and float(table["sail"][row]) == float(sail_scale)
                and float(table["fin"][row]) == float(fin_scale)
                and abs(float(table["uin"][row]) - float(u_in)) <= 1e-12
            ):
                match[row] = True
        rows = np.nonzero(match)[0]
        if rows.size == 0:
            raise BackendQueryError(
                f"design ({hull_type}, sail={sail_scale}, fin={fin_scale}, u_in={u_in}) "
                f"not present in replay fold {self.fold!r}"
            )
        res = table["re"][rows]
        order = np.argsort(res)
        return rows[order], res[order]

    def _rows_to_fold_pos(self, rows: np.ndarray) -> np.ndarray:
        """Positions of corpus rows within the fold test arrays."""
        pos = {int(r): i for i, r in enumerate(self._idx)}
        return np.array([pos[int(r)] for r in rows], dtype=np.int64)

    def predict(
        self,
        hull_type: str,
        sail_scale: float,
        fin_scale: float,
        re_grid: np.ndarray,
        *,
        u_in: float = 0.1,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Member C_D matrix ``(M, len(re_grid))`` for one design.

        Exact Re matches are served directly; queries between archived Re
        points are linearly interpolated in log10-Re independently per
        member; queries outside the archived sweep fall back to the
        nearest archived point and are marked ``n_extrapolated`` in the
        returned info dict.
        """
        re_grid = np.asarray(re_grid, dtype=np.float64)
        if re_grid.size == 0:
            raise ValueError("re_grid must be non-empty")
        rows, res = self._lookup_design(hull_type, float(sail_scale), float(fin_scale), float(u_in))
        fold_pos = self._rows_to_fold_pos(rows)
        member_at_rows = np.stack(
            [preds[fold_pos] for preds in self._member_preds.values()], axis=0
        )
        if rows.size == 1:
            member_out = np.repeat(member_at_rows, re_grid.size, axis=1)
            exact = np.isclose(re_grid, res[0], rtol=1e-9)
            info: dict[str, Any] = {
                "field_rows": rows.tolist(),
                "archived_re": res.tolist(),
                "mode": "single_point",
                "n_exact": int(exact.sum()),
                "n_interpolated": 0,
                "n_extrapolated": int((~exact).sum()),
            }
            return member_out, info
        log_re = np.log10(res)
        log_q = np.log10(re_grid)
        member_out = np.empty((self.n_members, re_grid.size), dtype=np.float64)
        for m in range(self.n_members):
            member_out[m] = np.interp(log_q, log_re, member_at_rows[m])
        inside = (re_grid >= res[0]) & (re_grid <= res[-1])
        exact = np.zeros(re_grid.size, dtype=bool)
        for r in np.unique(res):
            exact |= np.isclose(re_grid, r, rtol=1e-9)
        info = {
            "field_rows": rows.tolist(),
            "archived_re": res.tolist(),
            "mode": "log_re_interp",
            "n_exact": int((exact & inside).sum()),
            "n_interpolated": int((~exact & inside).sum()),
            "n_extrapolated": int((~inside).sum()),
        }
        return member_out, info


# ---------------------------------------------------------------------------
# Measured-curve quad3 fallback (fresh-Re campaign 2026-08-27)
# ---------------------------------------------------------------------------


def resolve_re_policy(re_policy: str | None = None) -> str:
    """Validate an Re serving policy name (fail loud, no silent aliases).

    ``None`` counts as :data:`RE_POLICY_NETWORK` so the default call path
    never touches the fallback logic.  Unknown names raise — a typoed
    policy must not silently degrade to the network path.
    """
    name = RE_POLICY_NETWORK if re_policy is None else str(re_policy)
    if name not in (RE_POLICY_NETWORK, RE_POLICY_QUAD3_FALLBACK):
        raise ValueError(
            f"unknown re_policy {re_policy!r}; expected "
            f"{RE_POLICY_NETWORK!r} or {RE_POLICY_QUAD3_FALLBACK!r}"
        )
    return name


def quad3_nearest3(
    cached_re: np.ndarray, cached_cd: np.ndarray, re_query: float
) -> tuple[float, np.ndarray] | None:
    """quad3 measured-curve value at ``re_query`` — the campaign definition.

    Exactly the ``quad3`` predictor adjudicated by the fresh-Re campaign
    (``analyze_extrap.py`` of 2026-08-27): fit the exact quadratic through
    the 3 cached levels nearest the query in ``|log10 Re|`` and evaluate it
    at ``x = log10(re_query)``::

        c2, c1, c0 = np.polyfit(log10(re_sel), log10(cd_sel), 2)
        cd = 10.0 ** (c2 * x * x + c1 * x + c0)

    For an out-of-window query the nearest 3 cached levels ARE the 3 edge
    levels on the query side (top-3 above, bottom-3 below), so this
    reproduces the campaign slices bit-for-bit on the same rows.  Rows are
    sorted by Re and the selected triple is evaluated in ascending order,
    the campaign ordering, so the float result is identical.

    Returns ``(cd_value, chosen_re_ascending)`` or ``None`` when the
    nearest triple does not contain 3 DISTINCT ``log10 Re`` levels
    (duplicate cached Re -> singular Vandermonde); the caller decides the
    fallback, nothing is silently interpolated.
    """
    re_arr = np.asarray(cached_re, dtype=np.float64)
    cd_arr = np.asarray(cached_cd, dtype=np.float64)
    if re_arr.ndim != 1 or cd_arr.ndim != 1 or re_arr.shape != cd_arr.shape:
        raise ValueError(
            f"cached_re/cached_cd must be matching 1-D arrays, got {re_arr.shape} vs {cd_arr.shape}"
        )
    if re_arr.size < 3:
        raise ValueError(f"quad3 needs >= 3 cached rows, got {re_arr.size}")
    if not np.isfinite(re_arr).all() or not (re_arr > 0.0).all():
        raise ValueError("cached_re entries must be finite and positive")
    if not np.isfinite(cd_arr).all() or not (cd_arr > 0.0).all():
        raise ValueError("cached_cd entries must be finite and positive")
    if not (np.isfinite(re_query) and re_query > 0.0):
        raise ValueError("re_query must be finite and positive")
    order = np.argsort(re_arr, kind="stable")
    re_s = re_arr[order]
    cd_s = cd_arr[order]
    lr = np.log10(re_s)
    x = np.log10(float(re_query))
    sel = np.sort(np.argsort(np.abs(lr - x), kind="stable")[:3])
    if np.unique(lr[sel]).size < 3:
        return None
    c2, c1, c0 = np.polyfit(lr[sel], np.log10(cd_s[sel]), 2)
    return float(10.0 ** (c2 * x * x + c1 * x + c0)), re_s[sel]


def quad3_loo_std(cached_re: np.ndarray, cached_cd: np.ndarray) -> float | None:
    """RMS *relative* leave-one-out residual of :func:`quad3_nearest3`.

    Each cached row ``j`` is treated as a pseudo-query served by the SAME
    estimator built on the remaining rows (nearest 3 to ``re_j`` in
    ``|log10 Re|``); the returned number is
    ``sqrt(mean_j ((cd_j - pred_j) / pred_j)^2)`` — a dimensionless
    relative accuracy of the quad3 estimator on THIS design curve.  The
    service multiplies it by the served value to obtain an absolute C_D
    std (the reported ensemble std is absolute C_D everywhere else).

    ``None`` when fewer than 4 cached rows exist (leave-one-out needs at
    least 3 remaining) or any pseudo-fit is degenerate (duplicate cached
    log10-Re levels — the measured pool381 curve, 157 rows / 126 distinct
    Re, hits this) — NO uncertainty is fabricated; the caller keeps the
    network ensemble std for the point and records why via ``std_source``.
    """
    re_arr = np.asarray(cached_re, dtype=np.float64)
    cd_arr = np.asarray(cached_cd, dtype=np.float64)
    if re_arr.ndim != 1 or cd_arr.ndim != 1 or re_arr.shape != cd_arr.shape:
        raise ValueError(
            f"cached_re/cached_cd must be matching 1-D arrays, got {re_arr.shape} vs {cd_arr.shape}"
        )
    if re_arr.size < 4:
        return None
    rel = np.empty(re_arr.size, dtype=np.float64)
    for j in range(re_arr.size):
        keep = np.ones(re_arr.size, dtype=bool)
        keep[j] = False
        out = quad3_nearest3(re_arr[keep], cd_arr[keep], float(re_arr[j]))
        if out is None:
            return None
        pred = out[0]
        rel[j] = abs(float(cd_arr[j]) - pred) / pred
    return float(np.sqrt(np.mean(rel**2)))


# ---------------------------------------------------------------------------
# Service facade
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DragCurveResult:
    """One served drag curve with UQ band and guard verdict."""

    re: np.ndarray
    cd: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    std: np.ndarray
    guard: GuardVerdict
    hull_type: str
    sail_scale: float
    fin_scale: float
    u_in: float
    backend: str
    members: tuple[str, ...]
    info: dict[str, Any] = field(default_factory=dict)

    def uq_dict(self) -> dict[str, Any]:
        return {
            "lo": self.lo.tolist(),
            "hi": self.hi.tolist(),
            "mean_std": float(np.mean(self.std)) if self.std.size else 0.0,
            "std": self.std.tolist(),
        }


def ensemble_stats(member_cd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deep-ensemble statistics: mean, std (ddof=1), min-max band."""
    member_cd = np.asarray(member_cd, dtype=np.float64)
    mean = member_cd.mean(axis=0)
    std = member_cd.std(axis=0, ddof=1) if member_cd.shape[0] > 1 else np.zeros_like(mean)
    return mean, std, member_cd.min(axis=0), member_cd.max(axis=0)


class DragSurrogateService:
    """Facade: design query -> condition rows -> guard -> backend -> UQ.

    Parameters
    ----------
    backend:
        :class:`ModelEnsembleBackend` (real checkpoints) or
        :class:`ReplayEnsembleBackend` (archived predictions).
    guard:
        Any :class:`Guardrail` over the served feature space (default:
        manual ``condition_v3`` space).
    corpus_cache:
        Optional ``(N, 5, ny, nx)`` field cache (B4 ``cache.npz`` ``x``
        array) resolving the reference field for model-backend queries;
        replay backends ignore it.
    grid:
        Voxel grid for the CAD-derived geometry features.  Defaults to the
        production 128-resolution placement; a coarser grid is accepted
        for tests (features stay self-consistent within the service).
    cache_re / cache_designs:
        Row-aligned Reynolds numbers and ``(hull, sail, fin, u_in)`` keys
        of ``corpus_cache``, used to resolve a design's nearest field.
    cache_cd:
        Row-aligned measured C_D labels of ``cache_re``/``cache_designs``
        (the corpus cache ``cd`` array).  Only consumer: the opt-in
        ``re_policy="quad3_fallback"`` of :meth:`predict`.  Absent (the
        default), the policy declines with
        ``reason="no_measured_curve_cache"`` and the network serves.
    u_in_default:
        Default inlet speed of the corpus (0.1 lattice units in B4).
    uq_temperature:
        Multiplier applied to the **reported** ensemble std only
        (``arg > TENSORLBM_DRAG_UQ_TEMPERATURE > 1.0``, see
        :func:`resolve_uq_temperature`).  ``1.0`` (the default) leaves every
        served number bit-identical to the pre-knob service; ``2.3`` is the
        #251-audited value for *new-design* semantics (see
        ``docs/uq_temperature_serving_20260827.md``).  The guard verdict and
        the member min-max band are never scaled.
    field_provider:
        Optional :class:`~tensorlbm.ai.field_provider.FieldProvider` pool
        enabling the opt-in ``field_policy="field_borrow"`` of
        :meth:`predict` (new-geometry queries with a query SDF but no
        cached reference field).  ``None`` (the default) leaves every
        existing behaviour untouched.
    """

    def __init__(
        self,
        backend: ModelEnsembleBackend | ReplayEnsembleBackend,
        guard: Guardrail,
        *,
        corpus_cache: np.ndarray | None = None,
        grid: SuboffGrid | None = None,
        cache_re: np.ndarray | None = None,
        cache_designs: list[tuple[str, float, float, float]] | None = None,
        cache_cd: np.ndarray | None = None,
        u_in_default: float = 0.1,
        uq_temperature: float | str | None = None,
        field_provider: FieldProvider | None = None,
    ) -> None:
        self.backend = backend
        self.guard = guard
        self.grid = grid if grid is not None else PRODUCTION_GRID
        self.corpus_cache = None if corpus_cache is None else np.asarray(corpus_cache)
        self.cache_re = None if cache_re is None else np.asarray(cache_re, dtype=np.float64)
        self.cache_designs = cache_designs
        self.cache_cd = None if cache_cd is None else np.asarray(cache_cd, dtype=np.float64)
        if (
            self.cache_cd is not None
            and self.cache_re is not None
            and self.cache_cd.shape != self.cache_re.shape
        ):
            raise ValueError(
                f"cache_cd {self.cache_cd.shape} must be row-aligned with "
                f"cache_re {self.cache_re.shape}"
            )
        self.u_in_default = float(u_in_default)
        self.uq_temperature = resolve_uq_temperature(uq_temperature)
        self.field_provider = field_provider

    # -- construction helpers -------------------------------------------------

    @classmethod
    def from_checkpoints(
        cls,
        paths: Sequence[str | Path],
        guard_features: np.ndarray,
        *,
        corpus_cache: np.ndarray | None = None,
        device: str = _DEFAULT_DEVICE,
        guard_names: Sequence[str] | None = None,
        grid: SuboffGrid | None = None,
        cache_re: np.ndarray | None = None,
        cache_designs: list[tuple[str, float, float, float]] | None = None,
        cache_cd: np.ndarray | None = None,
        backend_kind: str | None = None,
        backend_plan: str | Path | None = None,
        uq_temperature: float | str | None = None,
        field_provider: FieldProvider | None = None,
        **guard_kwargs: Any,
    ) -> DragSurrogateService:
        """Real-model service from member checkpoints + guard fit matrix.

        ``grid`` must match the grid the ``guard_features`` were computed
        on (default: the production grid — the B4 caches were built with
        it), for the same reason as :meth:`from_run_dir`.
        ``corpus_cache``/``cache_re``/``cache_designs`` wire up field
        resolution for callers that do not pass ``fields`` per query;
        ``cache_cd`` additionally wires the measured-curve quad3 fallback
        (``CorpusIndex.cd``).

        Backend selection (TRT slice 2026-08-27): ``backend_kind`` wins over
        ``TENSORLBM_DRAG_BACKEND`` (``torch`` default).  Any non-default
        kind is delegated to
        :func:`tensorlbm.ai.service_backends.make_backend` with
        ``artifact_path=backend_plan`` (falls back to
        ``TENSORLBM_DRAG_BACKEND_PLAN``) and
        ``TENSORLBM_DRAG_BACKEND_FALLBACK`` for degrade-to-torch.  With the
        argument unset and the env unset, the backend is built exactly as
        before — the default serving path is unchanged.
        """
        ckpts = [load_checkpoint(p) for p in paths]
        from .service_backends import (
            ENV_BACKEND_FALLBACK,
            ENV_BACKEND_KIND,
            make_backend,
            resolve_backend_kind,
        )

        kind = resolve_backend_kind(backend_kind)
        if (
            kind == "torch"
            and backend_kind is None
            and not os.environ.get(ENV_BACKEND_KIND, "").strip()
        ):
            backend = ModelEnsembleBackend(ckpts, device=device)
        else:
            backend = make_backend(
                kind,
                ckpts=ckpts,
                device=device,
                artifact_path=backend_plan,
                fallback=os.environ.get(ENV_BACKEND_FALLBACK),
            )
        guard = EnvelopeMahalanobisGuardrail(guard_features, guard_names, **guard_kwargs)
        return cls(
            backend,
            guard,
            corpus_cache=corpus_cache,
            grid=grid,
            cache_re=cache_re,
            cache_designs=cache_designs,
            cache_cd=cache_cd,
            uq_temperature=uq_temperature,
            field_provider=field_provider,
        )

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | Path,
        *,
        arm: str = "C_full",
        fold: str = "loho::full",
        guard_features: np.ndarray | None = None,
        guard_names: Sequence[str] | None = None,
        grid: SuboffGrid | None = None,
        uq_temperature: float | str | None = None,
        **guard_kwargs: Any,
    ) -> DragSurrogateService:
        """Replay service from a B4 run directory (preds + cache).

        ``grid`` must be the grid the ``guard_features`` (default: the
        run's ``cache_v3.npz`` geo block, i.e. the production grid) were
        computed on — query geometry channels are evaluated on the service
        grid and must live in the same feature space as the guard fit.
        """
        backend = ReplayEnsembleBackend(run_dir, arm=arm, fold=fold)
        if guard_features is None:
            guard_features = default_guard_features(run_dir)
        guard = EnvelopeMahalanobisGuardrail(guard_features, guard_names, **guard_kwargs)
        return cls(backend, guard, grid=grid, uq_temperature=uq_temperature)

    # -- queries ---------------------------------------------------------------

    def condition_rows(
        self,
        hull_type: str,
        sail_scale: float,
        fin_scale: float,
        re_grid: np.ndarray,
        *,
        u_in: float | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """``(N, 8)`` condition_v3 rows + the geometry-channel values."""
        u_in = self.u_in_default if u_in is None else float(u_in)
        features = suboff_geometry_features(hull_type, sail_scale, fin_scale, grid=self.grid)
        geo = geometry_channels(features)
        re_grid = np.asarray(re_grid, dtype=np.float64)
        cond = condition_v3(
            re_grid,
            np.full(re_grid.shape, u_in),
            np.full(re_grid.shape, float(sail_scale)),
            np.full(re_grid.shape, float(fin_scale)),
            np.broadcast_to(geo, (re_grid.size, 4)),
        )
        return cond, {
            "log_aproj_ratio": float(geo[0]),
            "sail_frac": float(geo[1]),
            "fin_frac": float(geo[2]),
            "solid_frac": float(geo[3]),
        }

    def _serve_model_members(
        self,
        hull_type: str,
        sail_scale: float,
        fin_scale: float,
        re_arr: np.ndarray,
        u_in: float,
        fields: np.ndarray | None,
        field_point: int | None,
        sdf: np.ndarray | None,
        field_policy: str,
        cond: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resolve the reference field, then serve the member C_D matrix.

        Default ``field_policy="cache"``: exactly the pre-flag composition
        — :meth:`_resolve_field` followed by ``backend.predict(field,
        cond)`` — byte-identical to the service before the borrow flag.

        ``field_policy="field_borrow"`` changes ONE thing: when normal
        field resolution would raise :class:`BackendQueryError` (a new
        geometry absent from the attached cache) and the query carries
        ``sdf`` while the service carries a ``field_provider``, the field
        is BORROWED from the pool via
        :func:`tensorlbm.ai.field_borrow.borrow_serving_field` and the
        response carries the retrieval provenance
        (``info["field_borrow"]``, never silent).  Cached designs and
        caller-supplied ``fields`` keep the resolution above; a missing
        provider or query SDF re-raises the original error.

        Two-stage backends (``kind == "per-member-model"``,
        :class:`tensorlbm.ai.ckpt_bundle.PerMemberEnsembleBackend`) take
        the query SDF as a model input and serve the corpus-convention
        param cond (:func:`tensorlbm.ai.field_borrow.param_cond_rows`)
        instead of the 8-channel condition_v3 — the composition validated
        by the 2026-09-04 e2e LODO campaign; the guard verdict above stays
        on condition_v3 either way.
        """
        backend = self.backend
        assert isinstance(backend, ModelEnsembleBackend)  # narrowed for mypy
        try:
            field, field_info = self._resolve_field(
                hull_type,
                sail_scale,
                fin_scale,
                float(re_arr[0]),
                u_in,
                fields,
                field_point,
            )
        except BackendQueryError:
            if field_policy != FIELD_POLICY_BORROW or sdf is None or self.field_provider is None:
                raise
            borrowed = borrow_serving_field(self.field_provider, sdf)
            field, field_info = borrowed.fields, borrowed.info
        if backend.kind == "per-member-model":
            if sdf is None:
                raise BackendQueryError(
                    "two-stage (per-member) backends need the query geometry volume: "
                    "pass sdf= to DragSurrogateService.predict"
                )
            member_cond = param_cond_rows(re_arr, u_in, sail_scale, fin_scale, backend.cond_dim)
            member_cd = backend.predict(field, sdf, member_cond)  # type: ignore[call-arg]
        else:
            member_cd = backend.predict(field, cond)
        return member_cd, field_info

    def _resolve_field(
        self,
        hull_type: str,
        sail_scale: float,
        fin_scale: float,
        re_hint: float,
        u_in: float,
        fields: np.ndarray | None,
        field_point: int | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if fields is not None:
            arr = np.asarray(fields, dtype=np.float32)
            if arr.shape != (5, self.grid.ny, self.grid.nx):
                raise ValueError(
                    f"fields must be (5, {self.grid.ny}, {self.grid.nx}), got {arr.shape}"
                )
            return arr, {"field_source": "caller"}
        if self.corpus_cache is None:
            raise BackendQueryError(
                "model backend needs `fields` or a corpus field cache to resolve the "
                "reference mid-plane field"
            )
        if field_point is not None:
            row = int(field_point)
            if not 0 <= row < self.corpus_cache.shape[0]:
                raise ValueError(f"field_point {row} out of cache range")
            return self.corpus_cache[row], {"field_source": "cache_row", "field_row": row}
        if self.cache_designs is None or self.cache_re is None:
            raise BackendQueryError("no design index attached to the field cache")
        best: tuple[float, int] | None = None
        for row, key in enumerate(self.cache_designs):
            if (
                key[0] == hull_type
                and key[1] == float(sail_scale)
                and key[2] == float(fin_scale)
                and abs(key[3] - u_in) <= 1e-12
            ):
                d = abs(
                    math.log10(max(self.cache_re[row], 1e-12)) - math.log10(max(re_hint, 1e-12))
                )
                if best is None or d < best[0]:
                    best = (d, row)
        if best is None:
            raise BackendQueryError(
                f"design ({hull_type}, {sail_scale}, {fin_scale}) not in the attached "
                f"field cache; pass fields= explicitly"
            )
        return self.corpus_cache[best[1]], {
            "field_source": "cache_design_nearest_re",
            "field_row": int(best[1]),
            "field_re": float(self.cache_re[best[1]]),
        }

    def _quad3_fallback(
        self,
        hull_type: str,
        sail_scale: float,
        fin_scale: float,
        u_in: float,
        re_arr: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Route per-point quad3 measured-curve serving (see :meth:`predict`).

        Routing order (first failure wins, the network value is kept and
        the reason recorded):

        1. no measured curve attached (``cache_designs`` / ``cache_re`` /
           ``cache_cd``) -> ``no_measured_curve_cache``;
        2. every query Re inside ``[min(cache_re), max(cache_re)]``
           (boundary values count as inside) -> ``re_inside_corpus_window``;
        3. no cached rows with the exact design key (hull string equal,
           sail/fin float-equal, ``|u_in - key| <= 1e-12`` — the
           ``_resolve_field`` convention) -> ``no_exact_design_match``;
        4. fewer than 3 cached rows -> ``insufficient_cached_rows``;
        5. degenerate nearest-3 (duplicate cached log10-Re) per point ->
           that point keeps the network value (``declined_points``).

        Applied points get ``cd = quad3`` and ``lo = hi = nan`` (no member
        band exists for a measured curve).  ``std = cd * quad3_loo_std``
        when the LOO rms exists — NOT rescaled by ``uq_temperature`` (that
        knob is calibrated on the deep-ensemble sigma of the network path,
        a different estimator).  When the LOO rms is unavailable (fewer
        than 4 cached rows, or duplicate cached log10-Re levels degenerating
        a leave-one-out pseudo-fit — the measured pool381 curve hits the
        latter), the point KEEPS the network ensemble std already computed
        for it — never ``nan`` — and ``info`` records why: ``std_source``
        plus, in the degenerate case, ``quad3_loo_degenerate = True`` and
        ``quad3_loo_duplicate_re_rows`` (cached rows minus distinct log10-Re
        levels).
        """
        info: dict[str, Any] = {
            "name": RE_POLICY_QUAD3_FALLBACK,
            "method": "network",
            "n_quad3_points": 0,
            "quad3_mask": [False] * re_arr.size,
        }
        if self.cache_designs is None or self.cache_re is None or self.cache_cd is None:
            info["reason"] = "no_measured_curve_cache"
            return mean, std, lo, hi, info
        window = (float(self.cache_re.min()), float(self.cache_re.max()))
        info["corpus_re_window"] = list(window)
        outside = (re_arr < window[0]) | (re_arr > window[1])
        if not outside.any():
            info["reason"] = "re_inside_corpus_window"
            return mean, std, lo, hi, info
        assert self.cache_designs is not None  # narrowed for mypy below
        rows = np.array(
            [
                r
                for r, key in enumerate(self.cache_designs)
                if key[0] == hull_type
                and key[1] == float(sail_scale)
                and key[2] == float(fin_scale)
                and abs(key[3] - u_in) <= 1e-12
            ],
            dtype=np.int64,
        )
        info["n_cached_rows"] = int(rows.size)
        if rows.size == 0:
            info["reason"] = "no_exact_design_match"
            return mean, std, lo, hi, info
        if rows.size < 3:
            info["reason"] = "insufficient_cached_rows"
            return mean, std, lo, hi, info
        order = np.argsort(self.cache_re[rows], kind="stable")
        curve_re = np.asarray(self.cache_re[rows][order], dtype=np.float64)
        curve_cd = np.asarray(self.cache_cd[rows][order], dtype=np.float64)
        info["cached_re"] = curve_re.tolist()
        loo_rel = quad3_loo_std(curve_re, curve_cd)
        info["loo_rel_rms"] = loo_rel
        if loo_rel is not None:
            info["std_source"] = "quad3_loo_relative_rms_times_cd"
        elif curve_re.size < 4:
            info["std_source"] = "network_ensemble_std_loo_needs_4_cached_rows"
        else:
            # >= 4 cached rows yet no LOO rms: duplicate cached log10-Re
            # levels degenerated a leave-one-out pseudo-fit.  The old flat
            # label "unavailable_fewer_than_4_cached_rows" was wrong here
            # (the pool381 curve has 157 rows) — record the true cause.
            n_dup = int(curve_re.size - np.unique(np.log10(curve_re)).size)
            info["std_source"] = "network_ensemble_std_loo_degenerate"
            info["quad3_loo_degenerate"] = True
            info["quad3_loo_duplicate_re_rows"] = n_dup
        mean = mean.copy()
        std = std.copy()
        lo = lo.copy()
        hi = hi.copy()
        mask = np.zeros(re_arr.size, dtype=bool)
        chosen: list[list[float] | None] = [None] * re_arr.size
        declined: list[str] = []
        for i in np.nonzero(outside)[0]:
            out = quad3_nearest3(curve_re, curve_cd, float(re_arr[i]))
            if out is None:
                declined.append(f"re={float(re_arr[i]):.6g}: duplicate log10 Re in nearest 3")
                continue
            value, sel_re = out
            mask[i] = True
            mean[i] = value
            if loo_rel is not None:
                std[i] = value * loo_rel
            # LOO rms unavailable (see std_source above): keep the network
            # ensemble std already computed for this point — never NaN.
            lo[i] = np.nan
            hi[i] = np.nan
            chosen[i] = [float(v) for v in sel_re]
        info["quad3_mask"] = mask.tolist()
        info["n_quad3_points"] = int(mask.sum())
        info["nearest_cached_re"] = chosen
        if declined:
            info["declined_points"] = declined
        if mask.any():
            info["method"] = RE_POLICY_QUAD3_FALLBACK
        else:
            info["reason"] = "degenerate_nearest3"
        return mean, std, lo, hi, info

    def predict(
        self,
        hull_type: str,
        sail_scale: float,
        fin_scale: float,
        re_grid: Sequence[float] | np.ndarray,
        *,
        u_in: float | None = None,
        fields: np.ndarray | None = None,
        field_point: int | None = None,
        re_policy: str | None = None,
        sdf: np.ndarray | None = None,
        field_policy: str | None = None,
    ) -> DragCurveResult:
        """Serve one design swept over ``re_grid`` with UQ + guard verdict.

        The guard verdict is always computed and attached; a ``reject``
        does not suppress the numbers — callers (e.g. the HTTP layer)
        decide how to present flagged results.

        ``re_policy`` (opt-in, default :data:`RE_POLICY_NETWORK` = the
        pre-flag behaviour, byte-identical including ``info``):
        ``"quad3_fallback"`` serves, per query point, the quad3
        measured-curve value (:func:`quad3_nearest3`) instead of the
        network when (a) the query design exactly matches a cached
        ``(hull, sail, fin, u_in)`` key with >= 3 cached Re rows AND
        (b) the query Re lies strictly outside the corpus Re window
        ``[min(cache_re), max(cache_re)]`` of the attached corpus.
        In-window points, unmatched designs and missing caches keep the
        network value; ``info["re_policy"]`` records the routing
        (method / reason / chosen cached Re / n_cached_rows).  The guard
        verdict is NOT softened: an out-of-window Re still flags
        ``reject`` on the ``log10_re`` envelope — the number is served
        AND flagged, mirroring the always-compute-and-attach semantics.

        ``sdf`` / ``field_policy`` (opt-in, default
        :data:`FIELD_POLICY_CACHE` = the pre-flag behaviour,
        byte-identical including ``info``): ``field_policy="field_borrow"``
        serves a query whose reference field cannot be resolved from the
        caller (``fields=``) or the attached cache by borrowing an
        in-manifold field from ``field_provider``, keyed on the query SDF
        (``sdf=``) — provenance and guard flags land in
        ``info["field_borrow"]``; a failed guard still serves, flagged and
        warning-logged (``docs/field_borrow_20260904.md``).  Two-stage
        (per-member) backends always take ``sdf=`` as a model input and
        serve the corpus-convention param cond instead of condition_v3
        (the composition validated by the 2026-09-04 e2e campaign); the
        guard stays on condition_v3 either way.
        """
        re_arr = np.asarray(re_grid, dtype=np.float64).ravel()
        if re_arr.size == 0:
            raise ValueError("re_grid must be non-empty")
        if not np.isfinite(re_arr).all() or not (re_arr > 0).all():
            raise ValueError("re_grid entries must be finite and positive")
        u_in = self.u_in_default if u_in is None else float(u_in)
        policy = resolve_re_policy(re_policy)
        field_pol = resolve_field_policy(field_policy)
        cond, geo_vals = self.condition_rows(hull_type, sail_scale, fin_scale, re_arr, u_in=u_in)
        verdict = self.guard.check(cond)
        info: dict[str, Any] = {"geometry": geo_vals}

        if isinstance(self.backend, ModelEnsembleBackend):
            member_cd, field_info = self._serve_model_members(
                hull_type,
                float(sail_scale),
                float(fin_scale),
                re_arr,
                u_in,
                fields,
                field_point,
                sdf,
                field_pol,
                cond,
            )
            info.update(field_info)
        else:
            member_cd, rep_info = self.backend.predict(
                hull_type, float(sail_scale), float(fin_scale), re_arr, u_in=u_in
            )
            info.update(rep_info)

        mean, std, lo, hi = ensemble_stats(member_cd)
        t = self.uq_temperature
        if t != 1.0:
            # #251 calibration knob: rescale the *reported* sigma only.  The
            # guard verdict (already computed above) and the member min-max
            # band stay raw, and at the default T = 1.0 this branch is dead
            # so the served arrays are bit-identical to the pre-knob service.
            std = std * t
        info["uq_temperature"] = t
        if policy != RE_POLICY_NETWORK:
            # Opt-in measured-curve fallback: overwrites ONLY the
            # out-of-window exact-design points below.  The network arrays
            # above (and the default call path without the flag) are
            # untouched — at default OFF this whole block is dead code.
            mean, std, lo, hi, policy_info = self._quad3_fallback(
                hull_type, float(sail_scale), float(fin_scale), u_in, re_arr, mean, std, lo, hi
            )
            info["re_policy"] = policy_info
        return DragCurveResult(
            re=re_arr,
            cd=mean,
            lo=lo,
            hi=hi,
            std=std,
            guard=verdict,
            hull_type=hull_type,
            sail_scale=float(sail_scale),
            fin_scale=float(fin_scale),
            u_in=u_in,
            backend=self.backend.kind,
            members=tuple(self.backend.member_labels()),
            info=info,
        )


def _find_corpus_cache(run_dir: Path) -> Path:
    """Locate the point cache of a B4 run dir (``cache.npz`` or v4 layout)."""
    for name in ("cache.npz", "cache_v4.npz"):
        p = run_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"no corpus cache (cache.npz / cache_v4.npz) under {run_dir}")


def _find_geo(run_dir: Path, cache: Any, cache_path: Path) -> np.ndarray:
    """Geometry-channel block: own file (v3 layout) or inside the cache (v4)."""
    if "geo" in cache.files:
        return np.asarray(cache["geo"], dtype=np.float64)
    for name in (cache_path.stem.replace("cache", "cache_v3") + ".npz", "cache_v3.npz"):
        p = run_dir / name
        if p.is_file():
            return np.asarray(np.load(p)["geo"], dtype=np.float64)
    raise FileNotFoundError(f"no geo block (cache_v3.npz / cache geo key) under {run_dir}")


def default_guard_features(run_dir: str | Path) -> np.ndarray:
    """Condition_v3 matrix of a run-dir corpus.

    Works with both layouts: ``cache.npz`` + ``cache_v3.npz`` (v3 runs)
    and the combined ``cache_v4.npz`` (v4 run, keys include ``geo``).
    Rebuilds the exact 8-channel training condition from the archived
    caches; used to fit the default manual-space guard.
    """
    run_dir = Path(run_dir)
    cache_path = _find_corpus_cache(run_dir)
    cache = np.load(cache_path)
    geo = _find_geo(run_dir, cache, cache_path)
    return condition_v3(cache["re"], cache["uin"], cache["sail"], cache["fin"], geo)


@dataclass(frozen=True)
class CorpusIndex:
    """Everything the service needs from a run-dir corpus.

    ``fields`` is the ``(N, 5, ny, nx)`` mid-plane cache, ``cond`` the
    ``(N, 8)`` condition_v3 matrix (guard fit + design lookup), and
    ``designs`` the per-row ``(hull, sail, fin, u_in)`` keys.
    """

    fields: np.ndarray
    re: np.ndarray
    designs: tuple[tuple[str, float, float, float], ...]
    cond: np.ndarray
    cd: np.ndarray | None = None


def load_corpus_index(run_dir: str | Path) -> CorpusIndex:
    """Load the serving-side view of a B4 run-dir corpus."""
    run_dir = Path(run_dir)
    cache_path = _find_corpus_cache(run_dir)
    z = np.load(cache_path)
    geo = _find_geo(run_dir, z, cache_path)
    cond = condition_v3(z["re"], z["uin"], z["sail"], z["fin"], geo)
    designs = tuple(
        (HULL_ORDER[int(h)], float(s), float(f), float(u))
        for h, s, f, u in zip(z["hull"], z["sail"], z["fin"], z["uin"])
    )
    return CorpusIndex(
        fields=np.asarray(z["x"], dtype=np.float32),
        re=np.asarray(z["re"], dtype=np.float64),
        designs=designs,
        cond=cond,
        cd=np.asarray(z["cd"], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# UQ + guard calibration helpers (offline analysis over archived preds)
# ---------------------------------------------------------------------------


def ensemble_picp(member_cd: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of points whose truth lies inside the member min-max band."""
    member_cd = np.asarray(member_cd, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if member_cd.ndim != 2 or member_cd.shape[1] != truth.size:
        raise ValueError("member_cd must be (M, N) aligned with truth (N,)")
    lo = member_cd.min(axis=0)
    hi = member_cd.max(axis=0)
    return float(np.mean((truth >= lo) & (truth <= hi)))


def error_std_spearman(std: np.ndarray, abs_rel_error: np.ndarray) -> float:
    """Spearman correlation between ensemble std and absolute error."""
    std = np.asarray(std, dtype=np.float64)
    err = np.asarray(abs_rel_error, dtype=np.float64)
    if std.size != err.size or std.size < 2:
        raise ValueError("std and abs_rel_error must be same-length (N>=2)")
    ra = np.argsort(np.argsort(std)).astype(np.float64)
    rb = np.argsort(np.argsort(err)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float(ra @ ra) * float(rb @ rb))
    if denom == 0.0:
        return 0.0
    return float(ra @ rb / denom)


@dataclass(frozen=True)
class CalibrationRow:
    """One point of the guard-threshold sweep."""

    threshold: float
    n_flagged: int
    flagged_frac: float
    n_large: int
    large_captured: int
    capture_rate: float
    precision: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "n_flagged": self.n_flagged,
            "flagged_frac": self.flagged_frac,
            "n_large": self.n_large,
            "large_captured": self.large_captured,
            "capture_rate": self.capture_rate,
            "precision": self.precision,
        }


def guard_threshold_sweep(
    scores: np.ndarray,
    errors: np.ndarray,
    *,
    large_error: float = 0.10,
    thresholds: np.ndarray | None = None,
) -> list[CalibrationRow]:
    """Flag-coverage vs large-error-capture table over guard thresholds.

    Parameters
    ----------
    scores:
        Guard severity per evaluated point (Mahalanobis distance in the
        default guard), shape ``(N,)``.
    errors:
        Relative error per point (``|pred-true|/true``, sign ignored).
    large_error:
        Error level counted as "large" (default 0.10 — the phase-1 bar).
    thresholds:
        Optional explicit threshold grid; defaults to score quantiles
        plus the built-in review/reject defaults (3.0 / 6.0).
    """
    scores = np.asarray(scores, dtype=np.float64)
    errors = np.abs(np.asarray(errors, dtype=np.float64))
    if scores.shape != errors.shape or scores.size == 0:
        raise ValueError("scores and errors must be same-length, non-empty")
    large = errors >= large_error
    if thresholds is None:
        qs = np.quantile(scores, np.linspace(0.0, 1.0, 21))
        thresholds = np.unique(np.concatenate([qs, np.array([3.0, 6.0])]))
    rows: list[CalibrationRow] = []
    n = scores.size
    n_large = int(large.sum())
    for t in thresholds:
        flagged = scores > float(t)
        n_flagged = int(flagged.sum())
        captured = int((flagged & large).sum())
        rows.append(
            CalibrationRow(
                threshold=float(t),
                n_flagged=n_flagged,
                flagged_frac=n_flagged / n,
                n_large=n_large,
                large_captured=captured,
                capture_rate=captured / n_large if n_large else float("nan"),
                precision=captured / n_flagged if n_flagged else float("nan"),
            )
        )
    return rows


def choose_threshold(
    rows: Sequence[CalibrationRow], *, target_capture: float = 0.8
) -> CalibrationRow:
    """Smallest-coverage row reaching ``target_capture`` (last row otherwise).

    Among rows whose large-error capture rate meets the target, take the
    one with the smallest flagged fraction (fewest false alarms); ties
    break on the tighter threshold.
    """
    eligible = [r for r in rows if r.n_large > 0 and r.capture_rate >= target_capture]
    if not eligible:
        return rows[-1]
    return min(eligible, key=lambda r: (r.flagged_frac, r.threshold))


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


class SpectralConv2dMatmul(nn.Module):
    """Real-arithmetic drop-in for :class:`tensorlbm.ai.fno.SpectralConv2d`.

    Computes the identical operator — low-frequency corner of
    ``rfft2(x, norm='ortho')``, complex weight multiply, ``irfft2`` back —
    using only real dense matmuls with precomputed DFT cosine/sine bases.
    With ``A_y[k, y] = exp(-2*pi*i*k*y/ny)/sqrt(ny)`` and
    ``B_x[x, k] = exp(-2*pi*i*k*x/nx)/sqrt(nx)``::

        X_r + i*X_i = A_y @ x @ B_x          (corner of rfft2, ortho)
        Y           = complex weight multiply of X
        g           = (Cy^T @ P - Sy^T @ Q) / sqrt(ny*nx)

    where ``P = Y_r @ (mu*Cx) - Y_i @ (mu*Sx)``,
    ``Q = Y_i @ (mu*Cx) + Y_r @ (mu*Sx)``, ``Cx/Sx`` are the inverse-DFT
    cosine/sine bases and ``mu`` folds the Hermitian mirror columns:
    ``mu = 2`` for interior columns ``0 < kx < nx/2`` and ``1`` for the
    self-conjugate columns ``kx = 0`` / ``kx = nx/2`` (each half-spectrum
    row of a self-conjugate column contributes its real part once —
    pinned empirically against ``torch.fft.irfft2``).

    Exists because ``torch.fft.rfft2`` + complex ``einsum`` block the
    ONNX exporter; parity with the fft module is pinned numerically by
    the test-suite (float64 max rel < 1e-12).
    """

    _MU_KX0 = 1.0

    # Declared buffer types (nn.Module resolves these via __getattr__ at
    # runtime; the annotations pin them for static typing).
    ay_r: torch.Tensor
    ay_i: torch.Tensor
    bx_r: torch.Tensor
    bx_i: torch.Tensor
    cy: torch.Tensor
    sy: torch.Tensor
    cx_mu: torch.Tensor
    sx_mu: torch.Tensor
    scale: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_y: int,
        modes_x: int,
        *,
        ny: int,
        nx: int,
    ) -> None:
        super().__init__()
        if modes_y > ny or modes_x > nx // 2 + 1:
            raise ValueError(f"modes ({modes_y}, {modes_x}) exceed rfft2 corner for ({ny}, {nx})")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_y = modes_y
        self.modes_x = modes_x
        self.ny = ny
        self.nx = nx
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_y, modes_x, 2)
        )
        f = 2.0 * math.pi
        ky = torch.arange(modes_y, dtype=torch.float64).view(-1, 1)
        yy = torch.arange(ny, dtype=torch.float64).view(1, -1)
        kx = torch.arange(modes_x, dtype=torch.float64).view(-1, 1)
        xx = torch.arange(nx, dtype=torch.float64).view(1, -1)
        ay = torch.exp(-1j * f * ky * yy / ny) / math.sqrt(float(ny))  # (my, ny)
        bx = torch.exp(-1j * f * xx.T * kx.T / nx) / math.sqrt(float(nx))  # (nx, mx)
        cy = torch.cos(f * ky * yy / ny)  # (my, ny), inverse basis
        sy = torch.sin(f * ky * yy / ny)
        cx = torch.cos(f * kx * xx / nx)  # (mx, nx)
        sx = torch.sin(f * kx * xx / nx)
        mirror = torch.full((modes_x,), 2.0, dtype=torch.float64)
        mirror[0] = self._MU_KX0
        if modes_x == nx // 2 + 1:
            mirror[-1] = 1.0
        self.register_buffer("ay_r", ay.real.float())
        self.register_buffer("ay_i", ay.imag.float())
        self.register_buffer("bx_r", bx.real.float())
        self.register_buffer("bx_i", bx.imag.float())
        self.register_buffer("cy", cy.float())
        self.register_buffer("sy", sy.float())
        self.register_buffer("cx_mu", (mirror.view(-1, 1) * cx).float())
        self.register_buffer("sx_mu", (mirror.view(-1, 1) * sx).float())
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(float(ny * nx))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, ny, nx = x.shape
        if ny != self.ny or nx != self.nx:
            raise ValueError(f"expected ({self.ny}, {self.nx}) field, got ({ny}, {nx})")
        xr = torch.einsum("kj,bcjx->bckx", self.ay_r, x)  # (B, C, my, nx)
        xi = torch.einsum("kj,bcjx->bckx", self.ay_i, x)
        x_r = xr @ self.bx_r - xi @ self.bx_i  # (B, C, my, mx) corner of rfft2
        x_i = xr @ self.bx_i + xi @ self.bx_r
        w_r = self.weight[..., 0]
        w_i = self.weight[..., 1]
        y_r = torch.einsum("bimn,iomn->bomn", x_r, w_r) - torch.einsum("bimn,iomn->bomn", x_i, w_i)
        y_i = torch.einsum("bimn,iomn->bomn", x_r, w_i) + torch.einsum("bimn,iomn->bomn", x_i, w_r)
        p = y_r @ self.cx_mu - y_i @ self.sx_mu  # (B, O, my, nx)
        q = y_i @ self.cx_mu + y_r @ self.sx_mu
        return self.scale * (torch.matmul(self.cy.t(), p) - torch.matmul(self.sy.t(), q))


def to_matmul_spectral(model: CondFNODrag, *, ny: int = 64, nx: int = 128) -> CondFNODrag:
    """Copy of ``model`` whose spectral layers use the matmul path.

    Weights and all non-spectral modules are copied verbatim; the returned
    model is a drop-in for ONNX export and is pinned to numerical parity
    with the fft original by the test-suite (``test_spectral_matmul_parity``).
    """
    from .fno import SpectralConv2d

    out = copy.deepcopy(model)
    out.eval()
    with torch.no_grad():
        for i, spec in enumerate(out.spectral):
            if isinstance(spec, SpectralConv2dMatmul):
                continue
            if not isinstance(spec, SpectralConv2d):
                raise TypeError(f"spectral[{i}] is {type(spec).__name__}, expected SpectralConv2d")
            twin = SpectralConv2dMatmul(
                spec.in_channels,
                spec.out_channels,
                spec.modes_y,
                spec.modes_x,
                ny=ny,
                nx=nx,
            )
            twin.weight.copy_(spec.weight)
            out.spectral[i] = twin
    return out


def export_cond_fno_onnx(
    model: CondFNODrag,
    path: str | Path,
    *,
    ny: int = 64,
    nx: int = 128,
    opset: int = 17,
) -> dict[str, Any]:
    """Export a :class:`CondFNODrag` member to ONNX with honest reporting.

    Tries the plain module first — ``torch.fft.rfft2``/``irfft2`` and the
    complex einsum block the legacy exporter in current torch, and the
    recorded blocker string says so — then falls back to the
    :func:`to_matmul_spectral` real-arithmetic twin.  Both inputs carry a
    dynamic batch axis.  Validation with ``onnx``/``onnxruntime`` runs
    only when those optional packages are installed; absence is recorded,
    not papered over.
    """
    report: dict[str, Any] = {
        "path": None,
        "plain_export_ok": False,
        "plain_blocker": None,
        "matmul_export_ok": False,
        "matmul_blocker": None,
        "opset": int(opset),
        "checker": "skipped",
        "runtime_parity": "skipped",
    }
    lift = model.lift
    cond_in = model.cond_embed[0]
    assert isinstance(lift, nn.Conv2d) and isinstance(cond_in, nn.Linear)
    example_x = torch.randn(2, lift.in_channels, ny, nx)
    example_p = torch.randn(2, cond_in.in_features)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _export(module: nn.Module) -> None:
        # TracerWarnings are expected here (shape checks / default args in
        # forward); the exported contract is fixed-HxW with dynamic batch.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=getattr(torch.jit, "TracerWarning", UserWarning)
            )
            torch.onnx.export(
                module,
                (example_x, example_p),
                str(out_path),
                opset_version=int(opset),
                input_names=["field", "cond"],
                output_names=["log10_cd"],
                dynamic_axes={
                    "field": {0: "batch"},
                    "cond": {0: "batch"},
                    "log10_cd": {0: "batch"},
                },
                dynamo=False,
            )

    try:
        _export(model)
        report["plain_export_ok"] = True
        report["path"] = str(out_path.resolve())
        with torch.no_grad():
            ref = model(example_x, example_p)
        return _maybe_validate_onnx(
            out_path, report, example_x=example_x, example_p=example_p, ref=ref
        )
    except Exception as exc:  # noqa: BLE001 — the blocker text is the deliverable
        report["plain_blocker"] = f"{type(exc).__name__}: {exc}"

    twin = to_matmul_spectral(model, ny=ny, nx=nx)
    try:
        _export(twin)
        report["matmul_export_ok"] = True
        report["path"] = str(out_path.resolve())
        with torch.no_grad():
            ref = model(example_x, example_p)
            got = twin(example_x, example_p)
        report["matmul_parity_max_abs"] = float((ref - got).abs().max())
    except Exception as exc:  # noqa: BLE001
        report["matmul_blocker"] = f"{type(exc).__name__}: {exc}"
        return report
    return _maybe_validate_onnx(out_path, report, example_x=example_x, example_p=example_p, ref=got)


def _maybe_validate_onnx(
    out_path: Path,
    report: dict[str, Any],
    *,
    example_x: torch.Tensor | None = None,
    example_p: torch.Tensor | None = None,
    ref: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Best-effort onnx checker + onnxruntime parity, recorded honestly."""
    try:
        import onnx as _onnx

        _onnx.checker.check_model(_onnx.load(str(out_path)))
        report["checker"] = "ok"
    except ImportError:
        report["checker"] = "skipped (onnx package not installed)"
    except Exception as exc:  # noqa: BLE001
        report["checker"] = f"failed: {type(exc).__name__}: {exc}"
    try:
        import onnxruntime as _ort

        session = _ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        report["runtime_providers"] = list(_ort.get_available_providers())
        if example_x is not None and example_p is not None and ref is not None:
            out = session.run(None, {"field": example_x.numpy(), "cond": example_p.numpy()})[0]
            report["runtime_parity"] = (
                f"max_abs={float(np.abs(out - ref.detach().numpy()).max()):.3e}"
            )
        else:
            report["runtime_parity"] = "ran (no reference outputs supplied)"
    except ImportError:
        report["runtime_parity"] = "skipped (onnxruntime not installed)"
    except Exception as exc:  # noqa: BLE001
        report["runtime_parity"] = f"failed: {type(exc).__name__}: {exc}"
    return report
