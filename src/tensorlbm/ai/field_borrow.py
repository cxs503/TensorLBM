"""Provider-backed reference-field borrowing wired into the inference service.

Serving a NEW geometry through the drag surrogate historically required a
cached reference mid-plane field of THAT design — field resolution in
:mod:`tensorlbm.ai.inference_service` raises :class:`BackendQueryError`
for a design absent from the attached cache, so a new-geometry query
(which has an SDF, no LBM reference run) could not be served at all.

The evidence that this requirement is artificial, and that the wiring here
is safe, is machine-recorded:

- 2026-09-04 field-sensitivity study
  (``/nfs/wangxi/runs/l2_field_sensitivity_20260904/``): swapping the
  reference field for ANY in-manifold donor is invisible on the held-25
  ensemble MAPE (cond-nearest donor x0.998 of baseline, corpus mean x0.998,
  random donor x1.004); feeding a DIFFERENT geometry's SDF is catastrophic
  (x57.9 ts2 / x59.3 ts4) — the provider never returns a donor SDF; the
  only field failure mode is a grossly out-of-manifold field (all-zeros
  probe: x7.7 ts2 / x17.9 ts4), guarded by the relative-L2 in-manifold
  check of :class:`~tensorlbm.ai.field_provider.FieldProvider`.
- 2026-09-04 e2e LODO campaign
  (``/nfs/wangxi/runs/l2_e2e_validation_20260904/``): the full composition
  this module implements — regenerated target SDF + ``FieldProvider``
  (``sdf_near``) borrowed field + target cond, served through the frozen
  pm20260831 10-member ensembles — reproduced oracle-level accuracy for
  every design whose SDF regenerates bit-exact (121/122 designs; ts2
  macro MAPE 0.215 % borrowed vs 0.217 % oracle, x0.99 excluding the one
  vintage-offset design; pure field borrowing x0.99 of oracle overall; all
  5 LODO donors passed the guard at rel-L2 0.026-0.132 vs the 0.15
  threshold).

This module holds the composition so the service hook stays thin:

- :func:`borrow_serving_field` — one borrowed reference field plus the
  honest-provenance ``info`` block the response carries;
- :func:`param_cond_rows` — the corpus-convention condition rows of the
  two-stage arms (``[log10 re, log10 u_in]`` for ts2, plus ``[log10 sail,
  log10 fin]`` for ts4), the exact ``load_fam`` assembly of the e2e driver;
- :func:`borrow_conditioning` — both in one call (the e2e
  ``A_regen_borrowed`` arm) for callers outside the service;
- :data:`FIELD_POLICY_CACHE` / :data:`FIELD_POLICY_BORROW` +
  :func:`resolve_field_policy` — the opt-in service flag, following the
  ``re_policy`` convention of :mod:`tensorlbm.ai.inference_service`.

Honest-serving contract (PR #275 philosophy): a borrowed field is NEVER
silent.  The response carries strategy / donor / guard numbers; a failed
in-manifold guard (``guard_ok=False``) still serves — flagged in the
provenance and warning-logged — never raises and never silently swaps
strategy.  Out-of-manifold fields are the known failure mode (x7.7), so
the flag IS the safety story.

Regeneration caveat: the e2e evidence reused the corpus's own CAD
parameterization (``suboff_cad`` + ``geom_encoder.sdf_volume``), so it
validates path consistency, not voxelizer independence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .field_provider import FieldProvider

__all__ = [
    "CORPUS_COND_KEYS",
    "E2E_PATH",
    "FIELD_POLICY_BORROW",
    "FIELD_POLICY_CACHE",
    "BorrowedConditioning",
    "BorrowedServingField",
    "borrow_conditioning",
    "borrow_serving_field",
    "param_cond_rows",
    "resolve_field_policy",
]

#: Machine-readable evidence this wiring implements (read-only).
E2E_PATH = "/nfs/wangxi/runs/l2_e2e_validation_20260904"

#: Corpus condition-column order (``load_fam`` of the e2e driver): ts2
#: serves columns 0:2, ts4 0:4 — :func:`param_cond_rows` slices this order,
#: never reorders it.
CORPUS_COND_KEYS = ("re", "uin", "sail", "fin")

#: Default field-serving policy of the drag service: resolve the reference
#: field from the caller (``fields=``) or the attached corpus cache only —
#: the pre-flag behaviour, byte-identical.
FIELD_POLICY_CACHE = "cache"

#: Opt-in field-serving policy: a model-backend query that cannot resolve a
#: reference field (new geometry absent from the attached cache) borrows an
#: in-manifold field from the attached
#: :class:`~tensorlbm.ai.field_provider.FieldProvider`, keyed on the query
#: SDF (e2e LODO validation 2026-09-04; ``docs/field_borrow_20260904.md``).
FIELD_POLICY_BORROW = "field_borrow"

_log = logging.getLogger(__name__)


def resolve_field_policy(field_policy: str | None = None) -> str:
    """Validate a field-serving policy name (fail loud, no silent aliases).

    Mirrors :func:`tensorlbm.ai.inference_service.resolve_re_policy`:
    ``None`` counts as :data:`FIELD_POLICY_CACHE` so the default call path
    never touches the borrow logic, and unknown names raise — a typoed
    policy must not silently degrade to the cached-field path.
    """
    name = FIELD_POLICY_CACHE if field_policy is None else str(field_policy)
    if name not in (FIELD_POLICY_CACHE, FIELD_POLICY_BORROW):
        raise ValueError(
            f"unknown field_policy {field_policy!r}; expected "
            f"{FIELD_POLICY_CACHE!r} or {FIELD_POLICY_BORROW!r}"
        )
    return name


@dataclass(frozen=True)
class BorrowedServingField:
    """One borrowed reference field plus its service ``info`` block.

    ``info`` is what the service merges into the response:
    ``field_source="field_borrow"`` and a nested ``field_borrow``
    provenance dict — the borrow is always visible, never silent.
    """

    fields: np.ndarray
    info: dict[str, Any]


def borrow_serving_field(
    provider: FieldProvider,
    target_sdf: np.ndarray,
    *,
    strategy: str = "sdf_near",
) -> BorrowedServingField:
    """Borrow one reference field for a new-geometry service query.

    Thin honest wrapper over :meth:`FieldProvider.borrow`: the TARGET keeps
    its own SDF (the provider never returns a donor SDF — the x57.9
    measured mistake), the response carries the full retrieval provenance,
    and a failed in-manifold guard SERVES, flagged and warning-logged —
    not raised, not re-strategied (out-of-manifold fields are the known
    failure mode, x7.7 ts2 held MAPE, so the flag IS the safety story).
    """
    borrowed = provider.borrow(target_sdf=target_sdf, strategy=strategy)
    prov = borrowed.provenance
    info: dict[str, Any] = {
        "field_source": "field_borrow",
        "field_borrow": {
            "strategy": borrowed.strategy,
            "donor_index": borrowed.donor_index,
            "donor_key": prov.get("donor_key"),
            "distance": borrowed.distance,
            "guard_ok": bool(borrowed.guard_ok),
            "guard_rel_l2": float(borrowed.guard_rel_l2),
            "guard_threshold": float(borrowed.guard_threshold),
            "pool_size": int(prov["pool_size"]),
            "e2e": E2E_PATH,
        },
    }
    if not borrowed.guard_ok:
        _log.warning(
            "field_borrow guard FAILED (strategy=%s donor_index=%s): "
            "guard_rel_l2=%.4f > threshold=%.4f — out-of-manifold reference "
            "field (known failure mode, x7.7 ts2 held MAPE); serving anyway, "
            "flags in info['field_borrow']",
            borrowed.strategy,
            borrowed.donor_index,
            borrowed.guard_rel_l2,
            borrowed.guard_threshold,
        )
    return BorrowedServingField(fields=borrowed.fields, info=info)


def param_cond_rows(
    re_grid: np.ndarray,
    u_in: float,
    sail_scale: float,
    fin_scale: float,
    param_dim: int,
) -> np.ndarray:
    """Corpus-convention condition rows for the two-stage arms (ts2/ts4).

    Exactly the ``load_fam`` assembly of the e2e driver: one row per query
    Reynolds number, ``[log10(re), log10(u_in), log10(sail), log10(fin)]``,
    sliced to the arm's param width (ts2 -> columns 0:2, ts4 -> 0:4) — the
    columns the frozen pm20260831 bodies were trained on.  The per-design
    scalars (``u_in`` / ``sail_scale`` / ``fin_scale``) are broadcast over
    the Re grid, matching a one-design serving sweep.
    """
    re_arr = np.asarray(re_grid, dtype=np.float64).ravel()
    if re_arr.size == 0:
        raise ValueError("re_grid must be non-empty")
    scalars = np.array([float(u_in), float(sail_scale), float(fin_scale)], dtype=np.float64)
    if not np.isfinite(scalars).all() or not (scalars > 0.0).all():
        raise ValueError("u_in / sail_scale / fin_scale must be finite and positive (log10 cond)")
    if not np.isfinite(re_arr).all() or not (re_arr > 0.0).all():
        raise ValueError("re_grid entries must be finite and positive (log10 cond)")
    if (
        not isinstance(param_dim, int)
        or isinstance(param_dim, bool)
        or not 1 <= param_dim <= len(CORPUS_COND_KEYS)
    ):
        raise ValueError(
            f"param_dim must be an int in 1..{len(CORPUS_COND_KEYS)} (ts2=2, ts4=4), "
            f"got {param_dim!r}"
        )
    full = np.stack(
        [re_arr, *np.broadcast_to(scalars[:, None], (3, re_arr.size))],
        axis=1,
    )
    return np.log10(full)[:, :param_dim]


@dataclass(frozen=True)
class BorrowedConditioning:
    """Assembled two-stage inputs for one new-geometry query + provenance.

    The e2e ``A_regen_borrowed`` composition: ``fields`` is the borrowed
    5-channel reference field, ``cond`` the ``(N, param_dim)`` corpus
    convention rows.  The TARGET's own SDF (the third model input) stays
    with the caller — the provider never returns a donor SDF by design, so
    this dataclass cannot accidentally carry one either.
    """

    fields: np.ndarray
    cond: np.ndarray
    provenance: dict[str, Any]


def borrow_conditioning(
    target_sdf: np.ndarray,
    re_grid: np.ndarray,
    u_in: float,
    sail_scale: float,
    fin_scale: float,
    provider: FieldProvider,
    *,
    strategy: str = "sdf_near",
    param_dim: int = 4,
) -> BorrowedConditioning:
    """One-call new-geometry composition (mirror of the e2e driver arm A).

    Given the target geometry's own SDF and the query Re sweep, return the
    borrowed reference field, the corpus-convention param cond rows and the
    retrieval provenance — feed them to a two-stage ensemble backend as
    ``backend.predict(fields, target_sdf, cond)``.  ``param_dim`` selects
    the arm (2 = ts2, 4 = ts4).
    """
    borrowed = borrow_serving_field(provider, target_sdf, strategy=strategy)
    cond = param_cond_rows(re_grid, u_in, sail_scale, fin_scale, param_dim)
    return BorrowedConditioning(
        fields=borrowed.fields, cond=cond, provenance=borrowed.info["field_borrow"]
    )
