"""Machine-checkable acceptance criteria for the TensorLBM validation matrix.

The matrix deliberately separates three things that are usually mixed together
in a benchmark README:

1. **the measured error** of a physical quantity against a frozen reference
   (``pct_error`` / ``judge_metric``);
2. **the convergence trend** over the declared level series (two grids for the
   CI tier, more when the case supports it) — a benchmark is only accepted if
   the error neither drifts nor stalls (``convergence_trend``);
3. **the three-level status** ``passed`` / ``failed`` / ``pending`` with an
   evidence chain (``judge_case`` returns the status, never a bare boolean).

Everything here is pure (no I/O), so the rules can be unit-tested and the same
code path is used for frozen artifacts and freshly executed runners.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_PENDING = "pending"
STATUSES = (STATUS_PASSED, STATUS_FAILED, STATUS_PENDING)

DEFAULT_ERROR_TOL_PCT = 3.0
DEFAULT_SPAN_TOL_PCT = 3.0


class CriterionError(ValueError):
    """Raised when a criterion is specified or evaluated inconsistently."""


def pct_error(value: float, reference: float) -> float:
    """Signed relative error in percent, ``(value - reference)/|reference|``."""
    value = float(value)
    reference = float(reference)
    if reference == 0.0:
        raise CriterionError("percentage error against a zero reference is undefined")
    return (value - reference) / abs(reference) * 100.0


def judge_metric(
    value: float,
    reference: float | None,
    tol_pct: float = DEFAULT_ERROR_TOL_PCT,
    *,
    kind: str = "value",
) -> dict:
    """Judge one metric at one level.

    ``kind="value"``           : ``value`` is a physical quantity, error is
                                 measured against ``reference``.
    ``kind="error_pct"``       : ``value`` is *already* the signed percentage
                                 error against the case anchor (this is how
                                 frozen result.json files record it); the sign
                                 is preserved so the table shows over/under
                                 prediction, the gate uses ``|err|``.
    """
    if kind == "value":
        if reference is None:
            raise CriterionError("kind='value' requires a reference value")
        err = pct_error(value, reference)
    elif kind == "error_pct":
        err = float(value)
    else:  # pragma: no cover - defensive
        raise CriterionError(f"unknown metric kind: {kind!r}")
    passed = abs(err) <= abs(float(tol_pct))
    return {
        "value": float(value),
        "reference": None if reference is None else float(reference),
        "err_pct": err,
        "tol_pct": abs(float(tol_pct)),
        "kind": kind,
        "pass": bool(passed),
    }


def convergence_trend(
    errors_pct: Sequence[float],
    *,
    max_span_pct: float = DEFAULT_SPAN_TOL_PCT,
    require_monotone: bool | None = None,
    min_levels: int = 2,
) -> dict:
    """Check the level-to-level trend of a metric series (coarse -> fine).

    ``max_span_pct``  : max(abs(err)) - min(abs(err)) must stay inside this
                        band, i.e. the answer does not move with resolution.
    ``require_monotone``: |err| must not grow from one level to the next.
                        Defaults to ``True`` for >= 3 levels, ``False`` for two
                        (a two-level series cannot distinguish noise from a
                        trend, and pretending otherwise would over-claim).
    """
    errs = [abs(float(e)) for e in errors_pct]
    if len(errs) < min_levels:
        raise CriterionError(
            f"convergence needs at least {min_levels} levels, got {len(errs)}"
        )
    if require_monotone is None:
        require_monotone = len(errs) >= 3
    span = max(errs) - min(errs)
    monotone = all(b <= a + 1e-12 for a, b in zip(errs, errs[1:]))
    span_ok = span <= abs(float(max_span_pct))
    trend_ok = (not require_monotone) or monotone
    return {
        "errors_abs_pct": errs,
        "max_span_pct": abs(float(max_span_pct)),
        "span_pct": span,
        "monotone": monotone,
        "require_monotone": bool(require_monotone),
        "span_ok": bool(span_ok),
        "trend_ok": bool(trend_ok),
        "pass": bool(span_ok and trend_ok),
    }


def _levels_of(metric_name: str, values: Sequence[float], spec: Mapping) -> dict:
    """Per-level judgments for one metric.

    ``tol_pct_by_level`` allows a case to gate only the resolutions its own
    pre-registration gates (e.g. "finest tier within 3% AND monotone"). An
    entry of ``None`` marks that level as *informational*: it is reported and
    used by the convergence check, but it cannot fail the metric on its own.
    """
    tol = float(spec.get("tol_pct", DEFAULT_ERROR_TOL_PCT))
    tol_by_level = spec.get("tol_pct_by_level")
    kind = spec.get("kind", "value")
    reference = spec.get("reference")
    rows = []
    for i, (level, v) in enumerate(zip(spec["levels"], values)):
        tol_i = tol if tol_by_level is None else tol_by_level[i]
        if tol_i is None:
            judged = judge_metric(v, reference, 0.0, kind=kind)
            judged["gated"] = False
        else:
            judged = judge_metric(v, reference, float(tol_i), kind=kind)
            judged["gated"] = True
        rows.append({"level": str(level), **judged})
    return {
        "metric": metric_name,
        "kind": kind,
        "reference": None if reference is None else float(reference),
        "tol_pct": tol,
        "tol_pct_by_level": tol_by_level,
        "levels": rows,
        "pass": all(r["pass"] for r in rows if r["gated"]),
        "max_abs_err_pct": max(abs(r["err_pct"]) for r in rows),
    }


def judge_case(
    metric_values: Mapping[str, Sequence[float]],
    metric_specs: Iterable[Mapping],
    convergence_specs: Iterable[Mapping] | None = None,
    *,
    levels: Sequence[str] | None = None,
    declared_status: str | None = None,
    blocker: str | None = None,
) -> dict:
    """Three-level verdict for one case.

    Parameters
    ----------
    metric_values
        ``{metric_name: [coarse, ..., fine]}`` measured errors/values.
    metric_specs
        One spec per metric: ``name``, ``kind``, ``tol_pct``, optional
        ``reference`` and ``levels`` (level labels, coarse first).
    convergence_specs
        ``[{"metric": name, "max_span_pct": x, "require_monotone": bool}]``.
    declared_status / blocker
        A case may be declared ``pending`` (evidence withheld / not yet run) or
        ``failed`` (honest negative result with a root cause). A declaration is
        *never* allowed to upgrade a machine FAIL to passed, and a ``pending``
        declaration wins over an absent artifact — but a machine FAIL with no
        declaration is reported as ``failed``, not hidden.
    """
    if declared_status is not None and declared_status not in STATUSES:
        raise CriterionError(f"invalid declared_status: {declared_status!r}")

    metric_checks = []
    for spec in metric_specs:
        name = spec["name"]
        if name not in metric_values:
            metric_checks.append(
                {
                    "metric": name,
                    "levels": [],
                    "pass": False,
                    "max_abs_err_pct": None,
                    "missing": True,
                }
            )
            continue
        spec = dict(spec)
        spec["levels"] = list(levels or spec.get("levels") or [])
        if len(spec["levels"]) != len(metric_values[name]):
            raise CriterionError(
                f"metric {name!r}: {len(metric_values[name])} values but "
                f"{len(spec['levels'])} level labels"
            )
        metric_checks.append(_levels_of(name, metric_values[name], spec))

    conv_checks = []
    for spec in convergence_specs or []:
        name = spec["metric"]
        if name not in metric_values:
            conv_checks.append({"metric": name, "pass": False, "missing": True})
            continue
        conv_checks.append(
            {
                "metric": name,
                **convergence_trend(
                    metric_values[name],
                    max_span_pct=spec.get("max_span_pct", DEFAULT_SPAN_TOL_PCT),
                    require_monotone=spec.get("require_monotone"),
                ),
            }
        )

    machine_pass = all(c["pass"] for c in metric_checks) and all(
        c["pass"] for c in conv_checks
    )

    if declared_status == STATUS_PENDING:
        status = STATUS_PENDING
        reason = blocker or "declared pending: evidence not sufficient"
    elif declared_status == STATUS_FAILED or not machine_pass:
        status = STATUS_FAILED
        reason = blocker or "one or more machine-checked criteria failed"
    else:
        status = STATUS_PASSED
        reason = "all metric and convergence criteria passed"

    return {
        "status": status,
        "machine_pass": bool(machine_pass),
        "declared_status": declared_status,
        "reason": reason,
        "metrics": metric_checks,
        "convergence": conv_checks,
    }