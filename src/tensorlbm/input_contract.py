"""Input-contract guards shared by TensorLBM's public entry points.

Motivation (audit ``docs/AUDIT_RUNTEST_6_TENSORLBM.md`` findings D5/D6/D7/D9):
the library used to accept physically meaningless scalars on some code paths and
reject the very same values on others — e.g. ``collide_advanced_3d`` raised on
``tau<=0.5`` while the core ``collide_bgk`` silently produced ``inf``/``nan``,
and a pressure outlet with ``rho_out=0`` returned an all-NaN field with no
diagnostic.  Illegal input that survives into the time loop is far more expensive
to debug than an immediate ``ValueError``: the field is already poisoned when the
first ``nan`` shows up in a macroscopic observable.

The guards below are deliberately cheap (a handful of Python float comparisons,
evaluated once per call — not per lattice node) and never touch tensors, so they
are safe inside ``torch.compile``-traced update functions and inside per-step
loops.

Contract summary
----------------
* ``check_relaxation_time`` — BGK/MRT relaxation time must be finite and ``> 0.5``
  (``tau = 0.5`` is zero lattice viscosity, ``tau < 0.5`` is negative viscosity).
* ``check_positive_scalar`` — densities, lengths, viscosities, temperatures used
  as denominators/scales must be finite and strictly positive.
* ``check_positive_int`` — grid resolutions and step counts must be integers with
  the documented lower bound.
"""

from __future__ import annotations

import math

__all__ = [
    "check_relaxation_time",
    "check_positive_scalar",
    "check_positive_int",
]


def _as_float(value, name: str, where: str) -> float | None:
    """Return ``value`` as float, or ``None`` when it is not a plain scalar.

    Non-scalars (tensors, arrays) are passed through untouched: those code paths
    own their own validation, and coercing a tensor here would silently strip a
    gradient or force a device sync on every step.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_relaxation_time(tau, *, where: str = "collision") -> float:
    """Validate a hydrodynamic relaxation time and return it as ``float``.

    Parameters
    ----------
    tau:
        Relaxation time.  Must be finite and strictly greater than ``0.5``;
        ``nu = (tau - 0.5) / 3`` is negative otherwise, which makes the BGK
        collision anti-dissipative and the scheme unconditionally unstable.
    where:
        Call-site label used in the error message.

    Raises
    ------
    ValueError
        If ``tau`` is non-finite or ``<= 0.5``.
    """
    t = _as_float(tau, "tau", where)
    if t is None:  # tensor / exotic scalar: leave validation to the caller
        return tau
    if not math.isfinite(t):
        raise ValueError(f"{where}: relaxation time tau={t!r} is not finite")
    if t <= 0.5:
        raise ValueError(
            f"{where}: relaxation time tau={t} <= 0.5 gives non-positive lattice "
            f"viscosity nu=(tau-0.5)/3={((t - 0.5) / 3.0):.3e}; use tau>0.5"
        )
    return t


def check_positive_scalar(value, name: str, *, where: str = "", allow_zero: bool = False):
    """Validate a scalar that is used as a denominator or a physical scale.

    Non-finite values are always rejected.  ``value < 0`` is rejected; ``value == 0``
    is rejected unless ``allow_zero`` is set.
    """
    v = _as_float(value, name, where)
    if v is None:
        return value
    label = f"{where}: " if where else ""
    if not math.isfinite(v):
        raise ValueError(f"{label}{name}={v!r} is not finite")
    if v < 0.0 or (v == 0.0 and not allow_zero):
        raise ValueError(f"{label}{name}={v!r} must be strictly positive")
    return v


def check_positive_int(value, name: str, *, where: str = "", min_value: int = 1) -> int:
    """Validate an integer count (resolution, steps) against a lower bound."""
    label = f"{where}: " if where else ""
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        else:
            raise ValueError(f"{label}{name}={value!r} must be an integer")
    if value < min_value:
        raise ValueError(f"{label}{name}={value} must be >= {min_value}")
    return value