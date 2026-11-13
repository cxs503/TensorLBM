"""Library-level collision-operator selection for D2Q9 LBM (XVAL-LBM-METHOD-SELECTION).

This module is the *production-side* home of the method-selection decision that
the ``xval/benchmark_method_selection`` harness cross-validates.  It is the
single place where "which collision operator + sponge" is decided, so the solver
and the case pool share one selection path instead of each hard-coding a choice.

The selection rule is the one the harness measured (see
``xval/benchmark_method_selection`` and ``docs/XVAL_LBM_BENCHMARK_METHOD_SELECTION.md``):

* **MRT + sponge** is the recorded TensorLBM selection for wall-bounded,
  vortex-shedding cases at moderate Re (the A1 arm).  It is the default.
* **BGK** is the OSS single-relaxation selection (the A2 arm); it is offered as
  an explicit alternative, not the default, because the harness measured a
  ~10 % drag shift and a ~47 % lift-amplitude shift relative to A1 at D=10.
* **TRT** with the OSS "magic" Lambda = 3/16 is *not* selectable as-is at
  lattice viscosity near the incompressible limit: the harness measured a
  blow-up (A3) because the magic Lambda forces tau_minus far outside the stable
  window.  TRT is only selectable with a Lambda re-derived into the stable
  tau_minus window (the A5 remediation).

The function is pure (no I/O, no global state) so it can be unit-tested and
called from the solver without side effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CollisionKind = Literal["mrt", "bgk", "trt"]

#: Recommended stable window for the TRT anti-symmetric relaxation time.
TRT_TAU_MINUS_WINDOW: tuple[float, float] = (0.5, 2.0)
#: The OSS "magic" Lambda (Ginzburg 2008) that eliminates Poiseuille wall error.
TRT_MAGIC_LAMBDA: float = 3.0 / 16.0


@dataclass(frozen=True)
class Selection:
    """A concrete method-selection decision (collision operator + sponge)."""

    collision: CollisionKind
    sponge: bool
    lambda_trt: float | None = None
    reason: str = field(default="", compare=False)

    def as_dict(self) -> dict:
        return {
            "collision": self.collision,
            "sponge": self.sponge,
            "lambda_trt": self.lambda_trt,
            "reason": self.reason,
        }


def tau_from_nu(nu: float) -> float:
    """Lattice relaxation time from kinematic viscosity: tau = 0.5 + 3 nu."""
    if nu <= 0.0:
        raise ValueError(f"nu must be positive, got {nu}")
    return 0.5 + 3.0 * nu


def trt_tau_minus(tau_plus: float, lambda_trt: float) -> float:
    """Anti-symmetric relaxation time from the magic parameter:
    tau_minus = 0.5 + Lambda / (tau_plus - 0.5)."""
    if tau_plus <= 0.5:
        raise ValueError(f"tau_plus must be > 0.5, got {tau_plus}")
    return 0.5 + lambda_trt / (tau_plus - 0.5)


def trt_lambda_for_tau_minus(tau_plus: float, tau_minus: float) -> float:
    """Invert :func:`trt_tau_minus`: the Lambda that yields a target tau_minus.

    This is the A5 remediation: instead of transplanting the OSS magic Lambda
    (which diverges at low lattice viscosity), pick the Lambda that lands
    tau_minus at the upper end of the recommended window.
    """
    if tau_plus <= 0.5:
        raise ValueError(f"tau_plus must be > 0.5, got {tau_plus}")
    return (tau_minus - 0.5) * (tau_plus - 0.5)


def select_collision(
    *,
    nu: float,
    wall_bounded: bool = True,
    sponge: bool | None = None,
    prefer: CollisionKind = "mrt",
    trt_tau_minus_target: float | None = None,
) -> Selection:
    """Choose the collision operator + sponge for a D2Q9 case.

    Args:
        nu: kinematic viscosity in lattice units (nu = U D / Re).
        wall_bounded: whether the case has no-slip walls (needs a sponge to
            absorb the reflected wave at the outlet).
        prefer: the requested operator; ``"trt"`` is only honoured when a stable
            Lambda can be derived (see ``trt_tau_minus_target``).
        trt_tau_minus_target: for ``prefer="trt"``, the target tau_minus inside
            the stable window.  Defaults to the window upper end (2.0).

    Returns:
        A :class:`Selection`.  The decision is deterministic and pure.
    """
    tau = tau_from_nu(nu)
    if sponge is None:
        sponge = wall_bounded

    if prefer == "bgk":
        return Selection(
            collision="bgk",
            sponge=sponge,
            reason="explicit BGK request (OSS single-relaxation selection)",
        )

    if prefer == "trt":
        target = trt_tau_minus_target if trt_tau_minus_target is not None \
            else TRT_TAU_MINUS_WINDOW[1]
        lo, hi = TRT_TAU_MINUS_WINDOW
        target = min(max(target, lo), hi)
        lam = trt_lambda_for_tau_minus(tau, target)
        return Selection(
            collision="trt",
            sponge=sponge,
            lambda_trt=lam,
            reason=(f"TRT with Lambda re-derived to tau_minus={target:.3f} "
                    f"(magic Lambda={TRT_MAGIC_LAMBDA:.4f} would give "
                    f"tau_minus={trt_tau_minus(tau, TRT_MAGIC_LAMBDA):.2f}, outside "
                    f"the stable window {TRT_TAU_MINUS_WINDOW})"),
        )

    # default: the recorded TensorLBM selection (MRT + sponge for wall-bounded)
    return Selection(
        collision="mrt",
        sponge=sponge,
        reason="recorded TensorLBM selection (MRT + absorbing sponge)",
    )


def collide_dispatch(selection: Selection, f, tau: float, tau_field=None):
    """Apply the selected collision step to the distribution ``f``.

    This is the single dispatch point the solver uses, so the selection decision
    and the kernel call live in one place.  ``tau_field`` is the per-cell
    relaxation field (sponge); it is only meaningful for the MRT arm.
    """
    # Imported lazily so the selection module has no import-time dependency on
    # the (heavier) solver module.
    from tensorlbm.solver import collide_bgk, collide_mrt, collide_trt

    if selection.collision == "mrt":
        return collide_mrt(f, tau, tau_field=tau_field)
    if selection.collision == "bgk":
        return collide_bgk(f, tau)
    if selection.collision == "trt":
        return collide_trt(f, tau, lambda_trt=selection.lambda_trt or TRT_MAGIC_LAMBDA)
    raise ValueError(f"unknown collision kind: {selection.collision!r}")
