"""Virk law / MDR reference machinery for the turbulent drag-reduction side.

What this module *is*
---------------------
An implementation of the classical Virk three-segment mean-velocity description
of polymer drag-reduced turbulent channel flow, plus the machinery needed to
judge a measured drag-reduction law:

* viscous sublayer ``u+ = y+``,
* Virk elastic (intercept-shifted log) segment ``u+ = 2.5 ln y+ + 5.5 + dB``,
* Maximum Drag Reduction (MDR) asymptote ``u+ = 11.7 ln y+ - 17.0``,

with the drag-reduced profile rendered as ``min(max(elastic, y+), MDR)`` so that
no admissible ``dB`` can ever exceed the MDR asymptote, and a log-law intercept
estimator (:func:`delta_b_from_log_law`) that turns a *measured* mean profile into
``dB``.

What this module is *not*
-------------------------
It contains **no DNS/LES of viscoelastic turbulence**.  ``dB(Wi)`` is therefore
*not measured* in this mission: what is verified here is the reference law, the
MDR cap and the (monotonicity + boundedness) acceptance machinery that a future
turbulent measurement must satisfy.  Evidence grade of the drag-reduction claim:
``WITHHELD`` -- see ``docs/VISCOELASTIC_P50_N06.md`` section 5.

Convention note
---------------
``dB`` is the intercept increment of the log law in wall units.  Virk's MDR
asymptote has a different slope (``1/11.7`` instead of ``1/2.5``), so the
intercept increment is not unique across conventions; the mission/BP bound used
here is ``dB <= 11.7`` (:data:`MDR_DELTA_B_MAX`) and it is applied as an upper
bound only.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "MDR_DELTA_B_MAX",
    "MDR_INTERCEPT",
    "MDR_SLOPE",
    "NEWTONIAN_INTERCEPT",
    "NEWTONIAN_SLOPE",
    "check_delta_b_bound",
    "delta_b_from_log_law",
    "drag_reduction_fraction",
    "virk_mdr_profile",
    "virk_newtonian_profile",
    "virk_segment_labels",
    "virk_three_segment",
]

NEWTONIAN_SLOPE = 2.5
NEWTONIAN_INTERCEPT = 5.5
MDR_SLOPE = 11.7
MDR_INTERCEPT = -17.0
MDR_DELTA_B_MAX = 11.7
VISCOUS_SUBLAYER_END = 11.6


def virk_newtonian_profile(y_plus):
    """Newtonian reference: ``u+ = y+`` for ``y+<11.6`` else ``2.5 ln y+ + 5.5``."""
    y = torch.as_tensor(y_plus, dtype=torch.float64)
    log_part = NEWTONIAN_SLOPE * torch.log(y) + NEWTONIAN_INTERCEPT
    return torch.where(y < VISCOUS_SUBLAYER_END, y, log_part)


def virk_mdr_profile(y_plus):
    """Virk MDR asymptote: ``u+ = y+`` for ``y+<11.6`` else ``11.7 ln y+ - 17``."""
    y = torch.as_tensor(y_plus, dtype=torch.float64)
    mdr = MDR_SLOPE * torch.log(y) + MDR_INTERCEPT
    return torch.where(y < VISCOUS_SUBLAYER_END, y, mdr)


def virk_three_segment(y_plus, delta_b: float = 0.0):
    """Three-segment drag-reduced profile (viscous / elastic / MDR).

    ``delta_b = 0`` recovers the Newtonian reference exactly; the profile is
    monotone in ``delta_b`` and can never exceed the MDR asymptote.
    """
    check_delta_b_bound(delta_b)
    y = torch.as_tensor(y_plus, dtype=torch.float64)
    viscous = y
    elastic = NEWTONIAN_SLOPE * torch.log(y) + NEWTONIAN_INTERCEPT + float(delta_b)
    mdr = MDR_SLOPE * torch.log(y) + MDR_INTERCEPT
    log_segment = torch.minimum(elastic, mdr)
    return torch.where(y < VISCOUS_SUBLAYER_END, viscous, log_segment)


def virk_segment_labels(y_plus, delta_b: float = 0.0) -> list[str]:
    """Label each ``y+`` with its active segment: ``viscous``/``elastic``/``mdr``."""
    y = torch.as_tensor(y_plus, dtype=torch.float64).reshape(-1)
    labels: list[str] = []
    for yi in y.tolist():
        elastic = NEWTONIAN_SLOPE * math.log(yi) + NEWTONIAN_INTERCEPT + float(delta_b)
        mdr = MDR_SLOPE * math.log(yi) + MDR_INTERCEPT
        if yi < VISCOUS_SUBLAYER_END:
            labels.append("viscous")
        elif elastic <= mdr:
            labels.append("elastic")
        else:
            labels.append("mdr")
    return labels


def check_delta_b_bound(delta_b: float, *, bound: float = MDR_DELTA_B_MAX) -> float:
    """Acceptance guard: ``0 <= dB <= 11.7`` (MDR cap).

    Raises:
        ValueError: if ``delta_b`` is negative or exceeds the MDR bound.
    """
    dB = float(delta_b)
    if not math.isfinite(dB):
        msg = f"delta_B must be finite (got {delta_b!r})"
        raise ValueError(msg)
    if dB < 0.0:
        msg = f"delta_B must be non-negative (got {dB})"
        raise ValueError(msg)
    if dB > bound:
        msg = f"delta_B={dB} exceeds the MDR bound {bound}"
        raise ValueError(msg)
    return dB


def delta_b_from_log_law(
    u_plus,
    y_plus,
    *,
    window: tuple[float, float] = (30.0, 100.0),
) -> dict:
    """Estimate ``dB`` from a measured mean profile by fitting the log-law offset.

    Regresses ``u+ - 2.5 ln y+`` on a constant over ``window`` (least squares);
    the constant is the apparent ``B`` and ``dB = B - 5.5``.  Returns the fit
    diagnostics so a caller can judge whether the window really is logarithmic.
    """
    y = torch.as_tensor(y_plus, dtype=torch.float64).reshape(-1)
    u = torch.as_tensor(u_plus, dtype=torch.float64).reshape(-1)
    mask = (y >= window[0]) & (y <= window[1]) & (u > 0)
    if int(mask.sum()) < 2:
        msg = f"not enough points in window {window} to fit the log law"
        raise ValueError(msg)
    y_w, u_w = y[mask], u[mask]
    residual_target = u_w - NEWTONIAN_SLOPE * torch.log(y_w)
    b_fit = float(residual_target.mean())
    scatter = float((residual_target - b_fit).abs().max())
    return {
        "b_fit": b_fit,
        "delta_b": b_fit - NEWTONIAN_INTERCEPT,
        "n_points": int(mask.sum()),
        "max_scatter": scatter,
        "window": list(window),
    }


def drag_reduction_fraction(q_polymer: float, q_newtonian: float) -> float:
    """``DR = 1 - Q_p/Q_N`` at matched pressure gradient (dimensionless)."""
    if q_newtonian == 0.0:
        msg = "Newtonian flow rate is zero"
        raise ValueError(msg)
    return 1.0 - float(q_polymer) / float(q_newtonian)