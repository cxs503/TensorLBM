"""One-dimensional wall-model-driven channel closure (wall units).

Why this module exists
----------------------
A wall-modelled LES exchanges one velocity sample per wall with a wall model and
gets back the friction velocity.  Between the wall and that exchange height the
wall model itself is the *only* statement about the flow, so the wall model can
be graded on its own by integrating the modelled layer over the whole channel
and comparing the result with a channel DNS.  Doing that costs milliseconds
instead of the hours a converged WMLES needs, and it separates two failure modes
that are otherwise entangled in a coarse LES:

* a **closure** failure (the modelled stress/velocity profile is physically
  wrong at any resolution), measured here;
* a **resolution** failure (the closure is fine but the grid cannot sample the
  feature of interest), measured by the LES benchmark.

Closures implemented
--------------------
``mixing_length`` (van Driest damped)
    ``l+ = kappa y+ (1 - exp(-y+/A+))``, ``nu_t+ = l+^2 du+/dy+`` and the
    channel stress balance ``tau+ = 1 - y+/h+ = du+/dy+ + nu_t+ du+/dy+``
    inverted in closed form for ``du+/dy+``.  This is the eddy-viscosity kernel
    used by :class:`tensorlbm.wall_model_les.models.NonEquilibriumWallModel`.

``log_layer`` (undamped)
    The same with ``A+ -> 0``: the classic Prandtl mixing length.  The pair is
    the controlled comparison "does the damping matter for the quantity of
    interest".

Every quantity is returned in wall units, so ``h+ = Re_tau`` and the molecular
viscous stress is simply ``du+/dy+`` (the ``1/Re_tau`` factor cancels between
``nu`` and the wall-unit normalisation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .models import B_DEFAULT, KAPPA_DEFAULT, VAN_DRIEST_A_PLUS

__all__ = [
    "ChannelClosureSolution",
    "CLOSURES",
    "solve_channel_closure",
    "log_layer_fit",
    "uv_monotonicity",
]


#: Closure identifiers accepted by :func:`solve_channel_closure`.
CLOSURES = ("mixing_length", "log_layer")


@dataclass(frozen=True)
class ChannelClosureSolution:
    """Modelled channel in wall units, from one wall out to the centreline."""

    re_tau: float
    closure: str
    kappa: float
    b: float
    van_driest_a_plus: float
    y_plus: np.ndarray
    u_plus: np.ndarray
    du_dy_plus: np.ndarray
    eddy_viscosity_plus: np.ndarray
    total_stress_plus: np.ndarray
    uv_plus: np.ndarray
    production_plus: np.ndarray
    extra: dict = field(default_factory=dict)

    # -- derived anchors -------------------------------------------------
    @property
    def uv_peak_y_plus(self) -> float:
        from .budget import peak_location

        mask = self.y_plus >= 1.0
        return peak_location(self.y_plus[mask], self.uv_plus[mask])

    @property
    def production_peak_y_plus(self) -> float:
        from .budget import peak_location

        mask = self.y_plus >= 1.0
        return peak_location(self.y_plus[mask], self.production_plus[mask])

    @property
    def stress_identity_max_abs_error(self) -> float:
        """``|du+/dy+ - <u'v'>+ - (1 - y+/h+)|``: zero by construction."""

        return float(np.max(np.abs(self.du_dy_plus + self.uv_plus - self.total_stress_plus)))

    def as_dict(self) -> dict:
        return {
            "re_tau": float(self.re_tau),
            "closure": self.closure,
            "kappa": float(self.kappa),
            "b": float(self.b),
            "van_driest_a_plus": float(self.van_driest_a_plus),
            "uv_peak_y_plus": float(self.uv_peak_y_plus),
            "uv_peak_value": float(np.max(self.uv_plus[self.y_plus >= 1.0])),
            "production_peak_y_plus": float(self.production_peak_y_plus),
            "production_peak_value": float(np.max(self.production_plus)),
            "stress_identity_max_abs_error": self.stress_identity_max_abs_error,
            "n_layers": int(self.y_plus.size),
            "u_plus_at_30": float(np.interp(30.0, self.y_plus, self.u_plus)),
            "u_plus_at_100": float(np.interp(100.0, self.y_plus, self.u_plus)),
            **self.extra,
        }


def _layer_faces(re_tau: float, n_layers: int) -> np.ndarray:
    """Quadratic clustering: fine at the wall, ``h+`` at the centreline."""

    s = np.linspace(0.0, 1.0, int(n_layers) + 1)
    return re_tau * s**2


def solve_channel_closure(
    re_tau: float,
    *,
    closure: str = "mixing_length",
    kappa: float = KAPPA_DEFAULT,
    b: float = B_DEFAULT,
    van_driest_a_plus: float = VAN_DRIEST_A_PLUS,
    n_layers: int = 4000,
) -> ChannelClosureSolution:
    """Integrate the modelled layer across a channel at ``Re_tau = h+``.

    The stress balance ``tau+ = 1 - y+/h+`` is imposed exactly; only the eddy
    viscosity is modelled.  Hence the returned profile satisfies
    ``du+/dy+ + nu_t+ du+/dy+ = 1 - y+/h+`` to round-off, and the Reynolds
    stress ``-<u'v'>+ = nu_t+ du+/dy+`` is a *prediction* of the closure rather
    than an imposed shape.
    """

    if closure not in CLOSURES:
        msg = f"unknown closure {closure!r}; expected one of {CLOSURES}"
        raise ValueError(msg)
    re_tau = float(re_tau)
    if re_tau <= 1.0:
        msg = "Re_tau must be > 1 (a wall unit grid needs h+ > 1)"
        raise ValueError(msg)

    y_plus = _layer_faces(re_tau, n_layers)
    dy = np.diff(y_plus)
    y_c = 0.5 * (y_plus[:-1] + y_plus[1:])
    tau_plus = np.clip(1.0 - y_c / re_tau, 1.0e-12, None)

    damping = (
        np.ones_like(y_c)
        if closure == "log_layer"
        else 1.0 - np.exp(-y_c / max(van_driest_a_plus, 1.0e-12))
    )
    ell_plus = kappa * y_c * damping

    # tau = du/dy + l^2 (du/dy)^2  ->  du/dy = 2 tau / (1 + sqrt(1 + 4 l^2 tau))
    du_dy = 2.0 * tau_plus / (1.0 + np.sqrt(1.0 + 4.0 * ell_plus**2 * tau_plus))
    nu_t_plus = ell_plus**2 * du_dy
    uv_plus = nu_t_plus * du_dy

    u_plus = np.concatenate([[0.0], np.cumsum(du_dy * dy)])

    # P+ = -<u'v'>+ dU+/dy+ evaluated on the same faces; pad to face count.
    production = uv_plus * du_dy
    production = np.concatenate([production[:1], production])

    # drop the (singular) first face value at the wall for the derivative arrays
    du_dy_full = np.concatenate([du_dy[:1], du_dy])
    nu_t_full = np.concatenate([nu_t_plus[:1], nu_t_plus])
    uv_full = np.concatenate([uv_plus[:1], uv_plus])
    tau_full = np.concatenate([tau_plus[:1], tau_plus])

    return ChannelClosureSolution(
        re_tau=re_tau,
        closure=closure,
        kappa=float(kappa),
        b=float(b),
        van_driest_a_plus=float(van_driest_a_plus),
        y_plus=y_plus,
        u_plus=u_plus,
        du_dy_plus=du_dy_full,
        eddy_viscosity_plus=nu_t_full,
        total_stress_plus=tau_full,
        uv_plus=uv_full,
        production_plus=production,
        extra={"dy_plus_first": float(dy[0])},
    )


def log_layer_fit(
    y_plus: np.ndarray,
    u_plus: np.ndarray,
    *,
    band: tuple[float, float] = (30.0, 300.0),
    h_plus: float | None = None,
    wall_fraction: float = 0.2,
) -> dict:
    """Least-squares ``u+ = (1/kappa) ln y+ + B`` over the criterion band.

    The band is clipped at ``wall_fraction * h+`` when ``h_plus`` is given: above
    roughly a fifth of the half-height the wake contribution is no longer a
    small correction and a log-law fit there measures the wake, not the slope.
    """

    y = np.asarray(y_plus, dtype=float)
    u = np.asarray(u_plus, dtype=float)
    hi = band[1]
    if h_plus is not None:
        hi = min(hi, wall_fraction * float(h_plus))
    mask = (y >= band[0]) & (y <= hi) & np.isfinite(u)
    if mask.sum() < 3:
        return {
            "n": int(mask.sum()),
            "y_plus_max": float(hi),
            "kappa": float("nan"),
            "b": float("nan"),
            "max_rel_error": float("nan"),
        }
    slope, intercept = np.polyfit(np.log(y[mask]), u[mask], 1)
    fitted = slope * np.log(y[mask]) + intercept
    rel = np.abs(u[mask] - fitted) / np.abs(u[mask])
    return {
        "n": int(mask.sum()),
        "y_plus_min": float(y[mask].min()),
        "y_plus_max": float(y[mask].max()),
        "kappa": float(1.0 / slope) if slope > 0 else float("nan"),
        "b": float(intercept),
        "max_rel_error": float(rel.max()),
        "rms_rel_error": float(np.sqrt(np.mean(rel**2))),
    }


def uv_monotonicity(y_plus: np.ndarray, uv_plus: np.ndarray, *, y_min: float = 5.0) -> dict:
    """Where does ``-<u'v'>+`` stop growing?

    An eddy-viscosity closure has ``-<u'v'>+ = nu_t+ dU+/dy+`` with ``nu_t+ >= 0``
    and ``dU+/dy+ >= 0``; the modelled stress can only fall back once the total
    stress ``1 - y+/h+`` starts to dominate the decay, i.e. the interior peak is
    a by-product of the *scale* growth ``l+ ~ kappa y+``.  Reporting the fraction
    of the layer where the modelled stress is still increasing makes the
    structural statement measurable instead of rhetorical.
    """

    y = np.asarray(y_plus, dtype=float)
    uv = np.asarray(uv_plus, dtype=float)
    mask = y >= y_min
    grad = np.gradient(uv, y)
    rising = grad[mask] > 0.0
    return {
        "y_plus_min": float(y_min),
        "rising_fraction": float(np.mean(rising)),
        "d_uv_dy_plus_at_wall_side": float(grad[mask][0]),
    }


def sweep(
    re_values: Iterable[float],
    *,
    closure: str = "mixing_length",
    **kwargs,
) -> list[ChannelClosureSolution]:
    """Solve several Reynolds numbers with the same closure."""

    return [solve_channel_closure(re, closure=closure, **kwargs) for re in re_values]