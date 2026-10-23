"""Pressure-gradient equilibrium ODE wall-model candidate.

The common inner-layer equation retained here is

``d[(nu + nu_t) du_t/dy]/dy = dp_t/(rho ds)``.

For an attached, locally steady wall layer its once-integrated form is

``du_t/dy = (u_tau**2 + a_p*y)/(nu + nu_t)``,

where ``a_p = dp_t/(rho ds)`` is positive for an adverse pressure gradient.
The exchange velocity is obtained by quadrature from the wall to ``y_m`` and
the non-negative friction velocity is solved by bisection.  Either a classic
Van-Driest-damped mixing length or the pressure-gradient velocity scale of
Duprat et al. (2011, DOI 10.1063/1.3529358) supplies ``nu_t``.

This module is deliberately a candidate closure, not a production force path.
If an adverse gradient already predicts an exchange velocity above the
observed value at zero wall shear, the non-negative attached solution does not
exist.  The solver reports that node as separated and returns zero shear
instead of silently clipping a negative-shear solution into an attached wall
law.  Reverse shear still requires a separately validated closure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PressureGradientWallModelResult:
    friction_velocity: torch.Tensor
    shear_stress_over_density: torch.Tensor
    reconstructed_exchange_speed: torch.Tensor
    residual: torch.Tensor
    attached: torch.Tensor
    separated: torch.Tensor
    finite: torch.Tensor
    iterations: int
    quadrature_points: int


def pressure_gradient_eddy_viscosity(
    wall_distance: torch.Tensor,
    friction_velocity: torch.Tensor,
    pressure_gradient_acceleration: torch.Tensor,
    nu: float,
    *,
    model: str = "van_driest",
    kappa: float = 0.4,
    a_plus: float = 18.0,
    duprat_beta: float = 0.78,
) -> torch.Tensor:
    """Evaluate classic or Duprat mixing-length eddy viscosity.

    ``pressure_gradient_acceleration`` is the kinematic pressure gradient
    ``|grad(p)|/rho`` for Duprat's pressure velocity.  Supplying the signed
    streamwise component is also valid because only its magnitude defines
    ``u_p``; its sign remains in the ODE source numerator.
    """
    fields = (wall_distance, friction_velocity, pressure_gradient_acceleration)
    if any(not field.is_floating_point() for field in fields):
        raise ValueError("eddy-viscosity fields must be floating point")
    if len({field.device for field in fields}) != 1:
        raise ValueError("eddy-viscosity fields must share a device")
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    if model not in {"van_driest", "duprat"}:
        raise ValueError("model must be 'van_driest' or 'duprat'")
    if not 0.0 < kappa < 1.0:
        raise ValueError("kappa must lie in (0,1)")
    if a_plus <= 0.0:
        raise ValueError("a_plus must be positive")
    if duprat_beta <= 0.0:
        raise ValueError("duprat_beta must be positive")
    try:
        y, u_tau, acceleration = torch.broadcast_tensors(*fields)
    except RuntimeError as error:
        raise ValueError("eddy-viscosity fields must be broadcastable") from error
    if bool((y < 0.0).any()) or bool((u_tau < 0.0).any()):
        raise ValueError("wall distance and friction velocity must be non-negative")
    if model == "van_driest":
        y_plus = y * u_tau / nu
        damping = -torch.expm1(-y_plus / a_plus)
        return kappa * y * u_tau * damping.square()

    pressure_velocity = (nu * acceleration.abs()).pow(1.0 / 3.0)
    combined_velocity = torch.sqrt(u_tau.square() + pressure_velocity.square())
    combined_squared = combined_velocity.square()
    alpha = torch.where(
        combined_squared > 0.0,
        u_tau.square()
        / combined_squared.clamp_min(
            torch.finfo(combined_squared.dtype).tiny,
        ),
        torch.ones_like(combined_squared),
    )
    y_star = y * combined_velocity / nu
    base = alpha + y_star * (1.0 - alpha).clamp_min(0.0).pow(1.5)
    damping = -torch.expm1(-y_star / (1.0 + a_plus * alpha.pow(3)))
    return nu * kappa * y_star * base.clamp_min(0.0).pow(duprat_beta) * damping.square()


def _validate_fields(
    exchange_speed: torch.Tensor,
    exchange_distance: torch.Tensor | float,
    pressure_gradient_acceleration: torch.Tensor | float,
    nu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not exchange_speed.is_floating_point():
        raise ValueError("exchange_speed must be floating point")
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    distance = torch.as_tensor(
        exchange_distance,
        dtype=exchange_speed.dtype,
        device=exchange_speed.device,
    )
    acceleration = torch.as_tensor(
        pressure_gradient_acceleration,
        dtype=exchange_speed.dtype,
        device=exchange_speed.device,
    )
    try:
        distance = torch.broadcast_to(distance, exchange_speed.shape)
        acceleration = torch.broadcast_to(acceleration, exchange_speed.shape)
    except RuntimeError as error:
        raise ValueError(
            "distance and pressure-gradient acceleration must broadcast to speed",
        ) from error
    if bool((distance <= 0.0).any()):
        raise ValueError("exchange_distance must be positive")
    return distance, acceleration


def pressure_gradient_equilibrium_velocity(
    friction_velocity: torch.Tensor,
    exchange_distance: torch.Tensor | float,
    pressure_gradient_acceleration: torch.Tensor | float,
    nu: float,
    *,
    pressure_gradient_magnitude_acceleration: torch.Tensor | float | None = None,
    kappa: float = 0.4,
    van_driest_a_plus: float = 18.0,
    eddy_viscosity_model: str = "van_driest",
    duprat_beta: float = 0.78,
    quadrature_points: int = 64,
    mixing_length: bool = True,
) -> torch.Tensor:
    """Integrate the attached ODE from the wall to the exchange location."""
    if not friction_velocity.is_floating_point():
        raise ValueError("friction_velocity must be floating point")
    if bool((friction_velocity < 0.0).any()):
        raise ValueError("friction_velocity must be non-negative")
    distance, acceleration = _validate_fields(
        friction_velocity,
        exchange_distance,
        pressure_gradient_acceleration,
        nu,
    )
    if pressure_gradient_magnitude_acceleration is None:
        acceleration_magnitude = acceleration.abs()
    else:
        acceleration_magnitude = torch.as_tensor(
            pressure_gradient_magnitude_acceleration,
            dtype=friction_velocity.dtype,
            device=friction_velocity.device,
        )
        try:
            acceleration_magnitude = torch.broadcast_to(
                acceleration_magnitude,
                friction_velocity.shape,
            )
        except RuntimeError as error:
            raise ValueError(
                "pressure-gradient magnitude must broadcast to friction velocity",
            ) from error
        if bool((acceleration_magnitude < 0.0).any()):
            raise ValueError("pressure-gradient magnitude must be non-negative")
    if not 0.0 < kappa < 1.0:
        raise ValueError("kappa must lie in (0,1)")
    if van_driest_a_plus <= 0.0:
        raise ValueError("van_driest_a_plus must be positive")
    if eddy_viscosity_model not in {"van_driest", "duprat"}:
        raise ValueError("eddy_viscosity_model must be 'van_driest' or 'duprat'")
    if duprat_beta <= 0.0:
        raise ValueError("duprat_beta must be positive")
    if (
        not isinstance(quadrature_points, int)
        or isinstance(
            quadrature_points,
            bool,
        )
        or quadrature_points < 2
    ):
        raise ValueError("quadrature_points must be an integer >= 2")

    # Midpoint quadrature integrates the laminar linear numerator exactly and
    # avoids evaluating the mixing length at the wall singular point.
    eta = (
        torch.arange(
            quadrature_points,
            dtype=friction_velocity.dtype,
            device=friction_velocity.device,
        )
        + 0.5
    ) / quadrature_points
    y = distance.unsqueeze(-1) * eta
    u_tau = friction_velocity.unsqueeze(-1)
    if mixing_length:
        nu_t = pressure_gradient_eddy_viscosity(
            y,
            u_tau,
            acceleration_magnitude.unsqueeze(-1),
            nu,
            model=eddy_viscosity_model,
            kappa=kappa,
            a_plus=van_driest_a_plus,
            duprat_beta=duprat_beta,
        )
    else:
        nu_t = torch.zeros_like(y)
    gradient = (u_tau.square() + acceleration.unsqueeze(-1) * y) / (nu + nu_t)
    return gradient.mean(dim=-1) * distance


def solve_pressure_gradient_equilibrium_wall_shear(
    exchange_speed: torch.Tensor,
    exchange_distance: torch.Tensor | float,
    pressure_gradient_acceleration: torch.Tensor | float,
    nu: float,
    *,
    pressure_gradient_magnitude_acceleration: torch.Tensor | float | None = None,
    kappa: float = 0.4,
    van_driest_a_plus: float = 18.0,
    eddy_viscosity_model: str = "van_driest",
    duprat_beta: float = 0.78,
    quadrature_points: int = 64,
    iterations: int = 48,
    mixing_length: bool = True,
) -> PressureGradientWallModelResult:
    """Solve non-negative attached wall shear and expose separation failures."""
    distance, acceleration = _validate_fields(
        exchange_speed,
        exchange_distance,
        pressure_gradient_acceleration,
        nu,
    )
    if pressure_gradient_magnitude_acceleration is None:
        acceleration_magnitude = acceleration.abs()
    else:
        acceleration_magnitude = torch.as_tensor(
            pressure_gradient_magnitude_acceleration,
            dtype=exchange_speed.dtype,
            device=exchange_speed.device,
        )
        try:
            acceleration_magnitude = torch.broadcast_to(
                acceleration_magnitude,
                exchange_speed.shape,
            )
        except RuntimeError as error:
            raise ValueError(
                "pressure-gradient magnitude must broadcast to exchange speed",
            ) from error
        if bool((acceleration_magnitude < 0.0).any()):
            raise ValueError("pressure-gradient magnitude must be non-negative")
    if (
        not isinstance(iterations, int)
        or isinstance(
            iterations,
            bool,
        )
        or iterations < 8
    ):
        raise ValueError("iterations must be an integer >= 8")
    finite = (
        torch.isfinite(exchange_speed)
        & torch.isfinite(distance)
        & torch.isfinite(acceleration)
        & torch.isfinite(acceleration_magnitude)
        & (exchange_speed >= 0.0)
    )
    safe_speed = torch.where(finite, exchange_speed, torch.zeros_like(exchange_speed))
    safe_acceleration = torch.where(
        finite,
        acceleration,
        torch.zeros_like(acceleration),
    )
    safe_acceleration_magnitude = torch.where(
        finite,
        acceleration_magnitude,
        torch.zeros_like(acceleration_magnitude),
    )
    zero = torch.zeros_like(safe_speed)
    minimum_profile = pressure_gradient_equilibrium_velocity(
        zero,
        distance,
        safe_acceleration,
        nu,
        pressure_gradient_magnitude_acceleration=safe_acceleration_magnitude,
        kappa=kappa,
        van_driest_a_plus=van_driest_a_plus,
        eddy_viscosity_model=eddy_viscosity_model,
        duprat_beta=duprat_beta,
        quadrature_points=quadrature_points,
        mixing_length=mixing_length,
    )
    # Equality is an attached zero-shear limit; strictly lower observed speed
    # requires reverse wall shear and is outside this candidate's contract.
    separated = finite & (safe_speed < minimum_profile)
    attached = finite & ~separated

    low = torch.zeros_like(safe_speed)
    high = torch.maximum(
        safe_speed,
        torch.sqrt((nu * safe_speed / distance).clamp_min(0.0)),
    ).clamp_min(torch.finfo(safe_speed.dtype).eps)
    high = torch.maximum(
        high,
        torch.sqrt((safe_acceleration.abs() * distance).clamp_min(0.0)),
    )
    for _ in range(16):
        high_profile = pressure_gradient_equilibrium_velocity(
            high,
            distance,
            safe_acceleration,
            nu,
            pressure_gradient_magnitude_acceleration=safe_acceleration_magnitude,
            kappa=kappa,
            van_driest_a_plus=van_driest_a_plus,
            eddy_viscosity_model=eddy_viscosity_model,
            duprat_beta=duprat_beta,
            quadrature_points=quadrature_points,
            mixing_length=mixing_length,
        )
        needs_expansion = attached & (high_profile < safe_speed)
        high = torch.where(needs_expansion, 2.0 * high, high)
    bracket_profile = pressure_gradient_equilibrium_velocity(
        high,
        distance,
        safe_acceleration,
        nu,
        pressure_gradient_magnitude_acceleration=safe_acceleration_magnitude,
        kappa=kappa,
        van_driest_a_plus=van_driest_a_plus,
        eddy_viscosity_model=eddy_viscosity_model,
        duprat_beta=duprat_beta,
        quadrature_points=quadrature_points,
        mixing_length=mixing_length,
    )
    attached &= bracket_profile >= safe_speed

    for _ in range(iterations):
        mid = 0.5 * (low + high)
        profile = pressure_gradient_equilibrium_velocity(
            mid,
            distance,
            safe_acceleration,
            nu,
            pressure_gradient_magnitude_acceleration=safe_acceleration_magnitude,
            kappa=kappa,
            van_driest_a_plus=van_driest_a_plus,
            eddy_viscosity_model=eddy_viscosity_model,
            duprat_beta=duprat_beta,
            quadrature_points=quadrature_points,
            mixing_length=mixing_length,
        )
        below = profile < safe_speed
        low = torch.where(attached & below, mid, low)
        high = torch.where(attached & below, high, mid)
    u_tau = torch.where(attached, 0.5 * (low + high), zero)
    reconstructed = pressure_gradient_equilibrium_velocity(
        u_tau,
        distance,
        safe_acceleration,
        nu,
        pressure_gradient_magnitude_acceleration=safe_acceleration_magnitude,
        kappa=kappa,
        van_driest_a_plus=van_driest_a_plus,
        eddy_viscosity_model=eddy_viscosity_model,
        duprat_beta=duprat_beta,
        quadrature_points=quadrature_points,
        mixing_length=mixing_length,
    )
    reconstructed = torch.where(finite, reconstructed, torch.full_like(reconstructed, float("nan")))
    residual = reconstructed - exchange_speed
    return PressureGradientWallModelResult(
        friction_velocity=u_tau,
        shear_stress_over_density=u_tau.square(),
        reconstructed_exchange_speed=reconstructed,
        residual=residual,
        attached=attached,
        separated=separated,
        finite=finite,
        iterations=iterations,
        quadrature_points=quadrature_points,
    )


__all__ = [
    "PressureGradientWallModelResult",
    "pressure_gradient_eddy_viscosity",
    "pressure_gradient_equilibrium_velocity",
    "solve_pressure_gradient_equilibrium_wall_shear",
]
