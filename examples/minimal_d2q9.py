"""Minimal CPU-first D2Q9 example for TensorLBM."""

from __future__ import annotations

from tensorlbm import collide_and_stream, initialize_equilibrium, macroscopic


def main() -> None:
    ny, nx = 48, 96
    omega = 1.0

    f = initialize_equilibrium(ny=ny, nx=nx, rho0=1.0, u0=(0.05, 0.0))
    for _ in range(20):
        f = collide_and_stream(f, omega=omega)

    rho, u = macroscopic(f)
    print(f"rho mean: {rho.mean().item():.6f}")
    print(f"u mean: ({u[..., 0].mean().item():.6f}, {u[..., 1].mean().item():.6f})")


if __name__ == "__main__":
    main()
