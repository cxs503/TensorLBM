"""Minimal runnable D2Q9 LBM example."""

from tensorlbm import LBMSimulation


def main() -> None:
    sim = LBMSimulation(nx=64, ny=32, tau=0.6, device="cpu")
    rho, u = sim.run(steps=20)

    print("Simulation complete")
    print(f"rho mean={rho.mean().item():.6f}, rho std={rho.std().item():.6f}")
    print(f"velocity mean magnitude={(u.pow(2).sum(dim=-1).sqrt().mean().item()):.6f}")


if __name__ == "__main__":
    main()
