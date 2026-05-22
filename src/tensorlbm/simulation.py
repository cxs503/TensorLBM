"""Simple LBM simulation entry point."""

import torch

from .constants import D2Q9
from .lattice import equilibrium, stream


class LBMSimulation:
    """Minimal D2Q9 BGK simulation on a 2D periodic domain."""

    def __init__(self, nx: int = 64, ny: int = 32, tau: float = 0.6, device: str = "cpu") -> None:
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.device = torch.device(device)

        rho0 = torch.ones((ny, nx), dtype=torch.float32, device=self.device)
        u0 = torch.zeros((ny, nx, 2), dtype=torch.float32, device=self.device)
        self.f = equilibrium(rho0, u0)

    def macroscopic(self) -> tuple[torch.Tensor, torch.Tensor]:
        rho = self.f.sum(dim=-1)
        c = D2Q9.c.to(dtype=self.f.dtype, device=self.device)
        momentum = torch.einsum("...q,qa->...a", self.f, c)
        u = momentum / rho.unsqueeze(-1).clamp_min(1e-12)
        return rho, u

    def step(self) -> None:
        rho, u = self.macroscopic()
        feq = equilibrium(rho, u)
        f_post_collision = self.f - (self.f - feq) / self.tau
        self.f = stream(f_post_collision)

    def run(self, steps: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(steps):
            self.step()
        return self.macroscopic()
