"""Simple LBM simulation entry point.

Two storages are supported for the populations:

``layout="dense"``   (default) TensorLBM's native ``(ny, nx, 9)`` array.
``layout="lbt_aa"``  the 2b-t/LB-t A-A layout pattern (single merged
                     ``NZ*NY*NX*NP*ND`` array with the component index ``p``
                     and the padded population slot), see
                     :mod:`tensorlbm.aa_layout`.  Both storages run the same
                     discrete scheme, so results agree to round-off; the option
                     exists to cross-validate against the OSS convention.
"""

import torch

from .constants import D2Q9
from .lattice import equilibrium, stream

#: D2Q9 discrete velocities in TensorLBM direction order for the NumPy path used
#: by the LB-t layout storage (``layout="lbt_aa"``).
AA_DIRS = D2Q9.c.numpy().astype("float64")


class LBMSimulation:
    """Minimal D2Q9 BGK simulation on a 2D periodic domain."""

    def __init__(
        self,
        nx: int = 64,
        ny: int = 32,
        tau: float = 0.6,
        device: str = "cpu",
        layout: str = "dense",
    ) -> None:
        if layout not in ("dense", "lbt_aa"):
            raise ValueError(f"layout must be 'dense' or 'lbt_aa' (got {layout!r})")
        if nx <= 0 or ny <= 0:
            raise ValueError(f"nx and ny must be positive integers (got nx={nx}, ny={ny})")
        if tau <= 0:
            raise ValueError(f"tau must be positive (got tau={tau})")
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.device = torch.device(device)
        self.layout = layout
        self._aa = None
        self._parity = 0

        if layout == "lbt_aa":
            import numpy as np

            from .aa_layout import AaPopulation, equilibrium_d2q9

            self._aa = AaPopulation(nx=nx, ny=ny, nz=1, npop=1, dtype=np.float64)
            ones = np.ones((1, ny, nx, 1))
            zeros = np.zeros((1, ny, nx, 1))
            self._aa.from_dense(equilibrium_d2q9(ones, zeros, zeros))
            self.f = self._aa.to_dense()[0, :, :, 0, :]
            return

        rho0 = torch.ones((ny, nx), dtype=torch.float32, device=self.device)
        u0 = torch.zeros((ny, nx, 2), dtype=torch.float32, device=self.device)
        self.f = equilibrium(rho0, u0)

    def macroscopic(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._aa is not None:
            rho_np, ux_np, uy_np = self._aa.macroscopic()
            rho = torch.as_tensor(rho_np[0, :, :, 0], dtype=torch.float32, device=self.device)
            u = torch.stack(
                [
                    torch.as_tensor(ux_np[0, :, :, 0], dtype=torch.float32, device=self.device),
                    torch.as_tensor(uy_np[0, :, :, 0], dtype=torch.float32, device=self.device),
                ],
                dim=-1,
            )
            return rho, u
        rho = self.f.sum(dim=-1)
        c = D2Q9.c.to(dtype=self.f.dtype, device=self.device)
        momentum = torch.einsum("...q,qa->...a", self.f, c)
        u = momentum / rho.unsqueeze(-1).clamp_min(1e-12)
        return rho, u

    def step(self) -> None:
        if self._aa is not None:
            from .aa_layout import equilibrium_d2q9

            f = self._aa.to_dense()
            rho_np = f.sum(axis=-1)
            ux_np = (f * AA_DIRS[:, 0]).sum(axis=-1) / rho_np
            uy_np = (f * AA_DIRS[:, 1]).sum(axis=-1) / rho_np
            feq = equilibrium_d2q9(rho_np, ux_np, uy_np)
            self._aa.push(f - (f - feq) / self.tau)
            self._parity ^= 1
            self.f = self._aa.to_dense()[0, :, :, 0, :]
            return
        rho, u = self.macroscopic()
        feq = equilibrium(rho, u)
        f_post_collision = self.f - (self.f - feq) / self.tau
        self.f = stream(f_post_collision)

    def run(self, steps: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(steps):
            self.step()
        return self.macroscopic()
