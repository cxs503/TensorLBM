"""Simple LBM simulation entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .constants import D2Q9
from .lattice import equilibrium, stream


class LBMSimulation:
    """Minimal D2Q9 BGK simulation on a 2D periodic domain.

    Long runs can be made restartable with :meth:`enable_checkpointing`, which
    attaches the MF-LBM-style job-status + wall-clock checkpoint state machine
    from :mod:`tensorlbm.job_checkpoint`.  The state machine is entirely optional
    and off by default, so plain ``LBMSimulation`` usage is unchanged.
    """

    def __init__(self, nx: int = 64, ny: int = 32, tau: float = 0.6, device: str = "cpu") -> None:
        if nx <= 0 or ny <= 0:
            raise ValueError(f"nx and ny must be positive integers (got nx={nx}, ny={ny})")
        if tau <= 0:
            raise ValueError(f"tau must be positive (got tau={tau})")
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.device = torch.device(device)

        rho0 = torch.ones((ny, nx), dtype=torch.float32, device=self.device)
        u0 = torch.zeros((ny, nx, 2), dtype=torch.float32, device=self.device)
        self.f = equilibrium(rho0, u0)
        self.step_count = 0
        self.checkpoints: Any | None = None
        self.finished = False

    # -- restart support ------------------------------------------------------
    def enable_checkpointing(
        self,
        directory: str | Path,
        *,
        interval_hours: float = 1.0,
        max_steps: int | None = None,
        volume_interval: float | None = None,
        dual_backup: bool = True,
        force_restart: bool = False,
        clock=None,
    ):
        """Attach the job-status/checkpoint state machine and resume if asked to.

        Returns the :class:`tensorlbm.job_checkpoint.ResumeState`; when it is a
        resume, the populations and step counter are restored *before* the first
        new step, so no step is recomputed (MF-LBM ``continue`` path).
        """
        from .job_checkpoint import JobCheckpointManager

        manager = JobCheckpointManager(
            directory,
            interval_hours=interval_hours,
            max_steps=max_steps,
            volume_interval=volume_interval,
            dual_backup=dual_backup,
            config_signature={"nx": self.nx, "ny": self.ny, "tau": self.tau, "lattice": "D2Q9"},
            **({} if clock is None else {"clock": clock}),
        )
        state = manager.begin(force_restart=force_restart)
        if state.resumed:
            self.f = state.fields["f"].to(device=self.f.device, dtype=self.f.dtype)
            self.step_count = state.step
        else:
            self.step_count = state.step
        self.checkpoints = manager
        self.finished = state.finished
        return state

    def run_with_checkpointing(self, steps: int) -> tuple[int, int]:
        """Advance up to ``steps`` new steps under the attached state machine.

        Returns ``(steps_executed, checkpoints_written)``.
        """
        if self.checkpoints is None:
            raise RuntimeError("enable_checkpointing() must be called first")
        from .job_checkpoint import run_with_checkpointing

        return run_with_checkpointing(self, steps, self.checkpoints)

    # -- physics --------------------------------------------------------------
    def macroscopic(self) -> tuple[torch.Tensor, torch.Tensor]:
        rho = self.f.sum(dim=-1)
        c = D2Q9.c.to(dtype=self.f.dtype, device=self.device)
        momentum = torch.einsum("...q,qa->...a", self.f, c)
        u = momentum / rho.unsqueeze(-1).clamp_min(1e-12)
        return rho, u

    def step(self) -> None:
        if self.finished:
            raise RuntimeError("simulation already reached a terminal checkpoint state")
        rho, u = self.macroscopic()
        feq = equilibrium(rho, u)
        f_post_collision = self.f - (self.f - feq) / self.tau
        self.f = stream(f_post_collision)
        self.step_count += 1
        if self.checkpoints is not None:
            self.checkpoints.maybe_checkpoint(self.step_count, {"f": self.f})
            if self.checkpoints.should_stop(self.step_count):
                self.checkpoints.finish(
                    self.step_count,
                    status="reached_max_step",
                    reason=f"max_steps={self.checkpoints.policy.max_steps}",
                    fields={"f": self.f},
                )
                self.finished = True

    def run(self, steps: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(steps):
            self.step()
        return self.macroscopic()