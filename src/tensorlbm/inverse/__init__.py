"""tensorlbm.inverse — differentiable inverse-problem optimisation infrastructure.

This package provides the reusable building blocks for inverse problems in
TensorLBM:

- :mod:`~.optimizer` — generic Adam + L-BFGS polish loop with convergence logging
- :mod:`~.losses` — composable loss functions (relative MSE, weighted MSE, L1)
- :mod:`~.convergence` — machine-readable convergence JSON schema and recorder
- :mod:`~.seed` — deterministic seeding and jittered initialisation

The Taylor-Green tau-recovery demo (:mod:`tensorlbm.opt_loop_tau_recovery`)
is the reference consumer of this infrastructure.

Quick start
-----------
::

    import torch.nn as nn
    from tensorlbm.inverse import (
        optimise_loop,
        LossComposition,
        default_relative_mse,
        ConvergenceLog,
        seed_everything,
        jittered_init,
    )

    seed_everything(42)

    # Your differentiable forward model
    log_nu = nn.Parameter(torch.tensor(math.log(0.1), requires_grad=True))
    amplitude = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward_fn():
        tau = 0.5 + 3.0 * torch.exp(log_nu)
        sim = my_solver(tau, amplitude)
        return sim, observations

    log = ConvergenceLog(problem="my_inverse", seed=42)
    result = optimise_loop(
        params=[log_nu, amplitude],
        param_names=["log_nu", "amplitude"],
        loss_fn=default_relative_mse(),
        forward_fn=forward_fn,
        iters=20,
        lr=0.05,
        polish=10,
        box_constraints={"log_nu": (math.log(0.001), math.log(1.0))},
        convergence_log=log,
    )
    log.write("artifacts/my_inverse_convergence.json")
    print(f"Recovered nu = {torch.exp(log_nu).item():.6f}")
"""

from __future__ import annotations

from .convergence import CONVERGENCE_SCHEMA, ConvergenceLog, IterationRecord, write_convergence_json
from .losses import (
    LossComponent,
    LossComposition,
    LossFn,
    default_relative_mse,
    l1_loss,
    relative_mse_loss,
    weighted_mse_loss,
)
from .optimizer import OptimiseResult, optimise_loop
from .seed import jittered_init, seed_everything

__all__ = [
    "CONVERGENCE_SCHEMA",
    "ConvergenceLog",
    "IterationRecord",
    "LossComponent",
    LossComposition,
    "LossFn",
    "OptimiseResult",
    "default_relative_mse",
    "jittered_init",
    "l1_loss",
    "optimise_loop",
    "relative_mse_loss",
    "seed_everything",
    "weighted_mse_loss",
    "write_convergence_json",
]

__version__ = "0.1.0"
