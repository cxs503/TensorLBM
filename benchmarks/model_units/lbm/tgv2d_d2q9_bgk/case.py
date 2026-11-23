"""Taylor-Green vortex decay, D2Q9 + BGK, fully periodic.

Reference: for ``u = U0 (sin kx cos ky, -cos kx sin ky)`` the incompressible
Navier-Stokes solution keeps the same shape and decays as

    U0(t) = U0(0) * exp(-2 nu k^2 t),      nu = (tau - 1/2)/3,  k = 2 pi / N

so the *amplitude decay rate* is a direct, analytic measurement of the solver's
effective viscosity.  The KPI is the relative error of the measured rate
``lambda = ln(A0/A(T)) / T`` against the analytic ``2 nu k^2``; a second metric
reports the field-level L2 error against the analytic field at ``T``.
"""

from __future__ import annotations

import math

import torch

from tensorlbm import d2q9, solver


def _amplitude(ux: torch.Tensor) -> float:
    """Mode amplitude of ``U0 sin(kx)cos(ky)`` from the velocity field."""
    return 2.0 * float(torch.sqrt((ux * ux).mean()))


def run(params: dict, *, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    dtype = torch.float64
    n = int(params.get("n", 32))
    u0 = float(params.get("u0", 0.04))
    tau = float(params.get("tau", 0.8))
    steps = int(params.get("steps", 400))

    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / n
    x = torch.arange(n, dtype=dtype)
    kx = (k * x).view(1, n)
    ky = (k * x).view(n, 1)
    ux = u0 * torch.sin(kx) * torch.cos(ky)
    uy = -u0 * torch.cos(kx) * torch.sin(ky)
    rho = torch.ones((n, n), dtype=dtype)
    f = d2q9.equilibrium(rho, ux, uy)

    a0 = _amplitude(ux)
    history = []
    for step in range(steps):
        f = solver.collide_bgk(f, tau)
        f = solver.stream(f)
        if (step + 1) % max(1, steps // 8) == 0:
            _, ux_t, _ = d2q9.macroscopic(f)
            history.append([step + 1, _amplitude(ux_t)])

    _, ux_t, uy_t = d2q9.macroscopic(f)
    a_t = _amplitude(ux_t)
    lam_meas = math.log(a0 / a_t) / steps
    lam_ana = 2.0 * nu * k * k
    decay_err = abs(lam_meas - lam_ana) / lam_ana

    a_ana = u0 * math.exp(-lam_ana * steps)
    ux_ana = a_ana * torch.sin(kx) * torch.cos(ky)
    uy_ana = -a_ana * torch.cos(kx) * torch.sin(ky)
    denom = float(torch.sqrt((ux_ana**2 + uy_ana**2).mean()))
    field_l2 = float(torch.sqrt(((ux_t - ux_ana) ** 2 + (uy_t - uy_ana) ** 2).mean()) / denom)
    # discrete incompressibility check: |div u| * N / (2 pi |u|)
    div = (
        torch.roll(ux_t, -1, dims=1)
        - torch.roll(ux_t, 1, dims=1)
        + torch.roll(uy_t, -1, dims=0)
        - torch.roll(uy_t, 1, dims=0)
    ) * 0.5
    div_norm = float(div.abs().max() * n / (2.0 * math.pi) / max(a_t, 1e-30))

    return {
        "kpis": {"decay_rate_rel_err": decay_err},
        "metrics": {
            "n": n,
            "u0": u0,
            "tau": tau,
            "nu": nu,
            "k": k,
            "steps": steps,
            "amplitude_initial": a0,
            "amplitude_final": a_t,
            "amplitude_analytic_final": a_ana,
            "lambda_measured": lam_meas,
            "lambda_analytic": lam_ana,
            "field_l2_rel_err": field_l2,
            "divergence_scaled": div_norm,
            "mach_max": float(a0 * math.sqrt(2.0) / math.sqrt(1.0 / 3.0)),
        },
        "curves": {
            "decay": {
                "step": [h[0] for h in history],
                "amplitude": [h[1] for h in history],
                "analytic": [u0 * math.exp(-lam_ana * h[0]) for h in history],
            },
            "profile": {
                "y": list(range(n)),
                "numeric_ux_at_x0": ux_t[:, 0].tolist(),
                "analytic_ux_at_x0": ux_ana[:, 0].tolist(),
            },
        },
    }
