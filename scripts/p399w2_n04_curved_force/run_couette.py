"""Analytic Stokes check of the curved-surface *traction* reconstruction.

Taylor–Couette flow between two concentric circular immersed boundaries has an
exact solution,

    u_θ(r) = A r + B / r,   A = −Ω R₁²/(R₂² − R₁²),  B = Ω R₁² R₂²/(R₂² − R₁²),

a *constant* wall shear on the inner cylinder,

    τ_θ(R₁) = μ (du_θ/dr − u_θ/r)|_{R₁} = −2 μ B / R₁²,

and the exact torque per unit length `T = −4π μ B = 4π μ Ω R₁² R₂²/(R₂² − R₁²)`.

Because both the traction *field* and its integral are known in closed form, this
case tests the two things the measure-consistent quadrature is supposed to
deliver and that a total-force check alone cannot separate:

* the reconstructed traction `τ_k = F_k / Δs_k` at each marker (pointwise), and
* the surface integral `∮ τ dS` / `∮ r × τ dS`.

`Re_Ω = Ω R₁ (R₂ − R₁)/ν ≪ 1` keeps the flow in the Stokes regime, where the
exact solution is valid.

    PYTHONPATH=src python scripts/p399w2_n04_curved_force/run_couette.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_validation import _git_revision  # noqa: E402

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.ibm_curved_force import circle_quadrature, curved_surface_forcing, polyline_arc_weights
from tensorlbm.ibm_force_transfer import body_source_2d, build_stencil, interior_mask_force
from tensorlbm.solver import collide_bgk, stream

torch.set_default_dtype(torch.float64)


def run_couette(
    *,
    n: int = 96,
    r1: float = 20.0,
    r2: float = 38.0,
    omega: float = 2e-4,
    nu: float = 0.5,
    spacing: float = 0.8,
    n_steps: int = 9000,
    n_transient: int = 4000,
) -> dict:
    tau = 0.5 + 3.0 * nu
    centre = (n / 2.0, n / 2.0)
    inner = circle_quadrature(centre, r1, max(16, int(round(2 * math.pi * r1 / spacing))))
    outer = circle_quadrature(centre, r2, max(24, int(round(2 * math.pi * r2 / spacing))))
    st_in = build_stencil(inner.points, (n, n), kernel="4pt")
    st_out = build_stencil(outer.points, (n, n), kernel="4pt")

    yy, xx = torch.meshgrid(
        torch.arange(n, dtype=torch.float64) - centre[1],
        torch.arange(n, dtype=torch.float64) - centre[0],
        indexing="ij",
    )
    r = torch.sqrt(xx * xx + yy * yy)
    solid = r < r1  # inner cylinder interior
    # rigid-body rotation of the inner cylinder (interior + surface)
    ub_in = torch.stack([-omega * yy, omega * xx], dim=-1).reshape(-1, 2)
    ub_in = ub_in.reshape(n, n, 2)[solid].reshape(-1, 2)
    u_in_target = torch.stack(
        [
            -omega * (inner.points[:, 1] - centre[1]),
            omega * (inner.points[:, 0] - centre[0]),
        ],
        dim=1,
    )
    u_out_target = torch.zeros((outer.n_markers, 2), dtype=torch.float64)

    rho0 = torch.ones((n, n), dtype=torch.float64)
    ux0 = torch.zeros((n, n), dtype=torch.float64)
    uy0 = torch.zeros((n, n), dtype=torch.float64)
    f = equilibrium(rho0, ux0, uy0)

    A = -omega * r1**2 / (r2**2 - r1**2)
    B = omega * r1**2 * r2**2 / (r2**2 - r1**2)
    mu = nu
    tau_exact = -2.0 * mu * B / r1**2  # tangential wall shear on r = R1 (signed)
    torque_exact = 4.0 * math.pi * mu * omega * r1**2 * r2**2 / (r2**2 - r1**2)
    q_scale = mu * omega * r1  # traction reference μ Ω R1

    trac_samples: list[list[float]] = []
    torque_samples: list[float] = []
    profile_err: list[float] = []
    tmp_samples: list[float] = []
    for step in range(n_steps):
        rho_f, uxf, uyf = macroscopic(f)
        u_grid = torch.stack([uxf, uyf])
        res_in = curved_surface_forcing(
            st_in, u_grid, u_in_target, inner, rho_grid=rho_f,
            n_correctors=2, n_normal_passes=1,
        )
        res_out = curved_surface_forcing(
            st_out, u_grid, u_out_target, outer, rho_grid=rho_f,
            n_correctors=2, n_normal_passes=1,
        )
        # interior of the inner cylinder rotates rigidly with the body
        u_solid = torch.stack([-omega * yy, omega * xx])
        fi = torch.where(solid.unsqueeze(0), u_solid - u_grid, torch.zeros_like(u_grid))
        fx = res_in.force_grid[0] + res_out.force_grid[0] + fi[0]
        fy = res_in.force_grid[1] + res_out.force_grid[1] + fi[1]
        f = collide_bgk(f, tau)
        f = body_source_2d(f, fx, fy)
        f = stream(f)

        if step >= n_transient and step % 200 == 0:
            # tangential traction on the inner cylinder (exact constant)
            rel = inner.points - torch.tensor(centre)
            tang = torch.stack([-rel[:, 1], rel[:, 0]], dim=1) / r1
            tt = (res_in.traction * tang).sum(dim=1)
            trac_samples.append([float(v) for v in tt.tolist()])
            torque_samples.append(float(inner.torque(res_in.marker_force, about=centre).item()))
            tmp_samples.append(res_in.traction.abs().max().item())
            # azimuthal velocity profile in the gap
            rho_n, uxn, uyn = macroscopic(f)
            th = torch.atan2(yy, xx)
            u_th = -uxn * torch.sin(th) + uyn * torch.cos(th)
            band = (r > r1 + 1.5) & (r < r2 - 1.5)
            u_ex = A * r + B / r
            profile_err.append(
                float((u_th[band] - u_ex[band]).abs().max().item()) / (omega * r1)
            )

    trac = torch.tensor(trac_samples)
    mean_trac = trac.mean(dim=0)
    # The reconstructed traction is the *load on the fluid* (reaction on the
    # body is −τ), so the sign is opposite to the analytic wall shear acting on
    # the body; amplitudes are what is compared.
    trac_err = float((mean_trac.abs() - abs(tau_exact)).abs().max().item()) / abs(tau_exact)
    torque_mean = float(torch.tensor(torque_samples).mean().item())
    torque_err = abs(abs(torque_mean) - torque_exact) / torque_exact
    return {
        "label": "couette_torque",
        "params": {
            "n": n, "r1": r1, "r2": r2, "omega": omega, "nu": nu, "tau": tau,
            "re_omega": omega * r1 * (r2 - r1) / nu,
            "n_markers_inner": inner.n_markers, "n_markers_outer": outer.n_markers,
            "measure_inner_rel_error": inner.measure_rel_error,
            "n_steps": n_steps, "n_transient": n_transient,
        },
        "metrics": {
            "traction_mean_inner": float(mean_trac.mean().item()),
            "traction_exact_inner": tau_exact,
            "traction_pointwise_rel_error": trac_err,
            "traction_rel_std_over_surface": float(mean_trac.std().item()) / abs(tau_exact),
            "traction_sign_convention": "load on the fluid (reaction on the body is −τ)",
            "torque_mean": torque_mean,
            "torque_exact": torque_exact,
            "torque_rel_error": torque_err,
            "profile_max_rel_error": max(profile_err) if profile_err else None,
            "traction_normalised_by_mu_omega_r1": float(mean_trac.mean().item()) / q_scale,
        },
        "series": {
            "traction_samples": trac_samples[::5],
            "torque_samples": torque_samples,
            "profile_error_samples": profile_err,
        },
        "provenance": {"git_revision": _git_revision()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/p399w2_n04/case_couette_torque.json")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--transient", type=int, default=4000)
    args = ap.parse_args()
    res = run_couette(n_steps=args.steps, n_transient=args.transient)
    path = Path(args.out)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    print(f"[couette] {path}")
    print(json.dumps(res["metrics"], indent=1))


if __name__ == "__main__":
    main()