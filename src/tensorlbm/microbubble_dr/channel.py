"""Turbulent-channel runner for the microbubble drag-reduction node (P106W2-N01).

One case = one Shan-Chen channel run at **matched bulk velocity**: an integral
controller adjusts the uniform per-volume drive ``F0`` until the volumetric mean
streamwise velocity equals ``U_target``, so the single-phase reference and every
two-phase case are compared at the same throughput (which is what makes
``Delta C_f`` a drag reduction rather than a flow-rate change).

Gas is injected as :math:`N` spherical bubbles of diameter :math:`d` next to the
**top** wall (the "hull plate"); a net buoyancy ``F_y = (rho_l - rho) g_b``
holds the bubbles against it.  The bottom wall is the control wall.

Statistics: (time, x, z)-averaged profiles of ``rho``, ``u``, ``<u'v'>`` and the
gas volume fraction ``alpha = (rho_l - rho)/(rho_l - rho_g)`` together with the
bounce-back momentum-exchange wall shear, which is an independent measurement of
``tau_w`` used to validate the profile-based FIK decomposition.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import torch

from .kernel import ChannelKernel, momentum, sc_force, psi_exp

__all__ = ["ChannelConfig", "coexistence", "probe_coexistence", "init_bubbles", "run_case"]


@dataclass
class ChannelConfig:
    # geometry (lattice units)
    nx: int = 96
    ny: int = 64
    nz: int = 32
    # target flow state
    Re_tau: float = 180.0
    u_tau: float = 0.006
    U_target: float = 0.108          # bulk velocity held by the controller
    # gas
    alpha: float = 0.01              # target gas volume fraction
    d_plus: float = 45.0             # bubble diameter in wall units
    rho_l: float = 1.38
    rho_g: float = 0.30
    G: float = 5.0
    g_b: float = -2.0e-4             # gas buoyancy per unit density deficit (pushes gas to y=0)
    # numerics
    lam_trt: float | None = None          # None -> omega_minus = 1 (robust for SC interfaces)
    rho_l_meas: float = 1.42
    rho_g_meas: float = 0.27
    n_warmup: int = 2500
    n_steps: int = 10000
    sample_every: int = 20
    pump_gain: float = 0.02
    seed: int = 0
    tag: str = "case"

    @property
    def nu(self) -> float:
        """Kinematic viscosity that puts the half-gap at ``Re_tau``."""
        return self.u_tau * (self.ny / 2.0) / self.Re_tau

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def d_lattice(self) -> float:
        """Bubble diameter in lattice units from the wall-unit target."""
        return self.d_plus * self.nu / self.u_tau


def coexistence(G: float = 5.0, rho0: float = 1.0, n: int = 24, steps: int = 4000, tau: float = 0.8) -> dict:
    """Measure the liquid/gas coexistence densities of the SC kernel (periodic box)."""
    torch.set_grad_enabled(False)
    k = ChannelKernel((n, n, n), tau=tau, G=G)
    rho = torch.full((n, n, n), float(rho0))
    # seed a density perturbation so the spinodal instability is not left to round-off
    gen = torch.Generator().manual_seed(7)
    rho = rho * (1.0 + 0.02 * torch.randn((n, n, n), generator=gen))
    u = torch.zeros((3, n, n, n))
    f = k.field(float(rho0), 0.0)
    noise = torch.randn((3, n, n, n), generator=gen) * 1e-3
    from .kernel import equilibrium

    f = equilibrium(rho, noise)
    # periodic box: emulate by using a kernel whose y walls are pushed outside the
    # statistics window -- instead run the SC-only relaxation with periodic rolls
    rho_t = rho.clone()
    for _ in range(steps):
        Fx, Fy, Fz = sc_force(rho_t, G)
        # simple pseudo-time relaxation of the density field (Cahn-Hilliard-like):
        # the SC force drives the separation; a diffusion term regularises it
        lap = (
            torch.roll(rho_t, 1, 0) + torch.roll(rho_t, -1, 0)
            + torch.roll(rho_t, 1, 1) + torch.roll(rho_t, -1, 1)
            + torch.roll(rho_t, 1, 2) + torch.roll(rho_t, -1, 2)
            - 6.0 * rho_t
        )
        div = (
            torch.roll(Fx, -1, 2) - torch.roll(Fx, 1, 2)
            + torch.roll(Fy, -1, 1) - torch.roll(Fy, 1, 1)
            + torch.roll(Fz, -1, 0) - torch.roll(Fz, 1, 0)
        ) * 0.5
        rho_t = rho_t + 0.02 * (div + 0.5 * lap)
        rho_t = torch.clamp(rho_t, 1e-3, 10.0)
    flat = rho_t.flatten().numpy()
    lo = float(np.percentile(flat, 5))
    hi = float(np.percentile(flat, 95))
    return {
        "G": G,
        "rho0": rho0,
        "rho_l": hi,
        "rho_g": lo,
        "density_ratio": hi / max(lo, 1e-12),
        "rho_mean": float(flat.mean()),
        "box": n,
        "steps": steps,
        "method": "SC spinodal relaxation (mean-field proxy for the LBM coexistence curve)",
    }


def probe_coexistence(cfg: ChannelConfig, rho0_list=(1.0, 1.3, 1.6), n: int = 16, steps: int = 1500) -> dict:
    """Measure the liquid/gas coexistence densities **with the production kernel**.

    The alpha field of every case uses these numbers, so they are measured in the
    same Shan-Chen kernel at the same relaxation time and recorded in the
    campaign JSON rather than assumed.
    """
    torch.set_grad_enabled(False)
    lam = cfg.lam_trt if cfg.lam_trt is not None else 0.5 * (cfg.tau - 0.5)
    out = {"tau": cfg.tau, "G": cfg.G, "lam_trt": lam, "box": [n, n * 3 // 2, n], "steps": steps, "points": []}
    for rho0 in rho0_list:
        k = ChannelKernel((n, n * 3 // 2, n), tau=cfg.tau, G=cfg.G, lam_trt=lam)
        gen = torch.Generator().manual_seed(11)
        shape = (n, n * 3 // 2, n)
        rho = torch.full(shape, float(rho0)) * (1 + 0.04 * torch.randn(shape, generator=gen))
        from .kernel import equilibrium

        f = equilibrium(rho, 0.01 * torch.randn((3,) + shape, generator=gen))
        for _ in range(steps):
            f, _d = k.step(f, 0.0)
        rh, _j = momentum(f)
        arr = rh.flatten().numpy()
        lo = float(np.percentile(arr, 0.5))
        hi = float(np.percentile(arr, 99.5))
        out["points"].append({"rho0": rho0, "rho_l": hi, "rho_g": lo,
                              "ratio": hi / max(lo, 1e-9), "mean": float(arr.mean())})
    sep = [p for p in out["points"] if p["ratio"] > 1.5]
    if sep:
        best = max(sep, key=lambda p: p["ratio"])
        out["rho_l"] = best["rho_l"]
        out["rho_g"] = best["rho_g"]
        out["density_ratio"] = best["ratio"]
        out["separated"] = True
    else:  # pragma: no cover
        out["rho_l"] = float(rho0_list[0])
        out["rho_g"] = 0.1 * float(rho0_list[0])
        out["separated"] = False
    return out


def init_bubbles(
    cfg: ChannelConfig,
    rho_l: float,
    rho_g: float,
    gen: torch.Generator,
) -> tuple[torch.Tensor, dict]:
    """Place ``N`` spherical gas bubbles next to the top wall; returns ``(rho, meta)``."""
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    d = cfg.d_lattice
    r = 0.5 * d
    vol_dom = nx * ny * nz
    vol_b = math.pi / 6.0 * d**3
    n_b = int(round(cfg.alpha * vol_dom / vol_b)) if cfg.alpha > 0 else 0
    z = torch.arange(nz).view(-1, 1, 1).float()
    y = torch.arange(ny).view(1, -1, 1).float()
    x = torch.arange(nx).view(1, 1, -1).float()
    field_r = torch.full((nz, ny, nx), float(rho_l))
    centres = []
    for _ in range(n_b):
        cz = float(torch.rand((), generator=gen) * nz)
        cx = float(torch.rand((), generator=gen) * nx)
        # bubbles sit against the BOTTOM wall (y = 0): that wall is the
        # bubble-side wall and the FIK decomposition is exact for its shear.
        # Physically this is the hull-bottom configuration: the gas is pressed
        # against a downward-facing plate by buoyancy.
        cy = float(1.5 * r + 1.0 + torch.rand((), generator=gen) * (1.0 * d))
        dz = torch.abs(z - cz)
        dz = torch.minimum(dz, nz - dz)
        dx = torch.abs(x - cx)
        dx = torch.minimum(dx, nx - dx)
        dist = torch.sqrt(dz**2 + (y - cy) ** 2 + dx**2)
        # tanh interface of ~1 lattice cell width
        prof = 0.5 * (1.0 - torch.tanh((dist - r) / 2.0))
        field_r = torch.minimum(field_r, rho_l - (rho_l - rho_g) * prof)
        centres.append((cz, cy, cx))
    return field_r, {"n_bubbles": n_b, "d_lattice": d, "alpha_geom": n_b * vol_b / vol_dom, "centres": centres}


def _turbulent_perturbation(shape: tuple[int, int, int], amp: float, gen: torch.Generator) -> torch.Tensor:
    """Random smooth perturbation from explicit Fourier modes (no FFT: the FFT
    path corrupted the heap on the loong64 torch build).

    The streamwise component carries the modes; a wall-normal component with a
    quarter-period shape supplies the initial <u'v'> production term that lets
    turbulence self-sustain.
    """
    nz, ny, nx = shape
    u = torch.zeros((3, nz, ny, nx))
    z = torch.arange(nz).view(-1, 1, 1).float()
    y = torch.arange(ny).view(1, -1, 1).float()
    x = torch.arange(nx).view(1, 1, -1).float()
    shape_y = torch.sin(math.pi * (y + 0.5) / ny)
    n_modes = 6
    for _ in range(n_modes):
        kx = 2 * math.pi * float(torch.randint(1, 4, (1,), generator=gen)) / nx
        kz = 2 * math.pi * float(torch.randint(1, 3, (1,), generator=gen)) / nz
        ph = 2 * math.pi * float(torch.rand((), generator=gen))
        mode = torch.cos(kx * x + kz * z + ph) * shape_y
        u[0] += mode
        u[1] += 0.35 * mode.roll(1, 2) * torch.cos(math.pi * y / ny)
    u = u / (u[0].abs().max() + 1e-30)
    return amp * u


def run_case(cfg: ChannelConfig, verbose: bool = False) -> dict[str, Any]:
    """Run one case; returns the record (profiles + diagnostics) -- no plotting here."""
    torch.set_grad_enabled(False)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    lam = cfg.lam_trt if cfg.lam_trt is not None else 0.5 * (cfg.tau - 0.5)  # -> omega_m = 1
    k = ChannelKernel((cfg.nz, cfg.ny, cfg.nx), tau=cfg.tau, G=cfg.G, lam_trt=lam)
    H = float(cfg.ny)
    t0 = time.time()

    rho, meta_b = init_bubbles(cfg, cfg.rho_l, cfg.rho_g, gen)
    u0 = _turbulent_perturbation((cfg.nz, cfg.ny, cfg.nx), amp=0.08 * cfg.U_target, gen=gen)
    u = u0.clone()
    u[0] = u[0] + cfg.U_target
    from .kernel import equilibrium

    f = equilibrium(rho, u)
    mass0 = float(rho.sum())
    # Initial drive from the Dean (1978) correlation -- only an initial guess for
    # the controller, never used as a result.  Starting from the *laminar* value
    # would be hopeless: the viscous response time of the channel is ~1e6 steps
    # at Re_tau = 180, so the controller could not climb the 20x gap.
    Re_b0 = cfg.U_target * H / cfg.nu
    Cf0 = 0.073 * Re_b0 ** (-0.25)
    F0 = 2.0 * (0.5 * cfg.rho_l * cfg.U_target**2 * Cf0) / H

    def plane_mean(t: torch.Tensor) -> np.ndarray:
        return t.mean(dim=(0, 2)).double().numpy()

    acc: dict[str, np.ndarray] = {}
    n_s = 0
    mem_b = []
    mem_t = []
    ux_hist = []
    mass_hist = []
    alpha_hist = []
    rho_hi_acc: list[float] = []
    rho_lo_acc: list[float] = []
    F0_hist = []

    for step in range(cfg.n_warmup + cfg.n_steps):
        f, d = k.step(f, F0, g_b=cfg.g_b, rho_l=cfg.rho_l)
        ux_hist.append(d["ux_mean"])
        F0_hist.append(F0)
        mem_b.append(d["mem_bottom"])
        mem_t.append(d["mem_top"])
        # integral controller on the uniform per-volume drive
        err = cfg.U_target - d["ux_mean"]
        F0 += cfg.pump_gain * (12.0 * cfg.rho_l * cfg.nu / H**2) * err
        F0 = max(F0, 0.0)
        if step >= cfg.n_warmup and (step - cfg.n_warmup) % cfg.sample_every == 0:
            rh, j = momentum(f)
            uu = (j[0] + 0.5 * torch.zeros_like(j[0])) * (1.0 / rh)
            uv = (j[1]) * (1.0 / rh)
            rho_p = rh
            u_p = uu
            v_p = uv
            for name, arr in (
                ("rho", rho_p),
                ("u", u_p),
                ("v", v_p),
                ("uu", (j[0] * j[0]) / (rh * rh)),
                ("uv", (j[0] * j[1]) / (rh * rh)),
                ("vv", (j[1] * j[1]) / (rh * rh)),
            ):
                acc[name] = acc.get(name, 0.0) + plane_mean(arr)
            sub = rh.flatten()[::7].numpy()
            rho_hi = float(np.percentile(sub, 99.0))
            rho_lo = float(np.percentile(sub, 1.0))
            alpha_field = torch.clamp((rho_hi - rh) / max(rho_hi - rho_lo, 1e-9), 0.0, 1.0)
            rho_hi_acc.append(rho_hi)
            rho_lo_acc.append(rho_lo)
            acc["alpha"] = acc.get("alpha", 0.0) + plane_mean(alpha_field)
            n_s += 1
            mass_hist.append(float(rh.sum()))
            alpha_hist.append(float(alpha_field.mean()))
        if verbose and step % 1000 == 0:
            print(
                f"[{cfg.tag}] step {step} F0={F0:.3e} ux={d['ux_mean']:.5f} "
                f"target={cfg.U_target:.5f} mem_b={d['mem_bottom']:.4e} mem_t={d['mem_top']:.4e}",
                flush=True,
            )

    prof = {key: val / max(n_s, 1) for key, val in acc.items()}
    y = np.arange(cfg.ny) + 0.5
    u_bar = prof["u"]
    v_bar = prof["v"]
    uv_re = prof["uv"] - u_bar * v_bar
    rho_bar = prof["rho"]
    rho_b = float(rho_bar.mean())
    record: dict[str, Any] = {
        "cfg": asdict(cfg),
        "shape": [cfg.nz, cfg.ny, cfg.nx],
        "H": H,
        "nu": k.nu,
        "tau": cfg.tau,
        "y": y,
        "rho": rho_bar,
        "u": u_bar,
        "uv": uv_re,
        "uu": prof["uu"] - u_bar**2,
        "vv": prof["vv"] - v_bar**2,
        "alpha": prof["alpha"],
        "rho_b": rho_b,
        "U_b": float(prof["u"].mean()),
        "U_target": cfg.U_target,
        "F0": float(np.mean(F0_hist[-max(n_s, 1) :])),
        "F0_final": F0,
        "mem_bottom": float(np.mean(mem_b[-max(n_s * cfg.sample_every, 1) :])),
        "mem_top": float(np.mean(mem_t[-max(n_s * cfg.sample_every, 1) :])),
        "ux_history": ux_hist[::50],
        "F0_history": F0_hist[::50],
        "mass0": mass0,
        "mass_drift_rel": float(abs(mass_hist[-1] - mass_hist[0]) / mass_hist[0]) if len(mass_hist) > 1 else 0.0,
        "alpha_integral": float(np.mean(alpha_hist)) if alpha_hist else 0.0,
        "rho_l_meas": float(np.mean(rho_hi_acc)) if rho_hi_acc else float("nan"),
        "rho_g_meas": float(np.mean(rho_lo_acc)) if rho_lo_acc else float("nan"),
        "alpha_integral_drift_rel": float(
            abs(alpha_hist[-1] - alpha_hist[0]) / max(alpha_hist[0], 1e-12)
        )
        if len(alpha_hist) > 1
        else 0.0,
        "n_samples": n_s,
        "init": meta_b,
        "wall_units": {
            "dy_plus": cfg.u_tau / k.nu,
            "Lx_plus": cfg.nx * cfg.u_tau / k.nu,
            "Lz_plus": cfg.nz * cfg.u_tau / k.nu,
            "h_plus": (cfg.ny / 2) * cfg.u_tau / k.nu,
        },
        "runtime_s": time.time() - t0,
    }
    # measured friction Reynolds number of the run
    u_tau_meas = math.sqrt(max(record["mem_top"], 0.0) / rho_b) if record["mem_top"] > 0 else 0.0
    record["u_tau_meas"] = u_tau_meas
    record["Re_tau_meas"] = u_tau_meas * (H / 2) / k.nu
    record["Re_bulk"] = rho_b * record["U_b"] * H / (rho_b * k.nu)
    record["d_plus_actual"] = cfg.d_lattice * record["u_tau_meas"] / k.nu
    return record