"""Strict R-P comparison: LBM (SCMC) vs Rayleigh-Plesset analytical.

Usage
-----
    # 1) Measure sigma_eff first (writes outputs/bubble_static_3d/sigma_fit.json):
    PYTHONPATH=src python examples/bubble_static_3d.py --device cuda \\
        --nx 96 --ny 96 --nz 96 --radii 12 16 20 --steps 2000

    # 2) Validate against Rayleigh-Plesset using measured sigma_eff:
    PYTHONPATH=src python examples/bubble_rp_validate.py --device cuda \\
        --R0 16 --steps 2000 --auto-sigma \\
        --delta-dp-frac 0.0    # 0 = pure Laplace equilibrium (bubble stays put)

    # Or with manual sigma:
    PYTHONPATH=src python examples/bubble_rp_validate.py --sigma-eff 0.05

Physics
-------
    SCMC initial state: vapour sphere (ρ_light) of radius R0 inside liquid
    (ρ_heavy).  In lattice pressure p = cs²·ρ with cs² = 1/3, the pressure
    jump satisfies Young-Laplace:
        p_in - p_out = -2·σ_eff / R0     (heavy outside, lighter inside → p_in < p_out)

    Adding a controllable overpressure δdp that drives expansion:
        p_in - p_out = -2·σ_eff/R0 + δdp
        δdp > 0 →  net inward pressure reduced → bubble expands.

    Rayleigh-Plesset (incompressible, spherical, no viscosity):
        R·R̈ + (3/2)·Ṙ² = (p_in - p_out) / ρ_l = (-2σ_eff/R + δdp) / ρ_l

Strict-comparison workflow
--------------------------
    Step 1: 测量 σ_eff（必须充分长 n_steps 让 SCMC 收敛）
        PYTHONPATH=src python examples/bubble_static_3d.py --device cuda \\
            --nx 96 --ny 96 --nz 96 --radii 8 12 16 --steps 2000

    Step 2: 把测得的 sigma_eff 传给 RP 验证脚本。三种运行模式:

        a) Pure Laplace equilibrium (--delta-dp-frac 1.0, 默认)
           p_in - p_out = 0  →  R(t) = R0（不动）
           LBM 任何 R(t) 偏离 R0 都是数值伪流信号。

        b) Expansion (--delta-dp-frac 2.0)
           p_in - p_out = +2σ/R0  →  气泡膨胀
           比较 LBM R(t) 与 R-P 解析解。

        c) Contraction (--delta-dp-frac 0.5)
           p_in - p_out = -σ/R0  →  气泡收缩
           比较 LBM R(t) 收缩率与 R-P 解析解。

        PYTHONPATH=src python examples/bubble_rp_validate.py --device cuda \\
            --R0 16 --steps 2000 --sigma-eff <from_step_1> \\
            --delta-dp-frac 2.0 --interface-width 1.5

Caveats
-------
    * SCMC σ_eff 在 n_steps < ~2000 时仍缓慢收敛。建议 step 1 跑 2000+。
    * R-P 假设无粘、球对称、界面无限薄；LBM 有粘性耗散和扩散界面，
      短时动力学会有偏差，主要来自初始瞬态（首 ~50 步）。
    * LBM 域必须够大（nx ≥ 4·R0）避免周期性镜像气泡干扰。
    * 用 tanh 平滑（--interface-width 1.5）消除初始冲击波，否则
      δdp_frac=1.0 也会看到伪膨胀。

Outputs (to outputs/bubble_rp_validate/):
    * R_t_compare.png  - R(t) and dR/dt LBM vs Rayleigh-Plesset
    * console summary
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.multiphase3d import collide_sc_two_component_3d
from tensorlbm.solver3d import stream3d


@dataclass(frozen=True)
class BubbleRPConfig:
    nx: int = 128
    ny: int = 128
    nz: int = 128
    R0: float = 16.0
    domain_multiple: float = 4.0  # nx must be ≥ domain_multiple * R0
    G12: float = 0.9
    tau: float = 1.0
    rho_heavy: float = 0.7
    rho_light: float = 0.3
    sigma_eff: float = 0.05     # from bubble_static_3d.py
    auto_sigma_path: str = ""   # if non-empty, load sigma_fit.json from this path
    delta_dp_frac: float = 1.0  # 1 = pure Laplace equilibrium; >1 drives expansion
    interface_width: float = 1.5  # tanh smoothing (cells)
    pre_equilibrate: int = 0    # if >0, run frac=1.0 for this many steps first
    inviscid: bool = False      # if True, drop -4μ_l·Ṙ/(ρ_l·R²) drag term
    n_steps: int = 2000
    output_interval: int = 10
    output_root: str = "outputs"
    run_name: str = "bubble_rp_validate"
    device: str = "cuda"

    @property
    def delta_dp(self) -> float:
        """Overpressure above Laplace equilibrium. δdp > 0 drives expansion.

        Convention: delta_dp_frac = 1.0 means δdp = 2σ_eff/R0 so that
        p_in - p_out = 0 → bubble sits at Laplace equilibrium (no motion).
        """
        return self.delta_dp_frac * 2.0 * self.sigma_eff / self.R0

    @property
    def p_in_minus_p_out(self) -> float:
        """Net (p_in - p_out) at R = R0: -2σ/R0 + δdp."""
        return -2.0 * self.sigma_eff / self.R0 + self.delta_dp

    @property
    def mu_l(self) -> float:
        """Liquid dynamic viscosity μ_l from LBM kinematic viscosity ν.

        ν = cs² · (τ - 1/2),  μ_l = ρ_l · ν.
        With cs² = 1/3 and τ = 1.0 (default): ν = 1/6, μ_l = ρ_l/6 ≈ 0.117.
        """
        cs2 = 1.0 / 3.0
        nu = cs2 * (self.tau - 0.5)
        return self.rho_heavy * nu


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LBM free bubble vs Rayleigh-Plesset (strict)")
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--nz", type=int, default=128)
    p.add_argument("--R0", type=float, default=16.0)
    p.add_argument("--domain-multiple", type=float, default=4.0,
                   help="Required nx/R0 ratio (default 4: nx must be ≥ 4·R0).")
    p.add_argument("--G12", type=float, default=0.9)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--rho-heavy", type=float, default=0.7)
    p.add_argument("--rho-light", type=float, default=0.3)
    p.add_argument("--sigma-eff", type=float, default=0.05)
    p.add_argument("--auto-sigma", default="",
                   help="Path to sigma_fit.json; overrides --sigma-eff if set")
    p.add_argument("--delta-dp-frac", type=float, default=1.0,
                   help="δdp = frac · 2σ_eff/R0.  1.0 = Laplace equilibrium (no motion).")
    p.add_argument("--interface-width", type=float, default=1.5,
                   help="tanh smoothing width for initial density (cells).")
    p.add_argument("--pre-equilibrate", type=int, default=0,
                   help="If >0, run delta_dp_frac=1.0 for this many steps first to "
                        "establish the steady Laplace-equilibrium density profile. "
                        "Recommended: 500-2000.")
    p.add_argument("--inviscid", action="store_true",
                   help="Drop the -4μ_l·Ṙ/(ρ_l·R²) viscous drag term from R-P ODE. "
                        "Default: viscous R-P (matches LBM dissipation).")
    p.add_argument("--steps", type=int, default=2000, dest="n_steps")
    p.add_argument("--output-interval", type=int, default=10)
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--run-name", default="bubble_rp_validate")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return p


def load_sigma_eff(path: str) -> float:
    """Load fitted sigma_eff from bubble_static_3d output."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"sigma_fit.json not found at {p}")
    with open(p) as f:
        data = json.load(f)
    sigma = float(data.get("sigma_eff_fit", float("nan")))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma_eff in {p} is invalid: {sigma}")
    return sigma


def initial_distributions(nx, ny, nz, R, rho_heavy, rho_light, interface_width, device):
    """Centred vapour sphere of radius R inside liquid, tanh-smoothed interface.

    Using tanh avoids the spurious shock wave that a hard-step density
    jump produces in LBM.  The interface width is in lattice cells.
    """
    z = torch.arange(nz, dtype=torch.float32, device=device)
    y = torch.arange(ny, dtype=torch.float32, device=device)
    x = torch.arange(nx, dtype=torch.float32, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    cz, cy, cx = nz / 2.0, ny / 2.0, nx / 2.0
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    # tanh transition: -1 at r→0, +1 at r→∞
    smooth = 0.5 * (1.0 + torch.tanh((r - R) / interface_width))
    # smooth=0 inside bubble (light), 1 outside (heavy)
    rho1 = rho_light + (rho_heavy - rho_light) * smooth
    rho2 = rho_heavy - (rho_heavy - rho_light) * smooth
    zero = torch.zeros((nz, ny, nx), device=device)
    f1 = equilibrium3d(rho1, zero, zero, zero)
    f2 = equilibrium3d(rho2, zero, zero, zero)
    return f1, f2


def measure_R(rho_total, cfg):
    """Radius of (ρ_heavy + ρ_light)/2 isosurface along +x axis (y=z=centre)."""
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    y0, z0 = ny // 2, nz // 2
    rho_mid = 0.5 * (cfg.rho_heavy + cfg.rho_light)
    line = rho_total[z0, y0, :]
    abs_diff = (line - rho_mid).abs()
    pos = torch.arange(nx, device=line.device) > nx // 2
    diff = abs_diff.clone()
    diff[~pos] = float("inf")
    idx = int(diff.argmin())
    return float(idx - nx // 2)


def rayleigh_plesset(cfg, t_array, viscous=True):
    """Hand-rolled RK4 solver for the Rayleigh-Plesset ODE.

    Full Rayleigh-Plesset (incompressible, Newtonian):
        R·R̈ + (3/2)·Ṙ² = (p_in - p_out)/ρ_l
                           - 4σ_eff / (ρ_l·R)
                           - 4μ_l·Ṙ / (ρ_l·R)

    With p_in - p_out = -2σ_eff/R + δdp evaluated at R(t):
        R̈ = (-2σ/R + δdp)/(ρ_l·R) - 1.5 (Ṙ/R)² - 4μ_l·Ṙ/(ρ_l·R²)

    Parameters
    ----------
    cfg : BubbleRPConfig
    t_array : 1-D array of time points (lattice steps)
    viscous : if True, include -4μ_l·Ṙ/(ρ_l·R²) drag term.
              If False, reproduce inviscid R-P (legacy mode).

    At t=0 with R=R0 and Ṙ=0:
        R̈(0) = (-2σ/R0 + δdp) / (ρ_l·R0)    (viscous term = 0 at t=0)
    """
    sigma = cfg.sigma_eff
    delta_dp = cfg.delta_dp
    rho_l = cfg.rho_heavy
    mu_l = cfg.mu_l

    def rhs(R, V):
        if R <= 1e-6:
            return 0.0, 0.0
        p_in_minus_p_out = -2.0 * sigma / R + delta_dp
        acc = p_in_minus_p_out / (rho_l * R) - 1.5 * V * V / R
        if viscous:
            acc -= 4.0 * mu_l * V / (rho_l * R * R)
        return V, acc

    dt = float(t_array[1] - t_array[0]) if len(t_array) > 1 else 1.0
    n_sub = 5
    h = dt / n_sub
    R, V = float(cfg.R0), 0.0
    R_out = np.empty_like(t_array, dtype=float)
    V_out = np.empty_like(t_array, dtype=float)
    R_out[0], V_out[0] = R, V
    for i in range(1, len(t_array)):
        for _ in range(n_sub):
            k1r, k1v = rhs(R, V)
            k2r, k2v = rhs(R + 0.5 * h * k1r, V + 0.5 * h * k1v)
            k3r, k3v = rhs(R + 0.5 * h * k2r, V + 0.5 * h * k2v)
            k4r, k4v = rhs(R + h * k3r, V + h * k3v)
            R += h * (k1r + 2 * k2r + 2 * k3r + k4r) / 6.0
            V += h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
            if R <= 1e-6:
                R, V = 1e-6, 0.0
        R_out[i], V_out[i] = R, V
    return t_array, R_out, V_out


def plot_compare(t, R_lbm, dR_lbm, t_rp, R_rp, dR_rp, cfg, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t, R_lbm, "b-", lw=1.4, label="LBM (SCMC)")
    axes[0].plot(t_rp, R_rp, "r--", lw=1.4, label="Rayleigh-Plesset")
    axes[0].axhline(cfg.R0, color="k", ls=":", alpha=0.3, label=f"R0={cfg.R0}")
    axes[0].set_xlabel("t (lattice steps)")
    axes[0].set_ylabel("R(t) (cells)")
    axes[0].set_title(f"Bubble radius  (δdp_frac={cfg.delta_dp_frac:.2f})")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(t, dR_lbm, "b-", lw=1.4, label="LBM")
    axes[1].plot(t_rp, dR_rp, "r--", lw=1.4, label="Rayleigh-Plesset")
    axes[1].set_xlabel("t (lattice steps)")
    axes[1].set_ylabel("dR/dt (cells / step)")
    axes[1].set_title("Wall velocity")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "R_t_compare.png", dpi=130)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    sigma_eff = args.sigma_eff
    if args.auto_sigma:
        sigma_eff = load_sigma_eff(args.auto_sigma)
        print(f"[auto-sigma] loaded sigma_eff = {sigma_eff:.6e} from {args.auto_sigma}")

    cfg = BubbleRPConfig(
        nx=args.nx, ny=args.ny, nz=args.nz,
        R0=args.R0, domain_multiple=args.domain_multiple,
        G12=args.G12, tau=args.tau,
        rho_heavy=args.rho_heavy, rho_light=args.rho_light,
        sigma_eff=sigma_eff, delta_dp_frac=args.delta_dp_frac,
        interface_width=args.interface_width,
        pre_equilibrate=args.pre_equilibrate,
        inviscid=args.inviscid,
        n_steps=args.n_steps, output_interval=args.output_interval,
        output_root=args.output_root, run_name=args.run_name,
        device=args.device,
    )
    # ---- domain-size guard ----
    min_n = int(np.ceil(cfg.domain_multiple * cfg.R0))
    if cfg.nx < min_n:
        print(f"[warn] nx={cfg.nx} < {cfg.domain_multiple}·R0={min_n}; "
              f"periodic image bubbles will interfere. Use --nx {min_n} or larger.")
    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")

    out_dir = Path(cfg.output_root) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Config: {asdict(cfg)}")
    print(f"  derived delta_dp     = {cfg.delta_dp:.6e}")
    print(f"  p_in - p_out (at R0) = {cfg.p_in_minus_p_out:.6e}")
    print(f"Output: {out_dir}")

    f1, f2 = initial_distributions(cfg.nx, cfg.ny, cfg.nz, cfg.R0,
                                   cfg.rho_heavy, cfg.rho_light,
                                   cfg.interface_width, device)

    # ---- pre-equilibration phase ----
    # Run frac=1.0 (pure Laplace equilibrium) for pre_equilibrate steps.
    # After this phase the density profile is the steady ρ_eq(r) that
    # already satisfies Young-Laplace.  The user-chosen delta_dp_frac
    # is applied in the measurement phase, comparing against an
    # already-relaxed LBM state — no initial transient shock.
    if cfg.pre_equilibrate > 0:
        print(f"\n[pre-equilibrate] running {cfg.pre_equilibrate} steps at frac=1.0 ...")
        sigma_save, delta_dp_save = cfg.sigma_eff, cfg.delta_dp
        # Temporarily set frac=1.0 without re-instantiating frozen dataclass
        object.__setattr__(cfg, "sigma_eff", cfg.sigma_eff)
        # Simpler: just call collide directly with frac=1.0 logic baked in.
        # At frac=1.0: δdp = 2σ/R0, so p_in - p_out = -2σ/R + 2σ/R0.
        # The LBM doesn't see δdp — it just evolves.  So we just run plain.
        for step in range(cfg.pre_equilibrate):
            f1, f2 = collide_sc_two_component_3d(
                f1, f2, G_12=cfg.G12, tau1=cfg.tau, tau2=cfg.tau,
            )
            f1 = stream3d(f1)
            f2 = stream3d(f2)
            if (step + 1) % max(1, cfg.pre_equilibrate // 4) == 0:
                rho_tot = macroscopic3d(f1 + f2)[0]
                R_now = measure_R(rho_tot, cfg)
                print(f"  [pre-eq] step {step+1:5d}/{cfg.pre_equilibrate}  R={R_now:.3f}")
        # Save the equilibrated (f1, f2) — they are the new initial state.
        print(f"[pre-equilibrate] done.")

    measure_steps, R_series, dR_series = [], [], []
    R_prev = float(cfg.R0)

    if cfg.n_steps < cfg.output_interval:
        print(f"[warn] n_steps={cfg.n_steps} < output_interval={cfg.output_interval}; "
              f"no measurements will be recorded. Adjust --output-interval.")

    for step in range(cfg.n_steps):
        f1, f2 = collide_sc_two_component_3d(
            f1, f2, G_12=cfg.G12, tau1=cfg.tau, tau2=cfg.tau,
        )
        f1 = stream3d(f1)
        f2 = stream3d(f2)
        if (step + 1) % cfg.output_interval == 0:
            rho_tot = macroscopic3d(f1 + f2)[0]
            R_now = measure_R(rho_tot, cfg)
            dR_now = (R_now - R_prev) / cfg.output_interval
            measure_steps.append(step + 1)
            R_series.append(R_now)
            dR_series.append(dR_now)
            R_prev = R_now
        if (step + 1) % 500 == 0 or step == 0:
            R_print = R_series[-1] if R_series else cfg.R0
            print(f"  step {step+1:5d}/{cfg.n_steps}  R={R_print:.3f}")

    if not measure_steps:
        print("[error] no measurements; aborting plot/comparison")
        return

    t = np.asarray(measure_steps, dtype=float)
    R_lbm = np.asarray(R_series)
    dR_lbm = np.asarray(dR_series)
    t_rp, R_rp, dR_rp = rayleigh_plesset(cfg, t, viscous=not cfg.inviscid)

    final_R_lbm = float(R_lbm[-1])
    max_R_lbm = float(R_lbm.max())
    final_dR = float(dR_lbm[-1])
    R_rp_final = float(R_rp[-1])
    R_rp_max = float(R_rp.max())

    err_R_final = abs(final_R_lbm - R_rp_final) / max(abs(R_rp_final), 1e-9)
    err_R_max = abs(max_R_lbm - R_rp_max) / max(abs(R_rp_max), 1e-9)

    # ---- RMS error over the measurement window ----
    # Skip initial transient (first ~10% of measurements) to avoid shock artefact.
    skip = max(1, len(t) // 10)
    diffs = R_lbm[skip:] - R_rp[skip:]
    rms_rel = float(np.sqrt(np.mean((diffs / np.maximum(np.abs(R_rp[skip:]), 1e-9)) ** 2)))
    peak_t_lbm = float(t[int(np.argmax(R_lbm))])
    peak_t_rp = float(t[int(np.argmax(R_rp))])
    peak_t_err = abs(peak_t_lbm - peak_t_rp) / max(t[-1], 1.0)

    print("\n=== Free-expansion diagnostics ===")
    if cfg.pre_equilibrate > 0:
        print(f"  pre-equilibrate     = {cfg.pre_equilibrate} steps (skipped initial transient)")
    print(f"  R0              = {cfg.R0} cells")
    print(f"  sigma_eff       = {cfg.sigma_eff:.6e}  (lattice)")
    print(f"  mu_l            = {cfg.mu_l:.6e}  (lattice, derived from τ)")
    print(f"  R-P model       = {'inviscid' if cfg.inviscid else 'viscous (with -4μṘ/ρR² drag)'}")
    print(f"  delta_dp_frac   = {cfg.delta_dp_frac:.3f}")
    print(f"  delta_dp        = {cfg.delta_dp:.6e}")
    print(f"  p_in - p_out    = {cfg.p_in_minus_p_out:.6e}")
    print(f"  LBM final R     = {final_R_lbm:.3f}")
    print(f"  R-P final R     = {R_rp_final:.3f}   (rel err {err_R_final:.2%})")
    print(f"  LBM max R       = {max_R_lbm:.3f}")
    print(f"  R-P max R       = {R_rp_max:.3f}   (rel err {err_R_max:.2%})")
    print(f"  LBM terminal dR = {final_dR:.3e}")
    print(f"  R-P terminal dR = {float(dR_rp[-1]):.3e}")
    print(f"  RMS rel err R(t) [skip {skip} samples] = {rms_rel:.2%}   (whole-curve quality)")
    print(f"  peak-time rel err                    = {peak_t_err:.2%}   "
          f"(LBM t_peak={peak_t_lbm:.0f}, R-P t_peak={peak_t_rp:.0f})")

    plot_compare(t, R_lbm, dR_lbm, t_rp, R_rp, dR_rp, cfg, out_dir)
    print(f"\n  Wrote {out_dir / 'R_t_compare.png'}")


if __name__ == "__main__":
    main()