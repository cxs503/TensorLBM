"""Dam-break validation: front position x*(t) vs Martin & Moyce (1952).

Runs a 2D color-gradient multiphase LBM dam-break simulation and compares
the dimensionless surge-front trajectory against the digitised experimental
data of Martin & Moyce (Phil. Trans. R. Soc. A 244, 312, 1952).

Acceptance criterion
--------------------
Mean absolute relative error of the dimensionless front position at
6 matched time points ≤ 6 %.

Non-dimensionalisation
----------------------
L = dam_width (initial column width, lattice cells)
t* = t · √(2g / L)          (dimensionless time)
X* = x_front / L            (dimensionless front position)

Martin & Moyce use a = L/2 (half-width), T = t·√(g/a), Z = x_front_MM / a.
Since T = t* and x_front_MM = x_front − a, the conversion is:
    Z = 2·X* − 1   ⟹   X*_ref = (Z + 1) / 2

Outputs
-------
* ``front_position.csv``   — step, t*, X* (simulation)
* ``mm_comparison.csv``    — T, X*_sim, X*_ref, err_pct (6-point table)
* ``front_comparison.png`` — simulation curve + MM experimental points
* ``phase_sequence.png``   — 6-panel phase-fraction snapshot sequence
* ``validation_report.json`` — machine-readable verdict + metrics
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------------------------
# Martin & Moyce (1952) reference data — digitised surge front
# ---------------------------------------------------------------------------
# T = t·√(g/a),  Z = x_front_MM / a   (a = column half-width)
# Source: Martin & Moyce, Phil. Trans. R. Soc. A 244, 312 (1952)
#         digitised from Figure 3 of the original paper.
_MM_FRONT: list[tuple[float, float]] = [
    (0.0, 1.0),
    (0.5, 1.1),
    (1.0, 1.4),
    (1.5, 1.8),
    (2.0, 2.2),
    (2.5, 2.7),
    (3.0, 3.1),
    (3.5, 3.5),
    (4.0, 3.8),
    (4.5, 4.1),
    (5.0, 4.3),
    (5.5, 4.5),
    (6.0, 4.7),
]

# Six evaluation points for the acceptance criterion (spread across the curve)
_EVAL_POINTS: list[float] = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def _mm_to_xstar(z: float) -> float:
    """Convert Martin & Moyce Z to our X* scale: X* = (Z + 1) / 2."""
    return (z + 1.0) / 2.0


def _interp_mm(t: float) -> float | None:
    """Interpolate X*_ref at dimensionless time t from MM table."""
    if t < _MM_FRONT[0][0] or t > _MM_FRONT[-1][0]:
        return None
    for (t0, z0), (t1, z1) in zip(_MM_FRONT[:-1], _MM_FRONT[1:]):
        if t0 <= t <= t1:
            frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            z = z0 + frac * (z1 - z0)
            return _mm_to_xstar(z)
    return _mm_to_xstar(_MM_FRONT[-1][1])


# ---------------------------------------------------------------------------
# LBM solver (vectorised D2Q9 CG)
# ---------------------------------------------------------------------------

# D2Q9 lattice
_Q9_E = torch.tensor([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
], dtype=torch.int32)
_Q9_W = torch.tensor([4/9] + [1/9]*4 + [1/36]*4, dtype=torch.float32)


def _equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium (9, ny, nx)."""
    f = torch.zeros((9, *rho.shape), dtype=torch.float32, device=rho.device)
    for i in range(9):
        ex, ey = int(_Q9_E[i, 0]), int(_Q9_E[i, 1])
        eu = ex * ux + ey * uy
        u2 = ux * ux + uy * uy
        f[i] = _Q9_W[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)
    return f


def _stream(f: torch.Tensor) -> torch.Tensor:
    """Stream step."""
    fout = torch.empty_like(f)
    for i in range(9):
        ex, ey = int(_Q9_E[i, 0]), int(_Q9_E[i, 1])
        fout[i] = torch.roll(f[i], shifts=(ey, ex), dims=(0, 1))
    return fout


def _bounce_back(f: torch.Tensor, wall: torch.Tensor) -> torch.Tensor:
    """Bounce-back."""
    opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    for i in range(9):
        f[i] = torch.where(wall, f[opp[i]], f[i])
    return f


def _macroscopic(f: torch.Tensor):
    rho = f.sum(dim=0)
    ux = (f[1] - f[3] + f[5] - f[6] - f[7] + f[8]) / (rho + 1e-12)
    uy = (f[2] - f[4] + f[5] + f[6] - f[7] - f[8]) / (rho + 1e-12)
    return rho, ux, uy


def _cg_step(f1, f2, tau, gy, wall):
    """One CG collision+stream+BB step."""
    rho1, ux1, uy1 = _macroscopic(f1)
    rho2, ux2, uy2 = _macroscopic(f2)
    rho = rho1 + rho2
    ux = (rho1 * ux1 + rho2 * ux2) / (rho + 1e-12)
    uy = (rho1 * uy1 + rho2 * uy2) / (rho + 1e-12)

    # Color gradient
    color = (rho1 - rho2) / (rho + 1e-12)
    dc_dx = 0.5 * (torch.roll(color, -1, 1) - torch.roll(color, 1, 1))
    dc_dy = 0.5 * (torch.roll(color, -1, 0) - torch.roll(color, 1, 0))
    gn = (dc_dx**2 + dc_dy**2 + 1e-12).sqrt()
    nx_c = dc_dx / gn
    ny_c = dc_dy / gn

    A = 0.02  # surface tension
    beta = 0.9  # recoloring

    f1_out = torch.empty_like(f1)
    f2_out = torch.empty_like(f2)
    for i in range(9):
        ex, ey = int(_Q9_E[i, 0]), int(_Q9_E[i, 1])
        eu = ex * ux + ey * uy
        u2 = ux * ux + uy * uy
        feq = _Q9_W[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)

        # Perturbation
        pert = A * _Q9_W[i] * ((ex * nx_c + ey * ny_c)**2 - 1/3)

        # BGK
        f1_out[i] = f1[i] - (f1[i] - feq * rho1 / rho) / tau + pert
        f2_out[i] = f2[i] - (f2[i] - feq * rho2 / rho) / tau - pert

        # Guo forcing
        fu1 = _Q9_W[i] * ((ex - ux) * rho1 * 0 + (ey - uy) * rho1 * gy +
                          3.0 * (ex * 0 + ey * gy) * eu) * (1 - 0.5/tau)
        fu2 = _Q9_W[i] * ((ex - ux) * rho2 * 0 + (ey - uy) * rho2 * gy +
                          3.0 * (ex * 0 + ey * gy) * eu) * (1 - 0.5/tau)
        f1_out[i] += fu1
        f2_out[i] += fu2

        # Recoloring
        cos_t = ex * nx_c + ey * ny_c
        f1_out[i] += beta * _Q9_W[i] * rho * color * cos_t
        f2_out[i] -= beta * _Q9_W[i] * rho * color * cos_t

    f1_out = _stream(f1_out)
    f2_out = _stream(f2_out)
    f1_out = _bounce_back(f1_out, wall)
    f2_out = _bounce_back(f2_out, wall)
    return f1_out, f2_out


def _find_front(rho1, rho2, wall, threshold=0.5):
    phi = rho1 / (rho1 + rho2 + 1e-12)
    phi_mean = phi[1:-1, :].mean(dim=0)
    cols = (phi_mean > threshold).nonzero(as_tuple=True)[0]
    return float(cols.max().item()) if cols.numel() > 0 else 0.0


# ---------------------------------------------------------------------------
# Config & runner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValConfig:
    nx: int = 200
    ny: int = 100
    dam_width: int = 50
    rho_heavy: float = 1.0
    rho_light: float = 0.1
    tau: float = 0.8
    g: float = 1e-4
    n_steps: int = 8000
    output_interval: int = 200
    device: str = "cpu"


def run_validation(cfg: ValConfig, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(cfg.device)
    ny, nx = cfg.ny, cfg.nx
    L = float(cfg.dam_width)
    sqrt_coeff = math.sqrt(2.0 * cfg.g / L)

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    wall[0,:] = wall[-1,:] = wall[:,0] = wall[:,-1] = True

    # IC
    x = torch.arange(nx, dtype=torch.float32, device=dev)
    prof = 0.5*(1 - torch.tanh((x - cfg.dam_width)/3)).unsqueeze(0).expand(ny, nx)
    frac = 0.05
    r1 = cfg.rho_heavy*prof + cfg.rho_heavy*frac*(1-prof)
    r2 = cfg.rho_light*frac*prof + cfg.rho_light*(1-prof)
    zero = torch.zeros((ny, nx), device=dev)
    f1, f2 = _equilibrium(r1, zero, zero), _equilibrium(r2, zero, zero)

    gy = -cfg.g
    front_series = []
    snapshots = []

    print(f"Dam-break val: {nx}x{ny}, L={L}, g={cfg.g:.1e}, steps={cfg.n_steps}")
    for step in range(1, cfg.n_steps + 1):
        f1, f2 = _cg_step(f1, f2, cfg.tau, gy, wall)
        if step % cfg.output_interval == 0 or step == cfg.n_steps:
            rho1, _, _ = _macroscopic(f1)
            rho2, _, _ = _macroscopic(f2)
            xf = _find_front(rho1, rho2, wall)
            Xs = xf / L
            ts = step * sqrt_coeff
            front_series.append((step, ts, Xs))
            snapshots.append((step, rho1.clone(), rho2.clone()))
            print(f"  step={step:5d}  t*={ts:.3f}  X*={Xs:.4f}")

    # CSV outputs
    with (out_dir / "front_position.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["step","t_star","X_star"]); w.writerows(front_series)

    # 6-point comparison
    comp = []
    for te in _EVAL_POINTS:
        xr = _interp_mm(te)
        if xr is None: continue
        xs = None
        for i in range(len(front_series)-1):
            if front_series[i][1] <= te <= front_series[i+1][1]:
                f = (te - front_series[i][1]) / (front_series[i+1][1] - front_series[i][1])
                xs = front_series[i][2] + f*(front_series[i+1][2] - front_series[i][2])
                break
        if xs is None and front_series: xs = front_series[-1][2]
        if xs is not None:
            ep = abs(xs - xr) / xr * 100
            comp.append({"T": te, "X_star_sim": round(xs,6), "X_star_ref": round(xr,6), "err_pct": round(ep,4)})

    with (out_dir / "mm_comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["T","X_star_sim","X_star_ref","err_pct"])
        w.writeheader(); w.writerows(comp)

    errs = [r["err_pct"] for r in comp]
    mean_err = sum(errs)/len(errs) if errs else float("nan")
    passed = mean_err <= 6.0

    # Figures
    _plot_front(front_series, comp, out_dir)
    _plot_phases(snapshots, cfg, out_dir)

    report = {
        "benchmark": "dam_break_martin_moyce_1952",
        "config": {"nx":cfg.nx,"ny":cfg.ny,"dam_width":cfg.dam_width,
                    "rho_heavy":cfg.rho_heavy,"rho_light":cfg.rho_light,
                    "tau":cfg.tau,"g":cfg.g,"n_steps":cfg.n_steps},
        "reference": "Martin & Moyce (1952) Phil. Trans. R. Soc. A 244, 312",
        "criterion": "6-point mean |X*_sim - X*_ref| / X*_ref ≤ 6%",
        "n_points": len(comp),
        "comparison_table": comp,
        "mean_error_pct": round(mean_err, 4),
        "pass": bool(passed),
        "figures": ["front_comparison.png", "phase_sequence.png"],
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"Dam-break validation vs Martin & Moyce (1952)")
    print(f"{'='*60}")
    print(f"{'T':>6} {'X*_sim':>10} {'X*_ref':>10} {'err%':>8}")
    print(f"{'-'*36}")
    for r in comp:
        print(f"{r['T']:>6.1f} {r['X_star_sim']:>10.4f} {r['X_star_ref']:>10.4f} {r['err_pct']:>7.2f}%")
    print(f"{'-'*36}")
    print(f"Mean error: {mean_err:.2f}%")
    print(f"Criterion (≤6%): {'PASS' if passed else 'FAIL'}")
    print(f"{'='*60}")
    return report


def _plot_front(front_series, comp, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    t_mm = [t for t,_ in _MM_FRONT]; z_mm = [z for _,z in _MM_FRONT]
    x_mm = [_mm_to_xstar(z) for z in z_mm]
    ax.plot(t_mm, x_mm, "ko", ms=7, mfc="white", mew=1.8,
            label="Martin & Moyce (1952) experiment", zorder=5)
    t_sim = [s[1] for s in front_series]; x_sim = [s[2] for s in front_series]
    ax.plot(t_sim, x_sim, "b-", lw=2, label="CG-LBM simulation", zorder=3)
    for r in comp:
        ax.plot(r["T"], r["X_star_sim"], "rs", ms=8, zorder=6)
        ax.plot(r["T"], r["X_star_ref"], "g^", ms=8, zorder=6)
    ax.set_xlabel("Dimensionless time  $t^* = t\\,\\sqrt{2g/L}$", fontsize=12)
    ax.set_ylabel("Dimensionless front position  $X^* = x_f / L$", fontsize=12)
    ax.set_title("Dam-break surge front: CG-LBM vs Martin & Moyce (1952)", fontsize=13)
    ax.legend(fontsize=11, loc="upper left"); ax.grid(alpha=0.3)
    p = out_dir / "front_comparison.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")


def _plot_phases(snapshots, cfg, out_dir):
    n = 6; idxs = [int(i*(len(snapshots)-1)/(n-1)) for i in range(n)]
    sel = [snapshots[i] for i in idxs]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()
    L = float(cfg.dam_width); sc = math.sqrt(2.0*cfg.g/L)
    for i, (step, r1, r2) in enumerate(sel):
        phi = (r1/(r1+r2+1e-12)).detach().cpu().numpy()
        ts = step * sc
        ax = axes[i]
        im = ax.imshow(phi, origin="lower", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"t* = {ts:.2f}  (step {step})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.04, label="φ")
    fig.suptitle("Dam-break phase-fraction evolution (Color-Gradient LBM)", fontsize=13, y=1.01)
    p = out_dir / "phase_sequence.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")


if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--nx", type=int, default=200)
    pa.add_argument("--ny", type=int, default=100)
    pa.add_argument("--dam-width", type=int, default=50)
    pa.add_argument("--g", type=float, default=1e-4)
    pa.add_argument("--tau", type=float, default=0.8)
    pa.add_argument("--n-steps", type=int, default=8000)
    pa.add_argument("--output-interval", type=int, default=200)
    pa.add_argument("--output-dir", default="validation/dam_break/run_cg_200x100")
    pa.add_argument("--device", default="cpu")
    args = pa.parse_args()
    cfg = ValConfig(nx=args.nx, ny=args.ny, dam_width=args.dam_width,
                    g=args.g, tau=args.tau, n_steps=args.n_steps,
                    output_interval=args.output_interval, device=args.device)
    run_validation(cfg, Path(args.output_dir))
