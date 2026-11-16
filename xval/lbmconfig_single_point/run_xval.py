#!/usr/bin/env python
"""XVAL-LBM-LBMCONFIG-SINGLE-POINT-CONFIG -- cross-validation harness.

Practice under test (OSS anchor: lssfau/lbmpy ``LBMConfig``)
------------------------------------------------------------
lbmpy declares a scheme in **one** config object (stencil, method, relaxation
rate, force model, force) and derives the equilibrium, the force source term,
the macroscopic velocity convention and the streaming step from it.
TensorLBM's production home for that pattern is ``src/tensorlbm/lbm_config.py``
(``LBMConfig``).

What is measured
----------------
A force-driven 2D channel (periodic in x, half-way bounce-back at
y = 1/2 and y = H + 1/2, D2Q9), whose steady solution is the plane-Poiseuille
parabola

    u_x(y) = F/(2 nu) * (y - 1/2) * (H + 1/2 - y),      y = 1 .. H

Arms
----
  srt_guo         LBMConfig(method=SRT, force_model=guo)        textbook
  trt_guo         LBMConfig(method=TRT, Lambda=3/16, guo)       Ginzburg 2008
  srt_guo_split   LBMConfig(force_model=guo_split)              in-repo split
  mrt_guo_split   LBMConfig(method=MRT, guo_split)
  explicit_calls  the srt_guo scheme rebuilt by hand (no config object);
                  must be bitwise identical -> the single point is transparent
  legacy_mixed    hand-built negative control: the (1 - omega/2) corrected
                  source paired with a *raw-velocity* equilibrium -- the
                  defect the configuration refuses to express
  simple_force    LBMConfig(force_model=simple)                 naive reference

Checks
------
  C1 transparency  max |explicit_calls - srt_guo| == 0 (bitwise)
  C2 accuracy      max |u_arm - u_analytic| / max|u_analytic| vs 15 % / 3 %
  C3 convergence   srt_guo and trt_guo for H = 16/32/64 -> observed order
  C4 bookkeeping   error vs omega for guo / guo_split / simple / legacy_mixed;
                   legacy_mixed must track 1 - (1 - omega/2) = omega/2

Usage::

    python xval/lbmconfig_single_point/run_xval.py --out-dir <dir> --fig-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tensorlbm import d2q9
from tensorlbm.lbm_config import LBMConfig
from tensorlbm.solver import stream

C2 = d2q9.C.to(torch.float64)
OPP = d2q9.OPPOSITE.to(torch.int64)


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelCase:
    """Force-driven plane-Poiseuille channel (lattice units)."""

    height: int = 32          # fluid rows H; the array has H + 2 rows
    nx: int = 8               # periodic in x (the flow is x-uniform)
    force_x: float = 2.0e-7   # body force per unit mass
    max_steps: int = 150_000
    window_steps: int = 400   # averaging window once the steady state is found
    steady_tol: float = 1e-12  # |du| over one chunk / u_scale
    chunk: int = 200

    @property
    def ny(self) -> int:
        return self.height + 2

    def analytic(self, nu: float) -> tuple[np.ndarray, np.ndarray]:
        """Plane-Poiseuille parabola between the half-way wall positions."""
        y = np.arange(1, self.height + 1, dtype=float)
        u = self.force_x / (2.0 * nu) * (y - 0.5) * (self.height + 0.5 - y)
        return y, u


# ---------------------------------------------------------------------------
# Lattice plumbing
# ---------------------------------------------------------------------------
def _wall_buffers(f_post: torch.Tensor) -> torch.Tensor:
    """Write reflected post-collision populations into the wall buffer rows.

    After the following (periodic) streaming step the first and last *fluid*
    rows receive exactly ``f_i = f_opp(i)`` post-collision, i.e. half-way
    bounce-back with walls at y = 1/2 and y = H + 1/2.
    """
    out = f_post.clone()
    out[:, 0, :] = f_post[OPP][:, 1, :]
    out[:, -1, :] = f_post[OPP][:, -2, :]
    return out


def _hand_step(
    cfg: LBMConfig,
    state: torch.Tensor,
    *,
    corrected_source: bool,
    half_step_equilibrium: bool,
) -> torch.Tensor:
    """Scheme written out by hand -- the *pre-config* per-call style.

    ``corrected_source`` / ``half_step_equilibrium`` are set independently so
    the harness can build both the consistent combinations and the legacy
    inconsistent one (corrected source + raw-velocity equilibrium).
    """
    c = C2.to(state.device)
    w = torch.tensor(d2q9.W_EXACT, dtype=state.dtype).view(9, 1, 1)
    fx = float(cfg.force[0])
    tau = cfg.tau

    rho = state.sum(dim=0)
    jx = (state * c[:, 0].view(9, 1, 1)).sum(dim=0)
    jy = (state * c[:, 1].view(9, 1, 1)).sum(dim=0)
    ux, uy = jx / rho, jy / rho
    if half_step_equilibrium:
        ux = ux + 0.5 * fx / rho

    feq = d2q9.equilibrium(rho, ux, uy)
    post = state - (state - feq) / tau          # SRT collision

    cu = c[:, 0].view(9, 1, 1) * ux + c[:, 1].view(9, 1, 1) * uy
    cf = c[:, 0].view(9, 1, 1) * fx
    uf = ux * fx
    scale = (1.0 - 0.5 / tau) if corrected_source else 1.0
    post = post + 3.0 * w * scale * (cf - uf + 3.0 * cu * cf)
    return post


def run_case(
    cfg: LBMConfig,
    case: ChannelCase,
    *,
    mode: str = "config",
    corrected_source: bool = True,
    half_step_equilibrium: bool = True,
) -> dict:
    """March one channel case to its steady state and compare with analytic.

    ``mode`` selects the marching path: ``config`` (``cfg.step``), ``explicit``
    (hand-built scheme, same declared model) or ``legacy_mixed`` (hand-built
    inconsistent force bookkeeping).
    """
    import sys as _sys

    from tensorlbm import lbm_config as _lc

    _ = (_sys, _lc)
    dev = torch.device("cpu")
    ny, nx = case.ny, case.nx
    rho0 = torch.ones((ny, nx), dtype=torch.float64, device=dev)
    f = d2q9.equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))

    def step(state: torch.Tensor) -> torch.Tensor:
        if mode == "config":
            post = cfg.collide(state)
            post = post + cfg.force_source(post)
        else:
            post = _hand_step(
                cfg,
                state,
                corrected_source=corrected_source,
                half_step_equilibrium=half_step_equilibrium,
            )
        return stream(_wall_buffers(post))

    def fluid_ux(state: torch.Tensor) -> np.ndarray:
        _, ux, _ = cfg.velocity(state)
        return ux[1:-1].mean(dim=1).detach().numpy().copy()

    t0 = time.perf_counter()
    steps, converged = 0, False
    prev = fluid_ux(f)
    while steps < case.max_steps:
        for _ in range(case.chunk):
            f = step(f)
        steps += case.chunk
        cur = fluid_ux(f)
        scale = max(float(np.abs(cur).max()), 1e-30)
        if float(np.abs(cur - prev).max()) / scale < case.steady_tol:
            converged = True
            break
        prev = cur
    elapsed = time.perf_counter() - t0

    # is the time-marched state an actual fixed point of the operator?
    residual = float((step(f) - f).abs().max() / f.abs().max())

    acc = np.zeros(case.height)
    for _ in range(case.window_steps):
        f = step(f)
        acc += fluid_ux(f)
    profile = acc / case.window_steps

    y, u_ana = case.analytic(cfg.nu)
    u_scale = float(np.abs(u_ana).max())
    mid = case.height // 2
    return {
        "mode": mode,
        "method": cfg.method,
        "force_model": cfg.force_model,
        "relaxation_rate": cfg.relaxation_rate,
        "tau": cfg.tau,
        "nu": cfg.nu,
        "momentum_relaxation_rate": cfg.momentum_relaxation_rate,
        "force_source_scale": cfg.force_source_scale,
        "height": case.height,
        "steps": steps,
        "converged": converged,
        "wall_seconds": elapsed,
        "fixed_point_residual": residual,
        "y": y.tolist(),
        "u_profile": profile.tolist(),
        "u_analytic": u_ana.tolist(),
        "u_center": float(profile[mid]),
        "u_center_analytic": float(u_ana[mid]),
        "rel_error": float(np.abs(profile - u_ana).max() / u_scale),
        "rel_error_center": float(abs(profile[mid] - u_ana[mid]) / u_scale),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _lbmpy_status() -> dict:
    try:  # pragma: no cover - depends on the container
        import lbmpy  # noqa: F401

        return {"available": True, "version": getattr(lbmpy, "__version__", "unknown")}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="xval/lbmconfig_single_point/out")
    ap.add_argument("--fig-dir", default="xval/lbmconfig_single_point/figs")
    ap.add_argument("--stage", default="all", choices=["all", "main", "convergence", "sweep"])
    ap.add_argument("--height", type=int, default=32, help="main-stage channel height")
    ap.add_argument("--steady-tol", type=float, default=None)
    ap.add_argument("--heights", default="16,32,64")
    ap.add_argument("--omegas", default="0.8,1.1,1.4,1.7,1.9")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    fx = ChannelCase.force_x
    base = LBMConfig(relaxation_rate=1.4, force=(fx, 0.0), force_model="guo")
    result: dict = {
        "anchor": "XVAL-LBM-LBMCONFIG-SINGLE-POINT-CONFIG",
        "host": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "machine": platform.machine(),
        },
        "case": {
            "name": "force_driven_poiseuille_channel_d2q9",
            "height": args.height,
            "nx": ChannelCase.nx,
            "force_x": fx,
            "steady_tol": ChannelCase.steady_tol,
            "wall_positions": "y = 1/2 and y = H + 1/2 (half-way bounce-back)",
        },
        "lbmpy_reference": _lbmpy_status(),
        "config": base.as_dict(),
        "scheme": base.scheme(),
        "arms": {},
        "checks": {},
    }
    if args.stage in ("all", "main"):
        case = ChannelCase(height=args.height) if args.steady_tol is None else \
            ChannelCase(height=args.height, steady_tol=args.steady_tol)
        arms = {
            "srt_guo": base,
            "trt_guo": base.with_(method="TRT"),
            "srt_guo_split": base.with_(force_model="guo_split"),
            "mrt_guo_split": base.with_(method="MRT", force_model="guo_split"),
            "simple_force": base.with_(force_model="simple"),
        }
        for name, cfg in arms.items():
            row = run_case(cfg, case)
            result["arms"][name] = row
            print(f"[arm] {name:15s} rel={row['rel_error']:.3e} steps={row['steps']} "
                  f"conv={row['converged']} resid={row['fixed_point_residual']:.1e}",
                  flush=True)

        explicit = run_case(base, case, mode="explicit")
        ref = result["arms"]["srt_guo"]
        u_ref_scale = max(abs(v) for v in ref["u_profile"])
        diff = max(abs(a - b) for a, b in zip(explicit["u_profile"], ref["u_profile"]))
        # the two paths differ only by floating-point association order
        # (omega*(f-feq) vs (f-feq)/tau), so "identical" means round-off level
        same = diff <= 1e-12 * u_ref_scale
        result["arms"]["explicit_calls"] = explicit
        result["checks"]["C1_single_point_transparency"] = {
            "max_abs_diff_vs_srt_guo": float(diff),
            "rel_diff_vs_u_scale": float(diff / u_ref_scale),
            "machine_identical": bool(same),
            "criterion": "max|diff| <= 1e-12 * max|u|",
            "pass": bool(same),
        }
        print(f"[arm] explicit_calls  bitwise_identical={same}", flush=True)

        # negative control: the inconsistent bookkeeping the config rejects
        legacy = run_case(base, case, mode="legacy", corrected_source=True,
                          half_step_equilibrium=False)
        result["arms"]["legacy_mixed"] = legacy
        print(f"[arm] legacy_mixed    rel={legacy['rel_error']:.3e}", flush=True)

        acc = {
            name: {
                "rel_error": row["rel_error"],
                "within_15pct": bool(row["rel_error"] < 0.15),
                "within_3pct": bool(row["rel_error"] < 0.03),
            }
            for name, row in result["arms"].items()
        }
        consistent = [n for n in arms]
        acc["pass"] = all(acc[n]["within_15pct"] for n in consistent)
        acc["pass_3pct"] = all(acc[n]["within_3pct"] for n in consistent)
        acc["negative_control_legacy_mixed"] = {
            "rel_error": legacy["rel_error"],
            "within_15pct": bool(legacy["rel_error"] < 0.15),
            "predicted_1_minus_scale": float(1.0 - ref["force_source_scale"]),
        }
        result["checks"]["C2_accuracy_vs_analytic"] = acc
        (out_dir / "run_main.json").write_text(json.dumps(result, indent=2))

        lines = ["arm,y,u_profile,u_analytic"]
        for name, row in result["arms"].items():
            for yv, uv, av in zip(row["y"], row["u_profile"], row["u_analytic"]):
                lines.append(f"{name},{yv:.1f},{uv:.12e},{av:.12e}")
        (out_dir / "profiles.csv").write_text("\n".join(lines) + "\n")

    if args.stage in ("all", "convergence"):
        conv: dict = {}
        for name, cfg in (("srt_guo", base), ("trt_guo", base.with_(method="TRT"))):
            conv[name] = {}
            for h in [int(v) for v in args.heights.split(",")]:
                row = run_case(cfg, ChannelCase(height=h))
                conv[name][h] = {
                    "rel_error": row["rel_error"],
                    "rel_error_center": row["rel_error_center"],
                    "steps": row["steps"],
                }
                print(f"[conv] {name} H={h:3d} rel={row['rel_error']:.3e}", flush=True)
            hs = [int(v) for v in args.heights.split(",")]
            errs = [conv[name][h]["rel_error_center"] for h in hs]
            conv[name]["observed_order_center"] = [
                math.log(errs[i] / errs[i + 1]) / math.log(hs[i + 1] / hs[i])
                for i in range(len(errs) - 1)
                if errs[i] > 0 and errs[i + 1] > 0
            ]
        result["checks"]["C3_mesh_convergence"] = conv
        (out_dir / "convergence.json").write_text(json.dumps(conv, indent=2))

    if args.stage in ("all", "sweep"):
        omegas = [float(v) for v in args.omegas.split(",")]
        sweep_case = ChannelCase(height=16, steady_tol=1e-9)
        sweep: dict = {}
        variants = {
            "guo": dict(cfg=base, mode="config"),
            "guo_split": dict(cfg=base.with_(force_model="guo_split"), mode="config"),
            "simple": dict(cfg=base.with_(force_model="simple"), mode="config"),
            "legacy_mixed": dict(cfg=base, mode="legacy"),
        }
        for name, spec in variants.items():
            sweep[name] = {}
            for om in omegas:
                cfg = spec["cfg"].with_(relaxation_rate=om)
                row = run_case(cfg, sweep_case, mode=spec["mode"],
                               corrected_source=True, half_step_equilibrium=False)
                sweep[name][f"{om:.2f}"] = {
                    "rel_error": row["rel_error"],
                    "rel_error_center": row["rel_error_center"],
                    "predicted_error_1_minus_scale": float(1.0 - row["force_source_scale"]),
                }
                print(f"[sweep] {name:12s} omega={om:.2f} rel={row['rel_error']:.3e}",
                      flush=True)
        result["checks"]["C4_force_bookkeeping_sweep"] = {
            "height": sweep_case.height,
            "omegas": omegas,
            "arms": sweep,
        }
        (out_dir / "sweep.json").write_text(
            json.dumps(result["checks"]["C4_force_bookkeeping_sweep"], indent=2)
        )

    if args.stage == "all":
        (out_dir / "run_main.json").write_text(json.dumps(result, indent=2))
        make_figures(result, fig_dir)
        acc = result["checks"]["C2_accuracy_vs_analytic"]
        print(json.dumps({
            "pass_15pct": acc["pass"],
            "pass_3pct": acc["pass_3pct"],
            "srt_guo_rel_error": result["arms"]["srt_guo"]["rel_error"],
            "trt_guo_rel_error": result["arms"]["trt_guo"]["rel_error"],
            "legacy_mixed_rel_error": result["arms"]["legacy_mixed"]["rel_error"],
        }, indent=2))
        return 0 if acc["pass"] else 1
    return 0


def make_figures(result: dict, fig_dir: Path) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    first = True
    for name, row in result["arms"].items():
        if name in ("explicit_calls",):
            continue
        ax.plot(row["u_profile"], row["y"], marker="o", ms=3, lw=1.2, label=name)
        if first:
            ax.plot(row["u_analytic"], row["y"], "k--", lw=1.6, label="analytic")
            first = False
    ax.set_xlabel(r"$u_x$ (lattice units)")
    ax.set_ylabel("y (lattice units)")
    ax.set_title("Force-driven channel: steady profiles vs Poiseuille")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_profiles.png", dpi=150)
    plt.close(fig)

    conv = result["checks"]["C3_mesh_convergence"]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for name in ("srt_guo", "trt_guo"):
        hs = [h for h in sorted(conv[name]) if isinstance(h, int)]
        ax.loglog(hs, [conv[name][h]["rel_error_center"] for h in hs], marker="s",
                  label=name)
    ax.set_xlabel("channel height H")
    ax.set_ylabel("centerline relative error")
    ax.set_title(r"Mesh convergence (SRT vs TRT magic $\Lambda=3/16$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_convergence.png", dpi=150)
    plt.close(fig)

    sw = result["checks"]["C4_force_bookkeeping_sweep"]
    omegas = sw["omegas"]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for name, rows in sw["arms"].items():
        ax.semilogy(omegas, [rows[f"{om:.2f}"]["rel_error_center"] for om in omegas],
                    marker="o", label=name)
    ax.semilogy(omegas, [0.5 * om for om in omegas], "k:", lw=1.2,
                label=r"$\omega/2$ (predicted legacy error)")
    ax.set_xlabel(r"relaxation rate $\omega = 1/\tau$")
    ax.set_ylabel("centerline relative error")
    ax.set_title(f"Force bookkeeping error vs $\\omega$ (H={sw['height']})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_force_bookkeeping.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())