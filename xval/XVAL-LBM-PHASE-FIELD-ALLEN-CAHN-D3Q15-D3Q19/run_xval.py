#!/usr/bin/env python3
"""XVAL-LBM-PHASE-FIELD-ALLEN-CAHN-D3Q15-D3Q19 — cross-lattice acceptance run.

Anchor: ``lssfau/walberla`` phase-field Allen-Cahn available on **D3Q15 and
D3Q19** (lbmpy ``phasefield_allen_cahn``).  Our implementation is
``tensorlbm.phasefield.ac_lattice_generic`` (one code path, lattice-parametric),
whose D3Q19 branch is machine-identical to the frozen, verified
``tensorlbm.cac_lbm``.

Judgement is against *analytic* anchors, never against our own output:

  * Young-Laplace for a static droplet, 3-D sphere:  dp = 2 sigma / R
    (Young 1805; Laplace 1806), read as sigma_rec = dp R / 2;
  * the model's own equilibrium interface solution
    phi(y) = 0.5 [1 - tanh(2 (y - y0) / W)]  (Liang et al., PRE 97, 033309
    (2018) Eq.(30)/(33); same profile family as lbmpy's Allen-Cahn app);
  * discrete conservation of the phase inventory (conservative AC form).

Arm definitions and thresholds live in ``predeclared.json``; this file contains
no threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.phasefield.ac_lattice_generic import (  # noqa: E402
    ACPhaseFieldSim,
    diagnostics,
    droplet_pressure_jump,
    init_phi_droplet,
    init_phi_layers,
    interface_width_fit,
)


def load_predeclared() -> dict:
    return json.loads((HERE / "predeclared.json").read_text())


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


def _common(cfg: dict) -> dict:
    ratio = float(cfg["rho_ratio"])
    rho_l = 1.0
    rho_g = rho_l / ratio
    return dict(
        rho_l=rho_l,
        rho_g=rho_g,
        mu_l=float(cfg["nu"]) * rho_l,            # equal kinematic viscosity
        mu_g=float(cfg["nu"]) * rho_g,
        sigma=float(cfg["sigma"]),
        width=float(cfg["width"]),
        mobility=float(cfg["mobility"]),
        dtype=torch.float64,
    )


def run_laplace_arm(name: str, cfg: dict, out_dir: Path) -> dict:
    n = int(cfg["grid"])
    R = float(cfg["radius"])
    shape = (n, n, n)
    sim = ACPhaseFieldSim(
        shape,
        lattice=cfg["lattice"],
        lattice_phase=cfg.get("lattice_phase"),
        lattice_hydro=cfg.get("lattice_hydro"),
        lap_scheme=cfg.get("lap_scheme", "lbmpy"),
        **_common(cfg),
    )
    phi0 = init_phi_droplet(shape, R, float(cfg["width"]), dtype=torch.float64)
    sim.initialize(phi0, p0=float(cfg["p0"]))
    phi_int0 = float(phi0.sum().item())

    max_steps = int(cfg["max_steps"])
    sample_every = int(cfg["sample_every"])
    tol = float(cfg["steady_rel_tol"])
    min_steps = int(cfg["min_steps"])
    series: list[dict] = []
    converged = False
    t0 = time.time()
    prev_dp = None
    steps_done = 0
    for it in range(1, max_steps + 1):
        sim.step()
        steps_done = it
        if it % sample_every == 0:
            phi, rho, grad, mu_phi, u, p = sim.macro()
            jm = droplet_pressure_jump(
                phi, p, radius=R, inner_band=tuple(cfg["inner_band"]), outer_band=tuple(cfg["outer_band"])
            )
            umax = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt().max().item()
            rec = {
                "step": it,
                "dp": jm["dp"],
                "p_in": jm["p_in"],
                "p_out": jm["p_out"],
                "sigma_rec": jm["dp"] * R / 2.0,
                "u_max": umax,
                "phi_integral": float(phi.sum().item()),
                "phi_drift_rel": abs(float(phi.sum().item()) - phi_int0) / abs(phi_int0),
                "nan": bool(torch.isnan(sim.f).any().item() or torch.isnan(sim.g).any().item()),
                "wall_s": time.time() - t0,
            }
            series.append(rec)
            print(
                f"[{name}] step {it} dp={rec['dp']:.6e} sig={rec['sigma_rec']:.6e} "
                f"umax={umax:.2e} drift={rec['phi_drift_rel']:.1e} {rec['wall_s']:.0f}s",
                flush=True,
            )
            if rec["nan"]:
                break
            if it >= min_steps and prev_dp is not None and abs(prev_dp) > 0:
                if abs(rec["dp"] - prev_dp) / abs(prev_dp) < tol:
                    converged = True
                    break
            prev_dp = rec["dp"]

    phi, rho, grad, mu_phi, u, p = sim.macro()
    jm = droplet_pressure_jump(
        phi, p, radius=R, inner_band=tuple(cfg["inner_band"]), outer_band=tuple(cfg["outer_band"])
    )
    sigma_in = float(cfg["sigma"])
    sigma_rec = jm["dp"] * R / 2.0

    # interface-shape anchor: radial profile vs the model's own tanh solution
    prof = _radial_profile(phi, R)
    fit = _fit_radial_tanh(prof, float(cfg["width"]))

    result = {
        "arm": name,
        "kind": "laplace",
        "config": {k: (v if not isinstance(v, torch.dtype) else "float64") for k, v in cfg.items()},
        "lattice": sim.label,
        "lap_scheme": cfg.get("lap_scheme", "lbmpy"),
        "tau_f": sim.tau_f,
        "beta": sim.beta,
        "kappa": sim.kappa,
        "steps_done": steps_done,
        "converged": converged,
        "max_steps": max_steps,
        "wall_s": time.time() - t0,
        "sigma_in": sigma_in,
        "sigma_rec": sigma_rec,
        "sigma_rel_err_pct": abs(sigma_rec - sigma_in) / sigma_in * 100.0,
        "dp": jm["dp"],
        "p_in": jm["p_in"],
        "p_out": jm["p_out"],
        "u_max_final": float(((u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt()).max().item()),
        "phi_drift_rel": abs(float(phi.sum().item()) - phi_int0) / abs(phi_int0),
        "nan": bool(torch.isnan(sim.f).any().item() or torch.isnan(sim.g).any().item()),
        "radial_profile": prof,
        "radial_fit": fit,
        "series": series,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1))
    return result


def _radial_profile(phi: torch.Tensor, radius: float, nbins: int = 48) -> dict:
    """Azimuthally averaged phi(r) around the box centre (r in lattice units)."""
    nz, ny, nx = phi.shape
    z = torch.arange(nz, dtype=torch.float64, device=phi.device).view(-1, 1, 1)
    y = torch.arange(ny, dtype=torch.float64, device=phi.device).view(1, -1, 1)
    x = torch.arange(nx, dtype=torch.float64, device=phi.device).view(1, 1, -1)
    r = ((z - nz / 2) ** 2 + (y - ny / 2) ** 2 + (x - nx / 2) ** 2).sqrt()
    edges = torch.linspace(0.0, 2.0 * radius, nbins + 1, dtype=torch.float64, device=phi.device)
    idx = torch.bucketize(r, edges) - 1
    valid = (idx >= 0) & (idx < nbins)
    sums = torch.zeros(nbins, dtype=torch.float64, device=phi.device).index_add_(
        0, idx[valid], phi.to(torch.float64)[valid]
    )
    cnt = torch.zeros(nbins, dtype=torch.float64, device=phi.device).index_add_(
        0, idx[valid], torch.ones(int(valid.sum()), dtype=torch.float64, device=phi.device)
    )
    mean = torch.where(cnt > 0, sums / cnt.clamp_min(1), torch.full_like(sums, float("nan")))
    centres = 0.5 * (edges[:-1] + edges[1:])
    return {"r": centres.tolist(), "phi": mean.tolist()}


def _fit_radial_tanh(prof: dict, width: float) -> dict:
    """Fit phi = 0.5 [1 - tanh(2 (r - R_fit)/W_fit)] on the interface band."""
    r = torch.tensor(prof["r"], dtype=torch.float64)
    ph = torch.tensor(prof["phi"], dtype=torch.float64)
    ok = torch.isfinite(ph)
    band = ok & (ph > 0.05) & (ph < 0.95)
    if int(band.sum()) < 4:
        return {"W_fit": float("nan"), "R_fit": float("nan"), "max_dev": float("nan"), "n_band": int(band.sum())}
    rr, pp = r[band], ph[band]
    arg = (1.0 - 2.0 * pp).clamp(-0.999999, 0.999999)
    art = 0.5 * torch.log((1.0 + arg) / (1.0 - arg))  # = 2 (r - R)/W
    rc = rr - rr.mean()
    slope = (rc * (art - art.mean())).sum() / (rc * rc).sum()
    W_fit = float(2.0 / slope.abs())
    R_fit = float(rr.mean() - art.mean() / slope)
    analytic = 0.5 * (1.0 - torch.tanh(2.0 * (rr - R_fit) / W_fit))
    return {
        "W_fit": W_fit,
        "W_input": width,
        "W_rel_err_pct": abs(W_fit - width) / width * 100.0,
        "R_fit": R_fit,
        "max_dev": float((pp - analytic).abs().max()),
        "n_band": int(band.sum()),
    }


def run_profile_arm(name: str, cfg: dict, out_dir: Path) -> dict:
    """Planar-interface arm: is the tanh profile of width W a fixed point?"""
    nz, ny, nx = cfg["shape"]
    W = float(cfg["width"])
    y0 = ny / 2.0
    sim = ACPhaseFieldSim(
        (nz, ny, nx),
        lattice=cfg["lattice"],
        lattice_phase=cfg.get("lattice_phase"),
        lattice_hydro=cfg.get("lattice_hydro"),
        lap_scheme=cfg.get("lap_scheme", "lbmpy"),
        **_common(cfg),
    )
    phi0 = init_phi_layers((nz, ny, nx), W, y0, dtype=torch.float64)
    sim.initialize(phi0, p0=float(cfg["p0"]))
    phi_int0 = float(phi0.sum().item())
    t0 = time.time()
    series = []
    for it in range(1, int(cfg["max_steps"]) + 1):
        sim.step()
        if it % int(cfg["sample_every"]) == 0:
            phi, *_ = sim.macro()
            fit = interface_width_fit(phi, W)
            series.append({"step": it, **fit, "phi_drift_rel": abs(float(phi.sum().item()) - phi_int0) / abs(phi_int0)})
            print(f"[{name}] step {it} W_fit={fit['W_fit']:.4f} dev={fit['max_dev']:.2e}", flush=True)
    phi, rho, grad, mu_phi, u, p = sim.macro()
    fit = interface_width_fit(phi, W)
    result = {
        "arm": name,
        "kind": "profile",
        "config": {k: v for k, v in cfg.items()},
        "lattice": sim.label,
        "lap_scheme": cfg.get("lap_scheme", "lbmpy"),
        "steps_done": int(cfg["max_steps"]),
        "wall_s": time.time() - t0,
        "W_input": W,
        "W_fit": fit["W_fit"],
        "W_rel_err_pct": abs(fit["W_fit"] - W) / W * 100.0,
        "max_dev": fit["max_dev"],
        "u_max_final": float(((u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt()).max().item()),
        "phi_drift_rel": abs(float(phi.sum().item()) - phi_int0) / abs(phi_int0),
        "nan": bool(torch.isnan(sim.f).any().item() or torch.isnan(sim.g).any().item()),
        "profile_line": phi[0, :, 0].tolist(),
        "series": series,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1))
    return result


def run_control_arm(name: str, cfg: dict, out_dir: Path) -> dict:
    """Single-phase control: phi == 1, so mu grad(phi) vanishes -> dp must be ~0."""
    n = int(cfg["grid"])
    shape = (n, n, n)
    sim = ACPhaseFieldSim(
        shape,
        lattice=cfg["lattice"],
        lattice_phase=cfg.get("lattice_phase"),
        lattice_hydro=cfg.get("lattice_hydro"),
        lap_scheme=cfg.get("lap_scheme", "lbmpy"),
        **_common(cfg),
    )
    phi0 = torch.ones(shape, dtype=torch.float64)
    sim.initialize(phi0, p0=float(cfg["p0"]))
    t0 = time.time()
    series = []
    for it in range(1, int(cfg["max_steps"]) + 1):
        sim.step()
        if it % int(cfg["sample_every"]) == 0:
            phi, rho, grad, mu_phi, u, p = sim.macro()
            umax = float(((u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt()).max().item())
            dp = float(p.max().item() - p.min().item())
            series.append({"step": it, "p_span": dp, "u_max": umax, "phi_min": float(phi.min().item())})
            print(f"[{name}] step {it} p_span={dp:.3e} umax={umax:.2e}", flush=True)
    phi, rho, grad, mu_phi, u, p = sim.macro()
    result = {
        "arm": name,
        "kind": "control",
        "config": cfg,
        "lattice": sim.label,
        "lap_scheme": cfg.get("lap_scheme", "lbmpy"),
        "steps_done": int(cfg["max_steps"]),
        "wall_s": time.time() - t0,
        "p_span": float(p.max().item() - p.min().item()),
        "u_max_final": float(((u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt()).max().item()),
        "phi_min": float(phi.min().item()),
        "nan": bool(torch.isnan(sim.f).any().item() or torch.isnan(sim.g).any().item()),
        "series": series,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1))
    return result


def run_arm(name: str, pre: dict, out_dir: Path) -> dict:
    cfg = pre["arms"][name]
    if cfg["kind"] == "laplace":
        return run_laplace_arm(name, cfg, out_dir)
    if cfg["kind"] == "profile":
        return run_profile_arm(name, cfg, out_dir)
    if cfg["kind"] == "control":
        return run_control_arm(name, cfg, out_dir)
    raise ValueError(f"unknown arm kind {cfg['kind']!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", help="single arm name")
    ap.add_argument("--shard", type=int, help="shard index (0-based)")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "artifacts"))
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default)")
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    pre = load_predeclared()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(pre["arms"].keys()) if args.arm is None else [args.arm]
    if args.shard is not None:
        names = [n for i, n in enumerate(names) if i % args.nshards == args.shard]
    print(f"=== running {len(names)} arm(s): {names} ===", flush=True)
    for n in names:
        run_arm(n, pre, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())