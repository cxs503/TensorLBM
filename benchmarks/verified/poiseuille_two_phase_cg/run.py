#!/usr/bin/env python3
"""Two-layer Poiseuille benchmark — two-tau D2Q9 Color-Gradient model.

Physics: gravity-driven planar two-layer Poiseuille flow of immiscible
fluids (equal densities) between halfway bounce-back walls.  Arms cover
the equal-viscosity case (eq) and a kinematic viscosity contrast
nu_lo/nu_hi = 10 (g10).  The steady analytic profile is evaluated in the
wall-plane convention with the interface plane at
Y_i = j_first_blue - 0.5 measured from the run's own steady phi column
(solid rows j=0 and j=ny-1 -> no-slip planes at Y=0.5 and ny-1.5).

Model: two-relaxation-time D2Q9 color-gradient LBM (Rothman-Keller /
Latva-Kokko recolor with segregation parameter beta, full Guo et al.
(2002) per-component body force, per-component tau_r/tau_b -> viscosity
contrast), fp64, nx=8 periodic streamwise column, A=0.04.

Arms (frozen):
  eq_ny64   : tau_r=tau_b=1.0, gx=5e-6   * (64/ny)^2,          60000 steps
  eq_ny128  : same with gx/4,                                     200000 steps
  g10_ny64  : tau=(2.5, 0.7), gx=1.5e-7 * (64/ny)^2, beta=0.9,  80000 steps
  g10_ny128 : same with gx/4,                                     250000 steps

Gates: relL2(u_sim, u_analytic) <= 3% on every arm; per-color mass drift
<= 1e-10 per step per color; NaN-free.  Convergence disclosure: u_max
relative change over the last 20% of samples is reported per arm.

Engine: additive module ``tensorlbm.color_gradient2d`` (two-tau color
gradient; the stock ``multiphase.color_gradient_step`` shares one tau and
cannot run the contrast arms).  If the module is absent from the repo,
the vendored copy in this directory is loaded under the same package
name.  The analytic evaluator ``cg_reference.py`` is vendored next to
this script; md5s of both are recorded in every result file.

Usage: python run.py [--arms eq_ny64,eq_ny128,g10_ny64,g10_ny128]
                     [--device auto] [--src REPO_SRC] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _resolve_src() -> Path:
    """Locate the repo src/ tree (case at benchmarks/verified/<name>/)."""
    cands: list[Path] = []
    if "--src" in sys.argv:
        i = sys.argv.index("--src")
        if i + 1 < len(sys.argv):
            cands.append(Path(sys.argv[i + 1]))
    if os.environ.get("TENSORLBM_SRC"):
        cands.append(Path(os.environ["TENSORLBM_SRC"]))
    cands.append(HERE.parents[2] / "src")
    for c in cands:
        if (c / "tensorlbm" / "d2q9.py").is_file():
            return c
    raise SystemExit(
        "repo src/ not found: pass --src PATH or set TENSORLBM_SRC (default lookup parents[2]/src)"
    )


SRC = _resolve_src()
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from cg_reference import (  # noqa: E402
    first_blue_row,
    single_phase_profile_wallplanes,
    two_layer_profile_wallplanes,
)

import tensorlbm  # noqa: E402,F401  (package context for the fallback load)
from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import C, equilibrium  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_engine() -> tuple[str, dict[str, str]]:
    """Return (color_gradient_two_tau_step, origin, md5s).

    Prefers a merged ``tensorlbm.color_gradient2d``; falls back to the
    vendored copy loaded under the same package name so its relative
    import ``from .d2q9 import C`` resolves against the repo package.
    """
    md5s = {"cg_reference.py": _md5(HERE / "cg_reference.py")}
    try:
        from tensorlbm.color_gradient2d import color_gradient_two_tau_step

        eng = Path(sys.modules["tensorlbm.color_gradient2d"].__file__ or "")
        md5s["color_gradient2d.py"] = _md5(eng)
        return color_gradient_two_tau_step, "repo tensorlbm.color_gradient2d", md5s
    except ImportError:
        vendored = HERE / "color_gradient2d.py"
        spec = importlib.util.spec_from_file_location("tensorlbm.color_gradient2d", vendored)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tensorlbm.color_gradient2d"] = mod
        spec.loader.exec_module(mod)
        md5s["color_gradient2d.py"] = _md5(vendored)
        return mod.color_gradient_two_tau_step, "vendored color_gradient2d.py", md5s


DT = torch.float64
NX = 8
A = 0.04
FROZEN_ARMS = [
    {
        "name": "eq_ny64",
        "ny": 64,
        "tau_r": 1.0,
        "tau_b": 1.0,
        "gx": 5e-6,
        "beta": 0.9,
        "steps": 60000,
    },
    {
        "name": "eq_ny128",
        "ny": 128,
        "tau_r": 1.0,
        "tau_b": 1.0,
        "gx": 5e-6 / 4,
        "beta": 0.9,
        "steps": 200000,
    },
    {
        "name": "g10_ny64",
        "ny": 64,
        "tau_r": 2.5,
        "tau_b": 0.7,
        "gx": 1.5e-7,
        "beta": 0.9,
        "steps": 80000,
    },
    {
        "name": "g10_ny128",
        "ny": 128,
        "tau_r": 2.5,
        "tau_b": 0.7,
        "gx": 1.5e-7 / 4,
        "beta": 0.9,
        "steps": 250000,
    },
]
GATE_RELL2 = 0.03
GATE_MASS_DRIFT = 1e-10


def run_arm(arm: dict, step_fn, dev: torch.device) -> dict:
    ny, gx = arm["ny"], arm["gx"]
    half = ny // 2
    j = torch.arange(ny, dtype=DT, device=dev).view(ny, 1)
    frac_r = 0.5 * (1.0 - torch.tanh((j - half) / 2.0)).expand(ny, NX).contiguous()
    z = torch.zeros((ny, NX), dtype=DT, device=dev)
    f_r = equilibrium(1.0 * frac_r, z, z)
    f_b = equilibrium(1.0 * (1.0 - frac_r), z, z)
    m_r0, m_b0 = float(f_r.sum()), float(f_b.sum())
    wall = torch.zeros((ny, NX), dtype=torch.bool, device=dev)
    wall[0, :] = True
    wall[-1, :] = True

    c9 = C.to(DT).to(dev)
    umax_hist = []
    t0 = time.perf_counter()
    for step in range(1, arm["steps"] + 1):
        f_r, f_b = step_fn(
            f_r,
            f_b,
            tau_r=arm["tau_r"],
            tau_b=arm["tau_b"],
            A=A,
            beta=arm["beta"],
            gx=gx,
            solid_mask=wall,
        )
        f_r = bounce_back_cells(stream(f_r), wall)
        f_b = bounce_back_cells(stream(f_b), wall)
        if not bool(torch.isfinite(f_r).all() and torch.isfinite(f_b).all()):
            return {**arm, "stable": False, "nan_step": step}
        if step % max(1, arm["steps"] // 20) == 0:
            f = f_r + f_b
            ux = (f * c9[:, 0].view(9, 1, 1)).sum(dim=0) / f.sum(dim=0)
            umax_hist.append((step, float(ux[:, NX // 2].max())))

    f = f_r + f_b
    rho = f.sum(dim=0)
    ux = (f * c9[:, 0].view(9, 1, 1)).sum(dim=0) / rho
    prof = ux[:, NX // 2]
    phi = (f_r.sum(0) - f_b.sum(0)) / rho
    phi_col = phi[:, NX // 2]

    if arm["tau_r"] == arm["tau_b"]:
        ana = single_phase_profile_wallplanes(ny, gx, (arm["tau_r"] - 0.5) / 3.0)
        jb = None
    else:
        jb = first_blue_row(phi_col.cpu())
        nu_lo = (arm["tau_r"] - 0.5) / 3.0
        nu_hi = (arm["tau_b"] - 0.5) / 3.0
        ana = two_layer_profile_wallplanes(ny, jb, gx, nu_lo, nu_hi, nu_lo / nu_hi)
    ana = ana.to(dev)
    rel = float(((prof[1:-1] - ana).norm() / ana.norm()))
    umax_ratio = float(prof.max() / ana.max())
    tail = umax_hist[-len(umax_hist) // 5 :]
    frozen = abs(tail[-1][1] - tail[0][1]) / abs(tail[-1][1])
    return {
        **{k: arm[k] for k in ("name", "ny", "tau_r", "tau_b", "gx", "beta", "steps")},
        "stable": True,
        "j_first_blue": jb,
        "relL2": rel,
        "u_max_ratio": umax_ratio,
        "u_max_hist": umax_hist,
        "u_max_frozen_rel_tail20": frozen,
        "mass_drift_r_per_step": abs(float(f_r.sum()) - m_r0) / arm["steps"],
        "mass_drift_b_per_step": abs(float(f_b.sum()) - m_b0) / arm["steps"],
        "profile": [round(float(v), 10) for v in prof.tolist()],
        "wall_s": time.perf_counter() - t0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="two-layer Poiseuille (two-tau CG)")
    p.add_argument("--arms", default=",".join(a["name"] for a in FROZEN_ARMS))
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--src", default=str(SRC), help="repo src/ tree")
    p.add_argument("--out", default=".", type=Path)
    args = p.parse_args()
    if Path(args.src).resolve() != SRC.resolve():
        raise SystemExit(f"--src must match the import tree used at startup ({SRC})")

    step_fn, origin, md5s = _load_engine()
    dev = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    selected = {a["name"]: a for a in FROZEN_ARMS}
    arms = []
    for name in [s.strip() for s in args.arms.split(",") if s.strip()]:
        if name not in selected:
            raise SystemExit(f"unknown arm {name!r}; frozen arms: {list(selected)}")
        arms.append(run_arm(selected[name], step_fn, dev))
        a = arms[-1]
        print(
            f"{a['name']}: relL2={a.get('relL2')} jb={a.get('j_first_blue')} stable={a['stable']}",
            flush=True,
        )

    rel2_max = max(a["relL2"] for a in arms if a.get("relL2") is not None)
    drift_max = max(
        max(a["mass_drift_r_per_step"], a["mass_drift_b_per_step"])
        for a in arms
        if a.get("mass_drift_r_per_step") is not None
    )
    verdict = {
        "gate_rell2": GATE_RELL2,
        "rell2_max": rel2_max,
        "gate_mass_drift": GATE_MASS_DRIFT,
        "mass_drift_max": drift_max,
        "all_stable": all(a["stable"] for a in arms),
        "pass": (
            rel2_max <= GATE_RELL2
            and drift_max <= GATE_MASS_DRIFT
            and all(a["stable"] for a in arms)
        ),
    }
    res = {
        "config": {"nx": NX, "A": A, "dtype": "float64", "device": str(dev)},
        "engine_origin": origin,
        "module_md5": md5s,
        "gate": verdict,
        "arms": arms,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "g2_rerun.json"
    dest.write_text(json.dumps(res, indent=1))
    print(f"saved -> {dest}")
    print(
        f"VERDICT: {'PASS' if verdict['pass'] else 'FAIL'} "
        f"(relL2_max={rel2_max:.4%}, mass_drift_max={drift_max:.3e})"
    )


if __name__ == "__main__":
    main()
