"""2D cylinder drag / shedding benchmark — real LBM execution, single source of truth.

Why this file exists
--------------------
`docs/LBM_VERIFICATION_TEACHING.md` carried a "Step E" cylinder-Cd
grid-convergence table (1.48 / 1.42 / 1.39 / 1.29) with **no executable
evidence anywhere in the repo**, and `tests/test_cylinder_grid_convergence.py`
merely feeds *synthetic* records (``cd = 1.33 + 10/D**2``) into the analyzer —
it validates the analyzer's arithmetic, not the solver.  The loose
``1.0 < Cd < 4.0`` band in `tests/test_cylinder_cd.py` has the same problem in
milder form: a 3x error still passes.

This module is the executable evidence.  It runs the repo's own D2Q9 solver
(``tensorlbm.d2q9`` + ``tensorlbm.solver`` + ``tensorlbm.boundary``) for
channel flow past a circular cylinder at Re=100 in three configurations:

  * ``dfg2d2``  — geometry of the DFG / Schaefer-Turek 2D-2 benchmark
                  (D/H = 0.2439, 2D upstream, 20D downstream), tight literature
                  anchor Cd ~ 3.22-3.24, St ~ 0.295-0.305.
  * ``free_B*``  — the same cylinder at low blockage (H = 8D, 16D) so that a
                  B -> 0 extrapolation can be compared with the **unconfined**
                  2D anchors Cd = 1.32..1.35, St = 0.164..0.166.

The two anchor families must never be mixed: DFG 2D-2 is *confined* (B=0.244)
with a parabolic inlet, the unconfined values are not.  That mix-up is exactly
what the old WITHHELD verdict was about, so both are recorded here with their
applicability flags.

Boundary conditions (repo defaults, deliberately unchanged):
  * inlet  — plug velocity (``apply_simple_channel_boundaries``), NOT the DFG
             parabolic profile; the resulting bias is quantified in the report.
  * outlet — zero-gradient copy (``f[:, :, -1] = f[:, :, -2]``).
  * walls/cylinder — halfway bounce-back (``bounce_back_cells``), force by the
             Ladd momentum-exchange formula (``compute_obstacle_forces``).

Performance note (matters on a loaded 128-core host): pinning
``torch.set_num_threads(1)`` turns ~500 ms/step into ~20 ms/step for these
~10^4-cell tensors, because the default 128-thread pool is pure overhead at
this size.  ``collide_bgk_matmul`` is used after a unit test proves it agrees
with ``collide_bgk`` to float32 round-off (1.3e-5 relative after 200 steps).

Usage
-----
    python -m tests.validation.cylinder_drag_benchmark --pilot
    python -m tests.validation.cylinder_drag_benchmark --full --emit-dir artifacts/cylinder_drag
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field

import torch

from tensorlbm.boundary.boundaries import (
    apply_simple_channel_boundaries,
    apply_zou_he_channel_boundaries,
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
)
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import collide_bgk_matmul, correct_mass, stream

# ---------------------------------------------------------------------------
# literature anchors (kept separate: different boundary conditions!)
# ---------------------------------------------------------------------------
ANCHORS = {
    "unconfined_re100_cd": {
        "value": 1.32,
        "range": [1.32, 1.35],
        "source": "Dennis & Chang 1970 JFM 42:471 (=1.32); 1.33-1.35 in later 2D studies",
        "comparable": True,
    },
    "unconfined_re100_st": {
        "value": 0.164,
        "range": [0.164, 0.166],
        "source": "Williamson 1989 / Norberg 1994, unconfined 2D cylinder",
        "comparable": True,
    },
    "dfg2d2_cd": {
        "value": 3.23,
        "range": [3.22, 3.24],
        "source": "Schafer & Turek, DFG 2D-2 (Re=100, D/H=0.2439, parabolic inlet)",
        "comparable": True,
        "note": "confined case: only comparable with the dfg2d2 configuration",
    },
    "dfg2d2_st": {
        "value": 0.300,
        "range": [0.295, 0.305],
        "source": "Schafer & Turek, DFG 2D-2",
        "comparable": True,
    },
}


@dataclass
class Config:
    """One cylinder-channel run (all lengths in lattice units)."""

    name: str
    re: float = 100.0
    diameter: int = 10
    height_cells: int = 81  # H, channel height in cells (D/H = blockage)
    upstream_d: float = 5.0  # inlet -> cylinder centre, in diameters
    downstream_d: float = 14.0
    u_in: float = 0.06
    min_warmup_steps: int = 8000
    stats_steps: int = 7000
    max_steps: int = 26000
    boundary: str = "simple"  # "simple" (repo default) or "zou_he" (pressure outlet)
    mass_correction: bool = True  # pin total mass: the simple BC drifts it (see docs)
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class Measurement:
    config: dict
    derived: dict = field(default_factory=dict)
    samples: dict = field(default_factory=dict)

    @property
    def cd(self) -> float:
        return self.derived["cd_mean"]

    @property
    def st(self) -> float:
        return self.derived["strouhal"]


def run_cfg(cfg: Config, verbose: bool = True) -> Measurement:
    """Run one cylinder configuration and return the measured Cd / St."""
    dev = torch.device(cfg.device)
    dt = getattr(torch, cfg.dtype)
    radius = cfg.diameter / 2.0
    ny = cfg.height_cells
    nx = int(round((cfg.upstream_d + cfg.downstream_d) * cfg.diameter))
    cx = cfg.upstream_d * cfg.diameter
    cy = ny // 2
    nu = cfg.u_in * cfg.diameter / cfg.re
    tau = 3.0 * nu + 0.5

    mask = cylinder_mask(nx, ny, cx, cy, radius, device=dev)
    wall = make_channel_wall_mask(ny, nx, mask, device=dev)
    none_mask = torch.zeros_like(mask)

    rho = torch.ones(ny, nx, device=dev, dtype=dt)
    ux = torch.full_like(rho, cfg.u_in)
    f = equilibrium(rho, ux, torch.zeros_like(rho), device=dev).to(dt)
    q = 0.5 * cfg.u_in**2 * cfg.diameter  # dynamic pressure x projected area
    target_mass = float(f.sum().item())  # total populations set by the initial state

    cd_hist: list[float] = []
    cl_hist: list[float] = []
    t0 = time.time()
    stop_reason = "max_steps"
    drift = float("nan")
    with torch.inference_mode():
        for step in range(1, cfg.max_steps + 1):
            f = collide_bgk_matmul(f, tau=tau)
            f = stream(f)
            fx, fy = compute_obstacle_forces(f, mask)
            f = _apply_bc(cfg.boundary, f, cfg.u_in, wall, none_mask)
            if cfg.mass_correction:
                f = correct_mass(f, target_mass)
            f = bounce_back_cells(f, mask)
            if step > cfg.min_warmup_steps:
                cd_hist.append(float(fx.item()) / q)
                cl_hist.append(float(fy.item()) / q)
            if verbose and step % 2000 == 0:
                cd = cd_hist[-1] if cd_hist else float("nan")
                print(
                    f"  [{cfg.name}] {step}/{cfg.max_steps} Cd~{cd:.4f} "
                    f"samples={len(cd_hist)} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            # stationarity gate: two-window mean drift on the drag history
            if len(cd_hist) >= cfg.stats_steps and len(cd_hist) % 2000 == 0:
                half = len(cd_hist) // 2
                drift = 100.0 * abs(_mean(cd_hist[half:]) - _mean(cd_hist[:half])) / _mean(cd_hist)
                if drift < 1.0:
                    stop_reason = "stationary"
                    break
    wall_s = time.time() - t0
    flow_through_steps = nx / cfg.u_in

    period_steps = _period_steps(cl_hist)
    st = (
        (cfg.diameter / (period_steps * cfg.u_in)) if period_steps == period_steps else float("nan")
    )
    half = len(cd_hist) // 2
    return Measurement(
        config=asdict(cfg),
        derived={
            "nx": nx,
            "ny": ny,
            "cells": nx * ny,
            "nu": nu,
            "tau": tau,
            "mach": cfg.u_in / (1.0 / 3.0) ** 0.5,
            "blockage": radius * 2.0 / ny,
            "cd_mean": _mean(cd_hist),
            "cd_half_drift_pct": 100.0
            * abs(_mean(cd_hist[half:]) - _mean(cd_hist[:half]))
            / _mean(cd_hist),
            "cd_amp": 0.5 * (max(cd_hist) - min(cd_hist)),
            "cl_amp": 0.5 * (max(cl_hist) - min(cl_hist)),
            "strouhal": st,
            "period_steps": period_steps,
            "samples": len(cd_hist),
            "wall_seconds": wall_s,
            "warmup_flow_throughs": cfg.min_warmup_steps / flow_through_steps,
            "stop_reason": stop_reason,
            "boundary": cfg.boundary,
            "mass_correction": cfg.mass_correction,
            "stationarity_drift_pct": drift,
        },
        samples={"cd_tail": cd_hist, "cl_tail": cl_hist},
    )


def _apply_bc(
    name: str, f: torch.Tensor, u_in: float, wall: torch.Tensor, obstacle: torch.Tensor
) -> torch.Tensor:
    """Channel boundary selector (see docs/cylinder_drag_benchmark_r1.md)."""
    if name in ("zou_he", "zouhe", "zh"):
        return apply_zou_he_channel_boundaries(f, u_in=u_in, wall_mask=wall, obstacle_mask=obstacle)
    if name == "simple":
        return apply_simple_channel_boundaries(f, u_in=u_in, wall_mask=wall, obstacle_mask=obstacle)
    raise ValueError(f"unknown boundary set {name!r}")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _period_steps(cl_hist: list[float]) -> float:
    """Shedding period in lattice steps from linear-interpolated up-crossings.

    Robust for the near-sinusoidal lift of a shedding cylinder and free of FFT
    bin resolution limits.  Returns nan for steady flow (St undefined).
    """
    if len(cl_hist) < 64:
        return float("nan")
    mean = _mean(cl_hist)
    sig = (_mean([(c - mean) ** 2 for c in cl_hist])) ** 0.5
    if sig < 1e-6:
        return float("nan")
    crossings: list[float] = []
    prev = cl_hist[0] - mean
    for i in range(1, len(cl_hist)):
        cur = cl_hist[i] - mean
        if prev < 0.0 <= cur:
            crossings.append(i - 1 + (-prev / (cur - prev)))
        prev = cur
    if len(crossings) < 3:
        return float("nan")
    return (crossings[-1] - crossings[0]) / (len(crossings) - 1)


def extrapolate_blockage(measurements: list[Measurement]) -> dict:
    """Least-squares B -> 0 extrapolation of Cd and St over a blockage series."""
    pts = [
        (m.derived["blockage"], m.derived["cd_mean"], m.derived["strouhal"]) for m in measurements
    ]
    pts = [p for p in pts if p[0] == p[0]]
    if len(pts) < 2:
        return {}
    n = len(pts)
    sb = sum(p[0] for p in pts)
    sbb = sum(p[0] ** 2 for p in pts)
    den = n * sbb - sb * sb
    if abs(den) < 1e-15:
        return {}
    out: dict = {"n_points": n, "blockages": [p[0] for p in pts]}
    for idx, key in ((1, "cd_b0"), (2, "st_b0")):
        vals = [p[idx] for p in pts]
        if any(v != v for v in vals):
            continue
        sy = sum(vals)
        sby = sum(p[0] * p[idx] for p in pts)
        slope = (n * sby - sb * sy) / den
        out[key] = (sy - slope * sb) / n
        out[key + "_slope"] = slope
    return out


def default_configs(device: str = "cpu") -> list[Config]:
    """Three configurations: one tight confined anchor + a low-blockage series."""
    return [
        # DFG 2D-2 geometry at lattice resolution D=10, H=41 cells (=4.1D)
        Config(
            name="dfg2d2",
            diameter=10,
            height_cells=41,
            upstream_d=2.0,
            downstream_d=20.0,
            u_in=0.1,
            min_warmup_steps=8000,
            stats_steps=6000,
            max_steps=30000,
            device=device,
        ),
        # low-blockage points for the B -> 0 extrapolation (H = 8D and 16D)
        Config(
            name="free_B0.125",
            diameter=10,
            height_cells=81,
            upstream_d=4.0,
            downstream_d=12.0,
            u_in=0.08,
            min_warmup_steps=8000,
            stats_steps=7000,
            max_steps=26000,
            device=device,
        ),
        Config(
            name="free_B0.0625",
            diameter=10,
            height_cells=161,
            upstream_d=4.0,
            downstream_d=12.0,
            u_in=0.08,
            min_warmup_steps=8000,
            stats_steps=7000,
            max_steps=26000,
            device=device,
        ),
    ]


def build_registry(measurements: list[Measurement]) -> dict:
    low_b = [m for m in measurements if m.config["name"].startswith("free_")]
    dfg = [m for m in measurements if m.config["name"] == "dfg2d2"]
    ext = extrapolate_blockage(low_b)
    reg = {
        "schema": "tensorlbm-cylinder-drag-benchmark-v1",
        "generated_by": "tests/validation/cylinder_drag_benchmark.py",
        "anchors": ANCHORS,
        "runs": [{"config": m.config, "derived": m.derived} for m in measurements],
        "extrapolation": ext,
        "comparisons": [],
    }
    if ext.get("cd_b0") is not None:
        ref = ANCHORS["unconfined_re100_cd"]
        reg["comparisons"].append(
            {
                "quantity": "Cd (B -> 0 extrapolated)",
                "measured": ext["cd_b0"],
                "reference": ref["value"],
                "reference_range": ref["range"],
                "rel_err_pct": 100.0 * (ext["cd_b0"] - ref["value"]) / ref["value"],
                "source": ref["source"],
            }
        )
    if ext.get("st_b0") is not None:
        ref = ANCHORS["unconfined_re100_st"]
        reg["comparisons"].append(
            {
                "quantity": "St (B -> 0 extrapolated)",
                "measured": ext["st_b0"],
                "reference": ref["value"],
                "reference_range": ref["range"],
                "rel_err_pct": 100.0 * (ext["st_b0"] - ref["value"]) / ref["value"],
                "source": ref["source"],
            }
        )
    for m in dfg:
        for key, qty in (("cd_mean", "dfg2d2_cd"), ("strouhal", "dfg2d2_st")):
            ref = ANCHORS[qty]
            val = m.derived[key]
            if val != val:
                continue
            reg["comparisons"].append(
                {
                    "quantity": f"{qty} ({m.config['name']})",
                    "measured": val,
                    "reference": ref["value"],
                    "reference_range": ref["range"],
                    "rel_err_pct": 100.0 * (val - ref["value"]) / ref["value"],
                    "source": ref["source"],
                }
            )
    return reg


def markdown_tables(reg: dict) -> str:
    lines = [
        "| run | D | H (cells) | B = D/H | grid | Ma | steps | Cd | St | wall s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reg["runs"]:
        c, d = r["config"], r["derived"]
        st = d["strouhal"]
        st_txt = "%.4f" % st if st == st else "steady"
        lines.append(
            "| {} | {} | {} | {:.4f} | {}\u00d7{} | {:.3f} | {} | {:.4f} | {} | {:.0f} |".format(
                c["name"],
                c["diameter"],
                d["ny"],
                d["blockage"],
                d["nx"],
                d["ny"],
                d["mach"],
                d["ny"] and c.get("min_warmup_steps", 0) + d["samples"],
                d["cd_mean"],
                st_txt,
                d["wall_seconds"],
            )
        )
    lines += [
        "",
        "| quantity | measured | reference | range | rel. err | source |",
        "|---|---|---|---|---|---|",
    ]
    for c in reg["comparisons"]:
        lines.append(
            f"| {c['quantity']} | {c['measured']:.4f} | {c['reference']:.4f} | "
            f"{c['reference_range'][0]}\u2013{c['reference_range'][1]} | "
            f"{c['rel_err_pct']:+.2f}% | {c['source']} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="cheap single-run sanity check")
    ap.add_argument("--full", action="store_true", help="run the 3 benchmark configurations")
    ap.add_argument("--only", default=None, help="run a single named configuration")
    ap.add_argument("--scale", type=float, default=1.0, help="scale step counts (pilots)")
    ap.add_argument("--boundary", default=None, help="override boundary set")
    ap.add_argument("--u-in", type=float, default=None, help="override inlet velocity")
    ap.add_argument("--emit-dir", metavar="DIR", help="write registry JSON + sample CSVs here")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TENSORLBM_THREADS", "1")))

    if args.pilot:
        m = run_cfg(
            Config(
                name="pilot",
                diameter=8,
                height_cells=65,
                u_in=0.08,
                min_warmup_steps=1500,
                stats_steps=800,
                max_steps=4000,
                device=args.device,
            )
        )
        print(json.dumps(m.derived, indent=1))
        return

    cfgs = default_configs(args.device)
    if args.only:
        cfgs = [c for c in cfgs if c.name == args.only]
        assert cfgs, f"no configuration named {args.only!r}"
    elif not args.full:
        cfgs = cfgs[:1]
    for c in cfgs:
        if args.boundary:
            c.boundary = args.boundary
        if args.u_in:
            c.u_in = args.u_in
        c.min_warmup_steps = int(c.min_warmup_steps * args.scale)
        c.stats_steps = int(c.stats_steps * args.scale)
        c.max_steps = int(c.max_steps * args.scale)
    out_dir = args.emit_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    measurements: list[Measurement] = []
    for cfg in cfgs:
        m = run_cfg(cfg)
        measurements.append(m)
        print(json.dumps({"name": cfg.name, **m.derived}, indent=1), flush=True)
        if out_dir:
            with open(os.path.join(out_dir, f"run_{cfg.name}.json"), "w") as fh:
                json.dump({"config": m.config, "derived": m.derived}, fh, indent=1)
    reg = build_registry(measurements)
    print(markdown_tables(reg))
    if out_dir:
        with open(os.path.join(out_dir, "cylinder_drag_registry.json"), "w") as fh:
            json.dump(reg, fh, indent=1)
        if len(measurements) == len(cfgs):
            import csv

            with open(os.path.join(out_dir, "samples.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["run", "step", "cd", "cl"])
                for m in measurements:
                    for i, (cd, cl) in enumerate(zip(m.samples["cd_tail"], m.samples["cl_tail"])):
                        w.writerow([m.config["name"], i, cd, cl])
        print("wrote", out_dir)


if __name__ == "__main__":
    main()
