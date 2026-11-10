#!/usr/bin/env python
"""plan-380 n04 campaign: Cassie -> Wenzel transition and droplet impact on a
micro-textured super-hydrophobic wall (SCMP lattice Boltzmann).

Stages
------
``flow``    flat-wall intrinsic-angle calibration, textured static angles, grid
            convergence, quasi-static impalement sweeps
``impact``  We x Oh impact matrix, radius scan for the contact-time exponent,
            energy audit
``report``  reduce everything into ``summary.json`` + figures + ``REPORT.md``

Run with ``--workers N`` (one torch thread per worker).  Results are written
incrementally to ``<out>/results.json`` so a killed run keeps its progress.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

SIGMA_LU = 0.05611
RHO_L = 1.957
RHO_V = 0.1596

#: texture family of the static / state-diagram campaign (P, D, H, theta_Y)
TEXTURES = {
    "T_cassie_130": dict(pitch=16, pillar_width=8, height=20, theta_y_deg=130.0),
    "T_nearcross_100": dict(pitch=16, pillar_width=8, height=20, theta_y_deg=100.0),
    "T_wenzel_80": dict(pitch=16, pillar_width=8, height=20, theta_y_deg=80.0),
    "T_sparse_120": dict(pitch=16, pillar_width=4, height=20, theta_y_deg=120.0),
}
#: impalement family (fixed pitch and height, solid fraction scanned)
IMPALE_TEXTURES = {
    "I_fs0.25": dict(pitch=16, pillar_width=4, height=20, theta_y_deg=120.0),
    "I_fs0.50": dict(pitch=16, pillar_width=8, height=20, theta_y_deg=120.0),
    "I_fs0.75": dict(pitch=16, pillar_width=12, height=20, theta_y_deg=120.0),
}


def _texture(spec: dict):
    from tensorlbm.contact_line.microtexture import PillarTexture

    return PillarTexture(**spec)


def _channel(spec: dict, *, radius: float, height_margin: float = 0.0):
    """Textured channel sized for a drop of ``radius`` (nx multiple of pitch)."""
    from tensorlbm.contact_line.microtexture import TexturedChannel

    tex = _texture(spec)
    d0 = 2.0 * radius
    nx = int(math.ceil(6.0 * d0 / tex.pitch)) * tex.pitch
    ny = int(math.ceil(tex.height + 2.0 * radius + height_margin + 4 * 2 + 12))
    ny = ny + (-ny) % 2
    return TexturedChannel(nx=nx, ny=ny, wall_rows=2, texture=tex)


# ---------------------------------------------------------------------------
# tasks (must be importable: run under the spawn start method)
# ---------------------------------------------------------------------------


def task_flat(spec: dict) -> dict:
    """Flat reference wall (H = 0) at a nominal intrinsic angle."""
    from tensorlbm.contact_line.microtexture import (
        TexturedChannel,
        apparent_angle,
        apparent_angle_n01,
        predicted_wetting_state,
        run_static_angle,
        PillarTexture,
    )

    flat = dict(pitch=16, pillar_width=8, height=0, theta_y_deg=float(spec["theta_y"]))
    tex = _texture(flat)
    r = float(spec.get("radius", 32.0))
    nx = int(math.ceil(6.0 * 2 * r / tex.pitch)) * tex.pitch
    ny = int(math.ceil(2 * r + 40))
    ch = TexturedChannel(nx=nx, ny=ny, wall_rows=2, texture=tex)
    res = run_static_angle(
        ch,
        drop_radius=r,
        relax_steps=int(spec.get("relax", 5000)),
        init_angle_deg=float(spec["theta_y"]),
        forcing=spec.get("forcing", "velocity_shift"),
        scheme="standard",
        checkpoint=None,
        every=10**9,
        field_path=spec.get("field_path"),
    )
    res.update({"kind": "flat", "theta_nominal": float(spec["theta_y"]), "radius": r})
    return res


def task_static(spec: dict) -> dict:
    """Textured static apparent angle + observed wetting state."""
    from tensorlbm.contact_line.microtexture import run_static_angle

    from tensorlbm.contact_line.microtexture import apparent_angle

    tex_spec = dict(spec["tex"])
    r = float(spec["radius"])
    ch = _channel(tex_spec, radius=r)
    trace: list[dict] = []
    trace_every = int(spec.get("trace_every", 0) or 0)
    solid = ch.solid_mask()

    def cb(step, f):
        from tensorlbm.d2q9 import macroscopic

        rho = macroscopic(f)[0]
        ang = apparent_angle(rho, solid, ch.y_wall())
        trace.append({"step": int(step), "theta_deg": ang["theta_deg"],
                      "theta_shape_deg": ang["theta_shape_deg"], "n_points": ang["n_points"]})

    res = run_static_angle(
        ch,
        drop_radius=r,
        relax_steps=int(spec.get("relax", 4000)),
        init_angle_deg=spec.get("init_angle"),
        fill_grooves=bool(spec.get("fill_grooves", False)),
        forcing=spec.get("forcing", "velocity_shift"),
        tau=float(spec.get("tau", 1.0)),
        checkpoint=cb if trace_every else None,
        every=trace_every if trace_every else 10**9,
        field_path=spec.get("field_path"),
    )
    res["trace"] = trace
    res.update(
        {
            "kind": "static",
            "name": spec.get("name", ""),
            "radius": r,
            "domain": [ch.nx, ch.ny],
            "tau": float(spec.get("tau", 1.0)),
        }
    )
    return res


def task_impale(spec: dict) -> dict:
    """Quasi-static Cassie -> Wenzel break-through sweep for one texture."""
    from tensorlbm.contact_line.microtexture import TexturedChannel, run_impalement

    tex_spec = dict(spec["tex"])
    tex = _texture(tex_spec)
    column = float(spec.get("column_height", 30.0))
    nx = int(4 * tex.pitch)
    ny = int(2 + tex.height + column + 10)
    ch = TexturedChannel(nx=nx, ny=ny, wall_rows=2, texture=tex)
    from tensorlbm.contact_line.microtexture import breakthrough_pressure

    p_ref = breakthrough_pressure(tex)["p_c_abs_2d"]
    g_est = p_ref / (RHO_L * column)
    factors = tuple(spec.get("factors", (0.4, 0.8, 1.2, 2.0, 3.0)))
    res = run_impalement(
        ch,
        column_height=column,
        g_values=tuple(f * g_est for f in factors),
        steps_per_g=int(spec.get("steps_per_g", 2500)),
        refine=int(spec.get("refine", 2)),
    )
    res.update(
        {
            "kind": "impale",
            "name": spec.get("name", ""),
            "g_estimate": g_est,
            "domain": [ch.nx, ch.ny],
        }
    )
    return res


def task_impact(spec: dict) -> dict:
    """One impact event (optionally saving density snapshots to ``<out>/frames``)."""
    from tensorlbm.contact_line.microtexture_impact import ImpactRunner

    tex_spec = dict(spec["tex"])
    r = float(spec["radius"])
    ch = _channel(tex_spec, radius=r, height_margin=14.0)
    runner = ImpactRunner(
        ch,
        radius=r,
        we=float(spec["we"]),
        sigma=float(spec.get("sigma", SIGMA_LU)),
        tau=float(spec.get("tau", 1.5)),
        forcing="guo",
        scheme="standard",
        width=float(spec.get("width", 1.5)),
        fill_grooves=bool(spec.get("fill_grooves", False)),
    )
    want_frames = bool(spec.get("snapshot_path"))
    res = runner.run(
        steps=int(spec.get("steps", 0)) or None,
        every=int(spec.get("every", 20)),
        clusters_every=int(spec.get("clusters_every", 100)),
        snapshot_path=spec.get("snapshot_path") if want_frames else None,
        snapshots=tuple(spec.get("snapshots", (0.25, 0.5, 1.0))),
    )
    frames = res.pop("snapshot_frames", {})
    if frames:
        path = Path(spec["snapshot_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            times=np.array([int(k) for k in frames]),
            phi=np.array([frames[k] for k in frames]),
        )
        res["snapshot_file"] = str(path)
        res["snapshot_times"] = [int(k) for k in frames]
    res.update(
        {
            "kind": "impact",
            "name": spec.get("name", ""),
            "name_str": spec.get("name", ""),
            "domain": [ch.nx, ch.ny],
            "texture": tex_spec,
        }
    )
    if not spec.get("keep_series", True):
        res.pop("series", None)
    return res


def task_n01_baseline(spec: dict) -> dict:
    """The smooth-wall ``wetting_n01`` pipeline on its own (regression anchor)."""
    from tensorlbm.wetting_n01 import run_single_case
    from tensorlbm.contact_line.microtexture import (
        TexturedChannel,
        apparent_angle_n01,
        run_static_angle,
    )
    from tensorlbm.contact_line.engine import WallStepper, sessile_drop_state
    from tensorlbm.wetting_n01 import angle_to_wall_fraction, _mm_to_lu
    from tensorlbm.contact_line.microtexture import PillarTexture

    theta = float(spec["theta"])
    r_mm = float(spec.get("r_mm", 1.0))
    relax = int(spec.get("relax", 5000))

    ref = run_single_case(theta, r_mm, relax_steps=relax)

    # same geometry / initialisation through the microtexture module (H = 0)
    r_lu = _mm_to_lu(r_mm)
    nx = max(int(4 * r_lu) + 32, 64)
    ny = max(int(3 * r_lu) + 24, 48)
    if nx % 2:
        nx += 1
    tex = PillarTexture(pitch=16, pillar_width=8, height=0, theta_y_deg=theta,
                        wall_fraction=angle_to_wall_fraction(theta))
    ch = TexturedChannel(nx=nx, ny=ny, wall_rows=2, texture=tex)
    drop_radius = max(r_lu * 0.8, 6.0)
    st = WallStepper(ch, G=5.0, tau=1.0, rho_l=RHO_L, rho_v=RHO_V,
                     wall_fraction=tex.wetting_fraction)
    f, _ = sessile_drop_state(ch, drop_radius, rho_l=RHO_L, rho_v=RHO_V)
    f = st.run(f, relax)
    rho = st.density(f)
    ang = apparent_angle_n01(rho, ch.solid_mask(), ch.y_wall())
    return {
        "kind": "n01_regression",
        "theta_nominal": theta,
        "r_mm": r_mm,
        "relax_steps": relax,
        "n01_theta_deg": ref.theta_measured_deg,
        "n01_error_deg": ref.theta_error_deg,
        "module_theta_deg": ang["theta_deg"],
        "delta_deg": abs(ang["theta_deg"] - ref.theta_measured_deg),
        "n01_capillary_rise": ref.capillary_rise_lu,
        "n01_capillary_rise_analytical": ref.capillary_rise_analytical_lu,
    }


TASKS = {
    "flat": task_flat,
    "static": task_static,
    "impale": task_impale,
    "impact": task_impact,
    "n01": task_n01_baseline,
}


def _run_one(spec: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    kind = spec.pop("task")
    t0 = time.time()
    try:
        out = TASKS[kind](spec)
    except Exception:  # pragma: no cover - reported, never silent
        return {"task": kind, "spec": spec, "error": traceback.format_exc(), "seconds": time.time() - t0}
    out["seconds"] = time.time() - t0
    out["spec"] = spec
    return out


# ---------------------------------------------------------------------------
# task list
# ---------------------------------------------------------------------------


def build_tasks(stages: set[str]) -> list[dict]:
    tasks: list[dict] = []
    if "flow" in stages:
        for theta in (80.0, 100.0, 120.0, 130.0):
            tasks.append({"task": "flat", "theta_y": theta, "radius": 32.0, "relax": 5000})
        tasks.append({"task": "n01", "theta": 120.0, "r_mm": 1.0, "relax": 5000})
        for name, tex in TEXTURES.items():
            init = None
            tasks.append(
                {
                    "task": "static",
                    "name": name,
                    "tex": tex,
                    "radius": 32.0,
                    "relax": 4000,
                    "init_angle": init,
                    "fill_grooves": name.startswith("T_wenzel"),
                }
            )
        # initial-condition independence: same texture started from theta+20 deg
        tasks.append(
            {
                "task": "static",
                "name": "T_cassie_130_init150",
                "tex": TEXTURES["T_cassie_130"],
                "radius": 32.0,
                "relax": 4000,
                "init_angle": 150.0,
            }
        )
        # grid convergence: coarse (H=20, P=16) vs refined (H=30, P=24), all
        # lengths scaled by 1.5 and the drop radius with them
        tasks.append(
            {
                "task": "static",
                "name": "T_cassie_130_fine",
                "tex": dict(pitch=24, pillar_width=12, height=30, theta_y_deg=130.0),
                "radius": 48.0,
                "relax": 4000,
            }
        )
        tasks.append(
            {
                "task": "flat",
                "theta_y": 130.0,
                "radius": 32.0,
                "relax": 8000,
                "name": "flat130_long",
                "field_path": str(REPO / "artifacts" / "p380_n04_wetting_transition" / "fields" / "flat130_long.npz"),
            }
        )
        tasks.append(
            {
                "task": "static",
                "name": "T_cassie_130_long",
                "tex": TEXTURES["T_cassie_130"],
                "radius": 32.0,
                "relax": 8000,
                "field_path": str(REPO / "artifacts" / "p380_n04_wetting_transition" / "fields" / "cassie130_long.npz"),
            }
        )
        for name, tex in IMPALE_TEXTURES.items():
            tasks.append(
                {"task": "impale", "name": name, "tex": tex, "column_height": 30.0,
                 "steps_per_g": 2500, "refine": 2}
            )
    if "impact" in stages:
        for we in (10.0, 20.0, 40.0, 60.0, 90.0, 120.0, 160.0, 200.0):
            task = {
                "task": "impact",
                "name": f"cassie_We{we:.0f}",
                "tex": TEXTURES["T_cassie_130"],
                "radius": 40.0,
                "we": we,
                "tau": 1.5,
            }
            if we == 40.0:
                task["snapshot_path"] = str(
                    REPO / "artifacts" / "p380_n04_wetting_transition" / "fields" / "cassie_We40_frames.npz"
                )
            tasks.append(task)
        for we in (40.0, 120.0):
            tasks.append(
                {
                    "task": "impact",
                    "name": f"wenzel_We{we:.0f}",
                    "tex": TEXTURES["T_wenzel_80"],
                    "radius": 40.0,
                    "we": we,
                    "tau": 1.5,
                    "fill_grooves": True,
                }
            )
        for r in (24.0, 32.0):
            tasks.append(
                {
                    "task": "impact",
                    "name": f"cassie_R{r:.0f}_We40",
                    "tex": TEXTURES["T_cassie_130"],
                    "radius": r,
                    "we": 40.0,
                    "tau": 1.5,
                }
            )
            # the radius scan for the contact-time exponent needs *detaching*
            # impacts, which on this viscous kernel happen at low We
            tasks.append(
                {
                    "task": "impact",
                    "name": f"cassie_R{r:.0f}_We20",
                    "tex": TEXTURES["T_cassie_130"],
                    "radius": r,
                    "we": 20.0,
                    "tau": 1.5,
                }
            )
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="flow,impact", help="comma separated subset of flow,impact,report")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(REPO / "artifacts" / "p380_n04_wetting_transition"))
    ap.add_argument("--only", default="", help="comma separated task names to run")
    args = ap.parse_args()

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"

    results: list[dict] = []
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text())
        except Exception:
            results = []

    tasks = build_tasks(stages)
    if args.only:
        keep = {t.strip() for t in args.only.split(",") if t.strip()}
        tasks = [t for t in tasks if t.get("name") in keep or t.get("task") in keep]

    done = {(r.get("task"), r.get("name")) for r in results if "error" not in r}
    tasks = [t for t in tasks if (t.get("task"), t.get("name")) not in done]

    print(f"[campaign] {len(tasks)} tasks, workers={args.workers}, out={out}", flush=True)
    if tasks:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=max(1, args.workers)) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_one, tasks, chunksize=1), 1):  # type: ignore[arg-type]
                results.append(res)
                tag = f"{res.get('task')}:{res.get('name', '')}"
                status = "ERROR" if "error" in res else "ok"
                print(f"[{i}/{len(tasks)}] {tag} {status} {res.get('seconds', 0):.0f}s", flush=True)
                results_path.write_text(json.dumps(results, indent=1, default=float))
    results_path.write_text(json.dumps(results, indent=1, default=float))
    print(f"[campaign] wrote {results_path} ({len(results)} records)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())