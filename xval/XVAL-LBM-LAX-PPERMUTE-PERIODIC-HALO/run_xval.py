"""Orchestrator for XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO.

Usage:
    python xval/XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO/run_xval.py [--quick]

Writes ``artifacts/result.json`` (every arm, every configuration) and prints a
compact table.  Nothing in the report is hand-copied from here on: the report
generator reads ``result.json`` only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import arms  # noqa: E402

NX, NY = 32, 24          # divisible by 2 and 4
TAU = 0.6
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------------
# 2-D vivsim arm
# ---------------------------------------------------------------------------


def gt_2d(f0, n_steps):
    import vivsim.lbm.basic as basic

    C, W, _ = arms.d2q9_tables()
    f = f0.copy()
    for _ in range(n_steps):
        f = arms.stream_global_2d(arms.collide(f, TAU, C, W), basic)
    return f


def err_2d(got, gt):
    num = float(np.linalg.norm(got - gt))
    den = float(np.linalg.norm(gt))
    return {"rel_l2": num / den, "max_abs": float(np.abs(got - gt).max()),
            "bitwise_equal": bool(np.array_equal(got, gt))}


ARMSPEC = (("REF-A", False), ("REF-B", False), ("REF-C", False), ("REF-CS", True))


def run_2d(n_steps: int, ranks_list, variants, seeds, kinds):
    rows = []
    cache = {}
    for kind in kinds:
        for seed in seeds:
            key = (kind, seed)
            f0 = arms.make_field_2d(NX, NY, kind, seed)
            if key not in cache:
                cache[key] = (f0, gt_2d(f0, n_steps))
            f0, gt = cache[key]
            for axis in ("y", "x"):
                for n in ranks_list:
                    for variant, comp in variants:
                        t0 = time.perf_counter()
                        got = arms.run_ref_arm(f0, axis, n, n_steps, TAU, variant,
                                               inplane_comp=comp)
                        rows.append(
                            {
                                "family": "REF",
                                "arm": variant,
                                "axis": axis,
                                "ranks": n,
                                "field": kind,
                                "seed": seed,
                                "n_steps": n_steps,
                                "grid_2d": [NX, NY],
                                "tau": TAU,
                                **err_2d(got, gt),
                                "wall_s": round(time.perf_counter() - t0, 3),
                            }
                        )
                        print(
                            f"2D {variant} axis={axis} n={n} field={kind} seed={seed}: "
                            f"rel_l2={rows[-1]['rel_l2']:.3e}",
                            flush=True,
                        )
    return rows


# ---------------------------------------------------------------------------
# 3-D GOT arm (real halo code, gloo)
# ---------------------------------------------------------------------------


def _worker_env(n, arm, field, seed, wire, steps, port):
    env = os.environ.copy()
    env.update(
        WORLD_SIZE=str(n),
        XVALS_ARM=arm,
        XVALS_FIELD=field,
        XVALS_SEED=str(seed),
        XVALS_WIRE=wire,
        XVALS_STEPS=str(steps),
        XVALS_MASTER=f"tcp://127.0.0.1:{port}",
        PYTHONPATH=str(HERE) + os.pathsep + env.get("PYTHONPATH", ""),
    )
    env.pop("RANK", None)
    return env


def run_3d_one(n, arm="GOT", field="rand", seed=0, wire="fp64", steps=20, port=29517):
    env = _worker_env(n, arm, field, seed, wire, steps, port)
    t0 = time.perf_counter()
    if arm == "GOT-W1FILL":
        arm = "GOT"
        env["XVALS_W1FILL"] = "1"
    if n == 1:
        # single-rank: real single-rank halo branch, no process group needed
        env["RANK"] = "0"
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(port)
        proc = subprocess.run(
            [sys.executable, str(HERE / "got_worker.py")],
            env=env, capture_output=True, text=True, timeout=1800,
        )
    else:
        proc = subprocess.run(
            [sys.executable, "-m", "torch.distributed.run", "--standalone",
             f"--nproc_per_node={n}", str(HERE / "got_worker.py")],
            env=env, capture_output=True, text=True, timeout=1800,
        )
    line = [l for l in proc.stdout.splitlines() if l.startswith("XVALS_METRICS=")]
    if not line:
        raise RuntimeError(
            f"no metrics (rc={proc.returncode})\nstdout:\n{proc.stdout[-3000:]}\n"
            f"stderr:\n{proc.stderr[-3000:]}"
        )
    m = json.loads(line[-1].split("=", 1)[1])
    m["family"] = "GOT" if arm == "GOT" else "CONTROL"
    m["wall_s"] = round(time.perf_counter() - t0, 3)
    if n == 1 and m["arm"] == "GOT":
        # gloo is not initialised at world=1; make the label explicit
        m["arm"] = "GOT"
    print(
        f"3D {arm} n={n} field={field} seed={seed} wire={wire}: "
        f"rel_l2={m['rel_l2']:.3e} bitwise={m['bitwise_equal']}",
        flush=True,
    )
    return m


def run_3d(steps: int, quick: bool):
    rows = []
    seeds = SEEDS if not quick else (0,)
    for seed in seeds:
        for n in (1, 2, 4):
            rows.append(run_3d_one(n, "GOT", "rand", seed, "fp64", steps, 29517 + n + seed))
    for seed in seeds:
        rows.append(run_3d_one(2, "GOT", "tgv", seed, "fp64", steps, 29540 + seed))
        rows.append(run_3d_one(4, "GOT", "tgv", seed, "fp64", steps, 29550 + seed))
    rows.append(run_3d_one(1, "GOT-W1FILL", "rand", 0, "fp64", steps, 29535))
    rows.append(run_3d_one(2, "GOT", "rand", 0, "fp16", steps, 29560))
    rows.append(run_3d_one(4, "GOT", "rand", 0, "fp16", steps, 29561))
    rows.append(run_3d_one(2, "CONTROL-LANE", "rand", 0, "fp64", steps, 29570))
    return rows


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    steps = args.steps
    ranks_list = (2, 4)
    variants = ARMSPEC
    seeds = SEEDS if not args.quick else (0,)
    kinds = ("asym", "sym")

    rows2d = run_2d(steps, ranks_list, variants, seeds, kinds)
    # step-count sensitivity: a wrong halo is a persistent bias, not noise
    sweep = []
    f0 = arms.make_field_2d(NX, NY, "asym", 0)
    import vivsim.lbm.basic as basic
    for nst in (1, 5, 20, 60):
        C, W, _ = arms.d2q9_tables()
        f = f0.copy()
        for _ in range(nst):
            f = arms.stream_global_2d(arms.collide(f, TAU, C, W), basic)
        gt = f
        for variant, comp in (("REF-A", False), ("REF-C", False), ("REF-C", True)):
            got = arms.run_ref_arm(f0, "y", 4, nst, TAU, variant, inplane_comp=comp)
            sweep.append({"family": "SWEEP", "arm": variant + ("+CS" if comp else ""),
                          "axis": "y", "ranks": 4, "field": "asym", "seed": 0,
                          "n_steps": nst, **err_2d(got, gt)})
    print("sweep done")
    rows3d = run_3d(steps, args.quick)

    out = {
        "anchor": "XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO",
        "mission_id": 17868,
        "generated_by": "run_xval.py",
        "torch": __import__("torch").__version__,
        "numpy": np.__version__,
        "jax": "emulated (jax_shim.py)",
        "rows_2d": rows2d,
        "rows_sweep": sweep,
        "rows_3d": rows3d,
    }
    (HERE / "artifacts").mkdir(exist_ok=True)
    (HERE / "artifacts" / "result.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote {HERE / 'artifacts' / 'result.json'} "
          f"({len(rows2d)} 2D rows, {len(rows3d)} 3D rows)")


if __name__ == "__main__":
    main()