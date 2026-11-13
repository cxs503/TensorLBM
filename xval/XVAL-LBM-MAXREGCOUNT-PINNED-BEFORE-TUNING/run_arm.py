"""XVAL arm runner: is the resource budget pinned *before* tuning?

Anchor: ``XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING`` (mission 17957).
Design and thresholds: ``predeclared.json`` next to this file -- read it first;
nothing in here redefines a threshold.

What one launch of this script does
-----------------------------------
1. **Tuning grid** (the thing that gets tuned): the D2Q9 BGK collide+stream
   whole step of ``tensorlbm.solver`` in three implementations
   (``collide_bgk`` / ``collide_bgk_fused`` / ``collide_bgk_matmul``) x two
   compile modes (eager / ``torch.compile('default')`` from
   :mod:`tensorlbm.compile_utils`) on a 128x128 periodic shear wave.
   For each configuration: ``warmup_steps`` warm-up, then ``repeats`` blocks of
   ``timed_steps`` steps; the KPI is the median per-step wall clock.
2. **Resource budget**: arm ``U`` (status quo) runs the grid at whatever thread
   budget the launch inherits from the predeclared drift set (the thread values
   actually present in this repo); arm ``P`` runs the grid inside
   :func:`tensorlbm.bench_pinning.pinned_resources` at the predeclared pin.
   Nothing else differs between the arms.
3. **Physics anchor** (proves the pin is physics-neutral): a plane Poiseuille
   channel of the same D2Q9/BGK lattice with Guo forcing and half-way
   bounce-back, compared with the analytic parabola; ``u_max`` and the
   cross-section L2 error are returned together with the pin audit.

Modes
-----
``--mode tune``    : one launch of the tuning grid + the physics anchor.
``--mode ladder``  : one process that sweeps the thread ladder for every
                    configuration (the direct measure of budget sensitivity).
``--mode physics`` : the physics anchor only (used for the pin-neutrality pair).

Every JSON written carries the pin audit (or an explicit ``pinned: false``
record for arm U), so no number can be read without the budget it came from.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tensorlbm.bench_pinning import (  # noqa: E402
    ResourcePin,
    assert_pinned,
    pin_audit,
    pinned_resources,
)
from tensorlbm.compile_utils import compile_step  # noqa: E402
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.solver import (  # noqa: E402
    collide_bgk,
    collide_bgk_fused,
    collide_bgk_matmul,
    stream,
)

# --------------------------------------------------------------------------
# Predeclared design constants (mirrored in predeclared.json)
# --------------------------------------------------------------------------
PIN_THREADS = 8          # the pinned budget of arm P
DRIFT_SET = [128, 16, 1, 4, 8]  # arm U: one inherited budget per launch, fixed order
N_TUNE = 128             # tuning grid resolution (periodic, square)
TAU = 0.8                # nu = cs^2 (tau - 1/2) = 0.1
WARMUP_STEPS = 10
TIMED_STEPS = 40
REPEATS = 5
COMPILE_MODES = (None, "default")
IMPLS = ("bgk", "bgk_fused", "bgk_matmul")
# Poiseuille anchor (predeclared)
CH_H = 16                # fluid rows between the two half-way walls
CH_NX = 8
CH_G = 1.0e-5            # body acceleration, lattice units
CH_STEPS = 6000
CH_AVG_STEPS = 200

_C = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
_OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]
_W = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]


def make_shear_state(n: int = N_TUNE, amp: float = 0.01) -> torch.Tensor:
    """Periodic D2Q9 state with a small sinusoidal shear: u_x = amp*sin(2 pi y/n)."""
    y = torch.arange(n, dtype=torch.float64).view(n, 1).expand(n, n)
    ux = amp * torch.sin(2.0 * math.pi * y / n)
    uy = torch.zeros_like(ux)
    rho = torch.ones_like(ux)
    return equilibrium(rho, ux, uy)


# The three base steps are module-level functions on purpose: torch.compile
# guards a compiled graph on the closed-over callable's code object, so three
# closures created by one factory share a code object and make Dynamo
# recompile (and, past the recompile limit, silently fall back to eager -- a
# measurement defect that would compare "compiled" against itself).  One
# function per implementation gives each graph its own guard and one compile.
def _step_bgk(f: torch.Tensor) -> torch.Tensor:
    return stream(collide_bgk(f, TAU))


def _step_bgk_fused(f: torch.Tensor) -> torch.Tensor:
    return stream(collide_bgk_fused(f, TAU))


def _step_bgk_matmul(f: torch.Tensor) -> torch.Tensor:
    return stream(collide_bgk_matmul(f, TAU))


_BASE_STEP = {
    "bgk": _step_bgk,
    "bgk_fused": _step_bgk_fused,
    "bgk_matmul": _step_bgk_matmul,
}


def make_step(impl: str, mode: str | None):
    """Whole LBM step ``f -> f'`` = collide (impl) then periodic stream."""
    return compile_step(_BASE_STEP[impl], mode)


def measure_step(step, f0: torch.Tensor) -> dict:
    """Timing protocol: warm-up + ``REPEATS`` blocks of ``TIMED_STEPS`` steps."""
    f = f0
    for _ in range(WARMUP_STEPS):
        f = step(f)
    per_step = []
    per_step_cpu = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        c0 = time.process_time()
        for _ in range(TIMED_STEPS):
            f = step(f)
        per_step.append((time.perf_counter() - t0) / TIMED_STEPS)
        # process CPU time is the noise-robustness cross-check (amendment 1):
        # it excludes the gaps where the shared host descheduled this process.
        per_step_cpu.append((time.process_time() - c0) / TIMED_STEPS)
    rho = f.sum(dim=0)
    finite = bool(torch.isfinite(f).all())
    return {
        # provenance of the measurement: was this really a compiled wrapper?
        "compiled": hasattr(step, "_torchdynamo_orig_callable"),
        "step_time_s": statistics.median(per_step),
        "step_cpu_s": statistics.median(per_step_cpu),
        "repeats_s": per_step,
        "repeats_cpu_s": per_step_cpu,
        "finite": finite,
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
    }


def tune_grid() -> list[dict]:
    """One tuning sweep = every (impl, compile mode) measured once, in order."""
    rows = []
    f0 = make_shear_state()
    for impl in IMPLS:
        for mode in COMPILE_MODES:
            step = make_step(impl, mode)
            r = measure_step(step, f0)
            rows.append({"collision_impl": impl, "compile_mode": mode, **r})
    return rows


def rank1(rows: list[dict]) -> dict:
    best = min(rows, key=lambda r: r["step_time_s"])
    return {"collision_impl": best["collision_impl"], "compile_mode": best["compile_mode"],
            "step_time_s": best["step_time_s"]}


# --------------------------------------------------------------------------
# Physics anchor: plane Poiseuille (Guo forcing + half-way bounce-back)
# --------------------------------------------------------------------------
def poiseuille_profile(steps: int = CH_STEPS) -> dict:
    """Steady plane-Poiseuille profile from the same D2Q9/BGK lattice.

    Layout: rows ``0`` and ``H+1`` are solid, rows ``1..H`` are fluid, so the
    half-way bounce-back walls sit at ``y = 0.5`` and ``y = H + 0.5``.  Guo
    forcing with acceleration ``g``: ``u = (sum c_i f_i + g/2) / rho`` and the
    source term ``(1 - 1/(2 tau)) w_i [3 (c_i - u) + 9 (c_i.u) c_i] . g``.
    """
    ny, nx = CH_H + 2, CH_NX
    dtype = torch.float64
    c = torch.tensor(_C, dtype=dtype)
    w = torch.tensor(_W, dtype=dtype).view(9, 1, 1)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    ux = torch.zeros(ny, nx, dtype=dtype)
    uy = torch.zeros(ny, nx, dtype=dtype)
    rho = torch.ones(ny, nx, dtype=dtype)
    f = equilibrium(rho, ux, uy)
    # zero the populations living inside the two solid rows
    mask = torch.ones(ny, dtype=dtype)
    mask[0] = 0.0
    mask[-1] = 0.0
    f = f * mask.view(1, ny, 1)
    g = torch.tensor([CH_G, 0.0], dtype=dtype)

    mask2d = mask.view(1, ny, 1)
    tail = []
    for step in range(steps):
        rho = f.sum(0)
        jx = (f * cx).sum(0)
        jy = (f * cy).sum(0)
        # force-shifted velocity (Guo); the solid rows carry rho = 0 so they are
        # guarded with rho_safe and zeroed again after the collision, otherwise a
        # 0/0 there would write NaN into the solid rows and outlive the mask.
        rho_safe = torch.clamp(rho, min=1e-12)
        u = jx / rho_safe + 0.5 * g[0]
        v = jy / rho_safe + 0.5 * g[1]
        feq = equilibrium(rho, u, v)
        cu = cx * u.unsqueeze(0) + cy * v.unsqueeze(0)
        cg = cx * g[0] + cy * g[1]
        # Guo source: 3 (c_i - u) . g + 9 (c_i.u)(c_i.g)
        src = (1.0 - 1.0 / (2.0 * TAU)) * w * (
            3.0 * (cg - (u.unsqueeze(0) * g[0] + v.unsqueeze(0) * g[1]))
            + 9.0 * cu * cg
        )
        f_post = (f - (f - feq) / TAU + src) * mask2d
        # periodic wrap + half-way bounce-back on the two solid rows
        f_new = stream(f_post)
        for k in range(9):
            if _C[k][1] > 0:      # incoming from the lower wall
                f_new[k, 1, :] = f_post[_OPP[k], 1, :]
            elif _C[k][1] < 0:    # incoming from the upper wall
                f_new[k, -2, :] = f_post[_OPP[k], -2, :]
        f = f_new * mask2d
        if step >= steps - CH_AVG_STEPS:
            tail.append((f.sum(0), (f * cx).sum(0)))

    u_sum = torch.zeros(ny, dtype=dtype)
    rho_flux = 0.0
    for rho_t, jx_t in tail:
        u_sum += jx_t.mean(dim=1) / rho_t.mean(dim=1).clamp(min=1e-12)
        rho_flux = float(rho_t[1:CH_H + 1].sum())
    u_mean = (u_sum / len(tail)).numpy()

    nu = (TAU - 0.5) / 3.0
    ys = torch.arange(1, CH_H + 1, dtype=dtype)
    u_analytic = (CH_G / (2.0 * nu)) * (ys - 0.5) * (CH_H + 0.5 - ys)
    u_num = torch.tensor(u_mean[1:CH_H + 1], dtype=dtype)
    u_max_num = float(u_num.max())
    u_max_an = float(u_analytic.max())
    prof_l2 = float(torch.linalg.norm(u_num - u_analytic) / torch.linalg.norm(u_analytic))
    return {
        "H_fluid_rows": CH_H,
        "g": CH_G,
        "nu": nu,
        "ma_max": u_max_num / math.sqrt(1.0 / 3.0),
        "u_max_numeric": u_max_num,
        "u_max_analytic": u_max_an,
        "u_max_rel_err": abs(u_max_num - u_max_an) / u_max_an,
        "profile_l2_rel": prof_l2,
        "profile_numeric": [float(x) for x in u_num],
        "profile_analytic": [float(x) for x in u_analytic],
        "rho_drift_fluid": abs(rho_flux - CH_H * CH_NX) / (CH_H * CH_NX),
    }


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def run_tune(arm: str, launch: int) -> dict:
    if arm == "P":
        pin = ResourcePin(
            label="xval-maxregcount-pinned/tune",
            threads=PIN_THREADS,
            dtype="fp64",
            compile_modes=COMPILE_MODES,
            warmup_steps=WARMUP_STEPS,
            timed_steps=TIMED_STEPS,
            repeats=REPEATS,
            maxnreg=None,  # CPU path: no register file; GPU spelling documented in bench_pinning
        )
        with pinned_resources(pin) as audit:
            assert_pinned(pin)
            rows = tune_grid()
            phys = poiseuille_profile()
            audit.update({"tuned_rows": rows, "physics": phys})
        record = {"arm": "P", "launch": launch, "pinned": True, "audit": audit}
    else:
        threads = DRIFT_SET[launch % len(DRIFT_SET)]
        torch.set_num_threads(threads)
        rows = tune_grid()
        phys = poiseuille_profile()
        record = {
            "arm": "U",
            "launch": launch,
            "pinned": False,
            "inherited_threads": threads,
            "audit": {
                "observed": {"torch_num_threads": torch.get_num_threads(),
                             "torch_version": torch.__version__},
                "drift_set": DRIFT_SET,
            },
            "tuned_rows": rows,
            "physics": phys,
        }
    record["rank1"] = rank1(record["audit"]["tuned_rows"] if arm == "P" else record["tuned_rows"])
    return record


def run_ladder(threads_ladder: list[int], sweeps: int = 2) -> dict:
    """Thread-ladder sweep; every cell is measured once per sweep, interleaved.

    The interleaving matters: a cell is compared with the same cell at another
    thread budget, never with a cell measured minutes later under a different
    host load.  The reported number is the median over the sweeps.
    """
    f0 = make_shear_state()
    cells: dict[tuple[int, str, str | None], list[dict]] = {}
    steps = {(impl, mode): make_step(impl, mode) for impl in IMPLS for mode in COMPILE_MODES}
    for _ in range(sweeps):
        for threads in threads_ladder:
            torch.set_num_threads(threads)
            for impl in IMPLS:
                for mode in COMPILE_MODES:
                    r = measure_step(steps[(impl, mode)], f0)
                    cells.setdefault((threads, impl, mode), []).append(r)
    out = []
    for (threads, impl, mode), rs in cells.items():
        out.append({
            "threads": threads,
            "collision_impl": impl,
            "compile_mode": mode,
            "step_time_s": statistics.median([r["step_time_s"] for r in rs]),
            "step_cpu_s": statistics.median([r["step_cpu_s"] for r in rs]),
            "sweeps": [{"step_time_s": r["step_time_s"], "step_cpu_s": r["step_cpu_s"],
                        "finite": r["finite"]} for r in rs],
        })
    return {"mode": "ladder", "threads_ladder": threads_ladder, "sweeps": sweeps, "rows": out}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("tune", "ladder", "physics"), default="tune")
    ap.add_argument("--arm", choices=("U", "P"), default="P")
    ap.add_argument("--launch", type=int, default=0)
    ap.add_argument("--threads-ladder", default="1,8,16,64,128")
    ap.add_argument("--sweeps", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    # defensive: a per-configuration compile must never hit the default
    # recompile limit and silently degrade to eager (see _BASE_STEP comment)
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.cache_size_limit = 64
        _dynamo.config.recompile_limit = 64
    except Exception:  # pragma: no cover - torch version without these knobs
        pass
    if args.mode == "tune":
        record = run_tune(args.arm, args.launch)
    elif args.mode == "ladder":
        record = run_ladder([int(t) for t in args.threads_ladder.split(",")], sweeps=args.sweeps)
    else:
        torch.set_num_threads(PIN_THREADS)
        record = {"mode": "physics", "physics": poiseuille_profile()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ("arm", "launch", "rank1") if k in record}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())