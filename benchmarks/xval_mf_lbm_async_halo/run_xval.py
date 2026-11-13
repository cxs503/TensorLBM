"""XVAL round 1: TensorLBM vs lanl/MF-LBM ``openacc-async-halo-overlap``.

Anchor ``XVAL-LBM-OPENACC-ASYNC-HALO-OVERLAP``.  Reference: ``lanl/MF-LBM`` @
``11be324`` (``multiphase_3D/0.src/Main_multiphase.F90:354-458``,
``Mpi.F90:5-346``, ``Module.F90:159-164``) -- read from source, cached copy at
``/tmp/mflbm2``; the practice is registered as ``oss_solutions`` id 200.

Arms
----
``blocking``    halo-before-compute (the schedule the exchange latency is fully
                exposed; TensorLBM's shipped ``D3Q19OverlapTransport`` baseline)
``cpu_order``   MF-LBM's non-OpenACC branch: ``mpi_send_req`` then interior kernel
``gpu_order``   MF-LBM's ``#ifdef _openacc`` branch: interior kernel then ``mpi_send_req``

Predeclared criteria (fixed before the run)
-------------------------------------------
C1 equivalence: max |f_mode - f_monolithic| <= 1e-12 absolute on O(1) populations.
C2 physics: quasi-2D Taylor-Green decay amplitude vs the analytic ``exp(-2 nu k^2 t)``
   must be within 15 % (owner bar); <3 % is the ecosystem stretch target.
C3 overlap: at modelled latency >= 100 us the overlapped arms must hide >= 50 % of
   the exchange time that ``blocking`` exposes (``hidden_fraction``), i.e. a
   >= 1.3x step-time reduction vs ``blocking`` on the same cost model.
Stopping rule: fixed grids/latencies, no tuning after the numbers are in; every
number printed here is written to ``results.json`` and re-plotted from it.

Run::

    PYTHONPATH=src python benchmarks/xval_mf_lbm_async_halo/run_xval.py --out artifacts/xval_mf_lbm_async_halo
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorlbm.comm_overlap.virtual_case import monolithic_step
from tensorlbm.mf_lbm_async_halo import (
    MF_LBM_MODES,
    bgk_collide,
    collision_counters,
    run_mf_virtual_case,
)

# ----------------------------------------------------------------- case definition
NX, NY, NZ = 48, 48, 4          # quasi-2D Taylor-Green in a periodic 3-D box
U0 = 0.04                       # peak velocity  -> Ma = U0/cs = 0.069
NU = 0.02                       # kinematic viscosity
TAU = 0.5 + 3.0 * NU            # BGK relaxation time
KX = 2.0 * math.pi / NX
KY = 2.0 * math.pi / NY
RE = U0 / (NU * KX)             # Reynolds number on the TG length scale 1/k
N_STEPS = 300

# cost model shared by every timing arm (microseconds)
MODEL = {
    "latency_us": 100.0,
    "bandwidth_bytes_per_us": 8000.0,          # 8 GB/s effective host<->device link
    "compute_us": {"collide": 1.5e-3, "stream": 1.5e-3, "pack": 1.0e-4},  # us per cell / per element
    "jitter_us": {},
}


def tgv_field(nx: int = NX, ny: int = NY, nz: int = NZ, u0: float = U0) -> torch.Tensor:
    """Equilibrium field of ``u = u0 (sin kx x cos ky y, -cos kx x sin ky y, 0)``."""
    x = torch.arange(nx, dtype=torch.float64) * KX
    y = torch.arange(ny, dtype=torch.float64) * KY
    z = torch.arange(nz, dtype=torch.float64)
    X, Y, _Z = torch.meshgrid(x, y, z, indexing="ij")
    ux = u0 * torch.sin(X) * torch.cos(Y)
    uy = -u0 * torch.cos(X) * torch.sin(Y)
    uz = torch.zeros_like(ux)
    u = torch.stack([ux, uy, uz], dim=0)                                   # (3,nx,ny,nz)
    u = u.permute(0, 3, 2, 1).contiguous()                                 # (3,nz,ny,nx)
    feq = torch.zeros((19, nz, ny, nx), dtype=torch.float64)
    from tensorlbm.d3q19 import C, W

    cs2 = 1.0 / 3.0
    for q in range(19):
        cu = (C[q, 0] * u[0] + C[q, 1] * u[1] + C[q, 2] * u[2]).double()
        usq = (u * u).sum(dim=0)
        feq[q] = W[q] * (1.0 + cu / cs2 + cu * cu / (2 * cs2 * cs2) - usq / (2 * cs2))
    return feq


def analytic_amplitude(t: float) -> float:
    """``u_peak(t)/u0`` of the 2-D Taylor-Green vortex (exact NS solution).

    Vorticity ``omega = 2 U0 k sin(kx) sin(ky) f(t)`` obeys ``df/dt = -nu (kx^2+ky^2) f``
    for ``kx = ky = k``, i.e. ``f = exp(-2 nu k^2 t)`` -- the decay rate is
    ``nu (kx^2 + ky^2)``, *not* twice that.
    """
    return math.exp(-NU * (KX * KX + KY * KY) * t)


# ------------------------------------------------------------------------ experiments
def equivalence_arm() -> list[dict]:
    """C1: every schedule/mesh split must reproduce the monolithic field."""
    full = tgv_field()
    rows: list[dict] = []
    ref = full.clone()
    for _ in range(20):
        ref = monolithic_step(ref, lambda b: bgk_collide(b, TAU))
    for world in (2, 4):
        for mode in MF_LBM_MODES:
            for strip_w in (1, 2):
                for lanes in ("split", "shared"):
                    res = run_mf_virtual_case(
                        full, mode=mode, world_size=world, n_steps=20,
                        collide_fn=lambda b: bgk_collide(b, TAU),
                        strip_width=strip_w, strip_lanes=lanes,
                    )
                    got = res["final"]
                    rows.append({
                        "world_size": world,
                        "mode": mode,
                        "strip_width": strip_w,
                        "strip_lanes": lanes,
                        "max_abs_error": res["max_abs_error"],
                        "sum_abs_error": float((torch.cat(got, dim=-1) - ref).abs().sum()),
                        "mismatched_values": res["mismatch"],
                        "pass_1e-12": bool(res["max_abs_error"] <= 1e-12),
                    })
    return rows


def physics_arm() -> dict:
    """C2: quasi-2-D Taylor-Green decay vs the analytic solution (serial + 4-slab)."""
    full = tgv_field()
    collide = lambda b: bgk_collide(b, TAU)  # noqa: E731

    serial = full.clone()
    serial_curve: dict[int, float] = {}
    t0 = time.perf_counter()
    watch = list(range(25, N_STEPS + 1, 25))
    for step in range(1, N_STEPS + 1):
        serial = monolithic_step(serial, collide)
        if step in watch:
            _, u_serial = collision_counters(serial)
            serial_curve[step] = float(u_serial.norm(dim=0).max())
    serial_wall = time.perf_counter() - t0

    t1 = time.perf_counter()
    split = run_mf_virtual_case(
        full, mode="gpu_order", world_size=4, n_steps=N_STEPS,
        collide_fn=collide, strip_width=1, strip_lanes="split", record_steps=watch,
    )
    split_wall = time.perf_counter() - t1

    curve: list[dict] = []
    for row in split["curve"]:
        step = row["step"]
        exact = U0 * analytic_amplitude(float(step))
        curve.append({
            "step": step,
            "u_peak_exact": exact,
            "u_peak_serial": serial_curve[step],
            "u_peak_split": row["peak_overlapped"],
            "err_serial_pct": 100.0 * abs(serial_curve[step] - exact) / exact,
            "err_split_pct": 100.0 * abs(row["peak_overlapped"] - exact) / exact,
            "serial_vs_split_abs": row["max_abs_error"],
        })
    last = curve[-1]
    return {
        "case": {
            "grid": [NX, NY, NZ], "world_size": 4, "steps": N_STEPS,
            "U0": U0, "nu": NU, "tau": TAU, "Re": RE,
            "Ma": U0 / math.sqrt(1.0 / 3.0),
            "k": [KX, KY], "schedule": "gpu_order, strip_width=1, split lanes",
            "slab_cuts": split["cuts"],
        },
        "curve": curve,
        "final_err_serial_pct": last["err_serial_pct"],
        "final_err_split_pct": last["err_split_pct"],
        "pass_15pct": bool(max(last["err_serial_pct"], last["err_split_pct"]) < 15.0),
        "pass_3pct": bool(max(last["err_serial_pct"], last["err_split_pct"]) < 3.0),
        "split_vs_serial_max_abs": max(r["serial_vs_split_abs"] for r in curve),
        "split_step_max_abs_error": split["max_abs_error"],
        "wall_seconds": serial_wall + split_wall,
    }


GRIDS = {
    "light": (32, 32, 8),     # per-rank compute (32 us) < latency (100 us)
    "heavy": (64, 64, 32),    # per-rank compute (110 us) > latency (100 us): the MF-LBM regime
}


def timing_arm(latencies=(5.0, 25.0, 100.0, 400.0), steps: int = 20) -> list[dict]:
    """C3: makespan vs modelled latency, and how much of the exchange is hidden.

    ``hidden_fraction`` is measured against the *same grid / strip / lane*
    configuration's ``blocking`` arm, so it isolates what the schedule buys.
    """
    collide = lambda b: bgk_collide(b, TAU)  # noqa: E731
    rows: list[dict] = []
    for label, (nx, ny, nz) in GRIDS.items():
        full = tgv_field(nx=nx, ny=ny, nz=nz)
        for lat in latencies:
            model = dict(MODEL)
            model["latency_us"] = float(lat)
            for mode in MF_LBM_MODES:
                for strip_w in (1, 2):
                    for lanes in ("split", "shared"):
                        res = run_mf_virtual_case(
                            full, mode=mode, world_size=4, n_steps=steps, model=model,
                            collide_fn=collide, strip_width=strip_w, strip_lanes=lanes,
                        )
                        rows.append({
                            "grid": label, "shape": [nx, ny, nz],
                            "latency_us": float(lat), "mode": mode, "strip_width": strip_w,
                            "strip_lanes": lanes,
                            "step_us": res["clock_us"],
                            "compute_us": res["compute_us"],
                            "max_abs_error": res["max_abs_error"],
                        })
    base = {}
    for row in rows:
        if row["mode"] == "blocking":
            base[(row["grid"], row["latency_us"], row["strip_width"], row["strip_lanes"])] = row["step_us"]
    for row in rows:
        b = base[(row["grid"], row["latency_us"], row["strip_width"], row["strip_lanes"])]
        exposed = b - row["compute_us"]
        row["blocking_step_us"] = b
        row["exposed_comm_us"] = exposed
        row["hidden_fraction"] = (
            0.0 if row["mode"] == "blocking" or exposed <= 1e-12
            else max(0.0, min(1.0, (b - row["step_us"]) / exposed))
        )
        row["speedup_vs_blocking"] = b / row["step_us"]
    return rows


# ---------------------------------------------------------------------------- figures
def make_figures(out: Path, physics: dict, timing: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names: list[str] = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    steps = [r["step"] for r in physics["curve"]]
    ax[0].plot(steps, [r["u_peak_exact"] for r in physics["curve"]], "k-", label="analytic $u_0e^{-2\\nu k^2t}$")
    ax[0].plot(steps, [r["u_peak_serial"] for r in physics["curve"]], "o", ms=4, label="TensorLBM monolithic")
    ax[0].plot(steps, [r["u_peak_split"] for r in physics["curve"]], "x", ms=5, label="TensorLBM 4-slab gpu_order")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("peak |u|"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"TGV decay, Re={RE:.0f}, Ma={physics['case']['Ma']:.3f}")
    ax[1].semilogy(steps, [max(r["err_serial_pct"], 1e-12) for r in physics["curve"]], "o-", label="monolithic")
    ax[1].semilogy(steps, [max(r["err_split_pct"], 1e-12) for r in physics["curve"]], "x-", label="4-slab overlapped")
    ax[1].axhline(15.0, color="r", ls="--", label="15 % owner bar")
    ax[1].axhline(3.0, color="orange", ls=":", label="3 % stretch")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("amplitude error [%]"); ax[1].legend(fontsize=8)
    ax[1].set_title("decay-amplitude error vs analytic")
    fig.tight_layout(); path = out / "tgv_decay.png"; fig.savefig(path, dpi=130); plt.close(fig)
    names.append(path.name)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for mode, style in (("blocking", "s-"), ("cpu_order", "o-"), ("gpu_order", "^-")):
        lane = [r for r in timing if r["mode"] == mode and r["grid"] == "heavy"
                and r["strip_width"] == 1 and r["strip_lanes"] == "split"]
        lane.sort(key=lambda r: r["latency_us"])
        ax[0].plot([r["latency_us"] for r in lane], [r["step_us"] for r in lane], style, label=mode)
    ax[0].set_xscale("log"); ax[0].set_xlabel("modelled exchange latency [us]")
    ax[0].set_ylabel("step makespan [us]"); ax[0].legend(fontsize=8)
    ax[0].set_title("heavy grid, strip_width=1, one queue per face")
    for lanes, style in (("split", "o-"), ("shared", "s--")):
        lane = [r for r in timing if r["mode"] == "gpu_order" and r["grid"] == "heavy"
                and r["strip_width"] == 1 and r["strip_lanes"] == lanes]
        lane.sort(key=lambda r: r["latency_us"])
        ax[1].plot([r["latency_us"] for r in lane], [100 * r["hidden_fraction"] for r in lane], style, label=f"{lanes} lanes")
    ax[1].axhline(50.0, color="r", ls="--", label="C3 predeclared 50 %")
    ax[1].set_xscale("log"); ax[1].set_xlabel("modelled exchange latency [us]")
    ax[1].set_ylabel("hidden fraction of exposed comm [%]"); ax[1].legend(fontsize=8)
    ax[1].set_title("gpu_order overlap efficiency")
    fig.tight_layout(); path = out / "overlap.png"; fig.savefig(path, dpi=130); plt.close(fig)
    names.append(path.name)
    return names


# -------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/xval_mf_lbm_async_halo")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    equiv = equivalence_arm()
    physics = physics_arm() if not args.quick else {"case": {}, "curve": [], "final_err_serial_pct": None,
                                                    "final_err_split_pct": None, "pass_15pct": None,
                                                    "pass_3pct": None, "split_vs_serial_max_abs": None,
                                                    "wall_seconds": 0.0}
    timing = timing_arm(steps=10)
    figures = make_figures(out, physics, timing) if physics["curve"] else []

    crit = {
        "C1_equivalence_max_abs_error": max(r["max_abs_error"] for r in equiv),
        "C1_pass": bool(all(r["pass_1e-12"] for r in equiv)),
        "C2_err_serial_pct": physics["final_err_serial_pct"],
        "C2_err_split_pct": physics["final_err_split_pct"],
        "C2_pass_15pct": physics["pass_15pct"],
        "C2_pass_3pct": physics["pass_3pct"],
        "C3_hidden_fraction_at_100us_gpu_order_heavy": next(
            (r["hidden_fraction"] for r in timing
             if r["mode"] == "gpu_order" and r["grid"] == "heavy" and r["latency_us"] == 100.0
             and r["strip_width"] == 1 and r["strip_lanes"] == "split"), None),
        "C3_speedup_at_100us_gpu_order_heavy": next(
            (r["speedup_vs_blocking"] for r in timing
             if r["mode"] == "gpu_order" and r["grid"] == "heavy" and r["latency_us"] == 100.0
             and r["strip_width"] == 1 and r["strip_lanes"] == "split"), None),
        "C3_hidden_fraction_at_100us_gpu_order_light": next(
            (r["hidden_fraction"] for r in timing
             if r["mode"] == "gpu_order" and r["grid"] == "light" and r["latency_us"] == 100.0
             and r["strip_width"] == 1 and r["strip_lanes"] == "split"), None),
    }
    crit["C3_pass"] = bool(
        crit["C3_hidden_fraction_at_100us_gpu_order_heavy"] is not None
        and crit["C3_hidden_fraction_at_100us_gpu_order_heavy"] >= 0.5
    )

    payload = {
        "anchor": "XVAL-LBM-OPENACC-ASYNC-HALO-OVERLAP",
        "oss_reference": {
            "repo": "lanl/MF-LBM", "commit": "11be324",
            "oss_solutions_id": 200,
            "files": [
                "multiphase_3D/0.src/Main_multiphase.F90:354-458",
                "multiphase_3D/0.src/Mpi.F90:5-346",
                "multiphase_3D/0.src/Module.F90:159-164",
                "multiphase_3D/0.src/Module.F90:263",
            ],
        },
        "arm": {"repo": "TensorLBM", "module": "src/tensorlbm/mf_lbm_async_halo.py"},
        "model": MODEL,
        "criteria": crit,
        "equivalence": equiv,
        "physics": {k: v for k, v in physics.items() if k != "final"},
        "timing": timing,
        "figures": figures,
        "wall_seconds": time.perf_counter() - t_start,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(crit, indent=2))
    print(f"artifacts -> {out}  ({payload['wall_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())