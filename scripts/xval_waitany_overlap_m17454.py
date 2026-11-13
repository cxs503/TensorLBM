#!/usr/bin/env python3
"""XVAL-LBM-ASYNC-COPY-WAITANY-OVERLAP experiment driver.

Replicates the TCLB halo-exchange schedule (``CROSS_MPI_WAITANY`` on/off) on the
TensorLBM x-slab D3Q19 transport and quantifies it against (a) the shipped
wait-all step and (b) the analytic overlap bound.

Arms
  blocking : shipped TensorLBM order (collide all -> post -> waitall -> stream all)
  waitall  : border collide/pack -> post all -> interior under the exchange -> waitall
  waitany  : same prefix, one buffer consumed per MPI_Waitany return

Outputs
  artifacts/xval_waitany_overlap_m17454.json
  figures/xval_waitany_fig1_stall_vs_latency.png
  figures/xval_waitany_fig2_hidden_window.png
  figures/xval_waitany_fig3_waitany_law.png
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.comm_overlap.virtual_case import run_virtual_case  # noqa: E402

ARMS = ("blocking", "waitall", "waitany")
COMPUTE = {"collide": 2.0, "stream": 2.0, "pack": 4.0}
NZ, NY = 2, 3
ITEMSIZE = 8


def collide_stub(f: torch.Tensor) -> torch.Tensor:
    """Pointwise affine map used as a stand-in collision operator (test fixture).

    Not a physics kernel: the study is about the communication schedule, and a
    pointwise map keeps collision splittable exactly as a cell-local LBM kernel
    is, so border/interior splitting is legitimate rather than assumed.
    """
    return f * 1.5 - 0.25


def field(nx: int) -> torch.Tensor:
    torch.manual_seed(20261113)
    return torch.randn(19, NZ, NY, nx, dtype=torch.float64)


def cuts_for(nx: int, world: int) -> list[int]:
    base, rem = divmod(nx, world)
    out = [0]
    for rank in range(world):
        out.append(out[-1] + base + (1 if rank < rem else 0))
    return out


def model(latency_us: float, bandwidth: float, jitter: dict[str, float] | None = None) -> dict:
    return {
        "latency_us": latency_us,
        "bandwidth_bytes_per_us": bandwidth,
        "compute_us": dict(COMPUTE),
        "jitter_us": dict(jitter or {}),
    }


def analytic_bound(latency_us: float, bandwidth: float, cuts: list[int]) -> dict:
    """Exposed stall predicted for the pipelined arms (per-rank, then averaged)."""
    payload = 19 * NZ * NY * ITEMSIZE
    t_comm = latency_us + payload / bandwidth
    n = len(cuts) - 1
    t_int = [(cuts[r + 1] - cuts[r] - 2) * COMPUTE["stream"] for r in range(n)]
    stalls = [max(0.0, t_comm - ti) for ti in t_int]
    blocked = [t_comm] * n
    return {
        "t_comm_us": t_comm,
        "pipelined_stall_us": sum(stalls) / n,
        "blocked_stall_us": sum(blocked) / n,
        "t_interior_us": sum(t_int) / n,
    }


def rel_err(got: float, ref: float) -> float:
    if ref == 0:
        return 0.0 if got == 0 else float("inf")
    return (got - ref) / ref


def main() -> int:
    results: dict = {"fixture": "pointwise affine collision (schedule study, not physics)"}
    nx, world = 16, 3
    cuts = cuts_for(nx, world)
    base_field = field(nx)

    # ---- A. correctness: decomposition invariance, every arm -----------------
    correctness = []
    for np_ in (2, 3, 4):
        for arm in ARMS:
            r = run_virtual_case(base_field, mode=arm, world_size=np_, n_steps=4,
                                 model=model(5.0, 32.0), collide_fn=collide_stub)
            correctness.append({
                "world_size": np_, "arm": arm, "mismatch": r["mismatch"],
                "max_abs_error": r["max_abs_error"], "cuts": r["cuts"],
            })
    results["correctness"] = correctness

    # ---- B. latency sweep -----------------------------------------------------
    sweep = []
    for lat in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
        m = model(lat, 32.0)
        bound = analytic_bound(lat, 32.0, cuts)
        row = {"latency_us": lat, "t_comm_us": bound["t_comm_us"],
               "ref_pipelined_stall_us": bound["pipelined_stall_us"],
               "ref_blocked_stall_us": bound["blocked_stall_us"]}
        for arm in ARMS:
            r = run_virtual_case(base_field, mode=arm, world_size=world, n_steps=3,
                                 model=m, collide_fn=collide_stub)
            row[f"{arm}_stall_us"] = r["stall_us"]
            row[f"{arm}_stall_fraction"] = r["stall_fraction"]
            row[f"{arm}_step_us"] = r["step_us"]
            row[f"{arm}_mismatch"] = r["mismatch"]
        row["pipelined_rel_err_vs_bound"] = rel_err(row["waitall_stall_us"],
                                                    bound["pipelined_stall_us"])
        row["blocking_rel_err_vs_bound"] = rel_err(row["blocking_stall_us"],
                                                   bound["blocked_stall_us"])
        row["speedup_pipelined_over_blocked"] = row["blocking_step_us"] / row["waitany_step_us"]
        sweep.append(row)
    results["latency_sweep"] = sweep

    # ---- C. arrival-jitter sweep (the CROSS_MPI_WAITANY regime) --------------
    jitter_rows = []
    consume = COMPUTE["stream"]  # one border column per consumed buffer
    for jit in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        m = model(5.0, 32.0, {"left": 0.0, "right": jit})
        rows = {arm: run_virtual_case(field(12), mode=arm, world_size=3, n_steps=3,
                                      model=m, collide_fn=collide_stub) for arm in ARMS}
        benefit = rows["waitall"]["stall_us"] - rows["waitany"]["stall_us"]
        ref_benefit = min(jit, consume)
        jitter_rows.append({
            "jitter_us": jit, "consume_us": consume,
            "waitany_benefit_us": benefit, "ref_benefit_us": ref_benefit,
            "rel_err": rel_err(benefit, ref_benefit),
            "waitany_mismatch": rows["waitany"]["mismatch"],
        })
    results["jitter_sweep"] = jitter_rows

    # ---- D. grid / compute-to-comm ratio sweep -------------------------------
    grid_rows = []
    for nx_ in (12, 16, 24, 32, 48, 64):
        cuts_ = cuts_for(nx_, world)
        bound = analytic_bound(5.0, 32.0, cuts_)
        m = model(5.0, 32.0)
        f = field(nx_)
        rows = {arm: run_virtual_case(f, mode=arm, world_size=world, n_steps=3,
                                      model=m, collide_fn=collide_stub) for arm in ARMS}
        grid_rows.append({
            "nx_local_mean": nx_ / world,
            "t_comm_us": bound["t_comm_us"],
            "t_interior_us": bound["t_interior_us"],
            "ratio_comm_over_interior": bound["t_comm_us"] / max(bound["t_interior_us"], 1e-9),
            "blocking_stall_us": rows["blocking"]["stall_us"],
            "pipelined_stall_us": rows["waitany"]["stall_us"],
            "ref_pipelined_stall_us": bound["pipelined_stall_us"],
            "rel_err_vs_bound": rel_err(rows["waitany"]["stall_us"], bound["pipelined_stall_us"]),
            "blocking_stall_fraction": rows["blocking"]["stall_fraction"],
            "pipelined_stall_fraction": rows["waitany"]["stall_fraction"],
            "mismatch": rows["waitany"]["mismatch"],
        })
    results["grid_sweep"] = grid_rows

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "xval_waitany_overlap_m17454.json").write_text(json.dumps(results, indent=2))

    # ---- figures -------------------------------------------------------------
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    lat = [row["latency_us"] for row in sweep]
    for arm, style in zip(ARMS, ("o-", "s--", "^-")):
        ax.plot(lat, [100 * row[f"{arm}_stall_fraction"] for row in sweep], style, label=arm)
    ax.plot(lat, [100 * row["ref_pipelined_stall_us"] / row["blocking_step_us"] for row in sweep],
            "k:", label="analytic bound (pipelined)")
    ax.set_xscale("log")
    ax.set_xlabel("one-way MPI latency [us]")
    ax.set_ylabel("exposed communication stall [% of step]")
    ax.set_title("XVAL wait-any overlap: stall vs latency (virtual clock, 3 ranks)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "xval_waitany_fig1_stall_vs_latency.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ratio = [row["ratio_comm_over_interior"] for row in grid_rows]
    ax.plot(ratio, [row["blocking_stall_fraction"] * 100 for row in grid_rows], "o-",
            label="blocking (shipped)")
    ax.plot(ratio, [row["pipelined_stall_fraction"] * 100 for row in grid_rows], "s-",
            label="waitany (TCLB schedule)")
    ax.plot(ratio, [row["ref_pipelined_stall_us"] / (row["ref_pipelined_stall_us"] + row["t_interior_us"]) * 100
                    for row in grid_rows], "k:", label="max(0, T_comm-T_int)/step bound")
    ax.set_xscale("log")
    ax.set_xlabel("T_comm / T_interior")
    ax.set_ylabel("exposed stall [% of step]")
    ax.set_title("Hidden comm window vs comm/compute ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "xval_waitany_fig2_hidden_window.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([row["jitter_us"] for row in jitter_rows],
            [row["waitany_benefit_us"] for row in jitter_rows], "o-", label="measured")
    art = [min(row["jitter_us"], consume) for row in jitter_rows]
    ax.plot([row["jitter_us"] for row in jitter_rows], art, "k--",
            label=f"min(arrival spread, per-buffer consume={consume:.0f} us)")
    ax.set_xlabel("arrival spread between the two directions [us]")
    ax.set_ylabel("waitany gain over waitall [us/step]")
    ax.set_title("MPI_Waitany gain law (CROSS_MPI_WAITANY switch)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "xval_waitany_fig3_waitany_law.png", dpi=150)
    plt.close(fig)

    # ---- console summary -----------------------------------------------------
    worst_correct = max(row["max_abs_error"] for row in correctness)
    print(f"correctness: max_abs_error over all arms/world sizes = {worst_correct}")
    for row in sweep:
        print(f"lat={row['latency_us']:5.1f}us  blocked={row['blocking_stall_us']:7.1f}  "
              f"waitany={row['waitany_stall_us']:7.1f}  ref={row['ref_pipelined_stall_us']:7.1f}  "
              f"err={100*row['pipelined_rel_err_vs_bound']:+.2f}%  "
              f"speedup={row['speedup_pipelined_over_blocked']:.3f}")
    for row in jitter_rows:
        print(f"jitter={row['jitter_us']:5.1f}  benefit={row['waitany_benefit_us']:5.2f}  "
              f"ref={row['ref_benefit_us']:5.2f}")
    for row in grid_rows:
        print(f"nx_local={row['nx_local_mean']:5.1f}  ratio={row['ratio_comm_over_interior']:6.2f}  "
              f"blocked%={100*row['blocking_stall_fraction']:6.1f}  "
              f"waitany%={100*row['pipelined_stall_fraction']:6.1f}  "
              f"err={100*row['rel_err_vs_bound']:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())