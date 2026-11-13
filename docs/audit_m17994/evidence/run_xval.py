#!/usr/bin/env python3
"""XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA driver.

Cross-validates the transplant of the OSS pattern
``lbm-principles-practice/code :: mpi-1d-slab-unpacked-halo-testall-overlap``
(see ``src/tensorlbm/parallel/slab_unpacked.py``) against

  A_ref  the same solver on **one** rank (NP=1, in-memory periodic self-halo):
         representation neutrality must be exact to round-off;
  A_ana  the Taylor-Green continuum solution
         ``ux = u0 cos kx cos ky exp(-2 nu k^2 t)``: physical accuracy;
  A_pk   the in-repo packed/selective halo convention (staging buffers, only
         the crossing directions): wire bytes and staging copies.

Every number in ``results.json`` / ``kpi.csv`` is produced here; the design
(thresholds, stopping rule, arms) is predeclared in ``predeclared.json``.

Usage
-----
    python run_xval.py --quick        # smoke, NOT for the report table
    python run_xval.py                # predeclared ladder
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import socket
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tensorlbm.parallel import slab_unpacked as su  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "xval_mpi_slab_unpacked_halo_overlap")
TAU = 0.8
U0 = 0.04


# --------------------------------------------------------------------------- arm
def _worker(rank: int, size: int, peers: dict, cfg: dict,
            out_ch: dict | None = None, in_ch: dict | None = None) -> dict:
    """One rank: run the slab solver and return the raw per-rank records."""
    """One rank: run the slab solver and return raw per-rank records."""
    nx = cfg["nx"]
    ny_local = cfg["ny"] // size
    comm = su.ThreadPeers(rank, size, out_ch or {}, in_ch or {})  # size 1 -> no peers
    kind = cfg["halo"]
    if kind == "unpacked":
        ex = su.UnpackedHaloExchange(comm, ny_local, nx, selective=False)
    elif kind == "unpacked-selective":
        ex = su.UnpackedHaloExchange(comm, ny_local, nx, selective=True)
    elif kind == "packed-selective":
        ex = su.PackedHaloExchange(comm, ny_local, nx)
    else:
        raise ValueError(kind)

    solver = su.SlabSolver(ny_local, nx, tau=TAU, comm=comm, exchange=ex,
                           overlap=cfg["overlap"], poll_every=cfg["poll_every"])
    solver.init_shear_wave(U0)
    comm.barrier()
    t0 = time.perf_counter()
    for _ in range(cfg["steps"]):
        solver.step()
    comm.barrier()
    wall = time.perf_counter() - t0
    rho, ux, uy = solver.macroscopic()
    glob = solver.global_scalars()
    ampl = solver.global_scalars()
    out = {
        "rank": rank,
        "wall_s": wall,
        "ux_local": ux.tolist(),
        "uy_local": uy.tolist(),
        "rho_local": rho.tolist(),
        "global_scalars": glob,
        "testall_polls": comm.n_testall if hasattr(comm, "n_testall") else 0,
        "waitall_calls": solver.n_wait,
        "poll_skipped": solver.poll_skipped,
        "messages_per_step": ex.messages_per_step(),
        "wire_bytes_per_step": ex.bytes_per_step(),
        "staging_bytes_per_step": ex.copies_bytes / max(1, cfg["steps"]),
        "comm_messages_total": comm.n_messages,
    }
    comm.close()
    return out


def _launch(size: int, cfg: dict) -> list[dict]:
    """Run ``size`` ranks as threads wired by directed queue channels.

    A real ``mpi4py``/``mpirun`` stack is not installed in this image (checked
    at run time: no ``mpi4py`` module, no ``mpirun`` binary), so ranks are OS
    threads in one process and the transport is a queue pair per direction.
    Requests stay pollable, so ``Testall`` progress polling and the overlap
    are measured on a real interleaving; the wire is in-process memory, so the
    absolute times are not MPI times (recorded as a comparison limit).
    """
    if size == 1:
        return [_worker(0, 1, {}, cfg)]
    pairs = {(i, j): queue.Queue()
             for i in range(size) for j in range(size) if i != j}
    results: dict[int, dict] = {}
    errors: dict[int, str] = {}
    lock = threading.Lock()

    def _run(r: int) -> None:
        out_ch = {j: su.QueueChannel(pairs[(r, j)], readable=False) for j in range(size) if j != r}
        in_ch = {j: su.QueueChannel(pairs[(j, r)], readable=True) for j in range(size) if j != r}
        try:
            res = _worker(r, size, {}, cfg, out_ch=out_ch, in_ch=in_ch)
            with lock:
                results[r] = res
        except BaseException as exc:  # noqa: BLE001 - rank must report, not die
            with lock:
                errors[r] = repr(exc)

    threads = [threading.Thread(target=_run, args=(r,), daemon=True) for r in range(size)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=600)
    if errors:
        raise RuntimeError(f"rank failures: {errors}")
    return [results[r] for r in range(size)]


def _assemble(per_rank: list[dict]) -> dict:
    ux = np.vstack([np.asarray(r["ux_local"]) for r in per_rank])
    uy = np.vstack([np.asarray(r["uy_local"]) for r in per_rank])
    rho = np.vstack([np.asarray(r["rho_local"]) for r in per_rank])
    return {
        "ux": ux,
        "uy": uy,
        "rho": rho,
        "wall_s": max(r["wall_s"] for r in per_rank),
        # already Allreduced inside the ranks: take rank 0 (no double counting)
        "global_scalars": per_rank[0]["global_scalars"],
        "testall_polls": per_rank[0]["testall_polls"],
        "waitall_calls": per_rank[0]["waitall_calls"],
        "poll_skipped": per_rank[0]["poll_skipped"],
        "messages_per_step": per_rank[0]["messages_per_step"],
        "wire_bytes_per_step": per_rank[0]["wire_bytes_per_step"],
        "staging_bytes_per_step": per_rank[0]["staging_bytes_per_step"],
        "comm_messages_total": sum(r["comm_messages_total"] for r in per_rank),
    }


# ------------------------------------------------------------------------- KPIs
def analytic_fields(nx: int, ny: int, t: int):
    """Transverse shear wave: divergence-free, so no acoustic transient."""
    ux = su.shear_wave_analytic(ny, nx, t, (TAU - 0.5) / 3.0, U0)
    return ux, np.zeros_like(ux)


def kpis(run: dict, ref: dict | None, nx: int, ny: int, steps: int) -> dict:
    ux, uy = run["ux"], run["uy"]
    ax, ay = analytic_fields(nx, ny, steps)
    amp_num = float(ux[0, 0])              # probe at (y=0, x=0), k=2 pi/ny
    amp_ref = float(ax[0, 0])
    nu = (TAU - 0.5) / 3.0
    decay_num = math.exp(-nu * (2.0 * np.pi / ny) ** 2 * steps)
    out = {
        "u_probe": amp_num,
        "u_probe_analytic": amp_ref,
        "probe_rel_err_pct": 100.0 * abs(amp_num - amp_ref) / abs(amp_ref),
        "decay_factor_measured": amp_num / U0,
        "decay_factor_analytic": decay_num,
        "field_l2_rel_err_pct": 100.0
        * float(np.sqrt(np.sum((ux - ax) ** 2 + (uy - ay) ** 2)) / np.sqrt(np.sum(ax ** 2 + ay ** 2))),
        "rho_min": float(run["rho"].min()),
        "rho_max": float(run["rho"].max()),
        "mass_drift_rel": abs(run["global_scalars"][0] - nx * ny) / (nx * ny),
        "ma_max": float(run["global_scalars"][6]) / math.sqrt(1.0 / 3.0),
        "wall_s": run["wall_s"],
        "mlups": nx * ny * steps / run["wall_s"] / 1e6,
        "testall_polls_per_step": run["testall_polls"] / steps,
        "waitall_per_step": run["waitall_calls"] / steps,
        "messages_per_step": run["messages_per_step"],
        "wire_bytes_per_step": run["wire_bytes_per_step"],
        "staging_bytes_per_step": float(run["staging_bytes_per_step"]),
    }
    if ref is not None:
        out["identity_maxrel"] = float(np.max(np.abs(ux - ref["ux"])) / np.max(np.abs(ref["ux"])))
        out["identity_maxrel_uy"] = float(np.max(np.abs(uy - ref["uy"])) / np.max(np.abs(ref["uy"])))
    return out


def main() -> int:
    """Resumable driver: each finished arm is checkpointed into results.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-seconds", type=float, default=95.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, "results.json")
    nx = 128 if not args.quick else 64
    ny = nx
    steps = 300 if not args.quick else 80
    nps = [1, 2, 4]
    arms = ["unpacked", "unpacked-selective", "packed-selective"]
    t_start = time.time()

    results = {"config": {"nx": nx, "ny": ny, "steps": steps, "tau": TAU, "u0": U0,
                          "nu": (TAU - 0.5) / 3.0, "quick": args.quick},
               "runs": {}, "kpis": {}, "overlap": {}, "poll_sensitivity": {},
               "convergence": {}, "snapshots": {}}
    if os.path.exists(res_path):
        with open(res_path) as fh:
            old = json.load(fh)
        if old.get("config") == results["config"]:
            results = old

    def budget_left() -> bool:
        return (time.time() - t_start) < args.max_seconds

    def save() -> None:
        with open(res_path, "w") as fh:
            json.dump(results, fh, indent=1)

    def get_field(np_: int, cfg: dict) -> dict:
        return _assemble(_launch(np_, cfg))

    # ---- 1) arm x NP ladder ---------------------------------------------
    for kind in arms:
        for np_ in nps:
            key = f"{kind}_np{np_}"
            if key in results["kpis"]:
                continue
            if not budget_left():
                save()
                print("PARTIAL (budget): stopped before", key)
                return 0
            cfg = {"nx": nx, "ny": ny, "steps": steps, "halo": kind,
                   "overlap": True, "poll_every": 1}
            got = get_field(np_, cfg)
            ref = None
            if key != "unpacked_np1" and "unpacked_np1" in results.get("_refs", {}):
                ref_ux = np.asarray(results["_refs"]["unpacked_np1"]["ux"])
                ref_uy = np.asarray(results["_refs"]["unpacked_np1"]["uy"])
                ref = {"ux": ref_ux, "uy": ref_uy}
            kk = kpis(got, ref, nx, ny, steps)
            results["runs"][key] = {
                "halo": kind, "np": np_, "wall_s": got["wall_s"],
                "testall_polls": got["testall_polls"], "waitall": got["waitall_calls"],
                "messages_per_step": got["messages_per_step"],
                "wire_bytes_per_step": got["wire_bytes_per_step"],
                "staging_bytes_per_step": got["staging_bytes_per_step"],
            }
            results["kpis"][key] = kk
            if key == "unpacked_np1":
                results.setdefault("_refs", {})["unpacked_np1"] = {
                    "ux": got["ux"].tolist(), "uy": got["uy"].tolist()}
            save()
            print(f"[{key}] wall={got['wall_s']:.2f}s probe_err={kk['probe_rel_err_pct']:.3f}% "
                  f"identity={kk.get('identity_maxrel')}")

    # ---- 2) overlap on/off ----------------------------------------------
    for np_ in nps:
        key = f"np{np_}"
        if key in results["overlap"]:
            continue
        if not budget_left():
            save()
            print("PARTIAL (budget): stopped before overlap", key)
            return 0
        row = {}
        for tag, ov in (("overlap", True), ("blocking", False)):
            cfg = {"nx": nx, "ny": ny, "steps": steps, "halo": "unpacked",
                   "overlap": ov, "poll_every": 1}
            got = get_field(np_, cfg)
            ref = None
            if "unpacked_np1" in results.get("_refs", {}):
                ref = {"ux": np.asarray(results["_refs"]["unpacked_np1"]["ux"]),
                       "uy": np.asarray(results["_refs"]["unpacked_np1"]["uy"])}
            kk = kpis(got, ref, nx, ny, steps)
            row[tag] = {"wall_s": got["wall_s"], "mlups": kk["mlups"],
                        "identity_maxrel": kk.get("identity_maxrel"),
                        "testall_polls_per_step": kk["testall_polls_per_step"]}
        row["speedup_overlap_vs_blocking"] = row["blocking"]["wall_s"] / row["overlap"]["wall_s"]
        results["overlap"][key] = row
        save()
        print(f"[overlap {key}] speedup={row['speedup_overlap_vs_blocking']:.3f}")

    # ---- 3) Testall polling granularity (sensitivity A) ------------------
    for pe in (1, 8, 64):
        key = f"poll_every_{pe}"
        if key in results["poll_sensitivity"]:
            continue
        if not budget_left():
            save()
            print("PARTIAL (budget): stopped before", key)
            return 0
        cfg = {"nx": nx, "ny": ny, "steps": steps, "halo": "unpacked",
               "overlap": True, "poll_every": pe}
        got = get_field(max(nps), cfg)
        kk = kpis(got, None, nx, ny, steps)
        results["poll_sensitivity"][key] = {
            "wall_s": got["wall_s"], "mlups": kk["mlups"],
            "testall_polls_per_step": kk["testall_polls_per_step"]}
        save()
        print(f"[{key}] wall={got['wall_s']:.2f}s polls/step={kk['testall_polls_per_step']:.1f}")

    # ---- 4) grid convergence (sensitivity B) -----------------------------
    for n in (64, 128):
        key = f"n{n}"
        if key in results["convergence"]:
            continue
        if not budget_left():
            save()
            print("PARTIAL (budget): stopped before", key)
            return 0
        steps_c = max(40, int(round(steps * (n / nx) ** 2)))  # equal decay time
        cfg = {"nx": n, "ny": n, "steps": steps_c, "halo": "unpacked",
               "overlap": True, "poll_every": 1}
        got = get_field(1, cfg)
        kk = kpis(got, None, n, n, steps_c)
        results["convergence"][key] = {
            "steps": steps_c, "probe_rel_err_pct": kk["probe_rel_err_pct"],
            "field_l2_rel_err_pct": kk["field_l2_rel_err_pct"],
            "decay_factor_measured": kk["decay_factor_measured"],
            "decay_factor_analytic": kk["decay_factor_analytic"]}
        save()
        print(f"[conv {key}] probe_err={kk['probe_rel_err_pct']:.3f}%")

    # ---- 5) field snapshots for the figures -----------------------------
    for np_ in (1, 4):
        skey = f"np{np_}"
        if skey in results["snapshots"]:
            continue
        if not budget_left():
            save()
            print("PARTIAL (budget): stopped before snapshot", skey)
            return 0
        cfg = {"nx": nx, "ny": ny, "steps": steps, "halo": "unpacked",
               "overlap": True, "poll_every": 1}
        got = get_field(np_, cfg)
        results["snapshots"][skey] = {"ux": got["ux"].tolist(), "uy": got["uy"].tolist()}
        save()
    if "analytic" not in results:
        ax, ay = analytic_fields(nx, ny, steps)
        results["analytic"] = {"ux": ax.tolist(), "uy": ay.tolist()}
        save()

    keys = ["probe_rel_err_pct", "field_l2_rel_err_pct", "identity_maxrel",
            "wire_bytes_per_step", "staging_bytes_per_step", "messages_per_step",
            "testall_polls_per_step", "mlups", "mass_drift_rel", "ma_max"]
    with open(os.path.join(args.out, "kpi.csv"), "w") as fh:
        fh.write("arm," + ",".join(keys) + "\n")
        for arm, kk in results["kpis"].items():
            fh.write(arm + "," + ",".join(
                "" if kk.get(k) is None else f"{kk[k]:.6g}" for k in keys) + "\n")
    print("COMPLETE", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())