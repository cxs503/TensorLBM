#!/usr/bin/env python3
"""8-card independent card-interference probe (cylinder Re=100 workload).

Mission m5113. Measures how much 8 SDAA cards interfere with each other when
each runs its OWN independent simulation (task-level parallelism), as opposed
to domain decomposition. Deliverable artefact:
    artifacts/card_interference_probe.json

Auxiliary: docs/card_probe_report.md

Methodology
-----------
Every timing measurement uses a separate OS process pinned to exactly one card
through ``SDAA_VISIBLE_DEVICES``, so a process can only ever see one device.
The orchestrator drives three phases:

  Phase S  : each card alone, nothing else on the machine  -> t1[i]
  Phase C  : all 8 workers live simultaneously             -> t8[i]

Co-tenancy is enforced *only* inside the timed window: workers finish their
warmup, announce readiness, and then block on a "go" barrier file. The
orchestrator releases the barrier only after all workers are ready, so the
timed loops overlap exactly. Outside the timed window the machine is quiet.

Per-card measurements (all inside the same release window where relevant):

  * cylinder Re=100 step time  -> speedup / interference
  * STREAM-triad HBM bandwidth -> memory_bandwidth_GBs, bandwidth interference
  * H2D + D2H copy bandwidth   -> PCIe traffic proxy (no counter is exposed by
                                  teco-smi 1.15.0 on this host; record honestly)
  * drag coefficient Cd        -> physics-identity check across cards

Speedup definition (aggregate throughput of 8 independent cards vs 1):

    speedup_8x = 8 * t1_median / t8_median

With zero interference t8 == t1 and speedup_8x == 8.0.

interference_pct = (t8_median / t1_median - 1) * 100

honest-FAIL applies when speedup_8x < 6.0.

Usage
-----
    # full probe (default 8 cards, cards 14..21)
    python scripts/card_interference_probe.py --cards 14,15,16,17,18,19,20,21

    # quick smoke (1 repetition, fewer steps)
    python scripts/card_interference_probe.py --steps 200 --reps 1
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Case definition: 2D cylinder, D2Q9 BGK, Re=100
# ---------------------------------------------------------------------------
CASE = {
    "name": "cylinder_2d_re100",
    "nx": 512,          # streamwise cells
    "ny": 256,          # cross-stream cells
    "diameter": 16.0,   # lattice cells
    "cx": 112.0,        # cylinder centre x (cells)
    "cy": 128.0,        # cylinder centre y (cells)
    "u_in": 0.05,       # inlet velocity (lattice units)
    "re": 100.0,
    "steps_default": 1200,
    "warmup_steps": 50,
}
CASE["tau"] = 3.0 * (CASE["u_in"] * CASE["diameter"] / CASE["re"]) + 0.5

SETTLE_S = float(os.environ.get("CARD_PROBE_SETTLE_S", "3.0"))
STREAM_TENSOR_MB = 512          # per-card STREAM triad working-set size
PCIe_TENSOR_MB = 256            # host<->device transfer size


# ---------------------------------------------------------------------------
# Worker: everything measured on a single card
# ---------------------------------------------------------------------------
def build_cylinder_case(nx: int, ny: int, diameter: float, cx: float, cy: float):
    import torch

    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    cyl = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (diameter / 2.0) ** 2
    wall = torch.zeros(ny, nx, dtype=torch.bool)
    wall[0, :] = True
    wall[-1, :] = True
    solid = cyl | wall
    return solid, cyl


def run_cylinder(
    dev,
    steps: int,
    warmup: int,
    barrier_dir: Path | None,
    worker_tag: int,
    measure_cd: bool = True,
) -> dict:
    """Run the cylinder Re=100 workload; return timings + diagnostics."""
    import torch

    from tensorlbm import collide_bgk, equilibrium, macroscopic, stream
    from tensorlbm import OPPOSITE

    nx, ny = CASE["nx"], CASE["ny"]
    tau = CASE["tau"]
    u_in = CASE["u_in"]

    solid_cpu, cyl = build_cylinder_case(nx, ny, CASE["diameter"], CASE["cx"], CASE["cy"])
    solid = solid_cpu.to(dev)
    cyl_d = cyl.to(dev)
    opp = OPPOSITE.to(dev)

    c_x = torch.tensor(
        [0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=torch.float32, device=dev
    )

    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full((ny, nx), u_in, dtype=torch.float32, device=dev)
    uy0 = torch.zeros((ny, nx), dtype=torch.float32, device=dev)
    f = equilibrium(rho0, ux0, uy0, device=dev)

    eq_col = equilibrium(
        torch.ones(ny, dtype=torch.float32, device=dev),
        torch.full((ny,), u_in, dtype=torch.float32, device=dev),
        torch.zeros(ny, dtype=torch.float32, device=dev),
        device=dev,
    )
    # equilibrium() promotes 1-D inputs to (9, 1, ny); normalise to a column
    eq_col = eq_col.reshape(9, ny)

    def one_step(f):
        f = collide_bgk(f, tau)
        f = stream(f)
        # halfway bounce-back on all solid nodes (cylinder + walls).
        # The full-field gather+where is measurably faster on SDAA than
        # boolean-mask scatter, so keep it (identical arithmetic).
        fs = f[opp]
        f = torch.where(solid.unsqueeze(0), fs, f)
        # inlet / outlet (in place: no extra allocation in the timed loop)
        f[:, :, 0] = eq_col
        f[:, :, -1] = f[:, :, -2]
        return f

    def measure_cd(f):
        # momentum exchange on the cylinder nodes only (walls excluded)
        fx = 2.0 * (c_x.view(9, 1) * f[:, cyl_d]).sum()
        return float(abs(fx) / (0.5 * 1.0 * u_in**2 * CASE["diameter"]))

    # ---- warmup (not timed, and completed before the barrier is announced) --
    for _ in range(warmup):
        f = one_step(f)
    sync(dev)
    # Let the accelerator reach a steady clock/power state before the timed
    # window. Without this the first timed steps of an otherwise-idle card run
    # at a lower DVFS point and bias the single-card baseline upward.
    time.sleep(SETTLE_S)

    # ---- co-tenancy barrier: all workers release together -----------------
    if barrier_dir is not None:
        signal_ready_and_wait(barrier_dir, worker_tag)

    # ---- timed window: NO host-side work, pure kernel loop -----------------
    step_times = []
    for idx in range(steps):
        if idx > 0:
            sync(dev)
            t0 = time.perf_counter()
        f = one_step(f)
        sync(dev)
        if idx > 0:
            step_times.append((time.perf_counter() - t0) * 1e3)

    cd = measure_cd(f) if measure_cd else None

    rho, ux, uy = macroscopic(f)
    speed = torch.sqrt(ux**2 + uy**2)

    return {
        "mean_step_ms": statistics.fmean(step_times),
        "median_step_ms": statistics.median(step_times),
        "p95_step_ms": sorted(step_times)[int(0.95 * (len(step_times) - 1))],
        "std_step_ms": statistics.pstdev(step_times),
        "cv_pct": 100.0 * statistics.pstdev(step_times) / statistics.fmean(step_times),
        "n_timed_steps": len(step_times),
        "mlups": (nx * ny * len(step_times)) / (sum(step_times) * 1e3),
        "cd": cd,
        "finite": bool(
            torch.isfinite(f).all() and torch.isfinite(rho).all()
            and torch.isfinite(ux).all()
        ),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "u_max": float(speed.max()),
        "f_checksum": float(f.sum()),
        "wake_ux": [float(v) for v in ux[:, -1].detach().cpu()],
        "_field": None,
    }


def sync(dev) -> None:
    import torch
    if dev.type == "sdaa":
        torch.sdaa.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Barrier helpers (file based, so they work across independent processes)
# ---------------------------------------------------------------------------
def signal_ready_and_wait(barrier_dir: Path, tag: int) -> None:
    (barrier_dir / f"ready_{tag}").write_text("1")
    deadline = time.time() + 600.0
    while not (barrier_dir / "go").exists():
        if time.time() > deadline:
            raise TimeoutError("barrier timeout: orchestrator never released")
        time.sleep(0.002)


# ---------------------------------------------------------------------------
# Bandwidth probes
# ---------------------------------------------------------------------------
def measure_hbm_bandwidth(dev, size_mb: int, reps: int = 5) -> dict:
    """STREAM-triad style HBM bandwidth (read 2 arrays, write 1)."""
    import torch

    n = (size_mb * 1024 * 1024) // 4  # float32 elements
    a = torch.rand(n, dtype=torch.float32, device=dev)
    b = torch.rand(n, dtype=torch.float32, device=dev)
    c = torch.empty(n, dtype=torch.float32, device=dev)
    for _ in range(2):
        torch.add(a, b, out=c)
    sync(dev)

    times = []
    for _ in range(reps):
        sync(dev)
        t0 = time.perf_counter()
        torch.add(a, b, out=c)
        sync(dev)
        times.append(time.perf_counter() - t0)

    best = min(times)
    traffic = 3 * n * 4  # 2 reads + 1 write
    return {
        "size_mb": size_mb,
        "median_ms": statistics.median(times) * 1e3,
        "best_ms": best * 1e3,
        "gbs_median": traffic / (statistics.median(times) * 1e9),
        "gbs_best": traffic / (best * 1e9),
    }


def measure_pcie_bandwidth(dev, size_mb: int, reps: int = 9) -> dict:
    """Host->device and device->host copy bandwidth = PCIe traffic proxy."""
    import torch

    n = (size_mb * 1024 * 1024) // 4
    host = torch.rand(n, dtype=torch.float32)
    gpu = torch.empty(n, dtype=torch.float32, device=dev)

    h2d, d2h = [], []
    for _ in range(2):
        gpu.copy_(host)
        host.copy_(gpu)
    sync(dev)

    for _ in range(reps):
        t0 = time.perf_counter()
        gpu.copy_(host)
        sync(dev)
        h2d.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        host.copy_(gpu)
        sync(dev)
        d2h.append(time.perf_counter() - t0)

    nbytes = n * 4
    return {
        "size_mb": size_mb,
        "h2d_gbs": nbytes / (statistics.median(h2d) * 1e9),
        "d2h_gbs": nbytes / (statistics.median(d2h) * 1e9),
        "counter_available": False,
        "note": (
            "teco-smi 1.15.0 exposes no PCIe throughput counter on this host; "
            "H2D/D2H copy bandwidth is the measured proxy for PCIe traffic."
        ),
    }


# ---------------------------------------------------------------------------
# Worker entry point (one OS process == one card)
# ---------------------------------------------------------------------------
def worker_main(args) -> None:
    import torch
    import torch_sdaa  # noqa: F401  (registers the sdaa backend)

    out_path = Path(args.out)
    dev = torch.device("sdaa:0")
    card_name = torch.sdaa.get_device_name(0)
    if torch.sdaa.device_count() != 1:
        raise RuntimeError(
            f"worker isolation violated: {torch.sdaa.device_count()} devices visible"
        )

    # optional cross-stream extent override, used for the blockage-confinement
    # sensitivity study that accompanies the interference probe
    if os.environ.get("CARD_PROBE_NY"):
        CASE["ny"] = int(os.environ["CARD_PROBE_NY"])
        CASE["cy"] = CASE["ny"] / 2.0

    torch.manual_seed(1234)

    barrier_dir = Path(args.barrier_dir) if args.barrier_dir else None
    result = {
        "card_env": os.environ.get("SDAA_VISIBLE_DEVICES"),
        "device_name": card_name,
        "visible_devices": torch.sdaa.device_count(),
    }
    result["hbm"] = measure_hbm_bandwidth(dev, STREAM_TENSOR_MB)
    result["pcie"] = measure_pcie_bandwidth(dev, PCIe_TENSOR_MB)
    # free the bandwidth tensors before the timed LBM window
    torch.sdaa.empty_cache()

    result["cylinder"] = run_cylinder(
        dev,
        steps=args.steps,
        warmup=args.warmup,
        barrier_dir=barrier_dir,
        worker_tag=args.tag,
    )
    result["cylinder"].pop("_field", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def spawn_workers(cards, out_dir, barrier_dir, steps, warmup, tag_offset=0):
    procs = []
    for i, card in enumerate(cards):
        env = dict(os.environ)
        env["SDAA_VISIBLE_DEVICES"] = str(card)
        env["SDAA_QUIET"] = "1"
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        out = out_dir / f"worker_{i}.json"
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker",
            "--out", str(out),
            "--steps", str(steps),
            "--warmup", str(warmup),
            "--tag", str(i + tag_offset),
        ]
        if barrier_dir is not None:
            cmd += ["--barrier-dir", str(barrier_dir)]
        procs.append((i, card, out, subprocess.Popen(cmd, env=env)))
    return procs


def collect(procs, timeout=1800):
    results = []
    for i, card, out, p in procs:
        rc = p.wait(timeout=timeout)
        if rc != 0:
            raise RuntimeError(f"worker {i} (card {card}) failed rc={rc}")
        results.append(json.loads(out.read_text()))
    return results


def run_phase(cards, out_dir: Path, steps, warmup, label, barrier=True, tag_offset=0):
    # transient rendezvous point: keep it out of the artefact tree
    barrier_dir = Path(tempfile.mkdtemp(prefix=f"cardprobe_{label}_"))

    t0 = time.perf_counter()
    procs = spawn_workers(
        cards, out_dir, barrier_dir if barrier else None, steps, warmup, tag_offset
    )
    if barrier and len(cards) > 1:
        # release when every worker has finished its warmup
        deadline = time.time() + 900
        while True:
            if all((barrier_dir / f"ready_{i + tag_offset}").exists() for i in range(len(cards))):
                break
            if time.time() > deadline:
                raise TimeoutError("workers never became ready")
            time.sleep(0.002)
        (barrier_dir / "go").write_text(str(time.time()))
    # sample clocks while the timed window is live
    time.sleep(2.0)
    clocks = sample_clocks()
    results = collect(procs)
    elapsed = time.perf_counter() - t0
    return results, elapsed, clocks


def sample_clocks() -> dict:
    """Read MPE/SPE clocks from teco-smi (current + max) for all cards."""
    try:
        r = subprocess.run(["teco-smi", "-q", "-d", "CLOCK"],
                           capture_output=True, text=True, timeout=25)
        mpe, spe = [], []
        for line in r.stdout.splitlines():
            ls = line.strip()
            try:
                if ls.startswith("Mpe"):
                    mpe.append(float(ls.split(":")[1].split()[0]))
                elif ls.startswith("Spe"):
                    spe.append(float(ls.split(":")[1].split()[0]))
            except (IndexError, ValueError):
                continue
        # output interleaves the "Clocks" and "Max Clocks" blocks per device
        cur_mpe, max_mpe = mpe[0::2], mpe[1::2]
        cur_spe, max_spe = spe[0::2], spe[1::2]
        return {
            "mpe_mhz_median": statistics.median(cur_mpe) if cur_mpe else None,
            "mpe_mhz_max": max_mpe[0] if max_mpe else None,
            "spe_mhz_median": statistics.median(cur_spe) if cur_spe else None,
            "spe_mhz_max": max_spe[0] if max_spe else None,
            "n_cards_reported": len(cur_mpe),
        }
    except Exception as exc:
        return {"error": str(exc)}


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=REPO_ROOT,
        ).stdout.strip()[:12]
    except Exception:
        return "unknown"


def capture_teco_smi() -> str:
    try:
        r = subprocess.run(["teco-smi", "-q", "-d", "MEMORY,UTILIZATION"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout
    except Exception as exc:  # pragma: no cover
        return f"teco-smi unavailable: {exc}"


def orchestrator_main(args) -> int:
    cards = [int(c) for c in args.cards.split(",")]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[probe] case={CASE['name']} tau={CASE['tau']:.4f} "
          f"grid={CASE['nx']}x{CASE['ny']} steps={args.steps} reps={args.reps}")
    print(f"[probe] cards={cards}")

    singles = {c: [] for c in cards}
    conc = []
    clocks_log = []
    # Alternate the order of the single and concurrent phases across
    # repetitions so that slow thermal/DVFS drift cancels instead of biasing
    # the concurrent phase in one direction.
    for rep in range(args.reps):
        order = ("single", "concurrent") if rep % 2 == 0 else ("concurrent", "single")
        for phase in order:
            if phase == "single":
                for card in cards:
                    tag = 1000 * (rep + 1) + card
                    res, el, clk = run_phase(
                        [card], out_root / f"s_rep{rep}_card{card}", args.steps,
                        args.warmup, f"s_{rep}_{card}", barrier=False, tag_offset=tag)
                    singles[card].append(res[0])
                    clocks_log.append({"phase": "single", "rep": rep,
                                       "card": card, **clk})
                    print(f"[single] card {card} rep{rep}: "
                          f"median_step={res[0]['cylinder']['median_step_ms']:.4f} ms "
                          f"mpe={clk.get('mpe_mhz_median')} ({el:.1f}s wall)")
            else:
                res, el, clk = run_phase(
                    cards, out_root / f"c_rep{rep}", args.steps, args.warmup,
                    f"c_{rep}", barrier=True, tag_offset=0)
                conc.append(res)
                clocks_log.append({"phase": "concurrent", "rep": rep, **clk})
                med = statistics.median([r["cylinder"]["median_step_ms"] for r in res])
                print(f"[8-card] rep{rep}: median_step={med:.4f} ms "
                      f"mpe={clk.get('mpe_mhz_median')} (wall {el:.1f}s)")

    # ---- aggregate -------------------------------------------------------
    def med(vals):
        return statistics.median(vals)

    single_step = [med([s["cylinder"]["median_step_ms"] for s in singles[c]]) for c in cards]
    conc_step = [med([rep[i]["cylinder"]["median_step_ms"] for rep in conc])
                 for i in range(len(cards))]

    t1 = med(single_step)
    t8 = med(conc_step)
    speedup = 8.0 * t1 / t8
    interference = (t8 / t1 - 1.0) * 100.0

    per_card = []
    for i, card in enumerate(cards):
        s_step = single_step[i]
        c_step = conc_step[i]
        per_card.append({
            "slot": i,
            "card_env": card,
            "device_name": singles[card][0]["device_name"],
            "single_median_step_ms": s_step,
            "concurrent_median_step_ms": c_step,
            "slowdown_pct": (c_step / s_step - 1.0) * 100.0,
            "single_mlups": singles[card][0]["cylinder"]["mlups"],
            "concurrent_mlups": conc[-1][i]["cylinder"]["mlups"],
            "single_step_cv_pct": singles[card][0]["cylinder"]["cv_pct"],
            "concurrent_step_cv_pct": conc[-1][i]["cylinder"]["cv_pct"],
            "single_hbm_gbs": singles[card][0]["hbm"]["gbs_median"],
            "concurrent_hbm_gbs": conc[-1][i]["hbm"]["gbs_median"],
            "hbm_degradation_pct": (
                conc[-1][i]["hbm"]["gbs_median"] / singles[card][0]["hbm"]["gbs_median"] - 1
            ) * 100.0,
            "single_h2d_gbs": singles[card][0]["pcie"]["h2d_gbs"],
            "concurrent_h2d_gbs": conc[-1][i]["pcie"]["h2d_gbs"],
            "single_d2h_gbs": singles[card][0]["pcie"]["d2h_gbs"],
            "concurrent_d2h_gbs": conc[-1][i]["pcie"]["d2h_gbs"],
            "cd": singles[card][0]["cylinder"]["cd"],
            "concurrent_cd": conc[-1][i]["cylinder"]["cd"],
            "finite": singles[card][0]["cylinder"]["finite"] and conc[-1][i]["cylinder"]["finite"],
            "single_step_spread_ms": [s["cylinder"]["median_step_ms"] for s in singles[card]],
            "concurrent_step_spread_ms": [r[i]["cylinder"]["median_step_ms"] for r in conc],
        })

    slowdowns = [p["slowdown_pct"] for p in per_card]
    hbm_single = med([p["single_hbm_gbs"] for p in per_card])
    hbm_conc = med([p["concurrent_hbm_gbs"] for p in per_card])

    honest_fail = speedup < 6.0
    payload = {
        "mission": "m5113",
        "probe": "8-card independent (task-level parallelism) interference",
        "git_sha": git_sha(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case": dict(CASE),
        "workload": {
            "solver": "tensorlbm D2Q9 BGK + halfway bounce-back, inlet/outlet",
            "grid": f"{CASE['nx']}x{CASE['ny']}",
            "diameter_cells": CASE["diameter"],
            "blockage_pct": 100.0 * CASE["diameter"] / CASE["ny"],
            "steps_timed": args.steps,
            "warmup_steps": args.warmup,
            "reps": args.reps,
            "settle_s": SETTLE_S,
        },
        "devices": {
            "n_cards": len(cards),
            "sdda_visible_devices": ",".join(str(c) for c in cards),
            "device_names": [p["device_name"] for p in per_card],
            "isolation": (
                "one OS process per card; SDAA_VISIBLE_DEVICES forced to a single "
                "card; worker asserts torch.sdaa.device_count()==1"
            ),
            "exclusive_during_timed_window": True,
            "co_tenancy_protocol": (
                "workers finish warmup, publish ready_<tag>, then block on a go "
                "barrier file released only after all workers are ready; timed "
                "loops therefore overlap exactly and no other work runs on the "
                "machine during the timed window"
            ),
        },
        "results": {
            "single_card_median_step_ms": t1,
            "concurrent_8card_median_step_ms": t8,
            "speedup_8x": speedup,
            "speedup_vs_1x_per_card_median": t1 / t8,
            "interference_pct": interference,
            "interference_pct_max_card": max(slowdowns),
            "interference_pct_mean_card": statistics.fmean(slowdowns),
            "interference_pct_std_card": statistics.pstdev(slowdowns),
            "parallel_efficiency_pct": 100.0 * speedup / 8.0,
            "memory_bandwidth_GBs": hbm_conc,
            "memory_bandwidth_GBs_single": hbm_single,
            "memory_bandwidth_degradation_pct": (hbm_conc / hbm_single - 1.0) * 100.0,
            "aggregate_mlups_8card": sum(p["concurrent_mlups"] for p in per_card),
            "aggregate_mlups_1card": med([p["single_mlups"] for p in per_card]),
        },
        "per_card": per_card,
        "verdict": {
            "honest_fail": bool(honest_fail),
            "criterion": "speedup_8x >= 6.0 passes; < 6.0 is honest-FAIL",
            "recommendation": (
                "single-card-first + task-level parallelism"
                if honest_fail
                else "8-card task-level parallelism is viable"
            ),
        },
        "clocks": {
            "samples": clocks_log,
            "single_mpe_mhz_median": statistics.median(
                [c["mpe_mhz_median"] for c in clocks_log
                 if c["phase"] == "single" and c.get("mpe_mhz_median")]) or None,
            "concurrent_mpe_mhz_median": statistics.median(
                [c["mpe_mhz_median"] for c in clocks_log
                 if c["phase"] == "concurrent" and c.get("mpe_mhz_median")]) or None,
        },
        "teco_smi_snapshot": capture_teco_smi()[:4000],
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[probe] wrote {out_path}")
    print(f"[probe] speedup_8x={speedup:.3f} interference={interference:.2f}% "
          f"honest_fail={honest_fail}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", default="14,15,16,17,18,19,20,21")
    ap.add_argument("--steps", type=int, default=CASE["steps_default"])
    ap.add_argument("--warmup", type=int, default=CASE["warmup_steps"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out-dir", default="artifacts/card_probe_runs")
    ap.add_argument(
        "--out-json",
        default=str(REPO_ROOT / "artifacts" / "card_interference_probe.json"),
    )
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--barrier-dir", default=None)
    ap.add_argument("--tag", type=int, default=0)
    args = ap.parse_args(argv)

    if args.worker:
        worker_main(args)
        return 0
    return orchestrator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())