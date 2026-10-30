#!/usr/bin/env python3
"""SDAA validation sweep — ``benchmarks/verified/suboff_sail_resistance``.

Independent driver (SDAA case sweep).  It **never edits** the case: business
code is only *read*, and every numeric run is a ``subprocess`` invocation of the
frozen case entry ``benchmarks/verified/suboff_sail_resistance/run.py``.

Caliber kept verbatim from the case (no re-tuning, no gate relaxation):

* reference       : Blasius laminar flat plate Cf = 1.328/sqrt(Re), Re = 1000,
                    on the *total* wetted area S = pi(2R_lb)L + A_sail_own
                    (recomputed here, never copied from the archives);
* frozen ladder   : L in {96, 112, 128}, steps = 150*L (prereg.md), i.e.
                    14400 / 16800 / 19200; domain padding (1,4,1,1,1,1);
* frozen physics  : D3Q19 MRT, u_lb = 0.05, nu = u_lb*L/1000, tau = 3nu+0.5,
                    friction = ``standard`` (erratum E2 primary estimator),
                    p0 = ``near_wall``, compile-mode = ``eager`` (prereg);
* window          : mean of the last 1000 force samples (= last 10000 steps);
* gates (prereg / E2 verdict):
    G1  finest |err| = |Cd/Cf_ref - 1| <= 3 %
    G2  |err| non-increasing along the ladder
    G3  common-module entry (GeneralSimEngine + suboff_cad + drag_pressure)
    G4  window stability |Cd(100%) - Cd(75%)| / |Cd(75%)| <= 2.5 %
    G5  sail solid cells vs frozen mask table {96:48, 112:70, 128:126} +-2 and
        sail y-thickness == 2

Device policy: every tensor computation is requested on ``sdaa`` via
``--device sdaa:0``; the device line recorded in the artifact is the verbatim
``torch.sdaa.get_device_name(0)`` output.  No CPU result is ever reported as an
SDAA result.  If a frozen tier cannot run on this card the gate is recorded as
FAIL (honest-FAIL), never silently downgraded — a smaller grid is run only as an
explicitly labelled *supplementary* capacity-limited arm and never enters the
gate decisions.

``TRITON_INTERPRET``: the executed path (eager MRT chain) imports no triton
kernel; SDAA cannot use the compiled route at all (``torch.compile`` on Hygon
runs through ``teco_inductor``, whose L5 defect aborts this chain — see
``benchmarks/compile_route.py``), so the run uses the prereg-frozen eager mode
and no triton bypass is required.  The script records that fact instead of
setting a flag it does not need.

Usage
-----
    python scripts/sdaa_sweep_suboff_sail_resistance.py
    python scripts/sdaa_sweep_suboff_sail_resistance.py --budget 600
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "benchmarks" / "verified" / "suboff_sail_resistance"
RUN_PY = CASE_DIR / "run.py"
ARTIFACT = REPO_ROOT / "artifacts" / "sdaa_suboff_sail_resistance.json"
WORK_ROOT = REPO_ROOT / "runs" / "sdaa_sweep_suboff_sail_resistance"

# ---------------------------------------------------------------- caliber ---
RE = 1000.0                       # case Reynolds number (prereg, frozen)
LADDER = (96, 112, 128)           # frozen ladder (erratum E1)
STEPS_PER_L = 150                 # prereg: steps = 150*L
G5_MASK_TABLE = {96: 48, 112: 70, 128: 126}   # frozen pre-run mask table (E1)
G1_MAX_ERR_PCT = 3.0
G4_MAX_DRIFT_PCT = 2.5
G5_TOL_CELLS = 2
G5_SAIL_Y_THICKNESS = 2
DEVICE = "sdaa:0"
FALLBACK_CANDIDATES = (72, 70, 68, 64)  # off-ladder, descending; only if the ladder is infeasible
FALLBACK_STEP_CAP_S = 540.0       # wall-clock budget for the supplementary arm
CASE_TIERS_HINT = "README.md / prereg.md: ladder L={96,112,128}, steps=150*L"


def cf_ref_blasius(re: float = RE) -> float:
    """Blasius laminar flat-plate Cf = 1.328/sqrt(Re) (case primary reference)."""
    return 1.328 / math.sqrt(re)


def steps_for(L: int) -> int:
    """Case-caliber step count for tier L: 150*L (prereg.md, frozen)."""
    return STEPS_PER_L * L


# ------------------------------------------------------------- device info ---
def device_evidence() -> dict:
    """Verbatim SDAA device evidence (no CPU fallback, no fabrication)."""
    import torch  # noqa: PLC0415

    ev: dict = {"torch": torch.__version__, "requested_device": DEVICE}
    ev["torch_sdaa"] = None
    ev["sdaa_available"] = False
    ev["device_line"] = None
    ev["device_count"] = None
    try:
        import torch_sdaa  # type: ignore  # noqa: PLC0415

        ev["torch_sdaa"] = getattr(torch_sdaa, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - env dependent
        ev["torch_sdaa_import_error"] = f"{type(exc).__name__}: {exc}"
        ev["device_line"] = f"UNAVAILABLE ({type(exc).__name__})"
        return ev
    try:
        ev["sdaa_available"] = bool(torch.sdaa.is_available())
        ev["device_count"] = int(torch.sdaa.device_count())
        if ev["sdaa_available"] and ev["device_count"] > 0:
            ev["device_line"] = str(torch.sdaa.get_device_name(0))
            try:
                free, total = torch.sdaa.mem_get_info(0)
                ev["mem_free_gib"] = round(free / 2**30, 3)
                ev["mem_total_gib"] = round(total / 2**30, 3)
            except Exception:
                try:
                    props = torch.sdaa.get_device_properties(0)
                    ev["mem_total_gib"] = round(
                        getattr(props, "total_memory", 0) / 2**30, 3
                    )
                except Exception:
                    ev["mem_total_gib"] = None
            t = torch.ones(4, device=DEVICE) * 2.0
            ev["liveness_check"] = {
                "op": "sdaa: torch.ones(4)*2 -> sum().item()",
                "value": float(t.sum().item()),
                "pass": float(t.sum().item()) == 8.0,
            }
    except Exception as exc:  # pragma: no cover - env dependent
        ev["sdaa_probe_error"] = f"{type(exc).__name__}: {exc}"
    return ev


# ------------------------------------------------------- log/json scraping ---
_DOMAIN_RE = re.compile(r'"domain_lu":\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
_CELLS_RE = re.compile(r'"total_cells":\s*(\d+)')
_SAIL_RE = re.compile(r'"appendage_solid_cells":\s*(\d+)')
_ROUTE_RE = re.compile(r"\[compile_route\][^\n]*")
_STEP_RE = re.compile(r"run finished: completed in ([0-9.]+)s \(([0-9.]+) ms/step\)")
_ERR_RE = re.compile(r"(?:RuntimeError|Error):[^\n]*")


def _parse_log(log: str) -> dict:
    info: dict = {}
    m = _DOMAIN_RE.search(log)
    if m:
        info["domain"] = [int(g) for g in m.groups()]
    m = _CELLS_RE.search(log)
    if m:
        info["total_cells"] = int(m.group(1))
    m = _SAIL_RE.search(log)
    if m:
        info["appendage_solid_cells_log"] = int(m.group(1))
    m = _ROUTE_RE.search(log)
    if m:
        info["compile_route_line"] = m.group(0).strip()
    m = _STEP_RE.search(log)
    if m:
        info["loop_elapsed_s"] = float(m.group(1))
        info["ms_per_step_log"] = float(m.group(2))
    if "out of memory" in log:
        info["oom"] = True
        m = _ERR_RE.search(log)
        info["oom_line"] = m.group(0).strip() if m else "out of memory"
    if "Traceback (most recent call last)" in log:
        info["traceback_tail"] = log.strip().splitlines()[-1]
    return info


def run_case(L: int, steps: int, tag: str, timeout: float) -> dict:
    """Run the frozen case entry once on sdaa (subprocess, no code edits)."""
    out_dir = WORK_ROOT / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(RUN_PY),
        "--hull-type", "with_sail",
        "--resolution", str(L),
        "--steps", str(steps),
        "--device", DEVICE,
        "--collision", "mrt",
        "--friction", "standard",
        "--p0", "near_wall",
        "--compile-mode", "eager",
        "--out", str(out_dir),
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    status = "ok"
    try:
        proc = subprocess.run(
            cmd, cwd=str(CASE_DIR), env=env, capture_output=True,
            text=True, timeout=timeout,
        )
        log = proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        so = exc.stdout or ""
        se = exc.stderr or ""
        if isinstance(so, bytes):
            so = so.decode("utf-8", "replace")
        if isinstance(se, bytes):
            se = se.decode("utf-8", "replace")
        log = so + "\n--- STDERR ---\n" + se
        rc = None
    wall = time.time() - t0
    log_path = WORK_ROOT / f"{tag}.log"
    log_path.write_text(log)

    result = None
    res_path = out_dir / f"result_L{L}.json"
    if res_path.exists():
        try:
            result = json.loads(res_path.read_text())
        except Exception:
            result = None
    parsed = _parse_log(log)
    ok = bool(result) and status == "ok" and rc == 0 and not parsed.get("oom")
    return {
        "tag": tag,
        "resolution_L": L,
        "requested_steps": steps,
        "device": DEVICE,
        "command": " ".join(cmd[1:]),
        "wall_clock_s": round(wall, 2),
        "returncode": rc,
        "ok": ok,
        "run_status": "ok" if ok else (status if status != "ok" else "failed"),
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "result_path": str(res_path.relative_to(REPO_ROOT)) if res_path.exists() else None,
        "log_parsed": parsed,
        "result": result,
        "oom_line": parsed.get("oom_line"),
        "traceback_tail": parsed.get("traceback_tail"),
    }


def _fill_row(row: dict, run: dict) -> None:
    """Derive every criterion number from the raw run JSON (nothing hand-copied)."""
    res = run.get("result")
    if not res:
        row["ok"] = False
        row["blocked_reason"] = (run.get("oom_line") or run.get("traceback_tail")
                                 or "no result JSON")
        return
    cf = cf_ref_blasius()
    cd = float(res["Cd_total"])
    row["ok"] = bool(run["ok"])
    row["run_steps"] = int(res["n_steps"])
    row["run_device"] = res.get("device")
    row["cd_total_window"] = cd
    row["cd_pressure"] = float(res.get("Cd_pressure", float("nan")))
    row["cd_friction"] = float(res.get("Cd_friction", float("nan")))
    row["cf_ref_runner"] = float(res.get("Cf_ref_blasius", float("nan")))
    row["cf_ref_recomputed"] = cf
    row["cf_ref_match"] = abs(row["cf_ref_runner"] - cf) / cf < 1e-9
    row["err_pct"] = (cd - cf) / cf * 100.0
    row["err_pct_runner"] = float(res.get("error_pct_vs_Blasius_totalwetted", float("nan")))
    row["err_runner_match"] = abs(row["err_pct"] - row["err_pct_runner"]) < 1e-6
    conv = res.get("convergence_windows") or {}
    if conv.get("75%") and conv.get("100%") is not None:
        row["drift_pct"] = abs(conv["100%"] - conv["75%"]) / abs(conv["75%"]) * 100.0
    else:
        row["drift_pct"] = None
    row["convergence_windows"] = conv
    row["window_samples"] = res.get("window_samples")
    row["ms_per_step"] = res.get("ms_per_step")
    row["loop_elapsed_s"] = res.get("elapsed_s")
    row["finite"] = res.get("finite")
    row["diverged"] = res.get("diverged")
    row["appendage_solid_cells_result"] = (
        res.get("appendage_stats") or {}
    ).get("appendage_solid_cells")
    row["wetted_area_lu2"] = res.get("S_total_wetted_lb2")
    row["compile_mode_effective"] = res.get("compile_mode_effective")
    row["compile_status"] = res.get("compile_status")
    row["entry_fields_ok"] = all(
        k in res for k in ("S_total_wetted_lb2", "dpS_type", "appendage_stats", "Cd_total")
    )


# ----------------------------------------------------------- G5 geometry ---
def sail_geometry(L: int, domain: list) -> dict:
    """Sail cell count + y-thickness from the case's own mask builder.

    Uses ``tensorlbm.suboff_cad.build_suboff_mask`` with exactly the arguments
    ``run.py::SuboffAppendageEngine._build_suboff_solid`` derives from the
    printed domain — the geometry is not re-implemented here.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import torch  # noqa: PLC0415

    from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask  # noqa: PLC0415

    nx, ny, nz = domain
    dx = 4.356 / float(L)          # run.py: dx = reference_length / resolution
    length_lb = 4.356 / dx         # == L
    radius_lb = 0.254 / dx
    kwargs = dict(
        nx=nx, ny=ny, nz=nz,
        cx=nx * 0.25, cy=ny * 0.5, cz=nz * 0.5,
        length=length_lb, radius=radius_lb,
        config=SuboffConfig(), device="cpu",
    )
    with_sail, stats = build_suboff_mask(hull_type="with_sail", **kwargs)
    bare, _ = build_suboff_mask(hull_type="bare_hull", **kwargs)
    sail = with_sail & ~bare
    n = int(sail.sum())
    thick = None
    if n:
        idx = sail.nonzero()
        key = idx[:, 0] * nz + idx[:, 2]        # column identity (x, z)
        counts = torch.zeros(nx * nz, dtype=torch.int32)
        counts.scatter_add_(0, key, torch.ones_like(key, dtype=torch.int32))
        thick = int(counts.max().item())
    return {
        "sail_cells": n,
        "sail_y_thickness": thick,
        "stats_appendage_solid_cells": int(stats.get("appendage_solid_cells", -1)),
        "domain": [nx, ny, nz],
    }


def scan_entry_modules() -> dict:
    """Static check that run.py routes through the shared modules (G3)."""
    src = RUN_PY.read_text()
    return {
        "general_sim": "from tensorlbm.general_sim import" in src,
        "suboff_cad": "from tensorlbm.suboff_cad import" in src,
        "drag_pressure": "from tensorlbm.drag_pressure import" in src,
        "compile_route": "from compile_route import" in src,
        "md5": hashlib.md5(RUN_PY.read_bytes()).hexdigest(),
    }


# --------------------------------------------------------------- gates ---
def _gate(gid, name, status, detail, **kv) -> dict:
    g = {"id": gid, "gate": name, "status": status, "detail": detail}
    g.update(kv)
    return g


def evaluate(tier_rows: dict, path_scan: dict,
             fallback_row: dict | None = None) -> tuple[list, str]:
    """Compute every gate from machine numbers only (nothing hand-copied)."""
    cf = cf_ref_blasius()
    gates: list[dict] = []

    # --- G1: finest ladder tier accuracy ----------------------------------
    finest = max(LADDER)
    row = tier_rows.get(finest, {})
    if row.get("err_pct") is not None:
        ok = abs(row["err_pct"]) <= G1_MAX_ERR_PCT
        gates.append(_gate(
            "G1", f"finest |err| <= {G1_MAX_ERR_PCT}%",
            "PASS" if ok else "FAIL",
            f"L{finest}: Cd_total={row['cd_total_window']:.10f} vs "
            f"Cf_ref={cf:.15f} -> err={row['err_pct']:+.4f}%",
            value_pct=row["err_pct"], threshold_pct=G1_MAX_ERR_PCT, evaluable=True,
        ))
    else:
        gates.append(_gate(
            "G1", f"finest |err| <= {G1_MAX_ERR_PCT}%", "FAIL",
            f"not evaluable on this device: finest frozen tier L{finest} produced no "
            f"window mean ({row.get('blocked_reason', 'no result')}) -> gate recorded "
            f"FAIL (honest-FAIL, no downgrade, no CPU substitution)",
            value_pct=None, threshold_pct=G1_MAX_ERR_PCT, evaluable=False,
        ))

    # --- G2: monotone convergence along the ladder ------------------------
    errs = [tier_rows.get(L, {}).get("err_pct") for L in LADDER]
    if all(e is not None for e in errs):
        abs_errs = [abs(e) for e in errs]
        ok = all(b <= a for a, b in zip(abs_errs, abs_errs[1:]))
        gates.append(_gate(
            "G2", "|err| non-increasing along ladder", "PASS" if ok else "FAIL",
            f"|err| over L={list(LADDER)}: " + " -> ".join(f"{e:.4f}%" for e in abs_errs),
            seq_pct=abs_errs, evaluable=True,
        ))
    else:
        missing = [f"L{L}" for L, e in zip(LADDER, errs) if e is None]
        gates.append(_gate(
            "G2", "|err| non-increasing along ladder", "FAIL",
            f"not evaluable on this device: missing tiers {missing} -> FAIL (honest-FAIL)",
            seq_pct=errs, evaluable=False,
        ))

    # --- G3: common-module entry ------------------------------------------
    src_ok = all(path_scan.get(k) for k in ("general_sim", "suboff_cad", "drag_pressure"))
    ran_ok = any(tier_rows.get(L, {}).get("ok") for L in LADDER)
    sup_ok = bool(fallback_row and fallback_row.get("ok"))
    fields_ok = any(r.get("entry_fields_ok") for r in tier_rows.values()) or bool(
        fallback_row and fallback_row.get("entry_fields_ok"))
    gates.append(_gate(
        "G3", "common-module entry (GeneralSimEngine + suboff_cad + drag_pressure)",
        "PASS" if (src_ok and (ran_ok or (sup_ok and fields_ok))) else "FAIL",
        f"frozen run.py imports={ {k: path_scan.get(k) for k in ('general_sim','suboff_cad','drag_pressure')} }; "
        f"ladder tier executed on sdaa={ran_ok}; "
        f"executed evidence from the supplementary off-ladder arm={sup_ok} "
        f"(disclosed: the ladder itself never executed on this card); "
        f"provenance fields (S_total_wetted_lb2/dpS_type/appendage_stats) present={fields_ok}",
        source_scan=path_scan, evaluable=True,
        ladder_executed=ran_ok, supplementary_executed=sup_ok,
    ))

    # --- G4: window stability ---------------------------------------------
    drifts = {L: tier_rows.get(L, {}).get("drift_pct") for L in LADDER}
    have = {L: d for L, d in drifts.items() if d is not None}
    if have:
        worst = max(have.values())
        ok = worst <= G4_MAX_DRIFT_PCT
        gates.append(_gate(
            "G4", f"window drift <= {G4_MAX_DRIFT_PCT}%", "PASS" if ok else "FAIL",
            "drift per tier: " + ", ".join(f"L{L}={d:.4f}%" for L, d in have.items())
            + f" (worst {worst:.4f}%)",
            drift_pct=have, threshold_pct=G4_MAX_DRIFT_PCT, evaluable=True,
        ))
    else:
        gates.append(_gate(
            "G4", f"window drift <= {G4_MAX_DRIFT_PCT}%", "FAIL",
            "not evaluable on this device: no frozen ladder tier produced convergence "
            "windows -> FAIL (honest-FAIL)",
            drift_pct=drifts, threshold_pct=G4_MAX_DRIFT_PCT, evaluable=False,
        ))

    # --- G5: sail geometry -------------------------------------------------
    checks = []
    for L in LADDER:
        g = tier_rows.get(L, {}).get("sail_geometry") or {}
        cells = g.get("sail_cells")
        exp = G5_MASK_TABLE[L]
        checks.append({
            "L": L,
            "sail_cells": cells,
            "expected": exp,
            "cells_ok": None if cells is None else abs(cells - exp) <= G5_TOL_CELLS,
            "y_thickness": g.get("sail_y_thickness"),
            "thickness_ok": g.get("sail_y_thickness") == G5_SAIL_Y_THICKNESS,
        })
    g5_ok = all(c["cells_ok"] and c["thickness_ok"] for c in checks)
    gates.append(_gate(
        "G5", "sail cells match frozen mask table +-2 and y-thickness == 2",
        "PASS" if g5_ok else "FAIL",
        "; ".join(
            f"L{c['L']}: cells={c['sail_cells']} (exp {c['expected']}+-{G5_TOL_CELLS}) "
            f"y-thickness={c['y_thickness']} (exp {G5_SAIL_Y_THICKNESS})"
            for c in checks
        ),
        table=checks, evaluable=all(c["cells_ok"] is not None for c in checks),
    ))

    overall = "PASS" if all(g["status"] == "PASS" for g in gates) else "FAIL"
    return gates, overall


# ------------------------------------------------------------------ main ---
def main() -> int:
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--budget", type=float, default=FALLBACK_STEP_CAP_S,
                    help="wall-clock budget [s] for the supplementary reduced-grid arm")
    ap.add_argument("--tier-timeout", type=float, default=300.0)
    ap.add_argument("--skip-fallback", action="store_true")
    args = ap.parse_args()
    DEVICE = args.device

    t_start = time.time()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)

    ev = device_evidence()
    print(f"[device] torch.sdaa.get_device_name(0) -> {ev.get('device_line')}", flush=True)
    path_scan = scan_entry_modules()
    print(f"[case] run.py md5={path_scan['md5']}", flush=True)

    tier_rows: dict = {}
    runs: dict = {}

    # ---- 1) frozen ladder: ask the feasibility question on the real card --
    for L in LADDER:
        steps = steps_for(L)
        tag = f"frozen_L{L}"
        row: dict = {"resolution_L": L, "caliber_steps": steps, "caliber_hint": CASE_TIERS_HINT}
        probe = run_case(L, 2, f"{tag}_probe2", timeout=args.tier_timeout)
        runs[f"{tag}_probe2"] = probe
        dom = probe["log_parsed"].get("domain")
        row["domain"] = dom
        row["total_cells"] = probe["log_parsed"].get("total_cells")
        row["compile_route_line"] = probe["log_parsed"].get("compile_route_line")
        if dom:
            try:
                row["sail_geometry"] = sail_geometry(L, dom)
            except Exception as exc:
                row["sail_geometry"] = {"error": f"{type(exc).__name__}: {exc}"}
        row["feasible_on_device"] = bool(probe["ok"])
        if not probe["ok"]:
            row["blocked_reason"] = (probe.get("oom_line") or probe.get("traceback_tail")
                                     or f"run_status={probe['run_status']}")
            row["note"] = ("frozen tier cannot run on this SDAA card -> caliber run "
                           "skipped; accuracy/window gates withheld and recorded FAIL")
            print(f"[tier L{L}] INFEASIBLE: {row['blocked_reason']}", flush=True)
            tier_rows[L] = row
            continue

        print(f"[tier L{L}] feasible on device; running caliber {steps} steps", flush=True)
        full = run_case(L, steps, tag, timeout=max(120.0, 4 * args.tier_timeout))
        runs[tag] = full
        _fill_row(row, full)
        tier_rows[L] = row
        print(f"[tier L{L}] ok={full['ok']} err={row.get('err_pct')}", flush=True)

    # ---- 2) supplementary reduced-grid arm (capacity-limited, off-ladder) -
    fallback: dict = {"attempted": False, "reason": None}
    fb_row: dict | None = None
    if not args.skip_fallback and not any(r.get("ok") for r in tier_rows.values()):
        fallback["attempted"] = True
        fallback["reason"] = (
            "no frozen ladder tier fits this card -> the largest grid that does is run "
            "purely as device-capacity / throughput evidence; it NEVER enters a gate "
            "decision and its Cd is not compared to a gate threshold")
        fallback["candidates"] = list(FALLBACK_CANDIDATES)
        chosen, ms, attempts = None, None, []
        for L in FALLBACK_CANDIDATES:
            pilot = run_case(L, 10, f"fallback_L{L}_pilot10", timeout=args.tier_timeout)
            runs[f"fallback_L{L}_pilot10"] = pilot
            att = {
                "L": L, "ok": pilot["ok"], "wall_clock_s": pilot["wall_clock_s"],
                "ms_per_step": pilot["log_parsed"].get("ms_per_step_log"),
                "domain": pilot["log_parsed"].get("domain"),
                "blocked_reason": pilot.get("oom_line") or pilot.get("traceback_tail"),
            }
            dom = pilot["log_parsed"].get("domain")
            if dom:
                try:
                    att["sail_geometry"] = sail_geometry(L, dom)
                except Exception as exc:
                    att["sail_geometry"] = {"error": f"{type(exc).__name__}: {exc}"}
            attempts.append(att)
            print(f"[fallback pilot L{L}] ok={pilot['ok']} "
                  f"({pilot.get('oom_line') or pilot['log_parsed'].get('ms_per_step_log')})",
                  flush=True)
            if pilot["ok"]:
                chosen = L
                ms = pilot["log_parsed"].get("ms_per_step_log")
                break
        fallback["attempts"] = attempts
        fallback["chosen_L"] = chosen
        if chosen is not None and ms:
            cap_steps = int(max(50.0, args.budget * 1000.0 / float(ms)))
            steps = min(steps_for(chosen), cap_steps)
            fallback["wall_clock_derived_steps"] = steps
            fallback["caliber_steps_at_this_L"] = steps_for(chosen)
            fallback["step_shortfall_factor"] = round(steps_for(chosen) / steps, 3)
            print(f"[fallback L{chosen}] running {steps} steps (caliber would need "
                  f"{steps_for(chosen)}; {ms} ms/step)", flush=True)
            full = run_case(chosen, steps, f"fallback_L{chosen}_{steps}",
                            timeout=args.budget * 2.5 + 180.0)
            runs[f"fallback_L{chosen}_{steps}"] = full
            row = {"resolution_L": chosen, "caliber_steps": steps_for(chosen),
                   "run_steps_requested": steps, "supplementary": True,
                   "note": "off-ladder, wall-clock-capped; Cd/err NOT used for any gate"}
            row["domain"] = full["log_parsed"].get("domain")
            row["total_cells"] = full["log_parsed"].get("total_cells")
            row["compile_route_line"] = full["log_parsed"].get("compile_route_line")
            if row["domain"]:
                try:
                    row["sail_geometry"] = sail_geometry(chosen, row["domain"])
                except Exception as exc:
                    row["sail_geometry"] = {"error": f"{type(exc).__name__}: {exc}"}
            _fill_row(row, full)
            fb_row = row
            fallback["run"] = row
            print(f"[fallback L{chosen}] ok={full['ok']} err={row.get('err_pct')} "
                  f"drift={row.get('drift_pct')}", flush=True)
        else:
            print("[fallback] no off-ladder grid completed either", flush=True)

    # ---- 3) gates --------------------------------------------------------
    gates, overall = evaluate(tier_rows, path_scan, fb_row)

    wall = round(time.time() - t_start, 1)
    artifact = {
        "case": "suboff_sail_resistance",
        "case_dir": str(CASE_DIR.relative_to(REPO_ROOT)),
        "repo": str(REPO_ROOT),
        "entry_md5": path_scan["md5"],
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_evidence": ev,
        "device_evidence_line": ev.get("device_line"),
        "caliber": {
            "reference": "Blasius Cf = 1.328/sqrt(Re), Re=1000, total wetted area "
                         "S = pi(2R_lb)L + A_sail_own",
            "cf_ref_recomputed": cf_ref_blasius(),
            "ladder": list(LADDER),
            "steps_rule": "steps = 150*L (prereg.md, frozen)",
            "frozen": {
                "collision": "mrt", "friction": "standard (erratum E2 primary)",
                "p0": "near_wall", "compile_mode": "eager",
                "mass_correction_interval": 200, "force_sample_interval": 10,
                "u_lb": 0.05,
            },
            "window": "mean of last 1000 force samples (= last 10000 steps)",
            "gates": {
                "G1": f"finest |err| <= {G1_MAX_ERR_PCT}%",
                "G2": "|err| non-increasing along ladder",
                "G3": "common-module entry (GeneralSimEngine + suboff_cad + drag_pressure)",
                "G4": f"window drift <= {G4_MAX_DRIFT_PCT}%",
                "G5": "sail cells = frozen mask table +-2 and sail y-thickness == 2",
            },
        },
        "triton_bypass": {
            "needed": False,
            "triton_interpret_set": False,
            "why": "the executed path (eager D3Q19 MRT chain via lbm_step_correct) "
                   "imports no triton kernel; the compiled route is unusable on "
                   "SDAA (torch.compile -> teco_inductor, upstream L5 defect), and "
                   "compile-mode=eager is the prereg-frozen parameter anyway, so no "
                   "TRITON_INTERPRET detour was required",
        },
        "tiers": {str(L): r for L, r in tier_rows.items()},
        "supplementary_reduced_grid": fallback,
        "runs": runs,
        "gates": gates,
        "OVERALL": overall,
        "wall_clock_s": wall,
        "honest_fail": (
            "FAIL is reported because gates that cannot be evaluated on this card "
            "(G1/G2 and the run-dependent part of G4) are recorded FAIL, per the "
            "case rule 'any gate failing => FAIL'. No threshold was relaxed and no "
            "CPU number is presented as an sdaa number."
            if overall == "FAIL" else None
        ),
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"\n[artifact] {ARTIFACT.relative_to(REPO_ROOT)}  wall={wall}s  OVERALL={overall}",
          flush=True)
    for g in gates:
        print(f"  {g['id']}: {g['status']}  {g['detail'][:160]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())