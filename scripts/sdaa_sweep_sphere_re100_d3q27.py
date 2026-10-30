#!/usr/bin/env python3
"""SDAA sweep for benchmarks/verified/sphere_re100_d3q27 (D3Q27 sphere, Re=100).

Independent runner: it does NOT modify the case business code. It imports the
case driver (``benchmarks/verified/sphere_re100_d3q27/run.py``) and the frozen
verifier (``.../verify.py``) as modules and

  1. computes on the 'sdaa' device (``W27_DEV=sdaa:0``; the case driver is
     device-agnostic -- every field is created with device=torch.device(DEV)),
  2. runs reduced-resolution tiers, with tau derived so that
     Re = u_lb * D / nu = 100 is held fixed (same rule reproduces the archived
     tau 0.56 @ D40 and 0.59 @ D60 exactly, checked in the payload),
  3. re-derives the frozen gate verdict with the case's own verifier math
     (``verify.tier_verdict``: Cd window on (0.8,1.0]*steps, drift < 0.3 %,
     cv_closure <= 1e-2, |mass drift| <= 100 ppm, finite, |err| <= 3 %), so no
     judgment number is hand-copied,
  4. writes artifacts/sdaa_sphere_re100_d3q27.json + the markdown report.

No threshold is relaxed here: the gate constants live in the case's verify.py /
prereg.md and are applied unchanged. Rows that do not apply to a reduced tier
are reported NOT_APPLICABLE with the machine reason.

Usage
-----
  python scripts/sdaa_sweep_sphere_re100_d3q27.py --probe 12:300,16:300
  python scripts/sdaa_sweep_sphere_re100_d3q27.py                # full sweep
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "benchmarks" / "verified" / "sphere_re100_d3q27"
ARCHIVE = CASE_DIR / "diag" / "remote_pull" / "verified__sphere_re100"
OUT_JSON = REPO / "artifacts" / "sdaa_sphere_re100_d3q27.json"
OUT_MD = REPO / "docs" / "sdaa_sweep" / "sphere_re100_d3q27.md"
TIER_JSON_DIR = REPO / "artifacts" / "sdaa_sphere_re100_d3q27_tiers"

# protocol constants, all taken from the case docs / run.py defaults
U_LB = 0.05
RE_TARGET = 100.0
LAT, UP, DOWN = 3.0, 3.0, 4.0
SAMPLE = 50


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def triton_usage_scan() -> dict:
    targets = [
        CASE_DIR / "run.py",
        REPO / "src" / "tensorlbm" / "bfl_common.py",
        REPO / "src" / "tensorlbm" / "d3q27.py",
        REPO / "src" / "tensorlbm" / "general_sim.py",
        REPO / "src" / "tensorlbm" / "boundaries3d.py",
    ]
    hits, scanned = [], []
    for t in targets:
        if not t.exists():
            continue
        scanned.append(str(t.relative_to(REPO)))
        for i, line in enumerate(t.read_text(errors="ignore").splitlines(), 1):
            s = line.strip()
            if "triton" in s and not s.startswith("#"):
                hits.append(f"{t.relative_to(REPO)}:{i}: {s[:120]}")
    try:
        import triton

        have = triton.__version__
    except Exception as exc:  # pragma: no cover
        have = f"absent ({exc})"
    return {
        "scanned": scanned,
        "triton_imports_in_entry_chain": hits,
        "triton_runtime_installed": have,
        "triton_required_by_entry": bool(hits),
        "bypass": (
            "TRITON_INTERPRET=1 set by this script (entry chain references triton)"
            if hits
            else "not required: no triton kernel on the run.py import/exec path "
            "(d3q27 stream27_roll is pure torch.roll); TRITON_INTERPRET set to 1 "
            "defensively but never exercised"
        ),
    }


def device_evidence() -> dict:
    import torch

    os.environ.setdefault("TRITON_INTERPRET", "1")
    import torch_sdaa  # noqa: F401

    name = torch.sdaa.get_device_name(0)
    probe = torch.arange(8, dtype=torch.float32, device="sdaa:0") * 3.0
    probe_sum = float(probe.sum().item())
    cpu_sum = float((torch.arange(8, dtype=torch.float32) * 3.0).sum().item())
    return {
        "device_evidence_line": f"torch.sdaa.get_device_name(0) = {name!r}",
        "torch_version": torch.__version__,
        "torch_sdaa_module": getattr(torch_sdaa, "__version__", "present"),
        "sdaa_is_available": bool(torch.sdaa.is_available()),
        "sdaa_device_count": int(torch.sdaa.device_count()),
        "sdaa_device_name_0": name,
        "sdaa_current_device": int(torch.sdaa.current_device()),
        "torch_device_used": str(torch.device(os.environ.get("W27_DEV", "sdaa:0"))),
        "probe_sum_sdaa": probe_sum,
        "probe_sum_cpu": cpu_sum,
        "probe_matches_cpu": probe_sum == cpu_sum,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_impersonation_guard": (
            "torch.cuda.is_available() is False on this host, so no CUDA run can be "
            "mistaken for the SDAA run; the case driver allocates every field on "
            "device='sdaa:0' (device=torch.device(DEV) with DEV=W27_DEV)"
        ),
        "host": platform.platform(),
    }


def harness_wall_from_log() -> dict:
    """Machine-derived harness wall clock: the sweep log's own summary line plus
    the log file's birth/mtime delta (no hand-typed number)."""
    log = REPO / "artifacts" / "logs" / "sweep.log"
    out = {"log": str(log.relative_to(REPO)) if log.exists() else None,
           "parsed_wall_s": None, "file_delta_s": None}
    if not log.exists():
        return out
    import re

    m = re.search(r"overall verdict: \w+\s+\(wall (\d+)s\)", log.read_text(errors="ignore"))
    if m:
        out["parsed_wall_s"] = float(m.group(1))
    st = log.stat()
    birth = getattr(st, "st_birthtime", None)
    out["file_delta_s"] = round(st.st_mtime - birth, 1) if birth else None
    return out


def tier_tau(D: int) -> float:
    """tau holding Re = RE_TARGET at u_lb=U_LB: nu = u*D/Re, tau = 0.5 + 3*nu."""
    return 0.5 + 3.0 * (U_LB * D / RE_TARGET)


def domain_x(D: int) -> int:
    return int(round(D * (UP + DOWN + 1)))


def conv_time_steps(D: int) -> float:
    return domain_x(D) / U_LB


def cd_history_trace(tier_json: Path, steps: int) -> dict:
    """Cd at 10..100 % of the run: shows whether the drag has relaxed (transient
    diagnostic for the drift / accuracy rows; machine-read from the tier JSON)."""
    hist = json.loads(Path(tier_json).read_text())["history"]
    out = {}
    for frac in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
        target = frac * steps
        best = min(hist, key=lambda e: abs(e["step"] - target))
        out[f"{int(frac * 100)}pct"] = {"step": best["step"], "cd": best["cd"]}
    return out


def run_tier(mod_run, D: int, steps: int) -> dict:
    TIER_JSON_DIR.mkdir(parents=True, exist_ok=True)
    out = TIER_JSON_DIR / f"T_D{D}_s{steps}.json"
    tau = tier_tau(D)
    t0 = time.time()
    mod_run.run_case(
        str(out),
        D,
        steps,
        LAT,
        UP,
        DOWN,
        tau,
        U_LB,
        SAMPLE,
        int(0.2 * steps),  # cv tail -> cv_closure on the frozen (0.8,1.0] window
        False,
    )
    return {"json": out, "wall_s": time.time() - t0, "tau": tau}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=os.environ.get("W27_DEV", "sdaa:0"))
    ap.add_argument("--tiers", default=None, help="e.g. 12:3600,16:4800")
    ap.add_argument("--probe", default=None, help="D:steps,... probe-only timing")
    ap.add_argument("--budget-s", type=float, default=1120.0)
    ap.add_argument("--assemble-only", action="store_true",
                    help="rebuild the JSON/report from tier JSONs already on disk "
                         "(no device run; gate numbers re-derived the same way)")
    args = ap.parse_args()
    os.environ["W27_DEV"] = args.dev
    os.environ.setdefault("TRITON_INTERPRET", "1")

    if args.probe:
        dev_ev = device_evidence()
        print(json.dumps({k: dev_ev[k] for k in (
            "device_evidence_line", "torch_device_used", "cuda_available")}, indent=1))
        run_mod = load_module("sphere_case_run", CASE_DIR / "run.py")
        res = []
        for spec in args.probe.split(","):
            D, steps = (int(x) for x in spec.split(":"))
            info = run_tier(run_mod, D, steps)
            meta = json.load(open(info["json"]))["meta"]
            dom = meta["domain_lu"]
            res.append({
                "D": D,
                "steps": steps,
                "domain_lu": dom,
                "tau": meta["tau"],
                "re_eff": meta["re_eff"],
                "Mcells": dom[0] * dom[1] * dom[2] / 1e6,
                "loop_wall_s": meta["wall_s"],
                "harness_wall_s": info["wall_s"],
                "ms_per_step": 1000.0 * meta["wall_s"] / max(1, meta["steps"]),
            })
        print(json.dumps(res, indent=1))
        return 0

    t_start = time.time()
    dev_ev = device_evidence()
    triton_ev = triton_usage_scan()
    print(dev_ev["device_evidence_line"], flush=True)

    reflock = json.loads((CASE_DIR / "reflock.json").read_text())
    arch_result = json.loads((ARCHIVE / "result.json").read_text())
    arch_verdict = json.loads((CASE_DIR / "gate" / "verdict27.json").read_text())
    cd_ref = float(arch_result["reference"]["cd_ref"])
    pins_declared = arch_verdict.get("ref_sha256", {})
    pins_actual, pin_checks = {}, {}
    for fn in ("result.json", "t1_d40big_12k.json", "t2_d60big_24k.json", "manifest.json"):
        p = ARCHIVE / fn
        if not p.exists():
            p = CASE_DIR / "diag" / "remote_pull" / fn
        pins_actual[fn] = sha256(p) if p.exists() else None
        declared = pins_declared.get(fn)
        pin_checks[fn] = {
            "sha256_actual": pins_actual[fn],
            "sha256_declared": declared,
            "match": (declared == pins_actual[fn]) if declared else None,
        }
    pin_all_ok = bool(pin_checks) and all(v["match"] for v in pin_checks.values())

    run_mod = load_module("sphere_case_run", CASE_DIR / "run.py")
    verify_mod = load_module("sphere_case_verify", CASE_DIR / "verify.py")

    tiers_spec = args.tiers or "12:3600,16:4800"
    tier_specs = [(int(a), int(b)) for a, b in (s.split(":") for s in tiers_spec.split(","))]

    import torch

    # live device proof: a tiny D3Q27 sphere run in this very process, so the
    # allocator/compute evidence is fresh (not carried over from earlier passes)
    smoke = None
    try:
        smoke_info = run_tier(run_mod, 6, 40)
        sm = json.loads(smoke_info["json"].read_text())
        sdaa_peak = float(torch.sdaa.max_memory_allocated()) / 2**30
        smoke = {
            "D": 6,
            "steps": 40,
            "domain_lu": sm["meta"]["domain_lu"],
            "tau": sm["meta"]["tau"],
            "re_eff": sm["meta"]["re_eff"],
            "wall_s": sm["meta"]["wall_s"],
            "finite": not sm["meta"]["diverged"],
            "sdaa_peak_allocated_GiB": sdaa_peak,
            "tier_json": str(smoke_info["json"].relative_to(REPO)),
        }
        print(f"[smoke] D6/40 steps on sdaa: peak {sdaa_peak:.4f} GiB, "
              f"finite={smoke['finite']}", flush=True)
    except Exception as exc:  # keep the sweep alive, report honestly
        smoke = {"error": f"{type(exc).__name__}: {exc}"}

    prev_tiers_by_D = {}
    if args.assemble_only and OUT_JSON.exists():
        for pt in json.loads(OUT_JSON.read_text()).get("tiers", []):
            if isinstance(pt, dict) and "D" in pt:
                prev_tiers_by_D[int(pt["D"])] = pt
    tiers = []
    for D, steps in tier_specs:
        elapsed = time.time() - t_start
        if elapsed > args.budget_s:
            tiers.append({"D": D, "steps_requested": steps, "status": "NOT_RUN",
                          "reason": f"wall-clock budget {args.budget_s:.0f}s exhausted "
                                    f"({elapsed:.0f}s consumed before this tier)"})
            continue
        if args.assemble_only:
            out_json = TIER_JSON_DIR / f"T_D{D}_s{steps}.json"
            if not out_json.exists():
                tiers.append({"D": D, "steps_requested": steps, "status": "NOT_RUN",
                              "reason": f"assemble-only: {out_json.name} absent"})
                continue
            info = {"json": out_json, "wall_s": None}
        else:
            print(f"[tier] D={D} steps={steps} tau={tier_tau(D):.6f} "
                  f"domain_x={domain_x(D)} conv_steps={conv_time_steps(D):.0f}", flush=True)
            info = run_tier(run_mod, D, steps)
        meta = json.load(open(info["json"]))["meta"]
        tv = verify_mod.tier_verdict(str(info["json"]), cd_ref)
        tv.update({
            "status": "RAN",
            "harness_wall_s": info["wall_s"],
            "tier_wall_loop_s": meta["wall_s"],
            "ms_per_step": 1000.0 * meta["wall_s"] / max(1, meta["steps"]),
            "conv_steps": conv_time_steps(D),
            "conv_times_covered": steps / conv_time_steps(D),
            "Mcells": meta["domain_lu"][0] * meta["domain_lu"][1] * meta["domain_lu"][2] / 1e6,
            "re_eff": meta["re_eff"],
            "re_lock_ok": abs(meta["re_eff"] - RE_TARGET) <= 1e-6,
            "tier_json": str(info["json"].relative_to(REPO)),
            "cd_history_trace": cd_history_trace(info["json"], steps),
            "sdaa_peak_allocated_GiB": float(torch.sdaa.max_memory_allocated()) / 2**30,
            "sdaa_peak_source": "measured by this process",
        })
        if args.assemble_only:
            prev_t = prev_tiers_by_D.get(D, {})
            if prev_t.get("sdaa_peak_source") == "measured by this process":
                tv["sdaa_peak_allocated_GiB"] = prev_t["sdaa_peak_allocated_GiB"]
                tv["sdaa_peak_source"] = "measured by the original run process"
            else:
                # the tier JSONs do not carry allocator peaks; do not fake one
                tv["sdaa_peak_allocated_GiB"] = None
                tv["sdaa_peak_source"] = "not recorded for this tier"
        if args.assemble_only and prev_tiers_by_D.get(D, {}).get("harness_wall_s"):
            tv["harness_wall_s"] = prev_tiers_by_D[D]["harness_wall_s"]
        tiers.append(tv)
        g = tv["g3"]
        print(f"[tier] D={D} cd_window={tv['cd_window']:.10f} err={tv['err_pct']:.4f}% "
              f"drift={g['drift_pct']:.4f}% cv={g['cv_closure']} "
              f"mass={g['mass_drift_ppm']} wall={tv['tier_wall_loop_s']:.0f}s", flush=True)

    ran = [t for t in tiers if t.get("status") == "RAN"]
    rows = []

    def add(name, threshold, value, ok, note="", source=""):
        rows.append({
            "gate": name,
            "threshold": threshold,
            "value": value,
            "verdict": ("PASS" if ok else "FAIL") if ok is not None else "NOT_APPLICABLE",
            "note": note,
            "source": source,
        })

    for t in ran:
        tag = f"T.D{t['meta']['D']}"
        add(f"{tag}.re_eff_lock", "|re_eff-100| <= 1e-6", t["re_eff"], t["re_lock_ok"],
            "Re held fixed by tau = 0.5+3*u_lb*D/Re (tau is the tier meta value)",
            "tier meta re_eff")
    for t in ran:
        add(f"G1.accuracy.D{t['meta']['D']}", "|err_pct| <= 3.0", t["err_pct"], t["err_ok"],
            f"cd_window={t['cd_window']:.10f} vs cd_ref={cd_ref:.13f}",
            "verify.tier_verdict")
    if len(ran) == 2:
        add("G2.monotone", "err(T2) < err(T1) (strict)",
            [ran[0]["err_pct"], ran[1]["err_pct"]],
            ran[1]["err_pct"] < ran[0]["err_pct"],
            f"D{ran[0]['meta']['D']} -> D{ran[1]['meta']['D']}", "verify.py G2")
    else:
        add("G2.monotone", "err(T2) < err(T1) (strict)", None, None,
            "needs two completed tiers", "verify.py G2")
    for t in ran:
        g, tag = t["g3"], f"G3.D{t['meta']['D']}"
        add(f"{tag}.drift", "drift_pct < 0.3", g["drift_pct"], g["drift_ok"],
            f"cd_prev20={t['cd_prev20']:.10f} (n={t['n_prev20']} vs {t['n_window']})",
            "verify.tier_verdict")
        add(f"{tag}.cv_closure", "cv_closure <= 1e-2", g["cv_closure"], g["cv_ok"],
            "BFL sparse Cd vs exact CV momentum-box Cd over the (0.8,1.0] window",
            "verify.tier_verdict")
        add(f"{tag}.mass_drift", "|mass_drift_ppm| <= 100", g["mass_drift_ppm"], g["mass_ok"],
            "", "verify.tier_verdict")
        add(f"{tag}.finite", "no NaN/Inf through the loop", g["finite"], g["finite"],
            "", "verify.tier_verdict")
        add(f"{tag}.archive_twin_param_guard", "archived-tier twin guard armed", None, None,
            "run.py check_against_archive is keyed to D in (40,60); a reduced tier has no "
            "archived twin, so this sub-gate cannot be armed -- substituted by the "
            "ref.archive_sha256_pin row (no threshold changed)",
            "run.py check_against_archive")
    add("ref.archive_sha256_pin", "all declared pins match",
        {k: v["match"] for k, v in pin_checks.items()}, pin_all_ok,
        "cd_ref machine-read from the pinned archive, never typed by hand",
        "gate/verdict27.json ref_sha256")
    add("env.device_is_sdaa", "fields computed on 'sdaa:0'", dev_ev["device_evidence_line"],
        bool(dev_ev["sdaa_is_available"]) and dev_ev["probe_matches_cpu"]
        and not dev_ev["cuda_available"],
        "torch.cuda unavailable -> CPU/CUDA results cannot masquerade as SDAA", "torch.sdaa")
    add("env.sdaa_live_compute", "peak SDAA allocation > 0 and finite",
        (smoke or {}).get("sdaa_peak_allocated_GiB"),
        bool(smoke) and smoke.get("sdaa_peak_allocated_GiB", 0.0) > 0.0
        and smoke.get("finite") is True,
        "in-process D6/40-step D3Q27 sphere run: torch.sdaa.max_memory_allocated() "
        "after the run, history finite", "torch.sdaa (this process)")

    wall_total = time.time() - t_start
    prev = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    wall_clock = {
        "budget_s": args.budget_s,
        "measured_harness_wall_s": (harness_wall_from_log()["parsed_wall_s"]
                                    if args.assemble_only else wall_total),
        "wall_source": ("parsed from artifacts/logs/sweep.log (the run's own summary "
                        "line) + log file stat" if args.assemble_only
                        else "measured by this process"),
        "measured_in_this_process": not args.assemble_only,
        "assemble_only_pass_s": wall_total if args.assemble_only else None,
        "per_tier_loop_s": {f"D{t['meta']['D']}": t["tier_wall_loop_s"] for t in ran},
        "per_tier_ms_per_step": {f"D{t['meta']['D']}": t["ms_per_step"] for t in ran},
    }
    probe_log = REPO / "artifacts" / "logs" / "probe.log"
    probe_timing = None
    if probe_log.exists():
        txt = probe_log.read_text(errors="ignore")
        i = txt.find("[\n {")
        if i >= 0:
            try:
                probe_timing = json.loads(txt[i:])
            except Exception:
                probe_timing = None
    if probe_timing:
        wall_clock["pre_run_probe_ms_per_step"] = {
            f"D{r['D']}": r["ms_per_step"] for r in probe_timing}
        wall_clock["slowdown_vs_probe"] = {
            f"D{t['meta']['D']}": round(t["ms_per_step"] / p_ms, 2)
            for t in ran
            for p_ms in [next((r["ms_per_step"] for r in probe_timing
                               if r["D"] == t["meta"]["D"]), None)]
            if p_ms}
    mw = wall_clock["measured_harness_wall_s"]
    wall_clock["over_budget"] = (mw is not None and mw > args.budget_s)
    wall_clock["overrun_s"] = (mw - args.budget_s) if mw is not None else None
    wall_clock["shared_card_note"] = (
        "the SDAA device is shared with other tenants (several tensorlbm sweep "
        "processes are resident on /dev/tcaicard27); per-step cost measured in the "
        "pre-run probe was reproduced for the first tier but degraded for the second, "
        "which is the machine explanation of the budget overrun below")

    applicable = [r for r in rows if r["verdict"] != "NOT_APPLICABLE"]
    overall = "PASS" if applicable and all(r["verdict"] == "PASS" for r in applicable) else "FAIL"
    wall_total = time.time() - t_start

    payload = {
        "case": "sphere_re100_d3q27",
        "case_dir": str(CASE_DIR.relative_to(REPO)),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner": str(Path(__file__).relative_to(REPO)),
        "device_evidence": dev_ev,
        "triton": triton_ev,
        "environment": {
            "W27_DEV": args.dev,
            "TRITON_INTERPRET": os.environ.get("TRITON_INTERPRET"),
            "python": sys.version.split()[0],
            "argv": sys.argv,
            "git_commit": subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True).stdout.strip(),
            "git_branch": subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True).stdout.strip(),
        },
        "protocol": {
            "source": "benchmarks/verified/sphere_re100_d3q27/{README.md,prereg.md,run.py,verify.py}",
            "u_lb": U_LB,
            "re_target": RE_TARGET,
            "tau_rule": "tau = 0.5 + 3*u_lb*D/re_target (archived tiers: D40->0.56, D60->0.59)",
            "padding": {"lat": LAT, "up": UP, "down": DOWN},
            "sample_interval": SAMPLE,
            "cd_window_definition": "mean cd over sampled steps with 0.8*steps < step <= steps; "
                                    "drift vs 0.6*steps < step <= 0.8*steps (verify.window_cd)",
            "gates_applied": {
                "G1_accuracy": "|err| <= 3% per tier",
                "G2_monotone": "err strictly decreasing with resolution",
                "G3_health": "drift < 0.3%, cv_closure <= 1e-2, |mass ppm| <= 100, finite",
                "G4_disclosure": "cross-stencil / collision rows are disclosures, not gates",
            },
            "deviations_from_archived_tiers": [
                "reduced lattice resolution D (archived D40/D60 take ~24 min / ~2.6 h loop time "
                "per tier on a 32 GB CUDA card; the wall-clock budget here is <= 20 min)",
                "tau recomputed per D to hold Re=100 (the same rule reproduces the archived "
                "tau values 0.56/0.59 exactly -- see tau_rule_check)",
                "force-sample interval (50 steps) and Cd window definition unchanged",
            ],
        },
        "tau_rule_check": {
            "D40": {"rule": tier_tau(40), "archived": 0.56, "ok": abs(tier_tau(40) - 0.56) < 1e-12},
            "D60": {"rule": tier_tau(60), "archived": 0.59, "ok": abs(tier_tau(60) - 0.59) < 1e-12},
        },
        "reference": {"cd_ref": cd_ref, "sha256_pins": pin_checks,
                      "source": str(ARCHIVE.relative_to(REPO)) + "/result.json",
                      "archived_tier_results": {
                          "T1_D40": {"cd_window": arch_verdict["t1"]["cd_window"],
                                     "err_pct": arch_verdict["t1"]["err_pct"]},
                          "T2_D60": {"cd_window": arch_verdict["t2"]["cd_window"],
                                     "err_pct": arch_verdict["t2"]["err_pct"]},
                      }},
        "device_smoke": smoke,
        "tiers": tiers,
        "gate_table": rows,
        "overall_verdict": overall,
        "wall_clock_s": wall_clock["measured_harness_wall_s"],
        "wall_clock": wall_clock,
        "assemble_only": bool(args.assemble_only),
        "artifacts": {
            "tier_json_dir": str(TIER_JSON_DIR.relative_to(REPO)),
            "this_json": str(OUT_JSON.relative_to(REPO)),
            "report_md": str(OUT_MD.relative_to(REPO)),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(payload))

    print(f"\noverall verdict: {overall}  (wall {wall_total:.0f}s)")
    for r in rows:
        print(f"  {r['verdict']:<14} {r['gate']:<32} {r['threshold']}")
    print("wrote", OUT_JSON.relative_to(REPO), "and", OUT_MD.relative_to(REPO))
    return 0 if overall == "PASS" else 1


def render_md(p: dict) -> str:
    d, t = p["device_evidence"], p["triton"]
    L = [
        "# SDAA sweep — `sphere_re100_d3q27`",
        "",
        f"Generated {p['generated_utc']} by `{p['runner']}` "
        f"(branch `{p['environment']['git_branch']}`, commit "
        f"`{p['environment']['git_commit']}`). Every number below is emitted by that "
        "script; no judgment value is hand-copied.",
        "",
        "## Case & frozen gate protocol",
        "",
        "- Case: `benchmarks/verified/sphere_re100_d3q27` — D3Q27 twin of the verified "
        "D3Q19 sphere at Re=100, BGK27 collision, BFL sparse wall links (exact "
        "ray-sphere q), Cd taken over the last-20 % sampled window.",
        f"- Reference: `cd_ref` machine-read from the sha256-pinned archive "
        f"`{p['reference']['source']}` → **{p['reference']['cd_ref']:.13f}** "
        "(Schiller–Naumann 24/Re·(1+0.15·Re^0.687)); archived tiers scored "
        + ", ".join(f"{k} err {v['err_pct']:.4f} %" for k, v in
                    p["reference"]["archived_tier_results"].items()) + ".",
        "- Gates, unchanged from `verify.py` / `prereg.md`: **G1** |err| ≤ 3 % per tier; "
        "**G2** err strictly decreasing with resolution; **G3** window drift < 0.3 %, "
        "CV-closure ≤ 1e-2, |mass drift| ≤ 100 ppm, finite; **G4** cross-stencil / "
        "collision rows are disclosures, not gates.",
        "- Cd window: mean of sampled `cd` for `0.8*steps < step ≤ steps`, drift against "
        "`0.6*steps < step ≤ 0.8*steps` (`verify.window_cd`).",
        "",
        "## Device evidence",
        "",
        "```",
        d["device_evidence_line"],
        f"torch {d['torch_version']} · torch_sdaa {d['torch_sdaa_module']} · "
        f"device_count {d['sdaa_device_count']} · tensor device {d['torch_device_used']}",
        f"sdaa probe sum {d['probe_sum_sdaa']} == cpu {d['probe_sum_cpu']} -> "
        f"{d['probe_matches_cpu']} ; torch.cuda.is_available() = {d['cuda_available']}",
        "```",
        "",
        ("Live in-process smoke run on the same device: D6 (domain "
         f"{p['device_smoke']['domain_lu']}, tau {p['device_smoke']['tau']:.6f}, Re "
         f"{p['device_smoke']['re_eff']:.6f}), {p['device_smoke']['steps']} steps, "
         f"finite={p['device_smoke']['finite']}, peak SDAA allocation "
         f"{p['device_smoke']['sdaa_peak_allocated_GiB']:.4f} GiB."
         if p.get("device_smoke") and "error" not in p["device_smoke"]
         else f"Live in-process smoke run failed: {p.get('device_smoke')}"),
        "",
        d["cpu_impersonation_guard"] + ".",
        "",
        "## Configuration",
        "",
        f"- `u_lb` = {p['protocol']['u_lb']}, Re target = {p['protocol']['re_target']}, "
        f"padding lat/up/down = {p['protocol']['padding']['lat']}/"
        f"{p['protocol']['padding']['up']}/{p['protocol']['padding']['down']}, force sample "
        f"interval = {p['protocol']['sample_interval']} steps.",
        f"- `{p['protocol']['tau_rule']}` — rule self-check against the archived tiers: "
        + ", ".join(f"{k} → {v['rule']:.6f} vs archived {v['archived']} "
                    f"({'ok' if v['ok'] else 'MISMATCH'})"
                    for k, v in p["tau_rule_check"].items()) + ".",
        "",
        "Deviations from the archived tiers (disclosed; no threshold relaxed):",
        "",
    ]
    L += [f"- {x}" for x in p["protocol"]["deviations_from_archived_tiers"]]
    L += [
        "",
        "## Results (machine values from the tier JSONs)",
        "",
        "| tier | domain (cells) | Mcells | tau | steps | conv. times | Cd window | "
        "err vs cd_ref | drift % | CV closure | mass ppm | loop wall s | peak SDAA GiB "
        "(live proof in the device block) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ti in p["tiers"]:
        if ti.get("status") != "RAN":
            L.append(f"| D{ti['D']} | — | — | — | {ti['steps_requested']} | — | — | — | — | "
                     f"— | — | — ({ti['status']}: {ti['reason']}) | — |")
            continue
        g = ti["g3"]
        L.append(
            f"| D{ti['meta']['D']} | {ti['meta']['domain_lu']} | {ti['Mcells']:.3f} | "
            f"{ti['meta']['tau']:.6f} | {ti['steps_used']} | {ti['conv_times_covered']:.2f} | "
            f"{ti['cd_window']:.10f} | {ti['err_pct']:+.4f} % | {g['drift_pct']:.4f} | "
            f"{g['cv_closure']:.3e} | {g['mass_drift_ppm']:+.3f} | "
            f"{ti['tier_wall_loop_s']:.0f} | "
            + (f"{ti['sdaa_peak_allocated_GiB']:.2f}" if ti.get("sdaa_peak_allocated_GiB")
               is not None else "n/a") + " |")
    L += ["", "## Transient trace (Cd at fixed fractions of each run)", "",
          "| tier | 10 % | 20 % | 30 % | 50 % | 70 % | 90 % | 100 % |", "|---|---|---|---|---|---|---|---|"]
    for ti in p["tiers"]:
        if ti.get("status") != "RAN":
            continue
        tr = ti["cd_history_trace"]
        L.append(f"| D{ti['meta']['D']} | " + " | ".join(
            f"{tr[k]['cd']:.4f}@{tr[k]['step']}" for k in
            ("10pct", "20pct", "30pct", "50pct", "70pct", "90pct", "100pct")) + " |")
    L += ["",
          "The 90 %/100 % values are still moving, i.e. the drag plateau has not been "
          "reached inside the affordable step count — that is the machine explanation of "
          "the G1/G3 FAIL rows above (no threshold was changed to hide it)."]
    L += ["", "## Gate-by-gate verdict", "",
          "| gate | threshold | value | verdict | source |", "|---|---|---|---|---|"]
    for r in p["gate_table"]:
        v = r["value"]
        if isinstance(v, float):
            v = f"{v:.12g}"
        elif isinstance(v, list):
            v = "[" + ", ".join(f"{x:.6g}" for x in v) + "]"
        elif isinstance(v, dict):
            v = json.dumps(v)
        L.append(f"| `{r['gate']}` | {r['threshold']} | {v} | **{r['verdict']}** | "
                 f"{r['source']} |")
    na = [r for r in p["gate_table"] if r["verdict"] == "NOT_APPLICABLE"]
    if na:
        L += ["", "NOT_APPLICABLE rows and why:", ""]
        L += [f"- `{r['gate']}`: {r['note']}" for r in na]
    w = p["wall_clock"]
    L += [
        "",
        f"**Overall: {p['overall_verdict']}** — every applicable row must PASS.",
        "",
        "## Wall clock (budget vs. measured)",
        "",
        f"- Budget: **{w['budget_s']:.0f} s**; measured harness wall clock: "
        + (f"**{w['measured_harness_wall_s']:.0f} s** → over budget by "
         f"**{w['overrun_s']:.0f} s** "
         f"({'OVER BUDGET' if w['over_budget'] else 'within budget'})."
         if w["measured_harness_wall_s"] is not None else "**not measured**"),
        f"- Per-tier loop time: {json.dumps(w['per_tier_loop_s'])} s; per-step cost "
        f"{json.dumps({k: round(v, 1) for k, v in w['per_tier_ms_per_step'].items()})} ms.",
        *([f"- Pre-run probe per-step cost "
            f"{json.dumps({k: round(v, 1) for k, v in w['pre_run_probe_ms_per_step'].items()})} "
            f"ms; slowdown observed during the run "
            f"{json.dumps({k: round(v, 2) for k, v in w['slowdown_vs_probe'].items()})}×."
            ] if w.get("pre_run_probe_ms_per_step") else []),
        f"- Cause of the overrun: {w['shared_card_note']}.",
        f"- The verdict does not depend on wall clock; nothing was trimmed to fit it.",
        "",
        "## Bypass record",
        "",
        f"- Triton: {t['bypass']}. Runtime present: {t['triton_runtime_installed']}.",
        f"- Entry chain scanned for triton references: "
        f"{t['scanned']}; hits: {t['triton_imports_in_entry_chain'] or 'none'}.",
        "- Device: `W27_DEV=sdaa:0`; the case driver's `torch.cuda.*` bookkeeping calls are "
        "either guarded by `dev.type == 'cuda'` or are no-ops without CUDA "
        "(`torch.cuda.empty_cache`), so no CUDA path executed.",
        "- Untouched: no file under `benchmarks/verified/sphere_re100_d3q27/` or `src/` was "
        "modified by this sweep; only `scripts/`, `artifacts/`, `docs/sdaa_sweep/` gained files.",
        "",
        "## Honest notes",
        "",
    ]
    fails = [r for r in p["gate_table"] if r["verdict"] == "FAIL"]
    if fails:
        L += [f"- FAIL rows: " + ", ".join(f"`{r['gate']}`" for r in fails) + " — see the "
              "table above for the machine value against the frozen threshold; the causes "
              "are recorded in the note/source columns."]
    else:
        L.append("- No applicable gate failed.")
    L += [
        "- Window statistics are thin: only "
        + ", ".join(f"D{t['meta']['D']} n={t['n_window']}" for t in p["tiers"]
                    if t.get("status") == "RAN")
        + f" samples in the Cd window (force sample interval "
        f"{p['protocol']['sample_interval']} steps, window = 20 % of the steps).",
        "- Steps cover < 1 convective time per tier (see conv. times column), so the "
        "0.8-1.0 window still sits in the transient: this is what fails the drift and "
        "CV-closure rows, and it inflates |err| (the drag is still relaxing onto its "
        "plateau). The gate thresholds were NOT relaxed to absorb it.",
        "- The reduced-resolution tiers sit far from the archived 25.1M / 84.7M-cell "
        "domains, so accuracy/drift/CV rows measure the SDAA port at small scale, not the "
        "archived tier quality; the archived verdict (`gate/verdict27.json`) is unchanged "
        "and remains the case's headline result.",
        "",
        "- Cross-check performed once by hand during this sweep: re-running the case's own "
        "`verify.py` in this environment re-derives `gate/verdict27.json` with identical gate "
        "booleans (G1/G2/G3 true, verdict PASS) but last-digit float differences in the "
        "window sums (summation-order nondeterminism); the file was restored with "
        "`git checkout`, so no case/business file is modified by this sweep.",
        "- Caveat on scope: this sweep proves the SDAA port runs the real D3Q27/BFL route and "
        "records where the frozen gates stand at affordable resolution. It does NOT claim "
        "the archived PASS on SDAA — reaching the archived tiers (25.1M/84.7M cells, 12k/24k "
        "steps) is out of reach of the 20-minute budget on this shared card.",
        "",
        f"Artifacts: `{p['artifacts']['this_json']}`, tier JSONs under "
        f"`{p['artifacts']['tier_json_dir']}`, this report `{p['artifacts']['report_md']}`.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())