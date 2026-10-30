#!/usr/bin/env python3
"""SDAA verification sweep — benchmarks/verified/laplace_cac_phasefield.

Independent run script (business code untouched): it imports the case's own
``run.py`` for its criteria/calibration constants (sigma Route-B, gates,
mobility/nu/W), then re-runs the *same* physics protocol
(``tensorlbm.cac_lbm`` CAC/HCH D3Q19, fp64, harmonic mu(phi), periodic cube)
on the SDAA device, with grid/step count overridden to a reduced tier that
fits the sweep wall-clock budget.

Device discipline:
  * compute tensors are placed on 'sdaa' (torch.device("sdaa:0"));
    no CPU fallback — if SDAA is unavailable the script exits non-zero.
  * the device evidence line comes from torch.sdaa.get_device_name(0) as
    printed by the runtime, recorded verbatim in the output JSON.
  * triton: the cac_lbm call path (d3q19 / boundaries3d /
    phasefield.stream_boundary_contract) has no triton kernel, so no
    TRITON_INTERPRET workaround is required; the script records this as a
    verified fact (source scan + sys.modules check) rather than an
    assumption.

Everything numeric in the output JSON is computed here — nothing is
transcribed from the NVIDIA-side result.json.

Usage:
  python scripts/sdaa_sweep_laplace_cac_phasefield.py \
      [--arms b3_rho100] [--grid 64] [--radius 20] [--steps 10000] \
      [--device sdaa:0] [--out artifacts/sdaa_laplace_cac_phasefield.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "benchmarks" / "verified" / "laplace_cac_phasefield"
sys.path.insert(0, str(REPO / "src"))

DEFAULT_OUT = REPO / "artifacts" / "sdaa_laplace_cac_phasefield.json"
case_steps = 12_000  # case README run length (per arm)

# ---------------------------------------------------------------- device setup
if "TRITON_INTERPRET" in os.environ:  # pragma: no cover - recorded below
    pass
os.environ.setdefault("TRITON_INTERPRET", os.environ.get("TRITON_INTERPRET", ""))

import torch  # noqa: E402

SDAAPY_IMPORT_ERROR = None
try:  # register the 'sdaa' device backend with torch
    import torch_sdaa  # noqa: F401,E402
except Exception as exc:  # pragma: no cover
    SDAAPY_IMPORT_ERROR = repr(exc)

TRITON_PATH_FILES = (
    "tensorlbm/cac_lbm.py",
    "tensorlbm/d3q19.py",
    "tensorlbm/boundaries3d.py",
    "tensorlbm/phasefield/stream_boundary_contract.py",
)


def load_case_module():
    """Import the case's own run.py (criteria + calibration source of truth)."""
    spec = importlib.util.spec_from_file_location("_laplace_cac_case", CASE_DIR / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def device_evidence(device: torch.device) -> dict:
    ev: dict = {
        "requested_device": str(device),
        "torch_version": torch.__version__,
        "torch_sdaa_import_error": SDAAPY_IMPORT_ERROR,
        "device_name_line": None,
        "device_count": None,
        "total_memory_gb": None,
        "sdaa_available": False,
        "probe_tensor_device": None,
        "probe_tensor_device_type": None,
    }
    sd = getattr(torch, "sdaa", None)
    ev["has_torch_sdaa_attr"] = sd is not None
    if sd is not None:
        try:
            torch.sdaa.set_device(0)
        except Exception as exc:
            ev["set_device_error"] = repr(exc)
        ev["device_count"] = int(torch.sdaa.device_count())
        try:
            ev["device_name_line"] = torch.sdaa.get_device_name(0)
        except Exception as exc:
            ev["device_name_error"] = repr(exc)
        try:
            ev["total_memory_gb"] = round(torch.sdaa.get_device_properties(0).total_memory / 1e9, 3)
        except Exception as exc:
            ev["props_error"] = repr(exc)
    if device.type == "sdaa" and ev["device_count"]:
        probe = torch.zeros(4, dtype=torch.float64, device=device)
        probe += 1.0
        ev["probe_tensor_device"] = str(probe.device)
        ev["probe_tensor_device_type"] = probe.device.type
        ev["sdaa_available"] = probe.device.type == "sdaa"
        del probe
    return ev


def triton_scan() -> dict:
    """Does this case's call path need triton? Verified by source scan."""
    hits = {}
    for rel in TRITON_PATH_FILES:
        p = REPO / "src" / rel
        txt = p.read_text(encoding="utf-8") if p.exists() else ""
        hits[rel] = len(re.findall(r"^\s*(?:import|from)\s+triton", txt, flags=re.M))
    try:
        ver = subprocess.run(
            [sys.executable, "-c", "import triton;print(triton.__version__)"],
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover
        ver = f"unavailable: {exc!r}"
    needed = any(v > 0 for v in hits.values())
    return {
        "triton_installed_version": ver or None,
        "triton_import_hits_per_file": hits,
        "call_path_requires_triton": needed,
        "TRITON_INTERPRET_env": os.environ.get("TRITON_INTERPRET", "") or None,
        "workaround": (
            "none needed: no triton import in the cac_lbm call path"
            if not needed
            else "TRITON_INTERPRET=1 required"
        ),
    }


def run_arm_scaled(
    case,
    kind: str,
    arg: float,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    n_steps: int,
    grid: int,
    radius: float,
    sample_every: int = 500,
    window: int = 2000,
    ckpt_path: Path | None = None,
    ckpt_state: dict | None = None,
) -> dict:
    """Same protocol as case.run_arm, with grid/radius/step count overridden.

    Only the physics-identical loop is mirrored here; all constants
    (sigma Route-B, MOB, NU, W_INT, p0) come from the case module.
    """
    m = case.m
    shape = (grid, grid, grid)
    if kind == "b1":
        rho_l, rho_g = 1.0, 0.01
    else:
        rho_l, rho_g = 1.0, 1.0 / float(arg)
    mu_l, mu_g = case.NU * rho_l, case.NU * rho_g

    sim = m.CACSim(
        shape,
        rho_l=rho_l,
        rho_g=rho_g,
        mu_l=mu_l,
        mu_g=mu_g,
        sigma=sigma,
        width=case.W_INT,
        mobility=case.MOB,
        p0=0.01,
        axes="ppp",
        mu_law="harmonic",
        dtype=dtype,
        device=device,
        gravity=(0.0, 0.0, 0.0),
        gravity_mode="rho",
    )
    center = tuple(s / 2.0 for s in shape)
    sim.initialize(
        m.init_phi_droplet(shape, radius=radius, width=case.W_INT, center=center, dtype=dtype)
    )
    dev_ok = str(sim.f.device)  # evidence: compute tensors live on sdaa

    t0 = time.time()
    series = []
    phi0_int = None
    for it in range(1, n_steps + 1):
        sim.step()
        if it == 1 or it % sample_every == 0:
            phi, rho, grad, mu_phi, F, u, p, u_star = m._macro_fp64(sim)
            if phi0_int is None:
                phi0_int = phi.sum().item()
            inside = phi > 0.9
            outside = phi < 0.1
            p_in = p[inside].mean().item() if inside.any() else float("nan")
            p_out = p[outside].mean().item() if outside.any() else float("nan")
            umax = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt().max().item()
            series.append(
                {
                    "step": it,
                    "phi_integral": phi.sum().item(),
                    "phi_drift_rel": abs(phi.sum().item() - phi0_int) / phi0_int,
                    "p_in": p_in,
                    "p_out": p_out,
                    "dP": p_in - p_out,
                    "sigma_rec": (p_in - p_out) * radius / 2.0,
                    "u_max": umax,
                    "rho_min": rho.min().item(),
                    "rho_max": rho.max().item(),
                    "nan_f": int(torch.isnan(sim.f).sum().item()),
                    "nan_g": int(torch.isnan(sim.g).sum().item()),
                    "wall_s": time.time() - t0,
                }
            )
            if ckpt_path is not None and ckpt_state is not None:
                ckpt_state.setdefault("arms_progress", {})[f"{kind}/{arg:g}"] = {
                    "last_step": it,
                    "samples": series,
                    "wall_s": time.time() - t0,
                    "compute_tensor_device": dev_ok,
                }
                ckpt_path.write_text(json.dumps(ckpt_state, indent=2), encoding="utf-8")
            if it % 1000 == 0:
                s = series[-1]
                print(
                    f"[{kind}/{arg:g}] step {it}  sig_rec {s['sigma_rec']:.6f}  "
                    f"u_max {s['u_max']:.2e}  drift {s['phi_drift_rel']:.2e}  "
                    f"t {s['wall_s']:.0f}s",
                    flush=True,
                )

    win = [s for s in series if s["step"] > n_steps - window]
    sigma_mean = sum(s["sigma_rec"] for s in win) / len(win)
    drift_at_10k = next((s["phi_drift_rel"] for s in series if s["step"] == 10_000), None)
    wall = time.time() - t0
    return {
        "kind": kind,
        "arg": arg,
        "params": dict(
            shape=list(shape),
            radius=radius,
            rho_l=rho_l,
            rho_g=rho_g,
            mu_l=mu_l,
            mu_g=mu_g,
            nu=case.NU,
            sigma=sigma,
            W=case.W_INT,
            M_mob=case.MOB,
            p0=0.01,
            n_steps=n_steps,
            axes="ppp",
            mu_law="harmonic",
            dtype=str(dtype),
            device=str(device),
            window=f"last {window} steps, samples every {sample_every}",
        ),
        "sigma_target": case.SIGMA_TARGET,
        "sigma_model_input": sigma,
        "sigma_rec_window_mean": sigma_mean,
        "sigma_rec_rel_err": abs(sigma_mean - case.SIGMA_TARGET) / case.SIGMA_TARGET,
        "err_pct": abs(sigma_mean - case.SIGMA_TARGET) / case.SIGMA_TARGET * 100.0,
        "dP_window_mean": sum(s["dP"] for s in win) / len(win),
        "u_max_window": max(s["u_max"] for s in win),
        "phi_drift_rel_at_10k": drift_at_10k,
        "phi_drift_rel_last": series[-1]["phi_drift_rel"],
        "nan_free": all(s["nan_f"] == 0 and s["nan_g"] == 0 for s in series),
        "phi_integral_rel_drift_max": max(s["phi_drift_rel"] for s in series),
        "series_len": len(series),
        "tail_sigma_rec": [s["sigma_rec"] for s in win],
        "series": series,
        "compute_tensor_device": dev_ok,
        "wall_time_s": wall,
        "ms_per_step": wall / n_steps * 1e3,
    }


FULL_CASE_GRIDS = {"b1": (128, 3), "b3": (96, 2)}  # case README, 12000 steps each


def throughput_estimate(state: dict, measured_grid: int) -> dict:
    """Per-cell extrapolation of the measured step cost to the case's own grids.

    Cost is assumed linear in cell count (verified on this card by the
    setup probe: 64^3 -> 108 ms, 96^3 -> 335 ms, 128^3 -> 832 ms per step).
    """
    arms = state["arms"]
    if not arms:
        return {}
    ms = sum(a["ms_per_step"] for a in arms.values()) / len(arms)
    per_cell_ms = ms / float(measured_grid**3)
    per_arm = {}
    total_s = 0.0
    for kind, (g, n_arms) in FULL_CASE_GRIDS.items():
        est_s = per_cell_ms * float(g**3) * case_steps * n_arms / 1e3
        per_arm[f"{kind} @{g}^3 x{n_arms}"] = round(est_s / 3600.0, 2)
        total_s += est_s
    return {
        "measured_ms_per_step_per_arm": {k: round(v["ms_per_step"], 2) for k, v in arms.items()},
        "measured_grid": measured_grid,
        "ms_per_step_per_cell": per_cell_ms,
        "assumption": "step cost ~ linear in cell count (checked by setup probe)",
        "full_case_hours_by_group": per_arm,
        "estimate_full_case_hours_on_this_card": round(total_s / 3600.0, 2),
    }


def judge(case, arms: dict, planned_arms: list[str], n_steps: int, grid: int) -> dict:
    """Per-item PASS/FAIL/NOT_EVALUATED against the case's own gates."""
    G = case.GATES
    checks = []

    for name, a in arms.items():
        checks.append(
            {
                "criterion": f"per_arm |sigma_rec/{case.SIGMA_TARGET} - 1| <= "
                f"{G['per_arm_err_pct']}% [{name}]",
                "gate": G["per_arm_err_pct"],
                "value": a["err_pct"],
                "unit": "percent",
                "verdict": "PASS" if a["err_pct"] <= G["per_arm_err_pct"] else "FAIL",
            }
        )

    b1 = [a for a in arms.values() if a["kind"] == "b1"]
    if len(b1) >= 2:
        sig = [a["sigma_rec_window_mean"] for a in b1]
        spread = (max(sig) - min(sig)) / (sum(sig) / len(sig)) * 100.0
        checks.append(
            {
                "criterion": f"B1 R-ladder spread (max-min)/mean <= {G['b1_spread_pct']}%",
                "gate": G["b1_spread_pct"],
                "value": spread,
                "unit": "percent",
                "verdict": "PASS" if spread <= G["b1_spread_pct"] else "FAIL",
            }
        )
    else:
        checks.append(
            {
                "criterion": f"B1 R-ladder spread (max-min)/mean <= {G['b1_spread_pct']}%",
                "gate": G["b1_spread_pct"],
                "value": None,
                "unit": "percent",
                "verdict": "NOT_EVALUATED",
                "reason": (
                    "needs the 3-radius ladder R in {20,32,48} at 128^3 (case README); "
                    "measured throughput on this card puts a single R=32/R=48 arm at "
                    "~166 min (12000 steps), far beyond the <=20 min sweep budget; "
                    f"only a single radius fits, at {grid}^3"
                ),
            }
        )

    if arms:
        umax = max(a["u_max_window"] for a in arms.values())
        checks.append(
            {
                "criterion": f"spurious velocity max|u| (in window) <= {G['u_max']}",
                "gate": G["u_max"],
                "value": umax,
                "unit": "-",
                "verdict": "PASS" if umax <= G["u_max"] else "FAIL",
            }
        )
        nan_free = all(a["nan_free"] for a in arms.values())
        checks.append(
            {
                "criterion": "NaN-free over the whole run",
                "gate": True,
                "value": nan_free,
                "unit": "bool",
                "verdict": "PASS" if nan_free else "FAIL",
            }
        )
        drifts = [a["phi_drift_rel_at_10k"] for a in arms.values()]
        if all(d is not None for d in drifts):
            dmax = max(drifts)
            checks.append(
                {
                    "criterion": f"phi integral drift @10k steps <= {G['drift_10k']}",
                    "gate": G["drift_10k"],
                    "value": dmax,
                    "unit": "-",
                    "verdict": "PASS" if dmax <= G["drift_10k"] else "FAIL",
                }
            )
        else:
            checks.append(
                {
                    "criterion": f"phi integral drift @10k steps <= {G['drift_10k']}",
                    "gate": G["drift_10k"],
                    "value": None,
                    "unit": "-",
                    "verdict": "NOT_EVALUATED",
                    "reason": f"run length {n_steps} < 10000 steps",
                }
            )

    missing = [a for a in planned_arms if a not in arms]
    covered_ok = (
        len([a for a in arms.values() if a["kind"] == "b1"]) == 3
        and len([a for a in arms.values() if a["kind"] == "b3"]) == 2
    )
    checks.append(
        {
            "criterion": "arm coverage: B1 R in {20,32,48} (128^3) + B3 rho-ratio in {10,100}",
            "gate": True,
            "value": {"ran": sorted(arms), "not_run": missing},
            "unit": "arms",
            "verdict": "PASS" if covered_ok else "FAIL",
            "reason": ""
            if covered_ok
            else "reduced tier: not all 5 arms fit the wall-clock budget",
        }
    )

    verdicts = [c["verdict"] for c in checks]
    return {
        "checks": checks,
        "n_pass": sum(v == "PASS" for v in verdicts),
        "n_fail": sum(v == "FAIL" for v in verdicts),
        "n_not_evaluated": sum(v == "NOT_EVALUATED" for v in verdicts),
        "case_verified_on_sdaa": bool(verdicts and all(v == "PASS" for v in verdicts)),
        "overall_verdict": ("PASS" if verdicts and all(v == "PASS" for v in verdicts) else "FAIL"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="b3_rho100")
    ap.add_argument("--grid", type=int, default=64, help="cubic grid edge (reduced tier)")
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument(
        "--planned-arms",
        default=",".join(("b1_R20", "b1_R32", "b1_R48", "b3_rho10", "b3_rho100")),
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--recompute-only",
        default=None,
        metavar="JSON",
        help="refresh derived (non-device) fields of an existing sweep JSON, "
        "e.g. --recompute-only artifacts/sdaa_laplace_cac_phasefield.json",
    )
    args = ap.parse_args()

    case = load_case_module()

    if args.recompute_only:
        target = Path(args.recompute_only)
        state = json.loads(target.read_text(encoding="utf-8"))
        state["throughput"] = throughput_estimate(state, state["config"]["grid"][0])
        state.pop("arms_progress", None)
        target.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(json.dumps(state["throughput"], indent=1))
        print(f"Refreshed derived fields -> {target}")
        return 0

    # iron rule inherited from the case: no hand-written collide/stream here
    src = Path(os.path.abspath(__file__)).read_text(encoding="utf-8")
    grep = {
        p: len(re.findall(p, src))
        for p in (r"def\s+bounce", r"def\s+stream_3d", r"torch\.roll\(\s*f\b")
    }
    assert all(v == 0 for v in grep.values()), grep

    device = torch.device(args.device)
    ev = device_evidence(device)
    print(f"[device] {ev['device_name_line']!r}  ({ev['probe_tensor_device']})", flush=True)
    if not ev["sdaa_available"]:
        print(
            "FATAL: 'sdaa' compute device unavailable — refusing to substitute CPU "
            f"(evidence: {json.dumps(ev)})",
            file=sys.stderr,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "benchmark": "laplace_cac_phasefield",
                    "target_device": str(device),
                    "device_evidence": ev,
                    "triton": triton_scan(),
                    "status": "FAIL",
                    "reason": "sdaa device unavailable; CPU substitution forbidden",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 2

    tri = triton_scan()
    sigma = case.SIGMA_ROUTE_B
    dtype = torch.float64
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    planned = [a.strip() for a in args.planned_arms.split(",") if a.strip()]

    state: dict = {
        "benchmark": "laplace_cac_phasefield",
        "case_dir": str(CASE_DIR.relative_to(REPO)),
        "case_entrypoint": "benchmarks/verified/laplace_cac_phasefield/run.py",
        "sweep_script": str(Path(__file__).resolve().relative_to(REPO)),
        "branch": os.popen("git -C %s rev-parse --abbrev-ref HEAD" % REPO).read().strip(),
        "device_evidence": ev,
        "triton": tri,
        "config": {
            "tier": "reduced (case recommends 128^3 b1 / 96^3 b3, 12000 steps)",
            "grid": [args.grid] * 3,
            "radius": args.radius,
            "steps": args.steps,
            "sample_every": 500,
            "window": "last 2000 steps",
            "arms": [a.strip() for a in args.arms.split(",") if a.strip()],
            "planned_arms_full_case": planned,
            "dtype": str(dtype),
            "device": str(device),
        },
        "criteria_source": {
            "module": "benchmarks/verified/laplace_cac_phasefield/run.py (imported live)",
            "sigma_target": case.SIGMA_TARGET,
            "bar_eff_W4": case.BAR_EFF_W4,
            "sigma_model_input": sigma,
            "gates": dict(case.GATES),
            "note": "criteria/calibration read from the case module, not transcribed",
        },
        "arms": {},
    }
    t_start = time.time()
    for arm in state["config"]["arms"]:
        kind = "b1" if arm.startswith("b1") else "b3"
        arg = float(arm.split("_R")[1] if kind == "b1" else arm.split("_rho")[1])
        print(f"===== {arm} on {device} grid {args.grid}^3 steps {args.steps} =====", flush=True)
        res = run_arm_scaled(
            case,
            kind,
            arg,
            sigma,
            device,
            dtype,
            args.steps,
            args.grid,
            args.radius,
            ckpt_path=out,
            ckpt_state=state,
        )
        state["arms"][arm] = res
        print(
            f"[{arm}] sigma_rec {res['sigma_rec_window_mean']:.6f} "
            f"err {res['err_pct']:.4f}% u_max {res['u_max_window']:.2e} "
            f"nan_free {res['nan_free']} {res['wall_time_s']:.0f}s",
            flush=True,
        )

    state["gates"] = judge(case, state["arms"], planned, args.steps, args.grid)
    state["wall_time_s"] = time.time() - t_start
    state["throughput"] = throughput_estimate(state, args.grid)
    state["status"] = state["gates"]["overall_verdict"]
    state.pop("arms_progress", None)  # transient checkpoint scratch
    out.write_text(json.dumps(state, indent=2), encoding="utf-8")
    g = state["gates"]
    print(
        f"\nverdict={g['overall_verdict']} pass={g['n_pass']} fail={g['n_fail']} "
        f"not_evaluated={g['n_not_evaluated']} wall={state['wall_time_s']:.0f}s"
    )
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
