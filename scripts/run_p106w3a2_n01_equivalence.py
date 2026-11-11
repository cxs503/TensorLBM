#!/usr/bin/env python3
"""P106W3A2-N01 driver: heterogeneous-backend equivalence ladder.

Stages (each stage is short enough to run under a per-command wall-time cap;
``l1`` is resumable through a checkpoint file):

    python scripts/run_p106w3a2_n01_equivalence.py --stage inventory
    python scripts/run_p106w3a2_n01_equivalence.py --stage l0 --tier cpu_fp64_serial
    python scripts/run_p106w3a2_n01_equivalence.py --stage l0-amp --tier cpu_fp64_serial
    python scripts/run_p106w3a2_n01_equivalence.py --stage l0-check
    python scripts/run_p106w3a2_n01_equivalence.py --stage l1 --profile cpu_fp64_torch --advance 1000
    python scripts/run_p106w3a2_n01_equivalence.py --stage l1-finalize --profile cpu_fp64_torch --tag full
    python scripts/run_p106w3a2_n01_equivalence.py --stage aggregate

``--stage refusals`` probes every registered tier and records the refusals
(fail-closed inventory).  ``--stage resume`` prints the state of a chunked run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import tensorlbm.hetero_equivalence as HE  # noqa: E402

ART = ROOT / "artifacts" / "p106w3a2_n01"
L0_DIR = ART / "l0"
L1_DIR = ART / "l1"
FIELDS_DIR = ART / "fields"

L0_TIERS = ("cpu_fp64_serial", "cpu_fp64_numpy_tree", "cpu_fp64_torch_gemm",
            "cpu_fp32_torch", "sdaa_fp64_torch", "sdaa_fp32_torch",
            # negative control: the shipped float32 weight table must fail L0/L2
            "cpu_fp64_shipped_w")


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(HE._jsonable(payload), indent=1), encoding="utf-8")


def _fields_path(tier: str, tag: str = "") -> Path:
    return FIELDS_DIR / f"{tier}{tag}.npz"


def save_fields(tier: str, run: dict, tag: str = "") -> str:
    FIELDS_DIR.mkdir(parents=True, exist_ok=True)
    path = _fields_path(tier, tag)
    np.savez_compressed(path, **{k: v for k, v in run["_fields"].items()})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fields(tier: str, tag: str = "") -> dict[str, np.ndarray]:
    with np.load(_fields_path(tier, tag)) as blob:
        return {k: blob[k] for k in blob.files}


def load_run(tier: str) -> dict:
    payload = json.loads((L0_DIR / f"{tier}.json").read_text(encoding="utf-8"))
    payload["_fields"] = load_fields(tier)
    return payload


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_inventory(args) -> None:
    ledger = HE.RefusalLedger()
    payload = HE.aggregate_ladder(ledger)
    payload["weight_table_audit"] = HE.weight_table_audit()
    payload["host"] = {
        "cpu_count": os.cpu_count(),
        "torch": _torch_version(),
        "numpy": np.__version__,
    }
    _dump(ART / "inventory.json", payload)
    print(json.dumps({"refusals": len(payload["refusals"]),
                      "available": [i["profile"] for i in payload["inventory"] if i["available"]]}))


def _torch_version() -> str:
    try:
        import torch

        return f"{torch.__version__} cuda_available={torch.cuda.is_available()}"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def stage_l0(args) -> None:
    tier = args.tier
    run = HE.l0_run(tier, n=args.grid, steps=args.steps, u0=args.u0, nu=args.nu)
    fields_sha = save_fields(tier, run)
    payload = {k: v for k, v in run.items() if not k.startswith("_")}
    payload["fields_npz_sha256"] = fields_sha
    payload["fields_npz"] = str(_fields_path(tier))
    _dump(L0_DIR / f"{tier}.json", payload)
    man = HE.freeze_golden(run, L0_DIR, name=f"golden_{tier}" if tier == HE.GOLDEN_PROFILE else f"trace_{tier}")
    print(json.dumps({"tier": tier, "elapsed_s": round(run["elapsed_seconds"], 2),
                      "mcells_s": round(run["mcell_steps_per_second"], 3),
                      "digest_chain": run["digest_chain"][:16],
                      "manifest": str(L0_DIR / (man["name"] + "_manifest.json"))}))


def stage_l0_amp(args) -> None:
    tier = args.tier
    prof = HE.resolve_profile(tier)
    base = load_run(tier)
    pert = HE.l0_run(tier, n=args.grid, steps=args.steps, u0=args.u0, nu=args.nu, seed_ulp=prof.eps)
    save_fields(tier, pert, tag="_amp")
    d0 = prof.eps * args.u0
    devs = {}
    for key in ("rho", "ux", "uy", "uz"):
        a = np.asarray(base["_fields"][key], dtype=np.float64)
        b = np.asarray(pert["_fields"][key], dtype=np.float64)
        devs[key] = float(np.max(np.abs(b - a)))
    dN = max(devs.values())
    payload = {
        "profile": tier,
        "dtype": prof.dtype,
        "eps": prof.eps,
        "seed_delta0": d0,
        "deltaN": dN,
        "deltaN_per_component": devs,
        "amplification": dN / d0,
        "steps": args.steps,
        "grid": [args.grid] * 3,
        "perturbed_digest_chain": pert["digest_chain"],
        "base_digest_chain": base["digest_chain"],
        "elapsed_seconds": pert["elapsed_seconds"],
    }
    _dump(L0_DIR / f"{tier}_amp.json", payload)
    print(json.dumps({"tier": tier, "amplification": payload["amplification"]}))


def stage_l0_check(args) -> None:
    golden = load_run(HE.GOLDEN_PROFILE)
    amps = {}
    for tier in L0_TIERS:
        path = L0_DIR / f"{tier}_amp.json"
        if path.exists():
            amps[tier] = json.loads(path.read_text(encoding="utf-8"))["amplification"]
    # Fall back to the amplification measured for the *same precision*: A_N is a
    # property of the trajectory/dtype, not of the kernel that reproduces it.
    for tier in L0_TIERS:
        if tier in amps:
            continue
        dtype = HE.PROFILES[tier].dtype
        same = [t for t, a in amps.items() if HE.PROFILES[t].dtype == dtype]
        if same:
            amps[tier] = amps[same[0]]
    checks = {}
    for tier in L0_TIERS:
        if tier == HE.GOLDEN_PROFILE:
            checks[tier] = HE.check_l0(golden, golden, amplification=amps.get(tier, 1.0))
        else:
            cand = load_run(tier)
            checks[tier] = HE.check_l0(golden, cand, amplification=amps.get(tier, 1.0))
    l2 = {tier: HE.check_l2(load_run(tier)) for tier in L0_TIERS}
    payload = {
        "schema": HE.SCHEMA,
        "level": "L0",
        "grid": golden["grid"],
        "steps": golden["steps"],
        "u0": golden["u0"],
        "nu": golden["nu"],
        "tau": golden["tau"],
        "golden_profile": HE.GOLDEN_PROFILE,
        "amplification": amps,
        "checks": checks,
        "l2": l2,
        "weight_table_audit": HE.weight_table_audit(),
        "criterion": {
            "L0": "bitwise digest identity OR max_rel_deviation_field <= A_N*N_OPS*eps",
            "L2": "mass/momentum drift per step per cell <= 1e-12",
        },
    }
    payload["all_l0_passed"] = all(c["passed"] for c in checks.values())
    payload["l2_passed_if_fp64"] = l2[HE.GOLDEN_PROFILE]["passed"]
    _dump(ART / "l0.json", payload)
    print(json.dumps({k: (v["passed"], v["bitwise_identical"]) for k, v in checks.items()}))
    print("L2:", {k: (v["passed"], v["mass"]["drift_per_step_per_cell"]) for k, v in l2.items()})


def _chunked(profile: str, args):
    return HE.ChunkedChannelRun(
        profile,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        steps=args.steps,
        warmup_steps=args.warmup,
        sample_interval=args.sample_interval,
        re_tau=args.re_tau,
        u_tau=args.u_tau,
        wall_scheme=args.wall_scheme,
        tau_minus_max=args.tau_minus_max,
        smagorinsky_cs=args.smagorinsky_cs,
        seed=args.seed,
        state_dir=L1_DIR / f"state_{profile}_{args.wall_scheme}_{args.tag}",
        label=f"{profile}:{args.wall_scheme}",
    )


def stage_l1(args) -> None:
    run = _chunked(args.profile, args)
    status = run.advance(args.advance)
    print(json.dumps(status))


def stage_l1_finalize(args) -> None:
    run = _chunked(args.profile, args)
    payload = run.finalize()
    payload["tag"] = args.tag
    payload["finalized_at_step"] = payload["steps_done"]
    _dump(L1_DIR / f"{args.profile}_{args.tag}.json", payload)
    print(json.dumps({k: payload[k] for k in ("profile", "steps_done", "samples", "re_tau_achieved",
                                              "elapsed_seconds", "mcell_steps_per_second")}, default=str))



def _representativeness(l1_run: dict, mkm) -> dict:
    """Is the probed state actually the turbulent channel the rung claims?

    The rung is a *paired* backend comparison: it can be satisfied by two
    identically-wrong solutions.  This block therefore reports the turbulence
    level of the golden run next to the MKM-1999 anchor, so a relaminarised
    (under-resolved) state cannot masquerade as a certified turbulent channel.
    """
    import numpy as np

    uv_peak = float(l1_run["reynolds_stress"]["uv_minus_peak_plus"])
    mkm_uv_peak = float(np.max(-np.asarray(mkm.r_uv, dtype=float)))
    ratio = uv_peak / mkm_uv_peak if mkm_uv_peak else float("inf")
    turbulent = bool(ratio >= 0.5)
    return {
        "uv_minus_peak_plus_golden": uv_peak,
        "uv_minus_peak_plus_mkm1999": mkm_uv_peak,
        "uv_peak_ratio_vs_mkm": ratio,
        "turbulent_state": turbulent,
        "verdict": ("turbulent: paired statistical comparison is representative"
                    if turbulent else
                    "RELAMINARISED: the backend comparison is valid but it probes a "
                    "laminar state, so it does not certify turbulent statistical "
                    "equivalence (see limitations)"),
        "dy_plus": float(l1_run["config"]["dy_plus"]),
        "domain_lx_over_h": float(l1_run["config"]["domain_lx_over_h"]),
        "eddy_turnovers_averaged": float(l1_run["config"]["eddy_turnovers_averaged"]),
    }

def stage_aggregate(args) -> None:
    """Build the acceptance JSON (criteria numbers + artifact paths + Pareto)."""
    inventory = json.loads((ART / "inventory.json").read_text(encoding="utf-8"))
    l0 = json.loads((ART / "l0.json").read_text(encoding="utf-8"))

    from tensorlbm.validation import mkm1999

    mkm = mkm1999.load_profile(180)
    golden_candidates = [L1_DIR / f"{args.l1_golden}_{args.l1_golden_tag}.json",
                         L1_DIR / f"{args.l1_golden}.json"]
    golden_path = next((p for p in golden_candidates if p.exists()), None)
    if golden_path is None:
        raise SystemExit(f"no golden L1 run: expected one of {[str(p) for p in golden_candidates]}")
    l1_golden = json.loads(golden_path.read_text(encoding="utf-8"))
    l1_checks = {}
    points = []
    for path in sorted(L1_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("level") != "L1" or payload.get("tag") != "full":
            continue
        name = payload["profile"]["name"]
        if name == args.l1_golden:
            continue
        check = HE.check_l1(l1_golden, payload, mkm=mkm)
        check["run_path"] = str(path)
        l1_checks[name] = check
        points.append({
            "name": name,
            "level": "L1",
            "dtype": payload["profile"]["dtype"],
            "error": check["worst_normalised_l2"],
            "cost": payload["seconds_per_convective_time"],
            "mcell_steps_per_second": payload["mcell_steps_per_second"],
        })
    # negative control (scheme change) is reported separately, never certified
    negative = {}
    for path in sorted(L1_DIR.glob("*_negative.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        negative[payload["profile"]["name"] + ":" + str(payload.get("wall_scheme"))] = {
            "run_path": str(path),
            "check_vs_golden": HE.check_l1(l1_golden, payload, mkm=mkm),
            "expected": "must FAIL the 2% gate (scheme difference)",
        }

    # cost/accuracy ledger including the L0 periodic box and the L1 channel
    ledger_points = []
    for tier, check in l0["checks"].items():
        run = json.loads((L0_DIR / f"{tier}.json").read_text(encoding="utf-8"))
        ledger_points.append({
            "name": tier,
            "level": "L0",
            "dtype": run["profile"]["dtype"],
            "error": check["max_rel_deviation_field"],
            "cost": run["seconds_per_step"],
            "mcell_steps_per_second": run["mcell_steps_per_second"],
        })
    golden_l1 = l1_golden
    ledger_points.append({
        "name": args.l1_golden,
        "level": "L1",
        "dtype": golden_l1["profile"]["dtype"],
        "error": 0.0,
        "cost": golden_l1["seconds_per_convective_time"],
        "mcell_steps_per_second": golden_l1["mcell_steps_per_second"],
    })
    ledger_points.extend(points)

    golden_cost = golden_l1["seconds_per_convective_time"]
    for p in ledger_points:
        if p.get("level") == "L1":
            p["speedup_vs_golden"] = golden_cost / p["cost"] if p["cost"] else None

    pareto_l1 = HE.pareto_front([p for p in ledger_points if p.get("level") == "L1"])
    pareto_l0 = HE.pareto_front([p for p in ledger_points if p.get("level") == "L0"])

    certificates = {tier: {"L0": "certified" if l0["checks"][tier]["passed"] else "failed",
                           "L2": "certified" if l0["l2"][tier]["passed"] else "failed"}
                    for tier in l0["checks"]}
    for name, check in l1_checks.items():
        certificates.setdefault(name, {})["L1"] = "certified" if check["passed"] else "failed"
    certificates.setdefault(args.l1_golden, {})["L1"] = "certified"

    report = {
        "schema": HE.SCHEMA,
        "node": "P106W3A2-N01",
        "mission_id": 12402,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": "TensorLBM",
        "module": "src/tensorlbm/hetero_equivalence.py",
        "inventory_path": str(ART / "inventory.json"),
        "l0_path": str(ART / "l0.json"),
        "criteria": {
            "L0": {
                "requirement": "64^3, 200 steps: bitwise identity or provable ULP bound",
                "numbers": {t: {"bitwise_identical": c["bitwise_identical"],
                                "max_rel_deviation_field": c["max_rel_deviation_field"],
                                "ulp_budget_relative": c["ulp_budget_relative"],
                                "max_ulp_distance": c["max_ulp_distance"],
                                "passed": c["passed"]} for t, c in l0["checks"].items()},
                "artifact": str(ART / "l0.json"),
            },
            "L1": {
                "requirement": "Re_tau=180 channel, 1e4 steps: U+/rms deviation vs golden <= 2%, MKM-1999 not worse",
                "numbers": {name: {"worst_normalised_l2": c["worst_normalised_l2"],
                                   "threshold": c["threshold"],
                                   "mkm_golden": c["mkm"].get("normalised_l2_deviation_golden"),
                                   "mkm_candidate": c["mkm"].get("normalised_l2_deviation_candidate"),
                                   "mkm_not_worse": c["mkm"].get("not_worse"),
                                   "passed": c["passed"]} for name, c in l1_checks.items()},
                "artifact": str(L1_DIR),
            },
            "L2": {
                "requirement": "mass/momentum drift per step per cell <= 1e-12",
                "numbers": {t: {"mass_drift": v["mass"]["drift_per_step_per_cell"],
                                "momentum_drift": v["momentum"]["drift_per_step_per_cell"],
                                "passed": v["passed"]} for t, v in l0["l2"].items()},
                "artifact": str(ART / "l0.json"),
            },
            "fail_closed": {
                "requirement": "missing backend / insufficient precision must raise and be registered",
                "numbers": {"refusals": len(inventory["refusals"]),
                            "refused_tiers": sorted({r["profile"] for r in inventory["refusals"]})},
                "artifact": str(ART / "inventory.json"),
            },
        },
        "physics_representativeness": _representativeness(l1_golden, mkm),
        "l1_checks": l1_checks,
        "negative_controls": negative,
        "cost_accuracy_ledger": {"points": ledger_points,
                                 "pareto_L0": pareto_l0,
                                 "pareto_L1": pareto_l1},
        "certificates": certificates,
        "refusals": inventory["refusals"],
        "weight_table_audit": l0["weight_table_audit"],
        "limitations": [
            "L0 budget is an engineering bound (measured amplification x local rounding budget), not a bit-exact forward-error proof.",
            "L1 is a paired comparison: identical IC/seed/grid; it detects implementation/precision drift, it does not certify physical accuracy by itself.",
            "No accelerator (CUDA/SDAA/Triton) is present on the certified host; those rungs are certified as fail-closed refusals.",
            "The channel is coarse (dy+ ~ 8) with a short averaging window, so the MKM-1999 agreement is limited by resolution, not by the backend.",
            "tensorlbm.d3q19.W ships truncated 4-decimal float32 weights (mass defect 1.5e-8/step); the ladder kernel uses exact rationals and the production table is left untouched.",
        ],
    }
    _dump(ART / "report.json", report)

    print(json.dumps({
        "L0": {t: v["passed"] for t, v in l0["checks"].items()},
        "L1": {n: v["passed"] for n, v in l1_checks.items()},
        "L2_golden": l0["l2"][HE.GOLDEN_PROFILE]["passed"],
        "refusals": len(inventory["refusals"]),
        "pareto_L1": pareto_l1,
        "report": str(ART / "report.json"),
    }, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["inventory", "l0", "l0-amp", "l0-check", "l1", "l1-finalize", "aggregate"])
    ap.add_argument("--tier", default=HE.GOLDEN_PROFILE)
    ap.add_argument("--profile", default=HE.GOLDEN_PROFILE)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--u0", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.003)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--nx", type=int, default=48)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--sample-interval", type=int, default=5)
    ap.add_argument("--re-tau", type=float, default=180.0)
    ap.add_argument("--u-tau", type=float, default=0.01)
    ap.add_argument("--wall-scheme", default="interpolated")
    ap.add_argument("--tau-minus-max", type=float, default=None)
    ap.add_argument("--smagorinsky-cs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--advance", type=int, default=1000)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--l1-golden", default="cpu_fp64_torch")
    ap.add_argument("--l1-golden-tag", default="full")
    args = ap.parse_args()

    try:
        import torch

        torch.set_num_threads(args.threads)
    except Exception:  # noqa: BLE001
        pass

    ART.mkdir(parents=True, exist_ok=True)
    {
        "inventory": stage_inventory,
        "l0": stage_l0,
        "l0-amp": stage_l0_amp,
        "l0-check": stage_l0_check,
        "l1": stage_l1,
        "l1-finalize": stage_l1_finalize,
        "aggregate": stage_aggregate,
    }[args.stage](args)


if __name__ == "__main__":
    main()