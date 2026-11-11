"""Negative-input probe for TensorLBM's core input contract (audit D5/D6/D7/D9 follow-up).

Each probe feeds a deliberately illegal value into a public entry point and records
* what the code did: raised a typed error (good) or silently produced garbage (bad),
* if silent, a compact fingerprint of the garbage (nan / inf / magnitude).

The same script is run against the pre-fix tree and the post-fix tree; the two JSON
files are the before/after evidence for docs/INPUT_CONTRACT_GUARDS_20261111.md.

Usage:  python scripts/contract_guards/negative_probe.py --out artifacts/contract_guards/probe_after.json [--label after]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def _fingerprint(t: torch.Tensor) -> dict:
    """Compact characterisation of a field that should never have been produced."""
    tf = t.float()
    return {
        "has_nan": bool(torch.isnan(tf).any()),
        "has_inf": bool(torch.isinf(tf).any()),
        "max_abs": None if bool(torch.isnan(tf).any()) else float(tf.abs().max()),
    }


def _record(probe_id: str, call, *, what: str) -> dict:
    try:
        out = call()
    except Exception as exc:  # noqa: BLE001 - probe deliberately catches everything
        return {
            "probe": probe_id,
            "input": what,
            "outcome": "raised",
            "exc_type": type(exc).__name__,
            "exc_msg": str(exc)[:200],
        }
    row = {"probe": probe_id, "input": what, "outcome": "silent"}
    if isinstance(out, torch.Tensor):
        row["field"] = _fingerprint(out)
    elif isinstance(out, dict):
        row["dict_keys"] = sorted(out.keys())[:8]
    return row


def probe_collide_bgk() -> list[dict]:
    from tensorlbm.solver import collide_bgk, collide_bgk_fused

    rho = torch.ones(8, 8)
    f = torch.stack([w * rho for w in torch.full((9,), 1.0 / 9.0)]) * 0.0 + 1.0 / 9.0
    rows = []
    for name, fn in (("collide_bgk", collide_bgk), ("collide_bgk_fused", collide_bgk_fused)):
        for tag, tau in (("tau_0.5", 0.5), ("tau_0.0", 0.0), ("tau_nan", float("nan")),
                         ("tau_neg", -0.7)):
            rows.append(_record(f"{name}:{tag}", lambda fn=fn, tau=tau: fn(f, tau), what=f"tau={tau}"))
    return rows


def probe_outlets() -> list[dict]:
    from tensorlbm.boundary.boundaries import zou_he_outlet_pressure
    from tensorlbm.boundary.boundaries3d import zou_he_outlet_pressure_3d
    from tensorlbm.boundary.boundaries_d3q27 import zou_he_outlet_pressure_27

    rows = []
    f2 = torch.full((9, 6, 6), 1.0 / 9.0)
    for tag, rho_out in (("rho_out_0", 0.0), ("rho_out_neg", -1.0), ("rho_out_nan", float("nan"))):
        rows.append(_record(f"zou_he_outlet_pressure:{tag}",
                            lambda r=rho_out: zou_he_outlet_pressure(f2, rho_out=r),
                            what=f"rho_out={rho_out}"))
    f3 = torch.full((19, 4, 4, 4), 1.0 / 19.0)
    for tag, rho_out in (("rho_out_0", 0.0), ("rho_out_neg", -1.0)):
        rows.append(_record(f"zou_he_outlet_pressure_3d:{tag}",
                            lambda r=rho_out: zou_he_outlet_pressure_3d(f3, rho_out=r),
                            what=f"rho_out={rho_out}"))
    f27 = torch.full((27, 4, 4, 4), 1.0 / 27.0)
    for tag, rho_out in (("rho_out_0", 0.0), ("rho_out_neg", -1.0)):
        rows.append(_record(f"zou_he_outlet_pressure_27:{tag}",
                            lambda r=rho_out: zou_he_outlet_pressure_27(f27, rho_out=r),
                            what=f"rho_out={rho_out}"))
    return rows


def probe_general_sim_config() -> list[dict]:
    from tensorlbm.general_sim import GeneralSimConfig
    from tensorlbm.general_sim import PhysicsConfig, SolverConfig

    rows = []

    def _validate(cfg):
        """Call cfg.validate() when the contract exists; record its absence honestly."""
        fn = getattr(cfg, "validate", None)
        if fn is None:
            return {"_no_validate_method": True}
        return fn()

    def build(**solver_kw):
        return lambda: _validate(GeneralSimConfig(solver=SolverConfig(**solver_kw)))

    rows.append(_record("GeneralSimConfig:resolution_0", build(resolution=0), what="resolution=0"))
    rows.append(_record("GeneralSimConfig:resolution_neg", build(resolution=-4), what="resolution=-4"))
    rows.append(_record("GeneralSimConfig:max_steps_neg", build(max_steps=-5), what="max_steps=-5"))
    rows.append(_record("GeneralSimConfig:viscosity_0",
                        lambda: _validate(GeneralSimConfig(physics=PhysicsConfig(viscosity=0.0))),
                        what="viscosity=0"))
    rows.append(_record("GeneralSimConfig:viscosity_neg",
                        lambda: _validate(GeneralSimConfig(physics=PhysicsConfig(viscosity=-1e-6))),
                        what="viscosity=-1e-6"))
    rows.append(_record("GeneralSimConfig:baseline_ok",
                        lambda: _validate(GeneralSimConfig()), what="defaults (must pass)"))
    return rows


def probe_poiseuille_2d_cli() -> list[dict]:
    """H=0 reaches a division by the wall-to-wall height inside the benchmark runner."""
    run = REPO / "benchmarks" / "verified" / "poiseuille_2d" / "run.py"
    if not run.exists():
        return [{"probe": "poiseuille_2d_cli:H_0", "outcome": "skipped", "exc_msg": "run.py absent"}]
    proc = subprocess.run(
        [sys.executable, str(run), "0", "/tmp/_probe_p2d_H0.json", "--min-steps", "2", "--max-steps", "4"],
        capture_output=True, text=True, timeout=300, cwd=str(REPO),
    )
    last = (proc.stderr.strip().splitlines() or [""])[-1]
    return [{
        "probe": "poiseuille_2d_cli:H_0",
        "input": "H=0",
        "outcome": "raised" if proc.returncode != 0 else "silent",
        "exc_type": last.split(":")[0].strip() if proc.returncode != 0 else "",
        "exc_msg": last[:200],
    }]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--repo-ref", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    rows: list[dict] = []
    rows += probe_collide_bgk()
    rows += probe_outlets()
    rows += probe_general_sim_config()
    rows += probe_poiseuille_2d_cli()

    payload = {"label": args.label, "repo_ref": args.repo_ref, "n_probes": len(rows), "probes": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    silent = [r for r in rows if r["outcome"] == "silent"]
    print(f"[{args.label}] {len(rows)} probes, {len(silent)} silent-illegal-input paths")
    for r in rows:
        tail = r.get("field") or r.get("exc_type") or ""
        print(f"  {r['probe']:<45} {r['outcome']:<8} {tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())