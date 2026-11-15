"""CASE-RT-INSTABILITY-ATWOOD -- finite-density-ratio arm (two-fluid spike growth).

The Boussinesq arm of this case (``run_case.py`` / ``run_late.py``) cannot answer
the anchor question *"what is the spike growth rate at A = 1/2"*, because a
Boussinesq two-layer model is invariant under ``(phi, y) -> (-phi, -y)``: its
spike and bubble fronts are the same front (measured ratio 1.0004 against
sqrt(3) = 1.732).  This executor drives the finite-density-ratio capability
``tensorlbm.rayleigh_taylor_twofluid`` (SCMP: two phases, one component, gravity
on mass) and measures the *spike* terminal velocity against the two-fluid
Layzer-Goncharov value at the Atwood number the model actually delivers.

Stages
------
``--stage calib``  scan the Shan-Chen coupling ``sc_g`` and interpolate to the
                   coupling whose measured coexistence ratio is 3 (A = 1/2);
                   writes ``artifacts_twofluid/calibration.{json,md}``.
``--stage run``    run the pre-declared case specs (base + grid ladder +
                   viscosity arms) with the calibrated densities, and write
                   ``artifacts_twofluid/runs/<case>/{history.csv,case.json}``
                   plus a summary + acceptance table.
``--smoke``        one coarse calibration point and one short run, for a
                   syntax/physics smoke test only (never used as evidence).

Gate (pre-declared, same tolerance as the case anchor)::

    |u_spike_sim / u_spike_ref - 1| <= 15 %      u_s = sqrt(2 A g / ((1-A) C k))

with ``C = 1`` (two-dimensional Layzer constant), ``A`` measured from the
coexistence run, and the fit window declared in *front position*
(``|h_spike| in [0.15, 0.45] lam``), not chosen from the result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from tensorlbm.rayleigh_taylor_twofluid import (
    ATWOOD_NOMINAL,
    SCMPRTConfig,
    calibrate_atwood,
    layzer_velocity,
    run_scmp_rt,
    scmp_coexistence,
)

OUT_ROOT = Path(__file__).resolve().parent / "artifacts_twofluid"
TARGET_RATIO = 3.0  # A = 1/2


@dataclass
class CaseSpec:
    name: str
    lam: int
    layers: float
    gravity: float
    tau_relax: float
    a0_over_lam: float
    steps: int
    sc_g: float
    rho_high: float
    rho_low: float
    role: str = "arm"


def base_specs(sc_g: float, rho_high: float, rho_low: float) -> tuple[CaseSpec, ...]:
    """Pre-declared case set (fixed before any run was inspected)."""
    g0, tau0 = 1.5e-4, 0.8
    return (
        CaseSpec("twofluid_l128_g1.5e-4", 128, 6.0, g0, tau0, 0.02, 4200, sc_g,
                 rho_high, rho_low, "primary"),
        CaseSpec("twofluid_l128_g1.5e-4_tau070", 128, 6.0, g0, 0.70, 0.02, 4200, sc_g,
                 rho_high, rho_low, "viscosity arm"),
        CaseSpec("twofluid_l128_g1.5e-4_tau100", 128, 6.0, g0, 1.00, 0.02, 4200, sc_g,
                 rho_high, rho_low, "viscosity arm"),
        CaseSpec("twofluid_l96_g1.125e-4", 96, 8.0, g0 * 96 / 128, tau0, 0.02, 5600, sc_g,
                 rho_high, rho_low, "grid ladder"),
        CaseSpec("twofluid_l64_g0.75e-4", 64, 10.0, g0 * 64 / 128, tau0, 0.02, 8400, sc_g,
                 rho_high, rho_low, "grid ladder"),
    )


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def stage_calib(out_root: Path = OUT_ROOT, quick: bool = False) -> dict:
    steps = 1200 if quick else 6000
    if quick:
        g, results = calibrate_atwood(target_ratio=TARGET_RATIO, grid=(0.45, 0.60), steps=steps)
    else:
        g, results = calibrate_atwood(target_ratio=TARGET_RATIO, steps=steps)
    best = min(
        [r for r in results if r.stable and math.isfinite(r.ratio)],
        key=lambda r: abs(r.ratio - TARGET_RATIO),
        default=None,
    )
    payload = {
        "target_ratio": TARGET_RATIO,
        "target_atwood": ATWOOD_NOMINAL,
        "selected_sc_g": g,
        "selected": best.as_dict() if best else None,
        "scan": [r.as_dict() for r in results],
        "tau_relax": 1.0,
        "psi": "exp",
        "steps": steps,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "calibration.json").write_text(json.dumps(payload, indent=2))
    lines = [
        "# SCMP coexistence calibration (target: density ratio 3, A = 1/2)",
        "",
        f"`psi = 1 - exp(-rho)`, `tau_relax = 1.0`, `{steps}` steps, 64 x 8 box, neutral walls.",
        "",
        "| sc_g | rho_high | rho_low | ratio | A | mass drift % | stable |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.sc_g:.4f} | {r.rho_high:.4f} | {r.rho_low:.4f} | {r.ratio:.4f} | "
            f"{r.atwood:.4f} | {r.mass_drift_pct:.4f} | {r.stable} |"
        )
    (out_root / "calibration.md").write_text("\n".join(lines) + "\n")
    return payload


def _load_calibration(out_root: Path) -> tuple[float, float, float]:
    path = out_root / "calibration.json"
    if not path.exists():
        raise SystemExit(f"missing {path}: run --stage calib first")
    calib = json.loads(path.read_text())
    sel = calib["selected"]
    return calib["selected_sc_g"], sel["rho_high"], sel["rho_low"]


def run_spec(spec: CaseSpec, out_root: Path = OUT_ROOT) -> dict:
    cfg = SCMPRTConfig(
        lam=spec.lam,
        layers=spec.layers,
        gravity=spec.gravity,
        tau_relax=spec.tau_relax,
        a0_over_lam=spec.a0_over_lam,
        steps=spec.steps,
        sc_g=spec.sc_g,
        name=spec.name,
    )
    res = run_scmp_rt(cfg, rho_high=spec.rho_high, rho_low=spec.rho_low)
    history = res.pop("history")
    case_dir = out_root / "runs" / spec.name
    _write_csv(
        case_dir / "history.csv",
        history,
        ["step", "t", "tau", "max_speed", "ma", "mass_drift_pct", "rho_min", "rho_max",
         "spike_tip_lam", "bubble_tip_lam", "spike_q10_lam", "bubble_q90_lam", "n_columns"],
    )
    res["role"] = spec.role
    res["gate_pass"] = bool(abs(res["err_spike_pct"]) <= 15.0)
    (case_dir / "case.json").write_text(json.dumps(res, indent=2))
    return res


def aggregate(results: list[dict], out_root: Path = OUT_ROOT) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps({"cases": results}, indent=2))
    head = (
        "| case | lam | g | tau | A_meas | grad_req | window tau | u_spike | u_s ref | err | gate | "
        "u_bubble | u_b ref | err | asym sim | asym ref | Re | Ma_peak | stop |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        span = r.get("window_tau_span") or [float("nan"), float("nan")]
        rows.append(
            f"| {r['name']} | {r['lam']} | {r['gravity']:.3g} | {r['tau_relax']:.2f} | "
            f"{r['atwood_measured']:.4f} | {r['window_position_lam'][0]}-{r['window_position_lam'][1]} lam | "
            f"{span[0]:.2f}-{span[1]:.2f} | {r['u_spike_sim']:.4f} | {r['u_spike_ref']:.4f} | "
            f"{r['err_spike_pct']:+.2f}% | {'PASS' if r['gate_pass'] else 'FAIL'} | "
            f"{r['u_bubble_sim']:.4f} | {r['u_bubble_ref']:.4f} | {r['err_bubble_pct']:+.2f}% | "
            f"{r['asym_sim']:.4f} | {r['asym_ref']:.4f} | {r['re_spike']:.1f} | "
            f"{r['ma_peak']:.3f} | {r['stop_reason']} |"
        )
    (out_root / "acceptance_table.md").write_text(head + "\n".join(rows) + "\n")
    return {"cases": results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("calib", "run"), default="calib")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None, help="comma separated case names to run")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    out_root = Path(args.out)

    if args.smoke:
        torch.set_num_threads(args.threads)
        co = scmp_coexistence(0.55, steps=600, ny=48, tau=1.0)
        print("smoke coexistence:", co.as_dict())
        cfg = SCMPRTConfig(lam=48, layers=8.0, gravity=2.0e-4, tau_relax=0.8,
                           a0_over_lam=0.04, steps=600, sample_every=10, name="smoke")
        res = run_scmp_rt(cfg, rho_high=co.rho_high, rho_low=co.rho_low)
        res.pop("history")
        print(json.dumps({k: res[k] for k in
                          ("name", "steps_run", "stop_reason", "ma_peak", "u_spike_sim",
                           "u_bubble_sim", "atwood_measured", "asym_sim", "asym_ref",
                           "mass_drift_pct", "rho_min_end", "rho_max_end")}, indent=2))
        return 0

    if args.stage == "calib":
        payload = stage_calib(out_root, quick=args.quick)
        print(json.dumps({k: payload[k] for k in ("selected_sc_g", "selected")}, indent=2))
        return 0

    sc_g, rho_high, rho_low = _load_calibration(out_root)
    specs = list(base_specs(sc_g, rho_high, rho_low))
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        specs = [s for s in specs if s.name in wanted]
    results = []
    for spec in specs:
        print(f"[run] {spec.name} lam={spec.lam} g={spec.gravity:.3g} tau={spec.tau_relax}",
              flush=True)
        res = run_spec(spec, out_root)
        print(f"      u_spike={res['u_spike_sim']:.4f} ref={res['u_spike_ref']:.4f} "
              f"err={res['err_spike_pct']:+.2f}% asym={res['asym_sim']:.3f}/"
              f"{res['asym_ref']:.3f} stop={res['stop_reason']}", flush=True)
        results.append(res)
    aggregate(results, out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())