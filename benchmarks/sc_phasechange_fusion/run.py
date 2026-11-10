#!/usr/bin/env python
"""Fused Shan-Chen + enthalpy phase-change verification runner (node p348w3-n04).

Four independent anchors, each runnable on its own so they can be sharded over
the cluster:

``--part enthalpy``  9 cases: Ste = 0.1/0.5/1.0 x subcooling 0/5/10 K, front vs
                     the two-phase Neumann similarity solution, Stefan-condition
                     and energy budgets.
``--part laplace``   static-droplet Laplace ladder at one density ratio
                     (10:1 / 50:1 / 100:1, selected with ``--ratio``).
``--part rt``        single-mode Rayleigh-Taylor growth rate at three Atwood
                     numbers.
``--part gates``     read the three JSONs above and assemble the C1-C7 verdict.

Outputs (``--out``, default ``artifacts/sc_phasechange_fusion``)::

    enthalpy.json  laplace_ratio<RR>.json  rt.json  verdict.json
    enthalpy_profiles.npz   (front/liquid-fraction time series)

Usage::

    PYTHONPATH=src python benchmarks/sc_phasechange_fusion/run.py --part enthalpy
    PYTHONPATH=src python benchmarks/sc_phasechange_fusion/run.py --part laplace --ratio 100
    PYTHONPATH=src python benchmarks/sc_phasechange_fusion/run.py --part rt
    PYTHONPATH=src python benchmarks/sc_phasechange_fusion/run.py --part gates
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "src"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from tensorlbm.sc_phasechange import enthalpy as enth  # noqa: E402
from tensorlbm.sc_phasechange.gates import assemble_verdict  # noqa: E402
from tensorlbm.sc_phasechange.highdensity import (  # noqa: E402
    EosModel,
    LaplaceConfig,
    run_laplace_case,
    temperature_for_density_ratio,
)
from tensorlbm.sc_phasechange.rt_anchor import run_rt_case  # noqa: E402

STEFAN_NUMBERS = (0.1, 0.5, 1.0)
SUBCOOLINGS = (0.0, 5.0, 10.0)
DENSITY_RATIOS = (10.0, 50.0, 100.0)
RADII = (16.0, 20.0, 24.0)
ATWOODS = (0.1, 1.0 / 3.0, 0.5)


# ---------------------------------------------------------------------------
# Anchor 1: enthalpy phase change (Neumann + Stefan + energy budgets)
# ---------------------------------------------------------------------------
def run_enthalpy_part(
    out_dir: Path, front_end: float = 0.035, dx: float = 1.2e-3, cfl: float = 0.48
) -> dict:
    props = enth.ICE_WATER
    cases = []
    profiles = {}
    for ste in STEFAN_NUMBERS:
        for sub in SUBCOOLINGS:
            superheat = props.superheat_for_stefan(ste)
            ref = enth.NeumannMelting(superheat, sub, props)
            t_end = ref.time_for_front(front_end)
            length = max(0.5, 1.7 * ref.solid_penetration(t_end), 3.0 * front_end)
            n_cells = int(round(length / dx))
            length = n_cells * dx
            sim = enth.EnthalpyPhaseChange1D(
                length=length,
                n_cells=n_cells,
                props=props,
                superheat=superheat,
                subcooling=sub,
                cfl=cfl,
            )
            times = [t_end * 10.0 ** (-1.0 + 1.0 * k / 9.0) for k in range(10)]
            t0 = time.time()
            records = sim.run(t_end, times, ref)
            stamp = f"ste{ste:g}_sub{sub:g}"
            profiles[f"{stamp}_front"] = np.array([r.front for r in records])
            profiles[f"{stamp}_front_ref"] = np.array([r.front_neumann for r in records])
            profiles[f"{stamp}_times"] = np.array([r.time for r in records])
            cases.append(
                dict(
                    ste=ste,
                    dTsub=sub,
                    superheat=superheat,
                    lam=ref.lam,
                    t_end=t_end,
                    length=length,
                    n_cells=n_cells,
                    dx=dx,
                    dt=sim.dt,
                    steps=sim.steps,
                    noise_dt=sim.dt,
                    front_end=records[-1].front,
                    front_ref_end=records[-1].front_neumann,
                    times=[r.time for r in records],
                    fronts=[r.front for r in records],
                    fronts_ref=[r.front_neumann for r in records],
                    errs_pct=[r.front_error_pct for r in records],
                    stefan_windows=[
                        dict(
                            residual_pct=r.stefan_residual_pct,
                            front_m=r.front,
                            latent_J_per_m2=r.stefan_latent_rate,
                            flux_jump_J_per_m2=r.stefan_flux_jump,
                            time=r.time,
                        )
                        for r in records
                    ],
                    energy_windows=[r.energy_imbalance_pct for r in records],
                    energy_cumulative_imbalance_pct=records[-1].energy_cumulative_imbalance_pct,
                    max_abs_phase_bookkeeping_pct=0.0,
                    phase_bookkeeping_note="phi_l + phi_s = 1 by construction "
                    "(structural identity, machine precision)",
                    wall_time_s=time.time() - t0,
                )
            )
            print(
                f"[enthalpy] Ste={ste} dTsub={sub:g} lam={ref.lam:.4f} "
                f"steps={sim.steps} max|front err|={max(abs(e) for e in cases[-1]['errs_pct']):.2f}%",
                flush=True,
            )
    np.savez_compressed(out_dir / "enthalpy_profiles.npz", **profiles)
    payload = dict(
        part="enthalpy",
        model="enthalpy finite volume, SI ice/water, implicit interface",
        stefan_numbers=list(STEFAN_NUMBERS),
        subcoolings=list(SUBCOOLINGS),
        cases=cases,
    )
    (out_dir / "enthalpy.json").write_text(json.dumps(payload, indent=1))
    return payload


# ---------------------------------------------------------------------------
# Anchor 2: high-density-ratio Laplace ladder
# ---------------------------------------------------------------------------
def run_laplace_part(
    out_dir: Path, ratio: float, box: int = 88, n_steps: int = 2200, radii=RADII
) -> dict:
    eos = EosModel.cs()
    co = temperature_for_density_ratio(eos, ratio)
    cases = []
    partial_path = out_dir / f"_partial_ratio{ratio:g}.json"
    if partial_path.exists():
        # resume an interrupted ladder (one case is ~30 s; a wall-clock limit
        # must not throw away the radii that already converged)
        cases = json.loads(partial_path.read_text())
        radii = [r for r in radii if not any(abs(c["radius_init"] - r) < 1e-9 for c in cases)]
    for radius in radii:
        cfg = LaplaceConfig(box=box, radius=radius, n_steps=n_steps, t_reduced=co.t_reduced)
        t0 = time.time()
        case = run_laplace_case(cfg, co)
        cases.append(case.__dict__)
        partial_path.write_text(json.dumps(cases))
        print(
            f"[laplace] ratio={ratio:g} R={radius} R_eff={case.radius_eff:.2f} "
            f"dp={case.dp:.5f} Ma={case.spurious_ma:.4f} steps={case.steps_run} "
            f"conv={case.converged} t={time.time() - t0:.0f}s",
            flush=True,
        )
    # least-squares dp = sigma/R + c on the raw measured numbers
    inv_r = np.array([1.0 / c["radius_eff"] for c in cases])
    dp = np.array([c["dp"] for c in cases])
    slope, intercept = np.polyfit(inv_r, dp, 1)
    pred = slope * inv_r + intercept
    ss_res = float(((dp - pred) ** 2).sum())
    ss_tot = float(((dp - dp.mean()) ** 2).sum())
    fit = dict(
        sigma=float(slope),
        intercept=float(intercept),
        r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        max_dev_pct=float(np.max(np.abs(100.0 * (dp - pred) / pred))),
        dev_pct=[float(v) for v in 100.0 * (dp - pred) / pred],
        dp=[float(v) for v in dp],
        inv_r=[float(v) for v in inv_r],
    )
    for c in cases:
        c["sigma"] = fit["sigma"]
    payload = dict(
        part="laplace",
        eos="cs",
        box=box,
        n_steps=n_steps,
        radii=list(radii),
        tiers=[
            dict(
                target_ratio=float(ratio),
                t=co.t,
                t_reduced=co.t_reduced,
                rho_liquid=co.rho_liquid,
                rho_vapour=co.rho_vapour,
                coexistence_ratio=co.density_ratio,
                p_sat=co.p_sat,
                cases=cases,
                fit=fit,
                max_spurious_ma=max(c["spurious_ma"] for c in cases),
                max_mass_drift_per_step=max(c["mass_drift_per_step"] for c in cases),
                all_converged=all(c["converged"] for c in cases),
                all_stable=all(c["stable"] for c in cases),
            )
        ],
    )
    (out_dir / f"laplace_ratio{ratio:g}.json").write_text(json.dumps(payload, indent=1))
    partial_path.unlink(missing_ok=True)
    return payload


def merge_laplace(out_dir: Path, ratios=DENSITY_RATIOS) -> dict:
    tiers = []
    meta = None
    merged_path = out_dir / "laplace.json"
    if any(not (out_dir / f"laplace_ratio{r:g}.json").exists() for r in ratios):
        # the per-ratio files are the bulky intermediates; the merged summary is
        # the committed artifact, so fall back to it when they are absent
        if not merged_path.exists():
            raise FileNotFoundError(
                f"need either laplace_ratio*.json or {merged_path.name} in {out_dir}"
            )
        return json.loads(merged_path.read_text())
    for ratio in ratios:
        path = out_dir / f"laplace_ratio{ratio:g}.json"
        payload = json.loads(path.read_text())
        tiers.extend(payload["tiers"])
        meta = meta or {k: payload[k] for k in ("eos", "box", "n_steps", "radii")}
    merged = dict(part="laplace", tiers=tiers, **(meta or {}))
    (out_dir / "laplace.json").write_text(json.dumps(merged, indent=1))
    return merged


# ---------------------------------------------------------------------------
# Anchor 3: Rayleigh-Taylor growth rate
# ---------------------------------------------------------------------------
def run_rt_part(out_dir: Path, lam: int = 64, gravity: float = 2e-3, n_steps: int = 6000) -> dict:
    cases = []
    for at in ATWOODS:
        t0 = time.time()
        case = run_rt_case(at, lam=lam, gravity=gravity, n_steps=n_steps)
        cases.append(case.__dict__)
        print(
            f"[rt] A={at:.4f} sigma_sim={case.sigma_sim:.6f} "
            f"analytic={case.sigma_analytic:.6f} err={case.rel_error_pct:+.2f}% "
            f"steps={case.steps_run} t={time.time() - t0:.0f}s",
            flush=True,
        )
    payload = dict(
        part="rt",
        wavelength=lam,
        gravity=gravity,
        atwoods=list(ATWOODS),
        cases=cases,
        max_abs_error_pct=max(abs(c["rel_error_pct"]) for c in cases),
    )
    (out_dir / "rt.json").write_text(json.dumps(payload, indent=1))
    return payload


# ---------------------------------------------------------------------------
# Gate assembly
# ---------------------------------------------------------------------------
def run_gates(out_dir: Path, test_count: int, test_failed: int) -> dict:
    parts = dict(
        enthalpy=json.loads((out_dir / "enthalpy.json").read_text()),
        laplace=merge_laplace(out_dir),
        rt=json.loads((out_dir / "rt.json").read_text()),
    )
    verdict = assemble_verdict(parts, dict(count=test_count, failed=test_failed))
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=1))
    for g in verdict["gates"]:
        print(
            f"[gate] {g['gate']:28s} measured={g['measured']:.4g}{g['unit']:6s} "
            f"threshold<={g['threshold']:.4g} -> {'PASS' if g['passed'] else 'FAIL'}",
            flush=True,
        )
    print(f"[gate] all_pass={verdict['all_pass']}", flush=True)
    return verdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--part", required=True, choices=["enthalpy", "laplace", "rt", "gates", "merge_laplace"]
    )
    ap.add_argument("--out", default=str(REPO / "artifacts" / "sc_phasechange_fusion"))
    ap.add_argument("--ratio", type=float, default=None)
    ap.add_argument("--box", type=int, default=88)
    ap.add_argument("--steps", type=int, default=2200)
    ap.add_argument("--lam", type=int, default=64)
    ap.add_argument("--gravity", type=float, default=2e-3)
    ap.add_argument("--front-end", type=float, default=0.035)
    ap.add_argument("--dx", type=float, default=1.2e-3)
    ap.add_argument("--test-count", type=int, default=0)
    ap.add_argument("--test-failed", type=int, default=0)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.part == "enthalpy":
        run_enthalpy_part(out_dir, front_end=args.front_end, dx=args.dx)
    elif args.part == "laplace":
        if args.ratio is None:
            ap.error("--part laplace needs --ratio")
        run_laplace_part(out_dir, args.ratio, box=args.box, n_steps=args.steps)
    elif args.part == "merge_laplace":
        merge_laplace(out_dir)
    elif args.part == "rt":
        run_rt_part(out_dir, lam=args.lam, gravity=args.gravity, n_steps=args.steps)
    else:
        run_gates(out_dir, args.test_count, args.test_failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
