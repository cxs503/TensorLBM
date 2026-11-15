#!/usr/bin/env python3
"""XVAL-LBM-POINTWISE-REFERENCE-REGRESSION, round 4: D3Q27 whole step.

Runs the predeclared arms in ``predeclared.json`` and writes ``summary.json``,
``table.csv``, per-arm raw metrics under ``runs/`` and the frozen reference
bundle under ``refdata/``.

The three earlier rounds covered D2Q9 (single step, then 30-step frozen
refdata) and D3Q19 (30-step frozen refdata).  D3Q27 -- the lattice used by the
Geier cumulant operator, i.e. the flagship 3-D path -- had no pointwise
independent reference at all; this round closes that gap.

Usage
-----
    PYTHONPATH=src python xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/d3q27/run_xval.py
    PYTHONPATH=src python .../run_xval.py --quick          # smoke test, no physics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XV = HERE.parent
REPO = XV.parents[1]
for _p in (str(REPO / "src"), str(XV), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms27  # noqa: E402
import ref27  # noqa: E402
import refcore27 as core  # noqa: E402

PREDECLARED = json.loads((HERE / "predeclared.json").read_text())
GATE = float(PREDECLARED["predeclared_thresholds"]["gate_fp64_code_equivalence"])
GATE_BASIS = float(PREDECLARED["predeclared_thresholds"]["mrt_basis_identity"])
GATE_CONTROL = float(PREDECLARED["predeclared_thresholds"]["control_detection_min"])
PHYS_ACCEPT = float(PREDECLARED["predeclared_thresholds"]["physics_decay_pct_acceptance"])
PHYS_TARGET = float(PREDECLARED["predeclared_thresholds"]["physics_decay_pct_target"])

RUNS = HERE / "runs"
REFDATA = HERE / "refdata"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# field construction
# ---------------------------------------------------------------------------
def make_field(shape: tuple[int, int, int], seed: int, amplitude: float, noise: float,
               rho_amp: float = 0.01) -> np.ndarray:
    """Near-equilibrium positive PDF field: discrete equilibrium + u/rho noise + neq noise."""
    rng = np.random.default_rng(seed)
    rho = 1.0 + rho_amp * rng.standard_normal(shape)
    u = amplitude * rng.standard_normal((3,) + shape)
    f = core.discrete_equilibrium(rho, u) + noise * rng.standard_normal((27,) + shape)
    return np.ascontiguousarray(f, dtype=np.float64)


# ---------------------------------------------------------------------------
# phase 0: sanity (permutation / basis / streaming / conservation)
# ---------------------------------------------------------------------------
def phase_sanity() -> dict:
    out: dict = {}
    vel = core.velocities()
    out["n_directions"] = len(vel)
    out["velocity_set_is_D3Q27"] = bool(
        sorted(map(tuple, vel.tolist())) == sorted(
            (x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
        )
    )
    perm = ref27.lpref.direction_permutation("D3Q27")
    out["permutation_is_bijection"] = bool(sorted(perm.tolist()) == list(range(27)))
    out["mrt_basis_gap"] = ref27.moment_basis_gap()
    out["mrt_basis_ok"] = bool(out["mrt_basis_gap"] < GATE_BASIS)

    # streaming pin: the reference rolling operator and the repo kernel must
    # move the same population by the same amount (exact data movement).
    rng = np.random.default_rng(12345)
    probe = rng.standard_normal((27, 6, 5, 7))
    a = arms27.stream(probe)
    b = arms27.stream_definition(probe)
    out["streaming_pin_abs_max"] = float(np.abs(a - b).max())
    out["streaming_pin_ok"] = bool(out["streaming_pin_abs_max"] == 0.0)

    # conservation drift of every arm collision half (mass/momentum)
    f = make_field((6, 5, 4), seed=3, amplitude=0.05, noise=1e-3)
    drift = {}
    for arm in arms27.COLLIDERS:
        post = arms27._to_numpy(arms27.COLLIDERS[arm](arms27._to_torch(f), 0.7))  # noqa: SLF001
        drift[arm] = core.drift(f, post)
    out["collision_conservation"] = drift
    out["conservation_ok"] = bool(
        max(max(d["mass_drift"], d["momentum_drift"]) for d in drift.values()) < 1e-12
    )
    return out


# ---------------------------------------------------------------------------
# phase 1: code-equivalence sweep (gated)
# ---------------------------------------------------------------------------
def compare_config(shape: tuple[int, int, int], tau: float, seed: int,
                   steps: int, keep: tuple[int, ...], amplitude: float,
                   noise: float, arms_to_run: list[str]) -> list[dict]:
    f0 = make_field(shape, seed, amplitude, noise)
    rows: list[dict] = []
    refs: dict[str, dict[int, np.ndarray]] = {}
    for arm in arms_to_run:
        model = arms27.REFERENCE_MODEL[arm]
        if model not in refs:
            t0 = time.time()
            collider = ref27.reference_collider(tau, model)
            refs[model] = ref27.whole_step(f0, collider, steps, keep=keep)
            log(f"    ref build+run {model} tau={tau}: {time.time() - t0:.1f}s")
        got = arms27.run_arm(arm, f0, tau, steps, keep=keep)
        for t in sorted(refs[model]):
            m = core.compare(got[t], refs[model][t])
            gated = arm in arms27.GATED_ARMS
            rows.append({
                "arm": arm,
                "reference_model": model,
                "grid": list(shape),
                "tau": tau,
                "seed": seed,
                "step": t,
                "gated": gated,
                "gate": GATE if gated else None,
                "eps_scaled_inf": m["eps_scaled_inf"],
                "eps_rel_rms": m["eps_rel_rms"],
                "abs_diff_max": m["abs_diff_max"],
                "pass": (bool(m["eps_scaled_inf"] < GATE) if gated else None),
            })
        del got
    return rows


# ---------------------------------------------------------------------------
# phase 2: positive controls (detection power)
# ---------------------------------------------------------------------------
def phase_controls(shape: tuple[int, int, int] = (8, 8, 8), tau: float = 0.7,
                   seed: int = 0, steps: int = 10) -> dict:
    f0 = make_field(shape, seed, 0.05, 1e-3)
    keep = (steps,)
    colliders = {m: ref27.reference_collider(tau, m) for m in ("srt",)}
    base_ref = ref27.whole_step(f0, colliders["srt"], steps, keep=keep)[steps]

    # C1: tau perturbation of the reference by 1e-3 relative
    c1_ref = ref27.whole_step(
        f0, ref27.reference_collider(tau * (1.0 + 1e-3), "srt"), steps, keep=keep
    )[steps]

    # C2: wrong equilibrium convention (continuous Maxwellian)
    import lbmpy_pointwise_ref as lpref  # noqa: PLC0415

    c2_collider = lpref.PointwiseReference(
        "D3Q27", "srt", tau=tau, continuous_equilibrium=True
    )
    c2_ref = ref27.whole_step(f0, c2_collider, steps, keep=keep)[steps]

    # C3: direction mis-permutation -- the arm collides a field whose
    # populations were rotated by one lattice direction.
    rot = np.roll(f0, shift=1, axis=0)
    got_rot = arms27.run_arm("A1_d3q27_bgk", rot, tau, steps, keep=keep)[steps]

    # C4: streaming omitted from the reference step
    f = f0.copy()
    for _ in range(steps):
        f = colliders["srt"].apply(f)
    c4_ref = f
    got_streamed = arms27.run_arm("A1_d3q27_bgk", f0, tau, steps, keep=keep)[steps]

    out = {
        "C1_tau_perturbation_1e-3": {
            "eps_scaled_inf": core.compare(base_ref, c1_ref)["eps_scaled_inf"],
        },
        "C2_wrong_equilibrium_convention": {
            "eps_scaled_inf": core.compare(base_ref, c2_ref)["eps_scaled_inf"],
        },
        "C3_direction_mispermutation": {
            "eps_scaled_inf": core.compare(got_rot, base_ref)["eps_scaled_inf"],
        },
        "C4_streaming_omission": {
            "eps_scaled_inf": core.compare(c4_ref, got_streamed)["eps_scaled_inf"],
        },
    }
    for k, v in out.items():
        v["detected"] = bool(v["eps_scaled_inf"] > GATE_CONTROL)
    out["all_detected"] = all(v["detected"] for v in out.values() if isinstance(v, dict))
    return out


# ---------------------------------------------------------------------------
# phase 3: physics anchor (3-D shear wave, closed-form decay)
# ---------------------------------------------------------------------------
def phase_physics(n: int = 16, tau: float = 0.7, u0: float = 0.02,
                  steps: int = 200) -> dict:
    k = core.lattice_wavenumber(n)
    nu = (tau - 0.5) / 3.0
    shape = (n, n, n)
    f0 = core.shear_wave_3d(shape, u0, k)
    a0 = core.shear_amplitude(f0)
    analytic = np.array([a0 * np.exp(-nu * k * k * t) for t in (steps // 2, steps)])
    out = {"n": n, "tau": tau, "nu": nu, "k": k, "u0": u0, "steps": steps,
           "amplitude_t0": a0, "arms": {}}
    for arm in ("A1_d3q27_bgk", "A4_d3q27_cumulant_geier"):
        f = np.array(f0, dtype=np.float64, copy=True)
        measured = {}
        for t in range(1, steps + 1):
            f = arms27.step(arm, f, tau)
            if t in (steps // 2, steps):
                measured[t] = core.shear_amplitude(f)
        errs = {}
        for t, val in measured.items():
            if t == steps:
                idx = 1
            else:
                idx = 0
            errs[t] = float(abs(val - analytic[idx]) / analytic[idx])
        out["arms"][arm] = {
            "measured": {str(t): v for t, v in measured.items()},
            "analytic": {str(t): float(analytic[0] if t == steps // 2 else analytic[1])
                         for t in measured},
            "rel_err": {str(t): e for t, e in errs.items()},
            "worst_rel_err": max(errs.values()),
            "pass_15pct": bool(max(errs.values()) < PHYS_ACCEPT),
            "within_3pct": bool(max(errs.values()) < PHYS_TARGET),
        }
    return out


# ---------------------------------------------------------------------------
# phase 4: convention attribution (amplitude scaling of the probe gap)
# ---------------------------------------------------------------------------
def phase_scaling(shape: tuple[int, int, int] = (8, 8, 8), tau: float = 0.7,
                  seed: int = 0, steps: int = 10) -> dict:
    amps = [0.002, 0.01, 0.05, 0.15]
    series: dict[str, list[dict]] = {
        "A4_geier_vs_cumulant_discrete_eq": [],
        "A4_geier_vs_cumulant_continuous_eq": [],
        "A5_legacy_vs_cumulant_continuous_eq": [],
    }
    for amp in amps:
        f0 = make_field(shape, seed, amp, 1e-3)
        got4 = arms27.run_arm("A4_d3q27_cumulant_geier", f0, tau, steps, keep=(steps,))[steps]
        got5 = arms27.run_arm("A5_d3q27_cumulant_legacy", f0, tau, steps, keep=(steps,))[steps]
        ref_d = ref27.whole_step(
            f0, ref27.reference_collider(tau, "cumulant_discrete_eq"), steps, keep=(steps,)
        )[steps]
        ref_c = ref27.whole_step(
            f0, ref27.reference_collider(tau, "cumulant"), steps, keep=(steps,)
        )[steps]
        series["A4_geier_vs_cumulant_discrete_eq"].append(
            {"amplitude": amp, "eps": core.compare(got4, ref_d)["eps_scaled_inf"]})
        series["A4_geier_vs_cumulant_continuous_eq"].append(
            {"amplitude": amp, "eps": core.compare(got4, ref_c)["eps_scaled_inf"]})
        series["A5_legacy_vs_cumulant_continuous_eq"].append(
            {"amplitude": amp, "eps": core.compare(got5, ref_c)["eps_scaled_inf"]})
        log(f"    scaling amp={amp}: " + " ".join(
            f"{k.split('_vs_')[0]}={v[-1]['eps']:.3e}" for k, v in series.items()))
    out = {}
    for key, pts in series.items():
        x = np.log([p["amplitude"] for p in pts])
        y = np.log([max(p["eps"], 1e-300) for p in pts])
        slope, intercept = np.polyfit(x, y, 1)
        out[key] = {
            "points": pts,
            "loglog_slope": float(slope),
            "floor_eps": float(np.exp(intercept)),
        }
    return out


# ---------------------------------------------------------------------------
# frozen bundle (so the gate runs without lbmpy)
# ---------------------------------------------------------------------------
def sha256_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def freeze_bundle(tau: float = 0.7, seed: int = 0, steps: int = 30) -> dict:
    REFDATA.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    keep = (1, 10, 30)
    meta = {"config": {"tau": tau, "seed": seed, "steps": steps,
                       "amplitude": 0.05, "noise": 1e-3, "keep": list(keep),
                       "grids": PREDECLARED["grids"]},
            "models": {}}
    for gi, grid in enumerate(PREDECLARED["grids"]):
        shape = tuple(grid)
        f0 = make_field(shape, seed, 0.05, 1e-3)
        arrays[f"f0_g{gi}"] = f0
        for arm in arms27.GATED_ARMS:
            model = arms27.REFERENCE_MODEL[arm]
            if model in meta["models"]:
                continue
            log(f"  freezing {model} on {shape}")
            ref = ref27.whole_step(f0, ref27.reference_collider(tau, model), steps, keep=keep)
            for t, field in ref.items():
                arrays[f"ref_{model}_g{gi}_t{t}"] = field
            meta["models"].setdefault(model, []).append({"grid": list(shape), "steps": list(ref)})
    npz_path = REFDATA / "pointwise_d3q27_v1.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest = {
        "bundle": npz_path.name,
        "config": meta["config"],
        "models": meta["models"],
        "arm_to_model": arms27.REFERENCE_MODEL,
        "gated_arms": list(arms27.GATED_ARMS),
        "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "arrays": {k: sha256_arr(v) for k, v in arrays.items()},
    }
    (REFDATA / "pointwise_d3q27_v1.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    log(f"  bundle {npz_path.name}: {len(arrays)} arrays, "
        f"{npz_path.stat().st_size / 1024:.0f} KiB")
    return {"path": str(npz_path.relative_to(REPO)), "n_arrays": len(arrays),
            "size_kib": round(npz_path.stat().st_size / 1024, 1),
            "sha256": manifest["sha256"]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test only")
    ap.add_argument("--skip-bundle", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    RUNS.mkdir(parents=True, exist_ok=True)
    if args.quick:
        grids = [(4, 4, 4)]
        taus = [0.7]
        seeds = [0]
        steps = 3
        keep = (1, 3)
        arms = ["A1_d3q27_bgk", "A4_d3q27_cumulant_geier"]
    else:
        grids = [tuple(g) for g in PREDECLARED["grids"]]
        taus = list(PREDECLARED["taus"])
        seeds = list(PREDECLARED["seeds"])
        steps = int(PREDECLARED["steps"])
        keep = (1, 10, steps)
        arms = list(arms27.COLLIDERS)
    amplitude = float(PREDECLARED["amplitude"])
    noise = float(PREDECLARED["noise"])

    log("phase 0: sanity")
    sanity = phase_sanity()
    log(f"  basis gap {sanity['mrt_basis_gap']:.2e}, stream pin "
        f"{sanity['streaming_pin_abs_max']:.1e}, conservation ok={sanity['conservation_ok']}")

    log("phase 1: code-equivalence sweep")
    rows: list[dict] = []
    for grid in grids:
        for tau in taus:
            for seed in seeds:
                log(f"  grid={grid} tau={tau} seed={seed}")
                rows.extend(compare_config(grid, tau, seed, steps, keep, amplitude, noise, arms))

    (RUNS / "equivalence_rows.json").write_text(json.dumps(rows, indent=1))
    with (HERE / "table.csv").open("w") as fh:
        fh.write("arm,reference_model,grid,tau,seed,step,gated,gate,eps_scaled_inf,"
                 "eps_rel_rms,abs_diff_max,pass\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in (
                "arm", "reference_model", "grid", "tau", "seed", "step", "gated", "gate",
                "eps_scaled_inf", "eps_rel_rms", "abs_diff_max", "pass")) + "\n")

    gated_rows = [r for r in rows if r["gated"]]
    failing = [r for r in gated_rows if not r["pass"]]
    log(f"  gated rows {len(gated_rows)}, failing {len(failing)}")

    log("phase 2: positive controls")
    controls = phase_controls()
    log("  " + ", ".join(f"{k.split('_')[0]}={v['eps_scaled_inf']:.2e}"
                         for k, v in controls.items() if isinstance(v, dict)))

    physics = scaling = None
    if not args.quick:
        log("phase 3: physics anchor (shear wave)")
        physics = phase_physics()
        for arm, v in physics["arms"].items():
            log(f"  {arm}: worst rel err {v['worst_rel_err'] * 100:.2f}%")
        log("phase 4: convention attribution (amplitude scaling)")
        scaling = phase_scaling()

    bundle = None
    if not args.quick and not args.skip_bundle:
        log("phase 5: freezing reference bundle")
        bundle = freeze_bundle(tau=0.7, seed=0, steps=steps)

    verdicts = {
        "C0_sanity": {
            "passed": bool(sanity["permutation_is_bijection"] and sanity["mrt_basis_ok"]
                           and sanity["streaming_pin_ok"] and sanity["conservation_ok"]),
            "detail": {k: sanity[k] for k in (
                "permutation_is_bijection", "mrt_basis_gap", "streaming_pin_abs_max",
                "conservation_ok")},
        },
        "C1_code_equivalence_gated": {
            "threshold": GATE,
            "worst_eps_scaled_inf": max((r["eps_scaled_inf"] for r in gated_rows), default=None),
            "n_rows": len(gated_rows),
            "n_failing": len(failing),
            "passed": bool(gated_rows and not failing),
        },
        "C2_positive_controls": controls,
        "C3_physics_shear_wave": (
            None if physics is None else {
                "threshold_pct": PHYS_ACCEPT,
                "target_pct": PHYS_TARGET,
                "worst_rel_err_pct": max(v["worst_rel_err"] for v in physics["arms"].values()) * 100,
                "passed": all(v["pass_15pct"] for v in physics["arms"].values()),
                "within_target": all(v["within_3pct"] for v in physics["arms"].values()),
            }
        ),
    }

    summary = {
        "anchor": PREDECLARED["anchor"],
        "round": PREDECLARED["round"],
        "lattice": PREDECLARED["lattice"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            **ref27.provenance(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "predeclared": {
            "grids": PREDECLARED["grids"],
            "taus": PREDECLARED["taus"],
            "steps": steps,
            "seeds": seeds,
            "amplitude": amplitude,
            "noise": noise,
            "gated_arms": list(arms27.GATED_ARMS),
            "probe_arms": list(arms27.PROBE_ARMS),
            "thresholds": PREDECLARED["predeclared_thresholds"],
        },
        "sanity": sanity,
        "code_equivalence": {
            "rows": rows,
            "worst_by_arm": {
                arm: max((r["eps_scaled_inf"] for r in rows if r["arm"] == arm), default=None)
                for arm in arms
            },
        },
        "controls": controls,
        "physics_shear_wave": physics,
        "convention_attribution": scaling,
        "frozen_bundle": bundle,
        "verdicts": verdicts,
        "all_gated_pass": verdicts["C1_code_equivalence_gated"]["passed"],
        "all_controls_detected": controls["all_detected"],
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"done in {summary['wall_seconds']}s; gated pass={summary['all_gated_pass']}, "
        f"controls detected={summary['all_controls_detected']}")
    return 0 if (summary["all_gated_pass"] and summary["all_controls_detected"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())